//! The bytecode interpreter.
//!
//! Design notes (from the discussion that shaped it):
//! * `match`-on-opcode dispatch. The hot loop state (pc, base, stack pointer,
//!   bytecode/constant pointers, multres) lives in *local variables* so the
//!   compiler keeps it in registers, mirroring LuaJIT's "VM state in
//!   registers" discipline. It is synced back to the `Interp` fields only
//!   around calls/returns (cold), never per instruction.
//! * All raw-pointer access is confined to a handful of macros (`reg!`,
//!   `setreg!`, `fetch!`, `kslot!`); the opcode arms and the public entry
//!   points are `unsafe`-free. The backing stack has fixed capacity and never
//!   reallocates, so those pointers stay valid.
//! * Register windows follow LuaJIT's FR2 layout (callee `base` =
//!   caller_base + A + 2; function at `base-2`), matching the bytecode.
//! * Errors use `LuaResult` with a fieldless error enum; the error object and
//!   yield count live on the `LuaState`. Hot paths return no `Result`.
//! * Lua->Lua calls do not recurse in Rust: `CALL` pushes a `Frame` and keeps
//!   looping; `RET` pops one. Only tail calls and re-entrant C calls recurse.

use crate::bc::*;

pub mod meta;

pub mod err;
use crate::err::{LuaError, LuaResult};
use crate::func::{GcFunc, LuaClosure, Upval};
use crate::gc::GcPtr;
use crate::jit::{HOTCOUNT_CALL, HOTCOUNT_LOOP};
use crate::proto::{KGc, PROTO_UV_IMMUTABLE, PROTO_UV_LOCAL, PROTO_VARARG, Proto};
use crate::runtime::meta::MM;
use crate::state::{LuaState, Suspend};
use crate::table::LuaTable;
use crate::value::*;

/// Frame type markers kept in the low bits of the frame-link slot at
/// `base-1`, exactly as in LuaJIT's `lj_frame.h` (FR2 layout):
///
/// ```text
///        base-2  base-1      |  base  base+1 ...
///       [func   PC/delta/ft] | [slots ...]
///       ^-- frame            | ^-- base    ^-- top
/// ```
///
/// * `FRAME_LUA`: the link is the caller's return PC (a 4-aligned pointer,
///   low bits 00). Caller base, wanted results and the result slot are all
///   recovered from the CALL instruction at `pc[-1]`.
/// * `FRAME_C`: a host (`execute`) entry; bits 3.. hold `want + 1`.
/// * `FRAME_VARG`: bits 3.. hold the slot delta back to the frame that
///   carries the real link; varargs live between the two frames.
const FRAME_LUA: u64 = 0;
const FRAME_C: u64 = 1;
pub const FRAME_VARG: u64 = 3;
const FRAME_CONT: u64 = 2;
pub const FRAME_TYPE_MASK: u64 = 3;

/// Continuation IDs stored in the cont-slot of a FRAME_CONT frame.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cont {
    Ra = 0,
    Nop = 1,
    Condt = 2,
    Condf = 3,
}
impl Cont {
    pub fn encode(self, extra: u32) -> u64 { ((self as u64) << 32) | (extra as u64) }
    pub fn decode(bits: u64) -> (Cont, u32) {
        (unsafe { std::mem::transmute::<u8, Cont>((bits >> 32) as u8) },
         (bits & 0xFFFF_FFFF) as u32)
    }
}

/// Call a value with the given arguments and collect all results.
/// The host entry point into the VM.
pub fn call(l: &mut LuaState, func: LuaValue, args: &[LuaValue]) -> LuaResult<Vec<LuaValue>> {
    l.stack_ensure(args.len() + STACK_SAFETY);
    l.top = 0;
    l.stack[0] = func;
    l.stack[1] = LuaValue::NIL;
    for (i, &a) in args.iter().enumerate() {
        l.stack[2 + i] = a;
    }
    let n = execute(l, 0, args.len(), -1)?;
    Ok((0..n).map(|i| l.stack[i]).collect())
}

/// Execute a call to the function at `func_slot` with `nargs` arguments
/// already placed at `func_slot + 2 ..`. Leaves the results at `func_slot`
/// and returns their count.
pub fn execute(l: &mut LuaState, func_slot: usize, nargs: usize, want: i32) -> LuaResult<usize> {
    l.c_depth += 1;
    // Ensure the stack can hold the entry frame (FR2: 2-slot header +
    // framesize, plus extra for call_c's arg region).
    l.stack_ensure(func_slot + nargs + STACK_SAFETY);
    let r = execute_inner(l, func_slot, nargs, want);
    l.c_depth -= 1;
    r
}

/// Safety margin added to every stack_ensure: protects against a few
/// extra slots written by CALL/VARG/TSETM frame setup.
const STACK_SAFETY: usize = 64;

fn execute_inner(l: &mut LuaState, func_slot: usize, nargs: usize, want: i32) -> LuaResult<usize> {
    let mut nargs = nargs;
    let f = l.stack[func_slot];
    let gf = match f.as_func() {
        Some(p) => p,
        None => {
            // lj_meta_call: try __call metamethod.
            nargs = meta::meta_call(l, func_slot, nargs)?;
            let f = l.stack[func_slot];
            f.as_func().expect("__call did not produce a function")
        }
    };
    if let GcFunc::C(cc) = gf.as_ref() {
        return call_c(l, cc.f, func_slot, nargs, want);
    }
    let mut vm = Interp::new(l);
    let link = (((want + 1) as u64) << 3) | FRAME_C;
    vm.enter_lua(gf, func_slot, nargs, link);
    vm.run()
}

/// Call a C function with `nargs` args at `func_slot + 2`, moving up to
/// `want` results back to `func_slot`.
fn call_c(
    l: &mut LuaState,
    f: crate::func::CFunction,
    func_slot: usize,
    nargs: usize,
    want: i32,
) -> LuaResult<usize> {
    let args_base = func_slot + 2;
    let saved_base = l.base;
    let saved_top = l.top;
    l.base = args_base;
    l.top = args_base + nargs;
    if l.global().heap.should_collect() {
        crate::gc::full_gc(l.global());
    }
    let r = f(l);
    let n = match r {
        Ok(n) => n as usize,
        Err(e) => {
            // Restore the frame even on error, so protected callers
            // (pcall) see a consistent base/top.
            l.base = saved_base;
            l.top = saved_top;
            return Err(e);
        }
    };
    for i in 0..n {
        l.stack[func_slot + i] = l.stack[args_base + i];
    }
    l.base = saved_base;
    l.top = saved_top;
    if want >= 0 {
        for i in n..(want as usize) {
            l.stack[func_slot + i] = LuaValue::NIL;
        }
        l.top = func_slot + want as usize;
        Ok(want as usize)
    } else {
        l.top = func_slot + n;
        Ok(n)
    }
}

/// Dispatch-loop exit reasons: a host-frame return, or a switch between
/// the plain and the recording interpreter (LuaJIT switches dispatch
/// tables instead).
enum Flow {
    Ret(usize),
    Rec,
}

/// Interpreter context. The per-instruction hot state is *not* kept here; it
/// lives in locals inside `run` and is synced to these fields only around
/// calls and returns.
struct Interp {
    l: *mut LuaState,
    sp: *mut LuaValue,
    base: usize,
    pc: usize,
    cl: GcPtr<GcFunc>,
    bcp: *const BCIns,
    knp: *const f64,
    ksp: *const LuaValue,
    multres: usize,
}

impl Interp {
    fn new(l: &mut LuaState) -> Interp {
        let sp = l.stack.as_mut_ptr();
        Interp {
            l,
            sp,
            base: 0,
            pc: 0,
            cl: GcPtr::from_addr(8).unwrap(), // placeholder; set by enter_lua
            bcp: std::ptr::null(),
            knp: std::ptr::null(),
            ksp: std::ptr::null(),
            multres: 0,
        }
    }

    /// The interpreter has exclusive access to the `LuaState` for the whole
    /// `run` invocation; the raw pointer (and this deliberate `&self ->
    /// &mut` escape hatch) exists so borrows of `Interp` fields and the
    /// state can overlap without fighting the borrow checker.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    fn l(&self) -> &mut LuaState {
        unsafe { &mut *self.l }
    }

    #[inline(always)]
    fn lua_cl(&self) -> &LuaClosure {
        match self.cl.as_ref() {
            GcFunc::Lua(c) => c,
            GcFunc::C(_) => unreachable!("C function in Lua frame"),
        }
    }

    #[inline(always)]
    fn proto(&self) -> &Proto {
        self.lua_cl().proto.as_ref()
    }

    #[inline(always)]
    fn at(&self, abs: usize) -> LuaValue {
        unsafe { *self.sp.add(abs) }
    }

    #[inline(always)]
    fn set_at(&self, abs: usize, v: LuaValue) {
        unsafe { *self.sp.add(abs) = v }
    }

    /// A string constant, read from the resolved KBASE table (`ksp`) — the
    /// value is precomputed at registration, so this is a single load, no
    /// interner lookup. Only valid for KSTR/GGET/GSET/TGETS/TSETS operands.
    #[inline(always)]
    fn kstr_at(&self, d: u32) -> LuaValue {
        let v = unsafe { *self.ksp.add(d as usize) };
        debug_assert!(v.is_string());
        v
    }

    /// Set up a Lua frame for the function at `func_slot` and switch the
    /// `Interp` fields to the callee. `link` is stored in the frame-link
    /// slot (`callbase - 1`); the caller must have synced its locals into
    /// the fields first. Mirrors LuaJIT's `ins_call` + FUNCF/FUNCV headers.
    fn enter_lua(&mut self, gf: GcPtr<GcFunc>, func_slot: usize, nargs: usize, link: u64) {
        let cl = match gf.as_ref() {
            GcFunc::Lua(c) => c,
            _ => unreachable!(),
        };
        let pt = cl.proto.as_ref();
        let numparams = pt.numparams as usize;
        let callbase = func_slot + 2;

        // Lazy stack growth: ensure the new frame's extent + margin fits.
        let need = if (pt.flags & PROTO_VARARG) != 0 {
            (callbase + nargs + 2) + numparams + pt.framesize as usize + 16
        } else {
            callbase + pt.framesize as usize + 16
        };
        let l = self.l();
        l.stack_ensure(need);
        // sp may have moved after Vec resize.
        self.sp = l.stack.as_mut_ptr();

        self.set_at(callbase - 1, LuaValue::from_bits(link));

        if (pt.flags & PROTO_VARARG) != 0 {
            // FUNCV: shift the fixed params up past the varargs and chain a
            // vararg frame back to the one holding the real link.
            let newbase = callbase + nargs + 2;
            self.set_at(newbase - 2, LuaValue::func(gf));
            for i in 0..numparams {
                let v = if i < nargs {
                    self.at(callbase + i)
                } else {
                    LuaValue::NIL
                };
                self.set_at(newbase + i, v);
            }
            let delta = (newbase - callbase) as u64;
            self.set_at(newbase - 1, LuaValue::from_bits((delta << 3) | FRAME_VARG));
            self.base = newbase;
        } else {
            for i in nargs..numparams {
                self.set_at(callbase + i, LuaValue::NIL);
            }
            self.base = callbase;
        }

        self.cl = gf;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
        self.ksp = pt.kstrv.as_ptr();
        self.pc = 1; // skip the FUNCF/FUNCV header
        self.l().top = self.base + pt.framesize as usize;
    }

    fn reload(&mut self, cl: GcPtr<GcFunc>) {
        let pt = match cl.as_ref() {
            GcFunc::Lua(c) => c.proto.as_ref(),
            _ => unreachable!(),
        };
        self.cl = cl;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
        self.ksp = pt.kstrv.as_ptr();
    }

    /// Generalised fast-return: walk the VARG chain inline and return the
    /// real frame's bp, along with `(want, ret_ip, caller_a)`.
    #[inline(always)]
    fn ret_fast_n(&self, mut bp: *const LuaValue) -> Option<(*const LuaValue, i32, *const BCIns, i32)> {
        let mut link = unsafe { (*bp.sub(1)).to_bits() };
        while link & FRAME_TYPE_MASK == FRAME_VARG {
            bp = unsafe { bp.sub((link >> 3) as usize) };
            link = unsafe { (*bp.sub(1)).to_bits() };
        }
        if link & FRAME_TYPE_MASK == FRAME_LUA && self.l().openuv.is_empty() {
            let ret_ip = link as *const BCIns;
            let call_ins = unsafe { *ret_ip.sub(1) };
            let want = bc_b(call_ins) as i32 - 1;
            let caller_a = bc_a(call_ins) as i32;
            Some((bp, want, ret_ip, caller_a))
        } else {
            None
        }
    }

    /// Whether a return from the frame at `bp` may take the inline fast
    /// path: a plain Lua frame link and no open upvalues to close. Returns
    /// the caller's wanted result count (from the CALL instruction's B).
    #[inline(always)]
    fn ret_fast(&self, bp: *const LuaValue) -> Option<i32> {
        let link = unsafe { (*bp.sub(1)).to_bits() };
        if link & FRAME_TYPE_MASK == FRAME_LUA && self.l().openuv.is_empty() {
            let ret_ip = link as *const BCIns;
            let call_ins = unsafe { *ret_ip.sub(1) };
            Some(bc_b(call_ins) as i32 - 1)
        } else {
            None
        }
    }

    /// Reload the interpreter for the closure owning the frame at `bp`.
    #[inline(always)]
    fn reload_at(&mut self, bp: *const LuaValue) {
        let cl = unsafe { *bp.sub(2) }.as_func().unwrap();
        self.reload(cl);
    }

    /// `mmcall` + FRAME_CONT (lj_meta.c's `mmcall` + `vm_call_dispatch_f`):
    /// set up a continuation frame above the current one and enter the Lua
    /// metamethod. Caller must be synced; afterwards the `Interp` fields
    /// point into the metamethod's frame (caller must `resync!()`).
    ///
    /// Stack layout (FR2):
    /// ```text
    ///  mmbase-4  mmbase-3   mmbase-2  mmbase-1  | mmbase  mmbase+1 ..
    /// [cont      saved PC] [mo        link|CONT] | [arg0   arg1 ..]
    /// ```
    fn mmcall_cont(&mut self, cont_id: Cont, extra: u32, mo: LuaValue, args: &[LuaValue]) {
        let saved_base = self.base;
        // curr_topL: scratch right above the running frame, +2 for cont/PC.
        let func_slot = saved_base + self.proto().framesize as usize + 2;
        let mmbase = func_slot + 2;
        {
            let need = mmbase + args.len() + 16;
            let l = self.l();
            l.stack_ensure(need);
            self.sp = l.stack.as_mut_ptr();
        }
        self.set_at(mmbase - 4, LuaValue::from_bits(cont_id.encode(extra)));
        self.set_at(mmbase - 3, LuaValue::from_bits(self.pc as u64));
        self.set_at(func_slot, mo);
        for (i, &v) in args.iter().enumerate() {
            self.set_at(mmbase + i, v);
        }
        let link = (((mmbase - saved_base) as u64) << 3) | FRAME_CONT;
        self.enter_lua(mo.as_func().unwrap(), func_slot, args.len(), link);
    }

    /// Call a C-function metamethod inline (no continuation frame): the
    /// result is available immediately. Uses scratch above the frame.
    fn call_c_fn(
        &mut self,
        f: crate::func::CFunction,
        mo: LuaValue,
        args: &[LuaValue],
    ) -> LuaResult<LuaValue> {
        let fs = self.base + self.proto().framesize as usize;
        {
            let need = fs + 2 + args.len() + 8;
            let l = self.l();
            l.stack_ensure(need);
            self.sp = l.stack.as_mut_ptr();
        }
        self.set_at(fs, mo);
        self.set_at(fs + 1, LuaValue::NIL);
        for (i, &v) in args.iter().enumerate() {
            self.set_at(fs + 2 + i, v);
        }
        let n = self.call_c_inline(f, fs, args.len())?;
        Ok(if n > 0 { self.at(fs) } else { LuaValue::NIL })
    }

    /// Continuation dispatch (LuaJIT's `->cont_dispatch` + `cont_*`
    /// handlers): a metamethod called through `mmcall_cont` has returned;
    /// `mmbase` is its frame base, results were copied to `mmbase - 2` by
    /// `do_return`. Restores the caller frame and applies the continuation.
    /// Returns `None`: execution always continues in the caller.
    fn cont_dispatch(&mut self, mmbase: usize, link: u64, n: usize) -> Option<usize> {
        let delta = (link >> 3) as usize;
        let caller_base = mmbase - delta;
        let (cont, extra) = Cont::decode(self.at(mmbase - 4).to_bits());
        let saved_pc = self.at(mmbase - 3).to_bits() as usize;
        // Ensure one valid result (cont_dispatch's "Ensure one valid arg").
        let result = if n > 0 { self.at(mmbase - 2) } else { LuaValue::NIL };

        self.base = caller_base;
        let cl = self.at(caller_base - 2).as_func().unwrap();
        self.reload(cl);
        self.l().top = caller_base + self.proto().framesize as usize;

        match cont {
            Cont::Ra => {
                // Store result in the A register of the triggering
                // instruction (encoded in `extra`).
                self.set_at(caller_base + extra as usize, result);
                self.pc = saved_pc;
            }
            Cont::Nop => {
                self.pc = saved_pc;
            }
            Cont::Condt => {
                // saved_pc points at the fused JMP.
                let jmp = unsafe { *self.bcp.add(saved_pc) };
                self.pc = saved_pc + 1;
                if result.is_truthy() {
                    self.pc = (self.pc as i64 + bc_j(jmp)) as usize;
                }
            }
            Cont::Condf => {
                let jmp = unsafe { *self.bcp.add(saved_pc) };
                self.pc = saved_pc + 1;
                if !result.is_truthy() {
                    self.pc = (self.pc as i64 + bc_j(jmp)) as usize;
                }
            }
        }
        None
    }

    /// The dispatch loop. The entire hot state is two locals — `bp` (the
    /// current base as a pointer, LuaJIT's BASE) and `ip` (a walking
    /// instruction pointer, LuaJIT's PC) — so both live in registers on
    /// every dispatch. Everything else (`self.knp`, `self.multres`, ...)
    /// is re-read from `self` in the arms that need it; keeping more locals
    /// alive across the dispatch forces spills (measured: rustc packs the
    /// extras into an XMM register and unpacks per instruction).
    /// `sync!`/`resync!` bridge to the fields around calls and returns.
    ///
    /// `run` is the mode trampoline: it re-enters `dispatch` whenever the
    /// trace recorder turns on or off, standing in for LuaJIT's dispatch
    /// table switching (`lj_dispatch_update`). `dispatch::<true>` is the
    /// recording interpreter: it feeds every instruction through
    /// `lj_record_ins` before executing it.
    fn run(&mut self) -> LuaResult<usize> {
        loop {
            let rec = self.l().global().jit.state == crate::jit::TraceState::Record;
            let r = if rec { self.dispatch::<true>() } else { self.dispatch::<false>() };
            match r {
                Ok(Flow::Ret(n)) => return Ok(n),
                Ok(Flow::Rec) => continue, // Recording toggled: switch modes.
                Err(e) => {
                    // An error raised while recording aborts the trace.
                    if rec {
                        crate::jit::trace::rec_abort_error(self.l().global());
                    }
                    return Err(e);
                }
            }
        }
    }

    fn dispatch<const REC: bool>(&mut self) -> LuaResult<Flow> {
        let mut bp = unsafe { self.sp.add(self.base) };
        let mut ip = unsafe { self.bcp.add(self.pc) };

        macro_rules! cur_base {
            () => {
                unsafe { bp.offset_from(self.sp) as usize }
            };
        }
        macro_rules! reg {
            ($i:expr) => {
                unsafe { *bp.add(($i) as usize) }
            };
        }
        macro_rules! setreg {
            ($i:expr, $v:expr) => {{
                let v = $v;
                unsafe { *bp.add(($i) as usize) = v }
            }};
        }
        macro_rules! kslot {
            ($d:expr) => {
                LuaValue::number_raw(unsafe { *self.knp.add(($d) as usize) })
            };
        }
        macro_rules! jump {
            ($ins:expr) => {
                ip = unsafe { ip.offset(bc_j($ins) as isize) }
            };
        }
        macro_rules! sync {
            () => {{
                self.base = cur_base!();
                self.pc = unsafe { ip.offset_from(self.bcp) as usize };
            }};
        }
        macro_rules! resync {
            () => {{
                bp = unsafe { self.sp.add(self.base) };
                ip = unsafe { self.bcp.add(self.pc) };
            }};
        }
        // Numeric binary op fast path; slow path calls meta_arith for
        // string coercion and metamethods (lj_meta_arith).
        macro_rules! arith {
            ($a:expr, $xv:expr, $yv:expr, $mm:expr, $x:ident, $y:ident, $body:expr) => {{
                let xv = $xv;
                let yv = $yv;
                if xv.is_number() && yv.is_number() {
                    let $x = xv.num();
                    let $y = yv.num();
                    setreg!($a, LuaValue::number_raw($body));
                } else {
                    sync!();
                    match self.meta_arith($mm, xv, yv, $a)? {
                        Some(r) => setreg!($a, r),
                        None => { resync!(); continue; }
                    }
                }
            }};
        }
        // reg `op` constant (VN form): check the register, constant is
        // always numeric.
        macro_rules! arith_vn {
            ($a:expr, $ins:expr, $mm:expr, $x:ident, $y:ident, $body:expr) => {{
                let xv = reg!(bc_b($ins));
                if xv.is_number() {
                    let $x = xv.num();
                    let $y = unsafe { *self.knp.add(bc_c($ins) as usize) };
                    setreg!($a, LuaValue::number_raw($body));
                } else {
                    sync!();
                    match self.meta_arith($mm, xv, kslot!(bc_c($ins)), $a)? {
                        Some(r) => setreg!($a, r),
                        None => { resync!(); continue; }
                    }
                }
            }};
        }
        // constant `op` reg (NV form): swap — constant is first argument
        // but LL semantics require it to be the *second* for ADD/NV (A=B+C
        // where B is reg, C is const). The macro body must do `y + x` etc.
        macro_rules! arith_nv {
            ($a:expr, $ins:expr, $mm:expr, $x:ident, $y:ident, $body:expr) => {{
                let $x = kslot!(bc_c($ins));
                let yv = reg!(bc_b($ins));
                if yv.is_number() {
                    let $y = yv.num();
                    setreg!($a, LuaValue::number_raw($body));
                } else {
                    sync!();
                    match self.meta_arith($mm, $x, yv, $a)? {
                        Some(r) => setreg!($a, r),
                        None => { resync!(); continue; }
                    }
                }
            }};
        }
        // Comparison + fused following JMP. `op` is the bytecode ordinal
        // (ISLT=0, ISGE=1, ISLE=2, ISGT=3), matching lj_meta_comp's
        // encoding. ISGE/ISGT are the *unordered* comparisons (NaN takes
        // the jump), because the parser emits them as the negation of
        // ISLT/ISLE — same as the dasc VMs and rec_comp.
        macro_rules! cmp {
            ($op:expr, $xv:expr, $yv:expr) => {{
                let xv = $xv;
                let yv = $yv;
                if xv.is_number() && yv.is_number() {
                    let x = xv.num();
                    let y = yv.num();
                    let cond = match $op {
                        0 => x < y, 1 => !(x < y), 2 => x <= y, 3 => !(x <= y),
                        _ => unreachable!(),
                    };
                    let jmp = unsafe { *ip };
                    ip = unsafe { ip.add(1) };
                    if cond { jump!(jmp); }
                } else {
                    sync!();
                    match self.meta_comp(xv, yv, $op)? {
                        Some(cond) => {
                            let jmp = unsafe { *ip };
                            ip = unsafe { ip.add(1) };
                            if cond { jump!(jmp); }
                        }
                        None => { resync!(); continue; }
                    }
                }
            }};
        }
        macro_rules! branch {
            ($cond:expr) => {{
                let jmp = unsafe { *ip };
                ip = unsafe { ip.add(1) };
                if $cond {
                    jump!(jmp);
                }
            }};
        }
        // FORL/ITERL loop-edge semantics, shared between the normal arm
        // and the "hot counter fired, recording just started" path (the
        // hot instruction itself runs before the trace entry).
        macro_rules! forl_body {
            ($ins:expr, $a:expr) => {{
                let i = reg!($a + FORL_IDX).num();
                let s = reg!($a + FORL_STOP).num();
                let st = reg!($a + FORL_STEP).num();
                let ni = i + st;
                let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                if cont {
                    let nv = LuaValue::number_raw(ni);
                    setreg!($a + FORL_IDX, nv);
                    setreg!($a + FORL_EXT, nv);
                    jump!($ins);
                }
            }};
        }
        macro_rules! iterl_body {
            ($ins:expr, $a:expr) => {{
                let first = reg!($a);
                if !first.is_nil() {
                    setreg!($a - 1, first);
                    jump!($ins);
                }
            }};
        }

        loop {
            if REC {
                // Recording dispatch: feed the instruction about to be
                // executed through the recorder (lj_trace_ins).
                sync!();
                let pt = self.lua_cl().proto;
                let (pc, base) = (self.pc, self.base);
                if !crate::jit::trace::rec_ins(self.l(), base, pt, pc) {
                    return Ok(Flow::Rec); // Recording ended: switch modes.
                }
                resync!();
            }
            let ins = unsafe { *ip };
            ip = unsafe { ip.add(1) };
            let a = bc_a(ins);
            match bc_op(ins) {
                // -- Comparisons (ORDER matters; see bc.rs) --
                BCOp::ISLT => cmp!(0u32, reg!(a), reg!(bc_d(ins))),
                BCOp::ISGE => cmp!(1u32, reg!(a), reg!(bc_d(ins))),
                BCOp::ISLE => cmp!(2u32, reg!(a), reg!(bc_d(ins))),
                BCOp::ISGT => cmp!(3u32, reg!(a), reg!(bc_d(ins))),
                BCOp::ISEQV => {
                    let x = reg!(a);
                    let y = reg!(bc_d(ins));
                    let cond = val_eq(x, y);
                    if cond {
                        branch!(true);
                    } else if x.is_table() && y.is_table() {
                        sync!();
                        match self.meta_equal(x, y, 0)? {
                            Some(eq) => branch!(eq),
                            None => { resync!(); continue; }
                        }
                    } else {
                        branch!(false);
                    }
                }
                BCOp::ISNEV => {
                    let x = reg!(a);
                    let y = reg!(bc_d(ins));
                    let cond = val_eq(x, y);
                    if cond {
                        branch!(false);
                    } else if x.is_table() && y.is_table() {
                        sync!();
                        match self.meta_equal(x, y, 1)? {
                            Some(eq) => branch!(!eq),
                            None => { resync!(); continue; }
                        }
                    } else {
                        branch!(true);
                    }
                }
                BCOp::ISEQS => {
                    let cond = val_eq(reg!(a), self.kstr_at(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISNES => {
                    let cond = !val_eq(reg!(a), self.kstr_at(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISEQN => {
                    let cond = val_eq(reg!(a), kslot!(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISNEN => {
                    let cond = !val_eq(reg!(a), kslot!(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISEQP => {
                    let cond = val_eq(reg!(a), PRI[bc_d(ins) as usize]);
                    branch!(cond);
                }
                BCOp::ISNEP => {
                    let cond = !val_eq(reg!(a), PRI[bc_d(ins) as usize]);
                    branch!(cond);
                }
                BCOp::ISTC => {
                    let d = reg!(bc_d(ins));
                    let cond = d.is_truthy();
                    if cond {
                        setreg!(a, d);
                    }
                    branch!(cond);
                }
                BCOp::ISFC => {
                    let d = reg!(bc_d(ins));
                    let cond = !d.is_truthy();
                    if cond {
                        setreg!(a, d);
                    }
                    branch!(cond);
                }
                BCOp::IST => {
                    let cond = reg!(bc_d(ins)).is_truthy();
                    branch!(cond);
                }
                BCOp::ISF => {
                    let cond = !reg!(bc_d(ins)).is_truthy();
                    branch!(cond);
                }

                // -- Unary and move --
                BCOp::MOV => setreg!(a, reg!(bc_d(ins))),
                BCOp::NOT => setreg!(a, LuaValue::boolean(!reg!(bc_d(ins)).is_truthy())),
                BCOp::UNM => {
                    let v = reg!(bc_d(ins));
                    if v.is_number() {
                        setreg!(a, LuaValue::number_raw(-v.num()));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Unm, v, v, a)? {
                            Some(r) => setreg!(a, r),
                            None => { resync!(); continue; }
                        }
                    }
                }
                BCOp::LEN => {
                    let v = reg!(bc_d(ins));
                    if let Some(sid) = v.as_string_id() {
                        let n = self.l().heap().strings.get(sid).len();
                        setreg!(a, LuaValue::number(n as f64));
                    } else if let Some(t) = v.as_table() {
                        setreg!(a, LuaValue::number(t.as_ref().len() as f64));
                    } else {
                        sync!();
                        let r = self.meta_len(v)?;
                        setreg!(a, r);
                    }
                }

                // -- Arithmetic --
                BCOp::ADDVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), MM::Add, x, y, x + y),
                BCOp::SUBVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), MM::Sub, x, y, x - y),
                BCOp::MULVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), MM::Mul, x, y, x * y),
                BCOp::DIVVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), MM::Div, x, y, x / y),
                BCOp::MODVV => {
                    arith!(
                        a,
                        reg!(bc_b(ins)),
                        reg!(bc_c(ins)),
                        MM::Mod,
                        x,
                        y,
                        x - (x / y).floor() * y
                    )
                }
                BCOp::ADDVN => arith_vn!(a, ins, MM::Add, x, y, x + y),
                BCOp::SUBVN => arith_vn!(a, ins, MM::Sub, x, y, x - y),
                BCOp::MULVN => arith_vn!(a, ins, MM::Mul, x, y, x * y),
                BCOp::DIVVN => arith_vn!(a, ins, MM::Div, x, y, x / y),
                BCOp::MODVN => arith_vn!(a, ins, MM::Mod, x, y, x - (x / y).floor() * y),
                BCOp::ADDNV => arith_nv!(a, ins, MM::Add, kv, y, kv.num() + y),
                BCOp::SUBNV => arith_nv!(a, ins, MM::Sub, kv, y, kv.num() - y),
                BCOp::MULNV => arith_nv!(a, ins, MM::Mul, kv, y, kv.num() * y),
                BCOp::DIVNV => arith_nv!(a, ins, MM::Div, kv, y, kv.num() / y),
                BCOp::MODNV => {
                    arith_nv!(a, ins, MM::Mod, kv, y, {
                        let x = kv.num();
                        x - (x / y).floor() * y
                    })
                }
                BCOp::POW => {
                    arith!(
                        a,
                        reg!(bc_b(ins)),
                        reg!(bc_c(ins)),
                        MM::Pow,
                        x,
                        y,
                        vm_pow(x, y)
                    )
                }
                BCOp::CAT => {
                    sync!();
                    self.gc_check();
                    let r = self.meta_cat(bc_b(ins), bc_c(ins))?;
                    setreg!(a, r);
                }

                // -- Constants --
                BCOp::KSTR => {
                    let v = self.kstr_at(bc_d(ins));
                    setreg!(a, v);
                }
                BCOp::KSHORT => setreg!(a, LuaValue::number(bc_d(ins) as i16 as f64)),
                BCOp::KNUM => setreg!(a, kslot!(bc_d(ins))),
                BCOp::KPRI => setreg!(a, PRI[bc_d(ins) as usize]),
                BCOp::KNIL => {
                    for i in a..=bc_d(ins) {
                        setreg!(i, LuaValue::NIL);
                    }
                }

                // -- Upvalues --
                BCOp::UGET => {
                    let uv = self.lua_cl().upvals[bc_d(ins) as usize];
                    setreg!(a, self.upval_get(uv));
                }
                BCOp::USETV => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = reg!(bc_d(ins));
                    self.upval_set(uv, v);
                }
                BCOp::USETS => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = self.kstr_at(bc_d(ins));
                    self.upval_set(uv, v);
                }
                BCOp::USETN => {
                    let uv = self.lua_cl().upvals[a as usize];
                    self.upval_set(uv, kslot!(bc_d(ins)));
                }
                BCOp::USETP => {
                    let uv = self.lua_cl().upvals[a as usize];
                    self.upval_set(uv, PRI[bc_d(ins) as usize]);
                }
                BCOp::UCLO => {
                    sync!();
                    self.close_upvals(cur_base!() + a as usize);
                    jump!(ins);
                }
                BCOp::FNEW => {
                    sync!();
                    self.gc_check();
                    let v = self.new_closure(bc_d(ins));
                    setreg!(a, v);
                }

                // -- Tables --
                BCOp::TNEW => {
                    // Inline: only sync when GC is due (cold path).
                    if self.l().global().heap.should_collect() {
                        sync!();
                        self.gc_check();
                    }
                    let t = self.l().heap().alloc_table(LuaTable::new(0, 0));
                    setreg!(a, LuaValue::table(t));
                }
                BCOp::TDUP => {
                    sync!();
                    self.gc_check();
                    let templ = match &self.proto().kgc[bc_d(ins) as usize] {
                        KGc::Table(t) => t.dup(),
                        KGc::TableRef(t) => t.as_ref().dup(),
                        _ => unreachable!("expected template table"),
                    };
                    let t = self.l().heap().alloc_table(templ);
                    setreg!(a, LuaValue::table(t));
                }
                BCOp::GGET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    let v = env.as_ref().get_str(key);
                    if !v.is_nil() || env.as_ref().metatable.is_none() {
                        setreg!(a, v);
                    } else {
                        sync!(); match self.meta_tget(LuaValue::table(env), key, a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                    }
                }
                BCOp::GSET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    let mt = env.as_ref().metatable;
                    if mt.is_none() || !env.as_ref().get_str(key).is_nil() {
                        env.as_mut().set_str(key, reg!(a));
                    } else {
                        sync!(); match self.meta_tset(LuaValue::table(env), key, reg!(a))? { Some(_) => {}, None => { resync!(); continue; } }
                    }
                }
                BCOp::TGETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    if let Some(tab) = t.as_table() {
                        let v = if k.is_string() {
                            tab.as_ref().get_str(k)
                        } else if k.is_number() {
                            let ki = k.num() as i32;
                            if ki as f64 == k.num() && ki >= 0 {
                                tab.as_ref().get_int(ki)
                            } else {
                                tab.as_ref().get(k)
                            }
                        } else {
                            tab.as_ref().get(k)
                        };
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            setreg!(a, v);
                        } else {
                            sync!(); match self.meta_tget(t, k, a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                        }
                    } else {
                        sync!(); match self.meta_tget(t, k, a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                    }
                }
                BCOp::TGETS => {
                    let t = reg!(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = self.kstr_at(bc_c(ins));
                        let v = tab.as_ref().get_str(k);
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            setreg!(a, v);
                        } else {
                            sync!(); match self.meta_tget(t, k, a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                        }
                    } else {
                        sync!(); match self.meta_tget(t, self.kstr_at(bc_c(ins)), a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                    }
                }
                BCOp::TGETB => {
                    let t = reg!(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = bc_c(ins) as i32;
                        let v = tab.as_ref().get_int(k);
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            setreg!(a, v);
                        } else {
                            sync!(); match self.meta_tget(t, LuaValue::number(k as f64), a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                        }
                    } else {
                        sync!(); match self.meta_tget(t, LuaValue::number(bc_c(ins) as f64), a)? { Some(v) => setreg!(a, v), None => { resync!(); continue; } }
                    }
                }
                BCOp::TSETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        if k.is_string() {
                            tab.as_mut().set_str(k, v);
                        } else if k.is_number() {
                            let ki = k.num() as i32;
                            if ki as f64 == k.num() && ki >= 0 {
                                tab.as_mut().set_int(ki, v);
                            } else {
                                tab.as_mut().set(k, v);
                            }
                        } else if k.is_nil() {
                            sync!();
                            return Err(self.l().runtime_error(b"table index is nil"));
                        } else {
                            tab.as_mut().set(k, v);
                        }
                    } else {
                        sync!(); match self.meta_tset(t, k, v)? { Some(_) => {}, None => { resync!(); continue; } }
                    }
                }
                BCOp::TSETS => {
                    let t = reg!(bc_b(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        let k = self.kstr_at(bc_c(ins));
                        tab.as_mut().set_str(k, v);
                    } else {
                        sync!(); match self.meta_tset(t, self.kstr_at(bc_c(ins)), v)? { Some(_) => {}, None => { resync!(); continue; } }
                    }
                }
                BCOp::TSETB => {
                    let t = reg!(bc_b(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        tab.as_mut().set_int(bc_c(ins) as i32, v);
                    } else {
                        sync!(); match self.meta_tset(t, LuaValue::number(bc_c(ins) as f64), v)? { Some(_) => {}, None => { resync!(); continue; } }
                    }
                }
                BCOp::TSETM => {
                    // Inline the fast path: table at R[a-1], values at
                    // R[a..a+multres-1], base_key from constant table.
                    let t = reg!(a as usize - 1);
                    if let Some(tab) = t.as_table() {
                        let base_key = unsafe { *self.knp.add(bc_d(ins) as usize) } as i64 - (1i64 << 52);
                        let mr = self.multres;
                        if mr > 0 && base_key == 1 {
                            let need = (base_key as u32).wrapping_add(mr as u32);
                            tab.as_mut().reasize(need);
                        }
                        for i in 0..mr {
                            let key = base_key + i as i64;
                            let v = reg!(a as usize + i);
                            if key >= 0 && key <= i32::MAX as i64 {
                                tab.as_mut().set_int(key as i32, v);
                            } else {
                                tab.as_mut().set(LuaValue::number(key as f64), v);
                            }
                        }
                    } else {
                        sync!();
                        self.tsetm(a, bc_d(ins))?;
                    }
                }

                // -- Calls / returns --
                BCOp::CALL => {
                    // Fast path (LuaJIT's ins_call): a Lua callee switches
                    // frames right here — one store for the frame link, no
                    // sync round-trip. C callees and vararg protos go slow.
                    let f = reg!(a);
                    if let Some(gf) = f.as_func()
                        && let GcFunc::Lua(cl) = gf.as_ref()
                    {
                        let ptref = cl.proto;
                        let pt = cl.proto.as_ref();
                        if (pt.flags & PROTO_VARARG) == 0 {
                            let nargs = bc_c(ins) as usize - 1;
                            let fs = pt.framesize as usize;
                            let need = cur_base!() + a as usize + 2 + fs + 8;
                            if need > self.l().stack.len() {
                                sync!();
                                self.l().stack_ensure(need);
                                self.sp = self.l().stack.as_mut_ptr();
                                resync!();
                            }
                            let newbp = unsafe { bp.add(a as usize + 2) };
                            unsafe { *newbp.sub(1) = LuaValue::from_bits(ip as u64) };
                            for i in nargs..pt.numparams as usize {
                                unsafe { *newbp.add(i) = LuaValue::NIL };
                            }
                            bp = newbp;
                            ip = unsafe { pt.bc.as_ptr().add(1) };
                            self.cl = gf;
                            self.bcp = pt.bc.as_ptr();
                            self.knp = pt.kn.as_ptr();
                            self.ksp = pt.kstrv.as_ptr();
                            self.l().top = cur_base!() + pt.framesize as usize;
                            // hotcall (vm_hotcall): count the FUNCF header.
                            // The other Lua-entry paths do not count until
                            // call recording lands (Phase 3).
                            if !REC
                                && bc_op(pt.bc[0]) == BCOp::FUNCF
                                && self.hot_count(ip as usize, HOTCOUNT_CALL)
                            {
                                sync!();
                                if self.hot_call(ptref) {
                                    // Record from the callee's first ins.
                                    return Ok(Flow::Rec);
                                }
                                resync!();
                            }
                            continue;
                        }
                    }
                    let nargs = bc_c(ins) as usize - 1;
                    sync!();
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                    resync!();
                }
                BCOp::CALLM => {
                    let nargs = bc_c(ins) as usize + self.multres;
                    sync!();
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                    resync!();
                }
                BCOp::CALLT => {
                    // Fast path: Lua callee, no vararg frame on either side
                    // — reuse the frame in place (BC_CALLT's hot route).
                    let f = reg!(a);
                    if let Some(gf) = f.as_func()
                        && let GcFunc::Lua(cl) = gf.as_ref()
                    {
                        let pt = cl.proto.as_ref();
                        let link = unsafe { (*bp.sub(1)).to_bits() };
                        if (pt.flags & PROTO_VARARG) == 0 && link & FRAME_TYPE_MASK != FRAME_VARG {
                            let nargs = bc_d(ins) as usize - 1;
                            let fs_need = cur_base!() + a.max(nargs as u32) as usize + pt.framesize as usize + 8;
                            if fs_need > self.l().stack.len() {
                                sync!();
                                self.l().stack_ensure(fs_need);
                                self.sp = self.l().stack.as_mut_ptr();
                                resync!();
                            }
                            let fs = unsafe { bp.add(a as usize) };
                            unsafe { *bp.sub(2) = f };
                            // Copy args in reverse to avoid overlap
                            // (same bug as do_tailcall's arg move).
                            for i in (0..nargs).rev() {
                                unsafe { *bp.add(i) = *fs.add(2 + i) };
                            }
                            for i in nargs..pt.numparams as usize {
                                unsafe { *bp.add(i) = LuaValue::NIL };
                            }
                            ip = unsafe { pt.bc.as_ptr().add(1) };
                            self.cl = gf;
                            self.bcp = pt.bc.as_ptr();
                            self.knp = pt.kn.as_ptr();
                            self.ksp = pt.kstrv.as_ptr();
                            self.l().top = cur_base!() + pt.framesize as usize;
                            continue;
                        }
                    }
                    let nargs = bc_d(ins) as usize - 1;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs)? {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::CALLMT => {
                    let nargs = bc_d(ins) as usize + self.multres;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs)? {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET0 => {
                    if let Some(want) = self.ret_fast(bp) {
                        let jmp = unsafe { (*bp.sub(1)).to_bits() } as *const BCIns;
                        let call_ins = unsafe { *jmp.sub(1) };
                        let dst = unsafe { bp.sub(2) };
                        for i in 0..want.max(0) as usize {
                            unsafe { *dst.add(i) = LuaValue::NIL };
                        }
                        self.multres = 0;
                        bp = unsafe { dst.sub(bc_a(call_ins) as usize) };
                        ip = jmp;
                        self.reload_at(bp);
                        self.l().top = if want >= 0 {
                            unsafe { dst.add(want as usize).offset_from(self.sp) as usize }
                        } else {
                            unsafe { dst.offset_from(self.sp) as usize }
                        };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(cur_base!(), 0) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET1 => {
                    if let Some(want) = self.ret_fast(bp) {
                        let jmp = unsafe { (*bp.sub(1)).to_bits() } as *const BCIns;
                        let call_ins = unsafe { *jmp.sub(1) };
                        let dst = unsafe { bp.sub(2) };
                        unsafe { *dst = *bp.add(a as usize) };
                        for i in 1..want.max(1) as usize {
                            unsafe { *dst.add(i) = LuaValue::NIL };
                        }
                        self.multres = 1;
                        bp = unsafe { dst.sub(bc_a(call_ins) as usize) };
                        ip = jmp;
                        self.reload_at(bp);
                        self.l().top = if want >= 0 {
                            unsafe { dst.add(want as usize).offset_from(self.sp) as usize }
                        } else {
                            unsafe { dst.add(1).offset_from(self.sp) as usize }
                        };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(cur_base!() + a as usize, 1) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET => {
                    let n = bc_d(ins) as usize - 1;
                    if let Some((wbp, want, ret_ip, ca)) = self.ret_fast_n(bp) {
                        let wbp = wbp as *mut LuaValue;
                        let src = unsafe { bp.add(a as usize) };
                        let dst = unsafe { wbp.sub(2) };
                        for i in 0..n {
                            unsafe { *dst.add(i) = *src.add(i) };
                        }
                        for i in n..(want.max(0) as usize) {
                            unsafe { *dst.add(i) = LuaValue::NIL };
                        }
                        self.multres = n;
                        bp = unsafe { dst.sub(ca as usize) };
                        ip = ret_ip;
                        self.reload_at(bp);
                        self.l().top = if want >= 0 {
                            unsafe { dst.add(want as usize).offset_from(self.sp) as usize }
                        } else {
                            unsafe { dst.add(n).offset_from(self.sp) as usize }
                        };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(cur_base!() + a as usize, n) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RETM => {
                    let n = self.multres + bc_d(ins) as usize;
                    if let Some((wbp, want, ret_ip, ca)) = self.ret_fast_n(bp) {
                        let wbp = wbp as *mut LuaValue;
                        let src = unsafe { bp.add(a as usize) };
                        let dst = unsafe { wbp.sub(2) };
                        for i in 0..n {
                            unsafe { *dst.add(i) = *src.add(i) };
                        }
                        for i in n..(want.max(0) as usize) {
                            unsafe { *dst.add(i) = LuaValue::NIL };
                        }
                        self.multres = n;
                        bp = unsafe { dst.sub(ca as usize) };
                        ip = ret_ip;
                        self.reload_at(bp);
                        self.l().top = if want >= 0 {
                            unsafe { dst.add(want as usize).offset_from(self.sp) as usize }
                        } else {
                            unsafe { dst.add(n).offset_from(self.sp) as usize }
                        };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(cur_base!() + a as usize, n) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }

                // -- Loops and branches --
                BCOp::FORI => {
                    let idx = reg!(a + FORL_IDX);
                    let stop = reg!(a + FORL_STOP);
                    let step = reg!(a + FORL_STEP);
                    if idx.is_number() && stop.is_number() && step.is_number() {
                        let (i, s, st) = (idx.num(), stop.num(), step.num());
                        setreg!(a + FORL_EXT, LuaValue::number_raw(i));
                        let enter = if st >= 0.0 { i <= s } else { i >= s };
                        if !enter {
                            jump!(ins);
                        }
                    } else {
                        sync!();
                        return Err(self
                            .l()
                            .runtime_error(b"'for' initial value must be a number"));
                    }
                }
                BCOp::JFORI => {
                    // FORI semantics; on loop entry go straight into the
                    // trace whose number sits in the JFORL at the target.
                    let idx = reg!(a + FORL_IDX);
                    let stop = reg!(a + FORL_STOP);
                    let step = reg!(a + FORL_STEP);
                    if idx.is_number() && stop.is_number() && step.is_number() {
                        let (i, s, st) = (idx.num(), stop.num(), step.num());
                        setreg!(a + FORL_EXT, LuaValue::number_raw(i));
                        let enter = if st >= 0.0 { i <= s } else { i >= s };
                        if enter {
                            sync!();
                            let jforl = (self.pc as i64 - 1 + bc_j(ins)) as usize;
                            let tno = bc_d(self.proto().bc[jforl]);
                            let r = crate::jit::exec::trace_exec(self.l(), self.base, tno);
                            self.pc = r.pc;
                            if r.baseslot != 2 {
                                self.trace_exit_frame(r.baseslot);
                            }
                            if self.rec_started() {
                                return Ok(Flow::Rec); // Hot exit: record a side trace.
                            }
                            resync!();
                            continue;
                        } else {
                            jump!(ins);
                        }
                    } else {
                        sync!();
                        return Err(self
                            .l()
                            .runtime_error(b"'for' initial value must be a number"));
                    }
                }
                BCOp::FORL | BCOp::IFORL => {
                    // FORL is the hot-counting variant (lj_vm's `hotloop`
                    // macro); IFORL is the blacklisted/non-counting one.
                    if !REC
                        && bc_op(ins) == BCOp::FORL
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            // Recording started: run this FORL un-recorded
                            // (it sits before the trace entry), then
                            // switch to the recording dispatch.
                            resync!();
                            forl_body!(ins, a);
                            sync!();
                            return Ok(Flow::Rec);
                        }
                        resync!();
                    }
                    forl_body!(ins, a);
                }
                BCOp::JFORL => {
                    // IFORL semantics; on loop-taken enter the compiled
                    // trace (the dasc VMs dispatch to BC_JLOOP).
                    let i = reg!(a + FORL_IDX).num();
                    let s = reg!(a + FORL_STOP).num();
                    let st = reg!(a + FORL_STEP).num();
                    let ni = i + st;
                    let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                    if cont {
                        let nv = LuaValue::number_raw(ni);
                        setreg!(a + FORL_IDX, nv);
                        setreg!(a + FORL_EXT, nv);
                        sync!();
                        let r = crate::jit::exec::trace_exec(self.l(), self.base, bc_d(ins));
                        self.pc = r.pc;
                        if r.baseslot != 2 {
                            self.trace_exit_frame(r.baseslot);
                        }
                        if self.rec_started() {
                            return Ok(Flow::Rec); // Hot exit: record a side trace.
                        }
                        resync!();
                    }
                }
                BCOp::LOOP | BCOp::ILOOP => {
                    // No-op apart from hot counting (ILOOP: not even that).
                    if !REC
                        && bc_op(ins) == BCOp::LOOP
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            return Ok(Flow::Rec); // pc already past LOOP.
                        }
                        resync!();
                    }
                }
                BCOp::JLOOP => {
                    // Enter the compiled trace; the interpreter resumes at
                    // whatever snapshot the trace exits through.
                    sync!();
                    let r = crate::jit::exec::trace_exec(self.l(), self.base, bc_d(ins));
                    self.pc = r.pc;
                    if r.baseslot != 2 {
                        self.trace_exit_frame(r.baseslot);
                    }
                    if self.rec_started() {
                        return Ok(Flow::Rec); // Hot exit: record a side trace.
                    }
                    resync!();
                }
                BCOp::JMP => jump!(ins),
                BCOp::ISNEXT => jump!(ins),
                BCOp::ITERC | BCOp::ITERN => {
                    sync!();
                    self.iter_call(a, bc_b(ins) as usize)?;
                    resync!();
                }
                BCOp::ITERL | BCOp::IITERL => {
                    if !REC
                        && bc_op(ins) == BCOp::ITERL
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            resync!();
                            iterl_body!(ins, a);
                            sync!();
                            return Ok(Flow::Rec);
                        }
                        resync!();
                    }
                    iterl_body!(ins, a);
                }
                BCOp::VARG => {
                    let link = unsafe { (*bp.sub(1)).to_bits() };
                    if link & FRAME_TYPE_MASK == FRAME_VARG {
                        let delta = (link >> 3) as usize;
                        let numparams = self.proto().numparams as usize;
                        let nvarg = (delta - 2).saturating_sub(numparams);
                        let dst = a as usize;
                        let src = unsafe { bp.sub(delta).add(numparams) };
                        if bc_b(ins) == 0 {
                            for i in 0..nvarg {
                                unsafe { *bp.add(dst + i) = *src.add(i) };
                            }
                            self.multres = nvarg;
                            self.l().top = cur_base!() + dst + nvarg;
                        } else {
                            let want = (bc_b(ins) - 1) as usize;
                            for i in 0..want {
                                unsafe {
                                    *bp.add(dst + i) =
                                        if i < nvarg { *src.add(i) } else { LuaValue::NIL };
                                }
                            }
                        }
                    }
                }

                // -- Bitwise ops (Lua 5.3+), lj_num2bit / lj_vm_tobit --
                BCOp::BNOT => {
                    let v = reg!(bc_d(ins));
                    let n = if v.is_number() { v.num() as i32 } else { 0 };
                    setreg!(a, LuaValue::number(n as f64));
                }
                BCOp::BAND => {
                    let xv = reg!(bc_b(ins));
                    let yv = reg!(bc_c(ins));
                    let x = if xv.is_number() { xv.num() as i32 } else { 0 };
                    let y = if yv.is_number() { yv.num() as i32 } else { 0 };
                    setreg!(a, LuaValue::number((x & y) as f64));
                }
                BCOp::BOR => {
                    let x = reg!(bc_b(ins)).num() as i32;
                    let y = reg!(bc_c(ins)).num() as i32;
                    setreg!(a, LuaValue::number((x | y) as f64));
                }
                BCOp::BXOR => {
                    let x = reg!(bc_b(ins)).num() as i32;
                    let y = reg!(bc_c(ins)).num() as i32;
                    setreg!(a, LuaValue::number((x ^ y) as f64));
                }
                BCOp::BSHL => {
                    let x = reg!(bc_b(ins)).num() as i32;
                    let y = (reg!(bc_c(ins)).num() as u32) & 31;
                    setreg!(a, LuaValue::number((x << y) as f64));
                }
                BCOp::BSHR => {
                    let x = reg!(bc_b(ins)).num() as i32;
                    let y = (reg!(bc_c(ins)).num() as u32) & 31;
                    setreg!(a, LuaValue::number(((x as u32) >> y) as f64));
                }
                BCOp::BSAR => {
                    let x = reg!(bc_b(ins)).num() as i32;
                    let y = (reg!(bc_c(ins)).num() as u32) & 31;
                    setreg!(a, LuaValue::number((x >> y) as f64));
                }

                // Explicit list (not `_`) so the match covers every opcode:
                // a wildcard shrinks the jump table and adds a bounds check
                // to every dispatch.
                other @ (BCOp::ISTYPE
                | BCOp::ISNUM
                | BCOp::KCDATA
                | BCOp::TGETR
                | BCOp::TSETR
                | BCOp::JITERL
                | BCOp::FUNCF
                | BCOp::IFUNCF
                | BCOp::JFUNCF
                | BCOp::FUNCV
                | BCOp::IFUNCV
                | BCOp::JFUNCV
                | BCOp::FUNCC
                | BCOp::FUNCCW) => {
                    sync!();
                    return Err(self
                        .l()
                        .runtime_error(format!("opcode {:?} not implemented", other).as_bytes()));
                }
            }
        }
    }

    // -- Cold slow paths -------------------------------------------------

    // -- JIT hot-path detection (lj_dispatch's hotloop/hotcall) -----------

    /// Decrement the hot counter hashed from `addr` (the interpreter PC
    /// *after* fetching the counting instruction, LuaJIT's offset-by-1
    /// convention). Returns true when it underflows: the path turned hot.
    /// Does nothing while the JIT is off.
    #[inline(always)]
    fn hot_count(&mut self, addr: usize, amount: crate::jit::HotCount) -> bool {
        let js = &mut self.l().global().jit;
        js.is_on() && js.hot_decrement(addr, amount)
    }

    /// `->vm_hotloop`: the FORL/ITERL/LOOP at `self.pc - 1` (locals must be
    /// synced) turned hot. Returns true when recording started and the
    /// caller must switch to the recording dispatch.
    #[cold]
    fn hot_loop(&mut self) -> bool {
        let pt = self.lua_cl().proto;
        let pc = self.pc - 1;
        let base = self.base;
        crate::jit::trace::trace_hot(self.l(), base, pt, pc);
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// `->vm_hotcall`: the FUNCF header of `pt` turned hot. Same contract
    /// as `hot_loop`; the frame must already be entered and synced.
    #[cold]
    fn hot_call(&mut self, pt: GcPtr<Proto>) -> bool {
        let base = self.base;
        crate::jit::trace::trace_hot(self.l(), base, pt, 0);
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// Did a trace exit just start a side-trace recording? The caller
    /// must then switch to the recording dispatch (`self.pc` is already
    /// at the exit's resume point).
    #[inline]
    fn rec_started(&mut self) -> bool {
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// A trace exited inside an inlined call frame: shift the base to
    /// the innermost frame (its slots — including the function and the
    /// frame link — were restored from the snapshot) and reload the
    /// interpreter for that frame's closure.
    #[cold]
    fn trace_exit_frame(&mut self, baseslot: usize) {
        self.base += baseslot - 2;
        let bp = unsafe { self.sp.add(self.base) };
        self.reload_at(bp);
        self.l().top = self.base + self.proto().framesize as usize;
    }

    /// `lj_gc_check` + `lj_gc_step_fixtop`: run a collection if the
    /// allocation debt is due. Only called from safe points (before an
    /// allocating opcode, with locals synced): the marker sees every live
    /// object through the stacks and roots. Fixes `l.top` up to the running
    /// frame's full extent first, since returns may have lowered it.
    #[inline]
    fn gc_check(&mut self) {
        if self.l().global().heap.should_collect() {
            self.gc_collect();
        }
    }

    #[cold]
    fn gc_collect(&mut self) {
        let need = self.base + self.proto().framesize as usize;
        let l = self.l();
        if l.top < need {
            l.top = need;
        }
        crate::gc::full_gc(l.global());
    }

    #[cold]
    fn len_op(&self, v: LuaValue) -> LuaResult<LuaValue> {
        if let Some(sid) = v.as_string_id() {
            let n = self.l().heap().strings.get(sid).len();
            return Ok(LuaValue::number(n as f64));
        }
        if let Some(t) = v.as_table() {
            return Ok(LuaValue::number(t.as_ref().len() as f64));
        }
        Err(self
            .l()
            .runtime_error(b"attempt to get length of a non-table value"))
    }

    #[cold]
    fn tsetm(&mut self, a: u32, d: u32) -> LuaResult<()> {
        let t = self.at(self.base + a as usize - 1);
        let base_key = unsafe { *self.knp.add(d as usize) } as i64 - (1i64 << 52);
        let tab = match t.as_table() {
            Some(t) => t,
            None => {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to index a non-table value"));
            }
        };
        // Pre-size array: we know the keys are base_key .. base_key+multres-1.
        // For a fresh table, this avoids per-value hash insertions.
        if self.multres > 0 && base_key == 1 {
            let need = (base_key as u32).wrapping_add(self.multres as u32);
            tab.as_mut().reasize(need);
        }
        for i in 0..self.multres {
            let key = base_key + i as i64;
            let v = self.at(self.base + a as usize + i);
            if key >= 0 && key <= i32::MAX as i64 {
                tab.as_mut().set_int(key as i32, v);
            } else {
                tab.as_mut().set(LuaValue::number(key as f64), v);
            }
        }
        Ok(())
    }

    // -- Calls -----------------------------------------------------------

    fn do_call(&mut self, a: u32, nargs: usize, want: i32) -> LuaResult<()> {
        let func_slot = self.base + a as usize;
        let mut nargs = nargs;
        let f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                // lj_meta_call: inject __call metamethod.
                nargs = meta::meta_call(self.l(), func_slot, nargs)?;
                let f = self.at(func_slot);
                f.as_func().expect("__call did not produce a function")
            }
        };
        match gf.as_ref() {
            GcFunc::Lua(_) => {
                // Frame link = the return PC; `want` is re-read from the
                // CALL/CALLM/ITERC instruction at `pc[-1]` on return.
                let link = unsafe { self.bcp.add(self.pc) } as u64;
                debug_assert!(link & FRAME_TYPE_MASK == FRAME_LUA);
                self.enter_lua(gf, func_slot, nargs, link);
                Ok(())
            }
            GcFunc::C(cc) => {
                let f = cc.f;
                let n = match self.call_c_inline(f, func_slot, nargs) {
                    Ok(n) => n,
                    Err(LuaError::Yield) => return Err(self.suspend_call(func_slot, want)),
                    Err(e) => return Err(e),
                };
                if want >= 0 {
                    for i in n..(want as usize) {
                        self.set_at(func_slot + i, LuaValue::NIL);
                    }
                } else {
                    self.multres = n;
                }
                Ok(())
            }
        }
    }

    /// A C function called from a Lua frame yielded (`coroutine.yield`):
    /// capture the resume point. Yield values move to `func_slot`.
    #[cold]
    fn suspend_call(&mut self, func_slot: usize, want: i32) -> LuaError {
        let ny = self.l().nyield as usize;
        for i in 0..ny {
            let v = self.at(func_slot + 2 + i);
            self.set_at(func_slot + i, v);
        }
        let l = self.l();
        l.suspend = Suspend::Call {
            pc: self.pc,
            cl: self.cl,
            base: self.base,
            slot: func_slot,
            want,
        };
        l.top = (self.base + self.proto().framesize as usize).max(func_slot + ny);
        l.base = self.base;
        LuaError::Yield
    }

    /// Same for a yield through a tail call (`return coroutine.yield(...)`).
    #[cold]
    fn suspend_return(&mut self, func_slot: usize) -> LuaError {
        let ny = self.l().nyield as usize;
        for i in 0..ny {
            let v = self.at(func_slot + 2 + i);
            self.set_at(func_slot + i, v);
        }
        let l = self.l();
        l.suspend = Suspend::Return { base: self.base, slot: func_slot };
        l.top = (self.base + self.proto().framesize as usize).max(func_slot + ny);
        l.base = self.base;
        LuaError::Yield
    }

    fn call_c_inline(
        &mut self,
        f: crate::func::CFunction,
        func_slot: usize,
        nargs: usize,
    ) -> LuaResult<usize> {
        let args_base = func_slot + 2;
        let l = self.l();
        let saved_base = l.base;
        let saved_top = l.top;
        l.base = args_base;
        l.top = args_base + nargs;
        // C-call boundary is a GC safe point (args anchored, frames below).
        if l.global().heap.should_collect() {
            crate::gc::full_gc(l.global());
        }
        let r = f(l);
        let n = match r {
            Ok(n) => n as usize,
            Err(e) => {
                l.base = saved_base;
                l.top = saved_top;
                return Err(e);
            }
        };
        for i in 0..n {
            l.stack[func_slot + i] = l.stack[args_base + i];
        }
        l.base = saved_base;
        l.top = saved_top;
        Ok(n)
    }

    /// True tail call, per LuaJIT's BC_CALLT: the callee *replaces* the
    /// current frame. Func and args move down to this frame's slots, the
    /// frame link stays untouched, and no Rust recursion happens, so tail
    /// recursion runs in constant stack. A tail call from a vararg function
    /// first drops its vararg frame (relocating to the frame that holds the
    /// real link). Returns `Some(n)` only when a C callee finishes a host
    /// (`FRAME_C`) frame.
    fn do_tailcall(&mut self, a: u32, nargs: usize) -> LuaResult<Option<usize>> {
        let func_slot = self.base + a as usize;
        let mut nargs = nargs;
        let mut f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                nargs = meta::meta_call(self.l(), func_slot, nargs)?;
                f = self.at(func_slot);
                f.as_func().expect("__call did not produce a function")
            }
        };

        // C callee: call inline (no c_depth bump) so yields propagate.
        if let GcFunc::C(cc) = gf.as_ref() {
            let r = call_c(self.l(), cc.f, func_slot, nargs, -1);
            return match r {
                Ok(n) => Ok(self.do_return(func_slot, n)),
                Err(LuaError::Yield) => Err(self.suspend_return(func_slot)),
                Err(e) => Err(e),
            };
        }

        let mut base = self.base;
        let link = self.at(base - 1).to_bits();
        if link & FRAME_TYPE_MASK == FRAME_VARG {
            let delta = (link >> 3) as usize;
            if base >= delta + 2 {
                base -= delta;
            } else {
                // Underflow: fall back.
                let n = execute(self.l(), func_slot, nargs, -1)?;
                return Ok(self.do_return(func_slot, n));
            }
        }
        // Move func and args down into the (possibly relocated) frame,
        // in reverse to avoid overlapping corruption.
        for i in (0..nargs).rev() {
            self.set_at(base + i, self.at(func_slot + 2 + i));
        }
        self.set_at(base - 2, f);

        match gf.as_ref() {
            GcFunc::Lua(cl) => {
                let pt = cl.proto.as_ref();
                if (pt.flags & PROTO_VARARG) != 0 {
                    let link = self.at(base - 1).to_bits();
                    self.enter_lua(gf, base - 2, nargs, link);
                } else {
                    for i in nargs..pt.numparams as usize {
                        self.set_at(base + i, LuaValue::NIL);
                    }
                    self.base = base;
                    self.cl = gf;
                    self.bcp = pt.bc.as_ptr();
                    self.knp = pt.kn.as_ptr();
                    self.ksp = pt.kstrv.as_ptr();
                    self.pc = 1;
                    self.l().top = base + pt.framesize as usize;
                }
                Ok(None)
            }
            GcFunc::C(_cc) => unreachable!("C path handled above"),
        }
    }

    /// Move `n` results to the caller's slot, restore the caller frame and
    /// continue. Everything is recovered from the frame links in the stack,
    /// as in LuaJIT's BC_RET: the caller base, wanted results and result
    /// slot all come from the CALL instruction at the stored return PC.
    /// Returns `Some(n)` when a host (`FRAME_C`) entry returns.
    fn do_return(&mut self, src: usize, n: usize) -> Option<usize> {
        if !self.l().openuv.is_empty() {
            self.close_upvals(self.base);
        }
        let mut base = self.base;
        let mut link = self.at(base - 1).to_bits();
        while link & FRAME_TYPE_MASK == FRAME_VARG {
            base -= (link >> 3) as usize;
            link = self.at(base - 1).to_bits();
        }
        let dst = base - 2; // results always land at the callee's func slot
        for i in 0..n {
            self.set_at(dst + i, self.at(src + i));
        }
        self.multres = n;

        if link & FRAME_TYPE_MASK == FRAME_CONT {
            return self.cont_dispatch(base, link, n);
        }

        if link & FRAME_TYPE_MASK == FRAME_C {
            let want = ((link >> 3) as i32) - 1;
            let got = if want >= 0 {
                for i in n..(want as usize) {
                    self.set_at(dst + i, LuaValue::NIL);
                }
                want as usize
            } else {
                n
            };
            self.l().top = dst + got;
            return Some(got);
        }

        // FRAME_LUA: the link is the return PC.
        let ret_ip = link as *const BCIns;
        let call_ins = unsafe { *ret_ip.sub(1) };
        let caller_base = dst - bc_a(call_ins) as usize;
        let want = bc_b(call_ins) as i32 - 1;

        self.base = caller_base;
        let cl = self.at(caller_base - 2).as_func().unwrap();
        self.reload(cl);
        self.pc = unsafe { ret_ip.offset_from(self.bcp) as usize };

        if want >= 0 {
            for i in n..(want as usize) {
                self.set_at(dst + i, LuaValue::NIL);
            }
            self.l().top = dst + want as usize;
        } else {
            self.l().top = dst + n;
        }
        None
    }

    // -- Generic for -----------------------------------------------------

    fn iter_call(&mut self, a: u32, nret: usize) -> LuaResult<()> {
        let fs = self.base + a as usize;
        let genf = self.at(fs - 3);
        let state = self.at(fs - 2);
        let ctl = self.at(fs - 1);
        self.set_at(fs, genf);
        self.set_at(fs + 2, state);
        self.set_at(fs + 3, ctl);
        self.do_call(a, 2, nret as i32 - 1)
    }

    // -- Varargs ---------------------------------------------------------

    /// Copy varargs into `dst..`, per BC_VARG. The varargs sit between the
    /// vararg frame and the frame below it (see `enter_lua`); their extent
    /// is recovered from the FRAME_VARG delta, LuaJIT-style.
    fn vararg(&mut self, a: u32, b: u32) {
        let base = self.base;
        let link = self.at(base - 1).to_bits();
        debug_assert!(link & FRAME_TYPE_MASK == FRAME_VARG);
        let delta = (link >> 3) as usize;
        let numparams = self.proto().numparams as usize;
        let varg_base = base - delta + numparams;
        let nvarg = (delta - 2).saturating_sub(numparams);

        let dst = base + a as usize;
        if b == 0 {
            for i in 0..nvarg {
                self.set_at(dst + i, self.at(varg_base + i));
            }
            self.multres = nvarg;
            self.l().top = dst + nvarg;
        } else {
            let want = (b - 1) as usize;
            for i in 0..want {
                let v = if i < nvarg {
                    self.at(varg_base + i)
                } else {
                    LuaValue::NIL
                };
                self.set_at(dst + i, v);
            }
        }
    }

    // -- Upvalues / closures ---------------------------------------------

    fn upval_get(&self, uv: GcPtr<Upval>) -> LuaValue {
        uv.as_ref().get()
    }

    fn upval_set(&self, uv: GcPtr<Upval>, v: LuaValue) {
        uv.as_mut().set(v);
    }

    /// Find or create an open upvalue for the stack slot `slot` (absolute
    /// index into this thread's stack). Identity is by value pointer,
    /// exactly like `lj_func_finduv`.
    fn find_upval(&mut self, slot: usize) -> GcPtr<Upval> {
        let ptr = unsafe { self.sp.add(slot) };
        for &uv in self.l().openuv.iter() {
            if uv.as_ref().value_ptr() == ptr {
                return uv;
            }
        }
        let nn = std::ptr::NonNull::new(ptr).unwrap();
        let uv = self
            .l()
            .heap()
            .alloc_upval(Upval::new_open(nn, false));
        self.l().openuv.push(uv);
        uv
    }

    /// Close every open upvalue at or above stack `level` (absolute index),
    /// per `lj_func_closeuv`.
    fn close_upvals(&mut self, level: usize) {
        let level_ptr = unsafe { self.sp.add(level) } as *const LuaValue;
        let l = self.l();
        let mut i = 0;
        while i < l.openuv.len() {
            let uv = l.openuv[i];
            if uv.as_ref().value_ptr() as *const LuaValue >= level_ptr {
                uv.as_mut().close();
                l.openuv.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn new_closure(&mut self, d: u32) -> LuaValue {
        let proto = match &self.proto().kgc[d as usize] {
            KGc::ProtoRef(p) => *p,
            _ => unreachable!("FNEW expects a registered child prototype"),
        };
        let pt = proto.as_ref();
        let nuv = pt.uv.len();
        let mut upvals = Vec::with_capacity(nuv);
        let parent_upvals: Vec<GcPtr<Upval>> = self.lua_cl().upvals.clone();
        for i in 0..nuv {
            let v = pt.uv[i];
            if (v & PROTO_UV_LOCAL) != 0 {
                let slot = self.base + (v & 0xff) as usize;
                let uv = self.find_upval(slot);
                if (v & PROTO_UV_IMMUTABLE) != 0 {
                    uv.as_mut().immutable = true;
                }
                upvals.push(uv);
            } else {
                upvals.push(parent_upvals[(v & 0xff) as usize]);
            }
        }
        let env = self.lua_cl().env;
        let fref = self
            .l()
            .heap()
            .alloc_func(GcFunc::Lua(LuaClosure { proto, env, upvals }));
        LuaValue::func(fref)
    }
}

/// The three primitive values, indexed by KPRI/ISEQP operand (0/1/2).
const PRI: [LuaValue; 3] = [LuaValue::NIL, LuaValue::FALSE, LuaValue::TRUE];

/// Raw equality used by ISEQ*/ISNE*: numbers compare by value (so `-0.0` and
/// NaN behave), everything else by bit pattern (interned strings and GC
/// pointers compare by identity).
#[inline(always)]
fn val_eq(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else {
        a.to_bits() == b.to_bits()
    }
}

/// `x ^ y` with a small-integer-exponent fast path (`lj_vm_powi`).
#[inline]
pub(crate) fn vm_pow(mut x: f64, y: f64) -> f64 {
    let k = y as i32;
    if k as f64 == y && k.unsigned_abs() <= 65536 {
        if k >= 1 {
            let mut n = k as u32;
            while n & 1 == 0 { x *= x; n >>= 1; }
            let mut z = x; n >>= 1;
            while n != 0 { x *= x; if n & 1 != 0 { z *= x; } n >>= 1; }
            z
        } else if k == 0 {
            1.0
        } else {
            1.0 / x.powf(-k as f64)
        }
    } else {
        x.powf(y)
    }
}

/// Resume a coroutine suspended via `Suspend::Call`. Rebuilds the Interp
/// from the saved state and re-enters the dispatch loop.
pub fn resume_continue(
    co: &mut LuaState,
    slot: usize,
    want: i32,
    nargs: usize,
    pc: usize,
    cl: GcPtr<GcFunc>,
    sbase: usize,
) -> LuaResult<usize> {
    co.c_depth += 1;
    // Note: no stack_ensure needed — the suspended frame was alive at
    // yield time and stack length never shrinks.
    if want >= 0 {
        let limit = nargs.min(want as usize);
        for i in 0..limit { co.stack[slot + i] = co.stack[slot + 2 + i]; }
        for i in limit..(want as usize) { co.stack[slot + i] = LuaValue::NIL; }
    } else {
        for i in 0..nargs { co.stack[slot + i] = co.stack[slot + 2 + i]; }
    }
    let mut vm = Interp::new(co);
    vm.base = sbase;
    vm.cl = cl;
    vm.reload(cl);
    vm.pc = pc;
    let pt = match cl.as_ref() {
        GcFunc::Lua(c) => c.proto.as_ref(),
        _ => unreachable!(),
    };
    co.top = sbase + pt.framesize as usize;
    co.base = sbase;
    co.status = crate::state::CoStatus::Running;
    let r = vm.run();
    co.c_depth -= 1;
    r
}

/// Finish a coroutine suspended via `Suspend::Return`. Delivers resume
/// args as a return from the saved frame, like `do_return`; if the return
/// lands in a Lua frame, the dispatch loop continues running.
pub fn resume_finish(co: &mut LuaState, slot: usize, nargs: usize, sbase: usize) -> LuaResult<usize> {
    co.c_depth += 1;
    let link = co.stack[slot + 1].to_bits();
    if link & FRAME_TYPE_MASK != FRAME_C && link & FRAME_TYPE_MASK != FRAME_LUA {
        co.stack[slot + 1] = LuaValue::from_bits(FRAME_C);
    }
    for i in 0..nargs {
        co.stack[slot + i] = co.stack[slot + 2 + i];
    }
    let mut vm = Interp::new(co);
    vm.base = sbase;
    co.status = crate::state::CoStatus::Running;
    let r = match vm.do_return(slot, nargs) {
        Some(n) => Ok(n),
        None => vm.run(),
    };
    co.c_depth -= 1;
    r
}
