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

pub mod err;
use crate::err::{LuaError, LuaResult};
use crate::func::{GcFunc, LuaClosure, Upval, UpvalState};
use crate::gc::GcPtr;
use crate::proto::{KGc, PROTO_UV_IMMUTABLE, PROTO_UV_LOCAL, PROTO_VARARG, Proto};
use crate::state::LuaState;
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
const FRAME_VARG: u64 = 3;
const FRAME_TYPE_MASK: u64 = 3;

/// Call a value with the given arguments and collect all results.
/// The host entry point into the VM.
pub fn call(l: &mut LuaState, func: LuaValue, args: &[LuaValue]) -> LuaResult<Vec<LuaValue>> {
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
    let f = l.stack[func_slot];
    let gf = match f.as_func() {
        Some(p) => p,
        None => return Err(l.runtime_error(b"attempt to call a non-function value")),
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
    let n = f(l)? as usize;
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

    /// The dispatch loop. The entire hot state is two locals — `bp` (the
    /// current base as a pointer, LuaJIT's BASE) and `ip` (a walking
    /// instruction pointer, LuaJIT's PC) — so both live in registers on
    /// every dispatch. Everything else (`self.knp`, `self.multres`, ...)
    /// is re-read from `self` in the arms that need it; keeping more locals
    /// alive across the dispatch forces spills (measured: rustc packs the
    /// extras into an XMM register and unpacks per instruction).
    /// `sync!`/`resync!` bridge to the fields around calls and returns.
    fn run(&mut self) -> LuaResult<usize> {
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
        // Numeric binary op fast path; falls to the arithmetic slow error.
        macro_rules! arith {
            ($a:expr, $xv:expr, $yv:expr, $x:ident, $y:ident, $body:expr) => {{
                let xv = $xv;
                let yv = $yv;
                if xv.is_number() && yv.is_number() {
                    let $x = xv.num();
                    let $y = yv.num();
                    setreg!($a, LuaValue::number_raw($body));
                } else {
                    sync!();
                    return Err(self.arith_err());
                }
            }};
        }
        // reg `op` constant (*VN) / constant `op` reg (*NV): number constants
        // need no type check, only the register operand does.
        macro_rules! arith_k {
            ($a:expr, $ins:expr, $x:ident, $y:ident, $body:expr) => {{
                let xv = reg!(bc_b($ins));
                if xv.is_number() {
                    let $x = xv.num();
                    let $y = unsafe { *self.knp.add(bc_c($ins) as usize) };
                    setreg!($a, LuaValue::number_raw($body));
                } else {
                    sync!();
                    return Err(self.arith_err());
                }
            }};
        }
        // Comparison + fused following JMP.
        macro_rules! cmp {
            ($op:expr, $xv:expr, $yv:expr, $x:ident, $y:ident, $fast:expr) => {{
                let xv = $xv;
                let yv = $yv;
                let cond = if xv.is_number() && yv.is_number() {
                    let $x = xv.num();
                    let $y = yv.num();
                    $fast
                } else {
                    sync!();
                    self.cmp_slow($op, xv, yv)?
                };
                let jmp = unsafe { *ip };
                ip = unsafe { ip.add(1) };
                if cond {
                    jump!(jmp);
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

        loop {
            let ins = unsafe { *ip };
            ip = unsafe { ip.add(1) };
            let a = bc_a(ins);
            match bc_op(ins) {
                // -- Comparisons (ORDER matters; see bc.rs) --
                BCOp::ISLT => cmp!(BCOp::ISLT, reg!(a), reg!(bc_d(ins)), x, y, x < y),
                BCOp::ISGE => cmp!(BCOp::ISGE, reg!(a), reg!(bc_d(ins)), x, y, x >= y),
                BCOp::ISLE => cmp!(BCOp::ISLE, reg!(a), reg!(bc_d(ins)), x, y, x <= y),
                BCOp::ISGT => cmp!(BCOp::ISGT, reg!(a), reg!(bc_d(ins)), x, y, x > y),
                BCOp::ISEQV => {
                    let cond = val_eq(reg!(a), reg!(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISNEV => {
                    let cond = !val_eq(reg!(a), reg!(bc_d(ins)));
                    branch!(cond);
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
                        return Err(self.arith_err());
                    }
                }
                BCOp::LEN => {
                    let v = reg!(bc_d(ins));
                    sync!();
                    let r = self.len_op(v)?;
                    setreg!(a, r);
                }

                // -- Arithmetic --
                BCOp::ADDVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x + y),
                BCOp::SUBVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x - y),
                BCOp::MULVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x * y),
                BCOp::DIVVV => arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x / y),
                BCOp::MODVV => {
                    arith!(
                        a,
                        reg!(bc_b(ins)),
                        reg!(bc_c(ins)),
                        x,
                        y,
                        x - (x / y).floor() * y
                    )
                }
                BCOp::ADDVN => arith_k!(a, ins, x, y, x + y),
                BCOp::SUBVN => arith_k!(a, ins, x, y, x - y),
                BCOp::MULVN => arith_k!(a, ins, x, y, x * y),
                BCOp::DIVVN => arith_k!(a, ins, x, y, x / y),
                BCOp::MODVN => arith_k!(a, ins, x, y, x - (x / y).floor() * y),
                BCOp::ADDNV => arith_k!(a, ins, y, x, x + y),
                BCOp::SUBNV => arith_k!(a, ins, y, x, x - y),
                BCOp::MULNV => arith_k!(a, ins, y, x, x * y),
                BCOp::DIVNV => arith_k!(a, ins, y, x, x / y),
                BCOp::MODNV => arith_k!(a, ins, y, x, x - (x / y).floor() * y),
                BCOp::POW => {
                    arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x.powf(y))
                }
                BCOp::CAT => {
                    sync!();
                    self.gc_check();
                    let r = self.concat(bc_b(ins), bc_c(ins))?;
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
                    sync!();
                    self.gc_check();
                    let t = self.l().heap().alloc_table(LuaTable::new(0, 0));
                    setreg!(a, LuaValue::table(t));
                }
                BCOp::TDUP => {
                    sync!();
                    self.gc_check();
                    let templ = match &self.proto().kgc[bc_d(ins) as usize] {
                        KGc::Table(t) => t.dup(),
                        _ => unreachable!("expected template table"),
                    };
                    let t = self.l().heap().alloc_table(templ);
                    setreg!(a, LuaValue::table(t));
                }
                BCOp::GGET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    setreg!(a, env.as_ref().get_str(key));
                }
                BCOp::GSET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    env.as_mut().set_str(key, reg!(a));
                }
                BCOp::TGETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    if let Some(tab) = t.as_table() {
                        if k.is_string() {
                            setreg!(a, tab.as_ref().get_str(k));
                        } else if k.is_number() {
                            let ki = k.num() as i32;
                            if ki as f64 == k.num() && ki >= 0 {
                                setreg!(a, tab.as_ref().get_int(ki));
                            } else {
                                setreg!(a, tab.as_ref().get(k));
                            }
                        } else {
                            setreg!(a, tab.as_ref().get(k));
                        }
                    } else {
                        sync!();
                        let v = self.index_get(t, k)?;
                        setreg!(a, v);
                    }
                }
                BCOp::TGETS => {
                    let t = reg!(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = self.kstr_at(bc_c(ins));
                        setreg!(a, tab.as_ref().get_str(k));
                    } else {
                        sync!();
                        let v = self.index_get(t, self.kstr_at(bc_c(ins)))?;
                        setreg!(a, v);
                    }
                }
                BCOp::TGETB => {
                    let t = reg!(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = bc_c(ins) as i32;
                        let v = tab.as_ref().get_int(k);
                        setreg!(a, v);
                    } else {
                        sync!();
                        let v = self.index_get(t, LuaValue::number(bc_c(ins) as f64))?;
                        setreg!(a, v);
                    }
                }
                BCOp::TSETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table() {
                        if k.is_string() {
                            tab.as_mut().set_str(k, v);
                        } else if k.is_number() {
                            let ki = k.num() as i32;
                            if ki as f64 == k.num() && ki >= 0 {
                                tab.as_mut().set_int(ki, v);
                            } else {
                                tab.as_mut().set(k, v);
                            }
                        } else {
                            tab.as_mut().set(k, v);
                        }
                    } else {
                        sync!();
                        self.index_set(t, k, v)?;
                    }
                }
                BCOp::TSETS => {
                    let t = reg!(bc_b(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table() {
                        let k = self.kstr_at(bc_c(ins));
                        tab.as_mut().set_str(k, v);
                    } else {
                        sync!();
                        self.index_set(t, self.kstr_at(bc_c(ins)), v)?;
                    }
                }
                BCOp::TSETB => {
                    let t = reg!(bc_b(ins));
                    let v = reg!(a);
                    if let Some(tab) = t.as_table() {
                        tab.as_mut().set_int(bc_c(ins) as i32, v);
                    } else {
                        sync!();
                        self.index_set(t, LuaValue::number(bc_c(ins) as f64), v)?;
                    }
                }
                BCOp::TSETM => {
                    sync!();
                    self.tsetm(a, bc_d(ins))?;
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
                        let pt = cl.proto.as_ref();
                        if (pt.flags & PROTO_VARARG) == 0 {
                            let nargs = bc_c(ins) as usize - 1;
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
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::CALLMT => {
                    let nargs = bc_d(ins) as usize + self.multres;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs)? {
                        return Ok(n);
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
                        return Ok(n);
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
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RET => {
                    let n = bc_d(ins) as usize - 1;
                    sync!();
                    if let Some(n) = self.do_return(cur_base!() + a as usize, n) {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RETM => {
                    let n = self.multres + bc_d(ins) as usize;
                    sync!();
                    if let Some(n) = self.do_return(cur_base!() + a as usize, n) {
                        return Ok(n);
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
                BCOp::FORL => {
                    let i = reg!(a + FORL_IDX).num();
                    let s = reg!(a + FORL_STOP).num();
                    let st = reg!(a + FORL_STEP).num();
                    let ni = i + st;
                    let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                    if cont {
                        let nv = LuaValue::number_raw(ni);
                        setreg!(a + FORL_IDX, nv);
                        setreg!(a + FORL_EXT, nv);
                        jump!(ins);
                    }
                }
                BCOp::LOOP => { /* hotcount hook: no-op for now */ }
                BCOp::JMP => jump!(ins),
                BCOp::ISNEXT => jump!(ins),
                BCOp::ITERC | BCOp::ITERN => {
                    sync!();
                    self.iter_call(a, bc_b(ins) as usize)?;
                    resync!();
                }
                BCOp::ITERL => {
                    let first = reg!(a);
                    if !first.is_nil() {
                        setreg!(a - 1, first);
                        jump!(ins);
                    }
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
                | BCOp::JFORI
                | BCOp::IFORL
                | BCOp::JFORL
                | BCOp::IITERL
                | BCOp::JITERL
                | BCOp::ILOOP
                | BCOp::JLOOP
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
    fn arith_err(&self) -> LuaError {
        self.l()
            .runtime_error(b"attempt to perform arithmetic on a non-number value")
    }

    #[cold]
    fn cmp_slow(&self, op: BCOp, x: LuaValue, y: LuaValue) -> LuaResult<bool> {
        if let (Some(a), Some(b)) = (self.as_bytes(x), self.as_bytes(y)) {
            return Ok(match op {
                BCOp::ISLT => a < b,
                BCOp::ISGE => a >= b,
                BCOp::ISLE => a <= b,
                BCOp::ISGT => a > b,
                _ => unreachable!(),
            });
        }
        Err(self
            .l()
            .runtime_error(b"attempt to compare incompatible values"))
    }

    fn as_bytes(&self, v: LuaValue) -> Option<Vec<u8>> {
        v.as_string_id()
            .map(|sid| self.l().heap().strings.get(sid).to_vec())
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
    fn concat(&mut self, from: u32, to: u32) -> LuaResult<LuaValue> {
        let mut buf = Vec::new();
        for i in from..=to {
            let v = self.at(self.base + i as usize);
            if let Some(sid) = v.as_string_id() {
                buf.extend_from_slice(self.l().heap().strings.get(sid));
            } else if let Some(n) = v.as_number() {
                buf.extend_from_slice(crate::strfmt::g14(n).as_bytes());
            } else {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to concatenate a non-string value"));
            }
        }
        let sid = self.l().heap().intern(&buf);
        Ok(self.l().heap().str_value(sid))
    }

    fn index_get(&self, t: LuaValue, k: LuaValue) -> LuaResult<LuaValue> {
        match t.as_table() {
            Some(tab) => Ok(tab.as_ref().get(k)),
            None => Err(self
                .l()
                .runtime_error(b"attempt to index a non-table value")),
        }
    }

    fn index_set(&self, t: LuaValue, k: LuaValue, v: LuaValue) -> LuaResult<()> {
        match t.as_table() {
            Some(tab) => {
                if k.is_nil() {
                    return Err(self.l().runtime_error(b"table index is nil"));
                }
                tab.as_mut().set(k, v);
                Ok(())
            }
            None => Err(self
                .l()
                .runtime_error(b"attempt to index a non-table value")),
        }
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
        let f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to call a non-function value"));
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
                let n = self.call_c_inline(f, func_slot, nargs)?;
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
        let n = f(l)? as usize;
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
        let f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to call a non-function value"));
            }
        };

        let mut base = self.base;
        let link = self.at(base - 1).to_bits();
        if link & FRAME_TYPE_MASK == FRAME_VARG {
            let delta = (link >> 3) as usize;
            if base >= delta + 2 {
                base -= delta;
            }
            // Else: base would underflow — the caller is so shallow that
            // dropping the vararg frame puts us past the stack origin.
            // Fall back to a regular (recursive) call: stack frame reuse
            // for a vararg tail call is incorrect when the delta pushes
            // past the stack origin.
            else if let GcFunc::C(_cc) = gf.as_ref() {
                let n = execute(self.l(), func_slot, nargs, -1)?;
                return Ok(self.do_return(func_slot, n));
            }
        }
        // Move func and args down into the (possibly relocated) frame.
        // Must copy args in reverse order: when func_slot + 2 > base,
        // the ranges overlap and a forward copy corrupts later arguments.
        self.set_at(base - 2, f);
        for i in (0..nargs).rev() {
            self.set_at(base + i, self.at(func_slot + 2 + i));
        }

        match gf.as_ref() {
            GcFunc::Lua(cl) => {
                let pt = cl.proto.as_ref();
                if (pt.flags & PROTO_VARARG) != 0 {
                    // FUNCV builds its own vararg frame on top, chained to
                    // the link already sitting at `base - 1`.
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
                    self.pc = 1; // skip the FUNCF header
                    self.l().top = base + pt.framesize as usize;
                }
                Ok(None)
            }
            GcFunc::C(_cc) => {
                // Tail call to a C function: run it as a regular
                // execute and return. True TCO for C callees is a
                // much narrower win (they're leaf calls) and the
                // frame-reuse path has subtle edge cases.
                let n = execute(self.l(), func_slot, nargs, -1)?;
                return Ok(self.do_return(func_slot, n));
            }
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
        match &uv.as_ref().state {
            UpvalState::Open(slot) => self.at(*slot),
            UpvalState::Closed(v) => *v,
        }
    }

    fn upval_set(&self, uv: GcPtr<Upval>, v: LuaValue) {
        match &mut uv.as_mut().state {
            UpvalState::Open(slot) => self.set_at(*slot, v),
            UpvalState::Closed(cv) => *cv = v,
        }
    }

    fn find_upval(&mut self, slot: usize) -> GcPtr<Upval> {
        for &uv in self.l().openuv.iter() {
            if let UpvalState::Open(s) = uv.as_ref().state
                && s == slot
            {
                return uv;
            }
        }
        let uv = self.l().heap().alloc_upval(Upval::new_open(slot, false));
        self.l().openuv.push(uv);
        uv
    }

    fn close_upvals(&mut self, level: usize) {
        let l = self.l();
        let mut i = 0;
        while i < l.openuv.len() {
            let uv = l.openuv[i];
            let close = match uv.as_ref().state {
                UpvalState::Open(s) => s >= level,
                UpvalState::Closed(_) => true,
            };
            if close {
                if let UpvalState::Open(s) = uv.as_ref().state {
                    let v = self.at(s);
                    uv.as_mut().state = UpvalState::Closed(v);
                }
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
