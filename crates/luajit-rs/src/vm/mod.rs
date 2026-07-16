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
use crate::proto::{KGc, Proto, PROTO_UV_IMMUTABLE, PROTO_UV_LOCAL, PROTO_VARARG};
use crate::state::{Frame, LuaState};
use crate::table::LuaTable;
use crate::value::*;

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
    let frame_floor = vm.l().frames.len();
    vm.enter_lua(gf, func_slot, nargs, want);
    vm.run(frame_floor)
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
    varg_base: usize,
    nvarg: usize,
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
            varg_base: 0,
            nvarg: 0,
            multres: 0,
        }
    }

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

    fn kstr_at(&self, d: u32) -> LuaValue {
        match &self.proto().kgc[d as usize] {
            KGc::Str(sid) => self.l().heap().str_value(*sid),
            _ => unreachable!("expected string constant"),
        }
    }

    /// Set up a Lua frame for the function at `func_slot` and switch the
    /// `Interp` fields to the callee. The caller must have synced its locals
    /// into the fields first.
    fn enter_lua(&mut self, gf: GcPtr<GcFunc>, func_slot: usize, nargs: usize, want: i32) {
        if !self.bcp.is_null() {
            self.l().frames.push(Frame {
                base: self.base,
                result_slot: func_slot,
                return_pc: self.pc,
                func: LuaValue::func(self.cl),
                nresults: want,
                varg_base: self.varg_base,
                nvarg: self.nvarg,
            });
        } else {
            self.l().frames.push(Frame {
                base: usize::MAX,
                result_slot: func_slot,
                return_pc: usize::MAX,
                func: LuaValue::NIL,
                nresults: want,
                varg_base: 0,
                nvarg: 0,
            });
        }

        let cl = match gf.as_ref() {
            GcFunc::Lua(c) => c,
            _ => unreachable!(),
        };
        let pt = cl.proto.as_ref();
        let numparams = pt.numparams as usize;
        let callbase = func_slot + 2;

        if (pt.flags & PROTO_VARARG) != 0 {
            let newbase = callbase + nargs + 2;
            self.set_at(newbase - 2, LuaValue::func(gf));
            for i in 0..numparams {
                let v = if i < nargs { self.at(callbase + i) } else { LuaValue::NIL };
                self.set_at(newbase + i, v);
            }
            self.varg_base = callbase + numparams;
            self.nvarg = nargs.saturating_sub(numparams);
            self.base = newbase;
        } else {
            for i in nargs..numparams {
                self.set_at(callbase + i, LuaValue::NIL);
            }
            self.varg_base = 0;
            self.nvarg = 0;
            self.base = callbase;
        }

        self.cl = gf;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
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
    }

    /// The dispatch loop. Hot state (pc/base/sp/bcp/knp/multres) is kept in
    /// locals; `sync!`/`resync!` bridge to the fields around cold operations.
    fn run(&mut self, frame_floor: usize) -> LuaResult<usize> {
        let sp = self.sp;
        let mut base = self.base;
        let mut pc = self.pc;
        let mut bcp = self.bcp;
        let mut knp = self.knp;
        let mut multres = self.multres;

        macro_rules! reg {
            ($i:expr) => {
                unsafe { *sp.add(base + ($i) as usize) }
            };
        }
        macro_rules! setreg {
            ($i:expr, $v:expr) => {
                unsafe { *sp.add(base + ($i) as usize) = $v }
            };
        }
        macro_rules! kslot {
            ($d:expr) => {
                LuaValue::number_raw(unsafe { *knp.add(($d) as usize) })
            };
        }
        macro_rules! sync {
            () => {{
                self.base = base;
                self.pc = pc;
                self.multres = multres;
            }};
        }
        macro_rules! resync {
            () => {{
                base = self.base;
                pc = self.pc;
                bcp = self.bcp;
                knp = self.knp;
                multres = self.multres;
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
                let jmp = unsafe { *bcp.add(pc) };
                pc += 1;
                if cond {
                    pc = (pc as i64 + bc_j(jmp)) as usize;
                }
            }};
        }
        macro_rules! branch {
            ($cond:expr) => {{
                let jmp = unsafe { *bcp.add(pc) };
                pc += 1;
                if $cond {
                    pc = (pc as i64 + bc_j(jmp)) as usize;
                }
            }};
        }

        loop {
            let ins = unsafe { *bcp.add(pc) };
            pc += 1;
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
                    arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x - (x / y).floor() * y)
                }
                BCOp::ADDVN => arith!(a, reg!(bc_b(ins)), kslot!(bc_c(ins)), x, y, x + y),
                BCOp::SUBVN => arith!(a, reg!(bc_b(ins)), kslot!(bc_c(ins)), x, y, x - y),
                BCOp::MULVN => arith!(a, reg!(bc_b(ins)), kslot!(bc_c(ins)), x, y, x * y),
                BCOp::DIVVN => arith!(a, reg!(bc_b(ins)), kslot!(bc_c(ins)), x, y, x / y),
                BCOp::MODVN => {
                    arith!(a, reg!(bc_b(ins)), kslot!(bc_c(ins)), x, y, x - (x / y).floor() * y)
                }
                BCOp::ADDNV => arith!(a, kslot!(bc_c(ins)), reg!(bc_b(ins)), x, y, x + y),
                BCOp::SUBNV => arith!(a, kslot!(bc_c(ins)), reg!(bc_b(ins)), x, y, x - y),
                BCOp::MULNV => arith!(a, kslot!(bc_c(ins)), reg!(bc_b(ins)), x, y, x * y),
                BCOp::DIVNV => arith!(a, kslot!(bc_c(ins)), reg!(bc_b(ins)), x, y, x / y),
                BCOp::MODNV => {
                    arith!(a, kslot!(bc_c(ins)), reg!(bc_b(ins)), x, y, x - (x / y).floor() * y)
                }
                BCOp::POW => {
                    arith!(a, reg!(bc_b(ins)), reg!(bc_c(ins)), x, y, x.powf(y))
                }
                BCOp::CAT => {
                    sync!();
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
                    self.close_upvals(base + a as usize);
                    pc = (pc as i64 + bc_j(ins)) as usize;
                }
                BCOp::FNEW => {
                    sync!();
                    let v = self.new_closure(bc_d(ins));
                    setreg!(a, v);
                }

                // -- Tables --
                BCOp::TNEW => {
                    sync!();
                    let t = self.l().heap().alloc_table(LuaTable::new(0, 0));
                    setreg!(a, LuaValue::table(t));
                }
                BCOp::TDUP => {
                    sync!();
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
                    setreg!(a, env.as_ref().get(key));
                }
                BCOp::GSET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    env.as_mut().set(key, reg!(a));
                }
                BCOp::TGETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    sync!();
                    let v = self.index_get(t, k)?;
                    setreg!(a, v);
                }
                BCOp::TGETS => {
                    let t = reg!(bc_b(ins));
                    let k = self.kstr_at(bc_c(ins));
                    sync!();
                    let v = self.index_get(t, k)?;
                    setreg!(a, v);
                }
                BCOp::TGETB => {
                    let t = reg!(bc_b(ins));
                    let k = LuaValue::number(bc_c(ins) as f64);
                    sync!();
                    let v = self.index_get(t, k)?;
                    setreg!(a, v);
                }
                BCOp::TSETV => {
                    let t = reg!(bc_b(ins));
                    let k = reg!(bc_c(ins));
                    let v = reg!(a);
                    sync!();
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETS => {
                    let t = reg!(bc_b(ins));
                    let k = self.kstr_at(bc_c(ins));
                    let v = reg!(a);
                    sync!();
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETB => {
                    let t = reg!(bc_b(ins));
                    let k = LuaValue::number(bc_c(ins) as f64);
                    let v = reg!(a);
                    sync!();
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETM => {
                    sync!();
                    self.tsetm(a, bc_d(ins))?;
                }

                // -- Calls / returns --
                BCOp::CALL => {
                    let nargs = bc_c(ins) as usize - 1;
                    sync!();
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                    resync!();
                }
                BCOp::CALLM => {
                    let nargs = bc_c(ins) as usize + multres;
                    sync!();
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                    resync!();
                }
                BCOp::CALLT => {
                    let nargs = bc_d(ins) as usize - 1;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs, frame_floor)? {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::CALLMT => {
                    let nargs = bc_d(ins) as usize + multres;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs, frame_floor)? {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RET0 => {
                    sync!();
                    if let Some(n) = self.do_return(base, 0, frame_floor) {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RET1 => {
                    sync!();
                    if let Some(n) = self.do_return(base + a as usize, 1, frame_floor) {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RET => {
                    let n = bc_d(ins) as usize - 1;
                    sync!();
                    if let Some(n) = self.do_return(base + a as usize, n, frame_floor) {
                        return Ok(n);
                    }
                    resync!();
                }
                BCOp::RETM => {
                    let n = multres + bc_d(ins) as usize;
                    sync!();
                    if let Some(n) = self.do_return(base + a as usize, n, frame_floor) {
                        return Ok(n);
                    }
                    resync!();
                }

                // -- Loops and branches --
                BCOp::FORI => {
                    let ai = base + a as usize;
                    let idx = self.at(ai + FORL_IDX as usize);
                    let stop = self.at(ai + FORL_STOP as usize);
                    let step = self.at(ai + FORL_STEP as usize);
                    if idx.is_number() && stop.is_number() && step.is_number() {
                        let (i, s, st) = (idx.num(), stop.num(), step.num());
                        self.set_at(ai + FORL_EXT as usize, LuaValue::number_raw(i));
                        let enter = if st >= 0.0 { i <= s } else { i >= s };
                        if !enter {
                            pc = (pc as i64 + bc_j(ins)) as usize;
                        }
                    } else {
                        sync!();
                        return Err(self.l().runtime_error(b"'for' initial value must be a number"));
                    }
                }
                BCOp::FORL => {
                    let ai = base + a as usize;
                    let i = self.at(ai + FORL_IDX as usize).num();
                    let s = self.at(ai + FORL_STOP as usize).num();
                    let st = self.at(ai + FORL_STEP as usize).num();
                    let ni = i + st;
                    let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                    if cont {
                        let nv = LuaValue::number_raw(ni);
                        self.set_at(ai + FORL_IDX as usize, nv);
                        self.set_at(ai + FORL_EXT as usize, nv);
                        pc = (pc as i64 + bc_j(ins)) as usize;
                    }
                }
                BCOp::LOOP => { /* hotcount hook: no-op for now */ }
                BCOp::JMP => pc = (pc as i64 + bc_j(ins)) as usize,
                BCOp::ISNEXT => pc = (pc as i64 + bc_j(ins)) as usize,
                BCOp::ITERC | BCOp::ITERN => {
                    sync!();
                    self.iter_call(a, bc_b(ins) as usize)?;
                    resync!();
                }
                BCOp::ITERL => {
                    let first = reg!(a);
                    if !first.is_nil() {
                        setreg!(a - 1, first);
                        pc = (pc as i64 + bc_j(ins)) as usize;
                    }
                }
                BCOp::VARG => {
                    sync!();
                    self.vararg(a, bc_b(ins));
                    resync!();
                }

                other => {
                    sync!();
                    return Err(self
                        .l()
                        .runtime_error(format!("opcode {:?} not implemented", other).as_bytes()));
                }
            }
        }
    }

    // -- Cold slow paths -------------------------------------------------

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
            None => Err(self.l().runtime_error(b"attempt to index a non-table value")),
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
            None => Err(self.l().runtime_error(b"attempt to index a non-table value")),
        }
    }

    #[cold]
    fn tsetm(&mut self, a: u32, d: u32) -> LuaResult<()> {
        let t = self.at(self.base + a as usize - 1);
        let base_key = unsafe { *self.knp.add(d as usize) } as i64 - (1i64 << 52);
        let tab = match t.as_table() {
            Some(t) => t,
            None => return Err(self.l().runtime_error(b"attempt to index a non-table value")),
        };
        for i in 0..self.multres {
            let v = self.at(self.base + a as usize + i);
            tab.as_mut()
                .set(LuaValue::number((base_key + i as i64) as f64), v);
        }
        Ok(())
    }

    // -- Calls -----------------------------------------------------------

    fn do_call(&mut self, a: u32, nargs: usize, want: i32) -> LuaResult<()> {
        let func_slot = self.base + a as usize;
        let f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => return Err(self.l().runtime_error(b"attempt to call a non-function value")),
        };
        match gf.as_ref() {
            GcFunc::Lua(_) => {
                self.enter_lua(gf, func_slot, nargs, want);
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
        let n = f(l)? as usize;
        for i in 0..n {
            l.stack[func_slot + i] = l.stack[args_base + i];
        }
        l.base = saved_base;
        l.top = saved_top;
        Ok(n)
    }

    fn do_tailcall(&mut self, a: u32, nargs: usize, floor: usize) -> LuaResult<Option<usize>> {
        // Simple (non-TCO) tail call: run the callee as a nested execution,
        // then return its results from the current frame. True TCO is later.
        let func_slot = self.base + a as usize;
        let n = execute(self.l(), func_slot, nargs, -1)?;
        Ok(self.do_return(func_slot, n, floor))
    }

    /// Move `n` results to the caller's slot, restore the caller frame and
    /// continue. Returns `Some(n)` when the entry frame returns to the host.
    fn do_return(&mut self, src: usize, n: usize, frame_floor: usize) -> Option<usize> {
        if !self.l().openuv.is_empty() {
            self.close_upvals(self.base);
        }
        let frame = self.l().frames.pop().expect("frame underflow");
        let dst = frame.result_slot;
        for i in 0..n {
            self.set_at(dst + i, self.at(src + i));
        }
        self.multres = n;

        if self.l().frames.len() < frame_floor || frame.return_pc == usize::MAX {
            let want = frame.nresults;
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

        self.base = frame.base;
        self.pc = frame.return_pc;
        self.varg_base = frame.varg_base;
        self.nvarg = frame.nvarg;
        let cl = frame.func.as_func().unwrap();
        self.reload(cl);

        let want = frame.nresults;
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
        execute(self.l(), fs, 2, nret as i32 - 1)?;
        Ok(())
    }

    // -- Varargs ---------------------------------------------------------

    fn vararg(&mut self, a: u32, b: u32) {
        let dst = self.base + a as usize;
        if b == 0 {
            for i in 0..self.nvarg {
                self.set_at(dst + i, self.at(self.varg_base + i));
            }
            self.multres = self.nvarg;
            self.l().top = dst + self.nvarg;
        } else {
            let want = (b - 1) as usize;
            for i in 0..want {
                let v = if i < self.nvarg {
                    self.at(self.varg_base + i)
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
            if let UpvalState::Open(s) = uv.as_ref().state {
                if s == slot {
                    return uv;
                }
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
        let fref = self.l().heap().alloc_func(GcFunc::Lua(LuaClosure {
            proto,
            env,
            upvals,
        }));
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
