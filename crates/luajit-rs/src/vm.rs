//! The bytecode interpreter.
//!
//! Design (see the discussion that led here):
//! * A `match`-on-opcode dispatch loop is the stable baseline. Register and
//!   bytecode access go through raw pointers into a fixed-capacity stack, so
//!   there are no per-instruction bounds checks; the stack never reallocates,
//!   which keeps those pointers valid.
//! * The register window follows LuaJIT's FR2 layout (callee `base` =
//!   caller_base + A + 2; function at `base-2`), matching the bytecode.
//! * Fast paths (arithmetic on numbers, comparisons, direct table access)
//!   are inline; failures fall through to `#[cold]` slow paths.
//! * Errors use `LuaResult` with a fieldless error enum; the error object and
//!   yield count live on the `LuaState`. The hot path returns no `Result`.
//! * Lua->Lua calls do not recurse in Rust: `CALL` pushes a `Frame` and keeps
//!   looping; `RET` pops one. Only tail calls and re-entrant C calls recurse.

use crate::bc::*;
use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, GcFunc, LuaClosure, Upval, UpvalState};
use crate::gc::GcPtr;
use crate::proto::{KGc, Proto, PROTO_UV_IMMUTABLE, PROTO_UV_LOCAL, PROTO_VARARG};
use crate::state::{Frame, LuaState};
use crate::table::LuaTable;
use crate::value::*;

/// Call a value with the given arguments and collect all results.
/// The host entry point into the VM.
pub fn call(l: &mut LuaState, func: LuaValue, args: &[LuaValue]) -> LuaResult<Vec<LuaValue>> {
    // Lay out an FR2 call frame at slot 0: [func, gap, args...].
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
pub fn execute(
    l: &mut LuaState,
    func_slot: usize,
    nargs: usize,
    want: i32,
) -> LuaResult<usize> {
    let f = l.stack[func_slot];
    let gf = match f.as_func() {
        Some(p) => p,
        None => return Err(l.runtime_error(b"attempt to call a non-function value")),
    };
    match gf.as_ref() {
        GcFunc::C(cc) => return call_c(l, cc.f, func_slot, nargs, want),
        GcFunc::Lua(_) => {}
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
    let n = f(l)?;
    let n = n as usize;
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

/// Interpreter loop state. Raw pointers into the (non-reallocating) stack and
/// current prototype avoid bounds checks and borrow friction.
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
            cl: GcPtr::from_addr(8).unwrap(), // placeholder, set by enter_lua
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
    unsafe fn r(&self, i: u32) -> LuaValue {
        *self.sp.add(self.base + i as usize)
    }

    #[inline(always)]
    unsafe fn set(&self, i: u32, v: LuaValue) {
        *self.sp.add(self.base + i as usize) = v;
    }

    #[inline(always)]
    unsafe fn at(&self, abs: usize) -> LuaValue {
        *self.sp.add(abs)
    }

    #[inline(always)]
    unsafe fn set_at(&self, abs: usize, v: LuaValue) {
        *self.sp.add(abs) = v;
    }

    fn kstr(&self, d: u32) -> LuaValue {
        match &self.proto().kgc[d as usize] {
            KGc::Str(sid) => self.l().heap().str_value(*sid),
            _ => unreachable!("expected string constant"),
        }
    }

    fn knum(&self, d: u32) -> f64 {
        unsafe { *self.knp.add(d as usize) }
    }

    /// Set up a Lua frame for the function at `func_slot`. Saves the current
    /// frame (unless this is the entry) and switches loop state to the callee.
    fn enter_lua(&mut self, gf: GcPtr<GcFunc>, func_slot: usize, nargs: usize, want: i32) {
        // Save the current frame if we are already inside one (bcp != null).
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
            // Entry frame: record only the result destination / want.
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
            unsafe {
                self.set_at(newbase - 2, LuaValue::func(gf));
                for i in 0..numparams {
                    let v = if i < nargs {
                        self.at(callbase + i)
                    } else {
                        LuaValue::NIL
                    };
                    self.set_at(newbase + i, v);
                }
            }
            self.varg_base = callbase + numparams;
            self.nvarg = nargs.saturating_sub(numparams);
            self.base = newbase;
        } else {
            unsafe {
                for i in nargs..numparams {
                    self.set_at(callbase + i, LuaValue::NIL);
                }
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

    /// Reload cached pointers after returning into a caller frame.
    fn reload(&mut self, cl: GcPtr<GcFunc>) {
        let pt = match cl.as_ref() {
            GcFunc::Lua(c) => c.proto.as_ref(),
            _ => unreachable!(),
        };
        self.cl = cl;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
    }

    #[inline(always)]
    fn fetch(&mut self) -> BCIns {
        let ins = unsafe { *self.bcp.add(self.pc) };
        self.pc += 1;
        ins
    }

    /// The main dispatch loop. Returns when the frame at `frame_floor` returns
    /// to the host.
    fn run(&mut self, frame_floor: usize) -> LuaResult<usize> {
        loop {
            let ins = self.fetch();
            let op = bc_op(ins);
            let a = bc_a(ins);
            match op {
                BCOp::ISLT | BCOp::ISGE | BCOp::ISLE | BCOp::ISGT => {
                    let x = unsafe { self.r(a) };
                    let y = unsafe { self.r(bc_d(ins)) };
                    let cond = self.compare(op, x, y)?;
                    self.cond_jump(cond);
                }
                BCOp::ISEQV | BCOp::ISNEV => {
                    let x = unsafe { self.r(a) };
                    let y = unsafe { self.r(bc_d(ins)) };
                    let eq = val_eq(x, y);
                    self.cond_jump(eq == (op == BCOp::ISEQV));
                }
                BCOp::ISEQS | BCOp::ISNES => {
                    let x = unsafe { self.r(a) };
                    let y = self.kstr(bc_d(ins));
                    let eq = val_eq(x, y);
                    self.cond_jump(eq == (op == BCOp::ISEQS));
                }
                BCOp::ISEQN | BCOp::ISNEN => {
                    let x = unsafe { self.r(a) };
                    let y = LuaValue::number(self.knum(bc_d(ins)));
                    let eq = val_eq(x, y);
                    self.cond_jump(eq == (op == BCOp::ISEQN));
                }
                BCOp::ISEQP | BCOp::ISNEP => {
                    let x = unsafe { self.r(a) };
                    let y = PRI[bc_d(ins) as usize];
                    let eq = val_eq(x, y);
                    self.cond_jump(eq == (op == BCOp::ISEQP));
                }
                BCOp::ISTC | BCOp::ISFC => {
                    let d = unsafe { self.r(bc_d(ins)) };
                    let cond = d.is_truthy() == (op == BCOp::ISTC);
                    if cond {
                        unsafe { self.set(a, d) };
                    }
                    self.cond_jump(cond);
                }
                BCOp::IST | BCOp::ISF => {
                    let d = unsafe { self.r(bc_d(ins)) };
                    let cond = d.is_truthy() == (op == BCOp::IST);
                    self.cond_jump(cond);
                }
                BCOp::MOV => {
                    let v = unsafe { self.r(bc_d(ins)) };
                    unsafe { self.set(a, v) };
                }
                BCOp::NOT => {
                    let v = unsafe { self.r(bc_d(ins)) };
                    unsafe { self.set(a, LuaValue::boolean(!v.is_truthy())) };
                }
                BCOp::UNM => {
                    let v = unsafe { self.r(bc_d(ins)) };
                    if let Some(n) = v.as_number() {
                        unsafe { self.set(a, LuaValue::number_raw(-n)) };
                    } else {
                        return Err(self.arith_err());
                    }
                }
                BCOp::LEN => {
                    let v = unsafe { self.r(bc_d(ins)) };
                    let r = self.len_op(v)?;
                    unsafe { self.set(a, r) };
                }
                BCOp::ADDVV | BCOp::SUBVV | BCOp::MULVV | BCOp::DIVVV | BCOp::MODVV => {
                    let x = unsafe { self.r(bc_b(ins)) };
                    let y = unsafe { self.r(bc_c(ins)) };
                    let r = self.arith_vv(op, x, y)?;
                    unsafe { self.set(a, r) };
                }
                BCOp::ADDVN | BCOp::SUBVN | BCOp::MULVN | BCOp::DIVVN | BCOp::MODVN => {
                    let x = unsafe { self.r(bc_b(ins)) };
                    let y = LuaValue::number(self.knum(bc_c(ins)));
                    let base = (op as u32 - BCOp::ADDVN as u32) + BCOp::ADDVV as u32;
                    let r = self.arith_vv(BCOp::from_u32(base), x, y)?;
                    unsafe { self.set(a, r) };
                }
                BCOp::ADDNV | BCOp::SUBNV | BCOp::MULNV | BCOp::DIVNV | BCOp::MODNV => {
                    let y = unsafe { self.r(bc_b(ins)) };
                    let x = LuaValue::number(self.knum(bc_c(ins)));
                    let base = (op as u32 - BCOp::ADDNV as u32) + BCOp::ADDVV as u32;
                    let r = self.arith_vv(BCOp::from_u32(base), x, y)?;
                    unsafe { self.set(a, r) };
                }
                BCOp::POW => {
                    let x = unsafe { self.r(bc_b(ins)) };
                    let y = unsafe { self.r(bc_c(ins)) };
                    match (x.as_number(), y.as_number()) {
                        (Some(x), Some(y)) => unsafe { self.set(a, LuaValue::number_raw(x.powf(y))) },
                        _ => return Err(self.arith_err()),
                    }
                }
                BCOp::CAT => {
                    let r = self.concat(bc_b(ins), bc_c(ins))?;
                    unsafe { self.set(a, r) };
                }
                BCOp::KSTR => {
                    let v = self.kstr(bc_d(ins));
                    unsafe { self.set(a, v) };
                }
                BCOp::KSHORT => {
                    let d = bc_d(ins) as i16 as f64;
                    unsafe { self.set(a, LuaValue::number(d)) };
                }
                BCOp::KNUM => {
                    let v = LuaValue::number(self.knum(bc_d(ins)));
                    unsafe { self.set(a, v) };
                }
                BCOp::KPRI => {
                    unsafe { self.set(a, PRI[bc_d(ins) as usize]) };
                }
                BCOp::KNIL => {
                    let to = bc_d(ins);
                    for i in a..=to {
                        unsafe { self.set(i, LuaValue::NIL) };
                    }
                }
                BCOp::UGET => {
                    let uv = self.lua_cl().upvals[bc_d(ins) as usize];
                    let v = self.upval_get(uv);
                    unsafe { self.set(a, v) };
                }
                BCOp::USETV => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = unsafe { self.r(bc_d(ins)) };
                    self.upval_set(uv, v);
                }
                BCOp::USETS => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = self.kstr(bc_d(ins));
                    self.upval_set(uv, v);
                }
                BCOp::USETN => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = LuaValue::number(self.knum(bc_d(ins)));
                    self.upval_set(uv, v);
                }
                BCOp::USETP => {
                    let uv = self.lua_cl().upvals[a as usize];
                    self.upval_set(uv, PRI[bc_d(ins) as usize]);
                }
                BCOp::UCLO => {
                    let level = self.base + a as usize;
                    self.close_upvals(level);
                    self.pc = self.jump_dest(ins);
                }
                BCOp::FNEW => {
                    let v = self.new_closure(bc_d(ins));
                    unsafe { self.set(a, v) };
                }
                BCOp::TNEW => {
                    let t = self.l().heap().alloc_table(LuaTable::new(0, 0));
                    unsafe { self.set(a, LuaValue::table(t)) };
                }
                BCOp::TDUP => {
                    let templ = match &self.proto().kgc[bc_d(ins) as usize] {
                        KGc::Table(t) => t.dup(),
                        _ => unreachable!("expected template table"),
                    };
                    let t = self.l().heap().alloc_table(templ);
                    unsafe { self.set(a, LuaValue::table(t)) };
                }
                BCOp::GGET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr(bc_d(ins));
                    let v = env.as_ref().get(key);
                    unsafe { self.set(a, v) };
                }
                BCOp::GSET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr(bc_d(ins));
                    let v = unsafe { self.r(a) };
                    env.as_mut().set(key, v);
                }
                BCOp::TGETV => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = unsafe { self.r(bc_c(ins)) };
                    let v = self.index_get(t, k)?;
                    unsafe { self.set(a, v) };
                }
                BCOp::TGETS => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = self.kstr(bc_c(ins));
                    let v = self.index_get(t, k)?;
                    unsafe { self.set(a, v) };
                }
                BCOp::TGETB => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = LuaValue::number(bc_c(ins) as f64);
                    let v = self.index_get(t, k)?;
                    unsafe { self.set(a, v) };
                }
                BCOp::TSETV => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = unsafe { self.r(bc_c(ins)) };
                    let v = unsafe { self.r(a) };
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETS => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = self.kstr(bc_c(ins));
                    let v = unsafe { self.r(a) };
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETB => {
                    let t = unsafe { self.r(bc_b(ins)) };
                    let k = LuaValue::number(bc_c(ins) as f64);
                    let v = unsafe { self.r(a) };
                    self.index_set(t, k, v)?;
                }
                BCOp::TSETM => {
                    self.tsetm(a, bc_d(ins))?;
                }
                BCOp::CALL => {
                    let nargs = bc_c(ins) as usize - 1;
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                }
                BCOp::CALLM => {
                    let nargs = bc_c(ins) as usize + self.multres;
                    self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                }
                BCOp::CALLT => {
                    let nargs = bc_d(ins) as usize - 1;
                    if let Some(n) = self.do_tailcall(a, nargs, frame_floor)? {
                        return Ok(n);
                    }
                }
                BCOp::CALLMT => {
                    let nargs = bc_d(ins) as usize + self.multres;
                    if let Some(n) = self.do_tailcall(a, nargs, frame_floor)? {
                        return Ok(n);
                    }
                }
                BCOp::RET0 => {
                    if let Some(n) = self.do_return(self.base, 0, frame_floor) {
                        return Ok(n);
                    }
                }
                BCOp::RET1 => {
                    if let Some(n) = self.do_return(self.base + a as usize, 1, frame_floor) {
                        return Ok(n);
                    }
                }
                BCOp::RET => {
                    let n = bc_d(ins) as usize - 1;
                    if let Some(n) = self.do_return(self.base + a as usize, n, frame_floor) {
                        return Ok(n);
                    }
                }
                BCOp::RETM => {
                    let n = self.multres + bc_d(ins) as usize;
                    if let Some(n) = self.do_return(self.base + a as usize, n, frame_floor) {
                        return Ok(n);
                    }
                }
                BCOp::FORI => {
                    self.for_init(a, ins)?;
                }
                BCOp::FORL => {
                    self.for_loop(a, ins)?;
                }
                BCOp::LOOP => { /* interpreter hotcount hook: no-op for now */ }
                BCOp::JMP => {
                    self.pc = self.jump_dest(ins);
                }
                BCOp::ISNEXT => {
                    // Generic-for start; treat like JMP to the ITERC/ITERN.
                    self.pc = self.jump_dest(ins);
                }
                BCOp::ITERC | BCOp::ITERN => {
                    self.iter_call(a, bc_b(ins) as usize, ins)?;
                }
                BCOp::ITERL => {
                    self.iter_loop(a, ins);
                }
                BCOp::VARG => {
                    self.vararg(a, bc_b(ins), ins);
                }
                other => {
                    return Err(self
                        .l()
                        .runtime_error(format!("opcode {:?} not implemented", other).as_bytes()));
                }
            }
        }
    }

    // -- Control flow helpers --------------------------------------------

    #[inline(always)]
    fn jump_dest(&self, ins: BCIns) -> usize {
        (self.pc as i64 + bc_j(ins)) as usize
    }

    /// After a comparison/test: the following instruction is a JMP. Take it if
    /// `cond`, otherwise skip it.
    #[inline(always)]
    fn cond_jump(&mut self, cond: bool) {
        let jmp = unsafe { *self.bcp.add(self.pc) };
        self.pc += 1;
        if cond {
            self.pc = (self.pc as i64 + bc_j(jmp)) as usize;
        }
    }

    // -- Arithmetic / comparison -----------------------------------------

    #[cold]
    fn arith_err(&self) -> LuaError {
        self.l()
            .runtime_error(b"attempt to perform arithmetic on a non-number value")
    }

    #[inline(always)]
    fn arith_vv(&self, op: BCOp, x: LuaValue, y: LuaValue) -> LuaResult<LuaValue> {
        if x.is_number() && y.is_number() {
            let a = x.num();
            let b = y.num();
            let r = match op {
                BCOp::ADDVV => a + b,
                BCOp::SUBVV => a - b,
                BCOp::MULVV => a * b,
                BCOp::DIVVV => a / b,
                BCOp::MODVV => a - (a / b).floor() * b,
                _ => unreachable!(),
            };
            Ok(LuaValue::number_raw(r))
        } else {
            Err(self.arith_err())
        }
    }

    fn compare(&self, op: BCOp, x: LuaValue, y: LuaValue) -> LuaResult<bool> {
        if x.is_number() && y.is_number() {
            let a = x.num();
            let b = y.num();
            return Ok(match op {
                BCOp::ISLT => a < b,
                BCOp::ISGE => a >= b,
                BCOp::ISLE => a <= b,
                BCOp::ISGT => a > b,
                _ => unreachable!(),
            });
        }
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
        v.as_string_id().map(|sid| self.l().heap().strings.get(sid).to_vec())
    }

    fn len_op(&self, v: LuaValue) -> LuaResult<LuaValue> {
        if let Some(sid) = v.as_string_id() {
            let n = self.l().heap().strings.get(sid).len();
            return Ok(LuaValue::number(n as f64));
        }
        if let Some(t) = v.as_table() {
            return Ok(LuaValue::number(t.as_ref().len() as f64));
        }
        Err(self.l().runtime_error(b"attempt to get length of a non-table value"))
    }

    fn concat(&mut self, from: u32, to: u32) -> LuaResult<LuaValue> {
        let mut buf = Vec::new();
        for i in from..=to {
            let v = unsafe { self.r(i) };
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

    // -- Table access ----------------------------------------------------

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

    /// TSETM: store the multres values into a table starting at integer key
    /// `d`'s biased base.
    fn tsetm(&mut self, a: u32, d: u32) -> LuaResult<()> {
        let t = unsafe { self.r(a - 1) };
        let base_key = self.knum(d) as i64 - (1i64 << 52);
        let tab = match t.as_table() {
            Some(t) => t,
            None => return Err(self.l().runtime_error(b"attempt to index a non-table value")),
        };
        for i in 0..self.multres {
            let v = unsafe { self.r(a + i as u32) };
            tab.as_mut()
                .set(LuaValue::number((base_key + i as i64) as f64), v);
        }
        Ok(())
    }

    // -- Calls -----------------------------------------------------------

    fn do_call(&mut self, a: u32, nargs: usize, want: i32) -> LuaResult<()> {
        let func_slot = self.base + a as usize;
        let f = unsafe { self.at(func_slot) };
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
                self.finish_c_results(func_slot, n, want);
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

    fn finish_c_results(&mut self, func_slot: usize, n: usize, want: i32) {
        if want >= 0 {
            for i in n..(want as usize) {
                unsafe { self.set_at(func_slot + i, LuaValue::NIL) };
            }
        } else {
            self.multres = n;
        }
    }

    fn do_tailcall(
        &mut self,
        a: u32,
        nargs: usize,
        _floor: usize,
    ) -> LuaResult<Option<usize>> {
        // Simple (non-TCO) tail call: run the callee as a nested execution,
        // then return its results from the current frame. Correct for finite
        // recursion; true TCO is a later optimization.
        let func_slot = self.base + a as usize;
        // Move func + args into place is already done by the compiler layout.
        let n = execute(self.l(), func_slot, nargs, -1)?;
        // Results are at func_slot; return them from this frame.
        Ok(self.do_return(func_slot, n, _floor))
    }

    // -- Return ----------------------------------------------------------

    /// Move `n` results from `src` to the caller's result slot, restore the
    /// caller frame and continue. Returns `Some(n)` when the entry frame
    /// returns to the host.
    fn do_return(&mut self, src: usize, n: usize, frame_floor: usize) -> Option<usize> {
        // Close any upvalues at or above this frame's base.
        if !self.l().openuv.is_empty() {
            self.close_upvals(self.base);
        }
        let frame = self.l().frames.pop().expect("frame underflow");
        let dst = frame.result_slot;
        for i in 0..n {
            unsafe { self.set_at(dst + i, self.at(src + i)) };
        }
        self.multres = n;

        if self.l().frames.len() < frame_floor || frame.return_pc == usize::MAX {
            // Returned to the host.
            let want = frame.nresults;
            let got = if want >= 0 {
                for i in n..(want as usize) {
                    unsafe { self.set_at(dst + i, LuaValue::NIL) };
                }
                want as usize
            } else {
                n
            };
            self.l().top = dst + got;
            return Some(got);
        }

        // Restore caller frame.
        self.base = frame.base;
        self.pc = frame.return_pc;
        self.varg_base = frame.varg_base;
        self.nvarg = frame.nvarg;
        let cl = frame.func.as_func().unwrap();
        self.reload(cl);

        let want = frame.nresults;
        if want >= 0 {
            for i in n..(want as usize) {
                unsafe { self.set_at(dst + i, LuaValue::NIL) };
            }
            self.l().top = dst + want as usize;
        } else {
            self.l().top = dst + n;
        }
        None
    }

    // -- Numeric for -----------------------------------------------------

    fn for_init(&mut self, a: u32, ins: BCIns) -> LuaResult<()> {
        let idx = unsafe { self.r(a + FORL_IDX) };
        let stop = unsafe { self.r(a + FORL_STOP) };
        let step = unsafe { self.r(a + FORL_STEP) };
        match (idx.as_number(), stop.as_number(), step.as_number()) {
            (Some(i), Some(s), Some(st)) => {
                let enter = if st >= 0.0 { i <= s } else { i >= s };
                unsafe { self.set(a + FORL_EXT, LuaValue::number_raw(i)) };
                if !enter {
                    self.pc = self.jump_dest(ins);
                }
                Ok(())
            }
            _ => Err(self.l().runtime_error(b"'for' initial value must be a number")),
        }
    }

    fn for_loop(&mut self, a: u32, ins: BCIns) -> LuaResult<()> {
        let idx = unsafe { self.r(a + FORL_IDX).num() };
        let stop = unsafe { self.r(a + FORL_STOP).num() };
        let step = unsafe { self.r(a + FORL_STEP).num() };
        let n = idx + step;
        let cont = if step >= 0.0 { n <= stop } else { n >= stop };
        if cont {
            unsafe {
                self.set(a + FORL_IDX, LuaValue::number_raw(n));
                self.set(a + FORL_EXT, LuaValue::number_raw(n));
            }
            self.pc = self.jump_dest(ins);
        }
        Ok(())
    }

    // -- Generic for -----------------------------------------------------

    fn iter_call(&mut self, a: u32, nret: usize, _ins: BCIns) -> LuaResult<()> {
        // Generic-for step. Slots A-3/A-2/A-1 hold generator/state/control;
        // results go to A .. A+nret-1. Set up an FR2 call at slot A and call
        // generator(state, control).
        let fs = self.base + a as usize;
        unsafe {
            let genf = self.at(fs - 3);
            let state = self.at(fs - 2);
            let ctl = self.at(fs - 1);
            self.set_at(fs, genf);
            self.set_at(fs + 2, state);
            self.set_at(fs + 3, ctl);
        }
        // Results land at fs (= R(A)), padded to `nret` by execute's `want`.
        execute(self.l(), fs, 2, nret as i32 - 1)?;
        Ok(())
    }

    fn iter_loop(&mut self, a: u32, ins: BCIns) {
        // If the first returned value (A) is non-nil, copy it to the control
        // slot (A-1) and jump back.
        let first = unsafe { self.r(a) };
        if !first.is_nil() {
            unsafe { self.set(a - 1, first) };
            self.pc = self.jump_dest(ins);
        }
    }

    // -- Varargs ---------------------------------------------------------

    fn vararg(&mut self, a: u32, b: u32, _ins: BCIns) {
        if b == 0 {
            // Copy all varargs; set multres.
            for i in 0..self.nvarg {
                let v = unsafe { self.at(self.varg_base + i) };
                unsafe { self.set(a + i as u32, v) };
            }
            self.multres = self.nvarg;
            self.l().top = self.base + a as usize + self.nvarg;
        } else {
            let want = (b - 1) as usize;
            for i in 0..want {
                let v = if i < self.nvarg {
                    unsafe { self.at(self.varg_base + i) }
                } else {
                    LuaValue::NIL
                };
                unsafe { self.set(a + i as u32, v) };
            }
        }
    }

    // -- Upvalues / closures ---------------------------------------------

    fn upval_get(&self, uv: GcPtr<Upval>) -> LuaValue {
        match &uv.as_ref().state {
            UpvalState::Open(slot) => unsafe { self.at(*slot) },
            UpvalState::Closed(v) => *v,
        }
    }

    fn upval_set(&self, uv: GcPtr<Upval>, v: LuaValue) {
        match &mut uv.as_mut().state {
            UpvalState::Open(slot) => unsafe { self.set_at(*slot, v) },
            UpvalState::Closed(cv) => *cv = v,
        }
    }

    /// Find or create an open upvalue for an absolute stack slot.
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

    /// Close all open upvalues at or above `level`.
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
                    let v = unsafe { self.at(s) };
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

/// Raw equality used by ISEQ*/ISNE*: numbers compare by value (so `-0.0`
/// and NaN behave), everything else by bit pattern (interned strings and GC
/// pointers compare by identity).
#[inline(always)]
fn val_eq(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else {
        a.to_bits() == b.to_bits()
    }
}

#[allow(dead_code)]
fn _cclosure_marker(_c: &CClosure) {}
