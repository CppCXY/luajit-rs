//! VM metamethod helpers, ported from LuaJIT's `lj_meta.c`.
//!
//! Hot paths (`meta_tget`, `meta_tset`, `meta_arith`, `meta_comp`,
//! `meta_equal`) set up a `FRAME_CONT` continuation frame via `mmcall_cont`
//! and return `None` to signal "resync into the metamethod".  Cold paths
//! (`meta_cat`, `meta_len`) still use `execute()` recursion.
//!
//! C metamethods are invoked inline through `call_c_fn`; only Lua
//! metamethods need continuation frames.

use super::*;
use crate::meta::{MM, meta_fast, meta_lookup, metatable_of};

const LJ_MAX_IDXCHAIN: usize = 100;

/// Lua type name for error messages ("a nil value", "a string value", ...).
fn lua_typename(v: LuaValue) -> &'static str {
    if v.is_nil() {
        "nil"
    } else if v.is_bool() {
        "boolean"
    } else if v.is_number() {
        "number"
    } else if v.is_string() {
        "string"
    } else if v.as_table().is_some() {
        "table"
    } else if v.as_func().is_some() {
        "function"
    } else if v.as_thread().is_some() {
        "thread"
    } else if v.as_userdata().is_some() {
        "userdata"
    } else {
        "value"
    }
}

/// `debug_varname`: the local variable name bound to `slot` at `pc`.
fn debug_varname(pt: &crate::runtime::proto::Proto, pc: usize, slot: u32) -> Option<String> {
    for (s, a, b, name) in &pt.varnames {
        if *s as u32 == slot && pc >= *a as usize && pc <= *b as usize {
            return Some(name.clone());
        }
    }
    None
}

/// LuaJIT's `lj_debug_slotname`: deduce the (kind, name) of the value in a
/// register slot by scanning backwards from `pc` for the instruction that
/// wrote it. Kinds: "global", "local", "upvalue", "field", "method".
fn debug_slotname(
    pt: &crate::runtime::proto::Proto,
    pc: usize,
    mut slot: u32,
) -> Option<(&'static str, String)> {
    if let Some(n) = debug_varname(pt, pc, slot) {
        return Some(("local", n));
    }
    let kstr = |idx: u32| -> String {
        pt.kstrv
            .get(idx as usize)
            .and_then(|v| v.as_string())
            .map(|s| String::from_utf8_lossy(s.as_ref().as_bytes()).into_owned())
            .unwrap_or_else(|| "?".into())
    };
    let mut i = pc as isize - 1;
    loop {
        if i < 1 {
            break;
        }
        let ins = pt.bc[i as usize];
        let op = crate::bc::bc_op(ins);
        let ra = crate::bc::bc_a(ins);
        // A base-mode instruction (JMP, ISEQ*, ...) can write the slot
        // range; LuaJIT stops tracing here (`bcmode_a == BCMbase` →
        // NULL) so `(aaa or aaa)` does not name the inner global.
        if crate::bc::bcmode_a(op) == crate::bc::BCMode::Base as u32 && slot >= ra {
            return None;
        }
        if crate::bc::bcmode_a(op) == crate::bc::BCMode::Dst as u32 && ra == slot {
            match op {
                crate::bc::BCOp::MOV => {
                    slot = crate::bc::bc_d(ins);
                    if let Some(n) = debug_varname(pt, pc, slot) {
                        return Some(("local", n));
                    }
                    i -= 1;
                    continue;
                }
                crate::bc::BCOp::GGET => {
                    return Some(("global", kstr(crate::bc::bc_d(ins))));
                }
                crate::bc::BCOp::TGETS => {
                    let name = kstr(crate::bc::bc_c(ins));
                    if i > 1 {
                        let insp = pt.bc[(i - 1) as usize];
                        // `a:bbbb(...)`: the object is moved to slot+2 (this
                        // VM's CALL args start at func+2, cf. do_call's
                        // args_base = func_slot + 2). LuaJIT uses slot+1
                        // because its CALL layout differs.
                        if crate::bc::bc_op(insp) == crate::bc::BCOp::MOV
                            && crate::bc::bc_a(insp) == ra + 2
                            && crate::bc::bc_d(insp) == crate::bc::bc_b(ins)
                        {
                            return Some(("method", name));
                        }
                    }
                    return Some(("field", name));
                }
                crate::bc::BCOp::UGET => {
                    let name = pt
                        .uvnames
                        .get(crate::bc::bc_d(ins) as usize)
                        .cloned()
                        .unwrap_or_else(|| "?".into());
                    return Some(("upvalue", name));
                }
                _ => return None,
            }
        }
        i -= 1;
    }
    None
}

/// Deduce the callee name for a failing call whose func slot is `func_slot`.
pub(super) fn debug_callname(l: &LuaState, func_slot: usize) -> Option<(&'static str, String)> {
    // The current Lua frame's closure sits at `base - 2` (the frame's func
    // slot), not at `func_slot - 2` (that is the *callee's* slot).
    let cl = l.stack.get(l.base.saturating_sub(2))?.as_func()?;
    let crate::func::GcFunc::Lua(c) = cl.as_ref() else {
        return None;
    };
    let pt = c.proto.as_ref();
    let pc = l.debug_pc;
    if pc >= 1 && pc - 1 < pt.bc.len() {
        let ins = pt.bc[pc - 1];
        let op = crate::bc::bc_op(ins);
        if matches!(
            op,
            crate::bc::BCOp::CALL | crate::bc::BCOp::CALLM | crate::bc::BCOp::CALLT
        ) {
            let ra = crate::bc::bc_a(ins);
            if func_slot >= l.base && func_slot - l.base == ra as usize {
                // Skip the CALL itself (a base-mode instruction stops the
                // scan) and trace the func slot from the instruction before.
                return debug_slotname(pt, pc - 1, ra);
            }
        }
    }
    None
}

impl Interp {
    #[inline]
    fn mm_frame(&self) -> usize {
        self.base + self.lua_cl().proto.as_ref().framesize as usize
    }

    /// `attempt to index <kind> '<name>' (a <type> value)`, deducing the
    /// object's name from the failing TGET instruction (LuaJIT's
    /// `lj_meta_tget` error path).
    fn index_error(&self, o: LuaValue) -> LuaError {
        let pt = self.proto();
        let pc = self.pc;
        let msg = if pc >= 2 && pc - 1 < pt.bc.len() {
            // The failing TGET is at `pc-1`; skip it (its result slot may
            // alias the object slot) and scan from the instruction before.
            let ins = pt.bc[pc - 1];
            let slot = crate::bc::bc_b(ins);
            debug_slotname(pt, pc - 1, slot).map(|(kind, name)| {
                format!(
                    "attempt to index {} '{}' (a {} value)",
                    kind,
                    name,
                    lua_typename(o)
                )
            })
        } else {
            None
        };
        match msg {
            Some(m) => self.l().runtime_error(m.as_bytes()),
            None => self.l().runtime_error(b"attempt to index a non-table value"),
        }
    }

    /// `lj_meta_tget`:
    /// resolved directly (raw hit, C metamethod) or `None` when a Lua
    /// metamethod continuation frame was set up.
    pub(super) fn meta_tget(
        &mut self,
        o: LuaValue,
        k: LuaValue,
        a: u32,
    ) -> LuaResult<Option<LuaValue>> {
        let mut cur = o;
        for _ in 0..LJ_MAX_IDXCHAIN {
            if let Some(t) = cur.as_table() {
                let tv = t.as_ref().get(k);
                if !tv.is_nil() {
                    return Ok(Some(tv));
                }
                let mo = match meta_fast(self.l().global(), t.as_ref().metatable, MM::Index) {
                    Some(mo) => mo,
                    None => return Ok(Some(LuaValue::NIL)),
                };
                if mo.is_func() {
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            let v = self.call_c_fn(cc.f, mo, &[cur, k])?;
                            return Ok(Some(v));
                        }
                        GcFunc::Lua(_) => {
                            self.l().mmname = Some(("__index", mo.to_bits()));
                            self.mmcall_cont(Cont::Ra, a, mo, &[cur, k]);
                            return Ok(None);
                        }
                    }
                }
                cur = mo;
            } else {
                let mo = meta_lookup(self.l().global(), cur, MM::Index);
                if mo.is_nil() {
                    return Err(self.index_error(cur));
                }
                if mo.is_func() {
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            let v = self.call_c_fn(cc.f, mo, &[cur, k])?;
                            return Ok(Some(v));
                        }
                        GcFunc::Lua(_) => {
                            self.l().mmname = Some(("__index", mo.to_bits()));
                            self.mmcall_cont(Cont::Ra, a, mo, &[cur, k]);
                            return Ok(None);
                        }
                    }
                }
                cur = mo;
            }
        }
        Err(self
            .l()
            .runtime_error(b"'__index' chain too long; possible loop"))
    }

    /// `lj_meta_tset`: `__newindex` chain.  Returns `Some(true)` if the
    /// raw set was done inline, `None` if a Lua metamethod was called.
    pub(super) fn meta_tset(
        &mut self,
        o: LuaValue,
        k: LuaValue,
        v: LuaValue,
    ) -> LuaResult<Option<bool>> {
        let mut cur = o;
        for _ in 0..LJ_MAX_IDXCHAIN {
            if let Some(t) = cur.as_table() {
                let tv = t.as_ref().get(k);
                if !tv.is_nil() {
                    t.as_mut().set(k, v);
                    return Ok(Some(true));
                }
                let mo = match meta_fast(self.l().global(), t.as_ref().metatable, MM::Newindex) {
                    Some(mo) => mo,
                    None => {
                        if k.is_nil() {
                            return Err(self.l().runtime_error(b"table index is nil"));
                        }
                        if let Some(n) = k.as_number()
                            && n.is_nan()
                        {
                            return Err(self.l().runtime_error(b"table index is NaN"));
                        }
                        t.as_mut().set(k, v);
                        return Ok(Some(true));
                    }
                };
                if mo.is_func() {
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            self.call_c_fn(cc.f, mo, &[cur, k, v])?;
                            return Ok(Some(true));
                        }
                        GcFunc::Lua(_) => {
                            self.mmcall_cont(Cont::Nop, 0, mo, &[cur, k, v]);
                            return Ok(None);
                        }
                    }
                }
                cur = mo;
            } else {
                let mo = meta_lookup(self.l().global(), cur, MM::Newindex);
                if mo.is_nil() {
                    return Err(self.index_error(cur));
                }
                if mo.is_func() {
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            self.call_c_fn(cc.f, mo, &[cur, k, v])?;
                            return Ok(Some(true));
                        }
                        GcFunc::Lua(_) => {
                            self.mmcall_cont(Cont::Nop, 0, mo, &[cur, k, v]);
                            return Ok(None);
                        }
                    }
                }
                cur = mo;
            }
        }
        Err(self
            .l()
            .runtime_error(b"'__newindex' chain too long; possible loop"))
    }

    /// `attempt to perform arithmetic on <kind> '<name>' (a <type> value)`,
    /// deducing the offending operand's slot from the arith instruction.
    fn arith_error(&self, rb: LuaValue, rc: LuaValue) -> LuaError {
        let pt = self.proto();
        let pc = self.pc;
        let msg = if pc >= 2 && pc - 1 < pt.bc.len() {
            let ins = pt.bc[pc - 1];
            let op = crate::bc::bc_op(ins);
            // The operand that cannot be coerced to a number is the one
            // reported (LuaJIT's `lj_meta_arith` slot).
            let rb_bad = str2num(self.l(), rb).is_none();
            let (bad, is_b) = if rb_bad { (rb, true) } else { (rc, false) };
            let slot = Self::arith_operand_slot(op, is_b, ins);
            if let Some(slot) = slot {
                debug_slotname(pt, pc - 1, slot).map(|(kind, name)| {
                    format!(
                        "attempt to perform arithmetic on {} '{}' (a {} value)",
                        kind,
                        name,
                        lua_typename(bad)
                    )
                })
            } else {
                None
            }
        } else {
            None
        };
        match msg {
            Some(m) => self.l().runtime_error(m.as_bytes()),
            None => self
                .l()
                .runtime_error(b"attempt to perform arithmetic on a non-number value"),
        }
    }

/// Map an arith instruction + which operand is bad to its register slot,
/// accounting for the bytecode's operand order (`*NV` puts the constant in
/// `c`, `*VN` the variable in `b`, VV/VV both in b/c).
fn arith_operand_slot(op: crate::bc::BCOp, is_b: bool, ins: BCIns) -> Option<u32> {
    use crate::bc::BCOp::*;
    match op {
        // Variable-variable (or unary).
        UNM | ADDVV | SUBVV | MULVV | DIVVV | MODVV | POW | BAND | BOR | BXOR | BSHL | BSHR => {
            Some(if is_b {
                crate::bc::bc_b(ins)
            } else {
                crate::bc::bc_c(ins)
            })
        }
        // Variable-constant: `rb` is the variable (b), `rc` is the constant.
        ADDVN | SUBVN | MULVN | DIVVN | MODVN => {
            if is_b {
                Some(crate::bc::bc_b(ins))
            } else {
                None
            }
        }
        // Constant-variable: `rb` is the constant (c), `rc` is the variable
        // (b) — the VM passes (kv, yv) with yv = fr.reg(bc_b).
        ADDNV | SUBNV | MULNV | DIVNV | MODNV => {
            if is_b {
                None
            } else {
                Some(crate::bc::bc_b(ins))
            }
        }
        _ => None,
    }
}

/// `lj_meta_arith`: coercion first, then arithmetic metamethod.
/// Returns `Some(val)` when resolved or `None` for Lua continuation.
pub(super) fn meta_arith(
        &mut self,
        mm: MM,
        rb: LuaValue,
        rc: LuaValue,
        a: u32,
    ) -> LuaResult<Option<LuaValue>> {
        if let (Some(b), Some(c)) = (str2num(self.l(), rb), str2num(self.l(), rc)) {
            return Ok(Some(LuaValue::number_raw(foldarith(mm, b, c))));
        }
        // Pointer arithmetic: a cdata +/- a number steps by the element
        // size (array/pointer cdata). The result is a cdata whose payload
        // is offset from the original.
        if let Some(cd) = rb.as_cdata()
            && let Some(n) = rc.as_number()
            && (mm == MM::Add || mm == MM::Sub)
            && let Some(v) = self.pointer_arith(cd, n, mm == MM::Add)
        {
            return Ok(Some(v));
        }
        if let Some(cd) = rc.as_cdata()
            && let Some(n) = rb.as_number()
            && mm == MM::Add
            && let Some(v) = self.pointer_arith(cd, n, true)
        {
            return Ok(Some(v));
        }
        // Pointer difference: `cdata - cdata` yields the element count.
        if mm == MM::Sub
            && let (Some(c1), Some(c2)) = (rb.as_cdata(), rc.as_cdata())
        {
            let g = self.l().global();
            if let Some(cts) = &g.cts
                && c1.as_ref().ctypeid == c2.as_ref().ctypeid
            {
                let raw = cts.raw(c1.as_ref().ctypeid);
                use crate::ffi::{ctype_cid, ctype_isarray, ctype_ispointer};
                if ctype_isarray(raw.info) || ctype_ispointer(raw.info) {
                    // Same storage → plain element difference of the
                    // alias offsets.
                    let (o1, s1) = crate::runtime::cdata::resolve_ptr(c1);
                    let (o2, s2) = crate::runtime::cdata::resolve_ptr(c2);
                    if s1 == s2 {
                        let elem_sz = cts.raw(ctype_cid(raw.info)).size as i64;
                        if elem_sz != 0 {
                            return Ok(Some(LuaValue::number(((o1 - o2) / elem_sz) as f64)));
                        }
                    }
                }
            }
        }
        let g = self.l().global();
        let mut mo = meta_lookup(g, rb, mm);
        if mo.is_nil() {
            mo = meta_lookup(g, rc, mm);
            if mo.is_nil() {
                return Err(self.arith_error(rb, rc));
            }
        }
        if mo.is_func() {
            match mo.as_func().unwrap().as_ref() {
                GcFunc::C(cc) => {
                    let v = self.call_c_fn(cc.f, mo, &[rb, rc])?;
                    return Ok(Some(v));
                }
                GcFunc::Lua(_) => {
                    self.mmcall_cont(Cont::Ra, a, mo, &[rb, rc]);
                    return Ok(None);
                }
            }
        }
        // Non-function metamethod: chain is unusual but valid — the
        // metamethod is itself called through its __call (the original
        // metamethod becomes the first argument).
        let Some(mo2) = resolve_callable(self.l(), mo) else {
            return Err(self.arith_error(rb, rc));
        };
        let args = [mo, rb, rc];
        match mo2.as_func().unwrap().as_ref() {
            GcFunc::C(cc) => {
                let v = self.call_c_fn(cc.f, mo2, &args)?;
                Ok(Some(v))
            }
            GcFunc::Lua(_) => {
                self.mmcall_cont(Cont::Ra, a, mo2, &args);
                Ok(None)
            }
        }
    }

    /// `lj_meta_comp`: ordered comparison slow path.  `op` follows the
    /// bytecode encoding: ISLT=0, ISGE=1, ISLE=2, ISGT=3.
    /// Returns `Some(cond)` when resolved inline, `None` for continuation.
    ///
    /// Only the first operand's metamethod is consulted (LuaJIT semantics;
    /// the second operand's metatable may differ). LE without `__le` falls
    /// back to `not (o2 < o1)` through the second operand's `__lt`.
    pub(super) fn meta_comp(
        &mut self,
        mut o1: LuaValue,
        mut o2: LuaValue,
        mut op: u32,
    ) -> LuaResult<Option<bool>> {
        if o1.itype() == o2.itype() || (o1.is_bool() && o2.is_bool()) {
            if o1.is_string() && o2.is_string() {
                let a_bytes = self.l().str_static(o1.as_string_id().unwrap());
                let b_bytes = self.l().str_static(o2.as_string_id().unwrap());
                let res = if (op & 2) != 0 {
                    a_bytes <= b_bytes
                } else {
                    a_bytes < b_bytes
                };
                return Ok(Some(res ^ ((op & 1) != 0)));
            }
            loop {
                let mm = if (op & 2) != 0 { MM::Le } else { MM::Lt };
                let g = self.l().global();
                // LuaJIT 2.1 (LJ_52) lj_meta_comp: the metamethod of the
                // first operand wins; only when both lack it does the
                // order error (mixed metamethods compare fine).
                let mut mo = meta_lookup(g, o1, mm);
                if mo.is_nil() {
                    mo = meta_lookup(g, o2, mm);
                }
                if mo.is_nil() {
                    if (op & 2) != 0 {
                        // MM_le not found: retry with MM_lt, swapped
                        // (i.e. `not (o2 < o1)`).
                        std::mem::swap(&mut o1, &mut o2);
                        op ^= 3;
                        continue;
                    }
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to compare incompatible values"));
                }
                let cont = if (op & 1) != 0 {
                    Cont::Condf
                } else {
                    Cont::Condt
                };
                if mo.is_func() {
                    let args = [o1, o2];
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            let v = self.call_c_fn(cc.f, mo, &args)?;
                            return Ok(Some(v.is_truthy() ^ ((op & 1) != 0)));
                        }
                        GcFunc::Lua(_) => {
                            self.mmcall_cont(cont, 0, mo, &args);
                            return Ok(None);
                        }
                    }
                }
                // Non-function metamethod: call it through its __call
                // (the metamethod value becomes the first argument).
                let Some(mo2) = resolve_callable(self.l(), mo) else {
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to compare incompatible values"));
                };
                let args = [mo, o1, o2];
                match mo2.as_func().unwrap().as_ref() {
                    GcFunc::C(cc) => {
                        let v = self.call_c_fn(cc.f, mo2, &args)?;
                        return Ok(Some(v.is_truthy() ^ ((op & 1) != 0)));
                    }
                    GcFunc::Lua(_) => {
                        self.mmcall_cont(cont, 0, mo2, &args);
                        return Ok(None);
                    }
                }
            }
        }
        Err(self
            .l()
            .runtime_error(b"attempt to compare incompatible values"))
    }

    /// `lj_meta_equal`: `__eq` for two tables/userdata that are not
    /// raw-equal. Returns `Some(is_equal)` or `None` for Lua continuation.
    /// `ne` is 0 for ISEQV, 1 for ISNEV (selects Condt/Condf).
    /// `cdata +/- number`: step by the element size of an array/pointer
    /// cdata. The result *aliases* the original storage (base + byte
    /// offset), so writes through the pointer land in the original object
    /// and pointer difference / comparisons resolve consistently.
    fn pointer_arith(
        &self,
        cd: crate::gc::GcPtr<crate::runtime::cdata::CData>,
        n: f64,
        add: bool,
    ) -> Option<LuaValue> {
        let g = self.l().global();
        let cts = g.cts.as_ref()?;
        let c = cd.as_ref();
        let raw = cts.raw(c.ctypeid);
        use crate::ffi::{ctype_cid, ctype_isarray, ctype_ispointer};
        if !ctype_isarray(raw.info) && !ctype_ispointer(raw.info) {
            return None; // Scalar/struct cdata: no pointer arithmetic.
        }
        let elem_sz = cts.raw(ctype_cid(raw.info)).size as i64;
        let mut delta = (n as i64) * elem_sz;
        if !add {
            delta = -delta;
        }
        // The alias chain: this cdata's own base/offset, else itself.
        let (base_off, root) = {
            let mut cur = cd;
            let mut off = 0i64;
            loop {
                let c = cur.as_ref();
                if let Some(b) = c.base {
                    off += c.offset;
                    cur = b;
                } else {
                    break;
                }
            }
            (off, cur)
        };
        let p = self
            .l()
            .global()
            .heap
            .cdatas
            .alloc(crate::runtime::cdata::CData {
                ctypeid: c.ctypeid,
                data: Box::new([]),
                base: Some(root),
                offset: base_off + delta,
            });
        Some(LuaValue::cdata(p))
    }

    pub(super) fn meta_equal(
        &mut self,
        o1: LuaValue,
        o2: LuaValue,
        ne: u32,
    ) -> LuaResult<Option<bool>> {
        let g = self.l().global();
        let mt1 = metatable_of(g, o1);
        let mo = match mt1.and_then(|mt| meta_fast(g, Some(mt), MM::Eq)) {
            Some(mo) => mo,
            None => return Ok(Some(false)),
        };
        if mt1 != metatable_of(g, o2) {
            match metatable_of(g, o2).and_then(|mt| meta_fast(g, Some(mt), MM::Eq)) {
                Some(mo2) if obj_equal(mo, mo2) => {}
                _ => return Ok(Some(false)),
            }
        }
        if mo.is_func() {
            let cont = if ne != 0 { Cont::Condf } else { Cont::Condt };
            match mo.as_func().unwrap().as_ref() {
                GcFunc::C(cc) => {
                    let v = self.call_c_fn(cc.f, mo, &[o1, o2])?;
                    let is_eq = v.is_truthy();
                    return Ok(Some(is_eq));
                }
                GcFunc::Lua(_) => {
                    self.mmcall_cont(cont, 0, mo, &[o1, o2]);
                    return Ok(None);
                }
            }
        }
        // Non-function metamethod: call it through its __call (the
        // metamethod value becomes the first argument).
        let Some(mo2) = resolve_callable(self.l(), mo) else {
            return Ok(Some(false));
        };
        let cont = if ne != 0 { Cont::Condf } else { Cont::Condt };
        let args = [mo, o1, o2];
        match mo2.as_func().unwrap().as_ref() {
            GcFunc::C(cc) => {
                let v = self.call_c_fn(cc.f, mo2, &args)?;
                let is_eq = v.is_truthy();
                Ok(Some(is_eq))
            }
            GcFunc::Lua(_) => {
                self.mmcall_cont(cont, 0, mo2, &args);
                Ok(None)
            }
        }
    }

    /// `lj_meta_len`: `__len` metamethod. Passes the object twice
    /// (LuaJIT 2.1 5.2-compat semantics). Returns `Some(len)` when
    /// resolved inline or `None` for a Lua continuation frame.
    pub(super) fn meta_len(&mut self, o: LuaValue, a: u32) -> LuaResult<Option<LuaValue>> {
        let mo = meta_lookup(self.l().global(), o, MM::Len);
        if mo.is_nil() {
            return Err(self
                .l()
                .runtime_error(b"attempt to get length of a non-table value"));
        }
        if mo.is_func() {
            let args = [o, o];
            match mo.as_func().unwrap().as_ref() {
                GcFunc::C(cc) => {
                    let v = self.call_c_fn(cc.f, mo, &args)?;
                    return Ok(Some(v));
                }
                GcFunc::Lua(_) => {
                    self.mmcall_cont(Cont::Ra, a, mo, &args);
                    return Ok(None);
                }
            }
        }
        // Non-function metamethod: call it through its __call (the
        // metamethod value becomes the first argument).
        let Some(mo2) = resolve_callable(self.l(), mo) else {
            return Err(self
                .l()
                .runtime_error(b"attempt to get length of a non-table value"));
        };
        let args = [mo, o, o];
        match mo2.as_func().unwrap().as_ref() {
            GcFunc::C(cc) => {
                let v = self.call_c_fn(cc.f, mo2, &args)?;
                Ok(Some(v))
            }
            GcFunc::Lua(_) => {
                self.mmcall_cont(Cont::Ra, a, mo2, &args);
                Ok(None)
            }
        }
    }

    /// `lj_meta_cat`: iterative concat over `b..=c` (absolute slots),
    /// right-to-left, with `__concat` (via execute recursion — cold).
    pub(super) fn meta_cat(&mut self, b: u32, c: u32) -> LuaResult<LuaValue> {
        let bottom = self.base + b as usize;
        let mut top = self.base + c as usize;
        loop {
            let o1 = self.at(top - 1);
            let o2 = self.at(top);
            if concat_ok(o1) && concat_ok(o2) {
                let mut o = top - 1;
                while o > bottom && concat_ok(self.at(o - 1)) {
                    o -= 1;
                }
                let mut buf: Vec<u8> = Vec::with_capacity(512);
                for i in o..=top {
                    let v = self.at(i);
                    if let Some(sid) = v.as_string_id() {
                        buf.extend_from_slice(self.l().str_static(sid));
                    } else {
                        let s = crate::strfmt::g14(v.num());
                        buf.extend_from_slice(s.as_bytes());
                    }
                }
                // LuaJIT's lj_str_new: strings longer than LJ_MAX_STR are
                // rejected with "string length overflow" (the concat would
                // otherwise allocate gigabytes before failing elsewhere).
                if buf.len() > 0xfffffe00 {
                    return Err(self.l().runtime_error(b"string length overflow"));
                }
                // Incremental FNV-1a for the common `s = s .. x` shape:
                // when the segment is exactly (previous result, x),
                // continue the stored stream state over x only.
                let heap = self.l().heap();
                let (state, hash) = if top == o + 1 {
                    if let Some(sid) = self.at(o).as_string_id()
                        && heap.cat_hash.is_some_and(|(id, _)| id == sid)
                    {
                        let st = heap.cat_hash.unwrap().1;
                        let st2 = crate::runtime::string::fnv1a_cont(
                            st,
                            buf.split_at(self.l().str_static(sid).len()).1,
                        );
                        (st2, crate::runtime::string::fnv1a_fold(st2))
                    } else {
                        let st = crate::runtime::string::fnv1a_state(&buf);
                        (st, crate::runtime::string::fnv1a_fold(st))
                    }
                } else {
                    let st = crate::runtime::string::fnv1a_state(&buf);
                    (st, crate::runtime::string::fnv1a_fold(st))
                };
                let sid = heap.intern_with_hash(&buf, hash);
                let v = self.l().heap().str_value(sid);
                self.l().heap().cat_hash = Some((sid, state));
                self.set_at(o, v);
                top = o;
            } else {
                let g = self.l().global();
                let mut mo = meta_lookup(g, o1, MM::Concat);
                if mo.is_nil() {
                    mo = meta_lookup(g, o2, MM::Concat);
                }
                if !mo.is_nil() {
                    // Non-function metamethods are called through their
                    // __call chain (the metamethod value becomes the
                    // first argument).
                    let (mo2, args, n) = if mo.is_func() {
                        (mo, [o1, o2, LuaValue::NIL], 2)
                    } else {
                        let Some(mo2) = resolve_callable(self.l(), mo) else {
                            return Err(self
                                .l()
                                .runtime_error(b"attempt to concatenate a non-string value"));
                        };
                        (mo2, [mo, o1, o2], 3)
                    };
                    let fs = self.mm_frame();
                    let st = self.l().top;
                    self.set_at(fs, mo2);
                    self.set_at(fs + 2, args[0]);
                    self.set_at(fs + 3, args[1]);
                    if n == 3 {
                        self.set_at(fs + 4, args[2]);
                    }
                    // Expose the metamethod name for debug.getinfo
                    // (the frame is a plain C-call frame).
                    let saved = self.l().mmname;
                    self.l().mmname = Some(("__concat", mo2.to_bits()));
                    let r = execute(self.l(), fs, n, 1);
                    self.l().mmname = saved;
                    let _r = r?;
                    self.sp = self.l().stack.as_mut_ptr();
                    let r = self.at(fs);
                    self.l().top = st;
                    top -= 1;
                    self.set_at(top, r);
                } else {
                    // No __concat: Lua 5.1 concat only accepts strings and
                    // numbers — report the offending operand (LuaJIT
                    // `lj_meta_cat`).
                    let bad_slot = if !concat_ok(o1) {
                        top - 1 - self.base
                    } else {
                        top - self.base
                    };
                    let bad = if !concat_ok(o1) { o1 } else { o2 };
                    let pt = self.proto();
                    let pc = self.pc;
                    let msg = if pc >= 2 && pc - 1 < pt.bc.len() {
                        debug_slotname(pt, pc - 1, bad_slot as u32).map(|(kind, name)| {
                            format!(
                                "attempt to concatenate {} '{}' (a {} value)",
                                kind,
                                name,
                                lua_typename(bad)
                            )
                        })
                    } else {
                        None
                    };
                    return Err(match msg {
                        Some(m) => self.l().runtime_error(m.as_bytes()),
                        None => self
                            .l()
                            .runtime_error(b"attempt to concatenate a non-string value"),
                    });
                }
            }
            if top <= bottom {
                return Ok(self.at(bottom));
            }
        }
    }
}

#[inline]
fn concat_ok(v: LuaValue) -> bool {
    v.is_string() || v.is_number()
}

#[inline]
fn obj_equal(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else {
        a.to_bits() == b.to_bits()
    }
}

fn str2num(l: &LuaState, o: LuaValue) -> Option<f64> {
    if let Some(n) = o.as_number() {
        return Some(n);
    }
    let sid = o.as_string_id()?;
    crate::strscan::scan_number(l.str_static(sid))
}

fn foldarith(mm: MM, b: f64, c: f64) -> f64 {
    match mm {
        MM::Add => b + c,
        MM::Sub => b - c,
        MM::Mul => b * c,
        MM::Div => b / c,
        MM::Mod => b - (b / c).floor() * c,
        MM::Pow => vm_pow(b, c),
        MM::Unm => -b,
        _ => unreachable!(),
    }
}

/// `lj_meta_call`: resolve `__call` for a non-function callee at
/// `func_slot`. Shifts the arguments up one slot, inserts the original
/// callee as the first argument and installs the metamethod as the callee.
/// Returns the new argument count.
pub(super) fn meta_call(l: &mut LuaState, func_slot: usize, nargs: usize) -> LuaResult<usize> {
    // __call frames are reported by their call-site name ("local t"),
    // not as a metamethod; clear any stale mmname first.
    l.mmname = None;
    let f = l.stack[func_slot];
    // The __call metamethod may itself be a non-function (a table with
    // its own __call, a primitive with a call metatable, ...). Resolve
    // the chain to the first function, like LuaJIT's repeated dispatch.
    let Some(mo) = resolve_callable(l, f) else {
        let msg = match debug_callname(l, func_slot) {
            Some((kind, name)) => format!(
                "attempt to call {} '{}' (a {} value)",
                kind,
                name,
                lua_typename(f)
            ),
            // Lua 5.1 fallback: report the callee's type
            // ("attempt to call a number value").
            None => format!("attempt to call a {} value", lua_typename(f)),
        };
        return Err(l.runtime_error(msg.as_bytes()));
    };
    for i in (0..nargs).rev() {
        l.stack[func_slot + 3 + i] = l.stack[func_slot + 2 + i];
    }
    l.stack[func_slot + 2] = f;
    l.stack[func_slot] = mo;
    Ok(nargs + 1)
}

/// Follow the `__call` chain of `v` (which must not be a function) to
/// the first function, e.g. a table whose `__call` is a table with a
/// `__call`, or a number whose `__call` comes from a debug-set metatable
/// that itself forwards. Returns None when the chain runs out (or
/// circles for more than a few levels).
fn resolve_callable(l: &LuaState, v: LuaValue) -> Option<LuaValue> {
    let mut mo = meta_lookup(l.global(), v, MM::Call);
    for _ in 0..4 {
        if mo.is_func() {
            return Some(mo);
        }
        if mo.is_nil() {
            return None;
        }
        mo = meta_lookup(l.global(), mo, MM::Call);
    }
    None
}
