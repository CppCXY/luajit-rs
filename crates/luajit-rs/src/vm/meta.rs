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
use crate::meta::{MM, meta_fast, meta_lookup};

const LJ_MAX_IDXCHAIN: usize = 100;

impl Interp {
    #[inline]
    fn mm_frame(&self) -> usize {
        self.base + self.lua_cl().proto.as_ref().framesize as usize
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
                            self.mmcall_cont(Cont::Ra, a, mo, &[cur, k]);
                            return Ok(None);
                        }
                    }
                }
                cur = mo;
            } else {
                let mo = meta_lookup(self.l().global(), cur, MM::Index);
                if mo.is_nil() {
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to index a non-table value"));
                }
                if mo.is_func() {
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            let v = self.call_c_fn(cc.f, mo, &[cur, k])?;
                            return Ok(Some(v));
                        }
                        GcFunc::Lua(_) => {
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
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to index a non-table value"));
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

    /// `lj_meta_arith`: coercion first, then arithmetic metamethod.
    /// Returns `Some(val)` when resolved or `None` for Lua continuation.
    #[cold]
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
        let g = self.l().global();
        let mut mo = meta_lookup(g, rb, mm);
        if mo.is_nil() {
            mo = meta_lookup(g, rc, mm);
            if mo.is_nil() {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to perform arithmetic on a non-number value"));
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
        // Non-function metamethod: chain is unusual but valid.
        // Retry by indexing the metamethod.
        Err(self
            .l()
            .runtime_error(b"attempt to perform arithmetic on a non-number value"))
    }

    /// `lj_meta_comp`: ordered comparison slow path.  `op` follows the
    /// bytecode encoding: ISLT=0, ISGE=1, ISLE=2, ISGT=3.
    /// Returns `Some(cond)` when resolved inline, `None` for continuation.
    #[cold]
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
                let mo = meta_lookup(g, o1, mm);
                let mo2 = meta_lookup(g, o2, mm);
                if mo.is_nil() || !obj_equal(mo, mo2) {
                    if (op & 2) != 0 {
                        // MM_le not found: retry with MM_lt, swapped.
                        std::mem::swap(&mut o1, &mut o2);
                        op ^= 3;
                        continue;
                    }
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to compare incompatible values"));
                }
                if mo.is_func() {
                    let cont = if (op & 1) != 0 {
                        Cont::Condf
                    } else {
                        Cont::Condt
                    };
                    match mo.as_func().unwrap().as_ref() {
                        GcFunc::C(cc) => {
                            let v = self.call_c_fn(cc.f, mo, &[o1, o2])?;
                            let cond = v.is_truthy();
                            return Ok(Some(cond ^ ((op & 1) != 0)));
                        }
                        GcFunc::Lua(_) => {
                            self.mmcall_cont(cont, 0, mo, &[o1, o2]);
                            return Ok(None);
                        }
                    }
                }
                // Non-function metamethod: shouldn't happen, error.
                return Err(self
                    .l()
                    .runtime_error(b"attempt to compare incompatible values"));
            }
        }
        Err(self
            .l()
            .runtime_error(b"attempt to compare incompatible values"))
    }

    /// `lj_meta_equal`: `__eq` for two tables that are not raw-equal.
    /// Returns `Some(is_equal)` or `None` for Lua continuation.  `ne` is
    /// 0 for ISEQV, 1 for ISNEV (selects Condt/Condf).
    #[cold]
    pub(super) fn meta_equal(
        &mut self,
        o1: LuaValue,
        o2: LuaValue,
        ne: u32,
    ) -> LuaResult<Option<bool>> {
        let t1 = o1.as_table().unwrap();
        let g = self.l().global();
        let mo = match meta_fast(g, t1.as_ref().metatable, MM::Eq) {
            Some(mo) => mo,
            None => return Ok(Some(false)),
        };
        if t1.as_ref().metatable != o2.as_table().unwrap().as_ref().metatable {
            match meta_fast(g, o2.as_table().unwrap().as_ref().metatable, MM::Eq) {
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
        Ok(Some(false))
    }

    /// `lj_meta_len`: `__len` metamethod (via execute recursion — cold).
    #[cold]
    pub(super) fn meta_len(&mut self, o: LuaValue) -> LuaResult<LuaValue> {
        let mo = meta_lookup(self.l().global(), o, MM::Len);
        if mo.is_nil() {
            return Err(self
                .l()
                .runtime_error(b"attempt to get length of a non-table value"));
        }
        let fs = self.mm_frame();
        let st = self.l().top;
        self.set_at(fs, mo);
        self.set_at(fs + 2, o);
        execute(self.l(), fs, 1, 1)?;
        let r = self.at(fs);
        self.l().top = st;
        Ok(r)
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
                let sid = self.l().heap().intern(&buf);
                let v = self.l().heap().str_value(sid);
                self.set_at(o, v);
                top = o;
            } else {
                let g = self.l().global();
                let mut mo = meta_lookup(g, o1, MM::Concat);
                if mo.is_nil() {
                    mo = meta_lookup(g, o2, MM::Concat);
                    if mo.is_nil() {
                        return Err(self
                            .l()
                            .runtime_error(b"attempt to concatenate a non-string value"));
                    }
                }
                let fs = self.mm_frame();
                let st = self.l().top;
                self.set_at(fs, mo);
                self.set_at(fs + 2, o1);
                self.set_at(fs + 3, o2);
                execute(self.l(), fs, 2, 1)?;
                let r = self.at(fs);
                self.l().top = st;
                top -= 1;
                self.set_at(top, r);
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
    let f = l.stack[func_slot];
    let mo = meta_lookup(l.global(), f, MM::Call);
    if !mo.is_func() {
        return Err(l.runtime_error(b"attempt to call a non-function value"));
    }
    for i in (0..nargs).rev() {
        l.stack[func_slot + 3 + i] = l.stack[func_slot + 2 + i];
    }
    l.stack[func_slot + 2] = f;
    l.stack[func_slot] = mo;
    Ok(nargs + 1)
}
