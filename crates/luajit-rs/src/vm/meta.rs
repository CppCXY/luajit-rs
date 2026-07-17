//! VM metamethod helpers, ported from LuaJIT's `lj_meta.c`.
//!
//! Divergence note: LuaJIT sets up a continuation frame (`FRAME_CONT` +
//! `lj_cont_ra`/`lj_cont_cond*`/`lj_cont_cat`) and lets the assembler VM run
//! the metamethod on the same dispatch loop. Here a metamethod call recurses
//! through `execute()` (like a C-frame call): the semantics — lookup order,
//! `__index`/`__newindex` chains, coercions, error cases — follow `lj_meta.c`
//! exactly, only the call mechanism differs.

use super::*;
use crate::meta::{MM, meta_fast, meta_lookup};

/// `__index`/`__newindex` chain limit (`LJ_MAX_IDXCHAIN`).
const LJ_MAX_IDXCHAIN: usize = 100;

impl Interp {
    /// Scratch slot just above the current frame, where metamethod call
    /// frames are built (LuaJIT's `mmcall` builds them at `curr_topL`).
    #[inline]
    fn mm_frame(&self) -> usize {
        self.base + self.proto().framesize as usize
    }

    /// `mmcall` + continuation `lj_cont_ra`: call `mo(a, b)` and return its
    /// single result.
    pub(super) fn mmcall(&mut self, mo: LuaValue, a: LuaValue, b: LuaValue) -> LuaResult<LuaValue> {
        let fs = self.mm_frame();
        let saved_top = self.l().top;
        self.set_at(fs, mo);
        self.set_at(fs + 1, LuaValue::NIL);
        self.set_at(fs + 2, a);
        self.set_at(fs + 3, b);
        execute(self.l(), fs, 2, 1)?;
        let r = self.at(fs);
        self.l().top = saved_top;
        Ok(r)
    }

    /// `mmcall` + `lj_cont_nop`: call `mo(a, b, c)` for `__newindex`.
    fn mmcall3(&mut self, mo: LuaValue, a: LuaValue, b: LuaValue, c: LuaValue) -> LuaResult<()> {
        let fs = self.mm_frame();
        let saved_top = self.l().top;
        self.set_at(fs, mo);
        self.set_at(fs + 1, LuaValue::NIL);
        self.set_at(fs + 2, a);
        self.set_at(fs + 3, b);
        self.set_at(fs + 4, c);
        execute(self.l(), fs, 3, 0)?;
        self.l().top = saved_top;
        Ok(())
    }

    /// `lj_meta_tget`: `__index` chain and metamethod.
    pub(super) fn meta_tget(&mut self, mut o: LuaValue, k: LuaValue) -> LuaResult<LuaValue> {
        for _ in 0..LJ_MAX_IDXCHAIN {
            let mo = if let Some(t) = o.as_table() {
                let tv = t.as_ref().get(k);
                if !tv.is_nil() {
                    return Ok(tv);
                }
                match meta_fast(self.l().global(), t.as_ref().metatable, MM::Index) {
                    Some(mo) => mo,
                    None => return Ok(LuaValue::NIL),
                }
            } else {
                let mo = meta_lookup(self.l().global(), o, MM::Index);
                if mo.is_nil() {
                    return Err(self.l().runtime_error(b"attempt to index a non-table value"));
                }
                mo
            };
            if mo.is_func() {
                return self.mmcall(mo, o, k);
            }
            o = mo;
        }
        Err(self.l().runtime_error(b"'__index' chain too long; possible loop"))
    }

    /// `lj_meta_tset`: `__newindex` chain and metamethod.
    pub(super) fn meta_tset(&mut self, mut o: LuaValue, k: LuaValue, v: LuaValue) -> LuaResult<()> {
        for _ in 0..LJ_MAX_IDXCHAIN {
            let mo = if let Some(t) = o.as_table() {
                let tv = t.as_ref().get(k);
                if !tv.is_nil() {
                    // Key exists: raw update (nomm invalidated inside set).
                    t.as_mut().set(k, v);
                    return Ok(());
                }
                match meta_fast(self.l().global(), t.as_ref().metatable, MM::Newindex) {
                    Some(mo) => mo,
                    None => {
                        // No metamethod: raw insert, with key checks
                        // (lj_err NILIDX/NANIDX).
                        if k.is_nil() {
                            return Err(self.l().runtime_error(b"table index is nil"));
                        }
                        if let Some(n) = k.as_number()
                            && n.is_nan()
                        {
                            return Err(self.l().runtime_error(b"table index is NaN"));
                        }
                        t.as_mut().set(k, v);
                        return Ok(());
                    }
                }
            } else {
                let mo = meta_lookup(self.l().global(), o, MM::Newindex);
                if mo.is_nil() {
                    return Err(self.l().runtime_error(b"attempt to index a non-table value"));
                }
                mo
            };
            if mo.is_func() {
                return self.mmcall3(mo, o, k, v);
            }
            o = mo;
        }
        Err(self
            .l()
            .runtime_error(b"'__newindex' chain too long; possible loop"))
    }

    /// `lj_meta_arith`: string-to-number coercion first, then the arithmetic
    /// metamethod of either operand.
    #[cold]
    pub(super) fn meta_arith(
        &mut self,
        mm: MM,
        rb: LuaValue,
        rc: LuaValue,
    ) -> LuaResult<LuaValue> {
        if let (Some(b), Some(c)) = (str2num(self.l(), rb), str2num(self.l(), rc)) {
            // Try coercion first (lj_vm_foldarith).
            return Ok(LuaValue::number_raw(foldarith(mm, b, c)));
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
        self.mmcall(mo, rb, rc)
    }

    /// `lj_meta_len`: `__len` metamethod (5.1: never consulted for tables).
    #[cold]
    pub(super) fn meta_len(&mut self, o: LuaValue) -> LuaResult<LuaValue> {
        let mo = meta_lookup(self.l().global(), o, MM::Len);
        if mo.is_nil() {
            return Err(self
                .l()
                .runtime_error(b"attempt to get length of a non-table value"));
        }
        self.mmcall(mo, o, LuaValue::NIL)
    }

    /// `lj_meta_comp`: ordered comparison slow path. `op` follows the
    /// bytecode encoding: ISLT=0, ISGE=1, ISLE=2, ISGT=3. Returns the
    /// branch condition.
    #[cold]
    pub(super) fn meta_comp(
        &mut self,
        mut o1: LuaValue,
        mut o2: LuaValue,
        mut op: u32,
    ) -> LuaResult<bool> {
        if o1.itype() == o2.itype() || (o1.is_bool() && o2.is_bool()) {
            // Never called with two numbers.
            if o1.is_string() && o2.is_string() {
                let a = self.l().str_static(o1.as_string_id().unwrap());
                let b = self.l().str_static(o2.as_string_id().unwrap());
                let res = if (op & 2) != 0 { a <= b } else { a < b };
                return Ok(res ^ ((op & 1) != 0));
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
                        op ^= 3; // Use LT and flip condition.
                        continue;
                    }
                    return Err(self
                        .l()
                        .runtime_error(b"attempt to compare incompatible values"));
                }
                let r = self.mmcall(mo, o1, o2)?;
                return Ok(r.is_truthy() ^ ((op & 1) != 0));
            }
        }
        Err(self
            .l()
            .runtime_error(b"attempt to compare incompatible values"))
    }

    /// `lj_meta_equal`: `__eq` for two tables that are not raw-equal.
    /// Returns the equality result (caller applies ISNEV negation).
    #[cold]
    pub(super) fn meta_equal(&mut self, o1: LuaValue, o2: LuaValue) -> LuaResult<bool> {
        let t1 = o1.as_table().unwrap();
        let t2 = o2.as_table().unwrap();
        let g = self.l().global();
        let mo = match meta_fast(g, t1.as_ref().metatable, MM::Eq) {
            Some(mo) => mo,
            None => return Ok(false),
        };
        if t1.as_ref().metatable != t2.as_ref().metatable {
            // Field metatable must agree: both operands need the same __eq.
            match meta_fast(g, t2.as_ref().metatable, MM::Eq) {
                Some(mo2) if obj_equal(mo, mo2) => {}
                _ => return Ok(false),
            }
        }
        let r = self.mmcall(mo, o1, o2)?;
        Ok(r.is_truthy())
    }

    /// `lj_meta_cat`: iterative concat over the CAT range `b..=c` (absolute
    /// stack slots relative to base), right-to-left, with `__concat`.
    pub(super) fn meta_cat(&mut self, b: u32, c: u32) -> LuaResult<LuaValue> {
        let bottom = self.base + b as usize;
        let mut top = self.base + c as usize;
        loop {
            let o1 = self.at(top - 1);
            let o2 = self.at(top);
            if concat_ok(o1) && concat_ok(o2) {
                // Pick as many strings as possible from the top and
                // concatenate them.
                let mut o = top - 1;
                while o > bottom && concat_ok(self.at(o - 1)) {
                    o -= 1;
                }
                let mut buf = Vec::new();
                for i in o..=top {
                    let v = self.at(i);
                    if let Some(sid) = v.as_string_id() {
                        buf.extend_from_slice(self.l().str_static(sid));
                    } else {
                        buf.extend_from_slice(crate::strfmt::g14(v.num()).as_bytes());
                    }
                }
                let sid = self.l().heap().intern(&buf);
                let v = self.l().heap().str_value(sid);
                self.set_at(o, v);
                top = o;
            } else {
                // One of the top two elements is not a string: __concat.
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
                let r = self.mmcall(mo, o1, o2)?;
                top -= 1;
                self.set_at(top, r);
            }
            if top <= bottom {
                return Ok(self.at(bottom));
            }
        }
    }
}

/// `tvisstr(o) || tvisnumber(o)`: directly concatenable.
#[inline]
fn concat_ok(v: LuaValue) -> bool {
    v.is_string() || v.is_number()
}

/// `lj_obj_equal` on metamethod values: numbers by value, others by bits.
#[inline]
fn obj_equal(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else {
        a.to_bits() == b.to_bits()
    }
}

/// `str2num`: numbers pass through, strings are scanned (`lj_strscan_num`).
fn str2num(l: &LuaState, o: LuaValue) -> Option<f64> {
    if let Some(n) = o.as_number() {
        return Some(n);
    }
    let sid = o.as_string_id()?;
    crate::strscan::scan_number(l.str_static(sid))
}

/// `lj_vm_foldarith` for the ORDER ARITH metamethods.
fn foldarith(mm: MM, b: f64, c: f64) -> f64 {
    match mm {
        MM::Add => b + c,
        MM::Sub => b - c,
        MM::Mul => b * c,
        MM::Div => b / c,
        MM::Mod => b - (b / c).floor() * c,
        MM::Pow => b.powf(c),
        MM::Unm => -b,
        _ => unreachable!("bad arith metamethod {:?}", mm),
    }
}

/// `lj_meta_call`: resolve `__call` for a non-function callee at
/// `func_slot`. Shifts the arguments up one slot, inserts the original
/// callee as the first argument and installs the metamethod as the callee.
/// Returns the new argument count.
pub(super) fn meta_call(
    l: &mut LuaState,
    func_slot: usize,
    nargs: usize,
) -> LuaResult<usize> {
    let f = l.stack[func_slot];
    let mo = meta_lookup(l.global(), f, MM::Call);
    if !mo.is_func() {
        return Err(l.runtime_error(b"attempt to call a non-function value"));
    }
    // for (p = top; p > func; p--) copyTV(L, p, p-1);
    for i in (0..nargs).rev() {
        l.stack[func_slot + 3 + i] = l.stack[func_slot + 2 + i];
    }
    l.stack[func_slot + 2] = f;
    l.stack[func_slot] = mo;
    Ok(nargs + 1)
}
