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
        // Non-function metamethod: chain is unusual but valid — the
        // metamethod is itself called through its __call (the original
        // metamethod becomes the first argument).
        let Some(mo2) = resolve_callable(self.l(), mo) else {
            return Err(self
                .l()
                .runtime_error(b"attempt to perform arithmetic on a non-number value"));
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
                let mo = meta_lookup(g, o1, mm);
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
                let sid = self.l().heap().intern(&buf);
                let v = self.l().heap().str_value(sid);
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
                    // No __concat available. Try __tostring on non-string operands
                    // and replace them, then continue the loop.
                    let mut replaced = false;
                    if !concat_ok(o1) {
                        let s = crate::stdlib::tostring_meta(self.l(), o1)?;
                        let sid = self.l().heap().intern(&s);
                        self.set_at(top - 1, self.l().heap().str_value(sid));
                        replaced = true;
                    }
                    if !concat_ok(o2) {
                        let s = crate::stdlib::tostring_meta(self.l(), o2)?;
                        let sid = self.l().heap().intern(&s);
                        self.set_at(top, self.l().heap().str_value(sid));
                        replaced = true;
                    }
                    if !replaced {
                        return Err(self
                            .l()
                            .runtime_error(b"attempt to concatenate a non-string value"));
                    }
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
        return Err(l.runtime_error(b"attempt to call a non-function value"));
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
