use crate::string::{LuaString, StrId};

/// A non-boxed 64-bit Lua value, bit-identical to LuaJIT's LJ_GC64 TValue
/// encoding (lj_obj.h):
///
/// ```text
///                     ------MSW------.------LSW------
/// primitive types    |1..1|itype|1..................1|
/// GC objects         |1..1|itype|-------GCRef--------|
/// number              ------------double-------------
/// ```
///
/// The upper 13 bits of a tagged value are all ones (a quiet NaN with sign),
/// the next 4 bits hold the internal tag, and the low 47 bits hold the
/// payload (all ones for primitives). Numbers are raw doubles and need no
/// tag. The itype is recovered with an arithmetic shift by 47, yielding the
/// full `LJ_T*` constant.
///
/// Divergence note: until a real GC exists, the 47-bit GC payload holds an
/// interned string id (or 0 for the template-table marker) instead of a
/// pointer. The encoding itself is unchanged.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LuaValue(u64);

/* Internal object tags, ORDER LJ_T (lj_obj.h). */
pub const LJ_TNIL: u32 = !0;
pub const LJ_TFALSE: u32 = !1;
pub const LJ_TTRUE: u32 = !2;
pub const LJ_TLIGHTUD: u32 = !3;
pub const LJ_TSTR: u32 = !4;
pub const LJ_TUPVAL: u32 = !5;
pub const LJ_TTHREAD: u32 = !6;
pub const LJ_TPROTO: u32 = !7;
pub const LJ_TFUNC: u32 = !8;
pub const LJ_TTRACE: u32 = !9;
pub const LJ_TCDATA: u32 = !10;
pub const LJ_TTAB: u32 = !11;
pub const LJ_TUDATA: u32 = !12;
pub const LJ_TNUMX: u32 = !13;

/// Integers have itype == LJ_TISNUM, doubles have itype < LJ_TISNUM.
pub const LJ_TISNUM: u32 = LJ_TNUMX;

/// Mask for the 47-bit GC payload (`LJ_GCVMASK`).
pub const LJ_GCVMASK: u64 = (1u64 << 47) - 1;

/// Primitive encoding, mirroring `setpriV`: `~((u64)~itype << 47)`.
const fn pri(itype: u32) -> u64 {
    !(((!itype) as u64) << 47)
}

/// GC object encoding, mirroring `setgcVraw`: `payload | ((u64)itype << 47)`.
const fn gcv(itype: u32, payload: u64) -> u64 {
    payload | ((itype as u64) << 47)
}

/// Bit scramble for hash keys, ported from LuaJIT's `hashrot`.
fn hashrot(mut lo: u32, mut hi: u32) -> u32 {
    lo ^= hi;
    hi = hi.rotate_left(14);
    lo = lo.wrapping_sub(hi);
    hi = hi.rotate_left(5);
    hi ^= lo;
    hi.wrapping_sub(lo.rotate_left(13))
}

impl LuaValue {
    pub const NIL: LuaValue = LuaValue(pri(LJ_TNIL));
    pub const FALSE: LuaValue = LuaValue(pri(LJ_TFALSE));
    pub const TRUE: LuaValue = LuaValue(pri(LJ_TTRUE));

    pub fn boolean(b: bool) -> LuaValue {
        if b {
            LuaValue::TRUE
        } else {
            LuaValue::FALSE
        }
    }

    pub fn number(n: f64) -> LuaValue {
        let bits = if n.is_nan() {
            f64::NAN.to_bits()
        } else if n == 0.0 {
            0 // normalize -0.0, like lj_tab_newkey's tvismzero check
        } else {
            n.to_bits()
        };
        LuaValue(bits)
    }

    pub fn string(s: &LuaString) -> LuaValue {
        LuaValue(gcv(LJ_TSTR, s.sid() as u64))
    }

    /// Placeholder reference used by template tables to preserve keys whose
    /// value is only known at runtime (LuaJIT stores the table itself).
    pub fn table_marker() -> LuaValue {
        LuaValue(gcv(LJ_TTAB, 0))
    }

    /// Recover the internal tag (`itype`): arithmetic shift by 47.
    pub fn itype(self) -> u32 {
        ((self.0 as i64) >> 47) as u32
    }

    pub fn is_nil(self) -> bool {
        self.itype() == LJ_TNIL
    }

    pub fn is_false(self) -> bool {
        self.itype() == LJ_TFALSE
    }

    pub fn is_true(self) -> bool {
        self.itype() == LJ_TTRUE
    }

    pub fn is_bool(self) -> bool {
        self.is_false() || self.is_true()
    }

    /// `tvisnumber`: itype <= LJ_TISNUM.
    pub fn is_number(self) -> bool {
        self.itype() <= LJ_TISNUM
    }

    pub fn is_string(self) -> bool {
        self.itype() == LJ_TSTR
    }

    pub fn as_number(self) -> Option<f64> {
        if self.is_number() {
            Some(f64::from_bits(self.0))
        } else {
            None
        }
    }

    pub fn as_string(self) -> Option<StrId> {
        if self.is_string() {
            Some((self.0 & LJ_GCVMASK) as u32)
        } else {
            None
        }
    }

    /// Exact conversion to int32, mirroring `lj_vm_num2int_check` semantics.
    pub fn as_int32_exact(self) -> Option<i32> {
        let n = self.as_number()?;
        let k = n as i32;
        if (k as f64) == n {
            Some(k)
        } else {
            None
        }
    }

    pub fn to_bits(self) -> u64 {
        self.0
    }

    /// Hash for use as a table key, ported from LuaJIT's `hashkey`:
    /// strings hash by their interned `sid` (`hashstr`), numbers by
    /// `hashrot` over their bit halves (`hashnum`), booleans map to 0/1
    /// (`hashmask(boolV)`), other GC objects by their payload (`hashgcref`).
    pub fn hash_key(self) -> u32 {
        if self.is_string() {
            (self.0 & LJ_GCVMASK) as u32
        } else if self.is_number() {
            hashrot(self.0 as u32, ((self.0 >> 32) as u32) << 1)
        } else if self.is_bool() {
            LJ_TFALSE - self.itype() // boolV: false -> 0, true -> 1
        } else {
            let payload = self.0 & LJ_GCVMASK;
            hashrot(payload as u32, (payload >> 32) as u32)
        }
    }
}

impl std::fmt::Debug for LuaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.itype() {
            LJ_TNIL => write!(f, "nil"),
            LJ_TFALSE => write!(f, "false"),
            LJ_TTRUE => write!(f, "true"),
            LJ_TSTR => write!(f, "str#{}", self.0 & LJ_GCVMASK),
            LJ_TTAB => write!(f, "table#{:#x}", self.0 & LJ_GCVMASK),
            _ => write!(f, "{}", f64::from_bits(self.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string::Interner;

    #[test]
    fn primitive_bit_patterns_match_luajit() {
        // setnilV: it64 = -1.
        assert_eq!(LuaValue::NIL.to_bits(), u64::MAX);
        // setpriV(LJ_TFALSE): ~((u64)1 << 47).
        assert_eq!(LuaValue::FALSE.to_bits(), !(1u64 << 47));
        // setpriV(LJ_TTRUE): ~((u64)2 << 47).
        assert_eq!(LuaValue::TRUE.to_bits(), !(2u64 << 47));
    }

    #[test]
    fn itype_recovery() {
        assert_eq!(LuaValue::NIL.itype(), LJ_TNIL);
        assert_eq!(LuaValue::FALSE.itype(), LJ_TFALSE);
        assert_eq!(LuaValue::TRUE.itype(), LJ_TTRUE);
        let mut strs = Interner::default();
        let sid = strs.intern(b"x");
        let v = LuaValue::string(strs.lookup(sid));
        assert_eq!(v.itype(), LJ_TSTR);
        assert_eq!(v.as_string(), Some(sid));
        assert_eq!(LuaValue::table_marker().itype(), LJ_TTAB);
    }

    #[test]
    fn numbers_are_raw_doubles() {
        let v = LuaValue::number(3.25);
        assert_eq!(v.to_bits(), 3.25f64.to_bits());
        assert!(v.is_number());
        assert_eq!(v.as_number(), Some(3.25));
        // Negative doubles (incl. -inf) stay below the tagged NaN space.
        assert!(LuaValue::number(f64::NEG_INFINITY).is_number());
        assert!(LuaValue::number(-1e308).is_number());
        assert!(LuaValue::number(f64::NAN).is_number());
        assert!(!LuaValue::number(1.0).is_string());
        // -0.0 is normalized for table-key identity.
        assert_eq!(LuaValue::number(-0.0).to_bits(), 0);
    }

    #[test]
    fn string_tag_encoding_matches_gc64() {
        let mut strs = Interner::default();
        let sid = strs.intern(b"abc");
        let v = LuaValue::string(strs.lookup(sid));
        assert_eq!(v.to_bits(), (sid as u64) | ((LJ_TSTR as u64) << 47));
    }
}
