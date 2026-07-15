use crate::lex::StrId;

/// A non-boxed, NaN-boxed 64-bit Lua value, mirroring LuaJIT's TValue
/// encoding: the high 32 bits hold `!itype` for tagged values, while any
/// other bit pattern is a raw IEEE-754 double.
///
/// Tagged values occupy the (sign + quiet-NaN + high mantissa) space that
/// no arithmetic result ever produces, so numbers need no tag at all.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LuaValue(u64);

const TAG_NIL: u32 = !0;
const TAG_FALSE: u32 = !1;
const TAG_TRUE: u32 = !2;
const TAG_STR: u32 = !4;
const TAG_TAB: u32 = !11;
const TAG_MIN: u32 = TAG_TAB;

const fn tagged(tag: u32, payload: u32) -> u64 {
    ((tag as u64) << 32) | payload as u64
}

impl LuaValue {
    pub const NIL: LuaValue = LuaValue(tagged(TAG_NIL, 0));
    pub const FALSE: LuaValue = LuaValue(tagged(TAG_FALSE, 0));
    pub const TRUE: LuaValue = LuaValue(tagged(TAG_TRUE, 0));

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
            0 // normalize -0.0 and +0.0 to the same key
        } else {
            n.to_bits()
        };
        debug_assert!((bits >> 32) < TAG_MIN as u64);
        LuaValue(bits)
    }

    pub fn string(sid: StrId) -> LuaValue {
        LuaValue(tagged(TAG_STR, sid))
    }

    /// Placeholder reference used by template tables to preserve keys whose
    /// value is only known at runtime (LuaJIT stores the table itself).
    pub fn table_marker() -> LuaValue {
        LuaValue(tagged(TAG_TAB, 0))
    }

    fn tag(self) -> u32 {
        (self.0 >> 32) as u32
    }

    pub fn is_nil(self) -> bool {
        self.tag() == TAG_NIL
    }

    pub fn is_number(self) -> bool {
        self.tag() < TAG_MIN
    }

    pub fn is_string(self) -> bool {
        self.tag() == TAG_STR
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
            Some(self.0 as u32)
        } else {
            None
        }
    }

    pub fn to_bits(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for LuaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.tag() {
            TAG_NIL => write!(f, "nil"),
            TAG_FALSE => write!(f, "false"),
            TAG_TRUE => write!(f, "true"),
            TAG_STR => write!(f, "str#{}", self.0 as u32),
            TAG_TAB => write!(f, "table"),
            _ => write!(f, "{}", f64::from_bits(self.0)),
        }
    }
}
