//! Metamethod definitions, ported from LuaJIT's `lj_obj.h` (ORDER MM) and
//! `lj_meta.c` (`lj_meta_init`).
//!
//! The first six metamethods (up to `MM_FAST` = `len`) are negative-cached
//! in the metatable's `nomm` bitmask, exactly as in LuaJIT: a set bit means
//! "this metatable has no such metamethod", so the hash lookup is skipped.
//! Any string-key write into a table clears its `nomm` cache.

use crate::gc::GcPtr;
use crate::state::GlobalState;
use crate::table::LuaTable;
use crate::value::LuaValue;

/// Metamethods, ORDER MM (lj_obj.h `MMDEF`). No FFI / 5.2 entries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MM {
    Index,
    Newindex,
    Gc,
    Mode,
    Eq,
    Len,
    // Only the above (fast) metamethods are negative cached (max. 8).
    Lt,
    Le,
    Concat,
    Call,
    // The following must be in ORDER ARITH.
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Unm,
    // The following are used in the standard libraries.
    Metatable,
    Tostring,
}

pub const MM_MAX: usize = MM::Tostring as usize + 1;

/// Last negative-cached metamethod (`MM_FAST = MM_len`).
pub const MM_FAST: u8 = MM::Len as u8;

/// Metamethod names, in ORDER MM.
pub const MM_NAMES: [&[u8]; MM_MAX] = [
    b"__index",
    b"__newindex",
    b"__gc",
    b"__mode",
    b"__eq",
    b"__len",
    b"__lt",
    b"__le",
    b"__concat",
    b"__call",
    b"__add",
    b"__sub",
    b"__mul",
    b"__div",
    b"__mod",
    b"__pow",
    b"__unm",
    b"__metatable",
    b"__tostring",
];

/// The metatable of a value: table's own, else the per-type base metatable
/// (`lj_meta_lookup`'s dispatch on the object type).
#[inline]
pub fn metatable_of(g: &GlobalState, o: LuaValue) -> Option<GcPtr<LuaTable>> {
    if let Some(t) = o.as_table() {
        t.as_ref().metatable
    } else {
        g.basemt_of(o.itype())
    }
}

/// `lj_meta_lookup`: the metamethod value for an object, or nil.
#[inline]
pub fn meta_lookup(g: &GlobalState, o: LuaValue, mm: MM) -> LuaValue {
    match metatable_of(g, o) {
        Some(mt) => mt.as_ref().get_str(g.mmname[mm as usize]),
        None => LuaValue::NIL,
    }
}

/// `lj_meta_fast` + `lj_meta_cache`: negative-cached lookup of a fast
/// metamethod (`mm <= MM_FAST`) in a metatable. Returns `None` when absent
/// and records the miss in the metatable's `nomm` bitmask.
#[inline]
pub fn meta_fast(g: &GlobalState, mt: Option<GcPtr<LuaTable>>, mm: MM) -> Option<LuaValue> {
    debug_assert!(mm as u8 <= MM_FAST, "bad fast metamethod {:?}", mm);
    let mt = mt?;
    if mt.as_ref().nomm & (1u8 << (mm as u8)) != 0 {
        return None; // Negative cache hit.
    }
    let mo = mt.as_ref().get_str(g.mmname[mm as usize]);
    if mo.is_nil() {
        mt.as_mut().nomm |= 1u8 << (mm as u8); // Set negative cache flag.
        None
    } else {
        Some(mo)
    }
}
