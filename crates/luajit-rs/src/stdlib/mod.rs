//! LuaJIT-compatible standard library.
//!
//! Each sub-module provides an `open(l: &mut LuaState)` function that
//! registers its functions, plus a `CFn`-typed entry for `make_lib`.
//! The `crate::func::CFunction` signature is `fn(&mut LuaState) ->
//! LuaResult<i32>`.  String args are read through
//! `LuaState::str_static` (zero-copy via pool-stable `'static` lifetimes)
//! so C functions never need to clone read-only string data.

pub mod base;
pub mod math;
pub mod os;
pub mod pattern;
pub mod reg;
pub mod sort;
pub mod string;
pub mod table;

pub use reg::{LibBuilder, LibTarget};

use crate::state::LuaState;
use crate::value::LuaValue;

/// `lua_push` a single result value.
#[inline]
pub fn push(l: &mut LuaState, v: LuaValue) {
    l.stack[l.base] = v;
    l.top = l.base + 1;
}

/// `lua_push` multiple results.
#[inline]
pub fn pushv(l: &mut LuaState, vs: &[LuaValue]) {
    for (i, &v) in vs.iter().enumerate() {
        l.stack[l.base + i] = v;
    }
    l.top = l.base + vs.len();
}

/// Argument `i` (0-based) of the current C call, or nil.
#[inline]
pub fn arg(l: &LuaState, i: usize) -> LuaValue {
    let slot = l.base + i;
    if slot < l.top {
        l.stack[slot]
    } else {
        LuaValue::NIL
    }
}

/// Number of arguments to the current C call.
#[inline]
pub fn nargs(l: &LuaState) -> usize {
    l.top - l.base
}

/// Convert a value to its display bytes (`tostring` without metamethods).
pub fn tostring_bytes(l: &mut LuaState, v: LuaValue) -> Vec<u8> {
    if let Some(sid) = v.as_string_id() {
        return l.heap().strings.get(sid).to_vec();
    }
    if let Some(n) = v.as_number() {
        return crate::strfmt::g14(n).into_bytes();
    }
    if v.is_nil() {
        return b"nil".to_vec();
    }
    if v.is_true() {
        return b"true".to_vec();
    }
    if v.is_false() {
        return b"false".to_vec();
    }
    let kind = if v.is_table() {
        "table"
    } else if v.is_func() {
        "function"
    } else {
        "userdata"
    };
    format!("{}: {:#x}", kind, v.gc_addr()).into_bytes()
}

/// Builtin error: `bad argument #N to 'func' (expected, got)`.
pub fn err_bad_arg(
    l: &mut LuaState,
    n: u32,
    func: &str,
    expected: &str,
    got: &str,
) -> crate::err::LuaError {
    let msg = format!(
        "bad argument #{} to '{}' ({} expected, got {})",
        n, func, expected, got
    );
    l.runtime_error(msg.as_bytes())
}

/// Create a named global table filled with C functions.
pub fn make_lib(l: &mut LuaState, name: &[u8], entries: &[(&[u8], crate::func::CFunction)]) {
    use crate::func::{CClosure, GcFunc};
    let t = l.heap().alloc_table(crate::table::LuaTable::new(
        0,
        (entries.len() as u32).next_power_of_two().trailing_zeros() as u32,
    ));
    for &(field, f) in entries {
        let sid = l.heap().intern(field);
        let env = l.global().globals;
        let fref = l.heap().alloc_func(GcFunc::C(CClosure {
            f,
            env,
            upvals: Vec::new(),
        }));
        let key = l.heap().str_value(sid);
        t.as_mut().set(key, LuaValue::func(fref));
    }
    let name_sid = l.heap().intern(name);
    let key = l.heap().str_value(name_sid);
    l.global().globals.as_mut().set(key, LuaValue::table(t));
}

/// Install every standard library.
pub fn open_libs(l: &mut LuaState) {
    base::open(l);
    string::open(l);
    table::open(l);
    math::open(l);
    os::open(l);
}
