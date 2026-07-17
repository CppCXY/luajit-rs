//! LuaJIT-compatible standard library.
//!
//! Each sub-module provides an `open(l: &mut LuaState)` function that
//! registers its functions, plus a `CFn`-typed entry for `make_lib`.
//! The `crate::func::CFunction` signature is `fn(&mut LuaState) ->
//! LuaResult<i32>`.  String args are read through
//! `LuaState::str_static` (zero-copy via pool-stable `'static` lifetimes)
//! so C functions never need to clone read-only string data.

pub mod base;
pub mod coroutine;
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
use crate::err::LuaResult;

/// `lua_push` a single result value.
#[inline]
pub fn push(l: &mut LuaState, v: LuaValue) {
    l.stack_ensure(l.base + 1);
    l.stack[l.base] = v;
    l.top = l.base + 1;
}

/// `lua_push` multiple results.
#[inline]
pub fn pushv(l: &mut LuaState, vs: &[LuaValue]) {
    l.stack_ensure(l.base + vs.len().max(1));
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
    use crate::strfmt::g14_to_buf;
    if let Some(sid) = v.as_string_id() {
        return l.heap().strings.get(sid).to_vec();
    }
    if let Some(n) = v.as_number() {
        let mut buf = [0u8; 64];
        let len = g14_to_buf(n, &mut buf);
        return buf[..len].to_vec();
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

/// Convert a value to its display bytes, checking `__tostring` first
/// (lj_meta_tostring). Falls back to the raw `tostring_bytes`.
pub fn tostring_meta(l: &mut LuaState, v: LuaValue) -> LuaResult<Vec<u8>> {
    use crate::meta::{MM, meta_lookup};
    let mo = meta_lookup(l.global(), v, MM::Tostring);
    if mo.is_nil() {
        return Ok(tostring_bytes(l, v));
    }
    // mmcall: place the metamethod and arg above the current frame.
    let saved_top = l.top;
    let fs = l.top + 16;
    assert!(fs + 16 < crate::state::STACK_MAX, "stack overflow in __tostring");
    l.stack[fs] = mo;
    // Args start at func_slot + 2.
    l.stack[fs + 2] = v;
    crate::vm::execute(l, fs, 1, 1)?;
    let r = l.stack[fs];
    l.top = saved_top;
    if let Some(sid) = r.as_string_id() {
        Ok(l.str_static(sid).to_vec())
    } else {
        Err(l.runtime_error(b"'__tostring' must return a string"))
    }
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
    coroutine::open(l);
    string::open(l);
    table::open(l);
    math::open(l);
    os::open(l);
}
