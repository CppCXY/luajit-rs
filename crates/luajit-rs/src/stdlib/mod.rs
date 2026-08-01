//! LuaJIT-compatible standard library.
//!
//! Each sub-module provides an `open(l: &mut LuaState)` function that
//! registers its functions, plus a `CFn`-typed entry for `make_lib`.
//! The `crate::func::CFunction` signature is `fn(&mut LuaState) ->
//! LuaResult<i32>`.  String args are read through
//! `LuaState::str_static` (zero-copy via pool-stable `'static` lifetimes)
//! so C functions never need to clone read-only string data.

pub mod base;
pub mod bit;
pub mod coroutine;
pub mod debug;
pub mod io;
pub mod jit;
pub mod math;
pub mod os;
pub mod package;
pub mod pattern;
pub mod reg;
pub mod sort;
pub mod string;
pub mod table;
mod time;

pub use reg::LibTarget;
pub(crate) use time::PlatformInstant;

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::{LuaError, ffi};

/// `lua_push` a single result value. Writes into the result area
/// at `base` and sets `top = base + 1`. Only safe for one-value
/// returns; for multiple results write directly to `stack[base..]`.
#[inline]
pub fn push(l: &mut LuaState, v: LuaValue) {
    l.stack_ensure(l.base + 1);
    l.stack[l.base] = v;
    l.top = l.base + 1;
}

/// `lua_push` multiple results. Writes at `base + i` and sets
/// `top = base + n`.
#[inline]
pub fn pushv(l: &mut LuaState, vs: &[LuaValue]) {
    let n = vs.len();
    if n == 0 {
        return;
    }
    l.stack_ensure(l.base + n);
    for (i, &v) in vs.iter().enumerate() {
        l.stack[l.base + i] = v;
    }
    l.top = l.base + n;
}

/// Argument `i` (0-based) of the current C call, or nil.
#[inline]
pub fn arg(l: &LuaState, i: usize) -> LuaValue {
    let slot = l.base + i;
    if slot < l.top && slot < l.stack.len() {
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
    if let Some(cd) = v.as_cdata() {
        let id = cd.as_ref().ctypeid;
        match id {
            id if id == crate::ffi::CTypeID::Int8 as u32 => {
                let val = cd.as_ref().data.first().copied().unwrap_or(0) as i8;
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::UInt8 as u32 => {
                let val = cd.as_ref().data.first().copied().unwrap_or(0);
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::Int16 as u32 => {
                let val = i16::from_le_bytes(cd.as_ref().data[..2].try_into().unwrap_or([0; 2]));
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::UInt16 as u32 => {
                let val = u16::from_le_bytes(cd.as_ref().data[..2].try_into().unwrap_or([0; 2]));
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::Int32 as u32 => {
                let val = i32::from_le_bytes(cd.as_ref().data[..4].try_into().unwrap_or([0; 4]));
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::UInt32 as u32 => {
                let val = u32::from_le_bytes(cd.as_ref().data[..4].try_into().unwrap_or([0; 4]));
                return format!("{}", val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::Int64 as u32 => {
                if cd.as_ref().data.len() >= 8 {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&cd.as_ref().data[..8]);
                    let val = i64::from_le_bytes(buf);
                    return format!("{}LL", val).into_bytes();
                }
            }
            id if id == crate::ffi::CTypeID::UInt64 as u32 => {
                if cd.as_ref().data.len() >= 8 {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&cd.as_ref().data[..8]);
                    let val = u64::from_le_bytes(buf);
                    return format!("{}ULL", val).into_bytes();
                }
            }
            id if id == crate::ffi::CTypeID::Float as u32 => {
                let val = f32::from_le_bytes(cd.as_ref().data[..4].try_into().unwrap_or([0; 4]));
                return crate::strfmt::g14(val as f64).into_bytes();
            }
            id if id == crate::ffi::CTypeID::Double as u32 => {
                let val = f64::from_le_bytes(cd.as_ref().data[..8].try_into().unwrap_or([0; 8]));
                return crate::strfmt::g14(val).into_bytes();
            }
            id if id == crate::ffi::CTypeID::Bool as u32 => {
                let val = cd.as_ref().data.first().copied().unwrap_or(0) != 0;
                return if val {
                    b"true".to_vec()
                } else {
                    b"false".to_vec()
                };
            }
            id if id == crate::ffi::CTypeID::Void as u32
                || id == crate::ffi::CTypeID::None as u32 =>
            {
                return b"cdata<void>"[..].into();
            }
            id if id == crate::ffi::CTypeID::PVoid as u32
                || id == crate::ffi::CTypeID::PCVoid as u32
                || id == crate::ffi::CTypeID::PCChar as u32
                || id == crate::ffi::CTypeID::PUInt8 as u32 =>
            {
                let ptr = cd.as_ref().get_ptr();
                if ptr == 0 {
                    return b"cdata<void *>: NULL"[..].into();
                }
                return format!("cdata<void *>: {:#x}", ptr).into_bytes();
            }
            _ => {}
        }
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
    let saved_base = l.base;
    let fs = l.top + 16;
    assert!(
        fs + 16 < crate::state::STACK_MAX,
        "stack overflow in __tostring"
    );
    l.stack[fs] = mo;
    // Args start at func_slot + 2.
    l.stack[fs + 2] = v;
    crate::vm::execute(l, fs, 1, 1)?;
    let r = l.stack[fs];
    l.top = saved_top;
    l.base = saved_base;
    if let Some(sid) = r.as_string_id() {
        Ok(l.str_static(sid).to_vec())
    } else {
        Err(l.runtime_error(b"'__tostring' must return a string"))
    }
}

/// Builtin error: `bad argument #N to 'func' (expected, got)`.
pub fn err_bad_arg(l: &mut LuaState, n: u32, func: &str, expected: &str, got: &str) -> LuaError {
    let msg = format!(
        "bad argument #{} to '{}' ({} expected, got {})",
        n, func, expected, got
    );
    l.runtime_error(msg.as_bytes())
}

/// Type name for `luaL_typename`-style messages; a nil argument is
/// spelled "no value" (LuaJIT's fast-function convention).
pub fn type_name(v: LuaValue) -> &'static str {
    if v.is_nil() {
        "no value"
    } else if v.is_number() {
        "number"
    } else if v.is_string() {
        "string"
    } else if v == LuaValue::FALSE || v == LuaValue::TRUE {
        "boolean"
    } else if v.is_table() {
        "table"
    } else if v.is_func() {
        "function"
    } else if v.is_thread() {
        "thread"
    } else if v.is_userdata() {
        "userdata"
    } else if v.is_cdata() {
        "cdata"
    } else {
        "userdata"
    }
}

/// `err_bad_arg` with the offending value's type filled in.
pub fn err_bad_arg_type(
    l: &mut LuaState,
    n: u32,
    func: &str,
    expected: &str,
    got: LuaValue,
) -> LuaError {
    err_bad_arg(l, n, func, expected, type_name(got))
}

/// Numeric cdata -> f64 (LuaJIT's `cdata_to_number` for bit ops and
/// arithmetic fallbacks). Pointers/voids are not convertible.
pub fn cdata_to_number(cd: &crate::runtime::cdata::CData) -> Option<f64> {
    use crate::ffi::CTypeID;
    let id = cd.ctypeid;
    let d = &cd.data;
    let v = match id {
        id if id == CTypeID::Int8 as u32 => d.first().copied().unwrap_or(0) as i8 as f64,
        id if id == CTypeID::UInt8 as u32 => d.first().copied().unwrap_or(0) as f64,
        id if id == CTypeID::CChar as u32 => d.first().copied().unwrap_or(0) as i8 as f64,
        id if id == CTypeID::Bool as u32 => {
            if d.first().copied().unwrap_or(0) != 0 {
                1.0
            } else {
                0.0
            }
        }
        id if id == CTypeID::Int16 as u32 => {
            i16::from_le_bytes(d[..2].try_into().unwrap_or([0; 2])) as f64
        }
        id if id == CTypeID::UInt16 as u32 => {
            u16::from_le_bytes(d[..2].try_into().unwrap_or([0; 2])) as f64
        }
        id if id == CTypeID::Int32 as u32 => {
            i32::from_le_bytes(d[..4].try_into().unwrap_or([0; 4])) as f64
        }
        id if id == CTypeID::UInt32 as u32 => {
            u32::from_le_bytes(d[..4].try_into().unwrap_or([0; 4])) as f64
        }
        id if id == CTypeID::Int64 as u32 => {
            if d.len() >= 8 {
                let mut b = [0u8; 8];
                b.copy_from_slice(&d[..8]);
                i64::from_le_bytes(b) as f64
            } else {
                return None;
            }
        }
        id if id == CTypeID::UInt64 as u32 => {
            if d.len() >= 8 {
                let mut b = [0u8; 8];
                b.copy_from_slice(&d[..8]);
                u64::from_le_bytes(b) as f64
            } else {
                return None;
            }
        }
        id if id == CTypeID::Float as u32 => {
            f32::from_le_bytes(d[..4].try_into().unwrap_or([0; 4])) as f64
        }
        id if id == CTypeID::Double as u32 => {
            f64::from_le_bytes(d[..8].try_into().unwrap_or([0; 8]))
        }
        _ => {
            // Enum types and other table-defined numerics: interpret by
            // payload size (enums are int-sized).
            match d.len() {
                1 => d.first().copied().unwrap_or(0) as i8 as f64,
                2 => i16::from_le_bytes(d[..2].try_into().unwrap_or([0; 2])) as f64,
                4 => i32::from_le_bytes(d[..4].try_into().unwrap_or([0; 4])) as f64,
                8 => {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&d[..8]);
                    i64::from_le_bytes(b) as f64
                }
                _ => return None,
            }
        }
    };
    Some(v)
}

/// Install every standard library.
pub fn open_libs(l: &mut LuaState) {
    base::open(l);
    coroutine::open(l);
    string::open(l);
    table::open(l);
    math::open(l);
    bit::open(l);
    jit::open(l);
    os::open(l);
    io::open(l);
    debug::open(l);
    package::open(l);
    ffi::open(l);
}
