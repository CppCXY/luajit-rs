use crate::api::lua_gettop;
use crate::err::{LuaError, LuaResult};
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, push};
use crate::lual_reg;

/// `lj_num2bit`: wrapping num -> int32. The bias add rounds to nearest
/// (ties to even) and leaves the wrapped 32-bit result in the low
/// mantissa bits. The JIT's TOBIT IR mirrors this exactly (within its
/// i32 range guards, cvtsd2si rounds identically).
#[inline]
pub fn num2bit(n: f64) -> i32 {
    let biased = n + 6755399441055744.0; // 2^52 + 2^51
    biased.to_bits() as u32 as i32
}

fn bitarg(l: &mut LuaState, i: usize, name: &str) -> Result<i32, LuaError> {
    let v = arg(l, i);
    match v.as_number() {
        Some(n) => Ok(num2bit(n)),
        None => match v.as_cdata() {
            Some(cd) => {
                let c = cd.as_ref();
                // 64-bit integers: take the low 32 bits exactly (the f64
                // conversion would round the high bits away).
                if (c.ctypeid == crate::ffi::CTypeID::Int64 as u32
                    || c.ctypeid == crate::ffi::CTypeID::UInt64 as u32)
                    && c.data.len() >= 8
                {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&c.data[..8]);
                    return Ok(i64::from_le_bytes(buf) as i32);
                }
                match super::cdata_to_number(c) {
                    Some(n) => Ok(num2bit(n)),
                    None => Err(err_bad_arg(l, i as u32 + 1, name, "number", "")),
                }
            }
            None => Err(err_bad_arg(l, i as u32 + 1, name, "number", "")),
        },
    }
}

fn ret(l: &mut LuaState, v: i32) -> LuaResult<i32> {
    push(l, LuaValue::number(v as f64));
    Ok(1)
}

// -- 64-bit mode (bit64) ------------------------------------------------------

fn is_64bit(v: LuaValue) -> bool {
    matches!(
        v.as_cdata(),
        Some(cd)
            if cd.as_ref().ctypeid == crate::ffi::CTypeID::Int64 as u32
                || cd.as_ref().ctypeid == crate::ffi::CTypeID::UInt64 as u32
    )
}

fn is_ull(v: LuaValue) -> bool {
    matches!(
        v.as_cdata(),
        Some(cd) if cd.as_ref().ctypeid == crate::ffi::CTypeID::UInt64 as u32
    )
}

fn bitarg64(l: &mut LuaState, i: usize, name: &str) -> Result<u64, LuaError> {
    let v = arg(l, i);
    match v.as_number() {
        Some(n) => Ok(num2bit(n) as i64 as u64),
        None => match v.as_cdata() {
            Some(cd) => {
                let c = cd.as_ref();
                if (c.ctypeid == crate::ffi::CTypeID::Int64 as u32
                    || c.ctypeid == crate::ffi::CTypeID::UInt64 as u32)
                    && c.data.len() >= 8
                {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&c.data[..8]);
                    return Ok(u64::from_le_bytes(buf));
                }
                match super::cdata_to_number(c) {
                    Some(n) => Ok(num2bit(n) as i64 as u64),
                    None => Err(err_bad_arg(l, i as u32 + 1, name, "number", "")),
                }
            }
            None => Err(err_bad_arg(l, i as u32 + 1, name, "number", "")),
        },
    }
}

fn ret64(l: &mut LuaState, v: u64, is_ull: bool) -> LuaResult<i32> {
    let id = if is_ull {
        crate::ffi::CTypeID::UInt64 as u32
    } else {
        crate::ffi::CTypeID::Int64 as u32
    };
    ret_cdata(l, v, id)
}

/// Push a numeric cdata of the given type holding the low bits of `v`.
fn ret_cdata(l: &mut LuaState, v: u64, ctypeid: u32) -> LuaResult<i32> {
    use crate::ffi::CTypeID;
    let size = match ctypeid {
        id if id == CTypeID::Int8 as u32 || id == CTypeID::UInt8 as u32 => 1,
        id if id == CTypeID::Int16 as u32 || id == CTypeID::UInt16 as u32 => 2,
        id if id == CTypeID::Int32 as u32 || id == CTypeID::UInt32 as u32 => 4,
        _ => 8,
    };
    let mut cd = crate::runtime::cdata::CData::new(ctypeid, size);
    let bytes = &v.to_le_bytes()[..size];
    cd.data[..size].copy_from_slice(bytes);
    let p = l.global().heap.alloc_cdata(cd);
    push(l, LuaValue::cdata(p));
    Ok(1)
}

/// Result type for a cdata-mode fold: UInt64 when any operand is one,
/// Int64 when any operand is 64-bit, else the first cdata operand's type.
fn fold_type(l: &LuaState, n: usize) -> u32 {
    use crate::ffi::CTypeID;
    let mut first_ct = 0u32;
    for i in 0..n {
        if let Some(cd) = arg(l, i).as_cdata() {
            let id = cd.as_ref().ctypeid;
            if first_ct == 0 {
                first_ct = id;
            }
            if id == CTypeID::UInt64 as u32 {
                return id;
            }
        }
    }
    if first_ct == CTypeID::Int64 as u32 || first_ct != 0 {
        return first_ct;
    }
    CTypeID::Int64 as u32
}

/// (pub: the JIT's fast-function recorder identifies builtins by their
/// function pointer.)
pub fn tobit(l: &mut LuaState) -> LuaResult<i32> {
    let x = bitarg(l, 0, "bit.tobit")?;
    ret(l, x)
}

macro_rules! bit_fold {
    ($name:ident, $lua:literal, $op:tt) => {
        pub fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let n = lua_gettop(l);
            // Any cdata operand puts the fold in cdata mode: the result
            // is a cdata of the fold type (bit64 semantics).
            if (0..n).any(|i| arg(l, i).as_cdata().is_some()) {
                let mut acc = bitarg64(l, 0, $lua)?;
                for i in 1..n {
                    #[allow(clippy::assign_op_pattern)]
                    { acc = acc $op bitarg64(l, i, $lua)?; }
                }
                ret_cdata(l, acc, fold_type(l, n))
            } else {
                let mut acc = bitarg(l, 0, $lua)?;
                for i in 1..n {
                    #[allow(clippy::assign_op_pattern)]
                    { acc = acc $op bitarg(l, i, $lua)?; }
                }
                ret(l, acc)
            }
        }
    };
}

bit_fold!(band, "bit.band", &);
bit_fold!(bor, "bit.bor", |);
bit_fold!(bxor, "bit.bxor", ^);

pub fn bnot(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    if is_64bit(v) {
        let x = bitarg64(l, 0, "bit.bnot")?;
        ret64(l, !x, is_ull(v))
    } else {
        let x = bitarg(l, 0, "bit.bnot")?;
        ret(l, !x)
    }
}

pub fn bswap(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    if is_64bit(v) {
        let x = bitarg64(l, 0, "bit.bswap")?;
        ret64(l, x.swap_bytes(), is_ull(v))
    } else {
        let x = bitarg(l, 0, "bit.bswap")?;
        ret(l, x.swap_bytes())
    }
}

macro_rules! bit_shift {
    ($name:ident, $lua:literal, $body32:expr, $body64:expr) => {
        pub fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let a = arg(l, 0);
            let b = arg(l, 1);
            if is_64bit(a) || is_64bit(b) {
                let x = bitarg64(l, 0, $lua)?;
                let n = (bitarg64(l, 1, $lua)? as u32) & 63;
                let f: fn(u64, u32) -> u64 = $body64;
                ret64(l, f(x, n), is_ull(a) || is_ull(b))
            } else {
                let x = bitarg(l, 0, $lua)?;
                let n = (bitarg(l, 1, $lua)? as u32) & 31;
                let f: fn(i32, u32) -> i32 = $body32;
                ret(l, f(x, n))
            }
        }
    };
}

bit_shift!(lshift, "bit.lshift", |x, n| x.wrapping_shl(n), |x, n| x
    .wrapping_shl(n));
bit_shift!(
    rshift,
    "bit.rshift",
    |x, n| ((x as u32).wrapping_shr(n)) as i32,
    |x, n| x.wrapping_shr(n)
);
bit_shift!(
    arshift,
    "bit.arshift",
    |x, n| x.wrapping_shr(n),
    |x, n| ((x as i64).wrapping_shr(n)) as u64
);
bit_shift!(
    rol,
    "bit.rol",
    |x, n| (x as u32).rotate_left(n) as i32,
    |x, n| x.rotate_left(n)
);
bit_shift!(
    ror,
    "bit.ror",
    |x, n| (x as u32).rotate_right(n) as i32,
    |x, n| x.rotate_right(n)
);

fn tohex(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let n = if lua_gettop(l) >= 2 {
        bitarg(l, 1, "bit.tohex")?
    } else if is_64bit(v) {
        16
    } else {
        8
    };
    let (digits, upper) = if n < 0 {
        ((-n) as usize, true)
    } else {
        (n as usize, false)
    };
    if digits == 0 {
        // tohex(x, 0) is the empty string.
        push(l, l.heap().str_value(l.heap().intern(b"")));
        return Ok(1);
    }
    let s = if is_64bit(v) {
        let x = bitarg64(l, 0, "bit.tohex")?;
        let digits = digits.clamp(1, 16);
        if upper {
            format!(
                "{:0width$X}",
                x & (!0u64 >> (64 - digits * 4)),
                width = digits
            )
        } else {
            format!(
                "{:0width$x}",
                x & (!0u64 >> (64 - digits * 4)),
                width = digits
            )
        }
    } else {
        let x = bitarg(l, 0, "bit.tohex")? as u32;
        let digits = digits.clamp(1, 8);
        if upper {
            format!(
                "{:0width$X}",
                x & (!0u32 >> (32 - digits * 4)),
                width = digits
            )
        } else {
            format!(
                "{:0width$x}",
                x & (!0u32 >> (32 - digits * 4)),
                width = digits
            )
        }
    };
    let sid = l.heap().intern(s.as_bytes());
    let v = l.heap().str_value(sid);
    push(l, v);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"bit", LibTarget::Global)
        .func(b"tobit", tobit)
        .func(b"bnot", bnot)
        .func(b"band", band)
        .func(b"bor", bor)
        .func(b"bxor", bxor)
        .func(b"lshift", lshift)
        .func(b"rshift", rshift)
        .func(b"arshift", arshift)
        .func(b"rol", rol)
        .func(b"ror", ror)
        .func(b"bswap", bswap)
        .func(b"tohex", tohex)
        .build();
}
