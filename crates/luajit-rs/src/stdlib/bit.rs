//! LuaJIT's `bit.*` library (lj_lib_bit). Every input goes through the
//! wrapping num -> int32 conversion (`lj_num2bit`: the 2^52+2^51 bias
//! trick, round-to-nearest-even, low 32 mantissa bits) — unlike the
//! Lua 5.3-style operators, which use a saturating truncation.

use crate::err::{LuaError, LuaResult};
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push};
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
    match arg(l, i).as_number() {
        Some(n) => Ok(num2bit(n)),
        None => Err(err_bad_arg(l, i as u32 + 1, name, "number", "")),
    }
}

fn ret(l: &mut LuaState, v: i32) -> LuaResult<i32> {
    push(l, LuaValue::number(v as f64));
    Ok(1)
}

/// (pub: the JIT's fast-function recorder identifies builtins by their
/// function pointer.)
pub fn tobit(l: &mut LuaState) -> LuaResult<i32> {
    let x = bitarg(l, 0, "bit.tobit")?;
    ret(l, x)
}

pub fn bnot(l: &mut LuaState) -> LuaResult<i32> {
    let x = bitarg(l, 0, "bit.bnot")?;
    ret(l, !x)
}

macro_rules! bit_fold {
    ($name:ident, $lua:literal, $op:tt) => {
        pub fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let n = nargs(l);
            let mut acc = bitarg(l, 0, $lua)?;
            for i in 1..n {
                acc = acc $op bitarg(l, i, $lua)?;
            }
            ret(l, acc)
        }
    };
}

bit_fold!(band, "bit.band", &);
bit_fold!(bor, "bit.bor", |);
bit_fold!(bxor, "bit.bxor", ^);

macro_rules! bit_shift {
    ($name:ident, $lua:literal, $body:expr) => {
        pub fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let x = bitarg(l, 0, $lua)?;
            let n = (bitarg(l, 1, $lua)? as u32) & 31;
            let f: fn(i32, u32) -> i32 = $body;
            ret(l, f(x, n))
        }
    };
}

bit_shift!(lshift, "bit.lshift", |x, n| x.wrapping_shl(n));
bit_shift!(rshift, "bit.rshift", |x, n| ((x as u32).wrapping_shr(n)) as i32);
bit_shift!(arshift, "bit.arshift", |x, n| x.wrapping_shr(n));
bit_shift!(rol, "bit.rol", |x, n| (x as u32).rotate_left(n) as i32);
bit_shift!(ror, "bit.ror", |x, n| (x as u32).rotate_right(n) as i32);

pub fn bswap(l: &mut LuaState) -> LuaResult<i32> {
    let x = bitarg(l, 0, "bit.bswap")?;
    ret(l, x.swap_bytes())
}

fn tohex(l: &mut LuaState) -> LuaResult<i32> {
    let x = bitarg(l, 0, "bit.tohex")? as u32;
    let n = if nargs(l) >= 2 { bitarg(l, 1, "bit.tohex")? } else { 8 };
    let (digits, upper) = if n < 0 { ((-n) as usize, true) } else { (n as usize, false) };
    let digits = digits.clamp(1, 8);
    let s = if upper {
        format!("{:0width$X}", x & (!0u32 >> (32 - digits * 4)), width = digits)
    } else {
        format!("{:0width$x}", x & (!0u32 >> (32 - digits * 4)), width = digits)
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
