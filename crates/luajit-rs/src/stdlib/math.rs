//! Math library.  Constants are set via `table.new` + `table.set`.

#![allow(dead_code)] // `atan`、`log` 由宏生成,在 lib 表中使用

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{err_bad_arg_type, LibTarget, arg, err_bad_arg, nargs, push, pushv};
use crate::lual_reg;

/// Tausworthe PRNG state (period 2^223), bit-exact with LuaJIT's
/// `lj_prng.c` / `lib_math.c:random_seed` so the test-suite sequences
/// match.
#[derive(Clone, Copy)]
pub struct RngState {
    u: [u64; 4],
}

impl RngState {
    /// The fixed initial state (lj_prng_seed_fixed).
    pub fn fixed() -> Self {
        RngState {
            u: [
                0xa0d277570a345b8c,
                0x764a296c5d4aa64f,
                0x51220704070adeaa,
                0x2a2717b5a7b7b927,
            ],
        }
    }

    /// `random_seed(rs, d)`: derive the four state words from a double.
    pub fn seed(&mut self, d: f64) {
        let mut r: u32 = 0x11090601; // 64-k[i] as four 8-bit constants.
        let mut d = d;
        for i in 0..4 {
            let m = 1u64 << (r & 255);
            r >>= 8;
            d = d * std::f64::consts::PI + std::f64::consts::E;
            let mut bits = d.to_bits();
            if bits < m {
                bits += m;
            }
            self.u[i] = bits;
        }
        for _ in 0..10 {
            self.next_u64();
        }
    }

    /// The TW223 step: update all four generators, xor their outputs.
    fn tw223_step(&mut self) -> u64 {
        const CFG: [(u32, u32, u32); 4] = [(63, 31, 18), (58, 19, 28), (55, 24, 7), (47, 21, 8)];
        let mut r = 0u64;
        for (i, &(k, q, s)) in CFG.iter().enumerate() {
            let mut z = self.u[i];
            z = (((z << q) ^ z) >> (k - s)) ^ ((z & (u64::MAX << (64 - k))) << s);
            r ^= z;
            self.u[i] = z;
        }
        r
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.tw223_step()
    }

    /// A double in [0, 1) (`lj_prng_u64d` - 1.0).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let r = self.tw223_step();
        f64::from_bits((r & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000) - 1.0
    }
}

macro_rules! math1 {
    ($name:ident, $fn:expr) => {
        /// (pub: the JIT's fast-function recorder identifies builtins by
        /// their function pointer.)
        pub fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let v = arg(l, 0);
            let x = match v.as_number() {
                Some(n) => n,
                None => {
                    // Lua-style string-to-number coercion.
                    if let Some(sid) = v.as_string_id() {
                        match crate::strscan::scan_number(l.str_static(sid)) {
                            Some(n) => n,
                            None => {
                                return Err(err_bad_arg_type(
                                    l,
                                    1,
                                    stringify!($name),
                                    "number",
                                    v,
                                ));
                            }
                        }
                    } else {
                        return Err(err_bad_arg_type(
                            l,
                            1,
                            stringify!($name),
                            "number",
                            v,
                        ));
                    }
                }
            };
            push(l, LuaValue::number($fn(x)));
            Ok(1)
        }
    };
}

math1!(abs, f64::abs);
math1!(acos, f64::acos);
math1!(asin, f64::asin);
math1!(atan, f64::atan);
math1!(ceil, f64::ceil);
math1!(cos, f64::cos);
math1!(cosh, f64::cosh);
math1!(deg, |x: f64| x * (180.0 / std::f64::consts::PI));
math1!(exp, f64::exp);
math1!(floor, f64::floor);
math1!(log, |x: f64| x.ln());
math1!(log10, f64::log10);
math1!(rad, |x: f64| x * (std::f64::consts::PI / 180.0));
math1!(sin, f64::sin);
math1!(sinh, f64::sinh);
math1!(sqrt, f64::sqrt);
math1!(tan, f64::tan);
math1!(tanh, f64::tanh);

fn math_atan2(l: &mut LuaState) -> LuaResult<i32> {
    let y = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.atan", "number", arg(l, 1-1))),
    };
    let x = arg(l, 1).as_number().unwrap_or(1.0);
    push(l, LuaValue::number(y.atan2(x)));
    Ok(1)
}

pub fn math_fmod(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.fmod", "number", arg(l, 1-1))),
    };
    let y = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 2, "math.fmod", "number", arg(l, 2-1))),
    };
    push(l, LuaValue::number(x % y));
    Ok(1)
}

fn math_frexp(l: &mut LuaState) -> LuaResult<i32> {
    // frexp: decompose x into m * 2^e where 0.5 <= |m| < 1.
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.frexp", "number", arg(l, 1-1))),
    };
    if x == 0.0 {
        pushv(l, &[LuaValue::number(0.0), LuaValue::number(0.0)]);
    } else {
        let bits = x.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i32 - 1022;
        let mant = f64::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000);
        pushv(l, &[LuaValue::number(mant), LuaValue::number(exp as f64)]);
    }
    Ok(2)
}

fn math_ldexp(l: &mut LuaState) -> LuaResult<i32> {
    let m = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.ldexp", "number", arg(l, 1-1))),
    };
    let e = match arg(l, 1).as_number() {
        Some(n) => n as i32,
        None => return Err(err_bad_arg_type(l, 2, "math.ldexp", "number", arg(l, 2-1))),
    };
    push(l, LuaValue::number(m * (2.0f64).powi(e)));
    Ok(1)
}

fn math_logx(l: &mut LuaState) -> LuaResult<i32> {
    // log(x [, base])
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.log", "number", arg(l, 1-1))),
    };
    let base = arg(l, 1).as_number();
    push(
        l,
        LuaValue::number(match base {
            Some(b) => x.log(b),
            None => x.ln(),
        }),
    );
    Ok(1)
}

pub fn math_max(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    if n == 0 {
        push(l, LuaValue::number(f64::NEG_INFINITY));
        return Ok(1);
    }
    let mut max = arg(l, 0).as_number().unwrap_or(f64::NEG_INFINITY);
    for i in 1..n {
        if let Some(n) = arg(l, i).as_number()
            && n > max
        {
            max = n;
        }
    }
    push(l, LuaValue::number(max));
    Ok(1)
}

pub fn math_min(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    if n == 0 {
        push(l, LuaValue::number(f64::INFINITY));
        return Ok(1);
    }
    let mut min = arg(l, 0).as_number().unwrap_or(f64::INFINITY);
    for i in 1..n {
        if let Some(n) = arg(l, i).as_number()
            && n < min
        {
            min = n;
        }
    }
    push(l, LuaValue::number(min));
    Ok(1)
}

fn math_modf(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.modf", "number", arg(l, 1-1))),
    };
    let int = x.trunc();
    pushv(l, &[LuaValue::number(int), LuaValue::number(x - int)]);
    Ok(2)
}

fn math_pow(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.pow", "number", arg(l, 1-1))),
    };
    let y = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 2, "math.pow", "number", arg(l, 2-1))),
    };
    push(l, LuaValue::number(x.powf(y)));
    Ok(1)
}

fn math_random(l: &mut LuaState) -> LuaResult<i32> {
    let rng = &mut l.global().rng;
    let n = nargs(l);
    let d = rng.next_f64();
    if n > 0 {
        let r1 = arg(l, 0).as_number().unwrap_or(0.0);
        if n == 1 {
            // random(r1): integer in [1, r1].
            push(l, LuaValue::number((d * r1).floor() + 1.0));
        } else {
            // random(r1, r2): integer in [r1, r2].
            let r2 = arg(l, 1).as_number().unwrap_or(0.0);
            if r1 > r2 {
                return Err(err_bad_arg(l, 1, "random", "number", "interval is empty"));
            }
            push(l, LuaValue::number((d * (r2 - r1 + 1.0)).floor() + r1));
        }
    } else {
        // random(): double in [0, 1).
        push(l, LuaValue::number(d));
    }
    Ok(1)
}

fn math_randomseed(l: &mut LuaState) -> LuaResult<i32> {
    if nargs(l) > 0 {
        let s = arg(l, 0).as_number().unwrap_or(0.0);
        l.global().rng.seed(s);
    } else {
        l.global().rng = RngState::fixed();
    }
    Ok(0)
}

fn math_tointeger(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };
    let i = x as i64;
    if i as f64 == x {
        push(l, LuaValue::number(i as f64));
    } else {
        push(l, LuaValue::NIL);
    }
    Ok(1)
}

fn math_type(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let sid = if v.is_number() {
        let n = v.num();
        if n == n.trunc() && n.is_finite() {
            l.heap().intern(b"integer")
        } else {
            l.heap().intern(b"float")
        }
    } else {
        l.heap().intern(b"")
    };
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn math_ult(l: &mut LuaState) -> LuaResult<i32> {
    let m = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 1, "math.ult", "number", arg(l, 1-1))),
    };
    let n = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg_type(l, 2, "math.ult", "number", arg(l, 2-1))),
    };
    push(l, LuaValue::boolean((m as u64) < (n as u64)));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"math", LibTarget::Global)
        .func(b"abs", abs)
        .func(b"acos", acos)
        .func(b"asin", asin)
        .func(b"atan", math_atan2)
        .func(b"atan2", math_atan2)
        .func(b"ceil", ceil)
        .func(b"cos", cos)
        .func(b"cosh", cosh)
        .func(b"deg", deg)
        .func(b"exp", exp)
        .func(b"floor", floor)
        .func(b"fmod", math_fmod)
        .func(b"frexp", math_frexp)
        .func(b"ldexp", math_ldexp)
        .func(b"log", math_logx)
        .func(b"max", math_max)
        .func(b"min", math_min)
        .func(b"modf", math_modf)
        .func(b"pow", math_pow)
        .func(b"rad", rad)
        .func(b"random", math_random)
        .func(b"randomseed", math_randomseed)
        .func(b"sin", sin)
        .func(b"sinh", sinh)
        .func(b"sqrt", sqrt)
        .func(b"tan", tan)
        .func(b"tanh", tanh)
        .value(b"pi", LuaValue::number(std::f64::consts::PI))
        .value(b"huge", LuaValue::number(f64::INFINITY))
        .build();
}
