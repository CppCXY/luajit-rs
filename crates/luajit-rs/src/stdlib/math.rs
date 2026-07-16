//! Math library: every function from LuaJIT's `math.*` except `random`
//! and `randomseed` (which need rng state).  Constants are set via
//! `table.new` + `table.set`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push, pushv};
use crate::lual_reg;

macro_rules! math1 {
    ($name:ident, $fn:expr) => {
        fn $name(l: &mut LuaState) -> LuaResult<i32> {
            let x = match arg(l, 0).as_number() {
                Some(n) => n,
                None => {
                    return Err(err_bad_arg(
                        l,
                        1,
                        concat!("math.", stringify!($name)),
                        "number",
                        "",
                    ))
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
        None => return Err(err_bad_arg(l, 1, "math.atan", "number", "")),
    };
    let x = match arg(l, 1).as_number() {
        Some(n) => n,
        None => 1.0,
    };
    push(l, LuaValue::number(y.atan2(x)));
    Ok(1)
}

fn math_fmod(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 1, "math.fmod", "number", "")),
    };
    let y = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 2, "math.fmod", "number", "")),
    };
    push(l, LuaValue::number(x % y));
    Ok(1)
}

fn math_frexp(l: &mut LuaState) -> LuaResult<i32> {
    // frexp: decompose x into m * 2^e where 0.5 <= |m| < 1.
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 1, "math.frexp", "number", "")),
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
        None => return Err(err_bad_arg(l, 1, "math.ldexp", "number", "")),
    };
    let e = match arg(l, 1).as_number() {
        Some(n) => n as i32,
        None => return Err(err_bad_arg(l, 2, "math.ldexp", "number", "")),
    };
    push(l, LuaValue::number(m * (2.0f64).powi(e)));
    Ok(1)
}

fn math_logx(l: &mut LuaState) -> LuaResult<i32> {
    // log(x [, base])
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 1, "math.log", "number", "")),
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

fn math_max(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    if n == 0 {
        push(l, LuaValue::number(f64::NEG_INFINITY));
        return Ok(1);
    }
    let mut max = match arg(l, 0).as_number() {
        Some(n) => n,
        None => f64::NEG_INFINITY,
    };
    for i in 1..n {
        if let Some(n) = arg(l, i).as_number() {
            if n > max {
                max = n;
            }
        }
    }
    push(l, LuaValue::number(max));
    Ok(1)
}

fn math_min(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    if n == 0 {
        push(l, LuaValue::number(f64::INFINITY));
        return Ok(1);
    }
    let mut min = match arg(l, 0).as_number() {
        Some(n) => n,
        None => f64::INFINITY,
    };
    for i in 1..n {
        if let Some(n) = arg(l, i).as_number() {
            if n < min {
                min = n;
            }
        }
    }
    push(l, LuaValue::number(min));
    Ok(1)
}

fn math_modf(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 1, "math.modf", "number", "")),
    };
    let int = x.trunc();
    pushv(l, &[LuaValue::number(int), LuaValue::number(x - int)]);
    Ok(2)
}

fn math_pow(l: &mut LuaState) -> LuaResult<i32> {
    let x = match arg(l, 0).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 1, "math.pow", "number", "")),
    };
    let y = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 2, "math.pow", "number", "")),
    };
    push(l, LuaValue::number(x.powf(y)));
    Ok(1)
}

fn math_random(l: &mut LuaState) -> LuaResult<i32> {
    // NYI: deterministic for now.
    let m = arg(l, 0).as_number();
    let n = arg(l, 1).as_number();
    push(
        l,
        LuaValue::number(match (m, n) {
            (None, _) => 0.0,
            (Some(u), None) => (u % 1.0),
            (Some(lo), Some(hi)) => lo,
        }),
    );
    Ok(1)
}

fn math_randomseed(l: &mut LuaState) -> LuaResult<i32> {
    push(l, LuaValue::number(0.0));
    Ok(1)
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
        None => return Err(err_bad_arg(l, 1, "math.ult", "number", "")),
    };
    let n = match arg(l, 1).as_number() {
        Some(n) => n,
        None => return Err(err_bad_arg(l, 2, "math.ult", "number", "")),
    };
    push(l, LuaValue::boolean((m as u64) < (n as u64)));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    let t = lual_reg!(l, b"math", LibTarget::Global)
        .func(b"abs", abs)
        .func(b"acos", acos)
        .func(b"asin", asin)
        .func(b"atan", math_atan2)
        .func(b"ceil", ceil)
        .func(b"cos", cos)
        .func(b"cosh", cosh)
        .func(b"deg", deg)
        .func(b"exp", exp)
        .func(b"floor", floor)
        .func(b"fmod", math_fmod)
        .func(b"frexp", math_frexp)
        .func(b"ldexp", math_ldexp)
        .func(b"log10", log10)
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
        .func(b"tointeger", math_tointeger)
        .func(b"type", math_type)
        .func(b"ult", math_ult)
        .build();
    // Per-session constants.
    let set = |t: &crate::gc::GcPtr<crate::table::LuaTable>, k: &str, v: LuaValue| {
        let sk = l.heap().str_value(l.heap().intern(k.as_bytes()));
        t.as_mut().set(sk, v);
    };
    set(&t, "pi", LuaValue::number(std::f64::consts::PI));
    set(&t, "huge", LuaValue::number(f64::MAX));
    set(&t, "maxinteger", LuaValue::number(i64::MAX as f64));
    set(&t, "mininteger", LuaValue::number(i64::MIN as f64));
}
