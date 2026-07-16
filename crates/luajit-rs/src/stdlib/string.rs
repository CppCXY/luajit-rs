//! String library: `string.byte`, `string.char`, `string.dump`,
//! `string.format`, `string.len`, `string.lower`, `string.rep`,
//! `string.reverse`, `string.sub`, `string.upper`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push, pushv, tostring_bytes};
use crate::lual_reg;

fn str_byte(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.byte", "string", "")),
    };
    let i = arg(l, 1).as_number().unwrap_or(1.0) as i64;
    let j = arg(l, 2).as_number().map_or(i, |n| n as i64);
    let len = s.len() as i64;
    let (lo, hi) = if i < 0 {
        (len + i, if j < 0 { len + j } else { j })
    } else {
        (i - 1, if j < 0 { len + j } else { j - 1 })
    };
    if lo < 0 || lo > hi || lo >= len {
        push(l, LuaValue::NIL);
        Ok(1)
    } else {
        let hi = hi.min(len - 1);
        for k in lo..=hi {
            l.stack[l.base + (k - lo) as usize] = LuaValue::number(s[k as usize] as f64);
        }
        Ok((hi - lo + 1) as i32)
    }
}

fn str_char(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = arg(l, i).as_number().unwrap_or(0.0) as u32;
        out.push((c & 0xff) as u8);
    }
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn str_dump(l: &mut LuaState) -> LuaResult<i32> {
    let fv = arg(l, 0);
    match fv.as_func() {
        Some(gf) => match gf.as_ref() {
            crate::func::GcFunc::Lua(cl) => {
                let pt = cl.proto.as_ref();
                let mut out = Vec::new();
                crate::dump::dump(pt, &l.heap().strings, "@dumped", &mut out);
                let sid = l.heap().intern(&out);
                push(l, l.heap().str_value(sid));
                Ok(1)
            }
            _ => Err(err_bad_arg(l, 1, "string.dump", "Lua function", "")),
        },
        None => Err(err_bad_arg(l, 1, "string.dump", "function", "")),
    }
}

fn str_format(l: &mut LuaState) -> LuaResult<i32> {
    let fmt = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 1, "string.format", "string", "")),
    };
    let n = nargs(l);
    enum Owned {
        Num(f64),
        Str(Vec<u8>),
    }
    let mut owned = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let v = arg(l, i);
        if let Some(n) = v.as_number() {
            owned.push(Owned::Num(n));
        } else if let Some(sid) = v.as_string_id() {
            owned.push(Owned::Str(l.heap().strings.get(sid).to_vec()));
        } else {
            owned.push(Owned::Str(tostring_bytes(l, v)));
        }
    }
    let args: Vec<crate::strfmt::FmtArg> = owned
        .iter()
        .map(|o| match o {
            Owned::Num(n) => crate::strfmt::FmtArg::Num(*n),
            Owned::Str(s) => crate::strfmt::FmtArg::Str(s),
        })
        .collect();
    match crate::strfmt::format(&fmt, &args) {
        Ok(bytes) => {
            let sid = l.heap().intern(&bytes);
            push(l, l.heap().str_value(sid));
            Ok(1)
        }
        Err(msg) => Err(l.runtime_error(msg.as_bytes())),
    }
}

fn str_len(l: &mut LuaState) -> LuaResult<i32> {
    match arg(l, 0).as_string_id() {
        Some(sid) => {
            push(l, LuaValue::number(l.str_static(sid).len() as f64));
            Ok(1)
        }
        None => Err(err_bad_arg(l, 1, "string.len", "string", "")),
    }
}

fn map_bytes(l: &mut LuaState, f: fn(u8) -> u8) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string case", "string", "")),
    };
    let out: Vec<u8> = s.iter().map(|&b| f(b)).collect();
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn str_lower(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_lowercase())
}
fn str_upper(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_uppercase())
}

fn str_rep(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.rep", "string", "")),
    };
    let n = arg(l, 1).as_number().unwrap_or(0.0) as i64;
    let sep = match arg(l, 2).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => b"" as &[u8],
    };
    let n = n.max(0) as usize;
    let mut out = Vec::with_capacity(s.len() * n + sep.len() * n.saturating_sub(1));
    for i in 0..n {
        if i > 0 {
            out.extend_from_slice(sep);
        }
        out.extend_from_slice(s);
    }
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn str_reverse(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.reverse", "string", "")),
    };
    let rev: Vec<u8> = s.iter().copied().rev().collect();
    let sid = l.heap().intern(&rev);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn str_sub(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.sub", "string", "")),
    };
    let i = arg(l, 1).as_number().unwrap_or(1.0) as i64;
    let j = arg(l, 2).as_number().map(|n| n as i64);
    let len = s.len() as i64;
    let a = if i < 0 {
        (len + i).max(0) as usize
    } else {
        (i - 1).max(0).min(len as i64) as usize
    };
    let b = match j {
        Some(j) => {
            let j = if j < 0 { len + j } else { j - 1 };
            (j.max(-1).min(len - 1) + 1) as usize
        }
        None => len as usize,
    };
    if a >= b {
        push(
            l,
            LuaValue::string(l.heap().strings.lookup_ptr(l.heap().intern(b""))),
        );
    } else {
        let sid = l.heap().intern(&s[a..b]);
        push(l, l.heap().str_value(sid));
    }
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"string", LibTarget::Global)
        .func(b"byte", str_byte)
        .func(b"char", str_char)
        .func(b"dump", str_dump)
        .func(b"format", str_format)
        .func(b"len", str_len)
        .func(b"lower", str_lower)
        .func(b"rep", str_rep)
        .func(b"reverse", str_reverse)
        .func(b"sub", str_sub)
        .func(b"upper", str_upper)
        .build();
}
