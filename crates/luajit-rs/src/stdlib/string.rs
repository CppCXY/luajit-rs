//! String library: `string.byte`, `string.char`, `string.dump`,
//! `string.find`, `string.format`, `string.gmatch`, `string.gsub`,
//! `string.len`, `string.lower`, `string.match`, `string.rep`,
//! `string.reverse`, `string.sub`, `string.upper`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push, tostring_bytes};
use crate::lual_reg;
use crate::stdlib::pattern::{CaptureValue, find, gsub};

/// Collect capture values into a vector of LuaValues, for pushing as
/// multiple return values.  `base` is the slot where results should start.
fn push_captures(l: &mut LuaState, captures: &[CaptureValue], text: &[u8], base: usize) {
    for (i, capture) in captures.iter().enumerate() {
        match capture {
            CaptureValue::Substring(start, end) => {
                let sid = l.heap().intern(&text[*start..*end]);
                l.stack[base + i] = l.heap().str_value(sid);
            }
            CaptureValue::Position(p) => {
                l.stack[base + i] = LuaValue::number((*p + 1) as f64);
            }
        }
    }
}

fn str_find(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.find", "string", "")),
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 2, "string.find", "string", "")),
    };
    let init = arg(l, 2).as_number().map_or(1, |n| n.max(1.0) as usize);
    let plain = arg(l, 3).is_truthy();

    if plain {
        if let Some(pos) = s[init.saturating_sub(1)..]
            .windows(pat.len())
            .position(|w| w == pat)
        {
            let start = init + pos;
            push(l, LuaValue::number(start as f64));
            push(l, LuaValue::number((start + pat.len() - 1) as f64));
            return Ok(2);
        }
        push(l, LuaValue::NIL);
        return Ok(1);
    }

    match find(s, pat, init.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            let caps_vec: Vec<CaptureValue> = caps.iter().cloned().collect();
            let n = caps_vec.len();
            l.stack[l.base] = LuaValue::number((start + 1) as f64);
            l.stack[l.base + 1] = LuaValue::number(end as f64);
            push_captures(l, &caps_vec, s, l.base + 2);
            l.top = l.base + 2 + n;
            Ok(2 + n as i32)
        }
        Ok(None) => {
            l.stack[l.base] = LuaValue::NIL;
            l.top = l.base + 1;
            Ok(1)
        }
        Err(e) => Err(l.runtime_error(e.as_bytes())),
    }
}

fn str_match(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "string.match", "string", "")),
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 2, "string.match", "string", "")),
    };
    let init = arg(l, 2).as_number().map_or(1, |n| n.max(1.0) as usize);

    match find(s, pat, init.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            let caps_vec: Vec<CaptureValue> = caps.iter().cloned().collect();
            let n = caps_vec.len();
            if n == 0 {
                let sid = l.heap().intern(&s[start..end]);
                l.stack[l.base] = l.heap().str_value(sid);
                l.top = l.base + 1;
                Ok(1)
            } else {
                push_captures(l, &caps_vec, s, l.base);
                l.top = l.base + n;
                Ok(n as i32)
            }
        }
        Ok(None) => {
            l.stack[l.base] = LuaValue::NIL;
            l.top = l.base + 1;
            Ok(1)
        }
        Err(e) => Err(l.runtime_error(e.as_bytes())),
    }
}

fn str_gmatch(l: &mut LuaState) -> LuaResult<i32> {
    let text = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 1, "string.gmatch", "string", "")),
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 2, "string.gmatch", "string", "")),
    };
    let text_sid = l.heap().intern(&text);
    let pat_sid = l.heap().intern(&pat);
    let closure = l
        .heap()
        .alloc_func(crate::func::GcFunc::C(crate::func::CClosure {
            f: gmatch_iter,
            env: l.global().globals,
            upvals: vec![
                l.heap().str_value(text_sid),
                l.heap().str_value(pat_sid),
                LuaValue::number(1.0),
            ],
        }));
    l.stack[l.base] = LuaValue::func(closure);
    l.top = l.base + 1;
    Ok(1)
}

fn gmatch_iter(l: &mut LuaState) -> LuaResult<i32> {
    let text_sid = match l.upvalue(0).as_string_id() {
        Some(sid) => sid,
        None => return Ok(0),
    };
    let pat_sid = match l.upvalue(1).as_string_id() {
        Some(sid) => sid,
        None => return Ok(0),
    };
    let pos = l.upvalue(2).as_number().unwrap_or(1.0) as usize;
    let text = l.heap().strings.get(text_sid).to_vec();
    let pat = l.heap().strings.get(pat_sid).to_vec();

    match find(&text, &pat, pos.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            l.set_upvalue(2, LuaValue::number((end + 1) as f64));
            let caps_vec: Vec<CaptureValue> = caps.iter().cloned().collect();
            if caps_vec.is_empty() {
                let sid = l.heap().intern(&text[start..end]);
                l.stack[l.base] = l.heap().str_value(sid);
                l.top = l.base + 1;
                Ok(1)
            } else {
                push_captures(l, &caps_vec, &text, l.base);
                l.top = l.base + caps_vec.len();
                Ok(caps_vec.len() as i32)
            }
        }
        Ok(None) => {
            l.stack[l.base] = LuaValue::NIL;
            l.top = l.base + 1;
            Ok(1)
        }
        Err(_) => {
            l.stack[l.base] = LuaValue::NIL;
            l.top = l.base + 1;
            Ok(1)
        }
    }
}

fn str_gsub(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 1, "string.gsub", "string", "")),
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 2, "string.gsub", "string", "")),
    };
    let repl = match arg(l, 2).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => return Err(err_bad_arg(l, 3, "string.gsub", "string", "")),
    };
    let max = arg(l, 3).as_number().map(|n| n as usize);

    match gsub(&s, &pat, &repl, max) {
        Ok((result, count)) => {
            let sid = l.heap().intern(&result);
            l.stack[l.base] = l.heap().str_value(sid);
            l.stack[l.base + 1] = LuaValue::number(count as f64);
            l.top = l.base + 2;
            Ok(2)
        }
        Err(e) => Err(l.runtime_error(e.as_bytes())),
    }
}

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
        .func(b"find", str_find)
        .func(b"format", str_format)
        .func(b"gmatch", str_gmatch)
        .func(b"gsub", str_gsub)
        .func(b"len", str_len)
        .func(b"lower", str_lower)
        .func(b"match", str_match)
        .func(b"rep", str_rep)
        .func(b"reverse", str_reverse)
        .func(b"sub", str_sub)
        .func(b"upper", str_upper)
        .build();
}
