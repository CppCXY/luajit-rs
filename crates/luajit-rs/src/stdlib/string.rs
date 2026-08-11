//! String library: `string.byte`, `string.char`, `string.dump`,
//! `string.find`, `string.format`, `string.gmatch`, `string.gsub`,
//! `string.len`, `string.lower`, `string.match`, `string.rep`,
//! `string.reverse`, `string.sub`, `string.upper`.

use crate::api::lua_gettop;
use crate::err::LuaResult;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg_type, push, tostring_bytes};
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
                // Already 1-based (engine adds +1).
                l.stack[base + i] = LuaValue::number(*p as f64);
            }
        }
    }
}

fn str_find(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => {
            return Err(err_bad_arg_type(l, 1, "string.find", "string", arg(l, 0)));
        }
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => {
            return Err(err_bad_arg_type(
                l,
                2,
                "string.find",
                "string",
                arg(l, 2 - 1),
            ));
        }
    };
    // init: negative counts from the end (1-based: len + init + 1).
    let init_n = arg(l, 2).as_number().unwrap_or(1.0);
    let init = if init_n < 0.0 {
        let len = s.len() as f64;
        (len + init_n + 1.0).max(1.0) as usize
    } else {
        init_n.max(1.0) as usize
    };
    let plain = arg(l, 3).is_truthy();

    if plain {
        if let Some(pos) = s[init.saturating_sub(1)..]
            .windows(pat.len())
            .position(|w| w == pat)
        {
            let start = init + pos;
            l.stack_ensure(l.base + 2);
            l.stack[l.base] = LuaValue::number(start as f64);
            l.stack[l.base + 1] = LuaValue::number((start + pat.len() - 1) as f64);
            l.top = l.base + 2;
            return Ok(2);
        }
        push(l, LuaValue::NIL);
        return Ok(1);
    }

    match find(s, pat, init.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            let caps_vec: Vec<CaptureValue> = caps.iter().cloned().collect();
            let n = caps_vec.len();
            l.stack_ensure(l.base + 2 + n);
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
        None => {
            return Err(err_bad_arg_type(l, 1, "string.match", "string", arg(l, 0)));
        }
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => {
            return Err(err_bad_arg_type(
                l,
                2,
                "string.match",
                "string",
                arg(l, 2 - 1),
            ));
        }
    };
    let init = arg(l, 2).as_number().map_or(1, |n| n.max(1.0) as usize);

    match find(s, pat, init.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            let n = caps.len();
            if n == 0 {
                let sid = l.heap().intern(&s[start..end]);
                l.stack[l.base] = l.heap().str_value(sid);
                l.top = l.base + 1;
                Ok(1)
            } else {
                for (i, capture) in caps.iter().enumerate() {
                    match capture {
                        CaptureValue::Substring(st, en) => {
                            let sid = l.heap().intern(&s[*st..*en]);
                            l.stack[l.base + i] = l.heap().str_value(sid);
                        }
                        CaptureValue::Position(p) => {
                            l.stack[l.base + i] = LuaValue::number(*p as f64);
                        }
                    }
                }
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
    let text_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "string.gmatch", "string", arg(l, 0)));
        }
    };
    let pat_sid = match arg(l, 1).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(
                l,
                2,
                "string.gmatch",
                "string",
                arg(l, 2 - 1),
            ));
        }
    };
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
    let text = l.str_static(text_sid);
    let pat = l.str_static(pat_sid);

    match find(text, pat, pos.saturating_sub(1)) {
        Ok(Some((start, end, caps))) => {
            // Advance past empty matches (5.1 gmatch_aux: `e == src` skips
            // one char so `()` on "abcde" yields positions 1..6).
            let next = if end == start { start + 1 } else { end };
            l.set_upvalue(2, LuaValue::number((next + 1) as f64));
            let n = caps.len();
            if n == 0 {
                let sid = l.heap().intern(&text[start..end]);
                l.stack[l.base] = l.heap().str_value(sid);
                l.top = l.base + 1;
                Ok(1)
            } else {
                for (i, capture) in caps.iter().enumerate() {
                    match capture {
                        CaptureValue::Substring(st, en) => {
                            let sid = l.heap().intern(&text[*st..*en]);
                            l.stack[l.base + i] = l.heap().str_value(sid);
                        }
                        CaptureValue::Position(p) => {
                            l.stack[l.base + i] = LuaValue::number(*p as f64);
                        }
                    }
                }
                l.top = l.base + n;
                Ok(n as i32)
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
        None => {
            return Err(err_bad_arg_type(l, 1, "string.gsub", "string", arg(l, 0)));
        }
    };
    let pat = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => {
            return Err(err_bad_arg_type(
                l,
                2,
                "string.gsub",
                "string",
                arg(l, 2 - 1),
            ));
        }
    };
    let repl_arg = arg(l, 2);
    let max = arg(l, 3).as_number().map(|n| n as usize);

    if repl_arg.is_func() {
        let (result, count) = gsub_fn(l, &s, &pat, repl_arg, max)?;
        let sid = l.heap().intern(&result);
        l.stack_ensure(l.base + 2);
        l.stack[l.base] = l.heap().str_value(sid);
        l.stack[l.base + 1] = LuaValue::number(count as f64);
        l.top = l.base + 2;
        Ok(2)
    } else if let Some(tab) = repl_arg.as_table() {
        let (result, count) = gsub_tab(l, &s, &pat, tab, max)?;
        let sid = l.heap().intern(&result);
        l.stack_ensure(l.base + 2);
        l.stack[l.base] = l.heap().str_value(sid);
        l.stack[l.base + 1] = LuaValue::number(count as f64);
        l.top = l.base + 2;
        Ok(2)
    } else if let Some(n) = repl_arg.as_number() {
        let repl = crate::strfmt::g14(n).as_bytes().to_vec();
        match gsub(&s, &pat, &repl, max) {
            Ok((result, count)) => {
                let sid = l.heap().intern(&result);
                l.stack_ensure(l.base + 2);
                l.stack[l.base] = l.heap().str_value(sid);
                l.stack[l.base + 1] = LuaValue::number(count as f64);
                l.top = l.base + 2;
                Ok(2)
            }
            Err(e) => Err(l.runtime_error(e.as_bytes())),
        }
    } else {
        let repl = match repl_arg.as_string_id() {
            Some(sid) => l.str_static(sid).to_vec(),
            None => {
                return Err(err_bad_arg_type(
                    l,
                    3,
                    "string.gsub",
                    "string or function",
                    arg(l, 3 - 1),
                ));
            }
        };
        match gsub(&s, &pat, &repl, max) {
            Ok((result, count)) => {
                let sid = l.heap().intern(&result);
                l.stack_ensure(l.base + 2);
                l.stack[l.base] = l.heap().str_value(sid);
                l.stack[l.base + 1] = LuaValue::number(count as f64);
                l.top = l.base + 2;
                Ok(2)
            }
            Err(e) => Err(l.runtime_error(e.as_bytes())),
        }
    }
}

fn gsub_fn(
    l: &mut LuaState,
    s: &[u8],
    pat: &[u8],
    func: LuaValue,
    max: Option<usize>,
) -> Result<(Vec<u8>, usize), crate::err::LuaError> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut count = 0;
    loop {
        if let Some(limit) = max
            && count >= limit
        {
            break;
        }
        match crate::stdlib::pattern::find(s, pat, pos) {
            Ok(Some((m_start, m_end, caps))) => {
                out.extend_from_slice(&s[pos..m_start]);
                let mut args: Vec<LuaValue> = Vec::new();
                if caps.is_empty() {
                    // No captures: the whole match is the single argument.
                    let sid = l.heap().intern(&s[m_start..m_end]);
                    args.push(l.heap().str_value(sid));
                }
                for i in 0..caps.len() {
                    match caps.get(i) {
                        Some(crate::stdlib::pattern::CaptureValue::Substring(cs, ce)) => {
                            let sid = l.heap().intern(&s[*cs..*ce]);
                            args.push(l.heap().str_value(sid));
                        }
                        Some(crate::stdlib::pattern::CaptureValue::Position(p)) => {
                            // Already 1-based relative to the source start.
                            args.push(LuaValue::number(*p as f64));
                        }
                        None => args.push(LuaValue::NIL),
                    }
                }
                let r = call_lua_fn(l, func, &args)?;
                if r.is_nil() || r.is_false() {
                    // Lua 5.1: a nil/false result keeps the original match.
                    out.extend_from_slice(&s[m_start..m_end]);
                } else if let Some(sid) = r.as_string_id() {
                    out.extend_from_slice(l.str_static(sid));
                } else {
                    let ts = crate::stdlib::tostring_bytes(l, r);
                    out.extend_from_slice(&ts);
                }
                count += 1;
                if m_end == m_start {
                    if m_end == s.len() {
                        break;
                    }
                    out.push(s[m_end]);
                    pos = m_end + 1;
                } else {
                    pos = m_end;
                }
            }
            Ok(None) => break,
            Err(e) => return Err(l.runtime_error(e.as_bytes())),
        }
    }
    out.extend_from_slice(&s[pos..]);
    Ok((out, count))
}

/// `string.gsub` with a table replacement: the key is the first capture
/// (or the whole match when there are no captures); nil/false values keep
/// the original match (Lua 5.1 `add_value`/`push_onecapture`).
fn gsub_tab(
    l: &mut LuaState,
    s: &[u8],
    pat: &[u8],
    tab: crate::gc::GcPtr<crate::table::LuaTable>,
    max: Option<usize>,
) -> Result<(Vec<u8>, usize), crate::err::LuaError> {
    let mut out = Vec::new();
    let mut pos = 0;
    let mut count = 0;
    loop {
        if let Some(limit) = max
            && count >= limit
        {
            break;
        }
        match crate::stdlib::pattern::find(s, pat, pos) {
            Ok(Some((m_start, m_end, caps))) => {
                out.extend_from_slice(&s[pos..m_start]);
                let kval = if !caps.is_empty() {
                    match caps.get(0) {
                        Some(CaptureValue::Substring(cs, ce)) => {
                            let sid = l.heap().intern(&s[*cs..*ce]);
                            l.heap().str_value(sid)
                        }
                        Some(CaptureValue::Position(p)) => LuaValue::number(*p as f64),
                        None => {
                            let sid = l.heap().intern(&s[m_start..m_end]);
                            l.heap().str_value(sid)
                        }
                    }
                } else {
                    let sid = l.heap().intern(&s[m_start..m_end]);
                    l.heap().str_value(sid)
                };
                let v = gettable(l, LuaValue::table(tab), kval)?;
                if v.is_nil() || v.is_false() {
                    // nil/false: keep the original match.
                    out.extend_from_slice(&s[m_start..m_end]);
                } else if let Some(sid) = v.as_string_id() {
                    out.extend_from_slice(l.str_static(sid));
                } else if let Some(n) = v.as_number() {
                    out.extend_from_slice(crate::strfmt::g14(n).as_bytes());
                } else {
                    let tn = crate::stdlib::type_name(v);
                    return Err(l.runtime_error(
                        format!("invalid replacement value (a {})", tn).as_bytes(),
                    ));
                }
                count += 1;
                if m_end == m_start {
                    if m_end == s.len() {
                        break;
                    }
                    out.push(s[m_end]);
                    pos = m_end + 1;
                } else {
                    pos = m_end;
                }
            }
            Ok(None) => break,
            Err(e) => return Err(l.runtime_error(e.as_bytes())),
        }
    }
    out.extend_from_slice(&s[pos..]);
    Ok((out, count))
}

/// `lua_gettable`-style lookup honoring `__index` chains (5.1 `gsub`
/// uses a plain `lua_gettable` on the replacement table).
fn gettable(
    l: &mut LuaState,
    mut cur: LuaValue,
    k: LuaValue,
) -> Result<LuaValue, crate::err::LuaError> {
    for _ in 0..100 {
        let Some(t) = cur.as_table() else {
            return Err(l.runtime_error(b"attempt to index a non-table value"));
        };
        let tv = t.as_ref().get(k);
        if !tv.is_nil() {
            return Ok(tv);
        }
        let mo = crate::meta::meta_fast(l.global(), t.as_ref().metatable, crate::meta::MM::Index);
        match mo {
            Some(mo) if mo.is_func() => return call_lua_fn(l, mo, &[cur, k]),
            Some(mo) => cur = mo,
            None => return Ok(LuaValue::NIL),
        }
    }
    Err(l.runtime_error(b"'__index' chain too long; possible loop"))
}

fn call_lua_fn(
    l: &mut LuaState,
    func: LuaValue,
    args: &[LuaValue],
) -> Result<LuaValue, crate::err::LuaError> {
    let saved_top = l.top;
    let saved_base = l.base;
    let fs = l.top + 16;
    l.stack_ensure(fs + 4 + args.len());
    l.stack[fs] = func;
    for (i, a) in args.iter().enumerate() {
        l.stack[fs + 2 + i] = *a;
    }
    let _ = crate::vm::execute(l, fs, args.len(), 1)?;
    let r = l.stack[fs];
    l.top = saved_top;
    l.base = saved_base;
    Ok(r)
}

pub fn str_byte(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => {
            return Err(err_bad_arg_type(l, 1, "string.byte", "string", arg(l, 0)));
        }
    };
    let i = arg(l, 1).as_number().unwrap_or(1.0) as i64;
    let j = arg(l, 2).as_number().map_or(i, |n| n as i64);
    let len = s.len() as i64;
    // Lua 5.1: negative indices count from the end, then clamp to
    // [1, len] (1-based).
    let mut lo = if i < 0 { len + i + 1 } else { i };
    let mut hi = if j < 0 { len + j + 1 } else { j };
    if lo < 1 {
        lo = 1;
    }
    if hi > len {
        hi = len;
    }
    if lo > hi {
        // LuaJIT: an empty interval returns no values (a 0-value
        // expression compares as nil, but char(byte(..., 1, 0)) gets 0
        // arguments).
        Ok(0)
    } else {
        let lo = (lo - 1) as usize;
        let hi = (hi - 1) as usize;
        l.stack_ensure(l.base + (hi - lo) + 1);
        for (k, &b) in s[lo..=hi].iter().enumerate() {
            l.stack[l.base + k] = LuaValue::number(b as f64);
        }
        Ok((hi - lo + 1) as i32)
    }
}

pub fn str_char(l: &mut LuaState) -> LuaResult<i32> {
    let n = lua_gettop(l);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = match num_arg_coerce(l, i) {
            Some(c) => c,
            None => {
                return Err(err_bad_arg_type(
                    l,
                    i as u32 + 1,
                    "char",
                    "number",
                    arg(l, i),
                ));
            }
        };
        if !(0.0..=255.0).contains(&c) || c.fract() != 0.0 {
            return Err(l.runtime_error(b"out of range"));
        }
        out.push(c as u8);
    }
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn str_dump(l: &mut LuaState) -> LuaResult<i32> {
    let fv = arg(l, 0);
    match fv.as_func() {
        Some(_gf) => {
            let g = l.global();
            // Cache the function in the registry table (GC-marked, and
            // invisible to the global namespace) for loadstring round-trip.
            let cache_key = g.heap.intern(b"__LUARS_DUMP_CACHE");
            let key = g.heap.str_value(cache_key);
            let registry = g.registry.as_mut();
            let cache = match registry.get(key) {
                v if v.is_table() => v.as_table().unwrap(),
                _ => {
                    let t = g.heap.alloc_table(LuaTable::new(0, 0));
                    registry.set(key, LuaValue::table(t));
                    t
                }
            };
            let idx = cache.as_ref().len();
            cache.as_mut().set_int(idx as i32, fv);
            let data = format!("\x1bLJ{}", idx);
            let sid = l.heap().intern(data.as_bytes());
            push(l, l.heap().str_value(sid));
            Ok(1)
        }
        None => Err(err_bad_arg_type(l, 1, "string.dump", "function", arg(l, 0))),
    }
}

pub fn str_format(l: &mut LuaState) -> LuaResult<i32> {
    let fmt_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "string.format", "string", arg(l, 0)));
        }
    };
    let fmt = l.str_static(fmt_sid);
    let n = lua_gettop(l);
    // Borrow string arguments (zero-copy); only non-string values are
    // materialized, and their buffers are pinned in `owned`.
    let mut owned: Vec<Vec<u8>> = Vec::new();
    for i in 1..n {
        let v = arg(l, i);
        if !v.as_number().is_some() && !v.is_string() {
            owned.push(tostring_bytes(l, v));
        }
    }
    let mut oi = 0;
    let mut args: Vec<crate::strfmt::FmtArg> = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let v = arg(l, i);
        if let Some(n) = v.as_number() {
            args.push(crate::strfmt::FmtArg::Num(n));
        } else if let Some(sid) = v.as_string_id() {
            args.push(crate::strfmt::FmtArg::Str(l.str_static(sid)));
        } else {
            args.push(crate::strfmt::FmtArg::Str(&owned[oi]));
            oi += 1;
        }
    }
    use std::sync::Arc;
    let parts = if let Some(p) = l.heap().fmt_cache.get(&fmt_sid) {
        Arc::clone(p)
    } else {
        let compiled = crate::strfmt::compile(fmt).map_err(|e| l.runtime_error(e.as_bytes()))?;
        let p = Arc::new(compiled);
        l.heap().fmt_cache.insert(fmt_sid, Arc::clone(&p));
        p
    };
    match crate::strfmt::format_parts(&parts, &args) {
        Ok(bytes) => {
            let sid = l.heap().intern(&bytes);
            push(l, l.heap().str_value(sid));
            Ok(1)
        }
        Err(msg) => Err(l.runtime_error(msg.as_bytes())),
    }
}

pub fn str_len(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let len = if let Some(sid) = v.as_string_id() {
        l.str_static(sid).len()
    } else if v.is_number() {
        // Lua 5.1: numbers are coerced to strings (luaL_checklstring).
        crate::stdlib::tostring_bytes(l, v).len()
    } else {
        return Err(err_bad_arg_type(l, 1, "len", "string", arg(l, 0)));
    };
    push(l, LuaValue::number(len as f64));
    Ok(1)
}

fn map_bytes(l: &mut LuaState, f: fn(u8) -> u8) -> LuaResult<i32> {
    let s = match str_arg_coerce(l, 0, "string") {
        Some(s) => s,
        None => return Err(err_bad_arg_type(l, 1, "string", "string", arg(l, 0))),
    };
    let out: Vec<u8> = s.iter().map(|&b| f(b)).collect();
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn str_lower(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_lowercase())
}
pub fn str_upper(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_uppercase())
}

fn str_rep(l: &mut LuaState) -> LuaResult<i32> {
    let s = match str_arg_coerce(l, 0, "rep") {
        Some(s) => s,
        None => {
            return Err(err_bad_arg_type(l, 1, "string.rep", "string", arg(l, 0)));
        }
    };
    let n = num_arg_coerce(l, 1).unwrap_or(0.0) as i64;
    let sep = str_arg_coerce(l, 2, "rep").unwrap_or_default();
    let n = n.max(0) as usize;
    let mut out = Vec::with_capacity(s.len() * n + sep.len() * n.saturating_sub(1));
    for i in 0..n {
        if i > 0 {
            out.extend_from_slice(&sep);
        }
        out.extend_from_slice(&s);
    }
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

/// Lua 5.1 string coercion: numbers are converted with tostring.
fn str_arg_coerce(l: &mut LuaState, i: usize, _name: &str) -> Option<Vec<u8>> {
    let v = arg(l, i);
    if let Some(sid) = v.as_string_id() {
        Some(l.str_static(sid).to_vec())
    } else if v.is_number() {
        Some(crate::stdlib::tostring_bytes(l, v))
    } else {
        None
    }
}

/// Numeric coercion for string-library indices (Lua 5.1's luaL_optint
/// accepts numeric strings).
fn num_arg_coerce(l: &mut LuaState, i: usize) -> Option<f64> {
    let v = arg(l, i);
    if let Some(n) = v.as_number() {
        return Some(n);
    }
    if let Some(sid) = v.as_string_id() {
        return crate::strscan::scan_number(l.str_static(sid));
    }
    None
}

pub fn str_reverse(l: &mut LuaState) -> LuaResult<i32> {
    let bytes = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => {
            let v = arg(l, 0);
            if v.is_number() {
                // Lua 5.1: numbers are coerced to strings.
                crate::stdlib::tostring_bytes(l, v)
            } else {
                return Err(err_bad_arg_type(l, 1, "reverse", "string", arg(l, 0)));
            }
        }
    };
    let rev: Vec<u8> = bytes.iter().copied().rev().collect();
    let sid = l.heap().intern(&rev);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

/// True when the running string-library function was invoked as a method
/// (`s:sub(...)`): the caller's bytecode is `MOV(self,obj); TGETS(func,
/// obj,key); CALL(func,..)` (lj_debug.c's "method" pattern). LuaJIT then
/// numbers the arguments from the receiver, so the self is not counted in
/// error messages.
fn is_method_call(l: &LuaState) -> bool {
    use crate::bc::{BCIns, BCOp, bc_a, bc_b, bc_d, bc_op};
    // The C function's frame: func at base-2, link (the caller's return
    // PC) at base-1.
    let Some(link) = l.stack.get(l.base.wrapping_sub(1)).map(|v| v.to_bits()) else {
        return false;
    };
    // FRAME_LUA only (frame type 0). The link is either the caller's
    // return PC or a caller-base delta (the C-call frames).
    if link & 0x7 != 0 || link == 0 {
        return false;
    }
    let a;
    let caller_base;
    let mut pc_from_link: Option<usize> = None;
    if ((link >> 3) as usize) < l.stack.len() {
        // C-call frame: link = the caller's base. The func sits at
        // l.base-2, which is caller_base + a.
        caller_base = (link >> 3) as usize;
        a = (l.base.saturating_sub(2).saturating_sub(caller_base)) as u32;
    } else {
        let ret_ip = link as *const BCIns;
        let call_ins = unsafe { *ret_ip.sub(1) };
        if bc_op(call_ins) != BCOp::CALL {
            return false;
        }
        a = bc_a(call_ins);
        caller_base = l.base.saturating_sub(4 + a as usize);
        pc_from_link = Some(ret_ip as usize);
    };
    if caller_base < 2 {
        return false;
    }
    let Some(cf) = l.stack.get(caller_base - 2).and_then(|v| v.as_func()) else {
        return false;
    };
    let crate::func::GcFunc::Lua(cl) = cf.as_ref() else {
        return false;
    };
    let pt = cl.proto.as_ref();
    let pc = match pc_from_link {
        Some(addr) => {
            let p = addr as *const BCIns;
            (unsafe { p.offset_from(pt.bc.as_ptr()) }) as usize
        }
        None => l.debug_pc,
    };
    if pc < 2 {
        return false;
    }
    // Scan back from the CALL over the argument-setup instructions to
    // the TGETS and the preceding MOV(self, obj) — the method pattern.
    let mut k = pc - 1;
    while k > 0 {
        match bc_op(pt.bc[k - 1]) {
            BCOp::TGETS | BCOp::TGETV => {
                let tgets = pt.bc[k - 1];
                if k >= 2 {
                    let mov = pt.bc[k - 2];
                    // The VM compiles methods in the frame-2 layout:
                    // MOV(self at func+1+fr2, obj); TGETS(func, obj, key).
                    return bc_op(mov) == BCOp::MOV
                        && bc_a(mov) == a + 1 + 1
                        && bc_d(mov) == bc_b(tgets);
                }
                return false;
            }
            BCOp::KSHORT | BCOp::KPRI | BCOp::KSTR | BCOp::KNUM | BCOp::KCDATA | BCOp::MOV => {
                k -= 1;
            }
            _ => return false,
        }
    }
    false
}

pub fn str_sub(l: &mut LuaState) -> LuaResult<i32> {
    let s: &[u8];
    let owned: Vec<u8>;
    match arg(l, 0).as_string_id() {
        Some(sid) => s = l.str_static(sid),
        None => {
            let v = arg(l, 0);
            if v.is_number() {
                owned = crate::stdlib::tostring_bytes(l, v);
                s = &owned;
            } else {
                return Err(err_bad_arg_type(l, 1, "sub", "string", arg(l, 0)));
            }
        }
    }
    // The indices must be numbers (or numeric strings); a present but
    // non-numeric argument is an error (luaL_checknumber semantics).
    let method = is_method_call(l);
    let argnum = |n: u32| if method { n - 1 } else { n };
    let coerce_index = |l: &mut LuaState, i: usize, name: &str| -> LuaResult<i64> {
        let v = arg(l, i);
        if v.is_nil() {
            return Ok(if i == 1 { 1 } else { i64::MAX });
        }
        match num_arg_coerce(l, i) {
            Some(n) => Ok(n as i64),
            None => Err(err_bad_arg_type(l, argnum(i as u32 + 1), name, "number", v)),
        }
    };
    let i = coerce_index(l, 1, "sub")?;
    let j = if arg(l, 2).is_nil() {
        None
    } else {
        Some(coerce_index(l, 2, "sub")?)
    };
    let len = s.len() as i64;
    let a = if i < 0 {
        (len + i).max(0) as usize
    } else {
        (i - 1).max(0).min(len) as usize
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
    let strtab = lual_reg!(l, b"string", LibTarget::Global)
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
    // LUA_COMPAT_GFIND: gfind is the *same* closure as gmatch
    // (lstrlib.c: `lua_getfield(gmatch); lua_setfield(gfind)`), so the
    // test `string.gfind == string.gmatch` holds.
    {
        let g = l.global();
        let sid = g.heap.intern(b"gmatch");
        let val = strtab.as_ref().get(g.heap.str_value(sid));
        let sid = g.heap.intern(b"gfind");
        strtab.as_mut().set(g.heap.str_value(sid), val);
    }

    // Base metatable for strings: __index = string table (lib_string.c's
    // LJLIB_MODULE mt setup: `s:upper()` etc. resolve through it).
    use crate::meta::MM;
    use crate::table::LuaTable;
    use crate::value::LJ_TSTR;
    let g = l.global();
    let mt = g.heap.alloc_table(LuaTable::new(0, 1));
    let key = g.mmname[MM::Index as usize];
    mt.as_mut().set_str(key, LuaValue::table(strtab));
    // Negative-cache everything except __index (lib_string.c does the same).
    mt.as_mut().nomm = !(1u8 << (MM::Index as u8));
    g.set_basemt(LJ_TSTR, Some(mt));
}
