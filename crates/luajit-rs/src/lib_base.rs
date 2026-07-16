//! A minimal base library: just enough builtins to run simple programs and
//! the benchmark scripts (`print`, `type`, `tostring`, `select`, `pairs`,
//! `ipairs`, `next`, plus `os.clock` and `string.format`).

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::strfmt::{self, FmtArg};
use crate::value::{LuaValue, LJ_TNIL};

/// Argument `i` (0-based) of the current C call, or nil.
fn arg(l: &LuaState, i: usize) -> LuaValue {
    let slot = l.base + i;
    if slot < l.top {
        l.stack[slot]
    } else {
        LuaValue::NIL
    }
}

fn nargs(l: &LuaState) -> usize {
    l.top - l.base
}

/// Convert a value to its display bytes (`tostring` without metamethods).
fn tostring_bytes(l: &mut LuaState, v: LuaValue) -> Vec<u8> {
    if let Some(sid) = v.as_string_id() {
        return l.heap().strings.get(sid).to_vec();
    }
    if let Some(n) = v.as_number() {
        return strfmt::g14(n).into_bytes();
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

fn type_name(v: LuaValue) -> &'static [u8] {
    if v.is_nil() {
        b"nil"
    } else if v.is_bool() {
        b"boolean"
    } else if v.is_number() {
        b"number"
    } else if v.is_string() {
        b"string"
    } else if v.is_table() {
        b"table"
    } else if v.is_func() {
        b"function"
    } else {
        b"userdata"
    }
}

fn lib_print(l: &mut LuaState) -> LuaResult<i32> {
    use std::io::Write;
    let n = nargs(l);
    let mut out = Vec::new();
    for i in 0..n {
        if i > 0 {
            out.push(b'\t');
        }
        let v = arg(l, i);
        out.extend_from_slice(&tostring_bytes(l, v));
    }
    out.push(b'\n');
    let stdout = std::io::stdout();
    let mut h = stdout.lock();
    let _ = h.write_all(&out);
    Ok(0)
}

fn lib_type(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let sid = l.heap().intern(type_name(v));
    l.stack[l.base] = l.heap().str_value(sid);
    Ok(1)
}

fn lib_tostring(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let bytes = tostring_bytes(l, v);
    let sid = l.heap().intern(&bytes);
    l.stack[l.base] = l.heap().str_value(sid);
    Ok(1)
}

fn lib_tonumber(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let r = if v.is_number() {
        v
    } else if let Some(sid) = v.as_string_id() {
        let bytes = l.heap().strings.get(sid).to_vec();
        match crate::strscan::scan_number(&bytes) {
            Some(n) => LuaValue::number(n),
            None => LuaValue::NIL,
        }
    } else {
        LuaValue::NIL
    };
    l.stack[l.base] = r;
    Ok(1)
}

fn lib_select(l: &mut LuaState) -> LuaResult<i32> {
    let first = arg(l, 0);
    let n = nargs(l);
    if let Some(sid) = first.as_string_id() {
        if l.heap().strings.get(sid) == b"#" {
            l.stack[l.base] = LuaValue::number((n - 1) as f64);
            return Ok(1);
        }
    }
    let k = match first.as_number() {
        Some(k) if k >= 1.0 => k as usize,
        _ => return Err(l.runtime_error(b"bad argument #1 to 'select'")),
    };
    let mut cnt = 0;
    for i in k..(n) {
        l.stack[l.base + cnt] = arg(l, i);
        cnt += 1;
    }
    Ok(cnt as i32)
}

fn lib_next(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(l.runtime_error(b"bad argument #1 to 'next' (table expected)")),
    };
    match tab.as_ref().next(k) {
        Some((nk, nv)) => {
            l.stack[l.base] = nk;
            l.stack[l.base + 1] = nv;
            Ok(2)
        }
        None => {
            l.stack[l.base] = LuaValue::NIL;
            Ok(1)
        }
    }
}

fn lib_pairs(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let next_sid = l.heap().intern(b"next");
    let key = l.heap().str_value(next_sid);
    let next_fn = l.global().globals.as_ref().get(key);
    l.stack[l.base] = next_fn;
    l.stack[l.base + 1] = t;
    l.stack[l.base + 2] = LuaValue::NIL;
    Ok(3)
}

fn lib_ipairs_iter(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let i = arg(l, 1).as_number().unwrap_or(0.0) + 1.0;
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(l.runtime_error(b"bad argument #1 to ipairs iterator")),
    };
    let v = tab.as_ref().get_int(i as i32);
    if v.is_nil() {
        l.stack[l.base] = LuaValue::NIL;
        Ok(1)
    } else {
        l.stack[l.base] = LuaValue::number(i);
        l.stack[l.base + 1] = v;
        Ok(2)
    }
}

fn lib_ipairs(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let iter_sid = l.heap().intern(b"__ipairs_iter");
    let key = l.heap().str_value(iter_sid);
    let iter = l.global().globals.as_ref().get(key);
    l.stack[l.base] = iter;
    l.stack[l.base + 1] = t;
    l.stack[l.base + 2] = LuaValue::number(0.0);
    Ok(3)
}

fn lib_setmetatable(l: &mut LuaState) -> LuaResult<i32> {
    // Metatables are not consulted yet; accept and return the table.
    let t = arg(l, 0);
    l.stack[l.base] = t;
    Ok(1)
}

fn lib_assert(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    if v.is_truthy() {
        let n = nargs(l);
        Ok(n as i32)
    } else {
        let msg = arg(l, 1);
        if msg.is_nil() {
            Err(l.runtime_error(b"assertion failed!"))
        } else {
            l.errval = msg;
            Err(crate::err::LuaError::Runtime)
        }
    }
}

fn os_clock(l: &mut LuaState) -> LuaResult<i32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    l.stack[l.base] = LuaValue::number(secs);
    Ok(1)
}

fn os_time(l: &mut LuaState) -> LuaResult<i32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    l.stack[l.base] = LuaValue::number(secs as f64);
    Ok(1)
}

fn str_format(l: &mut LuaState) -> LuaResult<i32> {
    let fmt = match arg(l, 0).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => return Err(l.runtime_error(b"bad argument #1 to 'format' (string expected)")),
    };
    let n = nargs(l);
    // Materialize args as owned bytes/numbers to avoid borrow conflicts.
    enum Owned {
        Num(f64),
        Str(Vec<u8>),
    }
    let mut owned = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let v = arg(l, i);
        if let Some(num) = v.as_number() {
            owned.push(Owned::Num(num));
        } else if let Some(sid) = v.as_string_id() {
            owned.push(Owned::Str(l.heap().strings.get(sid).to_vec()));
        } else {
            owned.push(Owned::Str(tostring_bytes(l, v)));
        }
    }
    let args: Vec<FmtArg> = owned
        .iter()
        .map(|o| match o {
            Owned::Num(n) => FmtArg::Num(*n),
            Owned::Str(s) => FmtArg::Str(s),
        })
        .collect();
    match strfmt::format(&fmt, &args) {
        Ok(bytes) => {
            let sid = l.heap().intern(&bytes);
            l.stack[l.base] = l.heap().str_value(sid);
            Ok(1)
        }
        Err(msg) => Err(l.runtime_error(msg.as_bytes())),
    }
}

fn str_len(l: &mut LuaState) -> LuaResult<i32> {
    let s = arg(l, 0);
    match s.as_string_id() {
        Some(sid) => {
            let n = l.heap().strings.get(sid).len();
            l.stack[l.base] = LuaValue::number(n as f64);
            Ok(1)
        }
        None => Err(l.runtime_error(b"bad argument #1 to 'len' (string expected)")),
    }
}

fn str_rep(l: &mut LuaState) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => return Err(l.runtime_error(b"bad argument #1 to 'rep' (string expected)")),
    };
    let n = arg(l, 1).as_number().unwrap_or(0.0) as i64;
    let mut out = Vec::new();
    for _ in 0..n.max(0) {
        out.extend_from_slice(&s);
    }
    let sid = l.heap().intern(&out);
    l.stack[l.base] = l.heap().str_value(sid);
    Ok(1)
}

fn str_upper(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_uppercase())
}

fn str_lower(l: &mut LuaState) -> LuaResult<i32> {
    map_bytes(l, |b| b.to_ascii_lowercase())
}

fn map_bytes(l: &mut LuaState, f: fn(u8) -> u8) -> LuaResult<i32> {
    let s = match arg(l, 0).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => return Err(l.runtime_error(b"bad argument #1 (string expected)")),
    };
    let out: Vec<u8> = s.iter().map(|&b| f(b)).collect();
    let sid = l.heap().intern(&out);
    l.stack[l.base] = l.heap().str_value(sid);
    Ok(1)
}

/// Create a global table `name` and populate it with `(field, fn)` entries.
fn make_lib(l: &mut LuaState, name: &[u8], entries: &[(&[u8], crate::func::CFunction)]) {
    let t = l.heap().alloc_table(crate::table::LuaTable::new(0, 4));
    for &(field, f) in entries {
        let sid = l.heap().intern(field);
        let env = l.global().globals;
        let fref = l.heap().alloc_func(crate::func::GcFunc::C(crate::func::CClosure {
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

/// Install the base library and the `os`/`string`/`table` subsets.
pub fn open_libs(l: &mut LuaState) {
    l.register(b"print", lib_print);
    l.register(b"type", lib_type);
    l.register(b"tostring", lib_tostring);
    l.register(b"tonumber", lib_tonumber);
    l.register(b"select", lib_select);
    l.register(b"next", lib_next);
    l.register(b"pairs", lib_pairs);
    l.register(b"ipairs", lib_ipairs);
    l.register(b"__ipairs_iter", lib_ipairs_iter);
    l.register(b"setmetatable", lib_setmetatable);
    l.register(b"assert", lib_assert);

    // Expose _G self-reference.
    let g = l.global().globals;
    let gsid = l.heap().intern(b"_G");
    let key = l.heap().str_value(gsid);
    g.as_mut().set(key, LuaValue::table(g));

    make_lib(l, b"os", &[(b"clock", os_clock), (b"time", os_time)]);
    make_lib(
        l,
        b"string",
        &[
            (b"format", str_format),
            (b"len", str_len),
            (b"rep", str_rep),
            (b"upper", str_upper),
            (b"lower", str_lower),
        ],
    );

    let _ = LJ_TNIL;
}
