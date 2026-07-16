//! Base library: `print`, `type`, `tostring`, `tonumber`, `select`,
//! `pairs`, `ipairs`, `next`, `assert`, `setmetatable`, `collectgarbage`,
//! `error`, `pcall`, `xpcall`, `rawequal`, `rawget`, `rawset`, `getmetatable`.

use crate::err::{LuaError, LuaResult};
use crate::state::LuaState;
use crate::value::{LJ_TNIL, LuaValue};

use super::{LibTarget, arg, err_bad_arg, nargs, push, pushv, tostring_bytes};
use crate::lual_reg;

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
    let _ = std::io::stdout().lock().write_all(&out);
    Ok(0)
}

fn lib_type(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let name: &[u8] = if v.is_nil() {
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
    };
    let sid = l.heap().intern(name);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn lib_tostring(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let bytes = tostring_bytes(l, v);
    let sid = l.heap().intern(&bytes);
    push(l, l.heap().str_value(sid));
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
    push(l, r);
    Ok(1)
}

fn lib_select(l: &mut LuaState) -> LuaResult<i32> {
    let first = arg(l, 0);
    let n = nargs(l);
    if let Some(sid) = first.as_string_id()
        && l.heap().strings.get(sid) == b"#"
    {
        push(l, LuaValue::number((n - 1) as f64));
        return Ok(1);
    }
    let k = match first.as_number() {
        Some(k) if k >= 1.0 => k as usize,
        _ => return Err(err_bad_arg(l, 1, "select", "number or '#'", "")),
    };
    let mut cnt = 0;
    for i in k..n {
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
        None => return Err(err_bad_arg(l, 1, "next", "table", "")),
    };
    match tab.as_ref().next(k) {
        Some((nk, nv)) => {
            pushv(l, &[nk, nv]);
            Ok(2)
        }
        None => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn lib_pairs(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let sid = l.heap().intern(b"next");
    let key = l.heap().str_value(sid);
    let next_fn = l.global().globals.as_ref().get(key);
    pushv(l, &[next_fn, t, LuaValue::NIL]);
    Ok(3)
}

fn lib_ipairs_iter(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let i = arg(l, 1).as_number().unwrap_or(0.0) + 1.0;
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "ipairs", "table", "")),
    };
    let v = tab.as_ref().get_int(i as i32);
    if v.is_nil() {
        push(l, LuaValue::NIL);
        Ok(1)
    } else {
        pushv(l, &[LuaValue::number(i), v]);
        Ok(2)
    }
}

fn lib_ipairs(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let sid = l.heap().intern(b"__ipairs_iter");
    let key = l.heap().str_value(sid);
    let iter = l.global().globals.as_ref().get(key);
    pushv(l, &[iter, t, LuaValue::number(0.0)]);
    Ok(3)
}

fn lib_setmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    push(l, t);
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
            Err(LuaError::Runtime)
        }
    }
}

fn lib_collectgarbage(l: &mut LuaState) -> LuaResult<i32> {
    let opt = match arg(l, 0).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => b"collect".to_vec(),
    };
    match opt.as_slice() {
        b"collect" | b"step" | b"full" => {
            crate::gc::full_gc(l.global());
            push(l, LuaValue::number(0.0));
            Ok(1)
        }
        b"count" => {
            let heap = &l.global().heap;
            let bytes = heap.total + heap.strings.bytes();
            push(l, LuaValue::number(bytes as f64 / 1024.0));
            Ok(1)
        }
        _ => Err(err_bad_arg(l, 1, "collectgarbage", "option string", "")),
    }
}

fn lib_rawget(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "rawget", "table", "")),
    };
    push(l, tab.as_ref().get(k));
    Ok(1)
}

fn lib_rawset(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let v = arg(l, 2);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "rawset", "table", "")),
    };
    tab.as_mut().set(k, v);
    push(l, t);
    Ok(1)
}

fn lib_rawequal(l: &mut LuaState) -> LuaResult<i32> {
    let a = arg(l, 0);
    let b = arg(l, 1);
    push(l, LuaValue::boolean(a.to_bits() == b.to_bits()));
    Ok(1)
}

fn lib_error(l: &mut LuaState) -> LuaResult<i32> {
    let msg = arg(l, 0);
    let level = arg(l, 1).as_number().unwrap_or(1.0) as i32;
    let _ = level; // stack-trace level not yet implemented
    l.errval = if msg.is_nil() { LuaValue::NIL } else { msg };
    Err(LuaError::Runtime)
}

/// `pcall(f [, arg...])` — protected call.
fn lib_pcall(l: &mut LuaState) -> LuaResult<i32> {
    let fv = arg(l, 0);
    let _gf = match fv.as_func() {
        Some(p) => p,
        None => return Err(err_bad_arg(l, 1, "pcall", "function", "")),
    };
    let n = nargs(l).saturating_sub(1);
    // Move `n` trailing args into call position right after `f`.
    for i in 0..n {
        l.stack[l.base + 2 + i] = arg(l, i + 1);
    }
    match crate::vm::execute(l, l.base, n, -1) {
        Ok(nret) => {
            // Shift results down so the true/false header can go first.
            for i in (0..nret).rev() {
                l.stack[l.base + i + 1] = l.stack[l.base + i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            Ok(nret as i32 + 1)
        }
        Err(LuaError::Runtime) => {
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = l.errval;
            Ok(2)
        }
        Err(e) => Err(e),
    }
}

/// `xpcall(f, msgh [, arg...])` — protected call with error handler.
fn lib_xpcall(l: &mut LuaState) -> LuaResult<i32> {
    let _msgh = arg(l, 1); // error handler (NYI: not invoked on error)
    let fv = arg(l, 0);
    let _gf = match fv.as_func() {
        Some(p) => p,
        None => return Err(err_bad_arg(l, 1, "xpcall", "function", "")),
    };
    let n = nargs(l).saturating_sub(2);
    for i in 0..n {
        l.stack[l.base + 2 + i] = arg(l, i + 2);
    }
    match crate::vm::execute(l, l.base, n, -1) {
        Ok(nret) => {
            for i in (0..nret).rev() {
                l.stack[l.base + i + 1] = l.stack[l.base + i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            Ok(nret as i32 + 1)
        }
        Err(LuaError::Runtime) => {
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = l.errval;
            Ok(2)
        }
        Err(e) => Err(e),
    }
}

fn lib_getmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let _v = arg(l, 0);
    // Metatables not yet implemented; return nil.
    push(l, LuaValue::NIL);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"", LibTarget::BaseLib)
        .func(b"print", lib_print)
        .func(b"type", lib_type)
        .func(b"tostring", lib_tostring)
        .func(b"tonumber", lib_tonumber)
        .func(b"select", lib_select)
        .func(b"next", lib_next)
        .func(b"pairs", lib_pairs)
        .func(b"ipairs", lib_ipairs)
        .func(b"__ipairs_iter", lib_ipairs_iter)
        .func(b"setmetatable", lib_setmetatable)
        .func(b"assert", lib_assert)
        .func(b"collectgarbage", lib_collectgarbage)
        .func(b"rawget", lib_rawget)
        .func(b"rawset", lib_rawset)
        .func(b"rawequal", lib_rawequal)
        .func(b"error", lib_error)
        .func(b"pcall", lib_pcall)
        .func(b"xpcall", lib_xpcall)
        .func(b"getmetatable", lib_getmetatable)
        .build();

    let gsid = l.heap().intern(b"_G");
    let key = l.heap().str_value(gsid);
    let g = l.global().globals;
    g.as_mut().set(key, LuaValue::table(g));

    let _ = LJ_TNIL;
}
