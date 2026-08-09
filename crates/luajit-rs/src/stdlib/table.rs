//! Table library: `table.concat`, `table.insert`, `table.move`,
//! `table.pack`, `table.remove`, `table.sort`, `table.unpack`.

use crate::api::{lua_gettop, lua_pop, lua_pushcfunction, lua_setfield};
use crate::err::LuaResult;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg_type, push};
use crate::lual_reg;

use super::sort::introsort;

pub fn tab_concat(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => {
            return Err(err_bad_arg_type(l, 1, "table.concat", "table", arg(l, 0)));
        }
    };
    let sep = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => Vec::new(),
    };
    let mut i = arg(l, 2).as_number().map_or(1.0, |n| n.max(1.0)) as usize;
    // j: keep the raw value — 0 (or j < i) means an empty range.
    let j = arg(l, 3).as_number().map(|n| n.max(0.0) as usize);

    let tab = t.as_ref();
    let mut out = Vec::new();
    let mut first = true;
    loop {
        if let Some(jv) = j
            && i > jv
        {
            break;
        }
        let v = tab.get_int(i as i32);
        if v.is_nil() {
            if j.is_some() {
                // An explicit range with a hole: LuaJIT reports the index.
                return Err(l.runtime_error(
                    format!("invalid value (nil) at index {i} in table for 'concat'").as_bytes(),
                ));
            }
            break;
        }
        if !first {
            out.extend_from_slice(&sep);
        }
        first = false;
        if let Some(sid) = v.as_string_id() {
            out.extend_from_slice(l.str_static(sid));
        } else if let Some(n) = v.as_number() {
            out.extend_from_slice(crate::strfmt::g14(n).as_bytes());
        } else {
            return Err(l.runtime_error(
                format!("invalid value (nil) at index {i} in table for 'concat'").as_bytes(),
            ));
        }
        i += 1;
    }
    let sid = l.heap().intern(&out);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn tab_insert(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => {
            return Err(err_bad_arg_type(l, 1, "table.insert", "table", arg(l, 0)));
        }
    };
    let n = lua_gettop(l);
    if n == 2 {
        let pos = (t.as_ref().len() as i32) + 1;
        t.as_mut().set_int(pos, arg(l, 1));
    } else if n >= 3 {
        let pos = arg(l, 1).as_number().unwrap_or(1.0) as i32;
        let val = arg(l, 2);
        let len = t.as_ref().len() as i32;
        if pos < 1 || pos > len + 1 {
            return Err(l.runtime_error(b"position out of bounds"));
        }
        // Shift elements [pos..len] up by one (Lua 5.1 semantics).
        for k in (pos..=len).rev() {
            let v = t.as_ref().get_int(k);
            t.as_mut().set_int(k + 1, v);
        }
        t.as_mut().set_int(pos, val);
    }
    Ok(0)
}

fn tab_move(l: &mut LuaState) -> LuaResult<i32> {
    let a1 = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "table.move", "table", arg(l, 0))),
    };
    let f = arg(l, 1).as_number().unwrap_or(1.0) as i64;
    let e = match arg(l, 2).as_number() {
        Some(n) => n as i64,
        None => {
            return Err(err_bad_arg_type(
                l,
                3,
                "table.move",
                "number",
                arg(l, 3 - 1),
            ));
        }
    };
    let t_pos = arg(l, 3).as_number().unwrap_or(1.0) as i64;
    let a2 = arg(l, 4).as_table();

    let src = a1.as_ref();
    let len = e - f + 1;
    for k in 0..len {
        let sv = src.get_int((f + k) as i32);
        let dst_idx = t_pos + k;
        if let Some(dst_tab) = a2 {
            dst_tab.as_mut().set_int(dst_idx as i32, sv);
        } else {
            a1.as_mut().set_int(dst_idx as i32, sv);
        }
    }
    push(
        l,
        match a2 {
            Some(dst) => LuaValue::table(dst),
            None => LuaValue::table(a1),
        },
    );
    Ok(1)
}

pub fn tab_pack(l: &mut LuaState) -> LuaResult<i32> {
    let n = lua_gettop(l);
    let t = l
        .heap()
        .alloc_table(crate::table::LuaTable::new(n as u32 + 1, 1));
    for i in 0..n {
        t.as_mut().set_int(i as i32 + 1, arg(l, i));
    }
    let sid = l.heap().intern(b"n");
    let key = l.heap().str_value(sid);
    t.as_mut().set(key, LuaValue::number(n as f64));
    push(l, LuaValue::table(t));
    Ok(1)
}

pub fn tab_remove(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => {
            return Err(err_bad_arg_type(l, 1, "table.remove", "table", arg(l, 0)));
        }
    };
    let len = t.as_ref().len() as i32;
    let pos = match arg(l, 1).as_number() {
        Some(n) if n >= 1.0 => n as i32,
        None => len,
        _ => {
            return Err(err_bad_arg_type(
                l,
                2,
                "table.remove",
                "number",
                arg(l, 2 - 1),
            ));
        }
    }
    .max(1)
    .min(len);

    if pos == 0 || pos > len || len == 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let v = t.as_ref().get_int(pos);
    for i in pos..len {
        t.as_mut().set_int(i, t.as_ref().get_int(i + 1));
    }
    t.as_mut().set_int(len, LuaValue::NIL);
    push(l, v);
    Ok(1)
}

fn tab_sort(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "table.sort", "table", arg(l, 0))),
    };
    let len = t.as_ref().len() as i32;
    let mut items: Vec<(i32, LuaValue)> = (1..=len).map(|i| (i, t.as_ref().get_int(i))).collect();
    let comp = arg(l, 1);
    if comp.is_func() {
        introsort(
            l,
            &mut items,
            crate::stdlib::sort::Comparator::Lua(comp.as_func().unwrap()),
        )?;
    } else {
        // No comparator: use the default `<` (numbers, strings, __lt).
        introsort(l, &mut items, crate::stdlib::sort::Comparator::Default)?;
    }
    for (idx, (_, v)) in items.iter().enumerate() {
        t.as_mut().set_int(idx as i32 + 1, *v);
    }
    Ok(0)
}

fn tab_unpack(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => {
            return Err(err_bad_arg_type(l, 1, "table.unpack", "table", arg(l, 0)));
        }
    };
    let i = arg(l, 1).as_number().unwrap_or(1.0) as i32;
    let j = arg(l, 2).as_number().unwrap_or(t.as_ref().len() as f64) as i32;
    let mut cnt = 0;
    l.stack_ensure(l.base + (j - i + 1) as usize);
    for k in i..=j {
        let v = t.as_ref().get_int(k);
        l.stack[l.base + cnt] = v;
        cnt += 1;
    }
    Ok(cnt as i32)
}

fn tab_new(l: &mut LuaState) -> LuaResult<i32> {
    let narr = arg(l, 0).as_number().unwrap_or(0.0) as u32;
    let nrec = arg(l, 1).as_number().unwrap_or(0.0) as u32;
    let hbits = if nrec == 0 {
        0
    } else {
        nrec.next_power_of_two().trailing_zeros()
    };
    let t = l.heap().alloc_table(LuaTable::new(narr, hbits));
    push(l, LuaValue::table(t));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"table", LibTarget::Global)
        .func(b"concat", tab_concat)
        .func(b"foreach", tab_foreach)
        .func(b"foreachi", tab_foreachi)
        .func(b"getn", tab_getn)
        .func(b"insert", tab_insert)
        .func(b"maxn", tab_maxn)
        .func(b"move", tab_move)
        .func(b"new", tab_new)
        .func(b"remove", tab_remove)
        .func(b"sort", tab_sort)
        .build();
    if l.compat52 {
        // `table.pack`/`table.unpack` are Lua 5.2 functions
        // (LuaJIT: `#if LJ_52`).
        lua_pushcfunction(l, tab_pack);
        lua_setfield(l, -2, "pack");
        lua_pushcfunction(l, tab_unpack);
        lua_setfield(l, -2, "unpack");
        lua_pop(l, 1);
    }
}

fn tab_foreach(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let f = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "foreach", "table", t)),
    };
    let obase = l.base;
    let otop = l.top;
    let mut k = LuaValue::NIL;
    while let Some((nk, v)) = tab.as_ref().next(k) {
        k = nk;
        let fs = l.top + 2;
        l.stack_ensure(fs + 6);
        l.stack[fs] = f;
        l.stack[fs + 2] = nk;
        l.stack[fs + 3] = v;
        match crate::vm::execute(l, fs, 2, 1) {
            Ok(_) => {
                let r = l.stack[fs];
                l.top = otop;
                l.base = obase;
                if !r.is_nil() {
                    push(l, r);
                    return Ok(1);
                }
            }
            _ => {
                l.top = otop;
                l.base = obase;
            }
        }
    }
    Ok(0)
}

fn tab_foreachi(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let f = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "foreachi", "table", t)),
    };
    let obase = l.base;
    let otop = l.top;
    for i in 1..=tab.as_ref().len() {
        let v = tab.as_ref().get_int(i as i32);
        if v.is_nil() {
            break;
        }
        let fs = l.top + 2;
        l.stack_ensure(fs + 6);
        l.stack[fs] = f;
        l.stack[fs + 2] = LuaValue::number(i as f64);
        l.stack[fs + 3] = v;
        match crate::vm::execute(l, fs, 2, 1) {
            Ok(_) => {
                let r = l.stack[fs];
                l.top = otop;
                l.base = obase;
                if !r.is_nil() {
                    push(l, r);
                    return Ok(1);
                }
            }
            _ => {
                l.top = otop;
                l.base = obase;
            }
        }
    }
    Ok(0)
}

fn tab_getn(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    match t.as_table() {
        Some(t) => {
            push(l, LuaValue::number(t.as_ref().len() as f64));
            Ok(1)
        }
        None => Err(err_bad_arg_type(l, 1, "getn", "table", t)),
    }
}

fn tab_maxn(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    match t.as_table() {
        Some(t) => {
            let mut max: f64 = 0.0;
            let mut k = LuaValue::NIL;
            while let Some((nk, _)) = t.as_ref().next(k) {
                k = nk;
                if let Some(n) = k.as_number() {
                    max = max.max(n);
                }
            }
            push(l, LuaValue::number(max));
            Ok(1)
        }
        None => Err(err_bad_arg_type(l, 1, "maxn", "table", t)),
    }
}
