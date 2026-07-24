//! Table library: `table.concat`, `table.insert`, `table.move`,
//! `table.pack`, `table.remove`, `table.sort`, `table.unpack`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push};
use crate::lual_reg;

use super::sort::introsort;

pub fn tab_concat(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "table.concat", "table", "")),
    };
    let sep = match arg(l, 1).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => Vec::new(),
    };
    let mut i = arg(l, 2).as_number().map_or(1.0, |n| n.max(1.0)) as usize;
    let j = arg(l, 3).as_number().map(|n| n.max(1.0) as usize);

    let tab = t.as_ref();
    let mut out = Vec::new();
    let mut first = true;
    loop {
        let v = tab.get_int(i as i32);
        if v.is_nil() || (j.is_some() && i > j.unwrap()) {
            break;
        }
        if !first {
            out.extend_from_slice(&sep);
        }
        first = false;
        if let Some(sid) = v.as_string_id() {
            out.extend_from_slice(
                tab.get_int(0)
                    .as_string_id()
                    .map_or(b"", |_| unreachable!()),
            );
            // actually just use str_static:
            out.extend_from_slice(l.str_static(sid));
        } else if let Some(n) = v.as_number() {
            out.extend_from_slice(crate::strfmt::g14(n).as_bytes());
        } else if !v.is_nil() {
            return Err(l.runtime_error(b"invalid value (%s) in table.concat"));
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
        None => return Err(err_bad_arg(l, 1, "table.insert", "table", "")),
    };
    let n = nargs(l);
    if n == 2 {
        let pos = (t.as_ref().len() as i32) + 1;
        t.as_mut().set_int(pos, arg(l, 1));
    } else if n >= 3 {
        let pos = arg(l, 1).as_number().unwrap_or(1.0) as i32;
        let val = arg(l, 2);
        t.as_mut().set_int(pos, val);
    }
    Ok(0)
}

fn tab_move(l: &mut LuaState) -> LuaResult<i32> {
    let a1 = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "table.move", "table", "")),
    };
    let f = arg(l, 1).as_number().unwrap_or(1.0) as i64;
    let e = match arg(l, 2).as_number() {
        Some(n) => n as i64,
        None => return Err(err_bad_arg(l, 3, "table.move", "number", "")),
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

fn tab_pack(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    let t = l
        .heap()
        .alloc_table(crate::table::LuaTable::new(n as u32, 0));
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
        None => return Err(err_bad_arg(l, 1, "table.remove", "table", "")),
    };
    let len = t.as_ref().len() as i32;
    let pos = match arg(l, 1).as_number() {
        Some(n) if n >= 1.0 => n as i32,
        None => len,
        _ => return Err(err_bad_arg(l, 2, "table.remove", "number", "")),
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
        None => return Err(err_bad_arg(l, 1, "table.sort", "table", "")),
    };
    let len = t.as_ref().len() as i32;
    let mut items: Vec<(i32, LuaValue)> = (1..=len).map(|i| (i, t.as_ref().get_int(i))).collect();
    let comp = arg(l, 1);
    if comp.is_func() {
        introsort(l, &mut items, comp.as_func().unwrap())?;
    } else {
        items.sort_unstable_by(|a, b| {
            a.1.num()
                .partial_cmp(&b.1.num())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    for (idx, (_, v)) in items.iter().enumerate() {
        t.as_mut().set_int(idx as i32 + 1, *v);
    }
    Ok(0)
}

fn tab_unpack(l: &mut LuaState) -> LuaResult<i32> {
    let t = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "table.unpack", "table", "")),
    };
    let i = arg(l, 1).as_number().unwrap_or(1.0) as i32;
    let j = arg(l, 2).as_number().unwrap_or(t.as_ref().len() as f64) as i32;
    let mut cnt = 0;
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
        .func(b"insert", tab_insert)
        .func(b"move", tab_move)
        .func(b"new", tab_new)
        .func(b"pack", tab_pack)
        .func(b"remove", tab_remove)
        .func(b"sort", tab_sort)
        .func(b"unpack", tab_unpack)
        .build();
}
