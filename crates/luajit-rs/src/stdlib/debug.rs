use crate::err::LuaResult;
use crate::func::GcFunc;
use crate::gc::GcPtr;
use crate::state::LuaState;
use crate::stdlib::{arg, err_bad_arg, nargs, push};
use crate::table::LuaTable;
use crate::value::{LJ_TFALSE, LJ_TTRUE, LuaValue};
use crate::vm::FRAME_TYPE_MASK;

fn set_basemt_for(l: &mut LuaState, o: &LuaValue, mt: Option<GcPtr<LuaTable>>) {
    let g = l.global();
    g.set_basemt(o.itype(), mt);
    if o.itype() == LJ_TFALSE {
        g.set_basemt(LJ_TTRUE, mt);
    } else if o.itype() == LJ_TTRUE {
        g.set_basemt(LJ_TFALSE, mt);
    }
}

fn lib_setmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let mt = arg(l, 1);
    if mt.is_nil() {
        if let Some(t) = o.as_table() {
            t.as_mut().metatable = None;
        } else {
            set_basemt_for(l, &o, None);
        }
    } else if let Some(mt_tab) = mt.as_table() {
        if let Some(t) = o.as_table() {
            t.as_mut().metatable = Some(mt_tab);
        } else {
            set_basemt_for(l, &o, Some(mt_tab));
        }
    } else {
        return Err(err_bad_arg(l, 2, "debug.setmetatable", "nil or table", ""));
    }
    push(l, o);
    Ok(1)
}

// ── Frame walking helpers ───────────────────────────────────────────────────

/// Walk `level` frames up from the current C frame, resolving vararg wrappers.
/// Returns `(slot, func)` or `None` if the level is out of range.
fn walk_frames(l: &LuaState, mut level: i32) -> Option<(usize, GcPtr<crate::func::GcFunc>)> {
    let mut slot = l.base;
    loop {
        if slot < 2 {
            return None;
        }
        let func = l.stack[slot - 2];
        let mut link = l.stack[slot - 1].to_bits();
        while (link & FRAME_TYPE_MASK) == 3
        /* FRAME_VARG */
        {
            slot = slot.saturating_sub((link >> 3) as usize);
            if slot < 2 {
                return None;
            }
            link = l.stack[slot - 1].to_bits();
        }
        let ft = link & FRAME_TYPE_MASK;
        if let Some(fv) = func.as_func() {
            if matches!(fv.as_ref(), crate::func::GcFunc::Lua(_)) {
                if level == 0 {
                    return Some((slot, fv));
                }
                level -= 1;
                if level == 0 {
                    return Some((slot, fv));
                }
            }
            // Walk to caller
            if ft == 0 /* FRAME_LUA */ && link != 0 {
                slot = (link >> 3) as usize;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    None
}

fn str_val(l: &mut LuaState, s: &str) -> LuaValue {
    let sid = l.heap().intern(s.as_bytes());
    l.heap().str_value(sid)
}

// ── debug.getinfo ───────────────────────────────────────────────────────────

// Flags for what-to-return:
const WHAT_S: u8 = 1; // source, short_src, linedefined, lastlinedefined, what
const WHAT_L: u8 = 2; // currentline
const WHAT_N: u8 = 4; // name, namewhat
const WHAT_U: u8 = 8; // nup, nparams, isvararg
const WHAT_F: u8 = 16; // func

fn parse_what(what: &str) -> u8 {
    let mut flags = 0u8;
    for c in what.chars() {
        match c {
            'S' => flags |= WHAT_S,
            'l' => flags |= WHAT_L,
            'n' => flags |= WHAT_N,
            'u' => flags |= WHAT_U,
            'f' => flags |= WHAT_F,
            _ => {}
        }
    }
    if flags == 0 {
        flags = WHAT_S | WHAT_N | WHAT_L | WHAT_U;
    }
    flags
}

fn lib_getinfo(l: &mut LuaState) -> LuaResult<i32> {
    let first = arg(l, 0);
    let what_str = if nargs(l) > 1 {
        match arg(l, 1).as_string_id() {
            Some(sid) => String::from_utf8_lossy(l.str_static(sid)).into_owned(),
            None => String::from(""),
        }
    } else {
        String::from("")
    };
    let flags = parse_what(&what_str);

    let (slot, gf) = if let Some(fv) = first.as_func() {
        // Given a function directly
        let mut slot = l.base;
        let link = l.stack[slot - 1].to_bits();
        let ft = link & FRAME_TYPE_MASK;
        if ft == 0 /* FRAME_LUA */ && link != 0 {
            slot = (link >> 3) as usize;
        }
        (slot, fv)
    } else if let Some(n) = first.as_number() {
        let level = n as i32;
        match walk_frames(l, level) {
            Some((s, f)) => (s, f),
            None => {
                push(l, LuaValue::NIL);
                return Ok(1);
            }
        }
    } else {
        return Err(err_bad_arg(l, 1, "getinfo", "function or level", ""));
    };

    let t = l.heap().alloc_table(LuaTable::new(0, 3));
    match gf.as_ref() {
        GcFunc::Lua(cl) => {
            let pt = cl.proto.as_ref();
            if flags & WHAT_S != 0 {
                let src = pt
                    .source
                    .and_then(|sid| {
                        l.heap().strings.try_lookup(sid).map(|_ptr| {
                            String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
                        })
                    })
                    .unwrap_or_else(|| "=?".to_string());

                let short_src = if src.starts_with('@') || src.starts_with('=') {
                    src[1..]
                        .rsplit(&['\\', '/'][..])
                        .next()
                        .unwrap_or(&src[1..])
                        .to_string()
                } else {
                    src.rsplit(&['\\', '/'][..])
                        .next()
                        .unwrap_or(&src)
                        .to_string()
                };

                t.as_mut().set_str(str_val(l, "source"), str_val(l, &src));
                t.as_mut()
                    .set_str(str_val(l, "short_src"), str_val(l, &short_src));
                t.as_mut().set_str(
                    str_val(l, "linedefined"),
                    LuaValue::number(pt.firstline as f64),
                );
                t.as_mut().set_str(
                    str_val(l, "lastlinedefined"),
                    LuaValue::number((pt.firstline + pt.numline - 1) as f64),
                );
                t.as_mut().set_str(
                    str_val(l, "what"),
                    if pt.source.is_some_and(|sid| {
                        let b = l.heap().strings.get(sid);
                        b.starts_with(b"@") || b.starts_with(b"=")
                    }) && pt.firstline == 0
                    {
                        str_val(l, "main")
                    } else if pt.firstline == 0 {
                        str_val(l, "C")
                    } else {
                        str_val(l, "Lua")
                    },
                );
            }
            if flags & WHAT_L != 0 {
                let cur_pc = l
                    .debug_pc
                    .saturating_sub(1)
                    .min(pt.lines.len().saturating_sub(1));
                let cur_line = if cur_pc < pt.lines.len() {
                    pt.lines[cur_pc] as f64
                } else {
                    pt.firstline as f64
                };
                t.as_mut()
                    .set_str(str_val(l, "currentline"), LuaValue::number(cur_line));
            }
            if flags & WHAT_U != 0 {
                t.as_mut()
                    .set_str(str_val(l, "nups"), LuaValue::number(cl.upvals.len() as f64));
                t.as_mut()
                    .set_str(str_val(l, "nparams"), LuaValue::number(pt.numparams as f64));
                t.as_mut().set_str(
                    str_val(l, "isvararg"),
                    LuaValue::boolean(pt.flags & crate::proto::PROTO_VARARG != 0),
                );
            }
            if flags & WHAT_F != 0 {
                t.as_mut().set_str(str_val(l, "func"), l.stack[slot - 2]);
            }
            if flags & WHAT_N != 0 {
                // Try to find name from caller
                t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
                t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
            }
        }
        crate::func::GcFunc::C(_) => {
            if flags & WHAT_S != 0 {
                t.as_mut().set_str(str_val(l, "source"), str_val(l, "=[C]"));
                t.as_mut()
                    .set_str(str_val(l, "short_src"), str_val(l, "[C]"));
                t.as_mut()
                    .set_str(str_val(l, "linedefined"), LuaValue::number(-1.0));
                t.as_mut()
                    .set_str(str_val(l, "lastlinedefined"), LuaValue::number(-1.0));
                t.as_mut().set_str(str_val(l, "what"), str_val(l, "C"));
            }
            if flags & WHAT_L != 0 {
                t.as_mut()
                    .set_str(str_val(l, "currentline"), LuaValue::number(-1.0));
            }
            if flags & WHAT_U != 0 {
                t.as_mut()
                    .set_str(str_val(l, "nups"), LuaValue::number(0.0));
                t.as_mut()
                    .set_str(str_val(l, "nparams"), LuaValue::number(0.0));
                t.as_mut().set_str(str_val(l, "isvararg"), LuaValue::FALSE);
            }
            if flags & WHAT_F != 0 {
                t.as_mut().set_str(str_val(l, "func"), l.stack[slot - 2]);
            }
            if flags & WHAT_N != 0 {
                t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
                t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
            }
        }
    }

    push(l, LuaValue::table(t));
    Ok(1)
}

// ── debug.getmetatable ──────────────────────────────────────────────────────

fn lib_getmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    if let Some(t) = o.as_table()
        && let Some(mt) = t.as_ref().metatable {
            push(l, LuaValue::table(mt));
            return Ok(1);
        }
    // Check base metatable for non-table types (string, number, etc.)
    let it = o.itype();
    if let Some(mt) = l.global().basemt_of(it) {
        push(l, LuaValue::table(mt));
        return Ok(1);
    }
    push(l, LuaValue::NIL);
    Ok(1)
}

// ── debug.getregistry ───────────────────────────────────────────────────────

fn lib_getregistry(l: &mut LuaState) -> LuaResult<i32> {
    push(l, LuaValue::table(l.global().registry));
    Ok(1)
}

// ── debug.getfenv / setfenv ─────────────────────────────────────────────────

fn lib_getfenv(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let env = match o.as_func() {
        Some(f) => match f.as_ref() {
            GcFunc::Lua(c) => c.env,
            GcFunc::C(c) => c.env,
        },
        _ => l.global().globals,
    };
    push(l, LuaValue::table(env));
    Ok(1)
}

fn lib_setfenv(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let tab = match arg(l, 1).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 2, "setfenv", "table", "")),
    };
    if let Some(f) = o.as_func() {
        match f.as_mut() {
            GcFunc::Lua(c) => c.env = tab,
            GcFunc::C(c) => c.env = tab,
        }
    }
    push(l, LuaValue::number(0.0));
    Ok(1)
}

// ── debug.traceback ─────────────────────────────────────────────────────────

fn lib_traceback(l: &mut LuaState) -> LuaResult<i32> {
    let msg = if nargs(l) > 0 {
        if let Some(sid) = arg(l, 0).as_string_id() {
            String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let mut trace = if msg.is_empty() {
        "stack traceback:\n".to_string()
    } else {
        format!("{}\nstack traceback:\n", msg)
    };

    let mut slot = l.base;
    let mut first = true;
    for _ in 0..64 {
        if slot < 2 {
            break;
        }
        let func = l.stack[slot - 2];
        let mut cur_link = l.stack[slot - 1].to_bits();
        while (cur_link & FRAME_TYPE_MASK) == 3
        /* FRAME_VARG */
        {
            slot = slot.saturating_sub((cur_link >> 3) as usize);
            if slot < 2 {
                break;
            }
            cur_link = l.stack[slot - 1].to_bits();
        }
        let frame_type = cur_link & FRAME_TYPE_MASK;

        if let Some(fv) = func.as_func() {
            match fv.as_ref() {
                crate::func::GcFunc::Lua(cl) => {
                    let pt = cl.proto.as_ref();
                    let src = pt
                        .source
                        .and_then(|sid| {
                            l.heap().strings.try_lookup(sid).map(|_| {
                                let bytes = l.heap().strings.get(sid);
                                if bytes.starts_with(b"@") || bytes.starts_with(b"=") {
                                    String::from_utf8_lossy(&bytes[1..]).into_owned()
                                } else {
                                    String::from_utf8_lossy(bytes).into_owned()
                                }
                            })
                        })
                        .unwrap_or_else(|| "(unknown)".to_string());

                    let pc = if first {
                        l.debug_pc.saturating_sub(1)
                    } else {
                        let ret_ip = cur_link as *const crate::bc::BCIns;
                        let call_ptr = unsafe { ret_ip.sub(1) };
                        let bcp = pt.bc.as_ptr();
                        if call_ptr >= bcp && call_ptr < unsafe { bcp.add(pt.bc.len()) } {
                            (call_ptr as usize - bcp as usize) / 4
                        } else {
                            0
                        }
                    };
                    let line = if pc < pt.lines.len() {
                        pt.lines[pc] as usize
                    } else {
                        pt.firstline as usize
                    };
                    trace.push_str(&format!(
                        "\t{}:{}: in {}\n",
                        src,
                        line,
                        if first { "main chunk" } else { "function" }
                    ));
                    first = false;
                }
                GcFunc::C(_) => {
                    trace.push_str("\t[C]: in function\n");
                    first = false;
                }
            }
        }

        // Walk to caller
        if frame_type == 0 && cur_link != 0 {
            slot = (cur_link >> 3) as usize;
        } else {
            break;
        }
    }

    let sid = l.heap().intern(trace.as_bytes());
    let v = l.heap().str_value(sid);
    push(l, v);
    Ok(1)
}

// ── debug.gethook / sethook (stubs) ─────────────────────────────────────────

fn lib_gethook(_l: &mut LuaState) -> LuaResult<i32> {
    Ok(0)
}

fn lib_sethook(_l: &mut LuaState) -> LuaResult<i32> {
    Ok(0)
}

fn lib_getupvalue(l: &mut LuaState) -> LuaResult<i32> {
    let f = arg(l, 0);
    let idx = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    match f.as_func() {
        Some(gf) => match gf.as_ref() {
            GcFunc::Lua(cl) => {
                if idx < 1 || idx > cl.upvals.len() {
                    push(l, LuaValue::NIL);
                    return Ok(1);
                }
                let uv_idx = idx - 1;
                let proto = cl.proto.as_ref();
                if uv_idx < proto.uvnames.len() && !proto.uvnames[uv_idx].is_empty() {
                    let sid = l.heap().intern(proto.uvnames[uv_idx].as_bytes());
                    push(l, l.heap().str_value(sid));
                } else {
                    push(l, l.heap().str_value(l.heap().intern(b"")));
                }
                let val = if uv_idx < cl.upvals.len() {
                    cl.upvals[uv_idx].as_ref().get()
                } else {
                    LuaValue::NIL
                };
                push(l, val);
                Ok(2)
            }
            GcFunc::C(_) => {
                push(l, LuaValue::NIL);
                Ok(1)
            }
        },
        None => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn lib_upvaluejoin(l: &mut LuaState) -> LuaResult<i32> {
    let _f1 = arg(l, 0);
    let _n1 = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let _f2 = arg(l, 2);
    let _n2 = arg(l, 3).as_number().unwrap_or(0.0) as usize;
    // NYI: stub — just succeed silently
    Ok(0)
}

// ── open ────────────────────────────────────────────────────────────────────

pub fn open(l: &mut LuaState) {
    crate::stdlib::reg::LibBuilder::new(l, b"debug", crate::stdlib::reg::LibTarget::Global)
        .func(b"setmetatable", lib_setmetatable)
        .func(b"getmetatable", lib_getmetatable)
        .func(b"getregistry", lib_getregistry)
        .func(b"getinfo", lib_getinfo)
        .func(b"traceback", lib_traceback)
        .func(b"getfenv", lib_getfenv)
        .func(b"setfenv", lib_setfenv)
        .func(b"gethook", lib_gethook)
        .func(b"sethook", lib_sethook)
        .func(b"getupvalue", lib_getupvalue)
        .func(b"upvaluejoin", lib_upvaluejoin)
        .build();
}
