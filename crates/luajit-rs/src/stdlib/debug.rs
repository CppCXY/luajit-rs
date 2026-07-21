use crate::err::LuaResult;
use crate::func::GcFunc;
use crate::state::LuaState;
use crate::stdlib::{arg, nargs, push};
use crate::value::LuaValue;

fn set_basemt_for(
    l: &mut LuaState,
    o: &LuaValue,
    mt: Option<crate::gc::GcPtr<crate::table::LuaTable>>,
) {
    let g = l.global();
    g.set_basemt(o.itype(), mt);
    if o.itype() == crate::value::LJ_TFALSE {
        g.set_basemt(crate::value::LJ_TTRUE, mt);
    } else if o.itype() == crate::value::LJ_TTRUE {
        g.set_basemt(crate::value::LJ_TFALSE, mt);
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
        return Err(crate::stdlib::err_bad_arg(l, 2, "debug.setmetatable", "nil or table", ""));
    }
    push(l, o);
    Ok(1)
}

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

    // Walk frames from the current base upward.
    let mut slot = l.base;
    let stack = &l.stack;

    loop {
        if slot < 2 {
            break;
        }

        let func = stack[slot - 2];
        let link_bits = stack[slot - 1].to_bits();

        if let Some(fv) = func.as_func() {
            match fv.as_ref() {
                GcFunc::Lua(cl) => {
                    let pt = cl.proto.as_ref();
                    let src = pt.source.map(|sid| {
                        String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
                    }).unwrap_or_else(|| "?".to_string());

                    // For the current frame, get line from debug_pc.
                    // For previous frames, get line from frame link (return PC).
                    let pc = if slot == l.base {
                        l.debug_pc
                    } else if link_bits & 0x3 == 0 {
                        let ret_pc = link_bits as usize;
                        if ret_pc >= 4 && ret_pc <= pt.bc.len() * 4 {
                            (ret_pc / 4).saturating_sub(1)
                        } else {
                            0
                        }
                    } else {
                        0
                    };

                    let line = if pc > 0 && pc < pt.lines.len() {
                        pt.lines[pc] as usize + pt.firstline as usize
                    } else {
                        pt.firstline as usize
                    };

                    trace.push_str(&format!(
                        "\t{}:{}: in main chunk\n",
                        src, line,
                    ));
                }
                GcFunc::C(_) => {
                    trace.push_str("\t[C]: in function\n");
                }
            }
        } else {
            trace.push_str("\t?: in ?\n");
        }

        // Follow frame link upward
        let frame_type = link_bits as u8 & 0x3;
        match frame_type {
            0 => {
                // FRAME_LUA: link is return PC. The previous frame's base
                // is at the call site. We can't directly get it from the link.
                break;
            }
            1 => {
                // FRAME_C: host entry - stop
                break;
            }
            3 => {
                // FRAME_VARG: skip vararg wrappers
                let delta = (link_bits >> 3) as usize;
                if delta == 0 || delta > slot { break; }
                slot -= delta;
            }
            _ => break,
        }
    }

    let sid = l.heap().intern(trace.as_bytes());
    let v = l.heap().str_value(sid);
    push(l, v);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    crate::stdlib::reg::LibBuilder::new(l, b"debug", crate::stdlib::reg::LibTarget::Global)
        .func(b"setmetatable", lib_setmetatable)
        .func(b"traceback", lib_traceback)
        .build();
}
