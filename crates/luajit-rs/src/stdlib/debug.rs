use crate::err::LuaResult;
use crate::state::LuaState;
use crate::stdlib::{arg, nargs, push};
use crate::value::LuaValue;
use crate::vm::FRAME_TYPE_MASK;

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

    let mut slot = l.base;
    let mut first = true;
    for _ in 0..64 {
        if slot < 2 {
            break;
        }
        // Skip FRAME_VARG wrappers.
        let mut cur_slot = slot;
        let mut cur_link = l.stack[cur_slot - 1].to_bits();
        while (cur_link & FRAME_TYPE_MASK) as u64 == 3 /* FRAME_VARG */ {
            cur_slot = cur_slot.saturating_sub((cur_link >> 3) as usize);
            if cur_slot < 2 { break; }
            cur_link = l.stack[cur_slot - 1].to_bits();
        }
        let func = l.stack[cur_slot - 2];
        let frame_type = cur_link & FRAME_TYPE_MASK;

        if let Some(fv) = func.as_func() {
            match fv.as_ref() {
                crate::func::GcFunc::Lua(cl) => {
                    let pt = cl.proto.as_ref();
                    let src = pt.source.and_then(|sid| {
                        l.heap().strings.try_lookup(sid).map(|_| {
                            String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
                        })
                    }).unwrap_or_else(|| "(unknown)".to_string());

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
                        pt.lines[pc] as usize + pt.firstline as usize
                    } else {
                        pt.firstline as usize
                    };
                    trace.push_str(&format!("\t{}:{}: in {}\n", src, line,
                        if first { "main chunk" } else { "function" }));
                    first = false;

                    // Walk to caller via frame link.
                    if (frame_type as u64) == 0 && cur_link != 0 {
                        let ret_ip = cur_link as *const crate::bc::BCIns;
                        let call_ins = unsafe { *ret_ip.sub(1) };
                        let a = crate::bc::bc_a(call_ins) as usize;
                        slot = slot.saturating_sub(2 + a);
                        continue;
                    }
                    break;
                }
                crate::func::GcFunc::C(_) => {
                    trace.push_str("\t[C]: in function\n");
                    first = false;
                    if (frame_type as u64) == 0 && cur_link != 0 {
                        slot = (cur_link >> 3) as usize;
                        continue;
                    }
                    break;
                }
            }
        }
        break;
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
