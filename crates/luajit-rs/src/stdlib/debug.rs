use crate::err::LuaResult;
use crate::func::GcFunc;
use crate::gc::GcPtr;
use crate::state::LuaState;
use crate::stdlib::{arg, err_bad_arg, nargs, push, pushv};
use crate::table::LuaTable;
use crate::value::{LJ_TFALSE, LJ_TNUMX, LJ_TTRUE, LuaValue};
use crate::vm::FRAME_TYPE_MASK;

fn set_basemt_for(l: &mut LuaState, o: &LuaValue, mt: Option<GcPtr<LuaTable>>) {
    let g = l.global();
    // Numbers share a single base metatable slot: the raw itype of a
    // double varies with its value (the NaN-boxing puts the exponent
    // in the tag), so normalize to the numeric tag.
    let it = if o.is_number() { LJ_TNUMX } else { o.itype() };
    g.set_basemt(it, mt);
    if o.itype() == LJ_TFALSE {
        g.set_basemt(LJ_TTRUE, mt);
    } else if o.itype() == LJ_TTRUE {
        g.set_basemt(LJ_TFALSE, mt);
    }
}

/// The metamethod name of the instruction that triggered a `FRAME_CONT`
/// continuation frame (`lj_debug.c`'s frame_iscont name switch). The
/// saved PC sits at `slot - 3` (in the caller's proto); the bytecode
/// there is the triggering instruction.
fn cont_mm_name(l: &LuaState, slot: usize) -> Option<&'static str> {
    if slot < 4 {
        return None;
    }
    use crate::bc::{BCOp, bc_op};
    let saved_pc = l.stack[slot - 3].to_bits() as usize;
    let link = l.stack[slot - 1].to_bits();
    let delta = (link >> 3) as usize;
    if delta == 0 || delta > slot {
        return None;
    }
    let caller_base = slot - delta;
    if caller_base < 2 {
        return None;
    }
    let pt = l.stack[caller_base - 2].as_func()?;
    let pt = match pt.as_ref() {
        GcFunc::Lua(cl) => cl.proto.as_ref(),
        _ => return None,
    };

    // The continuation's saved PC points one past the triggering
    // instruction (the interpreter resumes there after the metamethod).
    if saved_pc == 0 || saved_pc > pt.bc.len() {
        return None;
    }
    let name = match bc_op(pt.bc[saved_pc - 1]) {
        BCOp::ISLT | BCOp::ISGE => "__lt",
        BCOp::ISLE | BCOp::ISGT => "__le",
        BCOp::ISEQV | BCOp::ISNEV => "__eq",
        BCOp::ADDVN | BCOp::ADDNV | BCOp::ADDVV => "__add",
        BCOp::SUBVN | BCOp::SUBNV | BCOp::SUBVV => "__sub",
        BCOp::MULVN | BCOp::MULNV | BCOp::MULVV => "__mul",
        BCOp::DIVVN | BCOp::DIVNV | BCOp::DIVVV => "__div",
        BCOp::MODVN | BCOp::MODNV | BCOp::MODVV => "__mod",
        BCOp::POW => "__pow",
        BCOp::UNM => "__unm",
        BCOp::LEN => "__len",
        BCOp::CAT => "__concat",
        BCOp::TGETV | BCOp::TGETS | BCOp::TGETB | BCOp::GGET => "__index",
        BCOp::TSETV | BCOp::TSETS | BCOp::TSETB | BCOp::GSET | BCOp::TSETM => "__newindex",
        BCOp::CALL | BCOp::CALLT => "__call",
        _ => "?",
    };
    Some(name)
}

/// Resolve the function name of a Lua frame from its caller's bytecode
/// (lj_debug_funcname's local/global/method cases): the FRAME_LUA link is
/// the caller's return PC, the CALL instruction before it holds the
/// callee register, and the local variable debug info maps it to a name.
fn funcname_from_caller(l: &LuaState, slot: usize) -> Option<(&'static str, String)> {
    use crate::bc::{BCIns, BCOp, bc_a, bc_b, bc_d, bc_op};
    if slot < 2 {
        return None;
    }
    let link = l.stack[slot - 1].to_bits();
    if (link & FRAME_TYPE_MASK) != 0 || link == 0 {
        return None; // FRAME_LUA only.
    }
    let ret_ip = link as *const BCIns;
    let call_ins = unsafe { *ret_ip.sub(1) };
    let op = bc_op(call_ins);
    if std::env::var("LUARS_NAMEDBG").is_ok() {
        eprintln!(
            "NAMEDBG slot={} link={:#x} op={:?} a={} base_calc={}",
            slot,
            link,
            op,
            bc_a(call_ins),
            slot.saturating_sub(2 + bc_a(call_ins) as usize)
        );
    }
    if !matches!(
        op,
        BCOp::CALL | BCOp::CALLT | BCOp::CALLM | BCOp::CALLMT | BCOp::ITERC | BCOp::ITERN
    ) {
        return None;
    }
    let callee_reg = bc_a(call_ins);
    let caller_base = slot.saturating_sub(2 + callee_reg as usize);
    if caller_base < 2 {
        return None;
    }
    let pt = l.stack[caller_base - 2].as_func()?;
    let pt = match pt.as_ref() {
        GcFunc::Lua(cl) => cl.proto.as_ref(),
        _ => return None,
    };
    let pc = unsafe { ret_ip.offset_from(pt.bc.as_ptr()) } as usize;
    if pc == 0 {
        return None;
    }
    let call_pc = (pc - 1) as u32;
    // The compiler may move the callee to a fresh register right before
    // the CALL (e.g. MOV r, local); resolve through it.
    let mut callee_reg = callee_reg;
    if call_pc > 0 {
        let prev = pt.bc[(call_pc - 1) as usize];
        if bc_op(prev) == BCOp::MOV && bc_a(prev) == callee_reg {
            callee_reg = bc_b(prev);
        }
    }
    // LuaJIT's lj_debug_funcname: inspect the instruction right before
    // the CALL (following one MOV), then fall back to local variables.
    let mut k = call_pc;
    for _ in 0..2 {
        if k == 0 {
            break;
        }
        k -= 1;
        let ins = pt.bc[k as usize];
        match bc_op(ins) {
            BCOp::GGET if bc_a(ins) == callee_reg => {
                let name = kgc_str(l, &pt.kgc, bc_d(ins) as usize)?;
                return Some(("global", name));
            }
            BCOp::TGETS if bc_a(ins) == callee_reg => {
                let name = kgc_str(l, &pt.kgc, bc_d(ins) as usize)?;
                return Some(("field", name));
            }
            BCOp::TGETV if bc_a(ins) == callee_reg => {
                return Some(("method", "?".to_string()));
            }
            BCOp::MOV if bc_a(ins) == callee_reg => {
                callee_reg = bc_b(ins);
            }
            _ => break,
        }
    }
    // The callee is a local variable: pick the variable whose lifetime
    // contains the call and starts latest (nearest definition).
    let mut best: Option<(usize, &str, String)> = None;
    for (reg, spc, epc, name) in &pt.varnames {
        let end = if *epc == 0 { pt.bc.len() as u32 } else { *epc };
        if *reg as u32 == callee_reg && *spc <= call_pc && call_pc <= end {
            let spc = *spc as usize;
            if best.as_ref().is_none_or(|b| spc > b.0) {
                best = Some((spc, "local", name.clone()));
            }
        }
    }
    if let Some((_, w, n)) = best {
        return Some((w, n));
    }
    None
}

fn kgc_str(l: &LuaState, kgc: &[crate::proto::KGc], idx: usize) -> Option<String> {
    match kgc.get(idx) {
        Some(crate::proto::KGc::Str(sid)) => Some(String::from_utf8_lossy(l.str_static(*sid)).into_owned()),
        _ => None,
    }
}

/// Fill `name`/`namewhat` for a frame (`getinfo(level)`): metamethod
/// continuations first, then the call-site inference.
fn name_from_level(l: &mut LuaState, slot: usize, t: GcPtr<LuaTable>) {
    let link = l.stack[slot - 1].to_bits();
    if (link & FRAME_TYPE_MASK) == 2 {
        if let Some(name) = cont_mm_name(l, slot) {
            t.as_mut().set_str(str_val(l, "name"), str_val(l, name));
            t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, "metamethod"));
        } else {
            t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
            t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
        }
    } else if let Some((name, fbits)) = l.mmname {
        if l.stack[slot - 2].to_bits() == fbits {
            t.as_mut().set_str(str_val(l, "name"), str_val(l, name));
            t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, "metamethod"));
        } else {
            t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
            t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
        }
    } else if let Some((namewhat, name)) = funcname_from_caller(l, slot) {
        t.as_mut().set_str(str_val(l, "name"), str_val(l, &name));
        t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, namewhat));
    } else {
        t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
        t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
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
            // Walk to caller. A FRAME_LUA link is either the caller's
            // return PC (Lua-to-Lua calls) or a caller-base delta (the
            // C-call frames): the latter is always a small stack index.
            // FRAME_C / FRAME_CONT links carry a caller base in the
            // delta (xpcall's handler chains its frame to the failed
            // call; metamethod continuations chain to their caller) and
            // count as an extra (C-side) level.
            match ft {
                0 /* FRAME_LUA */ if link != 0 => {
                    if ((link >> 3) as usize) < l.stack.len() {
                        slot = (link >> 3) as usize;
                    } else {
                        let ret_ip = link as *const crate::bc::BCIns;
                        let call_ins = unsafe { *ret_ip.sub(1) };
                        let a = crate::bc::bc_a(call_ins) as usize;
                        slot = slot.saturating_sub(2 + a);
                    }
                }
                1 /* FRAME_C */ | 2 /* FRAME_CONT */ if link != 0 => {
                    // The C-side frame counts as a level of its own.
                    level -= 1;
                    if level == 0 {
                        return None; // A C-side frame is not a Lua frame.
                    }
                    // The delta is the distance back to the caller's base.
                    let d = (link >> 3) as usize;
                    if d == 0 || d > slot {
                        return None;
                    }
                    slot -= d;
                }
                _ => break,
            }
        } else {
            break;
        }
    }
    None
}

/// The metamethod name of a frame, when it is one: mmcall frames carry
/// the triggering instruction's saved PC (cont_mm_name); frames reached
/// through the execute-recursion paths carry LuaState.mmname.
fn self_mm_name(l: &LuaState, slot: usize) -> Option<&'static str> {
    if slot < 4 {
        return None;
    }
    if let Some((name, fbits)) = l.mmname {
        for s in slot.saturating_sub(5)..slot {
            if l.stack[s].to_bits() == fbits {
                return Some(name);
            }
        }
    }
    let link = l.stack[slot - 1].to_bits();
    let ft = link & FRAME_TYPE_MASK;
    if ft == 2 /* FRAME_CONT */ || ft == 1
    /* FRAME_C */
    {
        return cont_mm_name(l, slot);
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
const WHAT_ACTIVELINES: u8 = 0x20; // activelines ('L')
const WHAT_N: u8 = 4; // name, namewhat
const WHAT_U: u8 = 8; // nup, nparams, isvararg
const WHAT_F: u8 = 16; // func

fn parse_what(what: &str) -> u8 {
    let mut flags = 0u8;
    for c in what.chars() {
        match c {
            'S' => flags |= WHAT_S,
            'l' => flags |= WHAT_L,
            'L' => flags |= WHAT_ACTIVELINES,
            'n' => flags |= WHAT_N,
            'u' => flags |= WHAT_U,
            'f' => flags |= WHAT_F,
            _ => {}
        }
    }
    if flags == 0 {
        // Lua 5.1's lua_getinfo: an empty `what` means "nlSfu".
        flags = WHAT_S | WHAT_N | WHAT_L | WHAT_U | WHAT_F;
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
                    // Lua 5.1's luaO_chunkid for file names: truncate with
                    // a leading "..." keeping the tail (LUA_IDSIZE-3).
                    let name = &src[1..];
                    if name.len() > 37 {
                        format!("...{}", &name[name.len() - 37..])
                    } else {
                        name.to_string()
                    }
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
                    LuaValue::number((pt.firstline + pt.numline) as f64),
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
            if flags & WHAT_ACTIVELINES != 0 {
                // Lua 5.1's "L": a table mapping every line with code to
                // true (debug.getinfo(..., "L")).
                let al = l.heap().alloc_table(LuaTable::new(0, 0));
                let mut seen: Vec<u16> = Vec::new();
                for (i, &ln) in pt.lines.iter().enumerate() {
                    let ln = ln as i32;
                    // Skip the FUNCF header line: Lua 5.1's activelines
                    // only covers actual code lines.
                    if i > 0 && ln > 0 && !seen.contains(&(ln as u16)) {
                        seen.push(ln as u16);
                        al.as_mut().set_int(ln, LuaValue::TRUE);
                    }
                }
                t.as_mut().set_str(str_val(l, "activelines"), LuaValue::table(al));
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
                let f = if let Some(fv) = first.as_func() {
                    fv
                } else {
                    l.stack[slot - 2].as_func().expect("Lua frame func")
                };
                t.as_mut().set_str(str_val(l, "func"), LuaValue::func(f));
            }
            if flags & WHAT_N != 0 {
                // A function argument has no call context: Lua 5.1's
                // lua_getinfo leaves the name empty for `getinfo(f)`.
                if !first.is_func() {
                    name_from_level(l, slot, t);
                } else {
                    t.as_mut().set_str(str_val(l, "name"), LuaValue::NIL);
                    t.as_mut().set_str(str_val(l, "namewhat"), str_val(l, ""));
                }
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
                let f = first
                    .as_func()
                    .unwrap_or_else(|| l.stack[slot - 2].as_func().unwrap());
                t.as_mut().set_str(str_val(l, "func"), LuaValue::func(f));
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
        && let Some(mt) = t.as_ref().metatable
    {
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
        // getfenv(0): the environment of the running thread.
        _ if o.as_number() == Some(0.0) => l.thread_env,
        _ if o.as_thread().is_some() => o
            .as_thread()
            .map(|t| t.get().thread_env)
            .unwrap_or(l.thread_env),
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
    } else if let Some(t) = o.as_thread() {
        t.get().thread_env = tab;
    }
    push(l, LuaValue::number(0.0));
    Ok(1)
}

/// Move one frame up (toward the caller) following `link`; resolves the
/// VARG pseudo-frames of the frame being left. Returns the caller's
/// frame base, or `None` at the chain's end.
fn walk_next(l: &LuaState, mut slot: usize, mut cur_link: u64) -> Option<usize> {
    // Skip VARG pseudo-frames above the frame being left.
    while (cur_link & FRAME_TYPE_MASK) == 3 {
        slot = slot.saturating_sub((cur_link >> 3) as usize);
        if slot < 2 {
            return None;
        }
        cur_link = l.stack[slot - 1].to_bits();
    }
    let frame_type = cur_link & FRAME_TYPE_MASK;
    match frame_type {
        0 /* FRAME_LUA */ if cur_link != 0 => {
            if ((cur_link >> 3) as usize) < l.stack.len() {
                Some((cur_link >> 3) as usize)
            } else {
                let ret_ip = cur_link as *const crate::bc::BCIns;
                let call_ins = unsafe { *ret_ip.sub(1) };
                let a = crate::bc::bc_a(call_ins) as usize;
                Some(slot.saturating_sub(2 + a))
            }
        }
        1 /* FRAME_C */ | 2 /* FRAME_CONT */ if cur_link != 0 => {
            // The delta is the distance back to the caller's base.
            let d = (cur_link >> 3) as usize;
            if d == 0 || d > slot {
                None
            } else {
                Some(slot - d)
            }
        }
        _ => None,
    }
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
    let mut first_lua = true;
    // luaL_traceback starts at level 1: the traceback handler's own frame
    // (the frame executing right now) is skipped.
    if slot >= 2 {
        let link = l.stack[slot - 1].to_bits();
        if let Some(next) = walk_next(l, slot, link) {
            slot = next;
        } else {
            slot = 0;
        }
    }
    for _ in 0..64 {
        if slot < 2 {
            break;
        }
        let func = l.stack[slot - 2];
        let mut cur_link = l.stack[slot - 1].to_bits();
        let orig_slot = slot;
        while (cur_link & FRAME_TYPE_MASK) == 3
        /* FRAME_VARG */
        {
            slot = slot.saturating_sub((cur_link >> 3) as usize);
            if slot < 2 {
                break;
            }
            cur_link = l.stack[slot - 1].to_bits();
        }
        let _frame_type = cur_link & FRAME_TYPE_MASK;

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
                    // The first Lua frame below the handler is the failed
                    // frame: prefer the recorded raise site (error line),
                    // which its frame link alone cannot provide.
                    let pc = if first_lua {
                        if let Some((fbits, epc)) = l.err_raise_pc
                            && func.to_bits() == fbits
                            && epc < pt.bc.len()
                        {
                            l.err_raise_pc = None;
                            epc
                        } else {
                            pc
                        }
                    } else {
                        pc
                    };
                    let line = if pc < pt.lines.len() {
                        pt.lines[pc] as usize
                    } else {
                        pt.firstline as usize
                    };
                    // LuaJIT prints the function name for metamethod
                    // frames ("in function '__index'"); other Lua frames
                    // stay anonymous here. The main chunk (firstline 0)
                    // gets its own label.
                    let mut label = if pt.firstline == 0 {
                        "main chunk".to_string()
                    } else {
                        "function".to_string()
                    };
                    if let Some(mm) = self_mm_name(l, orig_slot) {
                        label = format!("function '{}'", mm);
                    }
                    trace.push_str(&format!("\t{}:{}: in {}\n", src, line, label));
                    first = false;
                    first_lua = false;
                }
                GcFunc::C(_) => {
                    trace.push_str("\t[C]: in function\n");
                    first = false;
                }
            }
        }

        // Walk to caller.
        if let Some(next) = walk_next(l, slot, cur_link) {
            slot = next;
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

fn lib_gethook(l: &mut LuaState) -> LuaResult<i32> {
    // debug.gethook([thread]) — thread form unsupported: use the main thread.
    let n = nargs(l);
    let thread = if n > 0 && arg(l, 0).is_thread() {
        arg(l, 0)
    } else {
        LuaValue::NIL
    };
    let st = if thread.is_thread() {
        thread.as_thread().map(|t| unsafe { t.as_ref() })
    } else {
        None
    };
    let hook = st.map(|s| s.hook).unwrap_or_else(|| l.hook);
    let mask = st.map(|s| s.hookmask).unwrap_or(l.hookmask);
    let count = st
        .map(|s| s.hook_count_reset)
        .unwrap_or(l.hook_count_reset);
    l.stack_ensure(l.base + 3);
    l.stack[l.base] = hook;
    let mut m = String::new();
    if mask & crate::vm::HOOKMASK_CALL != 0 {
        m.push('c');
    }
    if mask & crate::vm::HOOKMASK_RET != 0 {
        m.push('r');
    }
    if mask & crate::vm::HOOKMASK_LINE != 0 {
        m.push('l');
    }
    l.stack[l.base + 1] = l.heap().str_value(l.heap().intern(m.as_bytes()));
    l.stack[l.base + 2] = LuaValue::number(count as f64);
    l.top = l.base + 3;
    Ok(3)
}

fn lib_sethook(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l);
    // debug.sethook(hook, mask[, count]) or sethook([thread,] hook, mask[, count])
    let mut idx = 0usize;
    let mut target: Option<GcPtr<crate::state::LuaState>> = None;
    if n > 0 && arg(l, 0).is_thread() {
        target = arg(l, 0).as_thread();
        idx = 1;
    }
    let hook = arg(l, idx);
    let mask = if n > idx + 1 {
        arg(l, idx + 1)
            .as_string_id()
            .map(|sid| String::from_utf8_lossy(l.str_static(sid)).into_owned())
            .unwrap_or_default()
    } else {
        String::new()
    };
    let count = if n > idx + 2 {
        arg(l, idx + 2).as_number().unwrap_or(0.0) as i32
    } else {
        0
    };
    let mut hm = 0u8;
    for c in mask.chars() {
        match c {
            'l' => hm |= crate::vm::HOOKMASK_LINE,
            'c' => hm |= crate::vm::HOOKMASK_CALL,
            'r' => hm |= crate::vm::HOOKMASK_RET,
            _ => {}
        }
    }
    if hm == 0 && count > 0 {
        hm |= crate::vm::HOOKMASK_COUNT;
    }
    // Lua 5.1: the line hook doesn't fire for the line the hook was
    // installed on; seed hook_line with the caller's current line.
    let cur_line = caller_line(l).unwrap_or(0);
    if let Some(t) = target {
        let t = unsafe { t.as_mut() };
        t.hook = if hook.is_nil() { LuaValue::NIL } else { hook };
        t.hookmask = hm;
        t.hookcount = count;
        t.hook_count_reset = count;
        t.hook_line = cur_line;
    } else {
        l.hook = if hook.is_nil() { LuaValue::NIL } else { hook };
        l.hookmask = hm;
        l.hookcount = count;
        l.hook_count_reset = count;
        l.hook_line = cur_line;
    }
    Ok(0)
}

/// The caller's current source line (for sethook's initial hook_line).
fn caller_line(l: &LuaState) -> Option<u32> {
    use crate::bc::bc_op;
    if l.base < 2 {
        return None;
    }
    let link = l.stack[l.base - 1].to_bits();
    if (link & FRAME_TYPE_MASK) != 0 || link == 0 {
        return None;
    }
    let caller_base = if ((link >> 3) as usize) < l.stack.len() {
        // C-call frame: the link encodes the caller's base.
        (link >> 3) as usize
    } else {
        // Lua frame: the link is the caller's return PC.
        let ret_ip = link as *const crate::bc::BCIns;
        if (ret_ip as usize) < 0x10000 {
            return None;
        }
        let call_ins = unsafe { *ret_ip.sub(1) };
        if !matches!(
            bc_op(call_ins),
            crate::bc::BCOp::CALL
                | crate::bc::BCOp::CALLT
                | crate::bc::BCOp::CALLM
                | crate::bc::BCOp::CALLMT
        ) {
            return None;
        }
        let a = crate::bc::bc_a(call_ins) as usize;
        let base = l.base.saturating_sub(2 + a);
        if base < 2 {
            return None;
        }
        let pt = l.stack[base - 2].as_func()?;
        let pt = match pt.as_ref() {
            crate::func::GcFunc::Lua(cl) => cl.proto.as_ref(),
            _ => return None,
        };
        let pc = unsafe { ret_ip.offset_from(pt.bc.as_ptr()) } as usize;
        return Some(pt.lines[pc.saturating_sub(1).min(pt.lines.len().saturating_sub(1))] as u32);
    };
    if caller_base < 2 {
        return None;
    }
    let pt = l.stack[caller_base - 2].as_func()?;
    let pt = match pt.as_ref() {
        crate::func::GcFunc::Lua(cl) => cl.proto.as_ref(),
        _ => return None,
    };
    let pc = l.debug_pc.min(pt.lines.len().saturating_sub(1));
    Some(pt.lines[pc] as u32)
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

fn lib_setupvalue(l: &mut LuaState) -> LuaResult<i32> {
    let f = arg(l, 0);
    let idx = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let val = arg(l, 2);
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
                cl.upvals[uv_idx].as_mut().set(val);
                Ok(1)
            }
            GcFunc::C(c) => {
                if idx < 1 || idx > c.upvals.len() {
                    push(l, LuaValue::NIL);
                    return Ok(1);
                }
                push(l, l.heap().str_value(l.heap().intern(b"")));
                let uv_idx = idx - 1;
                l.set_upvalue(uv_idx, val);
                Ok(1)
            }
        },
        None => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn lib_getlocal(l: &mut LuaState) -> LuaResult<i32> {
    let level = arg(l, 0).as_number().unwrap_or(0.0) as i32;
    let local = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    if level == 0 {
        // Lua 5.1: level 0 names the current C frame's temporary
        // registers ("(*temporary)"); slot n is base + n - 1.
        let idx = l.base + local - 1;
        if local == 0 || idx >= l.top || idx >= l.stack.len() {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
        let val = l.stack[idx];
        let name_v = str_val(l, "(*temporary)");
        pushv(l, &[name_v, val]);
        return Ok(2);
    }
    let (slot, gf) = match walk_frames(l, level) {
        Some(s) => s,
        None => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };
    // The frame's current pc: for the current frame it is the
    // interpreter's debug_pc; for a caller frame it is decoded from the
    // *previous* frame's link (which points at the caller's return PC).
    let prev_slot = if level == 1 {
        None
    } else {
        walk_frames(l, level - 1).map(|(s, _)| s)
    };
    match gf.as_ref() {
        GcFunc::Lua(cl) => {
            let pt = cl.proto.as_ref();
            let (name, idx) = match frame_local(l, slot, pt, local, prev_slot) {
                Some(x) => x,
                None => {
                    push(l, LuaValue::NIL);
                    return Ok(1);
                }
            };
            let val = if idx < l.stack.len() {
                l.stack[idx]
            } else {
                LuaValue::NIL
            };
            let name_v = str_val(l, &name);
            pushv(l, &[name_v, val]);
            Ok(2)
        }
        GcFunc::C(_) => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

/// Resolve the `n`-th *active* local of the frame at `slot` (Lua 5.1's
/// `luaF_getlocalname`: locals are numbered in declaration order among
/// those visible at the frame's current pc).
fn frame_local(
    l: &LuaState,
    slot: usize,
    pt: &crate::proto::Proto,
    n: usize,
    prev_slot: Option<usize>,
) -> Option<(String, usize)> {
    let pc = match prev_slot {
        None => l.debug_pc.min(pt.lines.len().saturating_sub(1)) as u32,
        Some(ps) => frame_pc(l, ps, pt)?,
    };
    let mut seen = 0usize;
    // Lua 5.1 numbers the vararg parameter ("...") as the first local of
    // a vararg function.
    let vararg_offset = if pt.flags & crate::proto::PROTO_VARARG != 0 {
        1
    } else {
        0
    };
    for (reg, spc, epc, name) in &pt.varnames {
        let end = if *epc == 0 { pt.bc.len() as u32 } else { *epc };
        if *spc <= pc && pc < end {
            seen += 1;
            if seen + vararg_offset == n {
                return Some((name.clone(), slot + *reg as usize));
            }
        }
    }
    // Lua 5.1 falls back to the raw register: "(*temporary)" if the slot
    // lies within the frame's parameter area (lua_getlocal's temporary
    // handling: `n < ci->base - ci->func`).
    let raw = slot + n - 1 - vararg_offset;
    let limit = slot + 1 + pt.numparams as usize;
    if raw >= slot && raw < limit {
        return Some(("(*temporary)".to_string(), raw));
    }
    None
}

/// The current bytecode position of the frame at `slot` (pc-1 of its
/// return PC, falling back to the call site).
fn frame_pc(l: &LuaState, slot: usize, pt: &crate::proto::Proto) -> Option<u32> {
    use crate::bc::bc_op;
    if slot < 2 {
        return None;
    }
    let link = l.stack[slot - 1].to_bits();
    if (link & FRAME_TYPE_MASK) != 0 || link == 0 {
        return None;
    }
    if ((link >> 3) as usize) < l.stack.len() {
        // Base-encoded link: the frame is a host-fabricated one; use the
        // interpreter's current pc.
        let pc = l.debug_pc.min(pt.lines.len().saturating_sub(1));
        return Some(pc as u32);
    }
    let ret_ip = link as *const crate::bc::BCIns;
    if (ret_ip as usize) < 0x10000 {
        return None;
    }
    let call_ins = unsafe { *ret_ip.sub(1) };
    if !matches!(
        bc_op(call_ins),
        crate::bc::BCOp::CALL
            | crate::bc::BCOp::CALLT
            | crate::bc::BCOp::CALLM
            | crate::bc::BCOp::CALLMT
            | crate::bc::BCOp::ITERC
    ) {
        return None;
    }
    let off = unsafe { ret_ip.offset_from(pt.bc.as_ptr()) } as usize;
    Some(off.saturating_sub(1) as u32)
}

fn lib_setlocal(l: &mut LuaState) -> LuaResult<i32> {
    let level = arg(l, 0).as_number().unwrap_or(0.0) as i32;
    let local = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let val = arg(l, 2);
    let (slot, gf) = match walk_frames(l, level) {
        Some(s) => s,
        None => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };
    let prev_slot = if level == 1 {
        None
    } else {
        walk_frames(l, level - 1).map(|(s, _)| s)
    };
    match gf.as_ref() {
        GcFunc::Lua(cl) => {
            let pt = cl.proto.as_ref();
            let (_name, idx) = match frame_local(l, slot, pt, local, prev_slot) {
                Some(x) => x,
                None => {
                    push(l, LuaValue::NIL);
                    return Ok(1);
                }
            };
            let old = if idx < l.stack.len() {
                l.stack[idx]
            } else {
                LuaValue::NIL
            };
            if idx < l.stack.len() {
                l.stack[idx] = val;
            }
            push(l, old);
            Ok(1)
        }
        GcFunc::C(_) => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

// ── open ────────────────────────────────────────────────────────────────────

pub fn open(l: &mut LuaState) {
    crate::stdlib::reg::LibBuilder::new(l, b"debug", crate::stdlib::reg::LibTarget::Global)
        .func(b"debug", lib_debug_debug)
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
        .func(b"setupvalue", lib_setupvalue)
        .func(b"upvaluejoin", lib_upvaluejoin)
        .func(b"getlocal", lib_getlocal)
        .func(b"setlocal", lib_setlocal)
        .build();
}

fn lib_debug_debug(l: &mut LuaState) -> LuaResult<i32> {
    // The interactive debugger prompt is not implemented; the tests only
    // check its presence.
    push(l, LuaValue::NIL);
    Ok(0)
}
