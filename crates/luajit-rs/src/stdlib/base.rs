//! Base library: `print`, `type`, `tostring`, `tonumber`, `select`,
//! `pairs`, `ipairs`, `next`, `assert`, `setmetatable`, `collectgarbage`,
//! `error`, `pcall`, `xpcall`, `rawequal`, `rawget`, `rawset`, `getmetatable`,
//! `newproxy`.

use crate::err::{LuaError, LuaResult};
use crate::runtime::meta::MM;
use crate::runtime::userdata::GcUserData;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::{LJ_TNIL, LuaValue};

use super::{LibTarget, arg, err_bad_arg_type, nargs, push, pushv, tostring_meta};
use crate::lual_reg;

fn lib_print(l: &mut LuaState) -> LuaResult<i32> {
    use std::io::Write;
    let n = nargs(l);
    let mut out = Vec::new();
    // LuaJIT's print goes through the (thread) environment's tostring,
    // so an overridden one is honored.
    let ts_key = {
        let sid = l.heap().intern(b"tostring");
        l.heap().str_value(sid)
    };
    let env = LuaValue::table(l.thread_env);
    let ts_fn = match crate::meta::metatable_of(l.global(), env) {
        Some(_) => {
            let mo = crate::meta::meta_lookup(l.global(), env, crate::meta::MM::Index);
            match mo.as_table() {
                Some(mt2) => mt2.as_ref().get(ts_key),
                None => LuaValue::NIL,
            }
        }
        None => l.thread_env.as_ref().get(ts_key),
    };
    for i in 0..n {
        if i > 0 {
            out.push(b'\t');
        }
        let v = arg(l, i);
        let bytes = if ts_fn.is_func() {
            let saved_top = l.top;
            let saved_base = l.base;
            let fs = l.top + 16;
            l.stack_ensure(fs + 4);
            l.stack[fs] = ts_fn;
            l.stack[fs + 2] = v;
            crate::vm::execute(l, fs, 1, 1)?;
            let r = l.stack[fs];
            l.top = saved_top;
            l.base = saved_base;
            if let Some(sid) = r.as_string_id() {
                l.str_static(sid).to_vec()
            } else {
                return Err(l.runtime_error(b"'tostring' must return a string"));
            }
        } else {
            tostring_meta(l, v)?
        };
        out.extend_from_slice(&bytes);
    }
    out.push(b'\n');
    let _ = std::io::stdout().lock().write_all(&out);
    Ok(0)
}

pub fn lib_type(l: &mut LuaState) -> LuaResult<i32> {
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
    } else if v.is_cdata() {
        b"cdata"
    } else if v.is_thread() {
        b"thread"
    } else {
        b"userdata"
    };
    let sid = l.heap().intern(name);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn lib_tostring(l: &mut LuaState) -> LuaResult<i32> {
    if crate::stdlib::nargs(l) < 1 {
        return Err(l.runtime_error(b"bad argument #1 to 'tostring' (value expected)"));
    }
    let v = arg(l, 0);
    // Lua 5.1 luaB_tostring: the __tostring result is returned verbatim
    // (even nil); only when there is no metamethod is the raw fallback used.
    let mo = crate::meta::meta_lookup(l.global(), v, crate::meta::MM::Tostring);
    if !mo.is_nil() {
        let saved_top = l.top;
        let saved_base = l.base;
        let fs = l.top + 16;
        l.stack_ensure(fs + 4);
        l.stack[fs] = mo;
        l.stack[fs + 2] = v;
        crate::vm::execute(l, fs, 1, 1)?;
        let r = l.stack[fs];
        l.top = saved_top;
        l.base = saved_base;
        push(l, r);
        return Ok(1);
    }
    let bytes = crate::stdlib::tostring_bytes(l, v);
    let sid = l.heap().intern(&bytes);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn lib_tonumber(l: &mut LuaState) -> LuaResult<i32> {
    if crate::stdlib::nargs(l) < 1 {
        return Err(l.runtime_error(b"bad argument #1 to 'tonumber' (value expected)"));
    }
    let v = arg(l, 0);
    let r = if nargs(l) > 1 {
        // tonumber(e, base): parse e (string or number) in the given base.
        let base = arg(l, 1).as_number().unwrap_or(0.0) as u32;
        let s = if let Some(sid) = v.as_string_id() {
            l.heap().strings.get(sid).to_vec()
        } else if v.is_number() {
            crate::strfmt::g14(v.num()).into_bytes()
        } else {
            Vec::new()
        };
        let mut r = LuaValue::NIL;
        // Lua 5.1 luaB_tonumber with base: skip leading whitespace, parse
        // base digits, skip trailing whitespace, then require end of
        // string (`tonumber('  1010  ', 2) == 10`, `tonumber('  ', 9)`
        // is nil, `tonumber('99', 8)` is nil).
        if (2..=36).contains(&base) {
            let mut i = 0;
            while i < s.len() && (s[i] as char).is_ascii_whitespace() {
                i += 1;
            }
            let mut n: u64 = 0;
            let mut any = false;
            while i < s.len() {
                let c = s[i];
                let d = match c {
                    b'0'..=b'9' => (c - b'0') as u64,
                    b'a'..=b'z' => (c - b'a' + 10) as u64,
                    b'A'..=b'Z' => (c - b'A' + 10) as u64,
                    _ => break, // invalid digit
                };
                if d >= base as u64 {
                    break;
                }
                n = n.wrapping_mul(base as u64).wrapping_add(d);
                any = true;
                i += 1;
            }
            while i < s.len() && (s[i] as char).is_ascii_whitespace() {
                i += 1;
            }
            if any && i == s.len() {
                r = LuaValue::number(n as f64);
            }
        }
        r
    } else if v.is_number() {
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
    // 5.1 luaB_select: negative i counts from the end (i = n + i), then
    // clamps to [1, n] (select(10000, ...) yields nothing, no error).
    let mut k = match first.as_number() {
        Some(k) if k >= 1.0 => k as isize,
        Some(k) if k < 0.0 => n as isize + k as isize,
        _ => {
            return Err(err_bad_arg_type(l, 1, "select", "number or '#'", arg(l, 0)));
        }
    };
    if k > n as isize {
        k = n as isize;
    }
    if k < 1 {
        return Err(l.runtime_error(b"bad argument #1 to 'select' (index out of range)"));
    }
    let k = k as usize;
    // Args are 0-based on the stack; the value args sit at 1..n.
    let mut cnt = 0;
    l.stack_ensure(l.base + n.saturating_sub(k));
    for i in k..n {
        l.stack[l.base + cnt] = arg(l, i);
        cnt += 1;
    }
    Ok(cnt as i32)
}

/// (pub: the JIT's fast-function recorder identifies builtins by their
/// function pointer.)
pub fn lib_next(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "next", "table", arg(l, 0))),
    };
    // LuaJIT's lj_tab_next raises for a key that is not (or no longer) a
    // key of the table. A key deleted mid-traversal stays findable until
    // a rehash reclaims its node, so the strict check does not break
    // `for k in pairs(t) do t[k] = nil end` loops.
    if !k.is_nil() && !tab.as_ref().is_valid_key(k) {
        return Err(l.runtime_error(b"invalid key to 'next'"));
    }
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
    // __pairs metamethod (5.2): it returns the iterator triple.
    let mo = crate::meta::meta_lookup(l.global(), t, crate::meta::MM::Pairs);
    if !mo.is_nil() {
        let obase = l.base;
        let fs = l.top + 16;
        l.stack_ensure(fs + 4);
        l.stack[fs] = mo;
        l.stack[fs + 2] = t;
        crate::vm::execute(l, fs, 1, 3)?;
        // execute restores the caller's top; the three results (padded
        // with nil) are always at fs..fs+3.
        l.stack_ensure(obase + 3);
        for i in 0..3 {
            l.stack[obase + i] = l.stack[fs + i];
        }
        l.top = obase + 3;
        l.base = obase;
        return Ok(3);
    }
    let sid = l.heap().intern(b"next");
    let key = l.heap().str_value(sid);
    let next_fn = l.global().globals.as_ref().get(key);
    pushv(l, &[next_fn, t, LuaValue::NIL]);
    Ok(3)
}

/// (pub: the JIT's fast-function recorder identifies builtins by their
/// function pointer.)
pub fn lib_ipairs_iter(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let i = arg(l, 1).as_number().unwrap_or(0.0) + 1.0;
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "ipairs", "table", arg(l, 0))),
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
    // __ipairs metamethod (5.2): it returns the iterator triple.
    let mo = crate::meta::meta_lookup(l.global(), t, crate::meta::MM::Ipairs);
    if !mo.is_nil() {
        let obase = l.base;
        let fs = l.top + 16;
        l.stack_ensure(fs + 4);
        l.stack[fs] = mo;
        l.stack[fs + 2] = t;
        crate::vm::execute(l, fs, 1, 3)?;
        // execute restores the caller's top; the three results (padded
        // with nil) are always at fs..fs+3.
        l.stack_ensure(obase + 3);
        for i in 0..3 {
            l.stack[obase + i] = l.stack[fs + i];
        }
        l.top = obase + 3;
        l.base = obase;
        return Ok(3);
    }
    let sid = l.heap().intern(b"__ipairs_iter");
    let _key = l.heap().str_value(sid);
    let iter = l.global().ipairs_iter;
    pushv(l, &[iter, t, LuaValue::number(0.0)]);
    Ok(3)
}

fn lib_setmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let mt = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => {
            return Err(err_bad_arg_type(l, 1, "setmetatable", "table", arg(l, 0)));
        }
    };
    if !mt.is_table() && !mt.is_nil() {
        return Err(err_bad_arg_type(
            l,
            2,
            "setmetatable",
            "nil or table",
            arg(l, 2 - 1),
        ));
    }
    // Protected metatable check (lj_meta_lookup(o, MM_metatable)).
    if !crate::meta::meta_lookup(l.global(), t, MM::Metatable).is_nil() {
        return Err(l.runtime_error(b"cannot change a protected metatable"));
    }
    tab.as_mut().metatable = mt.as_table();
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
        // luaL_where(1): the direct caller's location. A C caller (e.g.
        // pcall) has no source line, so the message stays bare.
        let loc = assert_where(l);
        let full = if let Some(sid) = msg.as_string_id() {
            format!("{}{}", loc, String::from_utf8_lossy(l.str_static(sid)))
        } else {
            format!("{}assertion failed!", loc)
        };
        l.errval = l.heap().str_value(l.heap().intern(full.as_bytes()));
        Err(LuaError::Runtime)
    }
}

/// `luaL_where(L, 1)`: "file:line: " of the direct caller; empty when the
/// caller is a C function (pcall, the library itself, ...).
fn assert_where(l: &LuaState) -> String {
    if l.base < 2 {
        return String::new();
    }
    let link = l.stack[l.base - 1].to_bits();
    if (link & crate::vm::FRAME_TYPE_MASK) != 0 || link == 0 {
        return String::new();
    }
    let caller_base = (link >> 3) as usize;
    if caller_base < 2 || caller_base >= l.stack.len() {
        return String::new();
    }
    let Some(fv) = l.stack[caller_base - 2].as_func() else {
        return String::new();
    };
    let crate::func::GcFunc::Lua(cl) = fv.as_ref() else {
        return String::new();
    };
    let pt = cl.proto.as_ref();
    let pc = l
        .debug_pc
        .saturating_sub(1)
        .min(pt.lines.len().saturating_sub(1));
    let line = pt.lines[pc];
    let src = pt
        .source
        .and_then(|sid| {
            l.heap().strings.try_lookup(sid)?;
            let b = l.heap().strings.get(sid);
            if b.starts_with(b"@") || b.starts_with(b"=") {
                Some(&b[1..])
            } else {
                Some(b)
            }
        })
        .unwrap_or(b"?")
        .to_vec();
    format!("{}:{}: ", String::from_utf8_lossy(&src), line)
}

fn lib_collectgarbage(l: &mut LuaState) -> LuaResult<i32> {
    let opt = match arg(l, 0).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => b"collect".to_vec(),
    };
    match opt.as_slice() {
        b"collect" | b"full" => {
            // Run any pending finalizers before and after the collection
            // (a full cycle may separate new ones in its atomic phase).
            crate::vm::run_finalizers(l)?;
            crate::gc::full_gc(l.global());
            crate::vm::run_finalizers(l)?;
            // Lua 5.1's luaC_step calls setthreshold when the cycle
            // completes, which re-arms the automatic collector even after
            // collectgarbage("stop"). Mirror that: a full collection
            // un-stops the GC, or a later allocation loop (e.g. closure.lua's
            // `while x[1] do ... end` weak-table loop) would spin forever.
            l.global().heap.gc_stopped = false;
            push(l, LuaValue::number(0.0));
            Ok(1)
        }
        b"step" => {
            let size = arg(l, 1).as_number().unwrap_or(0.0) as usize;
            let g = l.global();
            // LuaJIT LUA_GCSTEP:
            //   GCSize a = data << 10;
            //   g->gc.threshold = (a <= g->gc.total) ? (g->gc.total - a) : 0;
            //   while (g->gc.total >= g->gc.threshold)
            //     if (lj_gc_step(L) > 0) { res = 1; break; }
            //
            // lj_gc_step does `do { lim -= gc_onestep } while lim > 0`.
            // We emulate this: `lim` gc_step() calls per collectgarbage().
            // Default size==0: 1 call (minimal step).  Larger size: more calls.
            let a = size * 1024;
            let live = g.heap.total + g.heap.strings.bytes() + g.heap.table_extra;
            g.heap.threshold = live.saturating_sub(a);
            let mut lim = if size == 0 { 1u64 } else { size as u64 };
            // Start a cycle if idle.
            if g.heap.gc_state == crate::runtime::gc::GcState::Pause {
                crate::gc::start_gc_cycle(g);
            }
            let mut done = false;
            loop {
                if lim == 0 {
                    break;
                }
                lim -= 1;
                let step_done = crate::gc::gc_step(&mut g.heap, crate::runtime::gc::GC_STEP_SIZE);
                // Detect completion: gc_step returned true, or the cycle finished
                // behind our back (C-call boundary completed it).
                if step_done || g.heap.gc_state == crate::runtime::gc::GcState::Pause {
                    done = true;
                    break;
                }
            }
            // Restore threshold so C-call boundaries don't fire full_gc
            // immediately.  LuaJIT's lj_gc_step debt logic does this.
            let new_threshold = if !done {
                g.heap.total + g.heap.strings.bytes() + crate::runtime::gc::GC_STEP_SIZE
            } else {
                g.heap.threshold
            };
            if l.global().heap.gc_state == crate::runtime::gc::GcState::Finalize {
                crate::vm::run_finalizers(l)?;
            }
            push(l, LuaValue::boolean(done));
            l.global().heap.threshold = new_threshold;
            Ok(1)
        }
        b"stop" => {
            l.global().heap.gc_stopped = true;
            Ok(0)
        }
        b"restart" => {
            l.global().heap.gc_stopped = false;
            Ok(0)
        }
        b"setpause" | b"setstepmul" => {
            // GC parameters — no-op for now.
            Ok(0)
        }
        b"count" => {
            let heap = &l.global().heap;
            let bytes = heap.total + heap.strings.bytes() + heap.table_extra;
            push(l, LuaValue::number(bytes as f64 / 1024.0));
            Ok(1)
        }
        _ => Err(err_bad_arg_type(
            l,
            1,
            "collectgarbage",
            "option string",
            arg(l, 0),
        )),
    }
}

fn lib_gcinfo(l: &mut LuaState) -> LuaResult<i32> {
    let heap = &l.global().heap;
    let bytes = heap.total + heap.strings.bytes() + heap.table_extra;
    push(l, LuaValue::number(bytes as f64 / 1024.0));
    Ok(1)
}

pub fn lib_rawget(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "rawget", "table", arg(l, 0))),
    };
    push(l, tab.as_ref().get(k));
    Ok(1)
}

pub fn lib_rawset(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let k = arg(l, 1);
    let v = arg(l, 2);
    let tab = match t.as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "rawset", "table", arg(l, 0))),
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
    if level == 0 {
        // level 0: the message is used verbatim (no position added).
        l.errval = if msg.is_nil() { LuaValue::NIL } else { msg };
        return Err(LuaError::Runtime);
    }
    // Remember the frame that called `error`: its stack slots survive the
    // unwind, so an enclosing xpcall handler can chain to it and debug
    // walks (getlocal levels) still see it below the error's C frame.
    if let Some(link) = l.stack.get(l.base.wrapping_sub(1)).map(|v| v.to_bits())
        && (link & 3) == 0
    {
        l.err_raise_slot = (link >> 3) as usize;
    }
    // The position prefix is added only when the direct caller is a Lua
    // function (LuaJIT's lj_ff_error: a C caller such as pcall/xpcall has
    // no location to report).
    let link = l.stack[l.base - 1].to_bits();
    let caller_is_lua = (link & 3) == 0 && {
        let sb = (link >> 3) as usize;
        sb >= 2
            && l.stack[sb - 2]
                .as_func()
                .is_some_and(|f| matches!(f.as_ref(), crate::func::GcFunc::Lua(_)))
    };
    if caller_is_lua && let Some(sid) = msg.as_string_id() {
        let bytes = l.heap().strings.get(sid).to_vec();
        return Err(l.runtime_error_level(&bytes, level as u32));
    }
    l.errval = if msg.is_nil() { LuaValue::NIL } else { msg };
    Err(LuaError::Runtime)
}

/// `pcall(f [, arg...])` — protected call. The callee may be any value:
/// `__call` resolution (or the call-type error) happens inside `execute`.
fn lib_pcall(l: &mut LuaState) -> LuaResult<i32> {
    let n = nargs(l).saturating_sub(1);
    // Move `n` trailing args into call position right after `f`.
    // Reverse order: dest overlaps src (dest = src + 1).
    l.stack_ensure(l.base + 2 + n);
    for i in (0..n).rev() {
        l.stack[l.base + 2 + i] = arg(l, i + 1);
    }
    let saved_base = l.base;
    match crate::vm::execute_yieldable(l, l.base, n, -1) {
        Ok(nret) => {
            // `execute` leaves `l.base` at the callee frame: restore it
            // before writing the results.
            l.base = saved_base;
            // Shift results down so the true/false header can go first.
            l.stack_ensure(l.base + nret + 1);
            for i in (0..nret).rev() {
                l.stack[l.base + i + 1] = l.stack[l.base + i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            Ok(nret as i32 + 1)
        }
        Err(LuaError::Runtime) => {
            l.base = saved_base;
            l.stack_ensure(l.base + 2);
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = l.errval;
            // The raised frame's slot/PC were recorded for traceback; the
            // error is now handled, so drop them — the recorded frame may
            // have been popped (and its closure collected) by the time the
            // caller runs a traceback, and a stale slot would point at a
            // recycled closure.
            l.err_trace_slot = None;
            l.err_raise_pc = None;
            Ok(2)
        }
        Err(LuaError::Yield) => {
            // The yield happened *inside* this protected call (e.g.
            // pcall(coroutine.yield, ...)): rewrite the captured suspend
            // so the resumption lands after pcall's own CALL and delivers
            // `true, <yield values>` as its results. The continuation
            // must run in the *caller's* frame (pcall's own CALL site),
            // not the frame that yielded.
            l.base = saved_base;
            let cl = l
                .stack
                .get(saved_base.saturating_sub(2))
                .and_then(|v| v.as_func())
                .unwrap_or(l.stack[0].as_func().unwrap());
            let value_slot = match l.suspend {
                crate::state::Suspend::Call { value_slot, .. } => value_slot,
                _ => saved_base,
            };
            l.suspend = crate::state::Suspend::Call {
                pc: l.debug_pc,
                cl,
                base: saved_base,
                slot: saved_base.saturating_sub(2),
                want: -1,
                protected: true,
                value_slot,
            };
            Err(LuaError::Yield)
        }
    }
}

/// `xpcall(f, msgh [, arg...])` — protected call with error handler.
fn lib_xpcall(l: &mut LuaState) -> LuaResult<i32> {
    let msgh = arg(l, 1);
    let n = nargs(l).saturating_sub(2);
    let saved_base = l.base;
    l.stack_ensure(l.base + 2 + n);
    for i in (0..n).rev() {
        l.stack[l.base + 2 + i] = arg(l, i + 2);
    }
    match crate::vm::execute_yieldable(l, l.base, n, -1) {
        Ok(nret) => {
            // `execute` leaves `l.base` at the callee frame: restore it
            // before writing the results (mirrors the Err branch).
            l.base = saved_base;
            l.stack_ensure(l.base + nret + 1);
            for i in (0..nret).rev() {
                l.stack[l.base + i + 1] = l.stack[l.base + i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            Ok(nret as i32 + 1)
        }
        Err(LuaError::Runtime) => {
            // Capture the innermost surviving frame *before* restoring
            // base: the error unwinds to it (call_c restores the caller's
            // frame, a Lua frame keeps its own base).
            let fail_base = l.base;
            // `execute` unwinds with `l.base` left at the callee frame:
            // restore it before writing the results (mirrors lib_pcall).
            l.base = saved_base;

            l.stack_ensure(l.base + 4);
            // Invoke the message handler with the error object; its
            // first result replaces the error (lj_ff_xpcall).
            if msgh.is_func() {
                // Preserve the failed call's frame chain below the handler
                // and chain the handler's frame link to it, so debug walks
                // (e.g. getlocal levels, traceback) can see the failed
                // frame. When the error was raised by `error()` from a
                // nested call, chain to the raise-time frame (its slots
                // survive the unwind) instead of the post-unwind base.
                let (chain_base, fs) = if l.err_raise_slot >= 2 {
                    // The raise slot is the frame's args base; the chain
                    // expects the target frame's func slot (below it).
                    let s = l.err_raise_slot - 2;
                    l.err_raise_slot = 0;
                    let fail_framesize = l.stack[s]
                        .as_func()
                        .and_then(|f| match f.as_ref() {
                            crate::func::GcFunc::Lua(cl) => {
                                Some(cl.proto.as_ref().framesize as usize)
                            }
                            _ => None,
                        })
                        .unwrap_or(0);
                    (s + 2, s + 2 + fail_framesize)
                } else {
                    // Place the handler above the failed frame's top so
                    // its frame chain (metamethod continuation frames,
                    // C frames) survives below the handler.
                    (fail_base, l.top + 2)
                };
                l.stack_ensure(fs + 4);
                l.stack[fs] = msgh;
                l.stack[fs + 1] = LuaValue::NIL;
                l.stack[fs + 2] = l.errval;
                // Chain to the failed frame: the delta is the distance
                // from the handler's base to the failed frame's base.
                let delta = (fs + 2).saturating_sub(chain_base);
                let link = ((delta as u64) << 3) | 1; // FRAME_C (vm/mod.rs)
                // Point `l.base` at the failed frame: the handler's own
                // C frame link (written by call_c) then references it, so
                // traceback walks reach the failed frame (and any
                // metamethod continuation frame above it) instead of
                // jumping straight to the xpcall frame.
                l.base = chain_base;
                let hr = crate::vm::execute_link(l, fs, 1, 1, link);

                if let Ok(n) = hr
                    && n >= 1
                {
                    l.errval = l.stack[fs];
                }
                l.base = saved_base;
            }
            // The error is handled; drop the recorded raise frame so a
            // later traceback cannot dereference a frame that was already
            // popped and collected.
            l.err_trace_slot = None;
            l.err_raise_pc = None;
            l.stack_ensure(l.base + 2);
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = l.errval;
            Ok(2)
        }
        Err(e) => Err(e),
    }
}

fn lib_getmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let mt = crate::meta::metatable_of(l.global(), v);
    match mt {
        Some(m) => {
            let mm = crate::meta::meta_lookup(l.global(), v, crate::runtime::meta::MM::Metatable);
            if mm.is_nil() {
                push(l, LuaValue::table(m));
            } else {
                push(l, mm);
            }
        }
        None => push(l, LuaValue::NIL),
    }
    Ok(1)
}

/// `setfenv(f, table)` — set the environment of a function (Lua 5.1).
fn lib_setfenv(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let tab = match arg(l, 1).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 2, "setfenv", "table", arg(l, 2 - 1))),
    };
    if let Some(f) = o.as_func() {
        match f.as_mut() {
            crate::func::GcFunc::Lua(c) => c.env = tab,
            crate::func::GcFunc::C(c) => c.env = tab,
        }
        push(l, o);
    } else if let Some(t) = o.as_thread() {
        t.get().thread_env = tab;
        push(l, o);
    } else if o.is_number() {
        let level = o.as_number().unwrap() as i32;
        if level == 0 {
            // setfenv(0, t): the environment of the running thread.
            l.thread_env = tab;
            push(l, LuaValue::TRUE);
        } else if let Some(func) = crate::stdlib::debug::frame_func(l, level) {
            match func.as_mut() {
                crate::func::GcFunc::Lua(c) => c.env = tab,
                crate::func::GcFunc::C(c) => c.env = tab,
            }
            // Lua 5.1's luaB_setfenv returns the function at the level.
            push(l, LuaValue::func(func));
        } else {
            return Err(l.runtime_error(b"`setfenv' cannot change environment of given object"));
        }
    } else {
        return Err(err_bad_arg_type(
            l,
            1,
            "setfenv",
            "function or number",
            arg(l, 0),
        ));
    }
    Ok(1)
}

/// `getfenv(f)` — the environment of a function (Lua 5.1).
fn lib_getfenv(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let env = match o.as_func() {
        Some(f) => match f.as_ref() {
            crate::func::GcFunc::Lua(c) => c.env,
            crate::func::GcFunc::C(c) => c.env,
        },
        // getfenv(0): the environment of the running thread.
        _ if o.as_number() == Some(0.0) => l.thread_env,
        // getfenv(n): the environment of the function at debug level n.
        _ if o.is_number() => {
            match crate::stdlib::debug::frame_func(l, o.as_number().unwrap() as i32) {
                Some(f) => match f.as_ref() {
                    crate::func::GcFunc::Lua(c) => c.env,
                    crate::func::GcFunc::C(c) => c.env,
                },
                None => l.global().globals,
            }
        }
        _ => l.global().globals,
    };
    push(l, LuaValue::table(env));
    Ok(1)
}

/// `rawlen(v)` — raw length without metamethods (Lua 5.2).
fn lib_rawlen(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let n = if let Some(t) = v.as_table() {
        t.as_ref().len() as f64
    } else if let Some(sid) = v.as_string_id() {
        l.str_static(sid).len() as f64
    } else if let Some(_ud) = v.as_userdata() {
        // Lua 5.2: userdata length is the size of its memory block.
        std::mem::size_of::<GcUserData>() as f64
    } else {
        return Err(err_bad_arg_type(l, 1, "rawlen", "table or string", v));
    };
    push(l, LuaValue::number(n));
    Ok(1)
}

fn call_reader(l: &mut LuaState, reader: LuaValue) -> Result<Vec<u8>, Vec<u8>> {
    let saved_top = l.top;
    let saved_base = l.base;
    let fs = l.top + 16;
    l.stack_ensure(fs + 4);
    l.stack[fs] = reader;
    // No args to reader
    match crate::vm::execute(l, fs, 0, 1) {
        Ok(_) => {
            let r = l.stack[fs];
            l.top = saved_top;
            l.base = saved_base;
            if r.is_nil() {
                Ok(Vec::new())
            } else if let Some(sid) = r.as_string_id() {
                Ok(l.str_static(sid).to_vec())
            } else {
                let s = crate::stdlib::tostring_bytes(l, r);
                Ok(s)
            }
        }
        Err(_e) => {
            // Preserve the reader's error message (error("hhi") → "hhi").
            let msg = if let Some(sid) = l.errval.as_string_id() {
                l.str_static(sid).to_vec()
            } else {
                crate::stdlib::tostring_bytes(l, l.errval)
            };
            l.top = saved_top;
            l.base = saved_base;
            Err(msg)
        }
    }
}

fn lib_load(l: &mut LuaState) -> LuaResult<i32> {
    let src = arg(l, 0);
    if let Some(s) = src.as_string() {
        let code = s.as_ref().as_bytes().to_vec();
        let chunkname = if nargs(l) >= 2 {
            let v = arg(l, 1);
            if let Some(s2) = v.as_string() {
                String::from_utf8_lossy(s2.as_ref().as_bytes()).into_owned()
            } else {
                "=(load)".to_string()
            }
        } else {
            "=(load)".to_string()
        };
        match crate::state::load(l, code, &chunkname) {
            Ok(v) => {
                push(l, v);
                Ok(1)
            }
            Err(msg) => {
                l.stack[l.base] = LuaValue::NIL;
                l.stack[l.base + 1] = l
                    .global()
                    .heap
                    .str_value(l.global().heap.intern(msg.as_bytes()));
                l.top = l.base + 2;
                Ok(2)
            }
        }
    } else if src.is_func() {
        let chunkname = if nargs(l) >= 2 {
            let v = arg(l, 1);
            if let Some(s2) = v.as_string() {
                String::from_utf8_lossy(s2.as_ref().as_bytes()).into_owned()
            } else {
                "=(load)".to_string()
            }
        } else {
            "=(load)".to_string()
        };
        // Collect chunks from reader function
        let mut code: Vec<u8> = Vec::new();
        loop {
            let r = match call_reader(l, src) {
                Ok(chunk) => chunk,
                Err(e) => {
                    l.stack_ensure(l.base + 2);
                    l.stack[l.base] = LuaValue::NIL;
                    l.stack[l.base + 1] = l.global().heap.str_value(
                        l.global().heap.intern(
                            format!(
                                "error calling reader: {}",
                                String::from_utf8_lossy(e.as_ref())
                            )
                            .as_bytes(),
                        ),
                    );
                    l.top = l.base + 2;
                    return Ok(2);
                }
            };
            if r.is_empty() {
                break;
            }
            code.extend_from_slice(&r);
        }
        match crate::state::load(l, code, &chunkname) {
            Ok(v) => {
                push(l, v);
                Ok(1)
            }
            Err(msg) => {
                l.stack_ensure(l.base + 2);
                l.stack[l.base] = LuaValue::NIL;
                l.stack[l.base + 1] = l
                    .global()
                    .heap
                    .str_value(l.global().heap.intern(msg.as_bytes()));
                l.top = l.base + 2;
                Ok(2)
            }
        }
    } else {
        Err(err_bad_arg_type(
            l,
            1,
            "load",
            "string or function",
            arg(l, 0),
        ))
    }
}

fn lib_loadstring(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let code = match v.as_string() {
        Some(s) => s.as_ref().as_bytes().to_vec(),
        None => {
            return Err(err_bad_arg_type(l, 1, "loadstring", "string", arg(l, 0)));
        }
    };
    // Lua 5.1's luaL_loadbuffer: no explicit name (or a nil one) means
    // the chunk name is the source itself; luaO_chunkid handles newlines
    // and truncation.
    let default_name =
        || String::from_utf8_lossy(v.as_string().unwrap().as_ref().as_bytes()).into_owned();
    let chunkname = if nargs(l) >= 2 {
        let nv = arg(l, 1);
        if let Some(s) = nv.as_string() {
            String::from_utf8_lossy(s.as_ref().as_bytes()).into_owned()
        } else if nv.is_nil() {
            default_name()
        } else {
            "=(loadstring)".to_string()
        }
    } else {
        default_name()
    };
    let chunkname = if chunkname.starts_with('@') || chunkname.starts_with('=') {
        chunkname
    } else {
        // Lua 5.1's luaO_chunkid: `[string "..."]` — a source with a
        // newline (or longer than 45 chars) is truncated: the tail of
        // its first line (up to 45 chars) plus trailing "...".
        let len = chunkname.find(['\n', '\r']).unwrap_or(chunkname.len());
        if len == chunkname.len() && chunkname.len() <= 45 {
            format!("[string \"{}\"]", chunkname)
        } else {
            let start = len.saturating_sub(45);
            format!("[string \"{}...\"]", &chunkname[start..len])
        }
    };
    match crate::state::load(l, code, &chunkname) {
        Ok(v) => {
            push(l, v);
            Ok(1)
        }
        Err(msg) => {
            l.stack_ensure(l.base + 2);
            l.stack[l.base] = LuaValue::NIL;
            l.stack[l.base + 1] = l
                .global()
                .heap
                .str_value(l.global().heap.intern(msg.as_bytes()));
            l.top = l.base + 2;
            Ok(2)
        }
    }
}

fn lib_loadfile(l: &mut LuaState) -> LuaResult<i32> {
    let filename = match arg(l, 0).as_string() {
        Some(s) => s.as_ref().as_bytes().to_vec(),
        None => return Err(err_bad_arg_type(l, 1, "loadfile", "string", arg(l, 0))),
    };
    let chunkname = format!(
        "@{}",
        std::str::from_utf8(&filename).unwrap_or("=(loadfile)")
    );
    let path = String::from_utf8_lossy(&filename);
    match std::fs::read(path.as_ref()) {
        Ok(code) => match crate::state::load(l, code, &chunkname) {
            Ok(v) => {
                push(l, v);
                Ok(1)
            }
            Err(msg) => {
                l.stack_ensure(l.base + 2);
                l.stack[l.base] = LuaValue::NIL;
                l.stack[l.base + 1] = l
                    .global()
                    .heap
                    .str_value(l.global().heap.intern(msg.as_bytes()));
                l.top = l.base + 2;
                Ok(2)
            }
        },
        Err(e) => {
            l.stack_ensure(l.base + 2);
            l.stack[l.base] = LuaValue::NIL;
            let msg = format!("cannot open {}: {}", path, e);
            l.stack[l.base + 1] = l
                .global()
                .heap
                .str_value(l.global().heap.intern(msg.as_bytes()));
            l.top = l.base + 2;
            Ok(2)
        }
    }
}

fn lib_unpack(l: &mut LuaState) -> LuaResult<i32> {
    let t = arg(l, 0);
    let i = arg(l, 1).as_number().unwrap_or(1.0) as usize;
    let j = arg(l, 2)
        .as_number()
        .map(|n| n as usize)
        .unwrap_or(usize::MAX);
    match t.as_table() {
        Some(tab) => {
            let n = tab.as_ref().len() as usize;
            let hi = j.min(n);
            let mut out = Vec::new();
            for k in i..=hi {
                out.push(tab.as_ref().get_int(k as i32));
            }
            pushv(l, &out);
            Ok(out.len() as i32)
        }
        None => Err(err_bad_arg_type(l, 1, "unpack", "table", t)),
    }
}

fn lib_module(l: &mut LuaState) -> LuaResult<i32> {
    let name = arg(l, 0);
    let modt = l.heap().alloc_table(LuaTable::new(0, 4));
    let g = l.global();
    let _env = g.globals;
    let name_v = name;
    let k = l.heap().str_value(l.heap().intern(b"_M"));
    modt.as_mut().set(k, name_v);
    let k = l.heap().str_value(l.heap().intern(b"_NAME"));
    modt.as_mut().set(k, name_v);
    let k = l.heap().str_value(l.heap().intern(b"_PACKAGE"));
    modt.as_mut()
        .set(k, l.heap().str_value(l.heap().intern(b"")));
    push(l, LuaValue::table(modt));
    Ok(1)
}

/// `dofile([filename])` — load and run a file.
fn lib_dofile(l: &mut LuaState) -> LuaResult<i32> {
    let filename = match arg(l, 0).as_string() {
        Some(s) => s.as_ref().as_bytes().to_vec(),
        None => return Err(err_bad_arg_type(l, 1, "dofile", "string", arg(l, 0))),
    };
    let chunkname = std::str::from_utf8(&filename)
        .unwrap_or("=(dofile)")
        .to_string();
    let path = String::from_utf8_lossy(&filename);
    let code = std::fs::read(path.as_ref())
        .map_err(|e| l.runtime_error(format!("cannot open {}: {}", path, e).as_bytes()))?;
    let v = crate::state::load(l, code, &chunkname).map_err(|e| l.runtime_error(e.as_bytes()))?;
    let fs = l.top + 4;
    l.stack_ensure(fs + 4);
    l.stack[fs] = v;
    let n = crate::vm::execute(l, fs, 0, -1)?;
    l.stack_ensure(l.base + n);
    for i in 0..n {
        l.stack[l.base + i] = l.stack[fs + i];
    }
    l.top = l.base + n;
    Ok(n as i32)
}

fn lib_newproxy(l: &mut LuaState) -> LuaResult<i32> {
    let arg = arg(l, 0);
    let data: Box<[u8]> = vec![0u8; 1].into_boxed_slice();
    let mt = if arg.is_truthy() && !arg.is_bool() {
        // newproxy(proxy) — share metatable of existing proxy
        if let Some(ud) = arg.as_userdata() {
            ud.as_ref().metatable
        } else {
            None
        }
    } else if arg.is_true() {
        // newproxy(true) — create empty metatable
        let t = l.global().heap.alloc_table(LuaTable::new(0, 0));
        Some(t)
    } else {
        // newproxy() or newproxy(false) — no metatable
        None
    };
    let ud = if let Some(m) = mt {
        GcUserData::with_metatable(data, m)
    } else {
        GcUserData::new(data)
    };
    let ptr = l.global().heap.alloc_userdata(ud);
    push(l, LuaValue::userdata(ptr));
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
        .func(b"dofile", lib_dofile)
        .func(b"setmetatable", lib_setmetatable)
        .func(b"assert", lib_assert)
        .func(b"collectgarbage", lib_collectgarbage)
        .func(b"gcinfo", lib_gcinfo)
        .func(b"rawget", lib_rawget)
        .func(b"rawset", lib_rawset)
        .func(b"rawequal", lib_rawequal)
        .func(b"error", lib_error)
        .func(b"pcall", lib_pcall)
        .func(b"xpcall", lib_xpcall)
        .func(b"getmetatable", lib_getmetatable)
        .func(b"rawlen", lib_rawlen)
        .func(b"setfenv", lib_setfenv)
        .func(b"getfenv", lib_getfenv)
        .func(b"loadstring", lib_loadstring)
        .func(b"load", lib_load)
        .func(b"loadfile", lib_loadfile)
        .func(b"unpack", lib_unpack)
        .func(b"module", lib_module)
        .func(b"newproxy", lib_newproxy)
        .build();

    // The internal ipairs iterator stays off the global namespace; the
    // registry table keeps it reachable for the GC.
    let iter_sid = l.heap().intern(b"__ipairs_iter");
    let iter_key = l.heap().str_value(iter_sid);
    let iter_fn = l.global().globals.as_ref().get(iter_key);
    l.global().ipairs_iter = iter_fn;
    l.global().registry.as_mut().set(iter_key, iter_fn);
    l.global().globals.as_mut().set(iter_key, LuaValue::NIL);

    let gsid = l.heap().intern(b"_G");
    let key = l.heap().str_value(gsid);
    let g = l.global().globals;
    g.as_mut().set(key, LuaValue::table(g));

    let vsid = l.heap().intern(b"_VERSION");
    let vkey = l.heap().str_value(vsid);
    let vsid2 = l.heap().intern(b"Lua 5.1");
    g.as_mut().set(vkey, l.heap().str_value(vsid2));

    let _ = LJ_TNIL;
}
