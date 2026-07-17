//! Coroutine library: `coroutine.create`, `coroutine.resume`,
//! `coroutine.yield`, `coroutine.status`, `coroutine.wrap`,
//! `coroutine.running`, `coroutine.isyieldable`.
//!
//! Design (see lib_base.c + lj_api.c `lua_resume`/`lua_yield`):
//! * A coroutine is a pool-allocated `LuaState`. `resume` runs it on the
//!   current OS stack via a nested `execute()`; `yield` unwinds back with
//!   `LuaError::Yield`, after the VM captured the resume point in
//!   `LuaState::suspend` (our stand-in for LuaJIT's cframe unwinding).
//! * The yield-across-C-boundary rule is enforced with `c_depth`/`c_base`:
//!   yielding is legal only when no `execute()` re-entry (pcall, C
//!   callback, nested resume) sits between the resume point and the yield.

use super::{LibTarget, arg, err_bad_arg, nargs, push};
use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, GcFunc};
use crate::lual_reg;
use crate::state::{CoStatus, LuaState, StateRef, Suspend};
use crate::value::LuaValue;

/// FRAME_C marker bits for the coroutine's outermost frame link.
const FRAME_C: u64 = 1;

/// Maximum resume nesting (Rust-stack protection, LJ_MAX_CSTACK-ish).
const MAX_RESUME_DEPTH: u32 = 200;

enum Outcome {
    /// Coroutine finished; `n` results at `co.stack[0..n]`.
    Done(usize),
    /// Coroutine yielded; `n` values at `co.stack[slot..slot+n]`.
    Yielded(usize, usize),
    /// Coroutine errored; message in `co.errval`.
    Failed,
}

/// The shared resume core: validates status, moves `nargs` arguments from
/// `l.stack[args_at..]` onto `co`'s stack, runs it and classifies the
/// outcome. Maintains `cur_l` and both status fields.
fn do_resume(l: &mut LuaState, co_ref: StateRef, args_at: usize, nargs: usize) -> LuaResult<Outcome> {
    let co = co_ref.get();
    match co.status {
        CoStatus::Dead => return Err(l.runtime_error(b"cannot resume dead coroutine")),
        CoStatus::Running | CoStatus::Normal => {
            return Err(l.runtime_error(b"cannot resume non-suspended coroutine"));
        }
        CoStatus::Suspended => {}
    }
    if l.c_depth >= MAX_RESUME_DEPTH {
        return Err(l.runtime_error(b"C stack overflow"));
    }

    let saved_cur = l.global().cur_l;
    l.global().cur_l = Some(co_ref);
    l.status = CoStatus::Normal;
    co.status = CoStatus::Running;
    co.c_base = co.c_depth + 1;

    let r = match co.suspend {
        Suspend::Start => {
            // First resume: entry function at stack[0], args at [2..].
            co.stack_ensure(nargs + 16);
            co.stack[1] = LuaValue::from_bits(FRAME_C);
            for i in 0..nargs {
                co.stack[2 + i] = l.stack[args_at + i];
            }
            crate::vm::execute(co, 0, nargs, -1)
        }
        Suspend::Call { pc, cl, base, slot, want } => {
            co.suspend = Suspend::Start;
            for i in 0..nargs {
                co.stack[slot + 2 + i] = l.stack[args_at + i];
            }
            crate::vm::resume_continue(co, slot, want, nargs, pc, cl, base)
        }
        Suspend::Return { base, slot } => {
            co.suspend = Suspend::Start;
            for i in 0..nargs {
                co.stack[slot + 2 + i] = l.stack[args_at + i];
            }
            crate::vm::resume_finish(co, slot, nargs, base)
        }
    };

    // Restore scheduler state.
    l.global().cur_l = saved_cur;
    l.status = CoStatus::Running;

    match r {
        Ok(n) => {
            co.status = CoStatus::Dead;
            Ok(Outcome::Done(n))
        }
        Err(LuaError::Yield) => {
            co.status = CoStatus::Suspended;
            let ny = co.nyield as usize;
            let slot = match co.suspend {
                Suspend::Call { slot, .. } | Suspend::Return { slot, .. } => slot,
                Suspend::Start => 0,
            };
            Ok(Outcome::Yielded(slot, ny))
        }
        Err(LuaError::Runtime) => {
            co.status = CoStatus::Dead;
            Ok(Outcome::Failed)
        }
    }
}

fn lib_create(l: &mut LuaState) -> LuaResult<i32> {
    let f = arg(l, 0);
    if !f.is_func() {
        return Err(err_bad_arg(l, 1, "coroutine.create", "function", ""));
    }
    let co_ref = crate::state::new_thread(l);
    let co = co_ref.get();
    co.stack[0] = f;
    co.top = 1; // Protect the entry function from the GC's stack wipe.
    push(l, LuaValue::thread(co_ref));
    Ok(1)
}

fn lib_resume(l: &mut LuaState) -> LuaResult<i32> {
    let tv = arg(l, 0);
    let co_ref = match tv.as_thread() {
        Some(p) => p,
        None => return Err(err_bad_arg(l, 1, "coroutine.resume", "thread", "")),
    };
    let n = nargs(l).saturating_sub(1);
    let outcome = match do_resume(l, co_ref, l.base + 1, n) {
        Ok(o) => o,
        // Status validation errors surface as `false, msg` from resume.
        Err(LuaError::Runtime) => {
            let msg = l.errval;
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = msg;
            l.top = l.base + 2;
            return Ok(2);
        }
        Err(e) => return Err(e),
    };
    let co = co_ref.get();
    match outcome {
        Outcome::Done(n) => {
            // true, results...
            for i in (0..n).rev() {
                l.stack[l.base + 1 + i] = co.stack[i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            l.top = l.base + 1 + n;
            Ok((1 + n) as i32)
        }
        Outcome::Yielded(slot, n) => {
            for i in 0..n {
                l.stack[l.base + 1 + i] = co.stack[slot + i];
            }
            l.stack[l.base] = LuaValue::TRUE;
            l.top = l.base + 1 + n;
            Ok((1 + n) as i32)
        }
        Outcome::Failed => {
            l.stack[l.base] = LuaValue::FALSE;
            l.stack[l.base + 1] = co.errval;
            l.top = l.base + 2;
            Ok(2)
        }
    }
}

fn lib_yield(l: &mut LuaState) -> LuaResult<i32> {
    if l.is_main() {
        return Err(l.runtime_error(b"attempt to yield from outside a coroutine"));
    }
    if !l.is_yieldable() {
        return Err(l.runtime_error(b"attempt to yield across C-call boundary"));
    }
    l.nyield = (l.top - l.base) as u32;
    Err(LuaError::Yield)
}

fn lib_status(l: &mut LuaState) -> LuaResult<i32> {
    let tv = arg(l, 0);
    let co = match tv.as_thread() {
        Some(p) => p,
        None => return Err(err_bad_arg(l, 1, "coroutine.status", "thread", "")),
    };
    let name: &[u8] = match co.get().status {
        CoStatus::Running => b"running",
        CoStatus::Suspended => b"suspended",
        CoStatus::Normal => b"normal",
        CoStatus::Dead => b"dead",
    };
    let sid = l.heap().intern(name);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn lib_running(l: &mut LuaState) -> LuaResult<i32> {
    match l.global().cur_l {
        Some(co) if !co.get().is_main() => {
            push(l, LuaValue::thread(co));
            Ok(1)
        }
        _ => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn lib_isyieldable(l: &mut LuaState) -> LuaResult<i32> {
    push(l, LuaValue::boolean(l.is_yieldable()));
    Ok(1)
}

fn lib_wrap(l: &mut LuaState) -> LuaResult<i32> {
    let f = arg(l, 0);
    if !f.is_func() {
        return Err(err_bad_arg(l, 1, "coroutine.wrap", "function", ""));
    }
    let co_ref = crate::state::new_thread(l);
    {
        let co = co_ref.get();
        co.stack[0] = f;
        co.top = 1; // Protect the entry function from the GC's stack wipe.
    }

    let env = l.global().globals;
    let fref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: wrap_call,
        env,
        upvals: vec![LuaValue::thread(co_ref)],
    }));
    push(l, LuaValue::func(fref));
    Ok(1)
}

/// The closure returned by `coroutine.wrap`: resumes the wrapped coroutine
/// with the call arguments, returns the results directly (no `true`
/// header) and rethrows errors.
fn wrap_call(l: &mut LuaState) -> LuaResult<i32> {
    let co_ref = match l.upvalue(0).as_thread() {
        Some(p) => p,
        None => return Err(l.runtime_error(b"coroutine.wrap: thread lost")),
    };
    let n = nargs(l);
    let outcome = do_resume(l, co_ref, l.base, n)?;
    let co = co_ref.get();
    match outcome {
        Outcome::Done(n) => {
            for i in 0..n {
                l.stack[l.base + i] = co.stack[i];
            }
            l.top = l.base + n;
            Ok(n as i32)
        }
        Outcome::Yielded(slot, n) => {
            for i in 0..n {
                l.stack[l.base + i] = co.stack[slot + i];
            }
            l.top = l.base + n;
            Ok(n as i32)
        }
        Outcome::Failed => {
            l.errval = co.errval;
            Err(LuaError::Runtime)
        }
    }
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"coroutine", LibTarget::Global)
        .func(b"create", lib_create)
        .func(b"resume", lib_resume)
        .func(b"yield", lib_yield)
        .func(b"status", lib_status)
        .func(b"wrap", lib_wrap)
        .func(b"running", lib_running)
        .func(b"isyieldable", lib_isyieldable)
        .build();
}
