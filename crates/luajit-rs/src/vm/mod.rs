//! The bytecode interpreter.
//!
//! Design notes (from the discussion that shaped it):
//! * `match`-on-opcode dispatch. The hot loop state (pc, base, stack pointer,
//!   bytecode/constant pointers, multres) lives in *local variables* so the
//!   compiler keeps it in registers, mirroring LuaJIT's "VM state in
//!   registers" discipline. It is synced back to the `Interp` fields only
//!   around calls/returns (cold), never per instruction.
//! * All raw-pointer access is confined to a handful of macros (`reg!`,
//!   `setreg!`, `fetch!`, `kslot!`); the opcode arms and the public entry
//!   points are `unsafe`-free. The backing stack has fixed capacity and never
//!   reallocates, so those pointers stay valid.
//! * Register windows follow LuaJIT's FR2 layout (callee `base` =
//!   caller_base + A + 2; function at `base-2`), matching the bytecode.
//! * Errors use `LuaResult` with a fieldless error enum; the error object and
//!   yield count live on the `LuaState`. Hot paths return no `Result`.
//! * Lua->Lua calls do not recurse in Rust: `CALL` pushes a `Frame` and keeps
//!   looping; `RET` pops one. Only tail calls and re-entrant C calls recurse.

use crate::{bc::*, jit};

pub mod meta;

pub mod err;
use crate::err::{LuaError, LuaResult};
use crate::func::{GcFunc, LuaClosure, Upval};
use crate::gc::GcPtr;
use crate::jit::{HOTCOUNT_CALL, HOTCOUNT_LOOP, rec_abort_error, rec_ins, trace_exec, trace_hot};
use crate::proto::{
    KGc, PROTO_UV_IMMUTABLE, PROTO_UV_LOCAL, PROTO_VARARG, PROTO_VARARG_NEEDSARG, Proto,
};
use crate::runtime::gc::barrier_back;
use crate::runtime::meta::MM;
use crate::state::{LuaState, Suspend};
use crate::table::LuaTable;
use crate::value::*;

/// Run all pending `__gc` finalizers (`lj_gc_finalize_udata`). Called at
/// safe points while the collector is in the `Finalize` state; resets it
/// to `Pause` once the list is drained. An error raised by a finalizer
/// propagates to the caller (LuaJIT's ERRFIN behavior without a handler).
pub fn run_finalizers(l: &mut LuaState) -> LuaResult<()> {
    loop {
        let g = l.global();
        let Some(o) = g.heap.mmudata.pop() else {
            // The cycle's finalizer stage is done: this is the true end of
            // the GC cycle. Reset the per-cycle debt counters here as well
            // as when the sweep finishes with no pending finalizers, or a
            // permanent finalizer chain (newproxy self-recycling) would
            // keep mmudata non-empty forever and `table_extra` would never
            // be cleared, inflating gcinfo()/GC debt without bound.
            g.heap.gc_state = crate::runtime::gc::GcState::Pause;
            g.heap.table_extra = 0;
            g.heap.debt = 0;
            return Ok(());
        };
        // The finalizer ran: the object is now dead for the *next* cycle.
        let mo = crate::meta::meta_lookup(g, o.value(), MM::Gc);
        o.mark_finalized(g.heap.current_white);
        if let crate::runtime::gc::Finalizable::CLib(cd) = &o {
            // A dead CLibrary cdata releases its library reference directly
            // (no Lua __gc; the library is dlclosed at refcount zero).
            let idx = u64::from_le_bytes(cd.as_ref().data[..8].try_into().unwrap()) as usize;
            crate::ffi::clib::gc_release(g, idx);
            continue;
        }
        if mo.is_nil() {
            continue;
        }
        let saved_top = l.top;
        let saved_base = l.base;
        let saved_frame_top = l.frame_top;
        // The finalizer frame must sit above the interpreter's live
        // frame area: `top` lags below the frame (framesize slots), and
        // the finalizer's result would overwrite the registers the
        // dispatch loop is about to use (same rule as call_hook).
        let base_slot = l.frame_top.max(l.top);
        l.stack_ensure(base_slot + 3 + STACK_SAFETY);
        l.stack[base_slot] = mo;
        l.stack[base_slot + 1] = LuaValue::NIL;
        l.stack[base_slot + 2] = o.value();
        match execute(l, base_slot, 1, -1) {
            Ok(_) => {
                l.top = saved_top;
                l.frame_top = saved_frame_top;
            }
            Err(e) => {
                // LuaJIT lj_gc_finalize rethrows the finalizer error to
                // the allocation/collectgarbage caller (`lj_err_run`);
                // gc.lua's `assert(not pcall(collectgarbage))` depends on
                // it. The object is already finalized.
                l.top = saved_top;
                l.frame_top = saved_frame_top;
                l.base = saved_base;
                return Err(e);
            }
        }
        // The finalizer's argument sits above the live frame and would
        // otherwise be marked (and keep the userdata, its metatable and
        // everything reachable through them alive) every cycle, delaying
        // the two-cycle weak-table removal indefinitely.
        l.stack[base_slot] = LuaValue::NIL;
        l.stack[base_slot + 1] = LuaValue::NIL;
        l.stack[base_slot + 2] = LuaValue::NIL;
        // execute runs a Lua frame through Interp, which mutates
        // l.base; restore it so a C function that triggered the
        // finalizer (collectgarbage) still pushes results correctly.
        l.base = saved_base;
    }
}

/// Invoke the debug hook (`debug.sethook`) with `(event[, line])`.
/// The hook runs on a scratch area above `top`; the stack is restored
/// afterwards. Hooks don't re-enter themselves. Cold: only ever reached
/// when a hook is installed.
#[cold]
pub fn call_hook(l: &mut LuaState, event: &str, line: Option<i32>) -> LuaResult<()> {
    if l.hook.is_nil() || l.hook_active {
        return Ok(());
    }
    let f = l.hook;
    let saved_base = l.base;
    let saved_top = l.top;
    let saved_active = l.hook_active;
    l.hook_active = true;
    let ev = l.heap().str_value(l.heap().intern(event.as_bytes()));
    // Place the hook frame above the interpreter's live frame area:
    // l.top may lag below the frame (framesize slots), and the hook must
    // not overwrite the registers the dispatch loop is about to use.
    let base_slot = l.frame_top.max(l.top);
    l.stack_ensure(base_slot + 4 + STACK_SAFETY);
    l.stack[base_slot] = f;
    l.stack[base_slot + 1] = LuaValue::NIL;
    l.stack[base_slot + 2] = ev;
    let nargs = if let Some(ln) = line {
        l.stack[base_slot + 3] = LuaValue::number(ln as f64);
        2
    } else {
        1
    };
    // Chain the hook frame to the interpreter frame (base-encoded link)
    // so debug.getinfo / getlocal from inside the hook can walk back.
    let link = ((saved_base as u64) << 3) | FRAME_LUA;
    let r = execute_link(l, base_slot, nargs, -1, link);
    l.base = saved_base;
    l.top = saved_top;
    l.hook_active = saved_active;
    r.map(|_| ())
}

/// Hook mask bits (debug.sethook mask string).
pub const HOOKMASK_LINE: u8 = 0x01;
pub const HOOKMASK_CALL: u8 = 0x02;
pub const HOOKMASK_RET: u8 = 0x04;
pub const HOOKMASK_COUNT: u8 = 0x08;

/// Check line/count hooks at an instruction boundary. Returns `true` if a
/// hook ran (the caller must resync its base/ip pointers). The hookmask
/// test itself is inline in the dispatch (two field loads); everything
/// beyond it is cold.
#[cold]
pub fn hook_check(l: &mut LuaState, line: u32) -> LuaResult<bool> {
    let mask = l.hookmask;
    if mask == 0 || l.hook_active {
        return Ok(false);
    }
    let mut ran = false;
    if mask & HOOKMASK_COUNT != 0 {
        l.hookcount -= 1;
        if l.hookcount <= 0 {
            l.hookcount = l.hook_count_reset;
            call_hook(l, "count", None)?;
            ran = true;
        }
    }
    if mask & HOOKMASK_LINE != 0 && line != 0 && line != l.hook_line {
        l.hook_line = line;
        call_hook(l, "line", Some(line as i32))?;
        ran = true;
    }
    Ok(ran)
}

/// Frame type markers kept in the low bits of the frame-link slot at
/// `base-1`, exactly as in LuaJIT's `lj_frame.h` (FR2 layout):
///
/// ```text
///        base-2  base-1      |  base  base+1 ...
///       [func   PC/delta/ft] | [slots ...]
///       ^-- frame            | ^-- base    ^-- top
/// ```
///
/// * `FRAME_LUA`: the link is the caller's return PC (a 4-aligned pointer,
///   low bits 00). Caller base, wanted results and the result slot are all
///   recovered from the CALL instruction at `pc[-1]`.
/// * `FRAME_C`: a host (`execute`) entry; bits 3.. hold `want + 1`.
/// * `FRAME_VARG`: bits 3.. hold the slot delta back to the frame that
///   carries the real link; varargs live between the two frames.
const FRAME_LUA: u64 = 0;
const FRAME_C: u64 = 1;
pub const FRAME_VARG: u64 = 3;
const FRAME_CONT: u64 = 2;
pub const FRAME_TYPE_MASK: u64 = 3;

/// Continuation IDs stored in the cont-slot of a FRAME_CONT frame.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cont {
    Ra = 0,
    Nop = 1,
    Condt = 2,
    Condf = 3,
}
impl Cont {
    pub fn encode(self, extra: u32) -> u64 {
        ((self as u64) << 32) | (extra as u64)
    }
    pub fn decode(bits: u64) -> (Cont, u32) {
        (
            unsafe { std::mem::transmute::<u8, Cont>((bits >> 32) as u8) },
            (bits & 0xFFFF_FFFF) as u32,
        )
    }
}

/// Call a value with the given arguments and collect all results.
/// The host entry point into the VM.
pub fn call(l: &mut LuaState, func: LuaValue, args: &[LuaValue]) -> LuaResult<Vec<LuaValue>> {
    l.stack_ensure(args.len() + STACK_SAFETY);
    l.top = 0;
    l.stack[0] = func;
    l.stack[1] = LuaValue::NIL;
    for (i, &a) in args.iter().enumerate() {
        l.stack[2 + i] = a;
    }
    let n = execute(l, 0, args.len(), -1)?;
    Ok((0..n).map(|i| l.stack[i]).collect())
}

/// Execute a call to the function at `func_slot` with `nargs` arguments
/// already placed at `func_slot + 2 ..`. Leaves the results at `func_slot`
/// and returns their count.
pub fn execute(l: &mut LuaState, func_slot: usize, nargs: usize, want: i32) -> LuaResult<usize> {
    l.c_depth += 1;
    l.stack_ensure(func_slot + nargs + STACK_SAFETY);
    let r = execute_inner(l, func_slot, nargs, want);
    l.c_depth -= 1;
    r
}

/// Like `execute`, but without counting as a C frame for yieldability:
/// the protected calls (pcall/xpcall) may be yielded through.
pub fn execute_yieldable(
    l: &mut LuaState,
    func_slot: usize,
    nargs: usize,
    want: i32,
) -> LuaResult<usize> {
    l.stack_ensure(func_slot + nargs + STACK_SAFETY);
    execute_inner(l, func_slot, nargs, want)
}

/// Like `execute`, with an explicit frame link for the callee frame
/// (xpcall's message handler chains to the failed call's frame).
pub fn execute_link(
    l: &mut LuaState,
    func_slot: usize,
    nargs: usize,
    want: i32,
    link: u64,
) -> LuaResult<usize> {
    l.c_depth += 1;
    l.stack_ensure(func_slot + nargs + STACK_SAFETY);
    let r = execute_inner_link(l, func_slot, nargs, want, Some(link));
    l.c_depth -= 1;
    r
}

/// Safety margin added to every stack_ensure: protects against a few
/// extra slots written by CALL/VARG/TSETM frame setup.
const STACK_SAFETY: usize = 64;

/// Check if a value is a NULL cdata (void pointer with null address).
fn is_cdata_null(v: LuaValue) -> bool {
    if let Some(cd) = v.as_cdata() {
        let c = cd.as_ref();
        if c.ctypeid != crate::ffi::CTypeID::PVoid as u32 {
            return false;
        }
        c.data.iter().all(|&b| b == 0)
    } else {
        false
    }
}
#[inline(always)]
fn num2bit(n: f64) -> i32 {
    (n as i64) as u32 as i32
}

fn cdata_u64(l: &LuaState, v: LuaValue) -> Option<u64> {
    if let Some(cd) = v.as_cdata() {
        let cd = cd.as_ref();
        // Only numeric cdata (int/uint/float/enum) act as raw values;
        // structs, arrays and pointers go through the metamethods.
        let numeric = l.global().cts.as_ref().is_none_or(|cts| {
            let raw = cts.raw(cd.ctypeid);
            crate::ffi::ctype_isnum(raw.info)
        });
        if numeric && cd.data.len() <= 8 {
            let mut buf = [0u8; 8];
            buf[..cd.data.len()].copy_from_slice(&cd.data);
            Some(u64::from_le_bytes(buf))
        } else {
            None
        }
    } else {
        None
    }
}

/// Numeric cdata (int/uint/float/enum) usable as a raw value in arithmetic
/// and bitwise ops. Structs, arrays and pointers must go through the
/// metamethod path (lj_carith / lj_meta_arith).
fn cdata_is_numeric(l: &LuaState, v: LuaValue) -> bool {
    if let Some(cd) = v.as_cdata() {
        l.global().cts.as_ref().is_none_or(|cts| {
            let raw = cts.raw(cd.as_ref().ctypeid);
            crate::ffi::ctype_isnum(raw.info)
        })
    } else {
        false
    }
}

/// The storage address of a pointer/array cdata: `(root, byte offset)`.
/// Non-pointer cdata yields `None` (its numeric value, not an address).
fn cdata_ptr_addr(l: &LuaState, v: LuaValue) -> Option<(u64, i64)> {
    let cd = v.as_cdata()?;
    let cts = l.global().cts.as_ref()?;
    let raw = cts.raw(cd.as_ref().ctypeid);
    if !crate::ffi::ctype_isarray(raw.info) && !crate::ffi::ctype_ispointer(raw.info) {
        return None;
    }
    let (off, root) = crate::runtime::cdata::resolve_ptr(cd);
    Some((root.addr(), off))
}

/// Pointer cdata comparisons and equality compare storage addresses.
fn cdata_ptr_eq(l: &LuaState, a: LuaValue, b: LuaValue) -> Option<bool> {
    Some(cdata_ptr_addr(l, a)? == cdata_ptr_addr(l, b)?)
}

fn make_cdata_result(l: &mut LuaState, bits: u64, is_ull: bool) -> LuaValue {
    let ctypeid = if is_ull {
        crate::ffi::CTypeID::UInt64 as u32
    } else {
        crate::ffi::CTypeID::Int64 as u32
    };
    let mut cd = crate::runtime::cdata::CData::new(ctypeid, 8);
    cd.data[..8].copy_from_slice(&bits.to_le_bytes());
    let p = l.global().heap.alloc_cdata(cd);
    LuaValue::cdata(p)
}

fn cdata_is_ull(v: LuaValue) -> bool {
    if let Some(cd) = v.as_cdata() {
        cd.as_ref().ctypeid == crate::ffi::CTypeID::UInt64 as u32
    } else {
        false
    }
}

/// Unsigned 64-bit comparison by opcode ordinal (ISLT=0, ISGE=1, ISLE=2,
/// ISGT=3). `op` 1/3 are the unordered negations (`!` of the ordered one).
#[inline(always)]
fn cmp_u(op: u32, x: u64, y: u64) -> bool {
    match op {
        0 => x < y,
        1 => !(x < y),
        2 => x <= y,
        3 => !(x <= y),
        _ => unreachable!(),
    }
}

/// Signed 64-bit comparison, same opcode encoding as `cmp_u`.
#[inline(always)]
fn cmp_s(op: u32, x: u64, y: u64) -> bool {
    let x = x as i64;
    let y = y as i64;
    match op {
        0 => x < y,
        1 => !(x < y),
        2 => x <= y,
        3 => !(x <= y),
        _ => unreachable!(),
    }
}

fn try_cdata_binop(
    l: &mut LuaState,
    xv: LuaValue,
    yv: LuaValue,
    op: impl Fn(u64, u64) -> u64,
) -> Option<LuaValue> {
    // Only numeric cdata participate in raw-value arithmetic.
    if (xv.is_cdata() && !cdata_is_numeric(l, xv)) || (yv.is_cdata() && !cdata_is_numeric(l, yv)) {
        return None;
    }
    let x_cd = cdata_u64(l, xv);
    let y_cd = cdata_u64(l, yv);
    match (x_cd, y_cd) {
        (Some(x), Some(y)) => {
            let is_ull = cdata_is_ull(xv) || cdata_is_ull(yv);
            Some(make_cdata_result(l, op(x, y), is_ull))
        }
        (Some(x), None) if yv.is_number() => {
            let is_ull = cdata_is_ull(xv);
            let y = (yv.num() as i64) as u64;
            Some(make_cdata_result(l, op(x, y), is_ull))
        }
        (None, Some(y)) if xv.is_number() => {
            let is_ull = cdata_is_ull(yv);
            let x = (xv.num() as i64) as u64;
            Some(make_cdata_result(l, op(x, y), is_ull))
        }
        _ => None,
    }
}

fn execute_inner(l: &mut LuaState, func_slot: usize, nargs: usize, want: i32) -> LuaResult<usize> {
    execute_inner_link(l, func_slot, nargs, want, None)
}

/// Like `execute_inner`, but with an explicit frame link for the callee
/// (used by xpcall's message handler to chain its frame to the failed
/// call's frame so debug walks can see it).
fn execute_inner_link(
    l: &mut LuaState,
    func_slot: usize,
    nargs: usize,
    want: i32,
    link: Option<u64>,
) -> LuaResult<usize> {
    let mut nargs = nargs;
    let f = l.stack[func_slot];
    let gf = match f.as_func() {
        Some(p) => p,
        None => {
            // lj_meta_call: try __call metamethod.
            nargs = meta::meta_call(l, func_slot, nargs)?;
            let f = l.stack[func_slot];
            f.as_func().expect("__call did not produce a function")
        }
    };
    if let GcFunc::C(cc) = gf.as_ref() {
        return call_c(l, cc.f, func_slot, nargs, want);
    }
    let saved_base = l.base;
    let saved_top = l.top;
    let saved_frame_top = l.frame_top;
    let mut vm = Interp::new(l);
    let link = link.unwrap_or_else(|| (((want + 1) as u64) << 3) | FRAME_C);
    vm.enter_lua(gf, func_slot, nargs, link);
    let r = vm.run();
    // The interpreter mutates l.base/l.top (and l.frame_top) for the
    // callee frame; restore them so the calling C function's results land
    // at `func_slot` (call_c's args_base convention). Leaving `frame_top`
    // at the callee's extent would make later incremental GC marks and
    // finalizer placement (base_slot = max(top, frame_top)) overshoot the
    // true frame, so live locals above the real frame get collected.
    //
    // A yield (coroutine suspend) must NOT restore: the suspended frame's
    // base/top/frame_top are the resume point, and clearing them to the
    // entry values (base=0/top=1) lets the next GC wipe the suspended
    // frame's locals (clear_from = max(top, frame_top) = 1).
    if !matches!(r, Err(LuaError::Yield)) {
        l.base = saved_base;
        l.top = saved_top;
        l.frame_top = saved_frame_top;
    }
    r
}

/// Call a C function with `nargs` args at `func_slot + 2`, moving up to
/// `want` results back to `func_slot`.
fn call_c(
    l: &mut LuaState,
    f: crate::func::CFunction,
    func_slot: usize,
    nargs: usize,
    want: i32,
) -> LuaResult<usize> {
    let args_base = func_slot + 2;
    // Ensure the stack can hold the args and results before touching it.
    l.stack_ensure(args_base + nargs + 8);
    let saved_base = l.base;
    let saved_top = l.top;
    let saved_frame_top = l.frame_top;
    // Set a frame link so error walkers can find the caller.
    l.stack[args_base - 1] = LuaValue::from_bits(((saved_base as u64) << 3) | FRAME_LUA);
    l.base = args_base;
    l.top = args_base + nargs;
    let (paused, collect) = {
        let g = l.global();
        (
            g.heap.gc_state == crate::runtime::gc::GcState::Pause,
            g.heap.should_collect() && !g.heap.gc_stopped,
        )
    };
    if collect {
        // lj_gc_step_fixtop: the collector may clear stack slots above
        // `top`; the caller's frame locals sit below frame_top, so raise
        // top past them for the duration of the step.
        l.top = l.top.max(l.frame_top);
        if paused {
            crate::gc::start_gc_cycle(l.global());
        }
        crate::gc::gc_step(&mut l.global().heap, l.global().heap.gc_step_size);
        l.top = args_base + nargs;
    }
    let r = f(l);
    let n = match r {
        Ok(nv) => nv as usize,
        Err(LuaError::Yield) => {
            // The C callee yielded (coroutine.yield through a C caller
            // like pcall): capture the resume point in the calling Lua
            // frame, like suspend_call.
            let ny = l.nyield as usize;
            for i in 0..ny {
                l.stack[func_slot + i] = l.stack[args_base + i];
            }
            let pc = l.debug_pc;
            let cl = l
                .stack
                .get(func_slot.saturating_sub(2))
                .and_then(|v| v.as_func())
                .unwrap_or(l.stack[0].as_func().unwrap());
            let a = match cl.as_ref() {
                crate::func::GcFunc::Lua(c) => {
                    let pt = c.proto.as_ref();
                    if pc >= 1 && pc - 1 < pt.bc.len() {
                        crate::bc::bc_a(pt.bc[pc - 1]) as usize
                    } else {
                        0
                    }
                }
                _ => 0,
            };
            let base = func_slot.saturating_sub(a);
            let (protected, value_slot) = match l.suspend {
                crate::state::Suspend::Call { protected: p, .. } => (p, func_slot),
                _ => (false, func_slot),
            };
            l.suspend = crate::state::Suspend::Call {
                pc,
                cl,
                base,
                slot: func_slot,
                want,
                protected,
                value_slot,
            };
            l.top = (l.base + 8).max(func_slot + ny);
            l.base = saved_base;
            l.top = saved_top;
            l.frame_top = saved_frame_top;
            return Err(LuaError::Yield);
        }
        Err(e) => {
            l.base = saved_base;
            l.top = saved_top;
            l.frame_top = saved_frame_top;
            return Err(e);
        }
    };
    l.stack_ensure((func_slot + n + 8).max(args_base + nargs + 8));
    for i in 0..n {
        l.stack[func_slot + i] = l.stack[args_base + i];
    }
    l.base = saved_base;
    l.top = saved_top;
    l.frame_top = saved_frame_top;
    if want >= 0 {
        for i in n..(want as usize) {
            l.stack[func_slot + i] = LuaValue::NIL;
        }
        l.top = func_slot + want as usize;
        Ok(want as usize)
    } else {
        l.top = func_slot + n;
        Ok(n)
    }
}

/// Dispatch-loop exit reasons: a host-frame return, or a switch between
/// the plain and the recording interpreter (LuaJIT switches dispatch
/// tables instead).
enum Flow {
    Ret(usize),
    Rec,
}

/// Result of the inlined `call_lua_fast` frame switch (shared by BC_CALL
/// and BC_CALLM).
enum CallFast {
    /// Fast path applied: keep dispatching in the callee frame.
    Applied,
    /// Fast path applied, but the dispatcher must return (trace enter or a
    /// hot call started recording).
    Trace(Flow),
    /// Callee is not a plain fixed-arity Lua function: use `do_call`.
    Slow,
}

/// The interpreter's register window over the Lua stack: `(sp, base)`
/// standing for LuaJIT's BASE pointer (`sp + base`). A value type, so the
/// dispatch loop keeps it in registers; recreate it from the interpreter
/// fields after any `stack_ensure` (the stack Vec may reallocate, which
/// is also why it cannot be a borrow).
///
/// All raw-pointer access to the stack is confined to this type — the
/// opcode arms and the rest of the VM never touch `sp` directly. `reg`/
/// `set` are relative to the frame base (bytecode registers), `at_abs`/
/// `set_abs` are absolute stack indices.
#[derive(Clone, Copy)]
struct Frame {
    sp: *mut LuaValue,
    /// The register window: `sp + base`. `reg`/`set` index off this single
    /// pointer, so the hot arms keep the window in a CPU register and every
    /// bytecode-register access is one indexed load/store (LuaJIT's BASE).
    window: *mut LuaValue,
}

impl Frame {
    #[inline(always)]
    fn new(sp: *mut LuaValue, base: usize) -> Frame {
        Frame {
            sp,
            window: unsafe { sp.add(base) },
        }
    }

    /// Read bytecode register `i` (relative to the frame base).
    #[inline(always)]
    fn reg(self, i: u32) -> LuaValue {
        unsafe { *self.window.add(i as usize) }
    }

    /// Write bytecode register `i`.
    #[inline(always)]
    fn set(self, i: u32, v: LuaValue) {
        unsafe { *self.window.add(i as usize) = v }
    }

    /// The current frame base as an absolute stack index (LuaJIT's BASE).
    #[inline(always)]
    fn cur_base(self) -> usize {
        unsafe { self.window.offset_from(self.sp) as usize }
    }

    /// Write the stack slot at absolute index `abs`.
    #[inline(always)]
    fn set_abs(self, abs: usize, v: LuaValue) {
        unsafe { *self.sp.add(abs) = v }
    }

    /// The frame-base pointer (LuaJIT's BASE). Only valid as long as the
    /// window is not moved; use `at_abs`/`set_abs` otherwise.
    #[inline(always)]
    fn bp(self) -> *mut LuaValue {
        self.window
    }

    /// The frame-link word at `base - 1`.
    #[inline(always)]
    fn frame_link(self) -> u64 {
        unsafe { *self.window.sub(1) }.to_bits()
    }

    /// Write the callee slot at `base - 2` (FR2 func slot).
    #[inline(always)]
    fn set_func(self, v: LuaValue) {
        unsafe { *self.window.sub(2) = v }
    }
}

/// Interpreter context. The per-instruction hot state is *not* kept here; it
/// lives in locals inside `run` and is synced to these fields only around
/// calls and returns.
struct Interp {
    l: *mut LuaState,
    sp: *mut LuaValue,
    base: usize,
    pc: usize,
    cl: GcPtr<GcFunc>,
    bcp: *const BCIns,
    knp: *const f64,
    ksp: *const LuaValue,
    multres: usize,
    /// Saved `hook_line` values for pending Lua frames: pushed on call,
    /// popped on return, so a line hook doesn't fire on the caller's
    /// resumption line (Lua 5.1 semantics).
    hook_lines: Vec<u32>,
}

impl Interp {
    fn new(l: &mut LuaState) -> Interp {
        let sp = l.stack.as_mut_ptr();
        Interp {
            l,
            sp,
            base: 0,
            pc: 0,
            cl: GcPtr::from_addr(0x100).unwrap(), // placeholder; set by enter_lua
            bcp: std::ptr::null(),
            knp: std::ptr::null(),
            ksp: std::ptr::null(),
            multres: 0,
            hook_lines: Vec::new(),
        }
    }

    /// The interpreter has exclusive access to the `LuaState` for the whole
    /// `run` invocation; the raw pointer (and this deliberate `&self ->
    /// &mut` escape hatch) exists so borrows of `Interp` fields and the
    /// state can overlap without fighting the borrow checker.
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    fn l(&self) -> &mut LuaState {
        unsafe { &mut *self.l }
    }

    #[inline(always)]
    fn lua_cl(&self) -> &LuaClosure {
        match self.cl.as_ref() {
            GcFunc::Lua(c) => c,
            GcFunc::C(_) => unreachable!("C function in Lua frame"),
        }
    }

    #[inline(always)]
    fn proto(&self) -> &Proto {
        self.lua_cl().proto.as_ref()
    }

    #[inline(always)]
    fn at(&self, abs: usize) -> LuaValue {
        unsafe { *self.sp.add(abs) }
    }

    #[inline(always)]
    fn set_at(&self, abs: usize, v: LuaValue) {
        unsafe { *self.sp.add(abs) = v }
    }

    /// A string constant, read from the resolved KBASE table (`ksp`) — the
    /// value is precomputed at registration, so this is a single load, no
    /// interner lookup. Only valid for KSTR/GGET/GSET/TGETS/TSETS operands.
    #[inline(always)]
    fn kstr_at(&self, d: u32) -> LuaValue {
        let v = unsafe { *self.ksp.add(d as usize) };
        debug_assert!(v.is_string());
        v
    }

    /// A number constant (`kslot!`): raw f64 load from the resolved KBASE
    /// table, no NaN-boxing canonicalization.
    #[inline(always)]
    fn knum(&self, d: u32) -> LuaValue {
        LuaValue::number_raw(unsafe { *self.knp.add(d as usize) })
    }

    /// Compute the stack slots needed for a Lua call frame at `func_slot`.
    fn enter_lua_need(&self, gf: GcPtr<GcFunc>, func_slot: usize, nargs: usize) -> usize {
        let cl = match gf.as_ref() {
            GcFunc::Lua(c) => c,
            _ => unreachable!(),
        };
        let pt = cl.proto.as_ref();
        let callbase = func_slot + 2;
        if (pt.flags & PROTO_VARARG) != 0 {
            (callbase + nargs + 2) + pt.numparams as usize + pt.framesize as usize + 16
        } else {
            callbase + pt.framesize as usize + 16
        }
    }

    /// Set up a Lua frame for the function at `func_slot` and switch the
    /// `Interp` fields to the callee. `link` is stored in the frame-link
    /// slot (`callbase - 1`); the caller must have synced its locals into
    /// the fields first. Mirrors LuaJIT's `ins_call` + FUNCF/FUNCV headers.
    fn enter_lua(&mut self, gf: GcPtr<GcFunc>, func_slot: usize, nargs: usize, link: u64) {
        let need = self.enter_lua_need(gf, func_slot, nargs);
        let l = self.l();
        l.stack_ensure(need);
        self.sp = l.stack.as_mut_ptr();
        self.enter_lua_sans_ensure(gf, func_slot, nargs, link);
    }

    /// Core of `enter_lua` without the `stack_ensure` — the caller has
    /// already ensured enough space (possibly combining with mmcall data).
    fn enter_lua_sans_ensure(
        &mut self,
        gf: GcPtr<GcFunc>,
        func_slot: usize,
        nargs: usize,
        link: u64,
    ) {
        let cl = match gf.as_ref() {
            GcFunc::Lua(c) => c,
            _ => unreachable!(),
        };
        let pt = cl.proto.as_ref();
        let numparams = pt.numparams as usize;
        let callbase = func_slot + 2;

        self.set_at(callbase - 1, LuaValue::from_bits(link));

        if (pt.flags & PROTO_VARARG) != 0 {
            // FUNCV: shift the fixed params up past the varargs and chain a
            // vararg frame back to the one holding the real link.
            let newbase = callbase + nargs + 2;
            self.set_at(newbase - 2, LuaValue::func(gf));
            for i in 0..numparams {
                let v = if i < nargs {
                    self.at(callbase + i)
                } else {
                    LuaValue::NIL
                };
                self.set_at(newbase + i, v);
            }
            let delta = (newbase - callbase) as u64;
            self.set_at(newbase - 1, LuaValue::from_bits((delta << 3) | FRAME_VARG));
            self.base = newbase;
            // Set TOP to the new frame before anything below may run GC
            // (alloc_table for the implicit `arg` table steps the collector,
            // which clears slots above TOP — the just-copied params would be
            // wiped while TOP still pointed at the caller's frame).
            self.l().top = newbase + pt.framesize as usize;
            self.l().frame_top = self.l().top;
            // Lua 5.1 LUA_COMPAT_VARARG: build the implicit `arg` local
            // ({varargs..., n = count}) unless the body uses `...` itself.
            if (pt.flags & PROTO_VARARG_NEEDSARG) != 0 {
                let l = self.l();
                let nvar = nargs.saturating_sub(numparams);
                let tab = l
                    .heap()
                    .alloc_table(crate::table::LuaTable::new(nvar as u32, 1));
                for i in 0..nvar {
                    tab.as_mut()
                        .set_int(i as i32 + 1, self.at(callbase + numparams + i));
                }
                let nsid = l.heap().intern(b"n");
                tab.as_mut()
                    .set(l.heap().str_value(nsid), LuaValue::number(nvar as f64));
                self.set_at(newbase + numparams, LuaValue::table(tab));
            } else {
                self.set_at(newbase + numparams, LuaValue::NIL);
            }
        } else {
            for i in nargs..numparams {
                self.set_at(callbase + i, LuaValue::NIL);
            }
            self.base = callbase;
        }

        self.cl = gf;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
        self.ksp = pt.kstrv.as_ptr();
        self.pc = 1; // skip the FUNCF/FUNCV header
        self.l().top = self.base + pt.framesize as usize;
        self.l().frame_top = self.l().top;
    }

    fn reload(&mut self, cl: GcPtr<GcFunc>) {
        let pt = match cl.as_ref() {
            GcFunc::Lua(c) => c.proto.as_ref(),
            _ => unreachable!(),
        };
        self.cl = cl;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
        self.ksp = pt.kstrv.as_ptr();
    }

    /// Generalised fast-return: walk the VARG chain inline and return the
    /// real frame's base offset, along with `(want, ret_ip, caller_a)`.
    #[inline(always)]
    fn ret_fast_n(&self, mut bp: *const LuaValue) -> Option<(usize, i32, *const BCIns, i32)> {
        let mut link = unsafe { (*bp.sub(1)).to_bits() };
        while (link & FRAME_TYPE_MASK) == FRAME_VARG {
            let sz = (link >> 3) as usize;
            if sz == 0 {
                break;
            }
            bp = unsafe { bp.sub(sz) };
            link = unsafe { (*bp.sub(1)).to_bits() };
        }
        // A FRAME_LUA link is the caller's return PC only when it is a real
        // code address; the size test separates it from the caller-base
        // encoding fabricated by `call_c` and `call_hook` (whose frames
        // carry `(saved_base << 3) | FRAME_LUA`, small enough to index the
        // stack). Treating an encoded link as a return PC would jump into a
        // tiny bogus address.
        if (link & FRAME_TYPE_MASK) == FRAME_LUA
            && ((link >> 3) as usize) >= self.l().stack.len()
            && self.l().openuv.is_empty()
        {
            let ret_ip = link as *const BCIns;
            let call_ins = unsafe { *ret_ip.sub(1) };
            let want = bc_b(call_ins) as i32 - 1;
            let caller_a = bc_a(call_ins) as i32;
            let base = unsafe { bp.offset_from(self.sp) } as usize;
            Some((base, want, ret_ip, caller_a))
        } else {
            None
        }
    }

    /// Whether a return from the frame at `bp` may take the inline fast
    /// path: a plain Lua frame link and no open upvalues to close. Returns
    /// the caller's wanted result count (from the CALL instruction's B).
    /// Reload the interpreter for the closure owning the frame at `fr`'s
    /// base.
    #[inline(always)]
    fn reload_at(&mut self, fr: Frame) {
        let cl = unsafe { *fr.bp().sub(2) }.as_func().unwrap();
        self.reload(cl);
    }

    /// Run a "call"/"return" hook event from a dispatch arm (cold). The
    /// caller must have synced the interpreter locals (`sync!`); the hook
    /// may grow the stack, so `sp` is refreshed for the caller's
    /// `resync!`.
    #[cold]
    fn hook_event(&mut self, event: &str) -> LuaResult<()> {
        call_hook(self.l(), event, None)?;
        self.sp = self.l().stack.as_mut_ptr();
        Ok(())
    }

    /// `mmcall` + FRAME_CONT (lj_meta.c's `mmcall` + `vm_call_dispatch_f`):
    /// set up a continuation frame above the current one and enter the Lua
    /// metamethod. Caller must be synced; afterwards the `Interp` fields
    /// point into the metamethod's frame (caller must `resync!()`).
    ///
    /// Stack layout (FR2):
    /// ```text
    ///  mmbase-4  mmbase-3   mmbase-2  mmbase-1  | mmbase  mmbase+1 ..
    /// [cont      saved PC] [mo        link|CONT] | [arg0   arg1 ..]
    /// ```
    fn mmcall_cont(&mut self, cont_id: Cont, extra: u32, mo: LuaValue, args: &[LuaValue]) {
        let saved_base = self.base;
        // curr_topL: scratch right above the running frame, +2 for cont/PC.
        let func_slot = saved_base + self.proto().framesize as usize + 2;
        let mmbase = func_slot + 2;
        {
            // Combine mmcall overhead with the Lua frame's need so we only
            // grow the stack once (avoid the double ensure in enter_lua).
            let gf = mo.as_func().unwrap();
            let mm_need = mmbase + args.len() + 16;
            let lua_need = self.enter_lua_need(gf, func_slot, args.len());
            let need = mm_need.max(lua_need);
            let l = self.l();
            l.stack_ensure(need);
            self.sp = l.stack.as_mut_ptr();
        }
        self.set_at(mmbase - 4, LuaValue::from_bits(cont_id.encode(extra)));
        self.set_at(mmbase - 3, LuaValue::from_bits(self.pc as u64));
        self.set_at(func_slot, mo);
        for (i, &v) in args.iter().enumerate() {
            self.set_at(mmbase + i, v);
        }
        let link = (((mmbase - saved_base) as u64) << 3) | FRAME_CONT;
        self.enter_lua_sans_ensure(mo.as_func().unwrap(), func_slot, args.len(), link);
    }

    /// Call a C-function metamethod inline (no continuation frame): the
    /// result is available immediately. Uses scratch above the frame.
    fn call_c_fn(
        &mut self,
        f: crate::func::CFunction,
        mo: LuaValue,
        args: &[LuaValue],
    ) -> LuaResult<LuaValue> {
        let fs = self.base + self.proto().framesize as usize;
        {
            let need = fs + 2 + args.len() + 8;
            let l = self.l();
            l.stack_ensure(need);
            self.sp = l.stack.as_mut_ptr();
        }
        self.set_at(fs, mo);
        self.set_at(fs + 1, LuaValue::NIL);
        for (i, &v) in args.iter().enumerate() {
            self.set_at(fs + 2 + i, v);
        }
        let n = self.call_c_inline(f, fs, args.len())?;
        self.sp = self.l().stack.as_mut_ptr();
        Ok(if n > 0 { self.at(fs) } else { LuaValue::NIL })
    }

    /// Continuation dispatch (LuaJIT's `->cont_dispatch` + `cont_*`
    /// handlers): a metamethod called through `mmcall_cont` has returned;
    /// `mmbase` is its frame base, results were copied to `mmbase - 2` by
    /// `do_return`. Restores the caller frame and applies the continuation.
    /// Returns `None`: execution always continues in the caller.
    fn cont_dispatch(&mut self, mmbase: usize, link: u64, n: usize) -> Option<usize> {
        let delta = (link >> 3) as usize;
        if delta == 0 || delta > mmbase {
            return None;
        }
        let caller_base = mmbase - delta;
        let (cont, extra) = Cont::decode(self.at(mmbase - 4).to_bits());
        let saved_pc = self.at(mmbase - 3).to_bits() as usize;
        // Ensure one valid result (cont_dispatch's "Ensure one valid arg").
        let result = if n > 0 {
            self.at(mmbase - 2)
        } else {
            LuaValue::NIL
        };

        self.base = caller_base;
        let cl = self.at(caller_base - 2).as_func().unwrap();
        self.reload(cl);
        self.l().top = caller_base + self.proto().framesize as usize;
        self.l().frame_top = self.l().top;

        match cont {
            Cont::Ra => {
                // Store result in the A register of the triggering
                // instruction (encoded in `extra`).
                self.set_at(caller_base + extra as usize, result);
                self.pc = saved_pc;
            }
            Cont::Nop => {
                self.pc = saved_pc;
            }
            Cont::Condt => {
                // saved_pc points at the fused JMP.
                let jmp = unsafe { *self.bcp.add(saved_pc) };
                self.pc = saved_pc + 1;
                if result.is_truthy() {
                    self.pc = (self.pc as i64 + bc_j(jmp)) as usize;
                }
            }
            Cont::Condf => {
                let jmp = unsafe { *self.bcp.add(saved_pc) };
                self.pc = saved_pc + 1;
                if !result.is_truthy() {
                    self.pc = (self.pc as i64 + bc_j(jmp)) as usize;
                }
            }
        }
        None
    }

    /// Numeric/cdata comparison fast lanes (`cmp_arm!`): returns
    /// `Some(cond)` when the comparison can be decided without
    /// metamethods, `None` when the operands need `__lt`/`__eq`&c (the
    /// caller takes the slow path). `op` is the bytecode ordinal
    /// (ISLT=0, ISGE=1, ISLE=2, ISGT=3); ISGE/ISGT are the *unordered*
    /// comparisons (NaN compares true), since the parser emits them as
    /// the negation of ISLT/ISLE — same as lj_meta_comp and rec_comp.
    #[inline(always)]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn cmp_fast(&self, op: u32, xv: LuaValue, yv: LuaValue) -> Option<bool> {
        if xv.is_number() && yv.is_number() {
            let x = xv.num();
            let y = yv.num();
            let cond = match op {
                0 => x < y,
                1 => !(x < y),
                2 => x <= y,
                3 => !(x <= y),
                _ => unreachable!(),
            };
            return Some(cond);
        }
        if let (Some(x), Some(y)) = (cdata_u64(self.l(), xv), cdata_u64(self.l(), yv)) {
            let x_is_ull = cdata_is_ull(xv);
            let y_is_ull = cdata_is_ull(yv);
            let cond = if x_is_ull == y_is_ull {
                if x_is_ull {
                    cmp_u(op, x, y)
                } else {
                    cmp_s(op, x, y)
                }
            } else if x_is_ull {
                // x is unsigned, y is signed: compare after sign check.
                if (y as i64) < 0 {
                    match op {
                        0 => false,
                        1 => true,
                        2 => false,
                        3 => true,
                        _ => unreachable!(),
                    }
                } else {
                    cmp_u(op, x, y)
                }
            } else {
                // x is signed, y is unsigned.
                if (x as i64) < 0 {
                    match op {
                        0 => true,
                        1 => false,
                        2 => true,
                        3 => false,
                        _ => unreachable!(),
                    }
                } else {
                    cmp_u(op, x, y)
                }
            };
            return Some(cond);
        }
        if let Some(x) = cdata_u64(self.l(), xv) {
            if !yv.is_number() {
                return None;
            }
            // cdata vs Lua number (e.g. `m1 < 0` with int64 cdata):
            // compare as the cdata's signedness.
            let x_is_ull = cdata_is_ull(xv);
            let yf = yv.num();
            let y = (yf as i64) as u64;
            let cond = if x_is_ull {
                let yn = y as i64;
                match op {
                    0 => yn < 0 || x < y,
                    1 => !(yn < 0 || x < y),
                    2 => yn < 0 || x <= y,
                    3 => !(yn < 0 || x <= y),
                    _ => unreachable!(),
                }
            } else {
                cmp_s(op, x, y)
            };
            return Some(cond);
        }
        if let Some(y) = cdata_u64(self.l(), yv) {
            if !xv.is_number() {
                return None;
            }
            // Lua number vs cdata.
            let y_is_ull = cdata_is_ull(yv);
            let xf = xv.num();
            let x = (xf as i64) as u64;
            let cond = if y_is_ull {
                let xn = x as i64;
                match op {
                    0 => xn < 0 || x < y,
                    1 => !(xn < 0 || x < y),
                    2 => xn < 0 || x <= y,
                    3 => !(xn < 0 || x <= y),
                    _ => unreachable!(),
                }
            } else {
                cmp_s(op, x, y)
            };
            return Some(cond);
        }
        if let (Some(xa), Some(ya)) = (cdata_ptr_addr(self.l(), xv), cdata_ptr_addr(self.l(), yv)) {
            // Pointer comparison: address order (root, offset).
            let x = (xa.0 as i128) << 64 | xa.1 as i128;
            let y = (ya.0 as i128) << 64 | ya.1 as i128;
            let cond = match op {
                0 => x < y,
                1 => !(x < y),
                2 => x <= y,
                3 => !(x <= y),
                _ => unreachable!(),
            };
            return Some(cond);
        }
        None
    }

    /// The dispatch loop. The entire hot state is two locals — `fr` (the
    /// register window, LuaJIT's BASE kept as a `Frame`) and `ip` (a
    /// walking instruction pointer, LuaJIT's PC) — so both live in
    /// registers on every dispatch. Everything else (`self.knp`,
    /// `self.multres`, ...) is re-read from `self` in the arms that need
    /// it; keeping more locals alive across the dispatch forces spills
    /// (measured: rustc packs the extras into an XMM register and unpacks
    /// per instruction).
    /// `sync!`/`resync!` bridge to the fields around calls and returns.
    ///
    /// `run` is the mode trampoline: it re-enters `dispatch` whenever the
    /// trace recorder turns on or off, standing in for LuaJIT's dispatch
    /// table switching (`lj_dispatch_update`). `dispatch::<true>` is the
    /// recording interpreter: it feeds every instruction through
    /// `lj_record_ins` before executing it.
    fn run(&mut self) -> LuaResult<usize> {
        loop {
            let rec = self.l().global().jit.state == crate::jit::TraceState::Record;
            let r = if rec {
                self.dispatch::<true>()
            } else {
                self.dispatch::<false>()
            };
            match r {
                Ok(Flow::Ret(n)) => return Ok(n),
                Ok(Flow::Rec) => continue, // Recording toggled: switch modes.
                Err(e) => {
                    // An error raised while recording aborts the trace.
                    if rec {
                        rec_abort_error(self.l().global());
                    }
                    // Error unwinding: this frame is being discarded, so
                    // close its open upvalues (they reference stack slots
                    // that die with the frame). LuaJIT does this in
                    // lj_err_throw; without it, closures captured before a
                    // `pcall(f, ...)`-style error keep pointing at reused
                    // stack memory. Yields must NOT close them: the
                    // coroutine's locals stay open across resume so
                    // closures keep sharing the live stack slots.
                    if e != LuaError::Yield {
                        self.close_upvals(self.base);
                    }
                    return Err(e);
                }
            }
        }
    }

    fn dispatch<const REC: bool>(&mut self) -> LuaResult<Flow> {
        let mut fr = Frame::new(self.sp, self.base);
        let mut ip = unsafe { self.bcp.add(self.pc) };
        macro_rules! jump {
            ($ins:expr) => {
                ip = unsafe { ip.offset(bc_j($ins) as isize) }
            };
        }
        macro_rules! sync {
            () => {{
                self.base = fr.cur_base();
                self.pc = unsafe { ip.offset_from(self.bcp) as usize };
                let l = self.l();
                l.debug_pc = self.pc;
                l.base = self.base;
            }};
        }
        macro_rules! resync {
            () => {{
                fr = Frame::new(self.sp, self.base);
                #[allow(unused_assignments)]
                {
                    ip = unsafe { self.bcp.add(self.pc) };
                }
            }};
        }
        // Comparison + fused following JMP. `op` is the bytecode ordinal
        // (ISLT=0, ISGE=1, ISLE=2, ISGT=3), matching lj_meta_comp's
        // encoding. ISGE/ISGT are the *unordered* comparisons (NaN takes
        // the jump), because the parser emits them as the negation of
        // ISLT/ISLE — same as the dasc VMs and rec_comp.
        //
        // The numeric/cdata fast lanes live in `Interp::cmp_fast` (an
        // `#[inline(always)]` method, inlined at every call site); only
        // the metamethod fallback is a macro, because it must sync the
        // locals and `continue` the dispatch loop.
        macro_rules! cmp_arm {
            ($op:expr, $xv:expr, $yv:expr) => {{
                let xv = $xv;
                let yv = $yv;
                if let Some(cond) = self.cmp_fast($op, xv, yv) {
                    branch!(cond);
                } else {
                    sync!();
                    match self.meta_comp(xv, yv, $op)? {
                        Some(cond) => branch!(cond),
                        None => {
                            resync!();
                            continue;
                        }
                    }
                }
            }};
        }
        macro_rules! branch {
            ($cond:expr) => {{
                let jmp = unsafe { *ip };
                ip = unsafe { ip.add(1) };
                if $cond {
                    jump!(jmp);
                }
            }};
        }
        // FORL/ITERL loop-edge semantics, shared between the normal arm
        // and the "hot counter fired, recording just started" path (the
        // hot instruction itself runs before the trace entry).
        macro_rules! forl_body {
            ($ins:expr, $a:expr) => {{
                let i = fr.reg($a + FORL_IDX).num();
                let s = fr.reg($a + FORL_STOP).num();
                let st = fr.reg($a + FORL_STEP).num();
                let ni = i + st;
                let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                if cont {
                    let nv = LuaValue::number_raw(ni);
                    fr.set($a + FORL_IDX, nv);
                    fr.set($a + FORL_EXT, nv);
                    // Force a line event on the loop body's first
                    // instruction of every iteration (Lua 5.1 semantics:
                    // each iteration fires, even on the same source line).
                    if self.l().hookmask & HOOKMASK_LINE != 0 {
                        self.l().hook_line = 0;
                    }
                    jump!($ins);
                }
            }};
        }
        macro_rules! iterl_body {
            ($ins:expr, $a:expr) => {{
                let first = fr.reg($a);
                if !first.is_nil() {
                    fr.set($a - 1, first);
                    jump!($ins);
                }
            }};
        }

        loop {
            if REC {
                // A nested call (or hook) may have aborted recording while
                // we were inside it; rec_ins consumes the recorder context,
                // so re-check before dispatching another instruction.
                if self.l().global().jit.state != crate::jit::TraceState::Record {
                    return Ok(Flow::Rec);
                }
                // Recording dispatch: feed the instruction about to be
                // executed through the recorder (lj_trace_ins).
                sync!();
                let pt = self.lua_cl().proto;
                let (pc, base) = (self.pc, self.base);
                if !rec_ins(self.l(), base, pt, pc) {
                    return Ok(Flow::Rec); // Recording ended: switch modes.
                }
                resync!();
            }
            // Debug hooks (debug.sethook): line and count events. KNIL
            // (register clearing) and LOOP (repeat-loop header) are
            // compiler-injected with no line of their own — skip them
            // entirely (no line or count event). A hook call runs
            // arbitrary Lua through the interpreter, which would clobber
            // an in-progress trace recording — abort it first.
            if self.l().hookmask != 0 && !self.l().hook_active {
                if REC {
                    jit::rec_abort_error(self.l().global());
                    return Ok(Flow::Rec);
                }
                sync!();
                let ins_hook = unsafe { *ip };
                let is_skip = matches!(bc_op(ins_hook), BCOp::KNIL | BCOp::LOOP);
                if !is_skip {
                    let pt = self.proto();
                    let pc = unsafe { ip.offset_from(self.bcp) as usize };
                    let pc = pc.min(pt.lines.len().saturating_sub(1));
                    let ln = pt.lines[pc];
                    if hook_check(self.l(), ln)? {
                        resync!();
                    }
                }
            }
            let ins = unsafe { *ip };
            ip = unsafe { ip.add(1) };
            // `debug_pc` is flushed lazily at the points that read it
            // (error raises, metamethod calls, sync to C) — not every
            // instruction — so the hot dispatch keeps `pc` in a register
            // (LuaJIT's interpreter tracks PC in a register and only
            // materializes it for error reporting).
            let a = bc_a(ins);
            match bc_op(ins) {
                // -- Comparisons (ORDER matters; see bc.rs) --
                BCOp::ISLT => cmp_arm!(0, fr.reg(a), fr.reg(bc_d(ins))),
                BCOp::ISGE => cmp_arm!(1, fr.reg(a), fr.reg(bc_d(ins))),
                BCOp::ISLE => cmp_arm!(2, fr.reg(a), fr.reg(bc_d(ins))),
                BCOp::ISGT => cmp_arm!(3, fr.reg(a), fr.reg(bc_d(ins))),
                BCOp::ISEQV => {
                    let x = fr.reg(a);
                    let y = fr.reg(bc_d(ins));
                    let cond = val_eq(self.l(), x, y);
                    if cond {
                        branch!(true);
                    } else if (x.is_table() || x.is_userdata()) && x.itype() == y.itype() {
                        sync!();
                        match self.meta_equal(x, y, 0)? {
                            Some(eq) => branch!(eq),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    } else {
                        branch!(false);
                    }
                }
                BCOp::ISNEV => {
                    let x = fr.reg(a);
                    let y = fr.reg(bc_d(ins));
                    let cond = val_eq(self.l(), x, y);
                    if cond {
                        branch!(false);
                    } else if (x.is_table() || x.is_userdata()) && x.itype() == y.itype() {
                        sync!();
                        match self.meta_equal(x, y, 1)? {
                            Some(eq) => branch!(!eq),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    } else {
                        branch!(true);
                    }
                }
                BCOp::ISEQS => {
                    let cond = val_eq(self.l(), fr.reg(a), self.kstr_at(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISNES => {
                    let cond = !val_eq(self.l(), fr.reg(a), self.kstr_at(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISEQN => {
                    let cond = val_eq(self.l(), fr.reg(a), self.knum(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISNEN => {
                    let cond = !val_eq(self.l(), fr.reg(a), self.knum(bc_d(ins)));
                    branch!(cond);
                }
                BCOp::ISEQP => {
                    let v = fr.reg(a);
                    let cond = if bc_d(ins) == 0 {
                        v.is_nil() || is_cdata_null(v)
                    } else {
                        val_eq(self.l(), v, PRI[bc_d(ins) as usize])
                    };
                    branch!(cond);
                }
                BCOp::ISNEP => {
                    let v = fr.reg(a);
                    let cond = if bc_d(ins) == 0 {
                        // Primitive 0 = nil; for ?? operator also treat NULL cdata as nil.
                        !v.is_nil() && !is_cdata_null(v)
                    } else {
                        !val_eq(self.l(), v, PRI[bc_d(ins) as usize])
                    };
                    branch!(cond);
                }
                BCOp::ISTC => {
                    let d = fr.reg(bc_d(ins));
                    let cond = d.is_truthy();
                    if cond {
                        fr.set(a, d);
                    }
                    branch!(cond);
                }
                BCOp::ISFC => {
                    let d = fr.reg(bc_d(ins));
                    let cond = !d.is_truthy();
                    if cond {
                        fr.set(a, d);
                    }
                    branch!(cond);
                }
                BCOp::IST => {
                    let cond = fr.reg(bc_d(ins)).is_truthy();
                    branch!(cond);
                }
                BCOp::ISF => {
                    let cond = !fr.reg(bc_d(ins)).is_truthy();
                    branch!(cond);
                }

                // -- Unary and move --
                BCOp::MOV => fr.set(a, fr.reg(bc_d(ins))),
                BCOp::NOT => fr.set(a, LuaValue::boolean(!fr.reg(bc_d(ins)).is_truthy())),
                BCOp::UNM => {
                    let v = fr.reg(bc_d(ins));
                    if let Some(bits) = cdata_u64(self.l(), v) {
                        let is_ull = cdata_is_ull(v);
                        let r = make_cdata_result(self.l(), (!bits).wrapping_add(1), is_ull);
                        fr.set(a, r);
                    } else if v.is_number() {
                        fr.set(a, LuaValue::number_raw(-v.num()));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Unm, v, v, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::LEN => {
                    let v = fr.reg(bc_d(ins));
                    if let Some(sid) = v.as_string_id() {
                        let n = self.l().heap().strings.get(sid).len();
                        fr.set(a, LuaValue::number(n as f64));
                    } else if let Some(t) = v.as_table() {
                        // Lua 5.2+: `__len` applies to tables when present.
                        let has_len_mm = crate::meta::meta_fast(
                            self.l().global(),
                            t.as_ref().metatable,
                            MM::Len,
                        )
                        .is_some();
                        if !has_len_mm {
                            fr.set(a, LuaValue::number(t.as_ref().len() as f64));
                        } else {
                            sync!();
                            match self.meta_len(v, a)? {
                                Some(r) => fr.set(a, r),
                                None => {
                                    resync!();
                                    continue;
                                }
                            }
                        }
                    } else {
                        sync!();
                        match self.meta_len(v, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }

                // -- Arithmetic --
                BCOp::ADDVV => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        fr.set(a, LuaValue::number_raw(xv.num() + yv.num()));
                    } else if let Some(r) =
                        try_cdata_binop(self.l(), xv, yv, |x, y| x.wrapping_add(y))
                    {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Add, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::SUBVV => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        fr.set(a, LuaValue::number_raw(xv.num() - yv.num()));
                    } else if let Some(r) =
                        try_cdata_binop(self.l(), xv, yv, |x, y| x.wrapping_sub(y))
                    {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Sub, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MULVV => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        fr.set(a, LuaValue::number_raw(xv.num() * yv.num()));
                    } else if let Some(r) =
                        try_cdata_binop(self.l(), xv, yv, |x, y| x.wrapping_mul(y))
                    {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mul, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::DIVVV => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        fr.set(a, LuaValue::number_raw(xv.num() / yv.num()));
                    } else if let Some(r) = try_cdata_binop(self.l(), xv, yv, |x, y| x / y) {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Div, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MODVV => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        let x = xv.num();
                        let y = yv.num();
                        fr.set(a, LuaValue::number_raw(x - (x / y).floor() * y));
                    } else if let Some(r) = try_cdata_binop(self.l(), xv, yv, |x, y| x % y) {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mod, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::ADDVN => {
                    let xv = fr.reg(bc_b(ins));
                    if xv.is_number() {
                        let x = xv.num();
                        let y = self.knum(bc_c(ins)).num();
                        fr.set(a, LuaValue::number_raw(x + y));
                    } else if let Some(x_cd) = cdata_u64(self.l(), xv) {
                        let is_ull = cdata_is_ull(xv);
                        let y = self.knum(bc_c(ins)).num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x_cd.wrapping_add(y), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Add, xv, self.knum(bc_c(ins)), a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::SUBVN => {
                    let xv = fr.reg(bc_b(ins));
                    if xv.is_number() {
                        let x = xv.num();
                        let y = self.knum(bc_c(ins)).num();
                        fr.set(a, LuaValue::number_raw(x - y));
                    } else if let Some(x_cd) = cdata_u64(self.l(), xv) {
                        let is_ull = cdata_is_ull(xv);
                        let y = self.knum(bc_c(ins)).num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x_cd.wrapping_sub(y), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Sub, xv, self.knum(bc_c(ins)), a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MULVN => {
                    let xv = fr.reg(bc_b(ins));
                    if xv.is_number() {
                        let x = xv.num();
                        let y = self.knum(bc_c(ins)).num();
                        fr.set(a, LuaValue::number_raw(x * y));
                    } else if let Some(x_cd) = cdata_u64(self.l(), xv) {
                        let is_ull = cdata_is_ull(xv);
                        let y = self.knum(bc_c(ins)).num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x_cd.wrapping_mul(y), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mul, xv, self.knum(bc_c(ins)), a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::DIVVN => {
                    let xv = fr.reg(bc_b(ins));
                    if xv.is_number() {
                        let x = xv.num();
                        let y = self.knum(bc_c(ins)).num();
                        fr.set(a, LuaValue::number_raw(x / y));
                    } else if let Some(x_cd) = cdata_u64(self.l(), xv) {
                        let is_ull = cdata_is_ull(xv);
                        let y = self.knum(bc_c(ins)).num() as i64 as u64;
                        let r = x_cd.checked_div(y).unwrap_or(0);
                        fr.set(a, make_cdata_result(self.l(), r, is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Div, xv, self.knum(bc_c(ins)), a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MODVN => {
                    let xv = fr.reg(bc_b(ins));
                    if xv.is_number() {
                        let x = xv.num();
                        let y = self.knum(bc_c(ins)).num();
                        fr.set(a, LuaValue::number_raw(x - (x / y).floor() * y));
                    } else if let Some(x_cd) = cdata_u64(self.l(), xv) {
                        let is_ull = cdata_is_ull(xv);
                        let y = self.knum(bc_c(ins)).num() as i64 as u64;
                        let r = if y == 0 { 0 } else { x_cd % y };
                        fr.set(a, make_cdata_result(self.l(), r, is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mod, xv, self.knum(bc_c(ins)), a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::ADDNV => {
                    let kv = self.knum(bc_c(ins));
                    let yv = fr.reg(bc_b(ins));
                    if yv.is_number() {
                        let y = yv.num();
                        fr.set(a, LuaValue::number_raw(kv.num() + y));
                    } else if let Some(y_cd) = cdata_u64(self.l(), yv) {
                        let is_ull = cdata_is_ull(yv);
                        let x = kv.num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x.wrapping_add(y_cd), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Add, kv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::SUBNV => {
                    let kv = self.knum(bc_c(ins));
                    let yv = fr.reg(bc_b(ins));
                    if yv.is_number() {
                        let y = yv.num();
                        fr.set(a, LuaValue::number_raw(kv.num() - y));
                    } else if let Some(y_cd) = cdata_u64(self.l(), yv) {
                        let is_ull = cdata_is_ull(yv);
                        let x = kv.num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x.wrapping_sub(y_cd), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Sub, kv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MULNV => {
                    let kv = self.knum(bc_c(ins));
                    let yv = fr.reg(bc_b(ins));
                    if yv.is_number() {
                        let y = yv.num();
                        fr.set(a, LuaValue::number_raw(kv.num() * y));
                    } else if let Some(y_cd) = cdata_u64(self.l(), yv) {
                        let is_ull = cdata_is_ull(yv);
                        let x = kv.num() as i64 as u64;
                        fr.set(a, make_cdata_result(self.l(), x.wrapping_mul(y_cd), is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mul, kv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::DIVNV => {
                    let kv = self.knum(bc_c(ins));
                    let yv = fr.reg(bc_b(ins));
                    if yv.is_number() {
                        let y = yv.num();
                        fr.set(a, LuaValue::number_raw(kv.num() / y));
                    } else if let Some(y_cd) = cdata_u64(self.l(), yv) {
                        let is_ull = cdata_is_ull(yv);
                        let x = kv.num() as i64 as u64;
                        let r = x.checked_div(y_cd).unwrap_or(0);
                        fr.set(a, make_cdata_result(self.l(), r, is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Div, kv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::MODNV => {
                    let kv = self.knum(bc_c(ins));
                    let yv = fr.reg(bc_b(ins));
                    if yv.is_number() {
                        let y = yv.num();
                        let x = kv.num();
                        fr.set(a, LuaValue::number_raw(x - (x / y).floor() * y));
                    } else if let Some(y_cd) = cdata_u64(self.l(), yv) {
                        let is_ull = cdata_is_ull(yv);
                        let x = kv.num() as i64 as u64;
                        let r = if y_cd == 0 { 0 } else { x % y_cd };
                        fr.set(a, make_cdata_result(self.l(), r, is_ull));
                    } else {
                        sync!();
                        match self.meta_arith(MM::Mod, kv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::POW => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    if xv.is_number() && yv.is_number() {
                        fr.set(a, LuaValue::number_raw(vm_pow(xv.num(), yv.num())));
                    } else if let Some(r) =
                        try_cdata_binop(self.l(), xv, yv, |x, y| (x as f64).powf(y as f64) as u64)
                    {
                        fr.set(a, r);
                    } else {
                        sync!();
                        match self.meta_arith(MM::Pow, xv, yv, a)? {
                            Some(r) => fr.set(a, r),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::CAT => {
                    sync!();
                    let r = self.op_cat(a, bc_b(ins), bc_c(ins))?;
                    // op_cat may run a nested execute (__concat /
                    // __tostring), which can reallocate the stack: rebuild
                    // the fast register window before writing the result.
                    resync!();
                    fr.set(a, r);
                }

                // -- Constants --
                BCOp::KSTR => {
                    let v = self.kstr_at(bc_d(ins));
                    fr.set(a, v);
                }
                BCOp::KSHORT => fr.set(a, LuaValue::number(bc_d(ins) as i16 as f64)),
                BCOp::KNUM => fr.set(a, self.knum(bc_d(ins))),
                BCOp::KPRI => fr.set(a, PRI[bc_d(ins) as usize]),
                BCOp::KCDATA => {
                    let idx = bc_d(ins) as usize;
                    let proto = self.lua_cl().proto;
                    let kgc = &proto.as_ref().kgc;
                    if idx >= kgc.len() {
                        return Err(self
                            .l()
                            .runtime_error(format!("invalid KCDATA index {}", idx).as_bytes()));
                    }
                    let cdata = match &kgc[idx] {
                        crate::proto::KGc::CData(cd) => {
                            let cd_heap = self.l().global().heap.alloc_cdata(cd.as_ref().clone());
                            LuaValue::cdata(cd_heap)
                        }
                        _ => {
                            return Err(self
                                .l()
                                .runtime_error("KCDATA with non-CData kgc entry".as_bytes()));
                        }
                    };
                    fr.set(a, cdata);
                }
                BCOp::KNIL => {
                    for i in a..=bc_d(ins) {
                        fr.set(i, LuaValue::NIL);
                    }
                }

                // -- Upvalues --
                BCOp::UGET => {
                    let uv = self.lua_cl().upvals[bc_d(ins) as usize];
                    fr.set(a, self.upval_get(uv));
                }
                BCOp::USETV => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = fr.reg(bc_d(ins));
                    self.upval_set(uv, v);
                }
                BCOp::USETS => {
                    let uv = self.lua_cl().upvals[a as usize];
                    let v = self.kstr_at(bc_d(ins));
                    self.upval_set(uv, v);
                }
                BCOp::USETN => {
                    let uv = self.lua_cl().upvals[a as usize];
                    self.upval_set(uv, self.knum(bc_d(ins)));
                }
                BCOp::USETP => {
                    let uv = self.lua_cl().upvals[a as usize];
                    self.upval_set(uv, PRI[bc_d(ins) as usize]);
                }
                BCOp::UCLO => {
                    sync!();
                    self.op_uclo(a);
                    jump!(ins);
                }
                BCOp::FNEW => {
                    sync!();
                    let v = self.op_fnew(a, bc_d(ins))?;
                    fr.set(a, v);
                }

                // -- Tables --
                BCOp::TNEW => {
                    // Inline: only sync when GC is due (cold path).
                    let g = self.l().global();
                    if g.heap.should_collect()
                        || g.heap.gc_state == crate::runtime::gc::GcState::Finalize
                    {
                        sync!();
                        self.gc_check(self.base + a as usize + 1)?;
                    }
                    // The compiler pre-allocates the array/hash size the
                    // table literal will need (D = asize | hbits << 11, see
                    // expr_table). Allocating it up front avoids repeated
                    // realloc-and-copy as the literal is filled.
                    let d = bc_d(ins);
                    let asize = d & 0x7ff;
                    let hbits = (d >> 11) & 0x1f;
                    let t = self.l().heap().alloc_table(LuaTable::new(asize, hbits));
                    fr.set(a, LuaValue::table(t));
                }
                BCOp::TDUP => {
                    sync!();
                    let t = self.op_tdup(a, bc_d(ins))?;
                    fr.set(a, t);
                }
                BCOp::GGET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    let v = env.as_ref().get_str(key);
                    if !v.is_nil() || env.as_ref().metatable.is_none() {
                        fr.set(a, v);
                    } else {
                        sync!();
                        match self.meta_tget(LuaValue::table(env), key, a)? {
                            Some(v) => fr.set(a, v),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::GSET => {
                    let env = self.lua_cl().env;
                    let key = self.kstr_at(bc_d(ins));
                    let mt = env.as_ref().metatable;
                    if mt.is_none() || !env.as_ref().get_str(key).is_nil() {
                        env.as_mut().set_str(key, fr.reg(a));
                    } else {
                        sync!();
                        match self.meta_tset(LuaValue::table(env), key, fr.reg(a))? {
                            Some(_) => {}
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TGETV => {
                    let t = fr.reg(bc_b(ins));
                    let k = fr.reg(bc_c(ins));
                    if let Some(tab) = t.as_table() {
                        let v = if k.is_string() {
                            tab.as_ref().get_str(k)
                        } else if k.is_number() {
                            let ki = k.num() as i32;
                            if ki as f64 == k.num() && ki >= 0 {
                                tab.as_ref().get_int(ki)
                            } else {
                                tab.as_ref().get(k)
                            }
                        } else {
                            tab.as_ref().get(k)
                        };
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            fr.set(a, v);
                        } else {
                            sync!();
                            match self.meta_tget(t, k, a)? {
                                Some(v) => fr.set(a, v),
                                None => {
                                    resync!();
                                    continue;
                                }
                            }
                        }
                    } else {
                        sync!();
                        match self.meta_tget(t, k, a)? {
                            Some(v) => fr.set(a, v),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TGETS => {
                    let t = fr.reg(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = self.kstr_at(bc_c(ins));
                        let v = tab.as_ref().get_str(k);
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            fr.set(a, v);
                        } else {
                            sync!();
                            match self.meta_tget(t, k, a)? {
                                Some(v) => fr.set(a, v),
                                None => {
                                    resync!();
                                    continue;
                                }
                            }
                        }
                    } else {
                        sync!();
                        match self.meta_tget(t, self.kstr_at(bc_c(ins)), a)? {
                            Some(v) => fr.set(a, v),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TGETB => {
                    let t = fr.reg(bc_b(ins));
                    if let Some(tab) = t.as_table() {
                        let k = bc_c(ins) as i32;
                        let v = tab.as_ref().get_int(k);
                        if !v.is_nil() || tab.as_ref().metatable.is_none() {
                            fr.set(a, v);
                        } else {
                            sync!();
                            match self.meta_tget(t, LuaValue::number(k as f64), a)? {
                                Some(v) => fr.set(a, v),
                                None => {
                                    resync!();
                                    continue;
                                }
                            }
                        }
                    } else {
                        sync!();
                        match self.meta_tget(t, LuaValue::number(bc_c(ins) as f64), a)? {
                            Some(v) => fr.set(a, v),
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TSETV => {
                    let t = fr.reg(bc_b(ins));
                    let k = fr.reg(bc_c(ins));
                    let v = fr.reg(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        if k.is_string() {
                            tab.as_mut().set_str(k, v);
                        } else if k.is_number() {
                            let n = k.num();
                            if n.is_nan() {
                                sync!();
                                return Err(self.l().runtime_error(b"table index is NaN"));
                            }
                            let ki = n as i32;
                            if ki as f64 == n && ki >= 0 {
                                tab.as_mut().set_int(ki, v);
                            } else {
                                tab.as_mut().set(k, v);
                            }
                        } else if k.is_nil() {
                            sync!();
                            return Err(self.l().runtime_error(b"table index is nil"));
                        } else {
                            tab.as_mut().set(k, v);
                        }
                        barrier_back(&mut self.l().global().heap, tab);
                    } else {
                        sync!();
                        match self.meta_tset(t, k, v)? {
                            Some(_) => {}
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TSETS => {
                    let t = fr.reg(bc_b(ins));
                    let v = fr.reg(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        let k = self.kstr_at(bc_c(ins));
                        tab.as_mut().set_str(k, v);
                        barrier_back(&mut self.l().global().heap, tab);
                    } else {
                        sync!();
                        match self.meta_tset(t, self.kstr_at(bc_c(ins)), v)? {
                            Some(_) => {}
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TSETB => {
                    let t = fr.reg(bc_b(ins));
                    let v = fr.reg(a);
                    if let Some(tab) = t.as_table()
                        && tab.as_ref().metatable.is_none()
                    {
                        tab.as_mut().set_int(bc_c(ins) as i32, v);
                        barrier_back(&mut self.l().global().heap, tab);
                    } else {
                        sync!();
                        match self.meta_tset(t, LuaValue::number(bc_c(ins) as f64), v)? {
                            Some(_) => {}
                            None => {
                                resync!();
                                continue;
                            }
                        }
                    }
                }
                BCOp::TSETM => {
                    // Inline the fast path: table at R[a-1], values at
                    // R[a..a+multres-1], base_key from constant table.
                    let t = fr.reg(a - 1);
                    if let Some(tab) = t.as_table() {
                        let base_key = self.knum(bc_d(ins)).num() as i64 - (1i64 << 52);
                        let mr = self.multres;
                        if mr > 0 && base_key == 1 {
                            let need = (base_key as u32).wrapping_add(mr as u32);
                            tab.as_mut().reasize(need);
                        }
                        for i in 0..mr {
                            let key = base_key + i as i64;
                            let v = fr.reg(a + i as u32);
                            if key >= 0 && key <= i32::MAX as i64 {
                                tab.as_mut().set_int(key as i32, v);
                            } else {
                                tab.as_mut().set(LuaValue::number(key as f64), v);
                            }
                        }
                        barrier_back(&mut self.l().global().heap, tab);
                    } else {
                        sync!();
                        self.tsetm(a, bc_d(ins))?;
                    }
                }

                // -- Calls / returns --
                BCOp::CALL => {
                    // Fast path (LuaJIT's ins_call): a Lua callee switches
                    // frames right here — one store for the frame link, no
                    // sync round-trip. C callees and vararg protos go slow.
                    let nargs = bc_c(ins) as usize - 1;
                    match self.call_lua_fast::<REC>(a, nargs, fr, ip)? {
                        (nf, nip, CallFast::Applied) => {
                            fr = nf;
                            ip = nip;
                            continue;
                        }
                        (nf, nip, CallFast::Trace(f)) => {
                            let _ = (nf, nip);
                            return Ok(f);
                        }
                        (nf, nip, CallFast::Slow) => {
                            fr = nf;
                            ip = nip;
                            sync!();
                            self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                            resync!();
                        }
                    }
                }
                BCOp::CALLM => {
                    // BC_CALLM: the callee's argument count includes the
                    // results of the previous multi-return call (`multres`).
                    // Plain Lua callees take the same inline fast path as
                    // BC_CALL (a previous call's results already sit in the
                    // right argument slots); only the arg count differs.
                    let nargs = bc_c(ins) as usize + self.multres;
                    match self.call_lua_fast::<REC>(a, nargs, fr, ip)? {
                        (nf, nip, CallFast::Applied) => {
                            fr = nf;
                            ip = nip;
                            continue;
                        }
                        (nf, nip, CallFast::Trace(f)) => {
                            let _ = (nf, nip);
                            return Ok(f);
                        }
                        (nf, nip, CallFast::Slow) => {
                            fr = nf;
                            ip = nip;
                            sync!();
                            self.do_call(a, nargs, bc_b(ins) as i32 - 1)?;
                            resync!();
                        }
                    }
                }
                BCOp::CALLT => {
                    // Fast path: Lua callee, no vararg frame on either side
                    // — reuse the frame in place (BC_CALLT's hot route).
                    let f = fr.reg(a);
                    if let Some(gf) = f.as_func()
                        && let GcFunc::Lua(cl) = gf.as_ref()
                    {
                        let pt = cl.proto.as_ref();
                        let link = fr.frame_link();
                        if (pt.flags & PROTO_VARARG) == 0 && (link & FRAME_TYPE_MASK) != FRAME_VARG
                        {
                            let nargs = bc_d(ins) as usize - 1;

                            let fs_need = fr.cur_base()
                                + a.max(nargs as u32) as usize
                                + pt.framesize as usize
                                + 8;
                            if fs_need > self.l().stack.len() {
                                sync!();
                                self.l().stack_ensure(fs_need);
                                self.sp = self.l().stack.as_mut_ptr();
                                resync!();
                            }
                            fr.set_func(f);
                            // Copy args down (dest [0,nargs) ⊂ src
                            // [2,nargs+2)): forward order is safe because
                            // each source slot is written only after being
                            // read (dest[i] = src[i+2]).
                            for i in 0..nargs {
                                fr.set(i as u32, fr.reg(a + 2 + i as u32));
                            }

                            for i in nargs..pt.numparams as usize {
                                fr.set(i as u32, LuaValue::NIL);
                            }
                            ip = unsafe { pt.bc.as_ptr().add(1) };
                            self.cl = gf;
                            self.bcp = pt.bc.as_ptr();
                            self.knp = pt.kn.as_ptr();
                            self.ksp = pt.kstrv.as_ptr();
                            self.l().top = fr.cur_base() + pt.framesize as usize;
                            self.l().frame_top = self.l().top;
                            let head = pt.bc[0];
                            if !REC && bc_op(head) == BCOp::JFUNCF {
                                sync!();
                                let r = trace_exec(self.l(), self.base, bc_d(head));
                                self.sp = self.l().stack.as_mut_ptr();
                                if r.stack_overflow {
                                    return Err(self.l().runtime_error(b"stack overflow"));
                                }
                                self.pc = r.pc;
                                if r.baseslot != 2 {
                                    self.trace_exit_frame(r.baseslot);
                                } else {
                                    fr = Frame::new(self.sp, self.base);
                                    self.reload_at(fr);
                                    self.l().top = self.base + self.proto().framesize as usize;
                                }
                                if self.rec_started() {
                                    return Ok(Flow::Rec); // Hot exit side trace.
                                }
                                self.gc_check(self.l().frame_top)?;
                                resync!();
                            }
                            continue;
                        }
                    }
                    let nargs = bc_d(ins) as usize - 1;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs)? {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::CALLMT => {
                    let nargs = bc_d(ins) as usize + self.multres;
                    sync!();
                    if let Some(n) = self.do_tailcall(a, nargs)? {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET0 => {
                    if self.l().hookmask & HOOKMASK_RET != 0 && !self.l().hook_active {
                        sync!();
                        self.hook_event("return")?;
                        resync!();
                    }
                    if let Some((wbase, want, ret_ip, ca)) = self.ret_fast_n(fr.bp()) {
                        if !self.hook_lines.is_empty() {
                            let _ = self.hook_lines.pop();
                        }
                        let dst = wbase - 2;
                        for i in 0..want.max(0) as usize {
                            fr.set_abs(dst + i, LuaValue::NIL);
                        }
                        self.multres = 0;
                        fr = Frame::new(self.sp, dst - ca as usize);
                        ip = ret_ip;
                        self.reload_at(fr);
                        // The inlined CALL fast path set `frame_top` to the
                        // callee frame; restore the caller's extent so later
                        // GC marks see the real frame (a small callee like a
                        // `return 1` closure would otherwise shrink it).
                        self.l().frame_top = fr.cur_base() + self.proto().framesize as usize;
                        // Skip a line event on the caller's resumption
                        // line (Lua 5.1: the caller's line was already
                        // reported before the call).
                        if self.l().hookmask & HOOKMASK_LINE != 0 {
                            let pt = self.proto();
                            let pc = unsafe { ip.offset_from(self.bcp) as usize };
                            let pc = pc.min(pt.lines.len().saturating_sub(1));
                            self.l().hook_line = pt.lines[pc];
                        }
                        self.l().top = dst + if want >= 0 { want as usize } else { 0 };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(fr.cur_base(), 0) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET1 => {
                    if self.l().hookmask & HOOKMASK_RET != 0 && !self.l().hook_active {
                        sync!();
                        self.hook_event("return")?;
                        resync!();
                    }
                    if let Some((wbase, want, ret_ip, ca)) = self.ret_fast_n(fr.bp()) {
                        if !self.hook_lines.is_empty() {
                            let _ = self.hook_lines.pop();
                        }
                        let dst = wbase - 2;
                        fr.set_abs(dst, fr.reg(a));
                        for i in 1..want.max(1) as usize {
                            fr.set_abs(dst + i, LuaValue::NIL);
                        }
                        self.multres = 1;
                        fr = Frame::new(self.sp, dst - ca as usize);
                        ip = ret_ip;
                        self.reload_at(fr);
                        self.l().frame_top = fr.cur_base() + self.proto().framesize as usize;
                        if self.l().hookmask & HOOKMASK_LINE != 0 {
                            let pt = self.proto();
                            let pc = unsafe { ip.offset_from(self.bcp) as usize };
                            let pc = pc.min(pt.lines.len().saturating_sub(1));
                            self.l().hook_line = pt.lines[pc];
                        }
                        self.l().top = dst + if want >= 0 { want as usize } else { 1 };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(fr.cur_base() + a as usize, 1) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RET => {
                    let n = bc_d(ins) as usize - 1;
                    if self.l().hookmask & HOOKMASK_RET != 0 && !self.l().hook_active {
                        sync!();
                        self.hook_event("return")?;
                        resync!();
                    }
                    if let Some((wbase, want, ret_ip, ca)) = self.ret_fast_n(fr.bp()) {
                        if !self.hook_lines.is_empty() {
                            self.l().hook_line = self.hook_lines.pop().unwrap();
                        }
                        if !self.hook_lines.is_empty() {
                            self.l().hook_line = self.hook_lines.pop().unwrap();
                        }
                        let src = a;
                        let dst = wbase - 2;
                        for i in 0..n {
                            fr.set_abs(dst + i, fr.reg(src + i as u32));
                        }
                        for i in n..(want.max(0) as usize) {
                            fr.set_abs(dst + i, LuaValue::NIL);
                        }
                        self.multres = n;
                        fr = Frame::new(self.sp, dst - ca as usize);
                        ip = ret_ip;
                        self.reload_at(fr);
                        self.l().frame_top = fr.cur_base() + self.proto().framesize as usize;
                        self.l().top = dst + if want >= 0 { want as usize } else { n };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(fr.cur_base() + a as usize, n) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }
                BCOp::RETM => {
                    let n = self.multres + bc_d(ins) as usize;
                    if let Some((wbase, want, ret_ip, ca)) = self.ret_fast_n(fr.bp()) {
                        if !self.hook_lines.is_empty() {
                            self.l().hook_line = self.hook_lines.pop().unwrap();
                        }
                        let src = a;
                        let dst = wbase - 2;
                        for i in 0..n {
                            fr.set_abs(dst + i, fr.reg(src + i as u32));
                        }
                        for i in n..(want.max(0) as usize) {
                            fr.set_abs(dst + i, LuaValue::NIL);
                        }
                        self.multres = n;
                        fr = Frame::new(self.sp, dst - ca as usize);
                        ip = ret_ip;
                        self.reload_at(fr);
                        self.l().frame_top = fr.cur_base() + self.proto().framesize as usize;
                        self.l().top = dst + if want >= 0 { want as usize } else { n };
                        continue;
                    }
                    sync!();
                    if let Some(n) = self.do_return(fr.cur_base() + a as usize, n) {
                        return Ok(Flow::Ret(n));
                    }
                    resync!();
                }

                // -- Loops and branches --
                BCOp::FORI => {
                    let idx = fr.reg(a + FORL_IDX);
                    let stop = fr.reg(a + FORL_STOP);
                    let step = fr.reg(a + FORL_STEP);
                    if let (Some(i), Some(s), Some(st)) = (
                        for_number(self.l(), idx),
                        for_number(self.l(), stop),
                        for_number(self.l(), step),
                    ) {
                        // LuaJIT coerces strings to numbers and writes the
                        // converted values back for the loop comparisons.
                        fr.set(a + FORL_IDX, LuaValue::number(i));
                        fr.set(a + FORL_STOP, LuaValue::number(s));
                        fr.set(a + FORL_STEP, LuaValue::number(st));
                        fr.set(a + FORL_EXT, LuaValue::number_raw(i));
                        let enter = if st >= 0.0 { i <= s } else { i >= s };
                        if !enter {
                            jump!(ins);
                        }
                    } else {
                        sync!();
                        return Err(self
                            .l()
                            .runtime_error(b"'for' initial value must be a number"));
                    }
                }
                BCOp::JFORI => {
                    // FORI semantics; on loop entry go straight into the
                    // trace whose number sits in the JFORL at the target.
                    let idx = fr.reg(a + FORL_IDX);
                    let stop = fr.reg(a + FORL_STOP);
                    let step = fr.reg(a + FORL_STEP);
                    if let (Some(i), Some(s), Some(st)) = (
                        for_number(self.l(), idx),
                        for_number(self.l(), stop),
                        for_number(self.l(), step),
                    ) {
                        fr.set(a + FORL_IDX, LuaValue::number(i));
                        fr.set(a + FORL_STOP, LuaValue::number(s));
                        fr.set(a + FORL_STEP, LuaValue::number(st));
                        fr.set(a + FORL_EXT, LuaValue::number_raw(i));
                        let enter = if st >= 0.0 { i <= s } else { i >= s };
                        if enter {
                            sync!();
                            let jforl = (self.pc as i64 - 1 + bc_j(ins)) as usize;
                            let tno = bc_d(self.proto().bc[jforl]);
                            let r = trace_exec(self.l(), self.base, tno);
                            self.sp = self.l().stack.as_mut_ptr();
                            self.pc = r.pc;
                            if r.baseslot != 2 {
                                self.trace_exit_frame(r.baseslot);
                            }
                            if self.rec_started() {
                                return Ok(Flow::Rec); // Hot exit: record a side trace.
                            }
                            self.gc_check(self.l().frame_top)?;
                            resync!();
                            continue;
                        } else {
                            jump!(ins);
                        }
                    } else {
                        sync!();
                        return Err(self
                            .l()
                            .runtime_error(b"'for' initial value must be a number"));
                    }
                }
                BCOp::FORL | BCOp::IFORL => {
                    // FORL is the hot-counting variant (lj_vm's `hotloop`
                    // macro); IFORL is the blacklisted/non-counting one.
                    if !REC
                        && bc_op(ins) == BCOp::FORL
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            // Recording started: run this FORL un-recorded
                            // (it sits before the trace entry), then
                            // switch to the recording dispatch.
                            resync!();
                            forl_body!(ins, a);
                            sync!();
                            return Ok(Flow::Rec);
                        }
                        resync!();
                    }
                    // Every iteration re-enters the loop body: force a
                    // line event on its first instruction (Lua 5.1 fires
                    // one per iteration, even on the same source line).
                    if self.l().hookmask & HOOKMASK_LINE != 0 {
                        self.l().hook_line = 0;
                    }
                    forl_body!(ins, a);
                }
                BCOp::JFORL => {
                    // A setmetatable call flagged invalidation (the trace
                    // specialized to a metatable's `__newindex`, which may
                    // have changed). Flush now — this JFORL entry runs in
                    // the interpreter, before any trace mcode is entered, so
                    // freeing the registry is safe. trace_flush_all reverted
                    // the JFORL in place to the original FORL; re-read it
                    // and take its loop-back jump.
                    if self.l().global().jit.invalidate_all {
                        self.l().global().jit.invalidate_all = false;
                        crate::jit::trace_flush_all(self.l());
                        #[cfg(debug_assertions)]
                        eprintln!("JFORL invalidate flush at pc={}", self.l().debug_pc);
                        // Reset the loop's hot counter so it won't instantly
                        // re-record (and re-flush) on every setmetatable — a
                        // loop that mutates metatables each iteration can
                        // never produce a stable trace. The interpreted loop
                        // runs until hotcount accumulates again.
                        let reset = (self.l().global().jit.param(crate::jit::JitParam::HotLoop)
                            as u32
                            * crate::jit::HOTCOUNT_LOOP as u32)
                            as crate::jit::HotCount;
                        self.l().global().jit.hotcount_set(ip as usize, reset);
                        let reverted = unsafe { *ip.sub(1) };
                        let a2 = bc_a(reverted);
                        forl_body!(reverted, a2);
                        continue;
                    }
                    // IFORL semantics; on loop-taken enter the compiled
                    // trace (the dasc VMs dispatch to BC_JLOOP).
                    let i = fr.reg(a + FORL_IDX).num();
                    let s = fr.reg(a + FORL_STOP).num();
                    let st = fr.reg(a + FORL_STEP).num();
                    let ni = i + st;
                    let cont = if st >= 0.0 { ni <= s } else { ni >= s };
                    if cont {
                        let nv = LuaValue::number_raw(ni);
                        fr.set(a + FORL_IDX, nv);
                        fr.set(a + FORL_EXT, nv);
                        sync!();
                        let r = trace_exec(self.l(), self.base, bc_d(ins));
                        self.sp = self.l().stack.as_mut_ptr();
                        self.pc = r.pc;
                        if r.baseslot != 2 {
                            self.trace_exit_frame(r.baseslot);
                        }
                        if self.rec_started() {
                            return Ok(Flow::Rec); // Hot exit: record a side trace.
                        }
                        self.gc_check(self.l().frame_top)?;
                        resync!();
                    }
                }
                BCOp::LOOP | BCOp::ILOOP => {
                    // No-op apart from hot counting (ILOOP: not even that).
                    if !REC
                        && bc_op(ins) == BCOp::LOOP
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            return Ok(Flow::Rec); // pc already past LOOP.
                        }
                        resync!();
                    }
                }
                BCOp::JLOOP => {
                    // Enter the compiled trace; the interpreter resumes at
                    // whatever snapshot the trace exits through.
                    sync!();
                    let r = trace_exec(self.l(), self.base, bc_d(ins));
                    self.sp = self.l().stack.as_mut_ptr();
                    self.pc = r.pc;
                    if r.baseslot != 2 {
                        self.trace_exit_frame(r.baseslot);
                    }
                    if self.rec_started() {
                        return Ok(Flow::Rec); // Hot exit: record a side trace.
                    }
                    self.gc_check(self.l().frame_top)?;
                    resync!();
                }
                BCOp::JMP => jump!(ins),
                BCOp::ISNEXT => jump!(ins),
                BCOp::ITERC | BCOp::ITERN => {
                    // Move the iterator triple (genf, state, ctl) into the
                    // call slots, then dispatch like a 2-arg call. Lua
                    // iterator closures take the inline fast path (no
                    // do_call round-trip) — this is the hot inner loop of
                    // `for k,v in <closure> do ... end`.
                    let nret = bc_b(ins) as usize;
                    let fs = fr.cur_base() + a as usize;
                    let genf = self.at(fs - 3);
                    let state = self.at(fs - 2);
                    let ctl = self.at(fs - 1);
                    self.set_at(fs, genf);
                    self.set_at(fs + 2, state);
                    self.set_at(fs + 3, ctl);
                    match self.call_lua_fast::<REC>(a, 2, fr, ip)? {
                        (nf, nip, CallFast::Applied) => {
                            fr = nf;
                            ip = nip;
                            continue;
                        }
                        (nf, nip, CallFast::Trace(f)) => {
                            let _ = (nf, nip);
                            return Ok(f);
                        }
                        (nf, nip, CallFast::Slow) => {
                            fr = nf;
                            ip = nip;
                            sync!();
                            self.do_call(a, 2, nret as i32 - 1)?;
                            resync!();
                        }
                    }
                }
                BCOp::ITERL | BCOp::IITERL => {
                    if !REC
                        && bc_op(ins) == BCOp::ITERL
                        && self.hot_count(ip as usize, HOTCOUNT_LOOP)
                    {
                        sync!();
                        if self.hot_loop() {
                            resync!();
                            iterl_body!(ins, a);
                            sync!();
                            return Ok(Flow::Rec);
                        }
                        resync!();
                    }
                    iterl_body!(ins, a);
                }
                BCOp::JITERL => {
                    // ITERL semantics; on loop-back enter the compiled
                    // trace (D holds the trace number, not a jump).
                    let first = fr.reg(a);
                    if !first.is_nil() {
                        fr.set(a - 1, first);
                        sync!();
                        let r = trace_exec(self.l(), self.base, bc_d(ins));
                        self.sp = self.l().stack.as_mut_ptr();
                        self.pc = r.pc;
                        if r.baseslot != 2 {
                            self.trace_exit_frame(r.baseslot);
                        }
                        if self.rec_started() {
                            return Ok(Flow::Rec); // Hot exit: record a side trace.
                        }
                        self.gc_check(self.l().frame_top)?;
                        resync!();
                    }
                }
                BCOp::VARG => {
                    sync!();
                    self.op_varg(a, bc_b(ins));
                    resync!();
                }
                // -- Bitwise ops (Lua 5.3+), lj_num2bit / lj_vm_tobit --
                BCOp::BNOT => {
                    let v = fr.reg(bc_d(ins));
                    if let Some(bits) = cdata_u64(self.l(), v) {
                        let is_ull = cdata_is_ull(v);
                        let r = make_cdata_result(self.l(), !bits, is_ull);
                        fr.set(a, r);
                    } else if v.is_number() {
                        let n = num2bit(v.num());
                        fr.set(a, LuaValue::number(!n as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BAND => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if x_cd.is_some() || y_cd.is_some() {
                        let x = x_cd.unwrap_or_else(|| num2bit(xv.num()) as u64);
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64);
                        let is_ull = cdata_is_ull(xv) || cdata_is_ull(yv);
                        let r = make_cdata_result(self.l(), x & y, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && yv.is_number() {
                        let x = num2bit(xv.num());
                        let y = num2bit(yv.num());
                        fr.set(a, LuaValue::number((x & y) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BOR => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if x_cd.is_some() || y_cd.is_some() {
                        let x = x_cd.unwrap_or_else(|| num2bit(xv.num()) as u64);
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64);
                        let is_ull = cdata_is_ull(xv) || cdata_is_ull(yv);
                        let r = make_cdata_result(self.l(), x | y, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && yv.is_number() {
                        let x = num2bit(xv.num());
                        let y = num2bit(yv.num());
                        fr.set(a, LuaValue::number((x | y) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BXOR => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if x_cd.is_some() || y_cd.is_some() {
                        let x = x_cd.unwrap_or_else(|| num2bit(xv.num()) as u64);
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64);
                        let is_ull = cdata_is_ull(xv) || cdata_is_ull(yv);
                        let r = make_cdata_result(self.l(), x ^ y, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && yv.is_number() {
                        let x = num2bit(xv.num());
                        let y = num2bit(yv.num());
                        fr.set(a, LuaValue::number((x ^ y) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BSHL => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if let Some(x) = x_cd {
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64) & 63;
                        let is_ull = cdata_is_ull(xv);
                        let r = make_cdata_result(self.l(), x << y, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && (yv.is_number() || y_cd.is_some()) {
                        let x = num2bit(xv.num());
                        let y = y_cd
                            .map_or_else(|| (num2bit(yv.num()) as u32) & 31, |v| (v as u32) & 31);
                        fr.set(a, LuaValue::number((x << y) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BSHR => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if let Some(x) = x_cd {
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64) & 63;
                        let is_ull = cdata_is_ull(xv);
                        let r = make_cdata_result(self.l(), x >> y, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && (yv.is_number() || y_cd.is_some()) {
                        let x = num2bit(xv.num());
                        let y = y_cd
                            .map_or_else(|| (num2bit(yv.num()) as u32) & 31, |v| (v as u32) & 31);
                        fr.set(a, LuaValue::number((((x as u32) >> y) as i32) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }
                BCOp::BSAR => {
                    let xv = fr.reg(bc_b(ins));
                    let yv = fr.reg(bc_c(ins));
                    let x_cd = cdata_u64(self.l(), xv);
                    let y_cd = cdata_u64(self.l(), yv);
                    if let Some(x) = x_cd {
                        let y = y_cd.unwrap_or_else(|| num2bit(yv.num()) as u64) & 63;
                        let is_ull = cdata_is_ull(xv);
                        let x_signed = x as i64;
                        let r = make_cdata_result(self.l(), (x_signed >> y) as u64, is_ull);
                        fr.set(a, r);
                    } else if xv.is_number() && (yv.is_number() || y_cd.is_some()) {
                        let x = num2bit(xv.num());
                        let y = y_cd
                            .map_or_else(|| (num2bit(yv.num()) as u32) & 31, |v| (v as u32) & 31);
                        fr.set(a, LuaValue::number((x >> y) as f64));
                    } else {
                        sync!();
                        let l = self.l();
                        l.runtime_error(b"attempt to perform arithmetic on a non-number value");
                        return Err(LuaError::Runtime);
                    }
                }

                // Explicit list (not `_`) so the match covers every opcode:
                // a wildcard shrinks the jump table and adds a bounds check
                // to every dispatch.
                other @ (BCOp::ISTYPE
                | BCOp::ISNUM
                | BCOp::TGETR
                | BCOp::TSETR
                | BCOp::FUNCF
                | BCOp::IFUNCF
                | BCOp::JFUNCF
                | BCOp::FUNCV
                | BCOp::IFUNCV
                | BCOp::JFUNCV
                | BCOp::FUNCC
                | BCOp::FUNCCW) => {
                    sync!();
                    return Err(self
                        .l()
                        .runtime_error(format!("opcode {:?} not implemented", other).as_bytes()));
                }
            }
        }
    }

    // -- Cold slow paths -------------------------------------------------

    // -- JIT hot-path detection (lj_dispatch's hotloop/hotcall) -----------

    /// Decrement the hot counter hashed from `addr` (the interpreter PC
    /// *after* fetching the counting instruction, LuaJIT's offset-by-1
    /// convention). Returns true when it underflows: the path turned hot.
    /// Does nothing while the JIT is off.
    #[inline(always)]
    fn hot_count(&mut self, addr: usize, amount: crate::jit::HotCount) -> bool {
        let js = &mut self.l().global().jit;
        js.is_on() && js.hot_decrement(addr, amount)
    }

    /// `->vm_hotloop`: the FORL/ITERL/LOOP at `self.pc - 1` (locals must be
    /// synced) turned hot. Returns true when recording started and the
    /// caller must switch to the recording dispatch.
    #[cold]
    fn hot_loop(&mut self) -> bool {
        let pt = self.lua_cl().proto;
        let pc = self.pc - 1;
        let base = self.base;
        trace_hot(self.l(), base, pt, pc);
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// `->vm_hotcall`: the FUNCF header of `pt` turned hot. Same contract
    /// as `hot_loop`; the frame must already be entered and synced.
    #[cold]
    fn hot_call(&mut self, pt: GcPtr<Proto>) -> bool {
        let base = self.base;
        trace_hot(self.l(), base, pt, 0);
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// Did a trace exit just start a side-trace recording? The caller
    /// must then switch to the recording dispatch (`self.pc` is already
    /// at the exit's resume point).
    #[inline]
    fn rec_started(&mut self) -> bool {
        self.l().global().jit.state == crate::jit::TraceState::Record
    }

    /// A trace exited inside an inlined call frame: shift the base to
    /// the innermost frame (its slots — including the function and the
    /// frame link — were restored from the snapshot) and reload the
    /// interpreter for that frame's closure.
    #[cold]
    #[inline(never)]
    fn trace_exit_frame(&mut self, baseslot: usize) {
        self.base += baseslot - 2;
        self.reload_at(Frame::new(self.sp, self.base));
        self.l().top = self.base + self.proto().framesize as usize;
        self.l().frame_top = self.l().top;
    }

    /// `lj_gc_check` + `lj_gc_step_fixtop`: run a collection if the
    /// allocation debt is due. Only called from safe points (before an
    /// allocating opcode, with locals synced): the marker sees every live
    /// object through the stacks and roots. `need` is the live register
    /// top of the allocating instruction (`base + regs`); the collector
    /// marks the stack up to it only, so dead temp slots above it cannot
    /// keep weak-table values alive.
    ///
    /// Runs pending `__gc` finalizers too: an error raised by one
    /// propagates from this allocating instruction, i.e. from inside any
    /// enclosing `pcall` (mirroring LuaJIT, where it surfaces from the
    /// allocation that triggered the collection).
    #[inline]
    fn gc_check(&mut self, need: usize) -> LuaResult<()> {
        let g = self.l().global();
        // Drain pending finalizers first: while the collector sits in
        // `Finalize`, every `gc_step` is a no-op, so if we skipped this the
        // heap would never be collected again — an allocation loop under a
        // `__gc` finalizer chain leaks indefinitely (the `while gcinfo()`
        // loop in gc.lua). Only the VM can run finalizers (they need a Lua
        // frame), so this is the single funnel.
        if g.heap.gc_state == crate::runtime::gc::GcState::Finalize {
            run_finalizers(self.l())?;
        }
        let g = self.l().global();
        // lj_gc_check: only `total >= threshold` triggers a step. The step
        // itself is budget-limited (`gc_step`), never a full cycle.
        if g.heap.should_collect() {
            self.gc_collect(need)?;
        }
        Ok(())
    }

    #[cold]
    fn gc_collect(&mut self, need: usize) -> LuaResult<()> {
        let l = self.l();
        // lj_gc_step_fixtop: the marker must see every live object, and
        // the atomic clear must not touch live registers. The frame
        // extends to `frame_top`; C-call results transiently lower `top`
        // while frame temporaries above it are still live, so never drop
        // below the frame extent.
        l.top = l.top.max(l.frame_top).max(need);
        let stopped = l.global().heap.gc_stopped;
        if stopped {
            return Ok(());
        }
        let g = l.global();
        if g.heap.gc_state == crate::runtime::gc::GcState::Pause {
            crate::gc::start_gc_cycle(g);
        }
        // One budget-limited incremental step (LuaJIT `lj_gc_step`). The
        // collector paces itself via debt: a fast allocation loop accrues
        // debt and triggers the next step sooner, eventually finishing the
        // cycle; it is never driven to completion here.
        let size = g.heap.gc_step_size;
        crate::gc::gc_step(&mut g.heap, size);
        Ok(())
    }

    // -- Cold opcode bodies ---------------------------------------------
    // The interpreter arms for these opcodes are inherently cold (they
    // allocate, re-enter the VM or walk frames), so their bodies live in
    // `#[cold]` methods instead of being inlined into the dispatch. The
    // caller is always synced (`sync!` before, `resync!` after).

    /// BC_CAT: string concatenation through the `__concat` metamethod path.
    #[cold]
    #[inline(never)]
    fn op_cat(&mut self, a: u32, b: u32, c: u32) -> LuaResult<LuaValue> {
        self.gc_check(self.base + a as usize + 1)?;
        self.meta_cat(b, c)
    }

    /// BC_TDUP: clone the template table in `kgc[d]`.
    #[cold]
    #[inline(never)]
    fn op_tdup(&mut self, a: u32, d: u32) -> LuaResult<LuaValue> {
        self.gc_check(self.base + a as usize + 1)?;
        let templ = match &self.proto().kgc[d as usize] {
            KGc::Table(t) => t.dup(),
            KGc::TableRef(t) => t.as_ref().dup(),
            _ => unreachable!("expected template table"),
        };
        Ok(LuaValue::table(self.l().heap().alloc_table(templ)))
    }

    /// BC_UCLO: close open upvalues at or above the register `a`.
    #[cold]
    #[inline(never)]
    fn op_uclo(&mut self, a: u32) {
        self.close_upvals(self.base + a as usize);
    }

    /// BC_FNEW: allocate a new closure from the prototype in `kgc[d]`.
    #[cold]
    #[inline(never)]
    fn op_fnew(&mut self, a: u32, d: u32) -> LuaResult<LuaValue> {
        self.gc_check(self.base + a as usize + 1)?;
        Ok(self.new_closure(d))
    }

    /// BC_VARG: copy the varargs into `a..`, per BC_VARG. The varargs sit
    /// between the vararg frame and the frame below it; their extent is
    /// recovered from the FRAME_VARG delta, LuaJIT-style.
    #[cold]
    #[inline(never)]
    fn op_varg(&mut self, a: u32, b: u32) {
        let base = self.base;
        let link = self.at(base - 1).to_bits();
        if link & FRAME_TYPE_MASK != FRAME_VARG {
            return;
        }
        let delta = (link >> 3) as usize;
        if delta < 2 {
            self.multres = 0;
            return;
        }
        let numparams = self.proto().numparams as usize;
        let nvarg = (delta - 2).saturating_sub(numparams);
        let dst = a as usize;
        let need = base + dst + nvarg + 8;
        if need > self.l().stack.len() {
            let l = self.l();
            l.stack_ensure(need);
            self.sp = l.stack.as_mut_ptr();
        }
        let src_base = base - delta + numparams;
        if b == 0 {
            for i in 0..nvarg {
                self.set_at(base + dst + i, self.at(src_base + i));
            }
            self.multres = nvarg;
            self.l().top = base + dst + nvarg;
        } else {
            let want = (b - 1) as usize;
            for i in 0..want {
                self.set_at(
                    base + dst + i,
                    if i < nvarg {
                        self.at(src_base + i)
                    } else {
                        LuaValue::NIL
                    },
                );
            }
        }
    }

    #[cold]
    #[allow(dead_code)]
    fn len_op(&self, v: LuaValue) -> LuaResult<LuaValue> {
        if let Some(sid) = v.as_string_id() {
            let n = self.l().heap().strings.get(sid).len();
            return Ok(LuaValue::number(n as f64));
        }
        if let Some(t) = v.as_table() {
            return Ok(LuaValue::number(t.as_ref().len() as f64));
        }
        Err(self
            .l()
            .runtime_error(b"attempt to get length of a non-table value"))
    }

    #[cold]
    fn tsetm(&mut self, a: u32, d: u32) -> LuaResult<()> {
        let t = self.at(self.base + a as usize - 1);
        let base_key = self.knum(d).num() as i64 - (1i64 << 52);
        let tab = match t.as_table() {
            Some(t) => t,
            None => {
                return Err(self
                    .l()
                    .runtime_error(b"attempt to index a non-table value"));
            }
        };
        // Pre-size array: we know the keys are base_key .. base_key+multres-1.
        // For a fresh table, this avoids per-value hash insertions.
        if self.multres > 0 && base_key == 1 {
            let need = (base_key as u32).wrapping_add(self.multres as u32);
            tab.as_mut().reasize(need);
        }
        for i in 0..self.multres {
            let key = base_key + i as i64;
            let v = self.at(self.base + a as usize + i);
            if key >= 0 && key <= i32::MAX as i64 {
                tab.as_mut().set_int(key as i32, v);
            } else {
                tab.as_mut().set(LuaValue::number(key as f64), v);
            }
        }
        Ok(())
    }

    // -- Calls -----------------------------------------------------------

    /// `sync!` for a helper that receives the frame/ip as parameters.
    #[inline(always)]
    fn sync_fr(&mut self, fr: Frame, ip: *const BCIns) {
        self.base = fr.cur_base();
        self.pc = unsafe { ip.offset_from(self.bcp) as usize };
        self.l().debug_pc = self.pc;
        self.l().base = self.base;
    }

    /// `resync!` for a helper: rebuild the frame/ip locals from the synced
    /// fields (returns them by value so the loop can keep them in
    /// registers).
    #[inline(always)]
    fn resync_fr(&mut self) -> (Frame, *const BCIns) {
        (Frame::new(self.sp, self.base), unsafe {
            self.bcp.add(self.pc)
        })
    }

    /// Inline fast path shared by BC_CALL and BC_CALLM (LuaJIT's
    /// `ins_call`): a plain (non-vararg) Lua callee switches frames right
    /// here — one store for the frame link, no sync round-trip. C callees
    /// and vararg protos fall back to `do_call`.
    #[allow(clippy::too_many_arguments)]
    fn call_lua_fast<const REC: bool>(
        &mut self,
        a: u32,
        nargs: usize,
        mut fr: Frame,
        mut ip: *const BCIns,
    ) -> LuaResult<(Frame, *const BCIns, CallFast)> {
        let f = fr.reg(a);
        let Some(gf) = f.as_func() else {
            return Ok((fr, ip, CallFast::Slow));
        };
        let GcFunc::Lua(cl) = gf.as_ref() else {
            return Ok((fr, ip, CallFast::Slow));
        };
        // "call" hook event (Lua 5.1 fires it only for Lua callees, not C
        // functions — a C call like collectgarbage() must not trip it).
        if self.l().hookmask & HOOKMASK_CALL != 0 && !self.l().hook_active {
            self.sync_fr(fr, ip);
            self.hook_event("call")?;
            (fr, ip) = self.resync_fr();
        }
        let ptref = cl.proto;
        let pt = cl.proto.as_ref();
        let fs = pt.framesize as usize;
        let numparams = pt.numparams as usize;
        let callbase = fr.cur_base() + a as usize + 2;
        let is_vararg = (pt.flags & PROTO_VARARG) != 0;
        // FUNCV shifts the fixed params up past the varargs, so its frame
        // base sits above the vararg area; FUNCF reuses the call slot.
        let newbase = if is_vararg {
            callbase + nargs + 2
        } else {
            callbase
        };
        let need = if is_vararg {
            newbase + numparams + fs + 16
        } else {
            callbase + fs + 8
        };
        // LuaJIT `lj_checkstack`: unbounded Lua recursion
        // (e.g. `function y() y() end`) must raise a Lua error, not grow
        // the stack past its limit.
        if need > self.l().max_stack() {
            self.sync_fr(fr, ip);
            return Err(self.l().runtime_error(b"stack overflow"));
        }
        if need > self.l().stack.len() {
            self.sync_fr(fr, ip);
            self.l().stack_ensure(need);
            self.sp = self.l().stack.as_mut_ptr();
            let (_, nip) = self.resync_fr();
            ip = nip;
        }
        // The caller's frame link (return PC) always sits at `callbase - 1`;
        // a vararg callee chains its FRAME_VARG link on top of it.
        self.set_at(callbase - 1, LuaValue::from_bits(ip as u64));
        if is_vararg {
            // FUNCV: shift the fixed params up past the varargs and chain a
            // vararg frame back to the one holding the real link.
            self.set_at(newbase - 2, LuaValue::func(gf));
            for i in 0..numparams {
                let v = if i < nargs {
                    self.at(callbase + i)
                } else {
                    LuaValue::NIL
                };
                self.set_at(newbase + i, v);
            }
            let delta = (newbase - callbase) as u64;
            self.set_at(newbase - 1, LuaValue::from_bits((delta << 3) | FRAME_VARG));
            self.base = newbase;
            // Set TOP to the new frame before anything below may run GC
            // (alloc_table for the implicit `arg` table steps the collector,
            // which clears slots above TOP — the just-copied params would be
            // wiped while TOP still pointed at the caller's frame).
            self.l().top = newbase + fs;
            self.l().frame_top = self.l().top;
            // Lua 5.1 LUA_COMPAT_VARARG: build the implicit `arg` local
            // ({varargs..., n = count}) unless the body uses `...` itself.
            if (pt.flags & PROTO_VARARG_NEEDSARG) != 0 {
                let nvar = nargs.saturating_sub(numparams);
                let l = self.l();
                let tab = l
                    .heap()
                    .alloc_table(crate::table::LuaTable::new(nvar as u32, 1));
                for i in 0..nvar {
                    tab.as_mut()
                        .set_int(i as i32 + 1, self.at(callbase + numparams + i));
                }
                let nsid = l.heap().intern(b"n");
                tab.as_mut()
                    .set(l.heap().str_value(nsid), LuaValue::number(nvar as f64));
                self.set_at(newbase + numparams, LuaValue::table(tab));
            } else {
                self.set_at(newbase + numparams, LuaValue::NIL);
            }
        } else {
            for i in nargs..numparams {
                self.set_at(callbase + i, LuaValue::NIL);
            }
            self.base = callbase;
        }
        fr = Frame::new(self.sp, self.base);
        ip = unsafe { pt.bc.as_ptr().add(1) };
        self.cl = gf;
        self.bcp = pt.bc.as_ptr();
        self.knp = pt.kn.as_ptr();
        self.ksp = pt.kstrv.as_ptr();
        self.l().top = fr.cur_base() + fs;
        self.l().frame_top = self.l().top;
        let head = pt.bc[0];
        // A compiled callee (JFUNCF): enter its trace from the fresh frame.
        if !REC && bc_op(head) == BCOp::JFUNCF {
            self.sync_fr(fr, ip);
            let r = trace_exec(self.l(), self.base, bc_d(head));
            self.sp = self.l().stack.as_mut_ptr();
            if r.stack_overflow {
                return Err(self.l().runtime_error(b"stack overflow"));
            }
            self.pc = r.pc;
            if r.baseslot != 2 {
                self.trace_exit_frame(r.baseslot);
            } else {
                fr = Frame::new(self.sp, self.base);
                self.reload_at(fr);
                self.l().top = self.base + self.proto().framesize as usize;
            }
            if self.rec_started() {
                return Ok((fr, ip, CallFast::Trace(Flow::Rec))); // Hot exit side trace.
            }
            self.gc_check(self.l().frame_top)?;
            (fr, ip) = self.resync_fr();
            return Ok((fr, ip, CallFast::Applied));
        }
        // hotcall (vm_hotcall): count the FUNCF header.
        if !REC && bc_op(head) == BCOp::FUNCF && self.hot_count(ip as usize, HOTCOUNT_CALL) {
            self.sync_fr(fr, ip);
            if self.hot_call(ptref) {
                return Ok((fr, ip, CallFast::Trace(Flow::Rec))); // Record from callee.
            }
            (fr, ip) = self.resync_fr();
        }
        Ok((fr, ip, CallFast::Applied))
    }

    #[inline(never)]
    fn do_call(&mut self, a: u32, nargs: usize, want: i32) -> LuaResult<()> {
        let func_slot = self.base + a as usize;
        let mut nargs = nargs;
        let f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                // lj_meta_call: inject __call metamethod.
                nargs = meta::meta_call(self.l(), func_slot, nargs)?;
                let f = self.at(func_slot);
                f.as_func().expect("__call did not produce a function")
            }
        };
        match gf.as_ref() {
            GcFunc::Lua(_) => {
                // Frame link = the return PC; `want` is re-read from the
                // CALL/CALLM/ITERC instruction at `pc[-1]` on return.
                let link = unsafe { self.bcp.add(self.pc) } as u64;
                debug_assert!((link & FRAME_TYPE_MASK) == FRAME_LUA);
                if self.l().hookmask != 0 {
                    self.hook_lines.push(self.l().hook_line);
                }
                self.enter_lua(gf, func_slot, nargs, link);
                Ok(())
            }
            GcFunc::C(cc) => {
                let f = cc.f;
                let n = match self.call_c_inline(f, func_slot, nargs) {
                    Ok(n) => n,
                    Err(LuaError::Yield) => return Err(self.suspend_call(func_slot, want)),
                    Err(e) => return Err(e),
                };
                if want >= 0 {
                    for i in n..(want as usize) {
                        self.set_at(func_slot + i, LuaValue::NIL);
                    }
                } else {
                    self.multres = n;
                }
                Ok(())
            }
        }
    }

    /// A C function called from a Lua frame yielded (`coroutine.yield`):
    /// capture the resume point. Yield values move to `func_slot`.
    #[cold]
    #[inline(never)]
    fn suspend_call(&mut self, func_slot: usize, want: i32) -> LuaError {
        let ny = self.l().nyield as usize;
        // A yield through pcall/xpcall rewrote the suspend with the
        // protected flag and moved the yield values to *its* slot; the
        // outer capture must keep both.
        let protected = matches!(
            self.l().suspend,
            Suspend::Call {
                protected: true,
                ..
            }
        );
        // A yield through pcall/xpcall: the yield values were moved to the
        // inner yield call's slot (the recorded value_slot).
        let (src, value_slot) = if protected {
            match self.l().suspend {
                Suspend::Call { value_slot: vs, .. } => (vs, func_slot),
                _ => (func_slot + 2, func_slot),
            }
        } else {
            (func_slot + 2, func_slot)
        };
        for i in 0..ny {
            let v = self.at(src + i);
            self.set_at(func_slot + i, v);
        }
        let l = self.l();
        l.suspend = Suspend::Call {
            pc: self.pc,
            cl: self.cl,
            base: self.base,
            slot: func_slot,
            want,
            protected,
            value_slot,
        };
        l.top = (self.base + self.proto().framesize as usize).max(func_slot + ny);
        l.base = self.base;
        LuaError::Yield
    }

    /// Same for a yield through a tail call (`return coroutine.yield(...)`).
    #[cold]
    fn suspend_return(&mut self, func_slot: usize) -> LuaError {
        let ny = self.l().nyield as usize;
        for i in 0..ny {
            let v = self.at(func_slot + 2 + i);
            self.set_at(func_slot + i, v);
        }
        let l = self.l();
        l.suspend = Suspend::Return {
            base: self.base,
            slot: func_slot,
        };
        l.top = (self.base + self.proto().framesize as usize).max(func_slot + ny);
        l.base = self.base;
        LuaError::Yield
    }

    fn call_c_inline(
        &mut self,
        f: crate::func::CFunction,
        func_slot: usize,
        nargs: usize,
    ) -> LuaResult<usize> {
        let args_base = func_slot + 2;
        // self.l() returns &mut LuaState, but we hold it as a raw pointer
        // so we can update self.sp mid-function if stack_ensure reallocs.
        let lp = self as *mut Interp;
        let l = self.l();
        l.stack_ensure(args_base + nargs + 8);
        let saved_base = l.base;
        let saved_top = l.top;
        // Set a frame link so error walkers can find the caller's Lua frame.
        l.stack[args_base - 1] = LuaValue::from_bits(((saved_base as u64) << 3) | FRAME_LUA);
        l.base = args_base;
        l.top = args_base + nargs;
        // C-call boundary is a GC safe point (args anchored, frames below).
        let (paused, collect) = {
            let g = l.global();
            (
                g.heap.gc_state == crate::runtime::gc::GcState::Pause,
                g.heap.should_collect() && !g.heap.gc_stopped,
            )
        };
        if collect {
            // lj_gc_step_fixtop: protect the caller's frame locals from
            // the collector's slot clearing (top was lowered to the arg
            // area for this C call).
            l.top = l.top.max(l.frame_top);
            if paused {
                crate::gc::start_gc_cycle(l.global());
            }
            crate::gc::gc_step(&mut l.global().heap, l.global().heap.gc_step_size);
            l.top = args_base + nargs;
        }
        let r = f(l);
        let n = match r {
            Ok(nv) => nv as usize,
            Err(e) => {
                l.base = saved_base;
                l.top = saved_top;
                return Err(e);
            }
        };
        // C function may return more results than nargs; ensure room and
        // refresh sp in case any internal stack_ensure reallocated the Vec.
        l.stack_ensure((func_slot + n + 8).max(args_base + nargs + 8));
        unsafe {
            (*lp).sp = l.stack.as_mut_ptr();
        }
        for i in 0..n {
            l.stack[func_slot + i] = l.stack[args_base + i];
        }
        l.base = saved_base;
        l.top = saved_top;
        Ok(n)
    }

    /// True tail call, per LuaJIT's BC_CALLT: the callee *replaces* the
    /// current frame. Func and args move down to this frame's slots, the
    /// frame link stays untouched, and no Rust recursion happens, so tail
    /// recursion runs in constant stack. A tail call from a vararg function
    /// first drops its vararg frame (relocating to the frame that holds the
    /// real link). Returns `Some(n)` only when a C callee finishes a host
    /// (`FRAME_C`) frame.
    fn do_tailcall(&mut self, a: u32, nargs: usize) -> LuaResult<Option<usize>> {
        let func_slot = self.base + a as usize;
        let mut nargs = nargs;
        let mut f = self.at(func_slot);
        let gf = match f.as_func() {
            Some(p) => p,
            None => {
                nargs = meta::meta_call(self.l(), func_slot, nargs)?;
                f = self.at(func_slot);
                f.as_func().expect("__call did not produce a function")
            }
        };

        // C callee: call inline (no c_depth bump) so yields propagate.
        if let GcFunc::C(cc) = gf.as_ref() {
            let r = call_c(self.l(), cc.f, func_slot, nargs, -1);
            return match r {
                Ok(n) => Ok(self.do_return(func_slot, n)),
                Err(LuaError::Yield) => Err(self.suspend_return(func_slot)),
                Err(e) => Err(e),
            };
        }

        let mut base = self.base;
        let link = self.at(base - 1).to_bits();
        if link & FRAME_TYPE_MASK == FRAME_VARG {
            let delta = (link >> 3) as usize;
            if base >= delta + 2 {
                base -= delta;
            } else {
                // Underflow: fall back.
                let n = execute(self.l(), func_slot, nargs, -1)?;
                return Ok(self.do_return(func_slot, n));
            }
        }
        // Move func and args down into the (possibly relocated) frame.
        // dest [base, base+nargs) ⊂ src [func_slot+2, ...): forward order
        // is safe (each source slot is read before it is overwritten).
        for i in 0..nargs {
            self.set_at(base + i, self.at(func_slot + 2 + i));
        }
        self.set_at(base - 2, f);

        match gf.as_ref() {
            GcFunc::Lua(cl) => {
                let pt = cl.proto.as_ref();
                if (pt.flags & PROTO_VARARG) != 0 {
                    let link = self.at(base - 1).to_bits();
                    self.enter_lua(gf, base - 2, nargs, link);
                } else {
                    for i in nargs..pt.numparams as usize {
                        self.set_at(base + i, LuaValue::NIL);
                    }
                    self.base = base;
                    self.cl = gf;
                    self.bcp = pt.bc.as_ptr();
                    self.knp = pt.kn.as_ptr();
                    self.ksp = pt.kstrv.as_ptr();
                    self.pc = 1;
                    self.l().top = base + pt.framesize as usize;
                }
                Ok(None)
            }
            GcFunc::C(_cc) => unreachable!("C path handled above"),
        }
    }

    /// Move `n` results to the caller's slot, restore the caller frame and
    /// continue. Everything is recovered from the frame links in the stack,
    /// as in LuaJIT's BC_RET: the caller base, wanted results and result
    /// slot all come from the CALL instruction at the stored return PC.
    /// Returns `Some(n)` when a host (`FRAME_C`) entry returns.
    #[inline(never)]
    fn do_return(&mut self, src: usize, n: usize) -> Option<usize> {
        // sp and the frame link at self.base-1 may be stale from a C-call
        // stack resize (push -> stack_ensure) inside a JIT helper or the
        // interpreter.  Re-read both from the canonical LuaState stack Vec.
        self.sp = self.l().stack.as_mut_ptr();
        if !self.l().openuv.is_empty() {
            self.close_upvals(self.base);
        }
        let mut base = self.base;
        if base < 2 {
            return None;
        }
        let mut link = self.l().stack[base - 1].to_bits();
        // A NIL link means we cannot determine the caller. Bail out so the
        // interpreter can continue with the next opcode (a Lua-level error
        // will surface if the state is too corrupted).
        if link == u64::MAX {
            return None;
        }
        while (link & FRAME_TYPE_MASK) == FRAME_VARG {
            let sz = (link >> 3) as usize;
            if sz == 0 || sz > base {
                break;
            }
            base -= sz;
            if base < 2 {
                break;
            }
            link = self.l().stack[base - 1].to_bits();
            if link == u64::MAX {
                return None;
            }
        }
        let dst = base - 2; // results always land at the callee's func slot
        for i in 0..n {
            self.set_at(dst + i, self.at(src + i));
        }
        self.multres = n;

        if (link & FRAME_TYPE_MASK) == FRAME_CONT {
            return self.cont_dispatch(base, link, n);
        }

        if link & FRAME_TYPE_MASK == FRAME_C {
            let want = ((link >> 3) as i32) - 1;
            // The custom xpcall-handler link encodes the failed frame's
            // base in the delta, so the apparent want may exceed the
            // stack: clamp the fill to what is actually there.
            let cap = self.l().stack.len().saturating_sub(dst);
            let want = want.min(cap as i32);
            let got = if want >= 0 {
                for i in n..(want as usize) {
                    self.set_at(dst + i, LuaValue::NIL);
                }
                want as usize
            } else {
                n
            };
            // Clear the C frame's dead slots above the results so stale
            // references never reach the next cycle (the atomic clear is
            // bounded by frame_top to protect live Lua locals). Never go
            // below the callee's base (dst+2): slots dst..dst+2 belong to
            // the caller (the func slot and the first argument positions
            // are the caller's registers).
            let top_now = self.l().top;
            let keep = (dst + got).min(top_now);
            let clear_from = (dst + 2).max(keep).min(top_now);
            for s in self.l().stack[clear_from..top_now].iter_mut() {
                *s = LuaValue::NIL;
            }
            self.l().top = dst + got;
            return Some(got);
        }

        if ((link >> 3) as usize) < self.l().stack.len() {
            // Base-encoded link (host-fabricated frame: C call, debug
            // hook): return to the caller without touching the
            // interpreter state — this run ends here. Note that the
            // encoding is *not* unique to C-call frames (the debug hook
            // frames fabricated by `call_hook` also carry it), so we must
            // keep the size test rather than keying off the frame's
            // function type.
            self.l().top = dst + n;
            return Some(n);
        }
        // FRAME_LUA: the link is the caller's return PC.
        let ret_ip = link as *const BCIns;
        let call_ins = unsafe { *ret_ip.sub(1) };
        let caller_base = dst - bc_a(call_ins) as usize;
        let want = bc_b(call_ins) as i32 - 1;
        // Callee frame extent (before reload switches proto to the caller).
        let callee_top = dst + 2 + self.proto().framesize as usize;

        if !self.hook_lines.is_empty() {
            let _ = self.hook_lines.pop();
        }
        self.base = caller_base;
        let cl = self.at(caller_base - 2).as_func().unwrap();
        self.reload(cl);
        self.pc = unsafe { ret_ip.offset_from(self.bcp) as usize };
        // Restore the caller's frame extent: the callee's CALL fast path
        // set `frame_top` to the callee frame, and a callee returning
        // through do_return (RET0/RET1 cold path, e.g. open upvalues)
        // would otherwise leave it at the callee's (possibly smaller)
        // extent, letting later GC marks/clears under-protect the
        // caller's live registers.
        self.l().frame_top = caller_base + self.proto().framesize as usize;
        // No line event on the caller's resumption line.
        if self.l().hookmask & HOOKMASK_LINE != 0 {
            let pt = self.proto();
            let pc = self.pc.min(pt.lines.len().saturating_sub(1));
            self.l().hook_line = pt.lines[pc];
        }

        let keep = dst
            + if want >= 0 {
                for i in n..(want as usize) {
                    self.set_at(dst + i, LuaValue::NIL);
                }
                want as usize
            } else {
                n
            };
        // Clear the callee frame's dead slots above the results (stale
        // references must not survive into the next GC cycle). Never go
        // below the callee's base (dst+2): with 0/1 results `keep` sits in
        // the caller's register area, which may still hold live values
        // (e.g. the pre-loaded callee/args of an enclosing call).
        let clear_from = (dst + 2).max(keep);
        let hi = callee_top.max(clear_from);
        for s in self.l().stack[clear_from..hi].iter_mut() {
            *s = LuaValue::NIL;
        }
        self.l().top = keep;
        None
    }

    // -- Upvalues / closures ---------------------------------------------

    fn upval_get(&self, uv: GcPtr<Upval>) -> LuaValue {
        uv.as_ref().get()
    }

    fn upval_set(&self, uv: GcPtr<Upval>, v: LuaValue) {
        uv.as_mut().set(v);
    }

    /// Find or create an open upvalue for the stack slot `slot` (absolute
    /// index into this thread's stack). Identity is by value pointer,
    /// exactly like `lj_func_finduv`.
    fn find_upval(&mut self, slot: usize) -> GcPtr<Upval> {
        let ptr = unsafe { self.sp.add(slot) };
        for &uv in self.l().openuv.iter() {
            if uv.as_ref().value_ptr() == ptr {
                return uv;
            }
        }
        let nn = std::ptr::NonNull::new(ptr).unwrap();
        let uv = self.l().heap().alloc_upval(Upval::new_open(nn, false));
        self.l().openuv.push(uv);
        uv
    }

    /// Close every open upvalue at or above stack `level` (absolute index),
    /// per `lj_func_closeuv`.
    fn close_upvals(&mut self, level: usize) {
        let level_ptr = unsafe { self.sp.add(level) } as *const LuaValue;
        let l = self.l();
        let mut i = 0;
        while i < l.openuv.len() {
            let uv = l.openuv[i];
            if uv.as_ref().value_ptr() as *const LuaValue >= level_ptr {
                uv.as_mut().close();
                l.openuv.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn new_closure(&mut self, d: u32) -> LuaValue {
        let proto = match &self.proto().kgc[d as usize] {
            KGc::ProtoRef(p) => *p,
            _ => unreachable!("FNEW expects a registered child prototype"),
        };
        let pt = proto.as_ref();
        let nuv = pt.uv.len();
        // Fast path: no upvalues — avoid cloning the parent's upvalue
        // vector (a hot `function() ... end` loop allocates nothing here).
        if nuv == 0 {
            let env = self.lua_cl().env;
            let fref = self.l().heap().alloc_func(GcFunc::Lua(LuaClosure {
                proto,
                env,
                upvals: crate::func::Upvals::empty(),
            }));
            return LuaValue::func(fref);
        }
        let env = self.lua_cl().env;
        // The parent closure lives at a stable heap address; borrow its
        // upvalue vector through a raw pointer so the mutable find_upval
        // below doesn't conflict (no clone — a hot factory inherits the
        // same cells every iteration).
        let parent = self.lua_cl() as *const LuaClosure;
        let mut upvals = crate::func::Upvals::empty();
        for i in 0..nuv {
            let v = pt.uv[i];
            if (v & PROTO_UV_LOCAL) != 0 {
                let slot = self.base + (v & 0xff) as usize;
                let uv = self.find_upval(slot);
                if (v & PROTO_UV_IMMUTABLE) != 0 {
                    uv.as_mut().immutable = true;
                }
                upvals.push(uv);
            } else {
                let inherited = unsafe { (*parent).upvals.get((v & 0xff) as usize) };
                upvals.push(inherited.copied().unwrap());
            }
        }
        let fref = self
            .l()
            .heap()
            .alloc_func(GcFunc::Lua(LuaClosure { proto, env, upvals }));
        LuaValue::func(fref)
    }
}

/// The three primitive values, indexed by KPRI/ISEQP operand (0/1/2).
const PRI: [LuaValue; 3] = [LuaValue::NIL, LuaValue::FALSE, LuaValue::TRUE];

/// Numeric coercion for `for` loop bounds (LuaJIT's `lj_tonumber` at the
/// FORI dispatch): numbers pass through, strings are parsed.
fn for_number(l: &LuaState, v: LuaValue) -> Option<f64> {
    if v.is_number() {
        return Some(v.num());
    }
    if let Some(sid) = v.as_string_id() {
        let bytes = l.global().heap.strings.get(sid).to_vec();
        return crate::strscan::scan_number(&bytes);
    }
    None
}

/// Raw equality used by ISEQ*/ISNE*: numbers compare by value (so `-0.0` and
/// NaN behave), everything else by bit pattern (interned strings and GC
/// pointers compare by identity).
#[inline(always)]
fn val_eq(l: &LuaState, a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else if let (Some(ca), _) = (a.as_cdata(), b.is_number()) {
        if b.is_number() {
            // Numeric value match, or the raw low-32 pattern when the
            // number only matches the truncated bits (unsigned small
            // types compared against a large Lua number).
            if let Some(cn) = crate::stdlib::cdata_to_number(ca.as_ref())
                && cn == b.num()
            {
                return true;
            }
            if let Some(bits) = cdata_u64(l, a) {
                let bv = num2bit(b.num()) as u64;
                return bits == bv;
            }
            return false;
        }
        if let Some(cb) = b.as_cdata() {
            // Numeric cdata compare by value (int vs uint of the same
            // magnitude are equal).
            let ca_num = crate::stdlib::cdata_to_number(ca.as_ref());
            let cb_num = crate::stdlib::cdata_to_number(cb.as_ref());
            if let (Some(a), Some(bn)) = (ca_num, cb_num) {
                return a == bn;
            }
            if ca.as_ref().ctypeid != cb.as_ref().ctypeid {
                return false;
            }
            // Pointer/array cdata compare by storage address (aliases of
            // the same storage compare by their offsets).
            if let Some(eq) = cdata_ptr_eq(l, a, b) {
                return eq;
            }
            return ca.as_ref().data == cb.as_ref().data;
        }
        a.to_bits() == b.to_bits()
    } else if a.is_number() && b.is_cdata() {
        if let Some(cb) = b.as_cdata() {
            if let Some(cn) = crate::stdlib::cdata_to_number(cb.as_ref())
                && cn == a.num()
            {
                return true;
            }
            if let Some(bits) = cdata_u64(l, b) {
                let av = num2bit(a.num()) as u64;
                return bits == av;
            }
            return false;
        }
        a.to_bits() == b.to_bits()
    } else {
        a.to_bits() == b.to_bits()
    }
}

/// `x ^ y` with a small-integer-exponent fast path (`lj_vm_powi`).
#[inline]
pub(crate) fn vm_pow(mut x: f64, y: f64) -> f64 {
    let k = y as i32;
    if k as f64 == y && k.unsigned_abs() <= 65536 {
        if k >= 1 {
            let mut n = k as u32;
            while n & 1 == 0 {
                x *= x;
                n >>= 1;
            }
            let mut z = x;
            n >>= 1;
            while n != 0 {
                x *= x;
                if n & 1 != 0 {
                    z *= x;
                }
                n >>= 1;
            }
            z
        } else if k == 0 {
            1.0
        } else {
            1.0 / x.powf(-k as f64)
        }
    } else {
        x.powf(y)
    }
}

/// Resume a coroutine suspended via `Suspend::Call`. Rebuilds the Interp
/// from the saved state and re-enters the dispatch loop.
#[allow(clippy::too_many_arguments)]
pub fn resume_continue(
    co: &mut LuaState,
    slot: usize,
    want: i32,
    nargs: usize,
    pc: usize,
    cl: GcPtr<GcFunc>,
    sbase: usize,
    protected: bool,
) -> LuaResult<usize> {
    co.c_depth += 1;
    // Note: no stack_ensure needed — the suspended frame was alive at
    // yield time and stack length never shrinks.
    let args_at = slot + 2;
    if protected {
        // The yield happened inside pcall/xpcall: the continuation reads
        // `true, <resume args>` as the protected call's results.
        co.stack[slot] = LuaValue::TRUE;
        for i in 0..nargs {
            co.stack[slot + 1 + i] = co.stack[args_at + i];
        }
    } else if want >= 0 {
        let limit = nargs.min(want as usize);
        for i in 0..limit {
            co.stack[slot + i] = co.stack[args_at + i];
        }
        for i in limit..(want as usize) {
            co.stack[slot + i] = LuaValue::NIL;
        }
    } else {
        for i in 0..nargs {
            co.stack[slot + i] = co.stack[args_at + i];
        }
    }
    let mut vm = Interp::new(co);
    if want < 0 {
        vm.multres = if protected { nargs + 1 } else { nargs };
    }
    vm.base = sbase;
    vm.cl = cl;
    vm.reload(cl);
    vm.pc = pc;
    let pt = match cl.as_ref() {
        GcFunc::Lua(c) => c.proto.as_ref(),
        _ => unreachable!(),
    };
    co.top = sbase + pt.framesize as usize;
    co.frame_top = co.top;
    co.base = sbase;
    co.status = crate::state::CoStatus::Running;
    let r = vm.run();
    co.c_depth -= 1;
    r
}

/// Finish a coroutine suspended via `Suspend::Return`. Delivers resume
/// args as a return from the saved frame, like `do_return`; if the return
/// lands in a Lua frame, the dispatch loop continues running.
pub fn resume_finish(
    co: &mut LuaState,
    slot: usize,
    nargs: usize,
    sbase: usize,
) -> LuaResult<usize> {
    co.c_depth += 1;
    let link = co.stack[slot + 1].to_bits();
    if (link & FRAME_TYPE_MASK) != FRAME_C && (link & FRAME_TYPE_MASK) != FRAME_LUA {
        co.stack[slot + 1] = LuaValue::from_bits(FRAME_C);
    }
    for i in 0..nargs {
        co.stack[slot + i] = co.stack[slot + 2 + i];
    }
    let mut vm = Interp::new(co);
    vm.base = sbase;
    // The suspended tail-call frame is still the current function; load
    // its proto so `do_return` can compute the callee extent (its
    // `self.proto()` is read before the caller reload).
    if let Some(cl) = co.stack[sbase.saturating_sub(2)].as_func() {
        vm.reload(cl);
    }
    co.status = crate::state::CoStatus::Running;
    let r = match vm.do_return(slot, nargs) {
        Some(n) => Ok(n),
        None => vm.run(),
    };
    co.c_depth -= 1;
    r
}
