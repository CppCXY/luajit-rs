//! Portable trace execution: a reference backend that runs a trace's IR
//! directly, plus the exit machinery shared with the future native
//! backends (snapshot restore = lj_snap_restore, exit accounting =
//! lj_trace_exit/trace_hotside).
//!
//! Backend organization (cross-platform by construction):
//! * `exec` (this file) is the arch-independent reference tier: it
//!   interprets the SSA IR of a compiled trace. It runs everywhere and
//!   doubles as the semantics oracle for the native backends.
//! * `mcode` provides W^X executable memory on every OS.
//! * `asm_x64` assembles the same IR to machine code and replaces the IR
//!   interpreter per trace when available; guards become branches to exit
//!   stubs, and the exit handler funnels back into the same
//!   snapshot-restore path below.
//!
//! Execution model for a self-linked loop trace (no IR_LOOP until
//! lj_opt_loop is ported): run the IR top to bottom; when the tail is
//! reached, materialize the final snapshot into the Lua stack (LuaJIT's
//! asm_tail_link stack sync) and restart from the top — the head SLOADs
//! then re-read the just-written slots. A failing guard exits through its
//! covering snapshot instead.

use crate::state::LuaState;
use crate::value::LuaValue;

use super::ir::*;
use super::{GCtrace, JitParam, SNAP_NORESTORE, SNAPCOUNT_DONE, TraceLink, TraceNo, snap_ref, snap_slot};

/// Result of running a trace: the bytecode index (into `startpt.bc`) at
/// which the interpreter resumes.
pub struct ExitResult {
    pub pc: usize,
    /// Exit (snapshot) number taken, for diagnostics and hot-exit checks.
    pub exitno: usize,
}

/// Execute trace `traceno` for the frame at `base`. Returns the resume pc.
/// The caller (JFORL/JLOOP dispatch arms) must have synced the frame.
pub fn trace_exec(l: &mut LuaState, base: usize, traceno: TraceNo) -> ExitResult {
    let g = l.global();
    // The trace is owned by the registry inside GlobalState; the executor
    // additionally mutates the Lua stack. Split the borrows via a raw
    // pointer — the registry never drops traces while one is running.
    let tr: &mut GCtrace = unsafe {
        let p: *mut GCtrace = &mut **g.jit.trace[traceno as usize]
            .as_mut()
            .expect("executing a freed trace");
        &mut *p
    };
    let mut env = std::mem::take(&mut g.jit.exec_env);
    let nins = (tr.ir.nins() - REF_BIAS) as usize;
    if env.len() < nins {
        // Stale contents are harmless: every slot is written before any
        // read (defs precede uses; snapshots only reference prior defs).
        env.resize(nins, 0u64);
    }

    let hotexit = g.jit.param(JitParam::HotExit) as u32;
    let r = if let Some(mc) = &tr.mcode {
        // Native backend: run the machine code; it returns the exit
        // snapshot index with all snapshot values parked in `env`.
        let entry: extern "C" fn(*mut LuaValue, *mut u64) -> u32 =
            unsafe { std::mem::transmute(mc.ptr()) };
        let exitno = entry(
            unsafe { l.stack.as_mut_ptr().add(base) },
            env.as_mut_ptr(),
        ) as usize;
        restore_snapshot(l, base, tr, &env, exitno);
        ExitResult { pc: tr.snap[exitno].pc as usize, exitno }
    } else {
        run_ir(l, base, tr, &mut env)
    };
    // Exit accounting (lj_trace_exit -> trace_hotside): count taken exits;
    // side traces spawn from hot exits in a later phase.
    if let Some(snap) = tr.snap.get_mut(r.exitno) {
        if snap.count != SNAPCOUNT_DONE {
            snap.count = snap.count.saturating_add(1);
            let _ = hotexit; // Phase 4: >= hotexit starts a side trace.
        }
    }
    let g = l.global();
    g.jit.exec_env = env;
    r
}

/// The IR interpreter proper.
fn run_ir(l: &mut LuaState, base: usize, tr: &GCtrace, env: &mut [u64]) -> ExitResult {
    let ir = &tr.ir;
    let nins = ir.nins();
    let looping = tr.linktype == TraceLink::Loop && tr.link == tr.traceno;

    'trace: loop {
        // Snapshot cursor: the guard at ref R exits through the last
        // snapshot with iref <= R.
        let mut snapidx = 0usize;
        let mut r = REF_FIRST;
        while r < nins {
            while snapidx + 1 < tr.snap.len() && tr.snap[snapidx + 1].iref <= r {
                snapidx += 1;
            }
            let ins = *ir.ir(r);
            let op = ins.op();
            let val = |env: &[u64], op: IRRef| -> u64 { read_ref(ir, env, op) };
            match op {
                IROp::NOP | IROp::BASE | IROp::LOOP => {}
                IROp::SLOAD => {
                    // op1 = absolute recorder slot (baseslot-based).
                    let slot = ins.op1 as usize - 2;
                    let v = l.stack[base + slot];
                    if ins.is_guard() && !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::POW
                | IROp::MIN | IROp::MAX => {
                    let x = f64::from_bits(val(env, ins.op1 as IRRef));
                    let y = f64::from_bits(val(env, ins.op2 as IRRef));
                    let z = super::opt_fold::fold_numarith(x, y, op);
                    env[(r - REF_BIAS) as usize] = z.to_bits();
                }
                IROp::NEG => {
                    let x = f64::from_bits(val(env, ins.op1 as IRRef));
                    env[(r - REF_BIAS) as usize] = (-x).to_bits();
                }
                IROp::ABS => {
                    let x = f64::from_bits(val(env, ins.op1 as IRRef));
                    env[(r - REF_BIAS) as usize] = x.abs().to_bits();
                }
                IROp::FPMATH => {
                    let x = f64::from_bits(val(env, ins.op1 as IRRef));
                    let z = match ins.op2 as u32 {
                        super::record::IRFPM_FLOOR => x.floor(),
                        super::record::IRFPM_CEIL => x.ceil(),
                        super::record::IRFPM_TRUNC => x.trunc(),
                        super::record::IRFPM_SQRT => x.sqrt(),
                        _ => unreachable!("bad FPMATH literal"),
                    };
                    env[(r - REF_BIAS) as usize] = z.to_bits();
                }
                IROp::LT | IROp::GE | IROp::LE | IROp::GT
                | IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT => {
                    let cond = if irt_isnum(ins.t()) {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef));
                        let y = f64::from_bits(val(env, ins.op2 as IRRef));
                        super::opt_fold::fold_numcmp(x, y, op)
                    } else {
                        unreachable!("non-num ordered comparison in trace")
                    };
                    debug_assert!(ins.is_guard());
                    if !cond {
                        return exit_snapshot(l, base, tr, env, snapidx);
                    }
                }
                IROp::EQ | IROp::NE => {
                    let cond = if irt_isnum(ins.t()) {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef));
                        let y = f64::from_bits(val(env, ins.op2 as IRRef));
                        if op == IROp::EQ { x == y } else { x != y }
                    } else {
                        // GC objects: reference identity on the value bits.
                        let x = val(env, ins.op1 as IRRef);
                        let y = val(env, ins.op2 as IRRef);
                        if op == IROp::EQ { x == y } else { x != y }
                    };
                    debug_assert!(ins.is_guard());
                    if !cond {
                        return exit_snapshot(l, base, tr, env, snapidx);
                    }
                }
                _ => unreachable!("unexpected IR op {:?} in phase-3 trace", op),
            }
            r += 1;
        }

        if looping {
            // Loop edge: sync the final snapshot into the stack, then
            // re-enter the trace head (asm_tail_link's stack sync).
            let lastidx = tr.snap.len() - 1;
            restore_snapshot(l, base, tr, env, lastidx);
            continue 'trace;
        }
        // Non-looping tail (TraceLink::None safety net): exit through the
        // final snapshot.
        let lastidx = tr.snap.len() - 1;
        return exit_snapshot(l, base, tr, env, lastidx);
    }
}

/// Read the 64-bit value of an operand ref (constant or instruction).
#[inline]
fn read_ref(ir: &IrBuf, env: &[u64], r: IRRef) -> u64 {
    if r >= REF_BIAS {
        env[(r - REF_BIAS) as usize]
    } else {
        const_bits(ir, r)
    }
}

/// The NaN-boxed bit pattern of a constant ref.
pub(super) fn const_bits(ir: &IrBuf, r: IRRef) -> u64 {
    let ins = ir.ir(r);
    match ins.op() {
        IROp::KNUM | IROp::KGC | IROp::KINT64 => ir.k64_val(r),
        IROp::KPRI => match irt_type(ins.t()) {
            IRT_NIL => LuaValue::NIL.to_bits(),
            IRT_FALSE => LuaValue::FALSE.to_bits(),
            _ => LuaValue::TRUE.to_bits(),
        },
        IROp::KINT => (ins.i() as f64).to_bits(),
        _ => unreachable!("bad constant op {:?}", ins.op()),
    }
}

/// `lj_snap_restore`: write the snapshot's slots back to the Lua stack.
fn restore_snapshot(l: &mut LuaState, base: usize, tr: &GCtrace, env: &[u64], snapidx: usize) {
    let snap = &tr.snap[snapidx];
    let map = &tr.snapmap[snap.mapofs as usize..(snap.mapofs as usize + snap.nent as usize)];
    for &sn in map {
        if sn & SNAP_NORESTORE != 0 {
            continue;
        }
        let s = snap_slot(sn) as usize;
        debug_assert!(s >= 2, "frame slots are never restored in phase 3");
        let bits = read_ref(&tr.ir, env, snap_ref(sn));
        l.stack[base + s - 2] = LuaValue::from_bits(bits);
    }
}

/// Take exit `snapidx`: restore the stack and resume at the snapshot pc.
fn exit_snapshot(
    l: &mut LuaState,
    base: usize,
    tr: &GCtrace,
    env: &[u64],
    snapidx: usize,
) -> ExitResult {
    restore_snapshot(l, base, tr, env, snapidx);
    ExitResult { pc: tr.snap[snapidx].pc as usize, exitno: snapidx }
}

/// Runtime type check against a (guarded) IR type — the SLOAD typecheck.
fn typecheck(v: LuaValue, t: u8) -> bool {
    match irt_type(t) {
        IRT_NUM => v.is_number(),
        IRT_NIL => v.is_nil(),
        IRT_FALSE => v.to_bits() == LuaValue::FALSE.to_bits(),
        IRT_TRUE => v.to_bits() == LuaValue::TRUE.to_bits(),
        ty => {
            // GC types: compare the negated itype tag.
            !v.is_number() && (!v.itype()) as u8 & IRT_TYPE == ty
        }
    }
}
