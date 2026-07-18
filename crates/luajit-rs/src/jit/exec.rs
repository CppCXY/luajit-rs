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
//! Execution model for a self-linked loop trace: for loop-optimized IR
//! (lj_opt_loop ported: LOOP + PHIs), the pre-roll runs once and the
//! variant part re-enters after LOOP with the PHI values carried over in
//! `env`. Traces without IR_LOOP fall back to: run top to bottom,
//! materialize the final snapshot into the Lua stack and restart from
//! the top. A failing guard exits through its covering snapshot.

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

/// Execute trace `traceno` for the frame at `base` and follow the trace
/// tree: exits with a linked side trace continue there, Root-linked side
/// traces re-enter their root — all without returning to the interpreter
/// (LuaJIT's mcode-to-mcode jumps). Returns the resume pc of the final,
/// unlinked exit. The caller (JFORL/JLOOP dispatch arms) must have
/// synced the frame and must switch to the recording dispatch when a
/// side trace recording was started (jit.state == Record).
pub fn trace_exec(l: &mut LuaState, base: usize, traceno: TraceNo) -> ExitResult {
    let g = l.global();
    let mut env = std::mem::take(&mut g.jit.exec_env);
    if env.len() < g.jit.env_need {
        // Machine-code chains switch traces without returning to Rust:
        // size the buffer for the largest stored trace up front. Stale
        // contents are harmless (defs precede uses).
        env.resize(g.jit.env_need, 0u64);
    }
    let hotexit = g.jit.param(JitParam::HotExit) as u8;
    let mut current = traceno;
    let r = loop {
        // The traces are owned by the registry inside GlobalState; the
        // executor additionally mutates the Lua stack. Split the borrows
        // via raw pointers — the registry never drops traces while one
        // is running.
        let tr: &mut GCtrace = unsafe {
            let p: *mut GCtrace = &mut **l.global().jit.trace[current as usize]
                .as_mut()
                .expect("executing a freed trace");
            &mut *p
        };
        let mut tr = tr;

        // Run the trace. The native tier leaves all snapshot values in
        // env (exit-stub register flush) and defers the Lua stack
        // restore: linked exits may hand the values straight to the next
        // trace. With patched exit chains the machine code may leave
        // from a different trace than it entered: the exit code is
        // `(traceno << 16) | snapidx`. The portable tier restores
        // eagerly.
        let (exitno, restored) = if let Some(mc) = &tr.mcode {
            let entry: extern "C" fn(*mut LuaValue, *mut u64) -> u32 =
                unsafe { std::mem::transmute(mc.ptr()) };
            let code = entry(
                unsafe { l.stack.as_mut_ptr().add(base) },
                env.as_mut_ptr(),
            ) as usize;
            let exit_trace = (code >> 16) as TraceNo;
            if exit_trace != current {
                // The chain left from a linked trace: re-resolve.
                current = exit_trace;
                tr = unsafe {
                    let p: *mut GCtrace = &mut **l.global().jit.trace[current as usize]
                        .as_mut()
                        .expect("exit from a freed trace");
                    &mut *p
                };
            }
            (code & 0xffff, false)
        } else {
            (run_ir(l, base, tr, &mut env).exitno, true)
        };

        let (nsnap, linktype, link) = (tr.snap.len(), tr.linktype, tr.link);
        // Follow a side trace linked to this exit. (Machine-code parents
        // jump to machine-code sides directly through the patched stubs;
        // this path covers the portable tiers and mixed pairs. The side
        // trace's own prelude performs the env hand-over.)
        let sidetrace = tr.snap[exitno].sidetrace;
        if sidetrace != 0 {
            let side_native = unsafe {
                l.global().jit.trace[sidetrace as usize]
                    .as_ref()
                    .expect("linked exit to a freed trace")
                    .mcode
                    .is_some()
            };
            if !side_native && !restored {
                // The portable side tier reads the Lua stack: materialize.
                restore_snapshot(l, base, tr, &env, exitno);
            }
            current = sidetrace;
            continue;
        }
        if !restored {
            restore_snapshot(l, base, tr, &env, exitno);
        }
        // A Root-linked side trace fell through its tail: re-enter the
        // root trace (the restored stack is exactly its entry state).
        if exitno == nsnap - 1 && linktype == TraceLink::Root && link != 0 {
            current = link;
            continue;
        }
        // Exit accounting (lj_trace_exit -> trace_hotside): count taken
        // exits and start recording a side trace on a hot one.
        let snap = &mut tr.snap[exitno];
        if snap.count != SNAPCOUNT_DONE {
            snap.count += 1;
            if snap.count >= hotexit {
                super::trace::trace_hot_side(l, base, current, exitno);
            }
        }
        break ExitResult { pc: tr.snap[exitno].pc as usize, exitno };
    };
    let g = l.global();
    g.jit.exec_env = env;
    r
}

/// The IR interpreter proper.
fn run_ir(l: &mut LuaState, base: usize, tr: &GCtrace, env: &mut [u64]) -> ExitResult {
    let ir = &tr.ir;
    let nins = ir.nins();
    let looping = tr.linktype == TraceLink::Loop && tr.link == tr.traceno;
    // Loop-optimized traces: re-enter at the LOOP instruction with the
    // PHI values carried over; legacy traces re-run from the top after a
    // final-snapshot restore.
    let loopref = if looping { ir.chain[IROp::LOOP as usize] as IRRef } else { 0 };
    let mut phis: Vec<(IRRef, IRRef)> = Vec::new();
    if loopref != 0 {
        // PHIs are the last instructions of the trace.
        let mut r = nins - 1;
        while r > loopref && ir.ir(r).op() == IROp::PHI {
            let ins = ir.ir(r);
            phis.push((ins.op1 as IRRef, ins.op2 as IRRef));
            r -= 1;
        }
    }
    let mut phivals: Vec<u64> = Vec::with_capacity(phis.len());

    let mut start = REF_FIRST;
    'trace: loop {
        // Snapshot cursor: the guard at ref R exits through the last
        // snapshot with iref <= R.
        let mut snapidx = 0usize;
        let mut r = start;
        while r < nins {
            while snapidx + 1 < tr.snap.len() && tr.snap[snapidx + 1].iref <= r {
                snapidx += 1;
            }
            let ins = *ir.ir(r);
            let op = ins.op();
            let val = |env: &[u64], op: IRRef| -> u64 { read_ref(ir, env, op) };
            match op {
                IROp::NOP | IROp::BASE | IROp::LOOP | IROp::PHI => {}
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
            if loopref != 0 {
                // Loop edge of an unrolled trace: parallel-assign the PHI
                // values (read all right refs before writing any left
                // ref), then re-enter the variant part after LOOP.
                phivals.clear();
                phivals.extend(phis.iter().map(|&(_, rref)| read_ref(ir, env, rref)));
                for (&(lref, _), &v) in phis.iter().zip(phivals.iter()) {
                    env[(lref - REF_BIAS) as usize] = v;
                }
                start = loopref + 1;
                continue 'trace;
            }
            // Legacy loop edge (no IR_LOOP): sync the final snapshot into
            // the stack, then re-enter the trace head (the head SLOADs
            // re-read the just-written slots).
            let lastidx = tr.snap.len() - 1;
            restore_snapshot(l, base, tr, env, lastidx);
            start = REF_FIRST;
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
