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
use super::{
    GCtrace, JitParam, SNAP_NORESTORE, SNAPCOUNT_DONE, TraceLink, TraceNo, snap_ref, snap_slot,
};

/// Result of running a trace: the bytecode index (into the resume
/// frame's proto) at which the interpreter resumes.
pub struct ExitResult {
    pub pc: usize,
    /// Exit (snapshot) number taken, for diagnostics and hot-exit checks.
    pub exitno: usize,
    /// Recorder base slot of the exit snapshot: 2 = the entry frame;
    /// higher values mean the exit lies in an inlined call frame and the
    /// interpreter must shift its base by `baseslot - 2` and reload the
    /// frame's closure before resuming.
    pub baseslot: usize,
    /// GC-debt exit (IR_GCSTEP): skip exit accounting and side traces —
    /// the caller must reach a collection point. These exits must never
    /// be patched over.
    pub gcexit: bool,
    /// Base shift (in slots) accumulated by recursive re-entries inside
    /// `run_ir` (Uprec frames pushed on the Lua stack). The exit's
    /// snapshot values are relative to `entry base + shift`.
    pub shift: usize,
}

/// Execute trace `traceno` for the frame at `base` and follow the trace
/// tree: exits with a linked side trace continue there, Root-linked side
/// traces re-enter their root — all without returning to the interpreter
/// (LuaJIT's mcode-to-mcode jumps). Returns the resume pc of the final,
/// unlinked exit. The caller (JFORL/JLOOP dispatch arms) must have
/// synced the frame and must switch to the recording dispatch when a
/// side trace recording was started (jit.state == Record).
pub fn trace_exec(l: &mut LuaState, base: usize, traceno: TraceNo) -> ExitResult {
    // Bind the heap for allocating helpers (string interning): traces
    // are executed strictly single-threaded per VM.
    JIT_HEAP.with(|c| c.set(l.heap() as *mut crate::state::GcHeap));
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
    // Current frame base: recursive traces (Uprec/Tailrec) and Root
    // links into call frames shift it as frames are pushed on trace.
    let mut cbase = base;
    let r = loop {
        // The recursive machine-code tails check their stack headroom
        // against this bound (re-bound each iteration: growth happens
        // only here in Rust).
        STACK_END.with(|c| c.set(unsafe { l.stack.as_ptr().add(l.stack.len()) } as u64));
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
        let (exitno, restored, gcexit) = if let Some(mc) = &tr.mcode {
            let entry: extern "C" fn(*mut LuaValue, *mut u64) -> u32 =
                unsafe { std::mem::transmute(mc.ptr()) };
            let code = entry(unsafe { l.stack.as_mut_ptr().add(cbase) }, env.as_mut_ptr()) as usize;
            // Recursive/call-link tails shift the base register inside
            // the mcode chain: recover the actual exit base from the
            // epilogue's report.
            let exit_base = EXIT_BASE.with(|c| c.get());
            cbase = (exit_base - l.stack.as_ptr() as u64) as usize / 8;
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
            (code & 0x7fff, false, code & 0x8000 != 0)
        } else {
            if execdump() {
                eprintln!(
                    "ENTER t{} cbase={} s0={:?} s2={:?} s3={:?}",
                    current,
                    cbase,
                    fmtv(l.stack[cbase - 2]),
                    fmtv(l.stack[cbase]),
                    fmtv(l.stack[cbase + 1]),
                );
            }
            let r = run_ir(l, cbase, tr, &mut env);
            cbase += r.shift; // Recursive re-entries pushed frames.
            if execdump() {
                eprintln!(
                    "EXIT  t{} exit#{} shift={} baseslot={} pc={}",
                    current, r.exitno, r.shift, r.baseslot, r.pc
                );
            }
            (r.exitno, true, r.gcexit)
        };

        if gcexit {
            // GC-debt exit: no accounting, no side traces — return to
            // the interpreter, whose boundary check collects.
            if !restored {
                restore_snapshot(l, cbase, tr, &env, exitno);
            }
            break ExitResult {
                pc: tr.snap[exitno].pc as usize,
                exitno,
                baseslot: (cbase - base) + tr.snap[exitno].baseslot as usize,
                gcexit: true,
                shift: cbase - base,
            };
        }

        let (nsnap, linktype, link) = (tr.snap.len(), tr.linktype, tr.link);
        // A native recursive tail (Uprec/Tailrec) ran out of stack
        // headroom and left through its final snapshot: grow the stack
        // and re-enter at the advanced base.
        if exitno == nsnap - 1
            && matches!(linktype, TraceLink::Uprec | TraceLink::Tailrec)
            && link == current
        {
            if !restored {
                restore_snapshot(l, cbase, tr, &env, exitno);
            }
            cbase += tr.snap[exitno].baseslot as usize - 2;
            l.stack_ensure(cbase + tr.startpt.as_ref().framesize as usize + 8);
            continue;
        }
        // Follow a side trace linked to this exit. (Machine-code parents
        // jump to machine-code sides directly through the patched stubs;
        // this path covers the portable tiers and mixed pairs. The side
        // trace's own prelude performs the env hand-over.)
        let sidetrace = tr.snap[exitno].sidetrace;
        if sidetrace != 0 {
            let side_native = l.global().jit.trace[sidetrace as usize]
                .as_ref()
                .expect("linked exit to a freed trace")
                .mcode
                .is_some();
            if !side_native && !restored {
                // The portable side tier reads the Lua stack: materialize.
                restore_snapshot(l, cbase, tr, &env, exitno);
            }
            current = sidetrace;
            continue;
        }
        if !restored {
            restore_snapshot(l, cbase, tr, &env, exitno);
        }
        // A Root-linked side trace fell through its tail: re-enter the
        // root trace (the restored stack is exactly its entry state).
        // Call-linked tails (a compiled callee) enter the target in the
        // just-materialized callee frame.
        if exitno == nsnap - 1 && linktype == TraceLink::Root && link != 0 {
            cbase += tr.snap[exitno].baseslot as usize - 2;
            if cbase != base {
                let need = {
                    let lk = l.global().jit.trace[link as usize]
                        .as_ref()
                        .expect("link to a freed trace")
                        .startpt;
                    cbase + lk.as_ref().framesize as usize + 8
                };
                l.stack_ensure(need);
            }
            current = link;
            continue;
        }
        // Exit accounting (lj_trace_exit -> trace_hotside): count taken
        // exits and start recording a side trace on a hot one.
        let snap = &mut tr.snap[exitno];
        if snap.count != SNAPCOUNT_DONE {
            snap.count += 1;
            if snap.count >= hotexit {
                super::trace::trace_hot_side(l, cbase, current, exitno);
            }
        }
        break ExitResult {
            pc: tr.snap[exitno].pc as usize,
            exitno,
            baseslot: (cbase - base) + tr.snap[exitno].baseslot as usize,
            gcexit: false,
            shift: cbase - base,
        };
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
    // Recursive self-links: the tail materializes the frames pushed on
    // trace and re-enters from the top at the callee's base.
    let recursing =
        matches!(tr.linktype, TraceLink::Uprec | TraceLink::Tailrec) && tr.link == tr.traceno;
    // Current base: advanced by the recursive tail as frames stack up.
    let mut cbase = base;
    // Loop-optimized traces: re-enter at the LOOP instruction with the
    // PHI values carried over; legacy traces re-run from the top after a
    // final-snapshot restore.
    let loopref = if looping {
        ir.chain[IROp::LOOP as usize] as IRRef
    } else {
        0
    };
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
                    // op1 = absolute recorder slot (baseslot-based; 0 is
                    // the frame-0 function slot at base-2).
                    let idx = (cbase as i64 + ins.op1 as i64 - 2) as usize;
                    let v = l.stack[idx];
                    if ins.is_guard() && !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::ULOAD => {
                    // op1 = KINT64 constant holding the closed cell address.
                    let p = const_bits(ir, ins.op1 as IRRef) as *const LuaValue;
                    let v = unsafe { *p };
                    if ins.is_guard() && !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::FLOAD => {
                    // Guarded `metatable == nil` check (IRFL_TAB_META).
                    debug_assert!(ins.is_guard());
                    let tv = LuaValue::from_bits(val(env, ins.op1 as IRRef));
                    let mt = tv
                        .as_table()
                        .expect("FLOAD on a non-table")
                        .as_ref()
                        .metatable;
                    if mt.is_some() {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                }
                IROp::HLOAD => {
                    let v = LuaValue::from_bits(jit_tget(
                        val(env, ins.op1 as IRRef),
                        val(env, ins.op2 as IRRef),
                    ));
                    if ins.is_guard() && !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::ALOAD => {
                    // Inlined array-part load: guard the key is an exact
                    // int inside the array, then read through aptr.
                    let tv = LuaValue::from_bits(val(env, ins.op1 as IRRef));
                    let t = tv.as_table().expect("ALOAD on a non-table");
                    let kn = f64::from_bits(val(env, ins.op2 as IRRef));
                    let ki = kn as i32;
                    if ki as f64 != kn || ki < 0 || (ki as u32) >= t.as_ref().asize {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    let v = unsafe { *t.as_ref().aptr.add(ki as usize) };
                    if !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::ASTORE => {
                    let carg = *ir.ir(ins.op2 as IRRef);
                    debug_assert_eq!(carg.op(), IROp::CARG);
                    let tv = LuaValue::from_bits(val(env, ins.op1 as IRRef));
                    let t = tv.as_table().expect("ASTORE on a non-table");
                    let kn = f64::from_bits(val(env, carg.op1 as IRRef));
                    let ki = kn as i32;
                    if ki as f64 != kn || ki <= 0 || (ki as u32) >= t.as_ref().asize {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    let v = LuaValue::from_bits(val(env, carg.op2 as IRRef));
                    unsafe { *t.as_ref().aptr.add(ki as usize) = v };
                }
                IROp::GCSTEP => {
                    // GC-debt guard: leave the trace when a collection
                    // is due (the boundary check then collects).
                    if l.global().heap.should_collect() {
                        let mut r = exit_snapshot(l, base, cbase, tr, env, snapidx);
                        r.gcexit = true;
                        return r;
                    }
                }
                IROp::CARG => {} // Consumed by HSTORE/CALLL.
                IROp::HSTORE => {
                    let carg = *ir.ir(ins.op2 as IRRef);
                    debug_assert_eq!(carg.op(), IROp::CARG);
                    jit_tset(
                        val(env, ins.op1 as IRRef),
                        val(env, carg.op1 as IRRef),
                        val(env, carg.op2 as IRRef),
                    );
                }
                IROp::TNEW => {
                    env[(r - REF_BIAS) as usize] = jit_tnew();
                }
                IROp::TDUP => {
                    env[(r - REF_BIAS) as usize] = jit_tdup(val(env, ins.op1 as IRRef));
                }
                IROp::CALLL => {
                    let idx = ins.op2 as u32;
                    let bits = match super::record::ircall_arity(idx) {
                        1 => {
                            let x = val(env, ins.op1 as IRRef);
                            match idx {
                                super::record::IRCALL_STR_LEN => jit_str_len(x),
                                super::record::IRCALL_STR_CHAR => jit_str_char(x),
                                super::record::IRCALL_TAB_LEN => jit_alen(x),
                                _ => unreachable!("bad IRCALL index"),
                            }
                        }
                        2 => {
                            let carg = *ir.ir(ins.op1 as IRRef);
                            debug_assert_eq!(carg.op(), IROp::CARG);
                            let (x, y) = (val(env, carg.op1 as IRRef), val(env, carg.op2 as IRRef));
                            match idx {
                                super::record::IRCALL_TAB_NEXTK => jit_tnextk(x, y),
                                super::record::IRCALL_FMOD => jit_fmod(x, y),
                                super::record::IRCALL_STR_CMP => jit_str_cmp(x, y),
                                super::record::IRCALL_STR_BYTE => jit_str_byte(x, y),
                                super::record::IRCALL_TAB_CONCAT => jit_tconcat(x, y),
                                super::record::IRCALL_CAT => jit_cat(x, y),
                                super::record::IRCALL_USET => jit_uset(x, y),
                                _ => unreachable!("bad IRCALL index"),
                            }
                        }
                        _ => {
                            let cargj = *ir.ir(ins.op1 as IRRef);
                            debug_assert_eq!(cargj.op(), IROp::CARG);
                            let cargi = *ir.ir(cargj.op1 as IRRef);
                            debug_assert_eq!(cargi.op(), IROp::CARG);
                            let (x, y, z) = (
                                val(env, cargi.op1 as IRRef),
                                val(env, cargi.op2 as IRRef),
                                val(env, cargj.op2 as IRRef),
                            );
                            match idx {
                                super::record::IRCALL_STR_SUB => jit_str_sub(x, y, z),
                                super::record::IRCALL_VARG => jit_varg(x, y, z),
                                _ => unreachable!("bad IRCALL index"),
                            }
                        }
                    };
                    let v = LuaValue::from_bits(bits);
                    if ins.is_guard() && !typecheck(v, ins.t()) {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                    env[(r - REF_BIAS) as usize] = v.to_bits();
                }
                IROp::TOBIT => {
                    // Wrapping num -> int32 (num2bit); the recorder's
                    // range guards keep this inside the exact window.
                    let x = f64::from_bits(val(env, ins.op1 as IRRef));
                    let z = crate::stdlib::bit::num2bit(x) as f64;
                    env[(r - REF_BIAS) as usize] = z.to_bits();
                }
                IROp::BSWAP => {
                    let x = f64::from_bits(val(env, ins.op1 as IRRef)) as i32;
                    env[(r - REF_BIAS) as usize] = ((x.swap_bytes()) as f64).to_bits();
                }
                IROp::BAND
                | IROp::BOR
                | IROp::BXOR
                | IROp::BSHL
                | IROp::BSHR
                | IROp::BSAR
                | IROp::BROL
                | IROp::BROR
                | IROp::BNOT => {
                    // Fused num -> int32 -> op -> num, mirroring the
                    // interpreter's coercions (operands are range-guarded).
                    let x = f64::from_bits(val(env, ins.op1 as IRRef)) as i32;
                    let z: f64 = match op {
                        IROp::BNOT => (!x) as f64,
                        IROp::BAND => {
                            (x & f64::from_bits(val(env, ins.op2 as IRRef)) as i32) as f64
                        }
                        IROp::BOR => (x | f64::from_bits(val(env, ins.op2 as IRRef)) as i32) as f64,
                        IROp::BXOR => {
                            (x ^ f64::from_bits(val(env, ins.op2 as IRRef)) as i32) as f64
                        }
                        _ => {
                            let sh = (f64::from_bits(val(env, ins.op2 as IRRef)) as u32) & 31;
                            match op {
                                IROp::BSHL => (x << sh) as f64,
                                IROp::BSHR => ((x as u32) >> sh) as f64,
                                IROp::BROL => ((x as u32).rotate_left(sh) as i32) as f64,
                                IROp::BROR => ((x as u32).rotate_right(sh) as i32) as f64,
                                _ => (x >> sh) as f64,
                            }
                        }
                    };
                    env[(r - REF_BIAS) as usize] = z.to_bits();
                }
                IROp::ADD
                | IROp::SUB
                | IROp::MUL
                | IROp::DIV
                | IROp::POW
                | IROp::MIN
                | IROp::MAX
                | IROp::MOD => {
                    if irt_isint(ins.t()) {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef)) as i32;
                        let y = f64::from_bits(val(env, ins.op2 as IRRef)) as i32;
                        let z = super::opt_fold::kfold_intop(x, y, op);
                        env[(r - REF_BIAS) as usize] = (z as f64).to_bits();
                    } else {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef));
                        let y = f64::from_bits(val(env, ins.op2 as IRRef));
                        let z = super::opt_fold::fold_numarith(x, y, op);
                        env[(r - REF_BIAS) as usize] = z.to_bits();
                    }
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
                IROp::LT
                | IROp::GE
                | IROp::LE
                | IROp::GT
                | IROp::ULT
                | IROp::UGE
                | IROp::ULE
                | IROp::UGT => {
                    let cond = if irt_isnum(ins.t()) {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef));
                        let y = f64::from_bits(val(env, ins.op2 as IRRef));
                        super::opt_fold::fold_numcmp(x, y, op)
                    } else if irt_isint(ins.t()) {
                        let x = f64::from_bits(val(env, ins.op1 as IRRef)) as i32;
                        let y = f64::from_bits(val(env, ins.op2 as IRRef)) as i32;
                        match op {
                            IROp::LT => x < y,
                            IROp::GE => x >= y,
                            IROp::LE => x <= y,
                            IROp::GT => x > y,
                            IROp::ULT => (x as u32) < (y as u32),
                            IROp::UGE => (x as u32) >= (y as u32),
                            IROp::ULE => (x as u32) <= (y as u32),
                            _ => (x as u32) > (y as u32),
                        }
                    } else {
                        unreachable!("non-num/int ordered comparison in trace")
                    };
                    debug_assert!(ins.is_guard());
                    if !cond {
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
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
                        return exit_snapshot(l, base, cbase, tr, env, snapidx);
                    }
                }
                IROp::CONV => {
                    // op2 encodes target type; for now only NUM↔INT.
                    let src = val(env, ins.op1 as IRRef);
                    let tgt = ins.op2 as u8;
                    let z = if irt_isnum(tgt) {
                        // INT → NUM: value already stored as f64 bits.
                        src
                    } else if irt_isint(tgt) && ins.is_guard() {
                        // NUM → INT (guarded): verify exact int32.
                        let x = f64::from_bits(src);
                        if super::ir::num_isint(x) {
                            (x as i32 as f64).to_bits()
                        } else {
                            return exit_snapshot(l, base, cbase, tr, env, snapidx);
                        }
                    } else {
                        // NUM → INT (unguarded).
                        let x = f64::from_bits(src);
                        (x as i32 as f64).to_bits()
                    };
                    env[(r - REF_BIAS) as usize] = z;
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
            restore_snapshot(l, cbase, tr, env, lastidx);
            start = REF_FIRST;
            continue 'trace;
        }
        if recursing {
            // Recursive tail (Uprec/Tailrec): materialize the frames the
            // trace pushed (the final snapshot covers the whole frame
            // stack), advance the base to the new callee frame and
            // re-enter from the top — the head SLOADs then read the new
            // frame. Tail recursion has baseslot 2: same-base restart.
            let lastidx = tr.snap.len() - 1;
            restore_snapshot(l, cbase, tr, env, lastidx);
            cbase += tr.snap[lastidx].baseslot as usize - 2;
            l.stack_ensure(cbase + tr.startpt.as_ref().framesize as usize + 8);
            start = REF_FIRST;
            continue 'trace;
        }
        // Non-looping tail (TraceLink::None safety net): exit through the
        // final snapshot.
        let lastidx = tr.snap.len() - 1;
        return exit_snapshot(l, base, cbase, tr, env, lastidx);
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

/// Debug: trace-execution event dump (LUAJIT_RS_EXECDUMP).
fn execdump() -> bool {
    use std::sync::atomic::{AtomicI32, Ordering};
    static LEFT: AtomicI32 = AtomicI32::new(-2);
    let left = LEFT.load(Ordering::Relaxed);
    if left == -2 {
        let v = std::env::var("LUAJIT_RS_EXECDUMP")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(-1);
        LEFT.store(v, Ordering::Relaxed);
        return v != -1 && v > 0;
    }
    if left <= 0 {
        return false;
    }
    LEFT.store(left - 1, Ordering::Relaxed);
    true
}

/// Debug: short value formatter for `execdump`.
fn fmtv(v: LuaValue) -> String {
    if let Some(n) = v.as_number() {
        format!("{}", n)
    } else if v.is_nil() {
        "nil".into()
    } else if v.is_table() {
        format!("tab:{:x}", v.gc_addr())
    } else if v.is_func() {
        format!("fun:{:x}", v.gc_addr())
    } else if v.is_string() {
        "str".into()
    } else {
        format!("raw:{:x}", v.to_bits())
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
        debug_assert!(s != 1, "the root frame link is never restored");
        let bits = read_ref(&tr.ir, env, snap_ref(sn));
        l.stack[base + s - 2] = LuaValue::from_bits(bits);
    }
}

/// Take exit `snapidx`: restore the stack and resume at the snapshot pc.
/// `cbase` is the current frame base (recursive re-entries may have
/// advanced it past the entry `base`).
fn exit_snapshot(
    l: &mut LuaState,
    base: usize,
    cbase: usize,
    tr: &GCtrace,
    env: &[u64],
    snapidx: usize,
) -> ExitResult {
    restore_snapshot(l, cbase, tr, env, snapidx);
    ExitResult {
        pc: tr.snap[snapidx].pc as usize,
        exitno: snapidx,
        baseslot: (cbase - base) + tr.snap[snapidx].baseslot as usize,
        gcexit: false,
        shift: cbase - base,
    }
}

/// Runtime type check against a (guarded) IR type — the SLOAD typecheck.
fn typecheck(v: LuaValue, t: u8) -> bool {
    match irt_type(t) {
        IRT_NUM => v.is_number(),
        IRT_INT => v.is_number(),
        IRT_NIL => v.is_nil(),
        IRT_FALSE => v.to_bits() == LuaValue::FALSE.to_bits(),
        IRT_TRUE => v.to_bits() == LuaValue::TRUE.to_bits(),
        ty => {
            // GC types: compare the negated itype tag.
            !v.is_number() && (!v.itype()) as u8 & IRT_TYPE == ty
        }
    }
}

// -- Table helpers shared by the recorder, run_ir and the machine code ------
//
// These mirror the interpreter's raw TGET/TSET dispatch bit for bit, so a
// recorded HLOAD/HSTORE is semantically identical to what the interpreter
// would have done. The traces guard the table type and (where required)
// `metatable == nil` before reaching them.

/// Raw table get (the TGETV dispatch). Returns the NaN-boxed result.
pub extern "C" fn jit_tget(tab_bits: u64, key_bits: u64) -> u64 {
    let t = LuaValue::from_bits(tab_bits)
        .as_table()
        .expect("HLOAD on a non-table");
    let k = LuaValue::from_bits(key_bits);
    let v = if k.is_string() {
        t.as_ref().get_str(k)
    } else if k.is_number() {
        let ki = k.num() as i32;
        if ki as f64 == k.num() && ki >= 0 {
            t.as_ref().get_int(ki)
        } else {
            t.as_ref().get(k)
        }
    } else {
        t.as_ref().get(k)
    };
    v.to_bits()
}

/// Raw table set (the TSETV dispatch, `metatable == nil` guarded).
pub extern "C" fn jit_tset(tab_bits: u64, key_bits: u64, val_bits: u64) {
    let t = LuaValue::from_bits(tab_bits)
        .as_table()
        .expect("HSTORE on a non-table");
    let k = LuaValue::from_bits(key_bits);
    let v = LuaValue::from_bits(val_bits);
    debug_assert!(!k.is_nil(), "nil key must be guarded at record time");
    if k.is_string() {
        t.as_mut().set_str(k, v);
    } else if k.is_number() {
        let ki = k.num() as i32;
        if ki as f64 == k.num() && ki >= 0 {
            t.as_mut().set_int(ki, v);
        } else {
            t.as_mut().set(k, v);
        }
    } else {
        t.as_mut().set(k, v);
    }
}

/// Table-traversal step (`lj_tab_next`'s key half): the next key after
/// `key`, or nil at the end. The value is re-fetched with a plain HLOAD
/// of the returned key.
pub extern "C" fn jit_tnextk(tab_bits: u64, key_bits: u64) -> u64 {
    let t = LuaValue::from_bits(tab_bits)
        .as_table()
        .expect("NEXTK on a non-table");
    match t.as_ref().next(LuaValue::from_bits(key_bits)) {
        Some((k, _)) => k.to_bits(),
        None => LuaValue::NIL.to_bits(),
    }
}

/// `x ^ y` with the interpreter's vm_pow semantics (raw result bits).
pub extern "C" fn jit_pow(a_bits: u64, b_bits: u64) -> u64 {
    crate::vm::vm_pow(f64::from_bits(a_bits), f64::from_bits(b_bits)).to_bits()
}

/// math.fmod: `x % y` pushed through LuaValue::number (normalizes -0.0
/// and NaN), mirroring the builtin bit for bit.
pub extern "C" fn jit_fmod(a_bits: u64, b_bits: u64) -> u64 {
    LuaValue::number(f64::from_bits(a_bits) % f64::from_bits(b_bits)).to_bits()
}

// -- String helpers ----------------------------------------------------------
//
// The read-only ones decode the string object straight from the value
// bits. The allocating ones (sub/char) intern through the heap bound by
// `trace_exec`; the growth is added to the GC debt cell so the on-trace
// GCSTEP guard sees it.

thread_local! {
    /// Heap of the VM currently executing a trace (set by `trace_exec`).
    static JIT_HEAP: std::cell::Cell<*mut crate::state::GcHeap> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
    /// One-past-the-end address of the current Lua stack buffer, for
    /// the machine-code recursive tails' headroom check.
    static STACK_END: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// BASE register value at the last machine-code exit: recursive and
    /// call-link tails shift the base inside mcode chains, invisibly to
    /// Rust — the epilogue reports it back through this cell.
    static EXIT_BASE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Address of the stack-end cell (embedded in recursive mcode tails).
#[allow(dead_code)]
pub(super) fn stack_end_cell_addr() -> u64 {
    STACK_END.with(|c| c.as_ptr() as u64)
}

/// Address of the exit-base cell (embedded in the mcode epilogue).
#[allow(dead_code)]
pub(super) fn exit_base_cell_addr() -> u64 {
    EXIT_BASE.with(|c| c.as_ptr() as u64)
}

#[inline]
fn str_bytes(bits: u64) -> &'static [u8] {
    let s = LuaValue::from_bits(bits)
        .as_string()
        .expect("string op on a non-string");
    unsafe { std::mem::transmute::<&[u8], &'static [u8]>(s.as_ref().as_bytes()) }
}

/// The heap bound by `trace_exec` (for allocating helpers).
fn jit_heap() -> &'static mut crate::state::GcHeap {
    unsafe {
        JIT_HEAP
            .with(|c| c.get())
            .as_mut()
            .expect("allocating helper outside trace_exec")
    }
}

/// Intern `bytes` in the bound heap, tracking the growth as GC debt.
fn jit_intern(bytes: &[u8]) -> u64 {
    let heap = jit_heap();
    let before = heap.strings.bytes();
    let sid = heap.strings.intern(bytes);
    let grown = heap.strings.bytes() - before;
    if grown > 0 {
        heap.table_extra += grown;
    }
    heap.str_value(sid).to_bits()
}

/// string.len / `#s`.
pub extern "C" fn jit_str_len(s_bits: u64) -> u64 {
    (str_bytes(s_bits).len() as f64).to_bits()
}

/// Lexicographic byte compare: -1/0/1 as a number (the interpreter's
/// `meta_comp` string path uses the same slice ordering).
pub extern "C" fn jit_str_cmp(a_bits: u64, b_bits: u64) -> u64 {
    let c = match str_bytes(a_bits).cmp(str_bytes(b_bits)) {
        std::cmp::Ordering::Less => -1.0f64,
        std::cmp::Ordering::Equal => 0.0,
        std::cmp::Ordering::Greater => 1.0,
    };
    c.to_bits()
}

/// string.byte(s, i) — the single-index case (j defaults to i): the
/// byte as a number, or nil when out of range (mirrors `str_byte`).
pub extern "C" fn jit_str_byte(s_bits: u64, i_bits: u64) -> u64 {
    let s = str_bytes(s_bits);
    let i = f64::from_bits(i_bits) as i64;
    let len = s.len() as i64;
    let idx = if i < 0 { len + i } else { i - 1 };
    if idx < 0 || idx >= len {
        LuaValue::NIL.to_bits()
    } else {
        LuaValue::number(s[idx as usize] as f64).to_bits()
    }
}

/// string.sub(s, i, j) — j always present (a missing j records as -1,
/// which selects the same suffix). Mirrors `str_sub`.
pub extern "C" fn jit_str_sub(s_bits: u64, i_bits: u64, j_bits: u64) -> u64 {
    let s = str_bytes(s_bits);
    let i = f64::from_bits(i_bits) as i64;
    let j = f64::from_bits(j_bits) as i64;
    let len = s.len() as i64;
    let a = if i < 0 {
        (len + i).max(0) as usize
    } else {
        (i - 1).max(0).min(len) as usize
    };
    let j = if j < 0 { len + j } else { j - 1 };
    let b = (j.max(-1).min(len - 1) + 1) as usize;
    if a >= b {
        jit_intern(b"")
    } else {
        jit_intern(&s[a..b])
    }
}

/// string.char(c) — the single-argument case (mirrors `str_char`).
pub extern "C" fn jit_str_char(c_bits: u64) -> u64 {
    let b = (f64::from_bits(c_bits) as u32 & 0xff) as u8;
    jit_intern(&[b])
}

// -- Table allocation and library helpers ------------------------------------

/// BC_TNEW: a fresh empty table (the interpreter ignores the size hint).
pub extern "C" fn jit_tnew() -> u64 {
    let t = jit_heap().alloc_table(crate::table::LuaTable::new(0, 0));
    LuaValue::table(t).to_bits()
}

/// BC_TDUP: duplicate a template table. `templ_addr` is the raw address
/// of the prototype's KGc template (stable: Box or pool).
pub extern "C" fn jit_tdup(templ_addr: u64) -> u64 {
    let templ = unsafe { &*(templ_addr as usize as *const crate::table::LuaTable) };
    let t = jit_heap().alloc_table(templ.dup());
    LuaValue::table(t).to_bits()
}

/// `#t` / table.insert boundary: the raw table length (no metamethods,
/// mirroring the interpreter's LEN fast path).
pub extern "C" fn jit_alen(tab_bits: u64) -> u64 {
    let t = LuaValue::from_bits(tab_bits)
        .as_table()
        .expect("ALEN on a non-table");
    (t.as_ref().len() as f64).to_bits()
}

/// table.concat(t [, sep]) — the (t, sep, 1, nil) form. Mirrors
/// `tab_concat`; an invalid element yields nil, which fails the
/// recorded STR guard so the interpreter re-runs the call and raises.
pub extern "C" fn jit_tconcat(tab_bits: u64, sep_bits: u64) -> u64 {
    let t = LuaValue::from_bits(tab_bits)
        .as_table()
        .expect("CONCAT on a non-table");
    let sep_v = LuaValue::from_bits(sep_bits);
    let sep: &[u8] = match sep_v.as_string() {
        Some(s) => s.as_ref().as_bytes(),
        None => b"",
    };
    let tab = t.as_ref();
    let mut out = Vec::new();
    let mut first = true;
    let mut i = 1usize;
    loop {
        let v = tab.get_int(i as i32);
        if v.is_nil() {
            break;
        }
        if !first {
            out.extend_from_slice(sep);
        }
        first = false;
        if let Some(s) = v.as_string() {
            out.extend_from_slice(s.as_ref().as_bytes());
        } else if let Some(n) = v.as_number() {
            out.extend_from_slice(crate::strfmt::g14(n).as_bytes());
        } else {
            return LuaValue::NIL.to_bits(); // Error path: guard exit.
        }
        i += 1;
    }
    jit_intern(&out)
}

/// String concatenation (.. operator) for two Lua values. Returns a
/// string GCref on success, or LuaValue::NIL if either operand is
/// neither a string nor a number (guard exit → interpreter).
pub extern "C" fn jit_cat(a_bits: u64, b_bits: u64) -> u64 {
    let a = LuaValue::from_bits(a_bits);
    let b = LuaValue::from_bits(b_bits);
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    match a.as_string() {
        Some(s) => buf.extend_from_slice(s.as_ref().as_bytes()),
        None if a.is_number() => buf.extend_from_slice(crate::strfmt::g14(a.num()).as_bytes()),
        _ => return LuaValue::NIL.to_bits(),
    }
    match b.as_string() {
        Some(s) => buf.extend_from_slice(s.as_ref().as_bytes()),
        None if b.is_number() => buf.extend_from_slice(crate::strfmt::g14(b.num()).as_bytes()),
        _ => return LuaValue::NIL.to_bits(),
    }
    jit_intern(&buf)
}

/// Upvalue write: store `val_bits` into the cell at `cell_ptr`.
/// The cell pointer is a stable heap address of an upvalue slot.
pub extern "C" fn jit_uset(cell_ptr: u64, val_bits: u64) -> u64 {
    unsafe {
        *(cell_ptr as *mut u64) = val_bits;
    }
    0
}

/// Vararg copy: reads `nvarg` values from the caller's frame and
/// writes them to slots at `dst`. `frame_link` is the link word at
/// bp-1, packed = (numparams<<16)|(dst<<8)|want.
pub extern "C" fn jit_varg(base_ptr: u64, frame_link: u64, packed: u64) -> u64 {
    use crate::vm::FRAME_TYPE_MASK;
    use crate::vm::FRAME_VARG;
    if frame_link & FRAME_TYPE_MASK != FRAME_VARG {
        return u64::MAX;
    }
    let numparams = (packed >> 16) as u32;
    let dst = ((packed >> 8) & 0xFF) as u32;
    let want = (packed & 0xFF) as u32;
    let delta = (frame_link >> 3) as usize;
    let nvarg = (delta - 2).saturating_sub(numparams as usize);
    let base = base_ptr as *mut LuaValue;
    let src = unsafe { base.sub(delta).add(numparams as usize) };
    let actual = if want == 0 {
        nvarg
    } else {
        nvarg.min(want as usize)
    };
    for i in 0..actual {
        unsafe { *base.add(dst as usize + i) = *src.add(i) };
    }
    if want > 0 {
        for i in actual..(want as usize) {
            unsafe { *base.add(dst as usize + i) = LuaValue::NIL };
        }
    }
    actual as u64
}
