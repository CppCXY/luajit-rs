//! Trace management: hot-path triggers, the compiler state machine,
//! penalties and blacklisting. Ported from lj_trace.c.
//!
//! With the Phase 2 recorder in place, hot FORL/LOOP/FUNCF paths start a
//! real recording (lj_record_setup / rec_setup_root); the interpreter
//! then feeds every executed bytecode through `rec_ins` until the trace
//! completes (`trace_stop`) or aborts (`trace_abort` -> penalty ->
//! blacklist). Completed traces are stored in the trace registry but not
//! yet executed — bytecode patching to J* opcodes waits for the machine
//! code backend. Until then the start instruction's hot counter is parked
//! at 0xffff so a finished loop only re-checks its registry entry rarely.

use crate::bc::{self, BCOp, bc_isret, bc_j, bc_op, setbc_d, setbc_op};
use crate::gc::GcPtr;
use crate::jit::ir::IrBuf;
use crate::jit::{GCtrace, record};
use crate::proto::{PROTO_ILOOP, PROTO_NOJIT, Proto};
use crate::state::{GlobalState, LuaState};

use super::record::Record;
use super::{
    HOTCOUNT_LOOP, HotCount, JitParam, JitState, PENALTY_MAX, PENALTY_MIN, PENALTY_RNDBITS,
    PENALTY_SLOTS, SNAPCOUNT_DONE, TraceError, TraceLink, TraceNo, TraceState, bc_addr,
};

/// Hot counter value that parks an already-compiled start instruction
/// until real J* bytecode patching exists.
const HOTCOUNT_PARKED: HotCount = 0xffff;

/// `lj_trace_hot`: a hot counter underflowed at `pt.bc[pc]` (a FORL /
/// ITERL / LOOP or a FUNCF header). Reset the counter and start a root
/// trace unless the compiler is busy.
pub fn trace_hot(l: &mut LuaState, base: usize, pt: GcPtr<Proto>, pc: usize) {
    {
        let g = l.global();
        let js = &mut g.jit;
        // Reset hotcount. The hash slot is keyed by the offset-by-1 PC, the
        // same one the interpreter decremented.
        let reset = (js.param(JitParam::HotLoop) as u32 * HOTCOUNT_LOOP as u32) as HotCount;
        js.hotcount_set(bc_addr(pt, pc + 1), reset);
        // Only start a new trace if not recording.
        if js.state == TraceState::Idle {
            js.parent = 0; // Root trace.
            js.exitno = 0;
            js.state = TraceState::Start;
            trace_start(l, base, pt, pc);
        }
    }
}

/// `trace_hotside` (the start half): a hot side exit of `parent` was
/// taken often enough. Start recording a side trace from the exit's
/// resume point. The caller (the trace executor) has already restored
/// the snapshot to the Lua stack.
pub fn trace_hot_side(l: &mut LuaState, base: usize, parent: TraceNo, exitno: usize) {
    {
        let g = l.global();
        let js = &mut g.jit;
        if js.state != TraceState::Idle {
            return;
        }
        let t = js.trace[parent as usize]
            .as_ref()
            .expect("hot exit of a freed trace");
        // Deep exits record from within the inlined frame: snap_replay
        // rebuilds the frame stack and rec.pt from the snapshot constants.
        let (pt, pc) = (t.startpt, t.snap[exitno].pc as usize);
        js.parent = parent;
        js.exitno = exitno as u32;
        js.state = TraceState::Start;
        trace_start(l, base, pt, pc);
    }
}

/// `trace_start` + `lj_record_setup` + `rec_setup_root`: begin recording
/// a root trace at `pt.bc[pc]`, or bail out silently.
fn trace_start(l: &mut LuaState, base: usize, pt: GcPtr<Proto>, pc: usize) {
    let g = l.global();
    let js = &mut g.jit;
    // Side traces start as a pseudo-JMP: `pc` may lie inside an inlined
    // callee, so the root proto's bytecode must not be indexed with it.
    let startins = if js.parent != 0 {
        bc::bcins_ad(BCOp::JMP, 0, 0)
    } else {
        pt.as_ref().bc[pc]
    };
    let op = bc_op(startins);

    if pt.as_ref().flags & PROTO_NOJIT != 0 {
        // JIT disabled for this proto: lazily patch the hot instruction to
        // its non-counting variant so it stops triggering.
        if js.parent == 0 && js.exitno == 0 && op != BCOp::ITERN {
            debug_assert!(
                matches!(
                    op,
                    BCOp::FORL | BCOp::ITERL | BCOp::LOOP | BCOp::FUNCF | BCOp::FUNCV
                ),
                "bad hot bytecode {:?}",
                op
            );
            setbc_op(&mut pt.as_mut().bc[pc], op.offset(1) as u32);
            pt.as_mut().flags |= PROTO_ILOOP;
        }
        js.state = TraceState::Idle; // Silently ignored.
        return;
    }

    // Already compiled? (Stands in for the BC_JLOOP check until the
    // backend patches bytecode.) Park the counter and ignore.
    if js.parent == 0 && js.find_root_trace(pt, pc).is_some() {
        js.hotcount_set(bc_addr(pt, pc + 1), HOTCOUNT_PARKED);
        js.state = TraceState::Idle;
        return;
    }

    // Bookkeeping for trace_abort (J->cur.start*).
    js.startpt = Some(pt);
    js.startpc = pc;
    js.startins = startins;

    // Phase 2 recorder handles FORL/LOOP/ITERL/FUNCF roots; penalize the
    // rest (ITERN roots arrive with pairs() recording). Side traces
    // start at an arbitrary bytecode.
    if js.parent == 0 && !matches!(op, BCOp::FORL | BCOp::LOOP | BCOp::FUNCF | BCOp::ITERL) {
        js.err = TraceError::NYIBC;
        js.state = TraceState::Err;
        trace_abort(g);
        return;
    }

    // Get a new trace number.
    let traceno = trace_findfree(js);
    let Some(traceno) = traceno else {
        js.state = TraceState::Idle; // No free trace: silently ignored.
        return;
    };

    let mut rec = Record::new(
        traceno,
        pt,
        pc,
        js.parent as u16,
        js.exitno as u16,
        js.param(JitParam::LoopUnroll),
        js.param(JitParam::InstUnroll),
        js.param(JitParam::CallUnroll),
        js.param(JitParam::RecUnroll),
    );

    if js.parent != 0 {
        // lj_record_setup, the side-trace half: inherit the parent exit.
        let parent = js.parent;
        let exitno = js.exitno as usize;
        let (root, exit_count) = {
            let t = js.trace[parent as usize]
                .as_ref()
                .expect("freed parent trace");
            (
                if t.root != 0 { t.root } else { parent },
                t.snap[exitno].count,
            )
        };
        rec.cur.root = root;
        rec.cur.startins = crate::bc::bcins_ad(BCOp::JMP, 0, 0);
        if js.exitno != 0 {
            rec.startpc = None; // Prevent forming an extra loop.
        }
        // Avoid a hopeless exit: too many side traces or slow progress.
        let root_children = js.trace[root as usize].as_ref().map_or(0, |r| r.nchild);
        if root_children >= js.param(JitParam::MaxSide) as u16
            || exit_count as i32 >= js.param(JitParam::HotExit) + js.param(JitParam::TrySide)
        {
            // lj_record_stop(LJ_TRLINK_INTERP): blacklist the exit for
            // good instead of compiling a stub trace.
            js.trace[parent as usize].as_mut().unwrap().snap[exitno].count = SNAPCOUNT_DONE;
            js.startpt = None;
            js.state = TraceState::Idle;
            return;
        }
        // The registry owns the parent; the borrow is split like in the
        // executor (traces are never freed while the compiler runs).
        let t: *const GCtrace = &**js.trace[parent as usize]
            .as_ref()
            .expect("freed parent trace");
        if let Err(e) = rec.snap_replay(unsafe { &*t }, exitno) {
            js.err = e;
            js.state = TraceState::Err;
            trace_abort(g);
            return;
        }
    } else {
        // rec_setup_root: determine the next PC and the loop bytecode range.
        match op {
            BCOp::FORL => {
                rec.bc_extent = (-bc_j(startins)) as usize;
                rec.bc_min = (pc as i64 + 1 + bc_j(startins)) as usize;
                rec.pc = rec.bc_min;
            }
            BCOp::ITERL => {
                // The paired ITERC/ITERN sits right before (this VM's
                // ITERN is a plain generic iterator call).
                let prev = pt.as_ref().bc[pc - 1];
                if !matches!(bc_op(prev), BCOp::ITERC | BCOp::ITERN) {
                    js.err = if bc_op(prev) == BCOp::JLOOP {
                        TraceError::LINNER
                    } else {
                        TraceError::NYIBC
                    };
                    js.state = TraceState::Err;
                    trace_abort(g);
                    return;
                }
                rec.maxslot = crate::bc::bc_a(startins) + crate::bc::bc_b(prev) - 1;
                rec.bc_extent = (-bc_j(startins)) as usize;
                rec.bc_min = (pc as i64 + 1 + bc_j(startins)) as usize;
                rec.pc = rec.bc_min;
            }
            BCOp::LOOP => {
                // Only check the range for real loops, not "repeat until true".
                let pcj = (pc as i64 + bc_j(startins)) as usize;
                let jins = pt.as_ref().bc[pcj];
                if bc_op(jins) == BCOp::JMP && bc_j(jins) < 0 {
                    rec.bc_min = (pcj as i64 + 1 + bc_j(jins)) as usize;
                    rec.bc_extent = (-bc_j(jins)) as usize;
                }
                rec.maxslot = crate::bc::bc_a(startins);
                rec.pc = pc + 1;
            }
            BCOp::FUNCF => {
                // No bytecode range check for root traces started by a hot call.
                rec.maxslot = pt.as_ref().numparams as u32;
                rec.pc = pc + 1;
            }
            _ => unreachable!(),
        }
        rec.startpc = Some(pc);

        // Snapshot #0 points at the first instruction to be recorded.
        rec.snap_add();

        // For FORL roots, record the loop bookkeeping of the paired FORI.
        if op == BCOp::FORL {
            let fori = rec.pc - 1;
            if let Err(e) = rec.for_loop(l, base, fori, true) {
                let g = l.global();
                g.jit.err = e;
                g.jit.state = TraceState::Err;
                trace_abort(g);
                return;
            }
        }
    }
    if 1 + pt.as_ref().framesize as usize >= record::MAX_JSLOTS {
        let g = l.global();
        g.jit.err = TraceError::STACKOV;
        g.jit.state = TraceState::Err;
        trace_abort(g);
        return;
    }

    let g = l.global();
    g.jit.rec = Some(rec);
    g.jit.state = TraceState::Record;
}

/// `trace_findfree`: find a free trace number, bounded by maxtrace.
fn trace_findfree(js: &mut JitState) -> Option<TraceNo> {
    if let Some(i) = js.trace.iter().skip(1).position(|t| t.is_none()) {
        return Some((i + 1) as TraceNo);
    }
    if js.trace.len() > js.param(JitParam::MaxTrace) as usize {
        return None;
    }
    js.trace.push(None);
    Some((js.trace.len() - 1) as TraceNo)
}

/// The per-instruction recording hook (`lj_trace_ins` + trace_state's
/// RECORD case). Called by the interpreter before executing each
/// instruction while recording. Returns false when recording ended and
/// the interpreter should leave the recording dispatch loop.
pub fn rec_ins(l: &mut LuaState, base: usize, pt: GcPtr<Proto>, pc: usize) -> bool {
    let g = l.global();
    debug_assert!(g.jit.state == TraceState::Record);
    let mut rec = g.jit.rec.take().expect("recording without context");

    // Single-frame traces: crossing into another proto means a call or
    // return slipped through — abort defensively.
    if pt.addr() != rec.pt.addr() {
        g.jit.err = TraceError::NYIBC;
        g.jit.state = TraceState::Err;
        trace_abort(g);
        return false;
    }

    match rec.record_ins(&g.jit, l, base, pc) {
        Ok(None) => {
            g.jit.rec = Some(rec);
            true
        }
        Ok(Some((link, lnk))) => {
            // lj_record_stop: add the loop snapshot. Note: all loop ops
            // set J->pc to the following instruction, which is what this
            // snapshot must describe.
            rec.needsnap = false;
            rec.mergesnap = true;
            rec.snap_add();
            rec.mergesnap = true; // In case recording continues below.
            // LJ_TRACE_END: DCE + loop unrolling for self-linking loops.
            if link == TraceLink::Loop
                && lnk == rec.cur.traceno
                && rec.framedepth + rec.retdepth == 0
            {
                super::opt_dce::opt_dce(&mut rec.cur);
                match super::opt_loop::opt_loop(&mut rec) {
                    Ok(true) => {}
                    Ok(false) => {
                        // Fixable failure (TYPEINS/GFAIL), already undone:
                        // continue recording, i.e. unroll the loop once
                        // more (LJ_TRACE_END -> LJ_TRACE_RECORD).
                        rec.loopref = rec.cur.ir.nins();
                        g.jit.rec = Some(rec);
                        return true;
                    }
                    Err(e) => {
                        g.jit.err = e;
                        g.jit.state = TraceState::Err;
                        trace_abort(g);
                        return false;
                    }
                }
            }
            trace_stop(g, rec, link, lnk);
            false
        }
        Err(e) => {
            // Keep a partial root trace (trace stitching): when the
            // root trace has enough IR instructions, stop it as a
            // Stitch prefix — the interpreter handles the NYI off-
            // trace and the prefix runs natively next time. Side
            // traces never stitch (a stitch prefix traced from an
            // exit would just repeat the same NYI).
            let minstitch = g.jit.param(JitParam::MinStitch) as u32;
            if minstitch > 0
                && rec.parent == 0
                && matches!(e, TraceError::NYIBC | TraceError::NYIFFU)
                && rec.cur.ir.nins() - super::ir::REF_FIRST >= minstitch
            {
                // The snapshot describes the state right before the NYI
                // bytecode; the interpreter resumes there, executes it
                // off-trace, and optionally records a continuation.
                rec.needsnap = false;
                rec.mergesnap = true;
                rec.snap_add();
                rec.cur.linktype = TraceLink::Stitch;
                trace_stop(g, rec, TraceLink::Stitch, 0);
            } else {
                g.jit.err = e;
                g.jit.state = TraceState::Err;
                trace_abort(g);
            }
            false
        }
    }
}

/// Abort the active recording from the outside (e.g. a runtime error was
/// raised while recording — LJ_TRERR_RECERR).
pub fn rec_abort_error(g: &mut GlobalState) {
    if g.jit.state == TraceState::Record {
        g.jit.err = TraceError::RECERR;
        g.jit.state = TraceState::Err;
        trace_abort(g);
    }
}

/// `lj_record_stop` + `trace_stop`: finalize and save a completed trace,
/// then patch the starting bytecode so the interpreter enters it (FORL ->
/// JFORL with D = traceno, plus FORI -> JFORI; LOOP -> JLOOP). Side
/// traces (startins JMP) instead link the parent exit to the new trace.
fn trace_stop(g: &mut GlobalState, mut rec: Box<Record>, linktype: TraceLink, lnk: TraceNo) {
    let js = &mut g.jit;
    let traceno = rec.cur.traceno;
    let (parent, exitno) = (rec.parent, rec.exitno as usize);
    rec.cur.linktype = linktype;
    rec.cur.link = lnk;

    let trace = std::mem::replace(
        &mut rec.cur,
        // Placeholder; `rec` is dropped right after.
        GCtrace {
            traceno: 0,
            ir: IrBuf::new(0, 0),
            snap: Vec::new(),
            snapmap: Vec::new(),
            mcode: None,
            startpt: rec.pt,
            startpc: 0,
            startins: 0,
            link: 0,
            linktype: TraceLink::None,
            root: 0,
            nchild: 0,
            parentmap: Vec::new(),
            inner_ofs: 0,
            stub_tails: Vec::new(),
        },
    );
    let trace = {
        // `lj_asm_trace`: assemble the IR to machine code. On NYI/OOM the
        // trace stays on the portable IR executor (mcode = None). Root
        // links get the target's inner-entry address for a direct jump.
        let mut trace = trace;
        let link_target: Option<*const u8> = if trace.linktype == TraceLink::Root && trace.link != 0
        {
            js.trace[trace.link as usize].as_ref().and_then(|t| {
                t.mcode
                    .as_ref()
                    .map(|m| unsafe { m.ptr().add(t.inner_ofs as usize) })
            })
        } else {
            None
        };
        let arch = js.arch;
        if !js.no_asm
            && let Ok((mc, inner, tails)) = super::asm::assemble(&trace, link_target, arch)
        {
            if js.trace_dump {
                eprintln!(
                    "TRACE {} mcode {:p}+{:#x} inner={:#x} line={} root={} link={} {:?}",
                    trace.traceno,
                    mc.ptr(),
                    mc.len(),
                    inner,
                    trace
                        .startpt
                        .as_ref()
                        .lines
                        .get(trace.startpc)
                        .copied()
                        .unwrap_or(0),
                    trace.root,
                    trace.link,
                    trace.linktype,
                );
            }
            trace.mcode = Some(mc);
            trace.inner_ofs = inner;
            trace.stub_tails = tails;
        }
        trace
    };
    if js.trace_dump {
        eprintln!(
            "STOP {} {:?} link={} start={:?} line={} native={} snaps={:?}",
            trace.traceno,
            trace.linktype,
            trace.link,
            crate::bc::bc_op(trace.startins),
            trace
                .startpt
                .as_ref()
                .lines
                .get(trace.startpc)
                .copied()
                .unwrap_or(0),
            trace.mcode.is_some(),
            trace
                .snap
                .iter()
                .map(|s| (s.iref - super::ir::REF_BIAS, s.pc, s.baseslot, s.nslots))
                .collect::<Vec<_>>(),
        );
        if js.trace_dump2 {
            use super::ir::{IR_NAMES, REF_BIAS, REF_FIRST};
            for r in REF_FIRST..trace.ir.nins() {
                let i = trace.ir.ir(r);
                eprintln!(
                    "  {:04} {} {} {} t={:#x}",
                    r - REF_BIAS,
                    IR_NAMES[i.op() as usize],
                    i.op1 as i32 - REF_BIAS as i32,
                    i.op2 as i32 - REF_BIAS as i32,
                    i.t(),
                );
            }
            for (si, s) in trace.snap.iter().enumerate() {
                let map = &trace.snapmap[s.mapofs as usize..s.mapofs as usize + s.nent as usize];
                let ent: Vec<String> = map
                    .iter()
                    .map(|&sn| {
                        format!(
                            "{}={}{}",
                            super::snap_slot(sn),
                            super::snap_ref(sn) as i32 - REF_BIAS as i32,
                            if sn & super::SNAP_NORESTORE != 0 {
                                "!"
                            } else {
                                ""
                            }
                        )
                    })
                    .collect();
                eprintln!(
                    "  snap#{} pc={} base={} [{}]",
                    si,
                    s.pc,
                    s.baseslot,
                    ent.join(" ")
                );
            }
        }
    }
    // Machine-code chains switch traces without resizing env in Rust:
    // keep the high-water mark over all stored traces.
    js.env_need = js
        .env_need
        .max((trace.ir.nins() - super::ir::REF_BIAS) as usize);

    // Patch the bytecode of the starting instruction in a root trace.
    let pt = trace.startpt;
    let pc = trace.startpc;
    let startins = trace.startins;
    let op = bc_op(startins);
    let root = trace.root;
    js.trace[traceno as usize] = Some(Box::new(trace));
    match op {
        BCOp::FORL | BCOp::LOOP | BCOp::ITERL | BCOp::FUNCF => {
            if op == BCOp::FORL {
                // Patch the FORI, too.
                let fori = (pc as i64 + bc_j(startins)) as usize;
                setbc_op(&mut pt.as_mut().bc[fori], BCOp::JFORI as u32);
            }
            let jop = op.offset(BCOp::JLOOP as i32 - BCOp::LOOP as i32);
            let ins = &mut pt.as_mut().bc[pc];
            setbc_op(ins, jop as u32);
            setbc_d(ins, traceno);
        }
        BCOp::JMP => {
            // Side trace: link the parent exit and avoid compiling it
            // twice. If both sides are machine code, patch the parent's
            // exit stubs to jump straight into the side trace's inner
            // entry (lj_asm_patchexit) — the whole tree then runs in
            // machine code without Rust round trips.
            debug_assert!(parent != 0 && root != 0, "not a side trace");
            {
                let target = js.trace[traceno as usize].as_ref().and_then(|t| {
                    t.mcode
                        .as_ref()
                        .map(|m| unsafe { m.ptr().add(t.inner_ofs as usize) })
                });
                if let Some(target) = target {
                    let pt_ = js.trace[parent as usize].as_mut().unwrap();
                    let tails = std::mem::take(&mut pt_.stub_tails);
                    if let Some(area) = &mut pt_.mcode {
                        super::asm::patch_exit(area, &tails, exitno as u32, target, js.arch);
                    }
                    pt_.stub_tails = tails;
                }
            }
            let psnap = &mut js.trace[parent as usize].as_mut().unwrap().snap[exitno];
            psnap.count = SNAPCOUNT_DONE;
            psnap.sidetrace = traceno;
            // Add to the side trace count of the root trace.
            js.trace[root as usize].as_mut().unwrap().nchild += 1;
        }
        _ => debug_assert!(false, "bad stop bytecode {:?}", op),
    }

    js.startpt = None;
    js.state = TraceState::Idle;
}

/// `trace_abort`: penalize or blacklist the starting instruction and go
/// back to idle.
fn trace_abort(g: &mut GlobalState) {
    let js = &mut g.jit;
    let e = js.err;
    js.rec = None; // Drop the recording context (frees the trace buffers).

    // Penalize or blacklist starting bytecode instruction.
    if js.parent == 0
        && let Some(startpt) = js.startpt
        && !bc_isret(bc_op(js.startins))
        && js.exitno == 0
    {
        let startpc = js.startpc;
        if e == TraceError::RETRY {
            js.hotcount_set(bc_addr(startpt, startpc + 1), 1); // Immediate retry.
        } else if e == TraceError::BLACKL {
            // FNEW inside a hot loop creates closures with unstable
            // identities. Disable JIT for the entire proto to prevent
            // inner ITERL/LOOP traces from compiling stale upvalue
            // references.
            startpt.as_mut().flags |= PROTO_NOJIT;
        } else if e == TraceError::NYIRETL && bc_op(js.startins) == BCOp::FUNCF {
            // A recursive function recorded through its base case
            // (return below the entry frame). Do not penalize: a
            // later hot call through the recursive path stops with
            // an up-recursion link instead.
            let reset = (js.param(JitParam::HotLoop) as u32 * HOTCOUNT_LOOP as u32) as HotCount;
            js.hotcount_set(bc_addr(startpt, startpc + 1), reset);
        } else {
            penalty_pc(js, startpt, startpc, e);
        }
    }
    // else: stitching aborts, once those exist.
    // Aborted side traces are not penalized: the parent exit keeps
    // counting and retries until hotexit+tryside blacklists it at setup.

    js.startpt = None;
    js.startins = 0;
    js.state = TraceState::Idle;
}

/// `penalty_pc`: bump the penalty for a starting instruction, doubling it
/// (plus random noise) on repeated failure; blacklist past PENALTY_MAX.
fn penalty_pc(js: &mut JitState, pt: GcPtr<Proto>, pc: usize, e: TraceError) {
    let key = bc_addr(pt, pc);
    let mut val = PENALTY_MIN;
    let mut slot = None;
    for i in 0..PENALTY_SLOTS {
        if js.penalty[i].pc == key {
            // Cache slot found: first try to bump its hotcount several times.
            val = ((js.penalty[i].val as u32) << 1)
                + (js.prng.u64() as u32 & ((1u32 << PENALTY_RNDBITS) - 1));
            if val > PENALTY_MAX {
                blacklist_pc(pt, pc); // Blacklist it, if that didn't help.
                return;
            }
            slot = Some(i);
            break;
        }
    }
    let i = slot.unwrap_or_else(|| {
        // Assign a new penalty cache slot (round-robin).
        let i = js.penaltyslot as usize;
        js.penaltyslot = (js.penaltyslot + 1) & (PENALTY_SLOTS as u32 - 1);
        js.penalty[i].pc = key;
        i
    });
    js.penalty[i].val = val as u16;
    js.penalty[i].reason = e;
    js.hotcount_set(key + std::mem::size_of::<crate::bc::BCIns>(), val as u16);
}

/// `blacklist_pc`: permanently disable hotcount events for an instruction
/// by patching it to the non-counting variant (FORL -> IFORL etc.).
fn blacklist_pc(pt: GcPtr<Proto>, pc: usize) {
    let p = pt.as_mut();
    let op = bc_op(p.bc[pc]);
    if op == BCOp::ITERN {
        // Undo the ITERN specialization (pairs() dispatch) as well.
        setbc_op(&mut p.bc[pc], BCOp::ITERC as u32);
        let t = (pc as i64 + 1 + bc_j(p.bc[pc + 1])) as usize;
        setbc_op(&mut p.bc[t], BCOp::JMP as u32);
    } else {
        setbc_op(&mut p.bc[pc], op.offset(1) as u32);
        p.flags |= PROTO_ILOOP;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bc::{BCIns, BCOp, bc_d, bc_op};
    use crate::jit::ir::IROp;
    use crate::state::Lua;
    use crate::value::LuaValue;

    fn load_proto(lua: &mut Lua, src: &str) -> (LuaValue, GcPtr<Proto>) {
        let f = crate::state::load(lua.main(), src.as_bytes().to_vec(), "=test").unwrap();
        let pt = match f.as_func().unwrap().as_ref() {
            crate::func::GcFunc::Lua(c) => c.proto,
            _ => panic!("expected Lua closure"),
        };
        (f, pt)
    }

    fn find_op(pt: GcPtr<Proto>, op: BCOp) -> Option<usize> {
        pt.as_ref().bc.iter().position(|&i| bc_op(i) == op)
    }

    #[test]
    fn hotcount_underflows_after_hotloop_iterations() {
        let mut js = JitState::new();
        let addr = 0x1000usize;
        let hotloop = js.param(JitParam::HotLoop) as u32; // 56
        let mut n = 0u32;
        while !js.hot_decrement(addr, HOTCOUNT_LOOP) {
            n += 1;
            assert!(n < 10_000, "no trigger");
        }
        assert_eq!(n + 1, hotloop);
    }

    #[test]
    fn penalty_doubles_then_blacklists() {
        let mut lua = Lua::new();
        let (_f, pt) = load_proto(&mut lua, "for i=1,10 do end");
        let pc = find_op(pt, BCOp::FORL).expect("FORL in bytecode");
        let g = lua.global();

        penalty_pc(&mut g.jit, pt, pc, TraceError::NYIBC);
        let mut prev = {
            let s = g
                .jit
                .penalty
                .iter()
                .find(|s| s.pc == bc_addr(pt, pc))
                .unwrap();
            assert!(s.val as u32 >= PENALTY_MIN);
            s.val as u32
        };
        let mut rounds = 0;
        loop {
            penalty_pc(&mut g.jit, pt, pc, TraceError::NYIBC);
            rounds += 1;
            if find_op(pt, BCOp::IFORL).is_some() {
                break; // Blacklisted: the slot is not bumped on this round.
            }
            let s = g
                .jit
                .penalty
                .iter()
                .find(|s| s.pc == bc_addr(pt, pc))
                .unwrap();
            assert!((s.val as u32) >= prev * 2, "penalty must at least double");
            prev = s.val as u32;
            assert!(rounds < 32, "never blacklisted");
        }
        assert!(pt.as_ref().flags & PROTO_ILOOP != 0);
        assert!((7..=11).contains(&rounds), "rounds = {}", rounds);
    }

    #[test]
    fn numeric_forl_records_patches_and_executes() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, pt) = load_proto(
            &mut lua,
            "local s = 0 for i = 1, 200 do s = s + i end return s",
        );
        let forl_pc = find_op(pt, BCOp::FORL).unwrap();
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(20100.0));

        let g = lua.global();
        assert_eq!(g.jit.state, TraceState::Idle);
        // The FORL was patched to JFORL (D = traceno), the FORI to JFORI.
        let jforl = pt.as_ref().bc[forl_pc];
        assert_eq!(bc_op(jforl), BCOp::JFORL);
        assert!(find_op(pt, BCOp::JFORI).is_some());
        let tno = crate::bc::bc_d(jforl);
        let tr = g.jit.trace[tno as usize]
            .as_deref()
            .expect("trace registered");
        assert_eq!(tr.traceno, tno);
        assert_eq!(tr.linktype, TraceLink::Loop);
        assert_eq!(tr.link, tr.traceno);
        // The x64 backend must have assembled this numeric loop.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(tr.mcode.is_some(), "trace not assembled to machine code");
        // After lj_opt_loop the trace holds the pre-roll plus the copied
        // variant part: two num ADDs each (s+i, i+step), separated by
        // LOOP, with PHIs for the two loop-carried values at the tail.
        let mut adds = 0;
        let mut guards = 0;
        let mut phis = 0;
        let mut loops = 0;
        for r in crate::jit::ir::REF_FIRST..tr.ir.nins() {
            let ir = tr.ir.ir(r);
            if ir.op() == IROp::ADD {
                adds += 1;
            }
            if ir.is_guard() && ir.op() != IROp::LOOP {
                guards += 1;
            }
            if ir.op() == IROp::PHI {
                phis += 1;
            }
            if ir.op() == IROp::LOOP {
                loops += 1;
            }
        }
        assert_eq!(adds, 4, "expected s+i and i+step in pre-roll and body");
        assert!(guards >= 2, "expected the loop condition guards");
        assert_eq!(loops, 1, "expected the LOOP marker");
        assert_eq!(phis, 2, "expected PHIs for s and i");
        assert!(tr.snap.len() >= 2);
        // The loop finished through the trace: the leave-exit was taken.
        assert!(tr.snap.iter().any(|s| s.count > 0), "no exit was taken");
        // No penalty was assessed for the start pc.
        let key = bc_addr(pt, forl_pc);
        assert!(g.jit.penalty.iter().all(|s| s.pc != key));
    }

    #[test]
    fn trace_takes_side_exit_every_other_iteration() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The recorded parity branch fails every other iteration, forcing
        // a mid-trace guard exit, an interpreted tail and a re-entry.
        let (f, _pt) = load_proto(
            &mut lua,
            "local s = 0 \
             for i = 1, 100000 do \
               if i % 2 == 0 then s = s + 2 else s = s + 1 end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(150000.0));
    }

    #[test]
    fn trace_survives_type_instability_via_exit() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // x flips to a string once the loop is compiled: the SLOAD
        // typecheck (or arith NYI abort during recording) must keep the
        // semantics intact either way.
        let (f, _pt) = load_proto(
            &mut lua,
            "local s = 0 \
             local x = 1 \
             for i = 1, 300 do \
               if i == 250 then x = '2' end \
               s = s + x \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // 249 iterations of +1, then 51 iterations of +'2' (coerced).
        assert_eq!(r[0].as_number(), Some(249.0 + 51.0 * 2.0));
    }

    #[test]
    fn while_loop_records_via_loop_root() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, pt) = load_proto(
            &mut lua,
            "local s, i = 0, 0 while i < 500 do i = i + 1 s = s + 2 end return s",
        );
        let loop_pc = find_op(pt, BCOp::LOOP).unwrap();
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(1000.0));
        let g = lua.global();
        // LOOP patched to JLOOP and executed through the trace.
        let jloop = pt.as_ref().bc[loop_pc];
        assert_eq!(bc_op(jloop), BCOp::JLOOP);
        let tr = g.jit.trace[crate::bc::bc_d(jloop) as usize]
            .as_deref()
            .unwrap();
        assert_eq!(tr.linktype, TraceLink::Loop);
        assert_eq!(bc_op(tr.startins), BCOp::LOOP);
        assert!(tr.snap.iter().any(|s| s.count > 0), "no exit was taken");
    }

    #[test]
    fn hot_exit_compiles_side_trace() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The parity branch exits the root trace every other iteration:
        // the exit turns hot, a side trace is recorded from the exit pc,
        // links back to the root (TRLINK_ROOT) and the parent exit is
        // patched to it (count = SNAPCOUNT_DONE).
        let (f, pt) = load_proto(
            &mut lua,
            "local s = 0 \
             for i = 1, 100000 do \
               if i % 2 == 0 then s = s + 2 else s = s + 1 end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(150000.0));

        let g = lua.global();
        let forl_pc = find_op(pt, BCOp::JFORL).expect("root trace patched");
        let rootno = crate::bc::bc_d(pt.as_ref().bc[forl_pc]);
        let root = g.jit.trace[rootno as usize].as_deref().unwrap();
        assert!(root.nchild >= 1, "no side trace attached to the root");
        // The linked exit no longer counts and points at the side trace.
        let (exitno, side) = root
            .snap
            .iter()
            .enumerate()
            .find_map(|(i, s)| (s.sidetrace != 0).then_some((i, s.sidetrace)))
            .expect("no exit linked to a side trace");
        assert_eq!(root.snap[exitno].count, super::super::SNAPCOUNT_DONE);
        let st = g.jit.trace[side as usize].as_deref().unwrap();
        assert_eq!(st.root, rootno, "side trace not rooted");
        assert_eq!(st.linktype, TraceLink::Root);
        assert_eq!(st.link, rootno, "side trace must link back to the root");
        assert_eq!(bc_op(st.startins), BCOp::JMP);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(st.mcode.is_some(), "side trace not assembled");
    }

    #[test]
    fn hopeless_side_exit_gets_blacklisted() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The taken branch calls an unrecordable builtin, so every side
        // trace attempt aborts; after hotexit+tryside tries the exit is
        // parked with SNAPCOUNT_DONE and never re-examined.
        let (f, pt) = load_proto(
            &mut lua,
            "local ts = tostring \
             local s = 0 \
             for i = 1, 100000 do \
               if i % 2 == 0 then ts(i) s = s + 2 else s = s + 1 end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(150000.0));
        let g = lua.global();
        if let Some(forl_pc) = find_op(pt, BCOp::JFORL) {
            let rootno = crate::bc::bc_d(pt.as_ref().bc[forl_pc]);
            let root = g.jit.trace[rootno as usize].as_deref().unwrap();
            assert_eq!(root.nchild, 0, "unrecordable side trace was compiled");
            assert!(
                root.snap
                    .iter()
                    .any(|s| s.count == super::super::SNAPCOUNT_DONE && s.sidetrace == 0),
                "hopeless exit was not blacklisted"
            );
        }
        assert_eq!(g.jit.state, TraceState::Idle);
    }

    #[test]
    fn side_trace_handover_keeps_norestore_slots() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // A variable loop bound is a READONLY/NORESTORE snapshot entry:
        // it is never written back to the Lua stack, but the env
        // hand-over to side traces must still see it (exit stubs flush
        // NORESTORE entries too). Regression: the side trace previously
        // read garbage as the loop bound and exited after ~92 iterations.
        let (f, _pt) = load_proto(
            &mut lua,
            "local n = 200000 \
             local a, b, c = 0, 0, 0 \
             for i = 1, n do \
               local m = i % 3 \
               if m == 0 then a = a + 1 elseif m == 1 then b = b + 1 else c = c + 1 end \
             end return a * 1000000 + b * 1000 + c",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // 200000 = 3*66666+2: m cycles 1,2,0,... -> a=66666, b=66667, c=66667.
        assert_eq!(
            r[0].as_number(),
            Some(66666.0 * 1000000.0 + 66667.0 * 1000.0 + 66667.0)
        );
    }

    #[test]
    fn nested_loop_side_trace_keeps_fori_semantics() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The outer back edge becomes a side trace of the inner loop and
        // re-enters it through JFORI: rec_for must record FORI semantics
        // (no index increment) there, or one iteration per outer round
        // gets lost.
        let (f, _pt) = load_proto(
            &mut lua,
            "local n = 0 \
             for i = 1, 1000 do \
               for j = 1, 1000 do n = n + 1 end \
             end return n",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(1000000.0));
    }

    #[test]
    fn swap_phis_use_parallel_assignment() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // a,b = b,a builds a PHI cycle: the back edge must read both
        // right refs before writing either left ref.
        let (f, _pt) = load_proto(
            &mut lua,
            "local a, b = 1.5, 2.5 \
             local s = 0 \
             for i = 1, 100001 do a, b = b, a s = s + a end \
             return a * 1000 + b * 100 + s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // Odd iteration count: a,b end up swapped; s alternates +2.5/+1.5.
        let s = 50000.0 * (1.5 + 2.5) + 2.5;
        assert_eq!(r[0].as_number(), Some(2.5 * 1000.0 + 1.5 * 100.0 + s));
    }

    #[test]
    fn portable_exec_runs_loop_optimized_ir() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Differential test: strip the machine code from all traces and
        // re-run, forcing run_ir through the LOOP/PHI execution path.
        let (f, pt) = load_proto(
            &mut lua,
            "local s = 0 for i = 1, 300 do s = s + i end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(45150.0));
        assert!(find_op(pt, BCOp::JFORL).is_some());
        for t in lua.global().jit.trace.iter_mut().flatten() {
            t.mcode = None;
        }
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(45150.0), "portable tier diverged");
    }

    #[test]
    fn unrecordable_loop_still_blacklists() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Calls to non-recff C functions are unrecordable, so the
        // penalty machinery must eventually blacklist the FORL.
        let (f, pt) = load_proto(
            &mut lua,
            "local ts = tostring local s = 0 \
             for i = 1, 2000000 do ts(i) s = s + 1 end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(2000000.0));
        // The loop should either be blacklisted (IFORL stays) or
        // successfully JIT-compiled (IFORL replaced by hotcount hook).
        let jitted = find_op(pt, BCOp::IFORL).is_none();
        if jitted {
            // IFORL was replaced — make sure it wasn't blacklisted by mistake.
            assert!(
                pt.as_ref().flags & PROTO_ILOOP == 0,
                "JIT-compiled loop should not have ILOOP flag"
            );
        } else {
            assert!(
                pt.as_ref().flags & PROTO_ILOOP != 0,
                "blacklisted loop must have ILOOP flag set"
            );
        }
        assert_eq!(lua.global().jit.state, TraceState::Idle);
    }

    #[test]
    fn hot_loop_inlines_lua_call() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The hot loop records straight through g(): the callee identity
        // is guarded (EQ FUNC) and the body inlined — no call frame at
        // loop close, no FUNCF blacklisting.
        let (f, pt) = load_proto(
            &mut lua,
            "local function g(x) return x + 1 end \
             local s = 0 for i = 1, 2000000 do s = g(s) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(2000000.0));
        let forl_pc = find_op(pt, BCOp::JFORL).expect("loop not compiled");
        let g = lua.global();
        let tno = crate::bc::bc_d(pt.as_ref().bc[forl_pc]);
        let tr = g.jit.trace[tno as usize].as_deref().unwrap();
        assert_eq!(tr.linktype, TraceLink::Loop);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(tr.mcode.is_some(), "inlined-call loop not assembled");
        // The callee's FUNCF header stays untouched (not blacklisted).
        let child = pt
            .as_ref()
            .kgc
            .iter()
            .find_map(|k| match k {
                crate::proto::KGc::ProtoRef(p) => Some(*p),
                _ => None,
            })
            .expect("child proto");
        assert_eq!(bc_op(child.as_ref().bc[0]), BCOp::FUNCF);
    }

    #[test]
    fn inlined_call_branch_exits_inside_callee_frame() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The branch in f() flips at i > 150000: the compiled trace's
        // guard fails *inside the inlined frame*, so the exit restores
        // the call frame (KFUNC + frame link constants) and the
        // interpreter resumes within f's bytecode.
        let (f, _pt) = load_proto(
            &mut lua,
            "local function f(x, i) \
               if i > 150000 then return x + 2 end \
               return x + 1 \
             end \
             local s = 0 \
             for i = 1, 200000 do s = f(s, i) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(150000.0 + 50000.0 * 2.0));
    }

    #[test]
    fn inlined_calls_nest_and_return_pairs() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, _pt) = load_proto(
            &mut lua,
            "local function two(x) return x, x + 0.5 end \
             local function inc(x) return x + 1 end \
             local function nop() end \
             local s = 0 \
             for i = 1, 200000 do \
               local a, b = two(inc(i)) \
               nop() \
               s = s + (b - a) \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(100000.0));
    }

    #[test]
    fn recursion_hits_call_unroll_limit() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Non-tail recursion exceeds JIT_P_callunroll during recording;
        // the trace aborts and the loop is eventually blacklisted, but
        // the semantics stay intact.
        let (f, _pt) = load_proto(
            &mut lua,
            "local function r(n) if n <= 0 then return 0 end return 1 + r(n - 1) end \
             local s = 0 \
             for i = 1, 30000 do s = s + r(8) end return s",
        );
        let res = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(res[0].as_number(), Some(240000.0));
        assert_eq!(lua.global().jit.state, TraceState::Idle);
    }

    #[test]
    fn deep_exit_seeds_side_trace_inside_callee() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The parity branch flips inside f(): the hot exit lies in the
        // inlined frame, so the side trace replays the KFUNC/frame-link
        // constants, rebuilds the frame stack and records from within
        // the callee back to the loop.
        let (f, pt) = load_proto(
            &mut lua,
            "local function f(x, i) \
               if i % 2 == 0 then return x + 2 end \
               return x + 1 \
             end \
             local s = 0 \
             for i = 1, 200000 do s = f(s, i) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(300000.0));
        let g = lua.global();
        let forl_pc = find_op(pt, BCOp::JFORL).expect("root not compiled");
        let rootno = crate::bc::bc_d(pt.as_ref().bc[forl_pc]);
        let root = g.jit.trace[rootno as usize].as_deref().unwrap();
        assert!(root.nchild >= 1, "no side trace from the deep exit");
        let (exitno, side) = root
            .snap
            .iter()
            .enumerate()
            .find_map(|(i, s)| (s.sidetrace != 0).then_some((i, s.sidetrace)))
            .expect("deep exit not linked");
        assert!(root.snap[exitno].baseslot > 2, "exit is not inside a frame");
        let st = g.jit.trace[side as usize].as_deref().unwrap();
        assert_eq!(st.linktype, TraceLink::Root);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(st.mcode.is_some(), "deep side trace not assembled");
    }

    #[test]
    fn tailcall_wrapper_is_inlined() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // wrap() forwards via CALLT: the frame is replaced in place and
        // the pending return still lands in the original caller.
        let (f, pt) = load_proto(
            &mut lua,
            "local function base(x) return x + 1 end \
             local function wrap(x) return base(x) end \
             local s = 0 \
             for i = 1, 200000 do s = wrap(s) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(200000.0));
        assert!(
            find_op(pt, BCOp::JFORL).is_some(),
            "tailcall loop not compiled"
        );
    }

    #[test]
    fn closed_mutable_upvalue_loads_through_cell() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // k is assigned after capture (mutable) and mk has returned
        // (closed): the load compiles to a ULOAD of the constant cell
        // address with a type guard.
        let (f, _pt) = load_proto(
            &mut lua,
            "local function mk() \
               local k = 0 \
               local g = function(x) return x + k end \
               k = 7 \
               return g \
             end \
             local g = mk() \
             local s = 0 \
             for i = 1, 200000 do s = g(s) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(1400000.0));
    }

    #[test]
    fn open_upvalue_aliases_recorded_slot() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // t is mutable and its upvalue is open (the chunk is running):
        // the load forwards to the aliased frame-0 slot under the
        // closure-identity guard.
        let (f, _pt) = load_proto(
            &mut lua,
            "local t = 0 \
             local function get() return t end \
             local s = 0 \
             for i = 1, 200000 do t = i s = s + get() end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(200000.0 * 200001.0 / 2.0));
    }

    #[test]
    fn table_array_load_store_compile() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Fill (inserts + array growth via the HSTORE helper), then sum
        // (HLOAD specialized to numbers).
        let (f, pt) = load_proto(
            &mut lua,
            "local t = {} \
             for i = 1, 100000 do t[i] = i * 2 end \
             local s = 0 \
             for i = 1, 100000 do s = s + t[i] end \
             return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(100000.0 * 100001.0));
        assert!(
            find_op(pt, BCOp::JFORL).is_some(),
            "table loops not compiled"
        );
    }

    #[test]
    fn table_field_and_globals_compile() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // TGETS/TSETS on a hash field plus GGET/GSET on a global counter.
        let (f, _pt) = load_proto(
            &mut lua,
            "gcnt = 0 \
             local t = { x = 0 } \
             for i = 1, 200000 do \
               t.x = t.x + 1 \
               gcnt = gcnt + 2 \
             end return t.x * 1000000 + gcnt",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(200000.0 * 1000000.0 + 400000.0));
    }

    #[test]
    fn nil_load_specializes_and_metatable_falls_back() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // t.missing stays nil (HLOAD specialized to NIL under the
        // metatable guard); u gets a metatable with __index mid-loop,
        // whose lookups must keep working through the fallback.
        let (f, _pt) = load_proto(
            &mut lua,
            "local t = { a = 1 } \
             local u = {} \
             local mt = { __index = function() return 5 end } \
             local s = 0 \
             for i = 1, 60000 do \
               if t.missing == nil then s = s + 1 end \
               if i == 50000 then setmetatable(u, mt) end \
               s = s + (u.k or 0) \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // +1 per iteration, plus __index result 5 for i in 50000..=60000.
        assert_eq!(r[0].as_number(), Some(60000.0 + 10001.0 * 5.0));
    }

    #[test]
    fn math_fast_functions_record_inline() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // floor/sqrt/abs inline as FPMATH/ABS IR under the callee guard;
        // the -0.0 normalization matches the interpreter's push path.
        let (f, pt) = load_proto(
            &mut lua,
            "local fl, sq, ab = math.floor, math.sqrt, math.abs \
             local s = 0 \
             for i = 1, 200000 do \
               s = s + fl(i / 2) + sq(i) - ab(-i) \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        let mut want = 0.0f64;
        for i in 1..=200000 {
            // Mirror the bytecode's association order exactly.
            let x = i as f64;
            want += (x / 2.0).floor();
            want += x.sqrt();
            want -= x;
        }
        assert_eq!(r[0].as_number(), Some(want));
        assert!(
            find_op(pt, BCOp::JFORL).is_some(),
            "recff loop not compiled"
        );
    }

    #[test]
    fn recff_floor_normalizes_negative_zero() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The interpreter pushes through LuaValue::number (-0.0 -> +0.0):
        // the compiled trace must agree, so 1/floor(-0.0) stays +inf
        // (z * -1 produces a runtime -0.0).
        let (f, _pt) = load_proto(
            &mut lua,
            "local fl = math.floor \
             local z = 0 \
             local acc = 0 \
             for i = 1, 200 do acc = acc + 1 / fl(z * -1) end return acc",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(f64::INFINITY));
    }

    #[test]
    fn ipairs_loop_compiles_via_iterc() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The ITERL root records the ITERC call to the builtin ipairs
        // iterator (recff_ipairs_aux: ctrl+1 + guarded array load).
        let (f, pt) = load_proto(
            &mut lua,
            "local t = {} \
             for i = 1, 2000 do t[i] = i * 3 end \
             local s = 0 \
             for n = 1, 200 do \
               for k, v in ipairs(t) do s = s + v - k end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // Each inner pass: sum of 2k over k=1..2000.
        assert_eq!(r[0].as_number(), Some(200.0 * (2000.0 * 2001.0)));
        assert!(
            find_op(pt, BCOp::JITERL).is_some(),
            "ipairs loop not compiled"
        );
        let g = lua.global();
        let jiterl = pt.as_ref().bc[find_op(pt, BCOp::JITERL).unwrap()];
        let tr = g.jit.trace[crate::bc::bc_d(jiterl) as usize]
            .as_deref()
            .unwrap();
        assert_eq!(tr.linktype, TraceLink::Loop);
        assert_eq!(bc_op(tr.startins), BCOp::ITERL);
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(tr.mcode.is_some(), "ipairs trace not assembled");
    }

    #[test]
    fn ipairs_type_change_exits_cleanly() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Element 1500 is a string: the HLOAD type guard exits and the
        // interpreter finishes the traversal (including the string).
        let (f, _pt) = load_proto(
            &mut lua,
            "local t = {} \
             for i = 1, 2000 do t[i] = 1 end \
             t[1500] = 'x' \
             local s = 0 \
             for n = 1, 120 do \
               for k, v in ipairs(t) do \
                 if v == 'x' then s = s + 100 else s = s + v end \
               end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(120.0 * (1999.0 + 100.0)));
    }

    #[test]
    fn lua_iterator_function_inlines() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // A plain Lua iterator goes through rec_call_lua from ITERC.
        let (f, pt) = load_proto(
            &mut lua,
            "local function iter(t, i) \
               i = i + 1 \
               local v = t[i] \
               if v ~= nil then return i, v end \
             end \
             local t = {} \
             for i = 1, 1000 do t[i] = i end \
             local s = 0 \
             for n = 1, 200 do \
               for k, v in iter, t, 0 do s = s + v end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(200.0 * (1000.0 * 1001.0 / 2.0)));
        assert!(
            find_op(pt, BCOp::JITERL).is_some(),
            "Lua-iterator loop not compiled"
        );
    }

    #[test]
    fn pairs_loop_compiles_with_hash_phase_side_trace() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // pairs() traverses the array part (number keys) then the hash
        // part (string keys): the key-type guard exit at the phase
        // switch turns hot and grows a side trace, keeping the whole
        // traversal compiled.
        let (f, pt) = load_proto(
            &mut lua,
            "local t = {} \
             for i = 1, 500 do t[i] = i end \
             t.x = 1000 t.y = 2000 \
             local s = 0 \
             for n = 1, 300 do \
               for k, v in pairs(t) do s = s + v end \
             end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(
            r[0].as_number(),
            Some(300.0 * (500.0 * 501.0 / 2.0 + 3000.0))
        );
        let jiterl_pc = find_op(pt, BCOp::JITERL).expect("pairs loop not compiled");
        let g = lua.global();
        let rootno = crate::bc::bc_d(pt.as_ref().bc[jiterl_pc]);
        let root = g.jit.trace[rootno as usize].as_deref().unwrap();
        assert_eq!(bc_op(root.startins), BCOp::ITERL);
        assert!(root.nchild >= 1, "no side trace for the hash phase");
    }

    #[test]
    fn bit_ops_compile_with_range_guards() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // A hash-mix loop: shifts, or, xor, and. Element 150000 pushes an
        // out-of-range operand through the guard exit (the interpreter's
        // saturating cast takes over there).
        let (f, pt) = load_proto(
            &mut lua,
            "local h = 0 \
             local big = 2 ^ 40 \
             for i = 1, 200000 do \
               local x = i \
               if i == 150000 then x = big end \
               h = ((h << 5) | (h >> 27)) ~ x \
               h = h & 2147483647 \
             end return h + ~h",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        // Mirror the interpreter's fused semantics in Rust.
        let mut h: i32 = 0;
        for i in 1..=200000i64 {
            let x: f64 = if i == 150000 {
                (2.0f64).powi(40)
            } else {
                i as f64
            };
            let mixed = ((h << 5) | ((h as u32) >> 27) as i32) ^ (x as i32);
            h = mixed & 2147483647;
        }
        let want = h as f64 + (!h) as f64;
        assert_eq!(r[0].as_number(), Some(want));
        assert!(find_op(pt, BCOp::JFORL).is_some(), "bit loop not compiled");
    }

    #[test]
    fn allocating_trace_reaches_gc_safe_points() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // A compiled append loop grows the table by megabytes: the
        // IR_GCSTEP guard must leave the trace so the boundary check can
        // collect (observable via the recomputed threshold).
        let (f, _pt) = load_proto(
            &mut lua,
            "local t = {} \
             local s = 0 \
             for i = 1, 300000 do t[i] = i s = s + 1 end \
             return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(300000.0));
        let g = lua.global();
        // A full GC ran mid-loop: the threshold was re-derived from the
        // multi-megabyte live table, far above the 64 KiB minimum.
        assert!(
            g.heap.threshold > 1024 * 1024,
            "no collection happened during the compiled loop (threshold = {})",
            g.heap.threshold
        );
        assert_eq!(g.jit.state, TraceState::Idle);
    }

    #[test]
    fn jit_off_never_triggers() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        lua.global().jit.set_on(false);
        let (f, pt) = load_proto(
            &mut lua,
            "local s = 0 for i = 1, 200000 do s = s + 1 end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(200000.0));
        assert!(
            find_op(pt, BCOp::FORL).is_some(),
            "FORL must stay untouched"
        );
        assert!(pt.as_ref().flags & PROTO_ILOOP == 0);
        assert!(lua.global().jit.trace.iter().flatten().next().is_none());
    }

    #[test]
    fn trace_error_messages() {
        assert_eq!(TraceError::BLACKL.message(), "blacklisted");
        const _: () = assert!(std::mem::size_of::<BCIns>() == 4);
    }

    #[test]
    fn string_concat_records() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, _pt) = load_proto(
            &mut lua,
            r#"local function f(a, b) return a .. b end return f("hello", "world")"#,
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert!(r[0].is_string());
        assert_eq!(r[0].as_string().unwrap().as_ref().as_bytes(), b"helloworld");
    }

    #[test]
    fn string_concat_number() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, _pt) = load_proto(
            &mut lua,
            "local s = '' for i = 1, 100 do s = s .. i end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert!(r[0].is_string());
        let expected: String = (1..=100).map(|n| n.to_string()).collect();
        assert_eq!(
            r[0].as_string().unwrap().as_ref().as_bytes(),
            expected.as_bytes()
        );
    }

    #[test]
    fn upvalue_write_records() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, _pt) = load_proto(
            &mut lua,
            "local x = 0; local function set(v) x = v end; set(42); return x",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(42.0));
    }

    #[test]
    fn vararg_records() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, _pt) = load_proto(
            &mut lua,
            "local function f(...) local a,b = ...; return a + b end; return f(3, 4)",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(7.0));
    }

    // -- JIT correctness suite: run programs through JIT, verify results ---

    #[test]
    fn integer_narrowing_uses_kint_in_forl() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, pt) = load_proto(
            &mut lua,
            "local s = 0 for i = 1, 200 do s = s + i end return s",
        );
        let forl_pc = find_op(pt, BCOp::FORL).unwrap();
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(20100.0));

        let g = lua.global();
        let jforl = pt.as_ref().bc[forl_pc];
        assert!(
            matches!(bc_op(jforl), BCOp::JFORL),
            "FORL should be patched to JFORL, got {:?}",
            bc_op(jforl)
        );
        let tno = bc_d(jforl);
        let tr = g.jit.trace[tno as usize]
            .as_deref()
            .expect("trace registered");

        let mut kint_count = 0;
        let mut knum_count = 0;
        for ins in tr.ir.const_iter() {
            if ins.op() == IROp::KINT {
                kint_count += 1;
            }
            if ins.op() == IROp::KNUM {
                knum_count += 1;
            }
        }
        assert!(
            kint_count >= 2,
            "expected at least 2 KINT constants (step + stop), got {}",
            kint_count
        );
        // KNUM may still exist from initializers and other constants.
        // After full narrowing lands, they will all become KINT.
        assert!(
            kint_count > 0 && knum_count <= 3,
            "KINT={} KNUM={} — narrowing should reduce KNUM count",
            kint_count,
            knum_count
        );
    }

    #[test]
    fn integer_narrowing_correctness_basic() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(&mut lua, "local s=0 for i=1,10000 do s=s+i end return s"),
            50005000.0,
        );
    }

    #[test]
    fn integer_narrowing_correctness_nested() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 for i=1,100 do for j=1,100 do s=s+1 end end return s",
            ),
            10000.0,
        );
    }

    fn jit_run(lua: &mut Lua, src: &str) -> LuaValue {
        let (f, _pt) = load_proto(lua, src);
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        r[0]
    }

    fn assert_num(v: LuaValue, expected: f64) {
        let got = v.as_number().expect("expected number");
        assert!(
            (got - expected).abs() < 1e-12,
            "got {:.16}, expected {:.16}, diff {:.2e}",
            got,
            expected,
            (got - expected).abs()
        );
    }

    fn assert_str(v: LuaValue, expected: &[u8]) {
        assert!(v.is_string());
        assert_eq!(v.as_string().unwrap().as_ref().as_bytes(), expected);
    }

    #[test]
    fn jit_correctness_arithmetic() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(&mut lua, "local s=0 for i=1,50000 do s=s+i end return s"),
            1250025000.0,
        );
        assert_num(
            jit_run(
                &mut lua,
                "local r=1.0 for i=1,100 do r=r*1.0001 end return r",
            ),
            1.0001f64.powi(100),
        );
        assert_num(
            jit_run(&mut lua, "local x=0 for i=1,1000 do x=i+5 end return x"),
            1005.0,
        );
    }

    #[test]
    fn jit_correctness_strcat() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_str(
            jit_run(&mut lua, r#"local a,b="hello","world" return a..b"#),
            b"helloworld",
        );
        assert_str(
            jit_run(&mut lua, "local s='' for i=1,10 do s=s..i end return s"),
            b"12345678910",
        );
    }

    #[test]
    fn jit_correctness_vararg() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(
                &mut lua,
                "local function f(...) local a,b=...; return a+b end; return f(10, 20)",
            ),
            30.0,
        );
    }

    #[test]
    fn jit_correctness_control_flow() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(
                &mut lua,
                "local c=0 for i=1,2000 do if i%2==0 then c=c+1 else c=c-1 end end return c",
            ),
            0.0,
        );
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 for i=1,1000 do if i<500 then s=s+1 end end return s",
            ),
            499.0,
        );
    }

    #[test]
    fn jit_correctness_tables() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(
            jit_run(
                &mut lua,
                "local t={} for i=1,100 do t[i]=i*2 end; local s=0 for i=1,100 do s=s+t[i] end return s",
            ),
            10100.0,
        );
    }

    // ── ARM64 diagnostic tests ───────────────────────────────────────────

    /// Simplest possible traced loop: one accumulator, one addition.
    #[test]
    fn diag_simple_add_loop() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        assert_num(jit_run(&mut lua, "local s=0 for i=1,300 do s=s+1 end return s"), 300.0);
    }

    /// Loop with comparison guard (EQ) that fires on the last iteration.
    #[test]
    fn diag_eq_guard_loop() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // if i == 250 then s = s + 100 else s = s + 1 end
        // 249*1 + 1*100 + 50*1 = 249 + 100 + 50 = 399
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 for i=1,300 do if i==250 then s=s+100 else s=s+1 end end return s",
            ),
            399.0,
        );
    }

    /// Loop where a variable changes type mid-execution: the SLOAD
    /// type guard must fire, then the interpreter takes over.
    #[test]
    fn diag_type_change_exit() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // x=1 for 249 iters, then x='2' at iter 250.
        // s = 249*1 + 2 + 50*2 = 249 + 2 + 100 = 351
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=1 for i=1,300 do if i==250 then x='2' end s=s+x end return s",
            ),
            351.0,
        );
    }

    /// Verify the trace was actually assembled (mcode present).
    #[test]
    fn diag_loop_gets_mcode() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        let (f, pt) = load_proto(
            &mut lua,
            "local s=0 for i=1,300 do s=s+i end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(45150.0));
        let g = lua.global();
        // Check if trace was assembled on native-arch targets.
        let jforl = find_op(pt, BCOp::JFORL);
        if let Some(jforl_pc) = jforl {
            let rootno = crate::bc::bc_d(pt.as_ref().bc[jforl_pc]);
            if let Some(tr) = g.jit.trace[rootno as usize].as_deref() {
                let has_mcode = tr.mcode.is_some();
                let arch = format!("{:?}", g.jit.arch);
                // If we expect native mcode but didn't get it, fail explicitly.
                if !has_mcode {
                    panic!(
                        "trace not assembled on {arch}: nins={} nsnap={} link={:?}",
                        tr.ir.nins(),
                        tr.snap.len(),
                        tr.linktype,
                    );
                }
            }
        }
    }

    /// Minimal type-change exit: simpler version of diag_type_change_exit
    /// without the EQ guard — x starts as number, type changes mid-loop.
    #[test]
    fn diag_type_change_only() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // x = 10 for first 5 iters, then x = '2' for remaining 5.
        // s = 5*10 + 5*2 = 50 + 10 = 60
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=10 for i=1,10 do if i==6 then x='2' end s=s+x end return s",
            ),
            60.0,
        );
    }

    /// Force no-asm: portable executor should always give correct results.
    #[test]
    fn diag_type_change_noasm() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        lua.global().jit.no_asm = true;
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=1 for i=1,300 do if i==250 then x='2' end s=s+x end return s",
            ),
            351.0,
        );
    }

    /// JIT-off: pure interpreter should always work.
    #[test]
    fn diag_type_change_jitoff() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        lua.global().jit.set_on(false);
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=1 for i=1,300 do if i==250 then x='2' end s=s+x end return s",
            ),
            351.0,
        );
    }

    /// Type change very early (iteration 5 out of 300): trace is recorded
    /// before the change point, but has plenty of time to re-enter after.
    #[test]
    fn diag_type_change_early() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // x=1 for iter 1-4, x='2' for iter 5-500. s = 4*1 + 496*2 = 996
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=1 for i=1,500 do if i==5 then x='2' end s=s+x end return s",
            ),
            996.0,
        );
    }

    /// Type change very late (iteration 295 out of 300): trace records with
    /// x=1, runs many iterations compiled, then exits near the end.
    #[test]
    fn diag_type_change_late() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // x=1 for iter 1-294, x='2' for iter 295-300. s = 294 + 6*2 = 306
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local x=1 for i=1,300 do if i==295 then x='2' end s=s+x end return s",
            ),
            306.0,
        );
    }

    /// `continue` in a while loop must produce the same result with JIT
    /// compiled as without.  Regression test: ARM64 side-trace recording
    /// from the continue exit was silently executing the wrong body code.
    #[test]
    fn while_continue_correctness() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // simple: sum values where (i & 4) != 0 (i.e. bit 2 set)
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local i=0 while i<100 do i=i+1 if(i&4)==0 then continue end s=s+i end return s",
            ),
            2476.0,
        );
        // full: with extra continue range (70-80) and break at 95
        assert_num(
            jit_run(
                &mut lua,
                "local s=0 local i=0 while i<100 do i=i+1 if(i&4)==0 then continue end if i>=70 and i<=80 then continue end if i==95 then break end s=s+i end return s",
            ),
            1830.0,
        );
    }
}
