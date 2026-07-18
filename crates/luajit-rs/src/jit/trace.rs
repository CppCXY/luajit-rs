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

use crate::bc::{BCOp, bc_isret, bc_j, bc_op, setbc_d, setbc_op};
use crate::gc::GcPtr;
use crate::proto::{PROTO_ILOOP, PROTO_NOJIT, Proto};
use crate::state::{GlobalState, LuaState};

use super::record::Record;
use super::{
    HOTCOUNT_LOOP, HotCount, JitParam, JitState, PENALTY_MAX, PENALTY_MIN, PENALTY_RNDBITS,
    PENALTY_SLOTS, TraceError, TraceLink, TraceNo, TraceState, bc_addr,
};

/// Hot counter value that parks an already-compiled start instruction
/// until real J* bytecode patching exists.
const HOTCOUNT_PARKED: HotCount = 0xffff;

/// `lj_trace_hot`: a hot counter underflowed at `pt.bc[pc]` (a FORL /
/// ITERL / LOOP or a FUNCF header). Reset the counter and start a root
/// trace unless the compiler is busy.
pub fn trace_hot(l: &mut LuaState, base: usize, pt: GcPtr<Proto>, pc: usize) {
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

/// `trace_start` + `lj_record_setup` + `rec_setup_root`: begin recording
/// a root trace at `pt.bc[pc]`, or bail out silently.
fn trace_start(l: &mut LuaState, base: usize, pt: GcPtr<Proto>, pc: usize) {
    let g = l.global();
    let js = &mut g.jit;
    let startins = pt.as_ref().bc[pc];
    let op = bc_op(startins);

    if pt.as_ref().flags & PROTO_NOJIT != 0 {
        // JIT disabled for this proto: lazily patch the hot instruction to
        // its non-counting variant so it stops triggering.
        if js.parent == 0 && js.exitno == 0 && op != BCOp::ITERN {
            debug_assert!(
                matches!(op, BCOp::FORL | BCOp::ITERL | BCOp::LOOP | BCOp::FUNCF | BCOp::FUNCV),
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

    // Phase 2 recorder handles FORL/LOOP/FUNCF roots; penalize the rest
    // (ITERL/ITERN roots arrive with iterator recording).
    if !matches!(op, BCOp::FORL | BCOp::LOOP | BCOp::FUNCF) {
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
    );

    // rec_setup_root: determine the next PC and the loop bytecode range.
    match op {
        BCOp::FORL => {
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
    if 1 + pt.as_ref().framesize as usize >= super::record::MAX_JSLOTS {
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
        Ok(Some(link)) => {
            trace_stop(g, rec, link);
            false
        }
        Err(e) => {
            g.jit.err = e;
            g.jit.state = TraceState::Err;
            trace_abort(g);
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
/// JFORL with D = traceno, plus FORI -> JFORI; LOOP -> JLOOP).
fn trace_stop(g: &mut GlobalState, mut rec: Box<Record>, linktype: TraceLink) {
    let js = &mut g.jit;
    let traceno = rec.cur.traceno;
    rec.cur.linktype = linktype;
    rec.cur.link = if linktype == TraceLink::Loop { traceno } else { 0 };
    // Note: all loop ops set J->pc to the following instruction, which is
    // what the final loop snapshot must describe.
    rec.needsnap = false;
    rec.mergesnap = true;
    rec.snap_add();

    let trace = std::mem::replace(
        &mut rec.cur,
        // Placeholder; `rec` is dropped right after.
        super::GCtrace {
            traceno: 0,
            ir: super::ir::IrBuf::new(0, 0),
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
        },
    );
    #[cfg(target_arch = "x86_64")]
    let trace = {
        // `lj_asm_trace`: assemble the IR to machine code. On NYI/OOM the
        // trace stays on the portable IR executor (mcode = None).
        let mut trace = trace;
        trace.mcode = super::asm_x64::assemble(&trace).ok();
        trace
    };

    // Patch the bytecode of the starting instruction in a root trace.
    let pt = trace.startpt;
    let pc = trace.startpc;
    let op = bc_op(trace.startins);
    match op {
        BCOp::FORL | BCOp::LOOP | BCOp::ITERL | BCOp::FUNCF => {
            if op == BCOp::FORL {
                // Patch the FORI, too.
                let fori = (pc as i64 + bc_j(trace.startins)) as usize;
                setbc_op(&mut pt.as_mut().bc[fori], BCOp::JFORI as u32);
            }
            let jop = op.offset(BCOp::JLOOP as i32 - BCOp::LOOP as i32);
            let ins = &mut pt.as_mut().bc[pc];
            setbc_op(ins, jop as u32);
            setbc_d(ins, traceno);
        }
        _ => debug_assert!(false, "bad stop bytecode {:?}", op),
    }

    js.trace[traceno as usize] = Some(Box::new(trace));
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
    {
        if js.exitno == 0 {
            let startpc = js.startpc;
            if e == TraceError::RETRY {
                js.hotcount_set(bc_addr(startpt, startpc + 1), 1); // Immediate retry.
            } else {
                penalty_pc(js, startpt, startpc, e);
            }
        }
        // else: side trace abort — blacklists the exit via a self-link,
        // once side traces exist.
    }

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
    use crate::bc::BCIns;
    use crate::jit::ir::{IROp, IRT_NUM, irt_type};
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
            let s = g.jit.penalty.iter().find(|s| s.pc == bc_addr(pt, pc)).unwrap();
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
            let s = g.jit.penalty.iter().find(|s| s.pc == bc_addr(pt, pc)).unwrap();
            assert!((s.val as u32) >= prev * 2, "penalty must at least double");
            prev = s.val as u32;
            assert!(rounds < 32, "never blacklisted");
        }
        assert!(pt.as_ref().flags & PROTO_ILOOP != 0);
        assert!((9..=11).contains(&rounds), "rounds = {}", rounds);
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
        let tr = g.jit.trace[tno as usize].as_deref().expect("trace registered");
        assert_eq!(tr.traceno, tno);
        assert_eq!(tr.linktype, TraceLink::Loop);
        assert_eq!(tr.link, tr.traceno);
        // The x64 backend must have assembled this numeric loop.
        #[cfg(target_arch = "x86_64")]
        assert!(tr.mcode.is_some(), "trace not assembled to machine code");
        // The loop body must contain two num ADDs (s+i and i+step) and a
        // loop-condition guard.
        let mut adds = 0;
        let mut guards = 0;
        for r in crate::jit::ir::REF_FIRST..tr.ir.nins() {
            let ir = tr.ir.ir(r);
            if ir.op() == IROp::ADD && irt_type(ir.t()) == IRT_NUM {
                adds += 1;
            }
            if ir.is_guard() {
                guards += 1;
            }
        }
        assert_eq!(adds, 2, "expected s+i and i+step");
        assert!(guards >= 1, "expected the loop condition guard");
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
        let tr = g.jit.trace[crate::bc::bc_d(jloop) as usize].as_deref().unwrap();
        assert_eq!(tr.linktype, TraceLink::Loop);
        assert_eq!(bc_op(tr.startins), BCOp::LOOP);
        assert!(tr.snap.iter().any(|s| s.count > 0), "no exit was taken");
    }

    #[test]
    fn unrecordable_loop_still_blacklists() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // The CALL to next() makes the loop body unrecordable (NYIBC),
        // so the penalty machinery must eventually blacklist the FORL.
        let (f, pt) = load_proto(
            &mut lua,
            "local t = {} local s = 0 \
             for i = 1, 2000000 do s = s + (t.x or 1) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(2000000.0));
        assert!(find_op(pt, BCOp::IFORL).is_some(), "FORL not blacklisted");
        assert!(pt.as_ref().flags & PROTO_ILOOP != 0);
        assert_eq!(lua.global().jit.state, TraceState::Idle);
    }

    #[test]
    fn hot_call_penalizes_funcf_header() {
        let mut lua = Lua::new();
        crate::open_libs(lua.main());
        // Function bodies end in RET*: call recording is NYI, so hot
        // functions get penalized and blacklisted at the FUNCF header.
        let (f, pt) = load_proto(
            &mut lua,
            "local function g(x) return x + 1 end \
             local s = 0 for i = 1, 2000000 do s = g(s) end return s",
        );
        let r = crate::vm::call(lua.main(), f, &[]).unwrap();
        assert_eq!(r[0].as_number(), Some(2000000.0));
        let child = pt
            .as_ref()
            .kgc
            .iter()
            .find_map(|k| match k {
                crate::proto::KGc::ProtoRef(p) => Some(*p),
                _ => None,
            })
            .expect("child proto");
        assert_eq!(bc_op(child.as_ref().bc[0]), BCOp::IFUNCF);
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
        assert!(find_op(pt, BCOp::FORL).is_some(), "FORL must stay untouched");
        assert!(pt.as_ref().flags & PROTO_ILOOP == 0);
        assert!(lua.global().jit.trace.iter().flatten().next().is_none());
    }

    #[test]
    fn trace_error_messages() {
        assert_eq!(TraceError::BLACKL.message(), "blacklisted");
        const _: () = assert!(std::mem::size_of::<BCIns>() == 4);
    }
}
