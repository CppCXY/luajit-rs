//! LOOP: Loop Optimizations. Ported from lj_opt_loop.c.
//!
//! Loop unrolling via copy-substitution: the recorded instruction stream
//! (the pre-roll, which performs exactly one iteration) is re-emitted to
//! the FOLD/CSE pipeline with substituted operands. The substitution
//! table is filled with the refs returned by re-emitting each
//! instruction. This works on-the-fly because the IR is in strict SSA
//! form: every ref is defined before its use.
//!
//! This generates two sections separated by the LOOP instruction:
//!
//! 1. The pre-roll: a mix of invariant and variant instructions running
//!    one full iteration, honoring the control dependencies for *both*
//!    itself and the loop body (so all invariant guards get hoisted).
//! 2. The loop body: only the variant instructions, running all
//!    remaining iterations.
//!
//! SLOADs re-emitted during substitution forward the recorder's current
//! slot value (the `LJFOLD(SLOAD any any) -> fwd_sload` rule) — that is
//! what turns stack slots into loop-carried SSA values. Loop-carried
//! dependencies are modeled with PHI instructions emitted *below* the
//! loop body; redundant ones are eliminated in several passes.
//!
//! On type instability (TYPEINS) or an always-failing guard (GFAIL) all
//! changes are undone and the caller continues recording: unrolling via
//! recording fixes many cases, e.g. a flipped boolean.

use super::ir::*;
use super::record::{IRSLOAD_FRAME, Record};
use super::{SnapEntry, SnapShot, TraceError, snap_ref, snap_slot};

/// Max. # of PHIs for a loop (LJ_MAX_PHI).
pub const LJ_MAX_PHI: usize = 64;

#[inline]
fn iidx(r: IRRef) -> usize {
    (r - REF_BIAS) as usize
}

/// `lj_opt_loop`: run the loop optimization on the just-closed trace.
/// `Ok(true)`: optimized. `Ok(false)`: failed with a fixable error, all
/// changes undone — the caller continues recording (one more unroll).
/// `Err`: abort the trace.
pub fn opt_loop(rec: &mut Record) -> Result<bool, TraceError> {
    let nins = rec.cur.ir.nins();
    let nsnap = rec.cur.snap.len();
    let nsnapmap = rec.cur.snapmap.len();
    match loop_unroll(rec) {
        Ok(()) => Ok(true),
        Err(e @ (TraceError::TYPEINS | TraceError::GFAIL)) => {
            rec.instunroll -= 1;
            if rec.instunroll < 0 {
                return Err(e); // But do not unroll forever.
            }
            loop_undo(rec, nins, nsnap, nsnapmap);
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

/// `loop_undo`: revert any partial changes made by the loop optimization.
fn loop_undo(rec: &mut Record, nins: IRRef, nsnap: usize, nsnapmap: usize) {
    rec.cur.snap.truncate(nsnap);
    rec.cur.snapmap.truncate(nsnapmap);
    rec.cur.ir.guardemit = 0;
    rec.cur.ir.rollback(nins);
    for r in REF_FIRST..nins {
        // Remove PHI/MARK flags from the surviving instructions.
        let ir = rec.cur.ir.ir_mut(r);
        ir.clear_phi();
        ir.clear_mark();
    }
}

/// `loop_subst_snap`: copy-substitute one pre-roll snapshot, merging in
/// the loop snapshot's slots as fallback substitutions.
fn loop_subst_snap(rec: &mut Record, osnapidx: usize, loopmap: &[SnapEntry], subst: &[IRRef1]) {
    let osnap = rec.cur.snap[osnapidx];
    let guarded = irt_isguard(rec.cur.ir.guardemit);
    rec.cur.ir.guardemit = 0;
    if !guarded {
        // No guard inbetween: overwrite the previous (copied) snapshot.
        let prev = rec.cur.snap.pop().unwrap();
        rec.cur.snapmap.truncate(prev.mapofs as usize);
    }
    let nmapofs = rec.cur.snapmap.len() as u32;
    let ofs = osnap.mapofs as usize;
    let oentries: Vec<SnapEntry> = rec.cur.snapmap[ofs..ofs + osnap.nent as usize].to_vec();
    // Merge the loop-entry slots with the substituted snapshot slots
    // (both are sorted by slot).
    let lslot = |ln: usize| -> u32 {
        if ln < loopmap.len() {
            snap_slot(loopmap[ln])
        } else {
            u32::MAX
        }
    };
    let (mut on, mut ln) = (0usize, 0usize);
    let mut nent = 0u32;
    while on < oentries.len() {
        let osn = oentries[on];
        if lslot(ln) < snap_slot(osn) {
            rec.cur.snapmap.push(loopmap[ln]); // Copy slot from loop map.
            ln += 1;
        } else {
            if lslot(ln) == snap_slot(osn) {
                ln += 1; // Shadowed loop slot.
            }
            let mut sn = osn;
            let r = snap_ref(osn);
            if !irref_isk(r) {
                // snap_setref with the substituted ref.
                sn = (osn & !0xffff) | subst[iidx(r)] as u32;
            }
            rec.cur.snapmap.push(sn);
            on += 1;
        }
        nent += 1;
    }
    while lslot(ln) < osnap.nslots as u32 {
        rec.cur.snapmap.push(loopmap[ln]); // Copy remaining loop slots.
        ln += 1;
        nent += 1;
    }
    rec.cur.snap.push(SnapShot {
        mapofs: nmapofs,
        iref: rec.cur.ir.nins(),
        pc: osnap.pc,
        sidetrace: 0,
        baseslot: osnap.baseslot,
        nslots: osnap.nslots,
        topslot: osnap.topslot,
        nent: nent as u8,
        count: 0,
    });
}

/// `loop_unroll`: unroll the loop with copy-substitution.
fn loop_unroll(rec: &mut Record) -> Result<(), TraceError> {
    // Only non-constant refs in [REF_BIAS, invar) are valid subst keys.
    let invar = rec.cur.ir.nins();
    let mut subst = vec![0 as IRRef1; iidx(invar)];
    subst[0] = REF_BASE as IRRef1;

    // LOOP separates the pre-roll from the loop body. Its guard bit also
    // forces the first copied snapshot to be a fresh one.
    rec.cur
        .ir
        .emit_ins(IRIns::new(irtg(IROp::LOOP, IRT_NIL), 0, 0));

    // The loop snapshot (the last one) provides fallback substitutions.
    let onsnap = rec.cur.snap.len();
    debug_assert!(onsnap >= 2, "missing loop snapshot");
    let loopsnap = rec.cur.snap[onsnap - 1];
    debug_assert_eq!(
        loopsnap.pc, rec.cur.snap[0].pc,
        "mismatched PC for loop snapshot"
    );
    let lofs = loopsnap.mapofs as usize;
    let loopmap: Vec<SnapEntry> = rec.cur.snapmap[lofs..lofs + loopsnap.nent as usize].to_vec();

    let mut phi: Vec<IRRef> = Vec::new();

    // Start substitution with snapshot #1 (#0 is empty for root traces).
    let mut osnapidx = 1usize;

    // Copy and substitute all recorded instructions and snapshots.
    let mut ins = REF_FIRST;
    while ins < invar {
        if osnapidx < onsnap && ins >= rec.cur.snap[osnapidx].iref {
            // Instruction belongs to the next snapshot: copy-substitute it.
            loop_subst_snap(rec, osnapidx, &loopmap, &subst);
            osnapidx += 1;
        }
        // Substitute instruction operands.
        let ir = *rec.cur.ir.ir(ins);
        let mut op1 = ir.op1 as IRRef;
        if !irref_isk(op1) {
            op1 = subst[iidx(op1)] as IRRef;
        }
        let mut op2 = ir.op2 as IRRef;
        if !irref_isk(op2) {
            op2 = subst[iidx(op2)] as IRRef;
        }
        if irm_kind(IR_MODE[ir.op() as usize]) == IRM_N
            && op1 == ir.op1 as IRRef
            && op2 == ir.op2 as IRRef
        {
            subst[iidx(ins)] = ins as IRRef1; // Regular invariant ins.
            ins += 1;
            continue;
        }
        // Re-emit the substituted instruction to the FOLD/CSE pipeline.
        // SLOADs instead forward the recorder's current slot value (the
        // LJFOLD SLOAD -> fwd_sload rule): this is what turns stack slots
        // into loop-carried SSA values.
        let t = ir.t();
        let r = if ir.op() == IROp::SLOAD {
            debug_assert!(ir.op2 as u32 & IRSLOAD_FRAME == 0);
            let tr = rec.slot[ir.op1 as usize];
            if tr != 0 { tref_ref(tr) } else { ins }
        } else {
            tref_ref(rec.cur.ir.emitir(ir.ot & !(IRT_ISPHI as u16), op1, op2)?)
        };
        subst[iidx(ins)] = r as IRRef1;
        if r != ins && r != REF_DROP && r < invar {
            // Loop-carried dependency: potential PHI.
            let irr = *rec.cur.ir.ir(r);
            if !irref_isk(r) && !irt_isphi(irr.t()) && !irt_ispri(irr.t()) {
                if phi.len() >= LJ_MAX_PHI {
                    return Err(TraceError::PHIOV);
                }
                rec.cur.ir.ir_mut(r).set_phi();
                phi.push(r);
            }
            // Check all loop-carried dependencies for type instability.
            // (No integer IR types in this VM yet, so no CONV fixups.)
            if !irt_sametype(t, irr.t()) {
                return Err(TraceError::TYPEINS);
            }
        }
        ins += 1;
    }

    // Drop a redundant last snapshot (no guard emitted after it).
    if !irt_isguard(rec.cur.ir.guardemit) {
        let prev = rec.cur.snap.pop().unwrap();
        rec.cur.snapmap.truncate(prev.mapofs as usize);
    }

    loop_emit_phi(rec, &subst, phi, onsnap)
}

/// `loop_emit_phi`: emit or eliminate the collected PHIs.
fn loop_emit_phi(
    rec: &mut Record,
    subst: &[IRRef1],
    mut phi: Vec<IRRef>,
    onsnap: usize,
) -> Result<(), TraceError> {
    let invar = rec.cur.ir.chain[IROp::LOOP as usize] as IRRef;
    let nins = rec.cur.ir.nins();
    let mut passx = false;

    // Pass #1: mark redundant and potentially redundant PHIs.
    let mut j = 0usize;
    for i in 0..phi.len() {
        let lref = phi[i];
        let rref = subst[iidx(lref)] as IRRef;
        if lref == rref || rref == REF_DROP {
            // Invariants are redundant.
            rec.cur.ir.ir_mut(lref).clear_phi();
        } else {
            phi[j] = lref;
            j += 1;
            let irr = *rec.cur.ir.ir(rref);
            if !(irr.op1 as IRRef == lref || irr.op2 as IRRef == lref) {
                // Quick check for simple recurrences failed: need pass #2.
                rec.cur.ir.ir_mut(lref).set_mark();
                passx = true;
            }
        }
    }
    phi.truncate(j);

    // Pass #2: traverse the variant part; clear marks of non-redundant
    // PHIs (i.e. those with uses besides the PHI itself).
    if passx {
        let mut i = nins - 1;
        while i > invar {
            // (No CALL/CARG argument chains in this IR subset yet.)
            let ir = *rec.cur.ir.ir(i);
            let op2 = ir.op2 as IRRef;
            if !irref_isk(op2) {
                rec.cur.ir.ir_mut(op2).clear_mark();
            }
            let op1 = ir.op1 as IRRef;
            if !irref_isk(op1) {
                rec.cur.ir.ir_mut(op1).clear_mark();
            }
            i -= 1;
        }
        for s in onsnap..rec.cur.snap.len() {
            let snap = rec.cur.snap[s];
            for n in 0..snap.nent as usize {
                let r = snap_ref(rec.cur.snapmap[snap.mapofs as usize + n]);
                if !irref_isk(r) {
                    rec.cur.ir.ir_mut(r).clear_mark();
                }
            }
        }
    }

    // Pass #3: add PHIs for variant slots without a corresponding SLOAD.
    let nslots = rec.baseslot + rec.maxslot as usize;
    for i in 1..nslots {
        let mut r = tref_ref(rec.slot[i]);
        while !irref_isk(r) && r != subst[iidx(r)] as IRRef {
            {
                let ir = rec.cur.ir.ir_mut(r);
                ir.clear_mark(); // Unmark potential uses, too.
                if irt_isphi(ir.t()) || irt_ispri(ir.t()) {
                    break;
                }
                ir.set_phi();
            }
            if phi.len() >= LJ_MAX_PHI {
                return Err(TraceError::PHIOV);
            }
            phi.push(r);
            r = subst[iidx(r)] as IRRef;
            if r > invar {
                break;
            }
        }
    }

    // Pass #4: propagate non-redundant PHIs.
    let mut retry = passx;
    while retry {
        retry = false;
        for &lref in &phi {
            if !rec.cur.ir.ir(lref).is_marked() {
                // Propagate only from unmarked PHIs.
                let rref = subst[iidx(lref)] as IRRef;
                if rref != REF_DROP && rec.cur.ir.ir(rref).is_marked() {
                    // Right ref points to another PHI: non-redundant.
                    rec.cur.ir.ir_mut(rref).clear_mark();
                    retry = true;
                }
            }
        }
    }

    // Pass #5: emit PHI instructions or eliminate PHIs.
    for &lref in &phi {
        if !rec.cur.ir.ir(lref).is_marked() {
            let rref = subst[iidx(lref)] as IRRef;
            debug_assert!(rref != REF_DROP);
            if rref > invar {
                rec.cur.ir.ir_mut(rref).set_phi();
            }
            let t = irt_type(rec.cur.ir.ir(lref).t());
            rec.cur
                .ir
                .emit_ins(IRIns::new(irt(IROp::PHI, t), lref, rref));
        } else {
            let ir = rec.cur.ir.ir_mut(lref);
            ir.clear_mark();
            ir.clear_phi();
        }
    }
    Ok(())
}
