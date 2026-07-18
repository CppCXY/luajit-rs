//! DCE: Dead Code Elimination. Ported from lj_opt_dce.c.
//!
//! Pre-LOOP optimization: scans all snapshots, marks the referenced
//! instructions, then propagates liveness backwards. Unused instructions
//! without side effects (per `IRIns::sideeff`: stores and non-weak
//! guards are kept) are replaced with NOPs and unlinked from their CSE
//! chains, so the FOLD/CSE engine never finds them during the subsequent
//! copy-substitution.

use super::GCtrace;
use super::ir::*;
use super::snap_ref;

/// `dce_marksnap`: mark all instructions referenced by snapshots.
fn dce_marksnap(t: &mut GCtrace) {
    for i in 0..t.snapmap.len() {
        let r = snap_ref(t.snapmap[i]);
        if r >= REF_FIRST {
            t.ir.ir_mut(r).set_mark();
        }
    }
}

/// `dce_propagate`: backwards propagate marks, replace dead instructions
/// with NOPs and reroute the per-opcode CSE chains around them.
fn dce_propagate(t: &mut GCtrace) {
    // pchain[op]: the instruction whose `prev` field links to the next
    // lower chain member of `op`; 0 means the chain root in `ir.chain`.
    let mut pchain = [0 as IRRef1; IR_MAX];
    let mut ins = t.ir.nins() - 1;
    while ins >= REF_FIRST {
        let ir = *t.ir.ir(ins);
        let op = ir.op() as usize;
        if ir.is_marked() {
            t.ir.ir_mut(ins).clear_mark();
            pchain[op] = ins as IRRef1;
        } else if !ir.sideeff() {
            // Reroute the original instruction chain and NOP it out.
            if pchain[op] == 0 {
                t.ir.chain[op] = ir.prev;
            } else {
                t.ir.ir_mut(pchain[op] as IRRef).prev = ir.prev;
            }
            t.ir.ir_mut(ins).set_nop();
            ins -= 1;
            continue;
        }
        // Instruction is live: its operands are live, too. Literal
        // operands are always below REF_FIRST.
        if ir.op1 as IRRef >= REF_FIRST {
            t.ir.ir_mut(ir.op1 as IRRef).set_mark();
        }
        if ir.op2 as IRRef >= REF_FIRST {
            t.ir.ir_mut(ir.op2 as IRRef).set_mark();
        }
        ins -= 1;
    }
}

/// `lj_opt_dce`: dead code elimination over the completed trace IR.
pub fn opt_dce(t: &mut GCtrace) {
    dce_marksnap(t);
    dce_propagate(t);
}
