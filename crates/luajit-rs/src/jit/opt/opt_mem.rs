//! Memory optimization: FWD (store-to-load forward substitution) and
//! DSE (dead store elimination) for array loads/stores and field
//! loads/stores within a straight-line trace region.
//!
//! Modeled on `lj_opt_mem.c`'s FWD/DSE rules. Runs inline during FOLD
//! as the recorder emits each instruction.

use super::super::ir::*;

/// Outgoing refs of a successfully scanned backward region: the store
/// instruction ref and the forwarded value ref.
struct FwdStore {
    store_ins: IRRef,
    val: IRRef,
}

/// Try to forward a load (ALOAD/FLOAD) from a preceding store
/// (ASTORE/FSTORE). Returns `Some(val_ref)` when forwarding succeeds.
pub fn try_fwd(buf: &IrBuf, fins: &IRIns) -> Option<IRRef> {
    match fins.op() {
        IROp::ALOAD => fwd_aload(buf, fins),
        IROp::FLOAD => fwd_fload(buf, fins),
        _ => None,
    }
}

/// Try to eliminate a dead store (store overwritten by a later store to
/// the same location with no intervening load).
pub fn try_dse(buf: &mut IrBuf, fins: &IRIns) -> bool {
    match fins.op() {
        IROp::ASTORE => dse_astore(buf, fins),
        _ => false,
    }
}

// ── FWD helpers ─────────────────────────────────────────────────────────

fn fwd_aload(buf: &IrBuf, fins: &IRIns) -> Option<IRRef> {
    let tab = fins.op1 as IRRef;
    let key = fins.op2 as IRRef;
    // Find nearest matching store: tab == store.tab && CARG.key == key.
    find_matching_store(buf, tab, key).map(|fwd| fwd.val)
}

fn fwd_fload(buf: &IrBuf, fins: &IRIns) -> Option<IRRef> {
    let obj = fins.op1 as IRRef;
    let fid = fins.op2; // field ID (literal or constant)
    find_matching_fstore(buf, obj, fid).map(|fwd| fwd.val)
}

// ── DSE helpers ─────────────────────────────────────────────────────────

fn dse_astore(buf: &mut IrBuf, fins: &IRIns) -> bool {
    let carg = buf.ir(fins.op2 as IRRef);
    debug_assert_eq!(carg.op(), IROp::CARG);
    let tab = fins.op1 as IRRef;
    let key = carg.op1 as IRRef;

    if let Some(fwd) = find_matching_store(buf, tab, key) {
        // Check that no load from (tab, key) occurred between
        // fwd.store_ins and the current point.
        let mut r = fwd.store_ins + 1;
        while r < buf.nins() {
            let ins = buf.ir(r);
            match ins.op() {
                IROp::ALOAD if ins.op1 as IRRef == tab && ins.op2 as IRRef == key => {
                    return false; // Load intervened: store is live.
                }
                _ => {}
            }
            r += 1;
        }
        // No intervening load: the previous store is dead.
        buf.ir_mut(fwd.store_ins).set_nop();
        return true;
    }
    false
}

// ── Core store-scanning infrastructure ──────────────────────────────────

/// Scan backward from `nins-1` to `REF_FIRST` looking for an ASTORE
/// whose table == `tab` and whose key (via CARG.op1) == `key`.
fn find_matching_store(buf: &IrBuf, tab: IRRef, key: IRRef) -> Option<FwdStore> {
    let mut r = buf.nins() - 1;
    while r >= REF_FIRST {
        let ins = buf.ir(r);
        match ins.op() {
            IROp::ASTORE => {
                if ins.op1 as IRRef == tab {
                    let carg = buf.ir(ins.op2 as IRRef);
                    if carg.op1 as IRRef == key {
                        return Some(FwdStore {
                            store_ins: r,
                            val: carg.op2 as IRRef,
                        });
                    }
                    // Different key in same table's array part:
                    // array stores to different indices do NOT alias,
                    // so we can continue scanning past them.
                }
            }
            IROp::HSTORE if ins.op1 as IRRef == tab => {
                // Hash store to the same table: may touch any key,
                // so we cannot forward across it.
                return None;
            }
            IROp::LOOP => {
                // Loop boundary: pre-roll stores should not forward
                // into the loop body (different iteration).
                return None;
            }
            IROp::CALLL => {
                let callee_tab = ins.op1 as IRRef;
                if callee_tab == tab {
                    // The table is passed as an argument to a call;
                    // the call might mutate it.
                    return None;
                }
            }
            _ => {}
        }
        r -= 1;
    }
    None
}

/// Scan backward for a matching FSTORE (obj, field_id).
fn find_matching_fstore(buf: &IrBuf, obj: IRRef, fid: IRRef1) -> Option<FwdStore> {
    let mut r = buf.nins() - 1;
    while r >= REF_FIRST {
        let ins = buf.ir(r);
        match ins.op() {
            IROp::FSTORE => {
                if ins.op1 as IRRef == obj {
                    let carg = buf.ir(ins.op2 as IRRef);
                    if carg.op1 as IRRef == fid as IRRef {
                        return Some(FwdStore {
                            store_ins: r,
                            val: carg.op2 as IRRef,
                        });
                    }
                    // Different field: continue (fields don't alias).
                }
            }
            IROp::LOOP => return None,
            IROp::CALLL if ins.op1 as IRRef == obj => return None,
            _ => {}
        }
        r -= 1;
    }
    None
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_buf() -> IrBuf {
        IrBuf::new(0, 0)
    }

    fn emit_raw(buf: &mut IrBuf, ot: u16, a: IRRef, b: IRRef) -> TRef {
        buf.emit_ins(IRIns::new(ot, a, b))
    }

    #[test]
    fn fwd_aload_simple() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx = buf.knum(1.0);
        let val = buf.knum(42.0);

        // ASTORE(tab, key=idx, val=42)
        let carg = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg),
        );

        // ALOAD(tab, idx) should forward to val=42
        let ins = IRIns::new(
            irt(IROp::ALOAD, IRT_GUARD | IRT_NUM),
            tref_ref(tab),
            tref_ref(idx),
        );
        let fwd = try_fwd(&buf, &ins);
        assert_eq!(fwd, Some(tref_ref(val)));
    }

    #[test]
    fn fwd_aload_different_key_no_block() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx1 = buf.knum(1.0);
        let idx2 = buf.knum(2.0);
        let val1 = buf.knum(10.0);
        let val2 = buf.knum(20.0);

        // ASTORE(tab, idx1, val1)
        let carg1 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx1),
            tref_ref(val1),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg1),
        );
        // ASTORE(tab, idx2, val2) — different key, no aliasing
        let carg2 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx2),
            tref_ref(val2),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg2),
        );

        // ALOAD(tab, idx1) should forward to val1 (second store was to a different key)
        let ins = IRIns::new(
            irt(IROp::ALOAD, IRT_GUARD | IRT_NUM),
            tref_ref(tab),
            tref_ref(idx1),
        );
        let fwd = try_fwd(&buf, &ins);
        assert_eq!(fwd, Some(tref_ref(val1)));
    }

    #[test]
    fn fwd_aload_blocked_by_hstore() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx = buf.knum(1.0);
        let val = buf.knum(42.0);

        // ASTORE(tab, idx, val)
        let carg = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg),
        );
        // HSTORE(tab) blocks forwarding — hash store might touch any key
        emit_raw(
            &mut buf,
            irt(IROp::HSTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(idx),
        );

        // ALOAD(tab, idx) should NOT forward (blocked by HSTORE)
        let ins = IRIns::new(
            irt(IROp::ALOAD, IRT_GUARD | IRT_NUM),
            tref_ref(tab),
            tref_ref(idx),
        );
        let fwd = try_fwd(&buf, &ins);
        assert_eq!(fwd, None);
    }

    #[test]
    fn dse_overwritten_store() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx = buf.knum(1.0);
        let val1 = buf.knum(10.0);
        let val2 = buf.knum(20.0);

        // ASTORE(tab, idx, val1)
        let carg1 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val1),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg1),
        );

        // ASTORE(tab, idx, val2) — overwrites previous store
        let carg2 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val2),
        );
        let store2_ins = IRIns::new(irt(IROp::ASTORE, IRT_NIL), tref_ref(tab), tref_ref(carg2));
        let nins_before = buf.nins();
        let dse_res = try_dse(&mut buf, &store2_ins);
        assert!(dse_res, "DSE should eliminate the overwritten first store");

        // The first store (at REF_FIRST + 0 + 3 = REF_BIAS + 3 after tab,idx,val,knum,knum,knum,carg1)
        // Actually, let's just verify that one instruction in the buffer was NOP'd.
        let nops = (REF_FIRST..buf.nins())
            .filter(|&r| buf.ir(r).op() == IROp::NOP)
            .count();
        assert_eq!(nops, 1);
        // Ensure total instruction count was NOT reduced (NOP, not removal).
        assert_eq!(buf.nins(), nins_before);
    }

    #[test]
    fn dse_no_elimination_when_load_intervenes() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx = buf.knum(1.0);
        let val1 = buf.knum(10.0);
        let val2 = buf.knum(20.0);

        // ASTORE(tab, idx, val1)
        let carg1 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val1),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg1),
        );

        // ALOAD(tab, idx) — reads val1, so the store is LIVE
        emit_raw(
            &mut buf,
            irt(IROp::ALOAD, IRT_GUARD | IRT_NUM),
            tref_ref(tab),
            tref_ref(idx),
        );

        // ASTORE(tab, idx, val2) — should NOT eliminate the first store
        let carg2 = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val2),
        );
        let store2_ins = IRIns::new(irt(IROp::ASTORE, IRT_NIL), tref_ref(tab), tref_ref(carg2));
        let dse_res = try_dse(&mut buf, &store2_ins);
        assert!(!dse_res, "DSE should NOT eliminate when a load intervenes");
    }

    #[test]
    fn fwd_blocked_by_loop_boundary() {
        let mut buf = test_buf();
        let tab = emit_raw(&mut buf, irtn(IROp::ADD), REF_NIL, REF_NIL);
        let idx = buf.knum(1.0);
        let val = buf.knum(42.0);

        // ASTORE(tab, idx, val) in pre-roll
        let carg = emit_raw(
            &mut buf,
            irt(IROp::CARG, IRT_NIL),
            tref_ref(idx),
            tref_ref(val),
        );
        emit_raw(
            &mut buf,
            irt(IROp::ASTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg),
        );

        // LOOP instruction separates pre-roll from loop body
        emit_raw(&mut buf, irt(IROp::LOOP, IRT_NIL), 0, 0);

        // ALOAD(tab, idx) in loop body — should NOT forward across LOOP
        let ins = IRIns::new(
            irt(IROp::ALOAD, IRT_GUARD | IRT_NUM),
            tref_ref(tab),
            tref_ref(idx),
        );
        let fwd = try_fwd(&buf, &ins);
        assert_eq!(fwd, None, "FWD must not forward across LOOP boundary");
    }
}
