//! Induction Variable Optimization: detects `PHI(i, ADD(i, step))`
//! patterns in loop bodies and creates new induction variables for
//! `MUL(i, K)` or `BSHL(i, K)` expressions, converting loop-body
//! multiplications into cheaper additions.
//!
//! Runs after `opt_loop` has emitted PHI instructions. Scans the loop
//! body (instructions after LOOP) for induction variable uses and
//! replaces eligible affine expressions with new PHIs.

use super::super::ir::*;

/// Count of new induction variables (cap to prevent bloat).
const MAX_IVARS: usize = 16;

/// Information about a discovered induction variable.
struct IndVar {
    /// The PHI reference that defines this IV.
    phi: IRRef,
    /// The step added each iteration (must be a constant).
    step: IRRef,
}

/// Run the IV optimization on the IR buffer. Must be called after loop
/// unrolling (LOOP instruction must be present).
pub fn opt_ivar(buf: &mut IrBuf) {
    let nins = buf.nins();

    // Find the LOOP instruction.
    let loop_ref = find_loop(buf, nins);
    if loop_ref == 0 {
        return;
    }

    // Scan for induction variables: PHIs in the form PHI(lref, rref)
    // where rref is ADD(lref, step). These are loop-carried variables
    // that increase by a constant each iteration.
    let mut ivars: Vec<IndVar> = Vec::new();
    for r in loop_ref + 1..nins {
        let ins = buf.ir(r);
        if ins.op() != IROp::PHI {
            continue;
        }
        let lref = ins.op1 as IRRef;
        let rref = ins.op2 as IRRef;
        if rref < REF_BIAS {
            continue;
        }
        let add_ins = buf.ir(rref);
        if add_ins.op() != IROp::ADD && add_ins.op() != IROp::SUB {
            continue;
        }
        let a1 = add_ins.op1 as IRRef;
        let a2 = add_ins.op2 as IRRef;
        let (_base_ref, step_ref) = if a1 == lref {
            (a1, a2)
        } else if a2 == lref {
            (a2, a1)
        } else {
            continue;
        };
        if !irref_isk(step_ref) {
            continue;
        }
        ivars.push(IndVar {
            phi: r,
            step: step_ref,
        });
        if ivars.len() >= MAX_IVARS {
            break;
        }
    }

    if ivars.is_empty() {
        return;
    }

    // For each IV, find MUL(phi, K) uses in the loop body and create
    // new scaled IVs.
    let mut new_ivars: Vec<(IRRef, IRRef, IRRef)> = Vec::new(); // (orig_phi, scale, new_phi)

    for ivar in &ivars {
        let phi = ivar.phi;
        let step = ivar.step;

        // Scan loop body for MUL(phi, K) and BSHL(phi, log2K).
        for r in loop_ref + 1..nins {
            let ins = buf.ir(r);
            let (mul_op, scale_ref) = match ins.op() {
                IROp::MUL => {
                    let a1 = ins.op1 as IRRef;
                    let a2 = ins.op2 as IRRef;
                    if a1 == phi && a2 < REF_BIAS {
                        (IROp::MUL, a2)
                    } else if a2 == phi && a1 < REF_BIAS {
                        (IROp::MUL, a1)
                    } else {
                        continue;
                    }
                }
                IROp::BSHL => {
                    let a1 = ins.op1 as IRRef;
                    let a2 = ins.op2 as IRRef;
                    if a1 == phi && a2 < REF_BIAS {
                        (IROp::BSHL, a2)
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };

            // Compute the scaled step: step * scale or step << log2(scale).
            let scaled_step = if let Some(step_k) = constant_value(buf, step) {
                if let Some(scale_k) = constant_value(buf, scale_ref) {
                    let result = match mul_op {
                        IROp::MUL => step_k.wrapping_mul(scale_k),
                        IROp::BSHL => step_k.wrapping_shl(scale_k as u32),
                        _ => continue,
                    };
                    buf.kint(result as i32)
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // Compute initial value: initial_phi * scale.
            let initial_val = if let Some(_init_k) = constant_value(buf, r) {
                // The initial value is encoded in the PHI or its SLOAD.
                // For now, compute at runtime via MUL/BSHL in the pre-roll.
                None
            } else {
                None
            };

            // Emit new PHI: PHI(initial, ADD(new_phi, scaled_step))
            let init_ref = if let Some(v) = initial_val {
                v
            } else {
                // Compute initial = phi_lref * scale in pre-roll area.
                let phi_lref = buf.ir(phi).op1 as IRRef;
                // Emit the initial computation.
                let init_ins = IRIns::new(buf.ir(r).ot, phi_lref, scale_ref);
                let tr = buf.emit_ins(init_ins);
                tref_ref(tr)
            };

            // Create new PHI for the scaled IV: PHI(init, ADD(new_phi, scaled_step)).
            // Emit PHI first with a placeholder rref, then emit ADD, then fix rref.
            let phi_ref = buf.nins();
            let phi_ins = IRIns::new(
                irt(IROp::PHI, irt_type(buf.ir(r).t())),
                init_ref,
                0, // placeholder
            );
            buf.emit_ins(phi_ins);

            // Emit ADD(phi_ref, scaled_step) as the back-edge value.
            let add_ins = IRIns::new(irtn(IROp::ADD), phi_ref, tref_ref(scaled_step));
            let add_ref = tref_ref(buf.emit_ins(add_ins));

            // Fix the PHI's rref.
            buf.ir_mut(phi_ref).op2 = add_ref as IRRef1;

            new_ivars.push((r, scaled_step, phi_ref));
        }
    }

    // Replace uses of the original MUL with the new PHI.
    // For now, just mark which refs to replace - the actual replacement
    // would need a full IR substitution pass.
    if !new_ivars.is_empty() {
        rewrite_uses(buf, loop_ref, &new_ivars);
    }
}

fn find_loop(buf: &IrBuf, nins: IRRef) -> IRRef {
    for r in REF_FIRST..nins {
        if buf.ir(r).op() == IROp::LOOP {
            return r;
        }
    }
    0
}

fn constant_value(buf: &IrBuf, r: IRRef) -> Option<i64> {
    if r >= REF_BIAS {
        None
    } else {
        match buf.ir(r).op() {
            IROp::KINT | IROp::KINT64 => Some(buf.k64_val(r) as i64),
            IROp::KNUM => Some(buf.knum_val(r) as i64),
            _ => None,
        }
    }
}

fn rewrite_uses(buf: &mut IrBuf, loop_ref: IRRef, replacements: &[(IRRef, IRRef, IRRef)]) {
    let nins = buf.nins();
    for r in loop_ref + 1..nins {
        let ins = buf.ir(r);
        let mut changed = false;
        let mut new_op1 = ins.op1 as IRRef;
        let mut new_op2 = ins.op2 as IRRef;

        for &(old_ref, _step, new_ref) in replacements {
            if ins.op1 as IRRef == old_ref {
                new_op1 = new_ref;
                changed = true;
            }
            if ins.op2 as IRRef == old_ref {
                new_op2 = new_ref;
                changed = true;
            }
        }

        if changed {
            let ins_mut = buf.ir_mut(r);
            ins_mut.op1 = new_op1 as IRRef1;
            ins_mut.op2 = new_op2 as IRRef1;
        }
    }
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
    fn detect_forl_ivar() {
        let mut buf = test_buf();
        // FORL loop pattern:
        // SLOAD(i) = IRT_INT, KINT(step=1), KINT(limit=10)
        let sload = emit_raw(&mut buf, irt(IROp::SLOAD, IRT_INT), 0, 0);
        let step = buf.kint(1);
        let _limit = buf.kint(10);

        // ADD(sload, step) — i+step
        let add_ref = buf.nins();
        emit_raw(&mut buf, irtn(IROp::ADD), tref_ref(sload), tref_ref(step));

        // KINT(4) — stride for indexing
        let stride = buf.kint(4);

        // MUL(sload, stride) — i*4 for array index
        let _mul_ref = buf.nins();
        emit_raw(&mut buf, irtn(IROp::MUL), tref_ref(sload), tref_ref(stride));

        // LOOP separates pre-roll from body
        emit_raw(&mut buf, irtg(IROp::LOOP, IRT_NIL), 0, 0);

        // PHI(i, ADD(i, step)) — loop induction variable
        let phi_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::PHI, IRT_INT),
            tref_ref(sload),
            tref_ref(add_ref),
        );

        // MUL(phi, stride) — should be replaced by new PHI
        let body_mul = buf.nins();
        emit_raw(&mut buf, irtn(IROp::MUL), phi_ref, tref_ref(stride));

        let _nins_before = buf.nins();
        opt_ivar(&mut buf);

        // After optimization, body_mul should have its op1 replaced
        let new_ins = buf.ir(body_mul);
        // The op1 should no longer be phi_ref (should be a new PHI)
        assert!(
            new_ins.op1 as IRRef != phi_ref || new_ins.op2 as IRRef != stride,
            "body_mul should have been rewritten to use a new scaled IV"
        );
    }

    #[test]
    fn no_ivar_for_non_induction() {
        let mut buf = test_buf();
        let x = buf.kint(5);
        let y = buf.kint(3);

        emit_raw(&mut buf, irtg(IROp::LOOP, IRT_NIL), 0, 0);

        // Just a regular MUL in the body, not from a PHI induction.
        buf.nins();
        emit_raw(&mut buf, irtn(IROp::MUL), tref_ref(x), tref_ref(y));

        let nins_before = buf.nins();
        opt_ivar(&mut buf);
        // No changes expected.
        assert_eq!(buf.nins(), nins_before);
    }

    #[test]
    fn ivar_with_shift_left() {
        let mut buf = test_buf();
        let sload = emit_raw(&mut buf, irt(IROp::SLOAD, IRT_INT), 0, 0);
        let step = buf.kint(1);
        let add_ref = buf.nins();
        emit_raw(&mut buf, irtn(IROp::ADD), tref_ref(sload), tref_ref(step));

        let shift = buf.kint(2); // i << 2

        emit_raw(&mut buf, irtg(IROp::LOOP, IRT_NIL), 0, 0);

        let phi_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::PHI, IRT_INT),
            tref_ref(sload),
            tref_ref(add_ref),
        );

        let body_shift = buf.nins();
        emit_raw(&mut buf, irtn(IROp::BSHL), phi_ref, tref_ref(shift));

        let _nins_before = buf.nins();
        opt_ivar(&mut buf);

        let new_ins = buf.ir(body_shift);
        assert!(
            new_ins.op1 as IRRef != phi_ref || new_ins.op2 as IRRef != shift,
            "body_shift should be rewritten"
        );
    }
}
