//! Integer narrowing pass: converts NUM operations to INT where both
//! operands are integer-typed, eliminating unnecessary FP arithmetic in
//! numeric loops.
//!
//! Modeled on `lj_opt_narrow.c`. Runs after DCE, before assembly.
//!
//! Phase 1 (forward): propagate INT-ness from KINT constants and
//! INT-typed SLOADs (FORL induction variables) through arithmetic,
//! PHI, and comparison nodes.
//!
//! Phase 2: narrow eligible instructions from IRT_NUM to IRT_INT.

use super::ir::*;

pub fn opt_narrow(buf: &mut IrBuf) {
    let nins = buf.nins();
    let ni = (nins - REF_BIAS) as usize;
    if ni == 0 {
        return;
    }

    // Phase 1: determine which instructions are narrowable to INT.
    let mut narrowable = vec![false; ni];

    for r in REF_FIRST..nins {
        let ins = buf.ir(r);
        let idx = (r - REF_BIAS) as usize;
        let op = ins.op();

        narrowable[idx] = match op {
            // Source: integer constants and INT-typed stack loads.
            IROp::KINT | IROp::KINT64 => true,
            IROp::SLOAD => irt_isint(ins.t()),

            // Arithmetic: narrow if both operands are narrowable.
            IROp::ADD | IROp::SUB | IROp::MUL => {
                irt_isnum(ins.t()) && narrowable_operand(buf, ins.op1 as IRRef, &narrowable)
                    && narrowable_operand(buf, ins.op2 as IRRef, &narrowable)
            }

            // Unary: narrow if the operand is narrowable.
            IROp::NEG => {
                irt_isnum(ins.t())
                    && narrowable_operand(buf, ins.op1 as IRRef, &narrowable)
            }

            // PHI: narrow if both incoming edges produce INT.
            IROp::PHI => {
                (irt_isnum(ins.t()) || irt_isint(ins.t()))
                    && narrowable_operand(buf, ins.op1 as IRRef, &narrowable)
                    && narrowable_operand(buf, ins.op2 as IRRef, &narrowable)
            }

            // Comparison guards: narrow if both operands are narrowable.
            IROp::LT | IROp::GE | IROp::LE | IROp::GT
            | IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT
            | IROp::EQ | IROp::NE => {
                narrowable_operand(buf, ins.op1 as IRRef, &narrowable)
                    && narrowable_operand(buf, ins.op2 as IRRef, &narrowable)
            }

            _ => false,
        };
    }

    // Phase 2: narrow the type of each eligible instruction.
    for r in REF_FIRST..nins {
        let idx = (r - REF_BIAS) as usize;
        if narrowable[idx] {
            let ins = buf.ir_mut(r);
            let t = ins.t();
            if irt_isnum(t) || irt_isint(t) {
                ins.ot = (ins.ot & 0xFF00) | (IRT_INT as u16 | ((t & !IRT_TYPE) as u16));
            }
        }
    }
}

/// Check whether operand `opr` is narrowable: constants are only
/// narrowable if they are KINT/KINT64; instructions must be marked in
/// the narrowable mask.
fn narrowable_operand(buf: &IrBuf, opr: IRRef, narrowable: &[bool]) -> bool {
    if opr >= REF_BIAS {
        narrowable[(opr - REF_BIAS) as usize]
    } else {
        // Constants: only KINT/KINT64 are narrowable.
        matches!(buf.ir(opr).op(), IROp::KINT | IROp::KINT64)
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
    fn narrow_add_of_two_kints() {
        let mut buf = test_buf();
        let k1 = buf.kint(3);
        let k2 = buf.kint(5);
        // ADD of two KINT values (initially emits as IRT_NUM via emitir)
        let a = buf.knum(0.0); // placeholder to get a ref
        let _ = a;
        let add_ref = buf.nins();
        // Manually emit a NUM-typed ADD with INT operands (simulating recorder output)
        emit_raw(
            &mut buf,
            irtn(IROp::ADD),
            tref_ref(k1),
            tref_ref(k2),
        );

        opt_narrow(&mut buf);

        // ADD should now be IRT_INT
        let ins = buf.ir(add_ref);
        assert_eq!(
            irt_type(ins.t()),
            IRT_INT,
            "ADD of two KINT should narrow to IRT_INT, got type {:?}",
            irt_type(ins.t())
        );
    }

    #[test]
    fn narrow_add_one_kint_one_num_does_not_narrow() {
        let mut buf = test_buf();
        let k1 = buf.kint(3);
        let k2 = buf.knum(5.0);
        let add_ref = buf.nins();
        emit_raw(
            &mut buf,
            irtn(IROp::ADD),
            tref_ref(k1),
            tref_ref(k2),
        );

        opt_narrow(&mut buf);

        let ins = buf.ir(add_ref);
        assert_eq!(
            irt_type(ins.t()),
            IRT_NUM,
            "ADD with one NUM operand should stay IRT_NUM"
        );
    }

    #[test]
    fn narrow_comparison_guard() {
        let mut buf = test_buf();
        let k1 = buf.kint(1);
        let k2 = buf.kint(10);
        let cmp_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::LT, IRT_GUARD | IRT_NUM),
            tref_ref(k1),
            tref_ref(k2),
        );

        opt_narrow(&mut buf);

        let ins = buf.ir(cmp_ref);
        assert!(
            irt_isint(irt_type(ins.t())),
            "LT guard of two KINT should narrow to IRT_INT"
        );
    }

    #[test]
    fn narrow_chain_add_then_cmp() {
        let mut buf = test_buf();
        let k1 = buf.kint(1);
        let k2 = buf.kint(2);
        let k3 = buf.kint(10);

        // ADD(INT, INT)
        let add_ref = buf.nins();
        emit_raw(
            &mut buf,
            irtn(IROp::ADD),
            tref_ref(k1),
            tref_ref(k2),
        );

        // LT(ADD_result, KINT) — comparison should narrow since ADD is narrowable
        let cmp_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::LT, IRT_GUARD | IRT_NUM),
            tref_ref(add_ref),
            tref_ref(k3),
        );

        opt_narrow(&mut buf);

        assert_eq!(irt_type(buf.ir(add_ref).t()), IRT_INT);
        assert!(irt_isint(irt_type(buf.ir(cmp_ref).t())));
    }

    #[test]
    fn phi_narrow() {
        let mut buf = test_buf();
        let k = buf.kint(0);
        // PHI(INT, INT)
        let phi_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::PHI, IRT_NUM),
            tref_ref(k),
            tref_ref(k),
        );

        opt_narrow(&mut buf);

        let ins = buf.ir(phi_ref);
        assert_eq!(
            irt_type(ins.t()),
            IRT_INT,
            "PHI of INT operands should narrow"
        );
    }

    #[test]
    fn no_narrow_across_non_narrowable_ops() {
        let mut buf = test_buf();
        let k1 = buf.kint(1);
        let k2 = buf.knum(2.5);
        // ADD with one NUM = stays NUM
        let add_ref = buf.nins();
        emit_raw(
            &mut buf,
            irtn(IROp::ADD),
            tref_ref(k1),
            tref_ref(k2),
        );

        let k3 = buf.kint(10);
        // LT with ADD result and INT — ADD is NUM, so comparison stays NUM
        let cmp_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::LT, IRT_GUARD | IRT_NUM),
            tref_ref(add_ref),
            tref_ref(k3),
        );

        opt_narrow(&mut buf);

        assert_eq!(irt_type(buf.ir(add_ref).t()), IRT_NUM);
        assert_eq!(irt_type(buf.ir(cmp_ref).t()) & IRT_TYPE, IRT_NUM);
    }

    #[test]
    fn forl_pattern_narrows() {
        let mut buf = test_buf();
        // Simulate FORL pattern: SLOAD(loopvar, IRT_INT), KINT(step=1), KINT(stop=10)
        let sload = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::SLOAD, IRT_INT),
            0, // slot
            0, // mode
        );

        let step = buf.kint(1);
        let stop = buf.kint(10);

        // ADD(SLOAD, step) → loop variable increment
        let add_ref = buf.nins();
        emit_raw(
            &mut buf,
            irtn(IROp::ADD),
            tref_ref(sload),
            tref_ref(step),
        );

        // LT(ADD, stop) → loop guard
        let cmp_ref = buf.nins();
        emit_raw(
            &mut buf,
            irt(IROp::LT, IRT_GUARD | IRT_NUM),
            tref_ref(add_ref),
            tref_ref(stop),
        );

        opt_narrow(&mut buf);

        assert_eq!(irt_type(buf.ir(add_ref).t()), IRT_INT);
        assert!(irt_isint(irt_type(buf.ir(cmp_ref).t())));
    }
}
