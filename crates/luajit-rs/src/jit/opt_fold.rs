//! FOLD: constant folding, algebraic simplifications and reassociation
//! plus CSE as the fallback. Ported from lj_opt_fold.c.
//!
//! LuaJIT drives rule dispatch through a generated semi-perfect hash of
//! `(ins-op, left-op, right-op)` keys; here the same rules are matched by
//! hand per opcode, ordered from most to least specific exactly like the
//! masked-key search would. The ported subset covers the arithmetic,
//! comparison and bit ops the recorder needs for numeric traces; loads,
//! stores and allocations are emitted raw until lj_opt_mem lands.
//!
//! Rule outcomes mirror the C macros: NEXTFOLD falls through to the next
//! rule, RETRYFOLD restarts with the modified instruction, LEFTFOLD /
//! RIGHTFOLD return an operand, INTFOLD/KNUM results intern a constant,
//! DROPFOLD elides an always-true guard and FAILFOLD aborts the trace
//! with `TraceError::GFAIL` (guard would always fail).

use super::TraceError;
use super::ir::*;

/// Terminal outcomes of one rule-matching pass.
enum Step {
    /// Instruction was modified in place: rerun the rules (RETRYFOLD).
    Retry,
    /// Return this (untagged) ref, no emission (LEFT/RIGHTFOLD etc.).
    Ref(IRRef),
    /// Return this tagged ref (interned constant results).
    TRef(TRef),
    /// Guard is always true: drop it (DROPFOLD).
    Drop,
    /// Pass on to CSE / raw emission.
    Cse,
}

/// `lj_opt_fold`: fold `fins`, returning the resulting tagged ref or
/// `REF_DROP` for an eliminated guard. `Err(GFAIL)` means the guard would
/// always fail, i.e. the trace is pointless (aborts recording).
pub fn opt_fold(buf: &mut IrBuf, fins: IRIns) -> Result<TRef, TraceError> {
    let mut fins = fins;
    loop {
        // Loads, stores and allocations must not reach plain CSE. LuaJIT
        // routes them to FWD/DSE any/any rules; until lj_opt_mem exists
        // they are emitted raw (conservative and sound).
        if irm_kind(IR_MODE[fins.op() as usize]) != IRM_N {
            return Ok(buf.emit_ins(fins));
        }
        match fold_step(buf, &mut fins)? {
            Step::Retry => continue,
            Step::Ref(r) => return Ok(tref(r, buf.ir(r).t())),
            Step::TRef(tr) => return Ok(tr),
            Step::Drop => return Ok(REF_DROP),
            Step::Cse => return Ok(opt_cse(buf, fins)),
        }
    }
}

/// `lj_opt_cse`: search the per-opcode chain for the same operands, bounded
/// below by `max(op1, op2)` (relies on literals < REF_BIAS). Emit if new.
pub fn opt_cse(buf: &mut IrBuf, fins: IRIns) -> TRef {
    let op12 = fins.op12();
    let lim = fins.op1.max(fins.op2) as IRRef;
    let mut r = buf.chain[fins.op() as usize] as IRRef;
    while r > lim {
        let ir = buf.ir(r);
        if ir.op12() == op12 {
            debug_assert!(ir.ot == fins.ot, "CSE across differing types");
            return tref(r, ir.t()); // Common subexpression found.
        }
        r = ir.prev as IRRef;
    }
    buf.emit_ins(fins)
}

/// `CONDFOLD(cond)`: drop the guard if true, otherwise the guard would
/// always fail.
fn condfold(cond: bool) -> Result<Step, TraceError> {
    if cond { Ok(Step::Drop) } else { Err(TraceError::GFAIL) }
}

/// `lj_vm_foldarith` for the FP ops (must match the interpreter exactly).
pub fn fold_numarith(x: f64, y: f64, op: IROp) -> f64 {
    match op {
        IROp::ADD => x + y,
        IROp::SUB => x - y,
        IROp::MUL => x * y,
        IROp::DIV => x / y,
        IROp::MOD => x - (x / y).floor() * y,
        IROp::POW => crate::vm::vm_pow(x, y),
        IROp::NEG => -x,
        IROp::ABS => x.abs(),
        IROp::MIN => if x < y { x } else { y },
        IROp::MAX => if x > y { x } else { y },
        _ => x,
    }
}

/// `lj_ir_numcmp`. The U* variants are the IEEE unordered comparisons.
pub fn fold_numcmp(a: f64, b: f64, op: IROp) -> bool {
    match op {
        IROp::EQ => a == b,
        IROp::NE => a != b,
        IROp::LT => a < b,
        IROp::GE => a >= b,
        IROp::LE => a <= b,
        IROp::GT => a > b,
        IROp::ULT => !(a >= b),
        IROp::UGE => !(a < b),
        IROp::ULE => !(a > b),
        IROp::UGT => !(a <= b),
        _ => unreachable!("bad IR op for numcmp"),
    }
}

/// `kfold_intop` (32 bit wrapping semantics, MOD is Lua floor-mod).
fn kfold_intop(k1: i32, k2: i32, op: IROp) -> i32 {
    match op {
        IROp::ADD => k1.wrapping_add(k2),
        IROp::SUB => k1.wrapping_sub(k2),
        IROp::MUL => k1.wrapping_mul(k2),
        IROp::MOD => {
            // lj_vm_modi: floor modulo; caller ensures k2 != 0.
            let r = k1.wrapping_rem(k2);
            if r != 0 && (r ^ k2) < 0 { r.wrapping_add(k2) } else { r }
        }
        IROp::NEG => (k1 as u32).wrapping_neg() as i32,
        IROp::BAND => k1 & k2,
        IROp::BOR => k1 | k2,
        IROp::BXOR => k1 ^ k2,
        IROp::BSHL => k1.wrapping_shl(k2 as u32 & 31),
        IROp::BSHR => ((k1 as u32).wrapping_shr(k2 as u32 & 31)) as i32,
        IROp::BSAR => k1.wrapping_shr(k2 as u32 & 31),
        IROp::BROL => (k1 as u32).rotate_left(k2 as u32 & 31) as i32,
        IROp::BROR => (k1 as u32).rotate_right(k2 as u32 & 31) as i32,
        IROp::MIN => k1.min(k2),
        IROp::MAX => k1.max(k2),
        _ => unreachable!("bad IR op for kfold_intop"),
    }
}

/// One pass over the fold rules for `fins`, most specific first.
fn fold_step(buf: &mut IrBuf, fins: &mut IRIns) -> Result<Step, TraceError> {
    let nk = buf.nk();
    let op = fins.op();
    // Operand instruction copies (fleft/fright); None for literals/none.
    let fleft = if fins.op1 as IRRef >= nk { Some(*buf.ir(fins.op1 as IRRef)) } else { None };
    let fright = if fins.op2 as IRRef >= nk { Some(*buf.ir(fins.op2 as IRRef)) } else { None };
    let lop = fleft.map(|i| i.op());
    let rop = fright.map(|i| i.op());
    let knumleft = |buf: &IrBuf| buf.knum_val(fins.op1 as IRRef);
    let knumright = |buf: &IrBuf| buf.knum_val(fins.op2 as IRRef);

    macro_rules! phibarrier {
        ($ir:expr) => {
            if irt_isphi($ir.t()) {
                return Ok(Step::Cse); // NEXTFOLD (no later rules in our sets).
            }
        };
    }

    match op {
        // -- Arithmetic ----------------------------------------------------
        IROp::ADD => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                // kfold_numarith
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                // kfold_intarith
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            if lop == Some(IROp::NEG) {
                // simplify_numadd_negx: (-a) + b ==> b - a
                phibarrier!(fleft.unwrap());
                fins.set_op(IROp::SUB);
                fins.op1 = fins.op2;
                fins.op2 = fleft.unwrap().op1;
                return Ok(Step::Retry);
            }
            if rop == Some(IROp::NEG) {
                // simplify_numadd_xneg: a + (-b) ==> a - b
                phibarrier!(fright.unwrap());
                fins.set_op(IROp::SUB);
                fins.op2 = fright.unwrap().op1;
                return Ok(Step::Retry);
            }
            // Note: x + 0 ==> x is INVALID for x = -0 (FP), int only.
            if rop == Some(IROp::KINT) && fright.unwrap().i() == 0 {
                return Ok(Step::Ref(fins.op1 as IRRef)); // simplify_intadd_k
            }
            comm_swap(fins)
        }
        IROp::SUB => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            if lop == Some(IROp::NEG) && rop == Some(IROp::KNUM) {
                // simplify_numsub_negk: (-x) - k ==> (-k) - x
                phibarrier!(fleft.unwrap());
                let nk = -knumright(buf);
                fins.op2 = fleft.unwrap().op1;
                fins.op1 = tref_ref(buf.knum(nk)) as IRRef1;
                return Ok(Step::Retry);
            }
            if rop == Some(IROp::KNUM) && buf.k64_val(fins.op2 as IRRef) == 0 {
                // simplify_numsub_k: x - (+0) ==> x
                return Ok(Step::Ref(fins.op1 as IRRef));
            }
            if rop == Some(IROp::NEG) {
                // simplify_numsub_xneg: a - (-b) ==> a + b
                phibarrier!(fright.unwrap());
                fins.set_op(IROp::ADD);
                fins.op2 = fright.unwrap().op1;
                return Ok(Step::Retry);
            }
            if rop == Some(IROp::KINT) {
                // simplify_intsub_k: i - 0 ==> i; i - k ==> i + (-k)
                let k = fright.unwrap().i();
                if k == 0 {
                    return Ok(Step::Ref(fins.op1 as IRRef));
                }
                fins.set_op(IROp::ADD);
                fins.op2 = tref_ref(buf.kint((k as u32).wrapping_neg() as i32)) as IRRef1;
                return Ok(Step::Retry);
            }
            if lop == Some(IROp::KINT) && fleft.unwrap().i() == 0 {
                // simplify_intsub_kleft: 0 - i ==> -i
                fins.set_op(IROp::NEG);
                fins.op1 = fins.op2;
                return Ok(Step::Retry);
            }
            Ok(Step::Cse)
        }
        IROp::MUL => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            if rop == Some(IROp::KNUM) {
                // simplify_nummuldiv_k (MUL cases; x * -1 needs KSIMD, NYI).
                let n = knumright(buf);
                if n == 1.0 {
                    return Ok(Step::Ref(fins.op1 as IRRef)); // x * 1 ==> x
                } else if n == 2.0 {
                    fins.set_op(IROp::ADD); // x * 2 ==> x + x
                    fins.op2 = fins.op1;
                    return Ok(Step::Retry);
                }
            }
            if rop == Some(IROp::KINT) {
                // simplify_intmul_k32 (k >= 0 only).
                let k = fright.unwrap().i();
                if k == 0 {
                    return Ok(Step::Ref(fins.op2 as IRRef)); // i * 0 ==> 0
                } else if k == 1 {
                    return Ok(Step::Ref(fins.op1 as IRRef)); // i * 1 ==> i
                } else if k > 0 && (k & (k - 1)) == 0 {
                    // i * 2^k ==> i << k
                    fins.set_op(IROp::BSHL);
                    fins.op2 = tref_ref(buf.kint(31 - k.leading_zeros() as i32)) as IRRef1;
                    return Ok(Step::Retry);
                }
            }
            comm_swap(fins)
        }
        IROp::DIV => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if rop == Some(IROp::KNUM) {
                // simplify_nummuldiv_k (DIV cases).
                let n = knumright(buf);
                if n == 1.0 {
                    return Ok(Step::Ref(fins.op1 as IRRef)); // x / 1 ==> x
                }
                // x / 2^k ==> x * 2^-k (exact reciprocal only).
                let u = buf.k64_val(fins.op2 as IRRef);
                let ex = ((u >> 52) & 0x7ff) as u32;
                if (u & 0x000f_ffff_ffff_ffff) == 0 && ex.wrapping_sub(1) < 0x7fd {
                    let recip = (u & (1u64 << 63)) | (((0x7fe - ex) as u64) << 52);
                    fins.set_op(IROp::MUL);
                    fins.op2 = tref_ref(buf.knum_u64(recip)) as IRRef1;
                    return Ok(Step::Retry);
                }
            }
            Ok(Step::Cse)
        }
        IROp::MOD => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) && fright.unwrap().i() != 0 {
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            if rop == Some(IROp::KINT) {
                // simplify_intmod_k: i % 2^k ==> i & (2^k-1)
                let k = fright.unwrap().i();
                if k > 0 && (k & (k - 1)) == 0 {
                    fins.set_op(IROp::BAND);
                    fins.op2 = tref_ref(buf.kint(k - 1)) as IRRef1;
                    return Ok(Step::Retry);
                }
            }
            Ok(Step::Cse)
        }
        IROp::POW => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if rop == Some(IROp::KNUM) {
                // simplify_numpow_k.
                let k = knumright(buf);
                if k == 0.0 {
                    return Ok(Step::TRef(buf.knum_one())); // x ^ 0 ==> 1.0
                } else if k == 1.0 {
                    return Ok(Step::Ref(fins.op1 as IRRef)); // x ^ 1 ==> x
                } else if k == 2.0 {
                    // x ^ 2 ==> x * x
                    let tr = buf.emitir(irtn(IROp::MUL), fins.op1 as IRRef, fins.op1 as IRRef)?;
                    return Ok(Step::TRef(tr));
                }
            }
            Ok(Step::Cse)
        }
        IROp::NEG => {
            if lop == Some(IROp::KNUM) {
                // kfold_numabsneg
                return Ok(Step::TRef(buf.knum(-knumleft(buf))));
            }
            if lop == Some(IROp::KINT) {
                return Ok(Step::TRef(buf.kint(kfold_intop(fleft.unwrap().i(), 0, op))));
            }
            if lop == Some(IROp::NEG) {
                // shortcut_leftleft: -(-x) ==> x
                phibarrier!(fleft.unwrap());
                return Ok(Step::Ref(fleft.unwrap().op1 as IRRef));
            }
            Ok(Step::Cse)
        }
        IROp::ABS => {
            if lop == Some(IROp::KNUM) {
                return Ok(Step::TRef(buf.knum(knumleft(buf).abs())));
            }
            Ok(Step::Cse)
        }
        IROp::MIN | IROp::MAX => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                let y = fold_numarith(knumleft(buf), knumright(buf), op);
                return Ok(Step::TRef(buf.knum(y)));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            if fins.op1 == fins.op2 {
                // comm_dup_minmax: x o x ==> x
                return Ok(Step::Ref(fins.op1 as IRRef));
            }
            Ok(Step::Cse)
        }

        // -- Overflow-checking arithmetic -----------------------------------
        IROp::ADDOV | IROp::SUBOV | IROp::MULOV => {
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                // kfold_intovarith: fold in 64 bit, fail on overflow.
                let base = match op {
                    IROp::ADDOV => IROp::ADD,
                    IROp::SUBOV => IROp::SUB,
                    _ => IROp::MUL,
                };
                let k = match base {
                    IROp::ADD => fleft.unwrap().i() as i64 + fright.unwrap().i() as i64,
                    IROp::SUB => fleft.unwrap().i() as i64 - fright.unwrap().i() as i64,
                    _ => fleft.unwrap().i() as i64 * fright.unwrap().i() as i64,
                };
                if k as i32 as i64 != k {
                    return Err(TraceError::GFAIL);
                }
                return Ok(Step::TRef(buf.kint(k as i32)));
            }
            if rop == Some(IROp::KINT) {
                let k = fright.unwrap().i();
                if op != IROp::MULOV && k == 0 {
                    // simplify_intadd_k: i +- 0 ==> i
                    return Ok(Step::Ref(fins.op1 as IRRef));
                }
                if op == IROp::MULOV {
                    // simplify_intmul_k (overflow-checked cases).
                    if k == 0 {
                        return Ok(Step::Ref(fins.op2 as IRRef)); // i * 0 ==> 0
                    } else if k == 1 {
                        return Ok(Step::Ref(fins.op1 as IRRef)); // i * 1 ==> i
                    } else if k == 2 {
                        fins.set_op(IROp::ADDOV); // i * 2 ==> i + i
                        fins.op2 = fins.op1;
                        return Ok(Step::Retry);
                    }
                }
            }
            if op != IROp::SUBOV { comm_swap(fins) } else { Ok(Step::Cse) }
        }

        // -- Bit ops ---------------------------------------------------------
        IROp::BNOT => {
            if lop == Some(IROp::KINT) {
                return Ok(Step::TRef(buf.kint(!fleft.unwrap().i())));
            }
            if lop == Some(IROp::BNOT) {
                phibarrier!(fleft.unwrap());
                return Ok(Step::Ref(fleft.unwrap().op1 as IRRef)); // ~~x ==> x
            }
            Ok(Step::Cse)
        }
        IROp::BSWAP => {
            if lop == Some(IROp::KINT) {
                return Ok(Step::TRef(buf.kint((fleft.unwrap().i() as u32).swap_bytes() as i32)));
            }
            if lop == Some(IROp::BSWAP) {
                phibarrier!(fleft.unwrap());
                return Ok(Step::Ref(fleft.unwrap().op1 as IRRef));
            }
            Ok(Step::Cse)
        }
        IROp::BAND | IROp::BOR | IROp::BXOR | IROp::BSHL | IROp::BSHR | IROp::BSAR
        | IROp::BROL | IROp::BROR => {
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                let y = kfold_intop(fleft.unwrap().i(), fright.unwrap().i(), op);
                return Ok(Step::TRef(buf.kint(y)));
            }
            match op {
                IROp::BAND | IROp::BOR => {
                    if fins.op1 == fins.op2 {
                        return Ok(Step::Ref(fins.op1 as IRRef)); // comm_dup
                    }
                    comm_swap(fins)
                }
                IROp::BXOR => {
                    if fins.op1 == fins.op2 {
                        return Ok(Step::TRef(buf.kint(0))); // comm_bxor
                    }
                    comm_swap(fins)
                }
                _ => Ok(Step::Cse),
            }
        }

        // -- Comparisons (guards) --------------------------------------------
        IROp::EQ | IROp::NE => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                // kfold_numcomp (not kfold_kref: NaN != NaN).
                return condfold(fold_numcmp(knumleft(buf), knumright(buf), op));
            }
            if (lop == Some(IROp::KINT) && rop == Some(IROp::KINT))
                || (lop == Some(IROp::KGC) && rop == Some(IROp::KGC))
            {
                // kfold_kref: constants are unique, same ref <=> same value.
                return condfold((fins.op1 == fins.op2) == (op == IROp::EQ));
            }
            // comm_equal: for non-numbers x == x ==> drop, x ~= x ==> fail.
            if fins.op1 == fins.op2 && !irt_isnum(fins.t()) {
                return condfold(op == IROp::EQ);
            }
            comm_swap_comp(fins)
        }
        IROp::LT | IROp::GE | IROp::LE | IROp::GT
        | IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT => {
            if lop == Some(IROp::KNUM) && rop == Some(IROp::KNUM) {
                return condfold(fold_numcmp(knumleft(buf), knumright(buf), op));
            }
            if lop == Some(IROp::KINT) && rop == Some(IROp::KINT) {
                // kfold_intcomp
                let (a, b) = (fleft.unwrap().i(), fright.unwrap().i());
                return condfold(match op {
                    IROp::LT => a < b,
                    IROp::GE => a >= b,
                    IROp::LE => a <= b,
                    IROp::GT => a > b,
                    IROp::ULT => (a as u32) < b as u32,
                    IROp::UGE => a as u32 >= b as u32,
                    IROp::ULE => a as u32 <= b as u32,
                    _ => a as u32 > b as u32,
                });
            }
            if op == IROp::UGE && rop == Some(IROp::KINT) && fright.unwrap().i() == 0 {
                return Ok(Step::Drop); // kfold_intcomp0: x >=u 0 is always true.
            }
            // comm_comp: for non-numbers x <=> x ==> drop, x <> x ==> fail.
            if fins.op1 == fins.op2 && !irt_isnum(fins.t()) {
                return condfold((op as u8 ^ (op as u8 >> 1)) & 1 != 0);
            }
            comm_swap_comp(fins)
        }

        _ => Ok(Step::Cse),
    }
}

/// `comm_swap`: canonicalize commutative ops — the lower ref goes right,
/// which also moves constants to the right and helps CSE.
fn comm_swap(fins: &mut IRIns) -> Result<Step, TraceError> {
    if fins.op1 < fins.op2 {
        fins.op1 = std::mem::replace(&mut fins.op2, fins.op1);
        return Ok(Step::Retry);
    }
    Ok(Step::Cse)
}

/// Comparison operand swap: also mirrors the opcode (GT <-> LT, GE <-> LE;
/// EQ/NE and the U* variants map onto themselves correctly via ^3/^0).
fn comm_swap_comp(fins: &mut IRIns) -> Result<Step, TraceError> {
    if fins.op1 < fins.op2 {
        fins.op1 = std::mem::replace(&mut fins.op2, fins.op1);
        let o = fins.op();
        if o != IROp::EQ && o != IROp::NE {
            fins.set_op(IROp::from_u8(o as u8 ^ 3));
        }
        return Ok(Step::Retry);
    }
    Ok(Step::Cse)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf() -> IrBuf {
        IrBuf::new(0, 0)
    }

    /// Emit a raw SLOAD-style opaque value to fold against.
    fn opaque_num(b: &mut IrBuf, slot: u16) -> IRRef {
        tref_ref(b.emit_ins(IRIns::new(irt(IROp::SLOAD, IRT_NUM), slot as IRRef, 0)))
    }

    #[test]
    fn knum_constants_fold_without_emission() {
        let mut b = buf();
        let k2 = tref_ref(b.knum(2.0));
        let k3 = tref_ref(b.knum(3.0));
        let n = b.nins();
        let tr = b.emitir(irtn(IROp::ADD), k2, k3).unwrap();
        assert_eq!(b.nins(), n, "no instruction may be emitted");
        assert!(tref_isnum(tr) && tref_isk(tr));
        assert_eq!(b.knum_val(tref_ref(tr)), 5.0);
        // Same for MUL/DIV/MOD/POW.
        let tr = b.emitir(irtn(IROp::MOD), k3, k2).unwrap();
        assert_eq!(b.knum_val(tref_ref(tr)), 1.0);
        let tr = b.emitir(irtn(IROp::POW), k2, k3).unwrap();
        assert_eq!(b.knum_val(tref_ref(tr)), 8.0);
    }

    #[test]
    fn constants_are_interned() {
        let mut b = buf();
        assert_eq!(b.knum(1.5), b.knum(1.5));
        assert_eq!(b.kint(42), b.kint(42));
        assert_ne!(b.kint(42), b.kint(43));
        // +0.0 and -0.0 must stay distinct (bit-pattern keyed).
        assert_ne!(b.knum(0.0), b.knum(-0.0));
        let nk = b.nk();
        b.knum(1.5);
        b.kint(42);
        assert_eq!(b.nk(), nk, "re-interning must not grow the buffer");
    }

    #[test]
    fn cse_finds_identical_instruction() {
        let mut b = buf();
        let x = opaque_num(&mut b, 1);
        let y = opaque_num(&mut b, 2);
        let t1 = b.emitir(irtn(IROp::ADD), x, y).unwrap();
        let n = b.nins();
        let t2 = b.emitir(irtn(IROp::ADD), x, y).unwrap();
        assert_eq!(t1, t2, "CSE must find the first ADD");
        assert_eq!(b.nins(), n);
    }

    #[test]
    fn commutative_canonicalization_helps_cse() {
        let mut b = buf();
        let x = opaque_num(&mut b, 1);
        let k = tref_ref(b.knum(7.0));
        // k + x and x + k must canonicalize to the same instruction
        // (constant moves right: lower ref to the right).
        let t1 = b.emitir(irtn(IROp::ADD), k, x).unwrap();
        let t2 = b.emitir(irtn(IROp::ADD), x, k).unwrap();
        assert_eq!(t1, t2);
        let ir = b.ir(tref_ref(t1));
        assert_eq!(ir.op1 as IRRef, x);
        assert_eq!(ir.op2 as IRRef, k);
    }

    #[test]
    fn algebraic_simplifications() {
        let mut b = buf();
        let x = opaque_num(&mut b, 1);
        // x - 0 ==> x
        let k0 = tref_ref(b.knum(0.0));
        assert_eq!(tref_ref(b.emitir(irtn(IROp::SUB), x, k0).unwrap()), x);
        // x * 1 ==> x
        let k1 = tref_ref(b.knum(1.0));
        assert_eq!(tref_ref(b.emitir(irtn(IROp::MUL), x, k1).unwrap()), x);
        // x * 2 ==> x + x
        let k2 = tref_ref(b.knum(2.0));
        let tr = b.emitir(irtn(IROp::MUL), x, k2).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::ADD);
        assert_eq!((ir.op1 as IRRef, ir.op2 as IRRef), (x, x));
        // x / 4 ==> x * 0.25 (exact reciprocal)
        let k4 = tref_ref(b.knum(4.0));
        let tr = b.emitir(irtn(IROp::DIV), x, k4).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::MUL);
        assert_eq!(b.knum_val(ir.op2 as IRRef), 0.25);
        // x ^ 2 ==> x * x
        let tr = b.emitir(irtn(IROp::POW), x, k2).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::MUL);
        assert_eq!((ir.op1 as IRRef, ir.op2 as IRRef), (x, x));
        // -(-x) ==> x
        let neg = b.emitir(irtn(IROp::NEG), x, k0).unwrap();
        let tr = b.emitir(irtn(IROp::NEG), tref_ref(neg), k0).unwrap();
        assert_eq!(tref_ref(tr), x);
    }

    #[test]
    fn int_folds_and_overflow_guard() {
        let mut b = buf();
        let k5 = tref_ref(b.kint(5));
        let k7 = tref_ref(b.kint(7));
        let tr = b.emitir(irti(IROp::ADD), k5, k7).unwrap();
        assert_eq!(b.ir(tref_ref(tr)).i(), 12);
        // Floor-mod semantics: -5 % 7 == 2.
        let km5 = tref_ref(b.kint(-5));
        let tr = b.emitir(irti(IROp::MOD), km5, k7).unwrap();
        assert_eq!(b.ir(tref_ref(tr)).i(), 2);
        // i - k ==> i + (-k)
        let x = tref_ref(b.emit_ins(IRIns::new(irt(IROp::SLOAD, IRT_INT), 1, 0)));
        let tr = b.emitir(irti(IROp::SUB), x, k5).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::ADD);
        assert_eq!(b.ir(ir.op2 as IRRef).i(), -5);
        // i % 8 ==> i & 7
        let k8 = tref_ref(b.kint(8));
        let tr = b.emitir(irti(IROp::MOD), x, k8).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::BAND);
        assert_eq!(b.ir(ir.op2 as IRRef).i(), 7);
        // ADDOV overflow ==> guard would always fail.
        let kmax = tref_ref(b.kint(i32::MAX));
        let k1 = tref_ref(b.kint(1));
        assert_eq!(
            b.emitir(irtgi(IROp::ADDOV), kmax, k1).unwrap_err(),
            TraceError::GFAIL
        );
    }

    #[test]
    fn guard_comparisons_drop_or_fail() {
        let mut b = buf();
        let k2 = tref_ref(b.knum(2.0));
        let k3 = tref_ref(b.knum(3.0));
        // 2 < 3: guard always true ==> dropped.
        assert_eq!(b.emitir(irtg(IROp::LT, IRT_NUM), k2, k3).unwrap(), REF_DROP);
        // 3 < 2: guard always false ==> trace error.
        assert_eq!(
            b.emitir(irtg(IROp::LT, IRT_NUM), k3, k2).unwrap_err(),
            TraceError::GFAIL
        );
        // x >=u 0 is always true.
        let x = tref_ref(b.emit_ins(IRIns::new(irt(IROp::SLOAD, IRT_INT), 1, 0)));
        let k0 = tref_ref(b.kint(0));
        assert_eq!(b.emitir(irtgi(IROp::UGE), x, k0).unwrap(), REF_DROP);
        // Table x == x drops; NUM x == x must NOT fold (NaN).
        let t = tref_ref(b.emit_ins(IRIns::new(irt(IROp::SLOAD, IRT_TAB), 2, 0)));
        assert_eq!(b.emitir(irtg(IROp::EQ, IRT_TAB), t, t).unwrap(), REF_DROP);
        let xn = tref_ref(b.emit_ins(IRIns::new(irt(IROp::SLOAD, IRT_NUM), 3, 0)));
        let tr = b.emitir(irtg(IROp::EQ, IRT_NUM), xn, xn).unwrap();
        assert_ne!(tr, REF_DROP, "NaN: x == x must stay");
    }

    #[test]
    fn comparison_swap_mirrors_opcode() {
        let mut b = buf();
        let x = opaque_num(&mut b, 1);
        let k = tref_ref(b.knum(7.0));
        // k < x (constant left) ==> x > k after canonicalization.
        let tr = b.emitir(irtg(IROp::LT, IRT_NUM), k, x).unwrap();
        let ir = *b.ir(tref_ref(tr));
        assert_eq!(ir.op(), IROp::GT);
        assert_eq!((ir.op1 as IRRef, ir.op2 as IRRef), (x, k));
    }

    #[test]
    fn fixed_refs_are_seeded() {
        let b = buf();
        assert_eq!(b.ir(REF_NIL).op(), IROp::KPRI);
        assert_eq!(irt_type(b.ir(REF_NIL).t()), IRT_NIL);
        assert_eq!(irt_type(b.ir(REF_FALSE).t()), IRT_FALSE);
        assert_eq!(irt_type(b.ir(REF_TRUE).t()), IRT_TRUE);
        assert_eq!(b.ir(REF_BASE).op(), IROp::BASE);
        assert_eq!(b.nins(), REF_FIRST);
        assert_eq!(b.nk(), REF_TRUE);
    }
}
