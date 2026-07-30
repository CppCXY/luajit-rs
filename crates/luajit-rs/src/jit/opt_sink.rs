//! Allocation sinking — scalar replacement of non-escaping allocations.
use crate::jit::ir::{REF_FIRST, REF_NIL, IROp, tref_isk};
use crate::jit::GCtrace;

pub fn opt_sink(t: &mut GCtrace) {
    let nins = t.ir.nins() as u32;

    for r in REF_FIRST..nins { t.ir.ir_mut(r).clear_mark(); }
    for &sn in &t.snapmap {
        let r = (sn & 0xffff) as u32;
        if r >= REF_FIRST && r < nins { t.ir.ir_mut(r).set_mark(); }
    }

    let mut allocs: Vec<u32> = Vec::new();
    for r in REF_FIRST..nins {
        let op = t.ir.ir(r).op();
        if (op == IROp::TDUP || op == IROp::CNEW) && !t.ir.ir(r).is_marked() {
            allocs.push(r);
        }
    }
    for r in REF_FIRST..nins {
        let ins = t.ir.ir(r);
        if ins.op() != IROp::TDUP && ins.op() != IROp::CNEW { continue; }
        if !ins.is_marked() { continue; }
        let mut escaped = false;
        for r2 in REF_FIRST..nins {
            let ins2 = t.ir.ir(r2);
            if (ins2.op1 == r as u16 || ins2.op2 == r as u16)
                && ins2.op() != IROp::ALOAD && ins2.op() != IROp::HLOAD
                && ins2.op() != IROp::ALEN && ins2.op() != IROp::GCSTEP
                && ins2.op() != IROp::TDUP && ins2.op() != IROp::CNEW
            { escaped = true; break; }
        }
        if !escaped { allocs.push(r); }
    }
    if allocs.is_empty() { return; }

    for &al_ref in &allocs {
        let al_ins = t.ir.ir(al_ref);
        let op = al_ins.op();

        if op == IROp::CNEW {
            // Closure: no loads, just NOP + snapmap patch.
            for sn in &mut t.snapmap {
                if (*sn & 0xffff) == (al_ref as u32) {
                    *sn = (*sn & 0xff00_0000) | (REF_NIL & 0xffff);
                }
            }
            if al_ref + 1 < nins && t.ir.ir(al_ref + 1).op() == IROp::GCSTEP {
                t.ir.ir_mut(al_ref + 1).set_nop();
            }
            t.ir.ir_mut(al_ref).set_nop();
            continue;
        }

        // TDUP: get template, forward loads, NOP.
        let kgc_ref = al_ins.op1 as u32;
        if !tref_isk(kgc_ref) { continue; }
        let kgc_bits = t.ir.k64_val(kgc_ref);
        let Some(templ) = crate::gc::GcPtr::<crate::table::LuaTable>::from_addr(
            kgc_bits & crate::value::LJ_GCVMASK) else { continue; };

        let mut key_to_val: Vec<(u16, u16)> = Vec::new();
        for i in 1i32..=64i32 {
            let v = templ.as_ref().get_int(i);
            if v.is_nil() { break; }
            if v.is_number() {
                key_to_val.push((t.ir.knum(i as f64) as u16, t.ir.knum(v.num()) as u16));
            }
        }
        if key_to_val.is_empty() { continue; }

        let mut forwards: Vec<(u32, u16)> = Vec::new();
        for r in REF_FIRST..nins {
            let ins = t.ir.ir(r);
            if (ins.op() == IROp::ALOAD || ins.op() == IROp::HLOAD) && ins.op1 == al_ref as u16 {
                for &(kref, vref) in &key_to_val {
                    if kref == ins.op2 { forwards.push((r, vref)); break; }
                }
            }
        }

        for &(load_ref, val_ref) in &forwards {
            for r in REF_FIRST..nins {
                let ins = t.ir.ir_mut(r);
                if ins.op1 == load_ref as u16 { ins.op1 = val_ref; }
                if ins.op2 == load_ref as u16 { ins.op2 = val_ref; }
            }
        }
        for sn in &mut t.snapmap {
            if (*sn & 0xffff) == (al_ref as u32) {
                *sn = (*sn & 0xff00_0000) | (REF_NIL & 0xffff);
            }
        }
        for &(lr, _) in &forwards { t.ir.ir_mut(lr).set_nop(); }
        if al_ref + 1 < nins && t.ir.ir(al_ref + 1).op() == IROp::GCSTEP {
            t.ir.ir_mut(al_ref + 1).set_nop();
        }
        t.ir.ir_mut(al_ref).set_nop();
    }
}
