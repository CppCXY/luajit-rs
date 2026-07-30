//! Allocation sinking — scalar replacement of non-escaping table allocations.
//!
//! When a trace creates a table via TDUP (or TNEW) and all reads from it
//! are local to the trace, eliminate the allocation and replace loads
//! with the stored values directly.

use crate::jit::GCtrace;
use crate::jit::ir::{IROp, REF_FIRST};

pub fn opt_sink(t: &mut GCtrace) {
    let nins = t.ir.nins();

    // Clear marks, then mark snapshot-referenced refs.
    for r in REF_FIRST..nins {
        t.ir.ir_mut(r).clear_mark();
    }
    // Mark instructions referenced by snapshots (they must survive).
    for &sn in &t.snapmap {
        let r = sn & 0xffff;
        if r >= REF_FIRST && r < nins {
            t.ir.ir_mut(r).set_mark();
        }
    }

    // Collect TDUP/TNEW allocations.
    let mut allocs: Vec<u32> = Vec::new();
    for r in REF_FIRST..nins {
        let ins = t.ir.ir(r);
        if matches!(ins.op(), IROp::TNEW | IROp::TDUP) {
            // Skip if marked by snapshot (IRT_MARK = 0x20).
            if !ins.is_marked() {
                allocs.push(r);
            }
        }
    }
    if allocs.is_empty() {
        return;
    }

    // For each allocation, collect ALOAD/HLOAD and check escape.
    for &tab_ref in &allocs {
        let tab_ref_u16 = tab_ref as u16;

        // Check escape: is the table ref used outside of ALOAD/HLOAD/ALEN?
        let mut escaped = false;
        for r in REF_FIRST..nins {
            let ins = t.ir.ir(r);
            let op = ins.op();
            if (ins.op1 == tab_ref_u16 || ins.op2 == tab_ref_u16)
                && op != IROp::ALOAD
                && op != IROp::HLOAD
                && op != IROp::ALEN
                && op != IROp::ASTORE
                && op != IROp::HSTORE
                && op != IROp::TDUP
                && op != IROp::TNEW
            {
                escaped = true;
                break;
            }
        }
        if escaped {
            continue;
        }

        // For TDUP, the template table KGC ref is in op1.
        // The constant values are in the template table's array.
        // We need to find ALOAD from this table and replace them with the
        // template's constants OR with the loop-carried PHI values.

        // Collect all ALOAD from this table.
        let mut loads: Vec<(u32, u16)> = Vec::new(); // (load_ref, key_ref)
        for r in REF_FIRST..nins {
            let ins = t.ir.ir(r);
            if (ins.op() == IROp::ALOAD || ins.op() == IROp::HLOAD) && ins.op1 == tab_ref_u16 {
                loads.push((r, ins.op2));
            }
        }
        if loads.is_empty() {
            continue;
        }

        // For each load, we need to find what value was stored at that key.
        // TDUP duplicates a template table — the values are embedded in the
        // proto's KGc::Table constant. We can look up the template by reading
        // the TDUP's op1 (which is a KGc ref).
        //
        // Simplification: since TDUP creates a table from a constant template
        // and no stores modify it, each ALOAD at a given key always returns
        // the same constant. We can replace the ALOAD with that constant.

        // Find the template table from the TDUP's KGC constant.
        // The KGC ref in op1 stores a NaN-boxed LuaValue whose lower 47
        // bits are the template table's GC pointer.
        let tdup_ins = t.ir.ir(tab_ref);
        let kgc_ref_raw = tdup_ins.op1;
        // Convert signed u16 ref to IRRef for k64 lookup.
        let kgc_bits = t.ir.k64_val(kgc_ref_raw as u32);
        let template_opt =
            crate::gc::GcPtr::<crate::table::LuaTable>::from_addr(kgc_bits & 0x7FFF_FFFF_FFFF);
        let Some(template) = template_opt else {
            continue;
        };

        // Build key→value mapping from the template table.
        let mut key_to_val: Vec<(u16, u16)> = Vec::new();
        for i in 1i32..=64i32 {
            let v = template.as_ref().get_int(i);
            if v.is_nil() {
                break;
            }
            if v.is_number() {
                let kref = t.ir.knum(i as f64);
                let vref = t.ir.knum(v.num());
                key_to_val.push((kref as u16, vref as u16));
            }
        }
        if key_to_val.is_empty() {
            continue;
        }

        // Rewrite each load: replace the load ref with the stored value.
        for &(load_ref, key_ref) in &loads {
            let mut forwarded_val: u16 = 0;
            for &(kref, vref) in &key_to_val {
                if kref == key_ref {
                    forwarded_val = vref;
                    break;
                }
            }
            if forwarded_val == 0 {
                continue;
            }

            // Patch all instructions that reference this load.
            for r in REF_FIRST..nins {
                let ins = t.ir.ir_mut(r);
                if ins.op1 == load_ref as u16 {
                    ins.op1 = forwarded_val;
                }
                if ins.op2 == load_ref as u16 {
                    ins.op2 = forwarded_val;
                }
            }
            // NOP the load.
            t.ir.ir_mut(load_ref).set_nop();
        }

        // NOP the TDUP and its GCSTEP.
        // GCSTEP is the next instruction; find and NOP it.
        t.ir.ir_mut(tab_ref).set_nop();
        if tab_ref + 1 < nins {
            let next = t.ir.ir(tab_ref + 1);
            if next.op() == IROp::GCSTEP {
                t.ir.ir_mut(tab_ref + 1).set_nop();
            }
        }
    }
}
