//! Trace recorder: translates executed bytecode to IR, one instruction at
//! a time. Ported from lj_record.c, with the snapshot-generation half of
//! lj_snap.c (snapshot_slots/snapshot_stack/lj_snap_add).
//!
//! Phase 2 scope: single-frame numeric traces — FORL and LOOP roots,
//! arithmetic, moves, constants and comparisons. Everything else aborts
//! with NYIBC and feeds the penalty/blacklist engine, exactly like LuaJIT
//! handles unrecordable bytecode. Calls, table access and returns arrive
//! with later phases, as does number->integer narrowing (all numeric
//! slots stay IRT_NUM for now, hence KSHORT interns a KNUM).

use crate::bc::*;
use crate::gc::GcPtr;
use crate::proto::Proto;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::ir::*;
use super::{
    GCtrace, JitParam, JitState, PENALTY_MIN, PENALTY_SLOTS, SNAP_CONT, SNAP_FRAME,
    SNAP_NORESTORE, SnapEntry, SnapShot, TraceError, TraceLink, TraceNo, snap_entry, snap_ref,
    snap_slot,
};

/// Maximum number of stack slots the recorder tracks (LJ_MAX_JSLOTS).
pub const MAX_JSLOTS: usize = 250;

/// SLOAD mode bits (lj_ir.h).
pub const IRSLOAD_PARENT: u32 = 0x01;
pub const IRSLOAD_FRAME: u32 = 0x02;
pub const IRSLOAD_TYPECHECK: u32 = 0x04;
pub const IRSLOAD_CONVERT: u32 = 0x08;
pub const IRSLOAD_READONLY: u32 = 0x10;
pub const IRSLOAD_INHERIT: u32 = 0x20;
pub const IRSLOAD_KEYINDEX: u32 = 0x40;

/// FPMATH sub-function literals (IRFPMDEF, ORDER FPM).
pub const IRFPM_FLOOR: u32 = 0;
pub const IRFPM_CEIL: u32 = 1;
pub const IRFPM_TRUNC: u32 = 2;
pub const IRFPM_SQRT: u32 = 3;

/// Loop event (LoopEvent).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoopEvent {
    Leave,   // Loop is left or not entered.
    EnterLo, // Loop is entered with a low iteration count left.
    Enter,   // Loop is entered.
}

/// Scalar evolution analysis cache of the FORL index (ScEvEntry).
#[derive(Clone, Copy)]
pub struct ScEv {
    /// Bytecode index of the matching FORI, or None if invalid.
    pub pc: Option<usize>,
    pub idx: IRRef,
    pub stop: IRRef,
    pub step: IRRef,
    pub t: u8,
    pub dir: bool,
}

impl Default for ScEv {
    fn default() -> ScEv {
        ScEv { pc: None, idx: REF_NIL, stop: 0, step: 0, t: IRT_NUM, dir: true }
    }
}

/// The three canonical primitive values, indexed by the ~itype operand.
const PRI_VALUES: [LuaValue; 3] = [LuaValue::NIL, LuaValue::FALSE, LuaValue::TRUE];

/// One inlined call frame of the recorder (the parts of LuaJIT's slot
/// bookkeeping that `lj_record_call`/`lj_record_ret` juggle).
struct FrameInfo {
    /// Caller prototype (recording continues there after the return).
    pt: GcPtr<Proto>,
    /// Callee prototype (for the recursion-unroll limit).
    callee: GcPtr<Proto>,
    /// The CALL's base slot A, relative to the caller frame.
    cbase: u32,
    /// Results wanted by the CALL (B-1; fixed, multres calls are NYI).
    want: u32,
    /// Caller's baseslot, restored on return.
    prev_baseslot: usize,
}

/// The recording context: the parts of `jit_State` that only live while
/// the recorder runs, plus the trace under construction (`J->cur`).
pub struct Record {
    pub cur: GCtrace,
    /// Parent trace of a side trace (J->parent, 0 = root).
    pub parent: TraceNo,
    /// Exit number in the parent trace (J->exitno).
    pub exitno: u32,
    /// TRef of each stack slot (J->slot); the frame base is at
    /// `slot[baseslot]`.
    pub slot: [TRef; MAX_JSLOTS],
    /// First base slot (1+LJ_FR2 = 2: the frame's function sits at
    /// `slot[baseslot-2]`, the frame link at `slot[baseslot-1]`).
    pub baseslot: usize,
    /// One above the highest used slot, relative to base (J->maxslot).
    pub maxslot: u32,
    pub framedepth: i32,
    pub retdepth: i32,
    /// Inlined call frames (parallel to framedepth).
    frames: Vec<FrameInfo>,
    /// Suppress a new snapshot if no guard was emitted (J->mergesnap).
    pub mergesnap: bool,
    /// Take a snapshot before recording the next bytecode (J->needsnap).
    pub needsnap: bool,
    pub scev: ScEv,
    /// Loop-formation PC (J->startpc): hitting it again closes the trace
    /// as a loop. None for side traces that cannot form an extra loop.
    pub startpc: Option<usize>,
    /// Root-trace bytecode range (in instruction indexes): leaving it
    /// aborts with LLEAVE. `bc_extent = !0` means no limit.
    pub bc_min: usize,
    pub bc_extent: usize,
    /// IR ref at the last inner-loop boundary (J->loopref).
    pub loopref: IRRef,
    pub loopunroll: i32,
    /// Remaining unroll attempts for unstable loops (J->instunroll).
    pub instunroll: i32,
    /// Max. depth of same-function call inlining (JIT_P_callunroll).
    pub callunroll: i32,
    /// Number of recorded tailcalls (J->tailcalled).
    pub tailcalled: i32,
    /// The proto being recorded (single-frame traces: always startpt).
    pub pt: GcPtr<Proto>,
    /// Current bytecode index, updated as recording progresses (J->pc).
    pub pc: usize,
}

impl Record {
    pub fn new(
        traceno: TraceNo,
        pt: GcPtr<Proto>,
        pc: usize,
        parent: u16,
        exitno: u16,
        loopunroll: i32,
        instunroll: i32,
        callunroll: i32,
    ) -> Box<Record> {
        Box::new(Record {
            cur: GCtrace {
                traceno,
                ir: IrBuf::new(parent, exitno),
                snap: Vec::with_capacity(4),
                snapmap: Vec::with_capacity(32),
                mcode: None,
                startpt: pt,
                startpc: pc,
                startins: pt.as_ref().bc[pc],
                link: 0,
                linktype: TraceLink::None,
                root: 0,
                nchild: 0,
                parentmap: Vec::new(),
                inner_ofs: 0,
                stub_tails: Vec::new(),
            },
            parent: parent as TraceNo,
            exitno: exitno as u32,
            slot: [0; MAX_JSLOTS],
            baseslot: 2,
            maxslot: 0,
            framedepth: 0,
            retdepth: 0,
            frames: Vec::new(),
            mergesnap: false,
            needsnap: false,
            scev: ScEv::default(),
            startpc: Some(pc),
            bc_min: 0,
            bc_extent: !0usize,
            loopref: 0,
            loopunroll,
            instunroll,
            callunroll,
            tailcalled: 0,
            pt,
            pc,
        })
    }

    // -- Slot handling ------------------------------------------------------

    #[inline]
    fn base_ref(&self, s: u32) -> TRef {
        self.slot[self.baseslot + s as usize]
    }
    #[inline]
    fn set_base(&mut self, s: u32, tr: TRef) {
        self.slot[self.baseslot + s as usize] = tr;
    }

    /// Runtime value of slot `s` in the current frame.
    #[inline]
    fn slot_val(&self, l: &LuaState, base: usize, s: u32) -> LuaValue {
        l.stack[base + s as usize]
    }

    /// `itype2irt`: map a runtime value to its IR type.
    fn value_irt(v: LuaValue) -> u8 {
        if v.is_number() { IRT_NUM } else { (!v.itype()) as u8 & IRT_TYPE }
    }

    /// `sloadt`: specialize a slot to a specific type, no typecheck guard.
    fn sloadt(&mut self, s: u32, t: u8, mode: u32) -> TRef {
        let tr = self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::SLOAD, t),
            self.baseslot as IRRef + s,
            mode,
        ));
        self.set_base(s, tr);
        tr
    }

    /// `sload`: specialize a slot to its runtime type, with a typecheck.
    fn sload(&mut self, l: &LuaState, base: usize, s: u32) -> TRef {
        let t = Self::value_irt(self.slot_val(l, base, s));
        let mut tr = self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::SLOAD, IRT_GUARD | t),
            self.baseslot as IRRef + s,
            IRSLOAD_TYPECHECK,
        ));
        if irt_ispri(t) {
            tr = tref_pri(t); // Canonicalize primitive refs.
        }
        self.set_base(s, tr);
        tr
    }

    /// `getslot`: load and specialize a slot if not done already.
    fn getslot(&mut self, l: &LuaState, base: usize, s: u32) -> TRef {
        let tr = self.base_ref(s);
        if tr != 0 { tr } else { self.sload(l, base, s) }
    }

    // -- Snapshots (lj_snap.c) ------------------------------------------------

    /// `snapshot_slots`: add all modified slots to the snap map.
    fn snapshot_slots(&self, map: &mut Vec<SnapEntry>, nslots: u32) -> u32 {
        let mut n = 0;
        for s in 0..nslots {
            let tr = self.slot[s as usize];
            let r = tref_ref(tr);
            if s == 1 {
                // Ignore the frame-link slot (FR2), except below a frame.
                if tr & TREF_FRAME != 0 {
                    map.push((1 << 24) + SNAP_FRAME + SNAP_NORESTORE + REF_NIL);
                    n += 1;
                }
                continue;
            }
            if r != 0 {
                let mut sn = snap_entry(s, tr);
                let ir = self.cur.ir.ir(r);
                if ir.op() == IROp::SLOAD && ir.op1 as u32 == s {
                    // No need to snapshot unmodified non-inherited slots.
                    if ir.op2 as u32 & IRSLOAD_INHERIT == 0 {
                        continue;
                    }
                    // No need to restore readonly slots.
                    if ir.op2 as u32 & IRSLOAD_READONLY != 0 {
                        sn |= SNAP_NORESTORE;
                    }
                }
                map.push(sn);
                n += 1;
            }
        }
        n
    }

    /// `snapshot_stack`: take a snapshot of the current stack.
    fn snapshot_stack(&mut self, nsnapmap: u32) -> SnapShot {
        let nslots = self.baseslot as u32 + self.maxslot;
        self.cur.snapmap.truncate(nsnapmap as usize);
        let mut map = std::mem::take(&mut self.cur.snapmap);
        let nent = self.snapshot_slots(&mut map, nslots);
        self.cur.snapmap = map;
        SnapShot {
            mapofs: nsnapmap,
            iref: self.cur.ir.nins(),
            pc: self.pc as u32,
            sidetrace: 0,
            baseslot: self.baseslot as u8,
            nslots: nslots as u8,
            topslot: self.pt.as_ref().framesize,
            nent: nent as u8,
            count: 0,
        }
    }

    /// `lj_snap_add`: add or merge a snapshot.
    pub fn snap_add(&mut self) {
        let mut nsnap = self.cur.snap.len();
        let mut nsnapmap = self.cur.snapmap.len() as u32;
        // Merge if no ins. in between, or if requested and no guard between.
        if (nsnap > 0 && self.cur.snap[nsnap - 1].iref == self.cur.ir.nins())
            || (self.mergesnap && !irt_isguard(self.cur.ir.guardemit))
        {
            if nsnap == 1 {
                // But preserve snap #0: its PC anchors the trace entry.
                self.cur.ir.emit_ins(IRIns::new(irt(IROp::NOP, IRT_NIL), 0, 0));
            } else {
                nsnap -= 1;
                nsnapmap = self.cur.snap[nsnap].mapofs;
                self.cur.snap.truncate(nsnap);
            }
        }
        self.mergesnap = false;
        self.cur.ir.guardemit = 0;
        let snap = self.snapshot_stack(nsnapmap);
        self.cur.snap.push(snap);
    }

    /// `snap_replay_const`: re-intern a constant from the parent trace.
    fn replay_const(&mut self, t: &GCtrace, r: IRRef) -> Result<TRef, TraceError> {
        let ir = t.ir.ir(r);
        match ir.op() {
            IROp::KPRI => Ok(tref_pri(irt_type(ir.t()))),
            IROp::KINT => Ok(self.cur.ir.kint(ir.i())),
            IROp::KGC => Ok(self.cur.ir.kgc(t.ir.k64_val(r), irt_type(ir.t()))),
            IROp::KNUM => Ok(self.cur.ir.knum_u64(t.ir.k64_val(r))),
            IROp::KINT64 => Ok(self.cur.ir.kint64(t.ir.k64_val(r))),
            _ => Err(TraceError::NYIBC), // Bad constant in a stack slot.
        }
    }

    /// `lj_snap_replay`: replay a parent snapshot to set up a side trace.
    /// Emits inherited SLOADs (or re-interned constants) for every slot
    /// of the parent exit, takes snapshot #0 and — for exits inside
    /// inlined call frames — rebuilds the recorder's frame stack from
    /// the KFUNC/frame-link constants (LuaJIT reads the real stack; our
    /// frame slots are compile-time constants, so decoding them is
    /// equivalent).
    pub fn snap_replay(&mut self, t: &GCtrace, exitno: usize) -> Result<(), TraceError> {
        let snap = t.snap[exitno];
        let ofs = snap.mapofs as usize;
        self.framedepth = 0;
        // Parent ref -> replayed tref, for de-duping aliased slots.
        let mut seen: Vec<(IRRef, TRef)> = Vec::new();
        // Frame-chain decoding state.
        let mut cur_pt = t.startpt;
        let mut prev_baseslot = 2usize;
        let mut prev_entry: Option<(u32, IRRef)> = None;
        for n in 0..snap.nent as usize {
            let sn = t.snapmap[ofs + n];
            let s = snap_slot(sn);
            let r = snap_ref(sn);
            if sn & SNAP_CONT != 0 {
                return Err(TraceError::NYIBC); // Continuation frames NYI.
            }
            let tr = if let Some(&(_, tr)) = seen.iter().find(|&&(pr, _)| pr == r) {
                tr
            } else {
                let tr = if irref_isk(r) {
                    // See the special treatment of the FR2 slot 1 in
                    // snapshot_slots.
                    if sn == (1 << 24) + SNAP_FRAME + SNAP_NORESTORE + REF_NIL {
                        0
                    } else {
                        self.replay_const(t, r)?
                    }
                } else {
                    let ir = t.ir.ir(r);
                    let ty = irt_type(ir.t());
                    let mut mode = IRSLOAD_INHERIT | IRSLOAD_PARENT;
                    if ir.op() == IROp::SLOAD {
                        mode |= ir.op2 as u32 & IRSLOAD_READONLY;
                    }
                    let tr = self.cur.ir.emit_ins(IRIns::new(irt(IROp::SLOAD, ty), s, mode));
                    // The executor pre-fills env[own] from the parent's
                    // env[r] on a linked exit (register-free hand-over).
                    self.cur.parentmap.push((tref_ref(tr) as IRRef1, r as IRRef1));
                    tr
                };
                seen.push((r, tr));
                tr
            };
            self.slot[s as usize] = tr | (sn & (SNAP_CONT | SNAP_FRAME));
            if sn & SNAP_FRAME != 0 && s != 1 {
                // An inlined frame's link slot: rebuild the FrameInfo
                // from the constants (the function sits in the entry at
                // slot s-1, the link is the caller's return address).
                if !irref_isk(r) {
                    return Err(TraceError::NYIBC);
                }
                let link_bits = t.ir.k64_val(r);
                let Some((fs, fref)) = prev_entry else { return Err(TraceError::NYIBC) };
                if fs != s - 1 || !irref_isk(fref) {
                    return Err(TraceError::NYIBC);
                }
                let fv = LuaValue::from_bits(t.ir.k64_val(fref));
                let Some(gf) = fv.as_func() else { return Err(TraceError::NYIBC) };
                let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
                    return Err(TraceError::NYIBC);
                };
                let bcbase = cur_pt.as_ref().bc.as_ptr() as u64;
                let Some(delta) = link_bits.checked_sub(bcbase) else {
                    return Err(TraceError::NYIBC);
                };
                let ret_pc = (delta / 4) as usize;
                if delta % 4 != 0 || ret_pc == 0 || ret_pc >= cur_pt.as_ref().bc.len() {
                    return Err(TraceError::NYIBC);
                }
                let call_ins = cur_pt.as_ref().bc[ret_pc - 1];
                if bc_op(call_ins) != BCOp::CALL || bc_b(call_ins) == 0 {
                    return Err(TraceError::NYIBC);
                }
                self.frames.push(FrameInfo {
                    pt: cur_pt,
                    callee: cl.proto,
                    cbase: bc_a(call_ins),
                    want: bc_b(call_ins) - 1,
                    prev_baseslot,
                });
                self.framedepth += 1;
                cur_pt = cl.proto;
                prev_baseslot = s as usize + 1;
                self.baseslot = s as usize + 1;
            }
            prev_entry = Some((s, r));
        }
        self.pt = cur_pt;
        if 1 + self.baseslot + cur_pt.as_ref().framesize as usize >= MAX_JSLOTS {
            return Err(TraceError::STACKOV);
        }
        self.maxslot = snap.nslots as u32 - self.baseslot as u32;
        self.pc = snap.pc as usize;
        self.snap_add();
        Ok(())
    }

    // -- FOR loops (rec_for*) ---------------------------------------------------

    /// `rec_for_direction`: sign bit of the step's high word.
    fn for_direction(v: LuaValue) -> bool {
        ((v.to_bits() >> 32) as i32) >= 0
    }

    /// `rec_for_iter`: simulate the runtime behavior of the loop iterator.
    fn for_iter(op: &mut IROp, l: &LuaState, base: usize, ra: u32, isforl: bool) -> LoopEvent {
        let stop = l.stack[base + (ra + FORL_STOP) as usize].num();
        let step = l.stack[base + (ra + FORL_STEP) as usize].num();
        let mut idx = l.stack[base + (ra + FORL_IDX) as usize].num();
        if isforl {
            idx += step;
        }
        if Self::for_direction(l.stack[base + (ra + FORL_STEP) as usize]) {
            if idx <= stop {
                *op = IROp::LE;
                return if idx + 2.0 * step > stop { LoopEvent::EnterLo } else { LoopEvent::Enter };
            }
            *op = IROp::GT;
        } else {
            if stop <= idx {
                *op = IROp::GE;
                return if idx + 2.0 * step < stop { LoopEvent::EnterLo } else { LoopEvent::Enter };
            }
            *op = IROp::LT;
        }
        LoopEvent::Leave
    }

    /// `find_kinit`: search bytecode backwards for a constant initializer
    /// of `slot` (KSHORT/KNUM with no forward jump across it).
    fn find_kinit(&mut self, endpc: usize, slot: u32) -> TRef {
        let pt = self.pt.as_ref();
        let mut pc = endpc;
        while pc > 1 {
            pc -= 1;
            let ins = pt.bc[pc];
            let op = bc_op(ins);
            if bcmode_a(op) == BCMode::Base as u32 && bc_a(ins) <= slot {
                return 0; // Multiple results, e.g. from a CALL or KNIL.
            } else if bcmode_a(op) == BCMode::Dst as u32 && bc_a(ins) == slot {
                if op == BCOp::KSHORT || op == BCOp::KNUM {
                    // Verify there's no forward jump across it.
                    let kpc = pc;
                    while pc > 1 {
                        pc -= 1;
                        if bc_op(pt.bc[pc]) == BCOp::JMP {
                            let target = (pc as i64 + bc_j(pt.bc[pc]) + 1) as usize;
                            if target > kpc && target <= endpc {
                                return 0; // Conditional assignment.
                            }
                        }
                    }
                    let n = if op == BCOp::KSHORT {
                        bc_d(ins) as i16 as f64
                    } else {
                        pt.kn[bc_d(ins) as usize]
                    };
                    return self.cur.ir.knum(n);
                }
                return 0; // Non-constant initializer.
            }
        }
        0
    }

    /// `fori_arg`: peek before the FORI for a constant initializer,
    /// otherwise load from the slot (readonly, inherited by side traces).
    fn fori_arg(&mut self, fori: usize, slot: u32) -> TRef {
        let tr = self.base_ref(slot);
        if tr != 0 {
            return tr;
        }
        let k = self.find_kinit(fori, slot);
        if k != 0 {
            return k;
        }
        self.sloadt(slot, IRT_NUM, IRSLOAD_INHERIT | IRSLOAD_READONLY)
    }

    /// `rec_for_check` (numeric mode): guard a non-constant step's sign.
    fn for_check(&mut self, dir: bool, step: TRef) -> Result<(), TraceError> {
        if !tref_isk(step) {
            let zero = self.cur.ir.knum(0.0);
            self.cur.ir.emitir(
                irtg(if dir { IROp::GE } else { IROp::LT }, IRT_NUM),
                tref_ref(step),
                tref_ref(zero),
            )?;
        }
        Ok(())
    }

    /// `rec_for_loop`: record the loop bookkeeping of the FORI at `fori`.
    /// `init` is true at trace setup, false when the FORL is re-recorded.
    pub fn for_loop(
        &mut self,
        l: &LuaState,
        base: usize,
        fori: usize,
        init: bool,
    ) -> Result<(), TraceError> {
        let ins = self.pt.as_ref().bc[fori];
        debug_assert!(matches!(bc_op(ins), BCOp::FORI | BCOp::JFORI));
        let ra = bc_a(ins);
        // Numeric mode only until narrowing lands.
        for i in FORL_IDX..=FORL_STEP {
            if !self.slot_val(l, base, ra + i).is_number() {
                return Err(TraceError::NYIBC);
            }
        }
        let mut idx = self.base_ref(ra + FORL_IDX);
        let stop = self.fori_arg(fori, ra + FORL_STOP);
        let step = self.fori_arg(fori, ra + FORL_STEP);
        let dir = Self::for_direction(self.slot_val(l, base, ra + FORL_STEP));
        self.scev.t = IRT_NUM;
        self.scev.dir = dir;
        self.scev.stop = tref_ref(stop);
        self.scev.step = tref_ref(step);
        self.for_check(dir, step)?;
        if idx == 0 {
            idx = self.sloadt(ra + FORL_IDX, IRT_NUM, IRSLOAD_INHERIT);
        }
        if !init {
            idx = self.cur.ir.emitir(irtn(IROp::ADD), tref_ref(idx), tref_ref(step))?;
            self.set_base(ra + FORL_IDX, idx);
        }
        self.set_base(ra + FORL_EXT, idx);
        self.scev.idx = tref_ref(idx);
        self.scev.pc = Some(fori);
        self.maxslot = ra + FORL_EXT + 1;
        Ok(())
    }

    /// `rec_for`: record a FORL/JFORL (`isforl`) or FORI/JFORI loop op.
    /// `fori` is the index of the (paired) FORI.
    fn rec_for(
        &mut self,
        l: &LuaState,
        base: usize,
        fori: usize,
        isforl: bool,
    ) -> Result<LoopEvent, TraceError> {
        let ins = self.pt.as_ref().bc[fori];
        let ra = bc_a(ins);
        // Avoid semantic mismatches and always-failing guards.
        for i in FORL_IDX..=FORL_STEP {
            let v = self.slot_val(l, base, ra + i);
            if !v.is_number() || v.num().is_nan() {
                return Err(TraceError::GFAIL);
            }
        }
        if self.slot_val(l, base, ra + FORL_STEP).to_bits() == (-0.0f64).to_bits() {
            return Err(TraceError::GFAIL);
        }

        let (stop, t);
        if isforl {
            // FORL/JFORL: move the loop variable forward.
            let idx = self.base_ref(ra + FORL_IDX);
            if self.scev.pc == Some(fori) && tref_ref(idx) == self.scev.idx {
                t = self.scev.t;
                stop = self.scev.stop;
                let step = self.scev.step;
                let nidx = self.cur.ir.emitir(irt(IROp::ADD, t), tref_ref(idx), step)?;
                self.set_base(ra + FORL_IDX, nidx);
                self.set_base(ra + FORL_EXT, nidx);
            } else {
                self.for_loop(l, base, fori, false)?;
                t = self.scev.t;
                stop = self.scev.stop;
            }
        } else {
            // FORI/JFORI: load the loop variables, no increment.
            let idx = self.getslot(l, base, ra + FORL_IDX);
            let stop_tr = self.getslot(l, base, ra + FORL_STOP);
            let step = self.getslot(l, base, ra + FORL_STEP);
            self.set_base(ra + FORL_EXT, idx);
            t = IRT_NUM;
            stop = tref_ref(stop_tr);
            let dir = Self::for_direction(self.slot_val(l, base, ra + FORL_STEP));
            self.for_check(dir, step)?;
        }

        let mut op = IROp::LE;
        let ev = Self::for_iter(&mut op, l, base, ra, isforl);
        // The bytecode index after the loop (FORI's forward jump target).
        let exit_pc = (fori as i64 + 1 + bc_j(ins)) as usize;
        // The first loop-body instruction.
        let body_pc = fori + 1;

        // Snapshot the *opposite* outcome: that is where a failing loop
        // condition guard resumes.
        if ev == LoopEvent::Leave {
            self.maxslot = ra + FORL_EXT + 1;
            self.pc = body_pc;
        } else {
            self.maxslot = ra;
            self.pc = exit_pc;
        }
        self.snap_add();

        let nidx = self.base_ref(ra + FORL_IDX);
        self.cur.ir.emitir(irtg(op, t), tref_ref(nidx), stop)?;

        // Now set the recording direction to the taken path.
        if ev == LoopEvent::Leave {
            self.maxslot = ra;
            self.pc = exit_pc;
        } else {
            self.maxslot = ra + FORL_EXT + 1;
            self.pc = body_pc;
        }
        self.needsnap = true;
        Ok(ev)
    }

    /// `rec_loop`: record LOOP. Now, that was easy.
    fn rec_loop(&mut self, ra: u32) -> LoopEvent {
        if ra < self.maxslot {
            self.maxslot = ra;
        }
        self.pc += 1;
        LoopEvent::Enter
    }

    /// `innerloopleft`: did this inner loop repeatedly fail to loop back?
    fn innerloopleft(js: &JitState, pt: GcPtr<Proto>, pc: usize) -> bool {
        let key = super::bc_addr(pt, pc);
        for i in 0..PENALTY_SLOTS {
            if js.penalty[i].pc == key {
                return matches!(js.penalty[i].reason, TraceError::LLEAVE | TraceError::LINNER)
                    && js.penalty[i].val as u32 >= 2 * PENALTY_MIN;
            }
        }
        false
    }

    /// `rec_loop_interp`: handle hitting an interpreted loop opcode.
    /// Returns the link (type, target) when the trace closes.
    fn loop_interp(
        &mut self,
        js: &JitState,
        pc: usize,
        ev: LoopEvent,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        if self.parent == 0 && self.exitno == 0 {
            if Some(pc) == self.startpc && self.framedepth + self.retdepth == 0 {
                if ev == LoopEvent::Leave {
                    return Err(TraceError::LLEAVE); // Must loop back.
                }
                return Ok(Some((TraceLink::Loop, self.cur.traceno))); // Looping root trace.
            } else if ev != LoopEvent::Leave {
                // Entering an inner loop: better wait until it is traced
                // itself, unless it repeatedly failed to loop back.
                if bc_j(self.pt.as_ref().bc[pc]) != -1 && !Self::innerloopleft(js, self.pt, pc) {
                    return Err(TraceError::LINNER);
                }
                if ev != LoopEvent::EnterLo
                    && self.loopref != 0
                    && self.cur.ir.nins() - self.loopref > 24
                {
                    return Err(TraceError::LUNROLL);
                }
                self.loopunroll -= 1;
                if self.loopunroll < 0 {
                    return Err(TraceError::LUNROLL);
                }
                self.loopref = self.cur.ir.nins();
            }
        }
        Ok(None)
    }

    /// `rec_loop_jit`: handle hitting an already compiled loop opcode.
    fn loop_jit(
        &mut self,
        lnk: TraceNo,
        ev: LoopEvent,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        if self.parent == 0 && self.exitno == 0 {
            // Root trace hit an inner loop: better let the inner loop
            // spawn a side trace back here.
            return Err(TraceError::LINNER);
        }
        if ev != LoopEvent::Leave {
            // Side trace enters a compiled loop.
            self.instunroll = 0; // Cannot continue across a compiled loop op.
            if Some(self.pc) == self.startpc && self.framedepth + self.retdepth == 0 {
                return Ok(Some((TraceLink::Loop, self.cur.traceno))); // Form an extra loop.
            }
            return Ok(Some((TraceLink::Root, lnk))); // Link to the loop.
        }
        // Side trace continues across a loop that's left or not entered.
        Ok(None)
    }

    // -- Comparisons --------------------------------------------------------------

    /// `rec_comp_prep`: add a snapshot before a comparison guard.
    fn comp_prep(&mut self) {
        // Prevent merging with snapshot #0, since its PC gets fixed up.
        if self.cur.snap.len() == 1 && self.cur.snap[0].iref == self.cur.ir.nins() {
            self.cur.ir.emit_ins(IRIns::new(irt(IROp::NOP, IRT_NIL), 0, 0));
        }
        self.snap_add();
    }

    /// `rec_comp_fixup`: point the last snapshot at the *opposite* branch
    /// target, so a failing guard resumes on the other path.
    fn comp_fixup(&mut self, pc: usize, cond: bool) {
        let jmpins = self.pt.as_ref().bc[pc + 1];
        let npc = (pc as i64 + 2 + if cond { bc_j(jmpins) } else { 0 }) as usize;
        self.cur.snap.last_mut().unwrap().pc = npc as u32;
        self.needsnap = true;
        if bc_a(jmpins) < self.maxslot {
            self.maxslot = bc_a(jmpins);
        }
    }

    /// `lj_record_objcmp`: record a raw object equality comparison.
    /// 0 = same, 1 = different but same type, 2 = different types.
    fn objcmp(&mut self, a: TRef, b: TRef, av: LuaValue, bv: LuaValue) -> Result<u32, TraceError> {
        let diff = !val_eq(av, bv);
        if !(tref_isk(a) && tref_isk(b)) {
            let ta = tref_type(a);
            let tb = tref_type(b);
            if ta != tb {
                return Ok(2); // Two different types are never equal.
            }
            if ta > IRT_TRUE {
                // For GC types the address identity is the equality.
                let o = if diff { IROp::NE } else { IROp::EQ };
                self.cur.ir.emitir(irtg(o, ta), tref_ref(a), tref_ref(b))?;
            }
        }
        Ok(diff as u32)
    }

    // -- Arithmetic ------------------------------------------------------------------

    /// Numeric arithmetic (no narrowing yet): both operands must be nums.
    fn arith(&mut self, rb: TRef, rc: TRef, op: IROp) -> Result<TRef, TraceError> {
        if !tref_isnum(rb) || !tref_isnum(rc) {
            return Err(TraceError::NYIBC); // Strings/metamethod arith NYI.
        }
        if op == IROp::MOD {
            // x % y ==> x - floor(x/y)*y (IR_MOD is integer-only).
            let tmp = self.cur.ir.emitir(irtn(IROp::DIV), tref_ref(rb), tref_ref(rc))?;
            let tmp = self.cur.ir.emitir(irtn(IROp::FPMATH), tref_ref(tmp), IRFPM_FLOOR)?;
            let tmp = self.cur.ir.emitir(irtn(IROp::MUL), tref_ref(tmp), tref_ref(rc))?;
            return self.cur.ir.emitir(irtn(IROp::SUB), tref_ref(rb), tref_ref(tmp));
        }
        self.cur.ir.emitir(irtn(op), tref_ref(rb), tref_ref(rc))
    }

    // -- Calls and returns (lj_record_call / lj_record_ret) -------------------

    /// Record a fixed-result CALL to a plain Lua closure, inlining the
    /// callee: guard the closure identity, lay down the frame slots
    /// (KFUNC + frame-link constant) and switch the recorder into the
    /// callee frame. Everything else (C functions, varargs, metacalls,
    /// multres) aborts with NYIBC and takes the penalty path.
    fn rec_call(&mut self, l: &LuaState, base: usize, pc: usize, ins: BCIns) -> Result<(), TraceError> {
        let a = bc_a(ins);
        let want = bc_b(ins) as i32 - 1;
        if want < 0 {
            return Err(TraceError::NYIBC); // Multres call (CALLM context).
        }
        let nargs = (bc_c(ins) as u32).wrapping_sub(1);
        let fv = self.slot_val(l, base, a);
        let Some(gf) = fv.as_func() else {
            return Err(TraceError::NYIBC); // __call metamethod NYI.
        };
        let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
            return Err(TraceError::NYIBC); // Fast functions NYI (recff).
        };
        let cpt = cl.proto;
        if cpt.as_ref().flags & crate::proto::PROTO_VARARG != 0 {
            return Err(TraceError::NYIBC);
        }
        // Bound true recursion (JIT_P_callunroll).
        if self.frames.iter().filter(|f| f.callee.addr() == cpt.addr()).count()
            >= self.callunroll as usize
        {
            return Err(TraceError::CUNROLL);
        }
        let framesize = cpt.as_ref().framesize as usize;
        let newbase = self.baseslot + a as usize + 2;
        if 1 + newbase + framesize >= MAX_JSLOTS {
            return Err(TraceError::STACKOV);
        }
        // Guard the callee identity, then specialize the slot to it.
        let ftr = self.getslot(l, base, a);
        let kf = self.cur.ir.kgc(fv.to_bits(), IRT_FUNC);
        if !tref_isk(ftr) {
            self.cur.ir.emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        self.set_base(a, kf);
        // The frame link the interpreter stores: the return ins address.
        let lk = self.cur.ir.kint64(super::bc_addr(self.pt, pc + 1) as u64);
        self.set_base(a + 1, lk | TREF_FRAME);
        // Load the arguments in the caller's context.
        for i in 0..nargs {
            self.getslot(l, base, a + 2 + i);
        }
        self.frames.push(FrameInfo {
            pt: self.pt,
            callee: cpt,
            cbase: a,
            want: want as u32,
            prev_baseslot: self.baseslot,
        });
        self.framedepth += 1;
        self.baseslot = newbase;
        self.pt = cpt;
        // rec_func_setup: pad missing arguments, clear the frame rest.
        let numparams = cpt.as_ref().numparams as u32;
        for s in nargs..numparams {
            self.set_base(s, TREF_NIL);
        }
        for s in numparams..framesize as u32 {
            self.set_base(s, 0);
        }
        self.maxslot = numparams;
        Ok(())
    }

    /// Record a tail call (`lj_record_tailcall`): replace the current
    /// frame in place — the pending return target (FrameInfo) stays.
    /// Mirrors the interpreter's CALLT fast path conditions.
    fn rec_callt(&mut self, l: &LuaState, base: usize, ins: BCIns) -> Result<(), TraceError> {
        let a = bc_a(ins);
        let nargs = (bc_d(ins) as u32).wrapping_sub(1);
        let fv = self.slot_val(l, base, a);
        let Some(gf) = fv.as_func() else {
            return Err(TraceError::NYIBC);
        };
        let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
            return Err(TraceError::NYIBC);
        };
        let cpt = cl.proto;
        if cpt.as_ref().flags & crate::proto::PROTO_VARARG != 0 {
            return Err(TraceError::NYIBC);
        }
        if self.framedepth == 0 {
            // Replacing the trace's entry frame would leave exits with a
            // stale closure context (pc in the new proto, baseslot 2).
            return Err(TraceError::NYIBC);
        }
        // Tailcalls can form a loop: count towards the unroll limit.
        self.tailcalled += 1;
        if self.tailcalled > self.loopunroll {
            return Err(TraceError::LUNROLL);
        }
        let framesize = cpt.as_ref().framesize as usize;
        if 1 + self.baseslot + framesize >= MAX_JSLOTS {
            return Err(TraceError::STACKOV);
        }
        // Guard the callee identity.
        let ftr = self.getslot(l, base, a);
        let kf = self.cur.ir.kgc(fv.to_bits(), IRT_FUNC);
        if !tref_isk(ftr) {
            self.cur.ir.emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        // Load the arguments, then move func + args down in place.
        let args: Vec<TRef> = (0..nargs).map(|i| self.getslot(l, base, a + 2 + i)).collect();
        self.slot[self.baseslot - 2] = kf; // Frame link (baseslot-1) stays.
        for (i, &tr) in args.iter().enumerate() {
            self.set_base(i as u32, tr);
        }
        let numparams = cpt.as_ref().numparams as u32;
        for s in nargs..numparams {
            self.set_base(s, TREF_NIL);
        }
        for s in numparams..framesize as u32 {
            self.set_base(s, 0);
        }
        self.maxslot = numparams;
        self.pt = cpt;
        if let Some(fr) = self.frames.last_mut() {
            fr.callee = cpt; // The pending return now belongs to the new callee.
        }
        Ok(())
    }

    /// Record a RET0/RET1/RET from an inlined frame: move the results to
    /// the caller's call base and pop the frame. Returning past the
    /// trace's entry frame is NYI (LJ_TRERR_NYIRETL).
    fn rec_ret(&mut self, l: &LuaState, base: usize, rbase: u32, gotres: u32) -> Result<(), TraceError> {
        if self.framedepth <= 0 {
            return Err(TraceError::NYIRETL);
        }
        // Load the results while still in the callee frame.
        let res: Vec<TRef> =
            (0..gotres).map(|i| self.getslot(l, base, rbase + i)).collect();
        let callee_top = self.baseslot + self.pt.as_ref().framesize as usize;
        let fr = self.frames.pop().expect("framedepth/frames mismatch");
        self.framedepth -= 1;
        self.baseslot = fr.prev_baseslot;
        self.pt = fr.pt;
        for i in 0..fr.want {
            let tr = if i < gotres { res[i as usize] } else { TREF_NIL };
            self.set_base(fr.cbase + i, tr);
        }
        // Clear the dead frame area (stale KFUNC/link/local trefs).
        for s in (self.baseslot + (fr.cbase + fr.want) as usize)..callee_top {
            self.slot[s] = 0;
        }
        self.maxslot = fr.cbase + fr.want;
        Ok(())
    }

    // -- Upvalues (lj_record.c rec_upvalue) -----------------------------------

    /// Late specialization of the current function (`getcurrf` + the EQ
    /// guard): required before anything derived from the closure (its
    /// upvalue cells) can be treated as a constant.
    fn specialize_curfn(
        &mut self,
        l: &LuaState,
        base: usize,
        fnval: LuaValue,
    ) -> Result<TRef, TraceError> {
        let fslot = self.baseslot - 2;
        let cur = self.slot[fslot];
        if cur != 0 && tref_isk(cur) {
            return Ok(cur); // Inlined frames are specialized already.
        }
        let kf = self.cur.ir.kgc(fnval.to_bits(), IRT_FUNC);
        let ftr = if cur != 0 { cur } else { self.getslot_abs(l, base, fslot) };
        if !tref_isk(ftr) {
            self.cur.ir.emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        self.slot[fslot] = kf;
        Ok(kf)
    }

    /// `getslot` by absolute recorder slot (may lie below the current
    /// frame, e.g. the frame-0 function slot or an aliased caller local).
    fn getslot_abs(&mut self, l: &LuaState, base: usize, abs: usize) -> TRef {
        let tr = self.slot[abs];
        if tr != 0 {
            return tr;
        }
        let frame0 = base - (self.baseslot - 2);
        let v = l.stack[frame0 + abs - 2];
        let t = Self::value_irt(v);
        let mut tr = self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::SLOAD, IRT_GUARD | t),
            abs as IRRef,
            IRSLOAD_TYPECHECK,
        ));
        if irt_ispri(t) {
            tr = tref_pri(t);
        }
        self.slot[abs] = tr;
        tr
    }

    /// Intern a runtime value as an IR constant (`lj_record_constify`).
    fn constify(&mut self, v: LuaValue) -> TRef {
        if v.is_number() {
            self.cur.ir.knum_u64(v.to_bits())
        } else {
            let t = Self::value_irt(v);
            if irt_ispri(t) { tref_pri(t) } else { self.cur.ir.kgc(v.to_bits(), t) }
        }
    }

    /// `rec_upvalue` (loads only): constify immutable upvalues, forward
    /// open upvalues aliasing recorded slots, and load closed cells
    /// through their (constant) address with a type guard.
    fn rec_uget(&mut self, l: &LuaState, base: usize, uvidx: u32) -> Result<TRef, TraceError> {
        let fnval = l.stack[base - 2];
        let Some(gf) = fnval.as_func() else { return Err(TraceError::NYIBC) };
        let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
            return Err(TraceError::NYIBC);
        };
        let Some(&uv) = cl.upvals.get(uvidx as usize) else {
            return Err(TraceError::NYIBC);
        };
        let uvp = uv.as_ref();
        let val = uvp.get();
        // rec_upvalue_constify: immutable upvalues become constants under
        // the closure-identity guard (skip memory-heavy objects).
        if uvp.immutable
            && !(val.is_table()
                || val.is_thread()
                || val.itype() == crate::value::LJ_TUDATA)
        {
            self.specialize_curfn(l, base, fnval)?;
            return Ok(self.constify(val));
        }
        self.specialize_curfn(l, base, fnval)?;
        if uvp.is_open() {
            // Open upvalue: if it aliases a slot of the recorded frames,
            // forward the slot. The closure-identity guard pins the
            // activation, so the alias is stable for this trace.
            let sp = l.stack.as_ptr() as usize;
            let ptr = uvp.value_ptr() as usize;
            let frame0 = base - (self.baseslot - 2);
            if ptr >= sp
                && ptr < sp + l.stack.len() * 8
                && (ptr - sp) % 8 == 0
            {
                let idx = (ptr - sp) / 8;
                if idx + 2 >= frame0 + 2 && idx - frame0 + 2 < MAX_JSLOTS {
                    let abs = idx - frame0 + 2;
                    if abs >= 2 {
                        return Ok(self.getslot_abs(l, base, abs));
                    }
                }
            }
            // Below the trace's frames or on another thread: the cell can
            // close behind our back.
            return Err(TraceError::NYIBC);
        }
        // Closed upvalue: the cell address is a constant (pool slots are
        // stable and closed cells never reopen); load with a type guard.
        let t = Self::value_irt(val);
        let cell = self.cur.ir.kint64(uvp.value_ptr() as u64);
        let mut tr = self
            .cur
            .ir
            .emit_ins(IRIns::new(irt(IROp::ULOAD, IRT_GUARD | t), tref_ref(cell), 0));
        if irt_ispri(t) {
            tr = tref_pri(t);
        }
        Ok(tr)
    }

    // -- Main recording entry (lj_record_ins) ---------------------------------------

    /// Record the instruction at `pc` *before* it is executed. Returns the
    /// link (type, target) when the trace just completed, None to keep
    /// recording.
    pub fn record_ins(
        &mut self,
        js: &JitState,
        l: &LuaState,
        base: usize,
        pc: usize,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        // Need a snapshot before recording the next bytecode (e.g. after
        // a loop condition guard).
        if self.needsnap {
            self.needsnap = false;
            self.snap_add();
            self.mergesnap = true;
        }

        self.pc = pc;
        // Record only closed loops for root traces.
        if self.framedepth == 0 && pc.wrapping_sub(self.bc_min) >= self.bc_extent {
            return Err(TraceError::LLEAVE);
        }

        let pt = self.pt;
        let ins = pt.as_ref().bc[pc];
        let op = bc_op(ins);
        let mut ra = bc_a(ins) as TRef;
        let mut rb = bc_b(ins) as TRef;
        let mut rc = bc_c(ins) as TRef;
        let mut rav = LuaValue::NIL;
        let mut rcv = LuaValue::NIL;

        // Preload var/num/pri/str operands, keeping runtime value copies.
        if bcmode_a(op) == BCMode::Var as u32 {
            rav = self.slot_val(l, base, ra);
            ra = self.getslot(l, base, ra);
        }
        if bcmode_b(op) == BCMode::None as u32 {
            rb = 0;
            rc = bc_d(ins);
        } else if bcmode_b(op) == BCMode::Var as u32 {
            rb = self.getslot(l, base, rb);
        }
        match bcmode_c(op) {
            m if m == BCMode::Var as u32 => {
                rcv = self.slot_val(l, base, rc);
                rc = self.getslot(l, base, rc);
            }
            m if m == BCMode::Pri as u32 => {
                rcv = PRI_VALUES[rc as usize];
                rc = tref_pri(IRT_NIL + rc as u8);
            }
            m if m == BCMode::Num as u32 => {
                let n = pt.as_ref().kn[rc as usize];
                rcv = LuaValue::number_raw(n);
                rc = self.cur.ir.knum(n);
            }
            m if m == BCMode::Str as u32 => {
                let sv = pt.as_ref().kstrv[rc as usize];
                rcv = sv;
                rc = self.cur.ir.kgc(sv.to_bits(), IRT_STR);
            }
            _ => {}
        }

        let mut result: TRef = 0;

        match op {
            // -- Comparison ops --------------------------------------------------
            BCOp::ISLT | BCOp::ISGE | BCOp::ISLE | BCOp::ISGT => {
                // Emit nothing for two numeric constants.
                if !(tref_isk(ra) && tref_isk(rc) && tref_isnum(ra) && tref_isnum(rc)) {
                    if !(tref_isnum(ra) && tref_isnum(rc)) {
                        return Err(TraceError::NYIBC); // Strings/metamethods NYI.
                    }
                    self.comp_prep();
                    let mut irop = IROp::from_u8(op as u8 - BCOp::ISLT as u8 + IROp::LT as u8);
                    if (irop as u8) & 1 != 0 {
                        // ISGE/ISGT are unordered (NaN behavior).
                        irop = IROp::from_u8(irop as u8 ^ 4);
                    }
                    if !super::opt_fold::fold_numcmp(rav.num(), rcv.num(), irop) {
                        irop = IROp::from_u8(irop as u8 ^ 5);
                    }
                    self.cur.ir.emitir(irtg(irop, IRT_NUM), tref_ref(ra), tref_ref(rc))?;
                    self.comp_fixup(pc, ((op as u8) ^ (irop as u8)) & 1 != 0);
                }
            }
            BCOp::ISEQV | BCOp::ISNEV | BCOp::ISEQS | BCOp::ISNES | BCOp::ISEQN
            | BCOp::ISNEN | BCOp::ISEQP | BCOp::ISNEP => {
                // Emit nothing for two non-table constants.
                if !(tref_isk(ra) && tref_isk(rc) && !tref_istab(ra)) {
                    if tref_isnum(ra) && tref_isnum(rc) {
                        // Number equality: guard the outcome.
                        self.comp_prep();
                        let diff = rav.num() != rcv.num();
                        let o = if diff { IROp::NE } else { IROp::EQ };
                        self.cur.ir.emitir(irtg(o, IRT_NUM), tref_ref(ra), tref_ref(rc))?;
                        self.comp_fixup(pc, ((op as u8) & 1 == 1) == !diff);
                    } else {
                        self.comp_prep();
                        let diff = self.objcmp(ra, rc, rav, rcv)?;
                        if diff == 1 && tref_istab(ra) {
                            return Err(TraceError::NYIBC); // __eq metamethod NYI.
                        }
                        self.comp_fixup(pc, ((op as u8) & 1 == 1) == (diff != 0));
                    }
                }
            }

            // -- Unary test and copy ops ------------------------------------------
            BCOp::ISTC | BCOp::ISFC => {
                let truecond = rcv.is_truthy();
                if ((op as u8) & 1 == 1) != truecond {
                    // Condition is true for this opcode: the copy happens.
                    self.set_base(bc_a(ins), rc);
                }
                if bc_a(pt.as_ref().bc[pc + 1]) < self.maxslot {
                    self.maxslot = bc_a(pt.as_ref().bc[pc + 1]);
                }
            }
            BCOp::IST | BCOp::ISF => {
                // Type specialization of the operand's SLOAD suffices.
                if bc_a(pt.as_ref().bc[pc + 1]) < self.maxslot {
                    self.maxslot = bc_a(pt.as_ref().bc[pc + 1]);
                }
            }

            // -- Unary and move ops --------------------------------------------------
            BCOp::MOV => result = rc,
            BCOp::NOT => {
                // Type specialization already forces a const result.
                result = if rcv.is_truthy() { TREF_FALSE } else { TREF_TRUE };
            }
            BCOp::UNM => {
                if !tref_isnum(rc) {
                    return Err(TraceError::NYIBC);
                }
                // op2 stands in for LuaJIT's KSIMD sign-flip constant.
                let signbit = self.cur.ir.knum(-0.0);
                result = self.cur.ir.emitir(irtn(IROp::NEG), tref_ref(rc), tref_ref(signbit))?;
            }

            // -- Constants -------------------------------------------------------------
            BCOp::KSTR | BCOp::KNUM | BCOp::KPRI => result = rc,
            BCOp::KSHORT => result = self.cur.ir.knum(bc_d(ins) as i16 as f64),
            BCOp::KNIL => {
                let mut s = bc_a(ins);
                let last = bc_d(ins);
                while s <= last {
                    self.set_base(s, TREF_NIL);
                    s += 1;
                }
                if last >= self.maxslot {
                    self.maxslot = last + 1;
                }
            }

            // -- Arithmetic --------------------------------------------------------------
            BCOp::ADDVN | BCOp::SUBVN | BCOp::MULVN | BCOp::DIVVN | BCOp::MODVN
            | BCOp::ADDVV | BCOp::SUBVV | BCOp::MULVV | BCOp::DIVVV | BCOp::MODVV => {
                result = self.arith(rb, rc, arith_irop(op))?;
            }
            BCOp::ADDNV | BCOp::SUBNV | BCOp::MULNV | BCOp::DIVNV | BCOp::MODNV => {
                // NV forms: constant op variable — operands swapped.
                result = self.arith(rc, rb, arith_irop(op))?;
            }
            BCOp::POW => result = self.arith(rb, rc, IROp::POW)?,

            // -- Loops and branches ---------------------------------------------------------
            BCOp::JMP => {
                if bc_a(ins) < self.maxslot {
                    self.maxslot = bc_a(ins); // Shrink used slots.
                }
            }
            BCOp::FORL => {
                let fori = (pc as i64 + bc_j(ins)) as usize;
                let ev = self.rec_for(l, base, fori, true)?;
                if let Some(link) = self.loop_interp(js, pc, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::LOOP => {
                let ev = self.rec_loop(bc_a(ins));
                if let Some(link) = self.loop_interp(js, pc, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::IFORL | BCOp::IITERL | BCOp::ILOOP | BCOp::IFUNCF | BCOp::IFUNCV => {
                return Err(TraceError::BLACKL);
            }
            BCOp::JFORI => {
                // The JFORI's jump targets the instruction after the JFORL.
                let jforl = (pc as i64 + bc_j(ins)) as usize;
                debug_assert_eq!(bc_op(pt.as_ref().bc[jforl]), BCOp::JFORL);
                if self.rec_for(l, base, pc, false)? != LoopEvent::Leave {
                    // Link to the existing loop.
                    return Ok(Some((TraceLink::Root, bc_d(pt.as_ref().bc[jforl]))));
                }
                // Continue tracing if the loop is not entered.
            }
            BCOp::JFORL => {
                // D holds the trace number; recover the FORI position from
                // the original FORL stored as the trace's start instruction.
                let lnk = bc_d(ins);
                let startins =
                    js.trace[lnk as usize].as_ref().ok_or(TraceError::NYIBC)?.startins;
                let fori = (pc as i64 + bc_j(startins)) as usize;
                let ev = self.rec_for(l, base, fori, true)?;
                if let Some(link) = self.loop_jit(lnk, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::JLOOP => {
                let lnk = bc_d(ins);
                let startins =
                    js.trace[lnk as usize].as_ref().ok_or(TraceError::NYIBC)?.startins;
                if bc_isret(bc_op(startins)) || bc_op(startins) == BCOp::ITERN {
                    return Err(TraceError::NYIBC); // Patched RET/ITERN loops.
                }
                let ev = self.rec_loop(bc_a(ins));
                if let Some(link) = self.loop_jit(lnk, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::JITERL => return Err(TraceError::NYIBC), // Iterators are NYI.

            // -- Upvalues ----------------------------------------------------------
            BCOp::UGET => result = self.rec_uget(l, base, bc_d(ins))?,

            // -- Calls and returns -------------------------------------------------
            BCOp::CALL => self.rec_call(l, base, pc, ins)?,
            BCOp::CALLT => self.rec_callt(l, base, ins)?,
            BCOp::RET0 => self.rec_ret(l, base, bc_a(ins), 0)?,
            BCOp::RET1 => self.rec_ret(l, base, bc_a(ins), 1)?,
            BCOp::RET => self.rec_ret(l, base, bc_a(ins), bc_d(ins) - 1)?,
            BCOp::FUNCF => {
                // Reached only through the slow call paths (the fast path
                // enters at bc[1]); those shapes are not recorded yet.
                return Err(TraceError::NYIBC);
            }

            // Everything else is NYI in Phase 2: calls, returns, tables,
            // upvalues, iterators, varargs, concat, bit ops, lengths.
            _ => return Err(TraceError::NYIBC),
        }

        // Store the result of dst-mode instructions.
        if bcmode_a(op) == BCMode::Dst as u32 && result != 0 {
            let a = bc_a(ins);
            if a > self.maxslot {
                // Clear the gap below (FR2 frame layout).
                self.slot[self.baseslot + self.maxslot as usize] = 0;
            }
            self.set_base(a, result);
            if a >= self.maxslot {
                self.maxslot = a + 1;
            }
        }

        // Limit the number of recorded IR instructions and constants.
        if self.cur.ir.nins() > REF_FIRST + js.param(JitParam::MaxRecord) as IRRef
            || self.cur.ir.nk() < REF_BIAS - js.param(JitParam::MaxIrConst) as IRRef
        {
            return Err(TraceError::TRACEOV);
        }
        Ok(None)
    }
}

/// BC arith opcode to IR op.
fn arith_irop(op: BCOp) -> IROp {
    match op {
        BCOp::ADDVN | BCOp::ADDNV | BCOp::ADDVV => IROp::ADD,
        BCOp::SUBVN | BCOp::SUBNV | BCOp::SUBVV => IROp::SUB,
        BCOp::MULVN | BCOp::MULNV | BCOp::MULVV => IROp::MUL,
        BCOp::DIVVN | BCOp::DIVNV | BCOp::DIVVV => IROp::DIV,
        BCOp::MODVN | BCOp::MODNV | BCOp::MODVV => IROp::MOD,
        _ => unreachable!("bad arith opcode"),
    }
}

/// Raw value equality, same semantics as the interpreter's `val_eq`.
fn val_eq(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() { a.num() == b.num() } else { a.to_bits() == b.to_bits() }
}
