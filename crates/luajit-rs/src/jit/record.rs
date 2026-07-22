//! Trace recorder: translates executed bytecode to IR, one instruction at
//! a time. Ported from lj_record.c, with the snapshot-generation half of
//! lj_snap.c (snapshot_slots/snapshot_stack/lj_snap_add).
//!
//! Phase 2 scope: single-frame numeric traces — FORL and LOOP roots,
#![allow(clippy::too_many_arguments)]
//! arithmetic, moves, constants and comparisons. Everything else aborts
//! with NYIBC and feeds the penalty/blacklist engine, exactly like LuaJIT
//! handles unrecordable bytecode. Calls, table access and returns arrive
//! with later phases, as does number->integer narrowing (integer-valued
//! constants and FORL induction variables use IRT_INT; general narrowing
//! will arrive with a dedicated backward narrowing pass).

use crate::bc::*;
use crate::gc::GcPtr;
use crate::proto::Proto;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::ir::*;
use super::{
    GCtrace, JitParam, JitState, PENALTY_MIN, PENALTY_SLOTS, SNAP_CONT, SNAP_FRAME, SNAP_NORESTORE,
    SnapEntry, SnapShot, TraceError, TraceLink, TraceNo, snap_entry, snap_ref, snap_slot,
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

/// FLOAD field literals (IRFLDEF subset).
pub const IRFL_TAB_META: u32 = 0;

/// IRCALL indexes (CALLL op2 literals).
pub const IRCALL_TAB_NEXTK: u32 = 0;
pub const IRCALL_FMOD: u32 = 1;
pub const IRCALL_STR_LEN: u32 = 2;
pub const IRCALL_STR_CMP: u32 = 3;
pub const IRCALL_STR_BYTE: u32 = 4;
pub const IRCALL_STR_SUB: u32 = 5;
pub const IRCALL_STR_CHAR: u32 = 6;
pub const IRCALL_TAB_LEN: u32 = 7;
pub const IRCALL_TAB_CONCAT: u32 = 8;
pub const IRCALL_CAT: u32 = 9;
pub const IRCALL_USET: u32 = 10;
pub const IRCALL_VARG: u32 = 11;

/// Argument count of an IRCALL: 1 = op1 is the argument, 2 = op1 is a
/// CARG pair, 3 = op1 is CARG(CARG(a, b), c).
pub fn ircall_arity(idx: u32) -> u32 {
    match idx {
        IRCALL_STR_LEN | IRCALL_STR_CHAR | IRCALL_TAB_LEN => 1,
        IRCALL_STR_SUB | IRCALL_VARG => 3,
        _ => 2,
    }
}

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
        ScEv {
            pc: None,
            idx: REF_NIL,
            stop: 0,
            step: 0,
            t: IRT_NUM,
            dir: true,
        }
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
    /// Min. unroll depth before stopping true recursion (JIT_P_recunroll).
    pub recunroll: i32,
    /// Number of recorded tailcalls (J->tailcalled).
    pub tailcalled: i32,
    /// The proto being recorded (single-frame traces: always startpt).
    pub pt: GcPtr<Proto>,
    /// Current bytecode index, updated as recording progresses (J->pc).
    pub pc: usize,
}

impl Record {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        traceno: TraceNo,
        pt: GcPtr<Proto>,
        pc: usize,
        parent: u16,
        exitno: u16,
        loopunroll: i32,
        instunroll: i32,
        callunroll: i32,
        recunroll: i32,
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
            recunroll,
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
        if v.is_number() {
            IRT_NUM
        } else {
            (!v.itype()) as u8 & IRT_TYPE
        }
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
                self.cur
                    .ir
                    .emit_ins(IRIns::new(irt(IROp::NOP, IRT_NIL), 0, 0));
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
                    let tr = self
                        .cur
                        .ir
                        .emit_ins(IRIns::new(irt(IROp::SLOAD, ty), s, mode));
                    // The executor pre-fills env[own] from the parent's
                    // env[r] on a linked exit (register-free hand-over).
                    self.cur
                        .parentmap
                        .push((tref_ref(tr) as IRRef1, r as IRRef1));
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
                let Some((fs, fref)) = prev_entry else {
                    return Err(TraceError::NYIBC);
                };
                if fs != s - 1 || !irref_isk(fref) {
                    return Err(TraceError::NYIBC);
                }
                let fv = LuaValue::from_bits(t.ir.k64_val(fref));
                let Some(gf) = fv.as_func() else {
                    return Err(TraceError::NYIBC);
                };
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
                return if idx + 2.0 * step > stop {
                    LoopEvent::EnterLo
                } else {
                    LoopEvent::Enter
                };
            }
            *op = IROp::GT;
        } else {
            if stop <= idx {
                *op = IROp::GE;
                return if idx + 2.0 * step < stop {
                    LoopEvent::EnterLo
                } else {
                    LoopEvent::Enter
                };
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
            if tref_isint(step) {
                let zero = self.cur.ir.kint(0);
                self.cur.ir.emitir(
                    irtg(if dir { IROp::GE } else { IROp::LT }, IRT_INT),
                    tref_ref(step),
                    tref_ref(zero),
                )?;
            } else {
                let zero = self.cur.ir.knum(0.0);
                self.cur.ir.emitir(
                    irtg(if dir { IROp::GE } else { IROp::LT }, IRT_NUM),
                    tref_ref(step),
                    tref_ref(zero),
                )?;
            }
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
        let stop_val = self.slot_val(l, base, ra + FORL_STOP).num();
        let step_val = self.slot_val(l, base, ra + FORL_STEP).num();
        let int_step = num_isint(step_val);
        let int_stop = num_isint(stop_val);
        let narrow = int_step && int_stop && tref_isk(step) && tref_isk(stop);
        if narrow {
            let step_int = self.cur.ir.kint(step_val as i32);
            let stop_int = self.cur.ir.kint(stop_val as i32);
            self.scev.t = IRT_NUM;
            self.scev.dir = dir;
            self.scev.stop = tref_ref(stop_int);
            self.scev.step = tref_ref(step_int);
            self.for_check(dir, step_int)?;
        } else {
            self.scev.t = IRT_NUM;
            self.scev.dir = dir;
            self.scev.stop = tref_ref(stop);
            self.scev.step = tref_ref(step);
            self.for_check(dir, step)?;
        }
        if idx == 0 {
            idx = self.sloadt(ra + FORL_IDX, IRT_NUM, IRSLOAD_INHERIT);
        }
        if !init {
            idx = self
                .cur
                .ir
                .emitir(irtn(IROp::ADD), tref_ref(idx), self.scev.step)?;
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
                return matches!(
                    js.penalty[i].reason,
                    TraceError::LLEAVE | TraceError::LINNER
                ) && js.penalty[i].val as u32 >= 2 * PENALTY_MIN;
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
            self.cur
                .ir
                .emit_ins(IRIns::new(irt(IROp::NOP, IRT_NIL), 0, 0));
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

    /// Numeric arithmetic with integer narrowing: if both operands are
    /// integers, emit an integer op; otherwise fall back to FP.
    fn arith(&mut self, rb: TRef, rc: TRef, op: IROp) -> Result<TRef, TraceError> {
        if !tref_isnum_or_int(rb) || !tref_isnum_or_int(rc) {
            return Err(TraceError::NYIBC);
        }
        let int_rb = tref_isint(rb);
        let int_rc = tref_isint(rc);
        let use_int =
            (int_rb && int_rc) && matches!(op, IROp::ADD | IROp::SUB | IROp::MUL | IROp::MOD);
        if use_int {
            if op == IROp::MOD {
                return self
                    .cur
                    .ir
                    .emitir(irti(IROp::MOD), tref_ref(rb), tref_ref(rc));
            }
            return self.cur.ir.emitir(irti(op), tref_ref(rb), tref_ref(rc));
        }
        if op == IROp::MOD {
            let tmp = self
                .cur
                .ir
                .emitir(irtn(IROp::DIV), tref_ref(rb), tref_ref(rc))?;
            let tmp = self
                .cur
                .ir
                .emitir(irtn(IROp::FPMATH), tref_ref(tmp), IRFPM_FLOOR)?;
            let tmp = self
                .cur
                .ir
                .emitir(irtn(IROp::MUL), tref_ref(tmp), tref_ref(rc))?;
            return self
                .cur
                .ir
                .emitir(irtn(IROp::SUB), tref_ref(rb), tref_ref(tmp));
        }
        self.cur.ir.emitir(irtn(op), tref_ref(rb), tref_ref(rc))
    }

    // -- Calls and returns (lj_record_call / lj_record_ret) -------------------

    /// Record a fixed-result CALL: dispatch on the callee — Lua closures
    /// are inlined as a frame, recordable builtins as IR (lj_ffrecord).
    /// Everything else (metacalls, varargs, multres) aborts with NYIBC.
    /// A `Some` result stops the trace (recursion or a compiled callee).
    fn rec_call(
        &mut self,
        l: &LuaState,
        base: usize,
        pc: usize,
        ins: BCIns,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        let a = bc_a(ins);
        let want = bc_b(ins) as i32 - 1;
        let nargs = (bc_c(ins) as u32).wrapping_sub(1);
        let fv = self.slot_val(l, base, a);
        self.getslot(l, base, a); // The identity guard reads the tref.
        let Some(gf) = fv.as_func() else {
            return Err(TraceError::NYIBC); // __call metamethod NYI.
        };
        match gf.as_ref() {
            crate::func::GcFunc::C(cc) => {
                let mut argv = [LuaValue::NIL; 3];
                for i in 0..nargs.min(3) {
                    argv[i as usize] = self.slot_val(l, base, a + 2 + i);
                }
                for i in 0..nargs {
                    self.getslot(l, base, a + 2 + i);
                }
                self.rec_callff_core(l, a, want, nargs, fv, cc.f, &argv)?;
                Ok(None)
            }
            crate::func::GcFunc::Lua(_) => self.rec_call_lua(l, base, pc, a, nargs, want, fv),
        }
    }

    /// Inline a call to a plain Lua closure: guard the closure identity,
    /// lay down the frame slots (KFUNC + frame-link constant) and switch
    /// the recorder into the callee frame. Stops the trace (`Some`) on
    /// up-recursion to the trace head (`check_call_unroll`) or when the
    /// callee is already compiled (`rec_func_jit`).
    #[allow(clippy::too_many_arguments)]
    fn rec_call_lua(
        &mut self,
        l: &LuaState,
        base: usize,
        pc: usize,
        a: u32,
        nargs: u32,
        want: i32,
        fv: LuaValue,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        if want < 0 {
            return Err(TraceError::NYIBC); // Multres call (CALLM context).
        }
        let gf = fv.as_func().expect("checked by the caller");
        let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
            unreachable!()
        };
        let cpt = cl.proto;
        if cpt.as_ref().flags & crate::proto::PROTO_VARARG != 0 {
            return Err(TraceError::NYIBC);
        }
        // Root traces cannot safely inline closure calls with closed
        // upvalues: closures re-created by FNEW on each outer loop
        // iteration get fresh upvalue cells whose addresses differ from
        // the recording-time constants, producing hangs or crashes.
        if self.parent == 0 {
            for &uv in &cl.upvals {
                if !uv.as_ref().is_open() {
                    return Err(TraceError::BLACKL);
                }
            }
        }
        // Up-recursion to the trace head (check_call_unroll): stop the
        // trace after `recunroll` inlined levels and link it to itself.
        let is_head_call = self.parent == 0
            && bc_op(self.cur.startins) == BCOp::FUNCF
            && cpt.addr() == self.cur.startpt.addr();
        let same_frames = self
            .frames
            .iter()
            .filter(|f| f.callee.addr() == cpt.addr())
            .count();
        let stop_uprec = is_head_call && same_frames >= self.recunroll as usize;
        // A compiled callee stops the trace with a link to its root
        // trace (rec_func_jit) once the frame is laid down.
        let callee_head = cpt.as_ref().bc[0];
        let stop_root = !stop_uprec && bc_op(callee_head) == BCOp::JFUNCF;
        // Bound plain recursion (JIT_P_callunroll).
        if !stop_uprec && !stop_root && same_frames >= self.callunroll as usize {
            return Err(TraceError::CUNROLL);
        }
        let framesize = cpt.as_ref().framesize as usize;
        let newbase = self.baseslot + a as usize + 2;
        if 1 + newbase + framesize >= MAX_JSLOTS {
            return Err(TraceError::STACKOV);
        }
        // Guard the callee identity, then specialize the slot to it.
        let ftr = self.base_ref(a);
        debug_assert!(ftr != 0, "callee tref not loaded");
        let kf = self.cur.ir.kgc(fv.to_bits(), IRT_FUNC);
        if !tref_isk(ftr) {
            self.cur
                .ir
                .emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        self.set_base(a, kf);
        // The frame link the interpreter stores: the return ins address.
        let lk = self.cur.ir.kint64(super::bc_addr(self.pt, pc + 1) as u64);
        self.set_base(a + 1, lk | TREF_FRAME);
        // Load the arguments in the caller's context (already-populated
        // trefs — e.g. the copied ITERC triple — are returned as-is).
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
        if stop_uprec || stop_root {
            // The final snapshot describes "frame pushed, about to run
            // the callee's first instruction" (snapshot pcs are relative
            // to the innermost frame's proto).
            self.pc = 1;
            if stop_uprec {
                return Ok(Some((TraceLink::Uprec, self.cur.traceno)));
            }
            return Ok(Some((TraceLink::Root, bc_d(callee_head) as TraceNo)));
        }
        Ok(None)
    }

    /// `rec_iterc`: mirror the interpreter's `iter_call` (copy the
    /// callable/state/control triple up), then dispatch like a CALL with
    /// two arguments.
    fn rec_iterc(
        &mut self,
        l: &LuaState,
        base: usize,
        pc: usize,
        ins: BCIns,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        let a = bc_a(ins);
        let want = bc_b(ins) as i32 - 1;
        let func = self.getslot(l, base, a - 3);
        let state = self.getslot(l, base, a - 2);
        let ctrl = self.getslot(l, base, a - 1);
        self.set_base(a, func);
        self.set_base(a + 1, 0);
        self.set_base(a + 2, state);
        self.set_base(a + 3, ctrl);
        let fv = self.slot_val(l, base, a - 3);
        let Some(gf) = fv.as_func() else {
            return Err(TraceError::NYIBC); // __call iterator NYI.
        };
        match gf.as_ref() {
            crate::func::GcFunc::C(cc) => {
                let argv = [self.slot_val(l, base, a - 2), self.slot_val(l, base, a - 1)];
                self.rec_callff_core(l, a, want, 2, fv, cc.f, &argv)?;
                Ok(None)
            }
            crate::func::GcFunc::Lua(_) => self.rec_call_lua(l, base, pc, a, 2, want, fv),
        }
    }

    /// `rec_iterl`: the loop-back decision is static — the nil-ness of
    /// the first result is encoded in its tref type (the iterator's load
    /// guard provides the runtime check).
    fn rec_iterl(&mut self, l: &LuaState, base: usize, pc: usize, iterins: BCIns) -> LoopEvent {
        let ra = bc_a(iterins);
        let tr = self.getslot(l, base, ra);
        if !tref_isnil(tr) {
            self.set_base(ra - 1, tr); // Copy the result to the control var.
            let iterc = self.pt.as_ref().bc[pc - 1];
            debug_assert!(matches!(bc_op(iterc), BCOp::ITERC | BCOp::ITERN));
            self.maxslot = ra - 1 + bc_b(iterc);
            self.pc = (pc as i64 + 1 + bc_j(iterins)) as usize;
            LoopEvent::Enter
        } else {
            self.maxslot = ra - 3;
            self.pc = pc + 1;
            LoopEvent::Leave
        }
    }

    /// Record a tail call (`lj_record_tailcall`): replace the current
    /// frame in place — the pending return target (FrameInfo) stays.
    /// Mirrors the interpreter's CALLT fast path conditions. Tail
    /// recursion to the trace head stops the trace (`Some`).
    fn rec_callt(
        &mut self,
        l: &LuaState,
        base: usize,
        ins: BCIns,
    ) -> Result<Option<(TraceLink, TraceNo)>, TraceError> {
        let a = bc_a(ins);
        let nargs = bc_d(ins).wrapping_sub(1);
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
        // Tail recursion back to the trace head: replace the entry frame
        // in place and stop with a self link (TAILREC).
        let stop_tailrec = self.framedepth == 0
            && self.parent == 0
            && bc_op(self.cur.startins) == BCOp::FUNCF
            && cpt.addr() == self.cur.startpt.addr();
        if self.framedepth == 0 && !stop_tailrec {
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
            self.cur
                .ir
                .emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        // Load the arguments, then move func + args down in place.
        let args: Vec<TRef> = (0..nargs)
            .map(|i| self.getslot(l, base, a + 2 + i))
            .collect();
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
        if stop_tailrec {
            self.pc = 1;
            return Ok(Some((TraceLink::Tailrec, self.cur.traceno)));
        }
        Ok(None)
    }

    /// Record a RET0/RET1/RET from an inlined frame: move the results to
    /// the caller's call base and pop the frame. Returning past the
    /// trace's entry frame is NYI (LJ_TRERR_NYIRETL).
    fn rec_ret(
        &mut self,
        l: &LuaState,
        base: usize,
        rbase: u32,
        gotres: u32,
    ) -> Result<(), TraceError> {
        if self.framedepth <= 0 {
            return Err(TraceError::NYIRETL);
        }
        // Load the results while still in the callee frame.
        let res: Vec<TRef> = (0..gotres)
            .map(|i| self.getslot(l, base, rbase + i))
            .collect();
        let callee_top = self.baseslot + self.pt.as_ref().framesize as usize;
        let fr = self.frames.pop().expect("framedepth/frames mismatch");
        self.framedepth -= 1;
        self.baseslot = fr.prev_baseslot;
        self.pt = fr.pt;
        for i in 0..fr.want {
            let tr = if i < gotres {
                res[i as usize]
            } else {
                TREF_NIL
            };
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
        let ftr = if cur != 0 {
            cur
        } else {
            self.getslot_abs(l, base, fslot)
        };
        if !tref_isk(ftr) {
            self.cur
                .ir
                .emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
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
            if irt_ispri(t) {
                tref_pri(t)
            } else {
                self.cur.ir.kgc(v.to_bits(), t)
            }
        }
    }

    /// `rec_upvalue` (loads only): constify immutable upvalues, forward
    /// open upvalues aliasing recorded slots, and load closed cells
    /// through their (constant) address with a type guard.
    fn rec_uget(&mut self, l: &LuaState, base: usize, uvidx: u32) -> Result<TRef, TraceError> {
        let fnval = l.stack[base - 2];
        let Some(gf) = fnval.as_func() else {
            return Err(TraceError::NYIBC);
        };
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
            && !(val.is_table() || val.is_thread() || val.itype() == crate::value::LJ_TUDATA)
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
            if ptr >= sp && ptr < sp + l.stack.len() * 8 && (ptr - sp).is_multiple_of(8) {
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
        let mut tr = self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::ULOAD, IRT_GUARD | t),
            tref_ref(cell),
            0,
        ));
        if irt_ispri(t) {
            tr = tref_pri(t);
        }
        Ok(tr)
    }

    // -- Table indexing (lj_record_idx, coarse helper-based form) -------------

    /// Guard `tab.metatable == nil` (FLOAD IRFL_TAB_META).
    fn meta_guard(&mut self, tab: TRef) {
        self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::FLOAD, IRT_GUARD | IRT_NIL),
            tref_ref(tab),
            IRFL_TAB_META,
        ));
    }

    /// Record a raw table load: HLOAD specialized to the runtime result
    /// type; a nil result additionally guards `metatable == nil` (the
    /// interpreter only consults `__index` for nil results). Metamethod
    /// paths abort with NYIBC.
    fn rec_tget(
        &mut self,
        tabv: LuaValue,
        tab: TRef,
        keyv: LuaValue,
        key: TRef,
    ) -> Result<TRef, TraceError> {
        if !tref_istab(tab) {
            return Err(TraceError::NYIBC); // __index on non-tables NYI.
        }
        let Some(t) = tabv.as_table() else {
            return Err(TraceError::NYIBC);
        };
        let v = LuaValue::from_bits(super::exec::jit_tget(tabv.to_bits(), keyv.to_bits()));
        if v.is_nil() {
            if t.as_ref().metatable.is_some() {
                return Err(TraceError::NYIBC); // __index metamethod NYI.
            }
            self.meta_guard(tab);
        }
        let ty = Self::value_irt(v);
        // Array fast path (get_int's inline case): integer key inside
        // the array part loads through a pointer, no helper call.
        let use_aload = tref_isnum(key)
            && keyv.as_number().is_some_and(|n| {
                let ki = n as i32;
                ki as f64 == n && ki >= 0 && (ki as u32) < t.as_ref().asize
            });
        let opk = if use_aload { IROp::ALOAD } else { IROp::HLOAD };
        let mut tr = self.cur.ir.emit_ins(IRIns::new(
            irt(opk, IRT_GUARD | ty),
            tref_ref(tab),
            tref_ref(key),
        ));
        if irt_ispri(ty) {
            tr = tref_pri(ty);
        }
        Ok(tr)
    }

    /// Record a raw table store: `metatable == nil` guard + HSTORE (the
    /// helper mirrors the interpreter's raw set dispatch, including
    /// inserts and resizes). Integer keys inside the array part use the
    /// inlined ASTORE instead. Forces a snapshot after the store so
    /// exits never replay a load/store sequence, and emits a GC-debt
    /// guard (IR_GCSTEP): stores may grow the table, and a compiled
    /// loop otherwise never reaches a GC safe point.
    fn rec_tset(
        &mut self,
        l: &LuaState,
        tabv: LuaValue,
        tab: TRef,
        keyv: LuaValue,
        key: TRef,
        val: TRef,
    ) -> Result<(), TraceError> {
        if !tref_istab(tab) {
            return Err(TraceError::NYIBC);
        }
        let Some(t) = tabv.as_table() else {
            return Err(TraceError::NYIBC);
        };
        if t.as_ref().metatable.is_some() {
            return Err(TraceError::NYIBC); // __newindex / protected path NYI.
        }
        if keyv.is_nil() {
            return Err(TraceError::NYIBC); // Runtime error path.
        }
        self.meta_guard(tab);
        let carg = self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::CARG, IRT_NIL),
            tref_ref(key),
            tref_ref(val),
        ));
        // Array fast path (set_int's inline case): integer key strictly
        // inside the array part — a plain store, no allocation.
        let ki = keyv.as_number().map(|n| n as i32);
        if tref_isnum(key)
            && let Some(ki) = ki
            && ki as f64 == keyv.num()
            && ki > 0
            && (ki as u32) < t.as_ref().asize
        {
            self.cur.ir.emit_ins(IRIns::new(
                irt(IROp::ASTORE, IRT_NIL),
                tref_ref(tab),
                tref_ref(carg),
            ));
            self.needsnap = true;
            return Ok(());
        }
        self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::HSTORE, IRT_NIL),
            tref_ref(tab),
            tref_ref(carg),
        ));
        self.rec_gcstep(l);
        Ok(())
    }

    /// GC-debt guard: exit when a collection is due (the boundary check
    /// in the trace-entry dispatch arms then collects). Must follow any
    /// on-trace allocation (table growth, string interning). One guard
    /// per trace suffices: the debt accumulates and a single iteration
    /// allocates far less than the threshold headroom (the loop peel
    /// keeps one copy in the loop body).
    fn rec_gcstep(&mut self, l: &LuaState) {
        if self.cur.ir.chain[IROp::GCSTEP as usize] != 0 {
            self.needsnap = true;
            return;
        }
        let heap = &l.global().heap;
        let ktotal = self.cur.ir.kint64(&heap.total as *const usize as u64);
        let kthres = self.cur.ir.kint64(&heap.threshold as *const usize as u64);
        self.cur.ir.emit_ins(IRIns::new(
            irt(IROp::GCSTEP, IRT_GUARD | IRT_NIL),
            tref_ref(ktotal),
            tref_ref(kthres),
        ));
        self.needsnap = true;
    }

    // -- Fast functions (lj_ffrecord.c, pointer-keyed subset) -----------------

    /// Record a call to a recordable builtin: the callee identity is
    /// guarded like a Lua closure, then the semantics are inlined as IR
    /// (no frame). The argument trefs must already sit in the call slots
    /// (a+2..); `argv` holds their runtime values. Unknown builtins
    /// abort with NYIBC.
    fn rec_callff_core(
        &mut self,
        l: &LuaState,
        a: u32,
        want: i32,
        nargs: u32,
        fv: LuaValue,
        f: crate::func::CFunction,
        argv: &[LuaValue],
    ) -> Result<(), TraceError> {
        let ff = recff_lookup(f).ok_or(TraceError::NYIBC)?;
        if want < 0 {
            return Err(TraceError::NYIBC); // Multres call NYI.
        }
        if nargs < 1 {
            return Err(TraceError::NYIBC); // Argument error path.
        }
        // Guard the callee identity.
        let ftr = self.base_ref(a);
        debug_assert!(ftr != 0, "callee tref not loaded");
        let kf = self.cur.ir.kgc(fv.to_bits(), IRT_FUNC);
        if !tref_isk(ftr) {
            self.cur
                .ir
                .emitir(irtg(IROp::EQ, IRT_FUNC), tref_ref(ftr), tref_ref(kf))?;
        }
        let mut res = [TREF_NIL; 2];
        let nres: u32;
        match ff {
            Recff::Abs | Recff::Floor | Recff::Ceil | Recff::Sqrt => {
                let arg0 = self.base_ref(a + 2);
                if !tref_isnum(arg0) {
                    return Err(TraceError::NYIBC); // Coercion/error path.
                }
                res[0] = match ff {
                    Recff::Abs => self.cur.ir.emitir(irtn(IROp::ABS), tref_ref(arg0), 0)?,
                    _ => {
                        let fpm = match ff {
                            Recff::Floor => IRFPM_FLOOR,
                            Recff::Ceil => IRFPM_CEIL,
                            _ => IRFPM_SQRT,
                        };
                        let r = self
                            .cur
                            .ir
                            .emitir(irtn(IROp::FPMATH), tref_ref(arg0), fpm)?;
                        // The builtins push through LuaValue::number,
                        // which normalizes -0.0 to +0.0: match that
                        // exactly (x + 0.0; raw-emitted so no fold rule
                        // may drop it).
                        let zero = self.cur.ir.knum(0.0);
                        self.cur.ir.emit_ins(IRIns::new(
                            irtn(IROp::ADD),
                            tref_ref(r),
                            tref_ref(zero),
                        ))
                    }
                };
                nres = 1;
            }
            Recff::IpairsAux => {
                // recff_ipairs_aux: k = ctrl+1; v = t[k]; nil v ends.
                if nargs < 2 {
                    return Err(TraceError::NYIBC);
                }
                let tab = self.base_ref(a + 2);
                let ctrl = self.base_ref(a + 3);
                if !tref_istab(tab) || !tref_isnum(ctrl) {
                    return Err(TraceError::NYIBC);
                }
                let one = self.cur.ir.knum(1.0);
                let k = self
                    .cur
                    .ir
                    .emitir(irtn(IROp::ADD), tref_ref(ctrl), tref_ref(one))?;
                let iv = argv[1].as_number().ok_or(TraceError::NYIBC)? + 1.0;
                let v = self.rec_tget(argv[0], tab, LuaValue::number(iv), k)?;
                if tref_isnil(v) {
                    res[0] = TREF_NIL;
                    nres = 1;
                } else {
                    res[0] = k;
                    res[1] = v;
                    nres = 2;
                }
            }
            Recff::Next => {
                // recff_next: k' = table-traversal step (a guarded helper
                // call, type-specialized per phase: number keys in the
                // array part, GC keys in the hash part — the phase
                // transition exits and grows a side trace), v = t[k'].
                let tab = self.base_ref(a + 2);
                if !tref_istab(tab) {
                    return Err(TraceError::NYIBC);
                }
                let key = if nargs >= 2 {
                    self.base_ref(a + 3)
                } else {
                    TREF_NIL
                };
                let tabv = argv[0];
                let keyv = if nargs >= 2 { argv[1] } else { LuaValue::NIL };
                let nkv =
                    LuaValue::from_bits(super::exec::jit_tnextk(tabv.to_bits(), keyv.to_bits()));
                let t_nk = Self::value_irt(nkv);
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(tab),
                    tref_ref(key),
                ));
                let mut nk = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_GUARD | t_nk),
                    tref_ref(carg),
                    IRCALL_TAB_NEXTK,
                ));
                if irt_ispri(t_nk) {
                    nk = tref_pri(t_nk);
                }
                if nkv.is_nil() {
                    res[0] = TREF_NIL;
                    nres = 1;
                } else {
                    let v = self.rec_tget(tabv, tab, nkv, nk)?;
                    res[0] = nk;
                    res[1] = v;
                    nres = 2;
                }
            }
            Recff::MathMin | Recff::MathMax => {
                // The stdlib folds `if next < acc { acc = next }`, which
                // is exactly MIN(next, acc) — including the NaN and ±0
                // tie behavior of minsd/maxsd. The push goes through
                // LuaValue::number, so normalize -0.0 at the end.
                let irop = if ff == Recff::MathMin {
                    IROp::MIN
                } else {
                    IROp::MAX
                };
                let mut acc = self.base_ref(a + 2);
                if !tref_isnum(acc) {
                    return Err(TraceError::NYIBC);
                }
                for i in 1..nargs {
                    let x = self.base_ref(a + 2 + i);
                    if !tref_isnum(x) {
                        return Err(TraceError::NYIBC);
                    }
                    acc = self.cur.ir.emitir(irtn(irop), tref_ref(x), tref_ref(acc))?;
                }
                let zero = self.cur.ir.knum(0.0);
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irtn(IROp::ADD),
                    tref_ref(acc),
                    tref_ref(zero),
                ));
                nres = 1;
            }
            Recff::Fmod => {
                // The helper mirrors math.fmod (x % y through
                // LuaValue::number) bit for bit; no guard needed.
                if nargs < 2 {
                    return Err(TraceError::NYIBC);
                }
                let x = self.base_ref(a + 2);
                let y = self.base_ref(a + 3);
                if !tref_isnum(x) || !tref_isnum(y) {
                    return Err(TraceError::NYIBC);
                }
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(x),
                    tref_ref(y),
                ));
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NUM),
                    tref_ref(carg),
                    IRCALL_FMOD,
                ));
                nres = 1;
            }
            Recff::Tobit => {
                res[0] = self.rec_tobit(self.base_ref(a + 2))?;
                nres = 1;
            }
            Recff::Bnot => {
                let x = self.rec_tobit(self.base_ref(a + 2))?;
                res[0] = self
                    .cur
                    .ir
                    .emit_ins(IRIns::new(irtn(IROp::BNOT), tref_ref(x), 0));
                nres = 1;
            }
            Recff::Bswap => {
                let x = self.rec_tobit(self.base_ref(a + 2))?;
                res[0] = self
                    .cur
                    .ir
                    .emit_ins(IRIns::new(irtn(IROp::BSWAP), tref_ref(x), 0));
                nres = 1;
            }
            Recff::Band | Recff::Bor | Recff::Bxor => {
                let irop = match ff {
                    Recff::Band => IROp::BAND,
                    Recff::Bor => IROp::BOR,
                    _ => IROp::BXOR,
                };
                let mut acc = self.rec_tobit(self.base_ref(a + 2))?;
                for i in 1..nargs {
                    let x = self.rec_tobit(self.base_ref(a + 2 + i))?;
                    acc = self
                        .cur
                        .ir
                        .emit_ins(IRIns::new(irtn(irop), tref_ref(acc), tref_ref(x)));
                }
                res[0] = acc;
                nres = 1;
            }
            Recff::Lshift | Recff::Rshift | Recff::Arshift | Recff::Rol | Recff::Ror => {
                if nargs < 2 {
                    return Err(TraceError::NYIBC);
                }
                let irop = match ff {
                    Recff::Lshift => IROp::BSHL,
                    Recff::Rshift => IROp::BSHR,
                    Recff::Arshift => IROp::BSAR,
                    Recff::Rol => IROp::BROL,
                    _ => IROp::BROR,
                };
                let x = self.rec_tobit(self.base_ref(a + 2))?;
                let n = self.rec_tobit(self.base_ref(a + 3))?;
                res[0] = self
                    .cur
                    .ir
                    .emit_ins(IRIns::new(irtn(irop), tref_ref(x), tref_ref(n)));
                nres = 1;
            }
            Recff::StrLen => {
                let s = self.base_ref(a + 2);
                if !tref_isstr(s) {
                    return Err(TraceError::NYIBC);
                }
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NUM),
                    tref_ref(s),
                    IRCALL_STR_LEN,
                ));
                nres = 1;
            }
            Recff::StrByte => {
                // Only the single-index form (j defaults to i): always
                // exactly one result, a number or nil — specialize on
                // the record-time outcome like Next does.
                if nargs > 2 {
                    return Err(TraceError::NYIBC);
                }
                let s = self.base_ref(a + 2);
                if !tref_isstr(s) {
                    return Err(TraceError::NYIBC);
                }
                let i = if nargs >= 2 {
                    let i = self.base_ref(a + 3);
                    if !tref_isnum(i) {
                        return Err(TraceError::NYIBC);
                    }
                    i
                } else {
                    self.cur.ir.knum(1.0)
                };
                let iv = if nargs >= 2 {
                    argv[1]
                } else {
                    LuaValue::number(1.0)
                };
                let outv =
                    LuaValue::from_bits(super::exec::jit_str_byte(argv[0].to_bits(), iv.to_bits()));
                let t_out = Self::value_irt(outv);
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(s),
                    tref_ref(i),
                ));
                let mut out = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_GUARD | t_out),
                    tref_ref(carg),
                    IRCALL_STR_BYTE,
                ));
                if irt_ispri(t_out) {
                    out = tref_pri(t_out);
                }
                res[0] = out;
                nres = 1;
            }
            Recff::StrSub => {
                // string.sub(s, i [, j]) — a missing j records as -1
                // (the same suffix). The result is always a string.
                if !(2..=3).contains(&nargs) {
                    return Err(TraceError::NYIBC);
                }
                let s = self.base_ref(a + 2);
                let i = self.base_ref(a + 3);
                if !tref_isstr(s) || !tref_isnum(i) {
                    return Err(TraceError::NYIBC);
                }
                let j = if nargs >= 3 {
                    let j = self.base_ref(a + 4);
                    if !tref_isnum(j) {
                        return Err(TraceError::NYIBC);
                    }
                    j
                } else {
                    self.cur.ir.knum(-1.0)
                };
                let cargi = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(s),
                    tref_ref(i),
                ));
                let cargj = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(cargi),
                    tref_ref(j),
                ));
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_STR),
                    tref_ref(cargj),
                    IRCALL_STR_SUB,
                ));
                self.rec_gcstep(l);
                nres = 1;
            }
            Recff::StrChar => {
                // Single argument only (multi-arg concatenation NYI).
                if nargs != 1 {
                    return Err(TraceError::NYIBC);
                }
                let c = self.base_ref(a + 2);
                if !tref_isnum(c) {
                    return Err(TraceError::NYIBC);
                }
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_STR),
                    tref_ref(c),
                    IRCALL_STR_CHAR,
                ));
                self.rec_gcstep(l);
                nres = 1;
            }
            Recff::TableInsert => {
                // Push form only: t[#t+1] = v (the builtin stores raw,
                // no metatable check). The ASTORE bounds guard exits on
                // the growth iterations; the interpreter then resizes.
                if nargs != 2 {
                    return Err(TraceError::NYIBC); // Positional insert NYI.
                }
                let tab = self.base_ref(a + 2);
                let val = self.base_ref(a + 3);
                if !tref_istab(tab) {
                    return Err(TraceError::NYIBC);
                }
                let Some(t) = argv[0].as_table() else {
                    return Err(TraceError::NYIBC);
                };
                let lenv = t.as_ref().len();
                // Record only the in-bounds fast path.
                if lenv + 1 >= t.as_ref().asize {
                    return Err(TraceError::NYIBC);
                }
                let len = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NUM),
                    tref_ref(tab),
                    IRCALL_TAB_LEN,
                ));
                let one = self.cur.ir.knum(1.0);
                let k = self
                    .cur
                    .ir
                    .emitir(irtn(IROp::ADD), tref_ref(len), tref_ref(one))?;
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(k),
                    tref_ref(val),
                ));
                self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::ASTORE, IRT_NIL),
                    tref_ref(tab),
                    tref_ref(carg),
                ));
                self.needsnap = true;
                nres = 0;
            }
            Recff::TableRemove => {
                // Pop form only: v = t[#t]; t[#t] = nil; return v.
                if nargs != 1 {
                    return Err(TraceError::NYIBC); // Positional remove NYI.
                }
                let tab = self.base_ref(a + 2);
                if !tref_istab(tab) {
                    return Err(TraceError::NYIBC);
                }
                let Some(t) = argv[0].as_table() else {
                    return Err(TraceError::NYIBC);
                };
                let lenv = t.as_ref().len();
                // Stay on the array fast path (the empty and hash-part
                // boundaries exit to the interpreter).
                if lenv == 0 || lenv >= t.as_ref().asize {
                    return Err(TraceError::NYIBC);
                }
                let vv = t.as_ref().get_int(lenv as i32);
                if vv.is_nil() {
                    return Err(TraceError::NYIBC);
                }
                let len = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NUM),
                    tref_ref(tab),
                    IRCALL_TAB_LEN,
                ));
                let zero = self.cur.ir.knum(0.0);
                self.cur
                    .ir
                    .emitir(irtg(IROp::GT, IRT_NUM), tref_ref(len), tref_ref(zero))?;
                let ty = Self::value_irt(vv);
                let v = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::ALOAD, IRT_GUARD | ty),
                    tref_ref(tab),
                    tref_ref(len),
                ));
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(len),
                    tref_ref(TREF_NIL),
                ));
                self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::ASTORE, IRT_NIL),
                    tref_ref(tab),
                    tref_ref(carg),
                ));
                self.needsnap = true;
                res[0] = v;
                nres = 1;
            }
            Recff::TableConcat => {
                // (t [, sep]) form only; the helper returns nil for an
                // invalid element, failing the STR guard so the
                // interpreter re-runs the call and raises the error.
                if nargs > 2 {
                    return Err(TraceError::NYIBC); // i/j range NYI.
                }
                let tab = self.base_ref(a + 2);
                if !tref_istab(tab) {
                    return Err(TraceError::NYIBC);
                }
                let sep = if nargs >= 2 {
                    let sep = self.base_ref(a + 3);
                    if !tref_isstr(sep) && !tref_isnil(sep) {
                        return Err(TraceError::NYIBC);
                    }
                    sep
                } else {
                    TREF_NIL
                };
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, IRT_NIL),
                    tref_ref(tab),
                    tref_ref(sep),
                ));
                res[0] = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_GUARD | IRT_STR),
                    tref_ref(carg),
                    IRCALL_TAB_CONCAT,
                ));
                self.rec_gcstep(l);
                nres = 1;
            }
        }
        // Write the results; pad/discard per the call's wanted count.
        for i in 0..want as u32 {
            let tr = if i < nres { res[i as usize] } else { TREF_NIL };
            self.set_base(a + i, tr);
        }
        // Clear the stale callee/arg slots above the results.
        for s in (a + want.max(0) as u32)..(a + 2 + nargs) {
            self.set_base(s, 0);
        }
        self.maxslot = a + want as u32;
        Ok(())
    }

    // -- Bit operators (fused num -> int32 -> op -> num semantics) ------------

    /// Guard that a bit-op operand stays in the i32 range (outside it
    /// the interpreter's saturating cast takes over off-trace; NaN fails
    /// the ordered compares too). CSE de-duplicates repeated guards.
    fn bitrange_guard(&mut self, x: TRef, lo: f64, hi: f64) -> Result<(), TraceError> {
        let kmin = self.cur.ir.knum(lo);
        let kmax = self.cur.ir.knum(hi);
        self.cur
            .ir
            .emitir(irtg(IROp::GE, IRT_NUM), tref_ref(x), tref_ref(kmin))?;
        self.cur
            .ir
            .emitir(irtg(IROp::LE, IRT_NUM), tref_ref(x), tref_ref(kmax))?;
        Ok(())
    }

    /// Bit-op results are exact int32s by construction: guards against
    /// the i32 range are redundant for them.
    fn is_bitop_ref(&self, x: TRef) -> bool {
        let r = tref_ref(x);
        r >= REF_BIAS
            && matches!(
                self.cur.ir.ir(r).op(),
                IROp::BAND
                    | IROp::BOR
                    | IROp::BXOR
                    | IROp::BSHL
                    | IROp::BSHR
                    | IROp::BSAR
                    | IROp::BNOT
                    | IROp::BROL
                    | IROp::BROR
                    | IROp::BSWAP
                    | IROp::TOBIT
            )
    }

    /// `bit.*` argument conversion: TOBIT (wrapping num -> int32 with
    /// round-to-nearest). Inside the guarded i32 range this equals the
    /// interpreter's `num2bit`; outside it the guard exits and the
    /// interpreter's wrap takes over.
    fn rec_tobit(&mut self, x: TRef) -> Result<TRef, TraceError> {
        if !tref_isnum(x) {
            return Err(TraceError::NYIBC);
        }
        if self.is_bitop_ref(x) {
            return Ok(x); // Already an exact int32.
        }
        self.bitrange_guard(x, -2147483648.0, 2147483647.0)?;
        Ok(self
            .cur
            .ir
            .emit_ins(IRIns::new(irtn(IROp::TOBIT), tref_ref(x), 0)))
    }

    /// Record BAND/BOR/BXOR/BSHL/BSHR/BSAR/BNOT: both operands must be
    /// numbers in i32 range (guarded); the op itself mirrors the
    /// interpreter's `as i32`/`as u32 & 31` coercions bit for bit.
    fn rec_bitop(&mut self, irop: IROp, x: TRef, y: TRef) -> Result<TRef, TraceError> {
        if !tref_isnum(x) || (y != 0 && !tref_isnum(y)) {
            return Err(TraceError::NYIBC); // Non-number coercion paths.
        }
        if !self.is_bitop_ref(x) {
            self.bitrange_guard(x, -2147483648.0, 2147483647.0)?;
        }
        if y != 0 && !self.is_bitop_ref(y) {
            if matches!(irop, IROp::BSHL | IROp::BSHR | IROp::BSAR) {
                // Shift counts go through `as u32`: negative saturates
                // to 0 in the interpreter, so stay on non-negative.
                self.bitrange_guard(y, 0.0, 2147483647.0)?;
            } else {
                self.bitrange_guard(y, -2147483648.0, 2147483647.0)?;
            }
        }
        // Raw-emitted: the integer fold rules do not apply to the fused
        // form (see fold_step), and CSE happens on re-emission anyway.
        Ok(self
            .cur
            .ir
            .emit_ins(IRIns::new(irtn(irop), tref_ref(x), tref_ref(y))))
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
        // a loop condition guard). The pc must be updated first: the
        // snapshot resumes at the instruction about to be recorded (the
        // stale previous pc could point back at a comparison whose
        // scratch operand slots are no longer covered by maxslot).
        self.pc = pc;
        if self.needsnap {
            self.needsnap = false;
            self.snap_add();
            self.mergesnap = true;
        }

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
                    let (x, y, xn, yn);
                    if tref_isnum(ra) && tref_isnum(rc) {
                        (x, y) = (ra, rc);
                        (xn, yn) = (rav.num(), rcv.num());
                    } else if tref_isstr(ra) && tref_isstr(rc) {
                        // Strings: compare through the helper, then
                        // guard the sign of the result against 0.
                        let carg = self.cur.ir.emit_ins(IRIns::new(
                            irt(IROp::CARG, IRT_NIL),
                            tref_ref(ra),
                            tref_ref(rc),
                        ));
                        x = self.cur.ir.emit_ins(IRIns::new(
                            irt(IROp::CALLL, IRT_NUM),
                            tref_ref(carg),
                            IRCALL_STR_CMP,
                        ));
                        y = self.cur.ir.knum(0.0);
                        xn = f64::from_bits(super::exec::jit_str_cmp(rav.to_bits(), rcv.to_bits()));
                        yn = 0.0;
                    } else {
                        return Err(TraceError::NYIBC); // Mixed/metamethods NYI.
                    }
                    self.comp_prep();
                    let mut irop = IROp::from_u8(op as u8 - BCOp::ISLT as u8 + IROp::LT as u8);
                    if (irop as u8) & 1 != 0 {
                        // ISGE/ISGT are unordered (NaN behavior).
                        irop = IROp::from_u8(irop as u8 ^ 4);
                    }
                    if !super::opt_fold::fold_numcmp(xn, yn, irop) {
                        irop = IROp::from_u8(irop as u8 ^ 5);
                    }
                    self.cur
                        .ir
                        .emitir(irtg(irop, IRT_NUM), tref_ref(x), tref_ref(y))?;
                    self.comp_fixup(pc, ((op as u8) ^ (irop as u8)) & 1 != 0);
                }
            }
            BCOp::ISEQV
            | BCOp::ISNEV
            | BCOp::ISEQS
            | BCOp::ISNES
            | BCOp::ISEQN
            | BCOp::ISNEN
            | BCOp::ISEQP
            | BCOp::ISNEP => {
                // Emit nothing for two non-table constants.
                if !(tref_isk(ra) && tref_isk(rc) && !tref_istab(ra)) {
                    if tref_isnum(ra) && tref_isnum(rc) {
                        // Number equality: guard the outcome.
                        self.comp_prep();
                        let diff = rav.num() != rcv.num();
                        let o = if diff { IROp::NE } else { IROp::EQ };
                        self.cur
                            .ir
                            .emitir(irtg(o, IRT_NUM), tref_ref(ra), tref_ref(rc))?;
                        self.comp_fixup(pc, ((op as u8) & 1 == 1) != diff);
                    } else {
                        self.comp_prep();
                        let diff = self.objcmp(ra, rc, rav, rcv)?;
                        if diff == 1 && tref_istab(ra) {
                            return Err(TraceError::NYIBC); // __eq metamethod NYI.
                        }
                        // lj_record.c: `(op & 1) == !diff` — the snapshot
                        // resumes on the branch the trace does not take.
                        self.comp_fixup(pc, ((op as u8) & 1 == 1) == (diff == 0));
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
                result = if rcv.is_truthy() {
                    TREF_FALSE
                } else {
                    TREF_TRUE
                };
            }
            BCOp::UNM => {
                if !tref_isnum(rc) {
                    return Err(TraceError::NYIBC);
                }
                // op2 stands in for LuaJIT's KSIMD sign-flip constant.
                let signbit = self.cur.ir.knum(-0.0);
                result = self
                    .cur
                    .ir
                    .emitir(irtn(IROp::NEG), tref_ref(rc), tref_ref(signbit))?;
            }
            BCOp::LEN => {
                if tref_isstr(rc) {
                    result = self.cur.ir.emit_ins(IRIns::new(
                        irt(IROp::CALLL, IRT_NUM),
                        tref_ref(rc),
                        IRCALL_STR_LEN,
                    ));
                } else if tref_istab(rc) {
                    // Raw length, mirroring the interpreter's table fast
                    // path (no __len for tables).
                    result = self.cur.ir.emit_ins(IRIns::new(
                        irt(IROp::CALLL, IRT_NUM),
                        tref_ref(rc),
                        IRCALL_TAB_LEN,
                    ));
                } else {
                    return Err(TraceError::NYIBC); // __len NYI.
                }
            }

            // -- Constants -------------------------------------------------------------
            BCOp::KSTR | BCOp::KNUM | BCOp::KPRI | BCOp::KCDATA => result = rc,
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
            BCOp::ADDVN
            | BCOp::SUBVN
            | BCOp::MULVN
            | BCOp::DIVVN
            | BCOp::MODVN
            | BCOp::ADDVV
            | BCOp::SUBVV
            | BCOp::MULVV
            | BCOp::DIVVV
            | BCOp::MODVV => {
                result = self.arith(rb, rc, arith_irop(op))?;
            }
            BCOp::ADDNV | BCOp::SUBNV | BCOp::MULNV | BCOp::DIVNV | BCOp::MODNV => {
                // NV forms: constant op variable — operands swapped.
                result = self.arith(rc, rb, arith_irop(op))?;
            }
            BCOp::POW => result = self.arith(rb, rc, IROp::POW)?,

            // -- Bit operators -----------------------------------------------------
            BCOp::BAND | BCOp::BOR | BCOp::BXOR | BCOp::BSHL | BCOp::BSHR | BCOp::BSAR => {
                let x = self.getslot(l, base, bc_b(ins));
                let y = self.getslot(l, base, bc_c(ins));
                let irop = match op {
                    BCOp::BAND => IROp::BAND,
                    BCOp::BOR => IROp::BOR,
                    BCOp::BXOR => IROp::BXOR,
                    BCOp::BSHL => IROp::BSHL,
                    BCOp::BSHR => IROp::BSHR,
                    _ => IROp::BSAR,
                };
                result = self.rec_bitop(irop, x, y)?;
            }
            BCOp::BNOT => {
                let x = self.getslot(l, base, bc_d(ins));
                result = self.rec_bitop(IROp::BNOT, x, 0)?;
            }

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
                let startins = js.trace[lnk as usize]
                    .as_ref()
                    .ok_or(TraceError::NYIBC)?
                    .startins;
                let fori = (pc as i64 + bc_j(startins)) as usize;
                let ev = self.rec_for(l, base, fori, true)?;
                if let Some(link) = self.loop_jit(lnk, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::JLOOP => {
                let lnk = bc_d(ins);
                let startins = js.trace[lnk as usize]
                    .as_ref()
                    .ok_or(TraceError::NYIBC)?
                    .startins;
                if bc_isret(bc_op(startins)) || bc_op(startins) == BCOp::ITERN {
                    return Err(TraceError::NYIBC); // Patched RET/ITERN loops.
                }
                let ev = self.rec_loop(bc_a(ins));
                if let Some(link) = self.loop_jit(lnk, ev)? {
                    return Ok(Some(link));
                }
            }
            BCOp::JITERL => {
                // rec_loop_jit for iterator loops: replay the original
                // ITERL semantics (the jump lives in the trace startins).
                let lnk = bc_d(ins);
                let startins = js.trace[lnk as usize]
                    .as_ref()
                    .ok_or(TraceError::NYIBC)?
                    .startins;
                let ev = self.rec_iterl(l, base, pc, startins);
                if let Some(link) = self.loop_jit(lnk, ev)? {
                    return Ok(Some(link));
                }
            }

            // -- Iterators ---------------------------------------------------------
            // This VM never specializes ITERN at runtime (ISNEXT is a
            // plain jump, ITERN a generic iterator call), so both forms
            // record identically.
            BCOp::ITERC | BCOp::ITERN => {
                if let Some(link) = self.rec_iterc(l, base, pc, ins)? {
                    return Ok(Some(link));
                }
            }
            BCOp::ISNEXT => {
                if bc_a(ins) < self.maxslot {
                    self.maxslot = bc_a(ins);
                }
            }
            BCOp::ITERL => {
                let ev = self.rec_iterl(l, base, pc, ins);
                if let Some(link) = self.loop_interp(js, pc, ev)? {
                    return Ok(Some(link));
                }
            }

            // -- Upvalues ----------------------------------------------------------
            BCOp::UGET => result = self.rec_uget(l, base, bc_d(ins))?,

            BCOp::VARG => {
                let bp = unsafe { l.stack.as_ptr().add(base) };
                let link = unsafe { (*bp.sub(1)).to_bits() };
                use crate::vm::FRAME_TYPE_MASK;
                use crate::vm::FRAME_VARG;
                if link & FRAME_TYPE_MASK != FRAME_VARG {
                    return Err(TraceError::NYIBC);
                }
                let delta = (link >> 3) as usize;
                let nparams = self.pt.as_ref().numparams as usize;
                let nvarg = (delta - 2).saturating_sub(nparams);
                let dst = bc_a(ins) as u32;
                let want = if bc_b(ins) == 0 {
                    nvarg as u32
                } else {
                    (bc_b(ins) - 1) as u32
                };
                let packed = ((nparams as u32) << 16) | ((dst as u32) << 8) | want;
                let k_link = self.cur.ir.kint64(link);
                let k_packed = self.cur.ir.kint64(packed as u64);
                let c1 = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, irt_type(IRT_NIL)),
                    REF_BIAS, // → base pointer via RBASE register
                    tref_ref(k_link),
                ));
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, irt_type(IRT_NIL)),
                    tref_ref(c1),
                    tref_ref(k_packed),
                ));
                self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NIL),
                    tref_ref(carg),
                    IRCALL_VARG,
                ));
            }

            BCOp::USETV | BCOp::USETS | BCOp::USETN | BCOp::USETP => {
                let fnval = l.stack[base - 2];
                let Some(gf) = fnval.as_func() else {
                    return Err(TraceError::NYIBC);
                };
                let crate::func::GcFunc::Lua(cl) = gf.as_ref() else {
                    return Err(TraceError::NYIBC);
                };
                let Some(&uv) = cl.upvals.get(bc_a(ins) as usize) else {
                    return Err(TraceError::NYIBC);
                };
                let cell = uv.as_ref().value_ptr() as u64;
                self.specialize_curfn(l, base, fnval)?;
                let ptr_k = self.cur.ir.kint64(cell);
                // rb is preloaded as Var for USETV; rc/D is preloaded for
                // USETS (str), USETN (num), USETP (pri).
                let val_ref = match op {
                    BCOp::USETV => rb,
                    _ => rc,
                };
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, irt_type(IRT_NIL)),
                    tref_ref(ptr_k),
                    tref_ref(val_ref),
                ));
                self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_NIL),
                    tref_ref(carg),
                    IRCALL_USET,
                ));
            }

            // -- Table indexing ----------------------------------------------------
            BCOp::TNEW => {
                // Mirror the interpreter: a fresh empty table (the size
                // hint in D is ignored there too).
                result = self
                    .cur
                    .ir
                    .emit_ins(IRIns::new(irt(IROp::TNEW, IRT_TAB), 0, 0));
                self.rec_gcstep(l);
            }
            BCOp::TDUP => {
                let templ_addr = match &pt.as_ref().kgc[bc_d(ins) as usize] {
                    crate::proto::KGc::Table(t) => &**t as *const crate::table::LuaTable,
                    crate::proto::KGc::TableRef(t) => t.as_ref() as *const crate::table::LuaTable,
                    _ => return Err(TraceError::NYIBC),
                };
                let k = self.cur.ir.kint64(templ_addr as u64);
                result = self
                    .cur
                    .ir
                    .emit_ins(IRIns::new(irt(IROp::TDUP, IRT_TAB), tref_ref(k), 0));
                self.rec_gcstep(l);
            }
            BCOp::TGETV | BCOp::TGETS | BCOp::TGETB => {
                let tabv = self.slot_val(l, base, bc_b(ins));
                let (key, keyv) = if op == BCOp::TGETB {
                    let n = bc_c(ins) as f64;
                    (self.cur.ir.knum(n), LuaValue::number(n))
                } else {
                    (rc, rcv)
                };
                result = self.rec_tget(tabv, rb, keyv, key)?;
            }
            BCOp::TSETV | BCOp::TSETS | BCOp::TSETB => {
                let tabv = self.slot_val(l, base, bc_b(ins));
                let tab = self.getslot(l, base, bc_b(ins));
                let (key, keyv) = if op == BCOp::TSETB {
                    let n = bc_c(ins) as f64;
                    (self.cur.ir.knum(n), LuaValue::number(n))
                } else {
                    (rc, rcv)
                };
                let val = self.getslot(l, base, bc_a(ins));
                self.rec_tset(l, tabv, tab, keyv, key, val)?;
            }
            BCOp::GGET | BCOp::GSET => {
                // The environment table of the (specialized) closure is a
                // constant; globals are then plain string-keyed lookups.
                let fnval = l.stack[base - 2];
                self.specialize_curfn(l, base, fnval)?;
                let env = fnval
                    .as_func()
                    .expect("current frame function")
                    .as_ref()
                    .env();
                let tabv = LuaValue::table(env);
                let tab = self.cur.ir.kgc(tabv.to_bits(), IRT_TAB);
                if op == BCOp::GGET {
                    result = self.rec_tget(tabv, tab, rcv, rc)?;
                } else {
                    let val = self.getslot(l, base, bc_a(ins));
                    self.rec_tset(l, tabv, tab, rcv, rc, val)?;
                }
            }

            // -- Calls and returns -------------------------------------------------
            BCOp::CALL => {
                if let Some(link) = self.rec_call(l, base, pc, ins)? {
                    return Ok(Some(link));
                }
            }
            BCOp::CALLT => {
                if let Some(link) = self.rec_callt(l, base, ins)? {
                    return Ok(Some(link));
                }
            }
            BCOp::RET0 => self.rec_ret(l, base, bc_a(ins), 0)?,
            BCOp::RET1 => self.rec_ret(l, base, bc_a(ins), 1)?,
            BCOp::RET => self.rec_ret(l, base, bc_a(ins), bc_d(ins) - 1)?,
            BCOp::FUNCF => {
                // Reached only through the slow call paths (the fast path
                // enters at bc[1]); those shapes are not recorded yet.
                return Err(TraceError::NYIBC);
            }

            BCOp::CAT => {
                // Two-value concatenation via jit_cat helper.
                // rb=first source, rc=second source (both already typed).
                if !(tref_isstr(rb) || tref_isnum(rb)) || !(tref_isstr(rc) || tref_isnum(rc)) {
                    return Err(TraceError::NYIBC);
                }
                let carg = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CARG, irt_type(IRT_NIL)),
                    tref_ref(rb),
                    tref_ref(rc),
                ));
                result = self.cur.ir.emit_ins(IRIns::new(
                    irt(IROp::CALLL, IRT_STR),
                    tref_ref(carg),
                    IRCALL_CAT,
                ));
            }

            // FNEW creates new closures with unstable identities inside
            // hot loops, making compiled traces with closure identity guards
            // useless. Immediately blacklist the parent trace.
            BCOp::FNEW => return Err(TraceError::BLACKL),

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

/// Recordable fast functions (the lj_ffrecord dispatch, keyed by the
/// builtin's function pointer instead of an ffid).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Recff {
    Floor,
    Ceil,
    Sqrt,
    Abs,
    /// The hidden ipairs() iterator (recff_ipairs_aux).
    IpairsAux,
    /// next(t, k) — the pairs() iterator (recff_next).
    Next,
    /// math.min / math.max (variadic compare-select chains).
    MathMin,
    MathMax,
    /// math.fmod (an IRCALL helper).
    Fmod,
    /// bit.* (TOBIT conversions + the fused int32 ops).
    Tobit,
    Bnot,
    Band,
    Bor,
    Bxor,
    Lshift,
    Rshift,
    Arshift,
    Rol,
    Ror,
    Bswap,
    /// string.* (IRCALL helpers; sub/char allocate + GCSTEP).
    StrLen,
    StrByte,
    StrSub,
    StrChar,
    /// table.* (raw array ops + IRCALL helpers).
    TableInsert,
    TableRemove,
    TableConcat,
}

fn recff_lookup(f: crate::func::CFunction) -> Option<Recff> {
    use crate::func::CFunction;
    use crate::stdlib::{base, bit, math, string, table};
    let entries: [(CFunction, Recff); 27] = [
        (math::floor, Recff::Floor),
        (math::ceil, Recff::Ceil),
        (math::sqrt, Recff::Sqrt),
        (math::abs, Recff::Abs),
        (math::math_min, Recff::MathMin),
        (math::math_max, Recff::MathMax),
        (math::math_fmod, Recff::Fmod),
        (base::lib_ipairs_iter, Recff::IpairsAux),
        (base::lib_next, Recff::Next),
        (bit::tobit, Recff::Tobit),
        (bit::bnot, Recff::Bnot),
        (bit::band, Recff::Band),
        (bit::bor, Recff::Bor),
        (bit::bxor, Recff::Bxor),
        (bit::lshift, Recff::Lshift),
        (bit::rshift, Recff::Rshift),
        (bit::arshift, Recff::Arshift),
        (bit::rol, Recff::Rol),
        (bit::ror, Recff::Ror),
        (bit::bswap, Recff::Bswap),
        (string::str_len, Recff::StrLen),
        (string::str_byte, Recff::StrByte),
        (string::str_sub, Recff::StrSub),
        (string::str_char, Recff::StrChar),
        (table::tab_insert, Recff::TableInsert),
        (table::tab_remove, Recff::TableRemove),
        (table::tab_concat, Recff::TableConcat),
    ];
    entries
        .iter()
        .find(|&&(g, _)| std::ptr::fn_addr_eq(f, g))
        .map(|&(_, id)| id)
}

/// Raw value equality, same semantics as the interpreter's `val_eq`.
fn val_eq(a: LuaValue, b: LuaValue) -> bool {
    if a.is_number() && b.is_number() {
        a.num() == b.num()
    } else {
        a.to_bits() == b.to_bits()
    }
}
