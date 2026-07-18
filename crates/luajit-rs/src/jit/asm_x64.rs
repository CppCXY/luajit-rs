//! x86-64 trace assembler: turns the SSA IR of a completed trace into
//! native machine code (the lj_asm.c/lj_asm_x86.h role, first phase).
//!
//! Deviations from LuaJIT's backend, by design:
//! * Forward single-pass code generation with a greedy scan allocator
//!   over xmm0-xmm4 (LuaJIT allocates backwards over all 16 registers).
//! * Every value referenced by any snapshot is stored to its spill slot
//!   in the `env` buffer at definition time. Exits therefore need no
//!   register maps or an ExitState: the Rust-side snapshot restore
//!   (`exec::restore_snapshot`) reads `env` directly, using the portable
//!   executor's layout (slot index = ref - REF_BIAS).
//! * Guards jump to per-snapshot exit stubs that return the snapshot
//!   index (LuaJIT's exit stubs + lj_vm_exit_handler, minus the state
//!   capture).
//! * The loop edge writes the final snapshot back to the Lua stack and
//!   jumps to the head, whose SLOADs re-read and re-check the slots —
//!   exactly the portable executor's semantics (no lj_opt_loop/PHIs yet).
//! * IR the backend cannot handle aborts with NYIIR; the trace then runs
//!   on the portable IR executor instead, so this stays a pure fast path.
//!
//! ABI of the emitted code:
//! `extern "C" fn(base: *mut LuaValue, env: *mut u64) -> u32` returning
//! the exit snapshot index. GPRs used are volatile in both the Win64 and
//! SysV ABIs (rax/rcx/rdx, r10/r11). The FP allocator prefers the
//! volatile xmm0-xmm4 and only then the Win64 callee-saved xmm6-xmm15;
//! the prologue/epilogue save exactly the callee-saved registers the
//! trace touches (usually none for short side traces).

use super::ir::*;
use super::mcode::McodeArea;
use super::record::{IRFPM_CEIL, IRFPM_FLOOR, IRFPM_SQRT, IRFPM_TRUNC};
use super::{GCtrace, SNAP_NORESTORE, TraceError, TraceLink, snap_ref, snap_slot};

/// Allocatable FP registers, in preference order: the always-volatile
/// xmm0-xmm4 first, then xmm6-xmm15 (Win64 callee-saved, spilled only
/// when used). xmm5 is the constant/mask scratch.
const ALLOC_REGS: [u8; 15] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const XMM_SCRATCH: u8 = 5;
/// Register state arrays are indexed by the xmm number.
const NREG: usize = 16;

/// Fixed-role GPRs, volatile in both ABIs.
const RAX: u8 = 0;
const RCX: u8 = 1;
#[cfg(windows)]
const RDX: u8 = 2;
#[cfg(not(windows))]
const RDI: u8 = 7;
#[cfg(not(windows))]
const RSI: u8 = 6;
const RBASE: u8 = 10; // r10 = Lua stack base of the frame.
const RENV: u8 = 11; // r11 = spill/exit value environment.

/// Condition codes (jcc near = 0F 80+cc).
const CC_B: u8 = 0x2;
const CC_AE: u8 = 0x3;
const CC_E: u8 = 0x4;
const CC_NE: u8 = 0x5;
const CC_BE: u8 = 0x6;
const CC_A: u8 = 0x7;
const CC_P: u8 = 0xA;

/// High dword of the smallest GC-tagged value (`LJ_TISNUM << 15`): any
/// value whose high dword is below this is a number (same check as
/// LuaJIT's GC64 asm_sload).
const TISNUM_HI: u32 = 0xFFF9_0000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    None,
    /// IR instruction result.
    Ins(IRRef),
    /// Materialized constant (cache only; re-materialized after eviction).
    Konst(IRRef),
}

/// A loop-carried value: `PHI(lref, rref)` — on the back edge the value
/// of `rref` becomes the next iteration's `lref`.
#[derive(Clone, Copy)]
struct PhiInfo {
    /// Ref of the PHI instruction itself; its env slot doubles as the
    /// scratch buffer for the parallel move on the back edge.
    phi: IRRef,
    lref: IRRef,
    rref: IRRef,
    /// FP value (lives in xmm registers); GC values are carried via env.
    num: bool,
}

/// Assemble a completed trace. On error the caller keeps `mcode = None`
/// and the portable executor runs the trace.
pub fn assemble(tr: &GCtrace) -> Result<McodeArea, TraceError> {
    Asm::new(tr)?.emit()
}

struct Asm<'a> {
    tr: &'a GCtrace,
    code: Vec<u8>,
    /// Current instruction ref and its covering snapshot.
    cur: IRRef,
    snapidx: usize,
    /// Last use site per instruction (index ref-REF_BIAS; 0 = unused).
    last_use: Vec<IRRef>,
    /// Last use site per constant (index REF_BIAS-1-ref).
    klast_use: Vec<IRRef>,
    /// Ref is referenced by a snapshot (or a GC-identity compare): its
    /// value must be in `env` from definition on.
    needs_env: Vec<bool>,
    /// The env slot currently holds the value (spilled or stored-on-def).
    env_valid: Vec<bool>,
    /// Register currently holding the instruction value.
    loc: Vec<Option<u8>>,
    owner: [Owner; NREG],
    /// (position of rel32, target snapshot index).
    fixups: Vec<(usize, usize)>,
    /// Loop-carried values (PHIs at the trace tail).
    phis: Vec<PhiInfo>,
    /// Code offset of the loop re-entry point (right after IR_LOOP).
    loop_pos: Option<usize>,
    /// Register state at IR_LOOP; the back edge restores exactly this.
    s0: [Owner; NREG],
    /// Bitmask of callee-saved xmm registers handed out by the allocator
    /// (Win64: xmm6-xmm15); only these are saved in the prologue.
    used_csr: u16,
}

impl<'a> Asm<'a> {
    #[inline]
    fn iidx(r: IRRef) -> usize {
        (r - REF_BIAS) as usize
    }
    #[inline]
    fn kidx(r: IRRef) -> usize {
        (REF_BIAS - 1 - r) as usize
    }
    #[inline]
    fn env_disp(r: IRRef) -> i32 {
        (Self::iidx(r) * 8) as i32
    }

    /// Scan the IR: reject NYI opcodes, record last-use positions and the
    /// set of refs that must live in `env` for exits.
    fn new(tr: &'a GCtrace) -> Result<Asm<'a>, TraceError> {
        let nins = Self::iidx(tr.ir.nins());
        let nk = (REF_BIAS - tr.ir.nk()) as usize;
        let mut a = Asm {
            tr,
            code: Vec::with_capacity(256),
            cur: 0,
            snapidx: 0,
            last_use: vec![0; nins],
            klast_use: vec![0; nk],
            needs_env: vec![false; nins],
            env_valid: vec![false; nins],
            loc: vec![None; nins],
            owner: [Owner::None; NREG],
            fixups: Vec::new(),
            phis: Vec::new(),
            loop_pos: None,
            s0: [Owner::None; NREG],
            used_csr: 0,
        };
        let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
        for r in REF_FIRST..tr.ir.nins() {
            let ins = tr.ir.ir(r);
            match ins.op() {
                IROp::NOP | IROp::BASE | IROp::LOOP | IROp::SLOAD => {}
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::MIN | IROp::MAX
                | IROp::NEG => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::ABS => a.mark_use(ins.op1 as IRRef, r),
                IROp::FPMATH => {
                    if ins.op2 as u32 != IRFPM_SQRT && !sse41 {
                        return Err(TraceError::NYIIR); // roundsd needs SSE4.1.
                    }
                    a.mark_use(ins.op1 as IRRef, r);
                }
                IROp::LT | IROp::GE | IROp::LE | IROp::GT | IROp::ULT | IROp::UGE
                | IROp::ULE | IROp::UGT => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::EQ | IROp::NE => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                    if !irt_isnum(ins.t()) {
                        // GC-identity compare reads the raw bits from env.
                        for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                            if op >= REF_BIAS {
                                a.needs_env[Self::iidx(op)] = true;
                            }
                        }
                    }
                }
                IROp::PHI => {
                    // Loop-carried: both sides stay live across the back
                    // edge; non-FP values are carried through env.
                    let (lref, rref) = (ins.op1 as IRRef, ins.op2 as IRRef);
                    let inf = tr.ir.nins();
                    a.mark_use(lref, inf);
                    a.mark_use(rref, inf);
                    let num = irt_isnum(ins.t());
                    if !num {
                        if lref >= REF_BIAS {
                            a.needs_env[Self::iidx(lref)] = true;
                        }
                        if rref >= REF_BIAS {
                            a.needs_env[Self::iidx(rref)] = true;
                        }
                    }
                    a.phis.push(PhiInfo { phi: r, lref, rref, num });
                }
                _ => return Err(TraceError::NYIIR), // POW and later-phase IR.
            }
        }
        for sn in &tr.snapmap {
            let rref = snap_ref(*sn);
            if rref >= REF_BIAS {
                a.needs_env[Self::iidx(rref)] = true;
            }
        }
        // The PHI's own env slot is the back-edge scratch buffer: it must
        // not double as a snapshot value.
        for p in &a.phis {
            if a.needs_env[Self::iidx(p.phi)] {
                return Err(TraceError::NYIIR);
            }
        }
        Ok(a)
    }

    // -- Main emission loop -------------------------------------------------

    fn emit(mut self) -> Result<McodeArea, TraceError> {
        // The body is emitted first; the prologue is prepended at the end
        // when the set of used callee-saved registers is known. All
        // branches are relative, so prepending is offset-neutral.
        let head = 0usize;

        let nins = self.tr.ir.nins();
        let mut r = REF_FIRST;
        while r < nins {
            // Same covering-snapshot rule as the portable executor.
            while self.snapidx + 1 < self.tr.snap.len()
                && self.tr.snap[self.snapidx + 1].iref <= r
            {
                self.snapidx += 1;
            }
            self.cur = r;
            let ins = *self.tr.ir.ir(r);
            match ins.op() {
                IROp::NOP | IROp::BASE => {}
                IROp::LOOP => self.asm_loop_head(),
                IROp::PHI => {} // Handled at the back edge.
                IROp::SLOAD => self.asm_sload(&ins)?,
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::MIN | IROp::MAX => {
                    self.asm_arith(&ins)?
                }
                IROp::NEG => {
                    let d = self.into_dst(ins.op1 as IRRef)?;
                    let m = self.fetch_xmm(ins.op2 as IRRef, pin(d))?;
                    self.sse_rr(0x66, 0x57, d, m); // xorpd: flip the sign bit.
                    self.def(d);
                }
                IROp::ABS => {
                    let d = self.into_dst(ins.op1 as IRRef)?;
                    self.mov_r64_imm64(RAX, 0x7fff_ffff_ffff_ffff);
                    self.movq_xmm_gpr(XMM_SCRATCH, RAX);
                    self.sse_rr(0x66, 0x54, d, XMM_SCRATCH); // andpd
                    self.def(d);
                }
                IROp::FPMATH => self.asm_fpmath(&ins)?,
                IROp::LT | IROp::GE | IROp::LE | IROp::GT | IROp::ULT | IROp::UGE
                | IROp::ULE | IROp::UGT => self.asm_comp(&ins)?,
                IROp::EQ | IROp::NE => self.asm_equal(&ins)?,
                _ => unreachable!("op rejected by the NYI scan"),
            }
            r += 1;
        }

        // Tail: loop-optimized traces jump back to IR_LOOP with the PHI
        // values moved into place; legacy loops re-materialize the final
        // snapshot and restart at the head; others leave through the
        // final snapshot.
        let lastsnap = self.tr.snap.len() - 1;
        let looping = self.tr.linktype == TraceLink::Loop && self.tr.link == self.tr.traceno;
        if looping {
            if let Some(lp) = self.loop_pos {
                self.asm_loop_back(lp);
            } else {
                self.tail_restore(lastsnap);
                self.code.push(0xE9); // jmp head
                let rel = head as i64 - (self.code.len() as i64 + 4);
                self.emit_u32(rel as i32 as u32);
            }
        } else {
            self.mov_eax_imm(lastsnap as u32);
            // Falls through into the epilogue.
        }

        // Common epilogue: restore the used callee-saved xmm (Win64),
        // return eax.
        let saves: Vec<u8> = (6..16u8).filter(|&r| self.used_csr & (1 << r) != 0).collect();
        let framesz = (saves.len() * 16) as i32;
        let epilogue = self.code.len();
        #[cfg(windows)]
        {
            for (k, &r) in saves.iter().enumerate() {
                self.movups_spill(r, (k as i32) * 16, false);
            }
            if framesz > 0 {
                self.code.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, n
                self.emit_u32(framesz as u32);
            }
        }
        self.code.push(0xC3);

        // Exit stubs: mov eax, snapidx; jmp ->epilogue.
        let mut stubs = Vec::with_capacity(self.tr.snap.len());
        for i in 0..self.tr.snap.len() {
            stubs.push(self.code.len());
            self.mov_eax_imm(i as u32);
            self.code.push(0xE9);
            let rel = epilogue as i64 - (self.code.len() as i64 + 4);
            self.emit_u32(rel as i32 as u32);
        }
        for (pos, si) in std::mem::take(&mut self.fixups) {
            let rel = (stubs[si] as i64 - (pos as i64 + 4)) as i32;
            self.code[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
        }

        // Prepend the prologue: capture the two arguments in r10/r11 and
        // save the callee-saved xmm registers this trace actually uses.
        let body = std::mem::take(&mut self.code);
        #[cfg(windows)]
        {
            self.mov_rr64(RBASE, RCX);
            self.mov_rr64(RENV, RDX);
            if framesz > 0 {
                self.code.extend_from_slice(&[0x48, 0x81, 0xEC]); // sub rsp, n
                self.emit_u32(framesz as u32);
                for (k, &r) in saves.iter().enumerate() {
                    self.movups_spill(r, (k as i32) * 16, true);
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = (&saves, framesz);
            self.mov_rr64(RBASE, RDI);
            self.mov_rr64(RENV, RSI);
        }
        self.code.extend_from_slice(&body);

        let mut area = McodeArea::alloc(self.code.len()).ok_or(TraceError::MCODEAL)?;
        area.as_mut_slice()[..self.code.len()].copy_from_slice(&self.code);
        if !area.protect_exec() {
            return Err(TraceError::MCODEAL);
        }
        Ok(area)
    }

    // -- Instruction emitters ------------------------------------------------

    /// SLOAD: optional typecheck guard + load from the Lua stack.
    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let disp = (ins.op1 as i32 - 2) * 8;
        let t = ins.t();
        let i = Self::iidx(self.cur);
        if irt_isnum(t) {
            if ins.is_guard() {
                // cmp dword [BASE+disp+4], LJ_TISNUM<<15; jae ->exit
                self.cmp_mem32_imm32(RBASE, disp + 4, TISNUM_HI);
                self.guard(CC_AE);
            }
            if self.last_use[i] != 0 || self.needs_env[i] {
                let d = self.alloc(0)?;
                self.movsd_load(d, RBASE, disp);
                self.def(d);
            }
            return Ok(());
        }
        // Non-number slots never live in xmm; keep the raw bits in env.
        if self.needs_env[i] {
            self.mov_r64_mem(RAX, RBASE, disp);
            self.mov_mem_r64(RENV, Self::env_disp(self.cur), RAX);
            self.env_valid[i] = true;
        }
        if !ins.is_guard() {
            return Ok(());
        }
        let ty = irt_type(t);
        if ty == IRT_NIL {
            // cmp qword [BASE+disp], -1; jne ->exit
            self.cmp_mem64_imm8(RBASE, disp, -1);
            self.guard(CC_NE);
        } else if ty <= IRT_TRUE {
            // Primitives: the high dword is (itype<<15)|0x7fff.
            let itype = !(ty as u32);
            self.cmp_mem32_imm32(RBASE, disp + 4, (itype << 15) | 0x7fff);
            self.guard(CC_NE);
        } else {
            // GC types: mov rax,[slot]; sar rax,47; cmp eax, itype; jne.
            self.mov_r64_mem(RAX, RBASE, disp);
            self.sar_r64_imm(RAX, 47);
            self.cmp_r32_imm8(RAX, !(ty as u32) as i8);
            self.guard(CC_NE);
        }
        Ok(())
    }

    fn asm_arith(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        let (mut a, mut b) = (ins.op1 as IRRef, ins.op2 as IRRef);
        // Reuse a dying register operand for commutative ops. MIN/MAX are
        // not symmetric (minsd picks the second operand on ties/NaN).
        if matches!(op, IROp::ADD | IROp::MUL)
            && !self.dying(a)
            && self.dying(b)
            && self.reg_of(b).is_some()
        {
            std::mem::swap(&mut a, &mut b);
        }
        let xo = match op {
            IROp::ADD => 0x58,
            IROp::MUL => 0x59,
            IROp::SUB => 0x5C,
            IROp::MIN => 0x5D,
            IROp::DIV => 0x5E,
            IROp::MAX => 0x5F,
            _ => unreachable!(),
        };
        let d = self.into_dst(a)?;
        let rhs = if b == a { d } else { self.fetch_xmm(b, pin(d))? };
        self.sse_rr(0xF2, xo, d, rhs);
        self.def(d);
        Ok(())
    }

    fn asm_fpmath(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let src = self.fetch_xmm(ins.op1 as IRRef, 0)?;
        let d = if self.dying(ins.op1 as IRRef) {
            self.steal_quiet(src);
            src
        } else {
            self.alloc(pin(src))?
        };
        match ins.op2 as u32 {
            IRFPM_SQRT => self.sse_rr(0xF2, 0x51, d, src), // sqrtsd
            fpm => {
                // roundsd d, src, 9/10/11 (floor/ceil/trunc, inexact
                // suppressed), like LuaJIT's asm_fpmath.
                let imm = match fpm {
                    IRFPM_FLOOR => 0x9,
                    IRFPM_CEIL => 0xA,
                    IRFPM_TRUNC => 0xB,
                    _ => unreachable!("bad FPMATH literal"),
                };
                self.roundsd(d, src, imm);
            }
        }
        self.def(d);
        Ok(())
    }

    /// Ordered/unordered FP comparison guards. The operand order and the
    /// jcc are chosen so a NaN always takes the correct side (see
    /// fold_numcmp: U* variants succeed on unordered).
    fn asm_comp(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(irt_isnum(ins.t()) && ins.is_guard());
        let (x, y) = (ins.op1 as IRRef, ins.op2 as IRRef);
        let (fst, snd, cc) = match ins.op() {
            IROp::LT => (y, x, CC_BE),
            IROp::GE => (x, y, CC_B),
            IROp::LE => (y, x, CC_B),
            IROp::GT => (x, y, CC_BE),
            IROp::ULT => (x, y, CC_AE),
            IROp::UGE => (y, x, CC_A),
            IROp::ULE => (x, y, CC_A),
            IROp::UGT => (y, x, CC_AE),
            _ => unreachable!(),
        };
        let f = self.fetch_xmm(fst, 0)?;
        let s = if snd == fst { f } else { self.fetch_xmm(snd, pin(f))? };
        self.sse_rr(0x66, 0x2E, f, s); // ucomisd
        self.guard(cc);
        Ok(())
    }

    fn asm_equal(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(ins.is_guard());
        let eq = ins.op() == IROp::EQ;
        if irt_isnum(ins.t()) {
            let f = self.fetch_xmm(ins.op1 as IRRef, 0)?;
            let s = if ins.op2 == ins.op1 {
                f
            } else {
                self.fetch_xmm(ins.op2 as IRRef, pin(f))?
            };
            self.sse_rr(0x66, 0x2E, f, s); // ucomisd
            if eq {
                // Fail on NaN (PF) or inequality (NZ).
                self.guard(CC_P);
                self.guard(CC_NE);
            } else {
                // Fail only on ordered equality: jp skips the exit.
                self.code.extend_from_slice(&[0x7A, 0x06]); // jp +6
                self.guard(CC_E);
            }
        } else {
            // GC objects/primitives: identity on the value bits.
            self.gpr_load_ref(RAX, ins.op1 as IRRef);
            self.gpr_load_ref(RCX, ins.op2 as IRRef);
            self.cmp_rr64(RAX, RCX);
            self.guard(if eq { CC_NE } else { CC_E });
        }
        Ok(())
    }

    /// The loop edge: write the final snapshot back to the Lua stack
    /// (asm_tail_link's stack sync in this VM's portable semantics).
    /// Only used for traces without IR_LOOP.
    fn tail_restore(&mut self, snapidx: usize) {
        let snap = &self.tr.snap[snapidx];
        let ofs = snap.mapofs as usize;
        for &sn in &self.tr.snapmap[ofs..ofs + snap.nent as usize] {
            if sn & SNAP_NORESTORE != 0 {
                continue;
            }
            let disp = (snap_slot(sn) as i32 - 2) * 8;
            let rref = snap_ref(sn);
            if rref >= REF_BIAS {
                if let Some(rg) = self.loc[Self::iidx(rref)] {
                    self.movsd_store(RBASE, disp, rg);
                } else {
                    debug_assert!(self.env_valid[Self::iidx(rref)]);
                    self.mov_r64_mem(RAX, RENV, Self::env_disp(rref));
                    self.mov_mem_r64(RBASE, disp, RAX);
                }
            } else {
                self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, rref));
                self.mov_mem_r64(RBASE, disp, RAX);
            }
        }
    }

    /// IR_LOOP: park every live value in env (their registers stay
    /// valid), then snapshot the register state — the back edge restores
    /// exactly this state, so the loop body can be entered from both the
    /// pre-roll fall-through and the back-edge jump.
    fn asm_loop_head(&mut self) {
        for rg in 0..NREG as u8 {
            let dead = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)] <= self.cur,
                Owner::Konst(k) => self.klast_use[Self::kidx(k)] <= self.cur,
                Owner::None => false,
            };
            if dead {
                self.steal_quiet(rg);
            }
        }
        for rg in 0..NREG as u8 {
            if let Owner::Ins(o) = self.owner[rg as usize] {
                let i = Self::iidx(o);
                if !self.env_valid[i] {
                    self.movsd_store(RENV, Self::env_disp(o), rg);
                    self.env_valid[i] = true;
                }
            }
        }
        self.s0 = self.owner;
        self.loop_pos = Some(self.code.len());
    }

    /// The back edge of a loop-optimized trace: carry the PHI values
    /// (rref -> lref home), restore the IR_LOOP register state and jump
    /// back. env[lref] is always refreshed, since body code and exits
    /// read the carried value through it.
    fn asm_loop_back(&mut self, loop_pos: usize) {
        let s0 = self.s0;
        let phis = std::mem::take(&mut self.phis);
        // The lref home registers (the parallel-move destinations).
        let homes: Vec<Option<u8>> = phis
            .iter()
            .map(|p| (0..NREG as u8).find(|&rg| s0[rg as usize] == Owner::Ins(p.lref)))
            .collect();
        let dstset: u16 = homes.iter().flatten().fold(0, |m, &rg| m | pin(rg));
        // Direct moves are safe iff no source lives in a destination
        // register and no source env slot is another PHI's lref slot.
        let direct = phis.iter().all(|p| {
            if p.rref < REF_BIAS {
                return true; // Constants are re-materialized.
            }
            if let Some(rg) = self.reg_of(p.rref) {
                if p.num {
                    return dstset & pin(rg) == 0;
                }
            }
            !phis.iter().any(|q| q.lref == p.rref)
        });
        if direct {
            for (p, home) in phis.iter().zip(homes.iter()) {
                self.phi_move(p, *home);
            }
        } else {
            // Buffered parallel move: read all right values into the
            // PHIs' own env slots first, then land them in their homes.
            for p in &phis {
                if p.rref >= REF_BIAS {
                    if p.num && let Some(rg) = self.reg_of(p.rref) {
                        self.movsd_store(RENV, Self::env_disp(p.phi), rg);
                    } else {
                        debug_assert!(self.env_valid[Self::iidx(p.rref)]);
                        self.mov_r64_mem(RAX, RENV, Self::env_disp(p.rref));
                        self.mov_mem_r64(RENV, Self::env_disp(p.phi), RAX);
                    }
                } else {
                    self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, p.rref));
                    self.mov_mem_r64(RENV, Self::env_disp(p.phi), RAX);
                }
            }
            for (p, home) in phis.iter().zip(homes.iter()) {
                if p.num && let Some(rg) = *home {
                    self.movsd_load(rg, RENV, Self::env_disp(p.phi));
                    self.movsd_store(RENV, Self::env_disp(p.lref), rg);
                } else {
                    self.mov_r64_mem(RAX, RENV, Self::env_disp(p.phi));
                    self.mov_mem_r64(RENV, Self::env_disp(p.lref), RAX);
                }
            }
        }
        // Restore the remaining IR_LOOP register state: invariants are
        // reloaded from env, constants re-materialized.
        for rg in 0..NREG as u8 {
            let so = s0[rg as usize];
            if so == Owner::None || so == self.owner[rg as usize] {
                continue;
            }
            if phis.iter().any(|p| so == Owner::Ins(p.lref)) {
                continue; // PHI homes were just written.
            }
            match so {
                Owner::Ins(x) => {
                    debug_assert!(self.env_valid[Self::iidx(x)]);
                    self.movsd_load(rg, RENV, Self::env_disp(x));
                }
                Owner::Konst(k) => {
                    self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, k));
                    self.movq_xmm_gpr(rg, RAX);
                }
                Owner::None => unreachable!(),
            }
        }
        self.phis = phis;
        self.code.push(0xE9); // jmp ->IR_LOOP
        let rel = loop_pos as i64 - (self.code.len() as i64 + 4);
        self.emit_u32(rel as i32 as u32);
    }

    /// One direct PHI move: load the right value into the lref home
    /// register (if any) and refresh env[lref].
    fn phi_move(&mut self, p: &PhiInfo, home: Option<u8>) {
        if p.num && let Some(rg) = home {
            if p.rref >= REF_BIAS {
                if let Some(src) = self.reg_of(p.rref) {
                    if src != rg {
                        self.movsd_rr(rg, src);
                    }
                } else {
                    debug_assert!(self.env_valid[Self::iidx(p.rref)]);
                    self.movsd_load(rg, RENV, Self::env_disp(p.rref));
                }
            } else {
                self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, p.rref));
                self.movq_xmm_gpr(rg, RAX);
            }
            self.movsd_store(RENV, Self::env_disp(p.lref), rg);
        } else if p.rref >= REF_BIAS {
            if p.num && let Some(src) = self.reg_of(p.rref) {
                self.movsd_store(RENV, Self::env_disp(p.lref), src);
            } else {
                debug_assert!(self.env_valid[Self::iidx(p.rref)]);
                self.mov_r64_mem(RAX, RENV, Self::env_disp(p.rref));
                self.mov_mem_r64(RENV, Self::env_disp(p.lref), RAX);
            }
        } else {
            self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, p.rref));
            self.mov_mem_r64(RENV, Self::env_disp(p.lref), RAX);
        }
    }

    // -- Register allocation ---------------------------------------------------

    fn last_use_of(&self, r: IRRef) -> IRRef {
        if r >= REF_BIAS { self.last_use[Self::iidx(r)] } else { self.klast_use[Self::kidx(r)] }
    }

    fn mark_use(&mut self, opr: IRRef, at: IRRef) {
        if opr >= REF_BIAS {
            self.last_use[Self::iidx(opr)] = at;
        } else {
            self.klast_use[Self::kidx(opr)] = at;
        }
    }

    /// The operand's last use is the current instruction.
    fn dying(&self, r: IRRef) -> bool {
        self.last_use_of(r) <= self.cur
    }

    fn reg_of(&self, r: IRRef) -> Option<u8> {
        if r >= REF_BIAS {
            self.loc[Self::iidx(r)]
        } else {
            (0..NREG as u8).find(|&i| self.owner[i as usize] == Owner::Konst(r))
        }
    }

    /// Drop a register's ownership without spilling (dying operand reuse).
    fn steal_quiet(&mut self, rg: u8) {
        if let Owner::Ins(o) = self.owner[rg as usize] {
            self.loc[Self::iidx(o)] = None;
        }
        self.owner[rg as usize] = Owner::None;
    }

    /// Allocate a register, evicting the least useful value if needed.
    /// `pinned` is a bitmask of registers that must not be touched.
    /// Volatile registers are preferred; handing out a callee-saved one
    /// is recorded for the prologue/epilogue.
    fn alloc(&mut self, pinned: u16) -> Result<u8, TraceError> {
        let rg = self.alloc_pick(pinned)?;
        if rg >= 6 {
            self.used_csr |= 1 << rg;
        }
        Ok(rg)
    }

    fn alloc_pick(&mut self, pinned: u16) -> Result<u8, TraceError> {
        // Free register first.
        for &rg in ALLOC_REGS.iter() {
            if pinned & pin(rg) == 0 && self.owner[rg as usize] == Owner::None {
                return Ok(rg);
            }
        }
        // Then any dead value (no further uses).
        for &rg in ALLOC_REGS.iter() {
            if pinned & pin(rg) != 0 {
                continue;
            }
            let dead = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)] < self.cur,
                Owner::Konst(o) => self.klast_use[Self::kidx(o)] < self.cur,
                Owner::None => unreachable!(),
            };
            if dead {
                self.steal_quiet(rg);
                return Ok(rg);
            }
        }
        // Evict the value with the farthest next use.
        let mut best: Option<(u8, IRRef)> = None;
        for &rg in ALLOC_REGS.iter() {
            if pinned & pin(rg) != 0 {
                continue;
            }
            let lu = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)],
                Owner::Konst(o) => self.klast_use[Self::kidx(o)],
                Owner::None => unreachable!(),
            };
            if best.is_none_or(|(_, b)| lu > b) {
                best = Some((rg, lu));
            }
        }
        let Some((rg, _)) = best else {
            return Err(TraceError::BADRA); // Everything pinned: cannot happen.
        };
        if let Owner::Ins(o) = self.owner[rg as usize] {
            let i = Self::iidx(o);
            if !self.env_valid[i] {
                self.movsd_store(RENV, Self::env_disp(o), rg);
                self.env_valid[i] = true;
            }
        }
        self.steal_quiet(rg);
        Ok(rg)
    }

    /// Bring an operand (instruction value or constant) into an xmm reg.
    fn fetch_xmm(&mut self, r: IRRef, pinned: u16) -> Result<u8, TraceError> {
        if let Some(rg) = self.reg_of(r) {
            return Ok(rg);
        }
        let rg = self.alloc(pinned)?;
        if r >= REF_BIAS {
            let i = Self::iidx(r);
            debug_assert!(self.env_valid[i], "live value neither in reg nor env");
            self.movsd_load(rg, RENV, Self::env_disp(r));
            self.owner[rg as usize] = Owner::Ins(r);
            self.loc[i] = Some(rg);
        } else {
            self.mov_r64_imm64(RAX, super::exec::const_bits(&self.tr.ir, r));
            self.movq_xmm_gpr(rg, RAX);
            self.owner[rg as usize] = Owner::Konst(r);
        }
        Ok(rg)
    }

    /// Fetch op1 as the (destroyed) destination of a two-address SSE op.
    fn into_dst(&mut self, a: IRRef) -> Result<u8, TraceError> {
        let r1 = self.fetch_xmm(a, 0)?;
        if self.dying(a) {
            self.steal_quiet(r1);
            Ok(r1)
        } else {
            let d = self.alloc(pin(r1))?;
            self.movsd_rr(d, r1);
            Ok(d)
        }
    }

    /// Bind the current instruction's result and store it to env if any
    /// snapshot needs it.
    fn def(&mut self, d: u8) {
        let i = Self::iidx(self.cur);
        self.owner[d as usize] = Owner::Ins(self.cur);
        self.loc[i] = Some(d);
        if self.needs_env[i] {
            self.movsd_store(RENV, Self::env_disp(self.cur), d);
            self.env_valid[i] = true;
        }
    }

    /// Raw 64-bit value of an operand into a GPR (GC-identity compares).
    fn gpr_load_ref(&mut self, gpr: u8, r: IRRef) {
        if r >= REF_BIAS {
            debug_assert!(self.env_valid[Self::iidx(r)]);
            self.mov_r64_mem(gpr, RENV, Self::env_disp(r));
        } else {
            self.mov_r64_imm64(gpr, super::exec::const_bits(&self.tr.ir, r));
        }
    }

    // -- Encoding primitives -----------------------------------------------

    #[inline]
    fn emit_u32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn modrm(&mut self, md: u8, reg: u8, rm: u8) {
        self.code.push((md << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    /// ModRM + displacement for [base + disp]. Bases are r10/r11 only,
    /// so no SIB byte is ever needed.
    fn mem(&mut self, reg: u8, base: u8, disp: i32) {
        let rm = base & 7;
        debug_assert!(rm != 4, "rsp/r12 base needs a SIB byte");
        if disp == 0 && rm != 5 {
            self.modrm(0, reg, rm);
        } else if (-128..=127).contains(&disp) {
            self.modrm(1, reg, rm);
            self.code.push(disp as u8);
        } else {
            self.modrm(2, reg, rm);
            self.emit_u32(disp as u32);
        }
    }

    /// SSE op with a memory operand: [pfx] [REX] 0F op /r.
    fn sse_mem(&mut self, pfx: u8, op: u8, reg: u8, base: u8, disp: i32) {
        if pfx != 0 {
            self.code.push(pfx);
        }
        let rex = 0x40 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x0F);
        self.code.push(op);
        self.mem(reg, base, disp);
    }

    /// SSE op, register-direct.
    fn sse_rr(&mut self, pfx: u8, op: u8, reg: u8, rm: u8) {
        if pfx != 0 {
            self.code.push(pfx);
        }
        let rex = 0x40 | (((reg >> 3) & 1) << 2) | ((rm >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x0F);
        self.code.push(op);
        self.modrm(3, reg, rm);
    }

    fn movsd_load(&mut self, dst: u8, base: u8, disp: i32) {
        self.sse_mem(0xF2, 0x10, dst, base, disp);
    }
    fn movsd_store(&mut self, base: u8, disp: i32, src: u8) {
        self.sse_mem(0xF2, 0x11, src, base, disp);
    }
    fn movsd_rr(&mut self, dst: u8, src: u8) {
        self.sse_rr(0xF2, 0x10, dst, src);
    }

    /// roundsd dst, src, imm (66 0F 3A 0B /r ib, SSE4.1).
    fn roundsd(&mut self, dst: u8, src: u8, imm: u8) {
        self.code.push(0x66);
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.extend_from_slice(&[0x0F, 0x3A, 0x0B]);
        self.modrm(3, dst, src);
        self.code.push(imm);
    }

    /// movups xmm, [rsp+disp] / movups [rsp+disp], xmm — the Win64
    /// callee-saved xmm spill area (rsp base needs a SIB byte).
    #[cfg(windows)]
    fn movups_spill(&mut self, xmm: u8, disp: i32, store: bool) {
        if xmm >= 8 {
            self.code.push(0x44); // REX.R
        }
        self.code.push(0x0F);
        self.code.push(if store { 0x11 } else { 0x10 });
        if disp == 0 {
            self.modrm(0, xmm, 4);
        } else if (-128..=127).contains(&disp) {
            self.modrm(1, xmm, 4);
            self.code.push(0x24); // SIB: base rsp, no index.
            self.code.push(disp as u8);
            return;
        } else {
            self.modrm(2, xmm, 4);
            self.code.push(0x24);
            self.emit_u32(disp as u32);
            return;
        }
        self.code.push(0x24);
    }

    /// movq xmm, r64 (66 REX.W 0F 6E /r).
    fn movq_xmm_gpr(&mut self, dst: u8, src: u8) {
        self.code.push(0x66);
        self.code.push(0x48 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1));
        self.code.push(0x0F);
        self.code.push(0x6E);
        self.modrm(3, dst, src);
    }

    fn mov_rr64(&mut self, dst: u8, src: u8) {
        self.code.push(0x48 | (((src >> 3) & 1) << 2) | ((dst >> 3) & 1));
        self.code.push(0x89);
        self.modrm(3, src, dst);
    }
    fn mov_r64_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.code.push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
        self.code.push(0x8B);
        self.mem(reg, base, disp);
    }
    fn mov_mem_r64(&mut self, base: u8, disp: i32, reg: u8) {
        self.code.push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
        self.code.push(0x89);
        self.mem(reg, base, disp);
    }
    fn mov_r64_imm64(&mut self, reg: u8, imm: u64) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0xB8 | (reg & 7));
        self.code.extend_from_slice(&imm.to_le_bytes());
    }
    fn mov_eax_imm(&mut self, v: u32) {
        self.code.push(0xB8);
        self.emit_u32(v);
    }
    fn sar_r64_imm(&mut self, reg: u8, sh: u8) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0xC1);
        self.modrm(3, 7, reg);
        self.code.push(sh);
    }
    fn cmp_rr64(&mut self, a: u8, b: u8) {
        self.code.push(0x48 | (((b >> 3) & 1) << 2) | ((a >> 3) & 1));
        self.code.push(0x39);
        self.modrm(3, b, a);
    }
    fn cmp_r32_imm8(&mut self, reg: u8, imm: i8) {
        if reg >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0x83);
        self.modrm(3, 7, reg);
        self.code.push(imm as u8);
    }
    fn cmp_mem32_imm32(&mut self, base: u8, disp: i32, imm: u32) {
        if base >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0x81);
        self.mem(7, base, disp);
        self.emit_u32(imm);
    }
    fn cmp_mem64_imm8(&mut self, base: u8, disp: i32, imm: i8) {
        self.code.push(0x48 | ((base >> 3) & 1));
        self.code.push(0x83);
        self.mem(7, base, disp);
        self.code.push(imm as u8);
    }

    /// Emit a guard branch to the covering snapshot's exit stub.
    fn guard(&mut self, cc: u8) {
        self.code.push(0x0F);
        self.code.push(0x80 | cc);
        self.fixups.push((self.code.len(), self.snapidx));
        self.emit_u32(0);
    }
}

#[inline]
fn pin(rg: u8) -> u16 {
    1u16 << rg
}
