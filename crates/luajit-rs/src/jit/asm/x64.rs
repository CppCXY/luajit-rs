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
//! the exit snapshot index. Every trace sets up the same outer frame
//! (Win64: 160 bytes with all of xmm6-xmm15 saved) so that linked traces
//! can `jmp` between each other's *inner* entries while staying inside
//! the frame of whichever trace was entered from Rust — the machine-code
//! equivalent of LuaJIT's trace linking. Exit stubs reserve a patchable
//! tail: once a side trace is compiled, `patch_exit` retargets them
//! straight into it (lj_asm_patchexit).

use super::super::ir::*;
use super::super::mcode::McodeArea;
use super::super::record::{IRFPM_CEIL, IRFPM_FLOOR, IRFPM_SQRT, IRFPM_TRUNC, IRSLOAD_PARENT};
use super::super::{GCtrace, SNAP_NORESTORE, TraceError, TraceLink, snap_ref, snap_slot};

/// Allocatable FP registers: xmm0-xmm4 and xmm6-xmm15 (callee-saved on
/// Win64; the uniform outer frame always saves them). xmm5 is the
/// constant/mask/cycle scratch.
const ALLOC_REGS: [u8; 15] = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const XMM_SCRATCH: u8 = 5;
/// Register state arrays are indexed by the xmm number.
const NREG: usize = 16;
/// Outer frame size: 10 saved xmm registers (Win64).
const FRAME_SIZE: u32 = 160;

/// Fixed-role GPRs, volatile in both ABIs.
const RAX: u8 = 0;
const RCX: u8 = 1;
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

/// Assemble a completed trace. `link` is the absolute address of the
/// linked trace's inner entry for TRLINK_ROOT tails (root already
/// compiled). On error the caller keeps `mcode = None` and the portable
/// executor runs the trace. Returns the code area, the inner entry
/// offset and the patchable exit-stub tail offsets.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn assemble(
    tr: &GCtrace,
    link: Option<*const u8>,
) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
    if tr.linktype == TraceLink::Downrec {
        return Err(TraceError::NYIIR); // Down-recursion tails NYI.
    }
    Asm::new(tr, link)?.emit()
}

/// `lj_asm_patchexit`: retarget every exit stub of `exitno` to jump
/// straight to `target` (a side trace's inner entry). The stub's flush
/// section stays; only the `mov eax; jmp epilogue` tail is rewritten to
/// `movabs rax, target; jmp rax`.
pub fn patch_exit(area: &mut McodeArea, stub_tails: &[(u32, u32)], exitno: u32, target: *const u8) {
    if !area.protect_rw() {
        return;
    }
    let code = area.as_mut_slice();
    for &(si, ofs) in stub_tails {
        if si == exitno {
            let p = ofs as usize;
            code[p] = 0x48;
            code[p + 1] = 0xB8; // movabs rax, target
            code[p + 2..p + 10].copy_from_slice(&(target as u64).to_le_bytes());
            code[p + 10] = 0xFF;
            code[p + 11] = 0xE0; // jmp rax
        }
    }
    area.protect_exec();
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
    /// Ref must be in `env` from definition on: non-FP values (GC bits
    /// never live in xmm) referenced by snapshots, GC-identity compares
    /// or env-carried PHIs. FP snapshot values are flushed to env by the
    /// exit stubs instead (the ExitState equivalent).
    needs_env: Vec<bool>,
    /// The env slot currently holds the value (spilled or stored-on-def).
    env_valid: Vec<bool>,
    /// Register currently holding the instruction value.
    loc: Vec<Option<u8>>,
    owner: [Owner; NREG],
    /// (position of rel32, target exit stub index).
    fixups: Vec<(usize, usize)>,
    /// Per-guard exit stubs: register flushes + snapshot number.
    stubs: Vec<Stub>,
    /// Loop-carried values (PHIs at the trace tail).
    phis: Vec<PhiInfo>,
    /// Code offset of the loop re-entry point (right after IR_LOOP).
    loop_pos: Option<usize>,
    /// Register state at IR_LOOP; the back edge restores exactly this.
    s0: [Owner; NREG],
    /// Absolute inner-entry address of the linked trace (Root tails).
    link: Option<*const u8>,
    /// Patchable stub tail offsets: (snapshot index, code offset).
    stub_tails: Vec<(u32, u32)>,
}

/// One exit: flush the snapshot values still held in registers at the
/// guard into their env slots, then leave with the snapshot number. This
/// replaces LuaJIT's ExitState + per-snapshot RegSP maps: after the
/// flush, the Rust-side `restore_snapshot` finds everything in env.
struct Stub {
    snapidx: usize,
    flush: Vec<(u8, IRRef)>,
    /// GC-debt exit (IR_GCSTEP): flagged in the exit code and never
    /// patched over by side traces.
    gc: bool,
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
    fn new(tr: &'a GCtrace, link: Option<*const u8>) -> Result<Asm<'a>, TraceError> {
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
            stubs: Vec::new(),
            phis: Vec::new(),
            loop_pos: None,
            s0: [Owner::None; NREG],
            link,
            stub_tails: Vec::new(),
        };
        #[cfg(target_arch = "x86_64")]
        let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
        #[cfg(not(target_arch = "x86_64"))]
        let sse41 = false;
        for r in REF_FIRST..tr.ir.nins() {
            let ins = tr.ir.ir(r);
            match ins.op() {
                IROp::NOP | IROp::BASE | IROp::LOOP | IROp::SLOAD => {}
                IROp::ULOAD => {} // op1 is a KINT64 address constant.
                IROp::FLOAD | IROp::HLOAD | IROp::CARG => {
                    // Helper-call arguments are read from env as raw bits.
                    for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                        if op >= REF_BIAS {
                            a.needs_env[Self::iidx(op)] = true;
                        }
                    }
                }
                IROp::HSTORE => {
                    // op2 is the CARG tuple (its operands are marked by
                    // the CARG arm); op1 is the table argument.
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                }
                IROp::CALLL => {
                    // Arity >= 2 arguments are marked by the CARG arm;
                    // single-argument calls take op1 directly.
                    if super::super::record::ircall_arity(ins.op2 as u32) == 1
                        && ins.op1 as IRRef >= REF_BIAS
                    {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                }
                IROp::TNEW | IROp::TDUP => {} // Literal / constant operands.
                IROp::ALOAD => {
                    // Table via env/GPR, key via xmm.
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::ASTORE => {
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                    // Key/value are marked (and env-pinned) by the CARG arm.
                }
                IROp::GCSTEP => {} // Both operands are address constants.
                IROp::BAND
                | IROp::BOR
                | IROp::BXOR
                | IROp::BSHL
                | IROp::BSHR
                | IROp::BSAR
                | IROp::BROL
                | IROp::BROR
                | IROp::BNOT => {
                    a.mark_use(ins.op1 as IRRef, r);
                    if ins.op2 != 0 {
                        a.mark_use(ins.op2 as IRRef, r);
                    }
                }
                IROp::ADD
                | IROp::SUB
                | IROp::MUL
                | IROp::DIV
                | IROp::MIN
                | IROp::MAX
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
                IROp::LT
                | IROp::GE
                | IROp::LE
                | IROp::GT
                | IROp::ULT
                | IROp::UGE
                | IROp::ULE
                | IROp::UGT => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::POW => {
                    // Helper call: both operands are passed as env bits.
                    for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                        a.mark_use(op, r);
                        if op >= REF_BIAS {
                            a.needs_env[Self::iidx(op)] = true;
                        }
                    }
                }
                IROp::TOBIT => a.mark_use(ins.op1 as IRRef, r),
                IROp::BSWAP => a.mark_use(ins.op1 as IRRef, r),
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
                    a.phis.push(PhiInfo {
                        phi: r,
                        lref,
                        rref,
                        num,
                    });
                }
                _ => return Err(TraceError::NYIIR), // POW and later-phase IR.
            }
        }
        // Snapshot references: a value must stay recoverable while any
        // guard covered by the snapshot can exit, i.e. until the next
        // snapshot begins. FP values are kept alive in registers (the
        // exit stub flushes them); non-FP values must live in env. Note:
        // NORESTORE entries are never written back to the Lua stack, but
        // the env hand-over to side traces still reads them.
        for (i, snap) in tr.snap.iter().enumerate() {
            let live_until = if i + 1 < tr.snap.len() {
                tr.snap[i + 1].iref
            } else {
                tr.ir.nins()
            };
            let ofs = snap.mapofs as usize;
            for sn in &tr.snapmap[ofs..ofs + snap.nent as usize] {
                let rref = snap_ref(*sn);
                if rref >= REF_BIAS {
                    a.mark_use(rref, live_until);
                    if !irt_isnum(tr.ir.ir(rref).t()) {
                        a.needs_env[Self::iidx(rref)] = true;
                    }
                }
            }
        }
        // Side traces: the inherited SLOADs are pre-filled env inputs
        // (the executor copies them from the parent's env slots), so no
        // load code is emitted for them at all.
        for &(own, _) in &tr.parentmap {
            a.env_valid[Self::iidx(own as IRRef)] = true;
        }
        // The PHI's own env slot is the back-edge scratch buffer: it must
        // not double as a snapshot value.
        for p in &a.phis {
            if a.last_use[Self::iidx(p.phi)] != 0 {
                return Err(TraceError::NYIIR);
            }
        }
        Ok(a)
    }

    // -- Main emission loop -------------------------------------------------

    #[allow(clippy::type_complexity)]
    fn emit(mut self) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
        // Uniform outer frame: capture the two arguments in r10/r11 and
        // (Win64) save all callee-saved xmm registers. Every trace uses
        // the identical frame, so linked traces jump between their inner
        // entries while reusing the frame of the Rust-entered trace; the
        // save cost is paid once per Rust entry, not per link.
        #[cfg(windows)]
        {
            self.mov_rr64(RBASE, RCX);
            self.mov_rr64(RENV, RDX);
            self.code.extend_from_slice(&[0x48, 0x81, 0xEC]); // sub rsp, n
            self.emit_u32(FRAME_SIZE);
            for k in 0..10u8 {
                self.movups_spill(6 + k, (k as i32) * 16, true);
            }
        }
        #[cfg(not(windows))]
        {
            self.mov_rr64(RBASE, RDI);
            self.mov_rr64(RENV, RSI);
        }
        let inner = self.code.len() as u32;

        // Side traces: the in-mcode hand-over prelude copies the
        // inherited values from the parent's env slots to the own ones
        // (a parallel move: ranges may overlap, even cyclically).
        if !self.tr.parentmap.is_empty() {
            self.emit_handover();
        }
        let head = self.code.len();

        let nins = self.tr.ir.nins();
        let mut r = REF_FIRST;
        while r < nins {
            // Same covering-snapshot rule as the portable executor.
            while self.snapidx + 1 < self.tr.snap.len() && self.tr.snap[self.snapidx + 1].iref <= r
            {
                self.snapidx += 1;
            }
            self.cur = r;
            let ins = *self.tr.ir.ir(r);
            if std::env::var("LUAJIT_RS_TRDUMP").as_deref() == Ok("2") {
                eprintln!(
                    "  tr{} {:#05x} {:04} {:?} {} {} t={}",
                    self.tr.traceno,
                    self.code.len(),
                    r - REF_BIAS,
                    ins.op(),
                    ins.op1 as i32 - REF_BIAS as i32,
                    ins.op2 as i32 - REF_BIAS as i32,
                    ins.t(),
                );
            }
            match ins.op() {
                IROp::NOP | IROp::BASE => {}
                IROp::LOOP => self.asm_loop_head(),
                IROp::PHI => {} // Handled at the back edge.
                IROp::SLOAD => self.asm_sload(&ins)?,
                IROp::ULOAD => self.asm_uload(&ins)?,
                IROp::FLOAD => self.asm_meta_guard(&ins),
                IROp::HLOAD => self.asm_hload(&ins)?,
                IROp::CARG => {} // Consumed by HSTORE/CALLL/ASTORE.
                IROp::HSTORE => self.asm_hstore(&ins),
                IROp::CALLL => self.asm_calll(&ins)?,
                IROp::TNEW => {
                    self.helper_call(
                        super::super::exec::jit_tnew as *const () as usize as u64,
                        &[],
                    );
                    self.ff_result(&ins)?;
                }
                IROp::TDUP => {
                    self.helper_call(
                        super::super::exec::jit_tdup as *const () as usize as u64,
                        &[ins.op1 as IRRef],
                    );
                    self.ff_result(&ins)?;
                }
                IROp::ALOAD => self.asm_aload(&ins)?,
                IROp::ASTORE => self.asm_astore(&ins)?,
                IROp::GCSTEP => self.asm_gcstep(&ins),
                IROp::POW => self.asm_pow(&ins)?,
                IROp::TOBIT => self.asm_tobit(&ins)?,
                IROp::BAND
                | IROp::BOR
                | IROp::BXOR
                | IROp::BSHL
                | IROp::BSHR
                | IROp::BSAR
                | IROp::BROL
                | IROp::BROR
                | IROp::BNOT
                | IROp::BSWAP => self.asm_bitop(&ins)?,
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
                IROp::LT
                | IROp::GE
                | IROp::LE
                | IROp::GT
                | IROp::ULT
                | IROp::UGE
                | IROp::ULE
                | IROp::UGT => self.asm_comp(&ins)?,
                IROp::EQ | IROp::NE => self.asm_equal(&ins)?,
                _ => unreachable!("op rejected by the NYI scan"),
            }
            r += 1;
        }

        // Tail: loop-optimized traces jump back to IR_LOOP with the PHI
        // values moved into place; legacy loops re-materialize the final
        // snapshot and restart at the head; Root-linked side traces sync
        // the stack and jump straight into the root trace's inner entry;
        // everything else leaves through the final snapshot.
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
        } else if matches!(self.tr.linktype, TraceLink::Uprec | TraceLink::Tailrec)
            && self.tr.link == self.tr.traceno
        {
            // Recursive tail: materialize the frames pushed on trace,
            // check the stack headroom (exit through the final snapshot
            // when it runs out — the Rust executor grows and re-enters),
            // advance BASE to the new callee frame and restart.
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let delta = (self.tr.snap[lastsnap].baseslot as i32 - 2) * 8;
            // rax = prospective frame top: max framesize (255) + margin.
            self.mov_rr64(RAX, RBASE);
            self.add_r64_imm32(RAX, delta + (255 + 8) * 8);
            self.mov_r64_imm64(RCX, super::super::exec::stack_end_cell_addr());
            self.cmp_r64_mem(RAX, RCX, 0);
            self.guard(CC_A);
            if delta != 0 {
                self.add_r64_imm32(RBASE, delta);
            }
            self.code.push(0xE9); // jmp inner (re-enter from the top)
            let rel = inner as i64 - (self.code.len() as i64 + 4);
            self.emit_u32(rel as i32 as u32);
        } else if self.tr.linktype == TraceLink::Root
            && let Some(target) = self.link
        {
            // asm_tail_link: materialize the final snapshot into the Lua
            // stack (the root re-reads it through its SLOADs), then jump
            // to the root's inner entry — no Rust round trip. Call-frame
            // tails (a compiled callee) advance BASE first, with the
            // same headroom check as the recursive tails.
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let delta = (self.tr.snap[lastsnap].baseslot as i32 - 2) * 8;
            if delta != 0 {
                self.mov_rr64(RAX, RBASE);
                self.add_r64_imm32(RAX, delta + (255 + 8) * 8);
                self.mov_r64_imm64(RCX, super::super::exec::stack_end_cell_addr());
                self.cmp_r64_mem(RAX, RCX, 0);
                self.guard(CC_A);
                self.add_r64_imm32(RBASE, delta);
            }
            self.mov_r64_imm64(RAX, target as u64);
            self.code.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
        } else {
            // Leave through the final snapshot: flush its register-held
            // values inline, then fall through into the epilogue.
            for (rg, fr) in self.exit_flush_set(lastsnap) {
                self.movsd_store(RENV, Self::env_disp(fr), rg);
            }
            self.mov_eax_imm(self.exit_code(lastsnap));
        }

        // Common epilogue: report the (possibly shifted) BASE register
        // back to the executor, restore the callee-saved xmm (Win64),
        // return eax.
        let epilogue = self.code.len();
        self.mov_r64_imm64(RCX, super::super::exec::exit_base_cell_addr());
        self.mov_mem_r64(RCX, 0, RBASE);
        #[cfg(windows)]
        {
            for k in 0..10u8 {
                self.movups_spill(6 + k, (k as i32) * 16, false);
            }
            self.code.extend_from_slice(&[0x48, 0x81, 0xC4]); // add rsp, n
            self.emit_u32(FRAME_SIZE);
        }
        self.code.push(0xC3);

        // Per-guard exit stubs: flush the live snapshot registers to env
        // (the ExitState equivalent), then leave with the snapshot index.
        // The `mov eax; jmp` tail is padded to 12 bytes so `patch_exit`
        // can later retarget it into a compiled side trace.
        let stubs = std::mem::take(&mut self.stubs);
        let mut stubpos = Vec::with_capacity(stubs.len());
        for st in &stubs {
            stubpos.push(self.code.len());
            for &(rg, fr) in &st.flush {
                self.movsd_store(RENV, Self::env_disp(fr), rg);
            }
            let tail = self.code.len();
            let mut code = self.exit_code(st.snapidx);
            if st.gc {
                code |= 0x8000; // GC exits are flagged and never patched.
            } else {
                self.stub_tails.push((st.snapidx as u32, tail as u32));
            }
            self.mov_eax_imm(code);
            self.code.push(0xE9);
            let rel = epilogue as i64 - (self.code.len() as i64 + 4);
            self.emit_u32(rel as i32 as u32);
            while self.code.len() < tail + 12 {
                self.code.push(0xCC); // Patch space (movabs rax; jmp rax).
            }
        }
        for (pos, si) in std::mem::take(&mut self.fixups) {
            let rel = (stubpos[si] as i64 - (pos as i64 + 4)) as i32;
            self.code[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
        }

        let mut area = McodeArea::alloc(self.code.len()).ok_or(TraceError::MCODEAL)?;
        area.as_mut_slice()[..self.code.len()].copy_from_slice(&self.code);
        if !area.protect_exec() {
            return Err(TraceError::MCODEAL);
        }
        Ok((area, inner, std::mem::take(&mut self.stub_tails)))
    }

    /// Exit return value: `(traceno << 16) | snapidx`. With patched exit
    /// chains, the trace that finally returns to Rust may not be the one
    /// entered, so the exit identifies itself.
    fn exit_code(&self, snapidx: usize) -> u32 {
        (self.tr.traceno << 16) | snapidx as u32
    }

    /// The hand-over prelude of a side trace: a parallel move of the
    /// parent env slots into the own (inherited SLOAD) slots. Emitted at
    /// the inner entry, so both the patched parent exits and the Rust
    /// executor path run it. Cycles are broken via the xmm scratch.
    fn emit_handover(&mut self) {
        let mut pending: Vec<(IRRef, IRRef)> = self
            .tr
            .parentmap
            .iter()
            .map(|&(o, p)| (o as IRRef, p as IRRef))
            .filter(|&(o, p)| o != p) // Same slot: already in place.
            .collect();
        let mut parked: Option<IRRef> = None; // Slot value held in xmm scratch.
        while !pending.is_empty() {
            let ready = pending
                .iter()
                .position(|&(d, _)| !pending.iter().any(|&(_, s)| s == d && parked != Some(s)));
            if let Some(i) = ready {
                let (d, s) = pending.remove(i);
                if parked == Some(s) {
                    self.movsd_store(RENV, Self::env_disp(d), XMM_SCRATCH);
                } else {
                    self.mov_r64_mem(RAX, RENV, Self::env_disp(s));
                    self.mov_mem_r64(RENV, Self::env_disp(d), RAX);
                }
                if parked == Some(s) && !pending.iter().any(|&(_, s2)| s2 == s) {
                    parked = None; // Last reader of the parked value.
                }
            } else {
                // Cycle: park the first destination's current value.
                debug_assert!(parked.is_none(), "one scratch, one cycle at a time");
                let d0 = pending[0].0;
                self.movsd_load(XMM_SCRATCH, RENV, Self::env_disp(d0));
                parked = Some(d0);
            }
        }
    }

    // -- Instruction emitters ------------------------------------------------

    /// SLOAD: optional typecheck guard + load from the Lua stack.
    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        if ins.op2 as u32 & IRSLOAD_PARENT != 0 {
            // Inherited from the parent trace: the value sits in this
            // ref's env slot (pre-filled by the executor on the linked
            // exit); it is fetched lazily on first use.
            return Ok(());
        }
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

    /// ULOAD: load a closed upvalue cell through its constant address
    /// (op1 = KINT64), with the same typecheck shapes as SLOAD.
    fn asm_uload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let t = ins.t();
        let i = Self::iidx(self.cur);
        self.mov_r64_imm64(RAX, addr);
        if irt_isnum(t) {
            if ins.is_guard() {
                self.cmp_mem32_imm32(RAX, 4, TISNUM_HI);
                self.guard(CC_AE);
            }
            if self.last_use[i] != 0 || self.needs_env[i] {
                let d = self.alloc(0)?;
                self.movsd_load(d, RAX, 0);
                self.def(d);
            }
            return Ok(());
        }
        if self.needs_env[i] {
            self.mov_r64_mem(RCX, RAX, 0);
            self.mov_mem_r64(RENV, Self::env_disp(self.cur), RCX);
            self.env_valid[i] = true;
        }
        if !ins.is_guard() {
            return Ok(());
        }
        let ty = irt_type(t);
        if ty == IRT_NIL {
            self.cmp_mem64_imm8(RAX, 0, -1);
            self.guard(CC_NE);
        } else if ty <= IRT_TRUE {
            let itype = !(ty as u32);
            self.cmp_mem32_imm32(RAX, 4, (itype << 15) | 0x7fff);
            self.guard(CC_NE);
        } else {
            self.mov_r64_mem(RCX, RAX, 0);
            self.sar_r64_imm(RCX, 47);
            self.cmp_r32_imm8(RCX, !(ty as u32) as i8);
            self.guard(CC_NE);
        }
        Ok(())
    }

    /// FLOAD IRFL_TAB_META as a guard: exit unless `tab.metatable` is
    /// None (the niche encoding makes that a plain null check).
    fn asm_meta_guard(&mut self, ins: &IRIns) {
        const META_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, metatable) as i32;
        self.gpr_load_ref(RAX, ins.op1 as IRRef);
        self.mov_r64_imm64(RCX, crate::value::LJ_GCVMASK);
        self.and_rr64(RAX, RCX); // NaN-boxed table value -> pointer.
        self.cmp_mem64_imm8(RAX, META_OFF, 0);
        self.guard(CC_NE);
    }

    /// HLOAD: raw table get through the shared helper, with the SLOAD
    /// typecheck shapes applied to the returned value bits in rax.
    fn asm_hload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_tget as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }

    /// CALLL: guarded helper calls selected by the IRCALL index in op2.
    /// op1 is the argument (arity 1) or a CARG chain (arity 2/3).
    fn asm_calll(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        use super::super::record as rec;
        let idx = ins.op2 as u32;
        let addr = match idx {
            rec::IRCALL_TAB_NEXTK => super::super::exec::jit_tnextk as *const () as u64,
            rec::IRCALL_FMOD => super::super::exec::jit_fmod as *const () as u64,
            rec::IRCALL_STR_LEN => super::super::exec::jit_str_len as *const () as u64,
            rec::IRCALL_STR_CMP => super::super::exec::jit_str_cmp as *const () as u64,
            rec::IRCALL_STR_BYTE => super::super::exec::jit_str_byte as *const () as u64,
            rec::IRCALL_STR_SUB => super::super::exec::jit_str_sub as *const () as u64,
            rec::IRCALL_VARG => super::super::exec::jit_varg as *const () as u64,
            rec::IRCALL_STR_CHAR => super::super::exec::jit_str_char as *const () as u64,
            rec::IRCALL_TAB_LEN => super::super::exec::jit_alen as *const () as u64,
            rec::IRCALL_TAB_CONCAT => super::super::exec::jit_tconcat as *const () as u64,
            rec::IRCALL_CAT => super::super::exec::jit_cat as *const () as u64,
            rec::IRCALL_USET => super::super::exec::jit_uset as *const () as u64,
            _ => unreachable!("bad IRCALL index"),
        };
        match rec::ircall_arity(idx) {
            1 => self.helper_call(addr, &[ins.op1 as IRRef]),
            2 => {
                let carg = *self.tr.ir.ir(ins.op1 as IRRef);
                debug_assert_eq!(carg.op(), IROp::CARG);
                self.helper_call(addr, &[carg.op1 as IRRef, carg.op2 as IRRef]);
            }
            _ => {
                let cargj = *self.tr.ir.ir(ins.op1 as IRRef);
                debug_assert_eq!(cargj.op(), IROp::CARG);
                let cargi = *self.tr.ir.ir(cargj.op1 as IRRef);
                debug_assert_eq!(cargi.op(), IROp::CARG);
                self.helper_call(
                    addr,
                    &[cargi.op1 as IRRef, cargi.op2 as IRRef, cargj.op2 as IRRef],
                );
            }
        }
        self.ff_result(ins)
    }

    /// Shared tail of the helper-returning ops: typecheck the value bits
    /// in rax against the instruction's guarded type, then land the
    /// result (xmm for numbers, env for GC bits).
    fn ff_result(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let t = ins.t();
        let i = Self::iidx(self.cur);
        if irt_isnum(t) {
            if ins.is_guard() {
                // High dword below the GC tag space = number.
                self.mov_rr64(RCX, RAX);
                self.shr_r64_imm(RCX, 32);
                self.cmp_r32_imm32(RCX, TISNUM_HI);
                self.guard(CC_AE);
            }
            if self.last_use[i] != 0 || self.needs_env[i] {
                let d = self.alloc(0)?;
                self.movq_xmm_gpr(d, RAX);
                self.def(d);
            }
            return Ok(());
        }
        if self.needs_env[i] {
            self.mov_mem_r64(RENV, Self::env_disp(self.cur), RAX);
            self.env_valid[i] = true;
        }
        if !ins.is_guard() {
            return Ok(());
        }
        let ty = irt_type(t);
        if ty <= IRT_TRUE {
            // Primitives: full bit compare against the canonical value.
            let bits = match ty {
                IRT_NIL => crate::value::LuaValue::NIL.to_bits(),
                IRT_FALSE => crate::value::LuaValue::FALSE.to_bits(),
                _ => crate::value::LuaValue::TRUE.to_bits(),
            };
            self.mov_r64_imm64(RCX, bits);
            self.cmp_rr64(RAX, RCX);
            self.guard(CC_NE);
        } else {
            self.mov_rr64(RCX, RAX);
            self.sar_r64_imm(RCX, 47);
            self.cmp_r32_imm8(RCX, !(ty as u32) as i8);
            self.guard(CC_NE);
        }
        Ok(())
    }

    /// Fused bit ops: convert the (range-guarded) operands with
    /// cvttsd2si, run the 32-bit ALU op, convert back.
    fn asm_bitop(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        // Fetch both operands before the rax conversion: materializing
        // an xmm constant goes through rax.
        let sx = self.fetch_xmm(ins.op1 as IRRef, 0)?;
        if !matches!(op, IROp::BNOT | IROp::BSWAP) {
            let sy = if ins.op2 == ins.op1 {
                sx
            } else {
                self.fetch_xmm(ins.op2 as IRRef, pin(sx))?
            };
            self.cvttsd2si_r64(RAX, sx);
            self.cvttsd2si_r64(RCX, sy);
        } else {
            self.cvttsd2si_r64(RAX, sx);
        }
        match op {
            IROp::BNOT => self.code.extend_from_slice(&[0xF7, 0xD0]), // not eax
            IROp::BSWAP => self.code.extend_from_slice(&[0x0F, 0xC8]), // bswap eax
            IROp::BAND => self.code.extend_from_slice(&[0x21, 0xC8]), // and eax, ecx
            IROp::BOR => self.code.extend_from_slice(&[0x09, 0xC8]),  // or eax, ecx
            IROp::BXOR => self.code.extend_from_slice(&[0x31, 0xC8]), // xor eax, ecx
            IROp::BSHL => self.code.extend_from_slice(&[0xD3, 0xE0]), // shl eax, cl
            IROp::BSHR => self.code.extend_from_slice(&[0xD3, 0xE8]), // shr eax, cl
            IROp::BROL => self.code.extend_from_slice(&[0xD3, 0xC0]), // rol eax, cl
            IROp::BROR => self.code.extend_from_slice(&[0xD3, 0xC8]), // ror eax, cl
            _ => self.code.extend_from_slice(&[0xD3, 0xF8]),          // sar eax, cl
        }
        let d = self.alloc(0)?;
        self.cvtsi2sd_r32(d, RAX);
        self.def(d);
        Ok(())
    }

    /// TOBIT: wrapping num -> int32 -> num. Inside the recorder's range
    /// guards, cvtsd2si (round-to-nearest-even) matches num2bit exactly.
    fn asm_tobit(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_xmm(ins.op1 as IRRef, 0)?;
        self.cvtsd2si_r32(RAX, sx);
        let d = self.alloc(0)?;
        self.cvtsi2sd_r32(d, RAX);
        self.def(d);
        Ok(())
    }

    /// POW: the interpreter's vm_pow via a helper call (raw bits in and
    /// out; the result is always a number, no guard).
    fn asm_pow(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_pow as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        let i = Self::iidx(self.cur);
        if self.last_use[i] != 0 || self.needs_env[i] {
            let d = self.alloc(0)?;
            self.movq_xmm_gpr(d, RAX);
            self.def(d);
        }
        Ok(())
    }

    /// HSTORE: raw table set through the shared helper. op2 is the CARG
    /// tuple holding (key, value).
    fn asm_hstore(&mut self, ins: &IRIns) {
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        debug_assert_eq!(carg.op(), IROp::CARG);
        let addr = super::super::exec::jit_tset as *const () as usize as u64;
        self.helper_call(
            addr,
            &[ins.op1 as IRRef, carg.op1 as IRRef, carg.op2 as IRRef],
        );
    }

    /// Emit a call to an `extern "C"` helper with up to three u64
    /// arguments (raw ref bits from env or constants). The volatile xmm
    /// registers are parked in env first (the callee-saved xmm6-xmm15
    /// survive); r10/r11 (BASE/ENV) are saved around the call.
    fn helper_call(&mut self, addr: u64, args: &[IRRef]) {
        // Park live values held in volatile xmm registers.
        for rg in 0..=4u8 {
            if let Owner::Ins(o) = self.owner[rg as usize] {
                let i = Self::iidx(o);
                if self.last_use[i] > self.cur && !self.env_valid[i] {
                    self.movsd_store(RENV, Self::env_disp(o), rg);
                    self.env_valid[i] = true;
                }
            }
            self.steal_quiet(rg);
        }
        // Load the arguments (from env or constants; r11 still live).
        #[cfg(windows)]
        const ARGREGS: [u8; 3] = [RCX, 2, 8]; // rcx, rdx, r8
        #[cfg(not(windows))]
        const ARGREGS: [u8; 3] = [7, 6, 2]; // rdi, rsi, rdx
        debug_assert!(args.len() <= 3);
        for (n, &r) in args.iter().enumerate() {
            self.gpr_load_ref(ARGREGS[n], r);
        }
        // push r10; push r11 (keeps 16-byte parity).
        self.code.extend_from_slice(&[0x41, 0x52, 0x41, 0x53]);
        // Align the stack for the call: body rsp is 8 mod 16; Win64 also
        // needs 32 bytes of shadow space.
        let adj: u8 = if cfg!(windows) { 40 } else { 8 };
        self.code.extend_from_slice(&[0x48, 0x83, 0xEC, adj]); // sub rsp, adj
        self.mov_r64_imm64(RAX, addr);
        self.code.extend_from_slice(&[0xFF, 0xD0]); // call rax
        self.code.extend_from_slice(&[0x48, 0x83, 0xC4, adj]); // add rsp, adj
        // pop r11; pop r10
        self.code.extend_from_slice(&[0x41, 0x5B, 0x41, 0x5A]);
    }

    /// Load the table pointer (NaN-boxed bits -> masked pointer) into rax
    /// and the (guarded exact-int) key into ecx. Shared head of the
    /// inlined array ops: exits when the key is not an exact int32 or
    /// (unsigned compare, covering negatives) not below tab.asize.
    fn asm_array_head(&mut self, tab: IRRef, key: IRRef) -> Result<(), TraceError> {
        const ASIZE_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, asize) as i32;
        // Fetch the key first: materializing an xmm constant goes
        // through rax, which must not hold the table pointer yet.
        let sk = self.fetch_xmm(key, 0)?;
        self.gpr_load_ref(RAX, tab);
        self.mov_r64_imm64(RCX, crate::value::LJ_GCVMASK);
        self.and_rr64(RAX, RCX);
        self.cvttsd2si_r32(RCX, sk);
        self.cvtsi2sd_r32(XMM_SCRATCH, RCX);
        self.sse_rr(0x66, 0x2E, XMM_SCRATCH, sk); // ucomisd: exact int?
        self.guard(CC_P);
        self.guard(CC_NE);
        self.cmp_r32_mem(RCX, RAX, ASIZE_OFF);
        self.guard(CC_AE);
        Ok(())
    }

    /// ALOAD: inlined array-part load (`aptr[key]`) plus the shared
    /// result typecheck.
    fn asm_aload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        self.asm_array_head(ins.op1 as IRRef, ins.op2 as IRRef)?;
        self.mov_r64_mem(RAX, RAX, APTR_OFF);
        self.mov_r64_sib(RAX, RAX, RCX); // value bits
        self.ff_result(ins)
    }

    /// ASTORE: inlined array-part store (set_int's `k > 0 && k < asize`
    /// fast path — never allocates).
    fn asm_astore(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        debug_assert_eq!(carg.op(), IROp::CARG);
        self.asm_array_head(ins.op1 as IRRef, carg.op1 as IRRef)?;
        // set_int additionally requires k > 0.
        self.code.extend_from_slice(&[0x85, 0xC9]); // test ecx, ecx
        self.guard(CC_E);
        self.mov_r64_mem(RAX, RAX, APTR_OFF);
        let val = carg.op2 as IRRef;
        if val >= REF_BIAS
            && let Some(sv) = self.reg_of(val)
        {
            self.movsd_store_sib(RAX, RCX, sv);
        } else {
            self.gpr_load_ref(RDX, val);
            self.mov_sib_r64(RAX, RCX, RDX);
        }
        Ok(())
    }

    /// IR_GCSTEP: exit when a collection is due. Mirrors the trigger sum
    /// minus the string bytes (strings never grow on-trace):
    /// `heap.total + TABLE_EXTRA >= heap.threshold`.
    fn asm_gcstep(&mut self, ins: &IRIns) {
        let total_addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let thres_addr = super::super::exec::const_bits(&self.tr.ir, ins.op2 as IRRef);
        // Thread-local cell: the trace is bound to this VM thread.
        let extra_addr = crate::table::TABLE_EXTRA.with(|c| c.as_ptr() as u64);
        self.mov_r64_imm64(RAX, total_addr);
        self.mov_r64_mem(RAX, RAX, 0);
        self.mov_r64_imm64(RCX, extra_addr);
        self.add_r64_mem(RAX, RCX, 0);
        self.mov_r64_imm64(RCX, thres_addr);
        self.cmp_r64_mem(RAX, RCX, 0);
        self.guard_gc(CC_AE);
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
        let rhs = if b == a {
            d
        } else {
            self.fetch_xmm(b, pin(d))?
        };
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
        debug_assert!((irt_isnum(ins.t()) || irt_isint(ins.t())) && ins.is_guard());
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
        let s = if snd == fst {
            f
        } else {
            self.fetch_xmm(snd, pin(f))?
        };
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
                self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, rref));
                self.mov_mem_r64(RBASE, disp, RAX);
            }
        }
    }

    /// IR_LOOP: park the live *invariant* values in env (their registers
    /// stay valid), then snapshot the register state — the back edge
    /// restores exactly this state, so the loop body can be entered from
    /// both the pre-roll fall-through and the back-edge jump.
    ///
    /// Register-homed loop-carried values (PHI lrefs) are *not* parked:
    /// their env slot would go stale after one iteration. Their env_valid
    /// is reset instead, so guards flush them from the register and body
    /// evictions store the current iteration's value.
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
                if self.phis.iter().any(|p| p.num && p.lref == o) {
                    continue; // Loop-carried: handled below.
                }
                let i = Self::iidx(o);
                if !self.env_valid[i] {
                    self.movsd_store(RENV, Self::env_disp(o), rg);
                    self.env_valid[i] = true;
                }
            }
        }
        for pi in 0..self.phis.len() {
            let p = self.phis[pi];
            let i = Self::iidx(p.lref);
            if p.num && self.loc[i].is_some() && !self.needs_env[i] {
                // A pre-roll env copy (def store or eviction) is stale
                // from the second iteration on. (needs_env lrefs stay
                // valid: the back edge refreshes their env slot.)
                self.env_valid[i] = false;
            }
        }
        self.s0 = self.owner;
        self.loop_pos = Some(self.code.len());
    }

    /// The back edge of a loop-optimized trace: carry the PHI values
    /// (rref -> lref home), restore the IR_LOOP register state and jump
    /// back. Register-homed FP values are carried in registers only;
    /// env[lref] is refreshed just for env-carried (homeless or non-FP)
    /// PHIs, whose readers go through env.
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
            if let Some(rg) = self.reg_of(p.rref)
                && p.num
            {
                return dstset & pin(rg) == 0;
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
                    if p.num
                        && let Some(rg) = self.reg_of(p.rref)
                    {
                        self.movsd_store(RENV, Self::env_disp(p.phi), rg);
                    } else {
                        debug_assert!(self.env_valid[Self::iidx(p.rref)]);
                        self.mov_r64_mem(RAX, RENV, Self::env_disp(p.rref));
                        self.mov_mem_r64(RENV, Self::env_disp(p.phi), RAX);
                    }
                } else {
                    self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, p.rref));
                    self.mov_mem_r64(RENV, Self::env_disp(p.phi), RAX);
                }
            }
            for (p, home) in phis.iter().zip(homes.iter()) {
                if p.num
                    && let Some(rg) = *home
                {
                    self.movsd_load(rg, RENV, Self::env_disp(p.phi));
                    if self.needs_env[Self::iidx(p.lref)] {
                        // Env-resident consumers (helper args) read the
                        // carried value through env[lref].
                        self.movsd_store(RENV, Self::env_disp(p.lref), rg);
                    }
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
                    self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, k));
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
    /// register (env[lref] is refreshed too when env-resident consumers
    /// exist), or refresh env[lref] for env-carried PHIs.
    fn phi_move(&mut self, p: &PhiInfo, home: Option<u8>) {
        if p.num
            && let Some(rg) = home
        {
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
                self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, p.rref));
                self.movq_xmm_gpr(rg, RAX);
            }
            if self.needs_env[Self::iidx(p.lref)] {
                self.movsd_store(RENV, Self::env_disp(p.lref), rg);
            }
        } else if p.rref >= REF_BIAS {
            if p.num
                && let Some(src) = self.reg_of(p.rref)
            {
                self.movsd_store(RENV, Self::env_disp(p.lref), src);
            } else {
                debug_assert!(self.env_valid[Self::iidx(p.rref)]);
                self.mov_r64_mem(RAX, RENV, Self::env_disp(p.rref));
                self.mov_mem_r64(RENV, Self::env_disp(p.lref), RAX);
            }
        } else {
            self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, p.rref));
            self.mov_mem_r64(RENV, Self::env_disp(p.lref), RAX);
        }
    }

    // -- Register allocation ---------------------------------------------------

    fn last_use_of(&self, r: IRRef) -> IRRef {
        if r >= REF_BIAS {
            self.last_use[Self::iidx(r)]
        } else {
            self.klast_use[Self::kidx(r)]
        }
    }

    fn mark_use(&mut self, opr: IRRef, at: IRRef) {
        if opr >= REF_BIAS {
            let i = Self::iidx(opr);
            self.last_use[i] = self.last_use[i].max(at);
        } else {
            let k = Self::kidx(opr);
            self.klast_use[k] = self.klast_use[k].max(at);
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
    fn alloc(&mut self, pinned: u16) -> Result<u8, TraceError> {
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
            self.mov_r64_imm64(RAX, super::super::exec::const_bits(&self.tr.ir, r));
            self.movq_xmm_gpr(rg, RAX);
            self.owner[rg as usize] = Owner::Konst(r);
        }
        Ok(rg)
    }

    /// Fetch op1 as the (destroyed) destination of a two-address SSE op.
    #[allow(clippy::wrong_self_convention)]
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

    /// Bind the current instruction's result. Values that must be
    /// env-resident (helper-call arguments, GC bits) are stored on def;
    /// FP snapshot values stay in registers (the exit stubs flush them).
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
        if r == REF_BIAS {
            // REF_BASE: load the Lua stack base pointer (r10).
            self.mov_rr64(gpr, RBASE);
        } else if r >= REF_BIAS {
            debug_assert!(self.env_valid[Self::iidx(r)]);
            self.mov_r64_mem(gpr, RENV, Self::env_disp(r));
        } else {
            self.mov_r64_imm64(gpr, super::super::exec::const_bits(&self.tr.ir, r));
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
        self.code
            .push(0x48 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1));
        self.code.push(0x0F);
        self.code.push(0x6E);
        self.modrm(3, dst, src);
    }

    /// cvttsd2si r32, xmm (F2 0F 2C /r): f64 -> i32, truncating.
    fn cvttsd2si_r32(&mut self, dst: u8, src: u8) {
        self.code.push(0xF2);
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x0F);
        self.code.push(0x2C);
        self.modrm(3, dst, src);
    }

    /// cvttsd2si r64, xmm (F2 REX.W 0F 2C /r): f64 -> i64, truncating.
    fn cvttsd2si_r64(&mut self, dst: u8, src: u8) {
        self.code.push(0xF2);
        let rex = 0x48 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1);
        self.code.push(rex);
        self.code.push(0x0F);
        self.code.push(0x2C);
        self.modrm(3, dst, src);
    }

    /// cvtsd2si r32, xmm (F2 0F 2D /r): f64 -> i32, round-to-nearest.
    fn cvtsd2si_r32(&mut self, dst: u8, src: u8) {
        self.code.push(0xF2);
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x0F);
        self.code.push(0x2D);
        self.modrm(3, dst, src);
    }

    /// cvtsi2sd xmm, r32 (F2 0F 2A /r): i32 -> f64 (always exact).
    fn cvtsi2sd_r32(&mut self, dst: u8, src: u8) {
        self.code.push(0xF2);
        let rex = 0x40 | (((dst >> 3) & 1) << 2) | ((src >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x0F);
        self.code.push(0x2A);
        self.modrm(3, dst, src);
    }

    fn mov_rr64(&mut self, dst: u8, src: u8) {
        self.code
            .push(0x48 | (((src >> 3) & 1) << 2) | ((dst >> 3) & 1));
        self.code.push(0x89);
        self.modrm(3, src, dst);
    }
    fn mov_r64_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.code
            .push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
        self.code.push(0x8B);
        self.mem(reg, base, disp);
    }
    fn mov_mem_r64(&mut self, base: u8, disp: i32, reg: u8) {
        self.code
            .push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
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
    fn shr_r64_imm(&mut self, reg: u8, sh: u8) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0xC1);
        self.modrm(3, 5, reg);
        self.code.push(sh);
    }
    fn and_rr64(&mut self, a: u8, b: u8) {
        // and a, b (REX.W 21 /r: rm = a, reg = b).
        self.code
            .push(0x48 | (((b >> 3) & 1) << 2) | ((a >> 3) & 1));
        self.code.push(0x21);
        self.modrm(3, b, a);
    }
    fn cmp_r32_imm32(&mut self, reg: u8, imm: u32) {
        if reg >= 8 {
            self.code.push(0x41);
        }
        self.code.push(0x81);
        self.modrm(3, 7, reg);
        self.emit_u32(imm);
    }
    fn cmp_rr64(&mut self, a: u8, b: u8) {
        self.code
            .push(0x48 | (((b >> 3) & 1) << 2) | ((a >> 3) & 1));
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
    /// cmp r32, dword [base+disp] (3B /r).
    fn cmp_r32_mem(&mut self, reg: u8, base: u8, disp: i32) {
        let rex = 0x40 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1);
        if rex != 0x40 {
            self.code.push(rex);
        }
        self.code.push(0x3B);
        self.mem(reg, base, disp);
    }
    /// cmp r64, qword [base+disp] (REX.W 3B /r).
    fn cmp_r64_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.code
            .push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
        self.code.push(0x3B);
        self.mem(reg, base, disp);
    }
    /// add r64, qword [base+disp] (REX.W 03 /r).
    fn add_r64_mem(&mut self, reg: u8, base: u8, disp: i32) {
        self.code
            .push(0x48 | (((reg >> 3) & 1) << 2) | ((base >> 3) & 1));
        self.code.push(0x03);
        self.mem(reg, base, disp);
    }
    /// add r64, imm32 (REX.W 81 /0 id).
    fn add_r64_imm32(&mut self, reg: u8, imm: i32) {
        self.code.push(0x48 | ((reg >> 3) & 1));
        self.code.push(0x81);
        self.modrm(3, 0, reg);
        self.emit_u32(imm as u32);
    }
    /// ModRM+SIB for [base + index*8] (both registers below r8, disp 0).
    fn sib8(&mut self, reg: u8, base: u8, index: u8) {
        debug_assert!(base < 8 && index < 8 && base & 7 != 5);
        self.modrm(0, reg, 4);
        self.code.push((3 << 6) | ((index & 7) << 3) | (base & 7));
    }
    /// mov r64, [base + index*8] (REX.W 8B /r + SIB).
    fn mov_r64_sib(&mut self, dst: u8, base: u8, index: u8) {
        self.code.push(0x48 | ((dst >> 3) << 2));
        self.code.push(0x8B);
        self.sib8(dst, base, index);
    }
    /// mov [base + index*8], r64 (REX.W 89 /r + SIB).
    fn mov_sib_r64(&mut self, base: u8, index: u8, src: u8) {
        self.code.push(0x48 | ((src >> 3) << 2));
        self.code.push(0x89);
        self.sib8(src, base, index);
    }
    /// movsd [base + index*8], xmm (F2 0F 11 /r + SIB).
    fn movsd_store_sib(&mut self, base: u8, index: u8, src: u8) {
        self.code.push(0xF2);
        if src >= 8 {
            self.code.push(0x44); // REX.R
        }
        self.code.push(0x0F);
        self.code.push(0x11);
        self.sib8(src, base, index);
    }

    /// Emit a guard branch to a fresh exit stub for the covering
    /// snapshot, capturing which snapshot values must be flushed from
    /// registers to env when this exact exit is taken.
    fn guard(&mut self, cc: u8) {
        let stub = self.make_stub(false);
        self.code.push(0x0F);
        self.code.push(0x80 | cc);
        self.fixups.push((self.code.len(), stub));
        self.emit_u32(0);
    }

    /// Like `guard`, but for GC-debt exits: the exit code carries the GC
    /// flag and the stub is never patched over by side traces.
    fn guard_gc(&mut self, cc: u8) {
        let stub = self.make_stub(true);
        self.code.push(0x0F);
        self.code.push(0x80 | cc);
        self.fixups.push((self.code.len(), stub));
        self.emit_u32(0);
    }

    fn make_stub(&mut self, gc: bool) -> usize {
        let flush = self.exit_flush_set(self.snapidx);
        self.stubs.push(Stub {
            snapidx: self.snapidx,
            flush,
            gc,
        });
        self.stubs.len() - 1
    }

    /// The register flush set of an exit through snapshot `snapidx` at
    /// the current emission position: every entry whose value lives in a
    /// register and has no valid env copy. NORESTORE entries are flushed
    /// too — the side-trace hand-over reads them from env.
    fn exit_flush_set(&self, snapidx: usize) -> Vec<(u8, IRRef)> {
        let snap = &self.tr.snap[snapidx];
        let ofs = snap.mapofs as usize;
        let mut flush: Vec<(u8, IRRef)> = Vec::new();
        for sn in &self.tr.snapmap[ofs..ofs + snap.nent as usize] {
            let r = snap_ref(*sn);
            if r < REF_BIAS {
                continue; // Constants are materialized by the restorer.
            }
            let i = Self::iidx(r);
            if self.env_valid[i] || flush.iter().any(|&(_, fr)| fr == r) {
                continue;
            }
            let rg = self.loc[i].expect("snapshot value neither in reg nor env");
            flush.push((rg, r));
        }
        flush
    }
}

#[inline]
fn pin(rg: u8) -> u16 {
    1u16 << rg
}
