#![allow(
    clippy::unusual_byte_groupings,
    dead_code,
    clippy::wrong_self_convention,
    clippy::type_complexity,
    clippy::needless_range_loop
)]
/// RISC-V 64 (RV64GC) trace assembler: translates SSA IR into RV64 native code.
///
/// Register conventions:
///   s0 (x8)  = RBASE  (Lua stack base, callee-saved)
///   s1 (x9)  = RENV   (spill/exit env buffer, callee-saved)
///   t0-t2 (x5-x7), t3-t6 (x28-x31) = scratch GPRs
///   f0-f7, f10-f17, f18-f25 = allocatable FP registers (f64)
///   f26-f27 = FP scratch (constant materialization)
///   a0 (x10) = return value (exit snapno)
///
/// Frame layout (16-aligned, 192 bytes):
///   Save callee-saved GPRs: s0-s11 (x8-x9, x18-x27) → 12 × 8 = 96 bytes
///   Save callee-saved FPRs: f8-f9, f18-f27 → 12 × 8 = 96 bytes
use super::super::ir::*;
use super::super::mcode::McodeArea;
use super::super::record::{IRFPM_SQRT};
use super::super::{GCtrace, TraceError, snap_ref};

// ── RISC-V 64 encoding helpers ──────────────────────────────────────────────

pub(crate) struct Emit(pub Vec<u8>);

#[allow(unused)]
impl Emit {
    fn new() -> Self { Emit(Vec::with_capacity(4096)) }
    fn len(&self) -> usize { self.0.len() }
    fn u32(&mut self, w: u32) { self.0.extend_from_slice(&w.to_le_bytes()); }
    fn patch_u32(&mut self, offset: usize, w: u32) {
        self.0[offset..offset + 4].copy_from_slice(&w.to_le_bytes());
    }
    fn nop(&mut self) { self.u32(0x00000013); }

    fn rtype(&mut self, f7: u32, rs2: u32, rs1: u32, f3: u32, rd: u32, op: u32) {
        self.u32((f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op);
    }
    fn itype(&mut self, imm: u32, rs1: u32, f3: u32, rd: u32, op: u32) {
        self.u32(((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op);
    }
    fn stype(&mut self, imm: u32, rs2: u32, rs1: u32, f3: u32, op: u32) {
        self.u32(((imm >> 5 & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1F) << 7) | op);
    }
    fn btype(&mut self, imm: u32, rs2: u32, rs1: u32, f3: u32, op: u32) {
        self.u32(((imm >> 12 & 1) << 31) | ((imm >> 5 & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm >> 1 & 0xF) << 8) | ((imm >> 11 & 1) << 7) | op);
    }
    fn utype(&mut self, imm: u32, rd: u32, op: u32) {
        self.u32(((imm >> 12 & 0xF_FFFF) << 12) | (rd << 7) | op);
    }
    fn jtype(&mut self, imm: u32, rd: u32, op: u32) {
        self.u32(((imm >> 20 & 1) << 31) | ((imm >> 1 & 0x3FF) << 21) | ((imm >> 11 & 1) << 20) | ((imm >> 12 & 0xFF) << 12) | (rd << 7) | op);
    }

    // Integer R-type
    fn add(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 0, rd as u32, 0b011_0011); }
    fn sub(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x20, rs2 as u32, rs1 as u32, 0, rd as u32, 0b011_0011); }
    fn sll(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 1, rd as u32, 0b011_0011); }
    fn srl(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 5, rd as u32, 0b011_0011); }
    fn sra(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x20, rs2 as u32, rs1 as u32, 5, rd as u32, 0b011_0011); }
    fn and(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 7, rd as u32, 0b011_0011); }
    fn or(&mut self, rd: u8, rs1: u8, rs2: u8)  { self.rtype(0, rs2 as u32, rs1 as u32, 6, rd as u32, 0b011_0011); }
    fn xor(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 4, rd as u32, 0b011_0011); }
    fn slt(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 2, rd as u32, 0b011_0011); }
    fn sltu(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0, rs2 as u32, rs1 as u32, 3, rd as u32, 0b011_0011); }
    fn mul(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(1, rs2 as u32, rs1 as u32, 0, rd as u32, 0b011_0011); }
    fn div(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(1, rs2 as u32, rs1 as u32, 4, rd as u32, 0b011_0011); }
    fn divu(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(1, rs2 as u32, rs1 as u32, 5, rd as u32, 0b011_0011); }
    fn remu(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(1, rs2 as u32, rs1 as u32, 7, rd as u32, 0b011_0011); }

    // Integer I-type
    fn addi(&mut self, rd: u8, rs1: u8, imm: u32) { self.itype(imm, rs1 as u32, 0, rd as u32, 0b001_0011); }
    fn slli(&mut self, rd: u8, rs1: u8, sh: u8) { self.itype(sh as u32, rs1 as u32, 1, rd as u32, 0b001_0011); }
    fn srli(&mut self, rd: u8, rs1: u8, sh: u8) { self.itype((0x00_000 | sh as u32), rs1 as u32, 5, rd as u32, 0b001_0011); }
    fn srai(&mut self, rd: u8, rs1: u8, sh: u8) { self.itype((0x20_000 | sh as u32), rs1 as u32, 5, rd as u32, 0b001_0011); }
    fn andi(&mut self, rd: u8, rs1: u8, imm: u32) { self.itype(imm, rs1 as u32, 7, rd as u32, 0b001_0011); }
    fn ori(&mut self, rd: u8, rs1: u8, imm: u32)  { self.itype(imm, rs1 as u32, 6, rd as u32, 0b001_0011); }
    fn xori(&mut self, rd: u8, rs1: u8, imm: u32) { self.itype(imm, rs1 as u32, 4, rd as u32, 0b001_0011); }
    fn slti(&mut self, rd: u8, rs1: u8, imm: u32) { self.itype(imm, rs1 as u32, 2, rd as u32, 0b001_0011); }

    fn mv(&mut self, rd: u8, rs1: u8) { self.addi(rd, rs1, 0); }

    // Load 32-bit immediate
    fn li(&mut self, rd: u8, imm: u64) {
        if imm == 0 { self.addi(rd, 0, 0); return; }
        let lo = (imm as i64 & 0xFFF) as i64;
        let hi = (imm as i64 - lo) as u64 & 0xFFFF_F000;
        self.utype(hi as u32, rd as u32, 0b011_0111);
        if lo != 0 { self.addi(rd, rd, lo as u32); }
    }

    /// Full 64-bit constant load: LUI+ADDI for low32, SLLI+ADDI chain for high32.
    fn mov64(&mut self, rd: u8, imm: u64) {
        if (imm >> 32) == 0 { self.li(rd, imm); return; }
        // Load low 32 bits
        let lo = (imm & 0xFFFF_FFFF) as i32;
        self.li(rd, lo as u64);
        // Load upper 32 bits into scratch, shift, add
        let hi = (imm >> 32) as i32;
        if hi != 0 {
            self.li(RSCRATCH3, hi as u64);
            self.slli(RSCRATCH3, RSCRATCH3, 32);
            // RISC-V ADDI sign-extends, so if hi is negative, slli sign-extends.
            // Use ADD (which adds full 64 bits) instead
            self.add(rd, rd, RSCRATCH3);
        }
    }
    fn mov32(&mut self, rd: u8, imm: u32) { self.li(rd, imm as u64); }

    // Load/Store
    fn ld(&mut self, rd: u8, rs1: u8, offset: i32) { self.itype(offset as u32, rs1 as u32, 3, rd as u32, 0b000_0011); }
    fn sd(&mut self, rs2: u8, rs1: u8, offset: i32) { self.stype(offset as u32, rs2 as u32, rs1 as u32, 3, 0b010_0011); }
    fn lwu(&mut self, rd: u8, rs1: u8, offset: i32) { self.itype(offset as u32, rs1 as u32, 6, rd as u32, 0b000_0011); }
    fn sw(&mut self, rs2: u8, rs1: u8, offset: i32) { self.stype(offset as u32, rs2 as u32, rs1 as u32, 2, 0b010_0011); }

    // FP load/store
    fn fld(&mut self, rd: u8, rs1: u8, offset: i32) { self.itype(offset as u32, rs1 as u32, 3, rd as u32, 0b000_0111); }
    fn fsd(&mut self, rs2: u8, rs1: u8, offset: i32) { self.stype(offset as u32, rs2 as u32, rs1 as u32, 3, 0b010_0111); }

    // FP arithmetic
    fn fadd_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(1, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fsub_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(5, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fmul_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(9, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fdiv_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0xD, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fsqrt_d(&mut self, rd: u8, rs1: u8) { self.rtype(0x2D, 0, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fmin_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x15, rs2 as u32, rs1 as u32, 1, rd as u32, 0b101_0011); }
    fn fmax_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x15, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fsgnj_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x11, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fsgnjn_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x11, rs2 as u32, rs1 as u32, 1, rd as u32, 0b101_0011); }
    fn fsgnjx_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x11, rs2 as u32, rs1 as u32, 2, rd as u32, 0b101_0011); }

    // FP compare
    fn feq_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x15, rs2 as u32, rs1 as u32, 2, rd as u32, 0b101_0011); }
    fn flt_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x15, rs2 as u32, rs1 as u32, 1, rd as u32, 0b101_0011); }
    fn fle_d(&mut self, rd: u8, rs1: u8, rs2: u8) { self.rtype(0x15, rs2 as u32, rs1 as u32, 0, rd as u32, 0b101_0011); }

    // FP convert
    fn fcvt_w_d(&mut self, rd: u8, rs1: u8) { self.rtype(0x60, 0, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fcvt_l_d(&mut self, rd: u8, rs1: u8) { self.rtype(0x60, 2, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fcvt_d_w(&mut self, rd: u8, rs1: u8) { self.rtype(0x68, 0, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fcvt_d_l(&mut self, rd: u8, rs1: u8) { self.rtype(0x68, 2, rs1 as u32, 0, rd as u32, 0b101_0011); }

    // FP/GPR move
    fn fmv_x_d(&mut self, rd: u8, rs1: u8) { self.rtype(0x70, 0, rs1 as u32, 0, rd as u32, 0b101_0011); }
    fn fmv_d_x(&mut self, rd: u8, rs1: u8) { self.rtype(0x78, 0, rs1 as u32, 0, rd as u32, 0b101_0011); }

    // Branches
    fn beq(&mut self, rs1: u8, rs2: u8, offset: i32)  { self.btype(offset as u32, rs2 as u32, rs1 as u32, 0, 0b110_0011); }
    fn bne(&mut self, rs1: u8, rs2: u8, offset: i32)  { self.btype(offset as u32, rs2 as u32, rs1 as u32, 1, 0b110_0011); }
    fn blt(&mut self, rs1: u8, rs2: u8, offset: i32)  { self.btype(offset as u32, rs2 as u32, rs1 as u32, 4, 0b110_0011); }
    fn bge(&mut self, rs1: u8, rs2: u8, offset: i32)  { self.btype(offset as u32, rs2 as u32, rs1 as u32, 5, 0b110_0011); }
    fn bltu(&mut self, rs1: u8, rs2: u8, offset: i32) { self.btype(offset as u32, rs2 as u32, rs1 as u32, 6, 0b110_0011); }
    fn bgeu(&mut self, rs1: u8, rs2: u8, offset: i32) { self.btype(offset as u32, rs2 as u32, rs1 as u32, 7, 0b110_0011); }

    // Jumps
    fn jal(&mut self, rd: u8, offset: i32) { self.jtype(offset as u32, rd as u32, 0b110_1111); }
    fn jalr(&mut self, rd: u8, rs1: u8, offset: i32) { self.itype(offset as u32 & 0xFFF, rs1 as u32, 0, rd as u32, 0b110_0111); }
    fn ret(&mut self) { self.jalr(0, 1, 0); }

    // Load f64 constant via GPR
    fn fli_d(&mut self, rd: u8, imm: u64) {
        self.mov64(RSCRATCH, imm);
        self.fmv_d_x(rd, RSCRATCH);
    }

    // Large-offset helpers
    fn ld_safe(&mut self, rt: u8, rn: u8, off: i32) {
        if off >= -2048 && off <= 2047 { self.ld(rt, rn, off); }
        else { self.mov64(RSCRATCH3, off as u64); self.add(RSCRATCH3, rn, RSCRATCH3); self.ld(rt, RSCRATCH3, 0); }
    }
    fn sd_safe(&mut self, rt: u8, rn: u8, off: i32) {
        if off >= -2048 && off <= 2047 { self.sd(rt, rn, off); }
        else { self.mov64(RSCRATCH3, off as u64); self.add(RSCRATCH3, rn, RSCRATCH3); self.sd(rt, RSCRATCH3, 0); }
    }
    fn fld_safe(&mut self, rd: u8, rn: u8, off: i32) {
        if off >= -2048 && off <= 2047 { self.fld(rd, rn, off); }
        else { self.mov64(RSCRATCH3, off as u64); self.add(RSCRATCH3, rn, RSCRATCH3); self.fld(rd, RSCRATCH3, 0); }
    }
    fn fsd_safe(&mut self, rd: u8, rn: u8, off: i32) {
        if off >= -2048 && off <= 2047 { self.fsd(rd, rn, off); }
        else { self.mov64(RSCRATCH3, off as u64); self.add(RSCRATCH3, rn, RSCRATCH3); self.fsd(rd, RSCRATCH3, 0); }
    }
}

// ── Constants ───────────────────────────────────────────────────────────────

const ALLOC_FP_REGS: [u8; 24] = [
    0, 1, 2, 3, 4, 5, 6, 7,
    10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25,
];
const FP_SCRATCH: u8 = 26;
const NREG: usize = 32;
const FRAME_SIZE: u32 = 192;

const RBASE: u8 = 8;       // s0
const RENV: u8 = 9;        // s1
const RSCRATCH: u8 = 5;    // t0
const RSCRATCH2: u8 = 6;   // t1
const RSCRATCH3: u8 = 7;   // t2
const RSP: u8 = 2;
const RRA: u8 = 1;
const RZERO: u8 = 0;

const TISNUM_HI: u32 = 0xFFF9_0000;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    None,
    Ins(IRRef),
    Konst(IRRef),
}

#[derive(Clone, Copy)]
struct PhiInfo {
    phi: IRRef,
    lref: IRRef,
    rref: IRRef,
    num: bool,
}
struct Stub {
    snapidx: usize,
    gc: bool,
}

#[inline]
fn pin(rg: u8) -> u32 { 1u32 << rg }

// ── Asm ─────────────────────────────────────────────────────────────────────

struct Asm<'a> {
    tr: &'a GCtrace,
    code: Emit,
    cur: IRRef,
    snapidx: usize,
    last_use: Vec<IRRef>,
    klast_use: Vec<IRRef>,
    needs_env: Vec<bool>,
    env_valid: Vec<bool>,
    loc: Vec<Option<u8>>,
    owner: [Owner; NREG],
    fixups: Vec<(usize, usize)>,    // (jal_position, stub_index)
    stubs: Vec<Stub>,
    phis: Vec<PhiInfo>,
    loop_pos: Option<usize>,
    s0: [Owner; NREG],
    phi_homes: Vec<(IRRef, u8)>,
    link: Option<*const u8>,
    stub_tails: Vec<(u32, u32)>,
}

impl<'a> Asm<'a> {
    #[inline] fn iidx(r: IRRef) -> usize { (r - REF_BIAS) as usize }
    #[inline] fn kidx(r: IRRef) -> usize { (REF_BIAS - 1 - r) as usize }
    #[inline] fn env_ofs(r: IRRef) -> i32 { (Self::iidx(r) * 8) as i32 }
    #[inline] fn cur_idx(&self) -> u32 { Self::iidx(self.cur) as u32 }

    fn mark_use(&mut self, r: IRRef, at: IRRef) {
        if r >= REF_BIAS {
            let ri = Self::iidx(r);
            if at > self.last_use[ri] { self.last_use[ri] = at; }
        } else if r > REF_FIRST {
            let ki = Self::kidx(r);
            if at > self.klast_use[ki] { self.klast_use[ki] = at; }
        }
    }

    // ═══ NYI scan ═══════════════════════════════════════════════════════════
    fn new(tr: &'a GCtrace, link: Option<*const u8>) -> Result<Asm<'a>, TraceError> {
        let nins = Self::iidx(tr.ir.nins());
        let nk = (REF_BIAS - tr.ir.nk()) as usize;
        let mut a = Asm {
            tr, code: Emit::new(), cur: 0, snapidx: 0,
            last_use: vec![0; nins], klast_use: vec![0; nk],
            needs_env: vec![false; nins], env_valid: vec![false; nins],
            loc: vec![None; nins], owner: [Owner::None; NREG],
            fixups: Vec::new(), stubs: Vec::new(),
            phis: Vec::new(), loop_pos: None,
            s0: [Owner::None; NREG], phi_homes: Vec::new(),
            link, stub_tails: Vec::new(),
        };
        for r in REF_FIRST..tr.ir.nins() {
            let ins = tr.ir.ir(r);
            match ins.op() {
                IROp::NOP | IROp::BASE | IROp::LOOP | IROp::SLOAD => {}
                IROp::ULOAD => {}
                IROp::FLOAD | IROp::HLOAD | IROp::CARG => {
                    for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                        if op >= REF_BIAS { a.needs_env[Self::iidx(op)] = true; }
                    }
                }
                IROp::HSTORE => {
                    if ins.op1 as IRRef >= REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)] = true; }
                }
                IROp::CALLL => {
                    if super::super::record::ircall_arity(ins.op2 as u32) == 1
                        && ins.op1 as IRRef >= REF_BIAS
                    { a.needs_env[Self::iidx(ins.op1 as IRRef)] = true; }
                }
                IROp::TNEW | IROp::TDUP => {}
                IROp::ALOAD => {
                    if ins.op1 as IRRef >= REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)] = true; }
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::ASTORE => {
                    if ins.op1 as IRRef >= REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)] = true; }
                }
                IROp::GCSTEP => {}
                IROp::BAND|IROp::BOR|IROp::BXOR|IROp::BSHL|IROp::BSHR|IROp::BSAR
                |IROp::BROL|IROp::BROR|IROp::BNOT|IROp::BSWAP => {
                    a.mark_use(ins.op1 as IRRef, r);
                    if ins.op2 != 0 { a.mark_use(ins.op2 as IRRef, r); }
                }
                IROp::ADD|IROp::SUB|IROp::MUL|IROp::DIV|IROp::MIN|IROp::MAX => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::NEG => { a.mark_use(ins.op1 as IRRef, r); a.mark_use(ins.op2 as IRRef, r); }
                IROp::ABS => a.mark_use(ins.op1 as IRRef, r),
                IROp::FPMATH => a.mark_use(ins.op1 as IRRef, r),
                IROp::LT|IROp::GE|IROp::LE|IROp::GT|IROp::ULT|IROp::UGE|IROp::ULE|IROp::UGT => {
                    a.mark_use(ins.op1 as IRRef, r); a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::POW => {
                    for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                        a.mark_use(op, r);
                        if op >= REF_BIAS { a.needs_env[Self::iidx(op)] = true; }
                    }
                }
                IROp::TOBIT => a.mark_use(ins.op1 as IRRef, r),
                IROp::EQ|IROp::NE => {
                    a.mark_use(ins.op1 as IRRef, r); a.mark_use(ins.op2 as IRRef, r);
                    if !irt_isnum(ins.t()) {
                        for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                            if op >= REF_BIAS { a.needs_env[Self::iidx(op)] = true; }
                        }
                    }
                }
                IROp::PHI => {
                    let (lref, rref) = (ins.op1 as IRRef, ins.op2 as IRRef);
                    a.mark_use(lref, tr.ir.nins());
                    a.mark_use(rref, tr.ir.nins());
                    if !irt_isnum(ins.t()) {
                        if lref >= REF_BIAS { a.needs_env[Self::iidx(lref)] = true; }
                        if rref >= REF_BIAS { a.needs_env[Self::iidx(rref)] = true; }
                    }
                    a.phis.push(PhiInfo { phi: r, lref, rref, num: irt_isnum(ins.t()) });
                }
                _ => return Err(TraceError::NYIIR),
            }
        }
        for (i, snap) in tr.snap.iter().enumerate() {
            let lu = if i + 1 < tr.snap.len() { tr.snap[i + 1].iref } else { tr.ir.nins() };
            let ofs = snap.mapofs as usize;
            for sn in &tr.snapmap[ofs..ofs + snap.nent as usize] {
                let rr = snap_ref(*sn);
                if rr >= REF_BIAS {
                    a.mark_use(rr, lu);
                    if !irt_isnum(tr.ir.ir(rr).t()) && !irt_isint(tr.ir.ir(rr).t()) {
                        a.needs_env[Self::iidx(rr)] = true;
                    }
                }
            }
        }
        Ok(a)
    }

    // ═══ Register allocation ═══════════════════════════════════════════════
    fn alloc(&mut self, pinned: u32) -> Result<u8, TraceError> {
        for &rg in ALLOC_FP_REGS.iter() {
            if pin(rg) & pinned != 0 { continue; }
            if self.owner[rg as usize] == Owner::None { self.owner[rg as usize] = Owner::Ins(0); return Ok(rg); }
        }
        let mut farthest: i32 = -1;
        let mut victim = 0xFFu8;
        for &rg in ALLOC_FP_REGS.iter() {
            if pin(rg) & pinned != 0 { continue; }
            let d = match self.owner[rg as usize] {
                Owner::Ins(r) => self.last_use[Self::iidx(r)] as i32,
                Owner::Konst(r) => self.klast_use[Self::kidx(r)] as i32,
                Owner::None => unreachable!(),
            };
            if d as usize == Self::iidx(self.cur) { self.steal_quiet(rg); return Ok(rg); }
            if d > farthest { farthest = d; victim = rg; }
        }
        if victim == 0xFF { return Err(TraceError::NYIIR); }
        self.spill(victim);
        Ok(victim)
    }

    fn spill(&mut self, rg: u8) {
        match self.owner[rg as usize] {
            Owner::Ins(r) => {
                if self.needs_env[Self::iidx(r)] && !self.env_valid[Self::iidx(r)] {
                    self.code.fsd_safe(rg, RENV, Self::env_ofs(r));
                    self.env_valid[Self::iidx(r)] = true;
                }
                self.loc[Self::iidx(r)] = None;
            }
            Owner::Konst(r) => { self.loc[Self::kidx(r)] = None; }
            Owner::None => unreachable!(),
        }
        self.owner[rg as usize] = Owner::None;
    }

    fn steal_quiet(&mut self, rg: u8) { self.owner[rg as usize] = Owner::None; }

    fn fetch_fp(&mut self, r: IRRef, pinned: u32) -> Result<u8, TraceError> {
        if r < REF_BIAS {
            let ki = Self::kidx(r);
            let bits = super::super::exec::const_bits(&self.tr.ir, r);
            if let Some(rg) = self.loc[ki] {
                if self.owner[rg as usize] == Owner::Konst(r) {
                    self.klast_use[ki] = self.cur_idx();
                    return Ok(rg);
                }
            }
            let d = self.alloc(pinned)?;
            self.code.fli_d(d, bits);
            self.owner[d as usize] = Owner::Konst(r);
            self.loc[ki] = Some(d);
            self.klast_use[ki] = self.cur_idx();
            Ok(d)
        } else {
            let ri = Self::iidx(r);
            if let Some(rg) = self.loc[ri] {
                if self.owner[rg as usize] == Owner::Ins(r) && (pin(rg) & pinned) == 0 {
                    self.last_use[ri] = self.cur_idx();
                    return Ok(rg);
                }
            }
            if self.env_valid[ri] {
                let d = self.alloc(pinned)?;
                self.code.fld_safe(d, RENV, Self::env_ofs(r));
                self.owner[d as usize] = Owner::Ins(r);
                self.loc[ri] = Some(d);
                self.last_use[ri] = self.cur_idx();
                Ok(d)
            } else {
                Err(TraceError::NYIIR)
            }
        }
    }

    fn into_dst(&mut self, a: IRRef) -> Result<u8, TraceError> {
        if a >= REF_BIAS && self.last_use[Self::iidx(a)] == self.cur_idx() {
            if let Some(rg) = self.loc[Self::iidx(a)] {
                self.steal_quiet(rg);
                return Ok(rg);
            }
        }
        self.alloc(0)
    }

    fn def(&mut self, d: u8) {
        self.loc[Self::iidx(self.cur)] = Some(d);
        self.owner[d as usize] = Owner::Ins(self.cur);
        if self.needs_env[Self::iidx(self.cur)] {
            self.code.fsd_safe(d, RENV, Self::env_ofs(self.cur));
            self.env_valid[Self::iidx(self.cur)] = true;
        }
    }

    fn reg_of(&self, r: IRRef) -> Option<u8> {
        if r >= REF_BIAS { self.loc.get(Self::iidx(r)).copied().flatten() }
        else { self.loc.get(Self::kidx(r)).copied().flatten() }
    }

    fn gpr_load_ref(&mut self, gpr: u8, r: IRRef) {
        if r < REF_BIAS {
            self.code.mov64(gpr, super::super::exec::const_bits(&self.tr.ir, r));
        } else if let Some(fr) = self.reg_of(r) {
            self.code.fmv_x_d(gpr, fr);
        } else {
            self.code.ld_safe(gpr, RENV, Self::env_ofs(r));
        }
    }

    // ═══ Guard / exit ══════════════════════════════════════════════════════
    /// Emit a guard: BEQ(condition=false) → skip JAL; JAL → exit stub (patched later).
    fn guard(&mut self) {
        // BEQ RSCRATCH, ZERO, +8  (skip JAL if condition false)
        self.code.beq(RSCRATCH, RZERO, 8);
        // JAL ZERO, 0  (placeholder — JAL offset patched in fixups)
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
    }

    /// Patch a JAL instruction at `jal_pos` to jump to `target`.
    fn patch_jal(code: &mut Emit, jal_pos: usize, target: usize) {
        let offset = target as i32 - jal_pos as i32;
        let _w = ((offset as u32 >> 1 & 0xFFFFF) << 12) | ((RZERO as u32) << 7) | 0b110_1111;
        // J-type: imm[20|10:1|11|19:12] rd opcode
        // The above is wrong; let me fix:
        // J-type: imm[20]<<31 | imm[10:1]<<21 | imm[11]<<20 | imm[19:12]<<12 | rd<<7 | opcode
        let i = offset as u32;
        let w = ((i >> 20 & 1) << 31)
              | ((i >> 1 & 0x3FF) << 21)
              | ((i >> 11 & 1) << 20)
              | ((i >> 12 & 0xFF) << 12)
              | ((RZERO as u32) << 7)
              | 0b110_1111u32;
        code.patch_u32(jal_pos, w);
    }

    // ═══ asm_arith ═════════════════════════════════════════════════════════
    fn asm_arith(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let sy = self.fetch_fp(ins.op2 as IRRef, pin(sx))?;
        let d = self.into_dst(ins.op1 as IRRef)?;
        match ins.op() {
            IROp::ADD => self.code.fadd_d(d, sx, sy),
            IROp::SUB => self.code.fsub_d(d, sx, sy),
            IROp::MUL => self.code.fmul_d(d, sx, sy),
            IROp::DIV => self.code.fdiv_d(d, sx, sy),
            IROp::MIN => self.code.fmin_d(d, sx, sy),
            IROp::MAX => self.code.fmax_d(d, sx, sy),
            _ => unreachable!(),
        }
        self.def(d);
        Ok(())
    }

    fn asm_neg(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.into_dst(ins.op1 as IRRef)?;
        self.code.fsgnjn_d(d, sx, sx);
        self.def(d);
        Ok(())
    }

    fn asm_abs(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.into_dst(ins.op1 as IRRef)?;
        self.code.fsgnjx_d(d, sx, sx);
        self.def(d);
        Ok(())
    }

    fn asm_fpmath(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.into_dst(ins.op1 as IRRef)?;
        match ins.op2 as u32 {
            IRFPM_SQRT => self.code.fsqrt_d(d, sx),
            _ => return Err(TraceError::NYIIR),
        }
        self.def(d);
        Ok(())
    }

    fn asm_tobit(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        self.code.fcvt_w_d(RSCRATCH, sx);
        let d = self.alloc(0)?;
        self.code.fcvt_d_w(d, RSCRATCH);
        self.def(d);
        Ok(())
    }

    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let refr = ins.op1 as IRRef;
        let _parent = ins.op2;
        self.code.li(RSCRATCH3, (Self::iidx(refr) as u64) << 3);
        self.code.add(RSCRATCH3, RBASE, RSCRATCH3);
        self.code.ld(RSCRATCH, RSCRATCH3, 0);
        if irt_isnum(ins.t()) {
            self.code.srli(RSCRATCH2, RSCRATCH, 48);
            self.code.mov32(RSCRATCH3, TISNUM_HI);
            self.code.sltu(RSCRATCH2, RSCRATCH2, RSCRATCH3);
            self.guard();
            let d = self.alloc(0)?;
            self.code.fmv_d_x(d, RSCRATCH);
            self.def(d);
        } else if irt_isint(ins.t()) {
            let d = self.alloc(0)?;
            self.code.fmv_d_x(d, RSCRATCH);
            self.def(d);
        } else {
            let d = self.alloc(0)?;
            self.code.fmv_d_x(d, RSCRATCH);
            self.def(d);
        }
        Ok(())
    }

    fn asm_comp(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        if matches!(op, IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT) {
            // Unsigned: convert both operands to u32 via GPR compare
            let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
            let sy = self.fetch_fp(ins.op2 as IRRef, pin(sx))?;
            self.code.fcvt_l_d(RSCRATCH, sx);
            self.code.fcvt_l_d(RSCRATCH2, sy);
            match op {
                IROp::ULT => self.code.sltu(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::UGE => self.code.sltu(RSCRATCH, RSCRATCH2, RSCRATCH), // RSCRATCH=1 if y<x → exit on GE (want x>=y)
                IROp::ULE => { self.code.sltu(RSCRATCH2, RSCRATCH2, RSCRATCH); self.code.xori(RSCRATCH, RSCRATCH2, 1); } // LE = !(y < x) i.e. !ULT
                IROp::UGT => self.code.sltu(RSCRATCH, RSCRATCH2, RSCRATCH), // RSCRATCH=1 if y < x → x > y
                _ => unreachable!(),
            }
            match op {
                IROp::ULT => self.code.beq(RSCRATCH, RZERO, 8),  // exit if not ULT (RSCRATCH==0)
                IROp::UGE => self.code.bne(RSCRATCH, RZERO, 8),  // exit if y<x (RSCRATCH==1) → not UGE
                IROp::ULE => self.code.beq(RSCRATCH, RZERO, 8),  // exit if not ULE
                IROp::UGT => self.code.bne(RSCRATCH, RZERO, 8),  // exit if not UGT
                _ => unreachable!(),
            }
            let jal_pos = self.code.len();
            self.fixups.push((jal_pos, self.stubs.len()));
            self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
            self.code.jal(RZERO, 0);
            return Ok(());
        }
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let sy = self.fetch_fp(ins.op2 as IRRef, pin(sx))?;
        match ins.op() {
            IROp::LT => self.code.flt_d(RSCRATCH, sx, sy),
            IROp::GE => { self.code.flt_d(RSCRATCH, sx, sy); }
            IROp::LE => self.code.fle_d(RSCRATCH, sx, sy),
            IROp::GT => { self.code.fle_d(RSCRATCH, sx, sy); }
            _ => return Err(TraceError::NYIIR),
        }
        match ins.op() {
            IROp::LT => { self.code.beq(RSCRATCH, RZERO, 8); }
            IROp::GE => { self.code.bne(RSCRATCH, RZERO, 8); }
            IROp::LE => { self.code.beq(RSCRATCH, RZERO, 8); }
            IROp::GT => { self.code.bne(RSCRATCH, RZERO, 8); }
            _ => unreachable!(),
        }
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
        Ok(())
    }

    fn asm_loop_head(&mut self) {
        self.loop_pos = Some(self.code.len());
        self.s0 = self.owner;
        self.phi_homes.clear();
        for p in &self.phis {
            if let Some(rg) = self.loc[Self::iidx(p.rref)] { self.phi_homes.push((p.phi, rg)); }
        }
    }

    fn asm_gcstep(&mut self, ins: &IRIns) {
        let total_addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let thres_addr = super::super::exec::const_bits(&self.tr.ir, ins.op2 as IRRef);
        let extra_addr = thres_addr + std::mem::size_of::<usize>() as u64;
        self.code.mov64(RSCRATCH, total_addr);
        self.code.ld(RSCRATCH, RSCRATCH, 0);
        self.code.mov64(RSCRATCH2, extra_addr);
        self.code.ld(RSCRATCH2, RSCRATCH2, 0);
        self.code.add(RSCRATCH, RSCRATCH, RSCRATCH2);
        self.code.mov64(RSCRATCH2, thres_addr);
        self.code.ld(RSCRATCH2, RSCRATCH2, 0);
        self.code.sltu(RSCRATCH, RSCRATCH, RSCRATCH2);
        // Exit if total >= threshold (RSCRATCH == 0 → not less than)
        self.code.bne(RSCRATCH, RZERO, 8);
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: true });
        self.code.jal(RZERO, 0);
    }

    fn asm_bitop(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        self.code.fmv_x_d(RSCRATCH, sx);
        if ins.op2 != 0 {
            let sy = self.fetch_fp(ins.op2 as IRRef, pin(sx))?;
            self.code.fmv_x_d(RSCRATCH2, sy);
            match ins.op() {
                IROp::BAND => self.code.and(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BOR  => self.code.or(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BXOR => self.code.xor(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BSHL => self.code.sll(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BSHR => self.code.srl(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BSAR => self.code.sra(RSCRATCH, RSCRATCH, RSCRATCH2),
                IROp::BROL => { self.code.mov64(RSCRATCH3, 64); self.code.sub(RSCRATCH2, RSCRATCH3, RSCRATCH2); self.code.srl(RSCRATCH3, RSCRATCH, RSCRATCH2); self.code.sll(RSCRATCH, RSCRATCH, RSCRATCH2); self.code.or(RSCRATCH, RSCRATCH, RSCRATCH3); }
                IROp::BROR => { self.code.mov64(RSCRATCH3, 64); self.code.sub(RSCRATCH2, RSCRATCH3, RSCRATCH2); self.code.sll(RSCRATCH3, RSCRATCH, RSCRATCH2); self.code.srl(RSCRATCH, RSCRATCH, RSCRATCH2); self.code.or(RSCRATCH, RSCRATCH, RSCRATCH3); }
                _ => return Err(TraceError::NYIIR),
            }
        } else {
            match ins.op() {
                IROp::BNOT => { self.code.mov64(RSCRATCH2, !0u64); self.code.xor(RSCRATCH, RSCRATCH, RSCRATCH2); }
                IROp::BSWAP => return Err(TraceError::NYIIR),
                _ => return Err(TraceError::NYIIR),
            }
        }
        let d = self.into_dst(ins.op1 as IRRef)?;
        self.code.fmv_d_x(d, RSCRATCH);
        self.def(d);
        Ok(())
    }

    fn asm_equal(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        if irt_isnum(ins.t()) {
            let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
            let sy = self.fetch_fp(ins.op2 as IRRef, pin(sx))?;
            self.code.feq_d(RSCRATCH, sx, sy);
            // EQ guard: exit if NOT equal (RSCRATCH == 0)
            // NE guard: exit if equal (RSCRATCH == 1)
            if ins.op() == IROp::EQ {
                self.code.beq(RSCRATCH, RZERO, 8); // exit if NE
            } else {
                self.code.bne(RSCRATCH, RZERO, 8); // exit if EQ
            }
            let jal_pos = self.code.len();
            self.fixups.push((jal_pos, self.stubs.len()));
            self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
            self.code.jal(RZERO, 0);
            return Ok(());
        }
        self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);
        self.gpr_load_ref(RSCRATCH2, ins.op2 as IRRef);
        self.code.xor(RSCRATCH, RSCRATCH, RSCRATCH2);
        if ins.op() == IROp::EQ {
            self.code.bne(RSCRATCH, RZERO, 8); // exit if not equal
        } else {
            self.code.beq(RSCRATCH, RZERO, 8); // exit if equal
        }
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
        Ok(())
    }

    // ═══ CONV: type conversion INT↔NUM ═══════════════════════════════════════
    fn asm_conv(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.into_dst(ins.op1 as IRRef)?;
        let tgt = ins.op2 as u8;
        if irt_isnum(tgt) {
            // INT → NUM: pass through (bits are already f64)
            self.code.fsgnj_d(d, sx, sx);
        } else {
            // NUM → INT: FCVT.L.D then FCVT.D.L
            // No guard — just conversion
            self.code.fcvt_l_d(RSCRATCH, sx);
            self.code.fcvt_d_l(d, RSCRATCH);
        }
        self.def(d);
        Ok(())
    }

    // ═══ BSWAP ═══════════════════════════════════════════════════════════
    fn asm_bswap(&mut self, _ins: &IRIns) -> Result<(), TraceError> {
        Err(TraceError::NYIIR)
    }

    fn asm_loop_back(&mut self) -> Result<(), TraceError> {
        for &(phi, rg) in &self.phi_homes {
            if let Some(dr) = self.loc[Self::iidx(phi)] {
                self.code.fsgnj_d(dr, rg, rg);
            }
        }
        self.owner = self.s0;
        let loop_ofs = self.loop_pos.unwrap();
        let offset = loop_ofs as i32 - self.code.len() as i32;
        self.code.jal(RZERO, offset);
        Ok(())
    }

    // ═══ helper_call / CALLL ═════════════════════════════════════════════════
    fn helper_call(&mut self, addr: u64, args: &[IRRef]) {
        let fmsk = -(FRAME_SIZE as i32);
        for (i, &arg) in args.iter().enumerate() {
            let arg_rg: u8 = [10, 11, 12][i];
            self.gpr_load_ref(arg_rg, arg);
        }
        self.code.mov64(RSCRATCH3, addr);
        self.code.sd(RBASE, RSP, fmsk - 8);
        self.code.sd(RENV, RSP, fmsk - 16);
        self.code.sd(RRA, RSP, fmsk - 24);
        self.code.jalr(RRA, RSCRATCH3, 0);
        self.code.ld(RBASE, RSP, fmsk - 8);
        self.code.ld(RENV, RSP, fmsk - 16);
        self.code.ld(RRA, RSP, fmsk - 24);
    }

    fn ff_result(&mut self, _ins: &IRIns) -> Result<(), TraceError> {
        let d = self.alloc(0)?;
        // Result is in a0 (x10) as raw bits from the helper
        self.code.fmv_d_x(d, 10);
        self.def(d);
        Ok(())
    }

    // ═══ HLOAD / HSTORE via helper_call ══════════════════════════════════════
    fn asm_hload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_tget as *const () as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }

    fn asm_hstore(&mut self, ins: &IRIns) {
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        let addr = super::super::exec::jit_tset as *const () as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, carg.op1 as IRRef, carg.op2 as IRRef]);
    }

    // ═══ CALLL ══════════════════════════════════════════════════════════════
    fn asm_calll(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        use super::super::record as rec;
        let idx = ins.op2 as u32;
        let addr = match idx {
            rec::IRCALL_TAB_NEXTK => super::super::exec::jit_tnextk as *const () as u64,
            rec::IRCALL_FMOD => super::super::exec::jit_fmod as *const () as u64,
            rec::IRCALL_STR_LEN => super::super::exec::jit_str_len as *const () as u64,
            rec::IRCALL_STR_CMP => super::super::exec::jit_str_cmp as *const () as u64,
            rec::IRCALL_STR_BYTE => super::super::exec::jit_str_byte as *const () as u64,
            rec::IRCALL_STR_CHAR => super::super::exec::jit_str_char as *const () as u64,
            rec::IRCALL_STR_SUB => super::super::exec::jit_str_sub as *const () as u64,
            rec::IRCALL_TAB_LEN => super::super::exec::jit_alen as *const () as u64,
            rec::IRCALL_TAB_CONCAT => super::super::exec::jit_tconcat as *const () as u64,
            rec::IRCALL_CAT => super::super::exec::jit_cat as *const () as u64,
            rec::IRCALL_USET => super::super::exec::jit_uset as *const () as u64,
            rec::IRCALL_VARG => super::super::exec::jit_varg as *const () as u64,
            _ => return Err(TraceError::NYIIR),
        };
        match super::super::record::ircall_arity(idx) {
            1 => {
                self.helper_call(addr, &[ins.op1 as IRRef]);
            }
            2 => {
                let carg = *self.tr.ir.ir(ins.op1 as IRRef);
                self.helper_call(addr, &[carg.op1 as IRRef, carg.op2 as IRRef]);
            }
            3 => {
                let cargj = *self.tr.ir.ir(ins.op1 as IRRef);
                let cargi = *self.tr.ir.ir(cargj.op1 as IRRef);
                self.helper_call(addr, &[cargi.op1 as IRRef, cargi.op2 as IRRef, cargj.op2 as IRRef]);
            }
            _ => return Err(TraceError::NYIIR),
        }
        self.ff_result(ins)
    }

    // ═══ TNEW / TDUP via helper_call ════════════════════════════════════════
    fn asm_tnew(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_tnew as *const () as u64;
        self.helper_call(addr, &[]);
        self.ff_result(ins)
    }

    fn asm_tdup(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_tdup as *const () as u64;
        self.helper_call(addr, &[ins.op1 as IRRef]);
        self.ff_result(ins)
    }

    fn asm_pow(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_pow as *const () as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }

    fn asm_fmod(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_fmod as *const () as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }

    // ═══ ULOAD: upvalue load ═══════════════════════════════════════════════
    fn asm_uload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        // op1 = constant holding the closed-upvalue cell address
        let bits = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        self.code.mov64(RSCRATCH3, bits);
        // Load the LuaValue (u64) from the cell
        self.code.ld(RSCRATCH, RSCRATCH3, 0);
        let d = self.alloc(0)?;
        self.code.fmv_d_x(d, RSCRATCH);
        self.def(d);
        Ok(())
    }

    // ═══ FLOAD: metatable == nil guard ═════════════════════════════════════
    fn asm_fload_meta(&mut self, ins: &IRIns) {
        const META_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, metatable) as i32;
        self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);
        self.code.mov64(RSCRATCH2, crate::value::LJ_GCVMASK);
        self.code.and(RSCRATCH, RSCRATCH, RSCRATCH2);
        self.code.ld(RSCRATCH2, RSCRATCH, META_OFF);
        self.code.bne(RSCRATCH2, RZERO, 8);
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
    }

    // ═══ ALOAD: inline array load ══════════════════════════════════════════
    fn asm_aload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        const ASIZE_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, asize) as i32;
        self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);  // table
        self.code.mov64(RSCRATCH3, crate::value::LJ_GCVMASK);
        self.code.and(RSCRATCH, RSCRATCH, RSCRATCH3);    // strip GC bits → raw ptr
        // Load key (f64) → int, check
        let sv = self.fetch_fp(ins.op2 as IRRef, 0)?;
        self.code.fcvt_l_d(RSCRATCH2, sv);               // key → i64
        // Guard: key >= 0
        self.code.blt(RSCRATCH2, RZERO, 8);
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
        // Guard: key < asize
        self.code.lwu(RSCRATCH3, RSCRATCH, ASIZE_OFF);
        self.code.bltu(RSCRATCH2, RSCRATCH3, 8);
        let jal_pos2 = self.code.len();
        self.fixups.push((jal_pos2, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
        // Load aptr, index with key*8
        self.code.ld(RSCRATCH, RSCRATCH, APTR_OFF);      // RSCRATCH = aptr
        self.code.slli(RSCRATCH2, RSCRATCH2, 3);         // key * 8
        self.code.add(RSCRATCH, RSCRATCH, RSCRATCH2);    // aptr + key*8
        self.code.ld(RSCRATCH, RSCRATCH, 0);             // load LuaValue
        let d = self.alloc(0)?;
        self.code.fmv_d_x(d, RSCRATCH);
        self.def(d);
        Ok(())
    }

    // ═══ ASTORE: inline array store ════════════════════════════════════════
    fn asm_astore(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        const ASIZE_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, asize) as i32;
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);   // table
        self.code.mov64(RSCRATCH3, crate::value::LJ_GCVMASK);
        self.code.and(RSCRATCH, RSCRATCH, RSCRATCH3);
        // Load key → int
        self.gpr_load_ref(RSCRATCH2, carg.op1 as IRRef);
        self.code.fmv_d_x(RSCRATCH2, RSCRATCH2);
        // Guard: key >= 0 (actually key > 0 for ASTORE since array is 1-based)
        self.code.bne(RSCRATCH2, RZERO, 8);
        // key < asize
        self.code.lwu(RSCRATCH3, RSCRATCH, ASIZE_OFF);
        self.code.bltu(RSCRATCH2, RSCRATCH3, 8);
        let jal_pos = self.code.len();
        self.fixups.push((jal_pos, self.stubs.len()));
        self.stubs.push(Stub { snapidx: self.snapidx, gc: false });
        self.code.jal(RZERO, 0);
        // Store value
        self.gpr_load_ref(RSCRATCH3, carg.op2 as IRRef);
        self.code.ld(RSCRATCH, RSCRATCH, APTR_OFF);
        self.code.slli(RSCRATCH2, RSCRATCH2, 3);
        self.code.add(RSCRATCH, RSCRATCH, RSCRATCH2);
        self.code.sd(RSCRATCH3, RSCRATCH, 0);
        Ok(())
    }

    // ═══ main emit ═════════════════════════════════════════════════════════
    fn emit(&mut self) -> Result<u32, TraceError> {
        // Prologue
        self.code.addi(RSP, RSP, -(FRAME_SIZE as i32) as u32);
        self.code.sd(8, RSP, 0);  self.code.sd(9, RSP, 8);
        self.code.sd(18, RSP, 16); self.code.sd(19, RSP, 24);
        self.code.sd(20, RSP, 32); self.code.sd(21, RSP, 40);
        self.code.sd(22, RSP, 48); self.code.sd(23, RSP, 56);
        self.code.sd(24, RSP, 64); self.code.sd(25, RSP, 72);
        self.code.sd(26, RSP, 80); self.code.sd(27, RSP, 88);
        self.code.fsd(8, RSP, 96);  self.code.fsd(9, RSP, 104);
        self.code.fsd(18, RSP, 112); self.code.fsd(19, RSP, 120);
        self.code.fsd(20, RSP, 128); self.code.fsd(21, RSP, 136);
        self.code.fsd(22, RSP, 144); self.code.fsd(23, RSP, 152);
        self.code.fsd(24, RSP, 160); self.code.fsd(25, RSP, 168);
        self.code.fsd(26, RSP, 176); self.code.fsd(27, RSP, 184);
        self.code.mv(RBASE, 10);
        self.code.mv(RENV, 11);

        let inner = self.code.len();

        // Instruction dispatch
        for r in REF_FIRST..self.tr.ir.nins() {
            self.cur = r;
            let ins = self.tr.ir.ir(r);
            match ins.op() {
                IROp::NOP => {}
                IROp::BASE => {}
                IROp::LOOP => self.asm_loop_head(),
                IROp::SLOAD => self.asm_sload(ins)?,
                IROp::ULOAD => self.asm_uload(ins)?,
                IROp::FLOAD => self.asm_fload_meta(ins),
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::MIN | IROp::MAX => self.asm_arith(ins)?,
                IROp::POW => self.asm_pow(ins)?,
                IROp::MOD => self.asm_fmod(ins)?,
                IROp::NEG => self.asm_neg(ins)?,
                IROp::ABS => self.asm_abs(ins)?,
                IROp::FPMATH => self.asm_fpmath(ins)?,
                IROp::TOBIT => self.asm_tobit(ins)?,
                IROp::LT | IROp::GE | IROp::LE | IROp::GT
                | IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT => self.asm_comp(ins)?,
                IROp::EQ | IROp::NE => self.asm_equal(ins)?,
                IROp::BAND | IROp::BOR | IROp::BXOR | IROp::BSHL | IROp::BSHR
                | IROp::BSAR | IROp::BROL | IROp::BROR | IROp::BNOT => self.asm_bitop(ins)?,
                IROp::BSWAP => self.asm_bswap(ins)?,
                IROp::GCSTEP => self.asm_gcstep(ins),
                IROp::CONV => self.asm_conv(ins)?,
                IROp::CALLL => self.asm_calll(ins)?,
                IROp::HLOAD => self.asm_hload(ins)?,
                IROp::HSTORE => self.asm_hstore(ins),
                IROp::ALOAD => self.asm_aload(ins)?,
                IROp::ASTORE => self.asm_astore(ins)?,
                IROp::TNEW => self.asm_tnew(ins)?,
                IROp::TDUP => self.asm_tdup(ins)?,
                IROp::PHI => {}
                IROp::CARG => {}
                _ => return Err(TraceError::NYIIR),
            }
        }

        // Shared epilogue
        let epilogue = self.code.len();
        self.code.ld(8, RSP, 0);  self.code.ld(9, RSP, 8);
        self.code.ld(18, RSP, 16); self.code.ld(19, RSP, 24);
        self.code.ld(20, RSP, 32); self.code.ld(21, RSP, 40);
        self.code.ld(22, RSP, 48); self.code.ld(23, RSP, 56);
        self.code.ld(24, RSP, 64); self.code.ld(25, RSP, 72);
        self.code.ld(26, RSP, 80); self.code.ld(27, RSP, 88);
        self.code.fld(8, RSP, 96);  self.code.fld(9, RSP, 104);
        self.code.fld(18, RSP, 112); self.code.fld(19, RSP, 120);
        self.code.fld(20, RSP, 128); self.code.fld(21, RSP, 136);
        self.code.fld(22, RSP, 144); self.code.fld(23, RSP, 152);
        self.code.fld(24, RSP, 160); self.code.fld(25, RSP, 168);
        self.code.fld(26, RSP, 176); self.code.fld(27, RSP, 184);
        self.code.addi(RSP, RSP, FRAME_SIZE);
        self.code.ret();

        // Exit stubs: load snapno in a0, jump to epilogue
        for fi in 0..self.fixups.len() {
            let (jal_pos, stub_idx) = self.fixups[fi];
            let exit_ofs = self.code.len();
            Asm::patch_jal(&mut self.code, jal_pos, exit_ofs);
            self.stub_tails.push((stub_idx as u32, exit_ofs as u32));
            let snap = self.stubs[stub_idx].snapidx;
            self.code.mov32(10, snap as u32);
            let epi_off = epilogue as i32 - self.code.len() as i32;
            self.code.jal(RZERO, epi_off);
        }

        Ok(inner as u32)
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

pub fn assemble(
    tr: &GCtrace,
    link: Option<*const u8>,
) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
    let mut a = Asm::new(tr, link)?;
    let inner = a.emit()?;
    let code = a.code.0;
    if code.is_empty() { return Err(TraceError::NYIIR); }
    let mut area = McodeArea::alloc(code.len()).ok_or(TraceError::NYIIR)?;
    area.as_mut_slice()[..code.len()].copy_from_slice(&code);
    area.protect_exec();
    Ok((area, inner, a.stub_tails))
}

pub fn patch_exit(
    area: &mut McodeArea,
    stub_tails: &[(u32, u32)],
    exitno: u32,
    target: *const u8,
) {
    for &(sn, ofs) in stub_tails {
        if sn == exitno {
            // The exit stub tail is a JAL to epilogue.
            // To retarget: replace it with a direct jump to `target`.
            let t = target as usize;
            let o = ofs as usize;
            let code = area.as_mut_slice();
            // Write JAL x0, target at ofs
            let offset = t.wrapping_sub(o) as i32;
            let i = offset as u32;
            let w = ((i >> 20 & 1) << 31)
                  | ((i >> 1 & 0x3FF) << 21)
                  | ((i >> 11 & 1) << 20)
                  | ((i >> 12 & 0xFF) << 12)
                  | (0u32 << 7)
                  | 0b110_1111u32;
            code[o..o + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
}
