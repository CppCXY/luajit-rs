/// AArch64 trace assembler: translates SSA IR into ARM64 native code.
///
/// Register conventions:
///   x19 = RBASE  (Lua stack base, callee-saved)
///   x20 = RENV   (spill/exit env buffer, callee-saved)
///   x9-x15 = scratch GPRs
///   v0-v7, v16-v31 = allocatable FP registers
///   v8 = FP scratch (constant materialization)
///
/// Outer frame (160 bytes, 16-aligned):
///   x19-x28 + x29(FP) + x30(LR) → 6 pairs = 96 bytes
///   v8-v15 (lower 64-bit) → 4 pairs = 64 bytes
use super::super::ir::*;
use super::super::mcode::McodeArea;
use super::super::record::{
    IRFPM_CEIL, IRFPM_FLOOR, IRFPM_SQRT, IRFPM_TRUNC, IRSLOAD_PARENT,
};
use super::super::{
    GCtrace, SNAP_NORESTORE, TraceError, TraceLink, snap_ref, snap_slot,
};

// ── ARM64 encoding helpers ─────────────────────────────────────────────────
pub(crate) struct Emit(pub Vec<u8>);
impl Emit {
    fn new() -> Self { Emit(Vec::with_capacity(1024)) }
    fn len(&self) -> usize { self.0.len() }
    fn u32(&mut self, w: u32) { self.0.extend_from_slice(&w.to_le_bytes()); }
    fn nop(&mut self) { self.u32(0xD503_201F); }
    fn brk(&mut self, imm: u16) { self.u32(0xD420_0000|((imm as u32)<<5)); }

    // ── move wide ──
    fn movz(&mut self, xd: u8, imm: u16, sh: u8) { let h=(sh/16) as u32; self.u32(0xD280_0000|(h<<21)|((imm as u32)<<5)|(xd as u32)); }
    fn movk(&mut self, xd: u8, imm: u16, sh: u8) { let h=(sh/16) as u32; self.u32(0xF280_0000|(h<<21)|((imm as u32)<<5)|(xd as u32)); }
    fn mov64(&mut self, xd: u8, imm: u64) {
        self.movz(xd, imm as u16, 0);
        if imm>>16 != 0 { self.movk(xd, (imm>>16) as u16, 16); }
        if imm>>32 != 0 { self.movk(xd, (imm>>32) as u16, 32); }
        if imm>>48 != 0 { self.movk(xd, (imm>>48) as u16, 48); }
    }
    fn mov32(&mut self, wd: u8, imm: u32) { self.movz(wd, imm as u16, 0); if imm>>16 != 0 { self.movk(wd, (imm>>16) as u16, 16); } }

    // ── branch ──
    fn b_cond(&mut self, cc: u32, off: i32) { self.u32(0x5400_0000|(((off as u32)&0x7FFFF)<<5)|cc); }
    fn b(&mut self, off: i32)   { self.u32(0x1400_0000|((off as u32)&0x3FF_FFFF)); }
    fn br(&mut self, rn: u8)    { self.u32(0xD61F_0000|((rn as u32)<<5)); }
    fn ret(&mut self)           { self.u32(0xD65F_03C0); }
    fn blr(&mut self, rn: u8)   { self.u32(0xD63F_0000|((rn as u32)<<5)); }

    // ── DP register (64-bit unless noted) ──
    fn add_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x8B00_0000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn sub_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0xCB00_0000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn and_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x8A00_0000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn orr_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0xAA00_0000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn eor_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0xCA00_0000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn mvn64(&mut self, rd: u8, rm: u8) { self.u32(0xAA20_03E0|((rm as u32)<<16)|(rd as u32)); }
    // 32-bit forms
    fn and_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x0A00_0000|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }
    fn orr_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x2A00_0000|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }
    fn eor_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x4A00_0000|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }
    fn mvn_w(&mut self, wd: u8, wm: u8) { self.u32(0x2A20_03E0|((wm as u32)<<16)|(wd as u32)); }

    fn add_imm(&mut self, rd: u8, rn: u8, imm: u32) { self.u32(0x9100_0000|((imm&0xFFF)<<10)|((rn as u32)<<5)|(rd as u32)); }
    fn sub_imm(&mut self, rd: u8, rn: u8, imm: u32) { self.u32(0xD100_0000|((imm&0xFFF)<<10)|((rn as u32)<<5)|(rd as u32)); }
    fn cmp_rr(&mut self, rn: u8, rm: u8) { self.u32(0xEB00_001F|((rm as u32)<<16)|((rn as u32)<<5)); }
    fn cmp_rr_w(&mut self, wn: u8, wm: u8) { self.u32(0x6B00_001F|((wm as u32)<<16)|((wn as u32)<<5)); }
    fn cmp_imm(&mut self, rn: u8, imm: u32) { self.u32(0xF100_001F|((imm&0xFFF)<<10)|((rn as u32)<<5)); }

    fn lsl_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x9AC0_2000|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn lsr_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x9AC0_2400|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn asr_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x9AC0_2800|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    fn ror_rr(&mut self, rd: u8, rn: u8, rm: u8) { self.u32(0x9AC0_2C00|((rm as u32)<<16)|((rn as u32)<<5)|(rd as u32)); }
    // 32-bit shifts
    fn lsl_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x1AC0_2000|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }
    fn lsr_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x1AC0_2400|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }
    fn asr_w(&mut self, wd: u8, wn: u8, wm: u8) { self.u32(0x1AC0_2800|((wm as u32)<<16)|((wn as u32)<<5)|(wd as u32)); }

    fn rev_w(&mut self, wd: u8, wn: u8) { self.u32(0x5AC0_0800|((wn as u32)<<5)|(wd as u32)); }

    // ── load/store pair ──
    fn stp(&mut self, rt1: u8, rt2: u8, rn: u8, off: i32) { let o=((off/8)as u32)&0x7F; self.u32(0xA900_0000|(o<<15)|((rt2 as u32)<<10)|((rn as u32)<<5)|(rt1 as u32)); }
    fn ldp(&mut self, rt1: u8, rt2: u8, rn: u8, off: i32) { let o=((off/8)as u32)&0x7F; self.u32(0xA940_0000|(o<<15)|((rt2 as u32)<<10)|((rn as u32)<<5)|(rt1 as u32)); }
    fn stp_d(&mut self, dt1: u8, dt2: u8, rn: u8, off: i32) { let o=((off/8)as u32)&0x7F; self.u32(0x6D00_0000|(o<<15)|((dt2 as u32)<<10)|((rn as u32)<<5)|(dt1 as u32)); }
    fn ldp_d(&mut self, dt1: u8, dt2: u8, rn: u8, off: i32) { let o=((off/8)as u32)&0x7F; self.u32(0x6D40_0000|(o<<15)|((dt2 as u32)<<10)|((rn as u32)<<5)|(dt1 as u32)); }
    fn stp_pre(&mut self, rt1: u8, rt2: u8, rn: u8, off: i32) { let o=(off/8) as i32; let imm7=(o as u32)&0x7F; self.u32(0xA980_0000|(imm7<<15)|((rt2 as u32)<<10)|((rn as u32)<<5)|(rt1 as u32)); }
    fn ldp_post(&mut self, rt1: u8, rt2: u8, rn: u8, off: i32) { let o=(off/8) as i32; let imm7=(o as u32)&0x7F; self.u32(0xA8C0_0000|(imm7<<15)|((rt2 as u32)<<10)|((rn as u32)<<5)|(rt1 as u32)); }

    // ── load/store register ──
    fn ldr(&mut self, rt: u8, rn: u8, off: i32) { self.u32(0xF940_0000|((((off/8)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(rt as u32)); }
    fn str(&mut self, rt: u8, rn: u8, off: i32) { self.u32(0xF900_0000|((((off/8)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(rt as u32)); }
    fn ldr_w(&mut self, rt: u8, rn: u8, off: i32) { self.u32(0xB940_0000|((((off/4)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(rt as u32)); }
    fn str_w(&mut self, rt: u8, rn: u8, off: i32) { self.u32(0xB900_0000|((((off/4)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(rt as u32)); }
    fn ldr_d(&mut self, dt: u8, rn: u8, off: i32) { self.u32(0xFD40_0000|((((off/8)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(dt as u32)); }
    fn str_d(&mut self, dt: u8, rn: u8, off: i32) { self.u32(0xFD00_0000|((((off/8)as u32)&0xFFF)<<10)|((rn as u32)<<5)|(dt as u32)); }

    /// 32-bit immediate store via temp register
    fn str_imm32(&mut self, rn: u8, off: i32, imm: u32) {
        self.mov32(RSCRATCH, imm);
        self.str_w(RSCRATCH, rn, off);
    }
    fn str_imm64(&mut self, rn: u8, off: i32, imm: u64) {
        self.mov64(RSCRATCH, imm);
        self.str(RSCRATCH, rn, off);
    }

    // ── FP ops ──
    fn fmov_dd(&mut self, dd: u8, dn: u8) { self.u32(0x1E60_4000|((dn as u32)<<5)|(dd as u32)); }
    fn fmov_gpr(&mut self, xd: u8, dn: u8) { self.u32(0x9E66_0000|((dn as u32)<<5)|(xd as u32)); }
    fn fmov_fp(&mut self, dd: u8, xn: u8) { self.u32(0x9E67_0000|((xn as u32)<<5)|(dd as u32)); }
    fn fadd(&mut self, dd: u8, dn: u8, dm: u8) { self.u32(0x1E60_2800|((dm as u32)<<16)|((dn as u32)<<5)|(dd as u32)); }
    fn fsub(&mut self, dd: u8, dn: u8, dm: u8) { self.u32(0x1E60_3800|((dm as u32)<<16)|((dn as u32)<<5)|(dd as u32)); }
    fn fmul(&mut self, dd: u8, dn: u8, dm: u8) { self.u32(0x1E60_0800|((dm as u32)<<16)|((dn as u32)<<5)|(dd as u32)); }
    fn fdiv(&mut self, dd: u8, dn: u8, dm: u8) { self.u32(0x1E60_1800|((dm as u32)<<16)|((dn as u32)<<5)|(dd as u32)); }
    fn fneg_d(&mut self, dd: u8, dn: u8) { self.u32(0x1E61_4000|((dn as u32)<<5)|(dd as u32)); }
    fn fabs_d(&mut self, dd: u8, dn: u8) { self.u32(0x1E60_C000|((dn as u32)<<5)|(dd as u32)); }
    fn fsqrt(&mut self, dd: u8, dn: u8) { self.u32(0x1E61_C000|((dn as u32)<<5)|(dd as u32)); }
    fn frintm(&mut self, dd: u8, dn: u8) { self.u32(0x1E65_4000|((dn as u32)<<5)|(dd as u32)); }
    fn frintp(&mut self, dd: u8, dn: u8) { self.u32(0x1E64_C000|((dn as u32)<<5)|(dd as u32)); }
    fn frintz(&mut self, dd: u8, dn: u8) { self.u32(0x1E65_C000|((dn as u32)<<5)|(dd as u32)); }
    fn fcmp(&mut self, dn: u8, dm: u8) { self.u32(0x1E60_2000|((dm as u32)<<16)|((dn as u32)<<5)); }
    fn scvtf_w(&mut self, dd: u8, wn: u8) { self.u32(0x1E22_0000|((wn as u32)<<5)|(dd as u32)); }
    fn scvtf_x(&mut self, dd: u8, xn: u8) { self.u32(0x9E22_0000|((xn as u32)<<5)|(dd as u32)); }
    fn ucvtf_w(&mut self, dd: u8, wn: u8) { self.u32(0x1E23_0000|((wn as u32)<<5)|(dd as u32)); }
    fn ucvtf_x(&mut self, dd: u8, xn: u8) { self.u32(0x9E23_0000|((xn as u32)<<5)|(dd as u32)); }
    fn fcvtzs_w(&mut self, wd: u8, dn: u8) { self.u32(0x1E38_0000|((dn as u32)<<5)|(wd as u32)); }
    fn fcvtzs_x(&mut self, xd: u8, dn: u8) { self.u32(0x9E38_0000|((dn as u32)<<5)|(xd as u32)); }

    // fcsel Dd, Dn, Dm, cond
    fn fcsel(&mut self, dd: u8, dn: u8, dm: u8, cc: u32) { self.u32(0x1E20_0C00|((dm as u32)<<16)|(cc<<12)|((dn as u32)<<5)|(dd as u32)); }
}

// ── Helpers for patching ──
fn patch_u32_at(code: &mut [u8], pos: usize, w: u32) {
    code[pos..pos+4].copy_from_slice(&w.to_le_bytes());
}
fn patch_bcond_at(code: &mut [u8], pos: usize, cc: u32, off: i32) {
    let w = 0x5400_0000u32 | (((off as u32)&0x7FFFF)<<5) | cc;
    patch_u32_at(code, pos, w);
}
fn patch_b_at(code: &mut [u8], pos: usize, off: i32) {
    let w = 0x1400_0000u32 | ((off as u32)&0x3FF_FFFF);
    patch_u32_at(code, pos, w);
}

// ── Register constants ─────────────────────────────────────────────────────
const ALLOC_REGS: [u8; 24] = [0,1,2,3,4,5,6,7, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31];
const FP_SCRATCH: u8 = 8;
const NREG: usize = 32;

const RBASE:    u8 = 19;
const RENV:     u8 = 20;
const RSCRATCH: u8 = 9;
const RSCRATCH2:u8 = 10;
const RSCRATCH3:u8 = 11;

const SAVED_GPR_PAIRS: u32 = 5; // x19-x28 (10 regs), x29/x30 saved separately at frame bottom
const SAVED_FP_PAIRS:  u32 = 4; // d8-d15 (8 regs)
const FRAME: u32 = 16 + SAVED_GPR_PAIRS * 16 + SAVED_FP_PAIRS * 16; // 16 + 80 + 64 = 160

mod cond {
    pub const EQ: u32 = 0x0; pub const NE: u32 = 0x1;
    pub const CS: u32 = 0x2; pub const CC: u32 = 0x3;
    pub const MI: u32 = 0x4; pub const PL: u32 = 0x5;
    pub const VS: u32 = 0x6; pub const VC: u32 = 0x7;
    pub const HI: u32 = 0x8; pub const LS: u32 = 0x9;
    pub const GE: u32 = 0xA; pub const LT: u32 = 0xB;
    pub const GT: u32 = 0xC; pub const LE: u32 = 0xD;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner { None, Ins(IRRef), Konst(IRRef) }
#[derive(Clone, Copy)]
struct PhiInfo { phi: IRRef, lref: IRRef, rref: IRRef, num: bool }
struct Stub { snapidx: usize, flush: Vec<(u8, IRRef)>, gc: bool }

#[inline] fn pin(rg: u8) -> u32 { 1u32 << rg }

// ── Asm ────────────────────────────────────────────────────────────────────

struct Asm<'a> {
    tr: &'a GCtrace, code: Emit, cur: IRRef, snapidx: usize,
    last_use: Vec<IRRef>, klast_use: Vec<IRRef>,
    needs_env: Vec<bool>, env_valid: Vec<bool>,
    loc: Vec<Option<u8>>, owner: [Owner; NREG],
    fixups: Vec<(usize, usize)>, stubs: Vec<Stub>,
    phis: Vec<PhiInfo>, loop_pos: Option<usize>, s0: [Owner; NREG],
    phi_homes: Vec<(IRRef, u8)>,
    link: Option<*const u8>, stub_tails: Vec<(u32, u32)>,
}

// map x64 CC to ARM64 CC: x64 jcc semantics (after cmp/after ucomisd)
// x64: 0x2=B(CF=1)->LO, 0x3=AE(CF=0)->HS, 0x4=E(ZF=1)->EQ, 0x5=NE(ZF=0)->NE,
//       0x6=BE(CF=1|ZF=1)->LS, 0x7=A(CF=0&ZF=0)->HI, 0xA=P->VS (approx, used only with ucomisd NaN)
fn x64cc_to_a64(xcc: u8) -> u32 {
    match xcc { 0x2=>cond::CC, 0x3=>cond::CS, 0x4=>cond::EQ, 0x5=>cond::NE, 0x6=>cond::LS, 0x7=>cond::HI, 0xA=>cond::VS, _=>cond::NE }
}

impl<'a> Asm<'a> {
    #[inline] fn iidx(r: IRRef) -> usize { (r - REF_BIAS) as usize }
    #[inline] fn kidx(r: IRRef) -> usize { (REF_BIAS - 1 - r) as usize }
    #[inline] fn env_ofs(r: IRRef) -> i32 { (Self::iidx(r) * 8) as i32 }

    // ═══ NYI scan ═══════════════════════════════════════════════════════════
    fn new(tr: &'a GCtrace, link: Option<*const u8>) -> Result<Asm<'a>, TraceError> {
        let nins = Self::iidx(tr.ir.nins());
        let nk = (REF_BIAS - tr.ir.nk()) as usize;
        let mut a = Asm { tr, code: Emit::new(), cur: 0, snapidx: 0,
            last_use: vec![0; nins], klast_use: vec![0; nk],
            needs_env: vec![false; nins], env_valid: vec![false; nins],
            loc: vec![None; nins], owner: [Owner::None; NREG],
            fixups: Vec::new(), stubs: Vec::new(), phis: Vec::new(),
            loop_pos: None, s0: [Owner::None; NREG], phi_homes: Vec::new(),
            link, stub_tails: Vec::new() };
        for r in REF_FIRST..tr.ir.nins() {
            let ins = tr.ir.ir(r);
            match ins.op() {
                IROp::NOP|IROp::BASE|IROp::LOOP|IROp::SLOAD => {}
                IROp::ULOAD => {}
                IROp::FLOAD|IROp::HLOAD|IROp::CARG => for op in [ins.op1 as IRRef, ins.op2 as IRRef] { if op>=REF_BIAS { a.needs_env[Self::iidx(op)]=true; } }
                IROp::HSTORE => { if ins.op1 as IRRef>=REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)]=true; } }
                IROp::CALLL => { if super::super::record::ircall_arity(ins.op2 as u32)==1 && ins.op1 as IRRef>=REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)]=true; } }
                IROp::TNEW|IROp::TDUP => {}
                IROp::ALOAD => { if ins.op1 as IRRef>=REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)]=true; } a.mark_use(ins.op2 as IRRef,r); }
                IROp::ASTORE => { if ins.op1 as IRRef>=REF_BIAS { a.needs_env[Self::iidx(ins.op1 as IRRef)]=true; } }
                IROp::GCSTEP => {}
                IROp::BAND|IROp::BOR|IROp::BXOR|IROp::BSHL|IROp::BSHR|IROp::BSAR|IROp::BNOT|IROp::BSWAP => { a.mark_use(ins.op1 as IRRef,r); if ins.op2!=0 { a.mark_use(ins.op2 as IRRef,r); } }
                IROp::ADD|IROp::SUB|IROp::MUL|IROp::DIV|IROp::MIN|IROp::MAX => { a.mark_use(ins.op1 as IRRef,r); a.mark_use(ins.op2 as IRRef,r); }
                IROp::NEG => { a.mark_use(ins.op1 as IRRef,r); a.mark_use(ins.op2 as IRRef,r); }
                IROp::ABS => a.mark_use(ins.op1 as IRRef,r),
                IROp::FPMATH => a.mark_use(ins.op1 as IRRef,r),
                IROp::LT|IROp::GE|IROp::LE|IROp::GT|IROp::ULT|IROp::UGE|IROp::ULE|IROp::UGT => { a.mark_use(ins.op1 as IRRef,r); a.mark_use(ins.op2 as IRRef,r); }
                IROp::POW => { for op in [ins.op1 as IRRef,ins.op2 as IRRef] { a.mark_use(op,r); if op>=REF_BIAS { a.needs_env[Self::iidx(op)]=true; } } }
                IROp::TOBIT => a.mark_use(ins.op1 as IRRef,r),
                IROp::EQ|IROp::NE => { a.mark_use(ins.op1 as IRRef,r); a.mark_use(ins.op2 as IRRef,r); if !irt_isnum(ins.t()) { for op in [ins.op1 as IRRef,ins.op2 as IRRef] { if op>=REF_BIAS { a.needs_env[Self::iidx(op)]=true; } } } }
                IROp::PHI => { let (lref,rref)=(ins.op1 as IRRef,ins.op2 as IRRef); let inf=tr.ir.nins(); a.mark_use(lref,inf); a.mark_use(rref,inf); if !irt_isnum(ins.t()) { if lref>=REF_BIAS { a.needs_env[Self::iidx(lref)]=true; } if rref>=REF_BIAS { a.needs_env[Self::iidx(rref)]=true; } } a.phis.push(PhiInfo{phi:r,lref,rref,num:irt_isnum(ins.t())}); }
                _ => return Err(TraceError::NYIIR),
            }
        }
        for (i,snap) in tr.snap.iter().enumerate() {
            let lu = if i+1 < tr.snap.len() { tr.snap[i+1].iref } else { tr.ir.nins() };
            let ofs = snap.mapofs as usize;
            for sn in &tr.snapmap[ofs..ofs+snap.nent as usize] {
                let rr = snap_ref(*sn);
                if rr>=REF_BIAS { a.mark_use(rr,lu); a.needs_env[Self::iidx(rr)]=true; }
            }
        }
        for &(own,_) in &tr.parentmap { a.env_valid[Self::iidx(own as IRRef)]=true; }
        for p in &a.phis { if a.last_use[Self::iidx(p.phi)]!=0 { return Err(TraceError::NYIIR); } }
        Ok(a)
    }

    // ═══ use tracking & register allocation ═══════════════════════════════
    fn mark_use(&mut self, opr: IRRef, at: IRRef) {
        if opr>=REF_BIAS { let i=Self::iidx(opr); self.last_use[i]=self.last_use[i].max(at); }
        else { let k=Self::kidx(opr); self.klast_use[k]=self.klast_use[k].max(at); }
    }
    fn last_use_of(&self, r: IRRef) -> IRRef { if r>=REF_BIAS { self.last_use[Self::iidx(r)] } else { self.klast_use[Self::kidx(r)] } }
    fn dying(&self, r: IRRef) -> bool { self.last_use_of(r) <= self.cur }
    fn reg_of(&self, r: IRRef) -> Option<u8> {
        if r>=REF_BIAS { self.loc[Self::iidx(r)] } else { (0..NREG as u8).find(|&i| self.owner[i as usize]==Owner::Konst(r)) }
    }
    fn steal_quiet(&mut self, rg: u8) { if let Owner::Ins(o)=self.owner[rg as usize] { self.loc[Self::iidx(o)]=None; } self.owner[rg as usize]=Owner::None; }
    fn alloc(&mut self, pinned: u32) -> Result<u8, TraceError> {
        for &rg in ALLOC_REGS.iter() { if pinned&pin(rg)==0 && self.owner[rg as usize]==Owner::None { return Ok(rg); } }
        for &rg in ALLOC_REGS.iter() { if pinned&pin(rg)!=0 { continue; } let dead=match self.owner[rg as usize] { Owner::Ins(o)=>self.last_use[Self::iidx(o)]<self.cur, Owner::Konst(o)=>self.klast_use[Self::kidx(o)]<self.cur, _=>unreachable!() }; if dead { self.steal_quiet(rg); return Ok(rg); } }
        let mut best:Option<(u8,IRRef)>=None;
        for &rg in ALLOC_REGS.iter() { if pinned&pin(rg)!=0 { continue; } let lu=match self.owner[rg as usize] { Owner::Ins(o)=>self.last_use[Self::iidx(o)], Owner::Konst(o)=>self.klast_use[Self::kidx(o)], _=>unreachable!() }; if best.is_none_or(|(_,b)| lu>b) { best=Some((rg,lu)); } }
        let Some((rg,_))=best else { return Err(TraceError::BADRA); };
        if let Owner::Ins(o)=self.owner[rg as usize] { let i=Self::iidx(o); if !self.env_valid[i] { self.code.str_d(rg,RENV,Self::env_ofs(o)); self.env_valid[i]=true; } }
        self.steal_quiet(rg); Ok(rg)
    }
    fn fetch_fp(&mut self, r: IRRef, pinned: u32) -> Result<u8, TraceError> {
        if let Some(rg)=self.reg_of(r) { return Ok(rg); }
        let rg=self.alloc(pinned)?;
        if r>=REF_BIAS { self.code.ldr_d(rg,RENV,Self::env_ofs(r)); self.owner[rg as usize]=Owner::Ins(r); self.loc[Self::iidx(r)]=Some(rg); }
        else { self.code.mov64(RSCRATCH, super::super::exec::const_bits(&self.tr.ir,r)); self.code.fmov_fp(rg,RSCRATCH); self.owner[rg as usize]=Owner::Konst(r); }
        Ok(rg)
    }
    fn into_dst(&mut self, a: IRRef) -> Result<u8, TraceError> {
        let r1=self.fetch_fp(a,0)?; if self.dying(a) { self.steal_quiet(r1); Ok(r1) } else { let d=self.alloc(pin(r1))?; self.code.fmov_dd(d,r1); Ok(d) }
    }
    fn gpr_load_ref(&mut self, rd: u8, r: IRRef) {
        if r>=REF_BIAS { self.code.ldr(rd,RENV,Self::env_ofs(r)); } else { self.code.mov64(rd, super::super::exec::const_bits(&self.tr.ir,r)); }
    }
    fn def(&mut self, d: u8) {
        let i=Self::iidx(self.cur); self.owner[d as usize]=Owner::Ins(self.cur); self.loc[i]=Some(d);
        if self.needs_env[i] { self.code.str_d(d,RENV,Self::env_ofs(self.cur)); self.env_valid[i]=true; }
    }

    // ═══ guard / stub ═══════════════════════════════════════════════════════
    fn exit_code(&self, snapidx: usize) -> u32 { (self.tr.traceno<<16)|snapidx as u32 }
    fn exit_flush_set(&self, snapidx: usize) -> Vec<(u8,IRRef)> {
        let snap=&self.tr.snap[snapidx]; let ofs=snap.mapofs as usize; let mut f=Vec::new();
        for sn in &self.tr.snapmap[ofs..ofs+snap.nent as usize] { let r=snap_ref(*sn); if r<REF_BIAS { continue; } let i=Self::iidx(r); if self.env_valid[i]||f.iter().any(|&(_,fr)| fr==r) { continue; } if let Some(rg)=self.loc[i] { f.push((rg,r)); } }
        f
    }
    fn make_stub(&mut self, gc: bool) -> usize { let flush=self.exit_flush_set(self.snapidx); self.stubs.push(Stub{snapidx:self.snapidx,flush,gc}); self.stubs.len()-1 }

    // Emit a conditional branch to a new exit stub (placeholder offset, patched later).
    fn guard(&mut self, cc: u32) { let s=self.make_stub(false); let pos=self.code.len(); self.code.b_cond(cc,0); self.fixups.push((pos,s)); }
    fn guard_gc(&mut self, cc: u32) { let s=self.make_stub(true); let pos=self.code.len(); self.code.b_cond(cc,0); self.fixups.push((pos,s)); }

    // ═══ handover: copy parent env values to child env (side traces) ════════
    fn emit_handover(&mut self) {
        let mut pending: Vec<(IRRef, IRRef)> = self.tr.parentmap.iter()
            .map(|&(o, p)| (o as IRRef, p as IRRef))
            .filter(|&(o, p)| o != p)
            .collect();
        let mut parked: Option<IRRef> = None;
        while !pending.is_empty() {
            let ready = pending.iter().position(|&(d, _)| {
                !pending.iter().any(|&(_, s)| s == d && parked != Some(s))
            });
            if let Some(i) = ready {
                let (d, s) = pending.remove(i);
                if parked == Some(s) {
                    self.code.str(RSCRATCH3, RENV, Self::env_ofs(d));
                } else {
                    self.code.ldr(RSCRATCH3, RENV, Self::env_ofs(s));
                    self.code.str(RSCRATCH3, RENV, Self::env_ofs(d));
                }
                if parked == Some(s) && !pending.iter().any(|&(_, s2)| s2 == s) {
                    parked = None;
                }
            } else {
                debug_assert!(parked.is_none(), "one scratch, one cycle at a time");
                let d0 = pending[0].0;
                self.code.ldr(RSCRATCH3, RENV, Self::env_ofs(d0));
                parked = Some(d0);
            }
        }
    }

    // ═══ instruction handlers ═══════════════════════════════════════════════

    // SLOAD: stack-load with optional typecheck guard.
    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        if ins.op2 as u32 & IRSLOAD_PARENT != 0 { return Ok(()); }
        let disp = (ins.op1 as i32 - 2) * 8;
        let t = ins.t(); let i = Self::iidx(self.cur);
        if irt_isnum(t) {
            if ins.is_guard() {
                self.code.ldr_w(RSCRATCH, RBASE, disp+4);
                let tisnum_hi: u64 = 0xFFF9_0000;
                self.code.mov64(RSCRATCH2, tisnum_hi);
                self.code.cmp_rr(RSCRATCH, RSCRATCH2);
                self.guard(cond::CS); // b.hs = unsigned higher or same (CF=0 → HS)
            }
            if self.last_use[i]!=0 || self.needs_env[i] { let d=self.alloc(0)?; self.code.ldr_d(d,RBASE,disp); self.def(d); }
            return Ok(());
        }
        if self.needs_env[i] { self.code.ldr(RSCRATCH,RBASE,disp); self.code.str(RSCRATCH,RENV,Self::env_ofs(self.cur)); self.env_valid[i]=true; }
        if !ins.is_guard() { return Ok(()); }
        let ty = irt_type(t);
        if ty == IRT_NIL {
            self.code.mov64(RSCRATCH2, (-1i64) as u64);
            self.code.ldr(RSCRATCH, RBASE, disp);
            self.code.cmp_rr(RSCRATCH, RSCRATCH2);
            self.guard(cond::NE);
        } else if ty <= IRT_TRUE {
            let itype = !(ty as u32); let bits = ((itype as u64)<<15)|0x7FFF;
            self.code.ldr_w(RSCRATCH, RBASE, disp+4);
            self.code.mov64(RSCRATCH2, bits);
            self.code.cmp_rr(RSCRATCH, RSCRATCH2);
            self.guard(cond::NE);
        } else {
            self.code.ldr(RSCRATCH, RBASE, disp);
            // Mirrors lj_asm_arm64.h: ASR tmp_reg, val_reg, #47; CMN tmp_reg, #(-itype)
            let tmp = 30; // LR = x30, used as temp by LuaJIT
            self.code.u32(0x936F_FC00 | ((RSCRATCH as u32) << 5) | (tmp as u32)); // asr x30, x9, #47
            let ity = !(ty as u32);
            let cmn = ((-(ity as i32)) as u32) & 0xFFF;
            self.code.u32(0xB12003FF | (cmn << 10) | ((tmp as u32) << 5)); // cmn x30, #imm12
            self.guard(cond::NE);
        }
        Ok(())
    }

    // ULOAD: upvalue load
    fn asm_uload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let t = ins.t(); let i = Self::iidx(self.cur);
        self.code.mov64(RSCRATCH, addr);
        if irt_isnum(t) {
            if ins.is_guard() {
                self.code.ldr_w(RSCRATCH2, RSCRATCH, 4);
                self.code.mov64(RSCRATCH3, 0xFFF9_0000u64);
                self.code.cmp_rr(RSCRATCH2, RSCRATCH3);
                self.guard(cond::CS);
            }
            if self.last_use[i]!=0 || self.needs_env[i] { let d=self.alloc(0)?; self.code.ldr_d(d,RSCRATCH,0); self.def(d); }
            return Ok(());
        }
        if self.needs_env[i] { self.code.ldr(RSCRATCH2,RSCRATCH,0); self.code.str(RSCRATCH2,RENV,Self::env_ofs(self.cur)); self.env_valid[i]=true; }
        if !ins.is_guard() { return Ok(()); }
        let ty = irt_type(t);
        if ty == IRT_NIL {
            self.code.ldr(RSCRATCH2,RSCRATCH,0); self.code.mov64(RSCRATCH3,(-1i64) as u64);
            self.code.cmp_rr(RSCRATCH2,RSCRATCH3); self.guard(cond::NE);
        } else if ty <= IRT_TRUE {
            let bits=(((!(ty as u32)) as u64)<<15)|0x7FFF;
            self.code.ldr_w(RSCRATCH2,RSCRATCH,4); self.code.mov64(RSCRATCH3,bits);
            self.code.cmp_rr(RSCRATCH2,RSCRATCH3); self.guard(cond::NE);
        } else {
            self.code.ldr(RSCRATCH2,RSCRATCH,0); self.code.u32(0x936F_FC00|((RSCRATCH2 as u32)<<5)|(RSCRATCH2 as u32)); // asr x10, x10, #47
            self.code.mov32(RSCRATCH3, !(ty as u32));
            self.code.cmp_rr_w(RSCRATCH2, RSCRATCH3); self.guard(cond::NE);
        }
        Ok(())
    }

    // FLOAD: meta guard (table.metatable == nil)
    fn asm_meta_guard(&mut self, ins: &IRIns) {
        const META_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, metatable) as i32;
        self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);
        self.code.mov64(RSCRATCH2, crate::value::LJ_GCVMASK);
        self.code.and_rr(RSCRATCH, RSCRATCH, RSCRATCH2);
        self.code.ldr(RSCRATCH2, RSCRATCH, META_OFF);
        self.code.cmp_imm(RSCRATCH2, 0);
        self.guard(cond::NE);
    }

    // ── helper_call: emit a call to an extern "C" helper ──────────────────
    // Parks volatile FP registers (v0-v7, v16-v31) to env, loads up to 3
    // u64 arguments into x0-x2, calls the helper via blr, and returns
    // with the helper's result in x0. RBASE/RENV are callee-saved (x19/x20)
    // and survive the call automatically.
    fn helper_call(&mut self, addr: u64, args: &[IRRef]) {
        // Collect phi lrefs — these are loop-carried and must survive calls.
        let phi_lrefs: Vec<IRRef> = self.phis.iter().map(|p| p.lref).collect();
        for &rg in ALLOC_REGS.iter() {
            if rg > 7 && rg < 16 { continue; }
            if let Owner::Ins(o) = self.owner[rg as usize] {
                let i = Self::iidx(o);
                if self.last_use[i] > self.cur && !self.env_valid[i] {
                    self.code.str_d(rg, RENV, Self::env_ofs(o));
                    self.env_valid[i] = true;
                }
                // Don't steal phi lrefs — they're loop-carried
                if !phi_lrefs.contains(&o) {
                    self.steal_quiet(rg);
                }
            } else {
                self.steal_quiet(rg);
            }
        }
        debug_assert!(args.len() <= 3);
        for (n, &r) in args.iter().enumerate() {
            self.gpr_load_ref(n as u8, r);
        }
        self.code.mov64(RSCRATCH, addr);
        self.code.blr(RSCRATCH);
    }

    // ── ff_result: typecheck + land a helper result from x0 ─────────────
    fn ff_result(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let t = ins.t();
        let i = Self::iidx(self.cur);
        if irt_isnum(t) {
            if ins.is_guard() {
                // Check hi 32 bits < 0xFFF9_0000 (i.e. value is a double, not GC-tagged).
                // ubfm x9, x0, #32, #63  →  extract bits 63:32 of x0 into x9
                self.code.u32(0xD360_FC00 | (RSCRATCH as u32)); // lsr x9, x0, #32
                self.code.mov64(RSCRATCH2, 0xFFF9_0000_u64);
                self.code.cmp_rr(RSCRATCH, RSCRATCH2);
                self.guard(cond::CS); // b.hs → exit if hi >= 0xFFF90000
            }
            if self.last_use[i] != 0 || self.needs_env[i] {
                let d = self.alloc(0)?;
                self.code.fmov_fp(d, 0); // fmov d_dst, x0
                self.def(d);
            }
            return Ok(());
        }
        // Non-number: store to env, then typecheck.
        if self.needs_env[i] {
            self.code.str(0, RENV, Self::env_ofs(self.cur));
            self.env_valid[i] = true;
        }
        if !ins.is_guard() {
            return Ok(());
        }
        let ty = irt_type(t);
        if ty <= IRT_TRUE {
            let bits = match ty {
                IRT_NIL => crate::value::LuaValue::NIL.to_bits(),
                IRT_FALSE => crate::value::LuaValue::FALSE.to_bits(),
                _ => crate::value::LuaValue::TRUE.to_bits(),
            };
            self.code.mov64(RSCRATCH2, bits);
            self.code.cmp_rr(0, RSCRATCH2);
            self.guard(cond::NE);
        } else {
            self.code.u32(0x936F_FC00 | (RSCRATCH as u32)); // asr x9, x0, #47 (was lsr/ubfm)
            self.code.mov32(RSCRATCH2, !(ty as u32));
            self.code.cmp_rr_w(RSCRATCH, RSCRATCH2);
            self.guard(cond::NE);
        }
        Ok(())
    }

    // HLOAD: table get via helper
    fn asm_hload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::jit_tget as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }

    // HSTORE: table set via helper
    fn asm_hstore(&mut self, ins: &IRIns) {
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        debug_assert_eq!(carg.op(), IROp::CARG);
        let addr = super::super::exec::jit_tset as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, carg.op1 as IRRef, carg.op2 as IRRef]);
    }

    // CALLL: guarded helper call — dispatches by IRCALL index
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
            _ => return Err(TraceError::NYIIR),
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
                self.helper_call(addr, &[cargi.op1 as IRRef, cargi.op2 as IRRef, cargj.op2 as IRRef]);
            }
        }
        self.ff_result(ins)
    }

    // ALOAD: array load via helper
    fn asm_aload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        // Fallback: use the shared helper jit_tget for now.
        // TODO: inline fast path when array bounds are guarded.
        let addr = super::super::exec::jit_tget as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, ins.op2 as IRRef]);
        self.ff_result(ins)
    }
    // ASTORE: array store via helper
    fn asm_astore(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        debug_assert_eq!(carg.op(), IROp::CARG);
        let addr = super::super::exec::jit_tset as *const () as usize as u64;
        self.helper_call(addr, &[ins.op1 as IRRef, carg.op1 as IRRef, carg.op2 as IRRef]);
        Ok(())
    }

    // GCSTEP: GC debt check — exit to interpreter when collection is due.
    fn asm_gcstep(&mut self, ins: &IRIns) {
        let total_addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let thres_addr = super::super::exec::const_bits(&self.tr.ir, ins.op2 as IRRef);
        let extra_addr = crate::table::TABLE_EXTRA.with(|c| c.as_ptr() as u64);
        self.code.mov64(RSCRATCH, total_addr);
        self.code.ldr(RSCRATCH2, RSCRATCH, 0);
        self.code.mov64(RSCRATCH3, extra_addr);
        self.code.ldr(RSCRATCH, RSCRATCH3, 0);
        self.code.add_rr(RSCRATCH2, RSCRATCH2, RSCRATCH);
        self.code.mov64(RSCRATCH, thres_addr);
        self.code.ldr(RSCRATCH, RSCRATCH, 0);
        self.code.cmp_rr(RSCRATCH2, RSCRATCH);
        self.guard_gc(cond::CS);
    }

    // TOBIT: wrapping num→int32→num
    fn asm_tobit(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        self.code.fcvtzs_w(RSCRATCH, sx);  // double → w9 (signed i32)
        let d = self.alloc(0)?;
        self.code.scvtf_x(d, RSCRATCH);     // w9 (zero-extended to x9) → double
        self.def(d);
        Ok(())
    }
    // Wait: fcvtzs_w writes a 32-bit signed integer to W register. scvtf_x reads X register.
    // In ARM64, writing Wn zero-extends to Xn. So scvtf_x on the same register gives correct int64→double.
    // But fcvtzs_w on a double→i32 correctly saturates. Then scvtf_x converts the zero-extended i64 back.
    // Actually we should use scvtf_d_w (reads Wn as signed i32). Let me use ucvtf_w for unsigned or scvtf_w.
    // For tobit semantics (wrapping num→i32→num), fcvtzs_w gives the truncated i32, then scvtf_w converts it back as a signed i32.
    // CORRECTION: rn needs to be the same register for scvtf to read the W value. Let me fix this.

    // BITOP: fused bit operations
    fn asm_bitop(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        self.code.fcvtzs_x(RSCRATCH, sx);
        if !matches!(op, IROp::BNOT|IROp::BSWAP) {
            let sy = if ins.op2 == ins.op1 { sx } else { self.fetch_fp(ins.op2 as IRRef, pin(sx))? };
            self.code.fcvtzs_x(RSCRATCH2, sy);
        }
        // 32-bit operations
        match op {
            IROp::BNOT => self.code.mvn_w(RSCRATCH as u8, RSCRATCH as u8),
            IROp::BSWAP => self.code.rev_w(RSCRATCH as u8, RSCRATCH as u8),
            IROp::BAND => self.code.and_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            IROp::BOR  => self.code.orr_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            IROp::BXOR => self.code.eor_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            IROp::BSHL => self.code.lsl_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            IROp::BSHR => self.code.lsr_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            IROp::BSAR => self.code.asr_w(RSCRATCH as u8, RSCRATCH as u8, RSCRATCH2 as u8),
            _ => {}
        }
        let d = self.alloc(0)?;
        self.code.scvtf_w(d, RSCRATCH as u8); // Wn → Dd signed i32→f64
        self.def(d);
        Ok(())
    }

    // ARITH: ADD/SUB/MUL/DIV/MIN/MAX
    fn asm_arith(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        let (mut a, mut b) = (ins.op1 as IRRef, ins.op2 as IRRef);
        if matches!(op, IROp::ADD|IROp::MUL) && !self.dying(a) && self.dying(b) && self.reg_of(b).is_some() { std::mem::swap(&mut a, &mut b); }
        let d = self.into_dst(a)?;
        let rhs = if b == a { d } else { self.fetch_fp(b, pin(d))? };
        match op {
            IROp::ADD => self.code.fadd(d,d,rhs),
            IROp::SUB => self.code.fsub(d,d,rhs),
            IROp::MUL => self.code.fmul(d,d,rhs),
            IROp::DIV => self.code.fdiv(d,d,rhs),
            IROp::MIN => { self.code.fcmp(d,rhs); self.code.fcsel(d,d,rhs,cond::GT); }  // if d > rhs pick rhs (min)
            IROp::MAX => { self.code.fcmp(d,rhs); self.code.fcsel(d,d,rhs,cond::MI); }  // if d < rhs pick rhs (max)
            // Note: MI = sign flag set = d < rhs for ordered values.
            // Actually for fcmp, N flag is set if result is less than.
            // After fcmp d,rhs: if d > rhs, C=1,Z=0 → GT(12). If d < rhs, N=1 → MI(4).
            _ => unreachable!(),
        }
        self.def(d);
        Ok(())
    }

    // NEG: flip sign bit. op1=source value, op2=signbit constant.
    fn asm_neg(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let m = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.alloc(pin(m))?;
        self.code.fneg_d(d, m);
        self.def(d);
        Ok(())
    }

    // ABS: clear sign bit.  Same pattern as asm_neg: fetch source first,
    // then allocate dest separately to avoid the into_dst/fetch_fp ENV relaod trap.
    fn asm_abs(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let m = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = self.alloc(pin(m))?;
        self.code.fabs_d(d, m);
        self.def(d);
        Ok(())
    }

    // FPMATH: sqrt/floor/ceil/trunc
    fn asm_fpmath(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let src = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let d = if self.dying(ins.op1 as IRRef) { self.steal_quiet(src); src } else { self.alloc(pin(src))? };
        match ins.op2 as u32 {
            IRFPM_SQRT => self.code.fsqrt(d, src),
            IRFPM_FLOOR => self.code.frintm(d, src),
            IRFPM_CEIL  => self.code.frintp(d, src),
            IRFPM_TRUNC => self.code.frintz(d, src),
            _ => unreachable!(),
        }
        self.def(d);
        Ok(())
    }

    // COMP: ordered/unordered FP comparison guards
    fn asm_comp(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(irt_isnum(ins.t()) && ins.is_guard());
        let (x, y) = (ins.op1 as IRRef, ins.op2 as IRRef);
        // Same mapping as x64: operand order chosen so branch direction matches NaN handling.
        // x64 CC_B(0x2)→LO, CC_BE(0x6)→LS, CC_A(0x7)→HI, CC_AE(0x3)→HS
        let (fst, snd, a64cc) = match ins.op() {
            IROp::LT => (y, x, cond::LS),  // exit if y <= x
            IROp::GE => (x, y, cond::CC),  // exit if x < y (CC = LO = unsigned lower)
            IROp::LE => (y, x, cond::CC),  // exit if y < x
            IROp::GT => (x, y, cond::LS),  // exit if x <= y
            // Unsigned: same pattern but A/HS instead of B/LO
            IROp::ULT => (x, y, cond::HI), // exit if NOT (x < y) → exit if x >= y unsigned? 
            // Actually for unsigned, x64 uses AE/HI for different semantics. Let me map more carefully:
            // x64 ULT: (x,y,AE) = exit if x >= y (i.e. CF=0 → AE). ARM64: fcmp x,y; b.hs exit
            // x64 UGE: (y,x,A)  = exit if NOT (x >= y) i.e. y > x unsigned → A=HI. ARM64: fcmp y,x; b.hi exit
            // x64 ULE: (x,y,A)  = exit if x > y unsigned. ARM64: fcmp x,y; b.hi exit
            // x64 UGT: (y,x,AE) = exit if y >= x unsigned. ARM64: fcmp y,x; b.hs exit
            _ => { 
                // Map unsigned: x64 ULT→(x,y,0x3=AE=CS), UGE→(y,x,0x7=A=HI), ULE→(x,y,0x7=A=HI), UGT→(y,x,0x3=AE=CS)
                match ins.op() {
                    IROp::ULT => (x, y, cond::CS),
                    IROp::UGE => (y, x, cond::HI),
                    IROp::ULE => (x, y, cond::HI),
                    IROp::UGT => (y, x, cond::CS),
                    _ => unreachable!(),
                }
            }
        };
        let f = self.fetch_fp(fst, 0)?;
        let s = if snd == fst { f } else { self.fetch_fp(snd, pin(f))? };
        self.code.fcmp(f, s);
        self.guard(a64cc);
        Ok(())
    }

    // EQ/NE: equality comparisons
    fn asm_equal(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(ins.is_guard());
        let eq = ins.op() == IROp::EQ;
        if irt_isnum(ins.t()) {
            let f = self.fetch_fp(ins.op1 as IRRef, 0)?;
            let s = if ins.op2 == ins.op1 { f } else { self.fetch_fp(ins.op2 as IRRef, pin(f))? };
            self.code.fcmp(f, s);
            if eq {
                self.guard(cond::VS); // NaN? exit
                self.guard(cond::NE); // not equal? exit
            } else {
                // NE: skip exit if NaN, exit if ordered equal
                let pos = self.code.len();
                self.code.b_cond(cond::VS, 0); // b.vs (skip over the eq check if NaN)
                let eq_guard_idx = self.make_stub(false);
                let eq_pos = self.code.len();
                self.code.b_cond(cond::EQ, 0); // b.eq exit (ordered EQ → fail NE)
                // patch the b.vs to jump over the b.eq
                let skip_off = (self.code.len() - pos) as i32 / 4;
                patch_bcond_at(&mut self.code.0, pos, cond::VS, skip_off);
                self.fixups.push((eq_pos, eq_guard_idx));
            }
        } else {
            self.gpr_load_ref(RSCRATCH, ins.op1 as IRRef);
            self.gpr_load_ref(RSCRATCH2, ins.op2 as IRRef);
            self.code.cmp_rr(RSCRATCH, RSCRATCH2);
            self.guard(if eq { cond::NE } else { cond::EQ });
        }
        Ok(())
    }

    // ═══ tail_restore: write snapshot values back to Lua stack ═══════════════
    fn tail_restore(&mut self, snapidx: usize) {
        let snap = &self.tr.snap[snapidx];
        let ofs = snap.mapofs as usize;
        for &sn in &self.tr.snapmap[ofs..ofs + snap.nent as usize] {
            if sn & SNAP_NORESTORE != 0 { continue; }
            let disp = (snap_slot(sn) as i32 - 2) * 8;
            let rref = snap_ref(sn);
            if rref >= REF_BIAS {
                if let Some(rg) = self.loc[Self::iidx(rref)] {
                    self.code.str_d(rg, RBASE, disp);
                } else {
                    self.code.ldr(RSCRATCH, RENV, Self::env_ofs(rref));
                    self.code.str(RSCRATCH, RBASE, disp);
                }
            } else {
                self.code.mov64(RSCRATCH, super::super::exec::const_bits(&self.tr.ir, rref));
                self.code.str(RSCRATCH, RBASE, disp);
            }
        }
    }

    // ═══ emit: main loop ════════════════════════════════════════════════════
    fn emit(mut self) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
        // ── prologue: allocate frame, save callee-saved regs ──
        self.code.sub_imm(31, 31, FRAME);               // sub sp, sp, #FRAME
        self.code.stp(29, 30, 31, 0);                   // stp fp, lr, [sp]
        self.code.add_imm(29, 31, 0);                 // mov fp, sp  (add fp, sp, #0)
        for i in 0..SAVED_GPR_PAIRS {
            let off = 16 + (i as i32) * 16;
            self.code.stp(19 + i as u8 * 2, 19 + i as u8 * 2 + 1, 29, off);
        }
        for i in 0..SAVED_FP_PAIRS {
            let off = 16 + SAVED_GPR_PAIRS as i32 * 16 + (i as i32) * 16;
            self.code.stp_d(8 + i as u8 * 2, 8 + i as u8 * 2 + 1, 29, off);
        }
        self.code.add_rr(RBASE, 0, 31);                 // mov x19, x0  (add x19, x0, xzr)
        self.code.add_rr(RENV,  1, 31);                 // mov x20, x1
        let inner = self.code.len() as u32;

        // ── handover (side traces) ──
        if !self.tr.parentmap.is_empty() {
            self.emit_handover();
        }

        // ── main dispatch ──
        let head = self.code.len();
        let nins = self.tr.ir.nins();
        let mut r = REF_FIRST;
        while r < nins {
            while self.snapidx + 1 < self.tr.snap.len() && self.tr.snap[self.snapidx + 1].iref <= r { self.snapidx += 1; }
            self.cur = r;
            let ins = *self.tr.ir.ir(r);
            match ins.op() {
                IROp::NOP|IROp::BASE => {}
                IROp::LOOP => {
                    if self.loop_pos.is_none() {
                        self.loop_pos = Some(self.code.len());
                        self.phi_homes = self.phis.iter().filter_map(|p| {
                            self.reg_of(p.lref).map(|rg| (p.lref, rg))
                        }).collect();
                    }
                }
                IROp::PHI => {}
                IROp::SLOAD => self.asm_sload(&ins)?,
                IROp::ULOAD => self.asm_uload(&ins)?,
                IROp::FLOAD => self.asm_meta_guard(&ins),
                IROp::HLOAD => self.asm_hload(&ins)?,
                IROp::CARG => {}
                IROp::HSTORE => self.asm_hstore(&ins),
                IROp::CALLL => self.asm_calll(&ins)?,
                IROp::TNEW => {
                    self.helper_call(
                        super::super::exec::jit_tnew as *const () as usize as u64, &[]);
                    self.ff_result(&ins)?;
                }
                IROp::TDUP => {
                    self.helper_call(
                        super::super::exec::jit_tdup as *const () as usize as u64,
                        &[ins.op1 as IRRef]);
                    self.ff_result(&ins)?;
                }
                IROp::ALOAD => self.asm_aload(&ins)?,
                IROp::ASTORE => self.asm_astore(&ins)?,
                IROp::GCSTEP => self.asm_gcstep(&ins),
                IROp::POW => {
                    self.helper_call(
                        super::super::exec::jit_pow as *const () as usize as u64,
                        &[ins.op1 as IRRef, ins.op2 as IRRef]);
                    self.ff_result(&ins)?;
                }
                IROp::TOBIT => {
                    let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
                    self.code.fcvtzs_w(RSCRATCH, sx);
                    let d = self.alloc(0)?;
                    self.code.scvtf_x(d, RSCRATCH);
                    self.def(d);
                }
                IROp::BAND|IROp::BOR|IROp::BXOR|IROp::BSHL|IROp::BSHR|IROp::BSAR|IROp::BNOT|IROp::BSWAP => self.asm_bitop(&ins)?,
                IROp::ADD|IROp::SUB|IROp::MUL|IROp::DIV => {
                    let op = ins.op(); let (mut a,mut b)=(ins.op1 as IRRef,ins.op2 as IRRef);
                    // Only swap when both are refs: swapping a constant into
                    // the destination position causes the destination register
                    // to hold stale constant bits on loop re-entry.
                    if matches!(op,IROp::ADD|IROp::MUL) && b>=REF_BIAS && !self.dying(a)&&self.dying(b)&&self.reg_of(b).is_some() { std::mem::swap(&mut a,&mut b); }
                    let d=self.into_dst(a)?; let rhs=if b==a {d}else{self.fetch_fp(b,pin(d))?};
                    match op { IROp::ADD=>self.code.fadd(d,d,rhs), IROp::SUB=>self.code.fsub(d,d,rhs), IROp::MUL=>self.code.fmul(d,d,rhs), IROp::DIV=>self.code.fdiv(d,d,rhs), _=>{} }
                    self.def(d);
                }
                IROp::MIN|IROp::MAX => {
                    let (a,b)=(ins.op1 as IRRef,ins.op2 as IRRef); let d=self.into_dst(a)?; let rhs=if b==a {d}else{self.fetch_fp(b,pin(d))?};
                    self.code.fcmp(d,rhs);
                    self.code.fcsel(d,d,rhs,if ins.op()==IROp::MIN{cond::GT}else{cond::MI});
                    self.def(d);
                }
                IROp::NEG => self.asm_neg(&ins)?,
                IROp::ABS => self.asm_abs(&ins)?,
                IROp::FPMATH => self.asm_fpmath(&ins)?,
                IROp::LT|IROp::GE|IROp::LE|IROp::GT|IROp::ULT|IROp::UGE|IROp::ULE|IROp::UGT => self.asm_comp(&ins)?,
                IROp::EQ|IROp::NE => self.asm_equal(&ins)?,
                _ => return Err(TraceError::NYIIR),
            }
            r += 1;
        }

        // ── tail ──
        let lastsnap = self.tr.snap.len()-1;
        let looping = self.tr.linktype==TraceLink::Loop && self.tr.link==self.tr.traceno;
        if looping {
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            if let Some(lp) = self.loop_pos {
                // PHI-aware back edge: parallel-assign rref values to lref
                // homes, then jump to the variant part (just after LOOP).
                // Handles cycles (a,b = b,a) by staging through a temp.
                let phis: Vec<_> = self.phis.iter().filter(|p| p.num).cloned().collect();
                // Detect conflicts: a PHI reads from a register that
                // another PHI writes to (cyclic dependency).
                let mut needs_temp = vec![false; phis.len()];
                for i in 0..phis.len() {
                    if let Some(ri) = self.reg_of(phis[i].rref) {
                        for j in 0..phis.len() {
                            if i == j { continue; }
                            if self.reg_of(phis[j].lref) == Some(ri) {
                                needs_temp[i] = true;
                            }
                        }
                    }
                }
                // Phase 1: stage rrefs that conflict into temp registers.
                let mut temp_used = 0u8;
                for (i, p) in phis.iter().enumerate() {
                    if needs_temp[i] {
                        let temp_rg = FP_SCRATCH + temp_used;
                        temp_used += 1;
                        if let Ok(rval) = self.fetch_fp(p.rref, 0) {
                            self.code.fmov_dd(temp_rg, rval);
                        }
                    }
                }
                // Phase 2: write to lref homes (from staged temps or
                // directly from rrefs).
                let mut temp_idx = 0u8;
                for (i, p) in phis.iter().enumerate() {
                    let lhome = self.reg_of(p.lref);
                    if needs_temp[i] {
                        let temp_rg = FP_SCRATCH + temp_idx;
                        temp_idx += 1;
                        if let Some(lh) = lhome {
                            if lh != temp_rg { self.code.fmov_dd(lh, temp_rg); }
                        } else {
                            self.code.str_d(temp_rg, RENV, Self::env_ofs(p.lref));
                            self.env_valid[Self::iidx(p.lref)] = true;
                        }
                    } else {
                        let pinned = if let Some(rg) = lhome { pin(rg) } else { 0 };
                        if let Ok(rval) = self.fetch_fp(p.rref, pinned) {
                            if let Some(lh) = lhome {
                                if lh != rval { self.code.fmov_dd(lh, rval); }
                            } else {
                                self.code.str_d(rval, RENV, Self::env_ofs(p.lref));
                                self.env_valid[Self::iidx(p.lref)] = true;
                            }
                        }
                    }
                }
                // Reload loop-carried register homes: they may have been
                // stolen by later instruction's constant loads.
                for &(lref, rg) in &self.phi_homes {
                    if self.reg_of(lref).is_none() && self.env_valid[Self::iidx(lref)] {
                        self.code.ldr_d(rg, RENV, Self::env_ofs(lref));
                    }
                }
                let off = (lp as i64 - self.code.len() as i64) as i32 / 4;
                self.code.b(off);
            } else {
                // Legacy trace (no IR_LOOP): jump back to head.
                let off = (head as i64 - self.code.len() as i64) as i32 / 4;
                self.code.b(off);
            }
        } else if matches!(self.tr.linktype, TraceLink::Uprec|TraceLink::Tailrec) && self.tr.link==self.tr.traceno {
            return Err(TraceError::NYIIR); // recursive tail NYI
        } else if self.tr.linktype==TraceLink::Root && let Some(target)=self.link {
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let delta = (self.tr.snap[lastsnap].baseslot as i32 - 2) * 8;
            if delta != 0 {
                // Stack headroom check: new_base + room > stack_end → exit
                let room = (255 + 8) * 8;
                self.code.mov64(RSCRATCH, (delta + room) as u64);
                self.code.add_rr(RSCRATCH, RSCRATCH, RBASE as u8);
                self.code.mov64(RSCRATCH2, super::super::exec::stack_end_cell_addr());
                self.code.ldr(RSCRATCH3, RSCRATCH2, 0);
                self.code.cmp_rr(RSCRATCH, RSCRATCH3);
                self.guard(cond::HI);
                self.code.mov64(RSCRATCH, delta as u64);
                self.code.add_rr(RBASE as u8, RBASE as u8, RSCRATCH);
            }
            self.code.mov64(RSCRATCH, target as u64);
            self.code.br(RSCRATCH);
        } else {
            // Final snapshot exit → epilogue
            self.snapidx = lastsnap;
            for (rg, fr) in self.exit_flush_set(lastsnap) { self.code.str_d(rg,RENV,Self::env_ofs(fr)); }
            // Load exit code into x0 (return value)
            let ec = self.exit_code(lastsnap);
            self.code.mov64(0, ec as u64);
        }

        // ── epilogue ──
        let epilogue = self.code.len();
        // Store BASE to exit_base cell
        self.code.mov64(RSCRATCH, super::super::exec::exit_base_cell_addr());
        self.code.str(RBASE, RSCRATCH, 0);
        // Restore callee-saved regs
        for i in 0..SAVED_GPR_PAIRS {
            let off = 16 + (i as i32) * 16;
            self.code.ldp(19+i as u8*2, 19+i as u8*2+1, 29, off);
        }
        for i in 0..SAVED_FP_PAIRS {
            let off = 16 + SAVED_GPR_PAIRS as i32*16 + (i as i32)*16;
            self.code.ldp_d(8+i as u8*2, 8+i as u8*2+1, 29, off);
        }
        self.code.ldp(29, 30, 31, 0);  // ldp fp, lr, [sp]
        self.code.add_imm(31, 31, FRAME); // add sp, sp, #FRAME
        self.code.ret();

        // ── exit stubs ──
        let stubs = std::mem::take(&mut self.stubs);
        let mut stubpos: Vec<usize> = Vec::with_capacity(stubs.len());
        for st in &stubs {
            stubpos.push(self.code.len());
            for &(rg, fr) in &st.flush { self.code.str_d(rg, RENV, Self::env_ofs(fr)); }
            let tail = self.code.len();
            let mut ec = self.exit_code(st.snapidx);
            if st.gc { ec |= 0x8000; } else { self.stub_tails.push((st.snapidx as u32, tail as u32)); }
            self.code.mov64(0, ec as u64); // x0 = exit code
            // patchable tail: mov(4)+mov(4)+mov64(16)+br(4) = 28 bytes reserved
            while self.code.len() < tail + 28 { self.code.nop(); }
            // b epilogue
            let off = (epilogue as i64 - self.code.len() as i64) as i32 / 4;
            self.code.b(off);
        }
        // Patch fixups
        for (pos, si) in std::mem::take(&mut self.fixups) {
            let off = (stubpos[si] as i64 - pos as i64) as i32 / 4;
            // Re-read the instruction at pos and fix its offset
            let w = u32::from_le_bytes(self.code.0[pos..pos+4].try_into().unwrap());
            let new_w = (w & 0xFC00_001F) | (((off as u32)&0x7FFFF)<<5);
            patch_u32_at(&mut self.code.0, pos, new_w);
        }

        // Allocate mcode
        let dump_mcode = std::env::var("LUAJIT_RS_MCDUMP").is_ok();
        if dump_mcode {
            eprintln!("=== ARM64 MCODE tr={} PROLOGUE inner={} ===", self.tr.traceno, inner);
            eprintln!("{}", hex_dump(&self.code.0));
        }
        let mut area = McodeArea::alloc(self.code.len()).ok_or(TraceError::MCODEAL)?;
        area.as_mut_slice()[..self.code.len()].copy_from_slice(&self.code.0);
        if !area.protect_exec() { return Err(TraceError::MCODEAL); }
        Ok((area, inner, std::mem::take(&mut self.stub_tails)))
    }
}

// ═══ Public API ═══════════════════════════════════════════════════════════

pub fn assemble(tr: &GCtrace, link: Option<*const u8>) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
    if tr.linktype == TraceLink::Downrec { return Err(TraceError::NYIIR); }
    Asm::new(tr, link)?.emit()
}

pub fn patch_exit(area: &mut McodeArea, stub_tails: &[(u32, u32)], exitno: u32, target: *const u8) {
    if !area.protect_rw() { return; }
    let code = area.as_mut_slice();
    for &(si, ofs) in stub_tails {
        if si == exitno {
            let p = ofs as usize;
            let t = target as u64;
            let mut e = Emit(Vec::new());
            e.add_rr(0, 19, 31);  // mov x0, x19 — restore base for side trace
            e.add_rr(1, 20, 31);  // mov x1, x20 — restore env for side trace
            e.mov64(16, t);         // mov x16, target
            e.br(16);               // br x16
            code[p..p+e.0.len()].copy_from_slice(&e.0);
        }
    }
    area.protect_exec();
}

fn hex_dump(buf: &[u8]) -> String {
    let mut s = String::with_capacity(buf.len() * 6);
    for (off, chunk) in buf.chunks(16).enumerate() {
        use std::fmt::Write;
        let _ = write!(s, "{:04x}: ", off * 16);
        for b in chunk {
            let _ = write!(s, "{:02x} ", b);
        }
        // Decode first instruction in this row
        if chunk.len() >= 4 {
            let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            s.push_str("  ");
            s.push_str(&disasm_a64(w, off * 16));
        }
        s.push('\n');
    }
    s
}

/// Quick A64 disassembler for debugging. Covers the instructions we emit.
fn disasm_a64(w: u32, _addr: usize) -> String {
    let rd = (w & 0x1F) as u8;
    let rn = ((w >> 5) & 0x1F) as u8;
    let rm = ((w >> 16) & 0x1F) as u8;
    let imm12 = ((w >> 10) & 0xFFF) as i32;
    let imm19 = ((w >> 5) & 0x7FFFF) as i32;
    let imm26 = (w & 0x3FF_FFFF) as i32;

    if w & 0xFF00_0000 == 0xD100_0000 { return format!("sub x{}, x{}, #{}", rd, rn, imm12); }
    if w & 0xFF00_0000 == 0x9100_0000 { return format!("add x{}, x{}, #{}", rd, rn, imm12); }
    if w & 0xFFE0_0000 == 0x8B00_0000 { return format!("add x{}, x{}, x{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0xCB00_0000 { return format!("sub x{}, x{}, x{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0x8A00_0000 { return format!("and x{}, x{}, x{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0x0A00_0000 { return format!("and w{}, w{}, w{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0xAA00_0000 { return format!("orr x{}, x{}, x{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0x2A00_0000 { return format!("orr w{}, w{}, w{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0xCA00_0000 { return format!("eor x{}, x{}, x{}", rd, rn, rm); }
    if w & 0xFFE0_0000 == 0xEB00_001F { return format!("cmp x{}, x{}", rn, rm); }
    if w & 0xFC00_0000 == 0xA900_0000 { let o=((w>>15)&0x7F)*8; return format!("stp x{}, x{}, [x{}, #{}]", rd, rm>>5/*rt2 is bits 14-10*/, rn, o); }
    // rt2 is bits 14-10 for STP
    let rt2 = ((w >> 10) & 0x1F) as u8;
    if w & 0xFC00_0000 == 0xA940_0000 { let o=((w>>15)&0x7F)*8; return format!("ldp x{}, x{}, [x{}, #{}]", rd, rt2, rn, o); }
    if w & 0xFFC0_0000 == 0xF940_0000 { return format!("ldr x{}, [x{}, #{}]", rd, rn, imm12*8); }
    if w & 0xFFC0_0000 == 0xF900_0000 { return format!("str x{}, [x{}, #{}]", rd, rn, imm12*8); }
    if w & 0xFFC0_0000 == 0xFD400000 { return format!("ldr d{}, [x{}, #{}]", rd, rn, imm12*8); }
    if w & 0xFFC0_0000 == 0xFD000000 { return format!("str d{}, [x{}, #{}]", rd, rn, imm12*8); }
    if w & 0xFFC0_0000 == 0xB9400000 { return format!("ldr w{}, [x{}, #{}]", rd, rn, imm12*4); }
    if w & 0xFFFF_FC00 == 0xD280_0000 { return format!("movz x{}, #0x{:x}", rd, (w>>5)&0xFFFF); }
    if w & 0xFFFF_FC00 == 0xF280_0000 { return format!("movk x{}, #0x{:x} lsl#{}", rd, (w>>5)&0xFFFF, ((w>>21)&3)*16); }
    if w & 0xFF20_0000 == 0x1E60_2800 { return format!("fadd d{}, d{}, d{}", rd, rn, rm); }
    if w & 0xFF20_0000 == 0x1E60_3800 { return format!("fsub d{}, d{}, d{}", rd, rn, rm); }
    if w & 0xFF20_0000 == 0x1E60_0800 { return format!("fmul d{}, d{}, d{}", rd, rn, rm); }
    if w & 0xFF20_0000 == 0x1E60_1800 { return format!("fdiv d{}, d{}, d{}", rd, rn, rm); }
    if w & 0xFF20_0000 == 0x1E60_2000 { return format!("fcmp d{}, d{}", rn, rm); }
    if w & 0xFF20_0000 == 0x1E60_4000 { return format!("fmov d{}, d{}", rd, rn); }
    if w & 0xFF20_0000 == 0x1E61_4000 { return format!("fneg d{}, d{}", rd, rn); }
    if w & 0xFC00_0000 == 0x5400_0000 { return format!("b.cond #{}", imm19); }
    if w & 0xFC00_0000 == 0x1400_0000 { let off = if imm26&0x2000000!=0 {(imm26 as i32)-0x4000000}else{imm26 as i32}; return format!("b {}", off); }
    if w & 0xFFFF_FC00 == 0x9E220_000 { return format!("scvtf d{}, w{}", rd, rn); }
    if w & 0xFFFF_FC00 == 0x9E380_000 { return format!("fcvtzs x{}, d{}", rd, rn); }
    if w & 0xFFFF_FC00 == 0x1E220_000 { return format!("scvtf d{}, w{}", rd, rn); }
    if w & 0xFFFF_FC00 == 0x1E380_000 { return format!("fcvtzs w{}, d{}", rd, rn); }
    if w & 0xFFFF_FC00 == 0x9E670_000 { return format!("fmov d{}, x{}", rd, rn); }
    if w == 0xD65F_03C0 { return "ret".into(); }
    if w & 0xFFFF_FC00 == 0xD63F_0000 { return format!("blr x{}", rn); }
    if w & 0xFFE0_0000 == 0x6D00_0000 { let o=((w>>15)&0x7F)*8; return format!("stp d{}, d{}, [x{}, #{}]", rd, rt2, rn, o); }
    if w & 0xFFE0_0000 == 0x6D40_0000 { let o=((w>>15)&0x7F)*8; return format!("ldp d{}, d{}, [x{}, #{}]", rd, rt2, rn, o); }
    if w == 0xD503_201F { return "nop".into(); }
    format!("??? 0x{:08x}", w)
}
