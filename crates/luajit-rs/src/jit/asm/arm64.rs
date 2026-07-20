//! ARM64 (AArch64) trace assembler: translates the SSA IR of a
//! completed trace into native ARM64 machine code.
//!
//! The external API (`assemble`, `patch_exit`) mirrors the x86-64
//! backend exactly — trace.rs calls through the unified `jit::asm`
//! facade with no architecture-specific logic.
//!
//! ## ABI of the emitted code
//! `extern "C" fn(base: *mut LuaValue, env: *mut u64) -> u32` returning
//! the exit snapshot index. AAPCS64: `base` in x0, `env` in x1, return
//! in w0. Every trace sets up the same outer frame (x19-x28 + x29/x30 +
//! v8-v15 saved on entry) so linked traces can jump between each
//! other's inner entries while staying inside the frame of whichever
//! trace was entered from Rust.
//!
//! ## Register assignment
//! | Role        | Reg | Notes                              |
//! |-------------|-----|------------------------------------|
//! | BASE        | x19 | Callee-saved (no save around calls)|
//! | ENV         | x20 | Callee-saved                       |
//! | Scratch GPR | x0  | Also return value                  |
//! | Scratch GPR | x1  |                                    |
//! | Scratch GPR | x2  |                                    |
//! | Scratch FP  | v0–v4  | Caller-saved, 5 allocatable  |
//! | Saved FP    | v8–v15 | Callee-saved, used for FP spills  |
//! | v16 (tmp)   | v16    | Scratch FP (never allocated)       |
//!
//! ## Encoding notes
//! ARM64 instructions are fixed-width 32 bits. 64-bit immediates use
//! `movz + movk` sequences (up to 4 × 16-bit halves). FP immediates
//! are loaded from a literal pool or via `fmov` from a GPR.
#![allow(unused_imports, dead_code)]

use std::mem::offset_of;

use super::super::ir::*;
use super::super::mcode::McodeArea;
use super::super::record::{IRFPM_CEIL, IRFPM_FLOOR, IRFPM_SQRT, IRFPM_TRUNC, IRSLOAD_PARENT};
use super::super::{GCtrace, SNAP_NORESTORE, TraceError, TraceLink, snap_ref, snap_slot};

// ---------------------------------------------------------------------------
// ARM64 instruction encoding primitives
// ---------------------------------------------------------------------------

/// Encode a 32-bit ARM64 instruction.
#[inline]
fn insn(bits: u32) -> [u8; 4] {
    bits.to_le_bytes()
}

/// Push a 32-bit instruction into the code buffer.
fn emit32(code: &mut Vec<u8>, bits: u32) {
    code.extend_from_slice(&bits.to_le_bytes());
}

/// Push a 64-bit word (literal pool entry).
fn emit64(code: &mut Vec<u8>, val: u64) {
    code.extend_from_slice(&val.to_le_bytes());
}

// -- GPR instructions -------------------------------------------------------

/// ADD (immediate): `add rd, rn, #imm` (12-bit unsigned, optionally shifted).
fn add_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u32, shift: u8) {
    debug_assert!(imm < 4096 && shift <= 1);
    let sf = 1u32 << 31; // 64-bit
    emit32(code, sf | 0x11000000 | ((shift as u32) << 22) | (imm << 10) | ((rn as u32) << 5) | rd as u32);
}

/// SUB (immediate): `sub rd, rn, #imm`.
fn sub_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u32, shift: u8) {
    debug_assert!(imm < 4096 && shift <= 1);
    let sf = 1u32 << 31;
    emit32(code, sf | 0x51000000 | ((shift as u32) << 22) | (imm << 10) | ((rn as u32) << 5) | rd as u32);
}

/// ADD (shifted register): `add rd, rn, rm, lsl #shift`.
fn add_reg_lsl(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8, shift: u8) {
    debug_assert!(shift < 64);
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x0B000000 | ((shift as u32) << 10) | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// MOV (register): `mov rd, rm` (alias of ORR with zero register).
fn mov_reg(code: &mut Vec<u8>, rd: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x2A0003E0 | ((rm as u32) << 16) | rd as u32);
}

/// MOVZ: `mov rd, #imm16, lsl #shift`.
fn movz(code: &mut Vec<u8>, rd: u8, imm: u16, shift: u8) {
    debug_assert!(shift % 16 == 0 && shift < 64);
    let sf = 1u32 << 31;
    let hw = (shift as u32 / 16) << 21;
    emit32(code, sf | 0x52800000 | hw | ((imm as u32) << 5) | rd as u32);
}

/// MOVK: `movk rd, #imm16, lsl #shift`.
fn movk(code: &mut Vec<u8>, rd: u8, imm: u16, shift: u8) {
    debug_assert!(shift % 16 == 0 && shift < 64);
    let sf = 1u32 << 31;
    let hw = (shift as u32 / 16) << 21;
    emit32(code, sf | 0x72800000 | hw | ((imm as u32) << 5) | rd as u32);
}

/// Load a 64-bit immediate into rd using movz + movk.
fn mov_imm64(code: &mut Vec<u8>, rd: u8, val: u64) {
    let mut v = val;
    let mut first = true;
    let mut shift = 0u8;
    while shift < 64 {
        let chunk = (v & 0xFFFF) as u16;
        if first {
            movz(code, rd, chunk, shift);
            first = false;
        } else if chunk != 0 || shift == 48 {
            // Always emit the top word to avoid canonical address
            // ambiguity (movz zero-extends, leaving high bits clear).
            movk(code, rd, chunk, shift);
        }
        v >>= 16;
        shift += 16;
    }
}

/// Load a 64-bit immediate into rd using movz + 3 × movk (always 16 bytes).
fn mov_imm64_full(code: &mut Vec<u8>, rd: u8, val: u64) {
    let mut v = val;
    for shift in [0u8, 16, 32, 48] {
        let chunk = (v & 0xFFFF) as u16;
        if shift == 0 {
            movz(code, rd, chunk, shift);
        } else {
            movk(code, rd, chunk, shift);
        }
        v >>= 16;
    }
}

/// CMP (immediate): `cmp rn, #imm` (alias of SUBS with zero register).
fn cmp_imm(code: &mut Vec<u8>, rn: u8, imm: u32, shift: u8) {
    debug_assert!(imm < 4096 && shift <= 1);
    let sf = 1u32 << 31;
    emit32(code, sf | 0x7100001F | ((shift as u32) << 22) | (imm << 10) | ((rn as u32) << 5));
}

/// CMP (register): `cmp rn, rm` (alias of SUBS with zero rd).
fn cmp_reg(code: &mut Vec<u8>, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x6B00001F | ((rm as u32) << 16) | ((rn as u32) << 5));
}

/// AND (immediate): `and rd, rn, #imm` (bitmask immediate).
fn and_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u64) {
    let (n, immr, imms) = encode_bitmask(imm).expect("invalid AND immediate");
    let sf = 1u32 << 31;
    let n_bit = (n as u32) << 22;
    emit32(
        code,
        sf | 0x12000000 | n_bit | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32,
    );
}

/// Encode a 64-bit bitmask immediate for AND/ORR/EOR. Returns (N, immr, imms).
fn encode_bitmask(imm: u64) -> Option<(u8, u32, u32)> {
    if imm == 0 || imm == u64::MAX {
        return Some((0, 0, 63));
    }
    // ARM64 bitmask immediates are a pattern of consecutive ones,
    // right-rotated by some amount, replicated across 2/4/8/16/32/64 bits.
    let ones = imm.count_ones();
    if ones == 0 { return None; }
    let r = imm.trailing_zeros();
    let len = 64 - imm.leading_zeros() - r;
    // The pattern must be a string of 1s of length `ones`, padded.
    if (len as u32) != ones { return None; }
    // Check that the 1s are consecutive and the rest is the repeating pattern.
    let mask = (1u64 << ones) - 1;
    let pattern = imm >> r;
    if pattern & mask != mask { return None; }
    // Find the smallest element size that fits the period.
    for &esize in &[2u32, 4, 8, 16, 32, 64] {
        if ones > esize { continue; }
        let period = esize;
        // Check that the pattern repeats every `period` bits.
        let repeating = (0..64).step_by(period as usize).all(|s| {
            (imm >> s) & ((1u64 << period) - 1) == (imm & ((1u64 << period) - 1))
        });
        if !repeating { continue; }
        let n = if esize == 64 { 1 } else { 0 };
        // imms encodes element-size and S (ones-1) in a single 6-bit field:
        //   esize=64: N=1, imms = ones-1
        //   esize=32: imms[5]=0,       imms[4:0] = ones-1
        //   esize=16: imms[5:4]=0b10,  imms[3:0] = ones-1
        //   esize=8:  imms[5:3]=0b100, imms[2:0] = ones-1
        //   esize=4:  imms[5:2]=0b1000,imms[1:0] = ones-1
        //   esize=2:  imms[5:1]=0b10000,imms[0]= ones-1
        let imms: u32 = match esize {
            64 => (ones - 1) & 0x3F,            // N=1 already
            32 => (ones - 1) & 0x1F,            // bit 5 = 0
            16 => 0x20 | ((ones - 1) & 0xF),    // bit[5:4] = 10
            8  => 0x30 | ((ones - 1) & 0x7),    // bit[5:3] = 100
            4  => 0x38 | ((ones - 1) & 0x3),    // bit[5:2] = 1000
            2  => 0x3C | ((ones - 1) & 0x1),    // bit[5:1] = 10000
            _  => unreachable!(),
        };
        let immr = (r as u32) % esize;
        return Some((n, immr, imms));
    }
    None
}

/// ORR (immediate): `orr rd, rn, #imm` (bitmask immediate).
fn orr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u64) {
    let (n, immr, imms) = encode_bitmask(imm).expect("invalid ORR immediate");
    let sf = 1u32 << 31;
    let n_bit = (n as u32) << 22;
    emit32(
        code,
        sf | 0x32000000 | n_bit | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32,
    );
}

/// AND (shifted register): `and rd, rn, rm, lsl/lsr/asr #shift`.
fn and_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x0A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// ORR (shifted register): `orr rd, rn, rm`.
fn orr_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x2A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// EOR (shifted register): `eor rd, rn, rm`.
fn eor_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x4A000000 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// LSL (register): `lsl rd, rn, rm`.
fn lsl_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x1AC02000 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// LSR (register): `lsr rd, rn, rm`.
fn lsr_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x1AC02400 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// LSR (immediate): `lsr rd, rn, #imm` (64-bit).
fn lsr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    let sf = 1u32 << 31;
    let n = 1u32 << 22;
    let immr = imm as u32;
    let imms = 63u32;
    // UBFM: sf|10|100110|N|immr|imms|Rn|Rd  (LSR allocates UBFM, not SBFM!)
    emit32(code, sf | 0x53000000 | n | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32);
}

/// ASR (immediate): `asr rd, rn, #imm` (proper encoding).
fn asr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    let sf = 1u32 << 31;
    let n = 1u32 << 22; // 64-bit
    let immr = imm as u32;
    let imms = 63u32; // sign extension
    emit32(code, sf | 0x13000000 | n | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32);
}

/// UBFX (unsigned bitfield extract): extract `width` bits starting at `lsb`,
/// zero-extended. `lsb + width <= 64`. Uses 64-bit UBFM.
fn ubfx(code: &mut Vec<u8>, rd: u8, rn: u8, lsb: u8, width: u8) {
    let sf = 1u32 << 31;
    let n = 1u32 << 22;
    let immr = lsb as u32;
    let imms = (lsb + width - 1) as u32;
    emit32(code, sf | 0x53000000 | n | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32);
}

/// ROR (register): `ror rd, rn, rm`.
fn ror_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x1AC02C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// EOR (imm): `eor rd, rn, #imm` via GPR load + eor (bitmask const
/// isn't always representable).
fn eor_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u64) {
    mov_imm64(code, 2, imm);
    eor_reg(code, rd, rn, 2);
}

/// ROR (immediate, 32-bit): `ror wd, wn, #imm`. Uses EXTR wd, wn, wn, #imm.
fn ror_imm32(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    emit32(code, 0 | 0x13800000 | 0 | ((rn as u32) << 16) | ((imm as u32) << 10) | ((rn as u32) << 5) | rd as u32);
}

/// NEG (32-bit): `neg wd, wn` (alias of SUB wd, WZR, wn).
fn neg_w(code: &mut Vec<u8>, rd: u8, rn: u8) {
    let sf = 0u32 << 31;
    // SUB (shifted register): WZR(w31) as Rn, wn as Rm.
    // Base 0x4B0003E0 has Rn=31(WZR), imm6=0, shift=00.
    emit32(code, sf | 0x4B0003E0 | ((rn as u32) << 16) | rd as u32);
}

/// AND (immediate, 32-bit): `and wd, wn, #imm` (bitmask).
fn and_imm32(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u32) {
    // Same issue: can't always encode as bitmask. Use GPR load + and_reg.
    mov_imm64(code, 2, imm as u64);
    let sf = 0u32 << 31;
    emit32(code, sf | 0x0A000000 | ((2u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// LDR (register, 64-bit, uxtw scaled): `ldr rd, [rn, rm, uxtw #3]`.
fn ldr_reg_uxtw(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    // (A64I_LDRx ^ A64I_LS_R) | A64I_LS_UXTWx | A64I_LS_SH = 0xF8605800
    emit32(code, 0xF8605800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// STR (register, 64-bit, uxtw scaled): `str rd, [rn, rm, uxtw #3]`.
fn str_reg_uxtw(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    // (A64I_STRx ^ A64I_LS_R) | A64I_LS_UXTWx | A64I_LS_SH = 0xF8205800
    emit32(code, 0xF8205800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// REV32 (reverse bytes in 32-bit word, zero-extending).
fn rev32(code: &mut Vec<u8>, rd: u8, rn: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x5AC00800 | ((rn as u32) << 5) | rd as u32);
}

/// LDR (GPR, unsigned offset): `ldr rd, [rn, #offset]`.
fn ldr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, offset: i32, size: u8) {
    let scale = (size / 8) as i32;
    debug_assert!(offset >= 0 && offset % scale == 0 && offset <= 32760);
    let base = if size == 64 { 0xF9400000u32 } else { 0xB9400000u32 };
    emit32(code, base | ((offset as u32 / scale as u32) << 10) | ((rn as u32) << 5) | rd as u32);
}

/// STR (FP, unsigned offset): `str dn, [rn, #offset]`.
fn str_fp(code: &mut Vec<u8>, dn: u8, rn: u8, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0 && offset <= 32760);
    // SIMD&FP store, unsigned offset, 64-bit: A64I_STRd = 0xFD000000
    emit32(code, 0xFD000000 | ((offset as u32 / 8) << 10) | ((rn as u32) << 5) | dn as u32);
}

/// LDR (FP, unsigned offset): `ldr dd, [rn, #offset]`.
fn ldr_fp(code: &mut Vec<u8>, dd: u8, rn: u8, offset: i32) {
    debug_assert!(offset >= 0 && offset % 8 == 0 && offset <= 32760);
    // SIMD&FP load, unsigned offset, 64-bit: A64I_LDRd = 0xFD400000
    emit32(code, 0xFD400000 | ((offset as u32 / 8) << 10) | ((rn as u32) << 5) | dd as u32);
}

/// STR (GPR, unsigned offset): `str rd, [rn, #offset]`.
fn str_imm(code: &mut Vec<u8>, rd: u8, rn: u8, offset: i32, size: u8) {
    let scale = (size / 8) as i32;
    debug_assert!(offset >= 0 && offset % scale == 0 && offset <= 32760);
    let base = if size == 64 { 0xF9000000u32 } else { 0xB9000000u32 };
    emit32(code, base | ((offset as u32 / scale as u32) << 10) | ((rn as u32) << 5) | rd as u32);
}

/// LDR (register, 64-bit, lsl scaled): `ldr rd, [rn, rm, lsl #3]`.
fn ldr_reg_lsl3(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    // (A64I_LDRx ^ A64I_LS_R) | A64I_LS_SXTXx | A64I_LS_SH = 0xF860F800
    emit32(code, 0xF860F800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// STR (register, 64-bit, lsl scaled): `str rd, [rn, rm, lsl #3]`.
fn str_reg_lsl3(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    // (A64I_STRx ^ A64I_LS_R) | A64I_LS_SXTXx | A64I_LS_SH = 0xF820F800
    emit32(code, 0xF820F800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32);
}

/// STP (store pair, offset): `stp rt1, rt2, [rn, #offset]`.
fn stp_offset(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset / 8) as u32 & 0x7F) << 15;
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x29000000 | (imm7) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
    );
}

/// LDP (load pair, offset): `ldp rt1, rt2, [rn, #offset]`.
fn ldp_offset(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset / 8) as u32 & 0x7F) << 15;
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x29400000 | (imm7) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
    );
}

/// B (unconditional): `b offset` (28-bit signed, 4-byte aligned).
fn b_imm(code: &mut Vec<u8>, offset: i32, link: bool) {
    debug_assert!(offset % 4 == 0 && offset >= -(1 << 27) && offset < (1 << 27));
    let imm26 = ((offset as u32) >> 2) & 0x3FFFFFF;
    let l_bit = if link { 1u32 << 31 } else { 0 };
    emit32(code, 0x14000000 | l_bit | imm26);
}

/// BR (register): `br rn` or `blr rn`.
fn br_reg(code: &mut Vec<u8>, rn: u8, link: bool) {
    let l_bit = if link { 1u32 << 21 } else { 0 };
    emit32(code, 0xD61F0000 | l_bit | ((rn as u32) << 5));
}

/// RET: `ret rn`.
fn ret(code: &mut Vec<u8>, rn: u8) {
    emit32(code, 0xD65F0000 | ((rn as u32) << 5));
}

/// B.cond: conditional branch (19-bit signed, 4-byte aligned).
fn b_cond(code: &mut Vec<u8>, cond: u8, offset: i32) {
    debug_assert!(offset % 4 == 0 && offset >= -(1 << 20) && offset < (1 << 20));
    let imm19 = ((offset as u32) >> 2) & 0x7FFFF;
    emit32(code, 0x54000000 | (imm19 << 5) | cond as u32);
}

/// CSET: `cset rd, cond` (conditional set).
fn cset(code: &mut Vec<u8>, rd: u8, cond: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x1A9F07E0 | ((cond as u32 - 1) << 12) | rd as u32);
}

/// Condition codes (ARM64, matching x86 convention).
#[allow(dead_code)]
const CC_EQ: u8 = 0;
#[allow(dead_code)]
const CC_NE: u8 = 1;
#[allow(dead_code)]
const CC_CS: u8 = 2;
#[allow(dead_code)]
const CC_CC: u8 = 3;
#[allow(dead_code)]
const CC_MI: u8 = 4;
#[allow(dead_code)]
const CC_PL: u8 = 5;
#[allow(dead_code)]
const CC_VS: u8 = 6;
#[allow(dead_code)]
const CC_VC: u8 = 7;
#[allow(dead_code)]
const CC_HI: u8 = 8;
#[allow(dead_code)]
const CC_LS: u8 = 9;
const CC_GE: u8 = 10;
const CC_LT: u8 = 11;
const CC_GT: u8 = 12;
const CC_LE: u8 = 13;
const CC_AL: u8 = 14;

/// Map x86 condition codes to ARM64.
fn to_arm64_cc(x86_cc: u8) -> u8 {
    match x86_cc {
        0x0 => CC_CC, // AE → CC (NB → LO inverse)
        0x1 => CC_CS, // A  → CS (NBE → HI)
        0x2 => CC_LE, // BE → LE (exact)
        0x3 => CC_GT, // B  → GT (NBE)
        0x4 => CC_EQ, // E  → EQ
        0x5 => CC_NE, // NE → NE
        0x6 => CC_LE, // LE → LE
        0x7 => CC_GT, // G  → GT (NLE)
        0x8 => CC_CC, // NP → VC (parity → overflow clear)
        0x9 => CC_CS, // P  → VS (parity → overflow set)
        _ => CC_AL,
    }
}

// -- FP/SIMD instructions ---------------------------------------------------

/// FMOV (register): `fmov dd, dn` (64-bit float).
fn fmov_reg(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E604000 | ((dn as u32) << 5) | dd as u32);
}

/// FMOV (GPR to FP): `fmov dd, xn`.
fn fmov_gpr_fp(code: &mut Vec<u8>, dd: u8, xn: u8) {
    emit32(code, 0x9E670000 | ((xn as u32) << 5) | dd as u32);
}

/// FMOV (FP to GPR): `fmov xd, dn`.
fn fmov_fp_gpr(code: &mut Vec<u8>, xd: u8, dn: u8) {
    emit32(code, 0x9E660000 | ((dn as u32) << 5) | xd as u32);
}

/// FADD: `fadd dd, dn, dm`.
fn fadd(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E602800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FSUB: `fsub dd, dn, dm`.
fn fsub(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E603800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FMUL: `fmul dd, dn, dm`.
fn fmul(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E600800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FDIV: `fdiv dd, dn, dm`.
fn fdiv(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E601800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FNEG: `fneg dd, dn`.
fn fneg(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E614000 | ((dn as u32) << 5) | dd as u32);
}

/// FABS: `fabs dd, dn`.
fn fabs(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E60C000 | ((dn as u32) << 5) | dd as u32);
}

/// FSQRT: `fsqrt dd, dn`.
fn fsqrt(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E61C000 | ((dn as u32) << 5) | dd as u32);
}

/// FMIN: `fmin dd, dn, dm`.
fn fmin(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E605800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FMAX: `fmax dd, dn, dm`.
fn fmax(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E604800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FRINTM (floor): `frintm dd, dn`.
fn frintm(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E654000 | ((dn as u32) << 5) | dd as u32);
}

/// FRINTP (ceil): `frintp dd, dn`.
fn frintp(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E64C000 | ((dn as u32) << 5) | dd as u32);
}

/// FRINTZ (trunc): `frintz dd, dn`.
fn frintz(code: &mut Vec<u8>, dd: u8, dn: u8) {
    emit32(code, 0x1E65C000 | ((dn as u32) << 5) | dd as u32);
}

/// FCVTZS (FP to int32, truncating): `fcvtzs wd, dn`. D-register
/// source (64-bit), W-register destination (32-bit), toward-zero.
fn fcvtzs_w(code: &mut Vec<u8>, wd: u8, dn: u8) {
    // A64I_FCVT_S32_F64 = 0x1e780000  (rmode=11 toward-zero)
    emit32(code, 0x1E780000 | ((dn as u32) << 5) | wd as u32);
}

/// FCVTNS (FP to int32, round-to-nearest-even): `fcvtns wd, dn`.
fn fcvtns_w(code: &mut Vec<u8>, wd: u8, dn: u8) {
    emit32(code, 0x1E700000 | ((dn as u32) << 5) | wd as u32);
}

/// SCVTF (int32 to FP, 64-bit): `scvtf dd, wn`. W-register source
/// (32-bit), D-register destination (64-bit), signed.
fn scvtf_w(code: &mut Vec<u8>, dd: u8, wn: u8) {
    // A64I_FCVT_F64_S32 = 0x1E620000 (32-bit Wn → 64-bit Dd)
    emit32(code, 0x1E620000 | ((wn as u32) << 5) | dd as u32);
}

/// FCMP: `fcmp dn, dm`.
fn fcmp(code: &mut Vec<u8>, dn: u8, dm: u8) {
    emit32(code, 0x1E602000 | ((dm as u32) << 16) | ((dn as u32) << 5));
}

/// FCMP with zero: `fcmp dn, #0.0`.
fn fcmp_zero(code: &mut Vec<u8>, dn: u8) {
    emit32(code, 0x1E602008 | ((dn as u32) << 5));
}

/// CSRINC: conditionally move a GPR.
fn csinc(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8, cond: u8) {
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x1A800400 | ((rm as u32) << 16) | ((cond as u32 - 1) << 12) | ((rn as u32) << 5) | rd as u32,
    );
}

/// LDR (literal): `ldr rd, #offset` (from the current PC, 19-bit).
fn ldr_literal(code: &mut Vec<u8>, rd: u8, offset: i32, size: u8) {
    debug_assert!(offset % 4 == 0 && offset >= -(1 << 20) && offset < (1 << 20));
    let opc = match size {
        64 => 0b01u32 << 30,
        32 => 0b00,
        _ => unreachable!(),
    };
    let imm19 = ((offset as u32) >> 2) & 0x7FFFF;
    emit32(code, opc | 0x18000000 | (imm19 << 5) | rd as u32);
}

// ---------------------------------------------------------------------------
// Assembler state
// ---------------------------------------------------------------------------

/// Number of allocatable FP registers (v0–v4).
const NREG: usize = 5;
/// Scratch FP register (never allocated).
const FP_SCRATCH: u8 = 16;
/// Scratch GPRs.
const RSCR: u8 = 0;
const RSCR2: u8 = 1;
const RSCR3: u8 = 2;

const RBASE: u8 = 19;
const RENV: u8 = 20;
/// Call-target scratch GPR (x3, not used for arguments).
const RCALL: u8 = 3;

/// Total frame size: 16 (fp+lr) + 10×8 (x21-x28, x19-x20) + 10×16 (v8-v15) = 256.
const FRAME_SIZE: i32 = 256;

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

struct StubIdx {
    snapidx: usize,
    flush: Vec<(u8, IRRef)>,
    gc: bool,
}

struct Asm<'a> {
    tr: &'a GCtrace,
    code: Vec<u8>,
    cur: IRRef,
    snapidx: usize,
    last_use: Vec<IRRef>,
    klast_use: Vec<IRRef>,
    needs_env: Vec<bool>,
    env_valid: Vec<bool>,
    owner: [Owner; NREG],
    /// Snapshot of `owner` at the LOOP instruction (for `asm_loop_back`).
    s0: [Owner; NREG],
    loc: Vec<Option<u8>>,
    loop_pos: Option<usize>,
    link: Option<*const u8>,
    stubs: Vec<StubIdx>,
    stub_tails: Vec<(u32, u32)>,
    stub_positions: Vec<usize>,
    phis: Vec<PhiInfo>,
    fixups: Vec<(usize, usize)>,
}

impl<'a> Asm<'a> {
    fn iidx(r: IRRef) -> usize {
        (r - REF_BIAS) as usize
    }
    fn kidx(r: IRRef) -> usize {
        (REF_BIAS - 1 - r) as usize
    }
    fn env_disp(r: IRRef) -> i32 {
        (Self::iidx(r) * 8) as i32
    }

    /// Pass-1: NYI scan + lifetime analysis (port of x64.rs pass1).
    fn new(tr: &'a GCtrace, link: Option<*const u8>) -> Result<Asm<'a>, TraceError> {
        let nins = tr.ir.nins();
        let mut a = Asm {
            tr,
            code: Vec::with_capacity(4096),
            cur: 0,
            snapidx: 0,
            last_use: vec![0; (nins - REF_BIAS) as usize],
            klast_use: vec![0; REF_BIAS as usize],
            needs_env: vec![false; (nins - REF_BIAS) as usize],
            env_valid: vec![false; (nins - REF_BIAS) as usize],
            owner: [Owner::None; NREG],
            s0: [Owner::None; NREG],
            loc: vec![None; (nins - REF_BIAS) as usize],
            loop_pos: None,
            link,
            stubs: Vec::new(),
            stub_tails: Vec::new(),
            stub_positions: Vec::new(),
            phis: Vec::new(),
            fixups: Vec::new(),
        };

        for r in REF_FIRST..nins {
            let ins = tr.ir.ir(r);
            match ins.op() {
                IROp::NOP
                | IROp::BASE
                | IROp::LOOP
                | IROp::SLOAD
                | IROp::ULOAD => {}
                IROp::FLOAD | IROp::HLOAD | IROp::CARG => {
                    // Helper-call arguments are read from env as raw bits.
                    for op in [ins.op1 as IRRef, ins.op2 as IRRef] {
                        if op >= REF_BIAS {
                            a.needs_env[Self::iidx(op)] = true;
                        }
                    }
                }
                IROp::HSTORE => {
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                }
                IROp::CALLL => {
                    if super::super::record::ircall_arity(ins.op2 as u32) == 1
                        && ins.op1 as IRRef >= REF_BIAS
                    {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                }
                IROp::ALOAD => {
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::ASTORE => {
                    if ins.op1 as IRRef >= REF_BIAS {
                        a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                    }
                }
                IROp::TNEW | IROp::TDUP
                | IROp::GCSTEP => {}
                IROp::POW | IROp::TOBIT | IROp::BSWAP => {
                    a.mark_use(ins.op1 as IRRef, r);
                }
                IROp::BAND | IROp::BOR | IROp::BXOR | IROp::BSHL | IROp::BSHR
                | IROp::BSAR | IROp::BROL | IROp::BROR | IROp::BNOT => {
                    a.mark_use(ins.op1 as IRRef, r);
                    if ins.op2 != 0 {
                        a.mark_use(ins.op2 as IRRef, r);
                    }
                }
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::MIN | IROp::MAX
                | IROp::NEG | IROp::ABS => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::FPMATH => a.mark_use(ins.op1 as IRRef, r),
                IROp::LT | IROp::GE | IROp::LE | IROp::GT | IROp::ULT | IROp::UGE
                | IROp::ULE | IROp::UGT => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                }
                IROp::EQ | IROp::NE => {
                    a.mark_use(ins.op1 as IRRef, r);
                    a.mark_use(ins.op2 as IRRef, r);
                    if !irt_isnum(ins.t()) {
                        if ins.op1 as IRRef >= REF_BIAS {
                            a.needs_env[Self::iidx(ins.op1 as IRRef)] = true;
                        }
                        if ins.op2 as IRRef >= REF_BIAS {
                            a.needs_env[Self::iidx(ins.op2 as IRRef)] = true;
                        }
                    }
                }
                IROp::PHI => {
                    let inf = tr.ir.nins();
                    a.mark_use(ins.op1 as IRRef, inf);
                    a.mark_use(ins.op2 as IRRef, inf);
                    // PHI result must be stored to env so the post-loop
                    // body (after LOOP) can load it via fetch_fp.
                    a.needs_env[Self::iidx(r)] = true;
                    let num = irt_isnum(ins.t());
                    a.phis.push(PhiInfo {
                        phi: r,
                        lref: ins.op1 as IRRef,
                        rref: ins.op2 as IRRef,
                        num,
                    });
                }
                _ => return Err(TraceError::NYIIR),
            }
        }

        // Mark snapshot references as needing env storage.
        for snap in &tr.snap {
            let ofs = snap.mapofs as usize;
            for sn in &tr.snapmap[ofs..ofs + snap.nent as usize] {
                let rref = snap_ref(*sn);
                if rref >= REF_BIAS {
                    a.mark_use(rref, nins);
                    if !irt_isnum(tr.ir.ir(rref).t()) {
                        a.needs_env[Self::iidx(rref)] = true;
                    }
                }
            }
        }
        // Side-trace handover: inherited SLOADs are pre-filled in env.
        for &(own, _) in &tr.parentmap {
            a.env_valid[Self::iidx(own as IRRef)] = true;
        }
        // The PHI result env slot doubles as the back-edge scratch buffer.
        // A PHI that is referenced by a later instruction (last_use != 0)
        // requires a more elaborate loop-carried-value machinery that NYI
        // for now (portable executor runs the trace instead). Match x64.
        for p in &a.phis {
            if a.last_use[Self::iidx(p.phi)] != 0 {
                return Err(TraceError::NYIIR);
            }
        }

        Ok(a)
    }

    fn mark_use(&mut self, r: IRRef, at: IRRef) {
        if r >= REF_BIAS {
            self.last_use[Self::iidx(r)] = self.last_use[Self::iidx(r)].max(at);
        } else {
            self.klast_use[Self::kidx(r)] = self.klast_use[Self::kidx(r)].max(at);
        }
    }

    // -- Register allocator (ARM64 ports of x64.rs equivalents) --------------

    #[inline]
    fn pin(rg: u8) -> u16 { 1u16 << (rg as u32) }

    #[inline]
    fn reg_of(&self, r: IRRef) -> Option<u8> {
        if r >= REF_BIAS { self.loc[Self::iidx(r)] } else { None }
    }

    #[inline]
    fn dying(&self, r: IRRef) -> bool {
        r >= REF_BIAS && self.last_use[Self::iidx(r)] <= self.cur
    }

    fn steal_quiet(&mut self, rg: u8) {
        match self.owner[rg as usize] {
            Owner::Ins(o) => self.loc[Self::iidx(o)] = None,
            Owner::Konst(_) => {}
            Owner::None => {}
        }
        self.owner[rg as usize] = Owner::None;
    }

    const ALLOC_REGS: [u8; 5] = [0, 1, 2, 3, 4];

    /// Allocate an FP register, evicting the farthest-use value if needed.
    fn alloc(&mut self, pinned: u16) -> Result<u8, TraceError> {
        for &rg in &Self::ALLOC_REGS {
            if pinned & Self::pin(rg) == 0 && self.owner[rg as usize] == Owner::None {
                return Ok(rg);
            }
        }
        for &rg in &Self::ALLOC_REGS {
            if pinned & Self::pin(rg) != 0 { continue; }
            let dead = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)] < self.cur,
                Owner::Konst(o) => self.klast_use[Self::kidx(o)] < self.cur,
                Owner::None => unreachable!(),
            };
            if dead { self.steal_quiet(rg); return Ok(rg); }
        }
        let mut best: Option<(u8, IRRef)> = None;
        for &rg in &Self::ALLOC_REGS {
            if pinned & Self::pin(rg) != 0 { continue; }
            let lu = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)],
                Owner::Konst(o) => self.klast_use[Self::kidx(o)],
                Owner::None => unreachable!(),
            };
            if best.is_none_or(|(_, b)| lu > b) { best = Some((rg, lu)); }
        }
        let Some((rg, _)) = best else { return Err(TraceError::BADRA); };
        if let Owner::Ins(o) = self.owner[rg as usize] {
            let i = Self::iidx(o);
            if !self.env_valid[i] {
                str_fp(&mut self.code, rg, RENV, Self::env_disp(o));
                self.env_valid[i] = true;
            }
        }
        self.steal_quiet(rg);
        Ok(rg)
    }

    /// Bring an operand into an FP register.
    fn fetch_fp(&mut self, r: IRRef, pinned: u16) -> Result<u8, TraceError> {
        if let Some(rg) = self.reg_of(r) { return Ok(rg); }
        let rg = self.alloc(pinned)?;
        if r >= REF_BIAS {
            let i = Self::iidx(r);
            debug_assert!(self.env_valid[i]);
            ldr_fp(&mut self.code, rg, RENV, Self::env_disp(r));
            self.owner[rg as usize] = Owner::Ins(r);
            self.loc[i] = Some(rg);
        } else {
            let bits = super::super::exec::const_bits(&self.tr.ir, r);
            mov_imm64(&mut self.code, RSCR, bits);
            fmov_gpr_fp(&mut self.code, rg, RSCR);
            self.owner[rg as usize] = Owner::Konst(r);
        }
        Ok(rg)
    }

    /// Fetch op1 as the (destroyed) destination of a two-address op.
    fn into_dst(&mut self, a: IRRef) -> Result<u8, TraceError> {
        let r1 = self.fetch_fp(a, 0)?;
        if self.dying(a) { self.steal_quiet(r1); Ok(r1) }
        else {
            let d = self.alloc(Self::pin(r1))?;
            fmov_reg(&mut self.code, d, r1);
            Ok(d)
        }
    }

    /// Bind the current instruction's result.
    fn def(&mut self, d: u8) {
        let i = Self::iidx(self.cur);
        self.owner[d as usize] = Owner::Ins(self.cur);
        self.loc[i] = Some(d);
        if self.needs_env[i] {
            str_fp(&mut self.code, d, RENV, Self::env_disp(self.cur));
            self.env_valid[i] = true;
        }
    }

    /// Raw 64-bit value of an operand into a GPR.
    fn gpr_load_ref(&mut self, gpr: u8, r: IRRef) {
        if r >= REF_BIAS {
            debug_assert!(self.env_valid[Self::iidx(r)]);
            ldr_imm(&mut self.code, gpr, RENV, Self::env_disp(r), 64);
        } else {
            mov_imm64(&mut self.code, gpr, super::super::exec::const_bits(&self.tr.ir, r));
        }
    }

    /// S-register (GPR) view of an FP register — not needed with str_fp/ldr_fp.
    #[allow(dead_code)]
    fn sreg_of(fp: u8) -> u8 { fp }

    // -- Exit stub helpers ---------------------------------------------------

    fn exit_flush_set(&self, snapidx: usize) -> Vec<(u8, IRRef)> {
        let snap = &self.tr.snap[snapidx];
        let ofs = snap.mapofs as usize;
        let mut flush = Vec::new();
        for sn in &self.tr.snapmap[ofs..ofs + snap.nent as usize] {
            let rref = snap_ref(*sn);
            if rref >= REF_BIAS && irt_isnum(self.tr.ir.ir(rref).t()) {
                if let Some(rg) = self.reg_of(rref) {
                    if !self.env_valid[Self::iidx(rref)] {
                        flush.push((rg, rref));
                    }
                }
            }
        }
        flush
    }

    fn exit_code(&self, snapidx: usize) -> u32 {
        (self.tr.traceno << 16) | snapidx as u32
    }

    fn make_stub(&mut self, gc: bool) -> usize {
        let flush = self.exit_flush_set(self.snapidx);
        self.stubs.push(StubIdx { snapidx: self.snapidx, flush, gc });
        self.stubs.len() - 1
    }

    /// Emit a conditional branch to a guard-exit stub (patched later).
    /// Takes an ARM64 condition code.
    fn guard(&mut self, cc: u8) {
        let stub = self.make_stub(false);
        let pos = self.code.len();
        b_cond(&mut self.code, cc, 0); // placeholder
        self.fixups.push((pos, stub));
    }

    /// GC-debt guard (never patched).
    fn guard_gc(&mut self, cc: u8) {
        let stub = self.make_stub(true);
        let pos = self.code.len();
        b_cond(&mut self.code, cc, 0);
        self.fixups.push((pos, stub));
    }

    // -- Snapshot tail restore (mirrors x64 tail_restore) --------------------

    fn tail_restore(&mut self, snapidx: usize) {
        let snap = &self.tr.snap[snapidx];
        let ofs = snap.mapofs as usize;
        for &sn in &self.tr.snapmap[ofs..ofs + snap.nent as usize] {
            if sn & SNAP_NORESTORE != 0 { continue; }
            let disp = (snap_slot(sn) as i32 - 2) * 8;
            let rref = snap_ref(sn);
            if rref >= REF_BIAS {
                if let Some(rg) = self.reg_of(rref) {
                    // FP value in register: store to Lua stack slot.
                    str_fp(&mut self.code, rg, RBASE, disp);
                } else {
                    debug_assert!(self.env_valid[Self::iidx(rref)]);
                    ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(rref), 64);
                    str_imm(&mut self.code, RSCR, RBASE, disp, 64);
                }
            } else {
                mov_imm64(&mut self.code, RSCR, super::super::exec::const_bits(&self.tr.ir, rref));
                str_imm(&mut self.code, RSCR, RBASE, disp, 64);
            }
        }
    }

    // -- Simple IR ops -------------------------------------------------------

    fn asm_arith(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        let (mut a, mut b) = (ins.op1 as IRRef, ins.op2 as IRRef);
        if matches!(op, IROp::ADD | IROp::MUL)
            && !self.dying(a) && self.dying(b) && self.reg_of(b).is_some()
        { std::mem::swap(&mut a, &mut b); }
        let d = self.into_dst(a)?;
        let rhs = if b == a { d } else { self.fetch_fp(b, Self::pin(d))? };
        match op {
            IROp::ADD => fadd(&mut self.code, d, d, rhs),
            IROp::SUB => fsub(&mut self.code, d, d, rhs),
            IROp::MUL => fmul(&mut self.code, d, d, rhs),
            IROp::DIV => fdiv(&mut self.code, d, d, rhs),
            IROp::MIN => fmin(&mut self.code, d, d, rhs),
            IROp::MAX => fmax(&mut self.code, d, d, rhs),
            _ => unreachable!(),
        }
        self.def(d);
        Ok(())
    }

    // -- Bit ops / TOBIT / ALOAD-ASTORE / POW --------------------------------

    /// Fused bit ops: convert the (range-guarded) FP operands with
    /// fcvtzs (truncate), run the 32-bit ALU op, convert back with scvtf.
    fn asm_bitop(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let op = ins.op();
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        // Fetch both before converting: the second fetch may clobber
        // scratch registers that the first conversion needs.
        let sy = if !matches!(op, IROp::BNOT | IROp::BSWAP) {
            if ins.op2 == ins.op1 { Some(sx) }
            else { Some(self.fetch_fp(ins.op2 as IRRef, Self::pin(sx))?) }
        } else { None };
        fcvtzs_w(&mut self.code, 0, sx); // w0 = trunc(sx)
        if let Some(s) = sy { fcvtzs_w(&mut self.code, 1, s); }
        match op {
            IROp::BNOT => eor_imm(&mut self.code, 0, 0, 0xFFFFFFFF),
            IROp::BSWAP => rev32(&mut self.code, 0, 0),
            IROp::BAND => and_reg(&mut self.code, 0, 0, 1),
            IROp::BOR  => orr_reg(&mut self.code, 0, 0, 1),
            IROp::BXOR => eor_reg(&mut self.code, 0, 0, 1),
            IROp::BSHL => lsl_reg(&mut self.code, 0, 0, 1),
            IROp::BSHR => lsr_reg(&mut self.code, 0, 0, 1),
            IROp::BROL => {
                neg_w(&mut self.code, 1, 1);     // w1 = -w1
                and_imm32(&mut self.code, 1, 1, 31); // w1 &= 31
                // 32-bit ror: sf=0
                let sf_w = 0u32;
                emit32(&mut self.code, sf_w | 0x1AC02C00 | ((1u32) << 16) | 0);
            }
            IROp::BROR => {
                // ROR w0, w0, w1 (32-bit)
                let sf_w = 0u32;
                emit32(&mut self.code, sf_w | 0x1AC02C00 | ((1u32) << 16) | 0);
            }
            _ => { /* BSAR: asr w0, w0, w1 (32-bit) */
                let sf_w = 0u32;
                emit32(&mut self.code, sf_w | 0x1AC02800 | ((1u32) << 16) | 0);
            }
        }
        let d = self.alloc(0)?;
        scvtf_w(&mut self.code, d, 0);
        self.def(d);
        Ok(())
    }

    /// TOBIT: wrapping num -> int32 -> num (round-to-nearest-even, same
    /// as num2bit in the guarded i32 range).
    fn asm_tobit(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let sx = self.fetch_fp(ins.op1 as IRRef, 0)?;
        fcvtns_w(&mut self.code, 0, sx);
        let d = self.alloc(0)?;
        scvtf_w(&mut self.code, d, 0);
        self.def(d);
        Ok(())
    }

    /// POW: vm_pow via a helper call.
    fn asm_pow(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        self.helper_call(
            super::super::exec::jit_pow as *const () as u64,
            &[ins.op1 as IRRef, ins.op2 as IRRef],
        );
        let i = Self::iidx(self.cur);
        if self.last_use[i] != 0 || self.needs_env[i] {
            let d = self.alloc(0)?;
            fmov_gpr_fp(&mut self.code, d, RSCR);
            self.def(d);
        }
        Ok(())
    }

    // -- ALOAD / ASTORE (array inlining) -------------------------------------

    /// Common head: load table pointer into x0, check key is exact int32
    /// in array range. key is in FP register, result: x0=table, w1=index.
    fn asm_array_head(&mut self, tab: IRRef, key: IRRef) -> Result<(), TraceError> {
        const ASIZE_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, asize) as i32;
        // Fetch key first (may use RAX/GPRs for constants).
        let sk = self.fetch_fp(key, 0)?;
        self.gpr_load_ref(0, tab);
        mov_imm64(&mut self.code, 1, crate::value::LJ_GCVMASK);
        and_reg(&mut self.code, 0, 0, 1);
        fcvtzs_w(&mut self.code, 1, sk);       // w1 = trunc(key)
        scvtf_w(&mut self.code, FP_SCRATCH, 1);
        fcmp(&mut self.code, FP_SCRATCH, sk);
        self.guard(CC_NE);                       // not exact int → exit
        // Check w1 < asize (unsigned).
        ldr_imm(&mut self.code, 2, 0, ASIZE_OFF, 32);
        cmp_reg(&mut self.code, 1, 2);
        self.guard(CC_CS);                       // unsigned >= → exit
        Ok(())
    }

    fn asm_aload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        self.asm_array_head(ins.op1 as IRRef, ins.op2 as IRRef)?;
        // Load aptr, then value.
        ldr_imm(&mut self.code, 2, 0, APTR_OFF, 64);
        // ldr x3, [x2, w1, uxtw #3]
        ldr_reg_uxtw(&mut self.code, 0, 2, 1);
        self.ff_result(ins)
    }

    fn asm_astore(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        const APTR_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, aptr) as i32;
        let carg = *self.tr.ir.ir(ins.op2 as IRRef);
        debug_assert_eq!(carg.op(), IROp::CARG);
        self.asm_array_head(ins.op1 as IRRef, carg.op1 as IRRef)?;
        // k > 0 guard (set_int requires).
        cmp_imm(&mut self.code, 1, 0, 0);
        self.guard(CC_EQ);
        ldr_imm(&mut self.code, 2, 0, APTR_OFF, 64); // x2 = aptr
        let val = carg.op2 as IRRef;
        if val >= REF_BIAS && let Some(sv) = self.reg_of(val) {
            fmov_fp_gpr(&mut self.code, 3, sv);
            str_reg_uxtw(&mut self.code, 3, 2, 1);
        } else {
            self.gpr_load_ref(3, val);
            str_reg_uxtw(&mut self.code, 3, 2, 1);
        }
        Ok(())
    }

    // -- Loop optimisation / Tail --------------------------------------------

    fn asm_loop_head(&mut self) {
        for rg in 0..NREG as u8 {
            let dead = match self.owner[rg as usize] {
                Owner::Ins(o) => self.last_use[Self::iidx(o)] <= self.cur,
                Owner::Konst(k) => self.klast_use[Self::kidx(k)] <= self.cur,
                Owner::None => false,
            };
            if dead { self.steal_quiet(rg); }
        }
        // Spill all live non-PHI regs into their env slots so the
        // post-loop body can find values even if registers were stolen.
        for rg in 0..NREG as u8 {
            if let Owner::Ins(o) = self.owner[rg as usize] {
                if self.phis.iter().any(|p| p.num && p.lref == o) { continue; }
                let i = Self::iidx(o);
                if !self.env_valid[i] {
                    str_fp(&mut self.code, rg, RENV, Self::env_disp(o));
                    self.env_valid[i] = true;
                }
            }
        }
        // Pre-fill PHI env slots with the current register value so
        // the post-loop body's first iteration can find the value.
        for p in &self.phis {
            let ii = Self::iidx(p.phi);
            if p.num && !self.env_valid[ii] {
                if let Some(lr) = self.loc[Self::iidx(p.lref)] {
                    // lref still has a register — store to phi env
                    str_fp(&mut self.code, lr, RENV, Self::env_disp(p.phi));
                    self.env_valid[ii] = true;
                } else {
                    // lref stolen — search all regs for a live value
                    for rg in 0..NREG as u8 {
                        if let Owner::Ins(_) = self.owner[rg as usize] {
                            str_fp(&mut self.code, rg, RENV, Self::env_disp(p.phi));
                            self.env_valid[ii] = true;
                            break;
                        }
                    }
                }
            }
        }
        // Reset PHI lref env_valid so the back-edge sees them as dirty.
        for p in &self.phis {
            let li = Self::iidx(p.lref);
            if p.num && self.loc[li].is_some() && !self.needs_env[li] {
                self.env_valid[li] = false;
            }
        }
        self.s0 = self.owner;
        self.loop_pos = Some(self.code.len());
    }

    fn asm_loop_back(&mut self, loop_pos: usize) {
        let s0 = self.s0;
        let phis = std::mem::take(&mut self.phis);
        let homes: Vec<Option<u8>> = phis.iter()
            .map(|p| (0..NREG as u8).find(|&rg| s0[rg as usize] == Owner::Ins(p.lref)))
            .collect();
        let dstset: u16 = homes.iter().flatten().fold(0, |m, &rg| m | Self::pin(rg));
        let direct = phis.iter().all(|p| {
            if p.rref < REF_BIAS { return true; }
            if let Some(rg) = self.reg_of(p.rref) && p.num {
                return dstset & Self::pin(rg) == 0;
            }
            !phis.iter().any(|q| q.lref == p.rref)
        });
        if direct {
            for (p, home) in phis.iter().zip(homes.iter()) {
                self.phi_move(p, *home);
            }
        } else {
            // Buffered: read right values into PHI env slots, then land.
            for p in &phis {
                if p.rref >= REF_BIAS {
                    if p.num && let Some(rg) = self.reg_of(p.rref) {
                        str_fp(&mut self.code, rg, RENV, Self::env_disp(p.phi));
                    } else {
                        ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.rref), 64);
                        str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.phi), 64);
                    }
                } else {
                    mov_imm64(&mut self.code, RSCR,
                        super::super::exec::const_bits(&self.tr.ir, p.rref));
                    str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.phi), 64);
                }
            }
            for (p, home) in phis.iter().zip(homes.iter()) {
                if p.num && let Some(rg) = *home {
                    ldr_fp(&mut self.code, rg, RENV, Self::env_disp(p.phi));
                    if self.needs_env[Self::iidx(p.lref)] {
                        str_fp(&mut self.code, rg, RENV, Self::env_disp(p.lref));
                    }
                    self.owner[rg as usize] = Owner::Ins(p.lref);
                    self.loc[Self::iidx(p.lref)] = Some(rg);
                } else {
                    ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.phi), 64);
                    str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.lref), 64);
                }
            }
        }
        // Restore invariants from env.
        for rg in 0..NREG as u8 {
            let so = s0[rg as usize];
            if so == Owner::None || so == self.owner[rg as usize] { continue; }
            if phis.iter().any(|p| so == Owner::Ins(p.lref)) { continue; }
            match so {
                Owner::Ins(x) => {
                    ldr_fp(&mut self.code, rg, RENV, Self::env_disp(x));
                    self.owner[rg as usize] = Owner::Ins(x);
                    self.loc[Self::iidx(x)] = Some(rg);
                }
                Owner::Konst(k) => {
                    mov_imm64(&mut self.code, RSCR,
                        super::super::exec::const_bits(&self.tr.ir, k));
                    fmov_gpr_fp(&mut self.code, rg, RSCR);
                    self.owner[rg as usize] = Owner::Konst(k);
                }
                Owner::None => unreachable!(),
            }
        }
        // Jump back.
        let offset = loop_pos as i32 - self.code.len() as i32;
        b_imm(&mut self.code, offset, false);
    }

    fn phi_move(&mut self, p: &PhiInfo, home: Option<u8>) {
        if p.rref < REF_BIAS {
            mov_imm64(&mut self.code, RSCR,
                super::super::exec::const_bits(&self.tr.ir, p.rref));
            if p.num {
                if let Some(rg) = home {
                    fmov_gpr_fp(&mut self.code, rg, RSCR);
                    if self.needs_env[Self::iidx(p.lref)] {
                        str_fp(&mut self.code, rg, RENV, Self::env_disp(p.lref));
                    }
                    if self.needs_env[Self::iidx(p.phi)] {
                        str_fp(&mut self.code, rg, RENV, Self::env_disp(p.phi));
                    }
                } else {
                    str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.lref), 64);
                    if self.needs_env[Self::iidx(p.phi)] {
                        str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.phi), 64);
                    }
                }
            }
        } else if p.num && let Some(rg) = self.reg_of(p.rref) {
            let d = home.unwrap_or(0);
            if d != rg { fmov_reg(&mut self.code, d, rg); }
            if self.needs_env[Self::iidx(p.lref)] {
                str_fp(&mut self.code, d, RENV, Self::env_disp(p.lref));
            }
            if self.needs_env[Self::iidx(p.phi)] {
                str_fp(&mut self.code, d, RENV, Self::env_disp(p.phi));
            }
            // After moving the value into `d`, restore the lref's register
            // mapping so that reg_of(lref) works in the next iteration.
            // Also map the pre-loop source ref (which opt_loop resolves
            // via tref_ref, e.g. MUL_pre_ref) to `d` so the post-loop body
            // can find the value via reg_of instead of loading from env.
            self.loc[Self::iidx(p.lref)] = Some(d);
            self.owner[d as usize] = Owner::Ins(p.lref);
            let post_ir = self.tr.ir.ir(p.rref);
            let pre_ref = post_ir.op1 as IRRef;
            if pre_ref >= REF_BIAS && p.num && self.loc[Self::iidx(pre_ref)].is_none() {
                self.loc[Self::iidx(pre_ref)] = Some(d);
            }
        } else {
            debug_assert!(self.env_valid[Self::iidx(p.rref)]);
            if p.num && let Some(rg) = home {
                ldr_fp(&mut self.code, rg, RENV, Self::env_disp(p.rref));
                self.loc[Self::iidx(p.lref)] = Some(rg);
                self.owner[rg as usize] = Owner::Ins(p.lref);
            }
            ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.rref), 64);
            str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.lref), 64);
            if self.needs_env[Self::iidx(p.phi)] {
                str_imm(&mut self.code, RSCR, RENV, Self::env_disp(p.phi), 64);
            }
        }
    }

    // -- Emit loop -----------------------------------------------------------

    fn emit(mut self) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
        // AAPCS64 prologue: x0=base, x1=env
        const FRAME: i32 = 256;
        sub_imm(&mut self.code, 31, 31, FRAME as u32, 0);   // sub sp, sp, #256
        stp_offset(&mut self.code, 29, 30, 31, 0);           // stp x29, x30, [sp, #0]
        // Save x19-x28 at [sp + 16*k]
        stp_offset(&mut self.code, 19, 20, 31, 16);
        stp_offset(&mut self.code, 21, 22, 31, 32);
        stp_offset(&mut self.code, 23, 24, 31, 48);
        stp_offset(&mut self.code, 25, 26, 31, 64);
        stp_offset(&mut self.code, 27, 28, 31, 80);
        // v8-v15: saved in area [sp+96..sp+256] — skip saves for now (traces
        // don't use callee-saved FP regs yet; all FP state is in v0-v4).
        mov_reg(&mut self.code, RBASE, 0);   // x19 = x0 (base)
        mov_reg(&mut self.code, RENV, 1);    // x20 = x1 (env)
        let inner = self.code.len() as u32;

        // Parentmap handover
        if !self.tr.parentmap.is_empty() {
            self.emit_handover();
        }

        let nins = self.tr.ir.nins();
        let mut r = REF_FIRST;
        while r < nins {
            while self.snapidx + 1 < self.tr.snap.len()
                && self.tr.snap[self.snapidx + 1].iref <= r
            { self.snapidx += 1; }
            self.cur = r;
            let ins = *self.tr.ir.ir(r);
            match ins.op() {
                IROp::NOP | IROp::BASE | IROp::PHI => {}
                IROp::LOOP => self.asm_loop_head(),
                IROp::SLOAD => self.asm_sload(&ins)?,
                IROp::ADD | IROp::SUB | IROp::MUL | IROp::DIV | IROp::MIN | IROp::MAX => {
                    self.asm_arith(&ins)?;
                }
                IROp::NEG => {
                    let d = self.into_dst(ins.op1 as IRRef)?;
                    fneg(&mut self.code, d, d);
                    self.def(d);
                }
                IROp::ABS => {
                    let d = self.into_dst(ins.op1 as IRRef)?;
                    fabs(&mut self.code, d, d);
                    self.def(d);
                }
                IROp::FPMATH => {
                    let src = self.fetch_fp(ins.op1 as IRRef, 0)?;
                    let d = if self.dying(ins.op1 as IRRef) {
                        self.steal_quiet(src); src
                    } else {
                        let d2 = self.alloc(Self::pin(src))?;
                        fmov_reg(&mut self.code, d2, src);
                        d2
                    };
                    match ins.op2 as u32 {
                        super::super::record::IRFPM_FLOOR => frintm(&mut self.code, d, d),
                        super::super::record::IRFPM_CEIL => frintp(&mut self.code, d, d),
                        super::super::record::IRFPM_TRUNC => frintz(&mut self.code, d, d),
                        super::super::record::IRFPM_SQRT => fsqrt(&mut self.code, d, d),
                        _ => unreachable!(),
                    }
                    self.def(d);
                }
                IROp::LT | IROp::GE | IROp::LE | IROp::GT
                | IROp::ULT | IROp::UGE | IROp::ULE | IROp::UGT => {
                    self.asm_comp(&ins)?;
                }
                IROp::EQ | IROp::NE => self.asm_equal(&ins)?,
                IROp::CARG => {} // Consumed by HSTORE/CALLL/ASTORE.
                IROp::CALLL => self.asm_calll(&ins)?,
                IROp::TNEW => {
                    self.helper_call(super::super::exec::jit_tnew as *const () as u64, &[]);
                    self.ff_result(&ins)?;
                }
                IROp::TDUP => {
                    self.helper_call(
                        super::super::exec::jit_tdup as *const () as u64,
                        &[ins.op1 as IRRef],
                    );
                    self.ff_result(&ins)?;
                }
                IROp::HSTORE => {
                    let carg = *self.tr.ir.ir(ins.op2 as IRRef);
                    debug_assert_eq!(carg.op(), IROp::CARG);
                    self.helper_call(
                        super::super::exec::jit_tset as *const () as u64,
                        &[ins.op1 as IRRef, carg.op1 as IRRef, carg.op2 as IRRef],
                    );
                }
                IROp::GCSTEP => self.asm_gcstep(&ins),
                IROp::FLOAD => self.asm_fload(&ins),
                IROp::HLOAD => {
                    self.helper_call(
                        super::super::exec::jit_tget as *const () as u64,
                        &[ins.op1 as IRRef, ins.op2 as IRRef],
                    );
                    self.ff_result(&ins)?;
                }
                IROp::ULOAD => self.asm_uload(&ins)?,
                IROp::BAND | IROp::BOR | IROp::BXOR | IROp::BSHL | IROp::BSHR
                | IROp::BSAR | IROp::BROL | IROp::BROR | IROp::BNOT | IROp::BSWAP => {
                    self.asm_bitop(&ins)?;
                }
                IROp::TOBIT => self.asm_tobit(&ins)?,
                IROp::POW => self.asm_pow(&ins)?,
                IROp::ALOAD => self.asm_aload(&ins)?,
                IROp::ASTORE => self.asm_astore(&ins)?,
                _ => return Err(TraceError::NYIIR),
            }
            r += 1;
        }

        // Tail section
        let lastsnap = self.tr.snap.len() - 1;
        let looping = self.tr.linktype == TraceLink::Loop && self.tr.link == self.tr.traceno;
        let recursing = matches!(self.tr.linktype, TraceLink::Uprec | TraceLink::Tailrec)
            && self.tr.link == self.tr.traceno;
        if looping {
            if let Some(lp) = self.loop_pos {
                self.asm_loop_back(lp);
            } else {
                self.snapidx = lastsnap;
                self.tail_restore(lastsnap);
                let head_ofs = inner as i32 - self.code.len() as i32;
                b_imm(&mut self.code, head_ofs, false);
            }
        } else if recursing {
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let delta = (self.tr.snap[lastsnap].baseslot as i32 - 2) * 8;
            // rax_equiv(x0) = RBASE + delta + headroom
            mov_reg(&mut self.code, 0, RBASE);
            add_imm(&mut self.code, 0, 0, (delta + (255 + 8) * 8) as u32, 0);
            mov_imm64(&mut self.code, 1, super::super::exec::stack_end_cell_addr());
            ldr_imm(&mut self.code, 1, 1, 0, 64);
            cmp_reg(&mut self.code, 0, 1);
            self.guard(CC_HI);
            if delta != 0 {
                add_imm(&mut self.code, RBASE, RBASE, delta as u32, 0);
            }
            let inner_ofs = inner as i32 - self.code.len() as i32;
            b_imm(&mut self.code, inner_ofs, false);
        } else if self.tr.linktype == TraceLink::Root && let Some(target) = self.link {
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let delta = (self.tr.snap[lastsnap].baseslot as i32 - 2) * 8;
            if delta != 0 {
                mov_reg(&mut self.code, 0, RBASE);
                add_imm(&mut self.code, 0, 0, (delta + (255 + 8) * 8) as u32, 0);
                mov_imm64(&mut self.code, 1, super::super::exec::stack_end_cell_addr());
                ldr_imm(&mut self.code, 1, 1, 0, 64);
                cmp_reg(&mut self.code, 0, 1);
                self.guard(CC_HI);
                add_imm(&mut self.code, RBASE, RBASE, delta as u32, 0);
            }
            mov_imm64(&mut self.code, 0, target as u64);
            br_reg(&mut self.code, 0, false); // br x0
        } else {
            // None/Stitch: exit through final snapshot
            self.snapidx = lastsnap;
            self.tail_restore(lastsnap);
            let flush = self.exit_flush_set(lastsnap);
            for (rg, rref) in &flush {
                str_fp(&mut self.code, *rg, RENV, Self::env_disp(*rref));
            }
            let ec = self.exit_code(lastsnap);
            mov_imm64(&mut self.code, 0, ec as u64);
            // Fall through to epilogue
        }

        // -- Epilogue --
        // x0 (w0) holds the exit code from the exit path above.
        // Report BASE back through the cell, then restore and return.
        let epilogue = self.code.len();
        mov_imm64(&mut self.code, RSCR2, super::super::exec::exit_base_cell_addr());
        str_imm(&mut self.code, RBASE, RSCR2, 0, 64);
        ldp_offset(&mut self.code, 27, 28, 31, 80);
        ldp_offset(&mut self.code, 25, 26, 31, 64);
        ldp_offset(&mut self.code, 23, 24, 31, 48);
        ldp_offset(&mut self.code, 21, 22, 31, 32);
        ldp_offset(&mut self.code, 19, 20, 31, 16);
        ldp_offset(&mut self.code, 29, 30, 31, 0);
        add_imm(&mut self.code, 31, 31, FRAME as u32, 0);    // add sp, sp, #256
        ret(&mut self.code, 30);

        // -- Guard-exit stubs --
        // Stub layout (matches x64 convention for patch_exit):
        //   [flush instructions...]  variable — live FP regs to env
        //   [patch tail: 20 bytes]   fixed — exit-code movz+k+b, patched by patch_exit
        // stubpos[i] → start of stub (guard b.cond target)
        // stub_tails.push((snapidx, tail)) where tail = position right after flush
        let stubs = std::mem::take(&mut self.stubs);
        let mut stubpos = Vec::with_capacity(stubs.len());
        for st in &stubs {
            stubpos.push(self.code.len());                           // guard branch target
            for (rg, rref) in &st.flush {
                str_fp(&mut self.code, *rg, RENV, Self::env_disp(*rref));
            }
            let tail = self.code.len();                              // patch-exit point
            if !st.gc {
                self.stub_tails.push((st.snapidx as u32, tail as u32));
            }
            let ec = if st.gc { self.exit_code(st.snapidx) | 0x8000 } else { self.exit_code(st.snapidx) };
            mov_imm64_full(&mut self.code, 0, ec as u64);            // 16 bytes (4 × movz/movk)
            let epi_off = epilogue as i32 - self.code.len() as i32;
            b_imm(&mut self.code, epi_off, false);                   // 4 bytes
            // Total mov+b = 20 bytes; patch_exit overwrites this region.
        }
        self.stub_positions = stubpos;

        // Fix up guard branches
        for (pos, si) in std::mem::take(&mut self.fixups) {
            let target = self.stub_positions[si];
            let offset = target as i32 - pos as i32;
            let insn = u32::from_le_bytes(self.code[pos..pos+4].try_into().unwrap());
            let imm19 = ((offset >> 2) as u32) & 0x7FFFF;
            let new_insn = (insn & 0xFF00001F) | (imm19 << 5);
            self.code[pos..pos+4].copy_from_slice(&new_insn.to_le_bytes());
        }

        let mut area = McodeArea::alloc(self.code.len()).ok_or(TraceError::MCODEAL)?;
        area.as_mut_slice()[..self.code.len()].copy_from_slice(&self.code);

        // Dump traces on aarch64 for debugging (TODO: remove after all fixed).
        {
            eprint!(
                "[arm64] TRACE {} mcode={:5}B nins={} inner={} link={:?}",
                self.tr.traceno,
                self.code.len(),
                self.tr.ir.nins() - REF_BIAS,
                inner,
                self.tr.linktype,
            );
            if let Some(lp) = self.loop_pos {
                eprint!(" loop_pos={}", lp);
            }
            eprint!(" hex=");
            for (i, chunk) in self.code.chunks(4).enumerate() {
                let w = u32::from_le_bytes(chunk.try_into().unwrap());
                if i > 0 { eprint!(","); }
                eprint!("{:08x}", w);
            }
            // Also dump IR for context
            eprint!(" ir=");
            for r in REF_FIRST..self.tr.ir.nins() {
                let ins = self.tr.ir.ir(r);
                eprint!("{:?}", ins.op());
                if ins.op() != IROp::NOP {
                    // skip NOP details
                }
                if r + 1 < self.tr.ir.nins() { eprint!(","); }
            }
            eprintln!();
        }

        if !area.protect_exec() { return Err(TraceError::MCODEAL); }
        Ok((area, inner, std::mem::take(&mut self.stub_tails)))
    }

    // -- Handover ------------------------------------------------------------

    fn emit_handover(&mut self) {
        let mut pending: Vec<(IRRef, IRRef)> = self
            .tr.parentmap.iter()
            .map(|&(o, p)| (o as IRRef, p as IRRef))
            .filter(|&(o, p)| o != p)
            .collect();
        while !pending.is_empty() {
            let ready = pending.iter().position(|&(d, _)| {
                !pending.iter().any(|&(_, s)| s == d)
            });
            let Some(i) = ready else {
                // Cycle: use scratch register to break
                let (d, s) = pending.remove(0);
                ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(s), 64);
                str_imm(&mut self.code, RSCR, RENV, Self::env_disp(d), 64);
                continue;
            };
            let (d, s) = pending.remove(i);
            ldr_imm(&mut self.code, RSCR, RENV, Self::env_disp(s), 64);
            str_imm(&mut self.code, RSCR, RENV, Self::env_disp(d), 64);
        }
    }

    // -- Helper call / GCSTEP / CALLL ----------------------------------------

    /// Call an `extern "C"` helper with up to three u64 args. ARM64
    /// x19/x20 are callee-saved (no save needed). Caller-saved FP regs
    /// are parked to env before the call.
    fn helper_call(&mut self, addr: u64, args: &[IRRef]) {
        for rg in 0..=4u8 {
            if let Owner::Ins(o) = self.owner[rg as usize] {
                let i = Self::iidx(o);
                if self.last_use[i] > self.cur && !self.env_valid[i] {
                    str_fp(&mut self.code, rg, RENV, Self::env_disp(o));
                    self.env_valid[i] = true;
                }
            }
            self.steal_quiet(rg);
        }
        const AARGS: [u8; 3] = [0, 1, 2];
        debug_assert!(args.len() <= 3);
        for (n, &r) in args.iter().enumerate() {
            self.gpr_load_ref(AARGS[n], r);
        }
        mov_imm64(&mut self.code, RCALL, addr);
        br_reg(&mut self.code, RCALL, true); // blr x3
    }

    /// GCSTEP guard: exit when a collection is due.
    fn asm_gcstep(&mut self, ins: &IRIns) {
        let total_addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        let thres_addr = super::super::exec::const_bits(&self.tr.ir, ins.op2 as IRRef);
        let extra_addr = crate::table::TABLE_EXTRA.with(|c| c.as_ptr() as u64);
        // total
        mov_imm64(&mut self.code, RSCR, total_addr);
        ldr_imm(&mut self.code, RSCR, RSCR, 0, 64);
        // total += TABLE_EXTRA
        mov_imm64(&mut self.code, RSCR2, extra_addr);
        ldr_imm(&mut self.code, RSCR3, RSCR2, 0, 64);
        add_reg_lsl(&mut self.code, RSCR, RSCR, RSCR3, 0);
        // cmp total, threshold
        mov_imm64(&mut self.code, RSCR2, thres_addr);
        ldr_imm(&mut self.code, RSCR2, RSCR2, 0, 64);
        cmp_reg(&mut self.code, RSCR, RSCR2);
        self.guard_gc(CC_CS);
    }

    /// CALLL: guarded helper call, selected by the IRCALL index in op2.
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
            rec::IRCALL_STR_CHAR => super::super::exec::jit_str_char as *const () as u64,
            rec::IRCALL_TAB_LEN => super::super::exec::jit_alen as *const () as u64,
            rec::IRCALL_TAB_CONCAT => super::super::exec::jit_tconcat as *const () as u64,
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

    /// Shared tail of helper-returning ops: typecheck and land the result.
    fn ff_result(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let t = ins.t();
        let i = Self::iidx(self.cur);
        if irt_isnum(t) {
            if ins.is_guard() {
                // Check whether the result is a number: the top 32 bits
                // of a raw IEEE 754 double are < 0xFFF9_0000, while all
                // NaN-boxed GC tags sit at or above that threshold.
                lsr_imm(&mut self.code, RSCR2, RSCR, 32);
                mov_imm64(&mut self.code, RSCR3, 0xFFF9_0000u64);
                cmp_reg(&mut self.code, RSCR2, RSCR3);
                self.guard(CC_CS);
            }
            if self.last_use[i] != 0 || self.needs_env[i] {
                let d = self.alloc(0)?;
                fmov_gpr_fp(&mut self.code, d, RSCR);
                self.def(d);
            }
            return Ok(());
        }
        if self.needs_env[i] {
            str_imm(&mut self.code, RSCR, RENV, Self::env_disp(self.cur), 64);
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
            mov_imm64(&mut self.code, RSCR2, bits);
            cmp_reg(&mut self.code, RSCR, RSCR2);
            self.guard(CC_NE);
        } else {
            ubfx(&mut self.code, RSCR2, RSCR, 47, 8);
            cmp_imm(&mut self.code, RSCR2, (!(ty as u32) & 0xFF) as u32, 0);
            self.guard(CC_NE);
        }
        Ok(())
    }

    /// FLOAD: guarded metatable-nil check.
    fn asm_fload(&mut self, ins: &IRIns) {
        debug_assert!(ins.is_guard());
        // ins.op1 is the table ref. Load TABLE obj, check metatable field.
        // metatable offset in LuaTable.
        const MT_OFF: i32 = std::mem::offset_of!(crate::table::LuaTable, metatable) as i32;
        self.gpr_load_ref(RSCR, ins.op1 as IRRef);
        mov_imm64(&mut self.code, RSCR2, crate::value::LJ_GCVMASK);
        and_reg(&mut self.code, RSCR, RSCR, RSCR2);
        ldr_imm(&mut self.code, RSCR, RSCR, MT_OFF, 64);
        // metatable is Option<GcPtr<LuaTable>> — for our purposes, None = 0.
        cmp_imm(&mut self.code, RSCR, 0, 0);
        self.guard(CC_NE);
    }

    /// ULOAD: closed upvalue read through a constant address.
    fn asm_uload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let addr = super::super::exec::const_bits(&self.tr.ir, ins.op1 as IRRef);
        mov_imm64(&mut self.code, RSCR, addr);
        ldr_imm(&mut self.code, RSCR, RSCR, 0, 64);
        if ins.is_guard() {
            ubfx(&mut self.code, RSCR2, RSCR, 47, 8);
            let expected = (!(irt_type(ins.t()) as u32)) as u8;
            cmp_imm(&mut self.code, RSCR2, expected as u32, 0);
            self.guard(CC_NE);
        }
        let r = self.cur;
        str_imm(&mut self.code, RSCR, RENV, Self::env_disp(r), 64);
        self.env_valid[Self::iidx(r)] = true;
        Ok(())
    }

    // -- SLOAD ---------------------------------------------------------------

    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let idx = ins.op1 as i32 - 2;
        let r = self.cur;
        let offset = idx as i64 * 8;
        if offset >= 0 && offset % 8 == 0 && offset <= 32760 {
            ldr_imm(&mut self.code, RSCR, RBASE, offset as i32, 64);
        } else {
            mov_imm64(&mut self.code, RSCR2, idx as u64);
            ldr_reg_lsl3(&mut self.code, RSCR, RBASE, RSCR2);
        }
        if irt_isnum(ins.t()) {
            if ins.is_guard() {
                // LuaJIT GC64 NaN-boxing: the low 32 bits of a number
                // are the IEEE-754 payload; the high 32 bits must be
                // less than 0xFFF9_0000 (the smallest GC-tagged value).
                // cmp dword [BASE+disp+4], LJ_TISNUM<<15; jae ->exit
                lsr_imm(&mut self.code, RSCR2, RSCR, 32);
                mov_imm64(&mut self.code, RSCR3, 0xFFF9_0000u64);
                cmp_reg(&mut self.code, RSCR2, RSCR3);
                self.guard(CC_CS);
            }
            let d = self.alloc(0)?;
            fmov_gpr_fp(&mut self.code, d, RSCR);
            self.owner[d as usize] = Owner::Ins(r);
            self.loc[Self::iidx(r)] = Some(d);
            if self.needs_env[Self::iidx(r)] {
                str_fp(&mut self.code, d, RENV, Self::env_disp(r));
                self.env_valid[Self::iidx(r)] = true;
            }
        } else {
            if ins.is_guard() {
                ubfx(&mut self.code, RSCR2, RSCR, 47, 8);
                let expected = (!(irt_type(ins.t()) as u32)) as u8;
                cmp_imm(&mut self.code, RSCR2, expected as u32, 0);
                self.guard(CC_NE);
            }
            str_imm(&mut self.code, RSCR, RENV, Self::env_disp(r), 64);
            self.env_valid[Self::iidx(r)] = true;
        }
        Ok(())
    }

    // -- Comparisons ---------------------------------------------------------

    /// Emit a comparison guard: `fcmp` followed by one or two conditional
    /// branches to the exit stub. Operand swapping follows the same
    /// convention as x64 for the ordered/unordered split.
    fn asm_comp(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(irt_isnum(ins.t()) && ins.is_guard());
        let x = self.fetch_fp(ins.op1 as IRRef, 0)?;
        let y = if ins.op2 == ins.op1 { x }
                else { self.fetch_fp(ins.op2 as IRRef, Self::pin(x))? };
        fcmp(&mut self.code, x, y);
        match ins.op() {
            // Ordered: guard fails on NaN.
            IROp::LT => { self.guard(CC_VS); self.guard(CC_GE); }
            IROp::GE => { self.guard(CC_VS); self.guard(CC_MI); }
            IROp::LE => { self.guard(CC_VS); self.guard(CC_GT); }
            IROp::GT => { self.guard(CC_VS); self.guard(CC_LS); }
            // Unordered: NaN passes the guard.
            IROp::ULT => self.guard(CC_GE),
            IROp::UGE => self.guard(CC_MI),
            IROp::ULE => self.guard(CC_GT),
            IROp::UGT => self.guard(CC_LS),
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Equality / inequality guard. For FP values NaN fails EQ and
    /// passes NE.
    fn asm_equal(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        debug_assert!(ins.is_guard());
        let eq = ins.op() == IROp::EQ;
        if irt_isnum(ins.t()) {
            let x = self.fetch_fp(ins.op1 as IRRef, 0)?;
            let y = if ins.op2 == ins.op1 { x }
                    else { self.fetch_fp(ins.op2 as IRRef, Self::pin(x))? };
            fcmp(&mut self.code, x, y);
            if eq {
                self.guard(CC_VS);
                self.guard(CC_NE);
            } else {
                // NE: exit on ordered equality. Skip the exit on NaN.
                let pos = self.code.len();
                b_cond(&mut self.code, CC_VS, 8); // b.vs +8 (skip exit on NaN)
                let stub = self.make_stub(false);
                let stub_pos = self.code.len();
                b_cond(&mut self.code, CC_EQ, 0); // placeholder → patched later
                // Fix up the first b.vs to skip past this b.eq
                let after = self.code.len() as i32 - pos as i32;
                self.code[pos..pos+4].copy_from_slice(
                    &(0x54000000 | (((after >> 2) as u32 & 0x7FFFF) << 5) | CC_VS as u32).to_le_bytes()
                );
                self.fixups.push((stub_pos, stub));
            }
        } else {
            self.gpr_load_ref(RSCR, ins.op1 as IRRef);
            self.gpr_load_ref(RSCR2, ins.op2 as IRRef);
            cmp_reg(&mut self.code, RSCR, RSCR2);
            self.guard(if eq { CC_NE } else { CC_EQ });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Instruction-level verification tests (compile on any arch for inspection)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn dump_prologue_epilogue() {
        let mut c = Vec::new();
        // Prologue
        sub_imm(&mut c, 31, 31, 256, 0);       // sub sp, sp, #256
        stp_offset(&mut c, 29, 30, 31, 0);      // stp x29, x30, [sp, #0]
        stp_offset(&mut c, 19, 20, 31, 16);
        stp_offset(&mut c, 21, 22, 31, 32);
        stp_offset(&mut c, 23, 24, 31, 48);
        stp_offset(&mut c, 25, 26, 31, 64);
        stp_offset(&mut c, 27, 28, 31, 80);
        mov_reg(&mut c, 19, 0);                  // mov x19, x0
        mov_reg(&mut c, 20, 1);                  // mov x20, x1
        // Epilogue
        mov_imm64(&mut c, 1, 0xFFFF_FFFF_FFFF_FFFFu64);
        str_imm(&mut c, 19, 1, 0, 64);           // str x19, [x1]
        ldp_offset(&mut c, 27, 28, 31, 80);
        ldp_offset(&mut c, 25, 26, 31, 64);
        ldp_offset(&mut c, 23, 24, 31, 48);
        ldp_offset(&mut c, 21, 22, 31, 32);
        ldp_offset(&mut c, 19, 20, 31, 16);
        ldp_offset(&mut c, 29, 30, 31, 0);
        add_imm(&mut c, 31, 31, 256, 0);         // add sp, sp, #256
        ret(&mut c, 30);
        for (i, chunk) in c.chunks(4).enumerate() {
            let w = u32::from_le_bytes(chunk.try_into().unwrap());
            eprintln!("  {:2}: {:02x?}  // {:08x}", i*4, chunk, w);
        }
        std::fs::write("dump_prologue.hex", &c).ok();
    }

    #[test]
    fn dump_fp_ops() {
        let mut c = Vec::new();
        fmov_gpr_fp(&mut c, 0, 0);    // fmov d0, x0
        fmov_fp_gpr(&mut c, 0, 0);    // fmov x0, d0
        fmov_reg(&mut c, 1, 0);       // fmov d1, d0
        fadd(&mut c, 0, 0, 1);        // fadd d0, d0, d1
        fsub(&mut c, 0, 0, 1);        // fsub d0, d0, d1
        fmul(&mut c, 0, 0, 1);        // fmul d0, d0, d1
        fdiv(&mut c, 0, 0, 1);        // fdiv d0, d0, d1
        fneg(&mut c, 0, 0);           // fneg d0, d0
        fabs(&mut c, 0, 0);           // fabs d0, d0
        fsqrt(&mut c, 0, 0);          // fsqrt d0, d0
        frintm(&mut c, 0, 0);         // frintm d0, d0
        frintp(&mut c, 0, 0);         // frintp d0, d0
        frintz(&mut c, 0, 0);         // frintz d0, d0
        fmin(&mut c, 0, 0, 1);        // fmin d0, d0, d1
        fmax(&mut c, 0, 0, 1);        // fmax d0, d0, d1
        fcmp(&mut c, 0, 1);           // fcmp d0, d1
        fcmp_zero(&mut c, 0);         // fcmp d0, #0.0
        fcvtzs_w(&mut c, 0, 0);       // fcvtzs w0, d0
        fcvtns_w(&mut c, 0, 0);       // fcvtns w0, d0
        scvtf_w(&mut c, 0, 0);        // scvtf d0, w0
        for (i, chunk) in c.chunks(4).enumerate() {
            let w = u32::from_le_bytes(chunk.try_into().unwrap());
            eprintln!("  {:2}: {:02x?}  // {:08x}", i*4, chunk, w);
        }
        std::fs::write("dump_fpops.hex", &c).ok();
    }

    #[test]
    fn dump_bit_ops() {
        let mut c = Vec::new();
        movz(&mut c, 0, 2, 0);            // movz x0, #2
        movk(&mut c, 0, 1, 16);           // movk x0, #1, lsl #16
        mov_imm64(&mut c, 0, 42);          // mov x0, #42
        mov_reg(&mut c, 0, 1);             // mov x0, x1
        add_reg_lsl(&mut c, 0, 0, 1, 3);  // add x0, x0, x1, lsl #3
        cmp_reg(&mut c, 0, 1);             // cmp x0, x1
        cmp_imm(&mut c, 0, 42, 0);        // cmp x0, #42
        and_reg(&mut c, 0, 0, 1);         // and x0, x0, x1
        eor_reg(&mut c, 0, 0, 1);         // eor x0, x0, x1
        orr_reg(&mut c, 0, 0, 1);         // orr x0, x0, x1
        lsl_reg(&mut c, 0, 0, 1);         // lsl x0, x0, x1
        lsr_reg(&mut c, 0, 0, 1);         // lsr x0, x0, x1
        asr_imm(&mut c, 0, 0, 47);        // asr x0, x0, #47
        lsr_imm(&mut c, 0, 0, 32);        // lsr x0, x0, #32
        rev32(&mut c, 0, 0);              // rev32 x0, x0
        neg_w(&mut c, 1, 1);              // neg w1, w1
        and_imm32(&mut c, 1, 1, 31);      // and w1, w1, #31
        // ror w0, w0, w1 (32-bit inline)
        emit32(&mut c, 0 | 0x1AC02C00 | ((1u32) << 16) | 0u32);
        // asr w0, w0, w1 (32-bit inline)
        emit32(&mut c, 0 | 0x1AC02800 | ((1u32) << 16) | 0u32);
        for (i, chunk) in c.chunks(4).enumerate() {
            let w = u32::from_le_bytes(chunk.try_into().unwrap());
            eprintln!("  {:2}: {:02x?}  // {:08x}", i*4, chunk, w);
        }
        std::fs::write("dump_bitops.hex", &c).ok();
    }

    #[test]
    fn dump_loads_stores_branches() {
        let mut c = Vec::new();
        ldr_imm(&mut c, 0, 19, 0, 64);          // ldr x0, [x19]
        str_imm(&mut c, 0, 19, 0, 64);          // str x0, [x19]
        ldr_imm(&mut c, 0, 19, 4, 32);          // ldr w0, [x19, #4]
        str_imm(&mut c, 0, 19, 4, 32);          // str w0, [x19, #4]
        ldr_fp(&mut c, 0, 20, 8);               // ldr d0, [x20, #8]
        str_fp(&mut c, 0, 20, 8);               // str d0, [x20, #8]
        ldr_reg_uxtw(&mut c, 0, 2, 1);          // ldr x0, [x2, w1, uxtw #3]
        str_reg_uxtw(&mut c, 0, 2, 1);          // str x0, [x2, w1, uxtw #3]
        ldr_reg_lsl3(&mut c, 0, 2, 1);          // ldr x0, [x2, x1, lsl #3]
        str_reg_lsl3(&mut c, 0, 2, 1);          // str x0, [x2, x1, lsl #3]
        ldr_literal(&mut c, 0, 16, 64);          // ldr x0, [pc, #16]
        stp_offset(&mut c, 19, 20, 31, 16);      // stp x19, x20, [sp, #16]
        ldp_offset(&mut c, 19, 20, 31, 16);      // ldp x19, x20, [sp, #16]
        b_imm(&mut c, 16, false);                // b #16
        b_imm(&mut c, 16, true);                 // bl #16
        b_cond(&mut c, 0, 8);                    // b.eq #8
        b_cond(&mut c, CC_VS, 8);                // b.vs #8
        br_reg(&mut c, 0, false);                // br x0
        br_reg(&mut c, 2, true);                 // blr x2
        ret(&mut c, 30);                         // ret x30
        for (i, chunk) in c.chunks(4).enumerate() {
            let w = u32::from_le_bytes(chunk.try_into().unwrap());
            eprintln!("  {:2}: {:02x?}  // {:08x}", i*4, chunk, w);
        }
        std::fs::write("dump_load_branch.hex", &c).ok();
    }
}

// ---------------------------------------------------------------------------
// External API
// ---------------------------------------------------------------------------

/// Translate a trace to ARM64 machine code. Returns the executable area,
/// the inner-entry offset and patchable tail positions.
pub fn assemble(
    tr: &GCtrace,
    link: Option<*const u8>,
) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
    Asm::new(tr, link)?.emit()
}

/// Retarget an exit stub to jump directly to a compiled side trace.
pub fn patch_exit(area: &mut McodeArea, tails: &[(u32, u32)], exitno: u32, target: *const u8) {
    if !area.protect_rw() {
        return;
    }
    let code = area.as_mut_slice();
    for &(si, ofs) in tails {
        if si == exitno {
            let p = ofs as usize;
            // Overwrite the exit-code movz+k+b (20 bytes) with:
            //   movz x9, #lo16; movk x9, #hi16, lsl #16;
            //   movk x9, #hi32, lsl #32; movk x9, #hi48, lsl #48; br x9
            let taddr = target as u64;
            let mut v = taddr;
            for (i, shift) in [0u32, 16, 32, 48].iter().enumerate() {
                let chunk = (v & 0xFFFF) as u16;
                let insn = if i == 0 {
                    0xD2800000u32 | ((*shift / 16) << 21) | ((chunk as u32) << 5) | 9u32
                } else {
                    0xF2800000u32 | ((*shift / 16) << 21) | ((chunk as u32) << 5) | 9u32
                };
                code[p + i * 4..p + i * 4 + 4].copy_from_slice(&insn.to_le_bytes());
                v >>= 16;
            }
            code[p + 16..p + 20].copy_from_slice(&0xD61F0120u32.to_le_bytes());
        }
    }
    area.protect_exec();
}
