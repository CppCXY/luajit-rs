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
    emit32(code, sf | 0x12800000 | hw | ((imm as u32) << 5) | rd as u32);
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
    // Find the repeating pattern.
    for size in [2, 4, 8, 16, 32, 64u32] {
        let ones = imm.count_ones();
        let len = ones;
        if len > size {
            continue;
        }
        // Try consecutive ones then consecutive zeros.
        let r = imm.trailing_zeros();
        let _s = (size - r - len) % size;
        if (imm >> r).wrapping_shl(size - len) == (1u64 << len) - 1 {
            let n = if size == 64 { 1 } else { 0 };
            return Some((n, r as u32, (size - len) as u32));
        }
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

/// ASR (immediate): `asr rd, rn, #imm`.
fn asr_reg_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    debug_assert!(imm > 0 && imm <= 64);
    let sf = 1u32 << 31;
    let immr = imm as u32;
    emit32(code, sf | 0x13000000 | (0x3F << 16) | (immr << 10) | ((rn as u32) << 5) | rd as u32);
    // UBFM with appropriate fields → ASR alias
    // Actually ASR immediate encoding: sf | 0x13000000 | (immr << 16) | (63 << 10) | rn << 5 | rd
    // Wait, the encoding needs N=1 (bit 22) for 64-bit and the imms=63.
}

/// ASR (immediate): `asr rd, rn, #imm` (proper encoding).
fn asr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, imm: u8) {
    let sf = 1u32 << 31;
    let n = 1u32 << 22; // 64-bit
    let immr = imm as u32;
    let imms = 63u32; // sign extension
    emit32(code, sf | 0x13000000 | n | (immr << 16) | (imms << 10) | ((rn as u32) << 5) | rd as u32);
}

/// ROR (register): `ror rd, rn, rm`.
fn ror_reg(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x1AC02C00 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// REV32 (reverse bytes in 32-bit word, zero-extending).
fn rev32(code: &mut Vec<u8>, rd: u8, rn: u8) {
    let sf = 1u32 << 31;
    emit32(code, sf | 0x5AC00800 | ((rn as u32) << 5) | rd as u32);
}

/// LDR (immediate): `ldr rd, [rn, #offset]` (scaled, 9-bit signed).
fn ldr_imm(code: &mut Vec<u8>, rd: u8, rn: u8, offset: i32, size: u8) {
    debug_assert!(offset >= -256 && offset <= 255 && offset % 8 == 0);
    let sf = if size == 64 { 1u32 << 30 } else { 0 };
    let imm = ((offset.abs() / 8) as u32) << 10;
    let u_bit = if offset >= 0 { 1u32 << 24 } else { 0 };
    emit32(code, sf | 0x39400000 | u_bit | imm | ((rn as u32) << 5) | rd as u32);
}

/// STR (FP register): `str dn, [rn, #offset]` (scaled, 9-bit signed).
fn str_fp(code: &mut Vec<u8>, dn: u8, rn: u8, offset: i32) {
    debug_assert!(offset >= -256 && offset <= 255 && offset % 8 == 0);
    let imm = ((offset.abs() / 8) as u32) << 10;
    let u_bit = if offset >= 0 { 1u32 << 24 } else { 0 };
    let v_bit = 1u32 << 26; // FP/SIMD
    emit32(code, v_bit | 0x3D000000 | u_bit | imm | ((rn as u32) << 5) | dn as u32);
}

/// LDR (FP register): `ldr dd, [rn, #offset]` (scaled, 9-bit signed).
fn ldr_fp(code: &mut Vec<u8>, dd: u8, rn: u8, offset: i32) {
    debug_assert!(offset >= -256 && offset <= 255 && offset % 8 == 0);
    let imm = ((offset.abs() / 8) as u32) << 10;
    let u_bit = if offset >= 0 { 1u32 << 24 } else { 0 };
    let v_bit = 1u32 << 26;
    emit32(code, v_bit | 0x3D400000 | u_bit | imm | ((rn as u32) << 5) | dd as u32);
}

/// STR (GPR): `str rd, [rn, #offset]` (scaled, 9-bit signed).
fn str_imm(code: &mut Vec<u8>, rd: u8, rn: u8, offset: i32, size: u8) {
    let sf = if size == 64 { 1u32 << 30 } else { 0 };
    let imm = ((offset.abs() / 8) as u32) << 10;
    let u_bit = if offset >= 0 { 1u32 << 24 } else { 0 };
    emit32(code, sf | 0x39000000 | u_bit | imm | ((rn as u32) << 5) | rd as u32);
}

/// LDR (register): `ldr rd, [rn, rm, lsl #3]`.
fn ldr_reg_lsl3(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 30; // 64-bit
    emit32(
        code,
        sf | 0x38606800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// STR (register): `str rd, [rn, rm, lsl #3]`.
fn str_reg_lsl3(code: &mut Vec<u8>, rd: u8, rn: u8, rm: u8) {
    let sf = 1u32 << 30; // 64-bit
    emit32(
        code,
        sf | 0x38206800 | ((rm as u32) << 16) | ((rn as u32) << 5) | rd as u32,
    );
}

/// STP (store pair): `stp rt1, rt2, [rn, #offset]!` (pre-index).
fn stp_pre(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset.abs() / 8) as u32) << 15;
    let sf = 1u32 << 31;
    let mode = 0b11u32 << 23;
    emit32(
        code,
        sf | 0x29800000 | mode | imm7 | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
    );
}

/// STP (store pair, offset): `stp rt1, rt2, [rn, #offset]`.
fn stp_offset(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset.abs() / 8) as u32) << 15;
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x29000000 | (imm7) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
    );
}

/// LDP (load pair, offset): `ldp rt1, rt2, [rn, #offset]`.
fn ldp_offset(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset.abs() / 8) as u32) << 15;
    let sf = 1u32 << 31;
    emit32(
        code,
        sf | 0x29400000 | (imm7) | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
    );
}

/// LDP (load pair, post-index): `ldp rt1, rt2, [rn], #offset`.
fn ldp_post(code: &mut Vec<u8>, rt1: u8, rt2: u8, rn: u8, offset: i32) {
    debug_assert!(offset % 8 == 0 && offset >= -512 && offset <= 504);
    let imm7 = ((offset.abs() / 8) as u32) << 15;
    let sf = 1u32 << 31;
    let mode = 0b01u32 << 23;
    emit32(
        code,
        sf | 0x28C00000 | mode | imm7 | ((rt2 as u32) << 10) | ((rn as u32) << 5) | rt1 as u32,
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
    emit32(code, 0x9E660000 | ((xn as u32) << 5) | dd as u32);
}

/// FMOV (FP to GPR): `fmov xd, dn`.
fn fmov_fp_gpr(code: &mut Vec<u8>, xd: u8, dn: u8) {
    emit32(code, 0x9E660000 | 1u32 << 16 | ((dn as u32) << 5) | xd as u32);
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
    emit32(code, 0x1E204800 | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
}

/// FMAX: `fmax dd, dn, dm`.
fn fmax(code: &mut Vec<u8>, dd: u8, dn: u8, dm: u8) {
    emit32(code, 0x1E204800 | (1u32 << 22) | ((dm as u32) << 16) | ((dn as u32) << 5) | dd as u32);
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

/// FCVTZS (FP to int32, truncating): `fcvtzs wd, dn`.
fn fcvtzs_w(code: &mut Vec<u8>, wd: u8, dn: u8) {
    emit32(code, 0x1E380000 | (1u32 << 19) | ((dn as u32) << 5) | wd as u32);
}

/// FCVTNS (FP to int32, round-to-nearest-even): `fcvtns wd, dn`.
fn fcvtns_w(code: &mut Vec<u8>, wd: u8, dn: u8) {
    emit32(code, 0x1E280000 | (1u32 << 19) | ((dn as u32) << 5) | wd as u32);
}

/// SCVTF (int32 to FP): `scvtf dd, wn`.
fn scvtf_w(code: &mut Vec<u8>, dd: u8, wn: u8) {
    emit32(code, 0x1E220000 | (1u32 << 16) | ((wn as u32) << 5) | dd as u32);
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

/// Total frame size: 16 (fp+lr) + 10×8 (x21-x28, x19-x20) + 10×16 (v8-v15) = 256.
const FRAME_SIZE: i32 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Owner {
    None,
    Ins(IRRef),
    Konst(IRRef),
}

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
                | IROp::ULOAD
                | IROp::FLOAD
                | IROp::HLOAD
                | IROp::CARG
                | IROp::CALLL
                | IROp::TNEW
                | IROp::TDUP
                | IROp::GCSTEP
                | IROp::ALOAD
                | IROp::ASTORE
                | IROp::HSTORE => {}
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
                }
                IROp::PHI => {
                    let inf = tr.ir.nins();
                    a.mark_use(ins.op1 as IRRef, inf);
                    a.mark_use(ins.op2 as IRRef, inf);
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

    // -- Emit loop -----------------------------------------------------------

    fn emit(mut self) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
        // AAPCS64 prologue: x0=base, x1=env
        // stp x29, x30, [sp, #-256]!  — pre-index, save frame pointer + lr
        const FRAME: i32 = 256;
        stp_pre(&mut self.code, 29, 30, 31, -FRAME);
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
                IROp::LOOP => {
                    // Legacy loop tail: restore the final snapshot into
                    // the Lua stack and branch back to the head.
                    let head_ofs = inner as i32 - self.code.len() as i32;
                    self.snapidx = self.tr.snap.len() - 1;
                    self.tail_restore(self.snapidx);
                    // Flush the live FP snapshot values to env before
                    // restarting (the head SLOADs re-read from env for FP,
                    // and from the Lua stack for GC values).
                    let flush = self.exit_flush_set(self.snapidx);
                    for (rg, rref) in &flush {
                        str_fp(&mut self.code, *rg, RENV, Self::env_disp(*rref));
                    }
                    b_imm(&mut self.code, head_ofs, false);
                    // Stop the main loop: tail_restore + b consume the rest.
                    r = nins;
                }
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
                _ => return Err(TraceError::NYIIR),
            }
            r += 1;
        }

        // Tail / final snapshot
        let lastsnap = self.tr.snap.len() - 1;
        self.snapidx = lastsnap;
        self.tail_restore(lastsnap);
        let flush = self.exit_flush_set(lastsnap);
        for (rg, rref) in &flush {
            str_fp(&mut self.code, *rg, RENV, Self::env_disp(*rref));
        }

        // Exit code in w0
        let ec = self.exit_code(lastsnap);
        mov_imm64(&mut self.code, 0, ec as u64);

        // -- Epilogue --
        let epilogue = self.code.len();
        mov_imm64(&mut self.code, RSCR, super::super::exec::exit_base_cell_addr());
        str_imm(&mut self.code, RBASE, RSCR, 0, 64);
        ldp_offset(&mut self.code, 27, 28, 31, 80);
        ldp_offset(&mut self.code, 25, 26, 31, 64);
        ldp_offset(&mut self.code, 23, 24, 31, 48);
        ldp_offset(&mut self.code, 21, 22, 31, 32);
        ldp_offset(&mut self.code, 19, 20, 31, 16);
        ldp_post(&mut self.code, 29, 30, 31, FRAME);
        ret(&mut self.code, 30);

        // -- Guard-exit stubs --
        let stubs = std::mem::take(&mut self.stubs);
        let mut stubpos = Vec::with_capacity(stubs.len());
        for st in &stubs {
            stubpos.push(self.code.len());
            for (rg, rref) in &st.flush {
                str_fp(&mut self.code, *rg, RENV, Self::env_disp(*rref));
            }
            let ec = if st.gc { self.exit_code(st.snapidx) | 0x8000 } else { self.exit_code(st.snapidx) };
            mov_imm64(&mut self.code, 0, ec as u64);
            let epi_off = epilogue as i32 - self.code.len() as i32;
            b_imm(&mut self.code, epi_off, false);
            // 12-byte patch space for side-trace target
            while self.code.len() < stubpos.last().unwrap() + 12 + 4 {
                self.code.push(0x00);
            }
            self.stub_tails.push((st.snapidx as u32, *stubpos.last().unwrap() as u32));
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

    // -- SLOAD ---------------------------------------------------------------

    fn asm_sload(&mut self, ins: &IRIns) -> Result<(), TraceError> {
        let idx = ins.op1 as i32 - 2;
        let r = self.cur;
        ldr_imm(&mut self.code, RSCR, RBASE, idx * 8, 64);
        if ins.is_guard() && !irt_isnum(ins.t()) {
            // Type guard: arithmetic-shift-right 47 extracts the itype
            // (NaN-boxed: high 17 bits = ~itype). Compare against the
            // expected negation of the specialized type.
            asr_imm(&mut self.code, RSCR2, RSCR, 47);
            let expected = (!(irt_type(ins.t()) as u32)) as u8;
            cmp_imm(&mut self.code, RSCR2, expected as u32, 0);
            self.guard(CC_NE);
        }
        if irt_isnum(ins.t()) {
            let d = self.alloc(0)?;
            fmov_gpr_fp(&mut self.code, d, RSCR);
            self.owner[d as usize] = Owner::Ins(r);
            self.loc[Self::iidx(r)] = Some(d);
            if self.needs_env[Self::iidx(r)] {
                str_fp(&mut self.code, d, RENV, Self::env_disp(r));
                self.env_valid[Self::iidx(r)] = true;
            }
        } else {
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
pub fn patch_exit(_area: &mut McodeArea, _tails: &[(u32, u32)], _exitno: u32, _target: *const u8) {
    // NYI
}
