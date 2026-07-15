#![allow(non_upper_case_globals)]

pub type BCIns = u32;
pub type BCPos = u32;
pub type BCReg = u32;
pub type BCLine = u32;

pub const BCMAX_A: u32 = 0xff;
pub const BCMAX_B: u32 = 0xff;
pub const BCMAX_C: u32 = 0xff;
pub const BCMAX_D: u32 = 0xffff;
pub const BCBIAS_J: u32 = 0x8000;
pub const NO_REG: u32 = BCMAX_A;
pub const NO_JMP: BCPos = !0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BCMode {
    None = 0,
    Dst,
    Base,
    Var,
    Rbase,
    Uv,
    Lit,
    Lits,
    Pri,
    Num,
    Str,
    Tab,
    Func,
    Jump,
    Cdata,
}

macro_rules! bcdef {
    ($handler:ident) => {
        $handler! {
            (ISLT, Var, None, Var),
            (ISGE, Var, None, Var),
            (ISLE, Var, None, Var),
            (ISGT, Var, None, Var),
            (ISEQV, Var, None, Var),
            (ISNEV, Var, None, Var),
            (ISEQS, Var, None, Str),
            (ISNES, Var, None, Str),
            (ISEQN, Var, None, Num),
            (ISNEN, Var, None, Num),
            (ISEQP, Var, None, Pri),
            (ISNEP, Var, None, Pri),
            (ISTC, Dst, None, Var),
            (ISFC, Dst, None, Var),
            (IST, None, None, Var),
            (ISF, None, None, Var),
            (ISTYPE, Var, None, Lit),
            (ISNUM, Var, None, Lit),
            (MOV, Dst, None, Var),
            (NOT, Dst, None, Var),
            (UNM, Dst, None, Var),
            (LEN, Dst, None, Var),
            (ADDVN, Dst, Var, Num),
            (SUBVN, Dst, Var, Num),
            (MULVN, Dst, Var, Num),
            (DIVVN, Dst, Var, Num),
            (MODVN, Dst, Var, Num),
            (ADDNV, Dst, Var, Num),
            (SUBNV, Dst, Var, Num),
            (MULNV, Dst, Var, Num),
            (DIVNV, Dst, Var, Num),
            (MODNV, Dst, Var, Num),
            (ADDVV, Dst, Var, Var),
            (SUBVV, Dst, Var, Var),
            (MULVV, Dst, Var, Var),
            (DIVVV, Dst, Var, Var),
            (MODVV, Dst, Var, Var),
            (POW, Dst, Var, Var),
            (CAT, Dst, Rbase, Rbase),
            (KSTR, Dst, None, Str),
            (KCDATA, Dst, None, Cdata),
            (KSHORT, Dst, None, Lits),
            (KNUM, Dst, None, Num),
            (KPRI, Dst, None, Pri),
            (KNIL, Base, None, Base),
            (UGET, Dst, None, Uv),
            (USETV, Uv, None, Var),
            (USETS, Uv, None, Str),
            (USETN, Uv, None, Num),
            (USETP, Uv, None, Pri),
            (UCLO, Rbase, None, Jump),
            (FNEW, Dst, None, Func),
            (TNEW, Dst, None, Lit),
            (TDUP, Dst, None, Tab),
            (GGET, Dst, None, Str),
            (GSET, Var, None, Str),
            (TGETV, Dst, Var, Var),
            (TGETS, Dst, Var, Str),
            (TGETB, Dst, Var, Lit),
            (TGETR, Dst, Var, Var),
            (TSETV, Var, Var, Var),
            (TSETS, Var, Var, Str),
            (TSETB, Var, Var, Lit),
            (TSETM, Base, None, Num),
            (TSETR, Var, Var, Var),
            (CALLM, Base, Lit, Lit),
            (CALL, Base, Lit, Lit),
            (CALLMT, Base, None, Lit),
            (CALLT, Base, None, Lit),
            (ITERC, Base, Lit, Lit),
            (ITERN, Base, Lit, Lit),
            (VARG, Base, Lit, Lit),
            (ISNEXT, Base, None, Jump),
            (RETM, Base, None, Lit),
            (RET, Rbase, None, Lit),
            (RET0, Rbase, None, Lit),
            (RET1, Rbase, None, Lit),
            (FORI, Base, None, Jump),
            (JFORI, Base, None, Jump),
            (FORL, Base, None, Jump),
            (IFORL, Base, None, Jump),
            (JFORL, Base, None, Lit),
            (ITERL, Base, None, Jump),
            (IITERL, Base, None, Jump),
            (JITERL, Base, None, Lit),
            (LOOP, Rbase, None, Jump),
            (ILOOP, Rbase, None, Jump),
            (JLOOP, Rbase, None, Lit),
            (JMP, Rbase, None, Jump),
            (BNOT, Dst, None, Var),
            (BAND, Dst, Var, Var),
            (BOR, Dst, Var, Var),
            (BXOR, Dst, Var, Var),
            (BSHL, Dst, Var, Var),
            (BSHR, Dst, Var, Var),
            (BSAR, Dst, Var, Var),
            (FUNCF, Rbase, None, None),
            (IFUNCF, Rbase, None, None),
            (JFUNCF, Rbase, None, Lit),
            (FUNCV, Rbase, None, None),
            (IFUNCV, Rbase, None, None),
            (JFUNCV, Rbase, None, Lit),
            (FUNCC, Rbase, None, None),
            (FUNCCW, Rbase, None, None),
        }
    };
}

macro_rules! bc_enum {
    ($(($name:ident, $ma:ident, $mb:ident, $mc:ident),)*) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        #[repr(u8)]
        pub enum BCOp {
            $($name,)*
        }

        pub const BC_NAMES: &[&str] = &[$(stringify!($name),)*];

        pub const BC_MODE: &[u16] = &[
            $((BCMode::$ma as u16) | ((BCMode::$mb as u16) << 3) | ((BCMode::$mc as u16) << 7),)*
        ];
    };
}

bcdef!(bc_enum);

pub const BC_MAX: u32 = BC_NAMES.len() as u32;

impl BCOp {
    #[inline]
    pub fn from_u32(op: u32) -> BCOp {
        debug_assert!(op < BC_MAX);
        unsafe { std::mem::transmute(op as u8) }
    }

    #[inline]
    pub fn offset(self, delta: i32) -> BCOp {
        BCOp::from_u32((self as i32 + delta) as u32)
    }
}

#[inline]
pub fn bc_op(i: BCIns) -> BCOp {
    BCOp::from_u32(i & 0xff)
}
#[inline]
pub fn bc_a(i: BCIns) -> BCReg {
    (i >> 8) & 0xff
}
#[inline]
pub fn bc_b(i: BCIns) -> BCReg {
    i >> 24
}
#[inline]
pub fn bc_c(i: BCIns) -> BCReg {
    (i >> 16) & 0xff
}
#[inline]
pub fn bc_d(i: BCIns) -> u32 {
    i >> 16
}
#[inline]
pub fn bc_j(i: BCIns) -> i64 {
    bc_d(i) as i64 - BCBIAS_J as i64
}

#[inline]
pub fn setbc_op(p: &mut BCIns, x: u32) {
    *p = (*p & !0xff) | (x & 0xff);
}
#[inline]
pub fn setbc_a(p: &mut BCIns, x: u32) {
    *p = (*p & !0xff00) | ((x & 0xff) << 8);
}
#[inline]
pub fn setbc_b(p: &mut BCIns, x: u32) {
    *p = (*p & 0x00ffffff) | ((x & 0xff) << 24);
}
#[inline]
pub fn setbc_c(p: &mut BCIns, x: u32) {
    *p = (*p & !0x00ff0000) | ((x & 0xff) << 16);
}
#[inline]
pub fn setbc_d(p: &mut BCIns, x: u32) {
    *p = (*p & 0xffff) | ((x & 0xffff) << 16);
}
#[inline]
pub fn setbc_j(p: &mut BCIns, x: i64) {
    setbc_d(p, (x + BCBIAS_J as i64) as u32);
}

#[inline]
pub fn bcins_abc(o: BCOp, a: BCReg, b: BCReg, c: BCReg) -> BCIns {
    (o as u32) | (a << 8) | (b << 24) | (c << 16)
}
#[inline]
pub fn bcins_ad(o: BCOp, a: BCReg, d: u32) -> BCIns {
    (o as u32) | (a << 8) | (d << 16)
}
#[inline]
pub fn bcins_aj(o: BCOp, a: BCReg, j: i64) -> BCIns {
    bcins_ad(o, a, (j + BCBIAS_J as i64) as u32)
}

#[inline]
pub fn bcmode_a(op: BCOp) -> u32 {
    (BC_MODE[op as usize] & 7) as u32
}
#[inline]
pub fn bcmode_b(op: BCOp) -> u32 {
    ((BC_MODE[op as usize] >> 3) & 15) as u32
}
#[inline]
pub fn bcmode_c(op: BCOp) -> u32 {
    ((BC_MODE[op as usize] >> 7) & 15) as u32
}

#[inline]
pub fn bc_isret(op: BCOp) -> bool {
    matches!(op, BCOp::RETM | BCOp::RET | BCOp::RET0 | BCOp::RET1)
}

#[inline]
pub fn bc_isret_or_tail(op: BCOp) -> bool {
    matches!(op, BCOp::CALLMT | BCOp::CALLT) || bc_isret(op)
}

pub const FORL_IDX: u32 = 0;
pub const FORL_STOP: u32 = 1;
pub const FORL_STEP: u32 = 2;
pub const FORL_EXT: u32 = 3;
