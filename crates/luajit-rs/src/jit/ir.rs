//! SSA IR (Intermediate Representation) format and buffer.
//!
//! Ported from lj_ir.h and the constant-interning half of lj_ir.c.
//!
//! Layout notes:
//! * `IRIns` is the same 8-byte record as LuaJIT's: op1/op2 (16 bit each),
//!   ot (opcode<<8 | type), prev (CSE chain, reused by the register
//!   allocator later).
//! * References: constants grow *down* from `REF_BIAS-1`, instructions
//!   grow *up* from `REF_BIAS`. Literal operands are < REF_BIAS, so
//!   `max(op1,op2)` bounds a CSE search and DCE can test `>= REF_BIAS`.
//! * Deviation from LuaJIT: 64-bit constants (KNUM/KINT64/KGC) occupy a
//!   *single* ref here, with the payload held in a parallel array — LuaJIT
//!   makes them take two IR slots, which is purely a C memory-layout
//!   trick (`ir[1].tv`).

use super::TraceError;

// -- IR opcodes (lj_ir.h IRDEF) --------------------------------------------

/// Operand mode of one operand (2 bits).
pub const IRM_REF: u8 = 0;
pub const IRM_LIT: u8 = 1;
pub const IRM_CST: u8 = 2;
pub const IRM_NONE: u8 = 3;

/// Mode bits: Commutative, {Normal/Ref, Alloc, Load, Store}, Non-weak guard.
pub const IRM_C: u8 = 0x10;
pub const IRM_N: u8 = 0x00;
pub const IRM_A: u8 = 0x20;
pub const IRM_L: u8 = 0x40;
pub const IRM_S: u8 = 0x60;
pub const IRM_W: u8 = 0x80;

macro_rules! irm_flag {
    (N) => {
        IRM_N
    };
    (C) => {
        IRM_C
    };
    (R) => {
        IRM_N
    };
    (A) => {
        IRM_A
    };
    (L) => {
        IRM_L
    };
    (S) => {
        IRM_S
    };
    (NW) => {
        IRM_N | IRM_W
    };
    (CW) => {
        IRM_C | IRM_W
    };
    (AW) => {
        IRM_A | IRM_W
    };
    (LW) => {
        IRM_L | IRM_W
    };
}
macro_rules! irm_mode {
    (ref) => {
        IRM_REF
    };
    (lit) => {
        IRM_LIT
    };
    (cst) => {
        IRM_CST
    };
    (___) => {
        IRM_NONE
    };
}

/// IR instruction definition. Order matters (ORDER IR): comparison
/// opposites flip with ^1, ordered/unordered with ^4, loads and stores
/// keep a fixed delta.
macro_rules! irdef {
    ($handler:ident) => {
        $handler! {
            // Guarded assertions.
            (LT, N, ref, ref),
            (GE, N, ref, ref),
            (LE, N, ref, ref),
            (GT, N, ref, ref),
            (ULT, N, ref, ref),
            (UGE, N, ref, ref),
            (ULE, N, ref, ref),
            (UGT, N, ref, ref),
            (EQ, C, ref, ref),
            (NE, C, ref, ref),
            (ABC, N, ref, ref),
            (RETF, S, ref, ref),
            // Miscellaneous ops.
            (NOP, N, ___, ___),
            (BASE, N, lit, lit),
            (PVAL, N, lit, ___),
            (GCSTEP, S, ___, ___),
            (HIOP, S, ref, ref),
            (LOOP, S, ___, ___),
            (USE, S, ref, ___),
            (PHI, S, ref, ref),
            (RENAME, S, ref, lit),
            (PROF, S, ___, ___),
            // Constants.
            (KPRI, N, ___, ___),
            (KINT, N, cst, ___),
            (KGC, N, cst, ___),
            (KPTR, N, cst, ___),
            (KKPTR, N, cst, ___),
            (KNULL, N, cst, ___),
            (KNUM, N, cst, ___),
            (KINT64, N, cst, ___),
            (KSLOT, N, ref, lit),
            // Bit ops. ORDER BIT.
            (BNOT, N, ref, ___),
            (BSWAP, N, ref, ___),
            (BAND, C, ref, ref),
            (BOR, C, ref, ref),
            (BXOR, C, ref, ref),
            (BSHL, N, ref, ref),
            (BSHR, N, ref, ref),
            (BSAR, N, ref, ref),
            (BROL, N, ref, ref),
            (BROR, N, ref, ref),
            // Arithmetic ops. ORDER ARITH.
            (ADD, C, ref, ref),
            (SUB, N, ref, ref),
            (MUL, C, ref, ref),
            (DIV, N, ref, ref),
            (MOD, N, ref, ref),
            (POW, N, ref, ref),
            (NEG, N, ref, ref),
            (ABS, N, ref, ref),
            (LDEXP, N, ref, ref),
            (MIN, N, ref, ref),
            (MAX, N, ref, ref),
            (FPMATH, N, ref, lit),
            // Overflow-checking arithmetic ops.
            (ADDOV, CW, ref, ref),
            (SUBOV, NW, ref, ref),
            (MULOV, CW, ref, ref),
            // Memory references.
            (AREF, R, ref, ref),
            (HREFK, R, ref, ref),
            (HREF, L, ref, ref),
            (NEWREF, S, ref, ref),
            (UREFO, LW, ref, lit),
            (UREFC, LW, ref, lit),
            (FREF, R, ref, lit),
            (TMPREF, S, ref, lit),
            (STRREF, N, ref, ref),
            (LREF, L, ___, ___),
            // Loads and stores. Same order (IRDELTA_L2S).
            (ALOAD, L, ref, ___),
            (HLOAD, L, ref, ___),
            (ULOAD, L, ref, ___),
            (FLOAD, L, ref, lit),
            (XLOAD, L, ref, lit),
            (SLOAD, L, lit, lit),
            (VLOAD, L, ref, lit),
            (ALEN, L, ref, ref),
            (ASTORE, S, ref, ref),
            (HSTORE, S, ref, ref),
            (USTORE, S, ref, ref),
            (FSTORE, S, ref, ref),
            (XSTORE, S, ref, ref),
            // Allocations.
            (SNEW, N, ref, ref),
            (XSNEW, A, ref, ref),
            (TNEW, AW, lit, lit),
            (TDUP, AW, ref, ___),
            (CNEW, AW, ref, ref),
            (CNEWI, NW, ref, ref),
            // Buffer operations.
            (BUFHDR, L, ref, lit),
            (BUFPUT, LW, ref, ref),
            (BUFSTR, AW, ref, ref),
            // Barriers.
            (TBAR, S, ref, ___),
            (OBAR, S, ref, ref),
            (XBAR, S, ___, ___),
            // Type conversions.
            (CONV, N, ref, lit),
            (TOBIT, N, ref, ref),
            (TOSTR, N, ref, lit),
            (STRTO, N, ref, ___),
            // Calls.
            (CALLN, NW, ref, lit),
            (CALLA, AW, ref, lit),
            (CALLL, LW, ref, lit),
            (CALLS, S, ref, lit),
            (CALLXS, S, ref, ref),
            (CARG, N, ref, ref),
        }
    };
}

macro_rules! ir_tables {
    ($(($name:ident, $m:tt, $m1:tt, $m2:tt),)*) => {
        /// IR opcodes (max. 256).
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        #[repr(u8)]
        pub enum IROp {
            $($name,)*
        }

        pub const IR_NAMES: &[&str] = &[$(stringify!($name),)*];

        /// Operand modes and flags, exactly LuaJIT's `lj_ir_mode` encoding:
        /// `(m1 | m2<<2 | flags) ^ IRM_W` (the XOR makes W a *non*-weak bit).
        pub const IR_MODE: &[u8] = &[
            $((irm_mode!($m1) | (irm_mode!($m2) << 2) | irm_flag!($m)) ^ IRM_W,)*
        ];

        const _: () = {
            let mut i = 0u8;
            $(
                assert!(IROp::$name as u8 == i);
                i += 1;
            )*
            assert!(i as usize == IR_NAMES.len());
        };
    };
}

irdef!(ir_tables);

pub const IR_MAX: usize = IR_NAMES.len();

impl IROp {
    #[inline]
    pub fn from_u8(op: u8) -> IROp {
        debug_assert!((op as usize) < IR_MAX);
        unsafe { std::mem::transmute(op) }
    }
}

// Comparison ops flip with ^1 (EQ<->NE, LT<->GE), ^3 swaps operands
// (LT<->GT), ^4 toggles ordered/unordered.
const _: () = {
    assert!(IROp::EQ as u8 ^ 1 == IROp::NE as u8);
    assert!(IROp::LT as u8 ^ 1 == IROp::GE as u8);
    assert!(IROp::LE as u8 ^ 1 == IROp::GT as u8);
    assert!(IROp::LT as u8 ^ 3 == IROp::GT as u8);
    assert!(IROp::LT as u8 ^ 4 == IROp::ULT as u8);
    // xLOAD -> xSTORE delta is constant.
    assert!(IROp::ASTORE as u8 - IROp::ALOAD as u8 == IROp::HSTORE as u8 - IROp::HLOAD as u8);
};

#[inline]
pub fn irm_op1(m: u8) -> u8 {
    m & 3
}
#[inline]
pub fn irm_op2(m: u8) -> u8 {
    (m >> 2) & 3
}
#[inline]
pub fn irm_iscomm(m: u8) -> bool {
    m & IRM_C != 0
}
#[inline]
pub fn irm_kind(m: u8) -> u8 {
    m & IRM_S
}

// -- IR types (lj_ir.h IRTDEF) ----------------------------------------------

pub const IRT_NIL: u8 = 0;
pub const IRT_FALSE: u8 = 1;
pub const IRT_TRUE: u8 = 2;
pub const IRT_LIGHTUD: u8 = 3;
pub const IRT_STR: u8 = 4;
pub const IRT_P32: u8 = 5;
pub const IRT_THREAD: u8 = 6;
pub const IRT_PROTO: u8 = 7;
pub const IRT_FUNC: u8 = 8;
pub const IRT_P64: u8 = 9;
pub const IRT_CDATA: u8 = 10;
pub const IRT_TAB: u8 = 11;
pub const IRT_UDATA: u8 = 12;
pub const IRT_FLOAT: u8 = 13;
pub const IRT_NUM: u8 = 14;
pub const IRT_I8: u8 = 15;
pub const IRT_U8: u8 = 16;
pub const IRT_I16: u8 = 17;
pub const IRT_U16: u8 = 18;
pub const IRT_INT: u8 = 19;
pub const IRT_U32: u8 = 20;
pub const IRT_I64: u8 = 21;
pub const IRT_U64: u8 = 22;
pub const IRT_SOFTFP: u8 = 23;

/// We are a GC64-style VM: pointers to GC objects are 64 bit.
pub const IRT_PTR: u8 = IRT_P64;
pub const IRT_PGC: u8 = IRT_P64;

// Additional flags.
pub const IRT_MARK: u8 = 0x20;
pub const IRT_ISPHI: u8 = 0x40;
pub const IRT_GUARD: u8 = 0x80;
// Masks.
pub const IRT_TYPE: u8 = 0x1f;

const _: () = assert!(IRT_GUARD == IRM_W);

#[inline]
pub fn irt_type(t: u8) -> u8 {
    t & IRT_TYPE
}
#[inline]
pub fn irt_isnum(t: u8) -> bool {
    irt_type(t) == IRT_NUM
}
#[inline]
pub fn irt_isint(t: u8) -> bool {
    irt_type(t) == IRT_INT
}
#[inline]
pub fn irt_ispri(t: u8) -> bool {
    irt_type(t) <= IRT_TRUE
}
#[inline]
pub fn irt_isguard(t: u8) -> bool {
    t & IRT_GUARD != 0
}
#[inline]
pub fn irt_isphi(t: u8) -> bool {
    t & IRT_ISPHI != 0
}
#[inline]
pub fn irt_isgcv(t: u8) -> bool {
    (IRT_STR..=IRT_UDATA).contains(&irt_type(t))
}
#[inline]
pub fn irt_isinteger(t: u8) -> bool {
    (IRT_I8..=IRT_INT).contains(&irt_type(t))
}
/// `irt_sametype`: same base type, ignoring the flag bits.
#[inline]
pub fn irt_sametype(t1: u8, t2: u8) -> bool {
    (t1 ^ t2) & IRT_TYPE == 0
}

/// Combined opcode + type (`IRT(o, t)`).
#[inline]
pub fn irt(o: IROp, t: u8) -> u16 {
    ((o as u16) << 8) | t as u16
}
/// `IRTI(o)`: opcode with int type.
#[inline]
pub fn irti(o: IROp) -> u16 {
    irt(o, IRT_INT)
}
/// `IRTN(o)`: opcode with number type.
#[inline]
pub fn irtn(o: IROp) -> u16 {
    irt(o, IRT_NUM)
}
/// `IRTG(o, t)`: guarded opcode.
#[inline]
pub fn irtg(o: IROp, t: u8) -> u16 {
    irt(o, IRT_GUARD | t)
}
/// `IRTGI(o)`: guarded opcode with int type.
#[inline]
pub fn irtgi(o: IROp) -> u16 {
    irt(o, IRT_GUARD | IRT_INT)
}

// -- IR references -----------------------------------------------------------

pub type IRRef1 = u16;
pub type IRRef = u32;

pub const REF_BIAS: IRRef = 0x8000;
pub const REF_TRUE: IRRef = REF_BIAS - 3;
pub const REF_FALSE: IRRef = REF_BIAS - 2;
pub const REF_NIL: IRRef = REF_BIAS - 1; // Constants grow downwards...
pub const REF_BASE: IRRef = REF_BIAS; // ...IR grows upwards.
pub const REF_FIRST: IRRef = REF_BIAS + 1;
pub const REF_DROP: IRRef = 0xffff;

#[inline]
pub fn irref_isk(r: IRRef) -> bool {
    r < REF_BIAS
}

/// Tagged IR reference: `irt << 24 | flags | ref`.
pub type TRef = u32;

pub const TREF_REFMASK: u32 = 0x0000_ffff;
pub const TREF_FRAME: u32 = 0x0001_0000;
pub const TREF_CONT: u32 = 0x0002_0000;
pub const TREF_KEYINDEX: u32 = 0x0010_0000;

#[inline]
pub fn tref(r: IRRef, t: u8) -> TRef {
    r + ((t as u32) << 24)
}
#[inline]
pub fn tref_ref(tr: TRef) -> IRRef {
    tr & TREF_REFMASK
}
#[inline]
pub fn tref_t(tr: TRef) -> u8 {
    (tr >> 24) as u8
}
#[inline]
pub fn tref_type(tr: TRef) -> u8 {
    ((tr >> 24) as u8) & IRT_TYPE
}
#[inline]
pub fn tref_isk(tr: TRef) -> bool {
    irref_isk(tref_ref(tr))
}
#[inline]
pub fn tref_isnum(tr: TRef) -> bool {
    tref_type(tr) == IRT_NUM
}
#[inline]
pub fn tref_isnil(tr: TRef) -> bool {
    tref_type(tr) == IRT_NIL
}
#[inline]
pub fn tref_isint(tr: TRef) -> bool {
    tref_type(tr) == IRT_INT
}

#[inline]
pub fn tref_isnum_or_int(tr: TRef) -> bool {
    let t = tref_type(tr);
    t == IRT_NUM || t == IRT_INT
}

#[inline]
pub fn num_isint(n: f64) -> bool {
    let i = n as i32;
    i as f64 == n
}
#[inline]
pub fn tref_istab(tr: TRef) -> bool {
    tref_type(tr) == IRT_TAB
}
#[inline]
pub fn tref_isstr(tr: TRef) -> bool {
    tref_type(tr) == IRT_STR
}
/// `TREF_PRI(t)`: the fixed refs for nil/false/true.
#[inline]
pub fn tref_pri(t: u8) -> TRef {
    tref(REF_NIL - t as IRRef, t)
}
pub const TREF_NIL: TRef = (REF_NIL - IRT_NIL as IRRef) + ((IRT_NIL as u32) << 24);
pub const TREF_FALSE: TRef = (REF_NIL - IRT_FALSE as IRRef) + ((IRT_FALSE as u32) << 24);
pub const TREF_TRUE: TRef = (REF_NIL - IRT_TRUE as IRRef) + ((IRT_TRUE as u32) << 24);

// -- IR instruction format ---------------------------------------------------

/// One IR instruction (8 bytes), LuaJIT's IRIns minus the union tricks:
///
/// ```text
///    16      16     8   8    16
/// +-------+-------+---+---+------+
/// |  op1  |  op2  | t | o | prev |
/// +-------+-------+---+---+------+
/// ```
///
/// `prev` links the per-opcode CSE chain and is reused as r/s by the
/// register allocator later.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IRIns {
    pub op1: IRRef1,
    pub op2: IRRef1,
    pub ot: u16,
    pub prev: IRRef1,
}

impl IRIns {
    #[inline]
    pub fn new(ot: u16, op1: IRRef, op2: IRRef) -> IRIns {
        debug_assert!(op1 <= 0xffff && op2 <= 0xffff);
        IRIns {
            op1: op1 as IRRef1,
            op2: op2 as IRRef1,
            ot,
            prev: 0,
        }
    }

    #[inline]
    pub fn op(&self) -> IROp {
        IROp::from_u8((self.ot >> 8) as u8)
    }
    #[inline]
    pub fn set_op(&mut self, o: IROp) {
        self.ot = (self.ot & 0x00ff) | ((o as u16) << 8);
    }
    /// Full type byte (including GUARD/PHI/MARK flags).
    #[inline]
    pub fn t(&self) -> u8 {
        self.ot as u8
    }
    #[inline]
    pub fn set_t(&mut self, t: u8) {
        self.ot = (self.ot & 0xff00) | t as u16;
    }
    /// op1 and op2 as one 32 bit key (CSE identity).
    #[inline]
    pub fn op12(&self) -> u32 {
        self.op1 as u32 | ((self.op2 as u32) << 16)
    }
    /// KINT payload: a 32 bit signed integer stored across op1/op2.
    #[inline]
    pub fn i(&self) -> i32 {
        self.op12() as i32
    }
    #[inline]
    pub fn set_i(&mut self, i: i32) {
        self.op1 = i as u32 as IRRef1;
        self.op2 = (i as u32 >> 16) as IRRef1;
    }
    #[inline]
    pub fn is_guard(&self) -> bool {
        irt_isguard(self.t())
    }
    /// A store or any other op with a non-weak guard has a side-effect.
    #[inline]
    pub fn sideeff(&self) -> bool {
        ((self.t() | !IRT_GUARD) & IR_MODE[self.op() as usize]) >= IRM_S
    }
    // IRT_MARK / IRT_ISPHI flag maintenance (irt_setmark & friends).
    #[inline]
    pub fn is_marked(&self) -> bool {
        self.t() & IRT_MARK != 0
    }
    #[inline]
    pub fn set_mark(&mut self) {
        self.ot |= IRT_MARK as u16;
    }
    #[inline]
    pub fn clear_mark(&mut self) {
        self.ot &= !(IRT_MARK as u16);
    }
    #[inline]
    pub fn set_phi(&mut self) {
        self.ot |= IRT_ISPHI as u16;
    }
    #[inline]
    pub fn clear_phi(&mut self) {
        self.ot &= !(IRT_ISPHI as u16);
    }
    /// Replace with NOP (`lj_ir_nop`).
    #[inline]
    pub fn set_nop(&mut self) {
        self.ot = irt(IROp::NOP, IRT_NIL);
        self.op1 = 0;
        self.op2 = 0;
        self.prev = 0;
    }
}

const _: () = assert!(std::mem::size_of::<IRIns>() == 8);

// -- IR buffer of the trace being recorded -----------------------------------

/// The growing IR of one trace plus the CSE chains: the parts of LuaJIT's
/// `jit_State`/`GCtrace` that the FOLD/CSE engine and the recorder write
/// into (J->cur.ir, J->cur.nins/nk, J->chain).
pub struct IrBuf {
    /// Instructions; `ins[0]` is REF_BASE.
    ins: Vec<IRIns>,
    /// Constants; `k[0]` is REF_NIL (= REF_BIAS-1), growing down.
    k: Vec<IRIns>,
    /// 64 bit payloads of the constants in `k` (KNUM/KINT64/KGC bits).
    k64: Vec<u64>,
    /// Per-opcode CSE chains (J->chain), 0 = empty.
    pub chain: [IRRef1; IR_MAX],
    /// Accumulated guard types since the last snapshot (J->guardemit).
    pub guardemit: u8,
}

impl IrBuf {
    /// Seed the buffer like `lj_record_setup`: KPRI nil/false/true at the
    /// fixed refs and the BASE instruction at REF_BASE.
    pub fn new(parent: u16, exitno: u16) -> IrBuf {
        let mut buf = IrBuf {
            ins: Vec::with_capacity(64),
            k: Vec::with_capacity(16),
            k64: Vec::with_capacity(16),
            chain: [0; IR_MAX],
            guardemit: 0,
        };
        for t in [IRT_NIL, IRT_FALSE, IRT_TRUE] {
            buf.k.push(IRIns::new(irt(IROp::KPRI, t), 0, 0));
            buf.k64.push(0);
        }
        buf.ins.push(IRIns::new(
            irt(IROp::BASE, IRT_PGC),
            parent as IRRef,
            exitno as IRRef,
        ));
        buf
    }

    /// Next instruction ref (J->cur.nins).
    #[inline]
    pub fn nins(&self) -> IRRef {
        REF_BIAS + self.ins.len() as IRRef
    }
    /// Lowest constant ref (J->cur.nk).
    #[inline]
    pub fn nk(&self) -> IRRef {
        REF_BIAS - self.k.len() as IRRef
    }

    #[inline]
    pub fn ir(&self, r: IRRef) -> &IRIns {
        if r >= REF_BIAS {
            &self.ins[(r - REF_BIAS) as usize]
        } else {
            &self.k[(REF_BIAS - 1 - r) as usize]
        }
    }
    #[inline]
    pub fn ir_mut(&mut self, r: IRRef) -> &mut IRIns {
        if r >= REF_BIAS {
            &mut self.ins[(r - REF_BIAS) as usize]
        } else {
            &mut self.k[(REF_BIAS - 1 - r) as usize]
        }
    }

    /// 64 bit payload of a constant (KNUM/KINT64/KGC).
    #[inline]
    pub fn k64_val(&self, r: IRRef) -> u64 {
        debug_assert!(irref_isk(r));
        self.k64[(REF_BIAS - 1 - r) as usize]
    }
    /// KNUM payload as f64.
    #[inline]
    pub fn knum_val(&self, r: IRRef) -> f64 {
        f64::from_bits(self.k64_val(r))
    }

    /// Iterate over all constant IR instructions (in reverse-ref order).
    pub fn const_iter(&self) -> impl Iterator<Item = &IRIns> {
        self.k.iter()
    }

    /// All KGC constants as NaN-boxed values, for GC root marking.
    pub fn kgc_values(&self) -> impl Iterator<Item = crate::value::LuaValue> + '_ {
        self.k
            .iter()
            .zip(self.k64.iter())
            .filter(|&(ins, &_bits)| ins.op() == IROp::KGC)
            .map(|(_ins, &bits)| crate::value::LuaValue::from_bits(bits))
    }

    #[inline]
    fn push_k(&mut self, ins: IRIns, payload: u64) -> IRRef {
        self.k.push(ins);
        self.k64.push(payload);
        REF_BIAS - self.k.len() as IRRef
    }

    /// Intern an int32 constant (`lj_ir_kint`).
    pub fn kint(&mut self, kv: i32) -> TRef {
        let mut r = self.chain[IROp::KINT as usize] as IRRef;
        while r != 0 {
            let ir = self.ir(r);
            if ir.i() == kv {
                return tref(r, IRT_INT);
            }
            r = ir.prev as IRRef;
        }
        let mut ins = IRIns::new(irt(IROp::KINT, IRT_INT), 0, 0);
        ins.set_i(kv);
        ins.prev = self.chain[IROp::KINT as usize];
        let nr = self.push_k(ins, 0);
        self.chain[IROp::KINT as usize] = nr as IRRef1;
        tref(nr, IRT_INT)
    }

    /// Intern a 64 bit constant by bit pattern (`lj_ir_k64`).
    pub fn k64(&mut self, op: IROp, u: u64) -> TRef {
        debug_assert!(op == IROp::KNUM || op == IROp::KINT64);
        let t = if op == IROp::KNUM { IRT_NUM } else { IRT_I64 };
        let mut r = self.chain[op as usize] as IRRef;
        while r != 0 {
            if self.k64_val(r) == u {
                return tref(r, t);
            }
            r = self.ir(r).prev as IRRef;
        }
        let mut ins = IRIns::new(irt(op, t), 0, 0);
        ins.prev = self.chain[op as usize];
        let nr = self.push_k(ins, u);
        self.chain[op as usize] = nr as IRRef1;
        tref(nr, t)
    }

    /// Intern an FP constant (`lj_ir_knum`). Keyed by exact bit pattern,
    /// so +0.0 and -0.0 stay distinct.
    #[inline]
    pub fn knum(&mut self, n: f64) -> TRef {
        self.k64(IROp::KNUM, n.to_bits())
    }
    #[inline]
    pub fn knum_u64(&mut self, u: u64) -> TRef {
        self.k64(IROp::KNUM, u)
    }
    #[inline]
    pub fn kint64(&mut self, u: u64) -> TRef {
        self.k64(IROp::KINT64, u)
    }
    /// `lj_ir_knum_one`.
    #[inline]
    pub fn knum_one(&mut self) -> TRef {
        self.knum(1.0)
    }

    /// Intern a GC object constant (`lj_ir_kgc`), keyed by the value bits
    /// (a NaN-boxed `LuaValue` payload in this VM).
    pub fn kgc(&mut self, bits: u64, t: u8) -> TRef {
        let mut r = self.chain[IROp::KGC as usize] as IRRef;
        while r != 0 {
            if self.k64_val(r) == bits {
                return tref(r, t);
            }
            r = self.ir(r).prev as IRRef;
        }
        let mut ins = IRIns::new(irt(IROp::KGC, t), 0, 0);
        ins.prev = self.chain[IROp::KGC as usize];
        let nr = self.push_k(ins, bits);
        self.chain[IROp::KGC as usize] = nr as IRRef1;
        tref(nr, t)
    }

    /// Emit without any optimization (`lj_ir_emit`): append, link the CSE
    /// chain and record emitted guard types.
    pub fn emit_ins(&mut self, fins: IRIns) -> TRef {
        let op = fins.op();
        let r = self.nins();
        debug_assert!(r < REF_DROP, "IR buffer overflow");
        let mut ins = fins;
        ins.prev = self.chain[op as usize];
        self.chain[op as usize] = r as IRRef1;
        self.guardemit |= ins.t();
        self.ins.push(ins);
        tref(r, ins.t())
    }

    /// `emitir`: route an instruction through FOLD (and CSE) like the
    /// `emitir` macro. The recorder's main emission entry point.
    #[inline]
    pub fn emitir(&mut self, ot: u16, a: IRRef, b: IRRef) -> Result<TRef, TraceError> {
        super::opt_fold::opt_fold(self, IRIns::new(ot, a, b))
    }

    /// `lj_ir_rollback`: undo all instructions emitted at or above `r`,
    /// unlinking them from the CSE chains via their `prev` fields.
    pub fn rollback(&mut self, r: IRRef) {
        while self.nins() > r {
            let ins = self.ins.pop().unwrap();
            self.chain[ins.op() as usize] = ins.prev;
        }
    }
}
