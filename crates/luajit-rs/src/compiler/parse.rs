mod funcstate;

use crate::bc::*;
use crate::lex::*;
use crate::proto::{
    KGc, PROTO_BITOP, PROTO_CHILD, PROTO_FIXUP_RETURN, PROTO_HAS_RETURN, PROTO_UV_IMMUTABLE,
    PROTO_UV_LOCAL, PROTO_VARARG, Proto,
};
use crate::table::LuaTable;
use crate::value::LuaValue;
use funcstate::{FuncScope, FuncState};

const LJ_MAX_LOCVAR: u32 = 200;
const LJ_MAX_UPVAL: u32 = 60;
const LJ_MAX_SLOTS: u32 = 250;
const LJ_MAX_XLEVEL: u32 = 200;
const LJ_MAX_VSTACK: u32 = 65536 - LJ_MAX_UPVAL;
const VINDEX_NONE: u16 = 0xffff;
const VINDEX_MASK: u32 = 31;

const VSTACK_VAR_RW: u8 = 0x01;
const VSTACK_GOTO: u8 = 0x02;
const VSTACK_LABEL: u8 = 0x04;
const VSTACK_CONST: u8 = 0x08;

const FSCOPE_LOOP: u8 = 0x01;
const FSCOPE_BREAK: u8 = 0x02;
const FSCOPE_GOLA: u8 = 0x04;
const FSCOPE_UPVAL: u8 = 0x08;
const FSCOPE_NOCLOSE: u8 = 0x10;
const FSCOPE_CONT: u8 = 0x20;

const EXPR_F_NORES: u32 = 0x01;
const EXPR_F_NOCOLON: u32 = 0x02;
const EXPR_F_NONAV: u32 = 0x04;
const EXPR_F_RET1: u32 = 0x08;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum ExpKind {
    VKNil,
    VKFalse,
    VKTrue,
    VKStr,
    VKNum,
    VLocal,
    VUpval,
    VGlobal,
    VIndexed,
    VJmp,
    VRelocable,
    VNonReloc,
    VCall,
    VCallNav,
    VVoid,
}
use ExpKind::*;

#[derive(Clone, Copy)]
struct ExpDesc {
    k: ExpKind,
    info: u32,
    aux: u32,
    nval: f64,
    sval: StrId,
    t: BCPos,
    f: BCPos,
}

impl ExpDesc {
    fn init(k: ExpKind, info: u32) -> ExpDesc {
        ExpDesc {
            k,
            info,
            aux: 0,
            nval: 0.0,
            sval: 0,
            t: NO_JMP,
            f: NO_JMP,
        }
    }
    fn hasjump(&self) -> bool {
        self.t != self.f
    }
    fn isk(&self) -> bool {
        self.k <= VKNum
    }
    fn isk_nojump(&self) -> bool {
        self.isk() && !self.hasjump()
    }
    fn isnumk(&self) -> bool {
        self.k == VKNum
    }
    fn isnumk_nojump(&self) -> bool {
        self.isnumk() && !self.hasjump()
    }
    fn isstrk(&self) -> bool {
        self.k == VKStr
    }
    fn const_pri(&self) -> u32 {
        debug_assert!(self.k <= VKTrue);
        self.k as u32
    }
    fn numiszero(&self) -> bool {
        self.nval == 0.0
    }
}

#[derive(Clone, PartialEq)]
enum VName {
    None,
    Break,
    Cont,
    Fixed(u8),
    Str(StrId),
}

#[derive(Clone)]
struct VarInfo {
    name: VName,
    startpc: BCPos,
    endpc: BCPos,
    slot: u8,
    info: u8,
    prev: u16,
}

#[derive(Clone, Copy)]
struct BCInsLine {
    ins: BCIns,
    line: BCLine,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    BAnd,
    BOr,
    BXor,
    BShl,
    BShr,
    BSar,
    Concat,
    Ne,
    Eq,
    Lt,
    Ge,
    Le,
    Gt,
    And,
    Or,
    Coal,
    NoBinOp,
}

const PRIORITY: [(u8, u8); 22] = [
    (10, 10),
    (10, 10),
    (11, 11),
    (11, 11),
    (11, 11),
    (14, 13),
    (6, 6),
    (4, 4),
    (5, 5),
    (7, 7),
    (7, 7),
    (7, 7),
    (9, 8),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (2, 2),
    (1, 1),
    (1, 1),
];

const UNARY_PRIORITY: u32 = 12;

fn token2binop(tok: Tok) -> BinOp {
    match tok {
        Tok::Char(b'+') => BinOp::Add,
        Tok::Char(b'-') => BinOp::Sub,
        Tok::Char(b'*') => BinOp::Mul,
        Tok::Char(b'/') => BinOp::Div,
        Tok::Char(b'%') => BinOp::Mod,
        Tok::Char(b'^') => BinOp::Pow,
        Tok::Char(b'&') => BinOp::BAnd,
        Tok::Char(b'|') => BinOp::BOr,
        Tok::Char(b'~') => BinOp::BXor,
        Tok::Shl => BinOp::BShl,
        Tok::Shr => BinOp::BShr,
        Tok::Sar => BinOp::BSar,
        Tok::Concat => BinOp::Concat,
        Tok::Ne | Tok::NeBang => BinOp::Ne,
        Tok::Eq => BinOp::Eq,
        Tok::Char(b'<') => BinOp::Lt,
        Tok::Le => BinOp::Le,
        Tok::Char(b'>') => BinOp::Gt,
        Tok::Ge => BinOp::Ge,
        Tok::And | Tok::AndAnd => BinOp::And,
        Tok::Or | Tok::OrOr => BinOp::Or,
        Tok::Coal => BinOp::Coal,
        _ => BinOp::NoBinOp,
    }
}

fn vm_foldarith(x: f64, y: f64, op: i32) -> f64 {
    match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => x / y,
        4 => x - (x / y).floor() * y,
        5 => x.powf(y),
        _ => unreachable!(),
    }
}

fn num2bit(n: f64) -> i32 {
    let d = n + 6755399441055744.0;
    d.to_bits() as u32 as i32
}

fn ksh_i16(n: f64) -> Option<u16> {
    let k = n as i32;
    if (k as f64) == n && (-32768..=32767).contains(&k) {
        return Some(k as u16);
    }
    None
}

fn kidx_u8(n: f64) -> Option<u32> {
    let k = n as i32;
    if (k as f64) == n && (0..=255).contains(&k) {
        return Some(k as u32);
    }
    None
}

fn hsize2hbits(s: u32) -> u32 {
    if s == 0 {
        0
    } else if s == 1 {
        1
    } else {
        1 + (31 - (s - 1).leading_zeros())
    }
}

fn lex_isname(tok: Tok) -> bool {
    matches!(tok, Tok::Name | Tok::Goto | Tok::Continue | Tok::Const)
}

fn parse_isend(tok: Tok) -> bool {
    matches!(
        tok,
        Tok::Else | Tok::Elseif | Tok::End | Tok::Until | Tok::Eof
    )
}

pub struct Parser {
    ls: LexState,
    vstack: Vec<VarInfo>,
    bcstack: Vec<BCInsLine>,
    vhash: [u16; 32],
    level: u32,
    fr2: u32,
    fs: Vec<FuncState>,
}

impl Parser {
    pub fn new(src: Vec<u8>, chunkname: String) -> Parser {
        Parser::with_interner(src, chunkname, Interner::default())
    }

    pub fn with_interner(src: Vec<u8>, chunkname: String, strs: Interner) -> Parser {
        Parser {
            ls: LexState::with_interner(src, chunkname, strs),
            vstack: Vec::new(),
            bcstack: Vec::new(),
            vhash: [VINDEX_NONE; 32],
            level: 0,
            fr2: 1,
            fs: Vec::new(),
        }
    }

    fn cur(&self) -> &FuncState {
        self.fs.last().unwrap()
    }

    fn cur_mut(&mut self) -> &mut FuncState {
        self.fs.last_mut().unwrap()
    }

    fn ins(&self, pc: BCPos) -> BCIns {
        self.bcstack[self.cur().bcbase + pc as usize].ins
    }

    fn ins_mut(&mut self, pc: BCPos) -> &mut BCIns {
        let b = self.cur().bcbase;
        &mut self.bcstack[b + pc as usize].ins
    }

    fn err_syntax(&self, msg: &str) -> ! {
        self.ls.err_near(msg)
    }

    fn err_token(&self, tok: Tok) -> ! {
        self.ls.err_near(&format!("'{}' expected", tok2str(tok)))
    }

    fn err_limit(&self, limit: u32, what: &str) -> ! {
        let fs = self.cur();
        if fs.linedefined == 0 {
            self.ls
                .error(&format!("main function has more than {} {}", limit, what));
        } else {
            self.ls.error(&format!(
                "function at line {} has more than {} {}",
                fs.linedefined, limit, what
            ));
        }
    }

    fn checklimit(&self, v: u32, l: u32, m: &str) {
        if v >= l {
            self.err_limit(l, m);
        }
    }

    #[allow(dead_code)]
    fn checklimitgt(&self, v: u32, l: u32, m: &str) {
        if v > l {
            self.err_limit(l, m);
        }
    }

    // -- Management of constants -------------------------------------------

    fn const_num(&mut self, e: &ExpDesc) -> u32 {
        debug_assert!(e.isnumk());
        let fs = self.cur_mut();
        let bits = e.nval.to_bits();
        if let Some(&idx) = fs.kn_map.get(&bits) {
            return idx;
        }
        let idx = fs.kn.len() as u32;
        fs.kn.push(e.nval);
        fs.kn_map.insert(bits, idx);
        idx
    }

    fn const_str_id(&mut self, sid: StrId) -> u32 {
        let fs = self.cur_mut();
        if let Some(&idx) = fs.kgc_map.get(&sid) {
            return idx;
        }
        let idx = fs.kgc.len() as u32;
        fs.kgc.push(KGc::Str(sid));
        fs.kgc_map.insert(sid, idx);
        idx
    }

    fn const_str(&mut self, e: &ExpDesc) -> u32 {
        debug_assert!(e.isstrk() || e.k == VGlobal);
        self.const_str_id(e.sval)
    }

    fn const_proto(&mut self, pt: Proto) -> u32 {
        let fs = self.cur_mut();
        let idx = fs.kgc.len() as u32;
        fs.kgc.push(KGc::Proto(Box::new(pt)));
        idx
    }

    // -- Jump list handling --------------------------------------------------

    fn jmp_next(&self, pc: BCPos) -> BCPos {
        let delta = bc_j(self.ins(pc));
        if delta == -1 {
            NO_JMP
        } else {
            ((pc + 1) as i64 + delta) as BCPos
        }
    }

    fn jmp_novalue(&self, mut list: BCPos) -> bool {
        while list != NO_JMP {
            let p = self.ins(if list >= 1 { list - 1 } else { list });
            if !(bc_op(p) == BCOp::ISTC || bc_op(p) == BCOp::ISFC || bc_a(p) == NO_REG) {
                return true;
            }
            list = self.jmp_next(list);
        }
        false
    }

    fn jmp_patchtestreg(&mut self, pc: BCPos, reg: BCReg) -> bool {
        let b = self.cur().bcbase;
        let i = b + (if pc >= 1 { pc - 1 } else { pc }) as usize;
        let ins = self.bcstack[i].ins;
        let op = bc_op(ins);
        if op == BCOp::ISTC || op == BCOp::ISFC {
            if reg != NO_REG && reg != bc_d(ins) {
                setbc_a(&mut self.bcstack[i].ins, reg);
            } else {
                let mut ins2 = ins;
                setbc_op(
                    &mut ins2,
                    op as u32 + (BCOp::IST as u32 - BCOp::ISTC as u32),
                );
                setbc_a(&mut ins2, 0);
                self.bcstack[i].ins = ins2;
            }
        } else if bc_a(ins) == NO_REG {
            if reg == NO_REG {
                let a = bc_a(self.bcstack[b + pc as usize].ins);
                self.bcstack[i].ins = bcins_aj(BCOp::JMP, a, 0);
            } else {
                setbc_a(&mut self.bcstack[i].ins, reg);
                if reg >= bc_a(self.bcstack[i + 1].ins) {
                    setbc_a(&mut self.bcstack[i + 1].ins, reg + 1);
                }
            }
        } else {
            return false;
        }
        true
    }

    fn jmp_dropval(&mut self, mut list: BCPos) {
        while list != NO_JMP {
            self.jmp_patchtestreg(list, NO_REG);
            list = self.jmp_next(list);
        }
    }

    fn jmp_patchins(&mut self, pc: BCPos, dest: BCPos) {
        debug_assert!(dest != NO_JMP);
        let offset = (dest as i64) - ((pc + 1) as i64) + BCBIAS_J as i64;
        if !(0..=BCMAX_D as i64).contains(&offset) {
            self.err_syntax("control structure too long");
        }
        setbc_d(self.ins_mut(pc), offset as u32);
    }

    fn jmp_append(&mut self, l1: &mut BCPos, l2: BCPos) {
        if l2 == NO_JMP {
            return;
        } else if *l1 == NO_JMP {
            *l1 = l2;
        } else {
            let mut list = *l1;
            loop {
                let next = self.jmp_next(list);
                if next == NO_JMP {
                    break;
                }
                list = next;
            }
            self.jmp_patchins(list, l2);
        }
    }

    fn jmp_patchval(&mut self, mut list: BCPos, vtarget: BCPos, reg: BCReg, dtarget: BCPos) {
        while list != NO_JMP {
            let next = self.jmp_next(list);
            if self.jmp_patchtestreg(list, reg) {
                self.jmp_patchins(list, vtarget);
            } else {
                self.jmp_patchins(list, dtarget);
            }
            list = next;
        }
    }

    fn jmp_tohere(&mut self, list: BCPos) {
        let pc = self.cur().pc;
        self.cur_mut().lasttarget = pc;
        let mut jpc = self.cur().jpc;
        self.jmp_append(&mut jpc, list);
        self.cur_mut().jpc = jpc;
    }

    fn jmp_patch(&mut self, list: BCPos, target: BCPos) {
        if target == self.cur().pc {
            self.jmp_tohere(list);
        } else {
            debug_assert!(target < self.cur().pc);
            self.jmp_patchval(list, target, NO_REG, target);
        }
    }

    // -- Bytecode register allocator -----------------------------------------

    fn bcreg_bump(&mut self, n: BCReg) {
        let sz = self.cur().freereg + n;
        if sz > self.cur().framesize as u32 {
            if sz >= LJ_MAX_SLOTS {
                self.err_syntax("function or expression too complex");
            }
            self.cur_mut().framesize = sz as u8;
        }
    }

    fn bcreg_reserve(&mut self, n: BCReg) {
        self.bcreg_bump(n);
        self.cur_mut().freereg += n;
    }

    fn bcreg_free(&mut self, reg: BCReg) {
        if reg >= self.cur().nactvar {
            self.cur_mut().freereg -= 1;
            debug_assert!(reg == self.cur().freereg);
        }
    }

    fn expr_free(&mut self, e: &ExpDesc) {
        if e.k == VNonReloc {
            self.bcreg_free(e.info);
        }
    }

    // -- Bytecode emitter ----------------------------------------------------

    fn bcemit(&mut self, ins: BCIns) -> BCPos {
        let pc = self.cur().pc;
        let jpc = self.cur().jpc;
        self.jmp_patchval(jpc, pc, NO_REG, pc);
        self.cur_mut().jpc = NO_JMP;
        let idx = self.cur().bcbase + pc as usize;
        if idx >= self.bcstack.len() {
            self.bcstack.resize(idx + 1, BCInsLine { ins: 0, line: 0 });
        }
        self.bcstack[idx] = BCInsLine {
            ins,
            line: self.ls.lastline,
        };
        self.cur_mut().pc = pc + 1;
        pc
    }

    fn bcemit_abc(&mut self, o: BCOp, a: BCReg, b: BCReg, c: BCReg) -> BCPos {
        self.bcemit(bcins_abc(o, a, b, c))
    }

    fn bcemit_ad(&mut self, o: BCOp, a: BCReg, d: u32) -> BCPos {
        self.bcemit(bcins_ad(o, a, d))
    }

    fn bcemit_aj(&mut self, o: BCOp, a: BCReg, j: i64) -> BCPos {
        self.bcemit(bcins_aj(o, a, j))
    }

    // -- Bytecode emitter for expressions ------------------------------------

    fn expr_discharge(&mut self, e: &mut ExpDesc) {
        let ins;
        if e.k == VUpval {
            ins = bcins_ad(BCOp::UGET, 0, e.info);
        } else if e.k == VGlobal {
            let idx = self.const_str(e);
            ins = bcins_ad(BCOp::GGET, 0, idx);
        } else if e.k == VIndexed {
            let rc = e.aux;
            if (rc as i32) < 0 {
                ins = bcins_abc(BCOp::TGETS, 0, e.info, !rc);
            } else if rc > BCMAX_C {
                ins = bcins_abc(BCOp::TGETB, 0, e.info, rc - (BCMAX_C + 1));
            } else {
                self.bcreg_free(rc);
                ins = bcins_abc(BCOp::TGETV, 0, e.info, rc);
            }
            self.bcreg_free(e.info);
        } else if e.k == VCall || e.k == VCallNav {
            e.info = e.aux;
            e.k = VNonReloc;
            return;
        } else if e.k == VLocal {
            e.k = VNonReloc;
            return;
        } else {
            return;
        }
        e.info = self.bcemit(ins);
        e.k = VRelocable;
    }

    fn bcemit_nil(&mut self, from: BCReg, n: BCReg) {
        let mut from = from;
        let mut n = n;
        if self.cur().pc > self.cur().lasttarget {
            let pc = self.cur().pc;
            let ip = self.ins(pc - 1);
            let pfrom = bc_a(ip);
            match bc_op(ip) {
                BCOp::KPRI => {
                    if bc_d(ip) == 0 {
                        if from == pfrom {
                            if n == 1 {
                                return;
                            }
                        } else if from == pfrom + 1 {
                            from = pfrom;
                            n += 1;
                        } else {
                            let _ = ();
                            self.bcemit(if n == 1 {
                                bcins_ad(BCOp::KPRI, from, VKNil as u32)
                            } else {
                                bcins_ad(BCOp::KNIL, from, from + n - 1)
                            });
                            return;
                        }
                        *self.ins_mut(pc - 1) = bcins_ad(BCOp::KNIL, from, from + n - 1);
                        return;
                    }
                }
                BCOp::KNIL => {
                    let pto = bc_d(ip);
                    if pfrom <= from && from <= pto + 1 {
                        if from + n - 1 > pto {
                            setbc_d(self.ins_mut(pc - 1), from + n - 1);
                        }
                        return;
                    }
                }
                _ => {}
            }
        }
        self.bcemit(if n == 1 {
            bcins_ad(BCOp::KPRI, from, VKNil as u32)
        } else {
            bcins_ad(BCOp::KNIL, from, from + n - 1)
        });
    }

    fn expr_toreg_nobranch(&mut self, e: &mut ExpDesc, reg: BCReg) {
        let ins;
        self.expr_discharge(e);
        if e.k == VKStr {
            let idx = self.const_str(e);
            ins = bcins_ad(BCOp::KSTR, reg, idx);
        } else if e.k == VKNum {
            if let Some(k) = ksh_i16(e.nval) {
                ins = bcins_ad(BCOp::KSHORT, reg, k as u32);
            } else {
                let idx = self.const_num(e);
                ins = bcins_ad(BCOp::KNUM, reg, idx);
            }
        } else if e.k == VRelocable {
            let pc = e.info;
            setbc_a(self.ins_mut(pc), reg);
            e.info = reg;
            e.k = VNonReloc;
            return;
        } else if e.k == VNonReloc {
            if reg == e.info {
                return;
            }
            ins = bcins_ad(BCOp::MOV, reg, e.info);
        } else if e.k == VKNil {
            self.bcemit_nil(reg, 1);
            e.info = reg;
            e.k = VNonReloc;
            return;
        } else if e.k <= VKTrue {
            ins = bcins_ad(BCOp::KPRI, reg, e.const_pri());
        } else {
            debug_assert!(e.k == VVoid || e.k == VJmp);
            return;
        }
        self.bcemit(ins);
        e.info = reg;
        e.k = VNonReloc;
    }

    fn expr_toreg(&mut self, e: &mut ExpDesc, reg: BCReg) {
        self.expr_toreg_nobranch(e, reg);
        if e.k == VJmp {
            let info = e.info;
            let mut t = e.t;
            self.jmp_append(&mut t, info);
            e.t = t;
        }
        if e.hasjump() {
            let mut jfalse = NO_JMP;
            let mut jtrue = NO_JMP;
            if self.jmp_novalue(e.t) || self.jmp_novalue(e.f) {
                let jval = if e.k == VJmp {
                    NO_JMP
                } else {
                    self.bcemit_jmp()
                };
                jfalse = self.bcemit_ad(BCOp::KPRI, reg, VKFalse as u32);
                let freereg = self.cur().freereg;
                self.bcemit_aj(BCOp::JMP, freereg, 1);
                jtrue = self.bcemit_ad(BCOp::KPRI, reg, VKTrue as u32);
                self.jmp_tohere(jval);
            }
            let jend = self.cur().pc;
            self.cur_mut().lasttarget = jend;
            self.jmp_patchval(e.f, jend, reg, jfalse);
            self.jmp_patchval(e.t, jend, reg, jtrue);
        }
        e.f = NO_JMP;
        e.t = NO_JMP;
        e.info = reg;
        e.k = VNonReloc;
    }

    fn expr_tonextreg(&mut self, e: &mut ExpDesc) {
        self.expr_discharge(e);
        self.expr_free(e);
        self.bcreg_reserve(1);
        let reg = self.cur().freereg - 1;
        self.expr_toreg(e, reg);
    }

    fn expr_toanyreg(&mut self, e: &mut ExpDesc) -> BCReg {
        self.expr_discharge(e);
        if e.k == VNonReloc {
            if !e.hasjump() {
                return e.info;
            }
            if e.info >= self.cur().nactvar {
                let reg = e.info;
                self.expr_toreg(e, reg);
                return e.info;
            }
        }
        self.expr_tonextreg(e);
        e.info
    }

    fn expr_toval(&mut self, e: &mut ExpDesc) {
        if e.hasjump() {
            self.expr_toanyreg(e);
        } else {
            self.expr_discharge(e);
        }
    }

    fn bcemit_store(&mut self, var: &ExpDesc, e: &mut ExpDesc) {
        let ins;
        if var.k == VLocal {
            self.vstack[var.aux as usize].info |= VSTACK_VAR_RW;
            self.expr_free(e);
            self.expr_toreg(e, var.info);
            return;
        } else if var.k == VUpval {
            self.vstack[var.aux as usize].info |= VSTACK_VAR_RW;
            self.expr_toval(e);
            if e.k <= VKTrue {
                ins = bcins_ad(BCOp::USETP, var.info, e.const_pri());
            } else if e.k == VKStr {
                let idx = self.const_str(e);
                ins = bcins_ad(BCOp::USETS, var.info, idx);
            } else if e.k == VKNum {
                let idx = self.const_num(e);
                ins = bcins_ad(BCOp::USETN, var.info, idx);
            } else {
                let ra = self.expr_toanyreg(e);
                ins = bcins_ad(BCOp::USETV, var.info, ra);
            }
        } else if var.k == VGlobal {
            let ra = self.expr_toanyreg(e);
            let idx = self.const_str(var);
            ins = bcins_ad(BCOp::GSET, ra, idx);
        } else {
            debug_assert!(var.k == VIndexed);
            let ra = self.expr_toanyreg(e);
            let rc = var.aux;
            if (rc as i32) < 0 {
                ins = bcins_abc(BCOp::TSETS, ra, var.info, !rc);
            } else if rc > BCMAX_C {
                ins = bcins_abc(BCOp::TSETB, ra, var.info, rc - (BCMAX_C + 1));
            } else {
                #[cfg(debug_assertions)]
                if e.k == VNonReloc && ra >= self.cur().nactvar && rc >= ra {
                    self.bcreg_free(rc);
                }
                ins = bcins_abc(BCOp::TSETV, ra, var.info, rc);
            }
        }
        self.bcemit(ins);
        self.expr_free(e);
    }

    fn bcemit_method(&mut self, e: &mut ExpDesc, key: &ExpDesc) {
        let obj = self.expr_toanyreg(e);
        self.expr_free(e);
        let func = self.cur().freereg;
        let fr2 = self.fr2;
        self.bcemit_ad(BCOp::MOV, func + 1 + fr2, obj);
        debug_assert!(key.isstrk());
        let idx = self.const_str_id(key.sval);
        if idx <= BCMAX_C {
            self.bcreg_reserve(2 + fr2);
            self.bcemit_abc(BCOp::TGETS, func, obj, idx);
        } else {
            self.bcreg_reserve(3 + fr2);
            self.bcemit_ad(BCOp::KSTR, func + 2 + fr2, idx);
            self.bcemit_abc(BCOp::TGETV, func, obj, func + 2 + fr2);
            self.cur_mut().freereg -= 1;
        }
        e.info = func;
        e.k = VNonReloc;
    }

    // -- Bytecode emitter for branches ---------------------------------------

    fn bcemit_jmp(&mut self) -> BCPos {
        let jpc = self.cur().jpc;
        let mut j = self.cur().pc - 1;
        self.cur_mut().jpc = NO_JMP;
        let ip = self.ins(j);
        if (j as i64) >= (self.cur().lasttarget as i64) && bc_op(ip) == BCOp::UCLO {
            setbc_j(self.ins_mut(j), -1);
            self.cur_mut().lasttarget = j + 1;
        } else {
            let freereg = self.cur().freereg;
            j = self.bcemit_aj(BCOp::JMP, freereg, -1);
        }
        let mut jl = j;
        self.jmp_append(&mut jl, jpc);
        jl
    }

    fn invertcond(&mut self, e: &ExpDesc) {
        let pc = e.info - 1;
        let ins = self.ins(pc);
        let op = bc_op(ins) as u32 ^ 1;
        setbc_op(self.ins_mut(pc), op);
    }

    fn bcemit_branch(&mut self, e: &mut ExpDesc, cond: bool) -> BCPos {
        if e.k == VRelocable {
            let pc = e.info;
            let ip = self.ins(pc);
            if bc_op(ip) == BCOp::NOT {
                *self.ins_mut(pc) = bcins_ad(if cond { BCOp::ISF } else { BCOp::IST }, 0, bc_d(ip));
                return self.bcemit_jmp();
            }
        }
        if e.k != VNonReloc {
            self.bcreg_reserve(1);
            let reg = self.cur().freereg - 1;
            self.expr_toreg_nobranch(e, reg);
        }
        self.bcemit_ad(if cond { BCOp::ISTC } else { BCOp::ISFC }, NO_REG, e.info);
        let pc = self.bcemit_jmp();
        self.expr_free(e);
        pc
    }

    fn bcemit_branch_t(&mut self, e: &mut ExpDesc) {
        self.expr_discharge(e);
        let pc;
        if e.k == VKStr || e.k == VKNum || e.k == VKTrue {
            pc = NO_JMP;
        } else if e.k == VJmp {
            self.invertcond(e);
            pc = e.info;
        } else if e.k == VKFalse || e.k == VKNil {
            self.expr_toreg_nobranch(e, NO_REG);
            pc = self.bcemit_jmp();
        } else {
            pc = self.bcemit_branch(e, false);
        }
        let mut f = e.f;
        self.jmp_append(&mut f, pc);
        e.f = f;
        self.jmp_tohere(e.t);
        e.t = NO_JMP;
    }

    fn bcemit_branch_f(&mut self, e: &mut ExpDesc) {
        self.expr_discharge(e);
        let pc;
        if e.k == VKNil || e.k == VKFalse {
            pc = NO_JMP;
        } else if e.k == VJmp {
            pc = e.info;
        } else if e.k == VKStr || e.k == VKNum || e.k == VKTrue {
            self.expr_toreg_nobranch(e, NO_REG);
            pc = self.bcemit_jmp();
        } else {
            pc = self.bcemit_branch(e, true);
        }
        let mut t = e.t;
        self.jmp_append(&mut t, pc);
        e.t = t;
        self.jmp_tohere(e.f);
        e.f = NO_JMP;
    }

    // -- Bytecode emitter for operators --------------------------------------

    fn foldarith(&self, opr: BinOp, e1: &mut ExpDesc, e2: &ExpDesc) -> bool {
        if !e1.isnumk_nojump() || !e2.isnumk_nojump() {
            return false;
        }
        let n = vm_foldarith(e1.nval, e2.nval, opr as i32 - BinOp::Add as i32);
        if n.is_nan() || n.to_bits() == (-0.0f64).to_bits() {
            return false;
        }
        e1.nval = n;
        true
    }

    fn foldbitop(&self, opr: BinOp, e1: &mut ExpDesc, e2: &ExpDesc) -> bool {
        if e1.isnumk_nojump() && e2.isnumk_nojump() {
            let mut k1 = num2bit(e1.nval);
            let k2 = num2bit(e2.nval);
            match opr {
                BinOp::BAnd => k1 &= k2,
                BinOp::BOr => k1 |= k2,
                BinOp::BXor => k1 ^= k2,
                BinOp::BShl => k1 = k1.wrapping_shl((k2 & 31) as u32),
                BinOp::BShr => k1 = ((k1 as u32) >> (k2 & 31)) as i32,
                BinOp::BSar => k1 >>= k2 & 31,
                _ => unreachable!(),
            }
            e1.nval = k1 as f64;
            return true;
        }
        false
    }

    fn bcemit_arith(&mut self, opr: BinOp, e1: &mut ExpDesc, e2: &mut ExpDesc) {
        let rb;
        let rc;
        let mut op;
        if opr == BinOp::Pow {
            op = BCOp::POW as u32;
            rc = self.expr_toanyreg(e2);
            rb = self.expr_toanyreg(e1);
        } else if opr >= BinOp::BAnd {
            op = BCOp::BAND as u32 + (opr as u32 - BinOp::BAnd as u32);
            rc = self.expr_toanyreg(e2);
            rb = self.expr_toanyreg(e1);
        } else {
            op = BCOp::ADDVV as u32 + (opr as u32 - BinOp::Add as u32);
            self.expr_toval(e2);
            let mut rc2;
            if e2.isnumk() && {
                rc2 = self.const_num(e2);
                rc2 <= BCMAX_C
            } {
                op -= BCOp::ADDVV as u32 - BCOp::ADDVN as u32;
            } else {
                rc2 = self.expr_toanyreg(e2);
            }
            debug_assert!(e1.isnumk() || e1.k == VNonReloc);
            self.expr_toval(e1);
            if e1.isnumk() && !e2.isnumk() {
                let t = self.const_num(e1);
                if t <= BCMAX_B {
                    rb = rc2;
                    rc = t;
                    op -= BCOp::ADDVV as u32 - BCOp::ADDNV as u32;
                } else {
                    rb = self.expr_toanyreg(e1);
                    rc = rc2;
                }
            } else {
                rb = self.expr_toanyreg(e1);
                rc = rc2;
            }
        }
        if e1.k == VNonReloc && e1.info >= self.cur().nactvar {
            self.cur_mut().freereg -= 1;
        }
        if e2.k == VNonReloc && e2.info >= self.cur().nactvar {
            self.cur_mut().freereg -= 1;
        }
        e1.info = self.bcemit_abc(BCOp::from_u32(op), 0, rb, rc);
        e1.k = VRelocable;
    }

    fn bcemit_comp(&mut self, opr: BinOp, e1: &mut ExpDesc, e2: &mut ExpDesc) {
        self.expr_toval(e1);
        let ins;
        let mut swapped = false;
        if opr == BinOp::Eq || opr == BinOp::Ne {
            let op = if opr == BinOp::Eq {
                BCOp::ISEQV
            } else {
                BCOp::ISNEV
            };
            if e1.isk() {
                std::mem::swap(e1, e2);
                swapped = true;
            }
            let ra = self.expr_toanyreg(e1);
            self.expr_toval(e2);
            ins = match e2.k {
                VKNil | VKFalse | VKTrue => bcins_ad(
                    op.offset(BCOp::ISEQP as i32 - BCOp::ISEQV as i32),
                    ra,
                    e2.const_pri(),
                ),
                VKStr => {
                    let idx = self.const_str(e2);
                    bcins_ad(op.offset(BCOp::ISEQS as i32 - BCOp::ISEQV as i32), ra, idx)
                }
                VKNum => {
                    let idx = self.const_num(e2);
                    bcins_ad(op.offset(BCOp::ISEQN as i32 - BCOp::ISEQV as i32), ra, idx)
                }
                _ => {
                    let rd = self.expr_toanyreg(e2);
                    bcins_ad(op, ra, rd)
                }
            };
        } else {
            let mut op = opr as u32 - BinOp::Lt as u32 + BCOp::ISLT as u32;
            let ra;
            let rd;
            if (op - BCOp::ISLT as u32) & 1 != 0 {
                std::mem::swap(e1, e2);
                swapped = true;
                op = ((op - BCOp::ISLT as u32) ^ 3) + BCOp::ISLT as u32;
                self.expr_toval(e1);
                ra = self.expr_toanyreg(e1);
                rd = self.expr_toanyreg(e2);
            } else {
                rd = self.expr_toanyreg(e2);
                ra = self.expr_toanyreg(e1);
            }
            ins = bcins_ad(BCOp::from_u32(op), ra, rd);
        }
        if e1.k == VNonReloc && e1.info >= self.cur().nactvar {
            self.cur_mut().freereg -= 1;
        }
        if e2.k == VNonReloc && e2.info >= self.cur().nactvar {
            self.cur_mut().freereg -= 1;
        }
        self.bcemit(ins);
        let jpc = self.bcemit_jmp();
        let _ = swapped;
        e1.info = jpc;
        e1.k = VJmp;
    }

    fn bcemit_binop_left(&mut self, op: BinOp, e: &mut ExpDesc) {
        if op == BinOp::And {
            self.bcemit_branch_t(e);
        } else if op == BinOp::Or {
            self.bcemit_branch_f(e);
        } else if op == BinOp::Coal {
            self.expr_tonextreg(e);
            let reg = e.info;
            self.bcemit(bcins_ad(BCOp::ISNEP, reg, VKNil as u32));
            e.aux = self.bcemit_jmp();
            self.bcreg_free(reg);
        } else if op == BinOp::Concat {
            self.expr_tonextreg(e);
        } else if op == BinOp::Eq || op == BinOp::Ne {
            if !e.isk_nojump() {
                self.expr_toanyreg(e);
            }
        } else {
            if !e.isnumk_nojump() {
                self.expr_toanyreg(e);
            }
        }
    }

    fn bcemit_binop(&mut self, op: BinOp, e1: &mut ExpDesc, e2: &mut ExpDesc) {
        if op <= BinOp::Pow {
            if !self.foldarith(op, e1, e2) {
                self.bcemit_arith(op, e1, e2);
            }
        } else if op <= BinOp::BSar {
            if !self.foldbitop(op, e1, e2) {
                self.cur_mut().flags |= PROTO_BITOP;
                self.bcemit_arith(op, e1, e2);
            }
        } else if op == BinOp::And {
            debug_assert!(e1.t == NO_JMP);
            self.expr_discharge(e2);
            let mut f = e2.f;
            self.jmp_append(&mut f, e1.f);
            e2.f = f;
            *e1 = *e2;
        } else if op == BinOp::Or {
            debug_assert!(e1.f == NO_JMP);
            self.expr_discharge(e2);
            let mut t = e2.t;
            self.jmp_append(&mut t, e1.t);
            e2.t = t;
            *e1 = *e2;
        } else if op == BinOp::Coal {
            self.expr_tonextreg(e2);
            self.jmp_tohere(e1.aux);
        } else if op == BinOp::Concat {
            self.expr_toval(e2);
            if e2.k == VRelocable && bc_op(self.ins(e2.info)) == BCOp::CAT {
                debug_assert!(e1.info == bc_b(self.ins(e2.info)) - 1);
                self.expr_free(e1);
                let pc = e2.info;
                let b = e1.info;
                setbc_b(self.ins_mut(pc), b);
                e1.info = e2.info;
            } else {
                self.expr_tonextreg(e2);
                self.expr_free(e2);
                self.expr_free(e1);
                e1.info = self.bcemit_abc(BCOp::CAT, 0, e1.info, e2.info);
            }
            e1.k = VRelocable;
        } else {
            debug_assert!(matches!(
                op,
                BinOp::Ne | BinOp::Eq | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Gt
            ));
            self.bcemit_comp(op, e1, e2);
        }
    }

    fn bcemit_unop(&mut self, op: BCOp, e: &mut ExpDesc) {
        if op == BCOp::NOT {
            std::mem::swap(&mut e.t, &mut e.f);
            self.jmp_dropval(e.f);
            self.jmp_dropval(e.t);
            self.expr_discharge(e);
            if e.k == VKNil || e.k == VKFalse {
                e.k = VKTrue;
                return;
            } else if e.isk() {
                e.k = VKFalse;
                return;
            } else if e.k == VJmp {
                self.invertcond(e);
                return;
            } else if e.k == VRelocable {
                self.bcreg_reserve(1);
                let reg = self.cur().freereg - 1;
                let pc = e.info;
                setbc_a(self.ins_mut(pc), reg);
                e.info = reg;
                e.k = VNonReloc;
            } else {
                debug_assert!(e.k == VNonReloc);
            }
        } else {
            debug_assert!(op == BCOp::UNM || op == BCOp::LEN || op == BCOp::BNOT);
            if !e.hasjump() {
                if op == BCOp::UNM {
                    if e.isnumk() && !e.numiszero() {
                        e.nval = f64::from_bits(e.nval.to_bits() ^ 0x8000000000000000);
                        return;
                    }
                } else if op == BCOp::BNOT && e.isnumk() {
                    e.nval = (!(num2bit(e.nval) as u32) as i32) as f64;
                    return;
                }
            }
            if op == BCOp::BNOT {
                self.cur_mut().flags |= PROTO_BITOP;
            }
            self.expr_toanyreg(e);
        }
        self.expr_free(e);
        e.info = self.bcemit_ad(op, 0, e.info);
        e.k = VRelocable;
    }

    // -- Lexer support -------------------------------------------------------

    fn lex_opt(&mut self, tok: Tok) -> bool {
        if self.ls.tok == tok {
            self.ls.next();
            return true;
        }
        false
    }

    fn lex_check(&mut self, tok: Tok) {
        if self.ls.tok != tok {
            self.err_token(tok);
        }
        self.ls.next();
    }

    fn lex_match(&mut self, what: Tok, who: Tok, line: BCLine) {
        if !self.lex_opt(what) {
            if line == self.ls.linenumber {
                self.err_token(what);
            } else {
                self.ls.err_near(&format!(
                    "'{}' expected (to close '{}' at line {})",
                    tok2str(what),
                    tok2str(who),
                    line
                ));
            }
        }
    }

    fn lex_str(&mut self) -> StrId {
        if !lex_isname(self.ls.tok) {
            self.err_token(Tok::Name);
        }
        let s = self.ls.tokval.str;
        self.ls.next();
        s
    }

    // -- Variable handling ---------------------------------------------------

    fn var_hash(&self, name: &VName) -> Option<u32> {
        match name {
            VName::Str(sid) => Some(sid & VINDEX_MASK),
            _ => None,
        }
    }

    fn var_new(&mut self, n: BCReg, name: VName) -> usize {
        let vtop = self.vstack.len();
        if let VName::Str(sid) = &name {
            let mut vidx = self.vhash[(sid & VINDEX_MASK) as usize];
            while vidx != VINDEX_NONE {
                let v = &self.vstack[vidx as usize];
                if v.name == name && (v.info & VSTACK_CONST) != 0 {
                    self.ls.error(&format!(
                        "attempt to redeclare constant '{}'",
                        String::from_utf8_lossy(self.ls.strs.get(*sid))
                    ));
                }
                vidx = v.prev;
            }
        }
        self.checklimit(self.cur().nactvar + n, LJ_MAX_LOCVAR, "local variables");
        if vtop as u32 >= LJ_MAX_VSTACK {
            self.ls.error("too many syntax levels");
        }
        self.vstack.push(VarInfo {
            name,
            startpc: 0,
            endpc: 0,
            slot: 0,
            info: 0,
            prev: VINDEX_NONE,
        });
        let nactvar = self.cur().nactvar;
        self.cur_mut().varmap[(nactvar + n) as usize] = vtop as u16;
        vtop
    }

    fn var_new_lit(&mut self, n: BCReg, name: &[u8]) -> usize {
        let sid = self.ls.strs.intern(name);
        self.var_new(n, VName::Str(sid))
    }

    fn var_add(&mut self, nvars: BCReg) {
        let mut nactvar = self.cur().nactvar;
        for _ in 0..nvars {
            let vidx = self.cur().varmap[nactvar as usize];
            let pc = self.cur().pc;
            let v = &mut self.vstack[vidx as usize];
            v.startpc = pc;
            v.slot = nactvar as u8;
            nactvar += 1;
            let hash = self.var_hash(&self.vstack[vidx as usize].name);
            if let Some(h) = hash {
                self.vstack[vidx as usize].prev = self.vhash[h as usize];
                self.vhash[h as usize] = vidx;
            }
        }
        self.cur_mut().nactvar = nactvar;
    }

    fn var_remove(&mut self, tolevel: BCReg) {
        while self.cur().nactvar > tolevel {
            self.cur_mut().nactvar -= 1;
            let nactvar = self.cur().nactvar;
            let vidx = self.cur().varmap[nactvar as usize];
            let pc = self.cur().pc;
            let v = &mut self.vstack[vidx as usize];
            v.endpc = pc;
            let hash = self.var_hash(&self.vstack[vidx as usize].name);
            if let Some(h) = hash {
                self.vhash[h as usize] = self.vstack[vidx as usize].prev;
            }
        }
    }

    fn var_lookup(&mut self, e: &mut ExpDesc, name: StrId) -> u32 {
        let mut vidx = self.vhash[(name & VINDEX_MASK) as usize];
        while vidx != VINDEX_NONE {
            let v = self.vstack[vidx as usize].clone();
            if v.name == VName::Str(name) {
                let top = self.fs.len() - 1;
                if vidx as usize >= self.fs[top].vbase {
                    *e = ExpDesc::init(VLocal, v.slot as u32);
                    e.aux = vidx as u32;
                } else {
                    e.aux = vidx as u32;
                    let nuv = self.fs[top].nuv as usize;
                    let mut found = false;
                    for uvidx in 0..nuv {
                        if self.fs[top].uvmap[uvidx] == vidx {
                            let aux = e.aux;
                            *e = ExpDesc::init(VUpval, uvidx as u32);
                            e.aux = aux;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let aux = e.aux;
                        *e = ExpDesc::init(VUpval, self.fs[top].nuv as u32);
                        e.aux = aux;
                        let mut fi = top;
                        loop {
                            let nuv = self.fs[fi].nuv as usize;
                            self.checklimit(nuv as u32, LJ_MAX_UPVAL, "upvalues");
                            self.fs[fi].uvmap[nuv] = vidx;
                            self.fs[fi].nuv = (nuv + 1) as u8;
                            let pend = (fi, nuv);
                            fi -= 1;
                            if vidx as usize >= self.fs[fi].vbase {
                                self.fs[pend.0].uvtmp[pend.1] = vidx;
                                self.fscope_uvmark(fi, v.slot as u32);
                                break;
                            }
                            let n = self.fs[fi].nuv as usize;
                            let mut uvi = None;
                            for uvidx in 0..n {
                                if self.fs[fi].uvmap[uvidx] == vidx {
                                    uvi = Some(uvidx);
                                    break;
                                }
                            }
                            if let Some(uvidx) = uvi {
                                self.fs[pend.0].uvtmp[pend.1] =
                                    (LJ_MAX_VSTACK as usize + uvidx) as u16;
                                break;
                            }
                            self.fs[pend.0].uvtmp[pend.1] = (LJ_MAX_VSTACK as usize + n) as u16;
                        }
                    }
                }
                return vidx as u32;
            }
            vidx = self.vstack[vidx as usize].prev;
        }
        *e = ExpDesc::init(VGlobal, 0);
        e.sval = name;
        VINDEX_NONE as u32
    }

    fn var_assign_check(&self, e: &ExpDesc) {
        if e.k == VLocal || e.k == VUpval {
            let v = &self.vstack[e.aux as usize];
            if (v.info & VSTACK_CONST) != 0 {
                if let VName::Str(sid) = &v.name {
                    self.ls.error(&format!(
                        "attempt to assign to constant '{}'",
                        String::from_utf8_lossy(self.ls.strs.get(*sid))
                    ));
                }
                self.ls.error("attempt to assign to constant");
            }
        }
    }

    // -- Goto and label handling ---------------------------------------------

    fn gola_new(&mut self, name: VName, info: u8, pc: BCPos) -> usize {
        let vtop = self.vstack.len();
        if vtop as u32 >= LJ_MAX_VSTACK {
            self.ls.error("too many syntax levels");
        }
        let nactvar = self.cur().nactvar;
        self.vstack.push(VarInfo {
            name,
            startpc: pc,
            endpc: 0,
            slot: nactvar as u8,
            info,
            prev: VINDEX_NONE,
        });
        vtop
    }

    fn gola_isgoto(&self, v: usize) -> bool {
        (self.vstack[v].info & VSTACK_GOTO) != 0
    }

    fn gola_islabel(&self, v: usize) -> bool {
        (self.vstack[v].info & VSTACK_LABEL) != 0
    }

    #[allow(dead_code)]
    fn gola_isgotolabel(&self, v: usize) -> bool {
        (self.vstack[v].info & (VSTACK_GOTO | VSTACK_LABEL)) != 0
    }

    fn gola_patch(&mut self, vg: usize, vl: usize) {
        let pc = self.vstack[vg].startpc;
        self.vstack[vg].name = VName::None;
        let slot = self.vstack[vl].slot as u32;
        setbc_a(self.ins_mut(pc), slot);
        let target = self.vstack[vl].startpc;
        self.jmp_patch(pc, target);
    }

    fn gola_close(&mut self, vg: usize) {
        let pc = self.vstack[vg].startpc;
        let slot = self.vstack[vg].slot as u32;
        let ins = self.ins(pc);
        debug_assert!(bc_op(ins) == BCOp::JMP || bc_op(ins) == BCOp::UCLO);
        setbc_a(self.ins_mut(pc), slot);
        if bc_op(ins) == BCOp::JMP {
            let next = self.jmp_next(pc);
            if next != NO_JMP {
                self.jmp_patch(next, pc);
            }
            setbc_op(self.ins_mut(pc), BCOp::UCLO as u32);
            setbc_j(self.ins_mut(pc), -1);
        }
    }

    fn gola_resolve(&mut self, bl_vstart: u32, idx: usize) {
        let mut vg = bl_vstart as usize;
        while vg < idx {
            if self.vstack[vg].name == self.vstack[idx].name && self.gola_isgoto(vg) {
                if self.vstack[vg].slot < self.vstack[idx].slot {
                    let slot = self.vstack[vg].slot as u32;
                    let vidx = self.cur().varmap[slot as usize];
                    let name = self.vstack[vidx as usize].name.clone();
                    let namestr = match name {
                        VName::Str(sid) => {
                            String::from_utf8_lossy(self.ls.strs.get(sid)).into_owned()
                        }
                        _ => "?".into(),
                    };
                    if self.vstack[vg].name == VName::Cont {
                        self.ls.error(&format!(
                            "'continue' jumps into the scope of local '{}'",
                            namestr
                        ));
                    } else {
                        let gname = match &self.vstack[vg].name {
                            VName::Str(sid) => {
                                String::from_utf8_lossy(self.ls.strs.get(*sid)).into_owned()
                            }
                            _ => "?".into(),
                        };
                        self.ls.error(&format!(
                            "<goto {}> jumps into the scope of local '{}'",
                            gname, namestr
                        ));
                    }
                }
                self.gola_patch(vg, idx);
            }
            vg += 1;
        }
    }

    fn gola_fixup(&mut self, bl_vstart: u32, bl_flags: u8, bl_nactvar: u8, has_prev: bool) {
        let mut v = bl_vstart as usize;
        while v < self.vstack.len() {
            let name = self.vstack[v].name.clone();
            if name != VName::None {
                if self.gola_islabel(v) {
                    self.vstack[v].name = VName::None;
                    let mut vg = v + 1;
                    while vg < self.vstack.len() {
                        if self.vstack[vg].name == name && self.gola_isgoto(vg) {
                            if (bl_flags & FSCOPE_UPVAL) != 0
                                && self.vstack[vg].slot > self.vstack[v].slot
                            {
                                self.gola_close(vg);
                            }
                            self.gola_patch(vg, v);
                        }
                        vg += 1;
                    }
                } else if self.gola_isgoto(v) {
                    if has_prev {
                        let flag = if name == VName::Break {
                            FSCOPE_BREAK
                        } else if name == VName::Cont {
                            FSCOPE_CONT
                        } else {
                            FSCOPE_GOLA
                        };
                        let fs = self.cur_mut();
                        fs.scopes.last_mut().unwrap().flags |= flag;
                        self.vstack[v].slot = bl_nactvar;
                        if (bl_flags & FSCOPE_UPVAL) != 0 {
                            self.gola_close(v);
                        }
                    } else {
                        self.ls.linenumber = self.line_of(self.vstack[v].startpc);
                        if name == VName::Break {
                            self.ls.error("break outside loop");
                        } else if name == VName::Cont {
                            self.ls.error("continue outside loop");
                        } else {
                            let gname = match &name {
                                VName::Str(sid) => {
                                    String::from_utf8_lossy(self.ls.strs.get(*sid)).into_owned()
                                }
                                _ => "?".into(),
                            };
                            self.ls.error(&format!("undefined label '{}'", gname));
                        }
                    }
                }
            }
            v += 1;
        }
    }

    fn line_of(&self, pc: BCPos) -> BCLine {
        self.bcstack[self.cur().bcbase + pc as usize].line
    }

    fn gola_findlabel(&self, name: &VName) -> Option<usize> {
        let vstart = self.cur().scopes.last().unwrap().vstart as usize;
        for v in vstart..self.vstack.len() {
            if self.vstack[v].name == *name && self.gola_islabel(v) {
                return Some(v);
            }
        }
        None
    }

    // -- Scope handling ------------------------------------------------------

    fn fscope_begin(&mut self, flags: u8) {
        let nactvar = self.cur().nactvar as u8;
        let vstart = self.vstack.len() as u32;
        let fs = self.cur_mut();
        fs.scopes.push(FuncScope {
            vstart,
            nactvar,
            flags,
        });
        debug_assert!(self.cur().freereg == self.cur().nactvar);
    }

    fn fscope_end(&mut self) {
        let mut bl = self.cur_mut().scopes.pop().unwrap();
        self.var_remove(bl.nactvar as u32);
        let nactvar = self.cur().nactvar;
        self.cur_mut().freereg = nactvar;
        debug_assert!(bl.nactvar as u32 == self.cur().nactvar);
        if (bl.flags & (FSCOPE_UPVAL | FSCOPE_NOCLOSE)) == FSCOPE_UPVAL {
            self.bcemit_aj(BCOp::UCLO, bl.nactvar as u32, 0);
        }
        debug_assert!((bl.flags & (FSCOPE_LOOP | FSCOPE_CONT)) != (FSCOPE_LOOP | FSCOPE_CONT));
        if (bl.flags & (FSCOPE_LOOP | FSCOPE_BREAK)) == (FSCOPE_LOOP | FSCOPE_BREAK) {
            bl.flags &= !FSCOPE_BREAK;
            let pc = self.cur().pc;
            let idx = self.gola_new(VName::Break, VSTACK_LABEL, pc);
            self.vstack.truncate(idx + 1);
            self.gola_resolve(bl.vstart, idx);
            self.vstack.truncate(idx);
        }
        if (bl.flags & (FSCOPE_GOLA | FSCOPE_BREAK | FSCOPE_CONT)) != 0 {
            let has_prev = !self.cur().scopes.is_empty();
            self.gola_fixup(bl.vstart, bl.flags, bl.nactvar, has_prev);
        }
    }

    fn fscope_continue(&mut self, cont: BCPos) {
        let bl_flags = self.cur().scopes.last().unwrap().flags;
        if (bl_flags & FSCOPE_CONT) != 0 {
            self.cur_mut().scopes.last_mut().unwrap().flags &= !FSCOPE_CONT;
            let vstart = self.cur().scopes.last().unwrap().vstart;
            let idx = self.gola_new(VName::Cont, VSTACK_LABEL, cont);
            self.vstack.truncate(idx + 1);
            self.gola_resolve(vstart, idx);
            self.vstack.truncate(idx);
        }
    }

    fn fscope_uvmark(&mut self, fi: usize, level: u32) {
        let fs = &mut self.fs[fi];
        for bl in fs.scopes.iter_mut().rev() {
            if (bl.nactvar as u32) <= level {
                bl.flags |= FSCOPE_UPVAL;
                return;
            }
        }
    }

    // -- Function state management -------------------------------------------

    fn fs_fixup_ret(&mut self) {
        let lastpc = self.cur().pc;
        let lastins = self.ins(lastpc - 1);
        if lastpc <= self.cur().lasttarget || !bc_isret_or_tail(bc_op(lastins)) {
            if (self.cur().scopes.first().unwrap().flags & FSCOPE_UPVAL) != 0 {
                self.bcemit_aj(BCOp::UCLO, 0, 0);
            }
            self.bcemit_ad(BCOp::RET0, 0, 1);
        }
        self.cur_mut().scopes.first_mut().unwrap().flags |= FSCOPE_NOCLOSE;
        debug_assert!(self.cur().scopes.len() == 1);
        self.fscope_end();
        debug_assert!(self.cur().scopes.is_empty());
        if (self.cur().flags & PROTO_FIXUP_RETURN) != 0 {
            let mut pc = 1;
            while pc < lastpc {
                let ins = self.ins(pc);
                match bc_op(ins) {
                    BCOp::CALLMT
                    | BCOp::CALLT
                    | BCOp::RETM
                    | BCOp::RET
                    | BCOp::RET0
                    | BCOp::RET1 => {
                        let offset = self.bcemit(ins);
                        let line = self.line_of(pc);
                        let b = self.cur().bcbase;
                        self.bcstack[b + offset as usize].line = line;
                        let offset = (offset as i64) - ((pc + 1) as i64) + BCBIAS_J as i64;
                        if offset > BCMAX_D as i64 {
                            self.err_syntax("function too long for return fixup");
                        }
                        *self.ins_mut(pc) = bcins_ad(BCOp::UCLO, 0, offset as u32);
                    }
                    BCOp::FNEW => return,
                    _ => {}
                }
                pc += 1;
            }
        }
    }

    fn fs_finish(&mut self, line: BCLine) -> Proto {
        self.fs_fixup_ret();
        let fs = self.fs.pop().unwrap();
        let numline = line - fs.linedefined;
        self.checklimitgt_static(fs.kn.len() as u32, BCMAX_D + 1, "constants", fs.linedefined);
        self.checklimitgt_static(
            fs.kgc.len() as u32,
            BCMAX_D + 1,
            "constants",
            fs.linedefined,
        );

        let n = fs.pc as usize;
        let mut bc = Vec::with_capacity(n);
        let mut lines = Vec::with_capacity(n);
        let op = if (fs.flags & PROTO_VARARG) != 0 {
            BCOp::FUNCV
        } else {
            BCOp::FUNCF
        };
        bc.push(bcins_ad(op, fs.framesize as u32, 0));
        lines.push(fs.linedefined);
        for i in 1..n {
            bc.push(self.bcstack[fs.bcbase + i].ins);
            lines.push(self.bcstack[fs.bcbase + i].line);
        }

        let mut kgc = fs.kgc;
        for k in kgc.iter_mut() {
            if let KGc::Proto(pt) = k {
                for uv in pt.uv.iter_mut() {
                    let vidx = *uv;
                    if vidx as u32 >= LJ_MAX_VSTACK {
                        *uv = vidx - LJ_MAX_VSTACK as u16;
                    } else if (self.vstack[vidx as usize].info & VSTACK_VAR_RW) != 0 {
                        *uv = (self.vstack[vidx as usize].slot as u16) | PROTO_UV_LOCAL;
                    } else {
                        *uv = (self.vstack[vidx as usize].slot as u16)
                            | PROTO_UV_LOCAL
                            | PROTO_UV_IMMUTABLE;
                    }
                }
            }
        }

        let nuv = fs.nuv as usize;
        let uv: Vec<u16> = fs.uvtmp[..nuv].to_vec();
        let mut uvnames = Vec::with_capacity(nuv);
        for i in 0..nuv {
            let vidx = fs.uvmap[i] as usize;
            let name = match &self.vstack[vidx].name {
                VName::Str(sid) => String::from_utf8_lossy(self.ls.strs.get(*sid)).into_owned(),
                VName::Fixed(n) => fixed_varname(*n).to_string(),
                _ => String::new(),
            };
            uvnames.push(name);
        }

        self.vstack.truncate(fs.vbase);

        Proto {
            bc,
            lines,
            kgc,
            kn: fs.kn,
            uv,
            flags: fs.flags & !(PROTO_HAS_RETURN | PROTO_FIXUP_RETURN),
            numparams: fs.numparams,
            framesize: fs.framesize,
            firstline: fs.linedefined,
            numline,
            uvnames,
        }
    }

    fn checklimitgt_static(&self, v: u32, l: u32, m: &str, linedefined: BCLine) {
        if v > l {
            if linedefined == 0 {
                self.ls
                    .error(&format!("main function has more than {} {}", l, m));
            } else {
                self.ls.error(&format!(
                    "function at line {} has more than {} {}",
                    linedefined, l, m
                ));
            }
        }
    }

    fn fs_init(&mut self) {
        let vbase = self.vstack.len();
        self.fs.push(FuncState::new(vbase));
    }

    // -- Expressions ---------------------------------------------------------

    fn expr_str_tok(&mut self, e: &mut ExpDesc) {
        *e = ExpDesc::init(VKStr, 0);
        e.sval = self.lex_str();
    }

    fn expr_index(&mut self, t: &mut ExpDesc, e: &mut ExpDesc) {
        t.k = VIndexed;
        if e.isnumk() {
            if let Some(k) = kidx_u8(e.nval) {
                t.aux = BCMAX_C + 1 + k;
                return;
            }
        } else if e.isstrk() {
            let idx = self.const_str(e);
            if idx <= BCMAX_C {
                t.aux = !idx;
                return;
            }
        }
        t.aux = self.expr_toanyreg(e);
    }

    fn expr_field(&mut self, v: &mut ExpDesc) {
        self.expr_toanyreg(v);
        let mut key = ExpDesc::init(VVoid, 0);
        self.expr_str_tok(&mut key);
        self.expr_index(v, &mut key);
    }

    fn expr_bracket(&mut self, v: &mut ExpDesc) {
        self.ls.next();
        self.expr(v, 0);
        self.expr_toval(v);
        self.lex_check(Tok::Char(b']'));
    }

    fn expr_kvalue(&self, e: &ExpDesc) -> LuaValue {
        if e.k <= VKTrue {
            match e.k {
                VKNil => LuaValue::NIL,
                VKFalse => LuaValue::FALSE,
                _ => LuaValue::TRUE,
            }
        } else if e.k == VKStr {
            LuaValue::string(self.ls.strs.lookup_ptr(e.sval))
        } else {
            debug_assert!(e.k == VKNum);
            LuaValue::number(e.nval)
        }
    }

    fn expr_table(&mut self, e: &mut ExpDesc) {
        let line = self.ls.linenumber;
        let mut t: Option<(u32, LuaTable)> = None;
        let mut vcall = false;
        let mut needarr = false;
        let mut narr: u32 = 1;
        let mut nhash: u32 = 0;
        let freg = self.cur().freereg;
        let pc = self.bcemit_ad(BCOp::TNEW, freg, 0);
        *e = ExpDesc::init(VNonReloc, freg);
        self.bcreg_reserve(1);
        let freg = freg + 1;
        self.lex_check(Tok::Char(b'{'));
        while self.ls.tok != Tok::Char(b'}') {
            let mut key = ExpDesc::init(VVoid, 0);
            let mut val = ExpDesc::init(VVoid, 0);
            vcall = false;
            if self.ls.tok == Tok::Char(b'[') {
                self.expr_bracket(&mut key);
                if !key.isk() {
                    self.expr_index(e, &mut key);
                }
                if key.isnumk() && key.numiszero() {
                    needarr = true;
                } else {
                    nhash += 1;
                }
                self.lex_check(Tok::Char(b'='));
            } else if lex_isname(self.ls.tok) && self.ls.peek() == Tok::Char(b'=') {
                self.expr_str_tok(&mut key);
                self.lex_check(Tok::Char(b'='));
                nhash += 1;
            } else {
                key = ExpDesc::init(VKNum, 0);
                key.nval = narr as f64;
                narr += 1;
                needarr = true;
                vcall = true;
            }
            self.expr(&mut val, 0);
            let mut nonconst = true;
            if key.isk() && key.k != VKNil && (key.k == VKStr || val.isk_nojump()) {
                if t.is_none() {
                    let kt = LuaTable::new(if needarr { narr } else { 0 }, hsize2hbits(nhash));
                    let fs = self.cur_mut();
                    let kidx = fs.kgc.len() as u32;
                    fs.kgc.push(KGc::Table(LuaTable::default()));
                    *self.ins_mut(pc) = bcins_ad(BCOp::TDUP, freg - 1, kidx);
                    t = Some((kidx, kt));
                }
                vcall = false;
                let kv = self.expr_kvalue(&key);
                let tab = &mut t.as_mut().unwrap().1;
                if val.isk_nojump() {
                    let mut vv = self.expr_kvalue(&val);
                    if key.k == VKStr && vv.is_nil() {
                        vv = LuaValue::table_marker();
                    }
                    tab.set(kv, vv);
                    nonconst = false;
                } else {
                    tab.set(kv, LuaValue::table_marker());
                }
            }
            if nonconst {
                if val.k != VCall {
                    self.expr_toanyreg(&mut val);
                    vcall = false;
                }
                if key.isk() {
                    self.expr_index(e, &mut key);
                }
                self.bcemit_store(&e.clone(), &mut val);
            }
            self.cur_mut().freereg = freg;
            if !self.lex_opt(Tok::Char(b',')) && !self.lex_opt(Tok::Char(b';')) {
                break;
            }
        }
        self.lex_match(Tok::Char(b'}'), Tok::Char(b'{'), line);
        if vcall {
            let fpc = self.cur().pc - 1;
            let ilp_ins = self.ins(fpc);
            debug_assert!(
                bc_a(ilp_ins) == freg
                    && bc_op(ilp_ins) == if narr > 256 { BCOp::TSETV } else { BCOp::TSETB }
            );
            let mut en = ExpDesc::init(VKNum, 0);
            en.nval = f64::from_bits((0x43300000u64 << 32) | (narr - 1) as u64);
            let mut fpc = fpc;
            if narr > 256 {
                self.cur_mut().pc -= 1;
                fpc -= 1;
            }
            let knum = self.const_num(&en);
            *self.ins_mut(fpc) = bcins_ad(BCOp::TSETM, freg, knum);
            setbc_b(self.ins_mut(fpc - 1), 0);
        }
        if pc == self.cur().pc - 1 {
            e.info = pc;
            self.cur_mut().freereg -= 1;
            e.k = VRelocable;
        } else {
            e.k = VNonReloc;
        }
        if let Some((kidx, mut kt)) = t {
            if needarr && kt.asize() < narr {
                kt.reasize(narr - 1);
            }
            self.cur_mut().kgc[kidx as usize] = KGc::Table(kt);
        } else {
            let mut narr = narr;
            if !needarr {
                narr = 0;
            } else if narr < 3 {
                narr = 3;
            } else if narr > 0x7ff {
                narr = 0x7ff;
            }
            setbc_d(self.ins_mut(pc), narr | (hsize2hbits(nhash) << 11));
        }
    }

    fn parse_params(&mut self, needself: bool, before: Tok, after: Tok) -> BCReg {
        let mut nparams: BCReg = 0;
        self.lex_check(before);
        if needself {
            self.var_new_lit(nparams, b"self");
            nparams += 1;
        }
        if self.ls.tok != after {
            loop {
                if lex_isname(self.ls.tok) {
                    let name = self.lex_str();
                    self.var_new(nparams, VName::Str(name));
                    nparams += 1;
                } else if self.ls.tok == Tok::Dots {
                    self.ls.next();
                    self.cur_mut().flags |= PROTO_VARARG;
                    break;
                } else {
                    self.err_syntax("<name> or '...' expected");
                }
                if !self.lex_opt(Tok::Char(b',')) {
                    break;
                }
            }
        }
        self.var_add(nparams);
        debug_assert!(self.cur().nactvar == nparams);
        self.bcreg_reserve(nparams);
        self.lex_check(after);
        nparams
    }

    fn proto_begin(&mut self, line: BCLine, nparams: BCReg) {
        let plen = self.fs.len();
        let (pbcbase, ppc) = {
            let pfs = &self.fs[plen - 2];
            (pfs.bcbase, pfs.pc)
        };
        let fs = self.cur_mut();
        fs.linedefined = line;
        fs.numparams = nparams as u8;
        fs.bcbase = pbcbase + ppc as usize;
        self.bcemit_ad(BCOp::FUNCF, 0, 0);
    }

    fn proto_finish(&mut self, e: &mut ExpDesc) {
        let flags = self.cur().flags & (PROTO_BITOP);
        let line = self.ls.linenumber;
        self.ls.lastline = line;
        let pt = self.fs_finish(line);
        let kidx = self.const_proto(pt);
        let pc = self.bcemit_ad(BCOp::FNEW, 0, kidx);
        *e = ExpDesc::init(VRelocable, pc);
        let pfs = self.cur_mut();
        pfs.flags |= flags;
        if (pfs.flags & PROTO_CHILD) == 0 {
            if (pfs.flags & PROTO_HAS_RETURN) != 0 {
                pfs.flags |= PROTO_FIXUP_RETURN;
            }
            pfs.flags |= PROTO_CHILD;
        }
    }

    fn parse_body(&mut self, e: &mut ExpDesc, needself: bool, line: BCLine) {
        self.fs_init();
        self.fscope_begin(0);
        let nparams = self.parse_params(needself, Tok::Char(b'('), Tok::Char(b')'));
        self.proto_begin(line, nparams);
        self.parse_chunk();
        if self.ls.tok != Tok::End {
            self.lex_match(Tok::End, Tok::Function, line);
        }
        self.proto_finish(e);
        self.ls.next();
    }

    fn parse_shortfunc(&mut self, e: &mut ExpDesc, name: Option<StrId>, eflags: u32, line: BCLine) {
        self.fs_init();
        self.fscope_begin(0);
        let mut nparams: BCReg = 0;
        if let Some(name) = name {
            self.var_new(nparams, VName::Str(name));
            nparams += 1;
            self.var_add(nparams);
            self.bcreg_reserve(1);
        } else if !self.lex_opt(Tok::OrOr) {
            nparams = self.parse_params(false, Tok::Char(b'|'), Tok::Char(b'|'));
        }
        self.lex_check(Tok::Arrow);
        self.proto_begin(line, nparams);
        if self.lex_opt(Tok::Do) {
            self.parse_chunk();
            if !self.lex_opt(Tok::End) {
                self.lex_match(Tok::End, Tok::Do, line);
            }
        } else {
            self.parse_return(eflags | EXPR_F_RET1);
        }
        self.proto_finish(e);
    }

    fn expr_list(&mut self, v: &mut ExpDesc) -> BCReg {
        let mut n = 1;
        self.expr(v, 0);
        while self.lex_opt(Tok::Char(b',')) {
            self.expr_tonextreg(v);
            self.expr(v, 0);
            n += 1;
        }
        n
    }

    fn parse_args(&mut self, e: &mut ExpDesc) {
        let mut args = ExpDesc::init(VVoid, 0);
        let line = self.ls.linenumber;
        if self.ls.tok == Tok::Char(b'(') {
            if line != self.ls.lastline {
                self.err_syntax("ambiguous syntax (function call x new statement)");
            }
            self.ls.next();
            if self.ls.tok == Tok::Char(b')') {
                args.k = VVoid;
            } else {
                self.expr_list(&mut args);
                if args.k == VCall {
                    let pc = args.info;
                    setbc_b(self.ins_mut(pc), 0);
                }
            }
            self.lex_match(Tok::Char(b')'), Tok::Char(b'('), line);
        } else if self.ls.tok == Tok::Char(b'{') {
            self.expr_table(&mut args);
        } else if self.ls.tok == Tok::Str {
            args = ExpDesc::init(VKStr, 0);
            args.sval = self.ls.tokval.str;
            self.ls.next();
        } else {
            self.err_syntax("function arguments expected");
        }
        debug_assert!(e.k == VNonReloc);
        let base = e.info;
        let ins;
        if args.k == VCall {
            ins = bcins_abc(BCOp::CALLM, base, 2, args.aux - base - 1 - self.fr2);
        } else {
            if args.k != VVoid {
                self.expr_tonextreg(&mut args);
            }
            ins = bcins_abc(BCOp::CALL, base, 2, self.cur().freereg - base - self.fr2);
        }
        *e = ExpDesc::init(VCall, self.bcemit(ins));
        e.aux = base;
        let pc = self.cur().pc - 1;
        let b = self.cur().bcbase;
        self.bcstack[b + pc as usize].line = line;
        self.cur_mut().freereg = base + 1;
    }

    fn expr_primary_nav(&mut self, v: &mut ExpDesc, eflags: u32) -> BCPos {
        let mut xpc = NO_JMP;
        if self.ls.tok == Tok::Char(b'(') {
            let line = self.ls.linenumber;
            self.ls.next();
            self.expr(v, 0);
            self.lex_match(Tok::Char(b')'), Tok::Char(b'('), line);
            self.expr_discharge(v);
        } else if lex_isname(self.ls.tok) {
            let line = self.ls.linenumber;
            let name = self.lex_str();
            if (eflags & EXPR_F_NORES) == 0 && self.ls.tok == Tok::Arrow {
                self.parse_shortfunc(v, Some(name), eflags, line);
                return xpc;
            }
            self.var_lookup(v, name);
        } else {
            self.err_syntax("unexpected symbol");
        }
        loop {
            let mut nav = false;
            if (eflags & EXPR_F_NONAV) == 0 && self.lex_opt(Tok::Nav) {
                nav = true;
                self.expr_toanyreg(v);
                self.bcemit(bcins_ad(BCOp::ISEQP, v.info, VKNil as u32));
                let j = self.bcemit_jmp();
                self.jmp_append(&mut xpc, j);
            }
            if self.ls.tok == Tok::Char(b'[') {
                let mut key = ExpDesc::init(VVoid, 0);
                self.expr_toanyreg(v);
                self.expr_bracket(&mut key);
                self.expr_index(v, &mut key);
            } else if self.ls.tok == Tok::Char(b':') {
                if (eflags & EXPR_F_NOCOLON) != 0 {
                    if nav {
                        self.err_syntax("unexpected symbol");
                    }
                    break;
                }
                self.ls.next();
                let mut key = ExpDesc::init(VVoid, 0);
                self.expr_str_tok(&mut key);
                self.bcemit_method(v, &key);
                nav = false;
                if self.lex_opt(Tok::Nav) {
                    nav = true;
                    self.bcemit(bcins_ad(BCOp::ISEQP, v.info, VKNil as u32));
                    let j = self.bcemit_jmp();
                    self.jmp_append(&mut xpc, j);
                }
                self.parse_args(v);
                if nav && (eflags & EXPR_F_NORES) == 0 && !self.suffix_follows() {
                    break;
                }
            } else if self.ls.tok == Tok::Char(b'(')
                || self.ls.tok == Tok::Str
                || self.ls.tok == Tok::Char(b'{')
            {
                self.expr_tonextreg(v);
                if self.fr2 != 0 {
                    self.bcreg_reserve(1);
                }
                self.parse_args(v);
                if nav && (eflags & EXPR_F_NORES) == 0 && !self.suffix_follows() {
                    break;
                }
            } else if nav || self.lex_opt(Tok::Char(b'.')) {
                self.expr_field(v);
            } else {
                break;
            }
            if nav && (eflags & EXPR_F_NORES) == 0 {
                self.expr_tonextreg(v);
            }
        }
        xpc
    }

    fn suffix_follows(&self) -> bool {
        matches!(
            self.ls.tok,
            Tok::Nav
                | Tok::Str
                | Tok::Char(b'[')
                | Tok::Char(b':')
                | Tok::Char(b'(')
                | Tok::Char(b'{')
                | Tok::Char(b'.')
        )
    }

    fn expr_primary(&mut self, v: &mut ExpDesc, eflags: u32) {
        let xpc = self.expr_primary_nav(v, eflags);
        if xpc != NO_JMP {
            let around = self.bcemit_jmp();
            self.jmp_tohere(xpc);
            if v.k == VCall {
                v.k = VCallNav;
                self.bcemit_ad(BCOp::KPRI, v.aux, VKNil as u32);
            } else {
                self.bcemit_ad(BCOp::KPRI, v.info, VKNil as u32);
            }
            self.jmp_tohere(around);
        }
    }

    fn expr_simple(&mut self, v: &mut ExpDesc, eflags: u32) {
        match self.ls.tok {
            Tok::Number => {
                *v = ExpDesc::init(VKNum, 0);
                v.nval = self.ls.tokval.num;
            }
            Tok::Str => {
                *v = ExpDesc::init(VKStr, 0);
                v.sval = self.ls.tokval.str;
            }
            Tok::Nil => {
                *v = ExpDesc::init(VKNil, 0);
            }
            Tok::True => {
                *v = ExpDesc::init(VKTrue, 0);
            }
            Tok::False => {
                *v = ExpDesc::init(VKFalse, 0);
            }
            Tok::Dots => {
                if (self.cur().flags & PROTO_VARARG) == 0 {
                    self.err_syntax("cannot use '...' outside a vararg function");
                }
                self.bcreg_reserve(1);
                let base = self.cur().freereg - 1;
                let numparams = self.cur().numparams as u32;
                let pc = self.bcemit_abc(BCOp::VARG, base, 2, numparams);
                *v = ExpDesc::init(VCall, pc);
                v.aux = base;
            }
            Tok::Char(b'{') => {
                self.expr_table(v);
                return;
            }
            Tok::Function => {
                self.ls.next();
                let line = self.ls.linenumber;
                self.parse_body(v, false, line);
                return;
            }
            Tok::Char(b'|') | Tok::OrOr => {
                let line = self.ls.linenumber;
                self.parse_shortfunc(v, None, eflags, line);
                return;
            }
            _ => {
                self.expr_primary(v, eflags);
                return;
            }
        }
        self.ls.next();
    }

    fn synlevel_begin(&mut self) {
        self.level += 1;
        if self.level >= LJ_MAX_XLEVEL {
            self.ls.error("chunk has too many syntax levels");
        }
    }

    fn synlevel_end(&mut self) {
        self.level -= 1;
    }

    fn expr_unop(&mut self, v: &mut ExpDesc, eflags: u32) {
        let op;
        if self.ls.tok == Tok::Not || self.ls.tok == Tok::Char(b'!') {
            op = BCOp::NOT;
        } else if self.ls.tok == Tok::Char(b'-') {
            op = BCOp::UNM;
        } else if self.ls.tok == Tok::Char(b'#') {
            op = BCOp::LEN;
        } else if self.ls.tok == Tok::Char(b'~') {
            op = BCOp::BNOT;
        } else {
            self.expr_simple(v, eflags);
            return;
        }
        self.ls.next();
        self.expr_binop(v, UNARY_PRIORITY, eflags);
        self.bcemit_unop(op, v);
    }

    fn expr_binop(&mut self, v: &mut ExpDesc, limit: u32, eflags: u32) -> BinOp {
        self.synlevel_begin();
        self.expr_unop(v, eflags);
        let mut opr = token2binop(self.ls.tok);
        while opr != BinOp::NoBinOp && PRIORITY[opr as usize].0 as u32 > limit {
            self.ls.next();
            self.bcemit_binop_left(opr, v);
            let mut v2 = ExpDesc::init(VVoid, 0);
            let nextop = self.expr_binop(&mut v2, PRIORITY[opr as usize].1 as u32, eflags);
            self.bcemit_binop(opr, v, &mut v2);
            opr = nextop;
        }
        self.synlevel_end();
        opr
    }

    fn expr(&mut self, v: &mut ExpDesc, eflags: u32) {
        self.expr_binop(v, 0, eflags);
        if self.lex_opt(Tok::Char(b'?')) {
            let mut escapelist = NO_JMP;
            self.bcemit_branch_t(v);
            let cond = v.f;
            self.expr(v, EXPR_F_NOCOLON);
            self.expr_tonextreg(v);
            let reg = v.info;
            let j = self.bcemit_jmp();
            self.jmp_append(&mut escapelist, j);
            self.jmp_tohere(cond);
            self.lex_check(Tok::Char(b':'));
            self.bcreg_free(reg);
            self.expr(v, 0);
            self.expr_tonextreg(v);
            self.jmp_tohere(escapelist);
        }
    }

    fn expr_next(&mut self) {
        let mut e = ExpDesc::init(VVoid, 0);
        self.expr(&mut e, 0);
        self.expr_tonextreg(&mut e);
    }

    fn expr_cond(&mut self) -> BCPos {
        let mut v = ExpDesc::init(VVoid, 0);
        self.expr(&mut v, 0);
        if v.k == VKNil {
            v.k = VKFalse;
        }
        self.bcemit_branch_t(&mut v);
        v.f
    }

    // -- Assignments ---------------------------------------------------------

    fn parse_compound(&mut self, e: &mut ExpDesc) -> bool {
        if !(e.k >= VLocal && e.k <= VIndexed) {
            return false;
        }
        let mut opr = token2binop(self.ls.tok);
        if opr > BinOp::Ne || opr == BinOp::Pow {
            return false;
        }
        self.var_assign_check(e);
        if opr == BinOp::Ne {
            if self.ls.tok != Tok::Ne {
                self.ls.err_near("'=' expected");
            }
            opr = BinOp::BXor;
        } else {
            if self.ls.c != b'=' as i32 {
                self.err_token(Tok::Char(b'='));
            }
            self.ls.next();
        }
        self.ls.next();
        let estore = *e;
        if e.k == VIndexed {
            let freg = self.cur().freereg;
            self.expr_discharge(e);
            self.cur_mut().freereg = freg;
        }
        if opr == BinOp::Concat {
            self.expr_tonextreg(e);
        } else {
            self.expr_toanyreg(e);
        }
        let mut v = ExpDesc::init(VVoid, 0);
        self.expr(&mut v, 0);
        self.bcemit_binop(opr, e, &mut v);
        self.bcemit_store(&estore, e);
        true
    }

    fn assign_hazard(&mut self, lhs: &mut [ExpDesc], v: &ExpDesc) {
        let reg = v.info;
        let tmp = self.cur().freereg;
        let mut hazard = false;
        for lh in lhs.iter_mut() {
            if lh.k == VIndexed {
                if lh.info == reg {
                    hazard = true;
                    lh.info = tmp;
                }
                if lh.aux == reg {
                    hazard = true;
                    lh.aux = tmp;
                }
            }
        }
        if hazard {
            self.bcemit_ad(BCOp::MOV, tmp, reg);
            self.bcreg_reserve(1);
        }
    }

    fn assign_adjust(&mut self, nvars: BCReg, nexps: BCReg, e: &mut ExpDesc) {
        let extra = nvars as i32 - nexps as i32;
        if e.k == VCall || e.k == VCallNav {
            let mut extra = extra + 1;
            if extra < 0 {
                extra = 0;
            }
            let pc = e.info;
            setbc_b(self.ins_mut(pc), (extra + 1) as u32);
            if extra > 1 {
                self.bcreg_reserve((extra - 1) as u32);
            }
            if e.k == VCallNav {
                let base = e.aux;
                debug_assert!({
                    let i0 = self.ins(pc);
                    let i1 = self.ins(pc + 1);
                    let i2 = self.ins(pc + 2);
                    (bc_op(i0) == BCOp::CALL || bc_op(i0) == BCOp::CALLM)
                        && bc_op(i1) == BCOp::JMP
                        && bc_op(i2) == BCOp::KPRI
                });
                let a = base + extra as u32;
                setbc_a(self.ins_mut(pc + 1), a);
                if extra > 1 {
                    *self.ins_mut(pc + 2) = bcins_ad(BCOp::KNIL, base, base + extra as u32 - 1);
                }
            }
        } else {
            if e.k != VVoid {
                self.expr_tonextreg(e);
            }
            if extra > 0 {
                let reg = self.cur().freereg;
                self.bcreg_reserve(extra as u32);
                self.bcemit_nil(reg, extra as u32);
            }
        }
        if nexps > nvars {
            self.cur_mut().freereg -= nexps - nvars;
        }
    }

    fn parse_assignment(&mut self, lhs: &mut Vec<ExpDesc>, nvars: BCReg) {
        let mut e = ExpDesc::init(VVoid, 0);
        {
            let v = lhs.last().unwrap();
            if !(v.k >= VLocal && v.k <= VIndexed) {
                self.err_syntax("syntax error");
            }
        }
        self.var_assign_check(lhs.last().unwrap());
        if self.lex_opt(Tok::Char(b',')) {
            let mut vl = ExpDesc::init(VVoid, 0);
            self.expr_primary(&mut vl, EXPR_F_NONAV);
            if vl.k == VLocal {
                self.assign_hazard(lhs, &vl);
            }
            self.checklimit(self.level + nvars, LJ_MAX_XLEVEL, "variable names");
            lhs.push(vl);
            self.parse_assignment(lhs, nvars + 1);
            lhs.pop();
        } else {
            self.lex_check(Tok::Char(b'='));
            let nexps = self.expr_list(&mut e);
            if nexps == nvars {
                if e.k == VCall {
                    if bc_op(self.ins(e.info)) == BCOp::VARG {
                        self.cur_mut().freereg -= 1;
                        e.k = VRelocable;
                    } else {
                        e.info = e.aux;
                        e.k = VNonReloc;
                    }
                }
                let var = *lhs.last().unwrap();
                self.bcemit_store(&var, &mut e);
                return;
            }
            self.assign_adjust(nvars, nexps, &mut e);
        }
        let freereg = self.cur().freereg;
        let mut e = ExpDesc::init(VNonReloc, freereg - 1);
        let var = *lhs.last().unwrap();
        self.bcemit_store(&var, &mut e);
    }

    fn parse_call_assign(&mut self) {
        let mut vl = ExpDesc::init(VVoid, 0);
        let xpc = self.expr_primary_nav(&mut vl, EXPR_F_NORES);
        if vl.k == VCall {
            let pc = vl.info;
            setbc_b(self.ins_mut(pc), 1);
        } else {
            debug_assert!(vl.k != VCallNav);
            if !(xpc == NO_JMP || self.ls.tok != Tok::Char(b',')) {
                self.err_syntax("syntax error");
            }
            if !self.parse_compound(&mut vl) {
                let mut lhs = vec![vl];
                self.parse_assignment(&mut lhs, 1);
            }
        }
        if xpc != NO_JMP {
            self.jmp_tohere(xpc);
        }
    }

    fn parse_local(&mut self, vinfo: u8) {
        self.ls.next();
        if self.lex_opt(Tok::Function) {
            let name = self.lex_str();
            let vidx = self.var_new(0, VName::Str(name));
            self.vstack[vidx].info = vinfo;
            let freereg = self.cur().freereg;
            let mut v = ExpDesc::init(VLocal, freereg);
            v.aux = self.cur().varmap[freereg as usize] as u32;
            self.bcreg_reserve(1);
            self.var_add(1);
            let mut b = ExpDesc::init(VVoid, 0);
            let line = self.ls.linenumber;
            self.parse_body(&mut b, false, line);
            self.expr_free(&b);
            self.expr_toreg(&mut b, v.info);
            let nactvar = self.cur().nactvar;
            let vidx2 = self.cur().varmap[(nactvar - 1) as usize];
            let pc = self.cur().pc;
            self.vstack[vidx2 as usize].startpc = pc;
        } else {
            let mut nvars: BCReg = 0;
            let mut e = ExpDesc::init(VVoid, 0);
            if vinfo != 0 {
                let vhsave = self.vhash;
                loop {
                    let name = self.lex_str();
                    let vidx = self.var_new(nvars, VName::Str(name));
                    nvars += 1;
                    let hash = self.var_hash(&self.vstack[vidx].name);
                    if let Some(h) = hash {
                        self.vstack[vidx].prev = self.vhash[h as usize];
                        self.vhash[h as usize] = vidx as u16;
                    }
                    self.vstack[vidx].info = vinfo;
                    if !self.lex_opt(Tok::Char(b',')) {
                        break;
                    }
                }
                self.vhash = vhsave;
            } else {
                loop {
                    let name = self.lex_str();
                    self.var_new(nvars, VName::Str(name));
                    nvars += 1;
                    if !self.lex_opt(Tok::Char(b',')) {
                        break;
                    }
                }
            }
            let nexps = if self.lex_opt(Tok::Char(b'=')) {
                self.expr_list(&mut e)
            } else {
                e.k = VVoid;
                0
            };
            self.assign_adjust(nvars, nexps, &mut e);
            self.var_add(nvars);
        }
    }

    fn parse_func(&mut self, line: BCLine) {
        let mut needself = false;
        self.ls.next();
        let mut v = ExpDesc::init(VVoid, 0);
        let name = self.lex_str();
        self.var_lookup(&mut v, name);
        while self.lex_opt(Tok::Char(b'.')) {
            self.expr_field(&mut v);
        }
        if self.lex_opt(Tok::Char(b':')) {
            needself = true;
            self.expr_field(&mut v);
        }
        self.var_assign_check(&v);
        let mut b = ExpDesc::init(VVoid, 0);
        self.parse_body(&mut b, needself, line);
        self.bcemit_store(&v, &mut b);
        let pc = self.cur().pc - 1;
        let bidx = self.cur().bcbase;
        self.bcstack[bidx + pc as usize].line = line;
    }

    // -- Control transfer statements -----------------------------------------

    fn parse_return(&mut self, eflags: u32) {
        let ins;
        self.cur_mut().flags |= PROTO_HAS_RETURN;
        if (eflags & EXPR_F_RET1) == 0
            && (parse_isend(self.ls.tok) || self.ls.tok == Tok::Char(b';'))
        {
            ins = bcins_ad(BCOp::RET0, 0, 1);
        } else {
            let mut e = ExpDesc::init(VVoid, 0);
            let nret;
            if (eflags & EXPR_F_RET1) != 0 {
                self.expr(&mut e, eflags);
                nret = 1;
            } else {
                nret = self.expr_list(&mut e);
            }
            if nret == 1 {
                if e.k == VCall && bc_op(self.ins(e.info)) != BCOp::VARG {
                    let ip = self.ins(e.info);
                    self.cur_mut().pc -= 1;
                    ins = bcins_ad(
                        bc_op(ip).offset(BCOp::CALLT as i32 - BCOp::CALL as i32),
                        bc_a(ip),
                        bc_c(ip),
                    );
                } else if e.k == VCall {
                    let pc = e.info;
                    setbc_b(self.ins_mut(pc), 0);
                    let nactvar = self.cur().nactvar;
                    ins = bcins_ad(BCOp::RETM, nactvar, e.aux - nactvar);
                } else {
                    let reg = self.expr_toanyreg(&mut e);
                    ins = bcins_ad(BCOp::RET1, reg, 2);
                }
            } else if e.k == VCall {
                let pc = e.info;
                setbc_b(self.ins_mut(pc), 0);
                let nactvar = self.cur().nactvar;
                ins = bcins_ad(BCOp::RETM, nactvar, e.aux - nactvar);
            } else {
                self.expr_tonextreg(&mut e);
                let nactvar = self.cur().nactvar;
                ins = bcins_ad(BCOp::RET, nactvar, nret + 1);
            }
        }
        if (self.cur().flags & PROTO_CHILD) != 0 {
            self.bcemit_aj(BCOp::UCLO, 0, 0);
        }
        self.bcemit(ins);
    }

    fn parse_break(&mut self) {
        let fs = self.cur_mut();
        fs.scopes.last_mut().unwrap().flags |= FSCOPE_BREAK;
        let pc = self.bcemit_jmp();
        self.gola_new(VName::Break, VSTACK_GOTO, pc);
    }

    fn parse_continue(&mut self) {
        let fs = self.cur_mut();
        fs.scopes.last_mut().unwrap().flags |= FSCOPE_CONT;
        let pc = self.bcemit_jmp();
        self.gola_new(VName::Cont, VSTACK_GOTO, pc);
    }

    fn parse_goto(&mut self) {
        let name = self.lex_str();
        if let Some(vl) = self.gola_findlabel(&VName::Str(name)) {
            let slot = self.vstack[vl].slot as u32;
            self.bcemit_aj(BCOp::LOOP, slot, -1);
        }
        let fs = self.cur_mut();
        fs.scopes.last_mut().unwrap().flags |= FSCOPE_GOLA;
        let pc = self.bcemit_jmp();
        self.gola_new(VName::Str(name), VSTACK_GOTO, pc);
    }

    fn parse_label(&mut self) {
        let pc = self.cur().pc;
        self.cur_mut().lasttarget = pc;
        {
            let fs = self.cur_mut();
            fs.scopes.last_mut().unwrap().flags |= FSCOPE_GOLA;
        }
        self.ls.next();
        let name = self.lex_str();
        if self.gola_findlabel(&VName::Str(name)).is_some() {
            self.ls.error(&format!(
                "duplicate label '{}'",
                String::from_utf8_lossy(self.ls.strs.get(name))
            ));
        }
        let pc = self.cur().pc;
        let idx = self.gola_new(VName::Str(name), VSTACK_LABEL, pc);
        self.lex_check(Tok::Label);
        loop {
            if self.ls.tok == Tok::Label {
                self.synlevel_begin();
                self.parse_label();
                self.synlevel_end();
            } else {
                break;
            }
        }
        if parse_isend(self.ls.tok) && self.ls.tok != Tok::Until {
            let nactvar = self.cur().scopes.last().unwrap().nactvar;
            self.vstack[idx].slot = nactvar;
        }
        let vstart = self.cur().scopes.last().unwrap().vstart;
        self.gola_resolve(vstart, idx);
    }

    // -- Blocks, loops and conditional statements ------------------------------

    fn parse_block(&mut self) {
        self.fscope_begin(0);
        self.parse_chunk();
        self.fscope_end();
    }

    fn parse_while(&mut self, line: BCLine) {
        self.ls.next();
        let start = self.cur().pc;
        self.cur_mut().lasttarget = start;
        let condexit = self.expr_cond();
        self.fscope_begin(FSCOPE_LOOP);
        self.lex_check(Tok::Do);
        let nactvar = self.cur().nactvar;
        let loop_pc = self.bcemit_ad(BCOp::LOOP, nactvar, 0);
        self.parse_block();
        let j = self.bcemit_jmp();
        self.jmp_patch(j, start);
        self.lex_match(Tok::End, Tok::While, line);
        self.fscope_continue(start);
        self.fscope_end();
        self.jmp_tohere(condexit);
        let pc = self.cur().pc;
        self.jmp_patchins(loop_pc, pc);
    }

    fn parse_repeat(&mut self, line: BCLine) {
        let loop_start = self.cur().pc;
        self.cur_mut().lasttarget = loop_start;
        self.fscope_begin(FSCOPE_LOOP);
        self.fscope_begin(0);
        self.ls.next();
        let nactvar = self.cur().nactvar;
        self.bcemit_ad(BCOp::LOOP, nactvar, 0);
        self.parse_chunk();
        self.lex_match(Tok::Until, Tok::Repeat, line);
        let pc = self.cur().pc;
        self.fscope_continue(pc);
        let mut condexit = self.expr_cond();
        let inner_upval = {
            let fs = self.cur();
            (fs.scopes.last().unwrap().flags & FSCOPE_UPVAL) != 0
        };
        if !inner_upval {
            self.fscope_end();
        } else {
            self.parse_break();
            self.jmp_tohere(condexit);
            self.fscope_end();
            condexit = self.bcemit_jmp();
        }
        self.jmp_patch(condexit, loop_start);
        let pc = self.cur().pc;
        self.jmp_patchins(loop_start, pc);
        self.fscope_end();
    }

    fn parse_for_num(&mut self, varname: StrId, line: BCLine) {
        let base = self.cur().freereg;
        self.var_new(FORL_IDX, VName::Fixed(1));
        self.var_new(FORL_STOP, VName::Fixed(2));
        self.var_new(FORL_STEP, VName::Fixed(3));
        self.var_new(FORL_EXT, VName::Str(varname));
        self.lex_check(Tok::Char(b'='));
        self.expr_next();
        self.lex_check(Tok::Char(b','));
        self.expr_next();
        if self.lex_opt(Tok::Char(b',')) {
            self.expr_next();
        } else {
            let freereg = self.cur().freereg;
            self.bcemit_ad(BCOp::KSHORT, freereg, 1);
            self.bcreg_reserve(1);
        }
        self.var_add(3);
        self.lex_check(Tok::Do);
        let loop_pc = self.bcemit_aj(BCOp::FORI, base, -1);
        self.fscope_begin(0);
        self.var_add(1);
        self.bcreg_reserve(1);
        self.parse_block();
        self.fscope_end();
        let pc = self.cur().pc;
        self.fscope_continue(pc);
        let loopend = self.bcemit_aj(BCOp::FORL, base, -1);
        let b = self.cur().bcbase;
        self.bcstack[b + loopend as usize].line = line;
        self.jmp_patchins(loopend, loop_pc + 1);
        let pc = self.cur().pc;
        self.jmp_patchins(loop_pc, pc);
    }

    fn predict_next(&mut self, pc: BCPos) -> bool {
        let ins = self.ins(pc);
        let name: Option<Vec<u8>> = match bc_op(ins) {
            BCOp::MOV => {
                if bc_d(ins) >= self.cur().nactvar {
                    return false;
                }
                let vidx = self.cur().varmap[bc_d(ins) as usize];
                match &self.vstack[vidx as usize].name {
                    VName::Str(sid) => Some(self.ls.strs.get(*sid).to_vec()),
                    _ => None,
                }
            }
            BCOp::UGET => {
                let vidx = self.cur().uvmap[bc_d(ins) as usize];
                match &self.vstack[vidx as usize].name {
                    VName::Str(sid) => Some(self.ls.strs.get(*sid).to_vec()),
                    _ => None,
                }
            }
            BCOp::GGET => {
                let pairs = self.ls.strs.intern(b"pairs");
                if let Some(&slot) = self.cur().kgc_map.get(&pairs) {
                    if slot == bc_d(ins) {
                        return true;
                    }
                }
                let next = self.ls.strs.intern(b"next");
                if let Some(&slot) = self.cur().kgc_map.get(&next) {
                    if slot == bc_d(ins) {
                        return true;
                    }
                }
                return false;
            }
            _ => return false,
        };
        match name {
            Some(n) => n == b"pairs" || n == b"next",
            None => false,
        }
    }

    fn parse_for_iter(&mut self, indexname: StrId) {
        let mut nvars: BCReg = 0;
        let base = self.cur().freereg + 3;
        let exprpc = self.cur().pc;
        self.var_new(nvars, VName::Fixed(4));
        nvars += 1;
        self.var_new(nvars, VName::Fixed(5));
        nvars += 1;
        self.var_new(nvars, VName::Fixed(6));
        nvars += 1;
        self.var_new(nvars, VName::Str(indexname));
        nvars += 1;
        while self.lex_opt(Tok::Char(b',')) {
            let name = self.lex_str();
            self.var_new(nvars, VName::Str(name));
            nvars += 1;
        }
        self.lex_check(Tok::In);
        let line = self.ls.linenumber;
        let mut e = ExpDesc::init(VVoid, 0);
        let nexps = self.expr_list(&mut e);
        self.assign_adjust(3, nexps, &mut e);
        self.bcreg_bump(3 + self.fr2);
        let isnext = nvars <= 5 && self.cur().pc > exprpc && self.predict_next(exprpc);
        self.var_add(3);
        self.lex_check(Tok::Do);
        let loop_pc = self.bcemit_aj(if isnext { BCOp::ISNEXT } else { BCOp::JMP }, base, -1);
        self.fscope_begin(0);
        self.var_add(nvars - 3);
        self.bcreg_reserve(nvars - 3);
        self.parse_block();
        self.fscope_end();
        let pc = self.cur().pc;
        self.jmp_patchins(loop_pc, pc);
        self.fscope_continue(pc);
        self.bcemit_abc(
            if isnext { BCOp::ITERN } else { BCOp::ITERC },
            base,
            nvars - 3 + 1,
            2 + 1,
        );
        let loopend = self.bcemit_aj(BCOp::ITERL, base, -1);
        let b = self.cur().bcbase;
        self.bcstack[b + (loopend - 1) as usize].line = line;
        self.bcstack[b + loopend as usize].line = line;
        self.jmp_patchins(loopend, loop_pc + 1);
    }

    fn parse_for(&mut self, line: BCLine) {
        self.fscope_begin(FSCOPE_LOOP);
        self.ls.next();
        let varname = self.lex_str();
        if self.ls.tok == Tok::Char(b'=') {
            self.parse_for_num(varname, line);
        } else if self.ls.tok == Tok::Char(b',') || self.ls.tok == Tok::In {
            self.parse_for_iter(varname);
        } else {
            self.err_syntax("'=' or 'in' expected");
        }
        self.lex_match(Tok::End, Tok::For, line);
        self.fscope_end();
    }

    fn parse_then(&mut self) -> BCPos {
        self.ls.next();
        let condexit = self.expr_cond();
        self.lex_check(Tok::Then);
        self.parse_block();
        condexit
    }

    fn parse_if(&mut self, line: BCLine) {
        let mut escapelist = NO_JMP;
        let mut flist = self.parse_then();
        while self.ls.tok == Tok::Elseif {
            let j = self.bcemit_jmp();
            self.jmp_append(&mut escapelist, j);
            self.jmp_tohere(flist);
            flist = self.parse_then();
        }
        if self.ls.tok == Tok::Else {
            let j = self.bcemit_jmp();
            self.jmp_append(&mut escapelist, j);
            self.jmp_tohere(flist);
            self.ls.next();
            self.parse_block();
        } else {
            self.jmp_append(&mut escapelist, flist);
        }
        self.jmp_tohere(escapelist);
        self.lex_match(Tok::End, Tok::If, line);
    }

    // -- Parse statements ------------------------------------------------------

    fn parse_stmt(&mut self) -> bool {
        let line = self.ls.linenumber;
        match self.ls.tok {
            Tok::If => {
                self.parse_if(line);
            }
            Tok::While => {
                self.parse_while(line);
            }
            Tok::Do => {
                self.ls.next();
                self.parse_block();
                self.lex_match(Tok::End, Tok::Do, line);
            }
            Tok::For => {
                self.parse_for(line);
            }
            Tok::Repeat => {
                self.parse_repeat(line);
            }
            Tok::Function => {
                self.parse_func(line);
            }
            Tok::Local => {
                self.parse_local(0);
            }
            Tok::Const => {
                let tokx = self.ls.peek();
                if !(lex_isname(tokx) || tokx == Tok::Function) {
                    self.parse_call_assign();
                } else {
                    self.parse_local(VSTACK_CONST);
                }
            }
            Tok::Return => {
                self.ls.next();
                self.parse_return(0);
                return true;
            }
            Tok::Break => {
                self.ls.next();
                self.parse_break();
                return true;
            }
            Tok::Continue => {
                if !parse_isend(self.ls.peek()) {
                    self.parse_call_assign();
                    return false;
                }
                self.ls.next();
                self.parse_continue();
                return true;
            }
            Tok::Label => {
                self.parse_label();
            }
            Tok::Goto => {
                if lex_isname(self.ls.peek()) {
                    self.ls.next();
                    self.parse_goto();
                } else {
                    self.parse_call_assign();
                }
            }
            _ => {
                self.parse_call_assign();
            }
        }
        false
    }

    fn parse_chunk(&mut self) {
        let mut islast = false;
        self.synlevel_begin();
        while !islast && !parse_isend(self.ls.tok) {
            islast = self.parse_stmt();
            self.lex_opt(Tok::Char(b';'));
            debug_assert!(
                self.cur().framesize as u32 >= self.cur().freereg
                    && self.cur().freereg >= self.cur().nactvar
            );
            let nactvar = self.cur().nactvar;
            self.cur_mut().freereg = nactvar;
        }
        self.synlevel_end();
    }

    pub fn parse(mut self) -> (Proto, Interner) {
        self.level = 0;
        self.fs_init();
        self.cur_mut().linedefined = 0;
        self.cur_mut().numparams = 0;
        self.cur_mut().bcbase = 0;
        self.cur_mut().flags |= PROTO_VARARG;
        self.fscope_begin(0);
        self.bcemit_ad(BCOp::FUNCV, 0, 0);
        self.ls.next();
        self.parse_chunk();
        if self.ls.tok != Tok::Eof {
            self.err_token(Tok::Eof);
        }
        let line = self.ls.linenumber;
        let pt = self.fs_finish(line);
        debug_assert!(self.fs.is_empty());
        debug_assert!(pt.uv.is_empty());
        (pt, self.ls.strs)
    }
}

pub fn fixed_varname(n: u8) -> &'static str {
    match n {
        1 => "(for index)",
        2 => "(for limit)",
        3 => "(for step)",
        4 => "(for generator)",
        5 => "(for state)",
        6 => "(for control)",
        _ => "?",
    }
}
