//! JIT engine state and hot-path detection.
//!
//! Ported from lj_jit.h (engine flags, parameters, penalty cache) and
//! lj_dispatch.h/c (hotcount table). The compiler pipeline so far:
//!
//! * hot-path detection: hotcount table + penalties + blacklisting,
//! * `ir`: the SSA IR format and buffer (lj_ir.h/c),
//! * `opt_fold`: FOLD/CSE engine (lj_opt_fold.c subset),
//! * `opt_dce`: dead code elimination (lj_opt_dce.c),
//! * `opt_loop`: loop unrolling via copy-substitution + PHIs (lj_opt_loop.c),
//! * `record`: the bytecode recorder for numeric single-frame traces
//!   (lj_record.c + lj_snap.c subsets),
//! * `trace`: the trace compiler state machine (lj_trace.c),
//! * `asm/`: native code generation backends (x86-64, ARM64) behind a
//!   unified API; traces they cannot handle fall back to the portable
//!   IR executor in `exec`.
//!
//! Differences from LuaJIT, by design:
//! * The hotcount table lives in `JitState` (LuaJIT puts it in `GG_State`
//!   so the assembler VM can reach it PC-relative; we have no such need).
//! * Hotcounts are hashed from the *address* of the bytecode instruction,
//!   like LuaJIT: `(addr >> 2) & (HOTCOUNT_SIZE-1)`. Proto bytecode lives
//!   in a `Vec` that is never resized after parsing, so the addresses are
//!   stable.
//! * Instead of switching dispatch tables, the interpreter checks
//!   `JIT_F_ON` on the counting opcodes; blacklisting still patches the
//!   bytecode to the non-counting I* variants, which removes the check
//!   from the hot path for good.

pub mod asm;
pub mod exec;
pub mod ir;
pub mod mcode;
pub mod opt_dce;
pub mod opt_fold;
pub mod opt_loop;
pub mod record;
pub mod trace;

use crate::bc::BCIns;
use crate::gc::GcPtr;
use crate::proto::Proto;

/// Type of hot counters (lj_dispatch.h). 16 bits: only ~0.0015% overhead
/// with the maximum slot penalty.
pub type HotCount = u16;

/// Number of hot counter hash slots. Must be a power of two.
pub const HOTCOUNT_SIZE: usize = 64;
/// Decrement per loop iteration (FORL/ITERL/LOOP).
pub const HOTCOUNT_LOOP: HotCount = 2;
/// Decrement per function call (FUNCF/FUNCV headers).
pub const HOTCOUNT_CALL: HotCount = 1;

/// Trace number. 0 = no trace.
pub type TraceNo = u32;

// -- JIT engine flags (lj_jit.h) ------------------------------------------

pub const JIT_F_ON: u32 = 0x0000_0001;

// -- JIT engine parameters (JIT_PARAMDEF) ---------------------------------

/// Optimization parameters, same order and defaults as lj_jit.h.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum JitParam {
    MaxTrace,   // Max. # of traces in cache.
    MaxRecord,  // Max. # of recorded IR instructions.
    MaxIrConst, // Max. # of IR constants of a trace.
    MaxSide,    // Max. # of side traces of a root trace.
    MaxSnap,    // Max. # of snapshots for a trace.
    MinStitch,  // Min. # of IR ins for a stitched trace.
    HotLoop,    // # of iterations to detect a hot loop/call.
    HotExit,    // # of taken exits to start a side trace.
    TrySide,    // # of attempts to compile a side trace.
    InstUnroll, // Max. unroll for instable loops.
    LoopUnroll, // Max. unroll for loop ops in side traces.
    CallUnroll, // Max. unroll for recursive calls.
    RecUnroll,  // Min. unroll for true recursion.
    SizeMcode,  // Size of each machine code area (KB).
    MaxMcode,   // Max. total size of all machine code areas (KB).
}
pub const JIT_P_MAX: usize = 15;

pub const JIT_PARAM_DEFAULT: [i32; JIT_P_MAX] = [
    1000, // maxtrace
    4000, // maxrecord
    500,  // maxirconst
    100,  // maxside
    500,  // maxsnap
    10,   // minstitch
    56,   // hotloop
    10,   // hotexit
    4,    // tryside
    4,    // instunroll
    15,   // loopunroll
    3,    // callunroll
    2,    // recunroll
    64,   // sizemcode
    2048, // maxmcode
];

// -- Trace compiler state (lj_jit.h) ---------------------------------------

/// Trace compiler state machine (`TraceState`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TraceState {
    Idle,      // Trace compiler idle.
    Record,    // Bytecode recording active.
    Record1st, // Record 1st instruction, too.
    Start,     // New trace started.
    End,       // End of trace.
    Asm,       // Assemble trace.
    Err,       // Trace aborted with error.
}

/// Trace linking modes (`TraceLink`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TraceLink {
    None,    // Incomplete trace. No link yet.
    Root,    // Link to other root trace.
    Loop,    // Loop to same trace.
    Tailrec, // Tail-recursion.
    Uprec,   // Up-recursion.
    Downrec, // Down-recursion.
    Interp,  // Fallback to interpreter.
    Return,  // Return to interpreter.
    Stitch,  // Trace stitching.
}

/// Snapshot of the host stack at a guard (`SnapShot`).
///
/// Deviation from LuaJIT: the snapshot PC (a bytecode index into
/// `startpt.bc`) lives in the struct instead of being packed into the tail
/// of the snap map.
#[derive(Clone, Copy, Debug)]
pub struct SnapShot {
    /// Offset into the snap map.
    pub mapofs: u32,
    /// First IR ref for this snapshot (guards after this use it).
    pub iref: ir::IRRef,
    /// Bytecode index to resume at (relative to `startpt.bc`).
    pub pc: u32,
    /// Side trace linked to this exit (0 = none). Stands in for LuaJIT's
    /// `lj_asm_patchexit` mcode patching: the executor follows it.
    pub sidetrace: TraceNo,
    /// Recorder base slot at the snapshot: 2 = the entry frame, higher
    /// values lie inside inlined call frames (the exit must re-enter the
    /// innermost frame before resuming).
    pub baseslot: u8,
    /// Number of valid slots.
    pub nslots: u8,
    /// Maximum frame extent.
    pub topslot: u8,
    /// Number of compressed entries.
    pub nent: u8,
    /// Count of taken exits for this snapshot.
    pub count: u8,
}

pub const SNAPCOUNT_DONE: u8 = 0xff;

/// Compressed snapshot entry: `(slot << 24) + flags + ref`.
pub type SnapEntry = u32;

pub const SNAP_FRAME: u32 = 0x01_0000;
pub const SNAP_CONT: u32 = 0x02_0000;
pub const SNAP_NORESTORE: u32 = 0x04_0000;
pub const SNAP_KEYINDEX: u32 = 0x10_0000;

#[inline]
pub fn snap_entry(slot: u32, tr: ir::TRef) -> SnapEntry {
    (slot << 24) + (tr & (ir::TREF_KEYINDEX | ir::TREF_CONT | ir::TREF_FRAME | ir::TREF_REFMASK))
}
#[inline]
pub fn snap_slot(sn: SnapEntry) -> u32 {
    sn >> 24
}
#[inline]
pub fn snap_ref(sn: SnapEntry) -> ir::IRRef {
    sn & 0xffff
}

/// A completed (or in-progress) trace: LuaJIT's `GCtrace`.
pub struct GCtrace {
    pub traceno: TraceNo,
    pub ir: ir::IrBuf,
    pub snap: Vec<SnapShot>,
    pub snapmap: Vec<SnapEntry>,
    /// Assembled machine code (None: run on the portable IR executor).
    pub mcode: Option<mcode::McodeArea>,
    /// Starting prototype and bytecode index.
    pub startpt: GcPtr<Proto>,
    pub startpc: usize,
    /// Original bytecode at the trace start (before any patching).
    pub startins: BCIns,
    /// Linked trace (self = loop, 0 = none/blacklisted).
    pub link: TraceNo,
    pub linktype: TraceLink,
    /// Root trace of a side trace (0 for root traces).
    pub root: TraceNo,
    /// Number of child traces (root trace only).
    pub nchild: u16,
    /// Side traces: pairs of (own inherited-SLOAD ref, parent snapshot
    /// ref). The machine-code prelude at `inner_ofs` copies the own env
    /// slots from the parent's, so a linked exit hands its values over
    /// without a Lua-stack round trip.
    pub parentmap: Vec<(ir::IRRef1, ir::IRRef1)>,
    /// Machine-code offset of the inner entry (after the outer-frame
    /// prologue). Patched exits and tail links jump here, staying inside
    /// the frame set up by the first trace of the chain.
    pub inner_ofs: u32,
    /// Machine-code offsets of the patchable exit-stub tails, per
    /// snapshot: (snapshot index, code offset). `lj_asm_patchexit`
    /// rewrites these to jump straight into a compiled side trace.
    pub stub_tails: Vec<(u32, u32)>,
}

/// Trace compiler error reasons (lj_traceerr.h).
macro_rules! trace_errors {
    ($(($name:ident, $msg:literal),)*) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        #[repr(u16)]
        pub enum TraceError {
            $($name,)*
        }
        impl TraceError {
            pub fn message(self) -> &'static str {
                match self {
                    $(TraceError::$name => $msg,)*
                }
            }
        }
    };
}

trace_errors! {
    // Recording.
    (RECERR, "error thrown or hook called during recording"),
    (TRACEUV, "trace too short"),
    (TRACEOV, "trace too long"),
    (STACKOV, "trace too deep"),
    (SNAPOV, "too many snapshots"),
    (BLACKL, "blacklisted"),
    (RETRY, "retry recording"),
    (NYIBC, "NYI: bytecode"),
    // Recording loop ops.
    (LLEAVE, "leaving loop in root trace"),
    (LINNER, "inner loop in root trace"),
    (LUNROLL, "loop unroll limit reached"),
    // Recording calls/returns.
    (BADTYPE, "bad argument type"),
    (CJITOFF, "JIT compilation disabled for function"),
    (CUNROLL, "call unroll limit reached"),
    (DOWNREC, "down-recursion, restarting"),
    (NYIFFU, "NYI: unsupported variant of FastFunc"),
    (NYIRETL, "NYI: return to lower frame"),
    // Recording indexed load/store.
    (STORENN, "store with nil or NaN key"),
    (NOMM, "missing metamethod"),
    (IDXLOOP, "looping index lookup"),
    (NYITMIX, "NYI: mixed sparse/dense table"),
    // Optimizations.
    (GFAIL, "guard would always fail"),
    (PHIOV, "too many PHIs"),
    (TYPEINS, "persistent type instability"),
    // Assembler.
    (MCODEAL, "failed to allocate mcode memory"),
    (MCODEOV, "machine code too long"),
    (MCODELM, "hit mcode limit (retrying)"),
    (SPILLOV, "too many spill slots"),
    (BADRA, "inconsistent register allocation"),
    (NYIIR, "NYI: cannot assemble IR instruction"),
    (NYIPHI, "NYI: PHI shuffling too complex"),
    (NYICOAL, "NYI: register coalescing too complex"),
}

// -- Penalty cache (lj_jit.h) ----------------------------------------------

/// Round-robin penalty cache for bytecodes leading to aborted traces.
#[derive(Clone, Copy)]
pub struct HotPenalty {
    /// Address of the starting bytecode instruction (hash key only, never
    /// dereferenced; stale entries after a GC merely perturb the heuristic,
    /// as in LuaJIT).
    pub pc: usize,
    /// Penalty value, i.e. the hotcount start.
    pub val: u16,
    /// Abort reason.
    pub reason: TraceError,
}

impl Default for HotPenalty {
    fn default() -> HotPenalty {
        HotPenalty {
            pc: 0,
            val: 0,
            reason: TraceError::RECERR,
        }
    }
}

pub const PENALTY_SLOTS: usize = 64; // Penalty cache slots. Power of two.
pub const PENALTY_MIN: u32 = 36 * 2; // Minimum penalty value.
pub const PENALTY_MAX: u32 = 6000; // Maximum penalty value.
pub const PENALTY_RNDBITS: u32 = 4; // # of random bits added to the penalty.

/// Small PRNG for penalty randomization (stands in for lj_prng's TW223;
/// only the low `PENALTY_RNDBITS` bits are consumed).
pub struct Prng(u64);

impl Prng {
    fn new() -> Prng {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Prng(seed | 1)
    }

    /// xorshift64* — plenty for penalty noise.
    pub fn u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// -- JIT compiler state (jit_State) ----------------------------------------

/// The JIT compiler state, hanging off the `GlobalState` (LuaJIT embeds it
/// in `GG_State`): profiling counters, the trace registry, penalties and
/// the active recording context.
pub struct JitState {
    /// JIT_F_* flags.
    pub flags: u32,
    /// Trace compiler state machine.
    pub state: TraceState,
    /// Parent trace of the side trace being recorded (0 = root trace).
    pub parent: TraceNo,
    /// Exit number in the parent trace.
    pub exitno: u32,
    /// Prototype holding the starting bytecode of the current trace.
    pub startpt: Option<GcPtr<Proto>>,
    /// Index of the starting instruction in `startpt.bc`.
    pub startpc: usize,
    /// Copy of the starting instruction.
    pub startins: BCIns,
    /// Pending abort reason (valid in `TraceState::Err`).
    pub err: TraceError,
    /// The active recording context (LuaJIT keeps these buffers inline in
    /// `jit_State`; boxing keeps the idle footprint small).
    pub rec: Option<Box<record::Record>>,
    /// Completed traces, indexed by trace number (slot 0 unused).
    pub trace: Vec<Option<Box<GCtrace>>>,
    /// Scratch value environment of the trace executors.
    pub exec_env: Vec<u64>,
    /// Required env size: the maximum instruction count over all stored
    /// traces. Machine-code chains switch traces without returning to
    /// Rust, so the buffer must cover the whole tree up front.
    pub env_need: usize,
    /// Engine parameters (JIT_P_*).
    pub param: [i32; JIT_P_MAX],
    /// Hot counter hash table (GG_State.hotcount).
    pub hotcount: [HotCount; HOTCOUNT_SIZE],
    /// Penalty slots.
    pub penalty: [HotPenalty; PENALTY_SLOTS],
    /// Round-robin index into the penalty slots.
    pub penaltyslot: u32,
    /// PRNG state for penalty randomization.
    pub prng: Prng,
    /// Target architecture for native code generation (cached from env/default).
    pub arch: self::asm::Arch,
    /// Skip native code generation (LUAJIT_RS_NOASM), cached at startup.
    pub no_asm: bool,
    /// Trace dump enabled (LUAJIT_RS_TRDUMP), cached at startup.
    pub trace_dump: bool,
    /// Trace dump level 2 (LUAJIT_RS_TRDUMP=2), cached at startup.
    pub trace_dump2: bool,
}

impl JitState {
    pub fn new() -> JitState {
        let flags = JIT_F_ON;
        let mut js = JitState {
            flags,
            state: TraceState::Idle,
            parent: 0,
            exitno: 0,
            startpt: None,
            startpc: 0,
            startins: 0,
            err: TraceError::RECERR,
            rec: None,
            trace: vec![None],
            exec_env: Vec::new(),
            env_need: 0,
            param: JIT_PARAM_DEFAULT,
            hotcount: [0; HOTCOUNT_SIZE],
            penalty: [HotPenalty::default(); PENALTY_SLOTS],
            penaltyslot: 0,
            prng: Prng::new(),
            arch: {
                let over = std::env::var("LUAJIT_RS_JIT_ARCH").unwrap_or_default();
                if over.eq_ignore_ascii_case("arm64") || over.eq_ignore_ascii_case("aarch64") {
                    self::asm::Arch::Arm64
                } else if over.eq_ignore_ascii_case("x64") || over.eq_ignore_ascii_case("x86_64") {
                    self::asm::Arch::X64
                } else {
                    self::asm::HOST_ARCH
                }
            },
            no_asm: std::env::var("LUAJIT_RS_NOASM").is_ok() || self::asm::HOST_ARCH == self::asm::Arch::Arm64,
            trace_dump: std::env::var("LUAJIT_RS_TRDUMP").is_ok(),
            trace_dump2: std::env::var("LUAJIT_RS_TRDUMP").as_deref() == Ok("2"),
        };
        js.init_hotcount();
        js
    }

    /// Find an existing root trace starting at `pt.bc[pc]`. Until the
    /// backend patches JLOOP/JFORL into the bytecode, this lookup is how
    /// re-triggered hot loops discover they are already compiled.
    pub fn find_root_trace(&self, pt: GcPtr<Proto>, pc: usize) -> Option<TraceNo> {
        let key = bc_addr(pt, pc);
        self.trace.iter().flatten().find_map(|t| {
            (t.root == 0 && bc_addr(t.startpt, t.startpc) == key).then_some(t.traceno)
        })
    }

    #[inline(always)]
    pub fn is_on(&self) -> bool {
        self.flags & JIT_F_ON != 0
    }

    /// `jit.on()` / `jit.off()`. Re-arms the hot counters on enable, like
    /// `lj_dispatch_update` -> `lj_dispatch_init_hotcount`.
    pub fn set_on(&mut self, on: bool) {
        if on {
            self.flags |= JIT_F_ON;
            self.init_hotcount();
        } else {
            self.flags &= !JIT_F_ON;
        }
    }

    #[inline(always)]
    pub fn param(&self, p: JitParam) -> i32 {
        self.param[p as usize]
    }

    /// `lj_dispatch_init_hotcount`: reset all counters to
    /// `hotloop * HOTCOUNT_LOOP - 1`.
    pub fn init_hotcount(&mut self) {
        let start = (self.param(JitParam::HotLoop) as u32 * HOTCOUNT_LOOP as u32 - 1) as HotCount;
        self.hotcount = [start; HOTCOUNT_SIZE];
    }

    /// `hotcount_get/set` hash: instruction address to slot index.
    #[inline(always)]
    fn hotcount_slot(addr: usize) -> usize {
        (addr >> 2) & (HOTCOUNT_SIZE - 1)
    }

    #[inline(always)]
    pub fn hotcount_set(&mut self, addr: usize, val: HotCount) {
        self.hotcount[Self::hotcount_slot(addr)] = val;
    }

    /// The `hotloop`/`hotcall` macro from the dasc VMs: `sub word [slot], N;
    /// jb ->vm_hot*`. Returns true when the counter underflows, i.e. the
    /// path just turned hot. `addr` is the interpreter PC *after* fetching
    /// the counting instruction, exactly like LuaJIT's offset-by-1 PC.
    #[inline(always)]
    pub fn hot_decrement(&mut self, addr: usize, amount: HotCount) -> bool {
        let slot = Self::hotcount_slot(addr);
        let (nv, underflow) = self.hotcount[slot].overflowing_sub(amount);
        self.hotcount[slot] = nv;
        underflow
    }
}

impl Default for JitState {
    fn default() -> JitState {
        JitState::new()
    }
}

/// Byte address of `pt.bc[idx]`, the hotcount/penalty hash key. Address
/// arithmetic only — never dereferenced here.
#[inline(always)]
pub fn bc_addr(pt: GcPtr<Proto>, idx: usize) -> usize {
    pt.as_ref().bc.as_ptr() as usize + idx * std::mem::size_of::<BCIns>()
}
