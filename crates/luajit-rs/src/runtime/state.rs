use std::ptr::NonNull;

use crate::compiler::lex::CompileError;
use crate::compiler::parse::Parser;
use crate::ffi::CTState;
use crate::func::{CClosure, CFunction, GcFunc, LuaClosure};
use crate::gc::{GcObjectKind, GcPtr, Pool};
use crate::jit::JitState;
use crate::proto::KGc;
use crate::proto::Proto;
use crate::runtime::cdata::CData;
use crate::runtime::func::Upval;
use crate::runtime::gc::{self, GcState, Gray};
use crate::runtime::userdata::GcUserData;
use crate::stdlib::PlatformInstant;
use crate::string::{Interner, StrId};
use crate::table::LuaTable;
use crate::value::{GcRef, LJ_TFUNC, LJ_TTAB, LuaValue};
use crate::vm::FRAME_TYPE_MASK;
use crate::{LuaError, meta};

/// The GC heap: stable-address object pools.
///
/// Every collectable type lives in its own `Pool`, which allocates objects in
/// fixed pages so their addresses never move (a `LuaValue` stores the raw
/// pointer in its 47-bit payload). The collector (`gc::full_gc`) marks from
/// the roots and sweeps these pools. `total`/`threshold` drive the trigger,
/// like LuaJIT's `gc.total`/`gc.threshold`.
///
/// `repr(C)` is required because the JIT backends emit loads at
/// compile-time-computed offsets from the addresses of `total` / `threshold`
/// (which are baked into trace IR as KINT64 constants).
#[repr(C)]
pub struct GcHeap {
    pub strings: Interner,
    pub protos: Pool<Proto>,
    pub tables: Pool<LuaTable>,
    pub funcs: Pool<GcFunc>,
    pub upvals: Pool<Upval>,
    pub cdatas: Pool<CData>,
    pub userdatas: Pool<GcUserData>,
    pub threads: Pool<LuaState>,
    pub total: usize,
    pub threshold: usize,
    pub table_extra: usize,
    pub debt: usize,
    pub gc_state: GcState,
    pub gc_gray: Vec<Gray>,
    pub gc_grayagain: Vec<Gray>,
    /// Weak tables found during marking, with their `__mode` bits. Cleared
    /// in the atomic phase (entries whose key/value is about to be swept
    /// are removed before any object is freed).
    pub gc_weak: Vec<(GcPtr<LuaTable>, u8)>,
    /// Objects waiting for their `__gc` finalizer. Filled by the atomic
    /// phase; the VM drains it at the next safe point.
    pub mmudata: Vec<crate::runtime::gc::Finalizable>,
    pub gc_sweep_pool: u8,
    pub gc_step_size: usize,
    /// Tri-color white bit (0 or 1), flips each GC cycle.
    pub current_white: u8,
    /// When true, the collector will not auto-start from boundaries.
    pub gc_stopped: bool,
    /// Incremental FNV hash state for the concatenation fast path: the
    /// string id of the last result produced by the concat helpers
    /// (`jit_cat` / `meta_cat`) and the FNV-1a stream state at its end.
    /// A subsequent `s .. x` continues from this state instead of
    /// re-hashing all of `s` — O(1) per iteration for `s = s .. x` loops
    /// (string ids are never recycled, so the id uniquely identifies the
    /// bytes).
    pub cat_hash: Option<(u32, u64)>,
    /// String-buffer slot for the trace-side concat fast path (lj_buf):
    /// `s = s .. x` loops accumulate into this buffer and intern once at
    /// exit instead of every iteration.
    pub cat_buf: Vec<u8>,
    /// The stack slot (relative to a trace's base) holding the buffer —
    /// exits flush `cat_buf` back into it. `u32::MAX` when inactive.
    pub cat_buf_slot: u32,
}

impl Default for GcHeap {
    fn default() -> GcHeap {
        GcHeap {
            strings: Interner::default(),
            protos: Pool::new(GcObjectKind::Proto),
            tables: Pool::new(GcObjectKind::Table),
            funcs: Pool::new(GcObjectKind::Func),
            upvals: Pool::new(GcObjectKind::Upval),
            cdatas: Pool::new(GcObjectKind::CData),
            userdatas: Pool::new(GcObjectKind::UserData),
            threads: Pool::new(GcObjectKind::Thread),
            total: 0,
            threshold: gc::GC_THRESHOLD_MIN,
            table_extra: 0,
            debt: 0,
            gc_state: gc::GcState::Pause,
            gc_gray: Vec::new(),
            gc_grayagain: Vec::new(),
            gc_weak: Vec::new(),
            mmudata: Vec::new(),
            gc_sweep_pool: 0,
            gc_step_size: gc::GC_STEP_SIZE,
            current_white: 0,
            gc_stopped: false,
            cat_hash: None,
            cat_buf: Vec::new(),
            cat_buf_slot: u32::MAX,
        }
    }
}

impl GcHeap {
    /// Track an allocation and accumulate GC debt.
    fn account_alloc(&mut self, size: usize) {
        let live = self.total + self.strings.bytes() + self.table_extra;
        if live >= self.threshold {
            self.debt += size;
            // Advance threshold proportionally (LuaJIT's GC_PAUSE: 200%).
            // This avoids re-triggering the GCSTEP guard on every allocation
            // while ensuring the next collection triggers at ~2x memory.
            self.threshold = live + ((live * gc::GC_PAUSE) / 100).max(16384);
        }
    }

    pub fn alloc_table(&mut self, mut t: LuaTable) -> GcPtr<LuaTable> {
        t.table_extra = &mut self.table_extra as *mut usize;
        t.heap = self as *const GcHeap;
        let size = t.gc_size();
        self.total += size;
        gc::gc_step(self, size); // GC first — may start a cycle
        self.account_alloc(size);
        self.tables.alloc(t)
    }

    /// JIT-safe alloc: track the size but skip the incremental GC step.
    /// The trace is responsible for its own GCSTEP guard.
    pub fn alloc_table_jit(&mut self, mut t: LuaTable) -> GcPtr<LuaTable> {
        t.table_extra = &mut self.table_extra as *mut usize;
        t.heap = self as *const GcHeap;
        let size = t.gc_size();
        self.total += size;
        // Advance the threshold so the GCSTEP guard does not fire
        // immediately just because the live set grew a little.
        self.account_alloc(size);
        self.tables.alloc(t)
    }

    pub fn alloc_proto(&mut self, p: Proto) -> GcPtr<Proto> {
        let sz = p.gc_size();
        self.total += sz;
        self.account_alloc(sz);
        gc::gc_step(self, sz);
        self.protos.alloc(p)
    }

    pub fn alloc_func(&mut self, f: GcFunc) -> GcPtr<GcFunc> {
        let size = gc::account_func(&f);
        self.total += size;
        self.account_alloc(size);
        gc::gc_step(self, size);
        self.funcs.alloc(f)
    }

    pub fn alloc_upval(&mut self, uv: Upval) -> GcPtr<Upval> {
        let size = gc::account_upval();
        self.total += size;
        self.account_alloc(size);
        gc::gc_step(self, size);
        let p = self.upvals.alloc(uv);
        p.as_mut().init_closed();
        p
    }

    pub fn alloc_thread(&mut self, th: LuaState) -> GcPtr<LuaState> {
        let size = gc::account_thread(&th);
        self.total += size;
        self.account_alloc(size);
        gc::gc_step(self, size);
        self.threads.alloc(th)
    }

    pub fn alloc_cdata(&mut self, cd: CData) -> GcPtr<CData> {
        let size = std::mem::size_of::<CData>() + cd.data.len();
        self.total += size;
        self.account_alloc(size);
        gc::gc_step(self, size);
        self.cdatas.alloc(cd)
    }

    pub fn alloc_userdata(&mut self, ud: GcUserData) -> GcPtr<GcUserData> {
        let size = std::mem::size_of::<GcUserData>();
        self.total += size;
        self.account_alloc(size);
        gc::gc_step(self, size);
        self.userdatas.alloc(ud)
    }

    pub fn intern(&mut self, s: &[u8]) -> StrId {
        let prev_bytes = self.strings.bytes();
        let sid = self.strings.intern(s);
        let new_bytes = self.strings.bytes();
        if new_bytes > prev_bytes {
            let sz = new_bytes - prev_bytes;
            self.account_alloc(sz);
            gc::gc_step(self, sz);
        }
        sid
    }

    /// Intern with a precomputed FNV hash (incremental concat fast path).
    pub fn intern_with_hash(&mut self, s: &[u8], hash: u32) -> StrId {
        let prev_bytes = self.strings.bytes();
        let sid = self.strings.intern_with_hash(s, hash);
        let new_bytes = self.strings.bytes();
        if new_bytes > prev_bytes {
            let sz = new_bytes - prev_bytes;
            self.account_alloc(sz);
            gc::gc_step(self, sz);
        }
        sid
    }

    /// A `LuaValue` for an interned string id.
    pub fn str_value(&self, sid: StrId) -> LuaValue {
        LuaValue::string(self.strings.lookup_ptr(sid))
    }

    /// `lj_gc_check`'s condition: is a collection due?
    #[inline]
    pub fn should_collect(&self) -> bool {
        self.total + self.strings.bytes() + self.table_extra >= self.threshold
    }
}

/// Number of internal itype tags, used to size the base-metatable array.
const ITYPE_COUNT: usize = 16;

/// Global state shared by all threads of a Lua universe, corresponding to
/// LuaJIT's `global_State`.
///
/// Not constructed directly: it is owned (boxed, at a fixed address) by the
/// top-level [`Lua`] object, which also owns every [`LuaState`]. Threads hold
/// a back-pointer to this via [`GlobalRef`].
pub struct GlobalState {
    pub heap: GcHeap,
    /// The globals table `_G` (default function environment).
    pub globals: GcPtr<LuaTable>,
    /// The registry table.
    pub registry: GcPtr<LuaTable>,
    /// Per-type base metatables, indexed by `~itype`.
    pub basemt: [Option<GcPtr<LuaTable>>; ITYPE_COUNT],
    /// Interned metamethod name strings, indexed by `MM` (LuaJIT's
    /// `GCROOT_MMNAME` roots, filled by `lj_meta_init`).
    pub mmname: [LuaValue; meta::MM_MAX],
    /// The currently running thread (LuaJIT's `cur_L`): the main thread or
    /// the innermost resumed coroutine.
    pub cur_l: Option<StateRef>,
    /// PRNG state for `math.random` (xoshiro256**).
    pub rng: crate::stdlib::math::RngState,
    /// The JIT compiler state (LuaJIT embeds `jit_State` in `GG_State`).
    pub jit: JitState,
    /// FFI C type system (lazy-initialised by `ffi.load` / first FFI call).
    pub cts: Option<CTState>,
    /// Per-ctype metatables registered by `ffi.metatype`, indexed by
    /// ctype id (each cdata's metatable lookup consults this first).
    pub ctype_mts: Vec<Option<GcPtr<LuaTable>>>,
    /// `ffi.errno` state (the tests set/read it through FFI).
    pub ffi_errno: i32,
    /// The internal ipairs iterator closure (not exposed as a global).
    pub ipairs_iter: LuaValue,
    /// `os.clock()` baseline: `Instant::now()` captured when the universe is
    /// created, so the reported time is relative to process start (matches
    /// LuaJIT's `luaopen_os` time).  Stored as `f64` seconds from epoch
    /// for cheap differencing at every `os.clock` call.
    pub(crate) boot_time: PlatformInstant,
    /// The main thread. Set once the owning [`Lua`] is pinned. The interpreter
    /// entry points use this when no explicit thread is supplied.
    main: Option<StateRef>,
}

impl GlobalState {
    pub fn basemt_of(&self, itype: u32) -> Option<GcPtr<LuaTable>> {
        self.basemt[(!itype) as usize & (ITYPE_COUNT - 1)]
    }

    pub fn set_basemt(&mut self, itype: u32, mt: Option<GcPtr<LuaTable>>) {
        self.basemt[(!itype) as usize & (ITYPE_COUNT - 1)] = mt;
    }

    /// The main thread. Panics if the `Lua` universe was not fully built.
    pub fn main(&self) -> StateRef {
        self.main.expect("main thread not initialized")
    }
}

/// A wrapped raw pointer to the [`GlobalState`], as held by every thread
/// (LuaJIT's `G(L)`). Confining the raw pointer here keeps `unsafe` localized;
/// the pointee is pinned inside a `Box` owned by the [`Lua`] object and
/// outlives all threads.
#[derive(Clone, Copy)]
pub struct GlobalRef(NonNull<GlobalState>);

impl GlobalRef {
    #[allow(clippy::mut_from_ref)]
    pub fn get<'a>(self) -> &'a mut GlobalState {
        unsafe { &mut *self.0.as_ptr() }
    }

    /// Shared reference with `'static` lifetime — the `Box<GlobalState>`
    /// outlives every thread, so the address is always valid.  Library
    /// functions use this to read string data without locking out
    /// mutable heap access.
    pub fn get_ref(self) -> &'static GlobalState {
        unsafe { &*self.0.as_ptr() }
    }
}

/// A reference to a [`LuaState`] in the thread pool (used for the stored
/// main thread and for thread `LuaValue`s). Being a `GcPtr`, it carries the
/// pool mark bit, so coroutines participate in GC like any other object.
pub type StateRef = GcPtr<LuaState>;

impl GcPtr<LuaState> {
    /// Legacy accessor kept from the old `StateRef` wrapper.
    #[allow(clippy::mut_from_ref)]
    pub fn get<'a>(self) -> &'a mut LuaState {
        self.as_mut()
    }
}

/// Maximum value-stack size (in slots) of the main thread. Fixed so the
/// backing `Vec` never reallocates during execution, keeping raw stack
/// pointers valid.
pub const STACK_MAX: usize = 1 << 16;

/// Value-stack size of a coroutine (16 KiB). Smaller than the main stack so
/// `coroutine.create` stays cheap; fixed for the same pointer-stability
/// reason.
pub const CO_STACK_MAX: usize = 1 << 11; // 2048 slots = 16 KiB

/// Coroutine status, mirroring `lua_State.status` + the distinctions
/// `coroutine.status` reports.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoStatus {
    /// Not started yet, or stopped in a `yield`.
    Suspended,
    /// Currently executing (`G->cur_l`).
    Running,
    /// Resumed somebody else and is waiting for them.
    Normal,
    /// Finished or stopped by an error.
    Dead,
}

/// Where and how a coroutine is suspended; consumed by `resume`.
#[derive(Clone, Copy)]
pub enum Suspend {
    /// Fresh coroutine: the entry function sits at `stack[0]`, resume
    /// arguments become its call arguments.
    Start,
    /// Yield from a `CALL coroutine.yield` in a Lua frame: continue at
    /// `pc` with `cl`, delivering the resume args at `slot` per `want`.
    /// `protected` marks a yield through a `pcall`/`xpcall` frame: the
    /// resume args must be delivered with a `true` prefix (the pcall's
    /// success flag), like LuaJIT's pcall continuation. `value_slot` is
    /// where the yield values live (may differ from `slot` when the
    /// protected call's own frame sits elsewhere).
    Call {
        pc: usize,
        cl: GcPtr<GcFunc>,
        base: usize,
        slot: usize,
        want: i32,
        protected: bool,
        value_slot: usize,
    },
    /// Yield through a tail call (`return coroutine.yield(...)`) or from
    /// the entry C function: resume performs a *return* of the resume args
    /// from `slot` in the frame at `base`.
    Return { base: usize, slot: usize },
}

impl Suspend {
    pub fn call_cl(&self) -> Option<GcPtr<GcFunc>> {
        match self {
            Suspend::Call { cl, .. } => Some(*cl),
            _ => None,
        }
    }
}

/// A Lua execution thread, corresponding to LuaJIT's `lua_State`.
///
/// Owns its value stack and open-upvalue list, and holds a back-pointer to
/// the shared [`GlobalState`]. Threads live in the heap's thread pool and
/// are collected by the GC (except the main thread, a permanent root).
/// There is no separate control stack: call frames live in the value stack
/// itself, LuaJIT-style (see `vm`'s frame-link encoding).
pub struct LuaState {
    g: GlobalRef,
    is_main: bool,
    /// The value stack / register file. Grows dynamically up to `_max_stack`.
    pub stack: Vec<LuaValue>,
    /// Top of the current Lua frame (`base + framesize`), kept for the GC:
    /// the collector must never clear slots below it even if `top` was
    /// lowered to a C-call result area (`lj_gc_step_fixtop`).
    pub frame_top: usize,
    _max_stack: usize,
    pub base: usize,
    pub top: usize,
    /// Open upvalues pointing into this thread's stack, kept sorted by slot
    /// (descending), mirroring LuaJIT's `L->openupval` list.
    pub openuv: Vec<GcPtr<Upval>>,
    /// The pending error object (`LuaError::Runtime`).
    pub errval: LuaValue,
    /// Args-base slot of the Lua frame that was executing when `error`
    /// was raised (set by lib_error, consumed by lib_xpcall's handler
    /// frame chain so debug walks see the raise-time frames below it).
    pub err_raise_slot: usize,
    /// The (function bits, bytecode index) of the frame where the current
    /// runtime error was raised — traceback uses it to report the error
    /// line on the failed frame (the frame link alone only shows the
    /// caller's call site).
    pub err_raise_pc: Option<(u64, usize)>,
    /// While a metamethod invoked through the cold execute-recursion
    /// paths (e.g. `__concat`) is running, the (name, function bits) of
    /// the active metamethod — debug.getinfo uses it to report
    /// `namewhat = "metamethod"` for the current frame.
    pub mmname: Option<(&'static str, u64)>,
    /// The number of yielded values (`LuaError::Yield`).
    pub nyield: u32,
    /// Coroutine status.
    pub status: CoStatus,
    /// Suspension point for `resume` (meaningful when `status == Suspended`).
    pub suspend: Suspend,
    /// Rust-recursion depth (incremented by every `execute` re-entry).
    /// LuaJIT's cframe-chain stand-in for the yield-across-C-boundary check.
    pub c_depth: u32,
    /// `c_depth` recorded when this coroutine was (re)entered; yielding is
    /// legal only while `c_depth == c_base` (no intervening C frames).
    pub c_base: u32,
    /// Current bytecode PC, updated by the VM for error location reporting.
    pub debug_pc: usize,
    /// Current chunk name for error location reporting.
    pub debug_chunkname: Vec<u8>,
    /// The environment table of this (possibly coroutine) thread
    /// (`debug.setfenv` on a thread; `getfenv(0)` reports it).
    pub thread_env: GcPtr<LuaTable>,
    /// Debug hook function (debug.sethook); nil when inactive.
    pub hook: LuaValue,
    /// Hook mask: bit0 = line, bit1 = call, bit2 = return, bit3 = count.
    pub hookmask: u8,
    /// Instruction counter for the count hook (remaining until trigger).
    pub hookcount: i32,
    /// Original count interval for the count hook (reset target).
    pub hook_count_reset: i32,
    /// Last line reported by the line hook (avoid repeats on same line).
    pub hook_line: u32,
    /// Set while a hook is running (hooks don't re-enter themselves).
    pub hook_active: bool,
}

impl LuaState {
    /// Create a thread bound to `g`. `is_main` marks the primary thread.
    /// Mirrors LuaJIT, where a `lua_State` always carries `G(L)`.
    ///
    /// The stack starts tiny (8 slots) and grows lazily via `stack_ensure`;
    /// `is_main` pre-allocates the full 512 KiB so the main thread never
    /// pays for `Vec::resize` during execution. Coroutines cost ~64 bytes
    /// at creation. The stack starts tiny and grows dynamically when the
    /// depth requires it. Open upvalue pointers are re-anchored after every
    /// resize so closures stay correct.
    pub fn new(g: GlobalRef, is_main: bool) -> LuaState {
        let max_stack = if is_main { STACK_MAX } else { CO_STACK_MAX };
        let initial_len = if is_main { 256 } else { 8 };
        LuaState {
            g,
            is_main,
            stack: {
                let mut v = Vec::with_capacity(initial_len);
                v.resize(initial_len, LuaValue::NIL);
                v
            },
            frame_top: 0,
            _max_stack: max_stack,
            base: 0,
            top: 0,
            openuv: Vec::new(),
            errval: LuaValue::NIL,
            err_raise_slot: 0,
            err_raise_pc: None,
            mmname: None,
            nyield: 0,
            status: if is_main {
                CoStatus::Running
            } else {
                CoStatus::Suspended
            },
            suspend: Suspend::Start,
            c_depth: 0,
            c_base: 0,
            debug_pc: 0,
            debug_chunkname: Vec::new(),
            thread_env: g.get_ref().globals,
            hook: LuaValue::NIL,
            hookmask: 0,
            hookcount: 0,
            hook_count_reset: 0,
            hook_line: 0,
            hook_active: false,
        }
    }

    /// Ensure the stack can hold at least `need` slots (absolute index).
    /// Grows dynamically by doubling, capped at `STACK_MAX`.  Open upvalue
    /// pointers that reference the old stack are patched after a reallocation.
    /// Also serves as a GC check point so loops that push values (e.g. table
    /// stores, string concatenation) eventually trigger collection.
    #[inline]
    pub fn stack_ensure(&mut self, need: usize) {
        if need >= self.stack.len() {
            let new_len = (self.stack.len() * 2).max(need + 16).min(self._max_stack);
            assert!(new_len <= self._max_stack, "stack overflow");
            let old_len = self.stack.len();
            let old_ptr = self.stack.as_mut_ptr();
            self.stack.resize(new_len, LuaValue::NIL);
            let new_ptr = self.stack.as_mut_ptr();
            if old_ptr != new_ptr {
                let delta_bytes = new_ptr as isize - old_ptr as isize;
                if delta_bytes != 0 {
                    for &uv in &self.openuv {
                        let uv_mut = uv.as_mut();
                        let p = uv_mut.value_ptr();
                        if p >= old_ptr && p < unsafe { old_ptr.add(old_len) } {
                            let new_p =
                                unsafe { (p as *mut u8).offset(delta_bytes) as *mut LuaValue };
                            uv_mut.repoint(unsafe { NonNull::new_unchecked(new_p) });
                        }
                    }
                }
            }
            // Keep the incremental GC accounting in sync: size_thread
            // measures stack.len(), so growth must land in heap.total.
            let grown = (new_len - old_len) * std::mem::size_of::<LuaValue>();
            if grown > 0 {
                self.g.get().heap.total += grown;
            }
        }
    }

    pub fn global(&self) -> &mut GlobalState {
        self.g.get()
    }

    pub fn heap(&self) -> &mut GcHeap {
        &mut self.g.get().heap
    }

    /// Get a string's content without cloning, using pool-stable `'static`
    /// lifetimes. This is the key zero-copy primitive for library functions:
    /// read args with `l.str_static(sid)`, intern results with
    /// `l.heap().intern(...)`, never a borrow conflict.
    #[inline]
    pub fn str_static(&self, sid: StrId) -> &'static [u8] {
        self.g.get_ref().heap.strings.get_static(sid)
    }

    /// `lua_upvalueindex(i)`: read the i-th upvalue (0-based) of the
    /// currently running C closure.  The closure lives at `base - 2`.
    #[inline]
    pub fn upvalue(&self, i: usize) -> LuaValue {
        let f = self.stack[self.base - 2];
        match f.as_func().map(|p| p.as_ref()) {
            Some(GcFunc::C(cc)) => cc.upvals.get(i).copied().unwrap_or(LuaValue::NIL),
            _ => LuaValue::NIL,
        }
    }

    /// `lua_setupvalue`: overwrite the i-th upvalue of the currently
    /// running C closure.
    pub fn set_upvalue(&mut self, i: usize, v: LuaValue) {
        let f = self.stack[self.base - 2];
        if let Some(gf) = f.as_func()
            && let GcFunc::C(cc) = gf.as_mut()
            && i < cc.upvals.len()
        {
            cc.upvals[i] = v;
        }
    }

    pub fn is_main(&self) -> bool {
        self.is_main
    }

    /// A `GcPtr` to this state itself (valid because every `LuaState`
    /// lives in the heap's thread pool at a stable address).
    pub fn self_ref(&self) -> StateRef {
        GcPtr::from_addr(self as *const LuaState as u64).unwrap()
    }

    /// Yield is legal when we're inside a coroutine and `c_depth == c_base`
    /// (no C frames between the resume point and the yield).
    pub fn is_yieldable(&self) -> bool {
        !self.is_main && self.c_depth == self.c_base
    }

    pub fn push(&mut self, v: LuaValue) {
        self.stack[self.top] = v;
        self.top += 1;
    }

    pub fn pop(&mut self) -> LuaValue {
        debug_assert!(self.top > 0);
        self.top -= 1;
        self.stack[self.top]
    }

    /// Raise a runtime error carrying a string message with source location.
    pub fn runtime_error(&mut self, msg: impl AsRef<[u8]>) -> LuaError {
        self.runtime_error_level(msg, 1)
    }

    /// `runtime_error` with `error()`-style `level` semantics: 1 = the
    /// direct caller's location, 2 = the caller's caller, etc.
    pub fn runtime_error_level(&mut self, msg: impl AsRef<[u8]>, level: u32) -> LuaError {
        let mut full = msg.as_ref().to_vec();

        let mut skip = level.saturating_sub(1) as usize;
        let mut slot = self.base;
        let mut pc_from_link: Option<usize> = None;
        for _ in 0..16 {
            if slot < 2 {
                break;
            }
            let func = self.stack[slot - 2];
            let link_bits = self.stack[slot - 1].to_bits();
            if let Some(fv) = func.as_func() {
                match fv.as_ref() {
                    GcFunc::Lua(cl) => {
                        let pt = cl.proto.as_ref();
                        if skip > 0 {
                            // Not the position frame: walk to the caller.
                            skip -= 1;
                            let ft = link_bits & FRAME_TYPE_MASK;
                            if ft == 0 && link_bits != 0 {
                                if ((link_bits >> 3) as usize) < self.stack.len() {
                                    slot = (link_bits >> 3) as usize;
                                } else {
                                    // The return PC lives in the *caller's*
                                    // proto, so carry the raw address and
                                    // resolve the offset against the
                                    // position frame's proto below.
                                    let ret_ip = link_bits as *const crate::bc::BCIns;
                                    pc_from_link = Some(ret_ip as usize);
                                    let call_ins = unsafe { *ret_ip.sub(1) };
                                    let a = crate::bc::bc_a(call_ins) as usize;
                                    slot = slot.saturating_sub(2 + a);
                                }
                                continue;
                            }
                            break;
                        }
                        let pc = match pc_from_link {
                            // Frame reached via a Lua link: the return PC
                            // points past the CALL; the CALL itself sits at
                            // pc-1 (both carry the same statement's line,
                            // but the RET0/jump at the return PC may sit on
                            // a later line).
                            Some(p) => {
                                let off = unsafe {
                                    (p as *const crate::bc::BCIns).offset_from(pt.bc.as_ptr())
                                } as usize;
                                off.saturating_sub(1)
                            }
                            None => self.debug_pc.saturating_sub(1),
                        }
                        .min(pt.lines.len().saturating_sub(1));
                        // Remember the raise site so the traceback can
                        // report the failed frame's error line.
                        self.err_raise_pc = Some((func.to_bits(), pc));
                        let line = if pc < pt.lines.len() {
                            pt.lines[pc] as usize
                        } else {
                            pt.firstline as usize
                        };
                        let src = pt
                            .source
                            .and_then(|sid| {
                                self.heap().strings.try_lookup(sid).map(|_ptr| {
                                    let bytes = self.heap().strings.get(sid);
                                    // Strip leading '@' or '=' for display.
                                    let s = if bytes.starts_with(b"@") || bytes.starts_with(b"=") {
                                        &bytes[1..]
                                    } else {
                                        bytes
                                    };
                                    String::from_utf8_lossy(s).into_owned()
                                })
                            })
                            .unwrap_or_else(|| "=?".to_string());
                        let msg_str = String::from_utf8_lossy(&full);
                        full = format!("{}:{}: {}", src, line, msg_str).into_bytes();
                        break;
                    }
                    GcFunc::C(_) => {
                        // Walk to caller via frame link. FRAME_LUA=0 means
                        // the link encodes the caller's base; otherwise it
                        // is the caller's return PC.
                        let ft = link_bits & FRAME_TYPE_MASK;
                        if ft == 0 && link_bits != 0 {
                            if ((link_bits >> 3) as usize) < self.stack.len() {
                                slot = (link_bits >> 3) as usize;
                            } else {
                                let ret_ip = link_bits as *const crate::bc::BCIns;
                                let call_ins = unsafe { *ret_ip.sub(1) };
                                let a = crate::bc::bc_a(call_ins) as usize;
                                slot = slot.saturating_sub(2 + a);
                            }
                            continue;
                        }
                        break;
                    }
                }
            }
            break;
        }

        let sid = self.heap().intern(&full);
        self.errval = self.heap().str_value(sid);
        LuaError::Runtime
    }

    /// Register a builtin function as a global under `name`.
    pub fn register(&mut self, name: &[u8], f: CFunction) {
        let g = self.global();
        let sid = g.heap.intern(name);
        let env = g.globals;
        let fref = g.heap.alloc_func(GcFunc::C(CClosure {
            f,
            env,
            upvals: Vec::new(),
        }));
        let key = g.heap.str_value(sid);
        g.globals.as_mut().set(key, LuaValue::func(fref));
    }
}

/// A Lua universe: the single owner of the [`GlobalState`] (and thus the
/// heap). Threads (main + coroutines) live in the heap's thread pool;
/// everything refers to them through `GcPtr`s, so their addresses stay
/// fixed and the GC can collect dead coroutines.
pub struct Lua {
    g: Box<GlobalState>,
}

impl Default for Lua {
    fn default() -> Self {
        Self::new()
    }
}

impl Lua {
    pub fn new() -> Lua {
        // Allocate GlobalState on the heap, then initialise it via a raw
        // pointer so that all alloc_table calls reference the heap-resident
        // GcHeap field from the start (no stack→heap migration).
        let boot = PlatformInstant::now();
        let g = Box::into_raw(Box::new(GlobalState {
            heap: GcHeap::default(),
            globals: GcPtr::new(NonNull::dangling()),
            registry: GcPtr::new(NonNull::dangling()),
            basemt: [None; ITYPE_COUNT],
            mmname: [LuaValue::NIL; meta::MM_MAX],
            cur_l: None,
            rng: crate::stdlib::math::RngState::fixed(),
            jit: JitState::new(),
            cts: None,
            ctype_mts: Vec::new(),
            ffi_errno: 0,
            ipairs_iter: LuaValue::NIL,
            boot_time: boot,
            main: None,
        }));
        let gs = unsafe { &mut *g };
        gs.globals = gs.heap.alloc_table(LuaTable::new(0, 1));
        gs.registry = gs.heap.alloc_table(LuaTable::new(0, 1));
        for (i, name) in meta::MM_NAMES.iter().enumerate() {
            let sid = gs.heap.intern(name);
            gs.mmname[i] = gs.heap.str_value(sid);
        }
        let mut lua = Lua {
            g: unsafe { Box::from_raw(g) },
        };
        let gref = GlobalRef(NonNull::from(&*lua.g));
        let main_ref = lua.g.heap.alloc_thread(LuaState::new(gref, true));
        lua.g.main = Some(main_ref);
        lua.g.cur_l = Some(main_ref);
        lua
    }

    pub fn global(&mut self) -> &mut GlobalState {
        &mut self.g
    }

    pub fn main(&mut self) -> &mut LuaState {
        self.g.main().get()
    }

    /// Spawn a new (coroutine) thread owned by this universe.
    pub fn new_thread(&mut self) -> StateRef {
        let gref = GlobalRef(NonNull::from(&*self.g));
        self.g.heap.alloc_thread(LuaState::new(gref, false))
    }
}

/// Spawn a coroutine thread from within a running state (`lua_newthread`).
pub fn new_thread(l: &LuaState) -> StateRef {
    let g = l.global();
    let gref = GlobalRef(NonNull::from(&*g));
    g.heap.alloc_thread(LuaState::new(gref, false))
}

pub fn load(l: &mut LuaState, src: Vec<u8>, chunkname: &str) -> Result<LuaValue, String> {
    let g = l.global();

    // Detect string.dump cache format: "\x1bLJ" + index
    let mut proto = if src.len() >= 3 && &src[..3] == b"\x1bLJ" {
        let idx_str = String::from_utf8_lossy(&src[3..]);
        if let Ok(idx) = idx_str.parse::<u32>() {
            let cache_key = g.heap.intern(b"__LUARS_DUMP_CACHE");
            let key = g.heap.str_value(cache_key);
            let registry = g.registry.as_ref();
            if let Some(fv) = registry.get(key).as_table()
                && fv.as_ref().get_int(idx as i32).is_func()
            {
                return Ok(fv.as_ref().get_int(idx as i32));
            }
            return Err("corrupted dump cache".to_string());
        }
        return Err("corrupted dump cache".to_string());
    } else {
        let mut parser = Parser::new(src, chunkname.to_string(), &mut g.heap.strings);
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parser.parse()));
        std::panic::set_hook(prev_hook);
        match result {
            Ok(p) => p,
            Err(e) => {
                let msg = if let Some(ce) = e.downcast_ref::<CompileError>() {
                    ce.0.clone()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else {
                    "unknown compile error".to_string()
                };
                return Err(msg);
            }
        }
    };

    debug_assert!(proto.uv.is_empty(), "main chunk must have no upvalues");
    // Every chunk carries a source name (Lua 5.1: `luaO_chunkid`), even
    // "=literal" names; the display layers strip the '@'/'=' prefix.
    if !chunkname.is_empty() {
        let source_sid = g.heap.strings.intern(chunkname.as_bytes());
        proto.source = Some(source_sid);
    }
    let proto_ref = register_proto(&mut g.heap, proto);
    let env = g.globals;
    let fref = g.heap.alloc_func(GcFunc::Lua(LuaClosure {
        proto: proto_ref,
        env,
        upvals: Vec::new(),
    }));
    Ok(LuaValue::func(fref))
}

/// Recursively register a prototype tree in the heap, turning each child
/// `KGc::Proto` constant into a `KGc::ProtoRef` pointing at the heap object
/// and resolving string constants into the `kstrv` fast-lookup table.
pub fn register_proto(heap: &mut GcHeap, mut proto: Proto) -> GcPtr<Proto> {
    for i in 0..proto.kgc.len() {
        if matches!(proto.kgc[i], KGc::Proto(_)) {
            let taken = std::mem::replace(&mut proto.kgc[i], KGc::Str(0));
            if let KGc::Proto(child) = taken {
                let r = register_proto(heap, *child);
                // Propagate parent source to child protos that don't have one.
                if r.as_ref().source.is_none() {
                    r.as_mut().source = proto.source;
                }
                proto.kgc[i] = KGc::ProtoRef(r);
            }
        }
    }
    proto.kstrv = proto
        .kgc
        .iter()
        .map(|k| match k {
            KGc::Str(sid) => {
                if let Some(ptr) = heap.strings.try_lookup(*sid) {
                    LuaValue::string(ptr)
                } else {
                    LuaValue::NIL
                }
            }
            _ => LuaValue::NIL,
        })
        .collect();
    heap.alloc_proto(proto)
}

/// Base-metatable itypes exposed for builtins.
pub const BASEMT_TAB: u32 = LJ_TTAB;
pub const BASEMT_FUNC: u32 = LJ_TFUNC;

/// Ensure `GcRef` remains the pointer-sized payload type.
const _: () = assert!(std::mem::size_of::<GcRef>() == 8);

#[cfg(test)]
mod tests {
    use crate::LuaResult;

    use super::*;

    #[test]
    fn load_produces_top_level_closure() {
        let mut lua = Lua::new();
        let f = load(lua.main(), b"local x = 1 return x".to_vec(), "@test").unwrap();
        assert!(f.is_func());
        match f.as_func().unwrap().as_ref() {
            GcFunc::Lua(c) => {
                assert!(c.upvals.is_empty());
                let pt = c.proto.as_ref();
                assert_eq!(pt.numparams, 0);
                assert!(!pt.bc.is_empty());
            }
            _ => panic!("expected Lua closure"),
        }
    }

    #[test]
    fn load_stores_source_on_proto() {
        let mut lua = Lua::new();
        let f = load(lua.main(), b"error('test')".to_vec(), "@test.lua").unwrap();
        match f.as_func().unwrap().as_ref() {
            GcFunc::Lua(c) => {
                let pt = c.proto.as_ref();
                assert!(pt.source.is_some(), "source should be set");
                let sid = pt.source.unwrap();
                let bytes = lua.global().heap.strings.get(sid);
                assert_eq!(bytes, b"@test.lua");
            }
            _ => panic!("expected Lua closure"),
        }
    }

    #[test]
    fn nested_proto_inherits_source() {
        let mut lua = Lua::new();
        let f = load(lua.main(), b"local function f() end".to_vec(), "@test.lua").unwrap();
        match f.as_func().unwrap().as_ref() {
            GcFunc::Lua(c) => {
                assert!(
                    c.proto.as_ref().source.is_some(),
                    "main chunk should have source"
                );
                let pt = c.proto.as_ref();
                for k in &pt.kgc {
                    if let KGc::ProtoRef(child) = k {
                        assert!(
                            child.as_ref().source.is_some(),
                            "child proto should inherit source"
                        );
                        let sid = child.as_ref().source.unwrap();
                        let bytes = lua.global().heap.strings.get(sid);
                        assert_eq!(bytes, b"@test.lua");
                        return;
                    }
                }
                panic!("no child proto found");
            }
            _ => panic!("expected Lua closure"),
        }
    }

    #[test]
    fn load_reports_syntax_errors() {
        let mut lua = Lua::new();
        let err = load(lua.main(), b"local = ".to_vec(), "@bad").unwrap_err();
        assert!(!err.is_empty());
        let f = load(lua.main(), b"return 1".to_vec(), "@ok").unwrap();
        assert!(f.is_func());
    }

    #[test]
    fn register_and_lookup_global() {
        fn dummy(_l: &mut LuaState) -> LuaResult<i32> {
            Ok(0)
        }
        let mut lua = Lua::new();
        lua.main().register(b"print", dummy);
        let g = lua.global();
        let sid = g.heap.intern(b"print");
        let key = g.heap.str_value(sid);
        let v = g.globals.as_ref().get(key);
        assert!(v.is_func());
    }

    #[test]
    fn object_addresses_are_stable() {
        let mut lua = Lua::new();
        let t0 = lua.global().heap.alloc_table(LuaTable::new(0, 1));
        let addr = t0.addr();
        for _ in 0..1000 {
            lua.global().heap.alloc_table(LuaTable::new(0, 1));
        }
        assert_eq!(t0.addr(), addr);
        let v = LuaValue::table(t0);
        assert_eq!(v.as_table().unwrap().addr(), addr);
    }

    #[test]
    fn threads_share_one_global() {
        let mut lua = Lua::new();
        let co = lua.new_thread();
        let main_g = lua.main().global() as *mut GlobalState;
        let co_g = co.get().global() as *mut GlobalState;
        assert_eq!(main_g, co_g);
        assert!(lua.main().is_main());
        assert!(!co.get().is_main());
    }

    #[test]
    fn stack_push_pop() {
        let mut lua = Lua::new();
        let l = lua.main();
        l.push(LuaValue::number(1.0));
        l.push(LuaValue::TRUE);
        assert_eq!(l.top, 2);
        assert!(l.pop().is_true());
        assert_eq!(l.pop().as_number(), Some(1.0));
        assert_eq!(l.top, 0);
    }
}
