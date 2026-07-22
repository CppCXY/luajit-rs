use std::ptr::NonNull;

use crate::func::{CClosure, CFunction, GcFunc, LuaClosure};
use crate::gc::{GcPtr, Pool};
use crate::proto::Proto;
use crate::string::{Interner, StrId};
use crate::table::LuaTable;
use crate::value::{GcRef, LJ_TFUNC, LJ_TTAB, LuaValue};
use crate::vm::FRAME_TYPE_MASK;

/// The GC heap: stable-address object pools.
///
/// Every collectable type lives in its own `Pool`, which allocates objects in
/// fixed pages so their addresses never move (a `LuaValue` stores the raw
/// pointer in its 47-bit payload). The collector (`gc::full_gc`) marks from
/// the roots and sweeps these pools. `total`/`threshold` drive the trigger,
/// like LuaJIT's `gc.total`/`gc.threshold`.
pub struct GcHeap {
    pub strings: Interner,
    pub protos: Pool<Proto>,
    pub tables: Pool<LuaTable>,
    pub funcs: Pool<GcFunc>,
    pub upvals: Pool<crate::func::Upval>,
    pub cdatas: Pool<crate::runtime::cdata::CData>,
    /// Threads (the main thread and all coroutines). Coroutines are
    /// collected by the GC like any other object; the main thread is a
    /// permanent root.
    pub threads: Pool<LuaState>,
    /// Allocation estimate for non-string objects (strings are tracked by
    /// the interner itself, which travels to the parser and back).
    pub total: usize,
    /// Next collection when `total + strings.bytes()` crosses this.
    pub threshold: usize,
}

impl Default for GcHeap {
    fn default() -> GcHeap {
        GcHeap {
            strings: Interner::default(),
            protos: Pool::with_page_size(16),
            tables: Pool::with_page_size(64),
            funcs: Pool::with_page_size(64),
            upvals: Pool::with_page_size(128),
            cdatas: Pool::with_page_size(32),
            threads: Pool::with_page_size(4),
            total: 0,
            threshold: crate::gc::GC_THRESHOLD_MIN,
        }
    }
}

impl GcHeap {
    pub fn alloc_table(&mut self, t: LuaTable) -> GcPtr<LuaTable> {
        self.total += t.gc_size();
        self.tables.alloc(t)
    }

    pub fn alloc_proto(&mut self, p: Proto) -> GcPtr<Proto> {
        self.total += p.gc_size();
        self.protos.alloc(p)
    }

    pub fn alloc_func(&mut self, f: GcFunc) -> GcPtr<GcFunc> {
        self.total += crate::gc::account_func(&f);
        self.funcs.alloc(f)
    }

    pub fn alloc_upval(&mut self, uv: crate::func::Upval) -> GcPtr<crate::func::Upval> {
        self.total += crate::gc::account_upval();
        let p = self.upvals.alloc(uv);
        // Closed upvalues point at their own inline slot; the address is
        // only stable after pool insertion, so patch it up here.
        p.as_mut().init_closed();
        p
    }

    pub fn alloc_thread(&mut self, th: LuaState) -> GcPtr<LuaState> {
        self.total += crate::gc::account_thread(&th);
        self.threads.alloc(th)
    }

    pub fn alloc_cdata(
        &mut self,
        cd: crate::runtime::cdata::CData,
    ) -> GcPtr<crate::runtime::cdata::CData> {
        self.total += std::mem::size_of::<crate::runtime::cdata::CData>() + cd.data.len();
        self.cdatas.alloc(cd)
    }

    pub fn intern(&mut self, s: &[u8]) -> StrId {
        self.strings.intern(s)
    }

    /// A `LuaValue` for an interned string id.
    pub fn str_value(&self, sid: StrId) -> LuaValue {
        LuaValue::string(self.strings.lookup_ptr(sid))
    }

    /// `lj_gc_check`'s condition: is a collection due?
    #[inline]
    pub fn should_collect(&self) -> bool {
        self.total + self.strings.bytes() + crate::table::TABLE_EXTRA.with(|c| c.get())
            >= self.threshold
    }

    /// Run a full GC cycle if the threshold has been exceeded.
    /// Call at allocation points to prevent unbounded growth.
    pub fn maybe_collect(g: &mut GlobalState) {
        if g.heap.should_collect() {
            crate::gc::full_gc(g);
        }
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
    pub mmname: [LuaValue; crate::runtime::meta::MM_MAX],
    /// The currently running thread (LuaJIT's `cur_L`): the main thread or
    /// the innermost resumed coroutine.
    pub cur_l: Option<StateRef>,
    /// The JIT compiler state (LuaJIT embeds `jit_State` in `GG_State`).
    pub jit: crate::jit::JitState,
    /// FFI C type system (lazy-initialised by `ffi.load` / first FFI call).
    pub cts: Option<crate::ffi::CTState>,
    /// `os.clock()` baseline: `Instant::now()` captured when the universe is
    /// created, so the reported time is relative to process start (matches
    /// LuaJIT's `luaopen_os` time).  Stored as `f64` seconds from epoch
    /// for cheap differencing at every `os.clock` call.
    pub boot_time: f64,
    /// The main thread. Set once the owning [`Lua`] is pinned. The interpreter
    /// entry points use this when no explicit thread is supplied.
    main: Option<StateRef>,
}

impl GlobalState {
    fn new() -> GlobalState {
        use std::time::UNIX_EPOCH;
        let boot = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let mut heap = GcHeap::default();
        let globals = heap.alloc_table(LuaTable::new(0, 1));
        let registry = heap.alloc_table(LuaTable::new(0, 1));
        // lj_meta_init: intern the metamethod names once.
        let mut mmname = [LuaValue::NIL; crate::runtime::meta::MM_MAX];
        for (i, name) in crate::runtime::meta::MM_NAMES.iter().enumerate() {
            let sid = heap.intern(name);
            mmname[i] = heap.str_value(sid);
        }
        GlobalState {
            heap,
            globals,
            registry,
            basemt: [None; ITYPE_COUNT],
            mmname,
            cur_l: None,
            jit: crate::jit::JitState::new(),
            cts: None,
            boot_time: boot,
            main: None,
        }
    }

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
    Call {
        pc: usize,
        cl: GcPtr<GcFunc>,
        base: usize,
        slot: usize,
        want: i32,
    },
    /// Yield through a tail call (`return coroutine.yield(...)`) or from
    /// the entry C function: resume performs a *return* of the resume args
    /// from `slot` in the frame at `base`.
    Return { base: usize, slot: usize },
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
    _max_stack: usize,
    pub base: usize,
    pub top: usize,
    /// Open upvalues pointing into this thread's stack, kept sorted by slot
    /// (descending), mirroring LuaJIT's `L->openupval` list.
    pub openuv: Vec<GcPtr<crate::func::Upval>>,
    /// The pending error object (`LuaError::Runtime`).
    pub errval: LuaValue,
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
        let initial_len = 1024;
        LuaState {
            g,
            is_main,
            stack: {
                let mut v = Vec::with_capacity(initial_len);
                v.resize(initial_len, LuaValue::NIL);
                v
            },
            _max_stack: max_stack,
            base: 0,
            top: 0,
            openuv: Vec::new(),
            errval: LuaValue::NIL,
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
        }
    }

    /// Ensure the stack can hold at least `need` slots (absolute index).
    /// Grows dynamically by doubling, capped at `STACK_MAX`.  Open upvalue
    /// pointers that reference the old stack are patched after a reallocation.
    /// Also serves as a GC check point so loops that push values (e.g. table
    /// stores, string concatenation) eventually trigger collection.
    #[inline]
    pub fn stack_ensure(&mut self, need: usize) {
        if need > self.stack.len() {
            let new_len = (self.stack.len() * 2).max(need + 16).min(self._max_stack);
            assert!(new_len <= self._max_stack, "stack overflow");
            let old_ptr = self.stack.as_mut_ptr();
            let old_len = self.stack.len();
            self.stack.resize(new_len, LuaValue::NIL);
            let new_ptr = self.stack.as_mut_ptr();
            if old_ptr != new_ptr {
                let delta_bytes = new_ptr as isize - old_ptr as isize;
                if delta_bytes != 0 {
                    for &uv in &self.openuv {
                        let uv_mut = uv.as_mut();
                        let p = uv_mut.value_ptr() as *mut LuaValue;
                        if p >= old_ptr && p < unsafe { old_ptr.add(old_len) } {
                            let new_p = unsafe { (p as *mut u8).offset(delta_bytes) as *mut LuaValue };
                            uv_mut.repoint(unsafe { NonNull::new_unchecked(new_p) });
                        }
                    }
                }
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
            Some(crate::func::GcFunc::C(cc)) => cc.upvals.get(i).copied().unwrap_or(LuaValue::NIL),
            _ => LuaValue::NIL,
        }
    }

    /// `lua_setupvalue`: overwrite the i-th upvalue of the currently
    /// running C closure.
    pub fn set_upvalue(&mut self, i: usize, v: LuaValue) {
        let f = self.stack[self.base - 2];
        if let Some(gf) = f.as_func()
            && let crate::func::GcFunc::C(cc) = gf.as_mut()
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
    pub fn runtime_error(&mut self, msg: impl AsRef<[u8]>) -> crate::err::LuaError {
        let mut full = msg.as_ref().to_vec();

        let mut slot = self.base;
        for _ in 0..8 {
            if slot < 2 {
                break;
            }
            let func = self.stack[slot - 2];
            let link_bits = self.stack[slot - 1].to_bits();
            if let Some(fv) = func.as_func() {
                match fv.as_ref() {
                    crate::func::GcFunc::Lua(cl) => {
                        let pt = cl.proto.as_ref();
                        let pc = self.debug_pc.saturating_sub(1).min(pt.lines.len().saturating_sub(1));
                        let line = if pc < pt.lines.len() {
                            pt.lines[pc] as usize
                        } else {
                            pt.firstline as usize
                        };
                        let src = pt.source.and_then(|sid| {
                            self.heap().strings.try_lookup(sid).map(|_ptr| {
                                let bytes = self.heap().strings.get(sid);
                                // Strip leading '@' or '=' for display.
                                let s = if bytes.starts_with(&[b'@']) || bytes.starts_with(&[b'=']) {
                                    &bytes[1..]
                                } else {
                                    bytes
                                };
                                String::from_utf8_lossy(s).into_owned()
                            })
                        }).unwrap_or_else(|| "=?".to_string());
                        let msg_str = String::from_utf8_lossy(&full);
                        full = format!("{}:{}: {}", src, line, msg_str).into_bytes();
                        break;
                    }
                    crate::func::GcFunc::C(_) => {
                        // Walk to caller via frame link.
                        // FRAME_LUA=0 means the link encodes the caller's base.
                        let ft = link_bits & FRAME_TYPE_MASK;
                        if ft == 0 /* FRAME_LUA */ {
                            slot = (link_bits >> 3) as usize;
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
        crate::err::LuaError::Runtime
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

impl Lua {
    pub fn new() -> Box<Lua> {
        let mut lua = Box::new(Lua {
            g: Box::new(GlobalState::new()),
        });
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

impl Default for Box<Lua> {
    fn default() -> Box<Lua> {
        Lua::new()
    }
}

pub fn load(l: &mut LuaState, src: Vec<u8>, chunkname: &str) -> Result<LuaValue, String> {
    let g = l.global();
    let mut parser = crate::parse::Parser::new(src, chunkname.to_string(), &mut g.heap.strings);
    // Suppress panic output for compile errors (caught by catch_unwind).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parser.parse()));
    std::panic::set_hook(prev_hook);
    let mut proto = match result {
        Ok(p) => p,
        Err(e) => {
            let msg = if let Some(ce) = e.downcast_ref::<crate::lex::CompileError>() {
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
    };

    debug_assert!(proto.uv.is_empty(), "main chunk must have no upvalues");
    if !chunkname.is_empty() && !chunkname.starts_with('=') {
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
        if matches!(proto.kgc[i], crate::proto::KGc::Proto(_)) {
            let taken = std::mem::replace(&mut proto.kgc[i], crate::proto::KGc::Str(0));
            if let crate::proto::KGc::Proto(child) = taken {
                let r = register_proto(heap, *child);
                // Propagate parent source to child protos that don't have one.
                if r.as_ref().source.is_none() {
                    r.as_mut().source = proto.source;
                }
                proto.kgc[i] = crate::proto::KGc::ProtoRef(r);
            }
        }
    }
    proto.kstrv = proto
        .kgc
        .iter()
        .map(|k| match k {
            crate::proto::KGc::Str(sid) => {
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
                assert!(c.proto.as_ref().source.is_some(), "main chunk should have source");
                let pt = c.proto.as_ref();
                for k in &pt.kgc {
                    if let crate::proto::KGc::ProtoRef(child) = k {
                        assert!(child.as_ref().source.is_some(), "child proto should inherit source");
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
        fn dummy(_l: &mut LuaState) -> crate::err::LuaResult<i32> {
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
