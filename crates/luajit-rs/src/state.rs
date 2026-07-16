use std::ptr::NonNull;

use crate::func::{CClosure, CFunction, GcFunc, LuaClosure};
use crate::gc::{GcPtr, Pool};
use crate::proto::Proto;
use crate::string::{Interner, StrId};
use crate::table::LuaTable;
use crate::value::{GcRef, LuaValue, LJ_TFUNC, LJ_TTAB};

/// The GC heap: stable-address object pools.
///
/// Every collectable type lives in its own `Pool`, which allocates objects in
/// fixed pages so their addresses never move (a `LuaValue` stores the raw
/// pointer in its 47-bit payload). This is the placement layer the future
/// collector will sweep.
#[derive(Default)]
pub struct GcHeap {
    pub strings: Interner,
    pub protos: Pool<Proto>,
    pub tables: Pool<LuaTable>,
    pub funcs: Pool<GcFunc>,
    pub upvals: Pool<crate::func::Upval>,
}

impl GcHeap {
    pub fn alloc_table(&mut self, t: LuaTable) -> GcPtr<LuaTable> {
        self.tables.alloc(t)
    }

    pub fn alloc_proto(&mut self, p: Proto) -> GcPtr<Proto> {
        self.protos.alloc(p)
    }

    pub fn alloc_func(&mut self, f: GcFunc) -> GcPtr<GcFunc> {
        self.funcs.alloc(f)
    }

    pub fn alloc_upval(&mut self, uv: crate::func::Upval) -> GcPtr<crate::func::Upval> {
        self.upvals.alloc(uv)
    }

    pub fn intern(&mut self, s: &[u8]) -> StrId {
        self.strings.intern(s)
    }

    /// A `LuaValue` for an interned string id.
    pub fn str_value(&self, sid: StrId) -> LuaValue {
        LuaValue::string(self.strings.lookup_ptr(sid))
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
    /// The main thread. Set once the owning [`Lua`] is pinned. The interpreter
    /// entry points use this when no explicit thread is supplied.
    main: Option<StateRef>,
}

impl GlobalState {
    fn new() -> GlobalState {
        let mut heap = GcHeap::default();
        let globals = heap.alloc_table(LuaTable::new(0, 1));
        let registry = heap.alloc_table(LuaTable::new(0, 1));
        GlobalState {
            heap,
            globals,
            registry,
            basemt: [None; ITYPE_COUNT],
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
}

/// A wrapped raw pointer to a [`LuaState`] (used for the stored main thread
/// and for thread `LuaValue`s).
#[derive(Clone, Copy)]
pub struct StateRef(NonNull<LuaState>);

impl StateRef {
    #[allow(clippy::mut_from_ref)]
    pub fn get<'a>(self) -> &'a mut LuaState {
        unsafe { &mut *self.0.as_ptr() }
    }
}

/// A call frame recording where to resume the caller on return.
pub struct Frame {
    pub func: LuaValue,
    pub base: usize,
    pub return_pc: usize,
}

/// A Lua execution thread, corresponding to LuaJIT's `lua_State`.
///
/// Owns its value stack and call frames, and holds a back-pointer to the
/// shared [`GlobalState`]. Threads are themselves owned by the top-level
/// [`Lua`] object.
pub struct LuaState {
    g: GlobalRef,
    is_main: bool,
    pub stack: Vec<LuaValue>,
    pub base: usize,
    pub top: usize,
    pub frames: Vec<Frame>,
}

impl LuaState {
    /// Create a thread bound to `g`. `is_main` marks the primary thread.
    /// Mirrors LuaJIT, where a `lua_State` always carries `G(L)`.
    pub fn new(g: GlobalRef, is_main: bool) -> LuaState {
        LuaState {
            g,
            is_main,
            stack: Vec::new(),
            base: 0,
            top: 0,
            frames: Vec::new(),
        }
    }

    pub fn global(&self) -> &mut GlobalState {
        self.g.get()
    }

    pub fn heap(&self) -> &mut GcHeap {
        &mut self.g.get().heap
    }

    pub fn is_main(&self) -> bool {
        self.is_main
    }

    pub fn push(&mut self, v: LuaValue) {
        if self.top == self.stack.len() {
            self.stack.push(v);
        } else {
            self.stack[self.top] = v;
        }
        self.top += 1;
    }

    pub fn pop(&mut self) -> LuaValue {
        debug_assert!(self.top > 0);
        self.top -= 1;
        self.stack[self.top]
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
/// heap) and of every [`LuaState`] thread. Everything else refers to these
/// through wrapped raw pointers, so their addresses stay fixed.
pub struct Lua {
    g: Box<GlobalState>,
    threads: Vec<Box<LuaState>>,
}

impl Lua {
    pub fn new() -> Box<Lua> {
        let mut lua = Box::new(Lua {
            g: Box::new(GlobalState::new()),
            threads: Vec::new(),
        });
        let gref = GlobalRef(NonNull::from(&*lua.g));
        let mut main = Box::new(LuaState::new(gref, true));
        let main_ref = StateRef(NonNull::from(&mut *main));
        lua.threads.push(main);
        lua.g.main = Some(main_ref);
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
        let mut t = Box::new(LuaState::new(gref, false));
        let r = StateRef(NonNull::from(&mut *t));
        self.threads.push(t);
        r
    }
}

impl Default for Box<Lua> {
    fn default() -> Box<Lua> {
        Lua::new()
    }
}

/// Load a Lua source chunk: compile it to bytecode, register the prototype in
/// the heap and build the top-level vararg closure. Returns the closure value
/// (not yet executed).
pub fn load(l: &mut LuaState, src: Vec<u8>, chunkname: &str) -> Result<LuaValue, String> {
    let g = l.global();
    let strs = std::mem::take(&mut g.heap.strings);
    let parser = crate::parse::Parser::with_interner(src, chunkname.to_string(), strs);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || parser.parse()));
    let (proto, strs) = match result {
        Ok(out) => out,
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
            g.heap.strings = Interner::default();
            return Err(msg);
        }
    };
    g.heap.strings = strs;

    debug_assert!(proto.uv.is_empty(), "main chunk must have no upvalues");
    let proto_ref = g.heap.alloc_proto(proto);
    let env = g.globals;
    let fref = g.heap.alloc_func(GcFunc::Lua(LuaClosure {
        proto: proto_ref,
        env,
        upvals: Vec::new(),
    }));
    Ok(LuaValue::func(fref))
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
    fn load_reports_syntax_errors() {
        let mut lua = Lua::new();
        let err = load(lua.main(), b"local = ".to_vec(), "@bad").unwrap_err();
        assert!(!err.is_empty());
        let f = load(lua.main(), b"return 1".to_vec(), "@ok").unwrap();
        assert!(f.is_func());
    }

    #[test]
    fn register_and_lookup_global() {
        fn dummy(_l: &mut LuaState) -> u32 {
            0
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
