use crate::gc::GcPtr;
use crate::proto::Proto;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;
use std::ptr::NonNull;

/// An upvalue object, corresponding to LuaJIT's `GCupval`.
///
/// Exactly like LuaJIT/PUC Lua, `v` always points at the live value:
/// while *open* it points into the owning thread's value stack (stable —
/// stacks never reallocate); once *closed* it points at the inline `tv`
/// field (the pool slot address is stable too). This makes cross-thread
/// upvalue access (a closure created on one coroutine running on another)
/// a plain pointer dereference, with no thread bookkeeping.
pub struct Upval {
    /// Pointer to the current value location (`uv->v`).
    /// `NonNull::dangling()` marks a freshly built closed upvalue whose
    /// pool address is not known yet; `GcHeap::alloc_upval` fixes it up.
    v: NonNull<LuaValue>,
    /// Inline storage used once closed (`uv->tv`).
    tv: LuaValue,
    /// Immutable upvalue (from `PROTO_UV_IMMUTABLE`).
    pub immutable: bool,
}

impl Upval {
    /// An open upvalue referring to a stack slot.
    pub fn new_open(slot: NonNull<LuaValue>, immutable: bool) -> Upval {
        Upval {
            v: slot,
            tv: LuaValue::NIL,
            immutable,
        }
    }

    /// `func_emptyuv`: an empty, already-closed upvalue holding nil.
    /// The `v` pointer is patched to `&tv` after pool insertion.
    pub fn new_closed(immutable: bool) -> Upval {
        Upval {
            v: NonNull::dangling(),
            tv: LuaValue::NIL,
            immutable,
        }
    }

    /// Fix-up after pool insertion: point a closed upvalue at its own
    /// (now stable) `tv` field.
    pub(crate) fn init_closed(&mut self) {
        if self.v == NonNull::dangling() {
            self.v = NonNull::from(&mut self.tv);
        }
    }

    #[inline]
    pub fn get(&self) -> LuaValue {
        unsafe { *self.v.as_ptr() }
    }

    #[inline]
    pub fn set(&mut self, val: LuaValue) {
        unsafe { *self.v.as_ptr() = val }
    }

    /// The raw location, used by `find_upval`'s identity check and the
    /// stack-level comparison in `close_upvals`.
    #[inline]
    pub fn value_ptr(&self) -> *mut LuaValue {
        self.v.as_ptr()
    }

    #[inline]
    pub fn is_open(&self) -> bool {
        !std::ptr::eq(self.v.as_ptr().cast_const(), &self.tv)
    }

    /// Close: copy the stack value into the inline slot and repoint
    /// (`lj_func_closeuv`'s flip of `uv->v` to `&uv->tv`).
    pub fn close(&mut self) {
        if self.is_open() {
            self.tv = self.get();
            self.v = NonNull::from(&mut self.tv);
        }
    }
}

/// A Lua closure, corresponding to LuaJIT's `GCfuncL`.
pub struct LuaClosure {
    /// The prototype this closure instantiates.
    pub proto: GcPtr<Proto>,
    /// Environment table for global accesses (GGET/GSET).
    pub env: GcPtr<LuaTable>,
    /// Upvalue objects (`uvptr`), shared between closures.
    pub upvals: Vec<GcPtr<Upval>>,
}

/// A builtin implemented in Rust, corresponding to LuaJIT's `GCfuncC`.
/// Receives the calling thread with arguments already on its stack frame
/// (`base..top`). Returns `Ok(n)` having left `n` results at the frame base,
/// or `Err(LuaError)` (error object / yield count are on the `LuaState`).
pub type CFunction = fn(&mut LuaState) -> crate::err::LuaResult<i32>;

/// A C-function closure, corresponding to LuaJIT's `GCfuncC`.
pub struct CClosure {
    pub f: CFunction,
    pub env: GcPtr<LuaTable>,
    pub upvals: Vec<LuaValue>,
}

/// A function object (`GCfunc`): either a Lua closure or a builtin.
pub enum GcFunc {
    Lua(LuaClosure),
    C(CClosure),
}

impl GcFunc {
    pub fn env(&self) -> GcPtr<LuaTable> {
        match self {
            GcFunc::Lua(l) => l.env,
            GcFunc::C(c) => c.env,
        }
    }
}
