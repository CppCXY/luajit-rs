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

    /// Repoint the value pointer (used after stack reallocation).
    pub fn repoint(&mut self, new_ptr: NonNull<LuaValue>) {
        self.v = new_ptr;
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
    pub upvals: Upvals,
}

/// Inline capacity for a Lua closure's upvalue references.
const INLINE_UV: usize = 4;

/// Upvalue references of a Lua closure. Up to `INLINE_UV` are stored inline
/// (no heap allocation); beyond that they spill to a heap `Vec`. Mirrors
/// LuaJIT's inline `uvptr` array inside `GCfuncL` — a hot
/// `function() ... end` loop then makes no per-closure upvalue allocation.
pub enum Upvals {
    /// Few upvalues: stored inline, no heap allocation.
    Inline {
        n: u8,
        uv: [GcPtr<Upval>; INLINE_UV],
    },
    /// More than `INLINE_UV` upvalues: spilled to the heap.
    Heap(Vec<GcPtr<Upval>>),
}

impl Upvals {
    pub fn empty() -> Upvals {
        Upvals::Inline {
            n: 0,
            uv: [GcPtr::new(NonNull::dangling()); INLINE_UV],
        }
    }
    pub fn from_vec(v: Vec<GcPtr<Upval>>) -> Upvals {
        if v.len() <= INLINE_UV {
            let mut uv = [GcPtr::new(NonNull::dangling()); INLINE_UV];
            for (i, &p) in v.iter().enumerate() {
                uv[i] = p;
            }
            Upvals::Inline { n: v.len() as u8, uv }
        } else {
            Upvals::Heap(v)
        }
    }
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Upvals::Inline { n, .. } => *n as usize,
            Upvals::Heap(v) => v.len(),
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn get(&self, i: usize) -> Option<&GcPtr<Upval>> {
        match self {
            Upvals::Inline { uv, .. } => uv.get(i),
            Upvals::Heap(v) => v.get(i),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &GcPtr<Upval>> {
        match self {
            Upvals::Inline { n, uv } => uv[..*n as usize].iter(),
            Upvals::Heap(v) => v.iter(),
        }
    }
    pub fn as_slice(&self) -> &[GcPtr<Upval>] {
        match self {
            Upvals::Inline { n, uv } => &uv[..*n as usize],
            Upvals::Heap(v) => v,
        }
    }
    pub fn push(&mut self, v: GcPtr<Upval>) {
        match self {
            Upvals::Inline { n, uv } => {
                let ni = *n as usize;
                if ni < INLINE_UV {
                    uv[ni] = v;
                    *n += 1;
                } else {
                    let mut heap = uv[..INLINE_UV].to_vec();
                    heap.push(v);
                    *self = Upvals::Heap(heap);
                }
            }
            Upvals::Heap(heap) => heap.push(v),
        }
    }
}

impl Default for Upvals {
    fn default() -> Self {
        Self::empty()
    }
}

impl std::ops::Index<usize> for Upvals {
    type Output = GcPtr<Upval>;
    fn index(&self, i: usize) -> &GcPtr<Upval> {
        &self.as_slice()[i]
    }
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
