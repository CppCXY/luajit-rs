use crate::gc::GcPtr;
use crate::proto::Proto;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

/// An upvalue object, corresponding to LuaJIT's `GCupval`.
///
/// While the enclosing function's activation record is still live, the
/// upvalue is *open* and refers to a stack slot; once the scope exits it is
/// *closed* and owns the value itself (LuaJIT flips `uv->v` from a stack
/// pointer to `&uv->tv`).
pub struct Upval {
    /// Immutable upvalue (from `PROTO_UV_IMMUTABLE`).
    pub immutable: bool,
    pub state: UpvalState,
}

pub enum UpvalState {
    /// Open: refers to an absolute slot in the thread's value stack.
    Open(usize),
    /// Closed: the value lives in the upvalue itself.
    Closed(LuaValue),
}

impl Upval {
    pub fn new_open(slot: usize, immutable: bool) -> Upval {
        Upval {
            immutable,
            state: UpvalState::Open(slot),
        }
    }

    /// `func_emptyuv`: an empty, already-closed upvalue holding nil.
    pub fn new_closed(immutable: bool) -> Upval {
        Upval {
            immutable,
            state: UpvalState::Closed(LuaValue::NIL),
        }
    }

    pub fn is_open(&self) -> bool {
        matches!(self.state, UpvalState::Open(_))
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
