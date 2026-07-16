use std::collections::HashMap;

use crate::bc::{BCLine, BCPos, BCReg, NO_JMP};
use crate::lex::StrId;
use crate::proto::KGc;

/// A lexical scope block within a function (loop/block/goto tracking).
#[derive(Clone, Copy)]
pub(crate) struct FuncScope {
    pub vstart: u32,
    pub nactvar: u8,
    pub flags: u8,
}

/// Per-function compiler state, corresponding to LuaJIT's `FuncState`.
///
/// Holds the constant tables, register/scope bookkeeping and upvalue maps for
/// the function currently being compiled. The bytecode itself lives in the
/// shared `bcstack` owned by the `Parser`, addressed via `bcbase`.
pub(crate) struct FuncState {
    /// Number constants, in emission order.
    pub kn: Vec<f64>,
    /// Deduplication map: number bit pattern -> `kn` index.
    pub kn_map: HashMap<u64, u32>,
    /// Collectable constants, in emission order.
    pub kgc: Vec<KGc>,
    /// Deduplication map: interned string -> `kgc` index.
    pub kgc_map: HashMap<StrId, u32>,
    /// Next bytecode position (relative to `bcbase`).
    pub pc: BCPos,
    /// Position of the last jump target (for peephole safety).
    pub lasttarget: BCPos,
    /// Pending jump list to be patched to the next instruction.
    pub jpc: BCPos,
    /// First free register.
    pub freereg: BCReg,
    /// Number of active local variables.
    pub nactvar: BCReg,
    /// First line of this function's definition.
    pub linedefined: BCLine,
    /// Base offset of this function's bytecode within the shared stack.
    pub bcbase: usize,
    /// Base of the variable stack for this function.
    pub vbase: usize,
    pub flags: u8,
    pub numparams: u8,
    pub framesize: u8,
    pub nuv: u8,
    /// Map from register slot to variable-stack index.
    pub varmap: [u16; 250],
    /// Map from upvalue index to variable-stack index.
    pub uvmap: [u16; 60],
    /// Temporary upvalue map (resolved into `Proto::uv`).
    pub uvtmp: [u16; 60],
    /// Stack of open scope blocks.
    pub scopes: Vec<FuncScope>,
}

impl FuncState {
    pub fn new(vbase: usize) -> FuncState {
        FuncState {
            kn: Vec::new(),
            kn_map: HashMap::new(),
            kgc: Vec::new(),
            kgc_map: HashMap::new(),
            pc: 0,
            lasttarget: 0,
            jpc: NO_JMP,
            freereg: 0,
            nactvar: 0,
            linedefined: 0,
            bcbase: 0,
            vbase,
            flags: 0,
            numparams: 0,
            framesize: 1,
            nuv: 0,
            varmap: [0; 250],
            uvmap: [0; 60],
            uvtmp: [0; 60],
            scopes: Vec::new(),
        }
    }
}
