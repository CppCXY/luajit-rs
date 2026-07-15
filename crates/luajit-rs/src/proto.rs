use crate::bc::{BCIns, BCLine};
use crate::lex::StrId;
use crate::table::LuaTable;

/// A collectable constant referenced from a prototype's `kgc` array.
pub enum KGc {
    Str(StrId),
    Proto(Box<Proto>),
    Table(LuaTable),
}

/// A function prototype: the output of the bytecode compiler, corresponding
/// to LuaJIT's `GCproto`.
pub struct Proto {
    /// Bytecode. `bc[0]` is the FUNCF/FUNCV header.
    pub bc: Vec<BCIns>,
    /// Absolute source line per instruction.
    pub lines: Vec<BCLine>,
    /// Collectable constants (strings, child prototypes, template tables).
    pub kgc: Vec<KGc>,
    /// Number constants.
    pub kn: Vec<f64>,
    /// Upvalue references: `PROTO_UV_LOCAL | slot` or parent upvalue index.
    pub uv: Vec<u16>,
    pub flags: u8,
    pub numparams: u8,
    pub framesize: u8,
    pub firstline: BCLine,
    pub numline: BCLine,
    /// Upvalue names for debug info and listings.
    pub uvnames: Vec<String>,
}

/// Prototype flags (subset of LuaJIT's PROTO_* used by the compiler).
pub const PROTO_CHILD: u8 = 0x01;
pub const PROTO_VARARG: u8 = 0x02;
#[allow(dead_code)]
pub const PROTO_FFI: u8 = 0x04;
pub const PROTO_HAS_RETURN: u8 = 0x20;
pub const PROTO_FIXUP_RETURN: u8 = 0x40;
pub const PROTO_BITOP: u8 = 0x80;

pub const PROTO_UV_LOCAL: u16 = 0x8000;
pub const PROTO_UV_IMMUTABLE: u16 = 0x4000;
