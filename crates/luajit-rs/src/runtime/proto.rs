use crate::bc::{BCIns, BCLine};
use crate::gc::GcPtr;
use crate::lex::StrId;
use crate::runtime::cdata::CData;
use crate::table::LuaTable;
use crate::value::LuaValue;

/// A collectable constant referenced from a prototype's `kgc` array.
pub enum KGc {
    Str(StrId),
    /// A child prototype, as produced by the compiler.
    Proto(Box<Proto>),
    /// A child prototype after the tree has been registered in the heap
    /// (see `GcHeap::alloc_proto`).
    ProtoRef(GcPtr<Proto>),
    /// A compiler-owned template table.  Replaced by `TableRef` during
    /// `register_proto`.
    Table(Box<LuaTable>),
    /// Template table in the GC pool (post-registration).
    TableRef(GcPtr<LuaTable>),
    /// C data constant (for LL/ULL suffixed integer literals, etc.).
    CData(Box<CData>),
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
    /// String constants resolved to values, indexed like `kgc` (nil for
    /// non-string entries). This is the interpreter's KBASE view: KSTR /
    /// GGET / TGETS load one slot instead of chasing `kgc` -> interner.
    /// Filled by `register_proto`; the GC marks these through `kgc`.
    pub kstrv: Vec<LuaValue>,
    /// Upvalue references: `PROTO_UV_LOCAL | slot` or parent upvalue index.
    pub uv: Vec<u16>,
    pub flags: u8,
    pub numparams: u8,
    pub framesize: u8,
    pub firstline: BCLine,
    pub numline: BCLine,
    /// Upvalue names for debug info and listings.
    pub uvnames: Vec<String>,
    /// Chunk name / source identifier (from load chunkname argument).
    pub source: Option<StrId>,
}

impl Proto {
    /// Approximate heap footprint in bytes, for GC accounting.
    pub fn gc_size(&self) -> usize {
        std::mem::size_of::<Proto>()
            + self.bc.capacity() * 4
            + self.lines.capacity() * 4
            + self.kgc.capacity() * std::mem::size_of::<KGc>()
            + self.kn.capacity() * 8
            + self.kstrv.capacity() * 8
            + self.uv.capacity() * 2
    }
}

/// Prototype flags (subset of LuaJIT's PROTO_* used by the compiler).
pub const PROTO_CHILD: u8 = 0x01;
pub const PROTO_VARARG: u8 = 0x02;
#[allow(dead_code)]
pub const PROTO_FFI: u8 = 0x04;
/// JIT disabled for this function.
pub const PROTO_NOJIT: u8 = 0x08;
/// Patched bytecode with ILOOP etc. (blacklisted by the trace compiler).
pub const PROTO_ILOOP: u8 = 0x10;
pub const PROTO_HAS_RETURN: u8 = 0x20;
pub const PROTO_FIXUP_RETURN: u8 = 0x40;
pub const PROTO_BITOP: u8 = 0x80;

pub const PROTO_UV_LOCAL: u16 = 0x8000;
pub const PROTO_UV_IMMUTABLE: u16 = 0x4000;
