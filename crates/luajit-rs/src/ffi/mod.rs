//! LuaJIT FFI (Foreign Function Interface).
//!
//! Port of LuaJIT's FFI subsystem: C type declarations, cdata objects,
//! C function calls, callbacks, and JIT support.
//!
//! Module layout follows LuaJIT's source organisation:
//! * `mod.rs`     — Core type definitions (CTInfo, CType, CTState)
//! * `parser.rs`  — C declaration parser (ffi.cdef)
//! * `ccall.rs`   — C function calling convention [TODO]
//! * `lib.rs`     — Lua FFI library API (ffi.new, ffi.cast, etc.) [TODO]
//!
//! CData objects live in `crate::runtime::cdata` — use `LuaValue::cdata()`
//! and `LuaValue::as_cdata()` for construction / access.
//! Reference: LuaJIT/src/lj_ctype.h, lj_cdata.h, lj_cparse.h

pub mod parser;
pub mod lib;

// ---------------------------------------------------------------------------
// C type numbers (enum from lj_ctype.h)
// ---------------------------------------------------------------------------

/// C type number: top 4 bits of CTInfo.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CT {
    Num = 0,        // Integer or floating-point numbers
    Struct = 1,     // Struct or union
    Ptr = 2,        // Pointer or reference
    Array = 3,      // Array or complex type
    Void = 4,       // Void type
    Enum = 5,       // Enumeration (also: last type where size holds actual size)
    Func = 6,       // Function
    Typedef = 7,    // Typedef
    Attrib = 8,     // Miscellaneous attributes
    // Internal element types
    Field = 9,      // Struct/union field or function parameter
    Bitfield = 10,  // Struct/union bitfield
    Constval = 11,  // Constant value
    Extern = 12,    // External reference
    Kw = 13,        // Keyword
}

// ---------------------------------------------------------------------------
// CTInfo: 32-bit type info packed word
// ---------------------------------------------------------------------------

/// C type info flags.
pub mod ctinfo {
    pub const BOOL: u32 = 0x0800_0000;
    pub const FP: u32 = 0x0400_0000;
    pub const CONST: u32 = 0x0200_0000;
    pub const VOLATILE: u32 = 0x0100_0000;
    pub const UNSIGNED: u32 = 0x0080_0000;
    pub const LONG: u32 = 0x0040_0000;
    pub const VLA: u32 = 0x0010_0000;
    pub const REF: u32 = 0x0080_0000; // Same bit as UNSIGNED — reuse depends on CT
    pub const VECTOR: u32 = 0x0800_0000;
    pub const COMPLEX: u32 = 0x0400_0000;
    pub const UNION: u32 = 0x0080_0000;
    pub const VARARG: u32 = 0x0080_0000;
    pub const SSEREGPARM: u32 = 0x0040_0000;

    pub const QUAL: u32 = CONST | VOLATILE;
    pub const UCHAR: u32 = if (i8::MIN as u8 as i8) < 0 { 0 } else { UNSIGNED };

    // Bitfield masks and shifts
    pub const MASK_CID: u32 = 0x0000_FFFF;
    pub const MASK_NUM: u32 = 0xF000_0000;
    pub const SHIFT_NUM: u32 = 28;
    pub const MASK_ALIGN: u32 = 15;
    pub const SHIFT_ALIGN: u32 = 16;
    pub const MASK_ATTRIB: u32 = 255;
    pub const SHIFT_ATTRIB: u32 = 16;
    pub const MASK_CCONV: u32 = 3;
    pub const SHIFT_CCONV: u32 = 16;
    pub const MASK_REGPARM: u32 = 3;
    pub const SHIFT_REGPARM: u32 = 18;

    // Bitfield positions
    pub const MASK_BITPOS: u32 = 127;
    pub const MASK_BITBSZ: u32 = 127;
    pub const MASK_BITCSZ: u32 = 127;
    pub const SHIFT_BITPOS: u32 = 0;
    pub const SHIFT_BITBSZ: u32 = 8;
    pub const SHIFT_BITCSZ: u32 = 16;
    pub const BSZ_MAX: u32 = 32;
    pub const BSZ_FIELD: u32 = 127;

    // Parser-only bitfields
    pub const MASK_VSIZEP: u32 = 15;
    pub const SHIFT_VSIZEP: u32 = 4;
    pub const MASK_MSIZEP: u32 = 255;
    pub const SHIFT_MSIZEP: u32 = 8;

    // Special sizes
    pub const SIZE_INVALID: u32 = 0xFFFF_FFFF;

    // Pre-computed alignment constants (shifted for CTInfo packing).
    pub const ALIGN0: u32 = 0;
    pub const ALIGN1: u32 = 1 << SHIFT_ALIGN;
    pub const ALIGN2: u32 = 2 << SHIFT_ALIGN;
    pub const ALIGN3: u32 = 3 << SHIFT_ALIGN;
    pub const ALIGN4: u32 = 4 << SHIFT_ALIGN;

    // Attribute constants.
    pub const ATTR_BAD: u32 = 5 << SHIFT_ATTRIB; // CTA_BAD = 5
}

/// Construct CTInfo: (type_number << 28) | flags.
#[inline]
pub fn ct_info(ct: CT, flags: u32) -> u32 {
    ((ct as u32) << ctinfo::SHIFT_NUM) | flags
}

/// Extract type number from CTInfo.
#[inline]
pub fn ctype_type(info: u32) -> CT {
    let t = (info >> ctinfo::SHIFT_NUM) & 0xF;
    // SAFETY: only 0-13 are valid
    match t {
        0 => CT::Num, 1 => CT::Struct, 2 => CT::Ptr, 3 => CT::Array,
        4 => CT::Void, 5 => CT::Enum, 6 => CT::Func, 7 => CT::Typedef,
        8 => CT::Attrib, 9 => CT::Field, 10 => CT::Bitfield, 11 => CT::Constval,
        12 => CT::Extern, 13 => CT::Kw,
        _ => unreachable!(),
    }
}

#[inline] pub fn ctype_cid(info: u32) -> u32 { info & ctinfo::MASK_CID }
#[inline] pub fn ctype_align(info: u32) -> u32 { (info >> ctinfo::SHIFT_ALIGN) & ctinfo::MASK_ALIGN }
#[inline] pub fn ctype_attrib(info: u32) -> u32 { (info >> ctinfo::SHIFT_ATTRIB) & ctinfo::MASK_ATTRIB }
#[inline] pub fn ctype_bitpos(info: u32) -> u32 { (info >> ctinfo::SHIFT_BITPOS) & ctinfo::MASK_BITPOS }
#[inline] pub fn ctype_bitbsz(info: u32) -> u32 { (info >> ctinfo::SHIFT_BITBSZ) & ctinfo::MASK_BITBSZ }
#[inline] pub fn ctype_bitcsz(info: u32) -> u32 { (info >> ctinfo::SHIFT_BITCSZ) & ctinfo::MASK_BITCSZ }
#[inline] pub fn ctype_cconv(info: u32) -> u32 { (info >> ctinfo::SHIFT_CCONV) & ctinfo::MASK_CCONV }

// Simple type checks
#[inline] pub fn ctype_isnum(info: u32) -> bool { ctype_type(info) == CT::Num }
#[inline] pub fn ctype_isvoid(info: u32) -> bool { ctype_type(info) == CT::Void }
#[inline] pub fn ctype_isptr(info: u32) -> bool { ctype_type(info) == CT::Ptr }
#[inline] pub fn ctype_isarray(info: u32) -> bool { ctype_type(info) == CT::Array }
#[inline] pub fn ctype_isstruct(info: u32) -> bool { ctype_type(info) == CT::Struct }
#[inline] pub fn ctype_isfunc(info: u32) -> bool { ctype_type(info) == CT::Func }
#[inline] pub fn ctype_istypedef(info: u32) -> bool { ctype_type(info) == CT::Typedef }
#[inline] pub fn ctype_isattrib(info: u32) -> bool { ctype_type(info) == CT::Attrib }
#[inline] pub fn ctype_isfield(info: u32) -> bool { ctype_type(info) == CT::Field }
#[inline] pub fn ctype_isbitfield(info: u32) -> bool { ctype_type(info) == CT::Bitfield }
#[inline] pub fn ctype_hassize(info: u32) -> bool { (ctype_type(info) as u32) <= CT::Enum as u32 }

// Combined type+flag checks
#[inline] pub fn ctype_isinteger(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::BOOL | ctinfo::FP)) == ct_info(CT::Num, 0)
}
#[inline] pub fn ctype_isinteger_or_bool(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::FP)) == ct_info(CT::Num, 0)
}
#[inline] pub fn ctype_isbool(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::BOOL)) == ct_info(CT::Num, ctinfo::BOOL)
}
#[inline] pub fn ctype_isfp(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::FP)) == ct_info(CT::Num, ctinfo::FP)
}
#[inline] pub fn ctype_ispointer(info: u32) -> bool {
    (ctype_type(info) as u32 >> 1) == (CT::Ptr as u32 >> 1) // Ptr or Array
}
#[inline] pub fn ctype_isref(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::REF)) == ct_info(CT::Ptr, ctinfo::REF)
}
#[inline] pub fn ctype_isrefarray(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::VECTOR | ctinfo::COMPLEX)) == ct_info(CT::Array, 0)
}
#[inline] pub fn ctype_isvector(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::VECTOR)) == ct_info(CT::Array, ctinfo::VECTOR)
}
#[inline] pub fn ctype_iscomplex(info: u32) -> bool {
    (info & (ctinfo::MASK_NUM | ctinfo::COMPLEX)) == ct_info(CT::Array, ctinfo::COMPLEX)
}

// ---------------------------------------------------------------------------
// Predefined type IDs (matching LuaJIT's CTID_* enum)
// ---------------------------------------------------------------------------

/// Public predefined C type IDs.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CTypeID {
    // From CTTYDEF macro expansion
    None = 0,
    Void,
    CVoid,
    Bool,
    CChar,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Int128,
    UInt128,
    Float,
    Double,
    ComplexFloat,
    ComplexDouble,
    PVoid,
    PCVoid,
    PCChar,
    PUInt8,
    ACChar,
    CTypeIDType, // CTID_CTYPEID — cdata holding a type ID
    Max = 65536,
}

impl CTypeID {
    pub const fn from_u32(v: u32) -> Self {
        if v > 32 { panic!("invalid CTypeID") }
        // SAFETY: 0..32 are valid discriminants
        unsafe { std::mem::transmute(v) }
    }
    pub const fn to_u32(self) -> u32 { self as u32 }
}

/// Platform-dependent type ID aliases.
#[cfg(target_pointer_width = "64")]
pub const CTID_INT_PSZ: CTypeID = CTypeID::Int64;
#[cfg(target_pointer_width = "64")]
pub const CTID_UINT_PSZ: CTypeID = CTypeID::UInt64;
#[cfg(target_pointer_width = "32")]
pub const CTID_INT_PSZ: CTypeID = CTypeID::Int32;
#[cfg(target_pointer_width = "32")]
pub const CTID_UINT_PSZ: CTypeID = CTypeID::UInt32;

// ---------------------------------------------------------------------------
// CType: single entry in the type table (8 bytes per entry in LJ)
// ---------------------------------------------------------------------------

/// A single C type table element.
#[derive(Debug, Clone, Default)]
pub struct CType {
    pub info: u32,
    pub size: u32,
    pub sib: u16,
    pub next: u16,
    pub name: u64, // GCRef<GCstr> — bit pattern, 0 = no name
}

pub const CTHASH_SIZE: usize = 128;

/// C type system state.
pub struct CTState {
    pub tab: Vec<CType>,
    pub top: u32,
    pub miscmap: u64,
    pub hash: [u16; CTHASH_SIZE],
    /// Typedef name → type ID index (for ffi.typeof/ffi.new name lookup).
    pub names: std::collections::HashMap<String, u32>,
}

impl CTState {
    pub fn new() -> Self {
        let mut cts = CTState {
            tab: Vec::with_capacity(256),
            top: 0,
            miscmap: 0,
            hash: [u16::MAX; CTHASH_SIZE],
            names: std::collections::HashMap::new(),
        };
        cts.init_predefined();
        cts
    }

    fn init_predefined(&mut self) {
        use ctinfo::*;

        // Size of long depends on platform
        #[cfg(target_pointer_width = "64")]
        let long_flag: u32 = LONG;
        #[cfg(target_pointer_width = "32")]
        let long_flag: u32 = 0;

        // Each entry: (info, size)
        let predefined: [(u32, i32); 25] = [
            // CTID_None
            (ct_info(CT::Attrib, ATTR_BAD), 0),
            // CTID_Void
            (ct_info(CT::Void, 0), -1),
            // CTID_CVoid
            (ct_info(CT::Void, CONST), -1),
            // CTID_Bool
            (ct_info(CT::Num, BOOL | UNSIGNED), 1),
            // CTID_CChar
            (ct_info(CT::Num, CONST | UCHAR), 1),
            // CTID_Int8
            (ct_info(CT::Num, 0), 1),
            // CTID_UInt8
            (ct_info(CT::Num, UNSIGNED), 1),
            // CTID_Int16
            (ct_info(CT::Num, ALIGN1), 2),
            // CTID_UInt16
            (ct_info(CT::Num, UNSIGNED | ALIGN1), 2),
            // CTID_Int32
            (ct_info(CT::Num, ALIGN2), 4),
            // CTID_UInt32
            (ct_info(CT::Num, UNSIGNED | ALIGN2), 4),
            // CTID_Int64
            (ct_info(CT::Num, long_flag | ALIGN3), 8),
            // CTID_UInt64
            (ct_info(CT::Num, UNSIGNED | long_flag | ALIGN3), 8),
            // CTID_Int128
            (ct_info(CT::Num, ALIGN4), 16),
            // CTID_UInt128
            (ct_info(CT::Num, UNSIGNED | ALIGN4), 16),
            // CTID_Float
            (ct_info(CT::Num, FP | ALIGN2), 4),
            // CTID_Double
            (ct_info(CT::Num, FP | ALIGN3), 8),
            // CTID_ComplexFloat
            (ct_info(CT::Array, COMPLEX | ALIGN2) | CTypeID::Float as u32, 8),
            // CTID_ComplexDouble
            (ct_info(CT::Array, COMPLEX | ALIGN3) | CTypeID::Double as u32, 16),
            // CTID_PVoid
            (ct_info(CT::Ptr, PTR_ALIGN) | CTypeID::Void as u32, PTR_SIZE),
            // CTID_PCVoid
            (ct_info(CT::Ptr, PTR_ALIGN) | CTypeID::CVoid as u32, PTR_SIZE),
            // CTID_PCChar
            (ct_info(CT::Ptr, PTR_ALIGN) | CTypeID::CChar as u32, PTR_SIZE),
            // CTID_PUInt8
            (ct_info(CT::Ptr, PTR_ALIGN) | CTypeID::UInt8 as u32, PTR_SIZE),
            // CTID_ACChar
            (ct_info(CT::Array, CONST) | CTypeID::CChar as u32, -1),
            // CTID_CTYPEID
            (ct_info(CT::Enum, ALIGN2) | CTypeID::Int32 as u32, 4),
        ];

        for (info, size) in &predefined {
            let ctype = CType {
                info: *info,
                size: *size as u32,
                sib: 0,
                next: 0,
                name: 0,
            };
            self.tab.push(ctype);
            self.top += 1;
        }
    }

    pub fn get(&self, id: u32) -> &CType {
        debug_assert!(id > 0 && id < self.top, "bad CTypeID {}", id);
        &self.tab[id as usize]
    }

    pub fn get_mut(&mut self, id: u32) -> &mut CType {
        debug_assert!(id > 0 && id < self.top, "bad CTypeID {}", id);
        &mut self.tab[id as usize]
    }

    pub fn child(&self, id: u32) -> &CType {
        let ct = self.get(id);
        debug_assert!(
            !ctype_isvoid(ct.info) && !ctype_isstruct(ct.info) && !ctype_isbitfield(ct.info),
            "ctype {:08x} has no children", ct.info
        );
        self.get(ctype_cid(ct.info))
    }

    pub fn raw(&self, mut id: u32) -> &CType {
        loop {
            let ct = self.get(id);
            if ctype_isattrib(ct.info) || ctype_istypedef(ct.info) {
                id = ctype_cid(ct.info);
            } else {
                return ct;
            }
        }
    }

    pub fn raw_child(&self, ct: &CType) -> &CType {
        let mut id = ctype_cid(ct.info);
        loop {
            let child = self.get(id);
            if !ctype_isattrib(child.info) { return child; }
            id = ctype_cid(child.info);
        }
    }
}

// Platform-specific constants
#[cfg(target_pointer_width = "64")]
const PTR_SIZE: i32 = 8;
#[cfg(target_pointer_width = "64")]
const PTR_ALIGN: u32 = 3 << ctinfo::SHIFT_ALIGN;
#[cfg(target_pointer_width = "32")]
const PTR_SIZE: i32 = 4;
#[cfg(target_pointer_width = "32")]
const PTR_ALIGN: u32 = 2 << ctinfo::SHIFT_ALIGN;
