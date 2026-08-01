//! C data objects — GC-managed raw C value storage.
//!
//! Every FFI value (pointer, integer, struct, etc.) is stored as a `CData`
//! object. The payload bytes are boxed on the heap since sizes vary per C type.
//!
//! Pointer arithmetic (`a + i`) produces an *alias* cdata: `base` refers to
//! the storage cdata and `offset` gives the byte delta. Indexing and pointer
//! difference resolve through the alias, so writes through the pointer are
//! visible in the original object (the cdata pool is never swept, so the
//! `base` reference cannot dangle).

/// C type ID — indexes into the `CTState` type table.
pub type CTypeID = u32;

use crate::gc::GcPtr;

// ---------------------------------------------------------------------------
// C data object
// ---------------------------------------------------------------------------

/// GC-managed cdata object. The payload holds raw C value bytes.
#[derive(Debug, Clone)]
pub struct CData {
    pub ctypeid: CTypeID,
    pub data: Box<[u8]>,
    /// For pointer-arith aliases: the storage cdata this one points into.
    pub base: Option<GcPtr<CData>>,
    /// Byte offset of this alias within `base` (or within `data` itself
    /// when `base` is `None`).
    pub offset: i64,
}

/// Resolve the storage bytes a cdata refers to: follows the alias chain.
/// Returns `(byte_offset, storage)`.
pub fn resolve_cdata(cd: &CData) -> (i64, &CData) {
    let mut cur = cd;
    let mut off = 0i64;
    loop {
        if let Some(b) = cur.base {
            let b = b.as_ref();
            off += cur.offset;
            cur = b;
        } else {
            return (off, cur);
        }
    }
}

/// Like `resolve_cdata`, returning the storage's `GcPtr` (for writes).
pub fn resolve_ptr(cd: GcPtr<CData>) -> (i64, GcPtr<CData>) {
    let mut cur = cd;
    let mut off = 0i64;
    loop {
        let c = cur.as_ref();
        if let Some(b) = c.base {
            off += c.offset;
            cur = b;
        } else {
            return (off, cur);
        }
    }
}

impl CData {
    pub fn new(ctypeid: CTypeID, sz: usize) -> Self {
        CData {
            ctypeid,
            data: vec![0u8; sz].into_boxed_slice(),
            base: None,
            offset: 0,
        }
    }

    /// Read a pointer from offset 0. Supports 32-bit ptrs on 64-bit.
    pub fn get_ptr(&self) -> usize {
        match self.data.len() {
            0 => 0,
            4 => u32::from_le_bytes(self.data[..4].try_into().unwrap()) as usize,
            _ => usize::from_ne_bytes({
                let mut b = [0u8; std::mem::size_of::<usize>()];
                let n = self.data.len().min(b.len());
                b[..n].copy_from_slice(&self.data[..n]);
                b
            }),
        }
    }

    /// Write a pointer at offset 0.
    pub fn set_ptr(&mut self, p: usize) {
        match self.data.len() {
            0 => {}
            4 => self.data[..4].copy_from_slice(&(p as u32).to_le_bytes()),
            _ => {
                let bytes = p.to_ne_bytes();
                let n = self.data.len().min(bytes.len());
                self.data[..n].copy_from_slice(&bytes[..n]);
            }
        }
    }
}
