//! C data objects — GC-managed raw C value storage.
//!
//! Every FFI value (pointer, integer, struct, etc.) is stored as a `CData`
//! object. The payload bytes are boxed on the heap since sizes vary per C type.

/// C type ID — indexes into the `CTState` type table.
pub type CTypeID = u32;

// ---------------------------------------------------------------------------
// C data object
// ---------------------------------------------------------------------------

/// GC-managed cdata object. The payload holds raw C value bytes.
#[derive(Debug, Clone)]
pub struct CData {
    pub ctypeid: CTypeID,
    pub data: Box<[u8]>,
}

impl CData {
    pub fn new(ctypeid: CTypeID, sz: usize) -> Self {
        CData { ctypeid, data: vec![0u8; sz].into_boxed_slice() }
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
