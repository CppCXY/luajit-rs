//! GcHeader — 12-byte header at a negative offset from every GC object.
//!
//! Layout:
//! ```text
//!   byte 0:   bits  [age:3][WHITE0:1][WHITE1:1][BLACK:1][rm:2]
//!   byte 1:   kind  (GcObjectKind tag for deallocation)
//!   bytes 2-3: _pad
//!   bytes 4-7: index (position in GcList vec)
//!   bytes 8-11: alloc_size (for GC pacing)
//! ```

use std::cell::Cell;

// ── Color bits ─────────────────────────────────────────────────────────

const BIT_WHITE0: u8 = 0b0000_1000; // bit 3
const BIT_WHITE1: u8 = 0b0001_0000; // bit 4
const BIT_BLACK:  u8 = 0b0010_0000; // bit 5
const COLOR_MASK: u8 = BIT_WHITE0 | BIT_WHITE1 | BIT_BLACK;
const AGE_MASK:   u8 = 0b0000_0111;

// ── Age ────────────────────────────────────────────────────────────────

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Age {
    New = 0,
    Survival = 1,
    Old0 = 2,
    Old1 = 3,
    Old = 4,
    Touched1 = 5,
    Touched2 = 6,
}

impl Age {
    pub fn is_old(self) -> bool {
        matches!(self, Age::Old | Age::Touched1 | Age::Touched2)
    }
}

// ── Type tag ────────────────────────────────────────────────────────────

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcObjectKind {
    String = 0,
    Table = 1,
    Func = 2,
    Upval = 3,
    Thread = 4,
    Proto = 5,
    CData = 6,
}

impl GcObjectKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::String),
            1 => Some(Self::Table),
            2 => Some(Self::Func),
            3 => Some(Self::Upval),
            4 => Some(Self::Thread),
            5 => Some(Self::Proto),
            6 => Some(Self::CData),
            _ => None,
        }
    }
}

// ── Header ─────────────────────────────────────────────────────────────

#[repr(C)]
pub struct GcHeader {
    /// Age (3 bits) + color bits (3 bits) + reserved (2 bits).
    bits: Cell<u8>,
    /// Object type for deallocation.
    pub(crate) kind: u8,
    _pad: [u8; 2],
    /// Position in the owning GcList vec (O(1) swap-remove).
    pub(crate) index: Cell<u32>,
    /// Allocation-time size estimate in bytes.
    pub(crate) alloc_size: Cell<u32>,
}

unsafe impl Send for GcHeader {}
unsafe impl Sync for GcHeader {}

impl GcHeader {
    pub fn new(current_white: u8, kind: GcObjectKind, alloc_size: u32) -> Self {
        debug_assert!(current_white == 0 || current_white == 1);
        let c = if current_white == 0 { BIT_WHITE0 } else { BIT_WHITE1 };
        Self {
            bits: Cell::new(c),
            kind: kind as u8,
            _pad: [0; 2],
            index: Cell::new(0),
            alloc_size: Cell::new(alloc_size),
        }
    }

    #[inline]
    fn rb(&self) -> u8 { self.bits.get() }
    #[inline]
    fn wb(&self, v: u8) { self.bits.set(v); }

    // -- Color --

    pub fn is_white(&self) -> bool {
        (self.rb() & COLOR_MASK) != BIT_BLACK && (self.rb() & COLOR_MASK) != 0
    }
    pub fn is_black(&self) -> bool {
        (self.rb() & BIT_BLACK) != 0
    }
    pub fn is_gray(&self) -> bool {
        (self.rb() & COLOR_MASK) == 0
    }
    pub fn change_white(&self) {
        let b = self.rb();
        if b & BIT_WHITE0 != 0 {
            self.wb((b & !BIT_WHITE0) | BIT_WHITE1);
        } else {
            self.wb((b & !BIT_WHITE1) | BIT_WHITE0);
        }
    }
    pub fn nw2black(&self) {
        self.wb((self.rb() & !COLOR_MASK) | BIT_BLACK);
    }
    pub fn make_gray(&self) {
        self.wb(self.rb() & !COLOR_MASK); // no white/black bits = gray
    }
    pub fn is_dead(&self, current_white: u8) -> bool {
        if current_white == 0 {
            self.rb() & BIT_WHITE1 != 0
        } else {
            self.rb() & BIT_WHITE0 != 0
        }
    }
    pub fn otherwhite(current_white: u8) -> u8 {
        if current_white == 0 { BIT_WHITE1 } else { BIT_WHITE0 }
    }

    // -- Age --

    pub fn age(&self) -> Age {
        match self.rb() & AGE_MASK {
            0 => Age::New,
            1 => Age::Survival,
            2 => Age::Old0,
            3 => Age::Old1,
            4 => Age::Old,
            5 => Age::Touched1,
            6 => Age::Touched2,
            _ => Age::Old,
        }
    }
    pub fn set_age(&self, age: Age) {
        self.wb((self.rb() & !AGE_MASK) | (age as u8));
    }
    pub fn is_old(&self) -> bool { self.age().is_old() }

    // -- Size --

    pub fn alloc_size(&self) -> u32 { self.alloc_size.get() }
    pub fn set_alloc_size(&self, s: u32) { self.alloc_size.set(s); }

    pub fn kind(&self) -> GcObjectKind {
        GcObjectKind::from_u8(self.kind).unwrap_or(GcObjectKind::Table)
    }
}
