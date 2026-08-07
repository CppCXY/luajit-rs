use std::cell::Cell;
use std::ptr::NonNull;

/// Maximum address for a valid GC pointer. On 64-bit hosts this is
/// 2^47 (for NaN-boxing), on 32-bit/wasm32 it's the full address space.
const ADDR_MAX: u64 = if cfg!(target_pointer_width = "64") {
    1u64 << 47
} else {
    u32::MAX as u64
};

// ── Re-export generational GC types ────────────────────────────────────
pub use super::gc_gen::header::{Age, GcObjectKind};

#[repr(C)]
pub struct GcHeader {
    /// Old `marked` flag — kept for backward compat during sweep.
    /// true = alive, false = dead (to be swept).
    pub(crate) marked: Cell<bool>,
    /// Age (3 bits) + tri-color (WHITE0/WHITE1/BLACK, 3 bits) + reserved.
    pub(crate) bits: Cell<u8>,
    /// Object type tag for deallocation.
    pub(crate) kind: u8,
    _pad: [u8; 2],
    /// Position in the owning GcList (for O(1) swap-remove).
    pub(crate) index: Cell<u32>,
    /// Allocation-time size estimate for GC pacing.
    pub(crate) alloc_size: Cell<u32>,
}

// ── Backward-compatible color/age helpers on GcHeader ──────────────────

const BIT_WHITE0: u8 = 0b0000_1000;
const BIT_WHITE1: u8 = 0b0001_0000;
const BIT_BLACK: u8 = 0b0010_0000;
const BIT_FINALIZED: u8 = 0b0100_0000;
const COLOR_MASK: u8 = BIT_WHITE0 | BIT_WHITE1 | BIT_BLACK;
const AGE_MASK: u8 = 0b0000_0111;

impl GcHeader {
    pub fn new(current_white: u8, kind: GcObjectKind, size: u32) -> Self {
        let c = if current_white == 0 {
            BIT_WHITE0
        } else {
            BIT_WHITE1
        };
        Self {
            marked: Cell::new(false),
            bits: Cell::new(c),
            kind: kind as u8,
            _pad: [0; 2],
            index: Cell::new(0),
            alloc_size: Cell::new(size),
        }
    }

    fn rb(&self) -> u8 {
        self.bits.get()
    }
    #[allow(dead_code)]
    pub(crate) fn raw_bits(&self) -> u8 {
        self.rb()
    }
    fn wb(&self, v: u8) {
        self.bits.set(v);
    }

    pub fn is_black(&self) -> bool {
        (self.rb() & BIT_BLACK) != 0
    }
    pub fn change_white(&self) {
        let b = self.rb();
        if b & BIT_WHITE0 != 0 {
            self.wb((b & !COLOR_MASK) | BIT_WHITE1);
        } else {
            self.wb((b & !COLOR_MASK) | BIT_WHITE0);
        }
    }
    pub fn nw2black(&self) {
        self.wb((self.rb() & !COLOR_MASK) | BIT_BLACK);
    }
    pub fn make_gray(&self) {
        self.wb(self.rb() & !COLOR_MASK);
    }
    pub fn is_dead(&self, current_white: u8) -> bool {
        if current_white == 0 {
            self.rb() & BIT_WHITE1 != 0
        } else {
            self.rb() & BIT_WHITE0 != 0
        }
    }
    pub fn otherwhite(current_white: u8) -> u8 {
        if current_white == 0 {
            BIT_WHITE1
        } else {
            BIT_WHITE0
        }
    }

    /// `isfinalized`: the `__gc` finalizer has already run; the object is
    /// just waiting for the sweep to collect it.
    pub fn is_finalized(&self) -> bool {
        self.rb() & BIT_FINALIZED != 0
    }
    pub fn set_finalized(&self) {
        self.wb(self.rb() | BIT_FINALIZED);
    }
    /// Unmark: clear the tri-color bits (the sweep then keeps the object
    /// alive without marking it, and weak tables keep its entries).
    pub fn make_undead(&self) {
        self.wb(self.rb() & !COLOR_MASK);
    }
    /// Set the color to the white bit of the *next* cycle, so the object
    /// is swept one cycle after its finalizer ran ("finalized keys are
    /// removed in two cycles").
    pub fn make_dead_next(&self, current_white: u8) {
        let color = if current_white == 0 {
            BIT_WHITE0
        } else {
            BIT_WHITE1
        };
        self.wb((self.rb() & !COLOR_MASK) | BIT_FINALIZED | color);
    }

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
    pub fn set_age(&self, a: Age) {
        self.wb((self.rb() & !AGE_MASK) | (a as u8));
    }
    pub fn is_old(&self) -> bool {
        self.age().is_old()
    }
    pub fn kind_tag(&self) -> GcObjectKind {
        GcObjectKind::from_u8(self.kind).unwrap_or(GcObjectKind::Table)
    }
}

// ── Low-address allocator ───────────────────────────────────────────────
#[cfg(not(target_arch = "wasm32"))]
mod lowmem {
    use std::alloc::Layout;
    use std::ptr::NonNull;
    const LIMIT: u64 = 1 << 47;
    pub fn alloc(layout: Layout) -> (NonNull<u8>, bool) {
        unsafe {
            let p = std::alloc::alloc(layout);
            if !p.is_null() {
                if (p as u64).saturating_add(layout.size() as u64) <= LIMIT {
                    return (NonNull::new_unchecked(p), false);
                }
                std::alloc::dealloc(p, layout);
            }
        }
        match os_alloc_low(layout.size().max(1)) {
            Some(p) => (p, true),
            None => panic!("cannot allocate below 2^47"),
        }
    }
    pub unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout, mapped: bool) {
        if mapped {
            os_free(ptr.as_ptr(), layout.size().max(1));
        } else {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }
    fn next_random_hint() -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEED: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
        let s = SEED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                Some(
                    s.wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407),
                )
            })
            .unwrap();
        ((1u64 << 38) + (s % ((1u64 << 46) - (1u64 << 38)))) & !0xFFFF
    }
    fn hint_state() -> &'static std::sync::atomic::AtomicU64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        &NEXT
    }
    fn probe<F: Fn(u64, usize) -> *mut u8>(
        size: usize,
        map: F,
        unmap: fn(*mut u8, usize),
    ) -> Option<NonNull<u8>> {
        use std::sync::atomic::Ordering;
        let mut hint = hint_state().load(Ordering::Relaxed);
        for _ in 0..1024 {
            if hint == 0 || hint.saturating_add(size as u64) > LIMIT {
                hint = next_random_hint();
            }
            let p = map(hint, size);
            if !p.is_null() && p as isize != -1 {
                if (p as u64).saturating_add(size as u64) <= LIMIT {
                    let end = (p as u64 + size as u64 + 0xFFFF) & !0xFFFF;
                    hint_state().store(end, Ordering::Relaxed);
                    return NonNull::new(p);
                }
                unmap(p, size);
            }
            hint = next_random_hint();
        }
        None
    }
    #[cfg(unix)]
    fn os_alloc_low(size: usize) -> Option<NonNull<u8>> {
        const PROT_RW: i32 = 3;
        const MAP_PRIVATE: i32 = 0x02;
        const MAP_ANON: i32 = if cfg!(any(target_os = "linux", target_os = "android")) {
            0x20
        } else {
            0x1000
        };
        const MAP_FNR: i32 = if cfg!(any(target_os = "linux", target_os = "android")) {
            0x10_0000
        } else {
            0
        };
        unsafe extern "C" {
            fn mmap(a: *mut u8, l: usize, p: i32, f: i32, fd: i32, o: i64) -> *mut u8;
        }
        probe(
            size,
            |h, sz| unsafe { mmap(h as _, sz, PROT_RW, MAP_PRIVATE | MAP_ANON | MAP_FNR, -1, 0) },
            os_free,
        )
    }
    #[cfg(unix)]
    fn os_free(p: *mut u8, s: usize) {
        unsafe extern "C" {
            fn munmap(a: *mut u8, l: usize) -> i32;
        }
        unsafe { munmap(p, s) };
    }
    #[cfg(windows)]
    fn os_alloc_low(s: usize) -> Option<NonNull<u8>> {
        const MC: u32 = 0x1000;
        const MR: u32 = 0x2000;
        const PRW: u32 = 0x04;
        unsafe extern "system" {
            fn VirtualAlloc(a: *mut u8, sz: usize, t: u32, p: u32) -> *mut u8;
        }
        probe(
            s,
            |h, sz| unsafe { VirtualAlloc(h as _, sz, MC | MR, PRW) },
            os_free,
        )
    }
    #[cfg(windows)]
    fn os_free(p: *mut u8, _: usize) {
        const MR: u32 = 0x8000;
        unsafe extern "system" {
            fn VirtualFree(a: *mut u8, sz: usize, t: u32) -> i32;
        }
        unsafe { VirtualFree(p, 0, MR) };
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn os_probe_returns_low_writable_memory() {
            let s = 1 << 20;
            let p = os_alloc_low(s).expect("fail");
            assert!((p.as_ptr() as u64) + s as u64 <= LIMIT);
            unsafe {
                p.as_ptr().write(0xAB);
                os_free(p.as_ptr(), s);
            }
        }
        #[test]
        fn alloc_dealloc_roundtrip() {
            let l = Layout::from_size_align(4096, 16).unwrap();
            let (p, m) = alloc(l);
            assert!((p.as_ptr() as u64) + 4096 <= LIMIT);
            unsafe {
                dealloc(p, l, m);
            }
        }
        #[test]
        fn os_probe_survives_hundreds_of_pages() {
            let s = 1 << 16;
            let mut v = Vec::new();
            for _i in 0..300 {
                let p = os_alloc_low(s).expect("fail");
                unsafe { p.as_ptr().write_bytes(0x77, s) };
                v.push(p);
            }
            for p in v {
                os_free(p.as_ptr(), s);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod lowmem {
    use std::alloc::Layout;
    use std::ptr::NonNull;
    pub fn alloc(layout: Layout) -> (NonNull<u8>, bool) {
        let p = unsafe { std::alloc::alloc(layout) };
        assert!(!p.is_null(), "alloc failed");
        unsafe { (NonNull::new_unchecked(p), false) }
    }
    pub unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout, _mapped: bool) {
        unsafe {
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
}

fn alloc_block<T>(
    v: T,
    kind: GcObjectKind,
    alloc_size: u32,
    current_white: u8,
) -> (NonNull<T>, bool) {
    // New objects inherit the current generation's white bit.
    // During the next sweep, `is_dead` checks the *opposite* bit,
    // so new objects survive their first sweep.  Current white
    // is flipped at cycle START, not END.
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    let (raw, mapped) = lowmem::alloc(layout);
    unsafe {
        (raw.as_ptr() as *mut GcHeader).write(GcHeader::new(current_white, kind, alloc_size));
    }
    let dp = unsafe { raw.as_ptr().add(data_offset) as *mut T };
    unsafe { dp.write(v) };
    (unsafe { NonNull::new_unchecked(dp) }, mapped)
}
fn dealloc_block<T>(data: NonNull<T>, mapped: bool) {
    let addr = data.as_ptr() as usize;
    if !(0x1000..ADDR_MAX as usize).contains(&addr) {
        eprintln!("DEALLOC-BAD-PTR: {:p}", data.as_ptr());
        std::process::abort();
    }
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    let ap = unsafe { (data.as_ptr() as *mut u8).sub(data_offset) };
    // Check that ap is 16-byte aligned (required by glibc fastbin).
    if (ap as usize) & 0xF != 0 {
        eprintln!(
            "DEALLOC-MISALIGNED: ap={:p} data={:p} offset={} T={}",
            ap,
            data.as_ptr(),
            data_offset,
            std::any::type_name::<T>()
        );
        std::process::abort();
    }
    let kind = unsafe { (*(ap as *const GcHeader)).kind };
    debug_assert!(
        kind <= 7,
        "dealloc_block: corrupt kind={} T={}",
        kind,
        std::any::type_name::<T>()
    );
    if kind > 7 {
        eprintln!(
            "DEALLOC-CORRUPT: kind={} T={}",
            kind,
            std::any::type_name::<T>()
        );
        std::process::abort();
    }
    unsafe { lowmem::dealloc(NonNull::new_unchecked(ap), layout, mapped) };
}
#[inline]
fn gc_header<T>(ptr: NonNull<T>) -> &'static GcHeader {
    let addr = ptr.as_ptr() as usize;
    if !(0x1000..ADDR_MAX as usize).contains(&addr) {
        static DUMMY: std::sync::atomic::AtomicPtr<GcHeader> =
            std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
        let p = DUMMY.load(std::sync::atomic::Ordering::Relaxed);
        if p.is_null() {
            let b = Box::new(GcHeader::new(0, GcObjectKind::Table, 0));
            let leaked = Box::leak(b) as *mut GcHeader;
            DUMMY.store(leaked, std::sync::atomic::Ordering::Relaxed);
            return unsafe { &*leaked };
        }
        return unsafe { &*p };
    }
    let (_, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    unsafe {
        let p = (ptr.as_ptr() as *const u8).sub(data_offset) as *const GcHeader;
        // Quick alignment check: must be at least 4-byte aligned
        if (p as usize) & 0x3 != 0 {
            eprintln!(
                "GCHEADER-MISALIGNED: p={:p} ptr={:p} off={} T={}",
                p,
                ptr.as_ptr(),
                data_offset,
                std::any::type_name::<T>()
            );
            std::process::abort();
        }
        let h = &*p;
        debug_assert!(
            h.kind <= 7,
            "gc_header: corrupt kind={} T={}",
            h.kind,
            std::any::type_name::<T>()
        );
        h
    }
}

// ── Pool ────────────────────────────────────────────────────────────────
pub struct Pool<T> {
    objects: Vec<NonNull<T>>,
    mapped: Vec<bool>,
    live: usize,
    kind: GcObjectKind,
    current_white: Cell<u8>,
}
impl<T> Pool<T> {
    pub fn with_page_size(_: usize) -> Self {
        Self::new(GcObjectKind::Table)
    }
    pub fn new(kind: GcObjectKind) -> Self {
        Self {
            objects: Vec::new(),
            mapped: Vec::new(),
            live: 0,
            kind,
            current_white: Cell::new(0),
        }
    }
    pub fn alloc(&mut self, v: T) -> GcPtr<T> {
        let cw = self.current_white.get();
        let (nn, m) = alloc_block(v, self.kind, std::mem::size_of::<T>() as u32, cw);
        self.objects.push(nn);
        self.mapped.push(m);
        self.live += 1;
        GcPtr::new(nn)
    }
    pub fn free(&mut self, p: GcPtr<T>) {
        unsafe { p.0.as_ptr().drop_in_place() };
        let idx = self.objects.iter().position(|&x| x == p.0).unwrap();
        dealloc_block(p.0, self.mapped[idx]);
        self.objects.swap_remove(idx);
        self.mapped.swap_remove(idx);
        self.live -= 1;
    }
    pub fn len(&self) -> usize {
        self.live
    }
    #[allow(dead_code)]
    pub(crate) fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.objects.iter().map(|nn| unsafe { nn.as_ref() })
    }
    /// Clear the per-cycle `marked` flag on every object. The flag records
    /// "marked in the current cycle"; without resetting it at a cycle
    /// boundary, an object marked by an interrupted/incomplete cycle keeps
    /// `marked == true` and is treated as live in the next cycle — weak
    /// tables then fail to drop it, and the sweep retains it forever.
    pub fn reset_marked(&mut self) {
        for nn in &self.objects {
            let h = gc_header(*nn);
            if h.marked.get() {
                h.marked.set(false);
            }
        }
    }
    pub fn sweep(&mut self, mut on_free: impl FnMut(&T)) {
        // Old marked-only sweep (backward compat).
        let mut i = 0;
        while i < self.objects.len() {
            let ptr = self.objects[i];
            let addr = ptr.as_ptr() as usize;
            if addr <= 0x1000 || addr >= ADDR_MAX as usize {
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
                continue;
            }
            if gc_header(ptr).marked.get() {
                gc_header(ptr).marked.set(false);
                i += 1;
            } else {
                unsafe {
                    on_free(ptr.as_ref());
                }
                unsafe {
                    ptr.as_ptr().drop_in_place();
                }
                dealloc_block(ptr, self.mapped[i]);
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
            }
        }
        self.live = self.objects.len();
    }
    /// Tri-color sweep: uses current_white to decide alive/dead.
    /// Surviving objects get change_white() and marked.set(false).
    pub fn sweep_tricolor(&mut self, current_white: u8, mut on_free: impl FnMut(&T)) {
        let mut i = 0;
        while i < self.objects.len() {
            let ptr = self.objects[i];
            let addr = ptr.as_ptr() as usize;
            if addr <= 0x1000 || addr >= ADDR_MAX as usize {
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
                continue;
            }
            let h = gc_header(ptr);
            let was_marked = h.marked.get();
            // Alive = marked this cycle, or a non-dead color. `marked` is
            // cleared at every cycle start, so `was_marked` only reflects
            // this cycle's mark; objects a stale BLACK left by an
            // interrupted earlier cycle are kept alive here (conservative),
            // and weak tables drop them via `may_clear`'s `!is_marked`.
            let alive = was_marked || (!h.is_dead(current_white) && !h.is_black());
            if alive {
                if was_marked {
                    h.change_white();
                }
                h.marked.set(false);
                i += 1;
            } else {
                unsafe {
                    on_free(ptr.as_ref());
                }
                unsafe {
                    ptr.as_ptr().drop_in_place();
                }
                dealloc_block(ptr, self.mapped[i]);
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
            }
        }
        self.live = self.objects.len();
    }
    pub fn update_current_white(&self, cw: u8) {
        self.current_white.set(cw);
    }
}
impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new(GcObjectKind::Table)
    }
}
impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        for (i, &nn) in self.objects.iter().enumerate() {
            unsafe { nn.as_ptr().drop_in_place() };
            dealloc_block(nn, self.mapped[i]);
        }
    }
}

// ── GcPtr ───────────────────────────────────────────────────────────────
pub struct GcPtr<T>(NonNull<T>);
impl<T> GcPtr<T> {
    pub(crate) fn new(p: NonNull<T>) -> Self {
        debug_assert!((p.as_ptr() as u64) < (1u64 << 47));
        GcPtr(p)
    }
    pub fn from_addr(addr: u64) -> Option<Self> {
        if addr != 0 && addr < 0x100 {
            return None;
        }
        NonNull::new(addr as *mut T).map(GcPtr)
    }
    pub fn addr(self) -> u64 {
        self.0.as_ptr() as u64
    }

    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    #[inline]
    pub fn as_mut<'a>(self) -> &'a mut T {
        unsafe { &mut *self.0.as_ptr() }
    }

    pub fn is_marked(self) -> bool {
        gc_header(self.0).marked.get()
    }

    #[allow(dead_code)]
    pub(crate) fn mark_bits(self) -> (bool, u8) {
        let h = gc_header(self.0);
        (h.marked.get(), h.raw_bits())
    }

    #[track_caller]
    pub fn set_marked(self) {
        let h = gc_header(self.0);
        h.marked.set(true);
        // Also mark in tri-color bits (may crash if bits is at wrong offset)
        h.nw2black();
    }
}

impl<T> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for GcPtr<T> {}

impl<T> PartialEq for GcPtr<T> {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<T> Eq for GcPtr<T> {}

impl<T> std::hash::Hash for GcPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, s: &mut H) {
        self.0.hash(s);
    }
}

impl<T> std::fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPtr({:p})", self.0.as_ptr())
    }
}

use crate::compiler::lex::Interner;
// ── Collector ───────────────────────────────────────────────────────────
use crate::func::{GcFunc, Upval};
use crate::proto::{KGc, Proto};
use crate::runtime::userdata::GcUserData;
use crate::state::{GcHeap, GlobalState, LuaState};
use crate::table::LuaTable;
use crate::value::{LJ_TCDATA, LJ_TFUNC, LJ_TSTR, LJ_TTAB, LJ_TTHREAD, LJ_TUDATA, LuaValue};

pub const GC_PAUSE: usize = 200;
pub(crate) const GC_THRESHOLD_MIN: usize = 64 * 1024;

/// Weak-table mode bits (mirror of LuaJIT's `LJ_GC_WEAKKEY`/`LJ_GC_WEAKVAL`).
pub const WEAKKEY: u8 = 0x08;
pub const WEAKVAL: u8 = 0x10;

// Incremental GC state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GcState {
    Pause,
    Propagate,
    Atomic,
    Sweep,
    /// Sweep finished but there are objects waiting for their `__gc`
    /// finalizers. The VM runs them at the next safe point
    /// (`vm::run_finalizers`), which returns the collector to `Pause`.
    Finalize,
}
pub const GC_STEP_SIZE: usize = 4096;

pub enum Gray {
    Tab(GcPtr<LuaTable>),
    Func(GcPtr<GcFunc>),
    Proto(GcPtr<Proto>),
    Thread(GcPtr<LuaState>),
    UserData(GcPtr<GcUserData>),
}

/// An object whose `__gc` finalizer must run. Collected during the atomic
/// phase into `GcHeap.mmudata` and executed by the VM at the next safe
/// point (after the sweep, so every object the finalizer may touch is
/// still alive).
pub enum Finalizable {
    UData(GcPtr<GcUserData>),
}

impl Finalizable {
    pub fn value(&self) -> LuaValue {
        match self {
            Finalizable::UData(u) => LuaValue::userdata(*u),
        }
    }
    pub fn metatable(&self) -> Option<GcPtr<LuaTable>> {
        match self {
            Finalizable::UData(u) => u.as_ref().metatable,
        }
    }
    pub fn mark_finalized(&self, current_white: u8) {
        match self {
            Finalizable::UData(u) => gc_header(u.0).make_dead_next(current_white),
        }
    }
}

struct Marker<'g> {
    gray: Vec<Gray>,
    strings: &'g Interner,
    heap: *const GcHeap,
    /// Atomic phase: stale stack slots above `top` may be cleared (the
    /// interpreter is not running, so `top` is exact). During propagation
    /// the interpreter may hold a lowered `top` (e.g. inside a C call),
    /// and clearing would destroy live frame locals.
    atomic: bool,
    /// Weak tables discovered during traversal, with their `__mode` bits.
    weak: Vec<(GcPtr<LuaTable>, u8)>,
}
impl<'g> Marker<'g> {
    #[track_caller]
    fn mark_value(&mut self, v: LuaValue) {
        match v.itype() {
            LJ_TSTR => {
                if let Some(p) = v.as_string() {
                    p.set_marked();
                }
            }
            LJ_TTAB => {
                if let Some(p) = v.as_table()
                    && !p.is_marked()
                {
                    p.set_marked();
                    self.gray.push(Gray::Tab(p));
                }
            }
            LJ_TFUNC => {
                if let Some(p) = v.as_func()
                    && !p.is_marked()
                {
                    p.set_marked();
                    self.gray.push(Gray::Func(p));
                }
            }
            LJ_TTHREAD => {
                if let Some(p) = v.as_thread()
                    && !p.is_marked()
                {
                    p.set_marked();
                    self.gray.push(Gray::Thread(p));
                }
            }
            LJ_TUDATA => {
                if let Some(p) = v.as_userdata()
                    && !p.is_marked()
                {
                    p.set_marked();
                    // Userdata itself holds no GC-reachable Lua objects
                    // (the inner Box<dyn Any> is managed by Rust).
                    // However, the metatable must be marked.
                    if let Some(mt) = p.as_ref().metatable {
                        self.mark_table(mt);
                    }
                }
            }
            _ => {}
        }
    }
    fn mark_thread(&mut self, th: GcPtr<LuaState>) {
        // Always mark the (main) thread and re-enqueue it: a stale
        // `marked` flag from an earlier cycle must never skip the stack
        // walk, or residual slots would go unmarked and be freed out from
        // under the stack (the interpreter's `frame_top` protects them
        // from the atomic clear).
        th.set_marked();
        self.gray.push(Gray::Thread(th));
    }
    fn mark_table(&mut self, t: GcPtr<LuaTable>) {
        if !t.is_marked() {
            t.set_marked();
            self.gray.push(Gray::Tab(t));
        }
    }
    fn mark_proto(&mut self, p: GcPtr<Proto>) {
        if !p.is_marked() {
            p.set_marked();
            self.gray.push(Gray::Proto(p));
        }
    }
    fn mark_upval(&mut self, uv: GcPtr<Upval>) {
        if !uv.is_marked() {
            uv.set_marked();
            self.mark_value(uv.as_ref().get());
        }
    }
    fn mark_kgc_slice(this: &mut Self, kgc: &[KGc]) {
        for k in kgc {
            match k {
                KGc::Str(sid) => {
                    if let Some(ptr) = this.strings.try_lookup(*sid) {
                        ptr.set_marked();
                    }
                }
                KGc::ProtoRef(c) => this.mark_proto(*c),
                // Template tables are strong (no metatable): ignore the
                // weak-mode return.
                KGc::Table(t) => {
                    t.gc_traverse(|v| this.mark_value(v));
                }
                KGc::TableRef(t) => {
                    // The template table is a heap object referenced by
                    // the prototype: it must be marked (and grayed for
                    // its contents), or the pool reclaims it while the
                    // proto's kgc still points at it.
                    this.mark_table(*t);
                }
                KGc::Proto(_) | KGc::CData(_) => {}
            }
        }
    }
    fn collect_weak(&mut self, t: GcPtr<LuaTable>, mode: u8) {
        if mode != 0 && !self.weak.iter().any(|(p, _)| *p == t) {
            self.weak.push((t, mode));
        }
    }

    fn propagate(&mut self) {
        while let Some(g) = self.gray.pop() {
            match g {
                Gray::Tab(t) => {
                    let mode = t.as_ref().gc_traverse(|v| self.mark_value(v));
                    self.collect_weak(t, mode);
                }
                Gray::Func(f) => match f.as_ref() {
                    GcFunc::Lua(c) => {
                        self.mark_table(c.env);
                        self.mark_proto(c.proto);
                        for &uv in &c.upvals {
                            self.mark_upval(uv);
                        }
                    }
                    GcFunc::C(c) => {
                        self.mark_table(c.env);
                        for &v in &c.upvals {
                            self.mark_value(v);
                        }
                    }
                },
                Gray::Proto(p) => {
                    let pt = p.as_ref();
                    if let Some(sid) = pt.source
                        && let Some(ptr) = self.strings.try_lookup(sid)
                    {
                        ptr.set_marked();
                    }
                    for k in &pt.kgc {
                        Self::mark_kgc_slice(self, std::slice::from_ref(k));
                    }
                }
                Gray::Thread(th) => {
                    let l = th.as_mut();
                    // lj_gc_step_fixtop: mark the whole current frame (`top`
                    // may be lowered to a C-call result area); stale slots
                    // inside the frame are kept alive rather than freed out
                    // from under us.
                    let mark_to = l.top.max(l.frame_top);
                    for i in 0..mark_to {
                        self.mark_value(l.stack[i]);
                    }
                    self.mark_value(l.errval);
                    for &uv in &l.openuv {
                        self.mark_upval(uv);
                    }
                    if let Some(cl) = l.suspend.call_cl() {
                        cl.set_marked();
                    }
                    if self.atomic {
                        // lj_gc_step_fixtop: never clear slots below the
                        // current instruction's live top (or the frame
                        // extent, which protects live temporaries above a
                        // transiently lowered top).
                        let clear_from = l.top.max(l.frame_top);
                        for s in l.stack[clear_from..].iter_mut() {
                            *s = LuaValue::NIL;
                        }
                    }
                    // Threads are never black (LuaJIT's black2gray +
                    // grayagain): the atomic phase re-traverses them and
                    // clears stale slots written after their first pass.
                    gc_header(th.0).make_gray();
                    unsafe {
                        (self.heap as *mut GcHeap)
                            .as_mut()
                            .unwrap()
                            .gc_grayagain
                            .push(Gray::Thread(th));
                    }
                }
                Gray::UserData(_) => {
                    // inner Box<dyn Any> contains no Lua GC objects.
                    // Metatable was already marked in mark_value.
                }
            }
        }
    }
    fn propagate_step(&mut self, work: usize) -> bool {
        for _ in 0..work {
            if self.gray.is_empty() {
                return true;
            }
            let g = self.gray.pop().unwrap();
            match g {
                Gray::Tab(t) => {
                    t.set_marked();
                    let mode = t.as_ref().gc_traverse(|v| self.mark_value(v));
                    self.collect_weak(t, mode);
                }
                Gray::Func(f) => {
                    f.set_marked();
                    match f.as_ref() {
                        GcFunc::Lua(c) => {
                            self.mark_table(c.env);
                            self.mark_proto(c.proto);
                            for &uv in &c.upvals {
                                self.mark_upval(uv);
                            }
                        }
                        GcFunc::C(c) => {
                            self.mark_table(c.env);
                            for &v in &c.upvals {
                                self.mark_value(v);
                            }
                        }
                    }
                }
                Gray::Proto(p) => {
                    p.set_marked();
                    let pt = p.as_ref();
                    if let Some(sid) = pt.source
                        && let Some(ptr) = self.strings.try_lookup(sid)
                    {
                        ptr.set_marked();
                    }
                    for k in &pt.kgc {
                        match k {
                            KGc::Str(sid) => {
                                if let Some(ptr) = self.strings.try_lookup(*sid) {
                                    ptr.set_marked();
                                }
                            }
                            KGc::ProtoRef(c) => self.mark_proto(*c),
                            KGc::Table(t) => {
                                t.gc_traverse(|v| self.mark_value(v));
                            }
                            KGc::TableRef(t) => {
                                // Same as mark_kgc_slice: the template
                                // table itself must survive.
                                self.mark_table(*t);
                            }
                            _ => {}
                        }
                    }
                }
                Gray::Thread(th) => {
                    th.set_marked();
                    let l = th.as_mut();
                    // lj_gc_step_fixtop: mark the whole current frame (see
                    // the propagate arm above).
                    let mark_to = l.top.max(l.frame_top);
                    for i in 0..mark_to {
                        self.mark_value(l.stack[i]);
                    }
                    self.mark_value(l.errval);
                    for &uv in &l.openuv {
                        self.mark_upval(uv);
                    }
                    if let Some(cl) = l.suspend.call_cl() {
                        cl.set_marked();
                    }
                    if self.atomic {
                        // lj_gc_step_fixtop: never clear slots below the
                        // current instruction's live top (or the frame
                        // extent, which protects live temporaries above a
                        // transiently lowered top).
                        let clear_from = l.top.max(l.frame_top);
                        for s in l.stack[clear_from..].iter_mut() {
                            *s = LuaValue::NIL;
                        }
                    }
                    // Threads are never black (LuaJIT's black2gray +
                    // grayagain): the atomic phase re-traverses them and
                    // clears stale slots written after their first pass.
                    gc_header(th.0).make_gray();
                    unsafe {
                        (self.heap as *mut GcHeap)
                            .as_mut()
                            .unwrap()
                            .gc_grayagain
                            .push(Gray::Thread(th));
                    }
                }
                Gray::UserData(_) => {}
            }
        }
        self.gray.is_empty()
    }
}

fn size_func(f: &GcFunc) -> usize {
    std::mem::size_of::<GcFunc>()
        + match f {
            GcFunc::Lua(c) => c.upvals.len() * 8,
            GcFunc::C(c) => c.upvals.len() * 8,
        }
}

const fn size_upval() -> usize {
    std::mem::size_of::<Upval>()
}

// ── Incremental GC ─────────────────────────────────────────────────────
/// Forward barrier: when a table (possibly BLACK) is written with a GC
/// value, ensure the value is at least GRAY so the sweep doesn't free it.
pub fn barrier_fwd(heap: &mut GcHeap, val: LuaValue) {
    if heap.gc_state == GcState::Pause || heap.gc_state == GcState::Finalize {
        return;
    }
    let gray = if heap.gc_state == GcState::Propagate {
        &mut heap.gc_gray
    } else {
        &mut heap.gc_grayagain
    };
    match val.itype() {
        LJ_TSTR => {
            if let Some(p) = val.as_string() {
                p.set_marked();
            }
        }
        LJ_TTAB => {
            if let Some(p) = val.as_table()
                && !p.is_marked()
            {
                p.set_marked();
                gray.push(Gray::Tab(p));
            }
        }
        LJ_TFUNC => {
            if let Some(p) = val.as_func()
                && !p.is_marked()
            {
                p.set_marked();
                gray.push(Gray::Func(p));
            }
        }
        LJ_TTHREAD => {
            if let Some(p) = val.as_thread()
                && !p.is_marked()
            {
                p.set_marked();
                gray.push(Gray::Thread(p));
            }
        }
        LJ_TUDATA => {
            if let Some(p) = val.as_userdata()
                && !p.is_marked()
            {
                p.set_marked();
                if let Some(mt) = p.as_ref().metatable {
                    barrier_fwd(heap, LuaValue::table(mt));
                }
            }
        }
        _ => {}
    }
}

/// Back barrier (LuaJIT `lj_gc_anybarriert`): called after every table
/// store. If the table was already scanned (black), mark it gray so the
/// collector rescans it during incremental propagation. No-op when the
/// table is white or gray.
///
/// Unlike `barrier_fwd`, this only accesses the table's GC header — always
/// a valid GC object pointer — so it is safe to call at every store site
/// including from the VM interpreter.
#[inline]
pub fn barrier_back(heap: &mut GcHeap, t: GcPtr<LuaTable>) {
    if heap.gc_state == GcState::Pause || heap.gc_state == GcState::Finalize {
        return;
    }
    let h = gc_header(t.0);
    if h.is_black() {
        h.make_gray();
        let gray = if heap.gc_state == GcState::Propagate {
            &mut heap.gc_gray
        } else {
            &mut heap.gc_grayagain
        };
        gray.push(Gray::Tab(t));
    }
}

/// Run one incremental GC step. Returns `true` when the cycle is complete
/// (state is Pause).
pub fn gc_step(heap: &mut GcHeap, size: usize) -> bool {
    let step = heap.gc_step_size.max(size);
    match heap.gc_state {
        GcState::Pause => {
            let live = heap.total + heap.strings.bytes() + heap.table_extra;
            if live >= heap.threshold {
                heap.debt += live - heap.threshold;
                heap.threshold = live + GC_STEP_SIZE;
            }
            true
        }
        GcState::Propagate => {
            let mut m = Marker {
                gray: std::mem::take(&mut heap.gc_gray),
                strings: &heap.strings,
                heap: heap as *const GcHeap,
                atomic: false,
                weak: Vec::new(),
            };
            let done = m.propagate_step((step / 64).max(1));
            m.gray.extend(std::mem::take(&mut heap.gc_gray));
            if done && m.gray.is_empty() {
                heap.gc_state = GcState::Atomic;
            }
            heap.gc_gray = m.gray;
            heap.gc_weak.extend(m.weak);
            false
        }
        GcState::Atomic => {
            // Atomic: mark the 2nd-chance list once (LuaJIT empties
            // grayagain into gray and propagates a single round; threads
            // re-register themselves and stay there until the next cycle).
            let mut gray = std::mem::take(&mut heap.gc_grayagain);
            gray.extend(std::mem::take(&mut heap.gc_gray));
            if !gray.is_empty() {
                let mut m = Marker {
                    gray,
                    strings: &heap.strings,
                    heap: heap as *const GcHeap,
                    atomic: true,
                    weak: Vec::new(),
                };
                m.propagate();
                heap.gc_weak.extend(m.weak);
            }
            heap.gc_gray.clear();
            heap.gc_grayagain.clear();
            // Separate objects that need a __gc finalizer. They are kept
            // unmarked (sweep keeps them; weak tables keep their entries,
            // so the finalizer can still reach them). Their metatables
            // are marked so the __gc function stays alive.
            let mmu = separate_finalizable(heap);
            if !mmu.is_empty() {
                let mut m = Marker {
                    gray: Vec::new(),
                    strings: &heap.strings,
                    heap: heap as *const GcHeap,
                    atomic: true,
                    weak: Vec::new(),
                };
                for o in &mmu {
                    if let Some(mt) = o.metatable() {
                        m.mark_table(mt);
                    }
                }
                m.propagate();
                heap.gc_gray.clear();
                heap.gc_grayagain.clear();
                heap.gc_weak.extend(m.weak);
                heap.mmudata.extend(mmu);
            }
            // All marking done: drop weak-table entries whose key or value
            // is about to be swept, before the sweep frees any object.
            clear_weak(heap);
            heap.gc_state = GcState::Sweep;
            heap.gc_sweep_pool = 0;
            false
        }
        GcState::Sweep => {
            // No propagation in the sweep phase (threads sit in grayagain
            // until the next atomic round); anything left over is from the
            // atomic round already handled there.
            let done = sweep_one_pool(heap);
            if done {
                heap.gc_state = if std::env::var("LUARS_NO_FIN").is_ok() || heap.mmudata.is_empty()
                {
                    GcState::Pause
                } else {
                    GcState::Finalize
                };
                let total = total_live(heap);
                heap.total = total;
                heap.threshold = ((total + heap.strings.bytes()) * GC_PAUSE / 100)
                    .max(GC_THRESHOLD_MIN)
                    .max(heap.threshold / 2);
                heap.table_extra = 0;
                heap.debt = 0;
                true
            } else {
                false
            }
        }
        GcState::Finalize => {
            // The VM runs pending finalizers at a safe point and resets
            // the state to Pause; until then every cycle attempt is a
            // no-op (new collections may start from run_finalizers).
            true
        }
    }
}

fn sweep_one_pool(heap: &mut GcHeap) -> bool {
    let done = match heap.gc_sweep_pool {
        0 => {
            heap.strings.sweep(heap.current_white);
            1
        }
        1 => {
            heap.tables.sweep_tricolor(heap.current_white, |_| {});
            2
        }
        2 => {
            heap.funcs.sweep_tricolor(heap.current_white, |_| {});
            3
        }
        3 => {
            heap.threads.sweep_tricolor(heap.current_white, |th| {
                for &uv in &th.openuv {
                    uv.as_mut().close();
                }
            });
            4
        }
        4 => {
            heap.upvals.sweep_tricolor(heap.current_white, |_| {});
            5
        }
        5 => {
            heap.protos.sweep_tricolor(heap.current_white, |_| {});
            6
        }
        6 => {
            heap.userdatas.sweep_tricolor(heap.current_white, |_| {});
            7
        }
        _ => {
            heap.gc_state = if heap.mmudata.is_empty() {
                GcState::Pause
            } else {
                GcState::Finalize
            };
            let total = total_live(heap);
            heap.total = total;
            heap.threshold = ((total + heap.strings.bytes()) * GC_PAUSE / 100)
                .max(GC_THRESHOLD_MIN)
                .max(heap.threshold / 2);
            heap.table_extra = 0;
            heap.debt = 0;
            return true;
        }
    };
    heap.gc_sweep_pool = done;
    false
}

fn total_live(heap: &GcHeap) -> usize {
    let mut total = 0usize;
    for t in heap.tables.iter() {
        total += t.gc_size();
    }
    for f in heap.funcs.iter() {
        total += size_func(f);
    }
    total += heap.upvals.len() * size_upval();
    for p in heap.protos.iter() {
        total += p.gc_size();
    }
    for th in heap.threads.iter() {
        total += size_thread(th);
    }
    for cd in heap.cdatas.iter() {
        total += std::mem::size_of::<crate::runtime::cdata::CData>() + cd.data.len();
    }
    total += heap.userdatas.len() * std::mem::size_of::<crate::runtime::userdata::GcUserData>();
    total
}

fn size_thread(th: &LuaState) -> usize {
    std::mem::size_of::<LuaState>() + th.stack.len() * std::mem::size_of::<LuaValue>()
}
pub(crate) fn account_thread(th: &LuaState) -> usize {
    size_thread(th)
}

/// `gc_mayclear`: can this weak-table entry slot be dropped? Only GC
/// objects can be weak references; strings cannot (they are marked and
/// kept, per the Lua 5.1 semantics of interned strings), and any other
/// object that was not marked this cycle (`is_white`) is cleared.
/// Finalized userdata is additionally dropped from *value* positions
/// (LuaJIT: `isfinalized(udata) && val`).
pub(crate) fn may_clear(v: LuaValue, is_val: bool, cw: u8) -> bool {
    match v.itype() {
        LJ_TSTR => {
            if let Some(p) = v.as_string() {
                p.set_marked();
            }
            false
        }
        LJ_TTAB => v.as_table().is_some_and(|p| !p.is_marked()),
        LJ_TFUNC => v.as_func().is_some_and(|p| !p.is_marked()),
        LJ_TTHREAD => v.as_thread().is_some_and(|p| !p.is_marked()),
        LJ_TUDATA => v.as_userdata().is_some_and(|p| {
            // Keys: cleared only when dead (other-white) — a finalized
            // key survives the cycle its finalizer ran in, then dies the
            // next ("finalized keys are removed in two cycles"). Values:
            // dead or already finalized.
            let h = gc_header(p.0);
            if is_val {
                h.is_dead(cw) || h.is_finalized()
            } else {
                h.is_dead(cw)
            }
        }),
        LJ_TCDATA => v.as_cdata().is_some_and(|p| !p.is_marked()),
        _ => false,
    }
}

/// `gc_clearweak`: atomic-phase cleanup of every weak table found during
/// marking. Runs after all marking is done and before the sweep frees
/// anything, so the `is_white` test is exact.
fn clear_weak(heap: &mut GcHeap) {
    let weak = std::mem::take(&mut heap.gc_weak);
    for (t, mode) in weak {
        t.as_mut().clear_weak_entries(mode, heap.current_white);
    }
}

/// Does this metatable provide a `__gc` finalizer (a non-nil `__gc`
/// value)? Mirrors `lj_meta_fastg(g, mt, MM_gc)`.
pub(crate) fn has_gc_meta(mt: Option<GcPtr<LuaTable>>) -> bool {
    mt.is_some_and(|mt| mt.as_ref().scan_str_key(b"__gc").is_some())
}

/// `lj_gc_separateudata`: collect every dead object with a `__gc`
/// metatable into the finalizer list. The objects are unmarked (the sweep
/// keeps them, and weak tables keep their entries) so they survive until
/// the VM runs the finalizer; `mark_finalized` then makes them die in the
/// *next* cycle.
fn separate_finalizable(heap: &GcHeap) -> Vec<Finalizable> {
    // Pool order is oldest-first; collect in that order so the vector is
    // [u, p1, ..., p10]. run_finalizers pops from the end, so the newest
    // object is finalized first — LuaJIT's mmudata LIFO, which the gc.lua
    // suite depends on (a[o] == 10-s).
    //
    // LuaJIT's gc_separateudata only separates userdata and threads —
    // tables with a __gc metatable are never finalized (the metatable
    // itself stays alive while its userdata is pending finalization).
    let mut out = Vec::new();
    for i in 0..heap.userdatas.objects.len() {
        let u = heap.userdatas.objects[i];
        let h = gc_header(u);
        if !h.marked.get() && !h.is_finalized() && has_gc_meta(unsafe { u.as_ref() }.metatable) {
            // LuaJIT's gc_separateudata: mark FINALIZED right here so the
            // atomic-phase clear_weak drops finalized userdata from weak
            // *value* positions (gc_mayclear's `isfinalized && val`).
            h.make_undead();
            h.set_finalized();
            out.push(Finalizable::UData(GcPtr(u)));
        }
    }
    out
}

pub fn full_gc(g: &mut GlobalState) {
    // Pending finalizers keep the collector in the Finalize state; the VM
    // runs them at a safe point (run_finalizers), not here.
    if g.heap.gc_state == GcState::Finalize {
        return;
    }
    // Drain any in-progress cycle.
    while !gc_step(&mut g.heap, usize::MAX) {}
    // Start a fresh cycle.
    start_gc_cycle(g);
    // Run it to completion.
    while !gc_step(&mut g.heap, usize::MAX) {}
    // Sweep clears the marked flag on every survivor, so liveness must be
    // checked by pool membership instead of the tri-color bits.
    debug_assert!(
        g.heap
            .tables
            .iter()
            .any(|t| std::ptr::eq(t, g.globals.as_ref())),
        "globals freed after full_gc"
    );
    debug_assert!(
        g.heap
            .tables
            .iter()
            .any(|t| std::ptr::eq(t, g.registry.as_ref())),
        "registry freed after full_gc"
    );
    debug_assert!(
        g.heap
            .threads
            .iter()
            .any(|t| std::ptr::eq(t, g.main().as_ref())),
        "main thread freed after full_gc"
    );
    g.heap.gc_gray.clear();
    g.heap.gc_grayagain.clear();
}

/// Start a new GC cycle from Pause state. Marks roots and transitions to
/// Propagate. Caller should then drive the cycle with gc_step().
pub fn start_gc_cycle(g: &mut GlobalState) {
    debug_assert!(g.heap.gc_state == GcState::Pause);
    // Flip current_white at cycle START so that sweep below finds
    // objects allocated with the *previous* white (which is now "dead").
    g.heap.current_white ^= 1;
    let cw = g.heap.current_white;
    g.heap.strings.update_current_white(cw);
    g.heap.tables.update_current_white(cw);
    g.heap.funcs.update_current_white(cw);
    g.heap.protos.update_current_white(cw);
    g.heap.upvals.update_current_white(cw);
    g.heap.threads.update_current_white(cw);
    g.heap.cdatas.update_current_white(cw);
    g.heap.userdatas.update_current_white(cw);
    // The `marked` flag records "marked in the current cycle". An
    // interrupted cycle (an allocation-driven step that stops mid-mark)
    // leaves it set; without a reset here, those objects would count as
    // live in this fresh cycle even though they are never re-marked —
    // weak tables would keep them and the sweep would retain them.
    g.heap.tables.reset_marked();
    g.heap.funcs.reset_marked();
    g.heap.threads.reset_marked();
    g.heap.upvals.reset_marked();
    g.heap.protos.reset_marked();
    g.heap.userdatas.reset_marked();
    g.heap.cdatas.reset_marked();
    g.heap.gc_state = GcState::Propagate;
    let mut m = Marker {
        gray: Vec::with_capacity(64),
        strings: &g.heap.strings,
        heap: &g.heap as *const GcHeap,
        atomic: false,
        weak: Vec::new(),
    };
    m.mark_table(g.globals);
    m.mark_table(g.registry);
    for mt in g.basemt.iter().flatten() {
        m.mark_table(*mt);
    }
    for &v in g.mmname.iter() {
        m.mark_value(v);
    }
    m.mark_thread(g.main());
    if let Some(cur) = g.cur_l {
        m.mark_thread(cur);
    }
    for t in g.jit.trace.iter().flatten() {
        m.mark_proto(t.startpt);
        for v in t.ir.kgc_values() {
            m.mark_value(v);
        }
    }
    if let Some(rec) = &g.jit.rec {
        #[cfg(not(target_arch = "wasm32"))]
        {
            m.mark_proto(rec.cur.startpt);
            for v in rec.cur.ir.kgc_values() {
                m.mark_value(v);
            }
        }
    }
    g.heap.gc_gray = m.gray;
}

pub(crate) fn account_func(f: &GcFunc) -> usize {
    size_func(f)
}
pub(crate) fn account_upval() -> usize {
    size_upval()
}

// ── Tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alloc_addresses_are_stable_across_growth() {
        let mut p: Pool<u64> = Pool::new(GcObjectKind::Table);
        let f = p.alloc(42);
        let a = f.addr();
        for i in 0..10000u64 {
            p.alloc(i);
        }
        assert_eq!(f.addr(), a);
        assert_eq!(*f.as_ref(), 42);
    }
    #[test]
    fn free_slots_are_reused() {
        let mut p: Pool<String> = Pool::new(GcObjectKind::String);
        let a = p.alloc("a".into());
        p.free(a);
        assert_eq!(p.len(), 0);
        let b = p.alloc("b".into());
        assert_eq!(b.as_ref(), "b");
        assert_eq!(p.len(), 1);
    }
    #[test]
    fn iter_visits_only_live() {
        let mut p: Pool<u32> = Pool::new(GcObjectKind::Table);
        let a = p.alloc(1);
        p.alloc(2);
        p.free(a);
        let mut v: Vec<u32> = p.iter().copied().collect();
        v.sort_unstable();
        assert_eq!(v, vec![2]);
    }
}
