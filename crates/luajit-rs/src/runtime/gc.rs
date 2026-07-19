use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// Default slots per pool page. `Pool::with_page_size` picks a per-type
/// count so pages stay near a sensible byte size: small objects (upvalues,
/// strings) pack many per page, huge ones (thread states) only a few.
const POOL_PAGE_DEFAULT: usize = 64;

/// A pool slot. `data` must stay the first field (`repr(C)`) so a pointer to
/// the payload can be cast back to the slot for freeing and for the mark
/// bit (the slot header is our stand-in for LuaJIT's `GCheader.marked`).
#[repr(C)]
struct Slot<T> {
    data: MaybeUninit<T>,
    live: bool,
    marked: bool,
}

/// Low-address page memory for the pools. GC object addresses travel in
/// the 47-bit payload of a NaN-boxed `LuaValue`; on platforms whose
/// user-space VA exceeds 47 bits (e.g. Linux/AArch64 with 48-bit VA) the
/// global allocator can return pointers above that limit, so the pages
/// fall back to hint-probed `mmap` like LuaJIT's `lj_alloc` does.
mod lowmem {
    use std::alloc::Layout;
    use std::ptr::NonNull;

    /// NaN-boxed pointers must fit the 47-bit LuaValue payload.
    const LIMIT: u64 = 1 << 47;

    /// Allocate `layout` with the whole block below 2^47. The flag in
    /// the result records whether the block came from the OS mapper
    /// (true) or the global allocator (false).
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
            None => panic!("cannot allocate GC pages below 2^47 (NaN-boxing limit)"),
        }
    }

    /// Free a block from `alloc`.
    ///
    /// # Safety
    /// `ptr`/`layout`/`mapped` must match a single previous `alloc`.
    pub unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout, mapped: bool) {
        if mapped {
            os_free(ptr.as_ptr(), layout.size().max(1));
        } else {
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
    }

    /// Pseudo-random probe hints in [2^38, 2^46], 64K aligned. The seed
    /// is global so successive allocations never replay the same hint
    /// sequence (a page already mapped at a hint would fail every later
    /// probe otherwise).
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

    /// Bump pointer past the last successful mapping: consecutive pages
    /// pack into one region instead of burning fresh probe hints.
    fn hint_state() -> &'static std::sync::atomic::AtomicU64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        &NEXT
    }

    /// Probe loop shared by the OS backends: `map(hint, size)` returns a
    /// mapping (kernel-placed anywhere on some systems) or null.
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
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_PRIVATE: i32 = 0x02;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        const MAP_ANON: i32 = 0x20;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        const MAP_ANON: i32 = 0x1000;
        /// The hint is only binding on Linux (elsewhere the kernel may
        /// place the mapping anywhere; the result is checked either way).
        #[cfg(any(target_os = "linux", target_os = "android"))]
        const MAP_FIXED_NOREPLACE: i32 = 0x10_0000;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        const MAP_FIXED_NOREPLACE: i32 = 0;

        unsafe extern "C" {
            fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
            -> *mut u8;
        }
        probe(
            size,
            |hint, size| unsafe {
                mmap(
                    hint as *mut u8,
                    size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON | MAP_FIXED_NOREPLACE,
                    -1,
                    0,
                )
            },
            os_free,
        )
    }

    #[cfg(unix)]
    fn os_free(ptr: *mut u8, size: usize) {
        unsafe extern "C" {
            fn munmap(addr: *mut u8, len: usize) -> i32;
        }
        unsafe { munmap(ptr, size) };
    }

    #[cfg(windows)]
    fn os_alloc_low(size: usize) -> Option<NonNull<u8>> {
        // Practically unreachable (Windows user space is 47-bit), kept
        // for completeness.
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RESERVE: u32 = 0x2000;
        const PAGE_READWRITE: u32 = 0x04;
        unsafe extern "system" {
            fn VirtualAlloc(addr: *mut u8, size: usize, ty: u32, prot: u32) -> *mut u8;
        }
        probe(
            size,
            |hint, size| unsafe {
                VirtualAlloc(
                    hint as *mut u8,
                    size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            },
            os_free,
        )
    }

    #[cfg(windows)]
    fn os_free(ptr: *mut u8, _size: usize) {
        const MEM_RELEASE: u32 = 0x8000;
        unsafe extern "system" {
            fn VirtualFree(addr: *mut u8, size: usize, ty: u32) -> i32;
        }
        unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn os_probe_returns_low_writable_memory() {
            let size = 1 << 20;
            let p = os_alloc_low(size).expect("probe failed");
            assert!((p.as_ptr() as u64) + size as u64 <= LIMIT);
            unsafe {
                p.as_ptr().write(0xAB);
                p.as_ptr().add(size - 1).write(0xCD);
                assert_eq!(p.as_ptr().read(), 0xAB);
                os_free(p.as_ptr(), size);
            }
        }

        #[test]
        fn alloc_dealloc_roundtrip() {
            let layout = Layout::from_size_align(4096, 16).unwrap();
            let (p, mapped) = alloc(layout);
            assert!((p.as_ptr() as u64) + 4096 <= LIMIT);
            unsafe {
                p.as_ptr().write_bytes(0x5A, 4096);
                dealloc(p, layout, mapped);
            }
        }

        #[test]
        fn os_probe_survives_hundreds_of_pages() {
            // Regression: the hint sequence must not replay (a replayed
            // hint hits its own earlier mapping and fails forever).
            let size = 1 << 16;
            let mut pages = Vec::new();
            for i in 0..300 {
                let p = os_alloc_low(size).expect("probe failed mid-run");
                assert!(
                    (p.as_ptr() as u64) + size as u64 <= LIMIT,
                    "page {i} too high"
                );
                unsafe { p.as_ptr().write_bytes(0x77, size) };
                pages.push(p);
            }
            for p in pages {
                os_free(p.as_ptr(), size);
            }
        }
    }
}

/// One pool page: a raw array of slots in low memory (see `lowmem`).
/// Live objects are dropped by `Pool::sweep`/`Pool::drop`; the page only
/// releases the memory.
struct RawPage<T> {
    ptr: NonNull<Slot<T>>,
    cap: usize,
    mapped: bool,
}

impl<T> RawPage<T> {
    fn layout(cap: usize) -> std::alloc::Layout {
        std::alloc::Layout::array::<Slot<T>>(cap).expect("pool page layout overflow")
    }

    fn new(cap: usize) -> RawPage<T> {
        let (raw, mapped) = lowmem::alloc(Self::layout(cap));
        let ptr = raw.cast::<Slot<T>>();
        unsafe {
            for i in 0..cap {
                ptr.as_ptr().add(i).write(Slot {
                    data: MaybeUninit::uninit(),
                    live: false,
                    marked: false,
                });
            }
        }
        RawPage { ptr, cap, mapped }
    }

    fn slots(&self) -> &[Slot<T>] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap) }
    }

    fn slots_mut(&mut self) -> &mut [Slot<T>] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap) }
    }
}

impl<T> Drop for RawPage<T> {
    fn drop(&mut self) {
        unsafe { lowmem::dealloc(self.ptr.cast(), Self::layout(self.cap), self.mapped) };
    }
}

/// A typed, stable-address object pool.
///
/// Objects are allocated inside fixed-size pages (`Box<[Slot<T>]>`); pages
/// are never reallocated or moved, so a `GcPtr<T>` stays valid for the life
/// of the pool (or until the object is explicitly freed). The per-page slot
/// count is chosen at construction so objects of different sizes get pages
/// of roughly the same byte size (small objects → many slots; huge objects
/// → few).
pub struct Pool<T> {
    pages: Vec<RawPage<T>>,
    free: Vec<NonNull<Slot<T>>>,
    live: usize,
    page_cap: usize,
}

impl<T> Pool<T> {
    pub fn with_page_size(page_cap: usize) -> Pool<T> {
        Pool {
            pages: Vec::new(),
            free: Vec::new(),
            live: 0,
            page_cap: page_cap.max(1),
        }
    }

    /// Legacy shortcut: 64 slots per page (medium-sized objects).
    pub fn new() -> Pool<T> {
        Pool::with_page_size(POOL_PAGE_DEFAULT)
    }

    fn add_page(&mut self) {
        let mut page = RawPage::new(self.page_cap);
        for s in page.slots_mut().iter_mut().rev() {
            self.free.push(NonNull::from(s));
        }
        self.pages.push(page);
    }

    pub fn alloc(&mut self, v: T) -> GcPtr<T> {
        if self.free.is_empty() {
            self.add_page();
        }
        let mut slot = self.free.pop().unwrap();
        self.live += 1;
        unsafe {
            let s = slot.as_mut();
            debug_assert!(!s.live);
            s.data.write(v);
            s.live = true;
            s.marked = false;
            GcPtr::new(NonNull::new_unchecked(s.data.as_mut_ptr()))
        }
    }

    /// Return an object's slot to the free list, dropping the object.
    /// The caller must guarantee no live `GcPtr` to it remains (this is the
    /// collector's job once implemented).
    pub fn free(&mut self, p: GcPtr<T>) {
        unsafe {
            let slot = p.0.as_ptr() as *mut Slot<T>;
            debug_assert!((*slot).live);
            (*slot).data.assume_init_drop();
            (*slot).live = false;
            self.free.push(NonNull::new_unchecked(slot));
        }
        self.live -= 1;
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Iterate all live objects (linear page walk, used by GC sweeps).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.pages
            .iter()
            .flat_map(|p| p.slots().iter())
            .filter(|s| s.live)
            .map(|s| unsafe { s.data.assume_init_ref() })
    }

    /// Sweep phase: free every live-but-unmarked object (calling `on_free`
    /// with it just before it is dropped) and clear the mark on survivors.
    /// The pool equivalent of LuaJIT's `gc_sweep` over the GC object chain.
    pub fn sweep(&mut self, mut on_free: impl FnMut(&T)) {
        let mut free = std::mem::take(&mut self.free);
        let mut live = 0;
        for page in &mut self.pages {
            for s in page.slots_mut() {
                if !s.live {
                    continue;
                }
                if s.marked {
                    s.marked = false;
                    live += 1;
                } else {
                    unsafe {
                        on_free(s.data.assume_init_ref());
                        s.data.assume_init_drop();
                    }
                    s.live = false;
                    free.push(NonNull::from(s));
                }
            }
        }
        self.free = free;
        self.live = live;
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Pool<T> {
        Pool::new()
    }
}

impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        for page in &mut self.pages {
            for s in page.slots_mut() {
                if s.live {
                    unsafe { s.data.assume_init_drop() };
                    s.live = false;
                }
            }
        }
    }
}

/// A pointer to a pool-allocated GC object.
///
/// This is the Rust stand-in for LuaJIT's `GCRef`: the raw address fits the
/// 47-bit payload of a `LuaValue`. Dereferencing is safe *by convention*:
/// objects live in stable pool pages, the collector only frees objects
/// proven unreachable, and the VM is single-threaded. All `unsafe` is
/// confined here. Every `GcPtr` must point into a `Pool` slot (the mark
/// bit lives in the slot header behind the payload).
pub struct GcPtr<T>(NonNull<T>);

impl<T> GcPtr<T> {
    pub(crate) fn new(p: NonNull<T>) -> GcPtr<T> {
        debug_assert!(
            (p.as_ptr() as u64) < (1u64 << 47),
            "pointer exceeds the 47-bit LuaValue payload"
        );
        GcPtr(p)
    }

    /// Reconstruct from a `LuaValue` payload. Returns `None` for a zero
    /// payload (e.g. the template-table marker).
    pub fn from_addr(addr: u64) -> Option<GcPtr<T>> {
        NonNull::new(addr as *mut T).map(GcPtr)
    }

    pub fn addr(self) -> u64 {
        self.0.as_ptr() as u64
    }

    #[allow(clippy::should_implement_trait)]
    pub fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_mut<'a>(self) -> &'a mut T {
        unsafe { &mut *self.0.as_ptr() }
    }

    #[inline]
    fn slot(self) -> *mut Slot<T> {
        self.0.as_ptr() as *mut Slot<T>
    }

    /// The mark bit in the pool-slot header (LuaJIT's `gch.marked`).
    #[inline]
    pub fn is_marked(self) -> bool {
        unsafe { (*self.slot()).marked }
    }

    #[inline]
    pub fn set_marked(self) {
        unsafe { (*self.slot()).marked = true }
    }
}

impl<T> Clone for GcPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcPtr<T> {}

impl<T> PartialEq for GcPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for GcPtr<T> {}

impl<T> std::hash::Hash for GcPtr<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> std::fmt::Debug for GcPtr<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcPtr({:p})", self.0.as_ptr())
    }
}

// -- The collector (port of lj_gc.c's mark & sweep) -------------------------
//
// Same algorithm and traversal order as LuaJIT's collector, minus the
// incremental machinery: LuaJIT interleaves propagation with the mutator
// (GCSpropagate + write barriers + a two-white scheme to tell "new since
// sweep started" from "dead"); we always run mark → sweep atomically at an
// allocation safe point, so one mark bit and no barriers suffice. Weak
// tables and finalizers do not exist yet (no __mode/__gc in this fork).
//
// Dead keys follow LuaJIT's policy (lj_obj.h): a hash node whose value is
// nil does not keep its key alive; the stale key reference is left in the
// node and is never dereferenced, only compared by identity. A false
// bit-identical match after address reuse yields the node whose value is
// nil, which is exactly the right answer.

use crate::func::{GcFunc, Upval};
use crate::proto::{KGc, Proto};
use crate::state::{GlobalState, LuaState};
use crate::table::LuaTable;
use crate::value::{LJ_TFUNC, LJ_TSTR, LJ_TTAB, LJ_TTHREAD, LuaValue};

/// GC pause: new threshold = live estimate * `GC_PAUSE` / 100 (LuaJIT's
/// default `LUAI_GCPAUSE`).
const GC_PAUSE: usize = 200;

/// Lower bound for the threshold, so tiny heaps do not collect constantly.
pub(crate) const GC_THRESHOLD_MIN: usize = 64 * 1024;

/// A gray object awaiting traversal (LuaJIT chains these through
/// `gch.gclist`; a worklist vector is the STW equivalent).
enum Gray {
    Tab(GcPtr<LuaTable>),
    Func(GcPtr<GcFunc>),
    Proto(GcPtr<Proto>),
    Thread(GcPtr<LuaState>),
}

struct Marker<'g> {
    gray: Vec<Gray>,
    strings: &'g crate::string::Interner,
}

impl<'g> Marker<'g> {
    /// `gc_marktv`: mark the object a value references, queueing
    /// traversable objects (tables/functions) on the gray list.
    fn mark_value(&mut self, v: LuaValue) {
        match v.itype() {
            LJ_TSTR => {
                if let Some(p) = v.as_string() {
                    p.set_marked(); // strings are leaves (black immediately)
                }
            }
            LJ_TTAB => {
                // `as_table` is None for the zero-payload template marker.
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
            _ => {}
        }
    }

    fn mark_thread(&mut self, th: GcPtr<LuaState>) {
        if !th.is_marked() {
            th.set_marked();
            self.gray.push(Gray::Thread(th));
        }
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

    /// `gc_mark` of a GCupval: reading through `uv->v` covers both the
    /// open (stack slot) and closed (inline `tv`) cases.
    fn mark_upval(&mut self, uv: GcPtr<Upval>) {
        if !uv.is_marked() {
            uv.set_marked();
            self.mark_value(uv.as_ref().get());
        }
    }

    /// `gc_propagate_gray`: empty the gray list, turning objects black.
    fn propagate(&mut self) {
        while let Some(g) = self.gray.pop() {
            match g {
                // gc_traverse_tab (no metatable field / weak modes yet).
                Gray::Tab(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
                // gc_traverse_func.
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
                // gc_traverse_proto: collectable constants.
                Gray::Proto(p) => {
                    for k in &p.as_ref().kgc {
                        match k {
                            KGc::Str(sid) => self.strings.lookup_ptr(*sid).set_marked(),
                            KGc::ProtoRef(child) => self.mark_proto(*child),
                            // Template tables are owned by the proto (not
                            // heap objects); mark their contents in place.
                            KGc::Table(t) => t.gc_traverse(|v| self.mark_value(v)),
                            KGc::TableRef(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
                            KGc::Proto(_) => unreachable!("unregistered child proto in heap"),
                        }
                    }
                }
                // gc_traverse_thread: the whole used stack (frame-link
                // slots decode as harmless numbers), the error value and
                // the open-upvalue list. Slots above `top` are cleared,
                // exactly like the GCSatomic branch of gc_traverse_thread:
                // anything below `top` survived the last cycle, so a later
                // `top` raise never exposes a dangling value.
                Gray::Thread(th) => {
                    let l = th.as_mut();
                    for i in 0..l.top {
                        self.mark_value(l.stack[i]);
                    }
                    self.mark_value(l.errval);
                    for &uv in &l.openuv {
                        self.mark_upval(uv);
                    }
                    // Suspend::Call's saved closure is reachable via
                    // stack[base-2], which is below top — already marked.
                    for slot in l.stack[l.top..].iter_mut() {
                        *slot = LuaValue::NIL;
                    }
                }
            }
        }
    }
}

/// Object size estimates for the allocation accounting (LuaJIT's
/// `gc.total`). Approximate: Rust-side reallocations (table rehash, vector
/// growth) are folded in when the total is recomputed after each sweep.
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

/// A full GC cycle: mark all roots, propagate, sweep every pool and reset
/// the threshold — `lj_gc_fullgc`, with the phases of `gc_onestep`
/// (mark start → propagate → atomic → sweepstring → sweep) run back to
/// back. Must only be called at a safe point: every live object reachable
/// from Rust locals must also be anchored on a stack or in a root.
pub fn full_gc(g: &mut GlobalState) {
    // -- Mark phase (gc_mark_start + propagate + atomic) --
    let mut m = Marker {
        gray: Vec::with_capacity(64),
        strings: &g.heap.strings,
    };
    m.mark_table(g.globals);
    m.mark_table(g.registry);
    for mt in g.basemt.iter().flatten() {
        m.mark_table(*mt);
    }
    // GCROOT_MMNAME: the interned metamethod name strings.
    for &v in g.mmname.iter() {
        m.mark_value(v);
    }
    // Thread roots: the main thread is permanent; the currently running
    // thread and every thread in the active resume chain are reachable
    // through the resumer's stack (the coroutine value is an argument of
    // the `resume` C frame), so marking main + cur_l covers everything.
    m.mark_thread(g.main());
    if let Some(cur) = g.cur_l {
        m.mark_thread(cur);
    }
    // JIT roots: completed traces and any active recording keep their
    // start prototype and KGC constants alive (a trace is a GC root in
    // LuaJIT, too).
    for t in g.jit.trace.iter().flatten() {
        m.mark_proto(t.startpt);
        for v in t.ir.kgc_values() {
            m.mark_value(v);
        }
    }
    if let Some(rec) = &g.jit.rec {
        m.mark_proto(rec.cur.startpt);
        for v in rec.cur.ir.kgc_values() {
            m.mark_value(v);
        }
    }
    m.propagate();

    // -- Sweep phase (GCSsweepstring + GCSsweep) --
    let heap = &mut g.heap;
    heap.strings.sweep();
    heap.tables.sweep(|_| {});
    heap.funcs.sweep(|_| {});
    // Threads are swept before upvalues: a dying coroutine first closes
    // its open upvalues (PUC's luaF_close on thread free), so surviving
    // closures keep valid values after the stack memory is dropped.
    heap.threads.sweep(|th| {
        for &uv in &th.openuv {
            uv.as_mut().close();
        }
    });
    heap.upvals.sweep(|_| {});
    heap.protos.sweep(|_| {});

    // -- Recompute the live estimate and set the next threshold --
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
    heap.total = total;
    heap.threshold = ((total + heap.strings.bytes()) * GC_PAUSE / 100).max(GC_THRESHOLD_MIN);
    // Table growth is now baked into the live estimate (gc_size counts
    // the grown capacities): reset the growth debt.
    crate::table::TABLE_EXTRA.with(|c| c.set(0));
}

/// Allocation-time cost bookkeeping (the `lj_mem_newgco` side).
pub(crate) fn account_func(f: &GcFunc) -> usize {
    size_func(f)
}

pub(crate) fn account_upval() -> usize {
    size_upval()
}

fn size_thread(th: &LuaState) -> usize {
    std::mem::size_of::<LuaState>() + th.stack.capacity() * std::mem::size_of::<LuaValue>()
}

pub(crate) fn account_thread(th: &LuaState) -> usize {
    size_thread(th)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_addresses_are_stable_across_growth() {
        let mut pool: Pool<u64> = Pool::new();
        let first = pool.alloc(42);
        let addr = first.addr();
        for i in 0..10_000u64 {
            pool.alloc(i);
        }
        assert_eq!(first.addr(), addr);
        assert_eq!(*first.as_ref(), 42);
        assert_eq!(pool.len(), 10_001);
    }

    #[test]
    fn free_slots_are_reused() {
        let mut pool: Pool<String> = Pool::new();
        let a = pool.alloc("a".to_string());
        let addr = a.addr();
        pool.free(a);
        assert_eq!(pool.len(), 0);
        let b = pool.alloc("b".to_string());
        assert_eq!(b.addr(), addr);
        assert_eq!(b.as_ref(), "b");
    }

    #[test]
    fn iter_visits_only_live() {
        let mut pool: Pool<u32> = Pool::new();
        let a = pool.alloc(1);
        let _b = pool.alloc(2);
        pool.free(a);
        let mut v: Vec<u32> = pool.iter().copied().collect();
        v.sort_unstable();
        assert_eq!(v, vec![2]);
    }
}
