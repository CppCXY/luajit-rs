use std::cell::Cell;
use std::ptr::NonNull;

/// Header placed immediately before every GC-allocated object. The
/// memory layout is [GcHeader | T], and `GcPtr<T>` points to T.
/// The mark bit replaces the old `Slot.marked`.
#[repr(C)]
struct GcHeader {
    marked: Cell<bool>,
}

/// Low-address allocator for GC objects. The global allocator (mimalloc
/// or system malloc) is tried first; if it returns a pointer above the
/// 47-bit NaN-boxing limit, we fall back to hinted OS mmap.
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
            None => panic!("cannot allocate GC pages below 2^47 (NaN-boxing limit)"),
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
        const PROT_READ: i32 = 1;
        const PROT_WRITE: i32 = 2;
        const MAP_PRIVATE: i32 = 0x02;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        const MAP_ANON: i32 = 0x20;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        const MAP_ANON: i32 = 0x1000;
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
            let layout = std::alloc::Layout::from_size_align(4096, 16).unwrap();
            let (p, mapped) = alloc(layout);
            assert!((p.as_ptr() as u64) + 4096 <= LIMIT);
            unsafe {
                p.as_ptr().write_bytes(0x5A, 4096);
                dealloc(p, layout, mapped);
            }
        }

        #[test]
        fn os_probe_survives_hundreds_of_pages() {
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

fn alloc_block<T>(v: T) -> (NonNull<T>, bool) {
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    let (raw, mapped) = lowmem::alloc(layout);
    let header_ptr = raw.as_ptr() as *mut GcHeader;
    unsafe {
        header_ptr.write(GcHeader {
            marked: Cell::new(false),
        });
    }
    let data_ptr = unsafe { raw.as_ptr().add(data_offset) as *mut T };
    unsafe { data_ptr.write(v) };
    (unsafe { NonNull::new_unchecked(data_ptr) }, mapped)
}

fn dealloc_block<T>(data: NonNull<T>, mapped: bool) {
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    // Reconstruct the start of the allocation block from the data pointer
    // and the precomputed offset (handles alignment padding correctly).
    let alloc_ptr =
        unsafe { (data.as_ptr() as *mut u8).sub(data_offset) };
    unsafe { lowmem::dealloc(NonNull::new_unchecked(alloc_ptr), layout, mapped) };
}

/// Get a reference to the GcHeader for a GC object. The header is at
/// a fixed `data_offset` before the data pointer (preserved from
/// allocation time). We reconstruct the offset by calling Layout::extend
/// again — the result is deterministic for a given T.
fn gc_header<T>(ptr: NonNull<T>) -> &'static GcHeader {
    let (_, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    unsafe {
        let p = (ptr.as_ptr() as *const u8).sub(data_offset);
        &*(p as *const GcHeader)
    }
}

/// Object pool: each object is individually allocated via the global
/// allocator, tracked in a Vec. No pages, no slots, no free-list
/// fragmentation. Sweep is O(allocated objects), memory is returned
/// to the allocator on free.
pub struct Pool<T> {
    objects: Vec<NonNull<T>>,
    mapped: Vec<bool>,
    live: usize,
}

impl<T> Pool<T> {
    pub fn with_page_size(_page_cap: usize) -> Pool<T> {
        Pool {
            objects: Vec::new(),
            mapped: Vec::new(),
            live: 0,
        }
    }

    pub fn new() -> Pool<T> {
        Pool::with_page_size(0)
    }

    pub fn alloc(&mut self, v: T) -> GcPtr<T> {
        let (nn, mapped) = alloc_block(v);
        self.objects.push(nn);
        self.mapped.push(mapped);
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

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.objects.iter().map(|nn| unsafe { nn.as_ref() })
    }

    pub fn sweep(&mut self, mut on_free: impl FnMut(&T)) {
        let mut i = 0;
        while i < self.objects.len() {
            let ptr = self.objects[i];
            let header = gc_header(ptr);
            if header.marked.get() {
                header.marked.set(false);
                i += 1;
            } else {
                unsafe {
                    on_free(ptr.as_ref());
                    ptr.as_ptr().drop_in_place();
                }
                dealloc_block(ptr, self.mapped[i]);
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
            }
        }
        self.live = self.objects.len();
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Pool<T> {
        Pool::new()
    }
}

impl<T> Drop for Pool<T> {
    fn drop(&mut self) {
        for (i, &nn) in self.objects.iter().enumerate() {
            unsafe { nn.as_ptr().drop_in_place() };
            dealloc_block(nn, self.mapped[i]);
        }
        self.objects.clear();
    }
}

/// A pointer to a GC-allocated object. Fits in the 47-bit NaN-boxed
/// payload of a `LuaValue`. The GC header (mark bit) lives immediately
/// before the pointed-to data.
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

    /// The mark bit in the GC header (LuaJIT's `gch.marked`).
    #[inline]
    pub fn is_marked(self) -> bool {
        gc_header(self.0).marked.get()
    }

    #[inline]
    pub fn set_marked(self) {
        gc_header(self.0).marked.set(true);
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
                    let pt = p.as_ref();
                    // Mark the source name string (not stored in kgc).
                    if let Some(sid) = pt.source
                        && let Some(ptr) = self.strings.try_lookup(sid) {
                            ptr.set_marked();
                        }
                    for k in &pt.kgc {
                        match k {
                            KGc::Str(sid) => {
                                if let Some(ptr) = self.strings.try_lookup(*sid) {
                                    ptr.set_marked();
                                }
                            }
                            KGc::ProtoRef(child) => self.mark_proto(*child),
                            // Template tables are owned by the proto (not
                            // heap objects); mark their contents in place.
                            KGc::Table(t) => t.gc_traverse(|v| self.mark_value(v)),
                            KGc::TableRef(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
                            KGc::Proto(_) => unreachable!("unregistered child proto in heap"),
                            KGc::CData(_) => {} // raw byte data, no GC references
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
    let new_threshold =
        ((total + heap.strings.bytes()) * GC_PAUSE / 100).max(GC_THRESHOLD_MIN);
    // Do not let the threshold collapse after a large heap was swept:
    // gradual deflation avoids excessive GC cycles when the live set
    // is small but the pool page count (and thus sweep cost) is large.
    heap.threshold = new_threshold.max(heap.threshold / 2);
    // Table growth is now baked into the live estimate (gc_size counts
    // the grown capacities): reset the growth debt.
    heap.table_extra = 0;
    heap.debt = 0;
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
        pool.free(a);
        assert_eq!(pool.len(), 0);
        let b = pool.alloc("b".to_string());
        assert_eq!(b.as_ref(), "b");
        assert_eq!(pool.len(), 1);
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
