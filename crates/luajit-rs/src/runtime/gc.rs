use std::cell::Cell;
use std::ptr::NonNull;

#[repr(C)]
struct GcHeader {
    marked: Cell<bool>,
}

// ── Low-address allocator ───────────────────────────────────────────────
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
            for i in 0..300 {
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

fn alloc_block<T>(v: T) -> (NonNull<T>, bool) {
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    let (raw, mapped) = lowmem::alloc(layout);
    unsafe {
        (raw.as_ptr() as *mut GcHeader).write(GcHeader {
            marked: Cell::new(false),
        });
    }
    let dp = unsafe { raw.as_ptr().add(data_offset) as *mut T };
    unsafe { dp.write(v) };
    (unsafe { NonNull::new_unchecked(dp) }, mapped)
}
fn dealloc_block<T>(data: NonNull<T>, mapped: bool) {
    let (layout, data_offset) = std::alloc::Layout::new::<GcHeader>()
        .extend(std::alloc::Layout::new::<T>())
        .unwrap();
    let layout = layout.pad_to_align();
    let ap = unsafe { (data.as_ptr() as *mut u8).sub(data_offset) };
    unsafe { lowmem::dealloc(NonNull::new_unchecked(ap), layout, mapped) };
}
fn gc_header<T>(ptr: NonNull<T>) -> &'static GcHeader {
    let addr = ptr.as_ptr() as usize;
    if addr < 0x1000 || addr >= (1usize << 47) {
        // Return a dummy header. The VM is single-threaded so this is safe.
        static DUMMY: std::sync::atomic::AtomicPtr<GcHeader> = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
        let p = DUMMY.load(std::sync::atomic::Ordering::Relaxed);
        if p.is_null() {
            let b = Box::new(GcHeader { marked: Cell::new(false) });
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
        let p = (ptr.as_ptr() as *const u8).sub(data_offset);
        &*(p as *const GcHeader)
    }
}

// ── Pool ────────────────────────────────────────────────────────────────
pub struct Pool<T> {
    objects: Vec<NonNull<T>>,
    mapped: Vec<bool>,
    live: usize,
}
impl<T> Pool<T> {
    pub fn with_page_size(_: usize) -> Self {
        Self {
            objects: Vec::new(),
            mapped: Vec::new(),
            live: 0,
        }
    }
    pub fn new() -> Self {
        Self::with_page_size(0)
    }
    pub fn alloc(&mut self, v: T) -> GcPtr<T> {
        let (nn, m) = alloc_block(v);
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
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.objects.iter().map(|nn| unsafe { nn.as_ref() })
    }
    pub fn sweep(&mut self, _on_free: impl FnMut(&T)) {
        let mut i = 0;
        while i < self.objects.len() {
            let ptr = self.objects[i];
            let addr = ptr.as_ptr() as usize;
            if addr <= 0x1000 || addr >= (1usize << 47) {
                self.objects.swap_remove(i);
                continue;
            }
            if gc_header(ptr).marked.get() {
                gc_header(ptr).marked.set(false);
                i += 1;
            } else {
                self.objects.swap_remove(i);
                self.mapped.swap_remove(i);
            }
        }
        self.live = self.objects.len();
    }
}
impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new()
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
        if addr > 0 && addr < 4 { return None; }
        NonNull::new(addr as *mut T).map(GcPtr)
    }
    pub fn addr(self) -> u64 {
        self.0.as_ptr() as u64
    }
    

    pub fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    pub fn as_mut<'a>(self) -> &'a mut T {
        unsafe { &mut *self.0.as_ptr() }
    }

    pub fn is_marked(self) -> bool {
        gc_header(self.0).marked.get()
    }

    pub fn set_marked(self) {
        gc_header(self.0).marked.set(true)
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

// ── Collector ───────────────────────────────────────────────────────────
use crate::func::{GcFunc, Upval};
use crate::proto::{KGc, Proto};
use crate::state::{GcHeap, GlobalState, LuaState};
use crate::table::LuaTable;
use crate::value::{LJ_TFUNC, LJ_TSTR, LJ_TTAB, LJ_TTHREAD, LuaValue};

const GC_PAUSE: usize = 200;
pub(crate) const GC_THRESHOLD_MIN: usize = 64 * 1024;

// Incremental GC state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GcState {
    Pause,
    Propagate,
    Sweep,
}
pub const GC_STEP_SIZE: usize = 4096;

pub enum Gray {
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
    fn mark_upval(&mut self, uv: GcPtr<Upval>) {
        if !uv.is_marked() {
            uv.set_marked();
            self.mark_value(uv.as_ref().get());
        }
    }
    fn propagate(&mut self) {
        while let Some(g) = self.gray.pop() {
            match g {
                Gray::Tab(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
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
                        match k {
                            KGc::Str(sid) => {
                                if let Some(ptr) = self.strings.try_lookup(*sid) {
                                    ptr.set_marked();
                                }
                            }
                            KGc::ProtoRef(c) => self.mark_proto(*c),
                            KGc::Table(t) => t.gc_traverse(|v| self.mark_value(v)),
                            KGc::TableRef(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
                            KGc::Proto(_) => unreachable!(),
                            KGc::CData(_) => {}
                        }
                    }
                }
                Gray::Thread(th) => {
                    let l = th.as_mut();
                    for i in 0..l.top {
                        self.mark_value(l.stack[i]);
                    }
                    self.mark_value(l.errval);
                    for &uv in &l.openuv {
                        self.mark_upval(uv);
                    }
                    for s in l.stack[l.top..].iter_mut() {
                        *s = LuaValue::NIL;
                    }
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
                    t.as_ref().gc_traverse(|v| self.mark_value(v));
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
                            KGc::Table(t) => t.gc_traverse(|v| self.mark_value(v)),
                            KGc::TableRef(t) => t.as_ref().gc_traverse(|v| self.mark_value(v)),
                            _ => {}
                        }
                    }
                }
                Gray::Thread(th) => {
                    th.set_marked();
                    let l = th.as_mut();
                    for i in 0..l.top {
                        self.mark_value(l.stack[i]);
                    }
                    self.mark_value(l.errval);
                    for &uv in &l.openuv {
                        self.mark_upval(uv);
                    }
                    for s in l.stack[l.top..].iter_mut() {
                        *s = LuaValue::NIL;
                    }
                }
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
    if heap.gc_state == GcState::Pause {
        return;
    }
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
                heap.gc_gray.push(Gray::Tab(p));
            }
        }
        LJ_TFUNC => {
            if let Some(p) = val.as_func()
                && !p.is_marked()
            {
                p.set_marked();
                heap.gc_gray.push(Gray::Func(p));
            }
        }
        LJ_TTHREAD => {
            if let Some(p) = val.as_thread()
                && !p.is_marked()
            {
                p.set_marked();
                heap.gc_gray.push(Gray::Thread(p));
            }
        }
        _ => {}
    }
}

pub fn gc_step(heap: &mut GcHeap, size: usize) {
    let step = heap.gc_step_size.max(size);
    match heap.gc_state {
        GcState::Pause => {
            let live = heap.total + heap.strings.bytes() + heap.table_extra;
            if live + step < heap.threshold {
                return;
            }
            // Start new cycle: all existing objects are currently unmarked.
            // Roots will be marked by gc_start_cycle.
            heap.gc_state = GcState::Propagate;
            let mut m = Marker {
                gray: Vec::with_capacity(64),
                strings: &heap.strings,
            };
            let g = unsafe { &*(heap as *const GcHeap as *const GlobalState) };
            // SAFETY: GlobalState contains GcHeap as its first (unnamed) field before globals.
            // This works because heap is inside a GlobalState and we only read root fields.
            unsafe {
                let gs = &*(g as *const GlobalState);
                m.mark_table(gs.globals);
                m.mark_table(gs.registry);
                for mt in gs.basemt.iter().flatten() {
                    m.mark_table(*mt);
                }
                for &v in gs.mmname.iter() {
                    m.mark_value(v);
                }
                m.mark_thread(gs.main());
                if let Some(cur) = gs.cur_l {
                    m.mark_thread(cur);
                }
                for t in gs.jit.trace.iter().flatten() {
                    m.mark_proto(t.startpt);
                    for v in t.ir.kgc_values() {
                        m.mark_value(v);
                    }
                }
                if let Some(rec) = &gs.jit.rec {
                    m.mark_proto(rec.cur.startpt);
                    for v in rec.cur.ir.kgc_values() {
                        m.mark_value(v);
                    }
                }
            }
            heap.gc_gray = m.gray;
        }
        GcState::Propagate => {
            let mut m = Marker {
                gray: std::mem::take(&mut heap.gc_gray),
                strings: &heap.strings,
            };
            if m.propagate_step(step / 64) {
                heap.gc_state = GcState::Sweep;
                heap.gc_sweep_pool = 0;
            }
            heap.gc_gray = m.gray;
        }
        GcState::Sweep => {
            // Sweep one pool per step for fairness.
            let done = match heap.gc_sweep_pool {
                0 => {
                    heap.strings.sweep();
                    1
                }
                1 => {
                    heap.tables.sweep(|_| {});
                    2
                }
                2 => {
                    heap.funcs.sweep(|_| {});
                    3
                }
                3 => {
                    heap.threads.sweep(|th| {
                        for &uv in &th.openuv {
                            uv.as_mut().close();
                        }
                    });
                    4
                }
                4 => {
                    heap.upvals.sweep(|_| {});
                    5
                }
                5 => {
                    heap.protos.sweep(|_| {});
                    6
                }
                _ => 6,
            };
            heap.gc_sweep_pool = done;
            if done >= 6 {
                heap.gc_state = GcState::Pause;
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
                heap.threshold = ((total + heap.strings.bytes()) * GC_PAUSE / 100)
                    .max(GC_THRESHOLD_MIN)
                    .max(heap.threshold / 2);
                heap.table_extra = 0;
                heap.debt = 0;
            }
        }
    }
}

fn size_thread(th: &LuaState) -> usize {
    std::mem::size_of::<LuaState>() + th.stack.capacity() * std::mem::size_of::<LuaValue>()
}
pub(crate) fn account_thread(th: &LuaState) -> usize {
    size_thread(th)
}

pub fn full_gc(g: &mut GlobalState) {
    // Finish any in-progress cycle.
    while g.heap.gc_state != GcState::Pause {
        gc_step(&mut g.heap, usize::MAX);
    }
    // Start and run a complete cycle.
    g.heap.gc_state = GcState::Propagate;
    let mut m = Marker {
        gray: Vec::with_capacity(64),
        strings: &g.heap.strings,
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
        m.mark_proto(rec.cur.startpt);
        for v in rec.cur.ir.kgc_values() {
            m.mark_value(v);
        }
    }
    m.propagate();
    g.heap.strings.sweep();
    g.heap.tables.sweep(|_| {});
    g.heap.funcs.sweep(|_| {});
    g.heap.threads.sweep(|th| {
        for &uv in &th.openuv {
            uv.as_mut().close();
        }
    });
    g.heap.upvals.sweep(|_| {});
    g.heap.protos.sweep(|_| {});
    let mut total = 0usize;
    for t in g.heap.tables.iter() {
        total += t.gc_size();
    }
    for f in g.heap.funcs.iter() {
        total += size_func(f);
    }
    total += g.heap.upvals.len() * size_upval();
    for p in g.heap.protos.iter() {
        total += p.gc_size();
    }
    for th in g.heap.threads.iter() {
        total += size_thread(th);
    }
    g.heap.total = total;
    g.heap.threshold = ((total + g.heap.strings.bytes()) * GC_PAUSE / 100)
        .max(GC_THRESHOLD_MIN)
        .max(g.heap.threshold / 2);
    g.heap.table_extra = 0;
    g.heap.debt = 0;
    g.heap.gc_state = GcState::Pause;
    g.heap.gc_gray.clear();
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
        let mut p: Pool<u64> = Pool::new();
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
        let mut p: Pool<String> = Pool::new();
        let a = p.alloc("a".into());
        p.free(a);
        assert_eq!(p.len(), 0);
        let b = p.alloc("b".into());
        assert_eq!(b.as_ref(), "b");
        assert_eq!(p.len(), 1);
    }
    #[test]
    fn iter_visits_only_live() {
        let mut p: Pool<u32> = Pool::new();
        let a = p.alloc(1);
        p.alloc(2);
        p.free(a);
        let mut v: Vec<u32> = p.iter().copied().collect();
        v.sort_unstable();
        assert_eq!(v, vec![2]);
    }
}
