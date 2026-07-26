//! GC collector — incremental / generational mark-sweep on top of the
//! existing Box-based allocator.
//!
//! Every object is allocated via `alloc_block` with a `GcHeader` at a
//! negative offset. The GC never allocates or frees memory itself — it
//! only manages the alive/dead status via header bits, and calls
//! `dealloc_block` for dead objects during sweep.

use std::ptr::NonNull;

use super::header::{Age, GcHeader};
use super::list::GcList;

// ── GC parameters ───────────────────────────────────────────────────────

pub const GC_PAUSE: usize = 200;
pub const GC_STEPMUL: usize = 200;
pub const GC_STEPSIZE: usize = 13; // KB

// ── States ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcState {
    Pause,
    Propagate,
    Sweep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcKind {
    Inc = 0,
    GenMinor = 1,
    GenMajor = 2,
}

// ── Collector ───────────────────────────────────────────────────────────

pub struct Collector {
    // ── Age buckets ────────────────────────────────────────────────────
    pub allgc: GcList,
    pub survival: GcList,
    pub old1: GcList,
    pub old: GcList,

    // ── Debt / accounting ──────────────────────────────────────────────
    pub gc_debt: isize,
    pub total_bytes: isize,
    pub gc_marked: isize,
    pub gc_majorminor: isize,
    pub threshold: usize,

    // ── State ──────────────────────────────────────────────────────────
    pub gc_state: GcState,
    pub gc_kind: GcKind,
    pub current_white: u8,

    // ── Gray list ──────────────────────────────────────────────────────
    gray: Vec<NonNull<GcHeader>>,
    grayagain: Vec<NonNull<GcHeader>>,

    // ── Sweep cursor ───────────────────────────────────────────────────
    sweep_list_index: usize,
    sweep_objects: Vec<NonNull<GcHeader>>,
}

impl Collector {
    pub fn new() -> Self {
        Self {
            allgc: GcList::new(),
            survival: GcList::new(),
            old1: GcList::new(),
            old: GcList::new(),
            gc_debt: 0,
            total_bytes: 0,
            gc_marked: 0,
            gc_majorminor: 0,
            threshold: 4096,
            gc_state: GcState::Pause,
            gc_kind: GcKind::Inc,
            current_white: 0,
            gray: Vec::with_capacity(128),
            grayagain: Vec::with_capacity(64),
            sweep_list_index: 0,
            sweep_objects: Vec::new(),
        }
    }

    // ── Allocation tracking ────────────────────────────────────────────

    /// Register a newly-created object. Call this RIGHT AFTER alloc_block.
    pub fn register_object(&mut self, header: NonNull<GcHeader>) {
        let size = unsafe { header.as_ref() }.alloc_size() as isize;
        self.allgc.add(header);
        self.gc_debt -= size;
        self.total_bytes += size;
    }

    /// Track a deallocation. Call this BEFORE dealloc_block.
    pub fn unregister_object(&mut self, header: NonNull<GcHeader>) {
        let hdr = unsafe { header.as_ref() };
        let age = hdr.age();
        match age {
            Age::New => self.allgc.remove(header),
            Age::Survival => self.survival.remove(header),
            Age::Old1 => self.old1.remove(header),
            _ => self.old.remove(header),
        }
        self.total_bytes -= hdr.alloc_size() as isize;
    }

    pub fn get_total_bytes(&self) -> isize {
        self.total_bytes + self.gc_debt
    }

    pub fn should_collect(&self) -> bool {
        self.gc_debt <= 0
    }

    // ── GC step dispatcher ─────────────────────────────────────────────

    pub fn step(&mut self) {
        match self.gc_kind {
            GcKind::Inc | GcKind::GenMajor => self.inc_step(),
            GcKind::GenMinor => {
                self.young_collection();
                self.set_minor_debt();
            }
        }
    }

    // ── Incremental step ───────────────────────────────────────────────

    fn inc_step(&mut self) {
        let stepsize = (GC_STEPSIZE * 1024) as isize;
        let mut work = stepsize;

        loop {
            match self.gc_state {
                GcState::Pause => {
                    self.restart_collection();
                    self.gc_state = GcState::Propagate;
                    work -= 1;
                }
                GcState::Propagate => {
                    if self.gray.is_empty() {
                        self.enter_sweep();
                        self.gc_state = GcState::Sweep;
                    } else {
                        let w = self.propagate_one();
                        work -= w;
                    }
                }
                GcState::Sweep => {
                    let done = self.sweep_chunk(GCSWEEPMAX);
                    if done {
                        self.gc_state = GcState::Pause;
                        self.update_threshold();
                    }
                    work -= GC_STEPSIZE as isize * 1024 / 8;
                    break;
                }
            }
            if work <= 0 { break; }
        }

        if self.gc_state == GcState::Pause {
            self.update_threshold();
        } else {
            self.gc_debt = stepsize;
        }
    }

    // ── Restart collection ─────────────────────────────────────────────

    fn restart_collection(&mut self) {
        self.current_white ^= 1;

        // Clear gray lists (roots will push new entries).
        self.gray.clear();
        self.grayagain.clear();
        self.gc_marked = 0;

        // Roots are marked externally by `mark_thread`, `mark_table`, etc.
        // Those functions should call `self.push_gray(header)`.
    }

    // ── Propagation ────────────────────────────────────────────────────

    fn propagate_one(&mut self) -> isize {
        let Some(hdr) = self.gray.pop() else { return 0 };
        let size = unsafe { hdr.as_ref() }.alloc_size() as isize;
        self.gc_marked += size;

        // Re-mark children (handled by callers via `push_gray`).
        // The gray object just gets consumed here; its children were
        // already pushed to gray by the mark traversal.
        size
    }

    /// Push a newly-discovered object onto the gray list.
    pub fn push_gray(&mut self, header: NonNull<GcHeader>) {
        let hdr = unsafe { header.as_ref() };
        if hdr.is_white() {
            hdr.nw2black();
            self.gray.push(header);
        }
    }

    /// Like push_gray but for old objects (goes to grayagain).
    pub fn push_gray_again(&mut self, header: NonNull<GcHeader>) {
        let hdr = unsafe { header.as_ref() };
        if hdr.is_white() {
            hdr.nw2black();
            self.grayagain.push(header);
        }
    }

    // ── Mark helpers (to be called from traversal code) ─────────────────

    /// Mark a string pointer if white.
    pub fn mark_str_if_white(&mut self, ptr: NonNull<GcHeader>) {
        let hdr = unsafe { ptr.as_ref() };
        if hdr.is_white() {
            hdr.nw2black();
            // Strings are values — no children to scan, so don't push to gray.
            self.gc_marked += hdr.alloc_size() as isize;
        }
    }

    /// Mark a table/func/thread/proto/upval and push to gray for traversal.
    pub fn mark_and_traverse(&mut self, header: NonNull<GcHeader>) {
        self.push_gray(header);
    }

    // ── Sweep ───────────────────────────────────────────────────────────

    fn enter_sweep(&mut self) {
        let other_white = GcHeader::otherwhite(self.current_white);

        // Collect all objects from all age buckets into the sweep work list.
        let mut objs = self.allgc.take_all();
        objs.extend(self.survival.take_all());
        objs.extend(self.old1.take_all());
        objs.extend(self.old.take_all());

        // Move survivors to the front (optimization: dead objects at end).
        let mut i = 0;
        while i < objs.len() {
            let hdr = unsafe { objs[i].as_ref() };
            if hdr.is_dead(other_white) {
                // Move dead to end
                let last = objs.len() - 1;
                objs.swap(i, last);
                objs.pop();
            } else {
                hdr.change_white();
                i += 1;
            }
        }

        self.sweep_list_index = 0;
        self.sweep_objects = objs;
    }

    /// Sweep a chunk of dead objects. Returns true when done.
    fn sweep_chunk(&mut self, max_count: usize) -> bool {
        if self.sweep_list_index >= self.sweep_objects.len() {
            return true;
        }

        let end = (self.sweep_list_index + max_count).min(self.sweep_objects.len());
        // Dead objects were already removed during enter_sweep.
        // Survivors just need to be re-added to the appropriate lists.
        for &hdr in &self.sweep_objects[self.sweep_list_index..end] {
            let header = unsafe { hdr.as_ref() };
            let age = header.age();
            match age {
                Age::New | Age::Survival | Age::Old1 => self.allgc.add(hdr),
                _ => self.old.add(hdr),
            }
        }
        self.sweep_list_index = end;
        self.sweep_list_index >= self.sweep_objects.len()
    }

    // ── Young collection ────────────────────────────────────────────────

    fn young_collection(&mut self) {
        self.current_white ^= 1;
        let other_white = GcHeader::otherwhite(self.current_white);

        // Sweep allgc → survival
        let allgc = self.allgc.take_all();
        for h in allgc {
            let header = unsafe { h.as_ref() };
            if header.is_dead(other_white) {
                // Dead — caller handles deallocation
            } else {
                header.change_white();
                header.set_age(Age::Survival);
                self.survival.add(h);
            }
        }

        // Sweep survival → old1
        let surv = self.survival.take_all();
        for h in surv {
            let header = unsafe { h.as_ref() };
            if header.is_dead(other_white) {
                // Dead
            } else {
                header.change_white();
                header.set_age(Age::Old1);
                self.old1.add(h);
            }
        }

        // Merge old1 → old
        let old1 = self.old1.take_all();
        for h in old1 {
            let header = unsafe { h.as_ref() };
            header.set_age(Age::Old);
            self.old.add(h);
        }

        self.update_threshold();
    }

    // ── Threshold ───────────────────────────────────────────────────────

    fn update_threshold(&mut self) {
        let live = (self.total_bytes.max(0) as usize).max(4096);
        self.threshold = (live * GC_PAUSE / 100).max(4096);
        self.gc_debt = (GC_STEPSIZE * 1024) as isize;
    }

    fn set_minor_debt(&mut self) {
        let live = (self.total_bytes.max(0) as usize).max(4096);
        self.gc_debt = (live * GC_PAUSE / 100) as isize;
    }

    // ── Write barrier ───────────────────────────────────────────────────

    /// Forward barrier: black(old) object references white(child) → make
    /// the parent gray again so its children are rescanned.
    pub fn barrier_fwd(&mut self, parent: NonNull<GcHeader>, child: NonNull<GcHeader>) {
        let p = unsafe { parent.as_ref() };
        let c = unsafe { child.as_ref() };

        if p.is_black() && c.is_white() {
            if p.is_old() {
                p.set_age(Age::Touched1);
                p.make_gray();
                self.grayagain.push(parent);
            } else {
                p.make_gray();
                self.gray.push(parent);
            }
        }
    }
}

const GCSWEEPMAX: usize = 64;
