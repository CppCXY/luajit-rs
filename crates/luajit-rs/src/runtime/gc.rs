use std::mem::MaybeUninit;
use std::ptr::NonNull;

/// Slots per pool page.
const POOL_PAGE: usize = 64;

/// A pool slot. `data` must stay the first field (`repr(C)`) so a pointer to
/// the payload can be cast back to the slot for freeing.
#[repr(C)]
struct Slot<T> {
    data: MaybeUninit<T>,
    live: bool,
}

/// A typed, stable-address object pool.
///
/// Objects are allocated inside fixed-size pages (`Box<[Slot<T>]>`); pages
/// are never reallocated or moved, so a `GcPtr<T>` stays valid for the life
/// of the pool (or until the object is explicitly freed). Freed slots go on
/// a free list and are reused by later allocations, so long-running churn
/// does not accumulate holes; unlike one `Box` per object, page allocation
/// keeps objects of the same type densely packed.
///
/// This is the placement layer for the future garbage collector: a sweep
/// walks the pages linearly and returns dead slots to the free list.
pub struct Pool<T> {
    pages: Vec<Box<[Slot<T>]>>,
    free: Vec<NonNull<Slot<T>>>,
    live: usize,
}

impl<T> Pool<T> {
    pub fn new() -> Pool<T> {
        Pool {
            pages: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    fn add_page(&mut self) {
        let mut page: Vec<Slot<T>> = Vec::with_capacity(POOL_PAGE);
        for _ in 0..POOL_PAGE {
            page.push(Slot {
                data: MaybeUninit::uninit(),
                live: false,
            });
        }
        let mut page = page.into_boxed_slice();
        for s in page.iter_mut().rev() {
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

    /// Iterate all live objects (linear page walk, used by future GC sweeps).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.pages
            .iter()
            .flat_map(|p| p.iter())
            .filter(|s| s.live)
            .map(|s| unsafe { s.data.assume_init_ref() })
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
            for s in page.iter_mut() {
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
/// objects live in stable pool pages, nothing is freed until a collector
/// exists, and the VM is single-threaded. All `unsafe` is confined here.
pub struct GcPtr<T>(NonNull<T>);

impl<T> GcPtr<T> {
    pub(crate) fn new(p: NonNull<T>) -> GcPtr<T> {
        debug_assert!(
            (p.as_ptr() as u64) < (1u64 << 47),
            "pointer exceeds the 47-bit LuaValue payload"
        );
        GcPtr(p)
    }

    pub fn from_ref(r: &T) -> GcPtr<T> {
        GcPtr::new(NonNull::from(r))
    }

    /// Reconstruct from a `LuaValue` payload. Returns `None` for a zero
    /// payload (e.g. the template-table marker).
    pub fn from_addr(addr: u64) -> Option<GcPtr<T>> {
        NonNull::new(addr as *mut T).map(GcPtr)
    }

    pub fn addr(self) -> u64 {
        self.0.as_ptr() as u64
    }

    pub fn as_ref<'a>(self) -> &'a T {
        unsafe { &*self.0.as_ptr() }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_mut<'a>(self) -> &'a mut T {
        unsafe { &mut *self.0.as_ptr() }
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
