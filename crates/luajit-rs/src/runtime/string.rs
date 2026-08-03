use crate::runtime::gc::{GcObjectKind, GcPtr, Pool};

pub type StrId = u32;

// -- String hashing (FNV-1a 64) -------------------------------------------
//
// Every interned string is hashed; concat-heavy workloads hash the whole
// accumulated string each iteration (measured: ~70% of intern time for
// `s = s .. x` loops). FNV-1a is a pure stream hash, so a concatenation
// `s .. x` can continue from the stored state of `s` and hash only the
// appended bytes — O(1) amortized per iteration instead of O(len(s)).
// The hash value is fully deterministic (no random seed), so an
// incrementally computed hash always matches a full re-hash of the same
// bytes.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One FNV-1a step.
#[inline(always)]
fn fnv_step(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(FNV_PRIME)
}

/// 8-step unrolled FNV-1a (the multiplicative chain is inherently
/// serial; unrolling removes the loop overhead per byte).
#[inline(always)]
fn fnv_8(h: u64, s: &[u8]) -> u64 {
    debug_assert!(s.len() >= 8);
    let h = fnv_step(h, s[0]);
    let h = fnv_step(h, s[1]);
    let h = fnv_step(h, s[2]);
    let h = fnv_step(h, s[3]);
    let h = fnv_step(h, s[4]);
    let h = fnv_step(h, s[5]);
    let h = fnv_step(h, s[6]);
    fnv_step(h, s[7])
}

/// Full FNV-1a 64 state for `s` (8-byte unrolled main loop).
pub fn fnv1a_state(s: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i + 8 <= s.len() {
        h = fnv_8(h, &s[i..i + 8]);
        i += 8;
    }
    while i < s.len() {
        h = fnv_step(h, s[i]);
        i += 1;
    }
    h
}

/// Continue an FNV-1a 64 stream with `s` (used for incremental concat).
#[inline]
pub fn fnv1a_cont(state: u64, s: &[u8]) -> u64 {
    let mut h = state;
    let mut i = 0;
    while i + 8 <= s.len() {
        h = fnv_8(h, &s[i..i + 8]);
        i += 8;
    }
    while i < s.len() {
        h = fnv_step(h, s[i]);
        i += 1;
    }
    h
}

/// Fold a 64-bit FNV state into the 32-bit interned hash.
#[inline]
pub fn fnv1a_fold(state: u64) -> u32 {
    (state as u32) ^ ((state >> 32) as u32)
}

const INLINE_CAP: usize = 40;

enum Repr {
    Inline { len: u8, buf: [u8; INLINE_CAP] },
    Heap(Box<[u8]>),
}

pub struct LuaString {
    sid: StrId,
    hash: u32,
    repr: Repr,
}

impl LuaString {
    fn new(bytes: &[u8], sid: StrId, hash: u32) -> LuaString {
        let repr = if bytes.len() <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..bytes.len()].copy_from_slice(bytes);
            Repr::Inline {
                len: bytes.len() as u8,
                buf,
            }
        } else {
            Repr::Heap(bytes.into())
        };
        LuaString { sid, hash, repr }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.repr {
            Repr::Inline { len, buf } => &buf[..*len as usize],
            Repr::Heap(b) => b,
        }
    }

    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Inline { len, .. } => *len as usize,
            Repr::Heap(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn sid(&self) -> StrId {
        self.sid
    }
    pub fn hash(&self) -> u32 {
        self.hash
    }
    pub fn gc_size(&self) -> usize {
        std::mem::size_of::<LuaString>()
            + match &self.repr {
                Repr::Inline { .. } => 0,
                Repr::Heap(b) => b.len(),
            }
    }
}

// -- Open-addressed hash table for string interning -----------------------

#[derive(Copy, Clone)]
enum Slot {
    Empty,
    Tombstone,
    Occupied(crate::gc::GcPtr<LuaString>),
}

pub struct Interner {
    slots: Vec<Slot>,
    nuse: usize,
    ndead: usize,
    by_id: Vec<Option<crate::gc::GcPtr<LuaString>>>,
    free_ids: Vec<StrId>,
    pool: crate::gc::Pool<LuaString>,
    bytes: usize,
    /// Pre-interned "" — always available, never collected.
    empty_sid: StrId,
}

impl Interner {
    pub fn new() -> Interner {
        let mut i = Interner {
            slots: vec![Slot::Empty; 512],
            nuse: 0,
            ndead: 0,
            by_id: Vec::new(),
            free_ids: Vec::new(),
            pool: Pool::new(GcObjectKind::String),
            bytes: 0,
            empty_sid: 0,
        };
        i.empty_sid = i.intern(b"");
        i
    }
}

impl Default for Interner {
    fn default() -> Interner {
        Interner::new()
    }
}

impl Interner {
    const MAX_LOAD_NUM: usize = 7;
    const MAX_LOAD_DEN: usize = 10;

    #[inline]
    fn should_grow(&self) -> bool {
        (self.nuse + self.ndead) * Self::MAX_LOAD_DEN >= self.slots.len() * Self::MAX_LOAD_NUM
    }

    pub fn intern(&mut self, s: &[u8]) -> StrId {
        // Compute hash once — consistent between lookup and insert.
        let state = fnv1a_state(s);
        let hash = fnv1a_fold(state);
        self.intern_with_hash(s, hash)
    }

    /// Intern with a precomputed (possibly incrementally continued) FNV
    /// hash. The caller must guarantee `hash` equals `fnv1a_fold(fnv1a_state(s))`.
    pub fn intern_with_hash(&mut self, s: &[u8], hash: u32) -> StrId {
        // Single probe: find existing entry or insertion slot.
        let mask = self.slots.len() - 1;
        let mut idx = (hash as usize) & mask;
        let mut first_dead: Option<usize> = None;
        loop {
            match self.slots[idx] {
                Slot::Empty => {
                    let ins = first_dead.unwrap_or(idx);
                    return self.insert_new(s, hash, ins);
                }
                Slot::Tombstone => {
                    if first_dead.is_none() {
                        first_dead = Some(idx);
                    }
                }
                Slot::Occupied(p) => {
                    let ls = p.as_ref();
                    if ls.hash() == hash && ls.as_bytes() == s {
                        // The string may carry a *previous* cycle's white
                        // (e.g. it was interned by an earlier parse before a
                        // GC sweep). A fresh object referencing it must not
                        // have it swept out from under it before the current
                        // cycle's marking reaches it, so mark eagerly.
                        // Marking is conservative: a dead-but-marked string
                        // is simply collected one cycle later.
                        p.set_marked();
                        return ls.sid();
                    }
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let new_size = self.slots.len() * 2;
        let mut new_slots = vec![Slot::Empty; new_size];
        let mask = new_size - 1;
        for slot in self.slots.iter().copied() {
            if let Slot::Occupied(p) = slot {
                let mut idx = (p.as_ref().hash() as usize) & mask;
                loop {
                    if matches!(new_slots[idx], Slot::Empty) {
                        new_slots[idx] = Slot::Occupied(p);
                        break;
                    }
                    idx = (idx + 1) & mask;
                }
            }
        }
        self.slots = new_slots;
        self.ndead = 0;
    }

    fn insert_new(&mut self, s: &[u8], hash: u32, mut slot: usize) -> StrId {
        if self.should_grow() {
            self.grow();
            // Re-probe after grow.
            let mask = self.slots.len() - 1;
            slot = (hash as usize) & mask;
            loop {
                match self.slots[slot] {
                    Slot::Empty | Slot::Tombstone => break,
                    _ => slot = (slot + 1) & mask,
                }
            }
        }
        let sid = match self.free_ids.pop() {
            Some(id) => id,
            None => {
                self.by_id.push(None);
                (self.by_id.len() - 1) as StrId
            }
        };
        let p = self.pool.alloc(LuaString::new(s, sid, hash));
        self.bytes += p.as_ref().gc_size();
        self.by_id[sid as usize] = Some(p);
        if matches!(self.slots[slot], Slot::Tombstone) {
            self.ndead -= 1;
        }
        self.slots[slot] = Slot::Occupied(p);
        self.nuse += 1;
        sid
    }

    pub fn get(&self, id: StrId) -> &[u8] {
        self.lookup(id).as_bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn try_lookup(&self, id: StrId) -> Option<GcPtr<LuaString>> {
        self.by_id.get(id as usize).and_then(|o| *o)
    }

    pub fn lookup(&self, id: StrId) -> &LuaString {
        self.try_lookup(id)
            .map(|p| p.as_ref())
            .unwrap_or_else(|| self.try_lookup(self.empty_sid).unwrap().as_ref())
    }

    pub fn lookup_ptr(&self, id: StrId) -> GcPtr<LuaString> {
        self.try_lookup(id)
            .unwrap_or_else(|| self.try_lookup(self.empty_sid).unwrap())
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    #[cfg(debug_assertions)]
    pub(crate) fn pool_len(&self) -> usize {
        self.pool.object_count()
    }

    #[cfg(not(debug_assertions))]
    pub(crate) fn pool_len(&self) -> usize {
        self.pool.object_count()
    }

    pub fn get_static(&self, id: StrId) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.get(id).as_ptr(), self.get(id).len()) }
    }

    pub(crate) fn sweep(&mut self, current_white: u8) {
        let by_id = &mut self.by_id;
        // Pin any string with empty bytes ("") by marking it before sweep.
        // The sentinel must survive all GC cycles.
        let empty_bytes: &[u8] = b"";
        for &slot in &self.slots {
            if let Slot::Occupied(p) = slot
                && p.as_ref().as_bytes() == empty_bytes
            {
                p.set_marked();
            }
        }
        self.pool.sweep_tricolor(current_white, |s| {
            let hash = s.hash();
            let bytes = s.as_bytes();
            let mask = self.slots.len() - 1;
            let mut idx = (hash as usize) & mask;
            loop {
                match self.slots[idx] {
                    Slot::Occupied(p)
                        if p.as_ref().hash() == hash && p.as_ref().as_bytes() == bytes =>
                    {
                        self.slots[idx] = Slot::Tombstone;
                        self.nuse -= 1;
                        self.ndead += 1;
                        break;
                    }
                    Slot::Occupied(_) | Slot::Tombstone => {
                        idx = (idx + 1) & mask;
                    }
                    Slot::Empty => break,
                }
            }
            by_id[s.sid() as usize] = None;
            // NOTE: we intentionally do NOT recycle the StringId via
            // free_ids. Reusing a dead StringId while stale references
            // (e.g. in a Proto's KGC list) still point to it would
            // cause the new string to silently replace the old one.
            // Dead StringIds are left as permanent holes in `by_id`.
        });
        self.bytes = self.pool.iter().map(|s| s.gc_size()).sum();
    }
    pub(crate) fn update_current_white(&self, cw: u8) {
        self.pool.update_current_white(cw);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_strings_are_inline() {
        let mut strs = Interner::default();
        let sid = strs.intern(b"hello");
        let s = strs.lookup(sid);
        assert!(matches!(s.repr, Repr::Inline { .. }));
        assert_eq!(s.as_bytes(), b"hello");
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn long_strings_are_heap() {
        let mut strs = Interner::default();
        let long = vec![b'x'; INLINE_CAP + 1];
        let sid = strs.intern(&long);
        let s = strs.lookup(sid);
        assert!(matches!(s.repr, Repr::Heap(_)));
        assert_eq!(s.as_bytes(), &long[..]);
    }

    #[test]
    fn interning_dedups() {
        let mut strs = Interner::default();
        let a = strs.intern(b"abc");
        let b = strs.intern(b"abc");
        let c = strs.intern(b"abd");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn string_values_share_the_str_tag() {
        use crate::value::LuaValue;
        let mut strs = Interner::default();
        let short = strs.intern(b"s");
        let long = strs.intern(&[b'y'; 100]);
        let vs = LuaValue::string(strs.lookup_ptr(short));
        let vl = LuaValue::string(strs.lookup_ptr(long));
        assert!(vs.is_string() && vl.is_string());
        assert_eq!(vs.as_string_id(), Some(short));
        assert_eq!(vl.as_string_id(), Some(long));
        assert_ne!(vs, vl);
    }
}
