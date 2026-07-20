use ahash::RandomState;

pub type StrId = u32;

const INLINE_CAP: usize = 22;

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
    hasher: RandomState,
    bytes: usize,
}

impl Default for Interner {
    fn default() -> Interner {
        Interner {
            slots: vec![Slot::Empty; 512],
            nuse: 0,
            ndead: 0,
            by_id: Vec::new(),
            free_ids: Vec::new(),
            pool: crate::gc::Pool::new(),
            hasher: RandomState::with_seeds(
                0x243f_6a88_85a3_08d3,
                0x1319_8a2e_0370_7344,
                0xa409_3822_299f_31d0,
                0x082e_fa98_ec4e_6c89,
            ),
            bytes: 0,
        }
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
        let hash = if s.len() <= 7 {
            let mut packed = [0u8; 8];
            packed[..s.len()].copy_from_slice(s);
            let key = u64::from_le_bytes(packed) | ((s.len() as u64) << 56);
            (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32
        } else {
            self.hasher.hash_one(s) as u32
        };
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

    pub fn try_lookup(&self, id: StrId) -> Option<crate::gc::GcPtr<LuaString>> {
        self.by_id.get(id as usize).and_then(|o| *o)
    }

    pub fn lookup(&self, id: StrId) -> &LuaString {
        self.by_id[id as usize].expect("dead string id").as_ref()
    }

    pub fn lookup_ptr(&self, id: StrId) -> crate::gc::GcPtr<LuaString> {
        self.by_id[id as usize].expect("dead string id")
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn get_static(&self, id: StrId) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.get(id).as_ptr(), self.get(id).len()) }
    }

    pub(crate) fn sweep(&mut self) {
        let by_id = &mut self.by_id;
        let free_ids = &mut self.free_ids;
        self.pool.sweep(|s| {
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
            free_ids.push(s.sid());
        });
        self.bytes = self.pool.iter().map(|s| s.gc_size()).sum();
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
