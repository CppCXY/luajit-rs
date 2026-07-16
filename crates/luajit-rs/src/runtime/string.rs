use std::collections::HashMap;

pub type StrId = u32;

/// Threshold for the inline small-string optimization. Strings up to this
/// length are stored inline; longer ones are heap-allocated. This is purely
/// an internal storage detail (like a smol-str): it is not observable at the
/// value level. LuaJIT itself has a single interned string type (`GCstr`,
/// itype `LJ_TSTR`) with no short/long split.
const INLINE_CAP: usize = 22;

enum Repr {
    Inline { len: u8, buf: [u8; INLINE_CAP] },
    Heap(Box<[u8]>),
}

/// A Lua string object, corresponding to LuaJIT's `GCstr`.
///
/// Carries the interned string id (`sid`, used for table hashing just like
/// LuaJIT's `hashstr`) and the cached content hash. Storage uses a
/// small-string optimization internally, which callers never observe.
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

    /// Approximate heap footprint in bytes, for GC accounting.
    pub fn gc_size(&self) -> usize {
        std::mem::size_of::<LuaString>()
            + match &self.repr {
                Repr::Inline { .. } => 0,
                Repr::Heap(b) => b.len(),
            }
    }
}

/// Keyed sparse ARX string hash, ported from LuaJIT's `hash_sparse`
/// (constants from Bob Jenkins' lookup3). Constant time.
pub fn hash_sparse(seed: u64, s: &[u8]) -> u32 {
    let len = s.len() as u32;
    if len == 0 {
        return seed as u32;
    }
    let getu32 = |i: usize| -> u32 { u32::from_le_bytes(s[i..i + 4].try_into().unwrap()) };
    let mut h = len ^ (seed as u32);
    let mut a;
    let mut b;
    if len >= 4 {
        a = getu32(0);
        h ^= getu32(s.len() - 4);
        b = getu32((len >> 1) as usize - 2);
        h ^= b;
        h = h.wrapping_sub(b.rotate_left(14));
        b = b.wrapping_add(getu32((len >> 2) as usize - 1));
    } else {
        a = s[0] as u32;
        h ^= s[s.len() - 1] as u32;
        b = s[(len >> 1) as usize] as u32;
        h ^= b;
        h = h.wrapping_sub(b.rotate_left(14));
    }
    a ^= h;
    a = a.wrapping_sub(h.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    h ^= b;
    h = h.wrapping_sub(b.rotate_left(16));
    h
}

/// The string intern table, corresponding to LuaJIT's global string table
/// (`lj_str_new`). Equal byte content always maps to the same `StrId`, so
/// string equality reduces to id equality. Dead ids are recycled by the GC
/// sweep; a live string's id never changes.
#[derive(Default)]
pub struct Interner {
    map: HashMap<Box<[u8]>, StrId>,
    by_id: Vec<Option<crate::gc::GcPtr<LuaString>>>,
    free_ids: Vec<StrId>,
    pool: crate::gc::Pool<LuaString>,
    seed: u64,
    bytes: usize,
}

impl Interner {
    pub fn intern(&mut self, s: &[u8]) -> StrId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let sid = match self.free_ids.pop() {
            Some(id) => id,
            None => {
                self.by_id.push(None);
                (self.by_id.len() - 1) as StrId
            }
        };
        let hash = hash_sparse(self.seed, s);
        let p = self.pool.alloc(LuaString::new(s, sid, hash));
        self.bytes += p.as_ref().gc_size();
        self.by_id[sid as usize] = Some(p);
        self.map.insert(s.into(), sid);
        sid
    }

    pub fn get(&self, id: StrId) -> &[u8] {
        self.lookup(id).as_bytes()
    }

    pub fn lookup(&self, id: StrId) -> &LuaString {
        self.by_id[id as usize].expect("dead string id").as_ref()
    }

    /// A stable pointer to the interned string object, for storing in a
    /// `LuaValue`.
    pub fn lookup_ptr(&self, id: StrId) -> crate::gc::GcPtr<LuaString> {
        self.by_id[id as usize].expect("dead string id")
    }

    /// Approximate bytes held by interned strings (GC accounting).
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Get string content with a `static` lifetime — pool pages never move,
    /// and an interned string stays alive as long as it is reachable.
    /// This lets C functions read string args without cloning, even while
    /// interning results on the same heap.
    pub fn get_static(&self, id: StrId) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.get(id).as_ptr(), self.get(id).len()) }
    }

    /// GC string sweep (`gc_sweepstr`): free unmarked strings, drop their
    /// intern-map entries and recycle their ids.
    pub(crate) fn sweep(&mut self) {
        let map = &mut self.map;
        let by_id = &mut self.by_id;
        let free_ids = &mut self.free_ids;
        self.pool.sweep(|s| {
            map.remove(s.as_bytes());
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
        // Inline vs heap storage is invisible at the value level: both are
        // plain LJ_TSTR values, distinguished only by their sid payload.
        assert!(vs.is_string() && vl.is_string());
        assert_eq!(vs.as_string_id(), Some(short));
        assert_eq!(vl.as_string_id(), Some(long));
        assert_ne!(vs, vl);
    }

    #[test]
    fn hash_sparse_matches_len_behavior() {
        let h1 = hash_sparse(0, b"abc");
        let h2 = hash_sparse(0, b"abd");
        assert_ne!(h1, h2);
        let h3 = hash_sparse(0, b"a longer string over four bytes");
        let h4 = hash_sparse(0, b"a longer string over four bytez");
        assert_ne!(h3, h4);
        assert_eq!(hash_sparse(7, b""), 7);
    }
}
