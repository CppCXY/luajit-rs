use crate::gc::GcPtr;
use crate::value::LuaValue;

/// Max. array part size (`LJ_MAX_ASIZE`) and max. array key bits.
const LJ_MAX_ABITS: u32 = 28;
const LJ_MAX_ASIZE: u32 = (1 << (LJ_MAX_ABITS - 1)) + 1;
const LJ_MAX_HBITS: u32 = 26;

/// Null hash-chain index (replaces LuaJIT's `NULL` MRef `next`).
const NIL_NODE: u32 = u32::MAX;

/// A hash node, mirroring LuaJIT's `Node`. The `next` chain pointer is an
/// index into `LuaTable::node` instead of a raw pointer (`MRef`), which keeps
/// the exact algorithm but stays memory-safe.
#[derive(Clone, Copy)]
struct Node {
    val: LuaValue,
    key: LuaValue,
    next: u32,
}

impl Node {
    const EMPTY: Node = Node {
        val: LuaValue::NIL,
        key: LuaValue::NIL,
        next: NIL_NODE,
    };
}

/// `hsize2hbits`: number of hash bits needed for `s` slots.
pub fn hsize2hbits(s: u32) -> u32 {
    if s == 0 {
        0
    } else if s == 1 {
        1
    } else {
        1 + (31 - (s - 1).leading_zeros())
    }
}

fn fls(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// A Lua table with a hybrid array + chained-hash layout, ported from
/// LuaJIT's `GCtab`/`lj_tab_*`. The array part covers integer keys
/// `0..asize`; everything else lives in the hash part, which uses Brent's
/// variation to keep chains short.
pub struct LuaTable {
    /// Array part. Slot `i` holds the value for integer key `i`.
    array: Vec<LuaValue>,
    /// Hash part; length is `hmask + 1` (or `1` when empty).
    node: Vec<Node>,
    asize: u32,
    hmask: u32,
    /// Top of the free-node search (index just past the last free node).
    freetop: u32,
    /// Negative metamethod cache (LuaJIT's `GCtab.nomm`): bit `mm` set means
    /// "this table, used as a metatable, has no metamethod `mm`". `!0` for
    /// fresh tables; cleared by any string-key write.
    pub nomm: u8,
    pub metatable: Option<GcPtr<LuaTable>>,
}

impl Default for LuaTable {
    fn default() -> LuaTable {
        LuaTable::new(0, 0)
    }
}

impl LuaTable {
    /// Create a table with `asize` array slots (keys `0..asize`) and a hash
    /// part of `2^hbits` slots (`hbits == 0` means no hash part). Mirrors
    /// `lj_tab_new` (array size is non-inclusive).
    pub fn new(asize: u32, hbits: u32) -> LuaTable {
        assert!(asize <= LJ_MAX_ASIZE, "table overflow");
        assert!(hbits <= LJ_MAX_HBITS, "table overflow");
        let mut t = LuaTable {
            array: vec![LuaValue::NIL; asize as usize],
            node: Vec::new(),
            asize,
            hmask: 0,
            freetop: 0,
            nomm: !0,
            metatable: None,
        };
        if hbits != 0 {
            t.new_hpart(hbits);
        } else {
            t.node = vec![Node::EMPTY]; // shared nil node
        }
        t
    }

    pub fn asize(&self) -> u32 {
        self.asize
    }

    /// Raw pointer to the array part for VM inline access (returns
    /// `*mut` from `&self` — the interpreter holds exclusive access).
    pub fn array_ptr(&self) -> *mut LuaValue {
        self.array.as_ptr() as *mut LuaValue
    }

    pub fn hbits(&self) -> u32 {
        if self.hmask == 0 {
            0
        } else {
            fls(self.hmask) + 1
        }
    }

    fn new_hpart(&mut self, hbits: u32) {
        assert!(hbits <= LJ_MAX_HBITS, "table overflow");
        let hsize = 1u32 << hbits;
        self.node = vec![Node::EMPTY; hsize as usize];
        self.hmask = hsize - 1;
        self.freetop = hsize;
    }

    fn has_hpart(&self) -> bool {
        self.hmask != 0
    }

    /// GC traversal, per `gc_traverse_tab`: all array values, and key+value
    /// of every non-empty hash node. Keys of nil-value nodes are *not*
    /// marked (LuaJIT's dead-key policy: the stale reference stays in the
    /// node but is only ever compared by identity, never dereferenced).
    pub(crate) fn gc_traverse(&self, mut mark: impl FnMut(LuaValue)) {
        if let Some(mt) = self.metatable {
            mark(LuaValue::table(mt));
        }
        for &v in &self.array {
            mark(v);
        }
        if self.has_hpart() {
            for n in &self.node {
                if !n.val.is_nil() {
                    debug_assert!(!n.key.is_nil(), "nil key in non-empty slot");
                    mark(n.key);
                    mark(n.val);
                }
            }
        }
    }

    /// Approximate heap footprint in bytes, for GC accounting.
    pub fn gc_size(&self) -> usize {
        std::mem::size_of::<LuaTable>()
            + self.array.capacity() * std::mem::size_of::<LuaValue>()
            + self.node.capacity() * std::mem::size_of::<Node>()
    }

    /// The main hash-slot index for `key` (`hashkey` + `hashmask`).
    fn hash_slot(&self, key: LuaValue) -> u32 {
        key.hash_key() & self.hmask
    }

    /// Array index for a key that is a non-negative integer within range.
    fn array_key(key: LuaValue) -> Option<u32> {
        let k = key.as_int32_exact()?;
        if k >= 0 && (k as u32) < LJ_MAX_ASIZE {
            Some(k as u32)
        } else {
            None
        }
    }

    // -- Getters ---------------------------------------------------------

    pub fn get(&self, key: LuaValue) -> LuaValue {
        if let Some(i) = LuaTable::array_key(key)
            && i < self.asize
        {
            return self.array[i as usize];
        }
        if key.is_nil() || !self.has_hpart() {
            return LuaValue::NIL;
        }
        let mut n = self.hash_slot(key);
        loop {
            let node = &self.node[n as usize];
            if node.key == key {
                return node.val;
            }
            if node.next == NIL_NODE {
                return LuaValue::NIL;
            }
            n = node.next;
        }
    }

    /// Fast integer-key get (`lj_tab_getint`).
    #[inline]
    pub fn get_int(&self, k: i32) -> LuaValue {
        if k >= 0 && (k as u32) < self.asize {
            return self.array[k as usize];
        }
        self.get(LuaValue::number(k as f64))
    }

    /// String-key get: direct hash-chain walk, no type dispatch. The key
    /// must already be a string (asserted in debug).
    #[inline]
    pub fn get_str(&self, key: LuaValue) -> LuaValue {
        debug_assert!(key.is_string());
        let mut n = self.hash_slot(key);
        loop {
            let node = &self.node[n as usize];
            if node.key == key {
                return node.val;
            }
            let next = node.next;
            if next == NIL_NODE {
                return LuaValue::NIL;
            }
            n = next;
        }
    }

    /// String-key set that reuses an existing node. Falls back to `set` for
    /// insertion (which may rehash).
    #[inline]
    pub fn set_str(&mut self, key: LuaValue, val: LuaValue) {
        debug_assert!(key.is_string());
        self.nomm = 0; // Clear metamethod cache (BC_TSETS does the same).
        let mut n = self.hash_slot(key);
        loop {
            let node = &mut self.node[n as usize];
            if node.key == key {
                node.val = val;
                return;
            }
            let next = node.next;
            if next == NIL_NODE {
                break;
            }
            n = next;
        }
        self.set(key, val);
    }

    /// Integer-key set with sequential insertion (push) fast path.
    /// When key == asize+1, expands the array in powers of two instead
    /// of routing through hash → rehash.
    #[inline]
    pub fn set_int(&mut self, k: i32, v: LuaValue) {
        if k > 0 && (k as u32) < self.asize {
            self.array[k as usize] = v;
            return;
        }
        if k > 0 && (k as u32) == self.asize {
            let new_sz = (self.asize * 2).max(4).min(LJ_MAX_ASIZE);
            self.reasize(new_sz - 1);
            self.array[k as usize] = v;
            return;
        }
        self.set(LuaValue::number(k as f64), v);
    }

    // -- Traversal and length --------------------------------------------

    /// The successor traversal index for `key` (`lj_tab_keyindex`).
    /// `0` starts the traversal, `!0` marks an invalid key.
    fn key_index(&self, key: LuaValue) -> u32 {
        if key.is_nil() {
            return 0;
        }
        if let Some(k) = LuaTable::array_key(key)
            && k < self.asize
        {
            return k + 1;
        }
        if self.has_hpart() {
            let mut n = self.hash_slot(key);
            loop {
                if self.node[n as usize].key == key {
                    return self.asize + n + 1;
                }
                let next = self.node[n as usize].next;
                if next == NIL_NODE {
                    break;
                }
                n = next;
            }
        }
        !0
    }

    /// The next key/value pair after `key` (`lj_tab_next`). `nil` starts the
    /// traversal; `None` ends it.
    pub fn next(&self, key: LuaValue) -> Option<(LuaValue, LuaValue)> {
        let ki = self.key_index(key);
        if ki == !0 {
            return None;
        }
        let mut idx = ki;
        while idx < self.asize {
            let v = self.array[idx as usize];
            if !v.is_nil() {
                return Some((LuaValue::number(idx as f64), v));
            }
            idx += 1;
        }
        idx -= self.asize;
        while self.has_hpart() && idx <= self.hmask {
            let nd = &self.node[idx as usize];
            if !nd.val.is_nil() {
                return Some((nd.key, nd.val));
            }
            idx += 1;
        }
        None
    }

    /// Table length (a border), ported from `lj_tab_len`.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        let mut hi = self.asize;
        hi = hi.saturating_sub(1);
        if hi > 0 && self.array[hi as usize].is_nil() {
            let mut lo = 0u32;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if self.array[mid as usize].is_nil() {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return lo;
        }
        if !self.has_hpart() {
            return hi;
        }
        self.len_hash(hi)
    }

    fn len_hash(&self, mut hi: u32) -> u32 {
        let mut lo = hi;
        hi += 1;
        while !self.get_int(hi as i32).is_nil() {
            lo = hi;
            if hi > (0x7fffffff - 2) / 2 {
                let mut i = 1u32;
                while !self.get_int(i as i32).is_nil() {
                    i += 1;
                }
                return i - 1;
            }
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.get_int(mid as i32).is_nil() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }

    /// Duplicate a template table (`lj_tab_dup`), replacing table-value markers
    /// (used to preserve keys with runtime values) with nil.
    pub fn dup(&self) -> LuaTable {
        let mut t = LuaTable {
            array: self.array.clone(),
            node: self.node.clone(),
            asize: self.asize,
            hmask: self.hmask,
            freetop: self.freetop,
            nomm: 0, // Keys with metamethod names may be present (lj_tab_dup).
            metatable: None,
        };
        for v in t.array.iter_mut() {
            if v.is_table() {
                *v = LuaValue::NIL;
            }
        }
        for nd in t.node.iter_mut() {
            if nd.val.is_table() {
                nd.val = LuaValue::NIL;
            }
        }
        t
    }

    // -- Setters ---------------------------------------------------------

    /// Set `key` to `val`, mirroring `lj_tab_set` + `lj_tab_newkey`. After a
    /// rehash the whole lookup is retried, including the array part (LuaJIT
    /// does this via `lj_tab_newkey` -> `lj_tab_set` recursion).
    pub fn set(&mut self, key: LuaValue, val: LuaValue) {
        debug_assert!(!key.is_nil());
        if key.is_string() {
            self.nomm = 0; // Invalidate negative metamethod cache.
        }
        loop {
            if let Some(i) = LuaTable::array_key(key)
                && i < self.asize
            {
                self.array[i as usize] = val;
                return;
            }
            if self.has_hpart() {
                let mut n = self.hash_slot(key);
                loop {
                    if self.node[n as usize].key == key {
                        self.node[n as usize].val = val;
                        return;
                    }
                    let next = self.node[n as usize].next;
                    if next == NIL_NODE {
                        break;
                    }
                    n = next;
                }
            }
            match self.try_new_key(key) {
                Some(slot) => {
                    self.node[slot as usize].val = val;
                    return;
                }
                None => continue, // Table was rehashed: retry insertion.
            }
        }
    }

    /// Insert a brand-new key and return its node index, or `None` after
    /// rehashing the table (caller must retry). Uses Brent's variation to
    /// keep chains short, ported from `lj_tab_newkey`.
    fn try_new_key(&mut self, key: LuaValue) -> Option<u32> {
        self.nomm = 0; // Keys with metamethod names may be added (lj_tab_newkey).
        if !self.has_hpart() {
            self.rehash(key);
            return None;
        }
        let n = self.hash_slot(key);
        if self.node[n as usize].val.is_nil() {
            // Main position is free: use it directly.
            self.node[n as usize].key = normalize_key(key);
            return Some(n);
        }

        // Find a free node, scanning downward from freetop.
        let mut freenode = self.freetop;
        loop {
            if freenode == 0 {
                self.rehash(key);
                return None;
            }
            freenode -= 1;
            if self.node[freenode as usize].key.is_nil() {
                break;
            }
        }
        self.freetop = freenode;

        let collide = self.hash_slot(self.node[n as usize].key);
        if collide != n {
            // Colliding node is not in its main position: move it away.
            let mut pred = collide;
            while self.node[pred as usize].next != n {
                pred = self.node[pred as usize].next;
            }
            self.node[pred as usize].next = freenode;
            self.node[freenode as usize].val = self.node[n as usize].val;
            self.node[freenode as usize].key = self.node[n as usize].key;
            self.node[freenode as usize].next = self.node[n as usize].next;
            self.node[n as usize].next = NIL_NODE;
            self.node[n as usize].val = LuaValue::NIL;
            self.node[n as usize].key = normalize_key(key);
            Some(n)
        } else {
            // Insert new node into the chain after the main position.
            self.node[freenode as usize].next = self.node[n as usize].next;
            self.node[n as usize].next = freenode;
            self.node[freenode as usize].key = normalize_key(key);
            Some(freenode)
        }
    }

    // -- Resizing --------------------------------------------------------

    /// Resize to the given array/hash sizes, reinserting existing entries.
    /// Ported from `lj_tab_resize`.
    pub fn resize(&mut self, asize: u32, hbits: u32) {
        assert!(asize <= LJ_MAX_ASIZE, "table overflow");
        let oldasize = self.asize;
        let oldnode = std::mem::take(&mut self.node);
        let oldhmask = self.hmask;

        if asize > oldasize {
            self.array.resize(asize as usize, LuaValue::NIL);
            self.asize = asize;
        }

        if hbits != 0 {
            self.new_hpart(hbits);
        } else {
            self.node = vec![Node::EMPTY];
            self.hmask = 0;
            self.freetop = 0;
        }

        if asize < oldasize {
            // Array part shrinks: reinsert dropped array values.
            self.asize = asize;
            let dropped: Vec<(u32, LuaValue)> = (asize..oldasize)
                .filter(|&i| !self.array[i as usize].is_nil())
                .map(|i| (i, self.array[i as usize]))
                .collect();
            self.array.truncate(asize as usize);
            for (i, v) in dropped {
                self.set(LuaValue::number(i as f64), v);
            }
        }

        // Reinsert entries from the old hash part.
        if oldhmask > 0 {
            for nd in &oldnode {
                if !nd.val.is_nil() {
                    self.set(nd.key, nd.val);
                }
            }
        }
    }

    /// Count integer array keys per power-of-two bucket, per `countarray`.
    fn count_array(&self, bins: &mut [u32]) -> u32 {
        if self.asize == 0 {
            return 0;
        }
        let mut na = 0;
        let mut i = 0u32;
        for b in 0..LJ_MAX_ABITS {
            let mut top = 2u32 << b;
            if top >= self.asize {
                top = self.asize - 1;
                if i > top {
                    break;
                }
            }
            let mut n = 0;
            while i <= top {
                if !self.array[i as usize].is_nil() {
                    n += 1;
                }
                i += 1;
            }
            bins[b as usize] += n;
            na += n;
        }
        na
    }

    /// Count an integer key into `bins`, per `countint`. Returns 1 if counted.
    fn count_int(key: LuaValue, bins: &mut [u32]) -> u32 {
        if let Some(k) = key.as_int32_exact()
            && (k as u32) < LJ_MAX_ASIZE
        {
            let idx = if k > 2 { fls((k - 1) as u32) } else { 0 };
            bins[idx as usize] += 1;
            return 1;
        }
        0
    }

    /// Count hash entries, folding integer keys into `bins`, per `counthash`.
    fn count_hash(&self, bins: &mut [u32], narray: &mut u32) -> u32 {
        let mut total = 0;
        let mut na = 0;
        if self.has_hpart() {
            for nd in &self.node {
                if !nd.val.is_nil() {
                    na += LuaTable::count_int(nd.key, bins);
                    total += 1;
                }
            }
        }
        *narray += na;
        total
    }

    /// Choose the best array size from the bucket histogram, per `bestasize`.
    fn best_asize(bins: &[u32], narray: &mut u32) -> u32 {
        let nn = *narray;
        let mut sum = 0;
        let mut na = 0;
        let mut sz = 0;
        let mut b = 0u32;
        while 2 * nn > (1u32 << b) && sum != nn {
            if bins[b as usize] > 0 {
                sum += bins[b as usize];
                if 2 * sum > (1u32 << b) {
                    sz = (2u32 << b) + 1;
                    na = sum;
                }
            }
            b += 1;
        }
        *narray = sz;
        na
    }

    /// Rehash the whole table to accommodate the extra key `ek`, per
    /// `rehashtab`.
    fn rehash(&mut self, ek: LuaValue) {
        let mut bins = [0u32; LJ_MAX_ABITS as usize];
        let mut asize = self.count_array(&mut bins);
        let mut total = 1 + asize;
        total += self.count_hash(&mut bins, &mut asize);
        asize += LuaTable::count_int(ek, &mut bins);
        let na = LuaTable::best_asize(&bins, &mut asize);
        total -= na;
        self.resize(asize, hsize2hbits(total));
    }

    /// Grow the array part to cover integer keys up to `nasize`, per
    /// `lj_tab_reasize`.
    pub fn reasize(&mut self, nasize: u32) {
        let hbits = if self.hmask > 0 {
            fls(self.hmask) + 1
        } else {
            0
        };
        self.resize(nasize + 1, hbits);
    }
}

/// Normalize `-0.0` keys to `+0.0`, matching `lj_tab_newkey`'s `tvismzero`
/// check. `LuaValue::number` already normalizes, so this is defensive.
fn normalize_key(key: LuaValue) -> LuaValue {
    if let Some(n) = key.as_number() {
        LuaValue::number(n)
    } else {
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string::Interner;

    #[test]
    fn int_keys_grow_array() {
        let mut t = LuaTable::new(0, 0);
        for i in 1..=100 {
            t.set(
                LuaValue::number(i as f64),
                LuaValue::number((i * 10) as f64),
            );
        }
        for i in 1..=100 {
            assert_eq!(
                t.get(LuaValue::number(i as f64)),
                LuaValue::number((i * 10) as f64),
                "key {}",
                i
            );
        }
        assert!(t.asize() > 1);
    }

    #[test]
    fn string_keys_chain_and_rehash() {
        let mut strs = Interner::default();
        let mut t = LuaTable::new(0, 1);
        let keys: Vec<_> = (0..200)
            .map(|i| strs.intern(format!("key_{}", i).as_bytes()))
            .collect();
        for (i, &sid) in keys.iter().enumerate() {
            t.set(
                LuaValue::string(strs.lookup_ptr(sid)),
                LuaValue::number(i as f64),
            );
        }
        for (i, &sid) in keys.iter().enumerate() {
            assert_eq!(
                t.get(LuaValue::string(strs.lookup_ptr(sid))),
                LuaValue::number(i as f64)
            );
        }
    }

    #[test]
    fn mixed_keys_and_overwrite() {
        let mut strs = Interner::default();
        let mut t = LuaTable::new(4, 1);
        let k = strs.intern(b"x");
        t.set(LuaValue::string(strs.lookup_ptr(k)), LuaValue::number(1.0));
        t.set(LuaValue::string(strs.lookup_ptr(k)), LuaValue::number(2.0));
        assert_eq!(
            t.get(LuaValue::string(strs.lookup_ptr(k))),
            LuaValue::number(2.0)
        );
        t.set(LuaValue::number(2.5), LuaValue::TRUE);
        assert_eq!(t.get(LuaValue::number(2.5)), LuaValue::TRUE);
        t.set(LuaValue::TRUE, LuaValue::FALSE);
        assert_eq!(t.get(LuaValue::TRUE), LuaValue::FALSE);
        assert_eq!(t.get(LuaValue::number(99.0)), LuaValue::NIL);
    }

    #[test]
    fn reasize_migrates_hash_to_array() {
        let mut t = LuaTable::new(0, 1);
        t.set(LuaValue::number(5.0), LuaValue::number(50.0));
        t.set(LuaValue::number(6.0), LuaValue::number(60.0));
        t.reasize(6);
        assert!(t.asize() >= 7);
        assert_eq!(t.get(LuaValue::number(5.0)), LuaValue::number(50.0));
        assert_eq!(t.get(LuaValue::number(6.0)), LuaValue::number(60.0));
    }

    #[test]
    fn negative_zero_key_normalized() {
        let mut t = LuaTable::new(2, 0);
        t.set(LuaValue::number(-0.0), LuaValue::number(1.0));
        assert_eq!(t.get(LuaValue::number(0.0)), LuaValue::number(1.0));
    }

    #[test]
    fn non_integer_number_keys() {
        let mut t = LuaTable::new(0, 0);
        t.set(LuaValue::number(1.5), LuaValue::number(15.0));
        t.set(LuaValue::number(2.5), LuaValue::number(25.0));
        t.set(LuaValue::number(1e100), LuaValue::number(1.0));
        assert_eq!(t.get(LuaValue::number(1.5)), LuaValue::number(15.0));
        assert_eq!(t.get(LuaValue::number(2.5)), LuaValue::number(25.0));
        assert_eq!(t.get(LuaValue::number(1e100)), LuaValue::number(1.0));
    }
}
