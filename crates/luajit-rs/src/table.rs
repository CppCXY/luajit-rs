use std::collections::HashMap;

use crate::value::LuaValue;

/// A Lua table split into an array part (dense integer keys `1..=asize`) and a
/// hash part, following LuaJIT's hybrid layout. This is currently used to build
/// the template tables baked into `TDUP` constants, but is meant to grow into
/// the runtime table type.
#[derive(Default)]
pub struct LuaTable {
    /// Array part. Slot `i` holds the value for integer key `i` (`0` unused).
    array: Vec<LuaValue>,
    /// Hash part for all remaining keys.
    hash: HashMap<LuaValue, LuaValue>,
    asize: u32,
    hbits: u32,
}

impl LuaTable {
    pub fn new(asize: u32, hbits: u32) -> LuaTable {
        LuaTable {
            array: vec![LuaValue::NIL; asize as usize],
            hash: HashMap::new(),
            asize,
            hbits,
        }
    }

    pub fn asize(&self) -> u32 {
        self.asize
    }

    pub fn hbits(&self) -> u32 {
        self.hbits
    }

    /// Return the array index for a key that is a non-negative integer.
    fn array_key(k: LuaValue) -> Option<u32> {
        let n = k.as_number()?;
        if n >= 0.0 && n == n.trunc() && n < u32::MAX as f64 {
            Some(n as u32)
        } else {
            None
        }
    }

    pub fn set(&mut self, k: LuaValue, v: LuaValue) {
        if let Some(i) = LuaTable::array_key(k) {
            if i < self.asize {
                self.array[i as usize] = v;
                return;
            }
        }
        self.hash.insert(k, v);
    }

    /// Grow the array part to cover integer keys up to `nasize`, migrating any
    /// now-in-range keys out of the hash part (mirrors `lj_tab_reasize`).
    pub fn reasize(&mut self, nasize: u32) {
        let asize = nasize + 1;
        if asize <= self.asize {
            return;
        }
        self.array.resize(asize as usize, LuaValue::NIL);
        self.asize = asize;
        let migrated: Vec<LuaValue> = self
            .hash
            .keys()
            .copied()
            .filter(|k| LuaTable::array_key(*k).is_some_and(|i| i < asize))
            .collect();
        for k in migrated {
            let v = self.hash.remove(&k).unwrap();
            let i = LuaTable::array_key(k).unwrap();
            self.array[i as usize] = v;
        }
    }
}
