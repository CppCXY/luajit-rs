//! Library registration builder.
//!
//! Mirrors LuaJIT's `luaL_Reg`-based library registration with explicit
//! placement (`LibTarget::Global` or `LibTarget::Preload`).  A builder
//! macro `lual_reg!` provides the fluid ergonomics:
//!
//! ```ignore
//! lual_reg!(l, b"string", LibTarget::Global)
//!     .func(b"byte", str_byte)
//!     .func(b"char", str_char)
//!     .build();
//! ```

use crate::func::{CClosure, CFunction, GcFunc};
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

/// Where the library table should be exposed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LibTarget {
    /// Register as a global variable (LuaJIT's default for `luaL_openlibs`).
    Global,
    /// Only insert into `package.preload[name]`, leaving the caller to
    /// `require` it.  Mirrors LuaJIT's `luaL_setfuncs` usage for modules
    /// that should not pollute the global namespace.
    Preload,
}

/// Builder for a named library table.  Created by [`lual_reg!`].
pub struct LibBuilder<'a> {
    l: &'a mut LuaState,
    name: &'a [u8],
    target: LibTarget,
    entries: Vec<(&'a [u8], CFunction)>,
    env: Option<crate::gc::GcPtr<LuaTable>>,
}

impl<'a> LibBuilder<'a> {
    pub fn new(l: &'a mut LuaState, name: &'a [u8], target: LibTarget) -> Self {
        LibBuilder {
            l,
            name,
            target,
            entries: Vec::new(),
            env: None,
        }
    }

    /// Override the environment table used for every closure in the library
    /// (defaults to `_G`).
    pub fn env(mut self, t: crate::gc::GcPtr<LuaTable>) -> Self {
        self.env = Some(t);
        self
    }

    /// Register one C function in the library table.
    pub fn func(mut self, fname: &'a [u8], f: CFunction) -> Self {
        self.entries.push((fname, f));
        self
    }

    /// Build the library table and expose it according to `target`.
    pub fn build(self) -> crate::gc::GcPtr<LuaTable> {
        let env = self.env.unwrap_or(self.l.global().globals);
        let t = self.l.heap().alloc_table(LuaTable::new(
            0,
            (self.entries.len() as u32)
                .next_power_of_two()
                .trailing_zeros() as u32,
        ));
        for &(field, f) in &self.entries {
            let sid = self.l.heap().intern(field);
            let fref = self.l.heap().alloc_func(GcFunc::C(CClosure {
                f,
                env,
                upvals: Vec::new(),
            }));
            t.as_mut()
                .set(self.l.heap().str_value(sid), LuaValue::func(fref));
        }
        match self.target {
            LibTarget::Global => {
                let name_sid = self.l.heap().intern(self.name);
                self.l
                    .global()
                    .globals
                    .as_mut()
                    .set(self.l.heap().str_value(name_sid), LuaValue::table(t));
            }
            LibTarget::Preload => {
                let g = self.l.global();
                let pack_sid = self.l.heap().intern(b"package");
                let pack = g.heap.str_value(pack_sid);
                let pack_tab = match g.globals.as_ref().get(pack).as_table() {
                    Some(pt) => pt,
                    None => {
                        let pt = g.heap.alloc_table(LuaTable::new(0, 2));
                        g.globals.as_mut().set(pack, LuaValue::table(pt));
                        pt
                    }
                };
                let pre_sid = self.l.heap().intern(b"preload");
                let pre = g.heap.str_value(pre_sid);
                let pre_tab = match pack_tab.as_ref().get(pre).as_table() {
                    Some(pt) => pt,
                    None => {
                        let pt = g.heap.alloc_table(LuaTable::new(0, 2));
                        pack_tab.as_mut().set(pre, LuaValue::table(pt));
                        pt
                    }
                };
                let name_sid = g.heap.intern(self.name);
                // Preload entry is simply a function that returns the table.
                let loader = g.heap.alloc_func(GcFunc::C(CClosure {
                    f: |l: &mut LuaState| {
                        let tab = match l.stack[l.base - 1].as_table() {
                            Some(t) => t,
                            None => return Ok(0),
                        };
                        l.stack[l.base] = LuaValue::table(tab);
                        Ok(1)
                    },
                    env,
                    upvals: Vec::new(),
                }));
                pre_tab
                    .as_mut()
                    .set(g.heap.str_value(name_sid), LuaValue::func(loader));
            }
        }
        t
    }
}

/// Convenience macro for the builder pattern:
///
/// `lual_reg!(l, b"string", LibTarget::Global).func(b"len", str_len).build();`
#[macro_export]
macro_rules! lual_reg {
    ($l:expr, $name:expr, $target:expr) => {
        $crate::stdlib::reg::LibBuilder::new($l, $name, $target)
    };
}
