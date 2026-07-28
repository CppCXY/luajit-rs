use crate::api as a;
use crate::func::CFunction;
use crate::gc::GcPtr;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LibTarget {
    BaseLib,
    Global,
    Preload,
}

pub struct LibBuilder<'a> {
    l: &'a mut LuaState,
    name: &'a [u8],
    target: LibTarget,
    entries: Vec<(&'a [u8], CFunction)>,
    constants: Vec<(&'a [u8], LuaValue)>,
}

impl<'a> LibBuilder<'a> {
    pub fn new(l: &'a mut LuaState, name: &'a [u8], target: LibTarget) -> Self {
        LibBuilder {
            l,
            name,
            target,
            entries: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub fn func(mut self, fname: &'a [u8], f: CFunction) -> Self {
        self.entries.push((fname, f));
        self
    }

    pub fn value(mut self, key: &'a [u8], val: LuaValue) -> Self {
        self.constants.push((key, val));
        self
    }

    pub fn build(self) -> GcPtr<LuaTable> {
        if matches!(self.target, LibTarget::BaseLib) {
            // Registration during init — use direct heap API to avoid
            // stack-index dependency on base/top being 0.
            let g = self.l.global();
            let env = g.globals;
            for &(field, f) in &self.entries {
                let sid = g.heap.intern(field);
                let fref = g
                    .heap
                    .alloc_func(crate::func::GcFunc::C(crate::func::CClosure {
                        f,
                        env,
                        upvals: Vec::new(),
                    }));
                env.as_mut()
                    .set(g.heap.str_value(sid), LuaValue::func(fref));
            }
            return env;
        }

        // Use api/ for table creation and population
        let top_before = a::lua_gettop(self.l);
        a::lua_newtable(self.l);
        // Stack: [ ...newtable ]
        let tidx = a::lua_gettop(self.l) as i32;

        for &(field, f) in &self.entries {
            a::lua_pushcfunction(self.l, f);
            // Stack: [ ...newtable, func ]
            a::lua_setfield(self.l, tidx, std::str::from_utf8(field).unwrap_or(""));
            // Stack: [ ...newtable ]
        }
        for &(key, val) in &self.constants {
            a::lua_pushraw(self.l, val);
            // Stack: [ ...newtable, val ]
            a::lua_setfield(self.l, tidx, std::str::from_utf8(key).unwrap_or(""));
            // Stack: [ ...newtable ]
        }

        let table = self.l.stack[tidx as usize - 1]
            .as_table()
            .expect("no table");

        match self.target {
            LibTarget::Global => {
                a::lua_setglobal(self.l, std::str::from_utf8(self.name).unwrap_or(""));
            }
            LibTarget::Preload => {
                // Keep internal API for complex closure construction
                let g = self.l.global();
                let env = g.globals;
                let pack_sid = g.heap.intern(b"package");
                let pack = g.heap.str_value(pack_sid);
                let pack_tab = env.as_ref().get(pack).as_table().unwrap_or_else(|| {
                    let pt = g.heap.alloc_table(LuaTable::new(0, 2));
                    env.as_mut().set(pack, LuaValue::table(pt));
                    pt
                });
                let pre_sid = g.heap.intern(b"preload");
                let pre = g.heap.str_value(pre_sid);
                let pre_tab = pack_tab.as_ref().get(pre).as_table().unwrap_or_else(|| {
                    let pt = g.heap.alloc_table(LuaTable::new(0, 2));
                    pack_tab.as_mut().set(pre, LuaValue::table(pt));
                    pt
                });
                let name_sid = g.heap.intern(self.name);
                let loader = g
                    .heap
                    .alloc_func(crate::func::GcFunc::C(crate::func::CClosure {
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
                a::lua_pop(self.l, 1);
            }
            LibTarget::BaseLib => unreachable!(),
        }

        // Restore stack to what it was before — table is now set as global,
        // and the stack copy is no longer needed
        a::lua_settop(self.l, top_before as i32);
        table
    }
}

#[macro_export]
macro_rules! lual_reg {
    ($l:expr, $name:expr, $target:expr) => {
        $crate::LibBuilder::new($l, $name, $target)
    };
}
