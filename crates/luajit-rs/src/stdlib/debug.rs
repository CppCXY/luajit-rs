use crate::err::LuaResult;
use crate::state::LuaState;
use crate::stdlib::{arg, err_bad_arg, push};
use crate::value::LuaValue;

fn set_basemt_for(
    l: &mut LuaState,
    o: &LuaValue,
    mt: Option<crate::gc::GcPtr<crate::table::LuaTable>>,
) {
    let g = l.global();
    g.set_basemt(o.itype(), mt);
    // Boolean: false and true share the same base metatable in LuaJIT.
    if o.itype() == crate::value::LJ_TFALSE {
        g.set_basemt(crate::value::LJ_TTRUE, mt);
    } else if o.itype() == crate::value::LJ_TTRUE {
        g.set_basemt(crate::value::LJ_TFALSE, mt);
    }
}

fn lib_setmetatable(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0);
    let mt = arg(l, 1);
    if mt.is_nil() {
        if let Some(t) = o.as_table() {
            t.as_mut().metatable = None;
        } else {
            set_basemt_for(l, &o, None);
        }
    } else if let Some(mt_tab) = mt.as_table() {
        if let Some(t) = o.as_table() {
            t.as_mut().metatable = Some(mt_tab);
        } else {
            set_basemt_for(l, &o, Some(mt_tab));
        }
    } else {
        return Err(err_bad_arg(l, 2, "debug.setmetatable", "nil or table", ""));
    }
    push(l, o);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    crate::stdlib::reg::LibBuilder::new(l, b"debug", crate::stdlib::reg::LibTarget::Global)
        .func(b"setmetatable", lib_setmetatable)
        .build();
}
