//! C-style Lua API — free functions with `&mut LuaState` as the first
//! argument, mirroring the Lua 5.1 / LuaJIT C API naming conventions.
//!
//! ```rust,no_run
//! use luajit_rs::*;
//!
//! let mut lua = Lua::new();
//! lual_openlibs(lua.main());
//! let l = lua.main();
//! lual_loadstring(l, b"return 1 + 2").unwrap();
//! lua_pcall(l, 0, 1, 0).unwrap();
//! assert!((lua_tonumber(l, -1) - 3.0).abs() < 0.001);
//! ```
use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, CFunction, GcFunc};
use crate::runtime::gc::{barrier_back, barrier_fwd, full_gc};
use crate::runtime::userdata::GcUserData;
use crate::state::{self, LuaState, StateRef};
use crate::stdlib::coroutine::{Outcome, do_resume};
use crate::stdlib::open_libs as open_stdlib;
use crate::table::LuaTable;
use crate::util::strfmt;
use crate::value::{LJ_TTAB, LJ_TUDATA};

pub use crate::value::LuaValue;

// ── Auxiliary library ─────────────────────────────────────────────────

/// Open all standard Lua libraries into the given state.
pub fn lual_openlibs(l: &mut LuaState) {
    open_stdlib(l);
}

// ── Error ──────────────────────────────────────────────────────────────

/// Extract the error object from a state as a human-readable string.
pub fn lua_error_message(l: &LuaState) -> String {
    let ev = l.errval;
    if let Some(sid) = ev.as_string_id() {
        String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
    } else if let Some(n) = ev.as_number() {
        strfmt::g14(n)
    } else {
        format!("{:?}", ev)
    }
}

/// Raise a Lua error. The error object must be on top of the stack;
/// it is popped and stored in `l.errval`.
pub fn lua_error(l: &mut LuaState) -> LuaResult<()> {
    if l.top > l.base {
        l.top -= 1;
        l.errval = l.stack[l.top];
        l.stack[l.top] = LuaValue::NIL;
    }
    Err(LuaError::Runtime)
}

/// Convenience: push a formatted error message then call `lua_error`.
/// Equivalent to `lua_pushstring(l, msg); lua_error(l)`.
pub fn lual_error(l: &mut LuaState, msg: impl AsRef<[u8]>) -> LuaResult<()> {
    lua_pushstring(l, msg.as_ref());
    lua_error(l)
}

// ── Load & execute ─────────────────────────────────────────────────────

pub fn lual_loadstring(l: &mut LuaState, src: &[u8]) -> LuaResult<()> {
    match state::load(l, src.to_vec(), "=(load)") {
        Ok(f) => {
            lua_pushraw(l, f);
            Ok(())
        }
        Err(msg) => {
            lua_pushstring(l, msg.as_bytes());
            Err(LuaError::Runtime)
        }
    }
}

pub fn lual_loadfile(l: &mut LuaState, path: &str) -> LuaResult<()> {
    let src = match std::fs::read(path) {
        Ok(s) => s,
        Err(e) => {
            lua_pushstring(l, format!("cannot open {path}: {e}").as_bytes());
            return Err(LuaError::Runtime);
        }
    };
    match lual_loadstring(l, &src) {
        Ok(()) => Ok(()),
        Err(e) => Err(e),
    }
}

pub fn lua_pcall(l: &mut LuaState, nargs: i32, nresults: i32, _errfunc: i32) -> LuaResult<()> {
    let func = lua_index(l, -(nargs + 1));
    let mut args = Vec::with_capacity(nargs as usize);
    for i in 0..nargs as usize {
        args.push(lua_index(l, -(nargs - i as i32)));
    }
    lua_settop(l, lua_gettop(l) as i32 - nargs - 1);
    crate::vm::call(l, func, &args).map(|results| {
        let want = if nresults < 0 {
            results.len()
        } else {
            results.len().min(nresults as usize)
        };
        for v in results.into_iter().take(want) {
            lua_pushraw(l, v);
        }
    })
}

pub fn lua_call(l: &mut LuaState, nargs: i32, nresults: i32) -> LuaResult<()> {
    let func = lua_index(l, -(nargs + 1));
    let mut args = Vec::with_capacity(nargs as usize);
    for i in 0..nargs as usize {
        args.push(lua_index(l, -(nargs - i as i32)));
    }
    lua_settop(l, lua_gettop(l) as i32 - nargs - 1);
    crate::vm::call(l, func, &args).map(|results| {
        for v in results.into_iter().take(nresults as usize) {
            lua_pushraw(l, v);
        }
    })
}

// ── Stack push ────────────────────────────────────────────────────────

pub fn lua_pushnil(l: &mut LuaState) {
    lua_pushraw(l, LuaValue::NIL);
}

pub fn lua_pushnumber(l: &mut LuaState, n: f64) {
    lua_pushraw(l, LuaValue::number(n));
}

pub fn lua_pushinteger(l: &mut LuaState, n: i64) {
    lua_pushraw(l, LuaValue::number(n as f64));
}

pub fn lua_pushstring(l: &mut LuaState, s: &[u8]) {
    let sid = l.global().heap.intern(s);
    let v = l.global().heap.str_value(sid);
    lua_pushraw(l, v);
}

pub fn lua_pushboolean(l: &mut LuaState, b: bool) {
    lua_pushraw(l, LuaValue::boolean(b));
}

/// Push a copy of the element at the given valid index onto the stack.
pub fn lua_pushvalue(l: &mut LuaState, idx: i32) {
    let v = lua_index(l, idx);
    lua_pushraw(l, v);
}

pub fn lua_pushcfunction(l: &mut LuaState, f: CFunction) {
    let g = l.global();
    let env = g.globals;
    let fref = g.heap.alloc_func(GcFunc::C(CClosure {
        f,
        env,
        upvals: vec![],
    }));
    lua_pushraw(l, LuaValue::func(fref));
}

pub fn lua_pushthread(l: &mut LuaState) {
    let co = l.self_ref();
    lua_pushraw(l, LuaValue::thread(co));
}

/// Push an arbitrary `LuaValue` without any conversion.
/// This is the low-level building block used by all other `lua_push*` functions.
pub fn lua_pushraw(l: &mut LuaState, v: LuaValue) {
    l.stack_ensure(l.top + 1);
    l.stack[l.top] = v;
    l.top += 1;
}

// ── Stack query ───────────────────────────────────────────────────────

pub fn lua_isnil(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_nil()
}

pub fn lua_isboolean(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_bool()
}

pub fn lua_isnumber(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_number()
}

pub fn lua_isstring(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_string()
}

pub fn lua_istable(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_table()
}

pub fn lua_isfunction(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_func()
}

pub fn lua_isthread(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_thread()
}

pub fn lua_isuserdata(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_userdata()
}

pub fn lua_iscdata(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_cdata()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuaType {
    None = -1,
    Nil = 0,
    Boolean = 1,
    LightUserdata = 2,
    Number = 3,
    String = 4,
    Table = 5,
    Function = 6,
    Userdata = 7,
    Thread = 8,
    CData = 10,
}

pub fn lua_type(l: &LuaState, idx: i32) -> LuaType {
    let v = lua_index(l, idx);
    if v.is_nil() {
        LuaType::Nil
    } else if v.is_bool() {
        LuaType::Boolean
    } else if v.is_number() {
        LuaType::Number
    } else if v.is_string() {
        LuaType::String
    } else if v.is_table() {
        LuaType::Table
    } else if v.is_func() {
        LuaType::Function
    } else if v.is_userdata() {
        LuaType::Userdata
    } else if v.is_thread() {
        LuaType::Thread
    } else if v.is_cdata() {
        LuaType::CData
    } else {
        LuaType::None
    }
}

pub fn lua_typename(_l: &LuaState, tp: LuaType) -> &'static str {
    match tp {
        LuaType::None => "no value",
        LuaType::Nil => "nil",
        LuaType::Boolean => "boolean",
        LuaType::LightUserdata => "lightuserdata",
        LuaType::Number => "number",
        LuaType::String => "string",
        LuaType::Table => "table",
        LuaType::Function => "function",
        LuaType::Userdata => "userdata",
        LuaType::Thread => "thread",
        LuaType::CData => "cdata",
    }
}

pub fn lua_tonumber(l: &LuaState, idx: i32) -> f64 {
    lua_index(l, idx).as_number().unwrap_or(0.0)
}

pub fn lua_tointeger(l: &LuaState, idx: i32) -> i64 {
    lua_index(l, idx).as_number().map(|n| n as i64).unwrap_or(0)
}

pub fn lua_toboolean(l: &LuaState, idx: i32) -> bool {
    let v = lua_index(l, idx);
    if v.is_bool() {
        v.is_true()
    } else {
        !v.is_nil()
    }
}

pub fn lua_tolstring(l: &LuaState, idx: i32) -> &[u8] {
    let v = lua_index(l, idx);
    if let Some(s) = v.as_string() {
        s.as_ref().as_bytes()
    } else {
        &[]
    }
}

pub fn lua_tothread(l: &LuaState, idx: i32) -> Option<StateRef> {
    lua_index(l, idx).as_thread()
}

pub fn lua_objlen(l: &LuaState, idx: i32) -> usize {
    let v = lua_index(l, idx);
    if let Some(s) = v.as_string() {
        s.as_ref().as_bytes().len()
    } else if let Some(t) = v.as_table() {
        t.as_ref().len() as usize
    } else {
        0
    }
}

// ── Stack management ──────────────────────────────────────────────────

pub fn lua_gettop(l: &LuaState) -> usize {
    l.top.saturating_sub(l.base)
}

pub fn lua_settop(l: &mut LuaState, idx: i32) {
    let new_top = if idx > 0 {
        l.base + idx as usize
    } else if idx == 0 {
        l.base
    } else {
        l.top.wrapping_add_signed(idx as isize + 1)
    };
    if new_top < l.base {
        return;
    }
    if new_top > l.top {
        l.stack_ensure(new_top);
        for i in l.top..new_top {
            l.stack[i] = LuaValue::NIL;
        }
    }
    l.top = new_top;
}

pub fn lua_pop(l: &mut LuaState, n: i32) {
    lua_settop(l, -(n + 1));
}

pub fn lua_remove(l: &mut LuaState, idx: i32) {
    let abs = lua_absindex(l, idx);
    if abs < l.base || abs >= l.top {
        return;
    }
    for i in abs..l.top - 1 {
        l.stack[i] = l.stack[i + 1];
    }
    l.top -= 1;
    l.stack[l.top] = LuaValue::NIL;
}

pub fn lua_replace(l: &mut LuaState, idx: i32) {
    let abs = lua_absindex(l, idx);
    if l.top <= l.base || abs < l.base || abs >= l.top {
        return;
    }
    l.top -= 1;
    l.stack[abs] = l.stack[l.top];
    l.stack[l.top] = LuaValue::NIL;
}

pub fn lua_insert(l: &mut LuaState, idx: i32) {
    let abs = lua_absindex(l, idx);
    if l.top <= l.base || abs < l.base || abs >= l.top {
        return;
    }
    let v = l.stack[l.top - 1];
    for i in (abs + 1..l.top).rev() {
        l.stack[i] = l.stack[i - 1];
    }
    l.stack[abs] = v;
}

pub fn lua_absindex(l: &LuaState, idx: i32) -> usize {
    if idx > 0 {
        l.base + (idx as usize) - 1
    } else {
        l.top.wrapping_add_signed(idx as isize)
    }
}

// ── Globals ───────────────────────────────────────────────────────────

pub fn lua_getglobal(l: &mut LuaState, name: &str) {
    let sid = l.global().heap.intern(name.as_bytes());
    let key = l.global().heap.str_value(sid);
    let globals = l.global().globals;
    let v = globals.as_ref().get(key);
    lua_pushraw(l, v);
}

pub fn lua_setglobal(l: &mut LuaState, name: &str) {
    let v = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let sid = l.global().heap.intern(name.as_bytes());
    let key = l.global().heap.str_value(sid);
    let globals = l.global().globals;
    globals.as_mut().set(key, v);
    gc_table_write(l, globals, v);
}

pub fn lua_register(l: &mut LuaState, name: &str, f: CFunction) {
    lua_pushcfunction(l, f);
    lua_setglobal(l, name);
}

// ── Tables ────────────────────────────────────────────────────────────

pub fn lua_newtable(l: &mut LuaState) {
    let t = l.global().heap.alloc_table(LuaTable::new(0, 0));
    lua_pushraw(l, LuaValue::table(t));
}

pub fn lua_getfield(l: &mut LuaState, idx: i32, k: &str) {
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let sid = l.global().heap.intern(k.as_bytes());
        let lk = l.global().heap.str_value(sid);
        let v = t.as_ref().get(lk);
        lua_pushraw(l, v);
    } else {
        lua_pushraw(l, LuaValue::NIL);
    }
}

pub fn lua_setfield(l: &mut LuaState, idx: i32, k: &str) {
    let val = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let sid = l.global().heap.intern(k.as_bytes());
        let lk = l.global().heap.str_value(sid);
        t.as_mut().set(lk, val);
        gc_table_write(l, t, val);
    }
}

pub fn lua_gettable(l: &mut LuaState, idx: i32) {
    let key = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let v = t.as_ref().get(key);
        lua_pushraw(l, v);
    } else {
        lua_pushraw(l, LuaValue::NIL);
    }
}

pub fn lua_settable(l: &mut LuaState, idx: i32) {
    let val = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let key = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        t.as_mut().set(key, val);
        gc_table_write(l, t, val);
    }
}

pub fn lua_rawgeti(l: &mut LuaState, idx: i32, n: i32) {
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let v = t.as_ref().get_int(n);
        lua_pushraw(l, v);
    } else {
        lua_pushraw(l, LuaValue::NIL);
    }
}

pub fn lua_rawseti(l: &mut LuaState, idx: i32, n: i32) {
    let val = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        t.as_mut().set_int(n, val);
        gc_table_write(l, t, val);
    }
}

// ── Raw get/set (bypass metamethods) ──────────────────────────────────

pub fn lua_rawget(l: &mut LuaState, idx: i32) {
    let key = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let v = t.as_ref().get(key);
        lua_pushraw(l, v);
    } else {
        lua_pushraw(l, LuaValue::NIL);
    }
}

pub fn lua_rawset(l: &mut LuaState, idx: i32) {
    let val = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let key = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        t.as_mut().set(key, val);
        gc_table_write(l, t, val);
    }
}

// ── Userdata ──────────────────────────────────────────────────────────

pub fn lua_newuserdata(l: &mut LuaState, size: usize) -> *mut u8 {
    let data = vec![0u8; size].into_boxed_slice();
    let ud = GcUserData::new(data);
    let ptr = l.global().heap.alloc_userdata(ud);
    let data_ptr = ptr
        .as_mut()
        .inner
        .downcast_mut::<Box<[u8]>>()
        .unwrap()
        .as_mut_ptr();
    lua_pushraw(l, LuaValue::userdata(ptr));
    data_ptr
}

pub fn lua_touserdata(l: &LuaState, idx: i32) -> *mut u8 {
    let v = lua_index(l, idx);
    if let Some(ud) = v.as_userdata() {
        ud.as_ref()
            .inner
            .downcast_ref::<Box<[u8]>>()
            .map(|d| d.as_ptr() as *mut u8)
            .unwrap_or(std::ptr::null_mut())
    } else {
        std::ptr::null_mut()
    }
}

// ── Metatables ────────────────────────────────────────────────────────

pub fn lual_newmetatable(l: &mut LuaState, tname: &str) -> i32 {
    let (registry, mt_val) = {
        let g = l.global();
        let sid = g.heap.intern(tname.as_bytes());
        let key = g.heap.str_value(sid);
        let registry = g.registry;
        if !registry.as_ref().get(key).is_nil() {
            return 0;
        }
        let mt = g.heap.alloc_table(LuaTable::new(0, 2));
        let mt_val = LuaValue::table(mt);
        registry.as_mut().set(key, mt_val);
        (registry, mt_val)
    };
    barrier_back(&mut l.global().heap, registry);
    barrier_fwd(&mut l.global().heap, mt_val);
    1
}

pub fn lual_getmetatable(l: &mut LuaState, tname: &str) {
    let sid = l.global().heap.intern(tname.as_bytes());
    let key = l.global().heap.str_value(sid);
    let registry = l.global().registry;
    let mt = registry.as_ref().get(key);
    lua_pushraw(l, mt);
}

pub fn lua_setmetatable(l: &mut LuaState, idx: i32) {
    let mt_v = if l.top > l.base {
        l.top -= 1;
        l.stack[l.top]
    } else {
        LuaValue::NIL
    };
    let mt = mt_v.as_table();
    let v = lua_index(l, idx);
    match v.itype() {
        LJ_TTAB => {
            if let Some(t) = v.as_table() {
                t.as_mut().metatable = mt;
            }
        }
        LJ_TUDATA => {
            if let Some(ud) = v.as_userdata() {
                ud.as_mut().metatable = mt;
            }
        }
        _ => {}
    }
}

pub fn lua_getmetatable(l: &mut LuaState, idx: i32) -> i32 {
    let v = lua_index(l, idx);
    let mt = match v.itype() {
        LJ_TTAB => v.as_table().and_then(|t| t.as_ref().metatable),
        LJ_TUDATA => v.as_userdata().and_then(|ud| ud.as_ref().metatable),
        _ => None,
    };
    if let Some(mt) = mt {
        lua_pushraw(l, LuaValue::table(mt));
        1
    } else {
        0
    }
}

pub fn lual_checkudata(l: &LuaState, idx: i32, tname: &str) -> *mut u8 {
    let v = lua_index(l, idx);
    let ud = match v.as_userdata() {
        Some(u) => u,
        None => return std::ptr::null_mut(),
    };
    if let Some(mt) = ud.as_ref().metatable {
        let g = l.global();
        let sid = g.heap.intern(tname.as_bytes());
        let key = g.heap.str_value(sid);
        let registry = g.registry;
        let reg_mt = registry.as_ref().get(key);
        if reg_mt.as_table() != Some(mt) {
            return std::ptr::null_mut();
        }
    } else {
        return std::ptr::null_mut();
    }
    ud.as_ref()
        .inner
        .downcast_ref::<Box<[u8]>>()
        .map(|d| d.as_ptr() as *mut u8)
        .unwrap_or(std::ptr::null_mut())
}

// ── Coroutines ────────────────────────────────────────────────────────

/// Create a new thread, push it on the stack, and return its `StateRef`.
pub fn lua_newthread(l: &mut LuaState) -> StateRef {
    let co = crate::state::new_thread(l);
    lua_pushraw(l, LuaValue::thread(co));
    co
}

/// Resume a coroutine `co` at stack index `-(nargs + 1)` with `nargs`
/// arguments. Returns one of: 0 (finished), LUA_YIELD, or an error code.
/// On success, results are on the calling thread's stack.
pub const LUA_YIELD: i32 = 1;

pub fn lua_resume(l: &mut LuaState, nargs: i32) -> LuaResult<i32> {
    let co_v = lua_index(l, -(nargs + 1));
    let co = match co_v.as_thread() {
        Some(c) => c,
        None => return Err(LuaError::Runtime),
    };
    let outcome = do_resume(l, co, l.top - nargs as usize, nargs as usize)?;
    match outcome {
        Outcome::Done(n) => {
            let co_state = co.as_mut();
            for i in 0..n {
                l.stack[l.base + i] = co_state.stack[i];
            }
            l.top = l.base + n;
            Ok(0)
        }
        Outcome::Yielded(slot, n) => {
            let co_state = co.as_mut();
            for i in 0..n {
                l.stack[l.base + i] = co_state.stack[slot + i];
            }
            l.top = l.base + n;
            Ok(LUA_YIELD)
        }
        Outcome::Failed => Err(LuaError::Runtime),
    }
}

/// Yield a coroutine. Returns `nresults` values to the resumer.
pub fn lua_yield(l: &mut LuaState, nresults: i32) -> LuaResult<()> {
    if l.is_main() {
        return Err(l.runtime_error("attempt to yield from outside a coroutine"));
    }
    if !l.is_yieldable() {
        return Err(l.runtime_error("attempt to yield across C-call boundary"));
    }
    if nresults >= 0 {
        lua_settop(l, lua_gettop(l) as i32);
        let base = l.base;
        l.nyield = (l.top - base).min(nresults as usize) as u32;
    }
    Err(LuaError::Yield)
}

pub enum CoroutineStatus {
    Running,
    Suspended,
    Normal,
    Dead,
}

pub fn lua_status(l: &LuaState) -> CoroutineStatus {
    use crate::state::CoStatus;
    match l.status {
        CoStatus::Running => CoroutineStatus::Running,
        CoStatus::Suspended => CoroutineStatus::Suspended,
        CoStatus::Normal => CoroutineStatus::Normal,
        CoStatus::Dead => CoroutineStatus::Dead,
    }
}

pub fn lua_isyieldable(l: &LuaState) -> bool {
    l.is_yieldable()
}

pub fn lua_xmove(from: &mut LuaState, to: &mut LuaState, n: i32) {
    if std::ptr::eq(from, to) {
        return;
    }
    let count = n as usize;
    let from_top = from.top;
    let start = from_top - count;
    for i in 0..count {
        let v = from.stack[start + i];
        to.stack_ensure(to.top + 1);
        to.stack[to.top] = v;
        to.top += 1;
    }
    from.top = start;
    for i in start..from_top {
        from.stack[i] = LuaValue::NIL;
    }
}

// ── Garbage collection ────────────────────────────────────────────────

pub const LUA_GCSTOP: i32 = 0;
pub const LUA_GCRESTART: i32 = 1;
pub const LUA_GCCOLLECT: i32 = 2;
pub const LUA_GCCOUNT: i32 = 3;
pub const LUA_GCSTEP: i32 = 5;

pub fn lua_gc(l: &mut LuaState, what: i32, _data: i32) -> i32 {
    match what {
        LUA_GCCOLLECT => {
            full_gc(l.global());
            0
        }
        LUA_GCCOUNT => (l.global().heap.total / 1024) as i32,
        _ => 0,
    }
}

// ── Miscellaneous ─────────────────────────────────────────────────────

pub fn lua_peek(l: &LuaState, idx: i32) -> LuaValue {
    lua_index(l, idx)
}

pub fn lua_next(l: &mut LuaState, idx: i32) -> i32 {
    let tab_v = lua_index(l, idx);
    if let Some(t) = tab_v.as_table() {
        let key = lua_index(l, -1);
        if let Some((k, v)) = t.as_ref().next(key) {
            lua_pop(l, 1);
            lua_pushraw(l, k);
            lua_pushraw(l, v);
            return 1;
        }
        lua_pop(l, 1);
        lua_pushraw(l, LuaValue::NIL);
        return 0;
    }
    lua_pop(l, 1);
    lua_pushraw(l, LuaValue::NIL);
    0
}

// ── Internal helpers ──────────────────────────────────────────────────

/// Apply both GC barriers after writing `val` into table `t`.
fn gc_table_write(l: &mut LuaState, t: crate::gc::GcPtr<crate::table::LuaTable>, val: LuaValue) {
    let g = l.global();
    barrier_back(&mut g.heap, t);
    barrier_fwd(&mut g.heap, val);
}

fn lua_index(l: &LuaState, idx: i32) -> LuaValue {
    let abs = lua_absindex(l, idx);
    if abs < l.base || abs >= l.top {
        LuaValue::NIL
    } else {
        l.stack[abs]
    }
}
