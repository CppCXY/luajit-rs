//! C-style Lua API — free functions with `&mut LuaState` as the first
//! argument, mirroring the Lua 5.1 / LuaJIT C API naming conventions.
//!
//! ```rust,no_run
//! use luajit_rs::api::*;
//!
//! let mut lua = lual_newstate();
//! let l = lua_main(&mut lua);
//! lual_loadstring(l, b"return 1 + 2").unwrap();
//! lua_pcall(l, 0, 1, 0).unwrap();
//! assert!((lua_tonumber(l, -1) - 3.0).abs() < 0.001);
//! ```
use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, CFunction, GcFunc};
use crate::runtime::gc::full_gc;
use crate::runtime::userdata::GcUserData;
use crate::state::{self, Lua, LuaState};
use crate::stdlib::open_libs as open_stdlib;
use crate::table::LuaTable;
use crate::value::{LJ_TTAB, LJ_TUDATA};

// Re-export LuaValue for convenience.
pub use crate::value::LuaValue;

// ── Universe handle ────────────────────────────────────────────────────

/// Extract an error message from a LuaState after a `Runtime` error.
pub fn error_message(l: &LuaState) -> String {
    let ev = l.errval;
    if let Some(sid) = ev.as_string_id() {
        String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
    } else if let Some(n) = ev.as_number() {
        crate::util::strfmt::g14(n)
    } else {
        format!("{:?}", ev)
    }
}

/// Opaque handle owning the entire Lua universe.
/// Destroyed by dropping — no explicit `lua_close` needed.

/// Create a new Lua universe with all standard libraries open.
pub fn lual_newstate() -> Lua {
    Lua::new()
}

/// Open all standard libraries into the given state.
pub fn lual_openlibs(l: &mut LuaState) {
    open_stdlib(l);
}

/// Return a mutable reference to the main thread's LuaState.
/// All `lua_*` API functions operate on this reference.
pub fn lua_main(lua: &mut Lua) -> &mut LuaState {
    lua.main()
}

// ── Load & execute ─────────────────────────────────────────────────────

/// Load a Lua chunk as a function onto the stack.
/// Returns `Ok(())` on success, `Err(LuaError::Runtime)` on syntax error.
pub fn lual_loadstring(l: &mut LuaState, src: &[u8]) -> LuaResult<()> {
    match state::load(l, src.to_vec(), "=(load)") {
        Ok(f) => {
            l.stack_ensure(l.top + 1);
            l.stack[l.top] = f;
            l.top += 1;
            Ok(())
        }
        Err(_) => Err(LuaError::Runtime),
    }
}

/// Load a Lua chunk from a file.
pub fn lual_loadfile(l: &mut LuaState, path: &str) -> LuaResult<()> {
    match std::fs::read(path) {
        Ok(src) => lual_loadstring(l, &src),
        Err(_) => Err(LuaError::Runtime),
    }
}

/// Protected call: pops `nargs` values + function, pushes results.
/// `errfunc` is the stack index of an error handler (0 = none).
/// Returns `Ok(())` on success, `Err` on runtime error.
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
            lua_pushvalue(l, v);
        }
    })
}

/// Unprotected call. On error, the error is propagated via the return
/// value (no longjmp — use `lua_pcall` for protected calls).
pub fn lua_call(l: &mut LuaState, nargs: i32, nresults: i32) -> LuaResult<()> {
    let func = lua_index(l, -(nargs + 1));
    let mut args = Vec::with_capacity(nargs as usize);
    for i in 0..nargs as usize {
        args.push(lua_index(l, -(nargs - i as i32)));
    }
    lua_settop(l, lua_gettop(l) as i32 - nargs - 1);
    crate::vm::call(l, func, &args).map(|results| {
        for v in results.into_iter().take(nresults as usize) {
            lua_pushvalue(l, v);
        }
    })
}

// ── Stack push ────────────────────────────────────────────────────────

pub fn lua_pushnil(l: &mut LuaState) {
    lua_pushvalue(l, LuaValue::NIL);
}

pub fn lua_pushnumber(l: &mut LuaState, n: f64) {
    lua_pushvalue(l, LuaValue::number(n));
}

pub fn lua_pushinteger(l: &mut LuaState, n: i64) {
    lua_pushvalue(l, LuaValue::number(n as f64));
}

pub fn lua_pushstring(l: &mut LuaState, s: &[u8]) {
    let sid = l.global().heap.intern(s);
    let v = l.global().heap.str_value(sid);
    lua_pushvalue(l, v);
}

pub fn lua_pushboolean(l: &mut LuaState, b: bool) {
    lua_pushvalue(l, LuaValue::boolean(b));
}

pub fn lua_pushvalue(l: &mut LuaState, v: LuaValue) {
    l.stack_ensure(l.top + 1);
    l.stack[l.top] = v;
    l.top += 1;
}

pub fn lua_pushcfunction(l: &mut LuaState, f: CFunction) {
    let g = l.global();
    let env = g.globals;
    let fref = g.heap.alloc_func(GcFunc::C(CClosure {
        f,
        env,
        upvals: vec![],
    }));
    lua_pushvalue(l, LuaValue::func(fref));
}

// ── Stack query ───────────────────────────────────────────────────────

pub fn lua_isnil(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_nil()
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

pub fn lua_isuserdata(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_userdata()
}

pub fn lua_iscdata(l: &LuaState, idx: i32) -> bool {
    lua_index(l, idx).is_cdata()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuaType {
    Nil,
    Boolean,
    Number,
    String,
    Table,
    Function,
    Userdata,
    Thread,
    Unknown,
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
    } else {
        LuaType::Unknown
    }
}

pub fn lua_typename(_l: &LuaState, tp: LuaType) -> &'static str {
    match tp {
        LuaType::Nil => "nil",
        LuaType::Boolean => "boolean",
        LuaType::Number => "number",
        LuaType::String => "string",
        LuaType::Table => "table",
        LuaType::Function => "function",
        LuaType::Userdata => "userdata",
        LuaType::Thread => "thread",
        LuaType::Unknown => "no value",
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

/// Returns a byte slice into the interned string data.
/// The pointer is stable (pool-allocated), so the caller may keep it
/// across other Lua calls.
pub fn lua_tolstring(l: &LuaState, idx: i32) -> &[u8] {
    let v = lua_index(l, idx);
    if let Some(s) = v.as_string() {
        s.as_ref().as_bytes()
    } else if let Some(_n) = v.as_number() {
        // In LuaJIT, tonumber gives a temporary string — we can't easily
        // return a stable reference here. Return empty slice for now.
        &[]
    } else {
        &[]
    }
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
    let abs = lua_absindex(l, idx);
    if abs < l.base {
        return;
    }
    if abs > l.top {
        l.stack_ensure(abs);
        l.top = abs;
    } else {
        for i in abs..l.top {
            l.stack[i] = LuaValue::NIL;
        }
        l.top = abs;
    }
}

pub fn lua_pop(l: &mut LuaState, n: i32) {
    lua_settop(l, -(n + 1));
}

/// Push a copy of the value at `idx`.
pub fn lua_pushvalue_at(l: &mut LuaState, idx: i32) {
    let v = lua_index(l, idx);
    lua_pushvalue(l, v);
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

// ── Abs index ─────────────────────────────────────────────────────────

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
    lua_pushvalue(l, v);
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
}

pub fn lua_register(l: &mut LuaState, name: &str, f: CFunction) {
    lua_pushcfunction(l, f);
    lua_setglobal(l, name);
}

// ── Tables ────────────────────────────────────────────────────────────

pub fn lua_newtable(l: &mut LuaState) {
    let t = l.global().heap.alloc_table(LuaTable::new(0, 0));
    lua_pushvalue(l, LuaValue::table(t));
}

pub fn lua_getfield(l: &mut LuaState, idx: i32, k: &str) {
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let sid = l.global().heap.intern(k.as_bytes());
        let lk = l.global().heap.str_value(sid);
        let v = t.as_ref().get(lk);
        lua_pushvalue(l, v);
    } else {
        lua_pushvalue(l, LuaValue::NIL);
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
        lua_pushvalue(l, v);
    } else {
        lua_pushvalue(l, LuaValue::NIL);
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
    }
}

pub fn lua_rawgeti(l: &mut LuaState, idx: i32, n: i32) {
    let tab = lua_index(l, idx);
    if let Some(t) = tab.as_table() {
        let v = t.as_ref().get_int(n);
        lua_pushvalue(l, v);
    } else {
        lua_pushvalue(l, LuaValue::NIL);
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
    }
}

// ── Userdata ──────────────────────────────────────────────────────────

/// Allocate `size` bytes, push a full userdata, return a mutable pointer
/// to the raw memory block.
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
    lua_pushvalue(l, LuaValue::userdata(ptr));
    data_ptr
}

/// Return the raw data pointer of the userdata at `idx`, or null if not
/// a userdata.
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

/// Create a new metatable `tname` in the registry. Returns 1 if newly
/// created, 0 if the name already exists.
pub fn lual_newmetatable(l: &mut LuaState, tname: &str) -> i32 {
    let g = l.global();
    let sid = g.heap.intern(tname.as_bytes());
    let key = g.heap.str_value(sid);
    let registry = g.registry;
    if !registry.as_ref().get(key).is_nil() {
        return 0;
    }
    let mt = g.heap.alloc_table(LuaTable::new(0, 2));
    registry.as_mut().set(key, LuaValue::table(mt));
    1
}

/// Push the metatable associated with `tname` from the registry, or nil.
pub fn lual_getmetatable(l: &mut LuaState, tname: &str) {
    let sid = l.global().heap.intern(tname.as_bytes());
    let key = l.global().heap.str_value(sid);
    let registry = l.global().registry;
    let mt = registry.as_ref().get(key);
    lua_pushvalue(l, mt);
}

/// Set the metatable of the value at `idx` from the value on top of the
/// stack. Pops the metatable.
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

/// Push the metatable of the value at `idx`, or false if none.
pub fn lua_getmetatable(l: &mut LuaState, idx: i32) -> i32 {
    let v = lua_index(l, idx);
    let mt = match v.itype() {
        LJ_TTAB => v.as_table().and_then(|t| t.as_ref().metatable),
        LJ_TUDATA => v.as_userdata().and_then(|ud| ud.as_ref().metatable),
        _ => None,
    };
    if let Some(mt) = mt {
        lua_pushvalue(l, LuaValue::table(mt));
        1
    } else {
        0
    }
}

/// Check the value at `idx` is userdata with metatable `tname`.
/// Returns the raw data pointer on success, null on failure.
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

/// Raise a Lua error with the value on top of the stack.
pub fn lua_error(l: &mut LuaState, err: &str) -> LuaResult<()> {
    Err(l.runtime_error(err))
}

// ── Garbage collection ────────────────────────────────────────────────

pub fn lua_gc(l: &mut LuaState, _what: i32, _data: i32) -> i32 {
    full_gc(l.global());
    0
}

/// Peek at the value at `idx` without modifying the stack.
pub fn lua_peek(l: &LuaState, idx: i32) -> LuaValue {
    lua_index(l, idx)
}

// ── Internal helpers ──────────────────────────────────────────────────

fn lua_index(l: &LuaState, idx: i32) -> LuaValue {
    let abs = lua_absindex(l, idx);
    if abs < l.base || abs >= l.top {
        LuaValue::NIL
    } else {
        l.stack[abs]
    }
}
