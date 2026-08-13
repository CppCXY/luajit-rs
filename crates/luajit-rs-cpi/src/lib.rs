//! C-style Lua API (`luajit-rs-cpi`) — a LuaJIT-compatible C ABI over the
//! `luajit-rs` engine, for embedding and for loading C extension modules.
//!
//! M2 scope (minimal core): `luaL_newstate` / `lua_close`, the essential
//! stack operations, `luaL_openlibs`, `luaL_loadstring`, `lua_pcall`,
//! `luaL_dostring`.
//!
//! M3 scope (C functions + errors): machine-code trampolines bridge
//! `lua_CFunction` pointers into the engine, and a small C shim
//! (`c/ljrs_shim.c`) implements longjmp-based `lua_error`/`luaL_error`
//! semantics with the invariant that a longjmp never crosses a live Rust
//! frame (every C function runs inside a C protection frame; see the shim
//! for the layering).
//!
//! # Safety contract
//!
//! `lua_State*` is an opaque pointer to the engine's GC-managed main
//! thread. Like LuaJIT:
//! - a state must not be used from multiple OS threads simultaneously;
//! - a state must not be used after `lua_close`;
//! - passing a `NULL` state or a pointer not created by `luaL_newstate`
//!   is undefined behaviour.
//!
//! `luaL_newstate` can panic on allocation failure (LuaJIT aborts on OOM
//! instead; panic-across-FFI hardening is tracked for a later milestone).

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int};
use std::sync::Mutex;

use luajit_rs::internal::func::{CClosure, CFunction, GcFunc};
use luajit_rs::{Lua, LuaError, LuaState, LuaType, LuaValue};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use luajit_rs::internal::mcode::McodeArea;

// ── Constants ──────────────────────────────────────────────────────────

pub const LUA_VERSION_NUM: c_int = 501;
pub const LUA_VERSION: &[u8] = b"Lua 5.1";

pub const LUA_MULTRET: c_int = -1;

pub const LUA_OK: c_int = 0;
pub const LUA_YIELD: c_int = 1;
pub const LUA_ERRRUN: c_int = 2;
pub const LUA_ERRSYNTAX: c_int = 3;
pub const LUA_ERRMEM: c_int = 4;
pub const LUA_ERRERR: c_int = 5;

/// Type tags for `lua_type` (LuaJIT-compatible values).
pub const LUA_TNONE: c_int = -1;
pub const LUA_TNIL: c_int = 0;
pub const LUA_TBOOLEAN: c_int = 1;
pub const LUA_TLIGHTUSERDATA: c_int = 2;
pub const LUA_TNUMBER: c_int = 3;
pub const LUA_TSTRING: c_int = 4;
pub const LUA_TTABLE: c_int = 5;
pub const LUA_TFUNCTION: c_int = 6;
pub const LUA_TUSERDATA: c_int = 7;
pub const LUA_TTHREAD: c_int = 8;
pub const LUA_TCDATA: c_int = 10;

/// Pseudo-indices for `lua_pushvalue` (5.1: 5.2's registry access goes
/// through `lua_rawgeti`/`lua_pushvalue` in practice).
pub const LUA_REGISTRYINDEX: c_int = -10000;
pub const LUA_ENVIRONINDEX: c_int = -10001;
pub const LUA_GLOBALSINDEX: c_int = -10002;

pub const LUA_REFNIL: c_int = -1;
pub const LUA_NOREF: c_int = -2;

/// One `luaL_Reg` entry (must match the C headers' layout).
#[repr(C)]
pub struct LuaReg {
    pub name: *const c_char,
    pub func: *const std::ffi::c_void,
}

/// The opaque C state type. In C headers this is `typedef struct lua_State
/// lua_State;` — here it is the engine's thread object.
#[allow(non_camel_case_types)]
pub type lua_State = LuaState;

// ── Universe registry ──────────────────────────────────────────────────
//
// `Lua` (the Box<GlobalState> owner) must outlive every `lua_State*`.
// The registry maps each universe's main-thread pointer to its owner;
// `lua_close` drops the entry (freeing the universe). The GC-pooled
// thread object itself never moves, so the exposed pointer is stable.

type UniverseMap = HashMap<usize, Box<Lua>>;

/// The universe registry. `Lua` is deliberately `!Send`/`!Sync` (like a
/// LuaJIT `lua_State`, a universe must only ever be used from one OS
/// thread at a time). The wrapper's `unsafe impl` is sound because the
/// `Mutex` serializes every access to the map, and the contract forbids
/// sharing a *universe* across threads — the registry itself is only
/// touched by `luaL_newstate` / `lua_close`, always under the lock.
struct UniverseRegistry(Mutex<UniverseMap>);
unsafe impl Send for UniverseRegistry {}
unsafe impl Sync for UniverseRegistry {}

static UNIVERSES: std::sync::LazyLock<UniverseRegistry> =
    std::sync::LazyLock::new(|| UniverseRegistry(Mutex::new(HashMap::new())));

unsafe fn state<'a>(l: *mut lua_State) -> &'a mut LuaState {
    // SAFETY: the C API contract — `l` is a live state owned by the
    // registry, and the caller does not alias it concurrently.
    unsafe { &mut *l }
}

// ── C shim bridge ──────────────────────────────────────────────────────
//
// Every `lua_CFunction` from C is wrapped in a machine-code trampoline
// that the engine calls as a regular Rust `CFunction`. The trampoline
// forwards to `ljrs_cfunc_invoke` (C), which installs a protection frame
// so `lua_error`/`luaL_error` can longjmp without crossing Rust frames.
// The invoke status is then converted to `LuaResult` by tail-jumping to
// the real Rust function `status_to_result` — so the result layout is
// produced by rustc itself, never hand-encoded.

unsafe extern "C" {
    fn ljrs_cfunc_invoke(l: *mut std::ffi::c_void, f: *const std::ffi::c_void) -> c_int;
}

fn status_to_result(l: &mut LuaState, status: c_int) -> luajit_rs::LuaResult<c_int> {
    if status < 0 {
        return Err(LuaError::Runtime);
    }
    if status > 0 {
        // The C API pushes results after the arguments; the engine's
        // CFunction ABI expects them to replace the arguments. Slide them
        // down (top = base + nargs + status).
        let n = status as usize;
        for i in 0..n {
            l.stack[l.base + i] = l.stack[l.top - n + i];
        }
        l.top = l.base + n;
    } else {
        l.top = l.base;
    }
    Ok(status)
}

/// Emit the C-function trampoline. Entry ABI: `fn(&mut LuaState)`, i.e.
/// the state pointer arrives in arg1 (RDI on SysV x64, RCX on Win64, x0
/// on AArch64 — matching the engine's `CFunction` call sites).
#[cfg(target_arch = "x86_64")]
fn emit_cfunc_trampoline(fn_addr: usize) -> Vec<u8> {
    let invoke = ljrs_cfunc_invoke as *const () as usize;
    let bridge = status_to_result as *const () as usize;
    let mut b: Vec<u8> = Vec::with_capacity(80);

    #[cfg(windows)]
    {
        // Entered with rsp%16 == 8. sub 0x38 → 0 mod 16 (callable).
        b.extend_from_slice(&[0x48, 0x83, 0xEC, 0x38]);
        // mov [rsp+0x30], rcx — save L above the 0x20 shadow space.
        b.extend_from_slice(&[0x48, 0x89, 0x4C, 0x24, 0x30]);
        // movabs rdx, fn_addr
        b.push(0x48);
        b.push(0xBA);
        b.extend_from_slice(&(fn_addr as u64).to_le_bytes());
        // movabs rax, invoke ; call rax
        b.push(0x48);
        b.push(0xB8);
        b.extend_from_slice(&(invoke as u64).to_le_bytes());
        b.extend_from_slice(&[0xFF, 0xD0]);
        // mov rcx, [rsp+0x30] ; mov rdx, rax  (bridge args: L, status)
        b.extend_from_slice(&[0x48, 0x8B, 0x4C, 0x24, 0x30]);
        b.extend_from_slice(&[0x48, 0x89, 0xC2]);
        // movabs rax, bridge ; add rsp, 0x38 ; jmp rax
        b.push(0x48);
        b.push(0xB8);
        b.extend_from_slice(&(bridge as u64).to_le_bytes());
        b.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38]);
        b.extend_from_slice(&[0xFF, 0xE0]);
    }
    #[cfg(not(windows))]
    {
        // push rdi — save L; entry rsp%16 == 8 → 0 (callable).
        b.push(0x57);
        // movabs rsi, fn_addr
        b.push(0x48);
        b.push(0xBE);
        b.extend_from_slice(&(fn_addr as u64).to_le_bytes());
        // movabs rax, invoke ; call rax
        b.push(0x48);
        b.push(0xB8);
        b.extend_from_slice(&(invoke as u64).to_le_bytes());
        b.extend_from_slice(&[0xFF, 0xD0]);
        // pop rdi ; mov rsi, rax  (bridge args: L, status)
        b.push(0x5F);
        b.extend_from_slice(&[0x48, 0x89, 0xC6]);
        // movabs rax, bridge ; jmp rax
        b.push(0x48);
        b.push(0xB8);
        b.extend_from_slice(&(bridge as u64).to_le_bytes());
        b.extend_from_slice(&[0xFF, 0xE0]);
    }
    b
}

#[cfg(target_arch = "aarch64")]
fn emit_cfunc_trampoline(fn_addr: usize) -> Vec<u8> {
    let invoke = ljrs_cfunc_invoke as *const () as usize;
    let bridge = status_to_result as *const () as usize;
    let mut b: Vec<u8> = Vec::with_capacity(96);
    let mov_imm = |b: &mut Vec<u8>, reg: u32, val: u64| {
        let imm = |i: u32| ((val >> (16 * i)) & 0xFFFF) as u32;
        b.extend_from_slice(&(0xD2800000 | (imm(0) << 5) | reg).to_le_bytes()); // movz
        b.extend_from_slice(&(0xF2A00000 | (imm(1) << 5) | reg).to_le_bytes()); // movk #16
        b.extend_from_slice(&(0xF2C00000 | (imm(2) << 5) | reg).to_le_bytes()); // movk #32
        b.extend_from_slice(&(0xF2E00000 | (imm(3) << 5) | reg).to_le_bytes()); // movk #48
    };
    // str x0, [sp, #-16]!  — save L (sp stays 16-aligned).
    b.extend_from_slice(&0xF81F0FE0u32.to_le_bytes());
    // x1 = fn_addr ; x16 = invoke ; blr x16
    mov_imm(&mut b, 1, fn_addr as u64);
    mov_imm(&mut b, 16, invoke as u64);
    b.extend_from_slice(&0xD63F0200u32.to_le_bytes());
    // ldr x0, [sp], #16 — restore L.
    b.extend_from_slice(&0xF84107E0u32.to_le_bytes());
    // mov w1, w0 (status) ; x16 = bridge ; br x16
    b.extend_from_slice(&0x2A0003E1u32.to_le_bytes());
    mov_imm(&mut b, 16, bridge as u64);
    b.extend_from_slice(&0xD61F0200u32.to_le_bytes());
    b
}

/// Trampoline registry: one executable stub per distinct C function
/// address (never freed — bounded by the number of distinct functions).
struct TrampolineRegistry(Mutex<HashMap<usize, McodeArea>>);
unsafe impl Send for TrampolineRegistry {}
unsafe impl Sync for TrampolineRegistry {}

static TRAMPOLINES: std::sync::LazyLock<TrampolineRegistry> =
    std::sync::LazyLock::new(|| TrampolineRegistry(Mutex::new(HashMap::new())));

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn cfunc_trampoline(fn_addr: usize) -> usize {
    // Get-or-create under the lock: a double-checked miss lets two threads
    // both allocate, and the second insert would drop the first area while
    // its caller is about to execute it (use-after-free).
    let mut map = TRAMPOLINES.0.lock().unwrap();
    if let Some(area) = map.get(&fn_addr) {
        return area.ptr() as usize;
    }
    let bytes = emit_cfunc_trampoline(fn_addr);
    let mut area = McodeArea::alloc(bytes.len().max(256))
        .expect("luajit-rs-cpi: out of memory for C function trampoline");
    area.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
    area.protect_exec();
    let entry = area.ptr() as usize;
    map.insert(fn_addr, area);
    entry
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn cfunc_trampoline(_fn_addr: usize) -> usize {
    panic!("luajit-rs-cpi: C functions are not supported on this architecture");
}

// ── C function registration ────────────────────────────────────────────

/// Push a C function onto the stack. `f` is a `lua_CFunction`
/// (`int (*)(lua_State *)`); errors it raises propagate through the
/// shim's protection frame.
///
/// # Safety
/// `f` must be a valid C function pointer with the `lua_CFunction`
/// signature.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushcfunction(l: *mut lua_State, f: *const std::ffi::c_void) {
    let tramp = cfunc_trampoline(f as usize);
    let cf: CFunction = unsafe { std::mem::transmute(tramp) };
    luajit_rs::lua_pushcfunction(unsafe { state(l) }, cf);
}

/// Push a C closure: pops `n` upvalues from the stack, which become the
/// closure's upvalues (C code reads them via `lua_upvalueindex(1)`...).
///
/// # Safety
/// `f` must be a valid `lua_CFunction`; `n` stack values must exist.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushcclosure(
    l: *mut lua_State,
    f: *const std::ffi::c_void,
    n: c_int,
) {
    let l = unsafe { state(l) };
    let n = n.max(0) as usize;
    let tramp = cfunc_trampoline(f as usize);
    let cf: CFunction = unsafe { std::mem::transmute(tramp) };
    let mut upvals = Vec::with_capacity(n);
    for i in 0..n {
        upvals.push(l.stack[l.top - n + i]);
    }
    l.top -= n;
    let g = l.global();
    let env = g.globals;
    let fref = g.heap.alloc_func(GcFunc::C(CClosure {
        f: cf,
        env,
        upvals,
    }));
    luajit_rs::lua_pushraw(l, LuaValue::func(fref));
}

/// `lua_pushcfunction` + `lua_setglobal`.
///
/// # Safety
/// `f` must be a valid `lua_CFunction`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_register(
    l: *mut lua_State,
    name: *const c_char,
    f: *const std::ffi::c_void,
) {
    unsafe { lua_pushcfunction(l, f) };
    if !name.is_null() {
        let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
        luajit_rs::lua_setglobal(unsafe { state(l) }, &String::from_utf8_lossy(bytes));
    }
}

// ── State creation / destruction ───────────────────────────────────────

/// Install the C-function trampoline factory into the engine so that
/// `package.loadlib` / `require` can load native modules. Idempotent.
pub fn install_factory() {
    luajit_rs::set_cfunc_factory(|p| {
        Some(unsafe { std::mem::transmute(cfunc_trampoline(p as usize)) })
    });
}

/// Create a new Lua universe; returns its main thread. `NULL` on failure
/// (currently always succeeds or aborts on OOM).
#[unsafe(no_mangle)]
pub extern "C" fn luaL_newstate() -> *mut lua_State {
    install_factory();
    let mut lua = Box::new(Lua::new());
    let lp: *mut LuaState = {
        let m = lua.main();
        m as *mut LuaState
    };
    UNIVERSES.0.lock().unwrap().insert(lp as usize, lua);
    lp
}

/// Destroy a universe and free all its resources. The state must not be
/// used afterwards.
#[unsafe(no_mangle)]
pub extern "C" fn lua_close(l: *mut lua_State) {
    if !l.is_null() {
        let _ = UNIVERSES.0.lock().unwrap().remove(&(l as usize));
    }
}

/// Open all standard libraries into the state (base, string, table, math,
/// bit, io, os, package, coroutine, debug, jit, ffi).
#[unsafe(no_mangle)]
pub extern "C" fn luaL_openlibs(l: *mut lua_State) {
    install_factory();
    luajit_rs::lual_openlibs(unsafe { state(l) });
}

// ── Stack management ───────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn lua_gettop(l: *mut lua_State) -> c_int {
    luajit_rs::lua_gettop(unsafe { state(l) }) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_settop(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_settop(unsafe { state(l) }, idx);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_pop(l: *mut lua_State, n: c_int) {
    luajit_rs::lua_pop(unsafe { state(l) }, n);
}

// ── Push operations ────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn lua_pushnil(l: *mut lua_State) {
    luajit_rs::lua_pushnil(unsafe { state(l) });
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_pushnumber(l: *mut lua_State, n: f64) {
    luajit_rs::lua_pushnumber(unsafe { state(l) }, n);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_pushinteger(l: *mut lua_State, n: i64) {
    luajit_rs::lua_pushinteger(unsafe { state(l) }, n);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_pushboolean(l: *mut lua_State, b: c_int) {
    luajit_rs::lua_pushboolean(unsafe { state(l) }, b != 0);
}

/// Push a NUL-terminated string.
#[unsafe(no_mangle)]
pub extern "C" fn lua_pushstring(l: *mut lua_State, s: *const c_char) {
    if s.is_null() {
        luajit_rs::lua_pushnil(unsafe { state(l) });
        return;
    }
    let bytes = unsafe { CStr::from_ptr(s) }.to_bytes();
    luajit_rs::lua_pushstring(unsafe { state(l) }, bytes);
}

/// Push a (possibly binary) string of `len` bytes.
///
/// # Safety
/// `s` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lua_pushlstring(l: *mut lua_State, s: *const c_char, len: usize) {
    let bytes = if s.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(s.cast::<u8>(), len) }
    };
    luajit_rs::lua_pushstring(unsafe { state(l) }, bytes);
}

// ── Query / conversion ─────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn lua_type(l: *mut lua_State, idx: c_int) -> c_int {
    match luajit_rs::lua_type(unsafe { state(l) }, idx) {
        LuaType::Nil => LUA_TNIL,
        LuaType::Boolean => LUA_TBOOLEAN,
        LuaType::Number => LUA_TNUMBER,
        LuaType::String => LUA_TSTRING,
        LuaType::Table => LUA_TTABLE,
        LuaType::Function => LUA_TFUNCTION,
        LuaType::Userdata => LUA_TUSERDATA,
        LuaType::Thread => LUA_TTHREAD,
        LuaType::CData => LUA_TCDATA,
        _ => LUA_TNONE,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_typename(l: *mut lua_State, tp: c_int) -> *const c_char {
    let name: &[u8] = match tp {
        LUA_TNIL => b"nil",
        LUA_TBOOLEAN => b"boolean",
        LUA_TNUMBER => b"number",
        LUA_TSTRING => b"string",
        LUA_TTABLE => b"table",
        LUA_TFUNCTION => b"function",
        LUA_TUSERDATA => b"userdata",
        LUA_TTHREAD => b"thread",
        LUA_TCDATA => b"cdata",
        _ => b"no value",
    };
    let _ = l;
    name.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isnil(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isnil(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isnumber(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isnumber(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isstring(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isstring(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_tonumber(l: *mut lua_State, idx: c_int) -> f64 {
    luajit_rs::lua_tonumber(unsafe { state(l) }, idx)
}

/// Return the string bytes at `idx` (no number→string coercion yet), or
/// `NULL` when the value is not a string. `len` (when non-NULL) receives
/// the byte length. The pointer is valid until the next GC collection.
#[unsafe(no_mangle)]
pub extern "C" fn lua_tolstring(l: *mut lua_State, idx: c_int, len: *mut usize) -> *const c_char {
    let l = unsafe { state(l) };
    if !luajit_rs::lua_isstring(l, idx) {
        if !len.is_null() {
            unsafe { *len = 0 };
        }
        return std::ptr::null();
    }
    let s = luajit_rs::lua_tolstring(l, idx);
    if !len.is_null() {
        unsafe { *len = s.len() };
    }
    s.as_ptr().cast()
}

// ── Load & call ────────────────────────────────────────────────────────

/// Compile the string as a Lua chunk; pushes the closure on success, the
/// error message on failure. Returns `LUA_OK` or `LUA_ERRSYNTAX`.
#[unsafe(no_mangle)]
pub extern "C" fn luaL_loadstring(l: *mut lua_State, s: *const c_char) -> c_int {
    let src = if s.is_null() {
        b"".as_slice()
    } else {
        unsafe { CStr::from_ptr(s) }.to_bytes()
    };
    match luajit_rs::lual_loadstring(unsafe { state(l) }, src) {
        Ok(()) => LUA_OK,
        Err(_) => LUA_ERRSYNTAX,
    }
}

/// Call a value in protected mode: pops the function and `nargs`
/// arguments, pushes `nresults` results (all with `LUA_MULTRET`), or the
/// error object on failure. `errfunc` (a stack index of a message handler)
/// is currently ignored.
#[unsafe(no_mangle)]
pub extern "C" fn lua_pcall(l: *mut lua_State, nargs: c_int, nresults: c_int, errfunc: c_int) -> c_int {
    let l = unsafe { state(l) };
    match luajit_rs::lua_pcall(l, nargs, nresults, errfunc) {
        Ok(()) => LUA_OK,
        Err(LuaError::Yield) => LUA_YIELD,
        Err(_) => {
            // C contract: the error object replaces function + arguments.
            let ev = l.errval;
            luajit_rs::lua_pushraw(l, ev);
            LUA_ERRRUN
        }
    }
}

/// `luaL_loadstring` + `lua_pcall(0, LUA_MULTRET)`. Returns `LUA_OK` or
/// the error code, with the results or the error object on the stack.
#[unsafe(no_mangle)]
pub extern "C" fn luaL_dostring(l: *mut lua_State, s: *const c_char) -> c_int {
    let status = luaL_loadstring(l, s);
    if status != LUA_OK {
        return status;
    }
    lua_pcall(l, 0, LUA_MULTRET, 0)
}

// ── Error helpers (called from the C shim) ─────────────────────────────
//
// These fully return before the shim longjmps — no Rust frame is ever
// left live above the raise.

/// Record a formatted error message as the pending error object.
#[unsafe(no_mangle)]
pub extern "C" fn ljrs_error_set(l: *mut lua_State, msg: *const c_char, len: usize) {
    let l = unsafe { state(l) };
    let bytes = if msg.is_null() {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(msg.cast::<u8>(), len) }
    };
    let sid = l.global().heap.intern(bytes);
    l.errval = l.global().heap.str_value(sid);
}

/// `lua_error`: pop the error object off the stack into the state.
#[unsafe(no_mangle)]
pub extern "C" fn ljrs_error_take(l: *mut lua_State) {
    let _ = luajit_rs::lua_error(unsafe { state(l) });
}

/// `lua_call` body: 0 = ok, non-zero = error pending (shim raises).
#[unsafe(no_mangle)]
pub extern "C" fn ljrs_call_impl(l: *mut lua_State, nargs: c_int, nresults: c_int) -> c_int {
    let l = unsafe { state(l) };
    match luajit_rs::lua_call(l, nargs, nresults) {
        Ok(()) => 0,
        Err(LuaError::Yield) => {
            let sid = l
                .global()
                .heap
                .intern(b"attempt to yield across C-call boundary");
            l.errval = l.global().heap.str_value(sid);
            1
        }
        Err(_) => 1,
    }
}

/// `luaL_checkudata` body: the payload pointer, or NULL on mismatch.
#[unsafe(no_mangle)]
pub extern "C" fn ljrs_checkudata(
    l: *mut lua_State,
    idx: c_int,
    tname: *const c_char,
) -> *mut std::ffi::c_void {
    let name = String::from_utf8_lossy(unsafe { CStr::from_ptr(tname) }.to_bytes()).into_owned();
    luajit_rs::lual_checkudata(unsafe { state(l) }, idx, &name) as *mut std::ffi::c_void
}

// ── Userdata & metatables ──────────────────────────────────────────────

/// Allocate `size` bytes of userdata, push it, and return the payload
/// pointer (stable until the userdata is collected).
#[unsafe(no_mangle)]
pub extern "C" fn lua_newuserdata(l: *mut lua_State, size: usize) -> *mut std::ffi::c_void {
    luajit_rs::lua_newuserdata(unsafe { state(l) }, size).cast()
}

/// The payload pointer of the full userdata at `idx`, or `NULL`.
#[unsafe(no_mangle)]
pub extern "C" fn lua_touserdata(l: *mut lua_State, idx: c_int) -> *mut std::ffi::c_void {
    luajit_rs::lua_touserdata(unsafe { state(l) }, idx).cast()
}

/// Create a registry-named metatable if absent and push it; returns 1 on
/// creation, 0 when it already existed.
#[unsafe(no_mangle)]
pub extern "C" fn luaL_newmetatable(l: *mut lua_State, tname: *const c_char) -> c_int {
    if tname.is_null() {
        return 0;
    }
    let name = String::from_utf8_lossy(unsafe { CStr::from_ptr(tname) }.to_bytes()).into_owned();
    luajit_rs::lual_newmetatable(unsafe { state(l) }, &name)
}

/// Push the registry-named metatable (nil if none).
#[unsafe(no_mangle)]
pub extern "C" fn luaL_getmetatable(l: *mut lua_State, tname: *const c_char) {
    if tname.is_null() {
        luajit_rs::lua_pushnil(unsafe { state(l) });
        return;
    }
    let name = String::from_utf8_lossy(unsafe { CStr::from_ptr(tname) }.to_bytes()).into_owned();
    luajit_rs::lual_getmetatable(unsafe { state(l) }, &name);
}

/// `luaL_getmetatable` + set it on the value at `-2`.
#[unsafe(no_mangle)]
pub extern "C" fn luaL_setmetatable(l: *mut lua_State, tname: *const c_char) {
    luaL_getmetatable(l, tname);
    lua_setmetatable(l, -2);
}

/// Pops a table (or nil) and sets it as the metatable of the value at
/// `idx`.
#[unsafe(no_mangle)]
pub extern "C" fn lua_setmetatable(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_setmetatable(unsafe { state(l) }, idx);
}

/// Push the metatable of the value at `idx`; returns 1, or 0 if none.
#[unsafe(no_mangle)]
pub extern "C" fn lua_getmetatable(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_getmetatable(unsafe { state(l) }, idx)
}

// ── Tables & globals ───────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn lua_createtable(l: *mut lua_State, _narr: c_int, _nrec: c_int) {
    luajit_rs::lua_newtable(unsafe { state(l) });
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_getglobal(l: *mut lua_State, name: *const c_char) {
    if name.is_null() {
        luajit_rs::lua_pushnil(unsafe { state(l) });
        return;
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    luajit_rs::lua_getglobal(unsafe { state(l) }, &String::from_utf8_lossy(bytes));
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_setglobal(l: *mut lua_State, name: *const c_char) {
    if name.is_null() {
        luajit_rs::lua_pop(unsafe { state(l) }, 1);
        return;
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    luajit_rs::lua_setglobal(unsafe { state(l) }, &String::from_utf8_lossy(bytes));
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_getfield(l: *mut lua_State, idx: c_int, k: *const c_char) {
    let bytes = if k.is_null() {
        &[][..]
    } else {
        unsafe { CStr::from_ptr(k) }.to_bytes()
    };
    luajit_rs::lua_getfield(unsafe { state(l) }, idx, &String::from_utf8_lossy(bytes));
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_setfield(l: *mut lua_State, idx: c_int, k: *const c_char) {
    let bytes = if k.is_null() {
        &[][..]
    } else {
        unsafe { CStr::from_ptr(k) }.to_bytes()
    };
    luajit_rs::lua_setfield(unsafe { state(l) }, idx, &String::from_utf8_lossy(bytes));
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_gettable(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_gettable(unsafe { state(l) }, idx);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_settable(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_settable(unsafe { state(l) }, idx);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_rawget(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_rawget(unsafe { state(l) }, idx);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_rawset(l: *mut lua_State, idx: c_int) {
    luajit_rs::lua_rawset(unsafe { state(l) }, idx);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_rawgeti(l: *mut lua_State, idx: c_int, n: c_int) {
    luajit_rs::lua_rawgeti(unsafe { state(l) }, idx, n);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_rawseti(l: *mut lua_State, idx: c_int, n: c_int) {
    luajit_rs::lua_rawseti(unsafe { state(l) }, idx, n);
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_next(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_next(unsafe { state(l) }, idx)
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_absindex(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_absindex(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_objlen(l: *mut lua_State, idx: c_int) -> usize {
    luajit_rs::lua_objlen(unsafe { state(l) }, idx)
}

/// Push the value at `idx` (supports the registry/globals pseudo-indices).
#[unsafe(no_mangle)]
pub extern "C" fn lua_pushvalue(l: *mut lua_State, idx: c_int) {
    let l = unsafe { state(l) };
    if idx == LUA_REGISTRYINDEX {
        let reg = l.global().registry;
        luajit_rs::lua_pushraw(l, LuaValue::table(reg));
    } else if idx == LUA_GLOBALSINDEX {
        let g = l.global().globals;
        luajit_rs::lua_pushraw(l, LuaValue::table(g));
    } else {
        luajit_rs::lua_pushvalue(l, idx);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_tointeger(l: *mut lua_State, idx: c_int) -> i64 {
    luajit_rs::lua_tointeger(unsafe { state(l) }, idx)
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_toboolean(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_toboolean(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isboolean(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isboolean(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_istable(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_istable(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isfunction(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isfunction(unsafe { state(l) }, idx) as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn lua_isuserdata(l: *mut lua_State, idx: c_int) -> c_int {
    luajit_rs::lua_isuserdata(unsafe { state(l) }, idx) as c_int
}

// ── References ─────────────────────────────────────────────────────────

/// Pop the top value and store it in the registry; returns the integer
/// reference (LuaJIT 5.1-style refs).
#[unsafe(no_mangle)]
pub extern "C" fn luaL_ref(l: *mut lua_State, t: c_int) -> c_int {
    let l = unsafe { state(l) };
    if t != LUA_REGISTRYINDEX || l.top <= l.base {
        return LUA_REFNIL;
    }
    let v = l.stack[l.top - 1];
    l.top -= 1;
    let reg = l.global().registry;
    let head = reg.as_ref().get_int(0).as_number().map(|n| n as i32).unwrap_or(0);
    let id = if head != 0 {
        let next = reg
            .as_ref()
            .get_int(head)
            .as_number()
            .map(|n| n as i32)
            .unwrap_or(0);
        reg.as_mut().set_int(0, LuaValue::number(next as f64));
        head
    } else {
        reg.as_ref().len() as i32 + 1
    };
    reg.as_mut().set_int(id, v);
    id
}

/// Release a reference created by `luaL_ref` (5.1: `luaL_unref(L, t, ref)`).
#[unsafe(no_mangle)]
pub extern "C" fn luaL_unref(l: *mut lua_State, t: c_int, r: c_int) {
    if t != LUA_REGISTRYINDEX || r == LUA_REFNIL || r == LUA_NOREF {
        return;
    }
    let l = unsafe { state(l) };
    let reg = l.global().registry;
    let head = reg.as_ref().get_int(0).as_number().map(|n| n as i32).unwrap_or(0);
    reg.as_mut().set_int(r, LuaValue::number(head as f64));
    reg.as_mut().set_int(0, LuaValue::number(r as f64));
}

/// Push the referenced registry value and return its type.
#[unsafe(no_mangle)]
pub extern "C" fn lua_rawget_ref(l: *mut lua_State, r: c_int) -> c_int {
    let l = unsafe { state(l) };
    if r == LUA_REFNIL || r == LUA_NOREF {
        luajit_rs::lua_pushnil(l);
        return LUA_TNIL;
    }
    let reg = l.global().registry;
    let v = reg.as_ref().get_int(r);
    luajit_rs::lua_pushraw(l, v);
    lua_type(l, -1)
}

// ── Library registration ───────────────────────────────────────────────

/// Fill the table at the stack top with the functions in `reg`
/// (NULL-terminated `luaL_Reg` array).
///
/// # Safety
/// `reg` must point to a NULL-terminated `luaL_Reg` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaL_setfuncs(l: *mut lua_State, reg: *const LuaReg) {
    if reg.is_null() {
        return;
    }
    let mut i = 0isize;
    loop {
        let r = unsafe { &*reg.offset(i) };
        if r.name.is_null() {
            break;
        }
        unsafe { lua_pushcfunction(l, r.func) };
        let bytes = unsafe { CStr::from_ptr(r.name) }.to_bytes();
        luajit_rs::lua_setfield(unsafe { state(l) }, -2, &String::from_utf8_lossy(bytes));
        i += 1;
    }
}

/// Create a new table and register `reg` in it (5.2-style `luaL_newlib`),
/// leaving the table on the stack. Returns 1.
///
/// # Safety
/// `reg` must point to a NULL-terminated `luaL_Reg` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaL_newlib(l: *mut lua_State, reg: *const LuaReg) -> c_int {
    luajit_rs::lua_newtable(unsafe { state(l) });
    unsafe { luaL_setfuncs(l, reg) };
    1
}

/// 5.1 `luaL_register`: create a new table, fill it from `reg`, and (when
/// `libname` is non-NULL) set it as a global. Returns 1 with the table on
/// the stack. (The 5.1 setfenv behaviour is not implemented yet.)
///
/// # Safety
/// `reg` must point to a NULL-terminated `luaL_Reg` array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn luaL_register(
    l: *mut lua_State,
    libname: *const c_char,
    reg: *const LuaReg,
) -> c_int {
    unsafe { luaL_newlib(l, reg) };
    if !libname.is_null() {
        // duplicate the table so it survives the global set (and stays on
        // the stack as the module result).
        luajit_rs::lua_pushvalue(unsafe { state(l) }, -1);
        lua_setglobal(l, libname);
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    unsafe extern "C" {
        fn ljrs_test_simple(l: *mut std::ffi::c_void) -> c_int;
        fn ljrs_test_error(l: *mut std::ffi::c_void) -> c_int;
        fn ljrs_test_check(l: *mut std::ffi::c_void) -> c_int;
        fn ljrs_test_ud(l: *mut std::ffi::c_void) -> c_int;
        fn ljrs_test_luaopen(l: *mut std::ffi::c_void) -> c_int;
    }

    fn cstr(s: &str) -> CString {
        CString::new(s).unwrap()
    }

    fn stack_string(l: *mut lua_State, idx: c_int) -> String {
        let mut len = 0usize;
        let p = lua_tolstring(l, idx, &mut len);
        if p.is_null() {
            return String::new();
        }
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) })
            .into_owned()
    }

    #[test]
    fn run_a_script_end_to_end() {
        let l = luaL_newstate();
        assert!(!l.is_null());
        luaL_openlibs(l);

        let status = luaL_dostring(l, cstr("return 6 * 7").as_ptr());
        assert_eq!(status, LUA_OK, "script must run");
        assert_eq!(lua_gettop(l), 1);
        assert_eq!(lua_type(l, -1), LUA_TNUMBER);
        assert!((lua_tonumber(l, -1) - 42.0).abs() < 1e-9);

        lua_close(l);
    }

    #[test]
    fn runtime_error_leaves_message_on_stack() {
        let l = luaL_newstate();
        luaL_openlibs(l);

        let status = luaL_dostring(l, cstr("error('boom')").as_ptr());
        assert_eq!(status, LUA_ERRRUN);
        assert_eq!(lua_type(l, -1), LUA_TSTRING);
        let mut len = 0usize;
        let msg = lua_tolstring(l, -1, &mut len);
        let msg = unsafe { std::slice::from_raw_parts(msg.cast::<u8>(), len) };
        assert!(String::from_utf8_lossy(msg).contains("boom"));

        lua_close(l);
    }

    #[test]
    fn syntax_error_reports_errsyntax() {
        let l = luaL_newstate();
        luaL_openlibs(l);

        let status = luaL_dostring(l, cstr("this is not lua ((").as_ptr());
        assert_eq!(status, LUA_ERRSYNTAX);
        assert_eq!(lua_type(l, -1), LUA_TSTRING);

        lua_close(l);
    }

    #[test]
    fn stack_roundtrip() {
        let l = luaL_newstate();
        luaL_openlibs(l);

        assert_eq!(lua_gettop(l), 0);
        lua_pushnumber(l, 3.5);
        lua_pushstring(l, cstr("hi").as_ptr());
        let raw = *b"ab\0cd";
        unsafe {
            lua_pushlstring(l, raw.as_ptr().cast(), 5);
        }
        assert_eq!(lua_gettop(l), 3);
        assert_eq!(lua_type(l, 1), LUA_TNUMBER);
        assert_eq!(lua_type(l, 2), LUA_TSTRING);
        assert!(lua_isstring(l, 3) == 1);

        let mut len = 0usize;
        let p = lua_tolstring(l, 3, &mut len);
        assert_eq!(len, 5);
        let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) };
        assert_eq!(bytes, b"ab\0cd");

        // non-string index → NULL
        assert!(lua_tolstring(l, 1, &mut len).is_null());
        assert_eq!(len, 0);

        lua_settop(l, 1);
        assert_eq!(lua_gettop(l), 1);
        lua_pop(l, 1);
        assert_eq!(lua_gettop(l), 0);

        lua_close(l);
    }

    #[test]
    fn universes_are_independent() {
        let a = luaL_newstate();
        let b = luaL_newstate();
        luaL_openlibs(a);
        // b has no libraries: the same script must fail there.

        let script = cstr("return string.format('%s', 'x')");
        assert_eq!(luaL_dostring(a, script.as_ptr()), LUA_OK);
        let mut len = 0usize;
        let p = lua_tolstring(a, -1, &mut len);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(p.cast::<u8>(), len) },
            b"x"
        );

        assert_eq!(luaL_dostring(b, script.as_ptr()), LUA_ERRRUN);

        lua_close(a);
        lua_close(b);
    }

    #[test]
    fn c_function_runs_through_trampoline() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        unsafe {
            lua_register(l, cstr("c_simple").as_ptr(), ljrs_test_simple as *const _);
        }
        assert_eq!(luaL_dostring(l, cstr("return c_simple()").as_ptr()), LUA_OK);
        assert_eq!(lua_gettop(l), 2);
        assert!((lua_tonumber(l, 1) - 6.0).abs() < 1e-9);
        assert!((lua_tonumber(l, 2) - 7.0).abs() < 1e-9);
        lua_close(l);
    }

    #[test]
    fn c_function_error_longjmps_and_state_survives() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        unsafe {
            lua_register(l, cstr("c_error").as_ptr(), ljrs_test_error as *const _);
        }
        assert_eq!(luaL_dostring(l, cstr("return c_error()").as_ptr()), LUA_ERRRUN);
        let msg = stack_string(l, -1);
        assert!(msg.contains("boom from C 42"), "msg = {msg}");

        // The state must remain fully usable after the longjmp.
        assert_eq!(luaL_dostring(l, cstr("return 1 + 1").as_ptr()), LUA_OK);
        assert!((lua_tonumber(l, -1) - 2.0).abs() < 1e-9);
        lua_close(l);
    }

    #[test]
    fn c_function_argument_checking() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        unsafe {
            lua_register(l, cstr("c_check").as_ptr(), ljrs_test_check as *const _);
        }
        assert_eq!(luaL_dostring(l, cstr("return c_check(3, 4)").as_ptr()), LUA_OK);
        assert!((lua_tonumber(l, -1) - 12.0).abs() < 1e-9);

        assert_eq!(
            luaL_dostring(l, cstr("return c_check('x', 1)").as_ptr()),
            LUA_ERRRUN
        );
        let msg = stack_string(l, -1);
        assert!(
            msg.contains("bad argument #1") && msg.contains("number expected, got string"),
            "msg = {msg}"
        );
        lua_close(l);
    }

    #[test]
    fn c_function_userdata_and_metatable() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        unsafe {
            lua_register(l, cstr("c_ud").as_ptr(), ljrs_test_ud as *const _);
        }
        assert_eq!(luaL_dostring(l, cstr("return c_ud()").as_ptr()), LUA_OK);
        assert_eq!(lua_type(l, -1), LUA_TUSERDATA);
        let p = lua_touserdata(l, -1);
        assert!(!p.is_null());
        assert_eq!(unsafe { (p.cast::<i32>()).read() }, 1234);

        // registry-named metatable: create, re-query, attach, verify
        // (both calls push the metatable, per LuaJIT semantics)
        let name = cstr("TESTUD");
        assert_eq!(luaL_newmetatable(l, name.as_ptr()), 1);
        assert_eq!(luaL_newmetatable(l, name.as_ptr()), 0);
        lua_pop(l, 1); // drop the duplicate
        // stack: [userdata, metatable]
        lua_setmetatable(l, -2);
        assert_eq!(lua_getmetatable(l, -1), 1);
        lua_pop(l, 1);
        assert_eq!(lua_gettop(l), 1);
        // re-fetch by name
        luaL_getmetatable(l, name.as_ptr());
        assert_eq!(lua_type(l, -1), LUA_TTABLE);
        lua_close(l);
    }

    #[test]
    fn c_closure_with_upvalues() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        // push an upvalue, then create a closure over it
        lua_pushnumber(l, 100.0);
        unsafe {
            lua_pushcclosure(l, ljrs_test_simple as *const _, 1);
        }
        // call it through Lua: the upvalue is ignored by the test fn but
        // the closure must still run.
        lua_setglobal(l, cstr("closure").as_ptr());
        assert_eq!(luaL_dostring(l, cstr("return closure()").as_ptr()), LUA_OK);
        assert_eq!(lua_gettop(l), 2);
        lua_close(l);
    }

    #[test]
    fn refs_roundtrip() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        lua_pushnumber(l, 3.25);
        let r1 = luaL_ref(l, LUA_REGISTRYINDEX);
        assert!(r1 > 0);
        assert_eq!(lua_gettop(l), 0);
        assert_eq!(lua_rawget_ref(l, r1), LUA_TNUMBER);
        assert!((lua_tonumber(l, -1) - 3.25).abs() < 1e-9);
        lua_pop(l, 1);
        luaL_unref(l, LUA_REGISTRYINDEX, r1);
        // 5.1 semantics: the freed slot holds the freelist head (a number,
        // not nil); the reference id must be reused by the next ref.
        lua_pushnumber(l, 9.5);
        let r2 = luaL_ref(l, LUA_REGISTRYINDEX);
        assert_eq!(r2, r1, "freed reference reused");
        assert_eq!(lua_rawget_ref(l, r2), LUA_TNUMBER);
        assert!((lua_tonumber(l, -1) - 9.5).abs() < 1e-9);
        lua_close(l);
    }

    #[test]
    fn registry_pseudo_index_and_tables() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        lua_pushvalue(l, LUA_REGISTRYINDEX);
        assert_eq!(lua_type(l, -1), LUA_TTABLE);
        lua_pop(l, 1);
        lua_pushvalue(l, LUA_GLOBALSINDEX);
        assert_eq!(lua_type(l, -1), LUA_TTABLE);
        lua_pop(l, 1);

        lua_createtable(l, 0, 0);
        lua_pushnumber(l, 42.0);
        lua_setfield(l, -2, cstr("answer").as_ptr());
        lua_getfield(l, -1, cstr("answer").as_ptr());
        assert!((lua_tonumber(l, -1) - 42.0).abs() < 1e-9);
        lua_close(l);
    }

    #[test]
    fn lual_register_registers_functions() {

        let l = luaL_newstate();
        luaL_openlibs(l);
        let reg = [
            LuaReg {
                name: c"test_fn".as_ptr(),
                func: ljrs_test_simple as *const std::ffi::c_void,
            },
            LuaReg {
                name: std::ptr::null(),
                func: std::ptr::null(),
            },
        ];
        unsafe {
            luaL_register(l, cstr("testmod").as_ptr(), reg.as_ptr());
        }
        assert_eq!(
            luaL_dostring(l, cstr("return testmod.test_fn()").as_ptr()),
            LUA_OK
        );
        assert_eq!(lua_gettop(l), 2);
        assert!((lua_tonumber(l, 1) - 6.0).abs() < 1e-9);
        lua_close(l);
    }

    #[test]
    fn luaopen_style_registration_via_c_function() {
        let l = luaL_newstate();
        luaL_openlibs(l);
        unsafe {
            lua_pushcfunction(l, ljrs_test_luaopen as *const _);
        }
        assert_eq!(lua_pcall(l, 0, 0, 0), LUA_OK, "luaopen must run");
        assert_eq!(
            luaL_dostring(l, cstr("return testmod.add(2, 3)").as_ptr()),
            LUA_OK
        );
        assert!((lua_tonumber(l, -1) - 5.0).abs() < 1e-9);
        assert_eq!(
            luaL_dostring(l, cstr("return testmod.greet('world')").as_ptr()),
            LUA_OK
        );
        assert_eq!(stack_string(l, -1), "hello, world");
        lua_close(l);
    }

    /// Windows-only: the Unix cc crate emits `lib<name>.so` (which does
    /// not match the `?.so` cpath pattern), and building a shared library
    /// from a test binary needs the cdylib to be present — both are
    /// Windows-only paths for now. The full native-module chain is
    /// exercised on Windows; Unix CI covers the rest of the C API.
    #[cfg(windows)]
    #[test]
    fn require_native_module_end_to_end() {
        // Build a real C extension as a shared library, then load it
        // through the engine's package.loadlib / require machinery.
        set_cc_env();
        let out_dir = std::env::temp_dir().join("luajit_rs_cpi_testmod");
        let _ = std::fs::create_dir_all(&out_dir);
        {
            // The module imports the C API from the built cdylib: keep the
            // dll loaded in the test process so the module's imports
            // resolve at LoadLibrary time.
            let dll = target_profile_dir().join("luajit_rs_cpi.dll");
            if !dll.exists() {
                eprintln!(
                    "SKIP require_native_module_end_to_end: cdylib not built; \
                     run `cargo build -p luajit-rs-cpi` first"
                );
                return;
            }
            let dll_c = CString::new(dll.to_str().unwrap()).unwrap();
            let h = load_library_raw(dll_c.as_ptr());
            assert!(h != 0, "LoadLibrary of the cdylib failed");

            // cargo test does not build the cdylib, so its import library
            // may be missing/stale. Generate one with lib.exe /DEF listing
            // exactly the symbols the module needs.
            let tool = cc::Build::new()
                .cargo_metadata(false)
                .target(&host_target())
                .try_get_compiler()
                .expect("find cl.exe");
            let cl_dir = tool.path().parent().unwrap();
            let def = out_dir.join("testmod.def");
            std::fs::write(
                &def,
                "LIBRARY luajit_rs_cpi\nEXPORTS\n  luaL_register\n  luaL_checknumber\n  luaL_checkstring\n  lua_pushfstring\n  lua_pushnumber\n",
            )
            .unwrap();
            let imp = out_dir.join("testmod_imp.lib");
            let lib_out = std::process::Command::new(cl_dir.join("lib.exe"))
                .arg(format!("/MACHINE:{}", machine()))
                .arg(format!("/DEF:{}", def.display()))
                .arg(format!("/OUT:{}", imp.display()))
                .current_dir(&out_dir)
                .output()
                .expect("run lib.exe");
            assert!(lib_out.status.success(), "lib.exe /DEF failed");

            let mut cmd = tool.to_command();
            cmd.arg("/nologo")
                .arg("/LD")
                .arg("/Fe:testmod.dll")
                .arg("/Fo:testmod")
                .arg(format!("/I{}", concat!(env!("CARGO_MANIFEST_DIR"), "/include")))
                .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/testmod.c"))
                .arg("/link")
                .arg(imp);
            cmd.current_dir(&out_dir);
            let out = cmd.output().expect("run cl.exe");
            assert!(
                out.status.success(),
                "cl /LD failed: {}",
                String::from_utf8_lossy(&out.stdout)
            );
        }
        #[cfg(not(windows))]
        {
            b.compile("testmod");
        }

        let l = luaL_newstate();
        luaL_openlibs(l);
        let dir = out_dir.to_str().unwrap().replace('\\', "/");
        let setup = format!("package.cpath = '{}/?.{};'", dir, module_ext());
        assert_eq!(luaL_dostring(l, cstr(&setup).as_ptr()), LUA_OK);
        let status = luaL_dostring(l, cstr("local m = require('testmod') assert(m.add(2, 3) == 5) assert(m.sayhi('x') == 'hi x') return m.add(20, 22)").as_ptr());
        assert_eq!(
            status,
            LUA_OK,
            "require must load the native module: {}",
            stack_string(l, -1)
        );
        assert!((lua_tonumber(l, -1) - 42.0).abs() < 1e-9);
        lua_close(l);
    }

    fn target_profile_dir() -> std::path::PathBuf {
        std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[cfg(windows)]
    fn module_ext() -> &'static str {
        #[cfg(windows)]
        {
            "dll"
        }
        #[cfg(target_os = "macos")]
        {
            "dylib"
        }
        #[cfg(not(any(windows, target_os = "macos")))]
        {
            "so"
        }
    }

    #[cfg(windows)]
    fn host_target() -> String {
        let os = if cfg!(windows) {
            "pc-windows-msvc"
        } else if cfg!(target_os = "macos") {
            "apple-darwin"
        } else {
            "unknown-linux-gnu"
        };
        format!("{}-{}", std::env::consts::ARCH, os)
    }

    /// `cc` (with `cargo_metadata(false)`) reads its configuration from
    /// the process environment; provide what cargo normally sets for
    /// build scripts. Only used by the module-compiling test.
    #[cfg(windows)]
    fn set_cc_env() {
        let t = host_target();
        let set = |k: &str, v: &str| {
            unsafe { std::env::set_var(k, v) };
        };
        set("TARGET", &t);
        set("HOST", &t);
        set("OPT_LEVEL", "0");
        set("PROFILE", "debug");
        set("DEBUG", "true");
        set("NUM_JOBS", "1");
    }

    #[cfg(windows)]
    fn load_library_raw(name: *const c_char) -> isize {
        unsafe extern "system" {
            fn LoadLibraryA(name: *const u8) -> isize;
        }
        unsafe { LoadLibraryA(name as *const u8) }
    }

    #[cfg(windows)]
    fn machine() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            "X64"
        }
        #[cfg(target_arch = "aarch64")]
        {
            "ARM64"
        }
    }
}
