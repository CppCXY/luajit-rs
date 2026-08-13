# luajit-rs-cpi — C API and Native Module Support

**Version:** 0.1 (experimental)
**Status:** core implemented and tested on Windows x64; x64 SysV and ARM64
trampolines included (ARM64 verified by compilation only).

---

## 1. Overview

`luajit-rs-cpi` is the C-facing API of the `luajit-rs` engine. It provides
a LuaJIT-compatible (Lua 5.1 surface) C ABI with three uses:

1. **Embedding** — C hosts create Lua states, run scripts, and exchange
   values through the standard `lua_*` / `luaL_*` API.
2. **C extension modules** — native shared libraries written against
   `lua.h`/`lauxlib.h` (e.g. a `luaL_register`-based `luaopen_*`) are
   loaded from Lua via `require` / `package.loadlib`.
3. **A stable ABI boundary** — the C symbols exported by this crate are
   the compatibility surface; the Rust API of `luajit-rs` is not part of
   the contract.

The engine (`luajit-rs`) remains dependency-free with respect to this
crate: the pieces of the C API that the engine needs (dynamic-library
loading for `package.loadlib`, a registration hook for bridging C function
pointers) live in the engine, while the machine-code bridge and the
longjmp machinery live here. A host that only links the engine gets a Lua
implementation without native-module loading; linking `luajit-rs-cpi`
enables the full C API surface.

## 2. Artifacts

```
crates/luajit-rs-cpi/
├── Cargo.toml          # crate-type = ["cdylib", "staticlib", "lib"]
├── build.rs            # compiles c/ljrs_shim.c with the cc crate
├── c/ljrs_shim.c       # C half of the error machinery + C-only entry points
├── include/
│   ├── lua.h           # core API declarations and constants
│   ├── lauxlib.h       # luaL_* auxiliary API
│   ├── lualib.h        # standard-library openers / names
│   └── luaconf.h       # LuaJIT 2.1 / Lua 5.1 feature macros
├── src/lib.rs          # the exported API (Rust), trampolines, registry
└── tests/testmod.c     # a real C module compiled at test time
```

Build products: `luajit_rs_cpi.{so,dylib,dll}` (+ import library on
Windows) and `libluajit_rs_cpi.a`.

### Exported symbols

Core: `luaL_newstate`, `lua_close`, `luaL_openlibs`, `lua_gettop`,
`lua_settop`, `lua_pop`, `lua_pushvalue`, `lua_absindex`,
`lua_pushnil`, `lua_pushnumber`, `lua_pushinteger`, `lua_pushboolean`,
`lua_pushstring`, `lua_pushlstring`, `lua_pushcfunction`,
`lua_pushcclosure`, `lua_pushfstring`, `lua_type`, `lua_typename`,
`lua_isnil`, `lua_isboolean`, `lua_isnumber`, `lua_isstring`,
`lua_istable`, `lua_isfunction`, `lua_isuserdata`, `lua_tonumber`,
`lua_tointeger`, `lua_toboolean`, `lua_tolstring`, `lua_objlen`,
`lua_touserdata`, `lua_newuserdata`, `lua_createtable`,
`lua_gettable`, `lua_settable`, `lua_rawget`, `lua_rawset`,
`lua_rawgeti`, `lua_rawseti`, `lua_getfield`, `lua_setfield`,
`lua_getglobal`, `lua_setglobal`, `lua_register`, `lua_next`,
`lua_getmetatable`, `lua_setmetatable`, `lua_call`, `lua_pcall`,
`lua_error`, `luaL_loadstring`, `luaL_dostring`, `luaL_ref`,
`luaL_unref`, `lua_rawget_ref` (LuaJIT extension).

Auxiliary: `luaL_newmetatable`, `luaL_getmetatable`, `luaL_setmetatable`,
`luaL_checkudata`, `luaL_checknumber`, `luaL_checkinteger`,
`luaL_checklstring`, `luaL_checkstring`, `luaL_error`, `luaL_argerror`,
`luaL_typerror`, `luaL_setfuncs`, `luaL_newlib`, `luaL_register`.

Constants and macros follow LuaJIT: `LUA_VERSION_NUM` is `501`,
`LUAJIT_VERSION` reports a 2.1 series string, `LUA_REGISTRYINDEX` /
`LUA_GLOBALSINDEX` / `LUA_ENVIRONINDEX`, `lua_upvalueindex`,
`LUA_NOREF` / `LUA_REFNIL`, `LUA_MULTRET`, and the usual `luaL_opt*`
macros.

## 3. Building and linking

### Embedding (static)

Compile the host against the headers and link `libluajit_rs_cpi.a`
(or link the rlib with the host's Rust build):

```c
#include "lua.h"
#include "lauxlib.h"

int main(void) {
    lua_State *L = luaL_newstate();
    luaL_openlibs(L);
    if (luaL_dostring(L, "print('hello from C host')") != LUA_OK) {
        fprintf(stderr, "error: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return 1;
    }
    lua_close(L);
    return 0;
}
```

### Embedding (dynamic)

Link against `luajit_rs_cpi.dll` (Windows) or `libluajit_rs_cpi.so` /
`.dylib`; the host keeps the library loaded for the process lifetime.

### C extension modules

A module is an ordinary shared library exporting `luaopen_<name>`:

```c
#include "lua.h"
#include "lauxlib.h"

static int add(lua_State *L) {
    double a = luaL_checknumber(L, 1);
    double b = luaL_checknumber(L, 2);
    lua_pushnumber(L, a + b);
    return 1;
}

static const luaL_Reg mod_funcs[] = {
    {"add", add},
    {NULL, NULL},
};

int luaopen_testmod(lua_State *L) {
    return luaL_register(L, "testmod", mod_funcs);
}
```

Compile it against the import library:

```sh
# Linux / macOS
cc -shared -fPIC -o testmod.so testmod.c -I<include> -L<lib> -lluajit_rs_cpi

# Windows (MSVC)
cl /LD /Fe:testmod.dll testmod.c /I<include> /link luajit_rs_cpi.dll.lib
```

Then, from Lua:

```lua
package.cpath = package.cpath .. ";./?.dll"   -- or ?.so / ?.dylib
local m = require("testmod")
print(m.add(20, 22))  --> 42
```

The `luajit-rs` CLI links this crate, so `require` works out of the box;
on Windows it also keeps `luajit_rs_cpi.dll` (next to the executable)
loaded so module imports resolve from the process.

## 4. Representation of `lua_State *`

`lua_State` is opaque in the headers. Internally:

- The pointer is the engine's GC-managed main thread (`GcPtr<LuaState>`).
  GC objects are individually heap-allocated and never move, so the
  pointer is stable for the lifetime of the universe.
- A universe registry (`LazyLock<Mutex<HashMap<ptr, Box<Lua>>>>`) owns
  each universe's `GlobalState`. `luaL_newstate` inserts the owner;
  `lua_close` removes it, which frees the universe. This is the only
  bookkeeping done behind the pointer.
- Values pushed through the API live on the same stack the GC scans
  (`l.stack[0 .. max(top, frame_top)]`), so no separate rooting protocol
  is required for stack values.

The registry is wrapped in an explicitly `unsafe impl Send + Sync`:
`Lua` is deliberately `!Send`/`!Sync` (a universe must be used from one
OS thread at a time, exactly like a LuaJIT `lua_State`), and soundness
rests on the mutex serializing map access plus that usage contract.

## 5. Error handling: longjmp without crossing Rust frames

LuaJIT's C API implements `lua_error`/`luaL_error` as a `longjmp` to the
nearest protected call. A literal port is unsound here: a longjmp that
skips live Rust frames skips their destructors.

The implementation enforces a single invariant:

> **A longjmp only ever unwinds C frames.**

The structure that guarantees it:

```
[Lua code] -> machine-code trampoline (mcode)
           -> ljrs_cfunc_invoke (C, sets a protection frame: setjmp)
              -> the user's lua_CFunction (C)
                 -> luaL_error (C) -> ljrs_error_set (Rust, returns)
                                    -> longjmp  → caught by the protection frame
```

- **Every** C function invoked from Lua runs inside `ljrs_cfunc_invoke`,
  which installs a thread-local jmp_buf on a jump chain. Errors raised
  inside the C function longjmp to this frame; the invoke wrapper then
  returns a negative status to the VM, and the error propagates upward as
  an ordinary Rust `Result` (so all Rust frames above unwind normally).
- **Error-raising entry points are C functions**: `lua_error`,
  `luaL_error`, `luaL_argerror`, `luaL_typerror`, `lua_call`, and the
  `luaL_check*` family. Each one (a) calls a Rust helper that fully
  returns — formatting the message and storing it as the pending error
  object in the state — and (b) only then longjmps. No Rust frame is ever
  live above the raise.
- **Unprotected errors abort the process** (`ljrs_raise` with an empty
  jump chain), mirroring LuaJIT's panic-on-unprotected-error behavior.
- `lua_pcall` itself needs no C protection frame: its Rust implementation
  returns a status code, and on error the pending error object is pushed
  onto the stack per the C contract (`LUA_ERRRUN`; `LUA_ERRSYNTAX` from
  `luaL_loadstring`; `LUA_YIELD` propagates).

The original two-layer design (a second setjmp at the `lua_pcall` entry
for host-raised errors) turned out to be unreachable: every C function is
already wrapped, so nothing can longjmp out of a `lua_pcall` frame except
through a per-call protection frame. The implementation is single-layer.

## 6. Bridging `lua_CFunction` into the VM

The engine's internal closure type stores `CFunction =
fn(&mut LuaState) -> LuaResult<i32>` (results *replace* the arguments on
the stack). A raw C function pointer has a different signature, calling
convention (results *appended* after the arguments), and error protocol
(longjmp). The bridge:

1. **One machine-code trampoline per distinct function address**
   (cached forever; x86-64 SysV and Win64, plus ARM64). The trampoline is
   entered exactly like a Rust `CFunction` — the state pointer arrives in
   the same argument register. It forwards `(L, fn_addr)` to the C
   `ljrs_cfunc_invoke`, then tail-jumps to a real Rust function
   `status_to_result(&mut LuaState, status) -> LuaResult<i32>`.
   Because the trampoline tail-jumps to a rustc-compiled function, the
   `Result` return layout is produced by the compiler, never hand-coded.
2. **Result slide.** The C API appends results above the arguments
   (`lua_push*` raises `top`); the engine's ABI expects them at the
   frame base. `status_to_result` moves the `status` results down to
   `base` and resets `top`. Negative statuses map to `Err(Runtime)` with
   the pending error object already in the state.
3. **Upvalues** (`lua_pushcclosure`) are ordinary stack values captured
   into the closure at creation time.

Windows note: `cc::Build` compiles the shim; all shim entry points are
marked `__declspec(dllexport)` so the cdylib exports them.

## 7. Threading model

- A `lua_State` (and its universe) must be used from one OS thread at a
  time, matching LuaJIT. No internal locking guards a state.
- Distinct universes are independent and may run on distinct threads;
  `luaL_newstate`/`lua_close` are internally synchronized (registry
  mutex, and `Once`-protected default-library initialization in the
  engine). Trampoline creation is lock-serialized (get-or-create under a
  single mutex — a double-checked variant had a use-after-free race).
- The shim's jump chain is thread-local C storage; `setjmp`/`longjmp`
  never leave the owning thread.

## 8. `require` / `package.loadlib`

The engine provides `package.loadlib(libname, funcname)` with LuaJIT
semantics:

- The library is loaded at the exact path (`dlopen`/`LoadLibraryA`, no
  suffix guessing); the handle is intentionally retained for the process
  lifetime (unloading a module library would invalidate its functions).
- The symbol is resolved and wrapped through the factory registered by
  this crate (`luajit_rs::set_cfunc_factory`, installed by
  `install_factory()`, which `luaL_newstate`, `luaL_openlibs` and the CLI
  all call).
- Failures return `(nil, message, "open" | "absent" | "init")`; the
  `path:func` shorthand is supported with an empty funcname argument.
- The `require` chain is the standard 5.1 one: preload → Lua path → C
  path (`luaopen_<name>` via `loader_C`) → C root loader. Without the
  factory (engine-only host), the C loaders report that dynamic loading
  is unavailable instead of silently failing.

## 9. Compatibility notes and known deviations

Fidelity is deliberately LuaJIT-shaped but not total. Current deviations:

- `lua_tolstring` does not coerce numbers to strings (5.2-style NULL
  instead of 5.1 coercion).
- `luaL_argerror` omits the calling function name in the message (no
  debug-name derivation yet).
- `luaL_register` does not set the module table as the caller's
  environment (the 5.1 setfenv behavior).
- `lua_pcall` ignores the message-handler argument (`errfunc`).
- `luaL_newstate` panics on allocation failure rather than returning
  `NULL`/aborting; panic-across-FFI hardening is planned.
- `lua_pushfstring` supports `%s %d %f %p %c %%` (small subset, 512-byte
  buffer).
- Not yet implemented: `lua_tocfunction`, `lua_getupvalue`/`lua_setupvalue`,
  `lua_dump`/`load`, `lua_sethook` family, `luaL_Buffer`, `lua_concat`,
  `luaL_loadfile`, `lua_cpcall`, `luaJIT_setmode`, coroutine API
  (`lua_newthread`/`lua_resume`), and cdata/lightuserdata accessors.
- WebAssembly targets are unsupported (no dynamic loading, no mcode).
- 32-bit targets are not supported by the engine itself.

Engine bugs found and fixed while building this crate (each covered by a
regression test): `package.open` leaking two tables onto a fresh stack;
the `api` setter family resolving negative indices *after* popping
(`lua_setfield`, `lua_settable`, `lua_rawset`, `lua_rawseti`,
`lua_setmetatable`, `lua_gettable`, `lua_rawget`); `luaL_newmetatable`
never pushing the metatable; unsynchronized default-library handle
initialization on Windows; `ctype_mts` metatables not being GC-rooted.

## 10. Testing

- 15 unit tests cover: end-to-end script execution, error object/message
  contracts, stack round-trips (including binary strings), universe
  isolation, trampoline execution, longjmp recovery and state re-use,
  argument checking, closures with upvalues, userdata + metatables,
  registry references (including 5.1 freelist reuse), pseudo-indices, and
  `luaL_register`-style registration from real C code.
- `require_native_module_end_to_end` compiles `tests/testmod.c` into a
  shared library at test time and loads it through the full
  `require`/`loadlib` chain. On Windows it drives `cl /LD` directly and
  generates an import library with `lib.exe /DEF` (because `cargo test`
  does not build the cdylib; the test skips with a hint when the dll is
  absent). On Unix it uses the `cc` crate's shared-library support.
- The three bundled conformance suites (Lua 5.1, LuaJIT 2, LuaJIT 3)
  continue to pass with the C API linked into the CLI.

## 11. Roadmap

- Headers: complete the remaining `lua.h` surface (`luaL_Buffer`,
  `lua_concat`, `luaL_loadfile`, `lua_cpcall`, `lua_sethook`).
- `lua_CFunction` yield support and `lua_newthread`/`lua_resume`.
- Panic isolation (`catch_unwind`) at the FFI boundary.
- CI: native-module tests on Linux/macOS runners.
- A C host embedding example and a `pkg-config` file.
