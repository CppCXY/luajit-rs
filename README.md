# luajit-rs

A LuaJIT3-compatible Lua 5.1 implementation written from scratch in Rust —
bytecode compiler, NaN-boxed interpreter, and a tracing JIT compiler with
x86-64 and ARM64 machine-code backends.

The entire codebase was written by **DeepSeek**, with a human providing
direction and review.

> **Status: experimental.** The core language and standard library are
> mostly functional. The tracing JIT works correctly on x86-64 and ARM64
> JIT is functional.

## Highlights

- **Lua 5.1 / LuaJIT3 Grammars** plus most of the LuaJIT standard library.
- **Passes the LuaJIT 2/3 test suite** — 424 tests, all passing.
- **Tracing JIT compiler** modeled on LuaJIT's design:
  - hot-path detection (hotcounts → penalties → blacklisting),
  - recording interpreter emitting SSA IR with snapshots,
  - FOLD / CSE / DCE and loop optimization (peeling + PHIs),
  - **x86-64 machine-code backend** (Windows & System V ABIs),
  - **ARM64 machine-code backend** (AArch64 ABI),
- **Precise garbage collector** with trace GC safe points.
- **FFI** with `cdef` parser, `new`/`cast`/`sizeof`/`alignof`, C types
  (struct/union/enum/complex/arrays/pointers), and native C function calls.
- **C-style API** (`luajit_rs::api`) — stack-based Lua C API mirror:
  `push_number`/`get_global`/`pcall`/`new_userdata`/`set_metatable` etc.
  A high-level Rust API (`IntoLua`/`FromLua`/`UserData` derive) is planned
  as a companion `luajit-rs-api` crate built on top of the C API.
- **Interactive REPL**, `-e`, stdin pipeline, script-file execution.

## Building

Requires stable Rust (edition 2024).

```sh
cargo build --release
./target/release/luajit-rs script.lua        # run a script
./target/release/luajit-rs -e 'print("hi")'  # run a chunk
./target/release/luajit-rs                   # REPL
cargo test --workspace                       # 125 unit tests
```

### Using the C API

```rust
use luajit_rs::api::Lua;
use luajit_rs::func::CFunction;
use luajit_rs::err::LuaResult;
use luajit_rs::state::LuaState;

let mut lua = Lua::new();

// Execute Lua code
lua.load(b"return 1 + 2", "test").unwrap();
lua.pcall(0, 1).unwrap();
assert_eq!(lua.to_number(-1), Some(3.0));

// Push and read values
lua.push_string(b"hello");
lua.set_global("greeting");
lua.get_global("greeting");
assert_eq!(&lua.to_string(-1), b"hello");

// Register a C function
fn double(l: &mut LuaState) -> LuaResult<i32> {
    let x = luajit_rs::stdlib::arg(l, 0).as_number().unwrap_or(0.0);
    luajit_rs::stdlib::push(l, luajit_rs::value::LuaValue::number(x * 2.0));
    Ok(1)
}
lua.register("double", double);

// Tables
lua.new_table();
lua.push_number(42.0);
lua.set_field(-2, "answer");
lua.get_field(-1, "answer");
assert_eq!(lua.to_number(-1), Some(42.0));

// Userdata
let ptr = lua.new_userdata(64);
unsafe { std::ptr::write(ptr as *mut f64, 3.14); }
let data = lua.check_userdata(-1);
assert!(!data.is_null());
assert_eq!(unsafe { *(data as *const f64) }, 3.14);
```

### LuaJIT test suite

```sh
cd crates/luajit-rs/tests/luajit2_test
../../../target/release/luajit-rs test.lua   # 424 tests (x86-64: all pass)
```

## Platform support

| Platform            | Interpreter | JIT (native code) | Notes |
|---------------------|:-----------:|:-----------------:|-------|
| Windows x64         | ✓           | ✓                 | Fully working |
| Linux x64           | ✓           | ✓                 | Fully working |
| macOS x64           | ✓           | ✓                 | Fully working |
| ARM64 (macOS/Linux) | ✓           | ✓                 | Fully working |

## Debugging & tuning

Environment variables:

| Variable              | Effect |
|-----------------------|--------|
| `LUAJIT_RS_NOASM=1`   | Skip native codegen; forces the portable IR executor on all architectures. |
| `LUAJIT_RS_TRDUMP=1`  | Print compiled trace summaries (IR + mcode offsets). |
| `LUAJIT_RS_TRDUMP=2`  | Also dump hex + disassembly of generated machine code. |
| `LUAJIT_RS_JIT_ARCH`  | Override the auto-detected target architecture for assembly (`x64` / `arm64`). |
| `LUA_PATH` / `LUA_CPATH` | Standard environment variables for `package.path` / `package.cpath`. |

From Lua: `jit.off()` / `jit.on()` / `jit.flush()` / `jit.status()` /
`jit.version` / `jit.arch` / `jit.os`.

## Standard library coverage

Fully or mostly implemented: `base`, `string` (full Lua patterns + `gsub`
with function replacement), `table` (incl. `sort` with custom comparator,
`new`, `pack`/`unpack`), `math`, `bit`, `coroutine`, `os` (subset), `io`
(subset incl. `read`/`write`/`open`/`close`/`lines`/`flush`/`tmpfile`),
`package` (with `config`/`cpath`/`searchpath`/`loadlib`/`seeall`/`loaders`),
`debug` (incl. `getinfo`/`traceback`/`getmetatable`/`setmetatable`/
`getfenv`/`setfenv`/`getupvalue`/`setupvalue`/`getlocal`/`setlocal`/
`upvalueid`/`upvaluejoin`), `jit` (`on`/`off`/`flush`/`status`/`version`/
`arch`/`os`).

Partially implemented: `ffi` (`cdef`/`new`/`sizeof`/`alignof`/`cast`/
`typeid`/`typeof`/`abi`/`arch`/`os`/`istype`/`metatype`/`string`/`copy`/
`fill`/`offsetof`/`errno` — no callbacks, no VLA, no `load`).

Not yet: `string.dump`, `debug.gethook`/`sethook` (stubs only), `dofile`/
`loadfile`, `_VERSION`.


## Architecture

```
crates/luajit-rs/src
├── api/             C-style Lua API (stack ops, tables, userdata, metatables)
├── compiler/        lexer, parser, bytecode emitter (extended LuaJIT BC format)
├── runtime/         NaN-boxed values, strings, tables, GC, userdata, coroutines
├── vm/              direct bytecode interpreter + metamethod dispatch
├── jit/
│   ├── trace.rs     hot counters, trace lifecycle, blacklisting, patching
│   ├── record.rs    recording interpreter → SSA IR (+ fast-function recorder)
│   ├── ir.rs        IR definitions and emission buffer
│   ├── opt_fold.rs  FOLD/CSE/DCE, opt_loop.rs loop peeling + PHIs
│   ├── asm/         x86-64 & ARM64 backends (regalloc, guards, exit stubs)
│   ├── exec.rs      portable IR executor + snapshot restore + helpers
│   └── mcode.rs     W^X executable memory management
├── stdlib/          base, string (+ patterns), table, math, bit, os, coroutine,
│                    io, package, debug, jit
└── ffi/             C type parser, cdata, metatype, C namespace
```

Key correspondences with LuaJIT sources: `trace.rs` ≈ `lj_trace.c`,
`record.rs` ≈ `lj_record.c`/`lj_ffrecord.c`, `opt_fold.rs` ≈
`lj_opt_fold.c`, `exec.rs` has no direct LuaJIT equivalent (LuaJIT always
runs native code; we fall back to IR interpretation).

## License

MIT — see [LICENSE](LICENSE).
