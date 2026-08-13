# luajit-rs

A LuaJIT-compatible Lua implementation written from scratch in Rust —
bytecode compiler, NaN-boxed interpreter, and a tracing JIT compiler with
x86-64 and ARM64 machine-code backends.

The entire codebase was written by **DeepSeek**, with a human providing
direction and review.

> **Status: experimental.** The core language, standard library, GC and
> tracing JIT are functional. The project passes all three bundled test
> suites (Lua 5.1 conformance, LuaJIT 2 and LuaJIT 3).

## Highlights

- **Lua 5.1 language with Lua 5.2+ conveniences enabled by default**:
  `goto`/labels, `;` empty statements, `table.pack`/`table.unpack`, the
  global `rawlen`, `__len` on tables, and 5.2 comparison-metamethod rules.
- **Passes the LuaJIT 2 test suite** — 424 tests, all passing.
- **Passes the LuaJIT 3 test suite** — 15 tests, all passing.
- **Passes the Lua 5.1 conformance test suite** — `final OK`.
- **Tracing JIT compiler** modeled on LuaJIT's design:
  - hot-path detection (hotcounts → penalties → blacklisting),
  - recording interpreter emitting SSA IR with snapshots,
  - FOLD / CSE / DCE, loop optimization (peeling + PHIs), narrowing,
    IV analysis, memory forwarding / DSE and allocation sinking,
  - **x86-64 machine-code backend** (Windows & System V ABIs),
  - **ARM64 machine-code backend** (AArch64 ABI),
  - portable IR executor fallback (also used on WebAssembly).
- **Precise incremental garbage collector** with trace GC safe points.
- **FFI** with `cdef` parser, `new`/`cast`/`sizeof`/`alignof`, C types
  (struct/union/enum/complex/arrays/pointers), native C function calls,
  callbacks, and `ffi.load` for named dynamic libraries.
- **C-style API** (`luajit_rs::api`) — stack-based Lua C API mirror:
  `push_number`/`get_global`/`pcall`/`new_userdata`/`set_metatable` etc.
- **Interactive REPL**, `-e`, `-l`, `-v`, stdin pipeline, script-file
  execution.

## Building

Requires stable Rust (edition 2024).

```sh
cargo build --release
./target/release/luajit-rs script.lua        # run a script
./target/release/luajit-rs -e 'print("hi")'  # run a chunk
./target/release/luajit-rs                   # REPL
cargo test --workspace                       # 125 unit tests
```

## Test suites

Three bundled suites cover conformance and JIT behavior. Run each from
its directory:

```sh
# Lua 5.1 conformance (official Lua 5.1 test suite)
cd tests/lua5.1_test
../../target/release/luajit-rs all.lua        # expect: final OK !!!

# LuaJIT 2 suite
cd tests/luajit2_test
../../target/release/luajit-rs test.lua       # 424 passed

# LuaJIT 3 suite
cd tests/luajit3_test
../../target/release/luajit-rs <file>.lua     # 15 files, all pass
```

## Platform support

| Platform            | Interpreter | JIT (native code) | Notes |
|---------------------|:-----------:|:-----------------:|-------|
| Windows x64         | ✓           | ✓                 | Fully working |
| Linux x64           | ✓           | ✓                 | Fully working |
| Linux ARM64         | ✓           | ✓                 | Fully working |
| macOS x64           | ✓           | ✓                 | Fully working |
| macOS ARM64         | ✓           | ✓                 | Fully working |
| WebAssembly         | ✓           | —                 | Portable IR executor (see `crates/luajit-wasm`) |

## Debugging & tuning

Environment variables:

| Variable              | Effect |
|-----------------------|--------|
| `LUAJIT_RS_NOASM=1`   | Skip native codegen; forces the portable IR executor on all architectures. |
| `LUAJIT_RS_TRDUMP=1`  | Print compiled trace summaries (IR + mcode offsets). |
| `LUAJIT_RS_TRDUMP=2`  | Also dump hex + disassembly of generated machine code. |
| `LUA_PATH` / `LUA_CPATH` | Standard environment variables for `package.path` / `package.cpath`. |

From Lua: `jit.off()` / `jit.on()` / `jit.flush()` / `jit.status()` /
`jit.version` / `jit.arch` / `jit.os`.

## Standard library coverage

Fully or mostly implemented: `base`, `string` (full Lua patterns + `gsub`
with function replacement), `table` (incl. `sort` with custom comparator,
`new`, `pack`/`unpack`), `math`, `bit`, `coroutine`, `os` (subset), `io`
(subset incl. `read`/`write`/`open`/`popen`/`close`/`lines`/`flush`/
`tmpfile`), `package` (with `config`/`cpath`/`searchpath`/`loadlib`/
`seeall`/`loaders`), `debug` (incl. `getinfo`/`traceback`/`getmetatable`/
`setmetatable`/`getfenv`/`setfenv`/`getupvalue`/`setupvalue`/`getlocal`/
`setlocal`/`upvalueid`/`upvaluejoin`), `jit` (`on`/`off`/`flush`/`status`/
`version`/`arch`/`os`).

Partially implemented: `ffi` (`cdef`/`new`/`sizeof`/`alignof`/`cast`/
`typeid`/`typeof`/`abi`/`arch`/`os`/`istype`/`metatype`/`string`/`copy`/
`fill`/`offsetof`/`errno`/`load` — no VLA).

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
│   ├── opt/         FOLD/CSE/DCE, loop peeling + PHIs, narrowing, IV, sink, mem
│   ├── asm/         x86-64 & ARM64 backends (regalloc, guards, exit stubs)
│   ├── exec.rs      portable IR executor + snapshot restore + helpers
│   └── mcode.rs     W^X executable memory management
├── stdlib/          base, string (+ patterns), table, math, bit, os, coroutine,
│                    io, package, debug, jit
├── ffi/             C type parser, cdata, metatype, C namespace
└── util/            shared helpers (strfmt, strscan)
```

Key correspondences with LuaJIT sources: `trace.rs` ≈ `lj_trace.c`,
`record.rs` ≈ `lj_record.c`/`lj_ffrecord.c`, `opt_fold.rs` ≈
`lj_opt_fold.c`, `exec.rs` has no direct LuaJIT equivalent (LuaJIT always
runs native code; we fall back to IR interpretation).

## Workspace

- `crates/luajit-rs` — the engine (library).
- `crates/luajit-rs-cli` — command-line frontend (`luajit-rs` binary).
- `crates/luajit-wasm` — WebAssembly build of the engine.

## License

MIT — see [LICENSE](LICENSE).
