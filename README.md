# luajit-rs

A LuaJIT3-compatible Lua 5.1 implementation written from scratch in Rust —
bytecode compiler, NaN-boxed interpreter, and a tracing JIT compiler with
x86-64 and ARM64 machine-code backends.

The entire codebase was written by **DeepSeek**, with a human providing
direction and review.

> **Status: experimental.** The core language and standard library are
> mostly functional. The tracing JIT works correctly on x86-64. ARM64
> JIT is functional but the number-type guard in type-changing traces
> has a known correctness bug (trace exits incorrectly produce wrong
> results). The portable IR executor serves as a reliable fallback on
> all architectures.

## Highlights

- **Lua 5.1 / LuaJIT3 dialect** plus most of the LuaJIT standard library.
- **Passes the LuaJIT 2/3 test suite** — 424 tests, all passing.
- **Tracing JIT compiler** modeled on LuaJIT's design:
  - hot-path detection (hotcounts → penalties → blacklisting),
  - recording interpreter emitting SSA IR with snapshots,
  - FOLD / CSE / DCE and loop optimization (peeling + PHIs),
  - **x86-64 machine-code backend** (Windows & System V ABIs),
  - **ARM64 machine-code backend** (type guards via UBFX, exit stubs),
- **Precise garbage collector** with trace GC safe points.
- **FFI** with `cdef` parser, `new`/`cast`/`sizeof`/`alignof`, C types
  (struct/union/enum/complex/arrays/pointers), and native C function calls.
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
| ARM64 (macOS/Linux) | ✓           | ~                 | Fully working |

On ARM64, tracing JIT works for many workloads but has a correctness issue
where the number SLOAD guard in type-changing traces does not fire correctly,
causing wrong computations. Use `LUAJIT_RS_NOASM=1` to force the portable
IR executor on ARM64, or `jit.off()` to use the interpreter.

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
├── compiler/       lexer, parser, bytecode emitter (extended LuaJIT BC format)
├── runtime/        NaN-boxed values, strings, tables, GC, coroutines, FuncState
├── vm/             direct bytecode interpreter + metamethod dispatch
├── jit/
│   ├── trace.rs    hot counters, trace lifecycle, blacklisting, patching
│   ├── record.rs   recording interpreter → SSA IR (+ fast-function recorder)
│   ├── ir.rs       IR definitions and emission buffer
│   ├── opt_fold.rs FOLD/CSE/DCE, opt_loop.rs loop peeling + PHIs
│   ├── asm/        x86-64 & ARM64 backends (regalloc, guards, exit stubs)
│   ├── exec.rs     portable IR executor + snapshot restore + helpers
│   └── mcode.rs    W^X executable memory management
├── stdlib/         base, string (+ patterns), table, math, bit, os, coroutine,
│                   io, package, debug, jit
└── ffi/            C type parser, cdata, metatype, C namespace
```

Key correspondences with LuaJIT sources: `trace.rs` ≈ `lj_trace.c`,
`record.rs` ≈ `lj_record.c`/`lj_ffrecord.c`, `opt_fold.rs` ≈
`lj_opt_fold.c`, `exec.rs` has no direct LuaJIT equivalent (LuaJIT always
runs native code; we fall back to IR interpretation).

## License

MIT — see [LICENSE](LICENSE).
