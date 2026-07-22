# luajit-rs

A LuaJIT-compatible Lua implementation written from scratch in Rust —
bytecode compiler, NaN-boxed interpreter, and a tracing JIT compiler
with an x86-64 machine-code backend.

The entire codebase was written by **DeepSeek**, with a human providing
direction and review.

> **Status: experimental.** The core language and standard library are
> mostly functional. The tracing JIT works on x86-64. ARM64 JIT is
> incomplete — traces run on the portable IR executor, which is slower
> than the interpreter on this architecture.  Use `jit.off()` on ARM64.

## Highlights

- **Lua 5.1 / LuaJIT dialect** plus most of the LuaJIT standard library.
- **Tracing JIT compiler** modeled on LuaJIT's design:
  - hot-path detection (hotcounts → penalties → blacklisting),
  - recording interpreter emitting SSA IR with snapshots,
  - FOLD / CSE / DCE and loop optimization (peeling + PHIs),
  - **x86-64 machine-code backend** (Windows & System V ABIs),
  - **portable IR executor** — interprets optimized SSA IR on any
    platform.  Used as fallback for NYI traces and as the default on
    ARM64 (where native code generation is not yet complete).
- **Precise garbage collector** with trace GC safe points.
- **Interactive REPL**, `-e`, stdin pipeline, script-file execution.

## Building

Requires stable Rust (edition 2024).

```sh
cargo build --release
./target/release/luajit-rs script.lua        # run a script
./target/release/luajit-rs -e 'print("hi")'  # run a chunk
./target/release/luajit-rs                   # REPL
cargo test --workspace                       # test suite
```

## Platform support

| Platform        | Interpreter | JIT (native code) | JIT (IR executor) |
|-----------------|:-----------:|:-----------------:|:-----------------:|
| Windows x64     | ✓           | ✓                 | ✓                 |
| Linux x64       | ✓           | ✓                 | ✓                 |
| macOS x64       | ✓           | ✓                 | ✓                 |
| ARM64 (macOS/Linux) | ✓       | —                 | ✓ (disabled by default) |

On ARM64, native code generation defaults to **off**:
traces are recorded and optimized but run on the portable IR executor.
The IR executor overhead is significant — **`jit.off()` often performs
better on ARM64** because the pure interpreter avoids IR dispatch
overhead.  Set `LUAJIT_RS_NOASM=0` to re-enable the ARM64 backend
(work in progress).

## Standard library coverage

Fully or mostly implemented: `base`, `string` (full Lua patterns + `gsub`
with function replacement), `table` (incl. `sort` with custom comparator,
`new`, `pack`/`unpack`), `math`, `bit`, `coroutine`, `os` (subset), `io`
(subset), `package` (with `config`/`cpath`/`searchpath`/`loadlib`/
`seeall`/`loaders`), `debug` (incl. `getinfo`/`traceback`/`getmetatable`/
`getfenv`/`setfenv`), `jit` (on/off/flush/status/version/arch/os).

Partially implemented: FFI (`cdef`/`new`/`sizeof`/`cast`/`abi`/`arch`/
`os` etc. — no callbacks, no VLA, no `load`).

Not yet: `io.lines` iterator from file handle, `os.execute` return value,
`string.dump`, `debug.gethook`/`sethook` (stubs only), `dofile`/
`loadfile`, `_VERSION`.

## Debugging & tuning

Environment variables:

| Variable              | Effect |
|-----------------------|--------|
| `LUAJIT_RS_NOASM=1`   | Record & optimize traces but skip native codegen; forces the portable IR executor on all architectures. |
| `LUAJIT_RS_TRDUMP=1`  | Print compiled trace summaries (IR + mcode offsets). |
| `LUAJIT_RS_TRDUMP=2`  | Also dump hex + disassembly of generated machine code (ARM64). |
| `LUAJIT_RS_JIT_ARCH`  | Override the auto-detected target architecture for assembly (`x64` / `arm64`). |
| `LUA_PATH` / `LUA_CPATH` | Standard environment variables for `package.path` / `package.cpath`. |

From Lua: `jit.off()` / `jit.on()`.

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
`lj_opt_fold.c`, `exec.rs` has no direct LuaJIT equivalent (LuaJIT
always runs native code; we fall back to IR interpretation).

## License

MIT — see [LICENSE](LICENSE).
