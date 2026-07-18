# luajit-rs

A LuaJIT-style Lua implementation written from scratch in Rust — bytecode
compiler, NaN-boxed interpreter, and a tracing JIT compiler — with **no
`unsafe`-heavy FFI dependencies** (one small dependency: `ahash`).

The entire codebase was written by **DeepSeek**, with a human providing
direction and review.

> **Status: experimental.** The core language, most of the standard
> library, and the tracing JIT work well enough to run and win common
> benchmarks, but this is not yet a drop-in LuaJIT replacement.

## Highlights

- **Lua 5.1 semantics** (the LuaJIT dialect), plus LuaJIT's `bit` library.
- **Tracing JIT compiler**, closely modeled on LuaJIT's design:
  - hot-path detection via hotcount tables, penalties and blacklisting,
  - a recording interpreter emitting SSA IR with snapshots,
  - FOLD / CSE / DCE and loop optimization (peeling + PHIs),
  - side traces with trace linking and machine-code exit patching,
  - an **x86-64 machine-code backend** (Windows & System V ABIs, W^X
    code areas),
  - a **portable IR executor** used on every other architecture
    (including ARM64) and as a fallback for NYI traces.
- **Precise garbage collector** with on-trace GC safe points.
- Interpreter performance beats the PUC Lua 5.4 C interpreter on many
  workloads even with the JIT disabled; with the JIT, compute-bound
  benchmarks run 1.3–6x faster than PUC Lua (allocation-heavy ones are
  still GC-bound, see below).

## Benchmarks

Measured on Windows x64 against PUC Lua (`lua`), lower is better:

| Benchmark          | luajit-rs | PUC Lua | Ratio |
|--------------------|----------:|--------:|------:|
| fannkuch-redux (9) |    0.096s |  0.319s | 0.30x |
| binary-trees (14)  |    1.560s |  1.211s | 1.29x |
| nbody (500k)       |    0.941s |  1.201s | 0.78x |
| spectral-norm (150)|    0.019s |  0.050s | 0.38x |
| mandelbrot (600)   |    0.043s |  0.254s | 0.17x |
| partial-sums (2M)  |    0.202s |  0.266s | 0.76x |

Run them yourself: `./run_lua_benchmarks.ps1` (needs a native `lua` on
`PATH`, or set `NATIVE_LUA`).

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

| Platform        | Interpreter | JIT (machine code) | JIT (IR executor) |
|-----------------|:-----------:|:------------------:|:-----------------:|
| Windows x64     | yes         | yes                | yes               |
| Linux x64       | yes         | yes                | yes               |
| macOS x64       | yes         | yes                | yes               |
| Linux/macOS ARM64 | yes       | not yet            | yes               |

On non-x86-64 targets, traces are still recorded and optimized but run
on the portable IR executor instead of native code — faster than pure
interpretation, slower than machine code. An ARM64 backend
(`jit/asm_arm64`) is on the roadmap.

## Debugging knobs

Environment variables understood by the `luajit-rs` binary:

| Variable            | Effect                                            |
|---------------------|---------------------------------------------------|
| `LUAJIT_RS_JIT=off` | Disable the JIT entirely (pure interpreter).      |
| `LUAJIT_RS_NOASM=1` | Record/optimize traces but skip the x64 backend (forces the portable IR executor). |
| `LUAJIT_RS_TRDUMP=1`| Dump compiled traces: mcode ranges plus a per-IR-instruction code-offset map. |

## Architecture

```
crates/luajit-rs/src
├── compiler/       lexer, parser, bytecode emitter (LuaJIT-format BC)
├── runtime/        NaN-boxed values, strings, tables, GC, coroutines
├── vm/             direct bytecode interpreter + metamethod dispatch
├── jit/
│   ├── trace.rs    hot counters, trace lifecycle, blacklisting, patching
│   ├── record.rs   recording interpreter -> SSA IR (+ fast-function recorder)
│   ├── ir.rs       IR definitions and emission buffer
│   ├── opt_fold.rs FOLD/CSE, opt_loop.rs loop peeling, opt_dce.rs DCE
│   ├── asm_x64.rs  x86-64 backend (regalloc, guards, exit stubs)
│   ├── exec.rs     portable IR executor + snapshot restore + helpers
│   └── mcode.rs    W^X executable memory management
├── stdlib/         base, string (+ patterns), table, math, bit, os, coroutine
└── util/           strtod/strfmt ports
```

Key correspondences with LuaJIT sources: `trace.rs` ≈ `lj_trace.c`,
`record.rs` ≈ `lj_record.c`/`lj_ffrecord.c`, `opt_fold.rs` ≈
`lj_opt_fold.c`, `asm_x64.rs` ≈ `lj_asm_x86.h`, `exec.rs` has no LuaJIT
equivalent (LuaJIT always JITs; we can fall back to interpreting IR).

## Standard library coverage

`base` (incl. `pcall`/`error`/metatable APIs), `string` with full Lua
patterns, `table` (+ `sort`), `math`, `bit` (LuaJIT semantics:
wrapping `tobit` conversion), `os` (subset), `coroutine`, `io`
(subset: `open`/`read`/`write`/`lines`/`close` and file methods),
`require` with `package.loaded`/`package.preload`/`package.path`.

Not yet implemented: full `io` (seek, popen, stdout objects),
`package.cpath`/C loaders, `debug`, FFI.

## Roadmap

- [x] String operations on trace (`string.len/sub/byte/char`, `#s`,
      string comparisons — IRCALL helpers)
- [x] Table allocation and library ops on trace (TNEW/TDUP, `#t`,
      `table.insert`/`remove` array fast paths, `table.concat`)
- [ ] Recursive-call traces (up/down recursion — the binary-trees
      hot path is recursive and currently stays in the interpreter)
- [ ] Trace stitching across NYI bytecodes/builtins
- [ ] ARM64 machine-code backend
- [x] `io` library subset and a `require` loader
- [ ] Cheaper table allocation + incremental GC (allocation-bound
      workloads are dominated by alloc/collect costs, not dispatch)

## License

MIT — see [LICENSE](LICENSE).
