This directory contains the test suite for **luajit-rs**, derived from
[LuaJIT-test-cleanup](https://github.com/LuaJIT/LuaJIT-test-cleanup). It has
been revised to match the behaviour of the current LuaJIT version and is
primarily used to verify the correctness of luajit-rs.

Large portions of the suite can also be run with any LuaJIT interpreter.

## Revisions from upstream

Notable changes made to align with the current LuaJIT release:

- **`math.mod`** — Removed from the expected pre-5.2 API (LuaJIT no longer
  provides it).
- **`string.gfind`** — Removed from the expected pre-5.2 API.
- **`table.move`** — Added to the expected table library contents (LuaJIT now
  includes this Lua 5.3 function).

## Running the test suite

Run `test.lua` with the LuaJIT interpreter:

```
luajit test.lua
```

If all tests pass, the final line will be `NNN passed` and the exit code will
be zero. Otherwise, a non-zero exit code is returned along with details of
every failure.

To run only specific tests, pass their numbers as arguments (the numbers are
printed after a failing run):

```
luajit test.lua 362 364 365
```

For more options:

```
luajit test.lua --help
```

## Structure

The suite is a directory tree. Each directory contains an `index` file listing
its members and optional metadata tags (e.g. `+ffi`, `-compat5.2`). Every
`.lua` file contains one or more tests.

A test looks like:

```lua
do --- <name> <metadata>
  <code>
end
```

Common definitions shared by tests in the same file go at the top. Code shared
across files lives in the `common/` directory and is loaded via `require`.

### Common metadata tags

| Tag | Meaning |
|-----|---------|
| `+lua<5.2` | Requires Lua < 5.2 |
| `+luajit>=2.1` | Requires LuaJIT 2.1+ |
| `+ffi` | Requires the FFI library |
| `+bit` | Requires the bit library |
| `+jit` | Requires JIT compilation enabled |
| `+compat5.2` | Requires Lua 5.2 compatibility mode |
| `!private_G` | Test mutates globals; needs a private `_G` copy |

## Adding tests

1. Choose or create the target `.lua` file.
2. Wrap each test in `do --- <name> <metadata> ... end`.
3. Add metadata tags for required features.
4. If creating new files/directories, update the relevant `index` files.

Each test should call `assert` or raise an error on misbehaviour. Tests must
not write to stdout/stderr or mutate global state (unless marked
`!private_G`).
