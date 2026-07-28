# luajit-wasm

LuaJIT-compatible Lua engine for WebAssembly, based on [luajit-rs](https://github.com/CppCXY/luajit-rs).

## Build

```bash
# Install wasm-pack
cargo install wasm-pack

# Build web target
wasm-pack build crates/luajit-wasm --target web --out-dir pkg
```

## Run locally

```bash
# Serve the directory (any static server works)
npx serve .
# or
python -m http.server 8080
```

Open `http://localhost:8080` and navigate to the editor page.

## API

```js
import { LuaWasm } from "./pkg/luajit_wasm.js";

// Initialize (async — loads wasm binary)
const lua = await new LuaWasm();

// Execute Lua code; returns the first result as a string
lua.do_string("return 1 + 2");          // "3"

// Register a global function, then call it
lua.do_string("function greet(n) return 'hello ' .. n end");
lua.call("greet", [42]);                // "hello 42"

// Set / get globals
lua.set_global("x", 100);
lua.get_global("x");                    // "100"

// GC control
lua.gc_collect();
lua.gc_count();                         // memory in KB
```

Values are converted between JS and Lua:

| JS | Lua |
|----|-----|
| `null`, `undefined` | `nil` |
| `boolean` | `boolean` |
| `number` | `number` |
| `string` | `string` |
| `Array` | `table` (1-indexed) |
| `Object` | `table` (key-value) |
