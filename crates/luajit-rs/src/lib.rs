pub mod compiler;
pub mod ffi;
pub mod jit;
pub mod runtime;
pub mod stdlib;
pub mod util;
pub mod vm;

pub use compiler::{bc, dump, lex, parse};
pub use runtime::{func, gc, meta, proto, state, string, table, value};
pub use stdlib::open_libs;
pub use util::{strfmt, strscan};
pub use vm::err;

use runtime::string::Interner;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub fn compile(src: Vec<u8>, chunkname: &str) -> Result<(proto::Proto, Interner), String> {
    let name = chunkname.to_string();
    let result = catch_unwind(AssertUnwindSafe(move || {
        let mut strs = Interner::default();
        let mut parser = parse::Parser::new(src, name, &mut strs);
        let pt = parser.parse();
        (pt, strs)
    }));
    match result {
        Ok(out) => Ok(out),
        Err(e) => {
            if let Some(ce) = e.downcast_ref::<lex::CompileError>() {
                Err(ce.0.clone())
            } else if let Some(s) = e.downcast_ref::<String>() {
                Err(s.clone())
            } else if let Some(s) = e.downcast_ref::<&str>() {
                Err((*s).to_string())
            } else {
                Err("unknown compile error".to_string())
            }
        }
    }
}

pub fn list_bytecode(src: Vec<u8>, chunkname: &str) -> Result<Vec<u8>, String> {
    let (pt, strs) = compile(src, chunkname)?;
    let mut out = Vec::new();
    dump::dump(&pt, &strs, chunkname, &mut out);
    Ok(out)
}

/// Compile and run a chunk on a fresh universe with the base library open.
/// Returns a human-readable error message on failure.
pub fn run_string(src: Vec<u8>, chunkname: &str) -> Result<(), String> {
    let mut lua = state::Lua::new();
    open_libs(lua.main());
    let f = state::load(lua.main(), src, chunkname)?;
    match vm::call(lua.main(), f, &[]) {
        Ok(_) => Ok(()),
        Err(err::LuaError::Runtime) => {
            let ev = lua.main().errval;
            Err(describe_value(lua.main(), ev))
        }
        Err(err::LuaError::Yield) => Err("attempt to yield from outside a coroutine".into()),
    }
}

fn describe_value(l: &mut state::LuaState, v: value::LuaValue) -> String {
    if let Some(sid) = v.as_string_id() {
        String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned()
    } else if let Some(n) = v.as_number() {
        strfmt::g14(n)
    } else {
        format!("(error object: {:?})", v)
    }
}
