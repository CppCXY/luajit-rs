//! `require` and the `package` table (a `lj_lib_package` subset:
//! `package.loaded`, `package.preload` and a `package.path` file
//! searcher — no C loaders, no `package.cpath`).

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{arg, err_bad_arg, push};

fn str_key(l: &mut LuaState, s: &[u8]) -> LuaValue {
    let sid = l.heap().intern(s);
    l.heap().str_value(sid)
}

fn package_table(l: &mut LuaState) -> crate::gc::GcPtr<LuaTable> {
    let k = str_key(l, b"package");
    match l.global().globals.as_ref().get_str(k).as_table() {
        Some(t) => t,
        None => {
            let t = l.heap().alloc_table(LuaTable::new(0, 2));
            l.global().globals.as_mut().set(k, LuaValue::table(t));
            t
        }
    }
}

fn sub_table(l: &mut LuaState, t: crate::gc::GcPtr<LuaTable>, name: &[u8]) -> crate::gc::GcPtr<LuaTable> {
    let k = str_key(l, name);
    match t.as_ref().get_str(k).as_table() {
        Some(s) => s,
        None => {
            let s = l.heap().alloc_table(LuaTable::new(0, 2));
            t.as_mut().set(k, LuaValue::table(s));
            s
        }
    }
}

/// Call `func(name, arg2)` above the current frame (re-entrant, unlike
/// `vm::call`) and return its first result.
fn call_loader(
    l: &mut LuaState,
    func: LuaValue,
    name: LuaValue,
    arg2: LuaValue,
) -> Result<LuaValue, crate::err::LuaError> {
    let saved_top = l.top;
    let fs = l.top + 16;
    l.stack_ensure(fs + 8);
    l.stack[fs] = func;
    l.stack[fs + 1] = LuaValue::NIL;
    l.stack[fs + 2] = name;
    l.stack[fs + 3] = arg2;
    crate::vm::execute(l, fs, 2, 1)?;
    let r = l.stack[fs];
    l.top = saved_top;
    Ok(r)
}

fn lib_require(l: &mut LuaState) -> LuaResult<i32> {
    let name_v = arg(l, 0);
    let Some(name_sid) = name_v.as_string_id() else {
        return Err(err_bad_arg(l, 1, "require", "string", ""));
    };
    let name = l.str_static(name_sid).to_vec();

    let pkg = package_table(l);
    let loaded = sub_table(l, pkg, b"loaded");
    let cached = loaded.as_ref().get_str(name_v);
    if !cached.is_nil() {
        push(l, cached);
        return Ok(1);
    }

    // Guard against recursive requires while the module runs.
    loaded.as_mut().set(name_v, LuaValue::TRUE);
    let finish = |l: &mut LuaState, result: LuaValue| -> LuaResult<i32> {
        let stored = if result.is_nil() {
            // Keep the sentinel `true`.
            loaded.as_ref().get_str(name_v)
        } else {
            loaded.as_mut().set(name_v, result);
            result
        };
        push(l, stored);
        Ok(1)
    };
    let fail = |_l: &mut LuaState, e: crate::err::LuaError| {
        loaded.as_mut().set(name_v, LuaValue::NIL);
        e
    };

    // 1. package.preload[name].
    let preload = sub_table(l, pkg, b"preload");
    let loader = preload.as_ref().get_str(name_v);
    if loader.is_func() {
        let r = call_loader(l, loader, name_v, LuaValue::NIL).map_err(|e| fail(l, e))?;
        return finish(l, r);
    }

    // 2. package.path search: '?' <- name with '.' -> '/'.
    let path_v = {
        let k = str_key(l, b"path");
        pkg.as_ref().get_str(k)
    };
    let path = match path_v.as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => b"./?.lua".to_vec(),
    };
    let mut modpath = name.clone();
    for b in modpath.iter_mut() {
        if *b == b'.' {
            *b = b'/';
        }
    }
    let mut tried = Vec::new();
    for tmpl in path.split(|&c| c == b';') {
        if tmpl.is_empty() {
            continue;
        }
        let mut fname = Vec::with_capacity(tmpl.len() + modpath.len());
        for &c in tmpl {
            if c == b'?' {
                fname.extend_from_slice(&modpath);
            } else {
                fname.push(c);
            }
        }
        let fname_str = String::from_utf8_lossy(&fname).into_owned();
        match std::fs::read(&fname_str) {
            Ok(src) => {
                let chunkname = format!("@{}", fname_str);
                let f = match crate::state::load(l, src, &chunkname) {
                    Ok(f) => f,
                    Err(e) => {
                        loaded.as_mut().set(name_v, LuaValue::NIL);
                        return Err(l.runtime_error(
                            format!("error loading module '{}': {}",
                                String::from_utf8_lossy(&name), e)
                            .as_bytes(),
                        ));
                    }
                };
                let fname_v = str_key(l, &fname);
                let r = call_loader(l, f, name_v, fname_v).map_err(|e| fail(l, e))?;
                return finish(l, r);
            }
            Err(_) => tried.push(format!("\n\tno file '{}'", fname_str)),
        }
    }

    loaded.as_mut().set(name_v, LuaValue::NIL);
    Err(l.runtime_error(
        format!("module '{}' not found:{}", String::from_utf8_lossy(&name), tried.concat())
            .as_bytes(),
    ))
}

pub fn open(l: &mut LuaState) {
    let pkg = package_table(l);
    // Defaults; LUA_PATH overrides like the reference implementations.
    let path_bytes = std::env::var("LUA_PATH")
        .map(|s| s.into_bytes())
        .unwrap_or_else(|_| b"./?.lua;./?/init.lua".to_vec());
    let path_v = {
        let sid = l.heap().intern(&path_bytes);
        l.heap().str_value(sid)
    };
    let k = str_key(l, b"path");
    pkg.as_mut().set(k, path_v);
    let loaded = sub_table(l, pkg, b"loaded");
    sub_table(l, pkg, b"preload");

    // Pre-register the built-in libraries (package.loaded.string etc).
    let g = l.global().globals;
    for lib in [
        b"string" as &[u8], b"table", b"math", b"os", b"io", b"bit", b"coroutine", b"package",
    ] {
        let k = str_key(l, lib);
        let v = g.as_ref().get_str(k);
        if !v.is_nil() {
            loaded.as_mut().set(k, v);
        }
    }
    let gk = str_key(l, b"_G");
    loaded.as_mut().set(gk, LuaValue::table(g));

    l.register(b"require", lib_require);
}
