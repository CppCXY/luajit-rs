//! `require` and the `package` table.
//!
//! Semantically close to LJ's `lib_package.c`:
//! `package.config`, `package.cpath`, `package.loaded`, `package.loadlib`,
//! `package.path`, `package.preload`, `package.searchpath`, `package.seeall`,
//! `package.loaders` (four searchers), plus the global `module` and `require`.

use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, GcFunc};
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{arg, err_bad_arg, push};

// ── constants ───────────────────────────────────────────────────────────────

#[cfg(windows)]
const LUA_DIRSEP: &[u8] = b"\\";
#[cfg(not(windows))]
const LUA_DIRSEP: &[u8] = b"/";

const LUA_PATHSEP: u8 = b';';
const LUA_PATH_MARK: u8 = b'?';
const LUA_EXECDIR: u8 = b'!';
const AUXMARK: u8 = b'\x01';

fn config_str() -> &'static [u8] {
    if cfg!(windows) {
        b"\\\n;\n?\n!\n-\n"
    } else {
        b"/\n;\n?\n!\n-\n"
    }
}

#[cfg(target_os = "macos")]
fn default_cpath() -> &'static [u8] {
    b"./?.dylib;/usr/local/lib/lua/5.1/?.dylib;/usr/local/lib/lua/5.1/loadall.dylib"
}
#[cfg(target_os = "linux")]
fn default_cpath() -> &'static [u8] {
    b"./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so"
}
#[cfg(windows)]
fn default_cpath() -> &'static [u8] {
    b".\\?.dll;"
}
#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn default_cpath() -> &'static [u8] {
    b""
}

fn default_path() -> &'static [u8] {
    b"./?.lua;./?/init.lua"
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn str_key(l: &mut LuaState, s: &[u8]) -> LuaValue {
    let sid = l.heap().intern(s);
    l.heap().str_value(sid)
}

fn package_table(l: &mut LuaState) -> crate::gc::GcPtr<LuaTable> {
    let k = str_key(l, b"package");
    match l.global().globals.as_ref().get_str(k).as_table() {
        Some(t) => t,
        None => {
            let t = l.heap().alloc_table(LuaTable::new(0, 8));
            l.global().globals.as_mut().set(k, LuaValue::table(t));
            t
        }
    }
}

fn sub_table(
    l: &mut LuaState,
    t: crate::gc::GcPtr<LuaTable>,
    name: &[u8],
) -> crate::gc::GcPtr<LuaTable> {
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

/// `luaL_pushmodule` equivalent: get or create the module table from
/// `_LOADED[modname]` / `_G[modname]`.
fn pushmodule(l: &mut LuaState, modname: &[u8]) -> crate::gc::GcPtr<LuaTable> {
    let loaded_k = str_key(l, b"_LOADED");
    let loaded = match l.global().registry.as_ref().get_str(loaded_k).as_table() {
        Some(t) => t,
        None => {
            let t = l.heap().alloc_table(LuaTable::new(0, 4));
            l.global()
                .registry
                .as_mut()
                .set(loaded_k, LuaValue::table(t));
            t
        }
    };
    let name_v = str_key(l, modname);
    if let Some(t) = loaded.as_ref().get_str(name_v).as_table() {
        return t;
    }
    if let Some(t) = l.global().globals.as_ref().get_str(name_v).as_table() {
        return t;
    }
    let t = l.heap().alloc_table(LuaTable::new(0, 4));
    l.global()
        .globals
        .as_mut()
        .set(name_v, LuaValue::table(t));
    t
}

/// Call `func` with `args` and return the first result.  Errors propagate.
fn call_func(
    l: &mut LuaState,
    func: LuaValue,
    args: &[LuaValue],
    nresults: i32,
) -> Result<LuaValue, LuaError> {
    let saved_top = l.top;
    let saved_base = l.base;
    let fs = l.top + 16;
    l.stack_ensure(fs + 4 + args.len());
    l.stack[fs] = func;
    // l.stack[fs + 1] is the frame link – set by call_c / enter_lua
    for (i, a) in args.iter().enumerate() {
        l.stack[fs + 2 + i] = *a;
    }
    let _ = crate::vm::execute(l, fs, args.len(), nresults)?;
    let r = l.stack[fs];
    l.top = saved_top;
    l.base = saved_base;
    Ok(r)
}

// ── path helpers ────────────────────────────────────────────────────────────

fn gsub(s: &[u8], pat: &[u8], repl: &[u8]) -> Vec<u8> {
    if pat.is_empty() {
        return s.to_vec();
    }
    let mut out = Vec::with_capacity(s.len() + repl.len());
    let mut i = 0;
    while i <= s.len().saturating_sub(pat.len()) {
        if s[i..i + pat.len()] == pat[..] {
            out.extend_from_slice(repl);
            i += pat.len();
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out.extend_from_slice(&s[i..]);
    out
}

/// Search for a file: replace `sep` with `dirsep` in `name`, then iterate
/// over semicolon-separated templates in `path`, replacing `?` with the
/// transformed name.  Return the resolved path on success, otherwise `None`.
fn searchpath(
    name: &[u8],
    path: &[u8],
    sep: &[u8],
    dirsep: &[u8],
    tried: &mut Vec<String>,
) -> Option<Vec<u8>> {
    let mut modpath = name.to_vec();
    if !sep.is_empty() && sep != dirsep {
        let mut i = 0;
        while i <= modpath.len().saturating_sub(sep.len()) {
            if modpath[i..i + sep.len()] == sep[..] {
                modpath.splice(i..i + sep.len(), dirsep.iter().copied());
                i += dirsep.len();
            } else {
                i += 1;
            }
        }
    }
    for tmpl in path.split(|&c| c == LUA_PATHSEP) {
        if tmpl.is_empty() {
            continue;
        }
        let mut fname = Vec::with_capacity(tmpl.len() + modpath.len());
        for &c in tmpl {
            if c == LUA_PATH_MARK {
                fname.extend_from_slice(&modpath);
            } else {
                fname.push(c);
            }
        }
        let fname_str = String::from_utf8_lossy(&fname).into_owned();
        if std::path::Path::new(&fname_str).is_file() {
            return Some(fname);
        }
        tried.push(format!("\n\tno file '{}'", fname_str));
    }
    None
}

/// Resolve a path string from env + default, with `!` → exe-dir and `;;` → `;AUX;`
fn resolve_path(l: &mut LuaState, envname: &str, def: &[u8]) -> Vec<u8> {
    let mut s = std::env::var(envname)
        .map(|v| v.into_bytes())
        .unwrap_or_else(|_| def.to_vec());
    {
        let pat: &[u8] = &[LUA_PATHSEP, LUA_PATHSEP];
        let repl: &[u8] = &[LUA_PATHSEP, AUXMARK, LUA_PATHSEP];
        s = gsub(&s, pat, repl);
    }
    {
        let pat: &[u8] = &[AUXMARK];
        s = gsub(&s, pat, def);
    }
    {
        let pat: &[u8] = &[LUA_EXECDIR];
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let dir = parent.to_string_lossy().into_owned().into_bytes();
                s = gsub(&s, pat, &dir);
            }
        }
    }
    let sid = l.heap().intern(&s);
    l.str_static(sid).to_vec()
}

// ── package.searchpath ──────────────────────────────────────────────────────

fn lib_searchpath(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => return Err(err_bad_arg(l, 1, "searchpath", "string", "")),
    };
    let name = l.str_static(name_sid);
    let path_sid = match arg(l, 1).as_string_id() {
        Some(sid) => sid,
        None => return Err(err_bad_arg(l, 2, "searchpath", "string", "")),
    };
    let path = l.str_static(path_sid);
    let sep = match arg(l, 2).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => b".",
    };
    let dirsep = match arg(l, 3).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => LUA_DIRSEP,
    };
    let mut tried = Vec::new();
    if let Some(found) = searchpath(name, path, sep, dirsep, &mut tried) {
        let sid = l.heap().intern(&found);
        push(l, l.heap().str_value(sid));
        return Ok(1);
    }
    // nil + error message
    let n = 2usize;
    l.stack_ensure(l.base + n);
    l.stack[l.base] = LuaValue::NIL;
    let msg = format!(
        "{}\n\tno file '{}'",
        tried.concat(),
        String::from_utf8_lossy(name)
    );
    l.stack[l.base + 1] = l.heap().str_value(l.heap().intern(msg.as_bytes()));
    l.top = l.base + n;
    Ok(2)
}

// ── package.loadlib ─────────────────────────────────────────────────────────

fn lib_loadlib(l: &mut LuaState) -> LuaResult<i32> {
    let n = 3usize;
    l.stack_ensure(l.base + n);
    l.stack[l.base] = LuaValue::NIL;
    l.stack[l.base + 1] = l
        .heap()
        .str_value(l.heap().intern(
            b"dynamic libraries not enabled; no support for target OS",
        ));
    l.stack[l.base + 2] = l.heap().str_value(l.heap().intern(b"absent"));
    l.top = l.base + n;
    Ok(3)
}

// ── package.seeall ──────────────────────────────────────────────────────────

fn lib_seeall(l: &mut LuaState) -> LuaResult<i32> {
    let tab = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg(l, 1, "seeall", "table", "")),
    };
    let mt = match tab.as_ref().metatable {
        Some(m) => m,
        None => {
            let m = l.heap().alloc_table(LuaTable::new(0, 1));
            tab.as_mut().metatable = Some(m);
            m
        }
    };
    let k = str_key(l, b"__index");
    mt.as_mut()
        .set(k, LuaValue::table(l.global().globals));
    Ok(0)
}

// ── module ──────────────────────────────────────────────────────────────────

fn lib_module(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => return Err(err_bad_arg(l, 1, "module", "string", "")),
    };
    let name = l.str_static(name_sid).to_vec();
    let nargs = super::nargs(l);

    let tab = pushmodule(l, &name);

    let k_m = str_key(l, b"_M");
    tab.as_mut().set(k_m, LuaValue::table(tab));
    let k_name = str_key(l, b"_NAME");
    tab.as_mut().set(k_name, str_key(l, &name));

    let dot = name.iter().rposition(|&c| c == b'.');
    let pkg_slice = match dot {
        Some(p) => &name[..p],
        None => &name[..],
    };
    let k_package = str_key(l, b"_PACKAGE");
    let pkg_sid = l.heap().intern(pkg_slice);
    tab.as_mut()
        .set(k_package, l.heap().str_value(pkg_sid));

    for i in 1..nargs {
        let opt = arg(l, i);
        if opt.is_func() {
            let _ = call_func(l, opt, &[LuaValue::table(tab)], 0)?;
        }
    }

    push(l, LuaValue::table(tab));
    Ok(0)
}

// ── loaders ─────────────────────────────────────────────────────────────────

fn loader_preload(l: &mut LuaState) -> LuaResult<i32> {
    let name_v = arg(l, 0);
    let pkg = package_table(l);
    let preload = sub_table(l, pkg, b"preload");
    let loader = preload.as_ref().get_str(name_v);
    if loader.is_func() {
        push(l, loader);
        return Ok(1);
    }
    let name_s = match name_v.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.str_static(sid)).into_owned(),
        None => String::from("?"),
    };
    let msg = format!("\n\tno field package.preload['{}']", name_s);
    push(
        l,
        l.heap().str_value(l.heap().intern(msg.as_bytes())),
    );
    Ok(1)
}

fn loader_lua(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => return Err(err_bad_arg(l, 1, "loader_lua", "string", "")),
    };
    let name = l.str_static(name_sid);
    let pkg = package_table(l);
    let path_k = str_key(l, b"path");
    let path_v = pkg.as_ref().get_str(path_k);
    let path = match path_v.as_string_id() {
        Some(sid) => l.str_static(sid),
        None => b"./?.lua",
    };
    let mut tried = Vec::new();
    let found = match searchpath(name, path, b".", LUA_DIRSEP, &mut tried) {
        Some(f) => f,
        None => {
            push(
                l,
                l.heap()
                    .str_value(l.heap().intern(tried.concat().as_bytes())),
            );
            return Ok(1);
        }
    };
    let fname_str = String::from_utf8_lossy(&found).into_owned();
    match std::fs::read(&fname_str) {
        Ok(src) => {
            let chunkname = format!("@{}", fname_str);
            match crate::state::load(l, src, &chunkname) {
                Ok(f) => {
                    push(l, f);
                    Ok(1)
                }
                Err(e) => Err(l.runtime_error(
                    format!(
                        "error loading module '{}' from file '{}':\n\t{}",
                        String::from_utf8_lossy(name),
                        fname_str,
                        e
                    )
                    .as_bytes(),
                )),
            }
        }
        Err(e) => Err(l.runtime_error(
            format!(
                "error loading module '{}' from file '{}':\n\t{}",
                String::from_utf8_lossy(name),
                fname_str,
                e
            )
            .as_bytes(),
        )),
    }
}

fn loader_c(l: &mut LuaState) -> LuaResult<i32> {
    let name_s = match arg(l, 0).as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.str_static(sid)).into_owned(),
        None => String::from("?"),
    };
    let msg = format!("\n\tno C loader for module '{}'", name_s);
    push(
        l,
        l.heap().str_value(l.heap().intern(msg.as_bytes())),
    );
    Ok(1)
}

fn loader_croot(l: &mut LuaState) -> LuaResult<i32> {
    let name_s = match arg(l, 0).as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.str_static(sid)).into_owned(),
        None => String::from("?"),
    };
    let msg = format!("\n\tno C root loader for module '{}'", name_s);
    push(
        l,
        l.heap().str_value(l.heap().intern(msg.as_bytes())),
    );
    Ok(1)
}

// ── require ─────────────────────────────────────────────────────────────────

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

    loaded.as_mut().set(name_v, LuaValue::TRUE);

    let loaders_tab = {
        let k = str_key(l, b"loaders");
        match pkg.as_ref().get_str(k).as_table() {
            Some(t) => t,
            None => {
                loaded.as_mut().set(name_v, LuaValue::NIL);
                return Err(l.runtime_error(b"'package.loaders' must be a table"));
            }
        }
    };

    let mut errs: Vec<Vec<u8>> = Vec::new();
    let mut found_loader = LuaValue::NIL;

    let mut idx: i32 = 1;
    loop {
        let loader = loaders_tab.as_ref().get_int(idx);
        if !loader.is_func() {
            loaded.as_mut().set(name_v, LuaValue::NIL);
            let e: String = errs
                .iter()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .collect::<Vec<_>>()
                .concat();
            return Err(l.runtime_error(
                format!(
                    "module '{}' not found:{}",
                    String::from_utf8_lossy(&name),
                    e
                )
                .as_bytes(),
            ));
        }
        match call_func(l, loader, &[name_v], 1) {
            Ok(r) => {
                if r.is_func() {
                    found_loader = r;
                    break;
                } else if let Some(sid) = r.as_string_id() {
                    errs.push(l.str_static(sid).to_vec());
                }
            }
            Err(_) => {
                loaded.as_mut().set(name_v, LuaValue::NIL);
                return Err(l.runtime_error(
                    format!(
                        "error loading module '{}'",
                        String::from_utf8_lossy(&name)
                    )
                    .as_bytes(),
                ));
            }
        }
        idx += 1;
    }

    let result = match call_func(l, found_loader, &[name_v], 1) {
        Ok(r) => r,
        Err(e) => {
            loaded.as_mut().set(name_v, LuaValue::NIL);
            return Err(e);
        }
    };

    if !result.is_nil() {
        loaded.as_mut().set(name_v, result);
        push(l, result);
    } else {
        loaded.as_mut().set(name_v, LuaValue::TRUE);
        push(l, LuaValue::TRUE);
    }
    Ok(1)
}

// ── open ────────────────────────────────────────────────────────────────────

fn tab_new_preload(l: &mut LuaState) -> LuaResult<i32> {
    let k_table = str_key(l, b"table");
    let table_tab = l.global().globals.as_ref().get_str(k_table).as_table().unwrap();
    let k_new = str_key(l, b"new");
    push(l, table_tab.as_ref().get_str(k_new));
    Ok(1)
}

fn jit_profile_preload(l: &mut LuaState) -> LuaResult<i32> {
    let t = l.heap().alloc_table(crate::table::LuaTable::new(0, 1));
    push(l, LuaValue::table(t));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    let pkg = package_table(l);

    // config
    {
        let k = str_key(l, b"config");
        let sid = l.heap().intern(config_str());
        pkg.as_mut().set(k, l.heap().str_value(sid));
    }

    // path & cpath
    {
        let k = str_key(l, b"path");
        let path = resolve_path(l, "LUA_PATH", default_path());
        let sid = l.heap().intern(&path);
        pkg.as_mut().set(k, l.heap().str_value(sid));
    }
    {
        let k = str_key(l, b"cpath");
        let cpath = resolve_path(l, "LUA_CPATH", default_cpath());
        let sid = l.heap().intern(&cpath);
        pkg.as_mut().set(k, l.heap().str_value(sid));
    }

    // loaded & preload
    let loaded = sub_table(l, pkg, b"loaded");
    let _preload = sub_table(l, pkg, b"preload");

    let g = l.global().globals;
    for lib in [
        b"string" as &[u8],
        b"table",
        b"math",
        b"os",
        b"io",
        b"bit",
        b"coroutine",
        b"package",
        b"jit",
    ] {
        let k = str_key(l, lib);
        let v = g.as_ref().get_str(k);
        if !v.is_nil() {
            loaded.as_mut().set(k, v);
        }
    }
    let gk = str_key(l, b"_G");
    loaded.as_mut().set(gk, LuaValue::table(g));

    // Register table.new and jit.profile as preload entries
    {
        let preload = sub_table(l, pkg, b"preload");
        let env = l.global().globals;
        let tab_new_val = LuaValue::func(l.heap().alloc_func(GcFunc::C(CClosure {
            f: tab_new_preload, env, upvals: Vec::new(),
        })));
        let jit_profile_val = LuaValue::func(l.heap().alloc_func(GcFunc::C(CClosure {
            f: jit_profile_preload, env, upvals: Vec::new(),
        })));
        let tab_new_k = str_key(l, b"table.new");
        preload.as_mut().set(tab_new_k, tab_new_val);
        let jit_profile_k = str_key(l, b"jit.profile");
        preload.as_mut().set(jit_profile_k, jit_profile_val);
    }

    // loaders (preload, lua, c, croot) indexed 1..4
    {
        let loaders_tab = l.heap().alloc_table(LuaTable::new(0, 0));
        let env = l.global().globals;

        let set_loader = |idx: i32, f: crate::func::CFunction| {
            let fref = l.heap().alloc_func(GcFunc::C(CClosure {
                f,
                env,
                upvals: Vec::new(),
            }));
            loaders_tab.as_mut().set_int(idx, LuaValue::func(fref));
        };
        set_loader(1, loader_preload);
        set_loader(2, loader_lua);
        set_loader(3, loader_c);
        set_loader(4, loader_croot);

        let k = str_key(l, b"loaders");
        pkg.as_mut().set(k, LuaValue::table(loaders_tab));
    }

    // searchpath, loadlib, seeall
    {
        let env = l.global().globals;
        let k = str_key(l, b"searchpath");
        pkg.as_mut().set(
            k,
            LuaValue::func(l.heap().alloc_func(GcFunc::C(CClosure {
                f: lib_searchpath,
                env,
                upvals: Vec::new(),
            }))),
        );
        let k = str_key(l, b"loadlib");
        pkg.as_mut().set(
            k,
            LuaValue::func(l.heap().alloc_func(GcFunc::C(CClosure {
                f: lib_loadlib,
                env,
                upvals: Vec::new(),
            }))),
        );
        let k = str_key(l, b"seeall");
        pkg.as_mut().set(
            k,
            LuaValue::func(l.heap().alloc_func(GcFunc::C(CClosure {
                f: lib_seeall,
                env,
                upvals: Vec::new(),
            }))),
        );
    }

    l.register(b"require", lib_require);
    l.register(b"module", lib_module);
}
