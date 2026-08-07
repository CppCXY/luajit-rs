//! `require` and the `package` table.
//!
//! Semantically close to LJ's `lib_package.c`:
//! `package.config`, `package.cpath`, `package.loaded`, `package.loadlib`,
//! `package.path`, `package.preload`, `package.searchpath`, `package.seeall`,
//! `package.loaders` (four searchers), plus the global `module` and `require`.

use crate::err::{LuaError, LuaResult};
use crate::state::LuaState;
use crate::stdlib::nargs;
use crate::table::LuaTable;
use crate::value::LuaValue;
use crate::{
    lua_getfield, lua_getglobal, lua_gettop, lua_isnil, lua_newtable, lua_pop, lua_pushcfunction,
    lua_pushstring, lua_pushvalue, lua_rawseti, lua_register, lua_setfield, lua_setglobal,
};

use super::{arg, err_bad_arg_type, push};

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

/// `luaL_findtable` equivalent: walk the dotted path `fname` from table
/// `start`, creating missing tables.  Returns `Err` on a name conflict
/// (an existing non-table value along the path).
fn findtable(
    l: &mut LuaState,
    start: crate::gc::GcPtr<LuaTable>,
    fname: &[u8],
) -> Result<crate::gc::GcPtr<LuaTable>, ()> {
    let mut cur = start;
    let mut rest = fname;
    loop {
        let dot = rest.iter().position(|&c| c == b'.');
        let (seg, more) = match dot {
            Some(p) => (&rest[..p], true),
            None => (rest, false),
        };
        let k = str_key(l, seg);
        let v = cur.as_ref().get_str(k);
        match v.as_table() {
            Some(t) => cur = t,
            None if v.is_nil() => {
                let t = l
                    .heap()
                    .alloc_table(LuaTable::new(0, if more { 2 } else { 4 }));
                cur.as_mut().set(k, LuaValue::table(t));
                cur = t;
            }
            None => return Err(()),
        }
        if !more {
            break;
        }
        rest = &rest[dot.unwrap() + 1..];
    }
    Ok(cur)
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
    if pat.is_empty() || pat.len() > s.len() {
        return s.to_vec();
    }
    let mut out = Vec::with_capacity(s.len() + repl.len());
    let mut i = 0;
    while i + pat.len() <= s.len() {
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
    let mut s = {
        #[cfg(target_arch = "wasm32")]
        {
            def.to_vec()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var(envname)
                .map(|v| v.into_bytes())
                .unwrap_or_else(|_| def.to_vec())
        }
    };
    {
        let pat: &[u8] = &[LUA_PATHSEP, LUA_PATHSEP];
        let repl: &[u8] = &[LUA_PATHSEP, AUXMARK, LUA_PATHSEP];
        s = gsub(&s, pat, repl);
    }
    {
        let pat: &[u8] = &[AUXMARK];
        s = gsub(&s, pat, def);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let pat: &[u8] = &[LUA_EXECDIR];
        if let Ok(exe) = std::env::current_exe()
            && let Some(parent) = exe.parent()
        {
            let dir = parent.to_string_lossy().into_owned().into_bytes();
            s = gsub(&s, pat, &dir);
        }
    }
    let sid = l.heap().intern(&s);
    l.str_static(sid).to_vec()
}

// ── package.searchpath ──────────────────────────────────────────────────────

fn lib_searchpath(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "searchpath", "string", arg(l, 0)));
        }
    };
    let name = l.str_static(name_sid);
    let path_sid = match arg(l, 1).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(
                l,
                2,
                "searchpath",
                "string",
                arg(l, 2 - 1),
            ));
        }
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
    l.stack[l.base + 1] = l.heap().str_value(
        l.heap()
            .intern(b"dynamic libraries not enabled; no support for target OS"),
    );
    l.stack[l.base + 2] = l.heap().str_value(l.heap().intern(b"absent"));
    l.top = l.base + n;
    Ok(3)
}

// ── package.seeall ──────────────────────────────────────────────────────────

fn lib_seeall(l: &mut LuaState) -> LuaResult<i32> {
    let tab = match arg(l, 0).as_table() {
        Some(t) => t,
        None => return Err(err_bad_arg_type(l, 1, "seeall", "table", arg(l, 0))),
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
    mt.as_mut().set(k, LuaValue::table(l.global().globals));
    Ok(0)
}

// ── module ──────────────────────────────────────────────────────────────────

fn lib_module(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => return Err(err_bad_arg_type(l, 1, "module", "string", arg(l, 0))),
    };
    let name = l.str_static(name_sid).to_vec();
    let nargs = nargs(l);

    // Get or create the module table: _LOADED[modname], else the dotted
    // global path (5.1's ll_module + luaL_findtable).
    let loaded = {
        let loaded_k = str_key(l, b"_LOADED");
        match l.global().registry.as_ref().get_str(loaded_k).as_table() {
            Some(t) => t,
            None => {
                let t = l.heap().alloc_table(LuaTable::new(0, 4));
                l.global()
                    .registry
                    .as_mut()
                    .set(loaded_k, LuaValue::table(t));
                t
            }
        }
    };
    let name_v = str_key(l, &name);
    let tab = match loaded.as_ref().get_str(name_v).as_table() {
        Some(t) => t,
        None => {
            let t = findtable(l, l.global().globals, &name).map_err(|_| {
                l.runtime_error(
                    format!(
                        "name conflict for module '{}'",
                        String::from_utf8_lossy(&name)
                    )
                    .as_bytes(),
                )
            })?;
            loaded.as_mut().set(name_v, LuaValue::table(t));
            t
        }
    };

    // modinit: initialize _M/_NAME/_PACKAGE unless the table was already
    // initialized (has a _NAME field).
    let k_name = str_key(l, b"_NAME");
    if tab.as_ref().get_str(k_name).is_nil() {
        tab.as_mut().set(str_key(l, b"_M"), LuaValue::table(tab));
        tab.as_mut().set(k_name, str_key(l, &name));
        // _PACKAGE: the full name up to and including the last dot.
        let pkg_slice = match name.iter().rposition(|&c| c == b'.') {
            Some(p) => &name[..p + 1],
            None => &name[..0],
        };
        tab.as_mut()
            .set(str_key(l, b"_PACKAGE"), str_key(l, pkg_slice));
    }

    // setfenv: the module table becomes the calling function's
    // environment; error when the caller is not a Lua function.
    let caller = crate::stdlib::debug::caller_lua_func(l)
        .ok_or_else(|| l.runtime_error(b"`module` not called from a Lua function"))?;
    match caller.as_mut() {
        crate::func::GcFunc::Lua(c) => c.env = tab,
        crate::func::GcFunc::C(_) => {
            return Err(l.runtime_error(b"`module` not called from a Lua function"));
        }
    }

    for i in 1..nargs {
        let opt = arg(l, i);
        let _ = call_func(l, opt, &[LuaValue::table(tab)], 0)?;
    }

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
    push(l, l.heap().str_value(l.heap().intern(msg.as_bytes())));
    Ok(1)
}

fn loader_lua(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "loader_lua", "string", arg(l, 0)));
        }
    };
    let name = l.str_static(name_sid);
    let name_str = String::from_utf8_lossy(name).into_owned();
    let pkg = package_table(l);

    // If the module name looks like an absolute/relative path (contains
    // directory separator), try it directly with and without .lua extension.
    // This matches the Lua 5.1 loader's behavior for file paths.
    if name_str.contains('/') || name_str.contains('\\') {
        let mut tried = Vec::new();
        for try_name in [name_str.as_str(), &format!("{}.lua", name_str)] {
            if std::path::Path::new(try_name).is_file() {
                match std::fs::read(try_name) {
                    Ok(src) => {
                        let chunkname = format!("@{}", try_name);
                        match crate::state::load(l, src, &chunkname) {
                            Ok(f) => {
                                push(l, f);
                                return Ok(1);
                            }
                            Err(e) => {
                                return Err(l.runtime_error(
                                    format!("error loading '{}': {}", try_name, e).as_bytes(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        return Err(l.runtime_error(
                            format!("cannot read '{}': {}", try_name, e).as_bytes(),
                        ));
                    }
                }
            }
            tried.push(format!("\n\tno file '{}'", try_name));
        }
        push(
            l,
            l.heap()
                .str_value(l.heap().intern(tried.concat().as_bytes())),
        );
        return Ok(1);
    }

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
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "loader_C", "string", arg(l, 0)));
        }
    };
    let name = l.str_static(name_sid);
    let pkg = package_table(l);
    let path_k = str_key(l, b"cpath");
    let path = match pkg.as_ref().get_str(path_k).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => default_cpath().to_vec(),
    };
    let mut tried = Vec::new();
    match searchpath(name, &path, b".", LUA_DIRSEP, &mut tried) {
        Some(fname) => {
            // The library exists, but there is no dynamic loader in this
            // VM; report it as a broken file (5.1's errorfile).
            let fname_str = String::from_utf8_lossy(&fname).into_owned();
            let msg = format!("\n\tno file '{}' (broken)", fname_str);
            push(l, l.heap().str_value(l.heap().intern(msg.as_bytes())));
            Ok(1)
        }
        None => {
            push(
                l,
                l.heap()
                    .str_value(l.heap().intern(tried.concat().as_bytes())),
            );
            Ok(1)
        }
    }
}

fn loader_croot(l: &mut LuaState) -> LuaResult<i32> {
    let name_sid = match arg(l, 0).as_string_id() {
        Some(sid) => sid,
        None => {
            return Err(err_bad_arg_type(l, 1, "loader_Croot", "string", arg(l, 0)));
        }
    };
    let name = l.str_static(name_sid);
    let name_str = String::from_utf8_lossy(name).into_owned();
    if !name_str.contains('.') {
        return Ok(0);
    }
    // 5.1's Croot loader: search the C path for the root module name.
    let root = name_str.split('.').next().unwrap_or(&name_str);
    let pkg = package_table(l);
    let path_k = str_key(l, b"cpath");
    let path = match pkg.as_ref().get_str(path_k).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => default_cpath().to_vec(),
    };
    let mut tried = Vec::new();
    match searchpath(root.as_bytes(), &path, b".", LUA_DIRSEP, &mut tried) {
        Some(fname) => {
            let fname_str = String::from_utf8_lossy(&fname).into_owned();
            let msg = format!("\n\tno file '{}' (broken)", fname_str);
            push(l, l.heap().str_value(l.heap().intern(msg.as_bytes())));
            Ok(1)
        }
        None => {
            push(
                l,
                l.heap()
                    .str_value(l.heap().intern(tried.concat().as_bytes())),
            );
            Ok(1)
        }
    }
}

// ── require ─────────────────────────────────────────────────────────────────

fn lib_require(l: &mut LuaState) -> LuaResult<i32> {
    let name_v = arg(l, 0);
    let Some(name_sid) = name_v.as_string_id() else {
        return Err(err_bad_arg_type(l, 1, "require", "string", arg(l, 0)));
    };
    let name = l.str_static(name_sid).to_vec();

    let pkg = package_table(l);
    let loaded = sub_table(l, pkg, b"loaded");
    let cached = loaded.as_ref().get_str(name_v);
    // Lua 5.1: only a *truthy* cached value counts as loaded; false or
    // nil means the module must be (re)loaded.
    if cached.is_truthy() {
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
    let found_loader;
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
                    format!("error loading module '{}'", String::from_utf8_lossy(&name)).as_bytes(),
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
        // The module ran: `module()` may have set _LOADED[name] itself.
        // Otherwise the in-progress marker (true) is the result (5.1's
        // sentinel handling).
        let v = loaded.as_ref().get_str(name_v);
        push(l, v);
    }
    Ok(1)
}

// ── open ────────────────────────────────────────────────────────────────────

fn tab_new_preload(l: &mut LuaState) -> LuaResult<i32> {
    let k_table = str_key(l, b"table");
    let table_tab = l
        .global()
        .globals
        .as_ref()
        .get_str(k_table)
        .as_table()
        .unwrap();
    let k_new = str_key(l, b"new");
    push(l, table_tab.as_ref().get_str(k_new));
    Ok(1)
}

fn jit_profile_preload(l: &mut LuaState) -> LuaResult<i32> {
    let t = l.heap().alloc_table(LuaTable::new(0, 1));
    push(l, LuaValue::table(t));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    // Ensure package table exists
    lua_getglobal(l, "package");
    if lua_isnil(l, -1) {
        lua_pop(l, 1);
        lua_newtable(l);
        lua_setglobal(l, "package");
        lua_getglobal(l, "package");
    }
    let pkg_idx = lua_gettop(l) as i32;

    // config, path, cpath
    lua_pushstring(l, config_str());
    lua_setfield(l, pkg_idx, "config");
    {
        let path = resolve_path(l, "LUA_PATH", default_path());
        lua_pushstring(l, &path);
    }
    lua_setfield(l, pkg_idx, "path");
    {
        let cpath = resolve_path(l, "LUA_CPATH", default_cpath());
        lua_pushstring(l, &cpath);
    }
    lua_setfield(l, pkg_idx, "cpath");

    // loaded & preload sub-tables
    ensure_sub_table(l, pkg_idx, "loaded");
    ensure_sub_table(l, pkg_idx, "preload");

    // 5.1: package.loaded is the registry's _LOADED table (module()
    // writes the registry, require reads package.loaded; same table).
    {
        let pkg = package_table(l);
        let loaded = sub_table(l, pkg, b"loaded");
        let k = str_key(l, b"_LOADED");
        l.global().registry.as_mut().set(k, LuaValue::table(loaded));
    }

    // Copy loaded libs into package.loaded
    lua_getfield(l, pkg_idx, "loaded");
    let loaded_idx = lua_gettop(l) as i32;
    for &lib in &[
        "string",
        "table",
        "math",
        "os",
        "io",
        "bit",
        "coroutine",
        "debug",
        "package",
        "jit",
    ] {
        lua_getglobal(l, lib);
        if !lua_isnil(l, -1) {
            lua_setfield(l, loaded_idx, lib);
        } else {
            lua_pop(l, 1);
        }
    }
    lua_getglobal(l, "_G");
    lua_setfield(l, loaded_idx, "_G");
    lua_pop(l, 1); // pop loaded

    // preload entries: table.new, jit.profile
    lua_getfield(l, pkg_idx, "preload");
    let preload_idx = lua_gettop(l) as i32;
    lua_pushcfunction(l, tab_new_preload);
    lua_setfield(l, preload_idx, "table.new");
    lua_pushcfunction(l, jit_profile_preload);
    lua_setfield(l, preload_idx, "jit.profile");
    lua_pop(l, 1); // pop preload

    // loaders (preload, lua, C, all-in-one) indexed 1..4
    lua_newtable(l);
    let loaders_idx = lua_gettop(l) as i32;
    lua_pushcfunction(l, loader_preload);
    lua_rawseti(l, loaders_idx, 1);
    lua_pushcfunction(l, loader_lua);
    lua_rawseti(l, loaders_idx, 2);
    lua_pushcfunction(l, loader_c);
    lua_rawseti(l, loaders_idx, 3);
    lua_pushcfunction(l, loader_croot);
    lua_rawseti(l, loaders_idx, 4);
    lua_setfield(l, pkg_idx, "loaders");

    // searchpath, loadlib, seeall
    lua_pushcfunction(l, lib_searchpath);
    lua_setfield(l, pkg_idx, "searchpath");
    lua_pushcfunction(l, lib_loadlib);
    lua_setfield(l, pkg_idx, "loadlib");
    lua_pushcfunction(l, lib_seeall);
    lua_setfield(l, pkg_idx, "seeall");

    lua_pop(l, 1); // pop package table

    lua_register(l, "require", lib_require);
    lua_register(l, "module", lib_module);
}

fn ensure_sub_table(l: &mut LuaState, pkg_idx: i32, name: &str) {
    lua_getfield(l, pkg_idx, name);
    if lua_isnil(l, -1) {
        lua_pop(l, 1);
        lua_newtable(l);
        lua_pushvalue(l, -1);
        lua_setfield(l, pkg_idx, name);
    }
}
