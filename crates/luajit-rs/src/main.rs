//! LuaJIT-compatible command-line frontend. Ported from luajit.c.
//!
//! Usage: luajit-rs [options] [script [args...]]
//!
//! Options:
//!   -e chunk   Execute string 'chunk'
//!   -l name    Require library 'name'
//!   -b[flags]  Save or list bytecode (same as -bl)
//!   -i         Enter interactive mode after running script
//!   -v         Show version information
//!   -E         Ignore environment variables
//!   --         Stop handling options
//!   -          Execute stdin (non-interactive)

use std::io::{self, BufRead, Read, Write};
use std::path::Path;
use std::process::exit;

const LUA_PROMPT: &str = "> ";
const LUA_PROMPT2: &str = ">> ";
const VERSION: &str = "luajit-rs (LuaJIT-compatible interpreter)";

struct Args {
    interactive: bool,
    version: bool,
    noenv: bool,
    exec: bool,
    argn: i32,
}

fn collectargs(argv: &[String]) -> Result<Args, String> {
    let mut interactive = false;
    let mut version = false;
    let mut noenv = false;
    let mut exec = false;
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        if !a.starts_with('-') || a == "--" {
            if a == "--" { i += 1; }
            break;
        }
        if a == "-" {
            break;
        }
        match a.chars().nth(1) {
            Some('i') => interactive = true,
            Some('v') => version = true,
            Some('E') => noenv = true,
            Some('e') => { exec = true; if a.len() <= 2 { i += 1; if i >= argv.len() { return Err("-e needs argument".into()); } } }
            Some('l') | Some('j') => { if a.len() <= 2 { i += 1; if i >= argv.len() { return Err("needs argument".into()); } } }
            Some('b') => { if exec { return Err("conflicting options".into()); } exec = true; break; }
            Some('O') => {}
            _ => return Err(format!("unrecognised option '{}'", a)),
        }
        i += 1;
    }
    Ok(Args { interactive, version, noenv, exec, argn: i as i32 })
}

fn create_arg_table(l: &mut luajit_rs::state::LuaState, args: &[String], argn: usize) {
    let g = l.global();
    let script_idx = argn.min(args.len().saturating_sub(1));
    let total = args.len() - script_idx.max(0);
    let t = g.heap.alloc_table(luajit_rs::runtime::table::LuaTable::new(0, 1));
    use luajit_rs::value::LuaValue;
    if script_idx < args.len() {
        let name = args[script_idx].as_str();
        let sid = g.heap.intern(name.as_bytes());
        let v = g.heap.str_value(sid);
        t.as_mut().set(LuaValue::number(-1.0), v);
        t.as_mut().set(LuaValue::number(0.0), v);
        for i in 1..total {
            let s = args[script_idx + i].as_str();
            let sid2 = g.heap.intern(s.as_bytes());
            let v2 = g.heap.str_value(sid2);
            t.as_mut().set(LuaValue::number(i as f64), v2);
        }
    }
    let key_sid = g.heap.intern(b"arg");
    let key = g.heap.str_value(key_sid);
    g.globals.as_mut().set(key, LuaValue::table(t));
}

#[cfg(windows)]
unsafe extern "C" { fn _isatty(fd: i32) -> i32; fn _setmode(fd: i32, mode: i32) -> i32; }

fn stdin_is_tty() -> bool {
    #[cfg(windows)]
    { unsafe { _isatty(0) != 0 } }
    #[cfg(not(windows))]
    { unsafe { extern "C" { fn isatty(fd: i32) -> i32; } isatty(0) != 0 } }
}

fn set_stdin_binary() {
    #[cfg(windows)]
    { unsafe { _setmode(0, 0x8000); } }
}

fn pushline(prompt: &str) -> Option<String> {
    let mut stdout = io::stdout();
    let _ = stdout.write_all(prompt.as_bytes());
    let _ = stdout.flush();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => {
            if line.ends_with('\n') { line.pop(); if line.ends_with('\r') { line.pop(); } }
            Some(line)
        }
        Err(_) => None,
    }
}

fn incomplete(err: &str) -> bool { err.contains("<eof>") }

fn loadline(lua: &mut luajit_rs::state::Lua) -> Result<Option<String>, String> {
    let ll = lua.main();
    let first = match pushline(LUA_PROMPT) { Some(s) => s, None => return Ok(None) };
    let mut buf = if first.starts_with('=') { format!("return {}", &first[1..]) } else { first };
    loop {
        match luajit_rs::state::load(ll, buf.as_bytes().to_vec(), "=stdin") {
            Ok(_) => return Ok(Some(buf)),
            Err(e) if incomplete(&e) => {
                match pushline(LUA_PROMPT2) {
                    Some(line) => { buf.push('\n'); buf.push_str(&line); }
                    None => return Ok(None),
                }
            }
            Err(e) => return Err(e),
        }
    }
}

fn dotty(lua: &mut luajit_rs::state::Lua) -> i32 {
    while let Ok(Some(chunk)) = loadline(lua) {
        let ll = lua.main();
        let f = match luajit_rs::state::load(ll, chunk.as_bytes().to_vec(), "=stdin") {
            Ok(v) => v,
            Err(e) => { eprintln!("luajit-rs: {}", e); continue; }
        };
        match luajit_rs::vm::call(ll, f, &[]) {
            Ok(results) => {
                if !results.is_empty() {
                    let key = ll.heap().str_value(ll.heap().intern(b"print"));
                    let print_fn = ll.global().globals.as_ref().get_str(key);
                    if print_fn.is_func() {
                        let mut args: Vec<luajit_rs::value::LuaValue> = results.into_iter().collect();
                        args.insert(0, print_fn);
                        let _ = luajit_rs::vm::call(ll, args[0], &args[1..]);
                    }
                }
            }
            Err(luajit_rs::err::LuaError::Runtime) => {
                let ev = ll.errval;
                let msg = if let Some(sid) = ev.as_string_id() {
                    String::from_utf8_lossy(ll.heap().strings.get(sid)).into_owned()
                } else { format!("{:?}", ev) };
                eprintln!("luajit-rs: {}", msg);
            }
            Err(luajit_rs::err::LuaError::Yield) => {
                eprintln!("luajit-rs: attempt to yield from outside a coroutine");
            }
        }
    }
    println!();
    0
}

fn error_msg(ll: &luajit_rs::state::LuaState) -> String {
    let ev = ll.errval;
    if let Some(sid) = ev.as_string_id() {
        String::from_utf8_lossy(ll.heap().strings.get(sid)).into_owned()
    } else { format!("{:?}", ev) }
}

fn dofile(lua: &mut luajit_rs::state::Lua, name: &str) -> i32 {
    let ll = lua.main();
    let src = match std::fs::read(name) {
        Ok(s) => s,
        Err(e) => { eprintln!("luajit-rs: cannot open {}: {}", name, e); return 1; }
    };
    let chunkname = format!("@{}", Path::new(name).file_name().and_then(|s| s.to_str()).unwrap_or(name));
    let f = match luajit_rs::state::load(ll, src, &chunkname) {
        Ok(v) => v,
        Err(e) => { eprintln!("luajit-rs: {}", e); return 1; }
    };
    match luajit_rs::vm::call(ll, f, &[]) {
        Ok(_) => 0,
        Err(luajit_rs::err::LuaError::Runtime) => { eprintln!("luajit-rs: {}", error_msg(ll)); 1 }
        Err(luajit_rs::err::LuaError::Yield) => { eprintln!("luajit-rs: attempt to yield"); 1 }
    }
}

fn dostring(lua: &mut luajit_rs::state::Lua, s: &str, name: &str) -> i32 {
    let ll = lua.main();
    let f = match luajit_rs::state::load(ll, s.as_bytes().to_vec(), name) {
        Ok(v) => v,
        Err(e) => { eprintln!("luajit-rs: {}", e); return 1; }
    };
    match luajit_rs::vm::call(ll, f, &[]) {
        Ok(_) => 0,
        Err(luajit_rs::err::LuaError::Runtime) => { eprintln!("luajit-rs: {}", error_msg(ll)); 1 }
        Err(luajit_rs::err::LuaError::Yield) => { eprintln!("luajit-rs: attempt to yield"); 1 }
    }
}

fn run_args(lua: &mut luajit_rs::state::Lua, argv: &[String], argn: usize) -> i32 {
    let mut i = 1;
    while i < argn {
        let a = argv[i].as_str();
        if !a.starts_with('-') { break; }
        match a.chars().nth(1) {
            Some('e') => {
                let chunk = if a.len() > 2 { &a[2..] } else { i += 1; argv[i].as_str() };
                if dostring(lua, chunk, "=(command line)") != 0 { return 1; }
            }
            Some('l') => {
                let name = if a.len() > 2 { &a[2..] } else { i += 1; argv[i].as_str() };
                let ll = lua.main();
                let require = ll.global().globals.as_ref().get_str(
                    ll.heap().str_value(ll.heap().intern(b"require")),
                );
                if require.is_func() {
                    let name_sid = ll.heap().intern(name.as_bytes());
                    let name_v = ll.heap().str_value(name_sid);
                    match luajit_rs::vm::call(ll, require, &[name_v]) {
                        Ok(_) => {}
                        Err(luajit_rs::err::LuaError::Runtime) => { eprintln!("luajit-rs: {}", error_msg(ll)); return 1; }
                        Err(luajit_rs::err::LuaError::Yield) => { eprintln!("luajit-rs: attempt to yield"); return 1; }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    0
}

fn handle_script(lua: &mut luajit_rs::state::Lua, argv: &[String], argn: usize) -> i32 {
    if argn >= argv.len() { return 0; }
    let name = argv[argn].as_str();
    if name == "-" {
        let mut src = Vec::new();
        if io::stdin().read_to_end(&mut src).is_err() {
            eprintln!("luajit-rs: cannot read stdin");
            return 1;
        }
        set_stdin_binary();
        return dostring(lua, &String::from_utf8_lossy(&src), "=stdin");
    }
    dofile(lua, name)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let flags = match collectargs(&args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("luajit-rs: {}", e);
            eprintln!("usage: {} [options] [script [args...]]", args[0]);
            exit(1);
        }
    };

    let mut lua = luajit_rs::state::Lua::new();
    luajit_rs::open_libs(lua.main());

    if !flags.noenv {
        if let Ok(init) = std::env::var("LUA_INIT") {
            if init.starts_with('@') { let _ = dofile(&mut lua, &init[1..]); }
            else { let _ = dostring(&mut lua, &init, "="); }
        }
    }

    if flags.version && !flags.interactive {
        println!("{}", VERSION);
    }

    create_arg_table(lua.main(), &args, flags.argn as usize);

    if run_args(&mut lua, &args, flags.argn as usize) != 0 { exit(1); }

    if (flags.argn as usize) < args.len() {
        let s = handle_script(&mut lua, &args, flags.argn as usize);
        if s != 0 { exit(s); }
    }

    if flags.interactive {
        if flags.version { println!("{}", VERSION); }
        dotty(&mut lua);
    } else if (flags.argn as usize) >= args.len() && !flags.exec && !flags.version {
        if stdin_is_tty() {
            println!("{}", VERSION);
            dotty(&mut lua);
        } else {
            let mut src = Vec::new();
            if io::stdin().read_to_end(&mut src).is_err() {
                eprintln!("luajit-rs: cannot read stdin");
                exit(1);
            }
            set_stdin_binary();
            exit(dostring(&mut lua, &String::from_utf8_lossy(&src), "=stdin"));
        }
    }

    exit(0);
}
