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

use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::exit;

use luajit_rs::internal::state::{Lua, load};
use luajit_rs::internal::table::LuaTable;
use luajit_rs::{
    LuaError, LuaState, LuaValue, internal, lua_error_message, lua_getglobal, lua_gettop,
    lua_pcall, lua_peek, lua_pushstring, lua_settop, lual_loadstring,
    lual_openlibs,
};

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
            if a == "--" {
                i += 1;
            }
            break;
        }
        if a == "-" {
            break;
        }
        match a.chars().nth(1) {
            Some('i') => interactive = true,
            Some('v') => version = true,
            Some('E') => noenv = true,
            Some('e') => {
                exec = true;
                if a.len() <= 2 {
                    i += 1;
                    if i >= argv.len() {
                        return Err("-e needs argument".into());
                    }
                }
            }
            Some('l') | Some('j') => {
                if a.len() <= 2 {
                    i += 1;
                    if i >= argv.len() {
                        return Err("needs argument".into());
                    }
                }
            }
            Some('b') => {
                if exec {
                    return Err("conflicting options".into());
                }
                exec = true;
                break;
            }
            Some('O') => {}
            _ => return Err(format!("unrecognised option '{}'", a)),
        }
        i += 1;
    }
    Ok(Args {
        interactive,
        version,
        noenv,
        exec,
        argn: i as i32,
    })
}

fn create_arg_table(l: &mut LuaState, args: &[String], argn: usize) {
    let g = l.global();
    let script_idx = argn.min(args.len().saturating_sub(1));
    let t = g.heap.alloc_table(LuaTable::new(0, 1));
    // Standard Lua 5.1: arg[-1] = interpreter path, arg[0] = script name
    if !args.is_empty() {
        let path = args[0].replace('\\', "/");
        let sid = g.heap.intern(path.as_bytes());
        let v = g.heap.str_value(sid);
        t.as_mut().set(LuaValue::number(-1.0), v);
    }
    if script_idx < args.len() {
        let name = args[script_idx].replace('\\', "/");
        let sid = g.heap.intern(name.as_bytes());
        let v = g.heap.str_value(sid);
        t.as_mut().set(LuaValue::number(0.0), v);
        let total = args.len() - script_idx;
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

fn stdin_is_tty() -> bool {
    io::stdin().is_terminal()
}

fn pushline(prompt: &str) -> Option<String> {
    let mut stdout = io::stdout();
    let _ = stdout.write_all(prompt.as_bytes());
    let _ = stdout.flush();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => None,
        Ok(_) => {
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            Some(line)
        }
        Err(_) => None,
    }
}

fn incomplete(err: &str) -> bool {
    err.contains("<eof>")
}

fn loadline(ll: &mut LuaState) -> Result<Option<Vec<u8>>, String> {
    let first = match pushline(LUA_PROMPT) {
        Some(s) => s,
        None => return Ok(None),
    };
    let mut buf = if let Some(rest) = first.strip_prefix('=') {
        format!("return {rest}")
    } else {
        first
    };
    loop {
        match load(ll, buf.as_bytes().to_vec(), "=stdin") {
            Ok(_) => return Ok(Some(buf.into_bytes())),
            Err(e) if incomplete(&e) => match pushline(LUA_PROMPT2) {
                Some(line) => {
                    buf.push('\n');
                    buf.push_str(&line);
                }
                None => return Ok(None),
            },
            Err(e) => return Err(e.to_string()),
        }
    }
}

fn error_msg(ll: &LuaState) -> String {
    lua_error_message(ll)
}

fn dotty(ll: &mut LuaState) -> i32 {
    while let Ok(Some(chunk)) = loadline(ll) {
        if lual_loadstring(ll, &chunk).is_err() {
            eprintln!("luajit-rs: compile error");
            continue;
        }
        match lua_pcall(ll, 0, -1, 0) {
            Ok(()) => {
                let nresults = lua_gettop(ll);
                if nresults > 0 {
                    let g = ll.global();
                    let print_sid = g.heap.intern(b"print");
                    let key = g.heap.str_value(print_sid);
                    let print_fn = g.globals.as_ref().get_str(key);
                    if print_fn.is_func() {
                        let mut args: Vec<LuaValue> = (0..nresults)
                            .map(|i| lua_peek(ll, (i + 1) as i32))
                            .collect();
                        args.insert(0, print_fn);
                        let _ = internal::call(ll, args[0], &args[1..]);
                    }
                }
                lua_settop(ll, 0);
            }
            Err(LuaError::Runtime) => {
                eprintln!("luajit-rs: {}", error_msg(ll));
            }
            Err(LuaError::Yield) => {
                eprintln!("luajit-rs: attempt to yield from outside a coroutine");
            }
        }
    }
    println!();
    0
}

fn dofile(lua: &mut Lua, name: &str) -> i32 {
    let ll = lua.main();
    let src = match std::fs::read(name) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("luajit-rs: cannot open {name}: {e}");
            return 1;
        }
    };
    let _ = lua_settop(ll, 0);
    if lual_loadstring(ll, &src).is_err() {
        eprintln!("luajit-rs: compile error in {name}: {}", String::from_utf8_lossy(&src));
        return 1;
    }
    match lua_pcall(ll, 0, 0, 0) {
        Ok(()) => 0,
        Err(LuaError::Runtime) => {
            eprintln!("luajit-rs: {}", error_msg(ll));
            1
        }
        Err(LuaError::Yield) => {
            eprintln!("luajit-rs: attempt to yield");
            1
        }
    }
}

fn dostring(lua: &mut Lua, s: &str, name: &str) -> i32 {
    let ll = lua.main();
    let _ = lua_settop(ll, 0);
    if lual_loadstring(ll, s.as_bytes()).is_err() {
        eprintln!("luajit-rs: compile error in {name}");
        return 1;
    }
    match lua_pcall(ll, 0, 0, 0) {
        Ok(()) => 0,
        Err(LuaError::Runtime) => {
            eprintln!("luajit-rs: {}", error_msg(ll));
            1
        }
        Err(LuaError::Yield) => {
            eprintln!("luajit-rs: attempt to yield");
            1
        }
    }
}

fn run_args(lua: &mut Lua, argv: &[String], argn: usize) -> i32 {
    let mut i = 1;
    while i < argn {
        let a = argv[i].as_str();
        if !a.starts_with('-') {
            break;
        }
        match a.chars().nth(1) {
            Some('e') => {
                let chunk = if a.len() > 2 {
                    &a[2..]
                } else {
                    i += 1;
                    argv[i].as_str()
                };
                if dostring(lua, chunk, "=(command line)") != 0 {
                    return 1;
                }
            }
            Some('l') => {
                let name = if a.len() > 2 {
                    &a[2..]
                } else {
                    i += 1;
                    argv[i].as_str()
                };
                // Exactly as LuaJIT's dolibrary():
                //   lua_getglobal(L, "require");
                //   lua_pushstring(L, name);
                //   return report(L, docall(L, 1, 1));
                let ll = lua.main();
                let _ = lua_settop(ll, 0);
                lua_getglobal(ll, "require");
                lua_pushstring(ll, name.as_bytes());
                if lua_pcall(ll, 1, 0, 0) != Ok(()) {
                    eprintln!("luajit-rs: {}", error_msg(ll));
                    return 1;
                }
            }
            Some('j') => {
                let cmd = if a.len() > 2 {
                    &a[2..]
                } else {
                    i += 1;
                    argv[i].as_str()
                };
                match cmd {
                    "on" => lua.global().jit.set_on(true),
                    "off" => lua.global().jit.set_on(false),
                    _ => {
                        eprintln!(
                            "luajit-rs: unknown luaJIT command or jit.* modules not installed"
                        );
                        return 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    0
}

#[cfg(windows)]
fn install_crash_handler() {
    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut u8,
        num_params: u32,
        info: [usize; 15],
    }
    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut u8,
    }
    unsafe extern "system" {
        fn AddVectoredExceptionHandler(
            first: u32,
            f: extern "system" fn(*mut ExceptionPointers) -> i32,
        ) -> usize;
    }
    extern "system" fn filter(ep: *mut ExceptionPointers) -> i32 {
        unsafe {
            let rec = &*(*ep).record;
            if rec.code != 0xC0000005 {
                return 0;
            }
            let rip = *((*ep).context.add(0xF8) as *const u64);
            let rsp = *((*ep).context.add(0x98) as *const u64);
            let fault = if rec.num_params >= 2 { rec.info[1] } else { 0 };
            eprintln!(
                "CRASH code={:#x} rip={:#x} rsp={:#x} access={} fault_addr={:#x}",
                rec.code,
                rip,
                rsp,
                if rec.num_params >= 1 { rec.info[0] } else { 99 },
                fault,
            );
            std::process::exit(3);
        }
    }
    unsafe {
        AddVectoredExceptionHandler(1, filter);
    }
}

fn handle_script(lua: &mut Lua, argv: &[String], argn: usize) -> i32 {
    if argn >= argv.len() {
        return 0;
    }
    let name = argv[argn].as_str();
    if name == "-" {
        let mut src = Vec::new();
        if io::stdin().read_to_end(&mut src).is_err() {
            eprintln!("luajit-rs: cannot read stdin");
            return 1;
        }
        return dostring(lua, &String::from_utf8_lossy(&src), "=stdin");
    }
    dofile(lua, name)
}

fn main() {
    #[cfg(windows)]
    install_crash_handler();
    let args: Vec<String> = std::env::args().collect();

    let flags = match collectargs(&args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("luajit-rs: {e}");
            eprintln!("usage: {} [options] [script [args...]]", args[0]);
            exit(1);
        }
    };

    let mut lua = Lua::new();
    lual_openlibs(lua.main());
    if std::env::var("LUAJIT_RS_JIT").as_deref() == Ok("off") {
        lua.global().jit.set_on(false);
    }

    if !flags.noenv
        && let Ok(init) = std::env::var("LUA_INIT")
    {
        if let Some(rest) = init.strip_prefix('@') {
            let _ = dofile(&mut lua, rest);
        } else {
            let _ = dostring(&mut lua, &init, "=");
        }
    }

    if flags.version && !flags.interactive {
        println!("{VERSION}");
    }

    create_arg_table(lua.main(), &args, flags.argn as usize);

    if run_args(&mut lua, &args, flags.argn as usize) != 0 {
        exit(1);
    }

    if (flags.argn as usize) < args.len() {
        let s = handle_script(&mut lua, &args, flags.argn as usize);
        if s != 0 {
            exit(s);
        }
    }

    if flags.interactive {
        if flags.version {
            println!("{VERSION}");
        }
        let ll = lua.main();
        dotty(ll);
    } else if (flags.argn as usize) >= args.len() && !flags.exec && !flags.version && flags.argn == 1 {
        if stdin_is_tty() {
            println!("{VERSION}");
            let ll = lua.main();
            dotty(ll);
        } else {
            let mut src = Vec::new();
            if io::stdin().read_to_end(&mut src).is_err() {
                eprintln!("luajit-rs: cannot read stdin");
                exit(1);
            }
            exit(dostring(&mut lua, &String::from_utf8_lossy(&src), "=stdin"));
        }
    }

    exit(0);
}
