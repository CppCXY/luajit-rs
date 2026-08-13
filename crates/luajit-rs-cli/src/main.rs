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

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use std::io::{self, BufRead, IsTerminal, Read, Write};
use std::process::exit;

use luajit_rs::internal::bc::{BC_MODE, BC_NAMES, bc_a, bc_b, bc_d};
use luajit_rs::internal::dump::dump;
use luajit_rs::internal::lex::Interner;
use luajit_rs::internal::parse::Parser;
use luajit_rs::internal::proto::{KGc, Proto};
use luajit_rs::internal::state::{Lua, load};
use luajit_rs::internal::table::LuaTable;
use luajit_rs::{
    LuaError, LuaState, LuaValue, internal, lua_error_message, lua_getglobal, lua_gettop,
    lua_pcall, lua_peek, lua_pushstring, lua_settop, lual_loadfile, lual_loadstring, lual_openlibs,
};

const LUA_PROMPT: &str = "> ";
const LUA_PROMPT2: &str = ">> ";
const VERSION: &str = "luajit-rs (LuaJIT-compatible interpreter)";

struct Args {
    interactive: bool,
    version: bool,
    noenv: bool,
    exec: bool,
    /// `-b` bytecode save/list requested (save unless `bc_list`).
    bc: bool,
    /// `-bl`: list (disassemble) bytecode instead of saving.
    bc_list: bool,
    /// `-bg`: keep debug info (default; our dump always retains debug info).
    #[allow(dead_code)]
    bc_keep_debug: bool,
    argn: i32,
}

fn collectargs(argv: &[String]) -> Result<Args, String> {
    let mut interactive = false;
    let mut version = false;
    let mut noenv = false;
    let mut exec = false;
    let mut bc = false;
    let mut bc_list = false;
    let mut bc_keep_debug = false;
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
                bc = true;
                // Parse the flags following `-b`: `-bl` (list), `-bs` (strip,
                // default), `-bg` (keep debug). The remaining letters (W/X/d
                // and value-taking options n/t/a/o/F) are accepted by LuaJIT
                // but not implemented here.
                for c in a[2..].chars() {
                    match c {
                        'l' => bc_list = true,
                        'g' => bc_keep_debug = true,
                        's' => {}
                        'W' | 'X' | 'd' => {}
                        _ => return Err(format!("unrecognised option flag '{}'", c)),
                    }
                }
                i += 1;
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
        bc,
        bc_list,
        bc_keep_debug,
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

fn dofile(lua: &mut Lua, name: &str, script_args: &[String]) -> i32 {
    let ll = lua.main();
    lua_settop(ll, 0);
    if lual_loadfile(ll, name).is_err() {
        eprintln!("luajit-rs: {}", error_msg(ll));
        return 1;
    }
    for a in script_args {
        lua_pushstring(ll, a.as_bytes());
    }
    match lua_pcall(ll, script_args.len() as i32, 0, 0) {
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
    lua_settop(ll, 0);
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
                lua_settop(ll, 0);
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

// -- Bytecode save/list (-b[flags]) ----------------------------------------

/// Read the input (file or `-` for stdin), compile to a prototype, then
/// either disassemble (`-bl`) or serialize (`-b`).
fn do_bcsave(args: &[String], argn: usize, flags: &Args) -> i32 {
    let input = args.get(argn).cloned().unwrap_or_else(|| "-".into());
    let output = args.get(argn + 1).cloned();

    if !flags.bc_list && output.is_none() {
        eprintln!("luajit-rs: -b requires both input and output");
        return 1;
    }

    let (src, chunkname) = if input == "-" {
        let mut buf = Vec::new();
        if io::stdin().read_to_end(&mut buf).is_err() {
            eprintln!("luajit-rs: cannot read stdin");
            return 1;
        }
        (buf, "=stdin".to_string())
    } else {
        match std::fs::read(&input) {
            Ok(b) => (b, format!("@{}", input)),
            Err(e) => {
                eprintln!("luajit-rs: cannot open {input}: {e}");
                return 1;
            }
        }
    };

    // The parser throws CompileError via panic_any; translate it back to an
    // error message instead of aborting with an unwind. Binary bytecode
    // (`\x1bLJ` magic) is undumped instead of parsed.
    let mut strs = Interner::new();
    let is_binary = src.len() >= 3 && &src[..3] == b"\x1bLJ";
    let result = if is_binary {
        luajit_rs::internal::undump::undump(&src, &mut strs)
            .map_err(|e| -> Box<dyn std::any::Any + Send> { Box::new(e) })
    } else {
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut parser = Parser::new(src, chunkname.clone(), &mut strs);
            parser.parse()
        }));
        std::panic::set_hook(prev_hook);
        r
    };
    let pt = match result {
        Ok(pt) => pt,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<luajit_rs::internal::lex::CompileError>()
                .map(|e| e.0.clone())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown compile error".into());
            eprintln!("luajit-rs: {msg}");
            return 1;
        }
    };

    let mut out = Vec::new();
    if flags.bc_list {
        let mut list = String::new();
        bclist(&pt, &strs, &chunkname, &mut list);
        out.extend_from_slice(list.as_bytes());
    } else {
        // `-bg` (keep debug) is the default for our dump format; `-b`/`-bs`
        // currently emit the same (debug info retained).
        dump(&pt, &strs, &chunkname, &mut out);
    }

    match output.as_deref() {
        None | Some("-") => {
            let mut so = io::stdout();
            let _ = so.write_all(&out);
            let _ = so.flush();
        }
        Some(name) => {
            if std::fs::write(name, &out).is_err() {
                eprintln!("luajit-rs: cannot write {name}");
                return 1;
            }
        }
    }
    0
}

/// BCMode encodings (must match `BCMode` in compiler/bc.rs). The 16-bit
/// BC_MODE packs A (bits 0-2), B (bits 3-6) and C/D (bits 7-10).
const BCM_NONE: u16 = 0;
const BCM_UV: u16 = 5;
const BCM_LITS: u16 = 7;
const BCM_NUM: u16 = 9;
const BCM_STR: u16 = 10;
const BCM_FUNC: u16 = 12;
const BCM_JUMP: u16 = 13;

/// Format a string constant like LuaJIT's `bcline`: control bytes escaped,
/// values over 40 bytes truncated with a trailing `~`.
fn fmt_str_const(s: &[u8]) -> String {
    let mut out = String::from("\"");
    for &c in s.iter().take(40) {
        match c {
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(c as char),
            other => out.push_str(&format!("\\{:03}", other)),
        }
    }
    if s.len() > 40 {
        out.push('~');
    }
    out.push('"');
    out
}

/// Render one instruction as a single bytecode-list line, LuaJIT-style.
fn bcline(pt: &Proto, strs: &Interner, pc: usize, prefix: &str) -> Option<String> {
    let ins = *pt.bc.get(pc)?;
    let op = (ins & 0xff) as usize;
    if op >= BC_NAMES.len() {
        return None;
    }
    let mode = BC_MODE[op];
    let ma = mode & 7;
    let mb = (mode >> 3) & 15;
    let mc = (mode >> 7) & 15;
    let a = bc_a(ins);
    let name = BC_NAMES[op];
    let a_fmt = if ma == BCM_NONE {
        String::new()
    } else {
        format!("{a}")
    };
    let s = format!("{pc:04} {prefix:2} {name:<6} {a_fmt:>3} ");
    let d = bc_d(ins);
    if mc == BCM_JUMP {
        return Some(format!(
            "{s}=> {pc_d:04}\n",
            pc_d = (pc as u32 + d - 0x7fff)
        ));
    }
    // For ABC-form instructions the 16-bit D field packs B (high byte) and
    // C (low byte); the low byte is the constant/operand selector here.
    let d_c = if mb != BCM_NONE { d & 0xff } else { d };
    if mb == BCM_NONE && mc == BCM_NONE {
        return Some(format!("{s}\n"));
    }
    let mut kc: Option<String> = None;
    match mc {
        BCM_STR => {
            if let Some(KGc::Str(sid)) = pt.kgc.get(d_c as usize) {
                kc = Some(fmt_str_const(strs.get(*sid)));
            }
        }
        BCM_NUM => {
            if let Some(&n) = pt.kn.get(d_c as usize) {
                kc = Some(format!("{n}"));
            }
        }
        BCM_FUNC => {
            if let Some(KGc::Proto(_)) = pt.kgc.get(d_c as usize) {
                kc = Some("proto".to_string());
            }
        }
        BCM_UV => {
            if let Some(u) = pt.uvnames.get(d_c as usize) {
                kc = Some(format!("uv:{u}"));
            }
        }
        _ => {}
    }
    if ma == BCM_UV {
        let ka = pt
            .uvnames
            .get(a as usize)
            .cloned()
            .unwrap_or_else(|| format!("uv:{a}"));
        kc = match kc {
            Some(k) => Some(format!("{ka} ; {k}")),
            None => Some(ka),
        };
    }
    if mb != BCM_NONE {
        let b = bc_b(ins);
        return Some(match kc {
            Some(k) => format!("{s}{b:>3} {d_c:>3}  ; {k}\n"),
            None => format!("{s}{b:>3} {d_c:>3}\n"),
        });
    }
    if let Some(k) = kc {
        return Some(format!("{s}{d_c:>3}      ; {k}\n"));
    }
    // BCMLits (KSHORT etc.) stores a signed 16-bit literal in D.
    let d_s = if mc == BCM_LITS && d_c > 0x7fff {
        d_c.wrapping_sub(0x10000)
    } else {
        d_c
    };
    Some(format!("{s}{d_s:>3}\n"))
}

/// Collect branch targets so jump destinations get a `=>` prefix.
fn bctargets(pt: &Proto) -> std::collections::HashSet<usize> {
    let mut t = std::collections::HashSet::new();
    for (pc, &ins) in pt.bc.iter().enumerate().skip(1) {
        let op = (ins & 0xff) as usize;
        if op >= BC_NAMES.len() {
            continue;
        }
        let mc = (BC_MODE[op] >> 7) & 15;
        if mc == BCM_JUMP {
            let d = bc_d(ins);
            t.insert((pc + d as usize - 0x7fff) & usize::MAX);
        }
    }
    t
}

/// Disassemble a prototype and all its child prototypes (recursively),
/// LuaJIT-style (`-- BYTECODE -- @file:first-last` headers).
fn bclist(pt: &Proto, strs: &Interner, chunkname: &str, out: &mut String) {
    for kgc in &pt.kgc {
        if let KGc::Proto(child) = kgc {
            bclist(child, strs, chunkname, out);
        }
    }
    out.push_str(&format!(
        "-- BYTECODE -- {}:{}-{}\n",
        chunkname.trim_start_matches('@'),
        pt.firstline,
        pt.numline
    ));
    let targets = bctargets(pt);
    for pc in 1..pt.bc.len() {
        let prefix = if targets.contains(&pc) { "=>" } else { "  " };
        if let Some(line) = bcline(pt, strs, pc, prefix) {
            out.push_str(&line);
        }
    }
    out.push('\n');
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
    dofile(lua, name, &argv[argn + 1..])
}

fn main() {
    // Deep Lua recursion through C boundaries (e.g. gsub callbacks) walks
    // the Rust stack; give the VM a generous stack instead of the OS
    // default.
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run_main)
        .expect("spawn main thread")
        .join()
        .expect("main thread panicked");
}

fn run_main() {
    // Enable loading native C modules through require/package.loadlib.
    luajit_rs_cpi::install_factory();
    // On Windows, keep the cdylib resident so modules compiled against its
    // import library resolve their imports from this process.
    #[cfg(windows)]
    {
        use std::ffi::CString;
        if let Ok(exe) = std::env::current_exe()
            && let Some(dir) = exe.parent()
        {
            let dll = dir.join("luajit_rs_cpi.dll");
            if dll.exists()
                && let Ok(c) = CString::new(dll.to_str().unwrap_or_default())
            {
                unsafe extern "system" {
                    fn LoadLibraryA(name: *const u8) -> isize;
                }
                let _ = unsafe { LoadLibraryA(c.as_ptr() as *const u8) };
            }
        }
    }
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
            let _ = dofile(&mut lua, rest, &[]);
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

    if flags.bc {
        exit(do_bcsave(&args, flags.argn as usize, &flags));
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
    } else if (flags.argn as usize) >= args.len()
        && !flags.exec
        && !flags.version
        && flags.argn == 1
    {
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
