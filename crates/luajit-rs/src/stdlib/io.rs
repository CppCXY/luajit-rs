//! `io` library subset. Since the runtime has no userdata, a file
//! handle is a plain table `{ __fd = id, read = ..., write = ...,
//! lines = ..., close = ... }`; the id indexes a process-wide file
//! registry (files are OS resources shared across VMs).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, Write};

use crate::api::lua_gettop;
use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, GcFunc};
use crate::runtime::userdata::GcUserData;
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, err_bad_arg_type, push, pushv, tostring_bytes};
use crate::lual_reg;

use crate::state::FileEntry;

fn registry_put(l: &mut LuaState, e: FileEntry) -> usize {
    // Never reuse a closed fd: an old file userdata's `__gc` would then
    // close a *new* file sharing the fd (LuaJIT fds are unique FILE*).
    let files = l.files_mut();
    files.push(Some(e));
    files.len() - 1
}

fn str_arg(l: &mut LuaState, i: usize, name: &str) -> Result<&'static [u8], LuaError> {
    match arg(l, i).as_string_id() {
        Some(sid) => Ok(l.str_static(sid)),
        None => Err(err_bad_arg(l, i as u32 + 1, name, "string", "")),
    }
}

fn ret_string(l: &mut LuaState, bytes: &[u8]) -> LuaResult<i32> {
    let sid = l.heap().intern(bytes);
    let v = l.heap().str_value(sid);
    push(l, v);
    Ok(1)
}

fn ret_fail(l: &mut LuaState, msg: &str) -> LuaResult<i32> {
    let sid = l.heap().intern(msg.as_bytes());
    let sv = l.heap().str_value(sid);
    pushv(l, &[LuaValue::NIL, sv]);
    Ok(2)
}

/// io.open failure: `nil, "<path>: <msg>", errno` (Lua 5.1).
fn ret_fail3(l: &mut LuaState, msg: &str) -> LuaResult<i32> {
    let sid = l.heap().intern(msg.as_bytes());
    let sv = l.heap().str_value(sid);
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as f64;
    pushv(l, &[LuaValue::NIL, sv, LuaValue::number(errno)]);
    Ok(3)
}

// -- Handle tables -----------------------------------------------------------

fn handle_fd(l: &LuaState, i: usize) -> Option<usize> {
    let u = arg(l, i).as_userdata()?;
    let fd = u.as_ref().borrow::<usize>()?;
    Some(*fd)
}

/// Build (or reuse) the file-handle userdata for a registered file id.
/// Only the standard streams (fds 0/1/2, registered first) are cached so
/// `io.output() == io.stdout`; regular files get a fresh userdata each time
/// so the GC can close them via `__gc` when unreferenced (LuaJIT semantics).
fn new_handle(l: &mut LuaState, id: usize) -> LuaValue {
    if id < 3
        && let Some(Some(v)) = l.global().io_file_cache.get(id)
    {
        return *v;
    }
    let ud = l.heap().alloc_userdata(GcUserData::new(id));
    push(l, LuaValue::userdata(ud)); // anchor for the metatable alloc
    let ud = arg(l, 0).as_userdata().unwrap();
    let env = l.global().globals;
    let mt = l.heap().alloc_table(LuaTable::new(0, 12));
    // LuaJIT's io metatable: methods live directly on it, __index points
    // back at it (so `pairs(getmetatable(f))` lists the methods too).
    let idx_key = l.heap().str_value(l.heap().intern(b"__index"));
    mt.as_mut().set(idx_key, LuaValue::table(mt));
    for (name, f) in [
        (
            b"close".as_slice(),
            handle_close_fd as crate::func::CFunction,
        ),
        (b"flush".as_slice(), handle_flush_fd),
        (b"lines".as_slice(), handle_lines_fd),
        (b"read".as_slice(), handle_read_fd),
        (b"seek".as_slice(), handle_seek_fd),
        (b"setvbuf".as_slice(), handle_setvbuf_fd),
        (b"write".as_slice(), handle_write_fd),
    ] {
        let k = l.heap().str_value(l.heap().intern(name));
        let fref = l.heap().alloc_func(GcFunc::C(CClosure {
            f,
            env,
            upvals: vec![],
        }));
        mt.as_mut().set(k, LuaValue::func(fref));
    }
    let ts_ref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: handle_tostring,
        env,
        upvals: vec![],
    }));
    let ts_key = l.heap().str_value(l.heap().intern(b"__tostring"));
    mt.as_mut().set(ts_key, LuaValue::func(ts_ref));
    let gc_ref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: handle_gc_fd,
        env,
        upvals: vec![],
    }));
    let gck = l.heap().str_value(l.heap().intern(b"__gc"));
    mt.as_mut().set(gck, LuaValue::func(gc_ref));
    ud.as_mut().metatable = Some(mt);

    l.top -= 1; // pop anchor
    let v = LuaValue::userdata(ud);
    if id < 3 {
        let cache = &mut l.global().io_file_cache;
        if cache.len() <= id {
            cache.resize(id + 1, None);
        }
        cache[id] = Some(v);
    }
    v
}

fn handle_flush_fd(l: &mut LuaState) -> LuaResult<i32> {
    let id = handle_fd_arg(l, 0)?;
    let files = l.files_mut();
    match files.get_mut(id).and_then(|e| e.as_mut()) {
        Some(FileEntry::Write(f)) => {
            let _ = f.get_ref().flush();
            push(l, LuaValue::TRUE);
            Ok(1)
        }
        Some(FileEntry::ReadWrite(f)) => {
            let _ = f.flush();
            push(l, LuaValue::TRUE);
            Ok(1)
        }
        _ => Ok(0),
    }
}

fn handle_seek_fd(l: &mut LuaState) -> LuaResult<i32> {
    let id = handle_fd_arg(l, 0)?;
    // whence (set/cur/end, default "cur") and offset (default 0). The
    // stack is [file, whence, offset]: whence = arg(1), offset = arg(2).
    let whence = match arg(l, 1).as_string_id() {
        Some(sid) => l.heap().strings.get(sid).to_vec(),
        None => b"cur".to_vec(),
    };
    let off = arg(l, 2).as_number().unwrap_or(0.0) as i64;
    let pos = match whence.as_slice() {
        b"set" => std::io::SeekFrom::Start(off.max(0) as u64),
        b"cur" => std::io::SeekFrom::Current(off),
        b"end" => std::io::SeekFrom::End(off),
        _ => return Err(l.runtime_error(b"invalid option 'whence'")),
    };
    let files = l.files_mut();
    let res = match files.get_mut(id).and_then(|e| e.as_mut()) {
        Some(FileEntry::Read(r)) => r.seek(pos),
        Some(FileEntry::Write(f)) => {
            let _ = f.flush();
            f.get_mut().seek(pos)
        }
        Some(FileEntry::ReadWrite(f)) => f.seek(pos),
        _ => return Err(l.runtime_error(b"attempt to use a closed file")),
    };
    match res {
        Ok(p) => {
            push(l, LuaValue::number(p as f64));
            Ok(1)
        }
        Err(e) => Err(l.runtime_error(e.to_string().as_bytes())),
    }
}

fn handle_setvbuf_fd(l: &mut LuaState) -> LuaResult<i32> {
    // setvbuf("no") = flush every write, "line" = flush on newline,
    // otherwise (full/default) = flush on close/flush.
    let fd = handle_fd_arg(l, 0)?;
    let mode = match arg(l, 1).as_string_id().map(|sid| l.str_static(sid)) {
        Some(b"no") => 2u8,
        Some(b"line") => 1u8,
        Some(b"full") | Some(b"") => 0u8,
        _ => return Err(l.runtime_error(b"bad argument #2 to 'setvbuf' (invalid option)")),
    };
    let m = &mut l.global().file_buf_mode;
    if m.len() <= fd {
        m.resize(fd + 1, 0);
    }
    m[fd] = mode;
    push(l, LuaValue::TRUE);
    Ok(1)
}

fn handle_fd_arg(l: &mut LuaState, i: usize) -> LuaResult<usize> {
    match handle_fd(l, i) {
        Some(fd) => Ok(fd),
        None => Err(l.runtime_error(b"attempt to use a closed file")),
    }
}
fn handle_read_fd(l: &mut LuaState) -> LuaResult<i32> {
    let fd = handle_fd_arg(l, 0)?;
    do_read(l, Some(fd), 1)
}
fn handle_write_fd(l: &mut LuaState) -> LuaResult<i32> {
    let fd = handle_fd_arg(l, 0)?;
    do_write(l, Some(fd), 1)
}
fn handle_lines_fd(l: &mut LuaState) -> LuaResult<i32> {
    let fd = handle_fd_arg(l, 0)?;
    let it = make_lines_iter(l, fd, false); // f:lines() shares the handle
    push(l, it);
    Ok(1)
}
fn handle_close_fd(l: &mut LuaState) -> LuaResult<i32> {
    let fd = handle_fd_arg(l, 0)?;
    let files = l.files_mut();
    match files.get_mut(fd) {
        Some(slot) if slot.is_some() => {
            // io.popen: wait for the child before dropping the entry
            // (dropping a Child without wait() leaves a zombie).
            if let Some(FileEntry::Pipe(child)) = slot.as_mut() {
                let _ = child.wait();
            }
            *slot = None;
        }
        // LuaJIT: closing an already-closed file is an error (a GC `__gc`
        // re-invocation is caught by the collector).
        _ => return Err(l.runtime_error(b"attempt to use a closed file")),
    }
    push(l, LuaValue::TRUE);
    Ok(1)
}

/// File-handle `__gc`: unlike an explicit `close`, finalizing an
/// already-closed handle is a silent no-op (Lua 5.1's `io_f_gc` clears the
/// file pointer and only closes when non-null). This matters for the
/// `io.close(io.output())` idiom: the temporary handle is collected later,
/// and its `__gc` must not raise "attempt to use a closed file".
fn handle_gc_fd(l: &mut LuaState) -> LuaResult<i32> {
    let fd = match handle_fd(l, 0) {
        Some(fd) => fd,
        // Lua 5.1's `io_gc` calls `tofilep` unconditionally: invoking
        // `getmetatable(io.stdin).__gc()` without a file argument raises
        // "bad argument #1 to '__gc' (FILE* expected, got no value)".
        None => return Err(err_bad_arg_type(l, 1, "__gc", "FILE*", arg(l, 0))),
    };
    // Never close the standard streams, or a file that is still the
    // current default input/output, via GC: a discarded
    // `io.input()`/`io.output()` handle must not close the file it
    // points at (files.lua reads a file to EOF while an earlier
    // temporary handle is collected).
    if fd < 3 {
        return Ok(0);
    }
    if l.global().default_input == Some(fd) || l.global().default_output == Some(fd) {
        return Ok(0);
    }
    let files = l.files_mut();
    if let Some(slot) = files.get_mut(fd)
        && slot.is_some() {
            if let Some(FileEntry::Pipe(child)) = slot {
                let _ = child.wait();
            }
            *slot = None;
        }
    Ok(0)
}

fn handle_tostring(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            let open = l.files_mut().get(fd).and_then(|e| e.as_ref()).is_some();
            let s = if open {
                format!("file ({:#x})", fd)
            } else {
                "file (closed)".to_string()
            };
            let sid = l.heap().intern(s.as_bytes());
            push(l, l.heap().str_value(sid));
            Ok(1)
        }
        None => Err(l.runtime_error(b"attempt to use a closed file")),
    }
}

// -- Reading -----------------------------------------------------------------

/// One `io.read`-style format applied to a buffered reader. Returns the
/// pushed value, or nil at EOF.
fn read_format(l: &mut LuaState, r: &mut dyn BufRead, fmt: LuaValue) -> Result<LuaValue, LuaError> {
    if let Some(n) = fmt.as_number() {
        // Read exactly n bytes.
        let want = n.max(0.0) as usize;
        if want == 0 {
            // io.read(0): Lua 5.1's `test_eof` — the empty string
            // mid-file, nil at end of file.
            let b = match r.fill_buf() {
                Ok(b) => b.to_vec(),
                Err(e) => return Err(l.runtime_error(e.to_string().as_bytes())),
            };
            if b.is_empty() {
                return Ok(LuaValue::NIL);
            }
            return Ok(l.heap().str_value(l.heap().intern(b"")));
        }
        let mut buf = vec![0u8; want];
        let mut got = 0;
        while got < want {
            match r.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(e) => return Err(l.runtime_error(e.to_string().as_bytes())),
            }
        }
        if got == 0 && want > 0 {
            return Ok(LuaValue::NIL);
        }
        let sid = l.heap().intern(&buf[..got]);
        return Ok(l.heap().str_value(sid));
    }
    let spec = fmt
        .as_string_id()
        .map(|sid| l.str_static(sid))
        .unwrap_or(b"*l");
    let kind = *spec.iter().find(|&&c| c != b'*').unwrap_or(&b'l');
    match kind {
        b'l' | b'L' => {
            let mut line = Vec::new();
            match r.read_until(b'\n', &mut line) {
                Ok(0) => Ok(LuaValue::NIL),
                Ok(_) => {
                    if kind == b'l' {
                        if line.last() == Some(&b'\n') {
                            line.pop();
                        }
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                    }
                    let sid = l.heap().intern(&line);
                    Ok(l.heap().str_value(sid))
                }
                Err(e) => Err(l.runtime_error(e.to_string().as_bytes())),
            }
        }
        b'a' => {
            let mut all = Vec::new();
            match r.read_to_end(&mut all) {
                Ok(_) => {
                    let sid = l.heap().intern(&all);
                    Ok(l.heap().str_value(sid))
                }
                Err(e) => Err(l.runtime_error(e.to_string().as_bytes())),
            }
        }
        b'n' => {
            // Skip whitespace, then scan a number token.
            let mut tok = Vec::new();
            loop {
                let (done, used) = {
                    let buf = match r.fill_buf() {
                        Ok(b) => b,
                        Err(e) => return Err(l.runtime_error(e.to_string().as_bytes())),
                    };
                    if buf.is_empty() {
                        (true, 0)
                    } else {
                        let mut used = 0;
                        let mut done = false;
                        for &c in buf {
                            let is_ws = c.is_ascii_whitespace();
                            if tok.is_empty() && is_ws {
                                used += 1;
                                continue;
                            }
                            if is_ws {
                                done = true;
                                break;
                            }
                            tok.push(c);
                            used += 1;
                        }
                        (done, used)
                    }
                };
                r.consume(used);
                if done || used == 0 {
                    break;
                }
            }
            match crate::strscan::scan_number(&tok) {
                Some(n) => Ok(LuaValue::number(n)),
                None => Ok(LuaValue::NIL),
            }
        }
        _ => Err(l.runtime_error(b"bad argument to 'read' (invalid format)")),
    }
}

fn do_read(l: &mut LuaState, fd: Option<usize>, first_fmt: usize) -> LuaResult<i32> {
    let n = lua_gettop(l);
    let mut fmts: Vec<LuaValue> = (first_fmt..n.max(first_fmt)).map(|i| arg(l, i)).collect();
    if fmts.is_empty() {
        fmts.push(LuaValue::NIL); // Default: one line.
    }
    let mut out = Vec::with_capacity(fmts.len());
    match fd {
        None => match l.global().default_input {
            Some(id) => {
                let files = l.files_mut();
                match files.get_mut(id).and_then(|e| e.as_mut()) {
                    Some(FileEntry::Read(r)) => {
                        for f in fmts {
                            out.push(read_format(l, r, f)?);
                        }
                    }
                    Some(FileEntry::ReadWrite(file)) => {
                        let mut r = BufReader::new(&mut *file);
                        for f in fmts {
                            out.push(read_format(l, &mut r, f)?);
                        }
                    }
                    Some(FileEntry::Stdin) => {
                        let stdin = std::io::stdin();
                        let mut lock = stdin.lock();
                        for f in fmts {
                            out.push(read_format(l, &mut lock, f)?);
                        }
                    }
                    // io.popen("cmd", "r"): read the child's stdout.
                    Some(FileEntry::Pipe(child)) => match child.stdout.as_mut() {
                        Some(stdout) => {
                            let mut r = BufReader::new(stdout);
                            for f in fmts {
                                out.push(read_format(l, &mut r, f)?);
                            }
                        }
                        None => {
                            for _ in &fmts {
                                out.push(LuaValue::NIL);
                            }
                        }
                    },
                    // LuaJIT: reading a write-mode handle yields nil (EOF).
                    Some(FileEntry::Write(_) | FileEntry::Stdout | FileEntry::Stderr) => {
                        for _ in &fmts {
                            out.push(LuaValue::NIL);
                        }
                    }
                    None => return Err(l.runtime_error(b"attempt to use a closed file")),
                }
            }
            None => {
                let stdin = std::io::stdin();
                let mut lock = stdin.lock();
                for f in fmts {
                    out.push(read_format(l, &mut lock, f)?);
                }
            }
        },
        Some(id) => {
            let files = l.files_mut();
            let entry = files.get_mut(id).and_then(|e| e.as_mut());
            match entry {
                Some(FileEntry::Read(r)) => {
                    for f in fmts {
                        out.push(read_format(l, r, f)?);
                    }
                }
                Some(FileEntry::ReadWrite(file)) => {
                    let mut r = BufReader::new(&mut *file);
                    for f in fmts {
                        out.push(read_format(l, &mut r, f)?);
                    }
                }
                Some(FileEntry::Stdin) => {
                    let stdin = std::io::stdin();
                    let mut lock = stdin.lock();
                    for f in fmts {
                        out.push(read_format(l, &mut lock, f)?);
                    }
                }
                // io.popen("cmd", "r"): read the child's stdout.
                Some(FileEntry::Pipe(child)) => {
                    match child.stdout.as_mut() {
                        Some(stdout) => {
                            let mut r = BufReader::new(stdout);
                            for f in fmts {
                                out.push(read_format(l, &mut r, f)?);
                            }
                        }
                        // Write-mode pipe ("w"): nothing to read.
                        None => {
                            for _ in &fmts {
                                out.push(LuaValue::NIL);
                            }
                        }
                    }
                }
                Some(FileEntry::Write(_) | FileEntry::Stdout | FileEntry::Stderr) => {
                    // LuaJIT: reading a write-mode handle yields nil (EOF).
                    for _ in &fmts {
                        out.push(LuaValue::NIL);
                    }
                }
                None => return Err(l.runtime_error(b"attempt to use a closed file")),
            }
        }
    }
    pushv(l, &out);
    Ok(out.len() as i32)
}

// -- Writing -----------------------------------------------------------------

/// Write `chunks`, then flush per the file's setvbuf mode: 2 ("no") flushes
/// every call, 1 ("line") flushes when a chunk contains a newline, 0
/// ("full") keeps the BufWriter's default (flush on close/flush).
fn write_chunks_buffered(
    w: &mut dyn std::io::Write,
    chunks: &[Vec<u8>],
    mode: u8,
) -> std::io::Result<()> {
    for c in chunks {
        w.write_all(c)?;
    }
    if mode == 2 || (mode == 1 && chunks.iter().any(|c| c.contains(&b'\n'))) {
        w.flush()?;
    }
    Ok(())
}

fn do_write(l: &mut LuaState, fd: Option<usize>, first: usize) -> LuaResult<i32> {
    let n = lua_gettop(l);
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(n.saturating_sub(first));
    for i in first..n {
        let v = arg(l, i);
        if v.as_string_id().is_none() && v.as_number().is_none() {
            return Err(err_bad_arg(l, i as u32 + 1, "write", "string", ""));
        }
        chunks.push(tostring_bytes(l, v));
    }
    let result: std::io::Result<()> = match fd {
        None => match l.global().default_output {
            Some(id) => {
                let mode = l.global().file_buf_mode.get(id).copied().unwrap_or(0);
                let files = l.files_mut();
                match files.get_mut(id).and_then(|e| e.as_mut()) {
                    Some(FileEntry::Write(f)) => write_chunks_buffered(f, &chunks, mode),
                    Some(FileEntry::ReadWrite(f)) => chunks.iter().try_for_each(|c| f.write_all(c)),
                    Some(FileEntry::Stdout) | Some(FileEntry::Stderr) => {
                        let mut so = std::io::stdout();
                        chunks
                            .iter()
                            .try_for_each(|c| so.write_all(c))
                            .and_then(|_| so.flush())
                    }
                    // io.popen("cmd", "w"): write to the child's stdin.
                    Some(FileEntry::Pipe(child)) => match child.stdin.as_mut() {
                        Some(stdin) => chunks.iter().try_for_each(|c| stdin.write_all(c)),
                        None => return ret_fail3(l, "file not opened for writing"),
                    },
                    Some(FileEntry::Read(_) | FileEntry::Stdin) => {
                        // LuaJIT: writing a read-mode handle returns
                        // (nil, msg, errno) instead of raising.
                        return ret_fail3(l, "file not opened for writing");
                    }
                    None => return Err(l.runtime_error(b"attempt to use a closed file")),
                }
            }
            None => {
                let mut so = std::io::stdout();
                chunks
                    .iter()
                    .try_for_each(|c| so.write_all(c))
                    .and_then(|_| so.flush())
            }
        },
        Some(id) => {
            let mode = l.global().file_buf_mode.get(id).copied().unwrap_or(0);
            let files = l.files_mut();
            match files.get_mut(id).and_then(|e| e.as_mut()) {
                Some(FileEntry::Write(f)) => write_chunks_buffered(f, &chunks, mode),
                Some(FileEntry::ReadWrite(f)) => chunks.iter().try_for_each(|c| f.write_all(c)),
                Some(FileEntry::Stdout) => {
                    let mut so = std::io::stdout();
                    chunks
                        .iter()
                        .try_for_each(|c| so.write_all(c))
                        .and_then(|_| so.flush())
                }
                Some(FileEntry::Stderr) => {
                    let mut se = std::io::stderr();
                    chunks
                        .iter()
                        .try_for_each(|c| se.write_all(c))
                        .and_then(|_| se.flush())
                }
                // io.popen("cmd", "w"): write to the child's stdin.
                Some(FileEntry::Pipe(child)) => match child.stdin.as_mut() {
                    Some(stdin) => chunks.iter().try_for_each(|c| stdin.write_all(c)),
                    None => return ret_fail3(l, "file not opened for writing"),
                },
                Some(FileEntry::Read(_) | FileEntry::Stdin) => {
                    // LuaJIT: writing a read-mode handle returns
                    // (nil, msg, errno) instead of raising.
                    return ret_fail3(l, "file not opened for writing");
                }
                None => return Err(l.runtime_error(b"attempt to use a closed file")),
            }
        }
    };
    match result {
        Ok(()) => {
            push(l, LuaValue::TRUE);
            Ok(1)
        }
        Err(e) => {
            let msg = e.to_string();
            ret_fail(l, &msg)
        }
    }
}

// -- Handle methods (self = arg 0) --------------------------------------------

#[allow(dead_code)]
fn handle_read(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => do_read(l, Some(fd), 1),
        None => Err(err_bad_arg_type(l, 1, "read", "file", arg(l, 0))),
    }
}

#[allow(dead_code)]
fn handle_write(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => do_write(l, Some(fd), 1),
        None => Err(err_bad_arg_type(l, 1, "write", "file", arg(l, 0))),
    }
}

fn handle_close(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            let files = l.files_mut();
            match files.get_mut(fd).and_then(|e| e.as_ref()) {
                Some(_) => {
                    files[fd] = None;
                }
                None => return Err(l.runtime_error(b"attempt to use a closed file")),
            }
            push(l, LuaValue::TRUE);
            Ok(1)
        }
        None => Err(err_bad_arg_type(l, 1, "close", "file", arg(l, 0))),
    }
}

/// Iterator state is a one-upvalue closure over the fd (as a number).
fn lines_iter(l: &mut LuaState) -> LuaResult<i32> {
    // The iterator holds the file userdata as its upvalue, so the GC closes
    // the file via __gc when the iterator (and thus the file) is dropped.
    let fd = match l.upvalue(0).as_userdata() {
        Some(u) => match u.as_ref().borrow::<usize>() {
            Some(fd) => *fd,
            None => {
                push(l, LuaValue::NIL);
                return Ok(1);
            }
        },
        None => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };
    // io.lines(filename) opens its own handle and closes it at EOF;
    // f:lines() shares the file's handle and must NOT close it (LuaJIT).
    let close_on_eof = l.upvalue(1).is_true();
    let line = {
        let files = l.files_mut();
        match files.get_mut(fd).and_then(|e| e.as_mut()) {
            Some(FileEntry::Read(r)) => {
                let mut line = Vec::new();
                match r.read_until(b'\n', &mut line) {
                    Ok(0) => None,
                    Ok(_) => {
                        if line.last() == Some(&b'\n') {
                            line.pop();
                        }
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        Some(line)
                    }
                    Err(e) => return Err(l.runtime_error(e.to_string().as_bytes())),
                }
            }
            // Already closed (EOF auto-close or an explicit close): a later
            // iteration is an error, like LuaJIT.
            None => return Err(l.runtime_error(b"attempt to use a closed file")),
            _ => None,
        }
    };
    match line {
        Some(bytes) => ret_string(l, &bytes),
        None => {
            if close_on_eof && let Some(slot) = l.global().files.get_mut(fd) {
                *slot = None;
            }
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn make_lines_iter(l: &mut LuaState, fd: usize, close_on_eof: bool) -> LuaValue {
    let env = l.global().globals;
    // Hold the userdata so the GC can close the file once the iterator is
    // collected (io.lines opens its own handle, not cached).
    let h = new_handle(l, fd);
    let fref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: lines_iter,
        env,
        upvals: vec![h, LuaValue::boolean(close_on_eof)],
    }));
    LuaValue::func(fref)
}

#[allow(dead_code)]
fn handle_lines(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            let it = make_lines_iter(l, fd, false);
            push(l, it);
            Ok(1)
        }
        None => Err(err_bad_arg_type(l, 1, "lines", "file", arg(l, 0))),
    }
}

// -- Library functions ---------------------------------------------------------

fn io_open(l: &mut LuaState) -> LuaResult<i32> {
    let path = String::from_utf8_lossy(str_arg(l, 0, "io.open")?).into_owned();
    let mode = if lua_gettop(l) >= 2 {
        String::from_utf8_lossy(str_arg(l, 1, "io.open")?).into_owned()
    } else {
        "r".to_string()
    };
    let m = mode.trim_end_matches('b');
    let entry = match m {
        "r" => File::open(&path).map(|f| FileEntry::Read(BufReader::new(f))),
        "w" => File::create(&path).map(|f| FileEntry::Write(BufWriter::new(f))),
        "a" => std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map(|f| FileEntry::Write(BufWriter::new(f))),
        "r+" => std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map(FileEntry::ReadWrite),
        "w+" => std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map(FileEntry::ReadWrite),
        "a+" => std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)
            .map(FileEntry::ReadWrite),
        _ => return ret_fail(l, &format!("invalid mode '{}'", mode)),
    };
    match entry {
        Ok(e) => {
            let id = registry_put(l, e);
            let h = new_handle(l, id);
            push(l, h);
            Ok(1)
        }
        Err(e) => ret_fail3(l, &format!("{}: {}", path, e)),
    }
}

fn io_read(l: &mut LuaState) -> LuaResult<i32> {
    do_read(l, None, 0)
}

fn io_write(l: &mut LuaState) -> LuaResult<i32> {
    do_write(l, None, 0)
}

fn io_lines(l: &mut LuaState) -> LuaResult<i32> {
    let (id, close_on_eof) = if lua_gettop(l) == 0 {
        // io.lines() iterates the default input (shared, not closed at EOF).
        match l.global().default_input {
            Some(id) => (id, false),
            None => {
                let id = registry_put(l, FileEntry::Stdin);
                l.global().default_input = Some(id);
                (id, false)
            }
        }
    } else {
        let path = String::from_utf8_lossy(str_arg(l, 0, "io.lines")?).into_owned();
        match File::open(&path) {
            Ok(f) => {
                let id = registry_put(l, FileEntry::Read(BufReader::new(f)));
                (id, true)
            }
            Err(e) => {
                return Err(l.runtime_error(format!("{}: {}", path, e).as_bytes()));
            }
        }
    };
    let it = make_lines_iter(l, id, close_on_eof);
    push(l, it);
    Ok(1)
}

fn io_close(l: &mut LuaState) -> LuaResult<i32> {
    if lua_gettop(l) >= 1 {
        return handle_close(l);
    }
    // Close default output: drop the file handle (BufWriter flushes on
    // drop). Keep `default_output` pointing at the id so a subsequent
    // io.write/io.output fails with "attempt to use a closed file" instead
    // of silently falling back to stdout (Lua 5.1: io.close() leaves the
    // default file closed until io.output is called again).
    let out_id = l.global().default_output;
    if let Some(id) = out_id {
        l.global().files[id] = None;
    }
    push(l, LuaValue::TRUE);
    Ok(1)
}

fn io_flush(l: &mut LuaState) -> LuaResult<i32> {
    let out = l.global().default_output;
    match out {
        Some(id) => {
            let files = l.files_mut();
            match files.get_mut(id).and_then(|e| e.as_mut()) {
                Some(FileEntry::Write(f)) => match f.flush() {
                    Ok(()) => {
                        push(l, LuaValue::TRUE);
                        Ok(1)
                    }
                    Err(e) => ret_fail(l, &e.to_string()),
                },
                Some(FileEntry::ReadWrite(f)) => match f.flush() {
                    Ok(()) => {
                        push(l, LuaValue::TRUE);
                        Ok(1)
                    }
                    Err(e) => ret_fail(l, &e.to_string()),
                },
                Some(FileEntry::Stdout | FileEntry::Stderr) => match std::io::stdout().flush() {
                    Ok(()) => {
                        push(l, LuaValue::TRUE);
                        Ok(1)
                    }
                    Err(e) => ret_fail(l, &e.to_string()),
                },
                _ => Err(l.runtime_error(b"default output not writable")),
            }
        }
        None => match std::io::stdout().flush() {
            Ok(()) => {
                push(l, LuaValue::TRUE);
                Ok(1)
            }
            Err(e) => ret_fail(l, &e.to_string()),
        },
    }
}

fn cache_handle(l: &mut LuaState, id: usize, v: LuaValue) {
    let cache = &mut l.global().io_file_cache;
    if cache.len() <= id {
        cache.resize(id + 1, None);
    }
    cache[id] = Some(v);
}

fn clear_handle_cache(l: &mut LuaState, id: usize) {
    let cache = &mut l.global().io_file_cache;
    if id < cache.len() {
        cache[id] = None;
    }
}

fn io_input(l: &mut LuaState) -> LuaResult<i32> {
    if lua_gettop(l) == 0 {
        let in_id = l.global().default_input;
        match in_id {
            Some(id) => {
                let cached = l.global().io_file_cache.get(id).and_then(|o| *o);
                let h = match cached {
                    Some(v) => v,
                    None => {
                        let h = new_handle(l, id);
                        cache_handle(l, id, h);
                        h
                    }
                };
                push(l, h);
                Ok(1)
            }
            None => {
                let id = registry_put(l, FileEntry::Stdin);
                l.global().default_input = Some(id);
                let h = new_handle(l, id);
                push(l, h);
                Ok(1)
            }
        }
    } else {
        let v = arg(l, 0);
        let id = if let Some(fd) = handle_fd(l, 0) {
            let files = l.files_mut();
            if files.get(fd).and_then(|e| e.as_ref()).is_none() {
                return Err(l.runtime_error(b"attempt to use a closed file"));
            }
            fd
        } else if let Some(s) = v.as_string() {
            let path = String::from_utf8_lossy(s.as_ref().as_bytes()).into_owned();
            match File::open(&path) {
                Ok(f) => registry_put(l, FileEntry::Read(BufReader::new(f))),
                Err(e) => return ret_fail(l, &format!("{}: {}", path, e)),
            }
        } else {
            return Err(err_bad_arg_type(l, 1, "input", "string or file", arg(l, 0)));
        };
        let old = l.global().default_input;
        l.global().default_input = Some(id);
        if let Some(fd) = handle_fd(l, 0) {
            cache_handle(l, fd, arg(l, 0));
        }
        match old {
            Some(old_id) if old_id == id => {
                // Re-selecting the current default input returns the very
                // same file object (Lua 5.1: io.input(io.stdin) == io.stdin).
                push(l, v);
                Ok(1)
            }
            Some(old_id) => {
                // The old default is no longer the default: let its cached
                // handle be collected (and the file closed) if unused.
                clear_handle_cache(l, old_id);
                push(l, v);
                Ok(1)
            }
            None => {
                push(l, v);
                Ok(1)
            }
        }
    }
}

fn io_output(l: &mut LuaState) -> LuaResult<i32> {
    if lua_gettop(l) == 0 {
        let out_id = l.global().default_output;
        match out_id {
            Some(id) => {
                let cached = l.global().io_file_cache.get(id).and_then(|o| *o);
                let h = match cached {
                    Some(v) => v,
                    None => {
                        let h = new_handle(l, id);
                        cache_handle(l, id, h);
                        h
                    }
                };
                push(l, h);
                Ok(1)
            }
            None => {
                let id = registry_put(l, FileEntry::Stdout);
                l.global().default_output = Some(id);
                let h = new_handle(l, id);
                push(l, h);
                Ok(1)
            }
        }
    } else {
        let v = arg(l, 0);
        let id = if let Some(fd) = handle_fd(l, 0) {
            let files = l.files_mut();
            if files.get(fd).and_then(|e| e.as_ref()).is_none() {
                return Err(l.runtime_error(b"attempt to use a closed file"));
            }
            fd
        } else if let Some(s) = v.as_string() {
            let path = String::from_utf8_lossy(s.as_ref().as_bytes()).into_owned();
            match File::create(&path) {
                Ok(f) => registry_put(l, FileEntry::Write(BufWriter::new(f))),
                Err(e) => return ret_fail(l, &format!("{}: {}", path, e)),
            }
        } else {
            return Err(err_bad_arg_type(
                l,
                1,
                "output",
                "string or file",
                arg(l, 0),
            ));
        };
        let old = l.global().default_output;
        l.global().default_output = Some(id);
        if let Some(fd) = handle_fd(l, 0) {
            cache_handle(l, fd, arg(l, 0));
        }
        match old {
            Some(old_id) if old_id == id => {
                push(l, v);
                Ok(1)
            }
            Some(old_id) => {
                clear_handle_cache(l, old_id);
                push(l, v);
                Ok(1)
            }
            None => {
                push(l, v);
                Ok(1)
            }
        }
    }
}

fn io_type(l: &mut LuaState) -> LuaResult<i32> {
    if lua_gettop(l) == 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let v = arg(l, 0);
    if let Some(u) = v.as_userdata() {
        let fd = match u.as_ref().borrow::<usize>() {
            Some(fd) => *fd,
            None => {
                push(l, LuaValue::NIL);
                return Ok(1);
            }
        };
        if l.files_mut().get(fd).and_then(|e| e.as_ref()).is_some() {
            let sid = l.heap().intern(b"file");
            push(l, l.heap().str_value(sid));
        } else {
            let sid = l.heap().intern(b"closed file");
            push(l, l.heap().str_value(sid));
        }
        return Ok(1);
    }
    push(l, LuaValue::NIL);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    // Register the standard streams first: their ids must be stable.
    let fdi = registry_put(l, FileEntry::Stdin);
    let fdo = registry_put(l, FileEntry::Stdout);
    let fde = registry_put(l, FileEntry::Stderr);
    let io_tab = lual_reg!(l, b"io", LibTarget::Global)
        .func(b"open", io_open)
        .func(b"popen", io_popen)
        .func(b"tmpfile", io_tmpfile)
        .func(b"read", io_read)
        .func(b"write", io_write)
        .func(b"lines", io_lines)
        .func(b"close", io_close)
        .func(b"flush", io_flush)
        .func(b"input", io_input)
        .func(b"output", io_output)
        .func(b"type", io_type)
        .build();
    for (name, fd) in [
        (b"stdin".as_slice(), fdi),
        (b"stdout", fdo),
        (b"stderr", fde),
    ] {
        let h = new_handle(l, fd);
        let k = l.heap().str_value(l.heap().intern(name));
        io_tab.as_mut().set(k, h);
    }
    // Lua 5.1: the default input/output start as stdin/stdout, so
    // `io.input(io.stdin) == io.stdin` holds and `io.input()` returns
    // the standard stream.
    l.global().default_input = Some(fdi);
    l.global().default_output = Some(fdo);
}

fn io_popen(l: &mut LuaState) -> LuaResult<i32> {
    let fname = str_arg(l, 0, "io.popen")?;
    let mode = if lua_gettop(l) >= 2 {
        String::from_utf8_lossy(str_arg(l, 1, "io.popen")?).into_owned()
    } else {
        "r".to_string()
    };
    let m = mode.trim_end_matches('b');
    let mut cmd = std::process::Command::new(shell_for_popen());
    cmd.arg(shell_flag_for_popen())
        .arg(String::from_utf8_lossy(fname).into_owned());
    match m {
        // "r": capture the child's stdout on the handle.
        "r" => cmd
            .stdout(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null()),
        // "w": the handle feeds the child's stdin.
        "w" => cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null()),
        _ => return ret_fail(l, &format!("invalid mode '{}'", mode)),
    };
    match cmd.spawn() {
        Ok(child) => {
            let id = registry_put(l, FileEntry::Pipe(child));
            let h = new_handle(l, id);
            push(l, h);
            Ok(1)
        }
        Err(e) => ret_fail3(l, &format!("{}: {}", String::from_utf8_lossy(fname), e)),
    }
}

/// The shell `io.popen` runs its command through. Windows uses `cmd /C`,
/// everything else uses `sh -c` (LuaJIT's `popen`/`_popen`).
#[cfg(target_os = "windows")]
fn shell_for_popen() -> &'static str {
    "cmd"
}
#[cfg(not(target_os = "windows"))]
fn shell_for_popen() -> &'static str {
    "sh"
}

#[cfg(target_os = "windows")]
fn shell_flag_for_popen() -> &'static str {
    "/C"
}
#[cfg(not(target_os = "windows"))]
fn shell_flag_for_popen() -> &'static str {
    "-c"
}

fn io_tmpfile(l: &mut LuaState) -> LuaResult<i32> {
    let tmp = std::env::temp_dir();
    let path = tmp.join(format!("luajit_rs_{}_{}.tmp", std::process::id(), l.base));
    // Lua 5.1 io.tmpfile uses tmpfile() (mode "w+b"): read-write.
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
    {
        Ok(f) => {
            let id = registry_put(l, FileEntry::ReadWrite(f));
            let h = new_handle(l, id);
            push(l, h);
            Ok(1)
        }
        Err(e) => Err(l.runtime_error(format!("tmpfile: {}", e).as_bytes())),
    }
}
