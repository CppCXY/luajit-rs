//! `io` library subset. Since the runtime has no userdata, a file
//! handle is a plain table `{ __fd = id, read = ..., write = ...,
//! lines = ..., close = ... }`; the id indexes a process-wide file
//! registry (files are OS resources shared across VMs).

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::Mutex;

use crate::err::{LuaError, LuaResult};
use crate::func::{CClosure, GcFunc};
use crate::state::LuaState;
use crate::table::LuaTable;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, nargs, push, pushv, tostring_bytes};
use crate::lual_reg;

enum Entry {
    Read(BufReader<File>),
    Write(File),
}

static FILES: Mutex<Vec<Option<Entry>>> = Mutex::new(Vec::new());

fn registry_put(e: Entry) -> usize {
    let mut files = FILES.lock().unwrap();
    for (i, slot) in files.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(e);
            return i;
        }
    }
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

// -- Handle tables -----------------------------------------------------------

fn handle_fd(l: &LuaState, i: usize) -> Option<usize> {
    let t = arg(l, i).as_table()?;
    let sid = l.heap().intern(b"__fd");
    let k = l.heap().str_value(sid);
    let n = t.as_ref().get_str(k).as_number()?;
    Some(n as usize)
}

/// Build a file-handle table for a registered file id.
fn new_handle(l: &mut LuaState, id: usize) -> LuaValue {
    let t = l.heap().alloc_table(LuaTable::new(0, 3));
    let entries: [(&[u8], crate::func::CFunction); 4] = [
        (b"read", handle_read),
        (b"write", handle_write),
        (b"lines", handle_lines),
        (b"close", handle_close),
    ];
    let env = l.global().globals;
    for (name, f) in entries {
        let sid = l.heap().intern(name);
        let k = l.heap().str_value(sid);
        let fref = l.heap().alloc_func(GcFunc::C(CClosure {
            f,
            env,
            upvals: Vec::new(),
        }));
        t.as_mut().set(k, LuaValue::func(fref));
    }
    let fd_sid = l.heap().intern(b"__fd");
    let fd_k = l.heap().str_value(fd_sid);
    t.as_mut().set(fd_k, LuaValue::number(id as f64));

    // Set __tostring metatable so the handle can be printed.
    let mt = l.heap().alloc_table(LuaTable::new(0, 2));
    let ts_ref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: handle_tostring,
        env,
        upvals: Vec::new(),
    }));
    let ts_key = l.heap().str_value(l.heap().intern(b"__tostring"));
    mt.as_mut().set(ts_key, LuaValue::func(ts_ref));
    t.as_mut().metatable = Some(mt);

    LuaValue::table(t)
}

fn handle_tostring(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            let s = format!("file ({:#x})", fd);
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
    let n = nargs(l);
    let mut fmts: Vec<LuaValue> = (first_fmt..n.max(first_fmt)).map(|i| arg(l, i)).collect();
    if fmts.is_empty() {
        fmts.push(LuaValue::NIL); // Default: one line.
    }
    let mut out = Vec::with_capacity(fmts.len());
    match fd {
        None => {
            let stdin = std::io::stdin();
            let mut lock = stdin.lock();
            for f in fmts {
                out.push(read_format(l, &mut lock, f)?);
            }
        }
        Some(id) => {
            let mut files = FILES.lock().unwrap();
            let entry = files.get_mut(id).and_then(|e| e.as_mut());
            match entry {
                Some(Entry::Read(r)) => {
                    // The registry lock is process-wide; nothing inside
                    // read_format touches FILES again.
                    for f in fmts {
                        out.push(read_format(l, r, f)?);
                    }
                }
                Some(Entry::Write(_)) => {
                    return Err(l.runtime_error(b"file not opened for reading"));
                }
                None => return Err(l.runtime_error(b"attempt to use a closed file")),
            }
        }
    }
    pushv(l, &out);
    Ok(out.len() as i32)
}

// -- Writing -----------------------------------------------------------------

fn do_write(l: &mut LuaState, fd: Option<usize>, first: usize) -> LuaResult<i32> {
    let n = nargs(l);
    let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(n.saturating_sub(first));
    for i in first..n {
        let v = arg(l, i);
        if v.as_string_id().is_none() && v.as_number().is_none() {
            return Err(err_bad_arg(l, i as u32 + 1, "write", "string", ""));
        }
        chunks.push(tostring_bytes(l, v));
    }
    let result: std::io::Result<()> = match fd {
        None => {
            let mut so = std::io::stdout();
            chunks
                .iter()
                .try_for_each(|c| so.write_all(c))
                .and_then(|_| so.flush())
        }
        Some(id) => {
            let mut files = FILES.lock().unwrap();
            match files.get_mut(id).and_then(|e| e.as_mut()) {
                Some(Entry::Write(f)) => chunks.iter().try_for_each(|c| f.write_all(c)),
                Some(Entry::Read(_)) => {
                    return Err(l.runtime_error(b"file not opened for writing"));
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

fn handle_read(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => do_read(l, Some(fd), 1),
        None => Err(err_bad_arg(l, 1, "read", "file", "")),
    }
}

fn handle_write(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => do_write(l, Some(fd), 1),
        None => Err(err_bad_arg(l, 1, "write", "file", "")),
    }
}

fn handle_close(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            if let Some(slot) = FILES.lock().unwrap().get_mut(fd) {
                *slot = None;
            }
            push(l, LuaValue::TRUE);
            Ok(1)
        }
        None => Err(err_bad_arg(l, 1, "close", "file", "")),
    }
}

/// Iterator state is a one-upvalue closure over the fd (as a number).
fn lines_iter(l: &mut LuaState) -> LuaResult<i32> {
    let fd = l.upvalue(0).as_number().unwrap_or(-1.0) as i64;
    if fd < 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let line = {
        let mut files = FILES.lock().unwrap();
        match files.get_mut(fd as usize).and_then(|e| e.as_mut()) {
            Some(Entry::Read(r)) => {
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
            _ => None,
        }
    };
    match line {
        Some(bytes) => ret_string(l, &bytes),
        None => {
            // Auto-close at EOF (io.lines semantics).
            if let Some(slot) = FILES.lock().unwrap().get_mut(fd as usize) {
                *slot = None;
            }
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn make_lines_iter(l: &mut LuaState, fd: usize) -> LuaValue {
    let env = l.global().globals;
    let fref = l.heap().alloc_func(GcFunc::C(CClosure {
        f: lines_iter,
        env,
        upvals: vec![LuaValue::number(fd as f64)],
    }));
    LuaValue::func(fref)
}

fn handle_lines(l: &mut LuaState) -> LuaResult<i32> {
    match handle_fd(l, 0) {
        Some(fd) => {
            let it = make_lines_iter(l, fd);
            push(l, it);
            Ok(1)
        }
        None => Err(err_bad_arg(l, 1, "lines", "file", "")),
    }
}

// -- Library functions ---------------------------------------------------------

fn io_open(l: &mut LuaState) -> LuaResult<i32> {
    let path = String::from_utf8_lossy(str_arg(l, 0, "io.open")?).into_owned();
    let mode = if nargs(l) >= 2 {
        String::from_utf8_lossy(str_arg(l, 1, "io.open")?).into_owned()
    } else {
        "r".to_string()
    };
    let m = mode.trim_end_matches('b');
    let entry = match m {
        "r" => File::open(&path).map(|f| Entry::Read(BufReader::new(f))),
        "w" => File::create(&path).map(Entry::Write),
        "a" => std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&path)
            .map(Entry::Write),
        _ => return ret_fail(l, &format!("invalid mode '{}'", mode)),
    };
    match entry {
        Ok(e) => {
            let id = registry_put(e);
            let h = new_handle(l, id);
            push(l, h);
            Ok(1)
        }
        Err(e) => ret_fail(l, &format!("{}: {}", path, e)),
    }
}

fn io_read(l: &mut LuaState) -> LuaResult<i32> {
    do_read(l, None, 0)
}

fn io_write(l: &mut LuaState) -> LuaResult<i32> {
    do_write(l, None, 0)
}

fn io_lines(l: &mut LuaState) -> LuaResult<i32> {
    if nargs(l) == 0 {
        return Err(l.runtime_error(b"io.lines() without a filename is not supported"));
    }
    let path = String::from_utf8_lossy(str_arg(l, 0, "io.lines")?).into_owned();
    match File::open(&path) {
        Ok(f) => {
            let id = registry_put(Entry::Read(BufReader::new(f)));
            let it = make_lines_iter(l, id);
            push(l, it);
            Ok(1)
        }
        Err(e) => Err(l.runtime_error(format!("{}: {}", path, e).as_bytes())),
    }
}

fn io_close(l: &mut LuaState) -> LuaResult<i32> {
    if nargs(l) >= 1 {
        return handle_close(l);
    }
    push(l, LuaValue::TRUE);
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"io", LibTarget::Global)
        .func(b"open", io_open)
        .func(b"read", io_read)
        .func(b"write", io_write)
        .func(b"lines", io_lines)
        .func(b"close", io_close)
        .build();
}
