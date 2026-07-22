//! `jit.*` library — LuaJIT-compatible JIT compiler control.
//!
//! Provides `jit.on()`, `jit.off()`, `jit.flush()`, and `jit.status()`
//! for runtime control of trace compilation from Lua code.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, push};
use crate::lual_reg;

fn jit_on(l: &mut LuaState) -> LuaResult<i32> {
    l.global().jit.set_on(true);
    Ok(0)
}

fn jit_off(l: &mut LuaState) -> LuaResult<i32> {
    l.global().jit.set_on(false);
    Ok(0)
}

fn jit_flush(l: &mut LuaState) -> LuaResult<i32> {
    let g = l.global();
    g.jit.set_on(false);
    for slot in g.jit.trace.iter_mut() {
        *slot = None;
    }
    g.jit.set_on(true);
    Ok(0)
}

fn jit_status(l: &mut LuaState) -> LuaResult<i32> {
    let on = l.global().jit.is_on();
    push(l, LuaValue::boolean(on));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    let version_str = l.heap().intern(b"LuaJIT 2.1.0-rs");
    let version_val = l.heap().str_value(version_str);
    let arch_str = l.heap().intern(if cfg!(target_arch = "x86_64") {
        b"x64"
    } else if cfg!(target_arch = "aarch64") {
        b"arm64"
    } else {
        b"unknown"
    });
    let arch_val = l.heap().str_value(arch_str);
    let os_str = l.heap().intern(if cfg!(windows) {
        b"Windows"
    } else if cfg!(target_os = "macos") {
        b"OSX"
    } else if cfg!(target_os = "linux") {
        b"Linux"
    } else {
        b"Other"
    });
    let os_val = l.heap().str_value(os_str);
    lual_reg!(l, b"jit", LibTarget::Global)
        .func(b"on", jit_on)
        .func(b"off", jit_off)
        .func(b"flush", jit_flush)
        .func(b"status", jit_status)
        .constant(b"version", version_val)
        .constant(b"version_num", LuaValue::number(20100.0))
        .constant(b"arch", arch_val)
        .constant(b"os", os_val)
        .build();
}
