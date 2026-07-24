#![allow(clippy::type_complexity)]
//! Native code generation backends, behind a unified architecture-agnostic API.
//! Both backends are always compiled so cross-platform codegen testing works.
//!
//! * `assemble(tr, link_target, arch)` – translate an IR trace into native
//!   `McodeArea` for the selected architecture.
//! * `patch_exit(area, tails, exitno, target, arch)` – retarget an exit stub.

use super::{GCtrace, TraceError, mcode::McodeArea};

mod arm64;
mod x64;

/// Target architecture for native code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X64,
    Arm64,
}

/// The native arch for the current compilation target.
pub const HOST_ARCH: Arch = if cfg!(target_arch = "aarch64") {
    Arch::Arm64
} else {
    Arch::X64
};

/// Assemble a completed trace for `arch`. On error the caller keeps
/// `mcode = None` and the portable executor runs the trace.
pub fn assemble(
    tr: &GCtrace,
    link: Option<*const u8>,
    arch: Arch,
) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
    match arch {
        Arch::X64 => x64::assemble(tr, link),
        Arch::Arm64 => arm64::assemble(tr, link),
    }
}

/// Retarget every exit stub of `exitno` to jump to `target` (a side
/// trace's inner entry), for the given architecture.
pub fn patch_exit(
    area: &mut McodeArea,
    stub_tails: &[(u32, u32)],
    exitno: u32,
    target: *const u8,
    arch: Arch,
) {
    match arch {
        Arch::X64 => x64::patch_exit(area, stub_tails, exitno, target),
        Arch::Arm64 => arm64::patch_exit(area, stub_tails, exitno, target),
    }
}
