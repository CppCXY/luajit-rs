//! Native code generation backends, behind a unified architecture-
//! agnostic API. Every target exports the same two entry points:
//!
//! * `assemble(tr, link_target)` – translate an IR trace into a native
//!   `McodeArea`, returning the inner-entry offset and patchable tail
//!   positions.
//! * `patch_exit(area, tails, exitno, target)` – retarget an exit stub
//!   directly to a compiled side-trace's inner entry.
//!
//! Traces the backend cannot handle (e.g. an unsupported link type) fail
//! with `NYIIR` and fall back to the portable IR executor.

#[cfg(target_arch = "x86_64")]
mod x64;
#[cfg(target_arch = "x86_64")]
pub use x64::{assemble, patch_exit};

#[cfg(target_arch = "aarch64")]
mod arm64;
#[cfg(target_arch = "aarch64")]
pub use arm64::{assemble, patch_exit};

/// Fallback for unsupported architectures: every trace runs on the
/// portable IR executor.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod stub {
    use super::super::super::{GCtrace, TraceError, mcode::McodeArea};

    pub fn assemble(
        _tr: &GCtrace,
        _link: Option<*const u8>,
    ) -> Result<(McodeArea, u32, Vec<(u32, u32)>), TraceError> {
        Err(TraceError::NYIIR)
    }

    pub fn patch_exit(
        _area: &mut McodeArea,
        _tails: &[(u32, u32)],
        _exitno: u32,
        _target: *const u8,
    ) {
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use stub::{assemble, patch_exit};
