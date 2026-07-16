//! The runtime error / control-flow model.
//!
//! Errors are propagated with Rust's `Result`, but the error type is a
//! fieldless enum so `LuaResult<T>` stays register-sized (8 bytes for
//! `T = i32`). The actual error object and yield count live on the
//! `LuaState`, matching how LuaJIT keeps them off the fast path.
//!
//! This is deliberately *not* stack unwinding: a future JIT emits machine
//! code whose call sites can check a status register, but cannot host a Rust
//! `panic` unwinder without hand-written unwind tables. The `VmStatus` type
//! is the layout-stable form used at that JIT <-> runtime boundary.

/// A runtime control-flow outcome. Carries no data; details are on the
/// `LuaState` (`errval` / `nyield`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LuaError {
    /// A raised error. The error object is in `LuaState::errval`.
    Runtime = 1,
    /// A `coroutine.yield`. The result count is in `LuaState::nyield` and the
    /// values are on the stack.
    Yield = 2,
}

/// The result type threaded through the interpreter and calls. For an `i32`
/// payload this is a single 8-byte value (asserted below).
pub type LuaResult<T> = Result<T, LuaError>;

const _: () = assert!(std::mem::size_of::<LuaResult<i32>>() == 8);
const _: () = assert!(std::mem::size_of::<LuaError>() == 1);

/// Layout-stable status word for the JIT <-> runtime ABI.
///
/// `0` means success; a non-zero low byte is the `LuaError` discriminant.
/// The high 32 bits can carry a small `i32` payload (e.g. a result count),
/// so a helper can return `LuaResult<i32>` losslessly across an `extern "C"`
/// boundary that must not depend on Rust's `Result` layout.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VmStatus(u64);

impl VmStatus {
    pub const OK: VmStatus = VmStatus(0);

    pub fn is_ok(self) -> bool {
        (self.0 & 0xff) == 0
    }

    pub fn error(self) -> Option<LuaError> {
        match self.0 & 0xff {
            0 => None,
            1 => Some(LuaError::Runtime),
            2 => Some(LuaError::Yield),
            _ => Some(LuaError::Runtime),
        }
    }

    pub fn payload(self) -> i32 {
        (self.0 >> 32) as i32
    }
}

impl From<LuaResult<i32>> for VmStatus {
    fn from(r: LuaResult<i32>) -> VmStatus {
        match r {
            Ok(n) => VmStatus((n as u32 as u64) << 32),
            Err(e) => VmStatus(e as u64),
        }
    }
}

impl From<VmStatus> for LuaResult<i32> {
    fn from(s: VmStatus) -> LuaResult<i32> {
        match s.error() {
            None => Ok(s.payload()),
            Some(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmstatus_roundtrips_results() {
        for r in [Ok(0), Ok(3), Ok(-1), Err(LuaError::Runtime), Err(LuaError::Yield)] {
            let s = VmStatus::from(r);
            let back: LuaResult<i32> = s.into();
            assert_eq!(back, r);
        }
        assert!(VmStatus::from(Ok::<i32, LuaError>(5)).is_ok());
        assert_eq!(VmStatus::from(Ok::<i32, LuaError>(5)).payload(), 5);
    }
}
