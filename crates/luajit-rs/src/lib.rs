mod api;
mod compiler;
mod ffi;
mod jit;
mod runtime;
mod stdlib;
mod util;
mod vm;

use compiler::{bc, dump, lex};
use runtime::{func, gc, meta, proto, state, string, table, value};
use util::{strfmt, strscan};
use vm::err;

// export the public API
pub use api::*;
pub use runtime::state::{Lua, LuaState};
pub use stdlib::reg::{LibBuilder, LibTarget};
pub use vm::err::{LuaError, LuaResult};

pub mod internal {
    pub use crate::compiler::*;
    pub use crate::ffi::*;
    pub use crate::jit::*;
    pub use crate::runtime::*;
    pub use crate::vm::{call, execute};
}
