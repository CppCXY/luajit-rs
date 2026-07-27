//! Internal APIs for advanced use cases (stack manipulation, direct VM
//! calls, raw GC access).  Most applications should use the higher-level
//! functions in `crate::api`.
//!
//! All types are also accessible via their canonical module paths
//! (`crate::state`, `crate::value`, etc.) — this module serves as a
//! convenient single-import point.

pub use crate::func::{CClosure, CFunction, GcFunc};
pub use crate::gc::{GcPtr, Pool};
pub use crate::state::{self, GlobalState, Lua as RawLua, LuaState};
pub use crate::stdlib::arg;
pub use crate::stdlib::push;
pub use crate::string::LuaString;
pub use crate::table::LuaTable;
pub use crate::value::LuaValue;
pub use crate::vm::call;
pub use crate::vm::err::{LuaError, LuaResult};
