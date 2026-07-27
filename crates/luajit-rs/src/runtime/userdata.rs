//! GC-managed userdata objects — Lua userdata wrapping Rust types.
//!
//! Every userdata value holds a `Box<dyn Any>` and an optional metatable.
//! The type-id is recovered via `Any::type_id()` for type-safe downcasting.

use std::any::Any;

use crate::gc::GcPtr;
use crate::table::LuaTable;

pub struct GcUserData {
    pub metatable: Option<GcPtr<LuaTable>>,
    pub inner: Box<dyn Any>,
}

impl GcUserData {
    pub fn new<T: Any>(val: T) -> Self {
        GcUserData {
            metatable: None,
            inner: Box::new(val),
        }
    }

    pub fn with_metatable<T: Any>(val: T, mt: GcPtr<LuaTable>) -> Self {
        GcUserData {
            metatable: Some(mt),
            inner: Box::new(val),
        }
    }

    pub fn type_id(&self) -> std::any::TypeId {
        (*self.inner).type_id()
    }

    pub fn is<T: Any>(&self) -> bool {
        self.type_id() == std::any::TypeId::of::<T>()
    }

    pub fn borrow<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }

    pub fn borrow_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.inner.downcast_mut::<T>()
    }
}
