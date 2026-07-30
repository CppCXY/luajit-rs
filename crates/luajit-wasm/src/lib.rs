use luajit_rs::{
    Lua, LuaState, lua_error_message, lua_gc, lua_getglobal, lua_gettop, lua_isfunction, lua_isnil,
    lua_newtable, lua_pcall, lua_peek, lua_pop, lua_pushboolean, lua_pushcfunction, lua_pushnil,
    lua_pushnumber, lua_pushstring, lua_rawseti, lua_setglobal, lua_settable, lua_settop,
    lual_loadstring, lual_openlibs,
};
use std::cell::RefCell;
use std::ptr;
use wasm_bindgen::prelude::*;

static mut BRIDGE_CB: *const RefCell<Option<js_sys::Function>> = ptr::null();

#[wasm_bindgen]
pub struct LuaWasm {
    inner: RefCell<Lua>,
    print_cb: RefCell<Option<js_sys::Function>>,
}

impl Default for LuaWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl LuaWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> LuaWasm {
        let mut lua = Lua::new();
        lual_openlibs(lua.main());
        LuaWasm {
            inner: RefCell::new(lua),
            print_cb: RefCell::new(None),
        }
    }

    pub fn on_print(&self, cb: JsValue) {
        let mut lua = self.inner.borrow_mut();
        let l = lua.main();
        if cb.is_null() || cb.is_undefined() {
            *self.print_cb.borrow_mut() = None;
            lua_getglobal(l, "__orig_print");
            if !lua_isnil(l, -1) {
                lua_setglobal(l, "print");
            } else {
                lua_pop(l, 1);
            }
            return;
        }
        let func = cb.unchecked_into::<js_sys::Function>();
        lua_getglobal(l, "print");
        if !lua_isnil(l, -1) && lua_isfunction(l, -1) {
            lua_setglobal(l, "__orig_print");
        } else {
            lua_pop(l, 1);
        }
        *self.print_cb.borrow_mut() = Some(func);
        unsafe {
            BRIDGE_CB = &self.print_cb;
        }
        lua_pushcfunction(l, print_bridge);
        lua_setglobal(l, "print");
    }

    pub fn do_string(&self, src: &str) -> Result<String, JsValue> {
        let mut lua = self.inner.borrow_mut();
        let l = lua.main();
        lua_settop(l, 0);
        lual_loadstring(l, src.as_bytes()).map_err(|_| js_error(l))?;
        lua_pcall(l, 0, -1, 0).map_err(|_| js_error(l))?;
        let n = lua_gettop(l);
        let result = Ok(if n == 0 {
            String::new()
        } else {
            value_to_literal(l, 1)
        });
        lua_settop(l, 0);
        result
    }

    pub fn call(&self, name: &str, args: js_sys::Array) -> Result<String, JsValue> {
        let mut lua = self.inner.borrow_mut();
        let l = lua.main();
        lua_settop(l, 0);
        lua_getglobal(l, name);
        if !lua_isfunction(l, -1) {
            lua_pop(l, 1);
            lua_settop(l, 0);
            return Err(JsValue::from_str(&format!("'{}' is not a function", name)));
        }
        let nargs = push_js_array(l, &args);
        lua_pcall(l, nargs, 1, 0).map_err(|_| js_error(l))?;
        let n = lua_gettop(l);
        let result = Ok(if n == 0 {
            String::new()
        } else {
            value_to_literal(l, 1)
        });
        lua_settop(l, 0);
        result
    }

    pub fn set_global(&self, name: &str, val: JsValue) -> Result<(), JsValue> {
        let mut lua = self.inner.borrow_mut();
        let l = lua.main();
        push_js_value(l, &val);
        lua_setglobal(l, name);
        Ok(())
    }

    pub fn get_global(&self, name: &str) -> Result<String, JsValue> {
        let mut lua = self.inner.borrow_mut();
        let l = lua.main();
        lua_getglobal(l, name);
        let result = value_to_literal(l, lua_gettop(l) as i32);
        lua_pop(l, 1);
        Ok(result)
    }

    pub fn gc_collect(&self) {
        let mut lua = self.inner.borrow_mut();
        lua_gc(lua.main(), 2, 0);
    }

    pub fn gc_count(&self) -> f64 {
        let mut lua = self.inner.borrow_mut();
        lua_gc(lua.main(), 5, 0) as f64
    }
}

fn print_bridge(l: &mut LuaState) -> luajit_rs::LuaResult<i32> {
    let n = lua_gettop(l);
    let mut parts = Vec::new();
    for i in 1..=n as i32 {
        parts.push(value_to_literal(l, i));
    }
    let line = parts.join("\t");
    let cb_ptr = unsafe { BRIDGE_CB };
    if !cb_ptr.is_null()
        && let Some(ref cb) = *unsafe { &*cb_ptr }.borrow()
    {
        let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(&line));
    }
    Ok(0)
}

fn value_to_literal(l: &LuaState, idx: i32) -> String {
    let v = lua_peek(l, idx);
    if v.is_nil() {
        "null".to_string()
    } else if v.is_bool() {
        (if v.is_true() { "true" } else { "false" }).to_string()
    } else if let Some(n) = v.as_number() {
        if n == n.trunc() && n.is_finite() && n >= i64::MIN as f64 && n <= i64::MAX as f64 {
            format!("{}", n as i64)
        } else {
            format!("{}", n)
        }
    } else if let Some(s) = v.as_string() {
        std::str::from_utf8(s.as_ref().as_bytes())
            .unwrap_or("")
            .to_string()
    } else {
        "null".to_string()
    }
}

fn push_js_array(l: &mut LuaState, arr: &js_sys::Array) -> i32 {
    let mut count = 0i32;
    for i in 0..arr.length() {
        push_js_value(l, &arr.get(i));
        count += 1;
    }
    count
}

fn push_js_value(l: &mut LuaState, val: &JsValue) {
    if val.is_null() || val.is_undefined() {
        lua_pushnil(l);
    } else if let Some(b) = val.as_bool() {
        lua_pushboolean(l, b);
    } else if let Some(n) = val.as_f64() {
        lua_pushnumber(l, n);
    } else if let Some(s) = val.as_string() {
        lua_pushstring(l, s.as_bytes());
    } else if js_sys::Array::is_array(val) {
        let arr = js_sys::Array::from(val);
        lua_newtable(l);
        for i in 0..arr.length() {
            push_js_value(l, &arr.get(i));
            lua_rawseti(l, -2, i as i32 + 1);
        }
    } else if val.is_object() {
        lua_newtable(l);
        let keys = js_sys::Object::keys(&val.clone().into());
        for i in 0..keys.length() {
            let key = keys.get(i);
            if let Some(k) = key.as_string() {
                lua_pushstring(l, k.as_bytes());
                let prop = js_sys::Reflect::get(val, &key).unwrap_or(JsValue::NULL);
                push_js_value(l, &prop);
                lua_settable(l, -3);
            }
        }
    } else {
        lua_pushnil(l);
    }
}

fn js_error(l: &LuaState) -> JsValue {
    JsValue::from_str(&lua_error_message(l))
}
