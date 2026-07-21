//! Lua FFI library — loaded via `require("ffi")`.
//!
//! Exposes the `ffi` global table with `cdef`, `new`, `sizeof`, `cast`, etc.
//! Also sets up the `ffi.C` namespace with lazy C symbol resolution and a
//! generic variadic call wrapper.

use std::ffi::CString;

use crate::err::{LuaError, LuaResult};
use crate::ffi::clib;
use crate::ffi::parser::parse;
use crate::ffi::{
    CT, CTState, CType, CTypeID, ct_info, ctype_align, ctype_cid, ctype_isnum, ctype_isptr,
};
use crate::func::{CClosure, CFunction, GcFunc};
use crate::meta::MM;
use crate::runtime::cdata::CData;
use crate::state::{GlobalState, LuaState};
use crate::stdlib::{arg, err_bad_arg, nargs, push};
use crate::table::LuaTable;
use crate::value::{LJ_TCDATA, LuaValue};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cts_of(l: &mut LuaState) -> &mut CTState {
    l.global().cts.get_or_insert_with(CTState::new)
}

/// Known C type names → predefined type IDs.
fn quick_type_id(name: &str) -> Option<u32> {
    Some(match name {
        "void" => CTypeID::Void as u32,
        "bool" | "_Bool" => CTypeID::Bool as u32,
        "char" => CTypeID::CChar as u32,
        "signed char" | "int8_t" => CTypeID::Int8 as u32,
        "unsigned char" | "uint8_t" => CTypeID::UInt8 as u32,
        "short" | "int16_t" => CTypeID::Int16 as u32,
        "unsigned short" | "uint16_t" => CTypeID::UInt16 as u32,
        "int" | "signed" | "int32_t" => CTypeID::Int32 as u32,
        "unsigned" | "unsigned int" | "uint32_t" => CTypeID::UInt32 as u32,
        "long" | "int64_t" => CTypeID::Int64 as u32,
        "unsigned long" | "uint64_t" => CTypeID::UInt64 as u32,
        "long long" => CTypeID::Int64 as u32,
        "unsigned long long" => CTypeID::UInt64 as u32,
        "float" => CTypeID::Float as u32,
        "double" => CTypeID::Double as u32,
        "void *" | "void*" => CTypeID::PVoid as u32,
        _ => return None,
    })
}

/// On-demand pointer type creation: returns a new `CT::Ptr` → `pointee_id`.
fn make_ptr_type(cts: &mut CTState, pointee_id: u32) -> u32 {
    let ptr_id = cts.top;
    let info = ct_info(CT::Ptr, 3 << 16) | pointee_id; // 8-byte alignment
    cts.tab.push(CType {
        info,
        size: 8,
        sib: 0,
        next: 0,
        name: 0,
    });
    cts.top += 1;
    ptr_id
}

/// Resolve a Lua string / cdata argument to a C type ID.
/// Handles `"type"`, `"type*"`, `"type[N]"`, and `"type[?]"` syntax.
fn check_ctype(l: &mut LuaState) -> LuaResult<u32> {
    let val = arg(l, 0);

    if val.is_cdata() {
        return Ok(val.as_cdata().unwrap().as_ref().ctypeid);
    }

    let sid = match val.as_string_id() {
        Some(s) => s,
        _ => return Err(err_bad_arg(l, 1, "ffi", "C type", "")),
    };

    let raw = l.heap().strings.get(sid).to_vec();
    let raw_str = std::str::from_utf8(&raw).map_err(|_| LuaError::Runtime)?;
    let name = raw_str.trim().to_string();

    // First try the full name including pointer/array suffixes.
    if let Some(id) = quick_type_id(&name) {
        return Ok(id);
    }
    if let Some(&id) = l.global().cts.as_ref().and_then(|c| c.names.get(&name)) {
        return Ok(id);
    }

    // Strip `[...]` suffix (VLA or fixed-size array).
    let base = name
        .find('[')
        .map(|i| name[..i].trim().to_string())
        .unwrap_or_else(|| name.clone());

    // Strip `*` suffix for pointer to custom types.
    let (base, is_ptr) = if let Some(s) = base.strip_suffix('*') {
        (s.trim().to_string(), true)
    } else if let Some(s) = base.strip_suffix(" *") {
        (s.trim().to_string(), true)
    } else {
        (base, false)
    };

    if let Some(id) = quick_type_id(&base) {
        return Ok(if is_ptr {
            make_ptr_type(cts_of(l), id)
        } else {
            id
        });
    }

    if let Some(&id) = l.global().cts.as_ref().and_then(|c| c.names.get(&base)) {
        return Ok(if is_ptr {
            make_ptr_type(cts_of(l), id)
        } else {
            id
        });
    }

    let cts = cts_of(l);
    let prev_top = cts.top;
    parse(cts, &base).map_err(|_| LuaError::Runtime)?;
    if cts.top > prev_top {
        Ok(cts.top - 1)
    } else {
        Err(err_bad_arg(l, 1, "ffi", "C type", ""))
    }
}

// ---------------------------------------------------------------------------
// ffi table functions
// ---------------------------------------------------------------------------

pub fn ffi_cdef(l: &mut LuaState) -> LuaResult<i32> {
    let sid = arg(l, 0)
        .as_string_id()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.cdef", "string", ""))?;
    let src = l.heap().strings.get(sid).to_vec();
    let text = std::str::from_utf8(&src).map_err(|_| LuaError::Runtime)?;
    parse(cts_of(l), text).map_err(|_| LuaError::Runtime)?;
    Ok(0)
}

pub fn ffi_new(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let ct = cts_of(l).raw(id);
    let size = if ct.size != u32::MAX {
        ct.size as usize
    } else {
        0
    };
    let mut cd = CData::new(id, size.max(1));

    if nargs(l) > 1 {
        let v2 = arg(l, 1);
        if let Some(tab) = v2.as_table() {
            // Initializer table: copy array elements into the struct.
            let n = tab.as_ref().len();
            for i in 0..n {
                let v = tab.as_ref().get_int(i as i32 + 1);
                let off = i as usize * 4;
                if off + 4 <= cd.data.len() {
                    let val = v.as_number().unwrap_or(0.0) as i32;
                    cd.data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        } else if let Some(count) = v2.as_number() {
            // Numeric argument: variable-length array.
            let count = count as usize;
            if count > 0 {
                cd = CData::new(id, count);
            }
        }
    }

    let ptr = l.global().heap.cdatas.alloc(cd);
    push(l, LuaValue::cdata(ptr));
    Ok(1)
}

pub fn ffi_sizeof(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let sz = cts_of(l).raw(id).size;
    push(l, LuaValue::number(sz as f64));
    Ok(1)
}

pub fn ffi_alignof(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let ct = cts_of(l).raw(id);
    let al = 1u32 << ctype_align(ct.info);
    push(l, LuaValue::number(al as f64));
    Ok(1)
}

pub fn ffi_typeof(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let mut cd = CData::new(CTypeID::CTypeIDType as u32, 4);
    cd.data[..4].copy_from_slice(&(id as u32).to_le_bytes());
    let ptr = l.global().heap.cdatas.alloc(cd);
    push(l, LuaValue::cdata(ptr));
    Ok(1)
}

pub fn ffi_istype(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let ok = arg(l, 1)
        .as_cdata()
        .is_some_and(|cd| cd.as_ref().ctypeid == id);
    push(l, if ok { LuaValue::TRUE } else { LuaValue::FALSE });
    Ok(1)
}

pub fn ffi_string(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.string", "cdata", ""))?;
    let ptr = cd.as_ref().get_ptr() as *const u8;

    let len = if nargs(l) > 1 {
        arg(l, 1).as_number().unwrap_or(0.0) as usize
    } else if ptr.is_null() {
        0
    } else {
        let mut n = 0;
        while n < 4096 && unsafe { *ptr.add(n) } != 0 {
            n += 1;
        }
        n
    };

    if ptr.is_null() || len == 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let h = l.heap();
    let sid = h.strings.intern(bytes);
    push(l, h.str_value(sid));
    Ok(1)
}

pub fn ffi_copy(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.copy", "cdata", ""))?;
    let src = arg(l, 1)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 2, "ffi.copy", "cdata", ""))?;
    let len = arg(l, 2).as_number().unwrap_or(0.0) as usize;

    let dp = dst.as_ref().get_ptr() as *mut u8;
    let sp = src.as_ref().get_ptr() as *const u8;
    if !dp.is_null() && !sp.is_null() && len > 0 {
        unsafe { std::ptr::copy_nonoverlapping(sp, dp, len) };
    }
    Ok(0)
}

pub fn ffi_fill(l: &mut LuaState) -> LuaResult<i32> {
    let dp = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.fill", "cdata", ""))?
        .as_ref()
        .get_ptr() as *mut u8;
    let len = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let byte = if nargs(l) > 2 {
        arg(l, 2).as_number().map(|n| n as u8).unwrap_or(0)
    } else {
        0
    };
    if !dp.is_null() && len > 0 {
        unsafe { std::ptr::write_bytes(dp, byte, len) };
    }
    Ok(0)
}

pub fn ffi_cast(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let val = arg(l, 1);

    if let Some(cd) = val.as_cdata() {
        let mut nc = CData::new(id, cd.as_ref().data.len().max(1));
        let n = nc.data.len().min(cd.as_ref().data.len());
        nc.data[..n].copy_from_slice(&cd.as_ref().data[..n]);
        let ptr = l.global().heap.cdatas.alloc(nc);
        push(l, LuaValue::cdata(ptr));
        return Ok(1);
    }

    if val.is_nil() {
        let sz = l
            .global()
            .cts
            .as_ref()
            .and_then(|c| {
                let r = c.raw(id);
                if r.size != u32::MAX {
                    Some(r.size as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0)
            .max(1);
        let ptr = l.global().heap.cdatas.alloc(CData::new(id, sz));
        push(l, LuaValue::cdata(ptr));
        return Ok(1);
    }

    if let Some(n) = val.as_number() {
        let sz = l
            .global()
            .cts
            .as_ref()
            .and_then(|c| {
                let r = c.raw(id);
                if r.size != u32::MAX {
                    Some(r.size as usize)
                } else {
                    None
                }
            })
            .unwrap_or(8)
            .max(1);
        let mut cd = CData::new(id, sz);
        let ptr = n as usize;
        let bytes = ptr.to_ne_bytes();
        let len = cd.data.len().min(bytes.len());
        cd.data[..len].copy_from_slice(&bytes[..len]);
        let gc_ptr = l.global().heap.cdatas.alloc(cd);
        push(l, LuaValue::cdata(gc_ptr));
        return Ok(1);
    }

    Err(err_bad_arg(l, 2, "ffi.cast", "cdata", ""))
}

// ---------------------------------------------------------------------------
// cdata metamethods: __index / __newindex
// ---------------------------------------------------------------------------

/// Look up a field offset in a struct type.
fn field_offset(cts: &CTState, ctypeid: u32, name: &str) -> Option<(u32, u32)> {
    let struct_id = cts.resolve_raw_id(ctypeid);
    cts.field_names.get(&(struct_id, name.to_string())).copied()
}

/// Read a numeric value from memory at a given offset with a given size.
unsafe fn read_field_value(ptr: *const u8, offset: u32, sz: usize) -> f64 {
    let p = unsafe { ptr.add(offset as usize) };
    match sz {
        1 => unsafe { *(p as *const i8) as f64 },
        2 => unsafe { *(p as *const i16) as f64 },
        4 => unsafe { *(p as *const i32) as f64 },
        8 => unsafe { *(p as *const i64) as f64 },
        _ => 0.0,
    }
}

/// Read a numeric value from a byte slice at a given offset.
fn read_field_from_slice(data: &[u8], offset: u32, sz: usize) -> f64 {
    let o = offset as usize;
    match sz {
        1 => data[o] as i8 as f64,
        2 => i16::from_le_bytes(data[o..o + 2].try_into().unwrap()) as f64,
        4 => i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f64,
        8 => i64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as f64,
        _ => 0.0,
    }
}

fn cdata_index(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0).as_cdata().ok_or(LuaError::Runtime)?;
    let key = arg(l, 1);
    let cts = l.global().cts.as_ref().ok_or(LuaError::Runtime)?;

    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };

    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    let (target_id, is_ptr) = if ctype_isptr(raw_ct.info) {
        (ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };

    let Some((field_type_id, offset)) = field_offset(cts, target_id, &name) else {
        push(l, LuaValue::NIL);
        return Ok(1);
    };

    let field_ct = cts.raw(field_type_id);
    if !ctype_isnum(field_ct.info) {
        push(l, LuaValue::NIL);
        return Ok(1);
    }

    let sz = field_ct.size as usize;
    let val = if is_ptr {
        let ptr = cd.as_ref().get_ptr();
        if ptr != 0 {
            unsafe { read_field_value(ptr as *const u8, offset, sz) }
        } else {
            0.0
        }
    } else {
        let data = &cd.as_ref().data;
        if offset as usize + sz <= data.len() {
            read_field_from_slice(data, offset, sz)
        } else {
            0.0
        }
    };

    push(l, LuaValue::number(val));
    Ok(1)
}

fn cdata_newindex(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0).as_cdata().ok_or(LuaError::Runtime)?;
    let key = arg(l, 1);
    let val = arg(l, 2);
    let cts = l.global().cts.as_ref().ok_or(LuaError::Runtime)?;

    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => return Err(LuaError::Runtime),
    };

    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    let (target_id, is_ptr) = if ctype_isptr(raw_ct.info) {
        (ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };

    let Some((_field_type_id, offset)) = field_offset(cts, target_id, &name) else {
        return Err(LuaError::Runtime);
    };

    let v = val.as_number().unwrap_or(0.0) as i32;

    if is_ptr {
        let ptr = cd.as_ref().get_ptr();
        if ptr != 0 {
            unsafe {
                let field_ptr = (ptr as *mut u8).add(offset as usize);
                *(field_ptr as *mut i32) = v;
            }
        }
    } else {
        let mut new_data = cd.as_ref().data.to_vec();
        if offset as usize + 4 <= new_data.len() {
            new_data[offset as usize..offset as usize + 4].copy_from_slice(&v.to_le_bytes());
            *cd.as_mut() = CData {
                ctypeid: cd.as_ref().ctypeid,
                data: new_data.into_boxed_slice(),
            };
        }
    }

    Ok(0)
}

// ---------------------------------------------------------------------------
// ffi.C: lazy symbol resolution and generic C call wrapper
// ---------------------------------------------------------------------------

/// `ffi.C.__index` — resolve a C symbol on first access and cache it.
fn clib_index(l: &mut LuaState) -> LuaResult<i32> {
    let key = arg(l, 1);
    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };

    let addr = clib::resolve_symbol(&name).unwrap_or(0) as f64;
    if addr == 0.0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }

    let g = l.global() as *mut GlobalState;
    let env = unsafe { (*g).globals };
    let clos = unsafe {
        (*g).heap.alloc_func(GcFunc::C(CClosure {
            f: call_c,
            env,
            upvals: vec![LuaValue::number(addr)],
        }))
    };
    let v = LuaValue::func(clos);

    // Cache in ffi.C table for next access.
    if let Some(ctab) = arg(l, 0).as_table() {
        let c_key = l.heap().str_value(l.heap().intern(name.as_bytes()));
        ctab.as_mut().set_str(c_key, v);
    }

    push(l, v);
    Ok(1)
}

/// Generic C call: read address from upvalue, marshal args as i64, call, return i64.
fn call_c(l: &mut LuaState) -> LuaResult<i32> {
    let addr = l.upvalue(0).as_number().unwrap() as usize;
    let narg = l.top - l.base;

    // Marshal each Lua argument to an i64 slot.
    let mut cstrs: Vec<CString> = Vec::new();
    let mut cargs: Vec<i64> = Vec::with_capacity(narg);
    for i in 0..narg {
        let a = arg(l, i);
        if let Some(sid) = a.as_string_id() {
            let bytes = l.heap().strings.get(sid).to_vec();
            let cs = CString::new(bytes).map_err(|_| LuaError::Runtime)?;
            cargs.push(cs.as_ptr() as i64);
            cstrs.push(cs);
        } else if let Some(cd) = a.as_cdata() {
            cargs.push(cd.as_ref().data.as_ptr() as i64);
        } else if let Some(n) = a.as_number() {
            cargs.push(n as i64);
        } else {
            cargs.push(0);
        }
    }

    type CFn = unsafe extern "system" fn(i64, i64, i64, i64, i64, i64) -> i64;
    let f: CFn = unsafe { std::mem::transmute(addr) };
    let pad = |i: usize| if i < cargs.len() { cargs[i] } else { 0 };
    let ret = unsafe { f(pad(0), pad(1), pad(2), pad(3), pad(4), pad(5)) };

    push(l, LuaValue::number(ret as f64));
    Ok(1)
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

pub fn open(l: &mut LuaState) {
    let g: *mut GlobalState = l.global() as *mut GlobalState;
    let env = unsafe { (*g).globals };

    // -- cdata metatable with __index / __newindex ----------------------------
    let cdata_mt = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 2)) };
    {
        let g = unsafe { &mut *g };
        let index_k = g.mmname[MM::Index as usize];
        let newindex_k = g.mmname[MM::Newindex as usize];
        let index_fn = g.heap.alloc_func(GcFunc::C(CClosure {
            f: cdata_index,
            env,
            upvals: vec![],
        }));
        let newindex_fn = g.heap.alloc_func(GcFunc::C(CClosure {
            f: cdata_newindex,
            env,
            upvals: vec![],
        }));
        cdata_mt.as_mut().set_str(index_k, LuaValue::func(index_fn));
        cdata_mt
            .as_mut()
            .set_str(newindex_k, LuaValue::func(newindex_fn));
        g.set_basemt(LJ_TCDATA, Some(cdata_mt));
    }

    let heap = unsafe { &mut (*g).heap };

    // -- ffi table ------------------------------------------------------------
    let ffi_tab = heap.alloc_table(LuaTable::new(0, 16));
    let builtins: [(&[u8], CFunction); 10] = [
        (b"cdef", ffi_cdef),
        (b"new", ffi_new),
        (b"sizeof", ffi_sizeof),
        (b"alignof", ffi_alignof),
        (b"typeid", ffi_typeof),
        (b"istype", ffi_istype),
        (b"string", ffi_string),
        (b"copy", ffi_copy),
        (b"fill", ffi_fill),
        (b"cast", ffi_cast),
    ];
    for &(name, func) in &builtins {
        let key_sid = heap.intern(name);
        let key = heap.str_value(key_sid);
        let f = heap.alloc_func(GcFunc::C(CClosure {
            f: func,
            env,
            upvals: vec![],
        }));
        ffi_tab.as_mut().set(key, LuaValue::func(f));
    }

    // -- ffi.C table with lazy __index ----------------------------------------
    let c_sid = heap.intern(b"C");
    let c_key = heap.str_value(c_sid);
    let c_tab = heap.alloc_table(LuaTable::new(0, 4));
    let cmt = heap.alloc_table(LuaTable::new(0, 1));
    {
        let g = unsafe { &mut *g };
        let cmt_index_k = g.mmname[MM::Index as usize];
        let cmt_index_fn = g.heap.alloc_func(GcFunc::C(CClosure {
            f: clib_index,
            env,
            upvals: vec![],
        }));
        cmt.as_mut()
            .set_str(cmt_index_k, LuaValue::func(cmt_index_fn));
    }
    c_tab.as_mut().metatable = Some(cmt);

    // -- package.preload.ffi loader -------------------------------------------
    let pk_sid = heap.intern(b"package");
    let pk_key = heap.str_value(pk_sid);
    let pr_sid = heap.intern(b"preload");
    let pr_key = heap.str_value(pr_sid);
    let ffi_sid = heap.intern(b"ffi");
    let ffi_key = heap.str_value(ffi_sid);
    let pk_tab = heap.alloc_table(LuaTable::new(2, 1));
    let pr_tab = heap.alloc_table(LuaTable::new(8, 3));
    let loader = heap.alloc_func(GcFunc::C(CClosure {
        f: preload_loader,
        env,
        upvals: vec![],
    }));

    // -- init default C lib handles (Windows) ---------------------------------
    #[cfg(windows)]
    unsafe {
        clib::init_default_libs();
    }

    // -- wire everything together ---------------------------------------------
    {
        let globals = unsafe { (*g).globals.as_mut() };

        globals.set(ffi_key, LuaValue::table(ffi_tab));
        ffi_tab.as_mut().set(c_key, LuaValue::table(c_tab));

        if globals.get(pk_key).as_table().is_none() {
            globals.set(pk_key, LuaValue::table(pk_tab));
        }
    }

    let pk = unsafe { (*g).globals.as_ref().get(pk_key).as_table().unwrap() };
    {
        let t = pk.as_mut();
        if t.get(pr_key).as_table().is_none() {
            t.set(pr_key, LuaValue::table(pr_tab));
        }
    }
    let pr = pk.as_ref().get(pr_key).as_table().unwrap();
    pr.as_mut().set(ffi_key, LuaValue::func(loader));
}

fn preload_loader(l: &mut LuaState) -> LuaResult<i32> {
    let g = l.global();
    let sid = g.heap.intern(b"ffi");
    let k = g.heap.str_value(sid);
    let t = g.globals.as_ref().get(k).as_table().unwrap();
    l.stack[l.base] = LuaValue::table(t);
    l.top = l.base + 1;
    Ok(1)
}
