//! Lua FFI library — loaded via `require("ffi")`.
//!
//! Exposes the `ffi` global table with `cdef`, `new`, `sizeof`, `cast`, etc.
//! Also sets up the `ffi.C` namespace with lazy C symbol resolution and a
//! generic variadic call wrapper.

use std::ffi::CString;

use crate::err::LuaResult;
use crate::ffi::clib;
use crate::ffi::parser::parse;
use crate::ffi::{
    CT, CTState, CType, CTypeID, ct_info, ctype_align, ctype_cid, ctype_isnum, ctype_ispointer,
};
use crate::func::{CClosure, CFunction, GcFunc};
use crate::gc::GcPtr;
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
pub(crate) fn quick_type_id(name: &str) -> Option<u32> {
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
        "complex" => CTypeID::ComplexDouble as u32,
        "complex float" => CTypeID::ComplexFloat as u32,
        "void *" | "void*" => CTypeID::PVoid as u32,
        _ => return None,
    })
}

/// On-demand pointer type creation: returns a new `CT::Ptr` → `pointee_id`.
fn make_ptr_type(cts: &mut CTState, pointee_id: u32) -> u32 {
    let info = ct_info(CT::Ptr, 3 << 16) | pointee_id; // 8-byte alignment
    for i in 0..cts.top as usize {
        if cts.tab[i].info == info {
            return i as u32;
        }
    }
    let ptr_id = cts.top;
    cts.tab.push(CType {
        info,
        size: 8,
        sib: 0,
        next: 0,
        name: 0,
    });
    cts.top = cts.top.saturating_add(1);
    ptr_id
}

/// Resolve a Lua string / cdata argument to a C type ID.
/// Handles `"type"`, `"type*"`, `"type[N]"`, and `"type[?]"` syntax.
fn check_ctype(l: &mut LuaState) -> LuaResult<u32> {
    let val = arg(l, 0);

    if val.is_cdata() {
        let cd = val.as_cdata().unwrap();
        let c = cd.as_ref();
        if c.ctypeid == crate::ffi::CTypeID::CTypeIDType as u32 && c.data.len() >= 4 {
            // A ffi.typeof value: the type id lives in the payload.
            return Ok(u32::from_le_bytes(c.data[..4].try_into().unwrap_or([0; 4])));
        }
        return Ok(c.ctypeid);
    }

    let sid = match val.as_string_id() {
        Some(s) => s,
        _ => return Err(err_bad_arg(l, 1, "ffi", "C type", "")),
    };

    let raw = l.heap().strings.get(sid).to_vec();
    let raw_str = std::str::from_utf8(&raw)
        .map_err(|_| l.runtime_error(b"ffi: invalid UTF-8 in type name"))?;
    let name = raw_str.trim().to_string();

    let is_complex_decl = name.trim_start().starts_with("struct")
        || name.trim_start().starts_with("union")
        || name.trim_start().starts_with("enum");

    let (base_name, array_count) = if is_complex_decl {
        (name.clone(), 1)
    } else if let Some(bracket) = name.rfind('[') {
        let close = name.rfind(']').unwrap_or(name.len());
        let base = name[..bracket].trim().to_string();
        let inside = name[bracket + 1..close].trim();
        // 0 = variable-length array ("[?]").
        let count: usize = if inside == "?" {
            0
        } else {
            inside.parse().unwrap_or(1)
        };
        (base, count)
    } else {
        (name.clone(), 1)
    };

    // Strip cv-qualifiers for type lookup ("const int" == "int").
    let base_name = strip_cv(&base_name);

    // First try the full base name including pointer suffixes.
    if let Some(id) = quick_type_id(&base_name) {
        return wrap_array(l, id, array_count);
    }
    if let Some(&id) = l
        .global()
        .cts
        .as_ref()
        .and_then(|c| c.names.get(&base_name))
    {
        return wrap_array(l, id, array_count);
    }

    // Strip `[...]` suffix from base_name (already done above).
    // Strip `*` suffix for pointer to custom types.
    let (base, is_ptr) = if let Some(s) = base_name.strip_suffix('*') {
        (s.trim().to_string(), true)
    } else if let Some(s) = base_name.strip_suffix(" *") {
        (s.trim().to_string(), true)
    } else {
        (base_name.clone(), false)
    };

    if let Some(id) = quick_type_id(&base) {
        let id = if is_ptr {
            make_ptr_type(cts_of(l), id)
        } else {
            id
        };
        return wrap_array(l, id, array_count);
    }

    if let Some(&id) = l.global().cts.as_ref().and_then(|c| c.names.get(&base)) {
        let id = if is_ptr {
            make_ptr_type(cts_of(l), id)
        } else {
            id
        };
        return wrap_array(l, id, array_count);
    }

    let cts = cts_of(l);
    let prev_top = cts.top;
    if let Err(e) = parse(cts, &base) {
        return Err(l.runtime_error(format!("ffi: cannot parse '{}': {}", base, e).as_bytes()));
    }
    let id = if cts.top > prev_top {
        cts.top - 1
    } else {
        return Err(err_bad_arg(l, 1, "ffi", "C type", ""));
    };
    wrap_array(l, id, array_count)
}

/// Strip leading/trailing `const`/`volatile` qualifiers for lookup.
fn strip_cv(name: &str) -> String {
    let mut s = name.trim().to_string();
    loop {
        let t = s.trim();
        let changed = if let Some(rest) = t
            .strip_prefix("const ")
            .or_else(|| t.strip_prefix("volatile "))
        {
            s = rest.trim().to_string();
            true
        } else if let Some(rest) = t
            .strip_suffix(" const")
            .or_else(|| t.strip_suffix(" volatile"))
        {
            s = rest.trim().to_string();
            true
        } else {
            false
        };
        if !changed {
            break;
        }
    }
    s
}

/// If `count > 1`, create or reuse an array `CType` wrapping the base type.
fn wrap_array(l: &mut LuaState, base_id: u32, count: usize) -> LuaResult<u32> {
    if count == 1 {
        return Ok(base_id);
    }
    let cts = cts_of(l);
    let base_sz = cts.raw(base_id).size;
    // count == 0 marks a variable-length array ("[?]").
    let total_sz = if count == 0 {
        u32::MAX
    } else {
        base_sz.saturating_mul(count as u32)
    };
    // Search existing array types for a match.
    let info = ct_info(CT::Array, 0) | base_id;
    for i in 0..cts.top as usize {
        if cts.tab[i].info == info && cts.tab[i].size == total_sz {
            return Ok(i as u32);
        }
    }
    let id = cts.top;
    cts.tab.push(CType {
        info,
        size: total_sz,
        sib: 0,
        next: 0,
        name: 0,
    });
    cts.top = id + 1;
    Ok(id)
}

// ---------------------------------------------------------------------------
// ffi table functions
// ---------------------------------------------------------------------------

pub fn ffi_cdef(l: &mut LuaState) -> LuaResult<i32> {
    let sid = arg(l, 0)
        .as_string_id()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.cdef", "string", ""))?;
    let src = l.heap().strings.get(sid).to_vec();
    let text = std::str::from_utf8(&src)
        .map_err(|e| l.runtime_error(format!("ffi.cdef: invalid UTF-8: {}", e).as_bytes()))?;
    parse(cts_of(l), text).map_err(|e| l.runtime_error(format!("ffi.cdef: {}", e).as_bytes()))?;
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
            // Initializer table: copy array/struct elements.
            let n = tab.as_ref().len() as usize;
            if n == 0 {
            } else {
                let first = tab.as_ref().get_int(1);
                if first.is_number() {
                    // Flat array of scalars: write 4-byte ints.
                    for i in 0..n {
                        let v = tab.as_ref().get_int(i as i32 + 1);
                        let off = i * 4;
                        if off + 4 <= cd.data.len() {
                            let val = v.as_number().unwrap_or(0.0) as i32;
                            cd.data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                        }
                    }
                } else if first.as_table().is_some() {
                    // Array of structs/tables: recursively fill each element.
                    let cts = cts_of(l);
                    let raw_ct = cts.raw(id);
                    let elem_id = if ctype_ispointer(raw_ct.info) {
                        ctype_cid(raw_ct.info)
                    } else {
                        id
                    };
                    let elem_sz = cts.raw(elem_id).size as usize;
                    if elem_sz > 0 {
                        for i in 0..n {
                            let sub = tab.as_ref().get_int(i as i32 + 1);
                            if let Some(st) = sub.as_table() {
                                let sub_n = st.as_ref().len() as usize;
                                for j in 0..sub_n {
                                    let fv = st.as_ref().get_int(j as i32 + 1);
                                    let off = i * elem_sz + j * 4;
                                    if off + 4 <= cd.data.len() {
                                        let val = fv.as_number().unwrap_or(0.0) as i32;
                                        cd.data[off..off + 4].copy_from_slice(&val.to_le_bytes());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if let Some(n) = v2.as_number() {
            // A scalar type initializes its value; a variable-length
            // array takes the element count; a fixed-size array is
            // filled with the (byte) value.
            let raw = cts_of(l).raw(id);
            if crate::ffi::ctype_isarray(raw.info) {
                if raw.size == u32::MAX {
                    let elem_id = crate::ffi::ctype_cid(raw.info);
                    let elem = cts_of(l).raw(elem_id).size as usize;
                    let count = n as usize;
                    if count > 0 {
                        cd = CData::new(id, count * elem.max(1));
                    }
                } else {
                    let mut d = vec![0u8; raw.size as usize];
                    d.fill(n as u8);
                    cd = CData {
                        ctypeid: id,
                        data: d.into_boxed_slice(),
                    };
                }
            } else if crate::ffi::ctype_ispointer(raw.info) {
                let count = n as usize;
                if count > 0 {
                    cd = CData::new(id, count);
                }
            } else {
                let mut d = cd.data.to_vec();
                write_scalar_value(&mut d, id, n);
                cd.data = d.into_boxed_slice();
            }
        }
    }

    let ptr = l.global().heap.cdatas.alloc(cd);
    push(l, LuaValue::cdata(ptr));
    Ok(1)
}

/// Write a numeric scalar into a cdata byte buffer for the given type.
fn write_scalar_value(data: &mut [u8], ctypeid: u32, n: f64) {
    use crate::ffi::CTypeID;
    let mut write = |off: usize, bytes: &[u8]| {
        if off + bytes.len() <= data.len() {
            data[off..off + bytes.len()].copy_from_slice(bytes);
        }
    };
    // Go through i64/u64 so the narrow casts wrap instead of saturating
    // (Rust's float->int `as` saturates).
    let i = n as i64;
    let u = n as u64;
    match ctypeid {
        id if id == CTypeID::Int8 as u32 => write(0, &(i as i8).to_le_bytes()),
        id if id == CTypeID::UInt8 as u32 => write(0, &(u as u8).to_le_bytes()),
        id if id == CTypeID::CChar as u32 => write(0, &(i as i8).to_le_bytes()),
        id if id == CTypeID::Bool as u32 => write(0, &(u as u8).to_le_bytes()),
        id if id == CTypeID::Int16 as u32 => write(0, &(i as i16).to_le_bytes()),
        id if id == CTypeID::UInt16 as u32 => write(0, &(u as u16).to_le_bytes()),
        id if id == CTypeID::Int32 as u32 => write(0, &(i as i32).to_le_bytes()),
        id if id == CTypeID::UInt32 as u32 => write(0, &(u as u32).to_le_bytes()),
        id if id == CTypeID::Int64 as u32 => write(0, &(i as i64).to_le_bytes()),
        id if id == CTypeID::UInt64 as u32 => write(0, &(u as u64).to_le_bytes()),
        id if id == CTypeID::Float as u32 => write(0, &(n as f32).to_le_bytes()),
        id if id == CTypeID::Double as u32 => write(0, &n.to_le_bytes()),
        _ => write(0, &(i as i32).to_le_bytes()),
    }
}
pub fn ffi_sizeof(l: &mut LuaState) -> LuaResult<i32> {
    let nargs = nargs(l);
    if let Some(cd) = arg(l, 0).as_cdata() {
        let c = cd.as_ref();
        if c.ctypeid == crate::ffi::CTypeID::CTypeIDType as u32 && c.data.len() >= 4 {
            // A ffi.typeof value: resolve the underlying type (a
            // variable-length array takes an explicit count).
            let id = u32::from_le_bytes(c.data[..4].try_into().unwrap_or([0; 4]));
            let ct = cts_of(l).raw(id);
            let (ct_size, ct_info) = (ct.size, ct.info);
            if nargs > 1 && ct_size == u32::MAX {
                let n = arg(l, 1).as_number().unwrap_or(0.0) as usize;
                let elem = cts_of(l).raw(ctype_cid(ct_info)).size as usize;
                push(l, LuaValue::number((n * elem) as f64));
            } else {
                push(l, LuaValue::number(ct_size as f64));
            }
            return Ok(1);
        }
        // A cdata instance reports its own (allocated) size.
        push(l, LuaValue::number(c.data.len() as f64));
        return Ok(1);
    }
    let id = check_ctype(l)?;
    let ct = cts_of(l).raw(id);
    let (ct_size, ct_info) = (ct.size, ct.info);
    if nargs > 1 && ct_size == u32::MAX {
        // Variable-length array: size = count * element size.
        let n = arg(l, 1).as_number().unwrap_or(0.0) as usize;
        let elem = cts_of(l).raw(ctype_cid(ct_info)).size as usize;
        push(l, LuaValue::number((n * elem) as f64));
    } else {
        push(l, LuaValue::number(ct_size as f64));
    }
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
    let id1 = check_ctype(l)?;
    let ok = match arg(l, 1).as_cdata() {
        Some(cd) => {
            let c = cd.as_ref();
            let id2 = if c.ctypeid == crate::ffi::CTypeID::CTypeIDType as u32
                && c.data.len() >= 4
            {
                u32::from_le_bytes(c.data[..4].try_into().unwrap_or([0; 4]))
            } else {
                c.ctypeid
            };
            let cts = cts_of(l);
            let ct1 = cts.raw(id1);
            let ct2 = cts.raw(id2);
            if std::env::var("LUAJIT_RS_FFIDBG").is_ok() {
                eprintln!(
                    "istype: id1={} id2={} info1={:#x} info2={:#x} sz1={} sz2={}",
                    id1, id2, ct1.info, ct2.info, ct1.size, ct2.size
                );
            }
            // LuaJIT's ffi_istype: identical types match; otherwise the
            // kind and size must agree (pointers check pointee
            // compatibility, numbers/voids compare modulo qualifiers).
            let same_kind_size = crate::ffi::ctype_type(ct1.info)
                == crate::ffi::ctype_type(ct2.info)
                && ct1.size == ct2.size;
            if ct1.info == ct2.info && ct1.size == ct2.size {
                true
            } else if crate::ffi::ctype_isstruct(ct1.info)
                && crate::ffi::ctype_isptr(ct2.info)
            {
                // A struct type matches a pointer-to-struct value.
                let p2 = crate::ffi::ctype_cid(ct2.info);
                crate::ffi::ctype_isstruct(cts.raw(p2).info)
            } else if same_kind_size {
                if crate::ffi::ctype_ispointer(ct1.info) {
                    let p1 = crate::ffi::ctype_cid(ct1.info);
                    let p2 = crate::ffi::ctype_cid(ct2.info);
                    let c1 = cts.raw(p1);
                    let c2 = cts.raw(p2);
                    crate::ffi::ctype_type(c1.info) == crate::ffi::ctype_type(c2.info)
                        && c1.size == c2.size
                } else if crate::ffi::ctype_isnum(ct1.info)
                    || crate::ffi::ctype_isvoid(ct1.info)
                {
                    // Ignore qualifiers (const/volatile) and the long flag.
                    const QUAL_LONG: u32 = 0x0340_0000;
                    (ct1.info ^ ct2.info) & !QUAL_LONG == 0
                } else {
                    false
                }
            } else {
                false
            }
        }
        None => false,
    };
    push(l, if ok { LuaValue::TRUE } else { LuaValue::FALSE });
    Ok(1)
}

pub fn ffi_string(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.string", "cdata", ""))?;
    let data = cd.as_ref().data.to_vec();

    let len = if nargs(l) > 1 {
        arg(l, 1).as_number().unwrap_or(0.0) as usize
    } else {
        // NUL-terminated scan over the payload.
        data.iter().position(|&b| b == 0).unwrap_or(data.len())
    };
    let n = len.min(data.len());
    let h = l.heap();
    let sid = h.strings.intern(&data[..n]);
    push(l, h.str_value(sid));
    Ok(1)
}

pub fn ffi_copy(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.copy", "cdata", ""))?;
    let src = arg(l, 1);
    let len = arg(l, 2).as_number().unwrap_or(0.0) as usize;
    let dlen = dst.as_ref().data.len();

    let mut d = dst.as_ref().data.to_vec();
    let n = len.min(dlen);
    if let Some(sc) = src.as_cdata() {
        let n = n.min(sc.as_ref().data.len());
        d[..n].copy_from_slice(&sc.as_ref().data[..n]);
    } else if let Some(sid) = src.as_string_id() {
        let bytes = l.heap().strings.get(sid);
        let copy_n = if len == 0 {
            // Copy the string plus its NUL terminator, bounded by the
            // destination size.
            bytes.len().min(dlen - 1).min(dlen)
        } else {
            n.min(bytes.len())
        };
        d[..copy_n].copy_from_slice(&bytes[..copy_n]);
        if len == 0 && copy_n < dlen {
            d[copy_n] = 0; // NUL terminator
        }
    } else {
        return Err(err_bad_arg(l, 2, "ffi.copy", "cdata/string", ""));
    }
    let ctypeid = dst.as_ref().ctypeid;
    *dst.as_mut() = CData {
        ctypeid,
        data: d.into_boxed_slice(),
    };
    Ok(0)
}

pub fn ffi_fill(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.fill", "cdata", ""))?;
    let len = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let byte = if nargs(l) > 2 {
        arg(l, 2).as_number().map(|n| n as u8).unwrap_or(0)
    } else {
        0
    };
    let mut d = dst.as_ref().data.to_vec();
    let n = len.min(d.len());
    d[..n].fill(byte);
    let ctypeid = dst.as_ref().ctypeid;
    *dst.as_mut() = CData {
        ctypeid,
        data: d.into_boxed_slice(),
    };
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

fn array_element(l: &mut LuaState, cd: GcPtr<CData>, idx: i32) -> LuaResult<i32> {
    let ctypeid = cd.as_ref().ctypeid;
    let cts = l.global().cts.as_ref().unwrap();
    let raw_ct = cts.raw(ctypeid);
    let elem_typeid = if ctype_ispointer(raw_ct.info) {
        ctype_cid(raw_ct.info)
    } else {
        ctypeid
    };
    let elem_sz = cts.raw(elem_typeid).size as usize;
    if idx < 0 || elem_sz == 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let data_len = cd.as_ref().data.len();
    let count = data_len / elem_sz;
    if idx as usize >= count {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let offset = idx as usize * elem_sz;
    let data = &cd.as_ref().data;
    let elem_bytes = data[offset..offset + elem_sz].to_vec();
    let sub = CData {
        ctypeid: elem_typeid,
        data: elem_bytes.into_boxed_slice(),
    };
    // Scalar elements read back as numbers (except 64-bit integers,
    // which stay cdata to preserve precision).
    if elem_typeid == crate::ffi::CTypeID::Int64 as u32
        || elem_typeid == crate::ffi::CTypeID::UInt64 as u32
    {
        let p = l.global().heap.cdatas.alloc(sub);
        push(l, LuaValue::cdata(p));
    } else if let Some(n) = crate::stdlib::cdata_to_number(&sub) {
        push(l, LuaValue::number(n));
    } else {
        let p = l.global().heap.cdatas.alloc(sub);
        push(l, LuaValue::cdata(p));
    }
    Ok(1)
}

fn cdata_index(l: &mut LuaState) -> LuaResult<i32> {
    let cd = match arg(l, 0).as_cdata() {
        Some(c) => c,
        None => return Err(l.runtime_error(b"ffi: expected cdata")),
    };
    let key = arg(l, 1);
    if l.global().cts.is_none() {
        return Err(l.runtime_error(b"ffi: no type state"));
    }
    let cts = l.global().cts.as_ref().unwrap();

    // Numeric key → array element access
    if let Some(idx) = key.as_number() {
        return array_element(l, cd, idx as i32);
    }

    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };

    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    let (target_id, is_ptr) = if ctype_ispointer(raw_ct.info) {
        (ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };

    let Some((field_type_id, offset)) = field_offset(cts, target_id, &name) else {
        return Err(l.runtime_error(
            format!("no member '{}' in cdata", name).as_bytes(),
        ));
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
    let cd = match arg(l, 0).as_cdata() {
        Some(c) => c,
        None => return Err(l.runtime_error(b"ffi: expected cdata")),
    };
    let key = arg(l, 1);
    let val = arg(l, 2);
    if l.global().cts.is_none() {
        return Err(l.runtime_error(b"ffi: no type state"));
    }
    let cts = l.global().cts.as_ref().unwrap();

    // Numeric key → array element write.
    if let Some(idx) = key.as_number() {
        let ctypeid = cd.as_ref().ctypeid;
        let raw_ct = cts.raw(ctypeid);
        let elem_typeid = if ctype_ispointer(raw_ct.info) {
            ctype_cid(raw_ct.info)
        } else {
            ctypeid
        };
        let elem_sz = cts.raw(elem_typeid).size as usize;
        if elem_sz == 0 {
            return Ok(0);
        }
        let offset = idx as i64 * elem_sz as i64;
        if offset < 0 {
            return Ok(0);
        }
        let mut data = cd.as_ref().data.to_vec();
        let mut ebuf = vec![0u8; elem_sz];
        write_scalar_value(&mut ebuf, elem_typeid, val.as_number().unwrap_or(0.0));
        let off = offset as usize;
        if off + elem_sz <= data.len() {
            data[off..off + elem_sz].copy_from_slice(&ebuf);
            *cd.as_mut() = CData {
                ctypeid,
                data: data.into_boxed_slice(),
            };
        }
        return Ok(0);
    }

    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => return Err(l.runtime_error(b"ffi: non-string key in cdata assignment")),
    };

    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    let (target_id, is_ptr) = if ctype_ispointer(raw_ct.info) {
        (ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };

    let Some((_field_type_id, offset)) = field_offset(cts, target_id, &name) else {
        return Err(l.runtime_error(format!("ffi: no member '{}' in cdata", name).as_bytes()));
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
            let cs = CString::new(bytes)
                .map_err(|_| l.runtime_error(b"ffi: string contains null byte"))?;
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
// Additional FFI exports
// ---------------------------------------------------------------------------

pub fn ffi_abi(l: &mut LuaState) -> LuaResult<i32> {
    let param = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => {
            push(l, LuaValue::FALSE);
            return Ok(1);
        }
    };
    let r = match param.as_slice() {
        b"le" => !cfg!(target_endian = "big"),
        b"be" => cfg!(target_endian = "big"),
        b"fpu" => true,
        b"softfp" => false,
        b"hardfp" => true,
        b"eabi" => false,
        b"win" => cfg!(windows),
        b"32bit" => cfg!(target_pointer_width = "32"),
        b"64bit" => cfg!(target_pointer_width = "64"),
        _ => false,
    };
    push(l, LuaValue::boolean(r));
    Ok(1)
}

pub fn ffi_arch(l: &mut LuaState) -> LuaResult<i32> {
    let s: &[u8] = if cfg!(target_arch = "x86_64") {
        b"x64"
    } else if cfg!(target_arch = "aarch64") {
        b"arm64"
    } else {
        b"unknown"
    };
    let sid = l.heap().intern(s);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn ffi_os(l: &mut LuaState) -> LuaResult<i32> {
    let s: &[u8] = if cfg!(windows) {
        b"Windows"
    } else if cfg!(target_os = "macos") {
        b"OSX"
    } else if cfg!(target_os = "linux") {
        b"Linux"
    } else {
        b"Other"
    };
    let sid = l.heap().intern(s);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn ffi_errno(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    if let Some(n) = v.as_int32_exact() {
        push(l, LuaValue::number(n as f64));
    } else if v.is_nil() {
        push(l, LuaValue::number(0.0));
    } else {
        return Err(err_bad_arg(l, 1, "ffi.errno", "nil or number", ""));
    }
    Ok(1)
}

pub fn ffi_gc(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0);
    push(l, cd);
    Ok(1)
}

pub fn ffi_load(l: &mut LuaState) -> LuaResult<i32> {
    push(l, LuaValue::NIL);
    let sid = l.heap().intern(b"ffi.load not implemented");
    push(l, l.heap().str_value(sid));
    Ok(2)
}

pub fn ffi_metatype(l: &mut LuaState) -> LuaResult<i32> {
    let id = check_ctype(l)?;
    let mt = arg(l, 1);
    if !mt.is_table() {
        return Err(err_bad_arg(l, 2, "ffi.metatype", "table", ""));
    }
    let mt = mt.as_table().unwrap();
    let g = l.global();
    if g.ctype_mts.len() <= id as usize {
        g.ctype_mts.resize(id as usize + 1, None);
    }
    g.ctype_mts[id as usize] = Some(mt);
    // The type value itself gets the ctype metatable (so it stays
    // callable); its payload holds the registered type id.
    let mut cd = CData::new(CTypeID::CTypeIDType as u32, 4);
    cd.data[..4].copy_from_slice(&id.to_le_bytes());
    let p = l.global().heap.cdatas.alloc(cd);
    push(l, LuaValue::cdata(p));
    Ok(1)
}

pub fn ffi_offsetof(l: &mut LuaState) -> LuaResult<i32> {
    push(l, LuaValue::number(0.0));
    Ok(1)
}

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------

pub fn open(l: &mut LuaState) {
    let g: *mut GlobalState = l.global() as *mut GlobalState;
    let env = unsafe { (*g).globals };

    // -- cdata metatable with __index / __newindex / __call -------------------
    let cdata_mt = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 3)) };
    {
        let g = unsafe { &mut *g };
        let index_k = g.mmname[MM::Index as usize];
        let newindex_k = g.mmname[MM::Newindex as usize];
        let call_k = g.mmname[MM::Call as usize];
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
        // Calling a type value allocates an instance (ffi.new).
        let call_fn = g.heap.alloc_func(GcFunc::C(CClosure {
            f: ffi_new,
            env,
            upvals: vec![],
        }));
        cdata_mt.as_mut().set_str(index_k, LuaValue::func(index_fn));
        cdata_mt
            .as_mut()
            .set_str(newindex_k, LuaValue::func(newindex_fn));
        cdata_mt.as_mut().set_str(call_k, LuaValue::func(call_fn));
        g.set_basemt(LJ_TCDATA, Some(cdata_mt));
    }

    let heap = unsafe { &mut (*g).heap };

    // -- ffi table ------------------------------------------------------------
    let ffi_tab = heap.alloc_table(LuaTable::new(0, 5)); // 20 entries → 32 nodes
    let builtins: [(&[u8], CFunction); 19] = [
        (b"cdef", ffi_cdef),
        (b"new", ffi_new),
        (b"sizeof", ffi_sizeof),
        (b"alignof", ffi_alignof),
        (b"typeid", ffi_typeof),
        (b"typeof", ffi_typeof),
        (b"istype", ffi_istype),
        (b"string", ffi_string),
        (b"copy", ffi_copy),
        (b"fill", ffi_fill),
        (b"cast", ffi_cast),
        (b"abi", ffi_abi),
        (b"arch", ffi_arch),
        (b"os", ffi_os),
        (b"errno", ffi_errno),
        (b"gc", ffi_gc),
        (b"load", ffi_load),
        (b"metatype", ffi_metatype),
        (b"offsetof", ffi_offsetof),
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
