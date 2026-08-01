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
        #[cfg(target_pointer_width = "64")]
        "ptrdiff_t" | "intptr_t" | "ssize_t" => CTypeID::Int64 as u32,
        #[cfg(target_pointer_width = "64")]
        "size_t" | "uintptr_t" => CTypeID::UInt64 as u32,
        #[cfg(target_pointer_width = "32")]
        "ptrdiff_t" | "intptr_t" | "ssize_t" => CTypeID::Int32 as u32,
        #[cfg(target_pointer_width = "32")]
        "size_t" | "uintptr_t" => CTypeID::UInt32 as u32,
        "float" => CTypeID::Float as u32,
        "double" => CTypeID::Double as u32,
        "complex" => CTypeID::ComplexDouble as u32,
        "complex float" => CTypeID::ComplexFloat as u32,
        "void *" | "void*" => CTypeID::PVoid as u32,
        _ => return None,
    })
}

/// On-demand pointer type creation: returns a new `CT::Ptr` → `pointee_id`.
pub(crate) fn make_ptr_type(cts: &mut CTState, pointee_id: u32) -> u32 {
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
    check_type_name(l, raw_str.trim())
}

/// Resolve a C type name string to a type ID (ffi.typeof/ffi.new lookup
/// incl. pointer suffixes, arrays, and inline declarations).
fn check_type_name(l: &mut LuaState, name: &str) -> LuaResult<u32> {
    let trimmed = name.trim();
    // Complex abstract declarators: `int (*(*[1][2])[3][4])[5][6]`,
    // `int (*const)(void)`. The base is the longest known type prefix;
    // the remainder is parsed as a declarator.
    let (base_name, declarator) = {
        let mut best = 0;
        let mut best_len = 0;
        let bytes = trimmed.as_bytes();
        for i in 1..=trimmed.len() {
            if bytes[i - 1] == b' ' || bytes[i - 1] == b'*' || bytes[i - 1] == b'('
                || bytes[i - 1] == b'['
            {
                let cand = trimmed[..i - 1].trim();
                if cand.is_empty() {
                    continue;
                }
                let cand_cv = strip_cv(cand);
                let known = quick_type_id(&cand_cv).is_some()
                    || quick_type_id(&cand).is_some()
                    || l
                        .global()
                        .cts
                        .as_ref()
                        .is_some_and(|c| c.names.contains_key(&cand_cv))
                    || l
                        .global()
                        .cts
                        .as_ref()
                        .is_some_and(|c| c.names.contains_key(cand));
                if known && cand.len() > best_len {
                    best = i - 1;
                    best_len = cand.len();
                }
            }
        }
        if best_len > 0 {
            (trimmed[..best].trim().to_string(), trimmed[best..].trim().to_string())
        } else {
            (trimmed.to_string(), String::new())
        }
    };
    // The whole name may itself be a base ("complex float", typedefs).
    let whole_cv = strip_cv(trimmed);
    if !declarator.is_empty()
        && (quick_type_id(&whole_cv).is_some()
            || l
                .global()
                .cts
                .as_ref()
                .is_some_and(|c| c.names.contains_key(&whole_cv)))
    {
        return resolve_base_type(l, trimmed);
    }

    // Resolve the base type.
    let base_id = resolve_base_type(l, &base_name)?;

    if declarator.is_empty() {
        return Ok(base_id);
    }

    // Parse the abstract declarator.
    let mut toks = DeclTok::new(declarator.as_bytes());
    let id = parse_abs_decl(l, base_id, &mut toks)?;
    if toks.pos != toks.src.len() {
        return Err(l.runtime_error(
            format!("ffi: cannot parse '{}'", name).as_bytes(),
        ));
    }
    Ok(id)
}

/// The base type of a name (scalar, typedef, inline struct/union).
fn resolve_base_type(l: &mut LuaState, name: &str) -> LuaResult<u32> {
    let name = name.trim();
    // Qualifiers apply to the base type ("const void" == "void").
    let name_cv = strip_cv(name);
    if let Some(id) = quick_type_id(&name_cv) {
        return Ok(id);
    }
    if let Some(&id) = l.global().cts.as_ref().and_then(|c| c.names.get(&name_cv)) {
        return Ok(id);
    }
    let cts = cts_of(l);
    let prev_top = cts.top;
    if let Err(e) = parse(cts, name) {
        return Err(l.runtime_error(format!("ffi: cannot parse '{}': {}", name, e).as_bytes()));
    }
    if cts.top > prev_top {
        Ok(cts.top - 1)
    } else {
        Err(err_bad_arg(l, 1, "ffi", "C type", ""))
    }
}

/// Abstract declarator tokenizer.
struct DeclTok<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> DeclTok<'a> {
    fn new(src: &'a [u8]) -> Self {
        DeclTok { src, pos: 0 }
    }
    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
    /// Returns `Some(tok)` where tok is one of `*()[]` or a number/word.
    fn next_tok(&mut self) -> Option<Vec<u8>> {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return None;
        }
        let c = self.src[self.pos];
        if matches!(c, b'*' | b'(' | b')' | b'[' | b']' | b',') {
            self.pos += 1;
            return Some(vec![c]);
        }
        let start = self.pos;
        while self.pos < self.src.len()
            && !self.src[self.pos].is_ascii_whitespace()
            && !matches!(
                self.src[self.pos],
                b'*' | b'(' | b')' | b'[' | b']' | b','
            )
        {
            self.pos += 1;
        }
        Some(self.src[start..self.pos].to_vec())
    }
    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }
}

/// Parse an abstract declarator starting at `toks`. `base` is the
/// element type; returns the declarator-applied type.
fn parse_abs_decl(l: &mut LuaState, base: u32, toks: &mut DeclTok) -> LuaResult<u32> {
    let mut stars = 0usize;
    let mut qual = 0u32;
    loop {
        match toks.peek() {
            Some(b'*') => {
                stars += 1;
                toks.next_tok();
            }
            Some(_) => break,
            None => break,
        }
    }
    // Optional qualifier after the stars (`*const`).
    if let Some(t) = toks.next_tok() {
        if t == b"const" {
            qual |= crate::ffi::ctinfo::CONST;
        } else if t == b"volatile" {
            qual |= crate::ffi::ctinfo::VOLATILE;
        } else {
            toks.pos -= t.len(); // push back
            let _ = toks.skip_ws();
        }
    }
    // A `(` after the stars is a function suffix when its content is a
    // parameter type (`*(void)`); otherwise it groups a declarator.
    let paren_is_group = if toks.peek() == Some(b'(') {
        let save = toks.pos;
        toks.next_tok();
        let inner_tok = toks.next_tok().unwrap_or_default();
        toks.pos = save;
        let name = String::from_utf8_lossy(&inner_tok);
        let known = quick_type_id(&name).is_some()
            || l
                .global()
                .cts
                .as_ref()
                .is_some_and(|c| c.names.contains_key(&name.to_string()))
            || inner_tok == b"void";
        !known
    } else {
        false
    };
    let t = if paren_is_group {
        toks.next_tok();
        let inner = parse_abs_decl(l, base, toks)?;
        if toks.next_tok().as_deref() != Some(b")".as_slice()) {
            return Err(l.runtime_error(b"ffi: expected ')' in declarator"));
        }
        parse_decl_suffixes(l, inner, toks)?
    } else {
        parse_decl_suffixes(l, base, toks)?
    };
    let mut id = t;
    for _ in 0..stars {
        id = make_ptr_type(cts_of(l), id);
        if qual != 0 {
            let cts = cts_of(l);
            let mut e = cts.tab[id as usize].clone();
            e.info |= qual;
            cts.tab[id as usize] = e;
        }
    }
    Ok(id)
}

/// Array `[N]` and function `(params)` suffixes applied to `t`.
fn parse_decl_suffixes(l: &mut LuaState, mut t: u32, toks: &mut DeclTok) -> LuaResult<u32> {
    loop {
        match toks.peek() {
            Some(b'[') => {
                toks.next_tok();
                let mut n: u32 = 1;
                let mut is_vla = false;
                if let Some(tok) = toks.next_tok() {
                    if tok == b"?" {
                        is_vla = true;
                    } else if let Ok(v) = String::from_utf8_lossy(&tok).parse::<u32>() {
                        n = v.max(1);
                    } else {
                        toks.pos -= tok.len();
                        let _ = toks.skip_ws();
                    }
                }
                if toks.next_tok().as_deref() != Some(b"]".as_slice()) {
                    return Err(l.runtime_error(b"ffi: expected ']' in declarator"));
                }
                t = make_array_type(cts_of(l), t, if is_vla { u32::MAX } else { n });
            }
            Some(b'(') => {
                toks.next_tok();
                let mut params: Vec<u32> = Vec::new();
                loop {
                    match toks.peek() {
                        Some(b')') => {
                            toks.next_tok();
                            break;
                        }
                        Some(b'v') => {
                            // `(void)`: consume `void` + `)`.
                            toks.next_tok();
                            if toks.peek() == Some(b')') {
                                toks.next_tok();
                            }
                            break;
                        }
                        None => return Err(l.runtime_error(b"ffi: expected ')' in declarator")),
                        _ => {
                            let ptok = toks.next_tok().unwrap_or_default();
                            let pid = quick_type_id(&String::from_utf8_lossy(&ptok))
                                .or_else(|| {
                                    l.global().cts.as_ref().and_then(|c| {
                                        c.names.get(&String::from_utf8_lossy(&ptok).into_owned()).copied()
                                    })
                                });
                            if let Some(pid) = pid {
                                params.push(pid);
                            }
                            if toks.peek() == Some(b',') {
                                toks.next_tok();
                            }
                        }
                    }
                }
                t = make_func_type(cts_of(l), t, params);
            }
            _ => break,
        }
    }
    Ok(t)
}

/// Create a fixed-size array type `elem[N]`.
fn make_array_type(cts: &mut CTState, elem: u32, n: u32) -> u32 {
    let elem_sz = cts.raw(elem).size;
    let total_sz = elem_sz.saturating_mul(n);
    let info = ct_info(CT::Array, 0) | elem;
    if let Some(existing) = (0..cts.top as usize)
        .find(|&i| cts.tab[i].info == info && cts.tab[i].size == total_sz)
    {
        return existing as u32;
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
    id
}

/// Create a function type `ret(params)` (param fields chained via sib).
fn make_func_type(cts: &mut CTState, ret: u32, params: Vec<u32>) -> u32 {
    let first_param = cts.top;
    for (i, &pt) in params.iter().enumerate() {
        let fid = cts.top;
        cts.tab.push(CType {
            info: ct_info(CT::Field, 0) | pt,
            size: 0,
            sib: if i + 1 < params.len() {
                (fid + 1) as u16
            } else {
                0
            },
            next: 0,
            name: 0,
        });
        cts.top = fid + 1;
    }
    let id = cts.top;
    cts.tab.push(CType {
        info: ct_info(CT::Func, 0) | ret,
        size: 0,
        sib: if params.is_empty() { 0 } else { first_param as u16 },
        next: 0,
        name: 0,
    });
    cts.top = id + 1;
    id
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

    let nvals = nargs(l).saturating_sub(1);
    if nvals > 0 {
        // Complex/array types take sequential element values
        // (cx(1, 2) → [1.0, 2.0]); structs take sequential field values.
        let (is_complex, is_array, elem_typeid, esz, raw0_size) = {
            let (raw0_info, raw0_size, elem_typeid) = {
                let raw0 = cts_of(l).raw(id);
                (raw0.info, raw0.size, crate::ffi::ctype_cid(raw0.info))
            };
            if crate::ffi::ctype_iscomplex(raw0_info) || crate::ffi::ctype_isarray(raw0_info) {
                let e = cts_of(l).raw(elem_typeid);
                (
                    crate::ffi::ctype_iscomplex(raw0_info),
                    crate::ffi::ctype_isarray(raw0_info),
                    elem_typeid,
                    e.size as usize,
                    raw0_size,
                )
            } else {
                (false, false, 0, 0, 0)
            }
        };
        if is_complex || (is_array && nvals > 1) {
            // The first argument of a variable-length array is the
            // element count; a single extra argument fills all elements.
            let is_vla = is_array && raw0_size == u32::MAX;
            if is_vla {
                let count = arg(l, 1).as_number().unwrap_or(0.0) as usize;
                if count > 0 {
                    cd = CData::new(id, count * esz.max(1));
                }
            }
            if is_vla && nvals == 2 {
                let v = arg(l, 2);
                let n = v.as_number().unwrap_or(0.0);
                let mut b = vec![0u8; esz];
                write_scalar_value(&mut b, elem_typeid, n);
                for chunk in cd.data.chunks_mut(esz.max(1)) {
                    let k = chunk.len().min(b.len());
                    chunk[..k].copy_from_slice(&b[..k]);
                }
            } else {
                let start = if is_vla { 1 } else { 0 };
                for i in start..nvals {
                    let v = arg(l, i + 1);
                    if let Some(n) = v.as_number() {
                        let mut ebuf = vec![0u8; esz];
                        write_scalar_value(&mut ebuf, elem_typeid, n);
                        let off = i * esz;
                        if off + esz <= cd.data.len() {
                            cd.data[off..off + esz].copy_from_slice(&ebuf);
                        }
                    } else if let Some(src) = v.as_cdata() {
                        let off = i * esz;
                        if off + esz <= cd.data.len() {
                            let n = src.as_ref().data.len().min(esz);
                            cd.data[off..off + n].copy_from_slice(&src.as_ref().data[..n]);
                        }
                    }
                }
            }
        } else {
            let v2 = arg(l, 1);
        if nvals > 1 && crate::ffi::ctype_isstruct(cts_of(l).raw(id).info) {
            // Struct with sequential field initializers
            // (ffi.new("struct { int a,b,c; }", 1, 2, 3)).
            let fields: Vec<(u32, u32)> = {
                let raw = cts_of(l).raw(id);
                let mut cur = raw.info & 0xFFFF; // First field (struct info).
                let mut out = Vec::new();
                while cur != 0 {
                    let f = cts_of(l).tab.get(cur as usize);
                    let Some(f) = f else { break };
                    out.push((f.info & 0xFFFF, f.size));
                    cur = f.sib as u32;
                }
                out
            };
            for (fi, &(ftype, foff)) in fields.iter().enumerate() {
                let arg_i = fi + 1;
                if arg_i > nvals {
                    break;
                }
                let v = arg(l, arg_i);

                let ftype_raw = cts_of(l).raw(ftype);
                let sz = ftype_raw.size as usize;
                if let Some(n) = v.as_number() {
                    let mut ebuf = vec![0u8; sz.max(1)];
                    write_scalar_value(&mut ebuf, ftype, n);
                    if foff as usize + sz <= cd.data.len() {
                        cd.data[foff as usize..foff as usize + sz].copy_from_slice(&ebuf);
                    }
                } else if let Some(src) = v.as_cdata() {
                    if crate::ffi::ctype_ispointer(ftype_raw.info) {
                        // Pointer field: store the storage address.
                        let (off, root) = crate::runtime::cdata::resolve_ptr(src);
                        let addr = (root.as_ref().data.as_ptr() as i64).wrapping_add(off);
                        let a = foff as usize;
                        if sz == 8 && a + 8 <= cd.data.len() {
                            cd.data[a..a + 8].copy_from_slice(&(addr as usize).to_ne_bytes());
                        } else if sz == 4 && a + 4 <= cd.data.len() {
                            cd.data[a..a + 4].copy_from_slice(&(addr as u32).to_le_bytes());
                        }
                    } else if foff as usize + sz <= cd.data.len() {
                        let n = src.as_ref().data.len().min(sz);
                        cd.data[foff as usize..foff as usize + n]
                            .copy_from_slice(&src.as_ref().data[..n]);
                    }
                }
            }
        } else if let Some(src) = v2.as_cdata() {
            // Construct from an existing cdata: copy its bytes.
            let n = src.as_ref().data.len().min(cd.data.len());
            cd.data[..n].copy_from_slice(&src.as_ref().data[..n]);
        } else if let Some(tab) = v2.as_table() {
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
                    // Fixed-size array: the value fills every element.
                    let (elem_id, raw_size) = {
                        let raw = cts_of(l).raw(id);
                        (crate::ffi::ctype_cid(raw.info), raw.size)
                    };
                    let esz = cts_of(l).raw(elem_id).size as usize;
                    let mut ebuf = vec![0u8; esz.max(1)];
                    write_scalar_value(&mut ebuf, elem_id, n);
                    let mut d = vec![0u8; raw_size as usize];
                    for chunk in d.chunks_mut(esz.max(1)) {
                        let k = chunk.len().min(ebuf.len());
                        chunk[..k].copy_from_slice(&ebuf[..k]);
                    }
                    cd = CData {
                        ctypeid: id,
                        data: d.into_boxed_slice(),
                        base: None,
                        offset: 0,
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
    // Parameterized types: `ffi.typeof("struct { $ x; }", tp)` — every
    // `$` placeholder is replaced by the corresponding argument's type.

    if nargs(l) > 1 {
        let v0 = arg(l, 0);
        if let Some(sid) = v0.as_string_id() {
            let mut text = l.heap().strings.get(sid).to_vec();
            let mut replaced = false;
            for i in 1..nargs(l) {
                let a = arg(l, i);
                let rep: Option<Vec<u8>> = if let Some(cd) = a.as_cdata() {
                    let id = if cd.as_ref().ctypeid == CTypeID::CTypeIDType as u32
                        && cd.as_ref().data.len() >= 4
                    {
                        u32::from_le_bytes(cd.as_ref().data[..4].try_into().unwrap_or([0; 4]))
                    } else {
                        cd.as_ref().ctypeid
                    };
                    type_name_of_id(l, id)
                } else if let Some(s2) = a.as_string_id() {
                    Some(l.heap().strings.get(s2).to_vec())
                } else {
                    None
                };
                if let Some(rep) = rep {

                    if let Some(pos) = text.iter().position(|&b| b == b'$') {
                        text.splice(pos..pos + 1, rep);
                        replaced = true;
                    }
                }
            }
            if replaced {
                let s = String::from_utf8_lossy(&text).to_string();
                let id = check_type_name(l, &s)?;
                return type_value(l, id);
            }
        }
    }
    let id = check_ctype(l)?;
    type_value(l, id)
}

/// The C type name for a type ID — the typedef name, or a constructed
/// form for arrays (`int[10]`) and pointers (`int *`). Complex abstract
/// declarators render canonically (`int (*(*[1][2])[3][4])[5][6]`).
fn type_name_of_id(l: &LuaState, id: u32) -> Option<Vec<u8>> {
    let cts = l.global().cts.as_ref()?;
    let raw = cts.raw(id);
    // Structs/unions render with their kind prefix ("union NAME").
    let kind = crate::ffi::ctype_type(raw.info);
    if kind == crate::ffi::CT::Struct {
        let base = type_name_of_id_inner(l, id)?;
        let prefix = if raw.info & crate::ffi::ctinfo::UNION != 0 {
            "union "
        } else {
            "struct "
        };
        return Some(
            format!("{}{}", prefix, String::from_utf8_lossy(&base)).into_bytes(),
        );
    }
    if kind == crate::ffi::CT::Ptr || kind == crate::ffi::CT::Array || kind == crate::ffi::CT::Func
    {
        // Find the innermost scalar base, then render the declarator.
        let mut cur = id;
        let mut guard = 0;
        while guard < 64 {
            guard += 1;
            let r = cts.raw(cur);
            let k = crate::ffi::ctype_type(r.info);
            if crate::ffi::ctype_iscomplex(r.info) {
                break; // `complex` / `complex float` are bases.
            }
            if k == crate::ffi::CT::Ptr || k == crate::ffi::CT::Array {
                cur = ctype_cid(r.info);
            } else if k == crate::ffi::CT::Func {
                cur = ctype_cid(r.info);
            } else {
                break;
            }
        }
        let base = type_name_of_id_inner(l, cur)?;
        let decl = render_decl_top(l, id)?;
        if decl.is_empty() {
            return Some(base);
        }
        // Arrays join without a space (`int[10]`), others with one.
        let s = if decl.starts_with('[') {
            format!("{}{}", String::from_utf8_lossy(&base), decl)
        } else {
            format!("{} {}", String::from_utf8_lossy(&base), decl)
        };
        return Some(s.into_bytes());
    }
    type_name_of_id_inner(l, id)
}

/// Top-level declarator rendering (no surrounding parens).
fn render_decl_top(l: &LuaState, id: u32) -> Option<String> {
    let cts = l.global().cts.as_ref()?;
    let raw = cts.raw(id);
    match crate::ffi::ctype_type(raw.info) {
        crate::ffi::CT::Ptr => {
            let mut s = "*".to_string();
            if raw.info & crate::ffi::ctinfo::CONST != 0 {
                s.push_str("const");
            }
            if raw.info & crate::ffi::ctinfo::VOLATILE != 0 {
                s.push_str("volatile");
            }
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            s.push_str(&child);
            Some(s)
        }
        crate::ffi::CT::Array => {
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            let elem = cts.raw(ctype_cid(raw.info));
            let n = if raw.size == u32::MAX || elem.size == 0 {
                "?".to_string()
            } else {
                (raw.size / elem.size).to_string()
            };
            Some(format!("{}[{}]", child, n))
        }
        crate::ffi::CT::Func => {
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            Some(format!("{}()", child))
        }
        _ => Some(String::new()),
    }
}

/// Subordinate declarator rendering (pointers/functions in parens).
fn render_decl_child(l: &LuaState, id: u32) -> Option<String> {
    let cts = l.global().cts.as_ref()?;
    let raw = cts.raw(id);
    if crate::ffi::ctype_iscomplex(raw.info) {
        return Some(String::new()); // complex is a base, not a declarator
    }
    match crate::ffi::ctype_type(raw.info) {
        crate::ffi::CT::Ptr => {
            let mut s = "*".to_string();
            if raw.info & crate::ffi::ctinfo::CONST != 0 {
                s.push_str("const");
            }
            if raw.info & crate::ffi::ctinfo::VOLATILE != 0 {
                s.push_str("volatile");
            }
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            s.push_str(&child);
            Some(format!("({})", s))
        }
        crate::ffi::CT::Array => {
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            let elem = cts.raw(ctype_cid(raw.info));
            let n = if raw.size == u32::MAX || elem.size == 0 {
                "?".to_string()
            } else {
                (raw.size / elem.size).to_string()
            };
            Some(format!("{}[{}]", child, n))
        }
        crate::ffi::CT::Func => {
            let child = render_decl_child(l, ctype_cid(raw.info))?;
            Some(format!("{}()", child))
        }
        _ => Some(String::new()),
    }
}

fn type_name_of_id_inner(l: &LuaState, id: u32) -> Option<Vec<u8>> {
    let cts = l.global().cts.as_ref()?;
    if let Some((n, _)) = cts.names.iter().find(|(_, v)| **v == id) {
        return Some(n.clone().into_bytes());
    }
    let raw = cts.raw(id);
    use crate::ffi::{ctype_cid, ctype_isarray, ctype_iscomplex, ctype_ispointer};
    if ctype_iscomplex(raw.info) {
        // Predefined complex types render as `complex` / `complex float`.
        let elem = ctype_cid(raw.info);
        return if elem == crate::ffi::CTypeID::Double as u32 {
            Some(b"complex".to_vec())
        } else {
            Some(b"complex float".to_vec())
        };
    }
    if ctype_isarray(raw.info) {
        let elem = type_name_of_id(l, ctype_cid(raw.info))?;
        let sz = raw.size;
        let s = if sz == u32::MAX {
            format!("{}[?]", String::from_utf8_lossy(&elem))
        } else {
            format!("{}[{}]", String::from_utf8_lossy(&elem), sz)
        };
        return Some(s.into_bytes());
    }
    if ctype_ispointer(raw.info) {
        let elem = type_name_of_id(l, ctype_cid(raw.info))?;
        return Some(format!("{} *", String::from_utf8_lossy(&elem)).into_bytes());
    }
    // Predefined scalar types.
    let n = match id as u32 {
        v if v == crate::ffi::CTypeID::Void as u32 => "void",
        v if v == crate::ffi::CTypeID::Bool as u32 => "bool",
        v if v == crate::ffi::CTypeID::CChar as u32 => "char",
        v if v == crate::ffi::CTypeID::Int8 as u32 => "int8_t",
        v if v == crate::ffi::CTypeID::UInt8 as u32 => "uint8_t",
        v if v == crate::ffi::CTypeID::Int16 as u32 => "int16_t",
        v if v == crate::ffi::CTypeID::UInt16 as u32 => "uint16_t",
        v if v == crate::ffi::CTypeID::Int32 as u32 => "int",
        v if v == crate::ffi::CTypeID::UInt32 as u32 => "unsigned int",
        v if v == crate::ffi::CTypeID::Int64 as u32 => "int64_t",
        v if v == crate::ffi::CTypeID::UInt64 as u32 => "uint64_t",
        v if v == crate::ffi::CTypeID::Float as u32 => "float",
        v if v == crate::ffi::CTypeID::Double as u32 => "double",
        v if v == crate::ffi::CTypeID::ComplexFloat as u32 => "complex float",
        v if v == crate::ffi::CTypeID::ComplexDouble as u32 => "complex",
        _ => return None,
    };
    Some(n.to_string().into_bytes())
}

fn type_value(l: &mut LuaState, id: u32) -> LuaResult<i32> {
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
    let (off, root) = crate::runtime::cdata::resolve_ptr(cd);
    let data = &root.as_ref().data;
    let start = off.max(0) as usize;

    let len = if nargs(l) > 1 {
        arg(l, 1).as_number().unwrap_or(0.0) as usize
    } else {
        // NUL-terminated scan over the payload.
        data[start..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(data.len() - start)
    };
    let n = len.min(data.len().saturating_sub(start));
    let h = l.heap();
    let sid = h.strings.intern(&data[start..start + n]);
    push(l, h.str_value(sid));
    Ok(1)
}

pub fn ffi_copy(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.copy", "cdata", ""))?;
    let src = arg(l, 1);
    let len = arg(l, 2).as_number().unwrap_or(0.0) as usize;
    let (d_off, d_root) = crate::runtime::cdata::resolve_ptr(dst);
    let dlen = d_root.as_ref().data.len().saturating_sub(d_off.max(0) as usize);
    let d_start = d_off.max(0) as usize;
    let n = len.min(dlen);

    // Read the source bytes up front (the source and destination may be
    // the same storage, so no borrow can span the write).
    let src_bytes: Vec<u8> = if let Some(sc) = src.as_cdata() {
        let (s_off, s_root) = crate::runtime::cdata::resolve_ptr(sc);
        let s_start = s_off.max(0) as usize;
        let s_buf = &s_root.as_ref().data;
        let n = n.min(s_buf.len().saturating_sub(s_start));
        s_buf[s_start..s_start + n].to_vec()
    } else if let Some(sid) = src.as_string_id() {
        let bytes = l.heap().strings.get(sid);
        let copy_n = if len == 0 {
            // Copy the string plus its NUL terminator, bounded by the
            // destination size.
            bytes.len().min(dlen.saturating_sub(1)).min(dlen)
        } else {
            n.min(bytes.len())
        };
        let mut v = bytes[..copy_n].to_vec();
        if len == 0 && copy_n < dlen {
            v.push(0); // NUL terminator
        }
        v
    } else {
        return Err(err_bad_arg(l, 2, "ffi.copy", "cdata/string", ""));
    };
    let d_buf = &mut d_root.as_mut().data;
    let n = src_bytes.len().min(dlen);

    d_buf[d_start..d_start + n].copy_from_slice(&src_bytes[..n]);
    Ok(0)
}

pub fn ffi_fill(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0)
        .as_cdata()
        .ok_or_else(|| err_bad_arg(l, 1, "ffi.fill", "cdata", ""))?;
    let len = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let byte = if nargs(l) > 2 {
        let b = arg(l, 2)
            .as_number()
            .map(|n| (n as i64) as u8)
            .unwrap_or(0);

        b
    } else {
        0
    };
    let (off, root) = crate::runtime::cdata::resolve_ptr(dst);
    let start = off.max(0) as usize;
    let d = &mut root.as_mut().data;
    let n = len.min(d.len().saturating_sub(start));
    d[start..start + n].fill(byte);
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
        // Go through i64 so negative values cast to the right bit pattern
        // (float -> usize `as` saturates).
        let ptr = (n as i64) as usize;
        let bytes = ptr.to_ne_bytes();
        let len = cd.data.len().min(bytes.len());
        cd.data[..len].copy_from_slice(&bytes[..len]);
        let gc_ptr = l.global().heap.cdatas.alloc(cd);
        push(l, LuaValue::cdata(gc_ptr));
        return Ok(1);
    }

    Err(err_bad_arg(l, 2, "ffi.cast", "cdata", ""))
}

/// `__tostring` for cdata: numbers render as `42LL` / `42ULL`, complex
/// as `re+imi`, pointers as addresses, structs/unions/arrays as
/// `cdata<type>: 0x...`, and type values as `ctype<type>`.
fn cdata_tostring(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    let bytes = cdata_tostring_bytes(l, v);
    let sid = l.heap().intern(&bytes);
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn cdata_tostring_bytes(l: &LuaState, v: LuaValue) -> Vec<u8> {
    let Some(cd) = v.as_cdata() else {
        return b"cdata".to_vec();
    };
    let c = cd.as_ref();
    let Some(cts) = l.global().cts.as_ref() else {
        return b"cdata".to_vec();
    };
    if c.ctypeid == CTypeID::CTypeIDType as u32 && c.data.len() >= 4 {
        // A type value: ctype<name>.
        let id = u32::from_le_bytes(c.data[..4].try_into().unwrap_or([0; 4]));
        let name = type_name_of_id(l, id).unwrap_or_else(|| b"void".to_vec());
        return format!("ctype<{}>", String::from_utf8_lossy(&name)).into_bytes();
    }
    let raw = cts.raw(c.ctypeid);
    // Numeric cdata (incl. enums).
    if crate::ffi::ctype_isnum(raw.info) && !crate::ffi::ctype_iscomplex(raw.info) {
        if let Some(n) = crate::stdlib::cdata_to_number(c) {
            let s = crate::strfmt::g14(n);
            let signed = (raw.info & crate::ffi::ctinfo::UNSIGNED) == 0 && raw.size == 8;
            let unsigned = (raw.info & crate::ffi::ctinfo::UNSIGNED) != 0 && raw.size == 8;
            if signed {
                return format!("{}LL", s).into_bytes();
            }
            if unsigned {
                // Print the full unsigned value (possibly > 2^53).
                let mut buf = [0u8; 8];
                buf[..c.data.len().min(8)].copy_from_slice(&c.data[..c.data.len().min(8)]);
                let u = u64::from_le_bytes(buf);
                return format!("{}ULL", u).into_bytes();
            }
            return s.into_bytes();
        }
    }
    // Complex values: `re+imi` (test: "12.5-753.125i", "-0-0i", "inf-infI").
    if crate::ffi::ctype_iscomplex(raw.info) {
        let elem = cts.raw(crate::ffi::ctype_cid(raw.info));
        let esz = elem.size as usize;
        let (off, root) = crate::runtime::cdata::resolve_cdata(c);
        let data = &root.data;
        let read = |rel: usize| -> f64 {
            let i = (off as usize) + rel;
            if esz == 8 && i + 8 <= data.len() {
                f64::from_le_bytes(data[i..i + 8].try_into().unwrap())
            } else if esz == 4 && i + 4 <= data.len() {
                f32::from_le_bytes(data[i..i + 4].try_into().unwrap()) as f64
            } else {
                0.0
            }
        };
        let re = read(0);
        let im = read(esz);
        let fmt = |x: f64| -> String {
            if x.is_nan() {
                "nan".to_string()
            } else if x.is_infinite() {
                if x > 0.0 { "inf" } else { "-inf" }.to_string()
            } else if x == 0.0 && x.is_sign_negative() {
                "-0".to_string()
            } else {
                crate::strfmt::g14(x)
            }
        };
        let sign = if im < 0.0 || (im == 0.0 && im.is_sign_negative()) { "-" } else { "+" };
        // Special values use a capital imaginary suffix ("inf-infI").
        let suffix = if re.is_finite() && im.is_finite() { "i" } else { "I" };
        return format!("{}{}{}{}", fmt(re), sign, fmt(im.abs()), suffix).into_bytes();
    }
    // Pointer / struct / union / array: cdata<type>: 0xADDR.
    let tname = type_name_of_id(l, c.ctypeid).unwrap_or_else(|| b"void".to_vec());
    let addr = c.data.as_ptr() as usize;
    let s = format!(
        "cdata<{}>: 0x{:x}",
        String::from_utf8_lossy(&tname),
        addr
    );
    s.into_bytes()
}

// ---------------------------------------------------------------------------
// cdata metamethods: __index / __newindex
// ---------------------------------------------------------------------------

/// True scalar types (Num/Enum) — arrays, structs, pointers and
/// complex values are returned as sub-cdata aliases.
fn ctype_isscalar(info: u32) -> bool {
    let t = crate::ffi::ctype_type(info);
    t == crate::ffi::CT::Num || t == crate::ffi::CT::Enum
}

/// Look up a field offset in a struct type. Anonymous struct/union
/// fields resolve through the nested type (`union { struct { int lo; }
/// ; ... }` exposes `lo` directly).
fn field_offset(cts: &CTState, ctypeid: u32, name: &str) -> Option<(u32, u32)> {
    let struct_id = cts.resolve_raw_id(ctypeid);
    if let Some(r) = cts.field_names.get(&(struct_id, name.to_string())) {
        return Some(*r);
    }
    // Anonymous nested struct/union fields.
    let st = cts.get(struct_id);
    let mut cur = st.info & 0xFFFF;
    let mut guard = 0;
    while cur != 0 && guard < 64 {
        guard += 1;
        let f = cts.get(cur);
        let ftype = ctype_cid(f.info);
        let ftype_raw = cts.raw(ftype);
        let fk = crate::ffi::ctype_type(ftype_raw.info);
        if fk == crate::ffi::CT::Struct || fk == crate::ffi::CT::Struct {
            if let Some((tid, off)) = field_offset(cts, ftype, name) {
                return Some((tid, f.size + off));
            }
        }
        cur = f.sib as u32;
    }
    None
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

/// An array index from a Lua number or a numeric cdata key (`a[i-1LL]`).
fn index_number(l: &LuaState, key: LuaValue) -> Option<i64> {
    if let Some(n) = key.as_number() {
        return Some(n as i64);
    }
    let cd = key.as_cdata()?;
    let cts = l.global().cts.as_ref()?;
    let raw = cts.raw(cd.as_ref().ctypeid);
    if crate::ffi::ctype_isnum(raw.info) {
        crate::stdlib::cdata_to_number(cd.as_ref()).map(|n| n as i64)
    } else {
        None
    }
}

fn array_element(l: &mut LuaState, cd: GcPtr<CData>, idx: i64) -> LuaResult<i32> {
    let ctypeid = cd.as_ref().ctypeid;
    let cts = l.global().cts.as_ref().unwrap();
    let raw_ct = cts.raw(ctypeid);
    let elem_typeid = if ctype_ispointer(raw_ct.info) {
        ctype_cid(raw_ct.info)
    } else {
        ctypeid
    };
    let elem_sz = cts.raw(elem_typeid).size as usize;
    if elem_sz == 0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    // Pointer-arith aliases resolve into their storage cdata; the
    // resolved address may be negative for `p[-1]` style access.
    let (base_off, root) = crate::runtime::cdata::resolve_ptr(cd);
    let data_len = root.as_ref().data.len() as i64;
    let addr = base_off + idx * elem_sz as i64;
    if addr < 0 || addr + elem_sz as i64 > data_len {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let data = &root.as_ref().data;
    let elem_bytes = data[addr as usize..(addr + elem_sz as i64) as usize].to_vec();
    // Non-scalar elements (structs/arrays) alias the storage so writes
    // through the sub-cdata propagate.
    let elem_raw = cts.raw(elem_typeid);
    if !crate::ffi::ctype_isnum(elem_raw.info) {
        let sub = CData {
            ctypeid: elem_typeid,
            data: Box::new([]),
            base: Some(root),
            offset: addr,
        };
        let p = l.global().heap.cdatas.alloc(sub);
        push(l, LuaValue::cdata(p));
        return Ok(1);
    }
    let sub = CData {
        ctypeid: elem_typeid,
        data: elem_bytes.into_boxed_slice(),
        base: None,
        offset: 0,
    };
    // Scalar elements read back as numbers (except 64-bit integers and
    // complex numbers, which stay cdata to preserve precision).
    let elem_raw = cts.raw(elem_typeid);
    if elem_typeid == crate::ffi::CTypeID::Int64 as u32
        || elem_typeid == crate::ffi::CTypeID::UInt64 as u32
        || crate::ffi::ctype_iscomplex(elem_raw.info)
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

    // Numeric key (or a numeric cdata key) → array element access.
    if let Some(idx) = index_number(l, key) {
        return array_element(l, cd, idx);
    }

    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => {
            push(l, LuaValue::NIL);
            return Ok(1);
        }
    };

    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    // Complex numbers expose `re` / `im` pseudo-fields (element 0/1).
    if crate::ffi::ctype_iscomplex(raw_ct.info) {
        let elem = cts.raw(ctype_cid(raw_ct.info));
        let sz = elem.size as usize;
        let off = match name.as_str() {
            "re" => Some(0u32),
            "im" => Some(sz as u32),
            _ => None,
        };
        if let Some(off) = off {
            let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
            let data = &root.as_ref().data;
            let a = (boff + off as i64) as usize;
            if a + sz <= data.len() {
                let val = if sz == 8 {
                    f64::from_le_bytes(data[a..a + 8].try_into().unwrap())
                } else if sz == 4 {
                    f32::from_le_bytes(data[a..a + 4].try_into().unwrap()) as f64
                } else {
                    read_field_from_slice(data, a as u32, sz)
                };
                push(l, LuaValue::number(val));
            } else {
                push(l, LuaValue::NIL);
            }
            return Ok(1);
        }
    }
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
    if !ctype_isscalar(field_ct.info) {
        // Non-scalar field (struct/array/complex/pointer): return an
        // alias or pointer-value cdata.
        let sz = field_ct.size as usize;
        if is_ptr {
            // A pointer-typed cdata: the field lives in the pointed-to
            // storage at (pointer value + field offset).
            let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
            let data = &root.as_ref().data;
            let a = boff as usize;
            let addr = if a + 8 <= data.len() {
                i64::from_ne_bytes(data[a..a + 8].try_into().unwrap())
            } else {
                0
            };
            let bytes: Vec<u8> = if addr != 0 {
                // The target lies in some cdata's storage or raw memory.
                let mut found: Option<Vec<u8>> = None;
                for cd2 in l.global().heap.cdatas.iter() {
                    let start = cd2.data.as_ptr() as i64;
                    let end = start + cd2.data.len() as i64;
                    let target = addr + offset as i64;
                    if target >= start && target + sz as i64 <= end {
                        let off = (target - start) as usize;
                        found = Some(cd2.data[off..off + sz].to_vec());
                        break;
                    }
                }
                found.unwrap_or_else(|| unsafe {
                    std::slice::from_raw_parts((addr as *const u8).add(offset as usize), sz).to_vec()
                })
            } else {
                vec![0u8; sz]
            };
            let sub = CData {
                ctypeid: field_type_id,
                data: bytes.into_boxed_slice(),
                base: None,
                offset: 0,
            };
            let p = l.global().heap.cdatas.alloc(sub);
            push(l, LuaValue::cdata(p));
            return Ok(1);
        }
        // Alias into this cdata's storage at the field offset.
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let sub = CData {
            ctypeid: field_type_id,
            data: Box::new([]),
            base: Some(root),
            offset: boff + offset as i64,
        };
        let p = l.global().heap.cdatas.alloc(sub);
        push(l, LuaValue::cdata(p));
        return Ok(1);
    }

    let sz = field_ct.size as usize;
    let field_is_fp = crate::ffi::ctype_isfp(field_ct.info);
    let val = if is_ptr {
        // The pointer cdata holds the target address; resolve through
        // the alias when it points into a cdata's storage.
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &root.as_ref().data;
        let base_addr = if boff + 8 <= data.len() as i64 {
            i64::from_ne_bytes(data[(boff as usize)..(boff as usize) + 8].try_into().unwrap())
        } else {
            0
        };
        if base_addr == 0 {
            0.0
        } else {
            // The stored address: if it lies within some cdata storage,
            // read from there; otherwise deref raw memory.
            let mut found: Option<f64> = None;
            for cd2 in l.global().heap.cdatas.iter() {
                let start = cd2.data.as_ptr() as i64;
                let end = start + cd2.data.len() as i64;
                if base_addr >= start && base_addr + sz as i64 <= end {
                    let off = (base_addr - start) as usize + offset as usize;

                    found = Some(if field_is_fp && sz == 8 {
                        f64::from_le_bytes(cd2.data[off..off + 8].try_into().unwrap())
                    } else if field_is_fp && sz == 4 {
                        f32::from_le_bytes(cd2.data[off..off + 4].try_into().unwrap()) as f64
                    } else {
                        read_field_from_slice(&cd2.data, off as u32, sz)
                    });
                    break;
                }
            }
            found.unwrap_or_else(|| unsafe { read_field_value(base_addr as *const u8, offset, sz) })
        }
    } else {
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &root.as_ref().data;
        let a = (boff + offset as i64) as usize;
        if a + sz <= data.len() {
            if field_is_fp && sz == 8 {
                f64::from_le_bytes(data[a..a + 8].try_into().unwrap())
            } else if field_is_fp && sz == 4 {
                f32::from_le_bytes(data[a..a + 4].try_into().unwrap()) as f64
            } else {
                read_field_from_slice(data, a as u32, sz)
            }
        } else {
            0.0
        }
    };

    // 64-bit integer fields stay cdata to preserve precision.
    if (field_type_id == crate::ffi::CTypeID::Int64 as u32
        || field_type_id == crate::ffi::CTypeID::UInt64 as u32)
        && sz == 8
    {
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &root.as_ref().data;
        let a = (boff + offset as i64) as usize;
        if a + 8 <= data.len() {
            let sub = CData {
                ctypeid: field_type_id,
                data: data[a..a + 8].to_vec().into_boxed_slice(),
                base: None,
                offset: 0,
            };
            let p = l.global().heap.cdatas.alloc(sub);
            push(l, LuaValue::cdata(p));
            return Ok(1);
        }
    }

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

    // Numeric key (or a numeric cdata key) → array element write.
    if let Some(idx) = index_number(l, key) {
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
        let (base_off, root) = crate::runtime::cdata::resolve_ptr(cd);
        let addr = base_off + idx * elem_sz as i64;
        let len = root.as_ref().data.len() as i64;
        if addr < 0 || addr + elem_sz as i64 > len {
            return Ok(0);
        }
        let mut ebuf = vec![0u8; elem_sz];
        if let Some(src) = val.as_cdata() {
            let src_raw = cts.raw(src.as_ref().ctypeid);
            let dst_raw = cts.raw(elem_typeid);
            let src_complex = crate::ffi::ctype_iscomplex(src_raw.info);
            let dst_complex = crate::ffi::ctype_iscomplex(dst_raw.info);
            if src_complex && dst_complex {
                // Element-wise float<->double conversion.
                let se = cts.raw(crate::ffi::ctype_cid(src_raw.info));
                let de = cts.raw(crate::ffi::ctype_cid(dst_raw.info));
                let se_sz = se.size as usize;
                let de_sz = de.size as usize;
                for j in 0..2 {
                    let s_off = j * se_sz;
                    let d_off = j * de_sz;
                    if s_off + se_sz <= src.as_ref().data.len() && d_off + de_sz <= ebuf.len() {
                        let v = if se_sz == 8 {
                            f64::from_le_bytes(src.as_ref().data[s_off..s_off + 8].try_into().unwrap())
                        } else {
                            f32::from_le_bytes(src.as_ref().data[s_off..s_off + 4].try_into().unwrap()) as f64
                        };
                        write_scalar_value(&mut ebuf[d_off..d_off + de_sz], crate::ffi::ctype_cid(dst_raw.info), v);
                    }
                }
            } else {
                let n = src.as_ref().data.len().min(elem_sz);
                ebuf[..n].copy_from_slice(&src.as_ref().data[..n]);
            }
        } else if let Some(tab) = val.as_table() {
            // Table initializer for the element (e.g. nested arrays).
            let n = tab.as_ref().len() as usize;
            let sub = cts.raw(elem_typeid);
            if crate::ffi::ctype_iscomplex(sub.info) || crate::ffi::ctype_isarray(sub.info) {
                let eid = crate::ffi::ctype_cid(sub.info);
                let esz = cts.raw(eid).size as usize;
                for j in 0..n {
                    let fv = tab.as_ref().get_int(j as i32 + 1);
                    if let Some(num) = fv.as_number() {
                        let mut b = vec![0u8; esz];
                        write_scalar_value(&mut b, eid, num);
                        let off = j * esz;
                        if off + esz <= ebuf.len() {
                            ebuf[off..off + esz].copy_from_slice(&b);
                        }
                    }
                }
            }
        } else {
            write_scalar_value(&mut ebuf, elem_typeid, val.as_number().unwrap_or(0.0));
        }
        let a = addr as usize;
        root.as_mut().data[a..a + elem_sz].copy_from_slice(&ebuf);
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

    // Complex pseudo-fields `re` / `im`.
    let target_raw = cts.raw(target_id);
    if crate::ffi::ctype_iscomplex(target_raw.info) {
        let off = match name.as_str() {
            "re" => Some(0i64),
            "im" => Some(cts.raw(ctype_cid(target_raw.info)).size as i64),
            _ => None,
        };
        if let Some(off) = off {
            let elem = cts.raw(ctype_cid(target_raw.info));
            let sz = elem.size as usize;
            let mut ebuf = vec![0u8; sz];
            write_scalar_value(&mut ebuf, ctype_cid(target_raw.info), val.as_number().unwrap_or(0.0));
            let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
            let data = &mut root.as_mut().data;
            let a = (boff + off) as usize;
            if a + sz <= data.len() {
                data[a..a + sz].copy_from_slice(&ebuf);
            }
            return Ok(0);
        }
    }

    let Some((field_type_id, offset)) = field_offset(cts, target_id, &name) else {
        return Err(l.runtime_error(format!("ffi: no member '{}' in cdata", name).as_bytes()));
    };

    // A complex field takes a scalar (its real part) or a complex cdata.
    let field_raw = cts.raw(field_type_id);
    if crate::ffi::ctype_iscomplex(field_raw.info) {
        let fe = cts.raw(crate::ffi::ctype_cid(field_raw.info));
        let fsz = fe.size as usize;
        let mut ebuf = vec![0u8; fsz * 2];
        if let Some(src) = val.as_cdata() {
            let n = src.as_ref().data.len().min(ebuf.len());
            ebuf[..n].copy_from_slice(&src.as_ref().data[..n]);
        } else if let Some(n) = val.as_number() {
            write_scalar_value(&mut ebuf, crate::ffi::ctype_cid(field_raw.info), n);
        }
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &mut root.as_mut().data;
        let a = (boff + offset as i64) as usize;
        if a + ebuf.len() <= data.len() {
            data[a..a + ebuf.len()].copy_from_slice(&ebuf);
        }
        return Ok(0);
    }

    let v = val.as_number().unwrap_or(0.0) as i32;
    let ptr_val: Option<i64> = if let Some(src) = val.as_cdata() {
        // A cdata value stored into a pointer field: its storage address.
        let (off, root) = crate::runtime::cdata::resolve_ptr(src);
        Some((root.as_ref().data.as_ptr() as i64).wrapping_add(off))
    } else {
        None
    };

    // A pointer field stores a cdata value's storage address.
    let field_is_ptr = crate::ffi::ctype_isptr(cts.raw(field_type_id).info);
    if field_is_ptr && ptr_val.is_some() {
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &mut root.as_mut().data;
        let a = (boff + offset as i64) as usize;
        let sz = cts.raw(field_type_id).size as usize;
        let pv = ptr_val.unwrap() as usize;

        if sz == 8 && a + 8 <= data.len() {
            data[a..a + 8].copy_from_slice(&pv.to_ne_bytes());
        } else if sz == 4 && a + 4 <= data.len() {
            data[a..a + 4].copy_from_slice(&(pv as u32).to_le_bytes());
        }
        return Ok(0);
    }

    // Pointer-typed cdata: resolve the stored address and write into
    // the pointed-to cdata's storage.
    if crate::ffi::ctype_ispointer(cts.raw(cd.as_ref().ctypeid).info) {
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &root.as_ref().data;
        let a = boff as usize;
        let addr = if a + 8 <= data.len() {
            i64::from_ne_bytes(data[a..a + 8].try_into().unwrap())
        } else {
            0
        };

        if addr != 0 {
            let fsz = cts.raw(field_type_id).size as usize;
            let mut ebuf = vec![0u8; fsz.max(1)];
            write_scalar_value(&mut ebuf, field_type_id, val.as_number().unwrap_or(0.0));
            // The address points into some cdata's (stable) storage or
            // into raw memory: write there directly.
            unsafe {
                let dp = (addr as *mut u8).add(offset as usize);
                std::ptr::copy_nonoverlapping(ebuf.as_ptr(), dp, ebuf.len());
            }
        }
        return Ok(0);
    }
    // Non-scalar value (e.g. a complex cdata) → copy the whole field.
    if let Some(src) = val.as_cdata() {
        let sz = cts.raw(field_type_id).size as usize;
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &mut root.as_mut().data;
        let a = (boff + offset as i64) as usize;
        if a + sz <= data.len() {
            let n = src.as_ref().data.len().min(sz);
            data[a..a + n].copy_from_slice(&src.as_ref().data[..n]);
        }
        return Ok(0);
    }
    {
        // Numeric field write (resolve pointer-arith aliases), using the
        // field's own type size.
        let (boff, root) = crate::runtime::cdata::resolve_ptr(cd);
        let data = &mut root.as_mut().data;
        let a = (boff + offset as i64) as usize;
        let fsz = cts.raw(field_type_id).size as usize;
        let mut ebuf = vec![0u8; fsz.max(1)];
        write_scalar_value(&mut ebuf, field_type_id, val.as_number().unwrap_or(0.0));
        if a + ebuf.len() <= data.len() {
            data[a..a + ebuf.len()].copy_from_slice(&ebuf);
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

    // If the symbol was declared in a cdef, keep its prototype so the
    // call wrapper can validate arguments against the parameter types
    // (lj_ccall's conversion; e.g. number -> pointer is not allowed).
    // The asm("name") override decides the C symbol to resolve.
    let ctypeid = cts_of(l).names.get(&name).copied();
    let sym_name = cts_of(l).symbols.get(&name).cloned().unwrap_or(name.clone());
    let addr = clib::resolve_symbol(&sym_name).unwrap_or(0) as f64;
    if addr == 0.0 {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    let mut upvals = vec![LuaValue::number(addr)];
    if let Some(id) = ctypeid {
        upvals.push(LuaValue::number(id as f64));
    }

    let g = l.global() as *mut GlobalState;
    let env = unsafe { (*g).globals };
    let clos = unsafe {
        (*g).heap.alloc_func(GcFunc::C(CClosure {
            f: call_c,
            env,
            upvals,
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
/// When the symbol was declared in a cdef (upvalue 1 = ctypeid), validate
/// each argument against the declared parameter types first (lj_cconv: a
/// plain number cannot convert to a pointer type without a cast).
fn call_c(l: &mut LuaState) -> LuaResult<i32> {
    let addr = l.upvalue(0).as_number().unwrap() as usize;
    let narg = l.top - l.base;
    let decl_id = l.upvalue(1).as_number().map(|n| n as u32);

    // Collect the declared parameter types: the func's field chain.
    let param_types: Vec<u32> = if let Some(fid) = decl_id {
        let mut types = Vec::new();
        let cts = l.global().cts.as_ref().unwrap();
        let raw = cts.raw(fid);

        if raw.info >> 28 == CT::Func as u32 {
            let mut cur = raw.sib as u32;
            while cur != 0 {
                let f = cts.tab.get(cur as usize);
                if let Some(f) = f {
                    types.push(f.info & 0xFFFF);
                    cur = f.sib as u32;
                } else {
                    break;
                }
            }
        }
        types
    } else {
        Vec::new()
    };

    // Marshal each Lua argument to an i64 slot.
    let mut cstrs: Vec<CString> = Vec::new();
    let mut cargs: Vec<i64> = Vec::with_capacity(narg);
    for i in 0..narg {
        let a = arg(l, i);
        if let Some(ptype) = param_types.get(i).copied() {
            // Declared parameter type: pointer params accept cdata /
            // string / nil / lightuserdata, but not plain numbers
            // (lj_cconv.c CCX(P,I) without CCF_CAST).
            let ptype = ptype & 0xFFFF;
            if let Some(cd) = a.as_cdata() {
                cargs.push(cd.as_ref().data.as_ptr() as i64);
                continue;
            }
            let is_ptr = {
                let cts = l.global().cts.as_ref().unwrap();
                let raw = cts.raw(ptype);
                raw.info >> 28 == CT::Ptr as u32 || raw.info >> 28 == CT::Func as u32
            };
            if is_ptr {
                if let Some(sid) = a.as_string_id() {
                    let bytes = l.heap().strings.get(sid).to_vec();
                    let cs = CString::new(bytes)
                        .map_err(|_| l.runtime_error(b"ffi: string contains null byte"))?;
                    cargs.push(cs.as_ptr() as i64);
                    cstrs.push(cs);
                    continue;
                }
                if a.is_nil() {
                    cargs.push(0);
                    continue;
                }
                let tname = type_name_of(l, a);
                let pname = ptr_name_of(l, ptype);
                return Err(l.runtime_error(
                    format!("cannot convert '{}' to '{}'", tname, pname).as_bytes(),
                ));
            }
            // Non-pointer declared type: numbers/cdata convert by value.
            if let Some(cd) = a.as_cdata() {
                let d = &cd.as_ref().data;
                let mut buf = [0u8; 8];
                let n = d.len().min(8);
                buf[..n].copy_from_slice(&d[..n]);
                cargs.push(i64::from_le_bytes(buf));
                continue;
            }
            if let Some(n) = a.as_number() {
                cargs.push(n as i64);
                continue;
            }
            if a.is_nil() {
                cargs.push(0);
                continue;
            }
            let tname = type_name_of(l, a);
            return Err(l.runtime_error(
                format!("cannot convert '{}' to '{}'", tname, "number").as_bytes(),
            ));
        }
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

fn type_name_of(l: &LuaState, v: LuaValue) -> String {
    if v.is_number() {
        "number".to_string()
    } else if v.is_string() {
        "string".to_string()
    } else if v.is_nil() {
        "nil".to_string()
    } else if is_boolean(v) {
        "boolean".to_string()
    } else if v.is_table() {
        "table".to_string()
    } else if v.is_func() {
        "function".to_string()
    } else if v.as_cdata().is_some() {
        "cdata".to_string()
    } else {
        "userdata".to_string()
    }
}

fn is_boolean(v: LuaValue) -> bool {
    v.itype() == crate::value::LJ_TTRUE || v.itype() == crate::value::LJ_TFALSE
}

fn ptr_name_of(l: &LuaState, ptype: u32) -> String {
    let cts = l.global().cts.as_ref().unwrap();
    let raw = cts.raw(ptype);
    let pointee = raw.info & 0xFFFF;
    let name = cts
        .names
        .iter()
        .find(|(_, id)| **id == pointee)
        .map(|(n, _)| n.clone());
    match name {
        Some(n) => format!("{} *", n),
        None => "void *".to_string(),
    }
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
        l.global().ffi_errno = n;
        push(l, LuaValue::number(n as f64));
    } else if v.is_nil() {
        push(l, LuaValue::number(l.global().ffi_errno as f64));
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
    let cdata_mt = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 4)) };
    {
        let g = unsafe { &mut *g };
        let index_k = g.mmname[MM::Index as usize];
        let newindex_k = g.mmname[MM::Newindex as usize];
        let call_k = g.mmname[MM::Call as usize];
        let tostring_k = g.mmname[MM::Tostring as usize];
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
        let tostring_fn = g.heap.alloc_func(GcFunc::C(CClosure {
            f: cdata_tostring,
            env,
            upvals: vec![],
        }));
        cdata_mt.as_mut().set_str(index_k, LuaValue::func(index_fn));
        cdata_mt
            .as_mut()
            .set_str(newindex_k, LuaValue::func(newindex_fn));
        cdata_mt.as_mut().set_str(call_k, LuaValue::func(call_fn));
        cdata_mt
            .as_mut()
            .set_str(tostring_k, LuaValue::func(tostring_fn));
        g.set_basemt(LJ_TCDATA, Some(cdata_mt));
    }

    let heap = unsafe { &mut (*g).heap };

    // -- ffi table ------------------------------------------------------------
    let ffi_tab = heap.alloc_table(LuaTable::new(0, 5)); // 20 entries → 32 nodes
    let builtins: [(&[u8], CFunction); 18] = [
        (b"cdef", ffi_cdef),
        (b"new", ffi_new),
        (b"sizeof", ffi_sizeof),
        (b"alignof", ffi_alignof),
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
        upvals: vec![LuaValue::table(ffi_tab)],
    }));

    // -- init default C lib handles (Windows) ---------------------------------
    #[cfg(windows)]
    unsafe {
        clib::init_default_libs();
    }

    // -- wire everything together ---------------------------------------------
    {
        let globals = unsafe { (*g).globals.as_mut() };

        // Note: LuaJIT exposes `ffi` as a global; the test suite's
        // contents check expects it *not* to be one, so only the
        // package.preload loader registers it.
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
    let t = l.upvalue(0).as_table().unwrap();
    l.stack[l.base] = LuaValue::table(t);
    l.top = l.base + 1;
    Ok(1)
}
