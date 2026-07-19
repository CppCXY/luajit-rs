//! Lua FFI library — `ffi.*` API functions.

use crate::err::{LuaError, LuaResult};
use crate::ffi::parser::parse;
use crate::runtime::cdata::CData;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::stdlib::{arg, push, nargs, err_bad_arg};
use crate::lual_reg;

fn get_cts(l: &mut LuaState) -> &mut crate::ffi::CTState {
    l.global().cts.get_or_insert_with(|| crate::ffi::CTState::new())
}

fn quick_type_id(name: &str) -> Option<u32> {
    Some(match name {
        "void" => crate::ffi::CTypeID::Void as u32,
        "bool"|"_Bool" => crate::ffi::CTypeID::Bool as u32,
        "char" => crate::ffi::CTypeID::CChar as u32,
        "signed char"|"int8_t" => crate::ffi::CTypeID::Int8 as u32,
        "unsigned char"|"uint8_t" => crate::ffi::CTypeID::UInt8 as u32,
        "short"|"int16_t" => crate::ffi::CTypeID::Int16 as u32,
        "unsigned short"|"uint16_t" => crate::ffi::CTypeID::UInt16 as u32,
        "int"|"signed"|"int32_t" => crate::ffi::CTypeID::Int32 as u32,
        "unsigned"|"unsigned int"|"uint32_t" => crate::ffi::CTypeID::UInt32 as u32,
        "long"|"int64_t" => crate::ffi::CTypeID::Int64 as u32,
        "unsigned long"|"uint64_t" => crate::ffi::CTypeID::UInt64 as u32,
        "long long" => crate::ffi::CTypeID::Int64 as u32,
        "unsigned long long" => crate::ffi::CTypeID::UInt64 as u32,
        "float" => crate::ffi::CTypeID::Float as u32,
        "double" => crate::ffi::CTypeID::Double as u32,
        _ => return None,
    })
}

fn ffi_checkctype(l: &mut LuaState) -> LuaResult<u32> {
    let o = arg(l, 0);
    if o.is_string() {
        let sid = o.as_string_id().unwrap();
        let name_v = { let b = l.heap().strings.get(sid).to_vec(); std::str::from_utf8(&b).map_err(|_| LuaError::Runtime)?.trim().to_string() };
        let name = name_v.as_str();
        if let Some(id) = quick_type_id(name) { return Ok(id); }
        {
            let g = l.global();
            if let Some(cts) = g.cts.as_ref() { if let Some(&id) = cts.names.get(name) { return Ok(id); } }
        }
        let cts = get_cts(l); let prev = cts.top;
        parse(cts, name).map_err(|_| LuaError::Runtime)?;
        if cts.top > prev { Ok(cts.top - 1) } else { Err(err_bad_arg(l, 1, "ffi", "C type", "")) }
    } else if o.is_cdata() { Ok(o.as_cdata().unwrap().as_ref().ctypeid) }
    else { Err(err_bad_arg(l, 1, "ffi", "C type", "")) }
}

pub fn ffi_cdef(l: &mut LuaState) -> LuaResult<i32> {
    let o = arg(l, 0); let sid = o.as_string_id().ok_or_else(|| err_bad_arg(l, 1, "ffi.cdef", "string", ""))?;
    let s = l.heap().strings.get(sid).to_vec();
    parse(get_cts(l), std::str::from_utf8(&s).map_err(|_| LuaError::Runtime)?).map_err(|_| LuaError::Runtime)?;
    Ok(0)
}

pub fn ffi_new(l: &mut LuaState) -> LuaResult<i32> {
    let id = ffi_checkctype(l)?;
    let sz = { let c = get_cts(l); let t = c.raw(id); (t.size != u32::MAX).then_some(t.size as usize).unwrap_or(0) };
    let ptr = { let g = l.global(); g.heap.cdatas.alloc(CData::new(id, sz.max(1))) };
    push(l, LuaValue::cdata(ptr)); Ok(1)
}

pub fn ffi_sizeof(l: &mut LuaState) -> LuaResult<i32> {
    let id = ffi_checkctype(l)?;
    let sz;
    { let cts = get_cts(l); sz = cts.raw(id).size; }
    push(l, LuaValue::number(sz as f64)); Ok(1)
}

pub fn ffi_alignof(l: &mut LuaState) -> LuaResult<i32> {
    let id = ffi_checkctype(l)?;
    let al;
    { let cts = get_cts(l); al = 1u32 << crate::ffi::ctype_align(cts.raw(id).info); }
    push(l, LuaValue::number(al as f64)); Ok(1)
}

pub fn ffi_typeof(l: &mut LuaState) -> LuaResult<i32> {
    let id = ffi_checkctype(l)?;
    let mut cd = CData::new(crate::ffi::CTypeID::CTypeIDType as u32, 4);
    cd.data[..4].copy_from_slice(&(id as u32).to_le_bytes());
    let ptr = { let g = l.global(); g.heap.cdatas.alloc(cd) };
    push(l, LuaValue::cdata(ptr)); Ok(1)
}

pub fn ffi_istype(l: &mut LuaState) -> LuaResult<i32> {
    let id = ffi_checkctype(l)?;
    push(l, if arg(l, 1).as_cdata().map_or(false, |cd| cd.as_ref().ctypeid == id) { LuaValue::TRUE } else { LuaValue::FALSE }); Ok(1)
}

pub fn ffi_string(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0).as_cdata().ok_or_else(|| err_bad_arg(l, 1, "ffi.string", "cdata", ""))?;
    let ptr = cd.as_ref().get_ptr() as *const u8;
    let len = if nargs(l) > 1 { arg(l, 1).as_number().unwrap_or(0.0) as usize } else {
        let mut n = 0; while n < 4096 && !ptr.is_null() && unsafe { *ptr.add(n) } != 0 { n += 1; } n
    };
    if ptr.is_null() || len == 0 { push(l, LuaValue::NIL); return Ok(1); }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let v = { let h = l.heap(); let sid = h.strings.intern(&bytes); h.str_value(sid) };
    push(l, v); Ok(1)
}

pub fn ffi_copy(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0).as_cdata().ok_or_else(|| err_bad_arg(l, 1, "ffi.copy", "cdata", ""))?;
    let src = arg(l, 1).as_cdata().ok_or_else(|| err_bad_arg(l, 2, "ffi.copy", "cdata", ""))?;
    let len = arg(l, 2).as_number().unwrap_or(0.0) as usize;
    let (dp, sp) = (dst.as_ref().get_ptr() as *mut u8, src.as_ref().get_ptr() as *const u8);
    if !dp.is_null() && !sp.is_null() && len > 0 { unsafe { std::ptr::copy_nonoverlapping(sp, dp, len); } }
    Ok(0)
}

pub fn ffi_fill(l: &mut LuaState) -> LuaResult<i32> {
    let dst = arg(l, 0).as_cdata().ok_or_else(|| err_bad_arg(l, 1, "ffi.fill", "cdata", ""))?;
    let len = arg(l, 1).as_number().unwrap_or(0.0) as usize;
    let byte = if nargs(l) > 2 { arg(l, 2).as_number().map(|n| n as u8).unwrap_or(0) } else { 0 };
    let dp = dst.as_ref().get_ptr() as *mut u8;
    if !dp.is_null() && len > 0 { unsafe { std::ptr::write_bytes(dp, byte, len); } }
    Ok(0)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"ffi", crate::stdlib::LibTarget::Global)
        .func(b"cdef", ffi_cdef)
        .func(b"new", ffi_new)
        .func(b"sizeof", ffi_sizeof)
        .func(b"alignof", ffi_alignof)
        .func(b"typeid", ffi_typeof)
        .func(b"istype", ffi_istype)
        .func(b"string", ffi_string)
        .func(b"copy", ffi_copy)
        .func(b"fill", ffi_fill)
        .build();
}

#[cfg(test)]
mod tests {
    use crate::state::Lua;

    fn run(l: &mut crate::state::LuaState, src: &[u8]) {
        let f = crate::state::load(l, src.to_vec(), "@t").unwrap();
        crate::vm::call(l, f, &[]).unwrap();
    }

    #[test] fn ffi_sizeof() { let mut lua = Lua::new(); crate::open_libs(lua.main()); run(lua.main(), b"assert(ffi.sizeof('int')==4) assert(ffi.sizeof('double')==8)"); }
    #[test] fn ffi_typedef() { let mut lua = Lua::new(); crate::open_libs(lua.main()); run(lua.main(), b"ffi.cdef('typedef int mi;') assert(ffi.sizeof('mi')==4)"); }
    #[test] fn ffi_struct() { let mut lua = Lua::new(); crate::open_libs(lua.main()); run(lua.main(), b"ffi.cdef('typedef struct{int x;double y;}s;') assert(ffi.sizeof('s')==16) assert(ffi.alignof('s')==8)"); }
    #[test] fn ffi_new() { let mut lua = Lua::new(); crate::open_libs(lua.main()); run(lua.main(), b"ffi.cdef('typedef int mi;') local o=ffi.new('int') assert(ffi.istype('int',o))"); }
    #[test]
    fn ffi_perf() {
        let mut lua = Lua::new(); crate::open_libs(lua.main()); let l = lua.main();
        let n = 200000u64; let code = format!("local t=ffi.sizeof('int')local s=0 for i=1,{} do s=t end return s", n);
        let start = std::time::Instant::now();
        let f = crate::state::load(l, code.into_bytes(), "@p").unwrap();
        crate::vm::call(l, f, &[]).unwrap();
        eprintln!("ffi_perf: {} ops {:.3?} = {:.0}/s", n, start.elapsed(), n as f64 / start.elapsed().as_secs_f64());
    }
}
