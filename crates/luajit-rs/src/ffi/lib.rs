//! Lua FFI library — loaded via `require("ffi")`.

use crate::err::{LuaError, LuaResult};
use crate::ffi::parser::parse;
use crate::runtime::cdata::CData;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::stdlib::{arg, push, nargs, err_bad_arg};

fn get_cts(l: &mut LuaState) -> &mut crate::ffi::CTState { l.global().cts.get_or_insert_with(crate::ffi::CTState::new) }
fn quick_type_id(n: &str) -> Option<u32> { Some(match n {
    "void"=>crate::ffi::CTypeID::Void as u32,"bool"|"_Bool"=>crate::ffi::CTypeID::Bool as u32,"char"=>crate::ffi::CTypeID::CChar as u32,
    "signed char"|"int8_t"=>crate::ffi::CTypeID::Int8 as u32,"unsigned char"|"uint8_t"=>crate::ffi::CTypeID::UInt8 as u32,
    "short"|"int16_t"=>crate::ffi::CTypeID::Int16 as u32,"unsigned short"|"uint16_t"=>crate::ffi::CTypeID::UInt16 as u32,
    "int"|"signed"|"int32_t"=>crate::ffi::CTypeID::Int32 as u32,"unsigned"|"unsigned int"|"uint32_t"=>crate::ffi::CTypeID::UInt32 as u32,
    "long"|"int64_t"=>crate::ffi::CTypeID::Int64 as u32,"unsigned long"|"uint64_t"=>crate::ffi::CTypeID::UInt64 as u32,
    "long long"=>crate::ffi::CTypeID::Int64 as u32,"unsigned long long"=>crate::ffi::CTypeID::UInt64 as u32,
    "float"=>crate::ffi::CTypeID::Float as u32,"double"=>crate::ffi::CTypeID::Double as u32,_=>return None
}) }

fn ffi_checkctype(l: &mut LuaState) -> LuaResult<u32> {
    let o=arg(l,0); if o.is_string() {
        let sid=o.as_string_id().unwrap();let n={let b=l.heap().strings.get(sid).to_vec();std::str::from_utf8(&b).map_err(|_|LuaError::Runtime)?.trim().to_string()};
        if let Some(id)=quick_type_id(&n){return Ok(id)}
        if let Some(&id)=l.global().cts.as_ref().and_then(|c|c.names.get(&n)){return Ok(id)}
        let cts=get_cts(l);let p=cts.top;parse(cts,&n).map_err(|_|LuaError::Runtime)?;if cts.top>p{Ok(cts.top-1)}else{Err(err_bad_arg(l,1,"ffi","C type",""))}
    }else if o.is_cdata(){Ok(o.as_cdata().unwrap().as_ref().ctypeid)}else{Err(err_bad_arg(l,1,"ffi","C type",""))}
}
pub fn ffi_cdef(l:&mut LuaState)->LuaResult<i32>{let sid=arg(l,0).as_string_id().ok_or_else(||err_bad_arg(l,1,"ffi.cdef","string",""))?;let s=l.heap().strings.get(sid).to_vec();parse(get_cts(l),std::str::from_utf8(&s).map_err(|_|LuaError::Runtime)?).map_err(|_|LuaError::Runtime)?;Ok(0)}
pub fn ffi_new(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let sz={let ct=get_cts(l).raw(id);if ct.size != u32::MAX { ct.size as usize } else { 0 }};let p={l.global().heap.cdatas.alloc(CData::new(id,sz.max(1)))};push(l,LuaValue::cdata(p));Ok(1)}
pub fn ffi_sizeof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let sz;{let c=get_cts(l);sz=c.raw(id).size}push(l,LuaValue::number(sz as f64));Ok(1)}
pub fn ffi_alignof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let al;{let c=get_cts(l);al=1u32<<crate::ffi::ctype_align(c.raw(id).info)}push(l,LuaValue::number(al as f64));Ok(1)}
pub fn ffi_typeof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let mut cd=CData::new(crate::ffi::CTypeID::CTypeIDType as u32,4);cd.data[..4].copy_from_slice(&(id as u32).to_le_bytes());let p={l.global().heap.cdatas.alloc(cd)};push(l,LuaValue::cdata(p));Ok(1)}
pub fn ffi_istype(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;push(l,if arg(l,1).as_cdata().is_some_and(|cd|cd.as_ref().ctypeid==id){LuaValue::TRUE}else{LuaValue::FALSE});Ok(1)}
pub fn ffi_string(l:&mut LuaState)->LuaResult<i32>{let cd=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.string","cdata",""))?;let p=cd.as_ref().get_ptr()as*const u8;let len=if nargs(l)>1{arg(l,1).as_number().unwrap_or(0.0)as usize}else{let mut n=0;while n<4096&&!p.is_null()&&unsafe{*p.add(n)}!=0{n+=1}n};if p.is_null()||len==0{push(l,LuaValue::NIL);return Ok(1)}let v={let h=l.heap();let sid=h.strings.intern(unsafe{std::slice::from_raw_parts(p,len)});h.str_value(sid)};push(l,v);Ok(1)}
pub fn ffi_copy(l:&mut LuaState)->LuaResult<i32>{let dp=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.copy","cdata",""))?.as_ref().get_ptr()as*mut u8;let sp=arg(l,1).as_cdata().ok_or_else(||err_bad_arg(l,2,"ffi.copy","cdata",""))?.as_ref().get_ptr()as*const u8;let len=arg(l,2).as_number().unwrap_or(0.0)as usize;if!dp.is_null()&&!sp.is_null()&&len>0{unsafe{std::ptr::copy_nonoverlapping(sp,dp,len)}}Ok(0)}
pub fn ffi_fill(l:&mut LuaState)->LuaResult<i32>{let dp=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.fill","cdata",""))?.as_ref().get_ptr()as*mut u8;let len=arg(l,1).as_number().unwrap_or(0.0)as usize;let b=if nargs(l)>2{arg(l,2).as_number().map(|n|n as u8).unwrap_or(0)}else{0};if!dp.is_null()&&len>0{unsafe{std::ptr::write_bytes(dp,b,len)}}Ok(0)}

pub fn open(l: &mut LuaState) {
    use crate::func::{CClosure, GcFunc};
    use crate::table::LuaTable;
    let g = l.global() as *mut crate::state::GlobalState;
    let env = unsafe { (*g).globals };

    // ---- phase 1: heap allocations ----
    let tab = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 16)) };
    let funcs: [(&[u8], crate::func::CFunction); 9] = [(b"cdef",ffi_cdef),(b"new",ffi_new),(b"sizeof",ffi_sizeof),(b"alignof",ffi_alignof),(b"typeid",ffi_typeof),(b"istype",ffi_istype),(b"string",ffi_string),(b"copy",ffi_copy),(b"fill",ffi_fill)];
    for &(n,f) in &funcs {
        let sid = unsafe { (*g).heap.intern(n) };
        let kv = unsafe { (*g).heap.str_value(sid) };
        let fr = unsafe { (*g).heap.alloc_func(GcFunc::C(CClosure{f,env,upvals:vec![]})) };
        tab.as_mut().set(kv, LuaValue::func(fr));
    }
    let ffi_k = unsafe { let s=(*g).heap.intern(b"ffi"); (*g).heap.str_value(s) };
    let pk_k = unsafe { let s=(*g).heap.intern(b"package"); (*g).heap.str_value(s) };
    let pr_k = unsafe { let s=(*g).heap.intern(b"preload"); (*g).heap.str_value(s) };
    let pk_tab = unsafe { (*g).heap.alloc_table(LuaTable::new(2,1)) };
    let pr_tab = unsafe { (*g).heap.alloc_table(LuaTable::new(8,3)) };
    let ld = unsafe { (*g).heap.alloc_func(GcFunc::C(CClosure{f:preload_loader,env,upvals:vec![]})) };
    let c_k = unsafe { let s=(*g).heap.intern(b"C"); (*g).heap.str_value(s) };
    let c_tab = unsafe { (*g).heap.alloc_table(LuaTable::new(0,4)) };

    // Load printf from default C library
    #[cfg(windows)] unsafe { crate::ffi::clib::init_default_libs(); }
    let addr = crate::ffi::clib::resolve_symbol("printf").unwrap_or(0) as f64;
    if addr != 0.0 {
        let printf_k = unsafe { let s=(*g).heap.intern(b"printf"); (*g).heap.str_value(s) };
        let printf_f = unsafe { (*g).heap.alloc_func(GcFunc::C(CClosure{
            f: call_printf, env, upvals: vec![LuaValue::number(addr)],
        })) };
        c_tab.as_mut().set(printf_k, LuaValue::func(printf_f));
    }

    // ---- phase 2: table mutations ----
    unsafe { (*g).globals.as_mut() }.set(ffi_k, LuaValue::table(tab));
    tab.as_mut().set(c_k, LuaValue::table(c_tab));
    unsafe {
        let gl = (*g).globals.as_mut();
        if gl.get(pk_k).as_table().is_none() { gl.set(pk_k, LuaValue::table(pk_tab)); }
    }
    let pk = unsafe { (*g).globals.as_ref().get(pk_k).as_table().unwrap() };
    { let t = pk.as_mut(); if t.get(pr_k).as_table().is_none() { t.set(pr_k, LuaValue::table(pr_tab)); } }
    let pr = pk.as_ref().get(pr_k).as_table().unwrap();
    pr.as_mut().set(ffi_k, LuaValue::func(ld));
}

fn preload_loader(l:&mut LuaState)->LuaResult<i32>{
    let g=l.global();let sid=g.heap.intern(b"ffi");let k=g.heap.str_value(sid);
    let t=g.globals.as_ref().get(k).as_table().unwrap();
    l.stack[l.base]=LuaValue::table(t);l.top=l.base+1;Ok(1)
}

fn call_printf(l: &mut LuaState) -> LuaResult<i32> {
    let addr = l.upvalue(0).as_number().unwrap() as usize;
    let fn_ptr: unsafe extern "system" fn(*const std::ffi::c_char, ...) -> i32 = unsafe { std::mem::transmute(addr) };
    let fmt_sid = arg(l,0).as_string_id().ok_or(LuaError::Runtime)?;
    let fmt_bytes = l.heap().strings.get(fmt_sid).to_vec();
    let fmt_cstr = std::ffi::CString::new(fmt_bytes).map_err(|_| LuaError::Runtime)?;
    let mut cstrs: Vec<std::ffi::CString> = Vec::new();
    for i in 1..(l.top - l.base) {
        if let Some(sid) = arg(l, i).as_string_id() {
            cstrs.push(std::ffi::CString::new(l.heap().strings.get(sid).to_vec()).unwrap());
        }
    }
    let r = match cstrs.len() { 0=>unsafe{fn_ptr(fmt_cstr.as_ptr())}, 1=>unsafe{fn_ptr(fmt_cstr.as_ptr(),cstrs[0].as_ptr())}, _=>unsafe{fn_ptr(fmt_cstr.as_ptr(),cstrs[0].as_ptr(),cstrs[1].as_ptr())} };
    push(l, LuaValue::number(r as f64)); Ok(1)
}
