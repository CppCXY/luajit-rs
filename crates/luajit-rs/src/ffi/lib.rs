//! Lua FFI library — loaded via `require("ffi")`.

use crate::err::{LuaError, LuaResult};
use crate::ffi::parser::parse;
use crate::runtime::cdata::CData;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::stdlib::{arg, push, nargs, err_bad_arg};
use crate::ffi::{ctype_isnum, ctype_isptr};

fn get_cts(l: &mut LuaState) -> &mut crate::ffi::CTState { l.global().cts.get_or_insert_with(crate::ffi::CTState::new) }
fn quick_type_id(n: &str) -> Option<u32> { Some(match n {
    "void"=>crate::ffi::CTypeID::Void as u32,"bool"|"_Bool"=>crate::ffi::CTypeID::Bool as u32,"char"=>crate::ffi::CTypeID::CChar as u32,
    "signed char"|"int8_t"=>crate::ffi::CTypeID::Int8 as u32,"unsigned char"|"uint8_t"=>crate::ffi::CTypeID::UInt8 as u32,
    "short"|"int16_t"=>crate::ffi::CTypeID::Int16 as u32,"unsigned short"|"uint16_t"=>crate::ffi::CTypeID::UInt16 as u32,
    "int"|"signed"|"int32_t"=>crate::ffi::CTypeID::Int32 as u32,"unsigned"|"unsigned int"|"uint32_t"=>crate::ffi::CTypeID::UInt32 as u32,
    "long"|"int64_t"=>crate::ffi::CTypeID::Int64 as u32,"unsigned long"|"uint64_t"=>crate::ffi::CTypeID::UInt64 as u32,
    "long long"=>crate::ffi::CTypeID::Int64 as u32,"unsigned long long"=>crate::ffi::CTypeID::UInt64 as u32,
    "float"=>crate::ffi::CTypeID::Float as u32,"double"=>crate::ffi::CTypeID::Double as u32,
    "void *"|"void*"=>crate::ffi::CTypeID::PVoid as u32,_=>return None
}) }

fn ffi_checkctype(l: &mut LuaState) -> LuaResult<u32> {
    let o=arg(l,0); if o.is_string() {
        let sid=o.as_string_id().unwrap();let n={let b=l.heap().strings.get(sid).to_vec();std::str::from_utf8(&b).map_err(|_|LuaError::Runtime)?.trim().to_string()};
        // Handle array suffix: "type[N]" or "type[?]" → return base type
        let base_name = if let Some(idx)=n.find('[') { n[..idx].trim().to_string() } else { n.clone() };
        // Handle pointer suffix: "Type*" or "Type *"
        let (base_name, is_ptr) = if let Some(stripped) = base_name.strip_suffix('*') {
            (stripped.trim().to_string(), true)
        } else if let Some(stripped) = base_name.strip_suffix(" *") {
            (stripped.trim().to_string(), true)
        } else {
            (base_name, false)
        };
        if let Some(id)=quick_type_id(&base_name){return if is_ptr {
            let cts=get_cts(l);
            let ptr_id=cts.top;
            let info=crate::ffi::ct_info(crate::ffi::CT::Ptr,3<<16)|id;
            cts.tab.push(crate::ffi::CType{info,size:8,sib:0,next:0,name:0});
            cts.top+=1;
            Ok(ptr_id)
        } else { Ok(id) }}
        if let Some(&id)=l.global().cts.as_ref().and_then(|c|c.names.get(&base_name)){return if is_ptr {
            let cts=get_cts(l);
            let ptr_id=cts.top;
            let info=crate::ffi::ct_info(crate::ffi::CT::Ptr,3<<16)|id;
            cts.tab.push(crate::ffi::CType{info,size:8,sib:0,next:0,name:0});
            cts.top+=1;
            Ok(ptr_id)
        } else { Ok(id) }}
        let cts=get_cts(l);let p=cts.top;parse(cts,&base_name).map_err(|_|LuaError::Runtime)?;if cts.top>p{Ok(cts.top-1)}else{Err(err_bad_arg(l,1,"ffi","C type",""))}
    }else if o.is_cdata(){Ok(o.as_cdata().unwrap().as_ref().ctypeid)}else{Err(err_bad_arg(l,1,"ffi","C type",""))}
}
pub fn ffi_cdef(l:&mut LuaState)->LuaResult<i32>{let sid=arg(l,0).as_string_id().ok_or_else(||err_bad_arg(l,1,"ffi.cdef","string",""))?;let s=l.heap().strings.get(sid).to_vec();parse(get_cts(l),std::str::from_utf8(&s).map_err(|_|LuaError::Runtime)?).map_err(|_|LuaError::Runtime)?;Ok(0)}
pub fn ffi_new(l:&mut LuaState)->LuaResult<i32>{
    let id=ffi_checkctype(l)?;
    let sz={let ct=get_cts(l).raw(id);if ct.size != u32::MAX { ct.size as usize } else { 0 }};
    let mut cd=CData::new(id,sz.max(1));
    if nargs(l)>1{
        let v2=arg(l,1);
        if let Some(tab)=v2.as_table(){
            let n=tab.as_ref().len() as u32;
            for i in 0u32..n{
                let v=tab.as_ref().get_int(i as i32 + 1);
                if i as usize*4+4<=cd.data.len(){
                    let val=v.as_number().unwrap_or(0.0)as i32;
                    cd.data[i as usize*4..i as usize*4+4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }else if let Some(count)=v2.as_number(){
            let count=count as usize;
            if count>0{
                cd=CData::new(id,count);
            }
        }
    }
    let p=l.global().heap.cdatas.alloc(cd);push(l,LuaValue::cdata(p));Ok(1)
}
pub fn ffi_sizeof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let sz;{let c=get_cts(l);sz=c.raw(id).size}push(l,LuaValue::number(sz as f64));Ok(1)}
pub fn ffi_alignof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let al;{let c=get_cts(l);al=1u32<<crate::ffi::ctype_align(c.raw(id).info)}push(l,LuaValue::number(al as f64));Ok(1)}
pub fn ffi_typeof(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;let mut cd=CData::new(crate::ffi::CTypeID::CTypeIDType as u32,4);cd.data[..4].copy_from_slice(&(id as u32).to_le_bytes());let p={l.global().heap.cdatas.alloc(cd)};push(l,LuaValue::cdata(p));Ok(1)}
pub fn ffi_istype(l:&mut LuaState)->LuaResult<i32>{let id=ffi_checkctype(l)?;push(l,if arg(l,1).as_cdata().is_some_and(|cd|cd.as_ref().ctypeid==id){LuaValue::TRUE}else{LuaValue::FALSE});Ok(1)}
pub fn ffi_string(l:&mut LuaState)->LuaResult<i32>{let cd=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.string","cdata",""))?;let p=cd.as_ref().get_ptr()as*const u8;let len=if nargs(l)>1{arg(l,1).as_number().unwrap_or(0.0)as usize}else{let mut n=0;while n<4096&&!p.is_null()&&unsafe{*p.add(n)}!=0{n+=1}n};if p.is_null()||len==0{push(l,LuaValue::NIL);return Ok(1)}let v={let h=l.heap();let sid=h.strings.intern(unsafe{std::slice::from_raw_parts(p,len)});h.str_value(sid)};push(l,v);Ok(1)}
pub fn ffi_copy(l:&mut LuaState)->LuaResult<i32>{let dp=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.copy","cdata",""))?.as_ref().get_ptr()as*mut u8;let sp=arg(l,1).as_cdata().ok_or_else(||err_bad_arg(l,2,"ffi.copy","cdata",""))?.as_ref().get_ptr()as*const u8;let len=arg(l,2).as_number().unwrap_or(0.0)as usize;if!dp.is_null()&&!sp.is_null()&&len>0{unsafe{std::ptr::copy_nonoverlapping(sp,dp,len)}}Ok(0)}
pub fn ffi_fill(l:&mut LuaState)->LuaResult<i32>{let dp=arg(l,0).as_cdata().ok_or_else(||err_bad_arg(l,1,"ffi.fill","cdata",""))?.as_ref().get_ptr()as*mut u8;let len=arg(l,1).as_number().unwrap_or(0.0)as usize;let b=if nargs(l)>2{arg(l,2).as_number().map(|n|n as u8).unwrap_or(0)}else{0};if!dp.is_null()&&len>0{unsafe{std::ptr::write_bytes(dp,b,len)}}Ok(0)}
pub fn ffi_cast(l:&mut LuaState)->LuaResult<i32>{
    let id=ffi_checkctype(l)?;let val=arg(l,1);
    if let Some(cd)=val.as_cdata(){
        let mut nc=CData::new(id,cd.as_ref().data.len().max(1));
        let n=nc.data.len().min(cd.as_ref().data.len());
        nc.data[..n].copy_from_slice(&cd.as_ref().data[..n]);
        let p=l.global().heap.cdatas.alloc(nc);push(l,LuaValue::cdata(p));Ok(1)
    }else if val.is_nil(){
        let sz=l.global().cts.as_ref().and_then(|c|{
            let r=c.raw(id);if r.size!=u32::MAX{Some(r.size as usize)}else{None}
        }).unwrap_or(0).max(1);
        let p=l.global().heap.cdatas.alloc(CData::new(id,sz));push(l,LuaValue::cdata(p));Ok(1)
    }else if let Some(n)=val.as_number(){
        // Cast a number to a pointer type
        let sz=l.global().cts.as_ref().and_then(|c|{
            let r=c.raw(id);if r.size!=u32::MAX{Some(r.size as usize)}else{None}
        }).unwrap_or(8).max(1);
        let mut cd=CData::new(id,sz);
        let ptr=n as usize;
        let bytes=ptr.to_ne_bytes();
        let len=cd.data.len().min(bytes.len());
        cd.data[..len].copy_from_slice(&bytes[..len]);
        let p=l.global().heap.cdatas.alloc(cd);push(l,LuaValue::cdata(p));Ok(1)
    }else{Err(err_bad_arg(l,2,"ffi.cast","cdata",""))}
}

fn get_field_offset(cts: &crate::ffi::CTState, ctypeid: u32, name: &str) -> Option<(u32, u32)> {
    let struct_id = cts.resolve_raw_id(ctypeid);
    cts.field_names.get(&(struct_id, name.to_string())).copied()
}

fn cdata_index(l: &mut LuaState) -> LuaResult<i32> {
    let cd = arg(l, 0).as_cdata().ok_or(LuaError::Runtime)?;
    let key = arg(l, 1);
    let cts = l.global().cts.as_ref().ok_or(LuaError::Runtime)?;
    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => { push(l, LuaValue::NIL); return Ok(1); }
    };
    // Resolve pointer types to their target
    let raw_ct = cts.raw(cd.as_ref().ctypeid);
    let (target_id, is_ptr) = if crate::ffi::ctype_isptr(raw_ct.info) {
        (crate::ffi::ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };
    if let Some((field_type_id, offset)) = get_field_offset(cts, target_id, &name) {
        if is_ptr {
            // For pointer cdata, read from the pointed-to memory
            let ptr = cd.as_ref().get_ptr();
            if ptr != 0 {
                let raw_ct = cts.raw(field_type_id);
                if ctype_isnum(raw_ct.info) {
                    let sz = raw_ct.size as usize;
                    let field_ptr = unsafe { (ptr as *const u8).add(offset as usize) };
                    let val = match sz {
                        1 => unsafe { *(field_ptr as *const i8) as f64 },
                        2 => unsafe { *(field_ptr as *const i16) as f64 },
                        4 => unsafe { *(field_ptr as *const i32) as f64 },
                        8 => unsafe { *(field_ptr as *const i64) as f64 },
                        _ => 0.0,
                    };
                    push(l, LuaValue::number(val));
                    return Ok(1);
                }
            }
            push(l, LuaValue::NIL);
            return Ok(1);
        }
        let data = &cd.as_ref().data;
        let raw_ct = cts.raw(field_type_id);
        if ctype_isnum(raw_ct.info) {
            let sz = raw_ct.size as usize;
            if offset as usize + sz <= data.len() {
                let val = match sz {
                    1 => data[offset as usize] as i8 as f64,
                    2 => i16::from_le_bytes(data[offset as usize..offset as usize + 2].try_into().unwrap()) as f64,
                    4 => i32::from_le_bytes(data[offset as usize..offset as usize + 4].try_into().unwrap()) as f64,
                    8 => i64::from_le_bytes(data[offset as usize..offset as usize + 8].try_into().unwrap()) as f64,
                    _ => 0.0,
                };
                push(l, LuaValue::number(val));
                return Ok(1);
            }
        }
    }
    push(l, LuaValue::NIL);
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
        (crate::ffi::ctype_cid(raw_ct.info), true)
    } else {
        (cd.as_ref().ctypeid, false)
    };
    if let Some((_field_type_id, offset)) = get_field_offset(cts, target_id, &name) {
        let v = val.as_number().unwrap_or(0.0) as i32;
        if is_ptr {
            let ptr = cd.as_ref().get_ptr();
            if ptr != 0 {
                let field_ptr = unsafe { (ptr as *mut u8).add(offset as usize) };
                unsafe { *(field_ptr as *mut i32) = v };
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
        return Ok(0);
    }
    Err(LuaError::Runtime)
}

// -- Generic ffi.C symbol resolution and C call wrapper --

/// ffi.C __index: lazily resolve a C symbol and return a callable wrapper.
fn clib_index(l: &mut LuaState) -> LuaResult<i32> {
    let _tab = arg(l, 0); // ffi.C table
    let key = arg(l, 1);
    let name = match key.as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.heap().strings.get(sid)).into_owned(),
        _ => { push(l, LuaValue::NIL); return Ok(1); }
    };
    let addr = crate::ffi::clib::resolve_symbol(&name).unwrap_or(0) as f64;
    if addr == 0.0 { push(l, LuaValue::NIL); return Ok(1); }
    let env = l.global().globals;
    let g = l.global() as *mut crate::state::GlobalState;
    let clos = unsafe {
        (*g).heap.alloc_func(crate::func::GcFunc::C(crate::func::CClosure {
            f: call_c, env, upvals: vec![LuaValue::number(addr)],
        }))
    };
    let v = LuaValue::func(clos);
    // Cache in the ffi.C table for next time
    if let Some(ctab) = arg(l, 0).as_table() {
        let c_k = l.heap().str_value(l.heap().intern(name.as_bytes()));
        ctab.as_mut().set_str(c_k, v);
    }
    push(l, v);
    Ok(1)
}

/// Generic C call wrapper: read address from upvalue, marshal args, call, return result.
fn call_c(l: &mut LuaState) -> LuaResult<i32> {
    let addr = l.upvalue(0).as_number().unwrap() as usize;
    let narg = l.top - l.base;
    let mut cstrs: Vec<std::ffi::CString> = Vec::new();
    let mut cargs: Vec<i64> = Vec::with_capacity(narg);
    for i in 0..narg {
        let a = arg(l, i);
        if let Some(sid) = a.as_string_id() {
            let bytes = l.heap().strings.get(sid).to_vec();
            let cs = std::ffi::CString::new(bytes).map_err(|_| LuaError::Runtime)?;
            cargs.push(cs.as_ptr() as i64);
            cstrs.push(cs);
        } else if let Some(cd) = a.as_cdata() {
            // For cdata, pass the address of the underlying data buffer
            let ptr = cd.as_ref().data.as_ptr() as i64;
            cargs.push(ptr);
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

pub fn open(l: &mut LuaState) {
    use crate::func::{CClosure, GcFunc};
    use crate::table::LuaTable;
    let g = l.global() as *mut crate::state::GlobalState;
    let env = unsafe { (*g).globals };

    // -- cdata metatable --
    let cdata_mt = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 2)) };
    {
        let g = unsafe { &mut *g };
        let index_k = g.mmname[crate::meta::MM::Index as usize];
        let newindex_k = g.mmname[crate::meta::MM::Newindex as usize];
        let index_f = g.heap.alloc_func(GcFunc::C(CClosure {
            f: cdata_index, env, upvals: vec![],
        }));
        let newindex_f = g.heap.alloc_func(GcFunc::C(CClosure {
            f: cdata_newindex, env, upvals: vec![],
        }));
        cdata_mt.as_mut().set_str(index_k, LuaValue::func(index_f));
        cdata_mt.as_mut().set_str(newindex_k, LuaValue::func(newindex_f));
        g.set_basemt(crate::value::LJ_TCDATA, Some(cdata_mt));
    }

    // ---- phase 1: heap allocations ----
    let tab = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 16)) };
    let funcs: [(&[u8], crate::func::CFunction); 10] = [(b"cdef",ffi_cdef),(b"new",ffi_new),(b"sizeof",ffi_sizeof),(b"alignof",ffi_alignof),(b"typeid",ffi_typeof),(b"istype",ffi_istype),(b"string",ffi_string),(b"copy",ffi_copy),(b"fill",ffi_fill),(b"cast",ffi_cast)];
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

    // -- ffi.C table with lazy __index --
    let c_k = unsafe { let s=(*g).heap.intern(b"C"); (*g).heap.str_value(s) };
    let c_tab = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 4)) };
    // Set up __index metatable for lazy symbol resolution
    let cmt = unsafe { (*g).heap.alloc_table(LuaTable::new(0, 1)) };
    {
        let g = unsafe { &mut *g };
        let index_k = g.mmname[crate::meta::MM::Index as usize];
        let index_f = g.heap.alloc_func(GcFunc::C(CClosure {
            f: clib_index, env, upvals: vec![],
        }));
        cmt.as_mut().set_str(index_k, LuaValue::func(index_f));
    }
    c_tab.as_mut().metatable = Some(cmt);

    #[cfg(windows)] unsafe { crate::ffi::clib::init_default_libs(); }

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
