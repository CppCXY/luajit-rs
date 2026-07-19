//! C library loading — `ffi.C` namespace.
//! Port of LuaJIT's `lj_clib.h/c`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::stdlib::{arg, push};
use std::collections::HashMap;

/// A loaded C library.
pub struct CLibrary {
    /// Opaque handle (HMODULE on Windows, void* on Unix).
    handle: *mut std::ffi::c_void,
    /// Cache of resolved symbol names → cdata pointers.
    cache: HashMap<String, crate::gc::GcPtr<crate::runtime::cdata::CData>>,
}

impl CLibrary {
    fn new(handle: *mut std::ffi::c_void) -> Self {
        CLibrary { handle, cache: HashMap::new() }
    }
}

/// Default C library handles (Windows: searches exe + system DLLs).
#[cfg(windows)]
static mut CLIB_DEF_HANDLES: [isize; 6] = [0; 6];

#[cfg(windows)]
const CLIB_HANDLE_EXE: usize = 0;
#[cfg(windows)]
const CLIB_HANDLE_DLL: usize = 1;
#[cfg(windows)]
const CLIB_HANDLE_CRT: usize = 2;
#[cfg(windows)]
const CLIB_HANDLE_KERNEL32: usize = 3;
#[cfg(windows)]
const CLIB_HANDLE_USER32: usize = 4;
#[cfg(windows)]
const CLIB_HANDLE_GDI32: usize = 5;

// ---------------------------------------------------------------------------
// Platform-specific dynamic library loading
// ---------------------------------------------------------------------------

#[cfg(unix)]
mod unix_ffi {
    use libc::{dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL};
    pub unsafe fn load(name: *const i8, global: bool) -> *mut std::ffi::c_void {
        let flags = RTLD_LAZY | if global { RTLD_GLOBAL } else { 0 };
        dlopen(name, flags) as *mut std::ffi::c_void
    }
    pub unsafe fn sym(handle: *mut std::ffi::c_void, name: *const i8) -> *mut std::ffi::c_void {
        dlsym(handle, name) as *mut std::ffi::c_void
    }
    pub unsafe fn close(handle: *mut std::ffi::c_void) { dlclose(handle); }
    /// Default libc handle for dlsym.
    pub fn def_handle() -> *mut std::ffi::c_void { std::ptr::null_mut() }
    pub unsafe fn resolve_default(name: *const i8) -> *mut std::ffi::c_void {
        dlsym(def_handle(), name) as *mut std::ffi::c_void
    }
}

#[cfg(windows)]
mod win_ffi {
    use std::ffi::CStr;
    use super::*;

    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> isize;
        fn GetProcAddress(h: isize, name: *const u8) -> *const std::ffi::c_void;
        fn FreeLibrary(h: isize) -> i32;
        fn GetModuleHandleExA(flags: u32, name: *const u8, out: *mut isize) -> i32;
    }
    pub unsafe fn load(name: &str) -> isize {
        let cname = CStr::from_ptr(format!("{}\0", name).as_ptr() as *const _);
        LoadLibraryA(cname.to_bytes_with_nul().as_ptr())
    }
    pub unsafe fn sym(h: isize, name: *const u8) -> *const std::ffi::c_void {
        GetProcAddress(h, name)
    }
    pub unsafe fn close(h: isize) { FreeLibrary(h); }

    /// Resolve default library handles (once, cached).
    pub unsafe fn init_default_handles() {
        let handles = &mut *std::ptr::addr_of_mut!(CLIB_DEF_HANDLES);
        // EXE
        GetModuleHandleExA(2, std::ptr::null(), &mut handles[CLIB_HANDLE_EXE]);
        // DLL (the module containing this function)
        GetModuleHandleExA(6, win_ffi::init_default_handles as *const u8, &mut handles[CLIB_HANDLE_DLL]);
        // CRT (via _fmode — not available, skip)
        handles[CLIB_HANDLE_CRT] = LoadLibraryA(b"msvcrt.dll\0".as_ptr());
        handles[CLIB_HANDLE_KERNEL32] = LoadLibraryA(b"kernel32.dll\0".as_ptr());
        handles[CLIB_HANDLE_USER32] = LoadLibraryA(b"user32.dll\0".as_ptr());
        handles[CLIB_HANDLE_GDI32] = LoadLibraryA(b"gdi32.dll\0".as_ptr());
    }

    /// Resolve symbol from default handle (searches all default libraries).
    pub unsafe fn resolve_default(name: &str) -> *const std::ffi::c_void {
        let cname = std::ffi::CString::new(name).unwrap();
        let handles = &*std::ptr::addr_of!(CLIB_DEF_HANDLES);
        for &h in handles.iter() {
            if h != 0 {
                let p = GetProcAddress(h, cname.as_ptr() as *const u8);
                if !p.is_null() { return p; }
            }
        }
        std::ptr::null()
    }
}

// ---------------------------------------------------------------------------
// Global CLibrary state
// ---------------------------------------------------------------------------

/// Global FFI C library cache (stored once in GlobalState).  
pub struct ClibState {
    /// The default C library (ffi.C).
    pub default_lib: CLibrary,
    /// Loaded named libraries.
    pub libs: HashMap<String, CLibrary>,
}

impl ClibState {
    pub fn init() -> Self {
        #[cfg(windows)]
        unsafe { win_ffi::init_default_handles(); }
        ClibState {
            default_lib: CLibrary::new(default_handle()),
            libs: HashMap::new(),
        }
    }
}

fn default_handle() -> *mut std::ffi::c_void {
    #[cfg(unix)] { unix_ffi::def_handle() }
    #[cfg(windows)] { std::ptr::null_mut() } // CLIB_DEFHANDLE marker
    #[cfg(not(any(unix, windows)))] { std::ptr::null_mut() }
}

/// Resolve a C symbol address by name (searches default lib on Unix,
/// all default handles on Windows).
fn resolve_symbol(name: &str) -> Option<usize> {
    let cname = std::ffi::CString::new(name).ok()?;
    #[cfg(unix)]
    {
        let p = unsafe { unix_ffi::resolve_default(cname.as_ptr()) };
        if !p.is_null() { return Some(p as usize); }
    }
    #[cfg(windows)]
    {
        let p = unsafe { win_ffi::resolve_default(name) };
        if !p.is_null() { return Some(p as usize); }
    }
    None
}

// ---------------------------------------------------------------------------
// `ffi.C.__index` metamethod — resolves symbols on access
// ---------------------------------------------------------------------------

pub fn clib_index(l: &mut LuaState) -> LuaResult<i32> {
    let key = arg(l, 1);
    let sid = key.as_string_id().expect("non-string clib key");
    let name_bytes = l.heap().strings.get(sid).to_vec();
    let name = std::str::from_utf8(&name_bytes).map_err(|_| crate::err::LuaError::Runtime)?;

    let g = l.global();
    let clib = g.clib_state.get_or_insert_with(ClibState::init);

    // Check cache for pre-created cdata
    if let Some(&cd) = clib.default_lib.cache.get(name) {
        push(l, LuaValue::cdata(cd));
        return Ok(1);
    }

    // Resolve symbol
    let addr = resolve_symbol(name).unwrap_or(0);
    if addr == 0 { push(l, LuaValue::NIL); return Ok(1); }

    // Create a C closure wrapping the function call
    let closure_f: crate::func::CFunction = match name {
        "printf" => make_printf_caller(addr),
        _ => {
            // General case: create cdata (not yet callable without __call metamethod)
            let mut cd = crate::runtime::cdata::CData::new(0, std::mem::size_of::<usize>());
            cd.data[..std::mem::size_of::<usize>()].copy_from_slice(&addr.to_ne_bytes());
            let ptr = g.heap.cdatas.alloc(cd);
            clib.default_lib.cache.insert(name.to_string(), ptr);
            push(l, LuaValue::cdata(ptr));
            return Ok(1);
        }
    };
    let closure = crate::func::CClosure { f: closure_f, env: g.globals, upvals: vec![] };
    let fp = g.heap.alloc_func(crate::func::GcFunc::C(closure));
    push(l, LuaValue::func(fp));
    Ok(1)
}

fn make_printf_caller(addr: usize) -> crate::func::CFunction {
    fn inner(l: &mut LuaState) -> LuaResult<i32> {
        let addr = 0usize; // will be captured differently
        let _ = addr;
        // Placeholder — we can't capture addr in a fn pointer
        push(l, LuaValue::NIL);
        Ok(1)
    }
    inner
}
