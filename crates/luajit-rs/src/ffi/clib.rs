//! C library loading — `ffi.C` namespace.
//! Port of LuaJIT's `lj_clib.h/c`.
//!
//! Provides cross-platform dynamic library loading (dlopen/LoadLibrary)
//! and symbol resolution (dlsym/GetProcAddress).

/// On Windows, searches these default libraries in order.
#[cfg(windows)]
pub static mut CLIB_DEF_HANDLES: [isize; 6] = [0; 6];

#[cfg(windows)]
pub const CLIB_HANDLE_EXE: usize = 0;
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

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryA(name: *const u8) -> isize;
    fn GetProcAddress(h: isize, name: *const u8) -> *const std::ffi::c_void;
    fn GetModuleHandleExA(flags: u32, name: *const u8, out: *mut isize) -> i32;
}

/// Initialise the default library handles on Windows.
/// Call once at startup, before any symbol resolution.
///
/// # Safety
/// Must be called on the main thread before any other clib operations.
#[cfg(windows)]
pub unsafe fn init_default_libs() {
    let handles = &raw mut CLIB_DEF_HANDLES;
    unsafe {
        GetModuleHandleExA(2, std::ptr::null(), &mut (*handles)[CLIB_HANDLE_EXE]);
        GetModuleHandleExA(6, init_default_libs as *const u8, &mut (*handles)[CLIB_HANDLE_DLL]);
        let msvcrt = cstr("msvcrt.dll"); (*handles)[CLIB_HANDLE_CRT] = LoadLibraryA(msvcrt.as_ptr() as *const u8);
        let k32 = cstr("kernel32.dll"); (*handles)[CLIB_HANDLE_KERNEL32] = LoadLibraryA(k32.as_ptr() as *const u8);
        let u32 = cstr("user32.dll"); (*handles)[CLIB_HANDLE_USER32] = LoadLibraryA(u32.as_ptr() as *const u8);
        let g32 = cstr("gdi32.dll"); (*handles)[CLIB_HANDLE_GDI32] = LoadLibraryA(g32.as_ptr() as *const u8);
    }
}

#[cfg(windows)]
fn cstr(s: &str) -> std::ffi::CString { std::ffi::CString::new(s).unwrap() }

/// Resolve a symbol from the default C library.
/// Cross-platform: Windows searches all default handles; Unix uses RTLD_DEFAULT.
pub fn resolve_symbol(name: &str) -> Option<usize> {
    let cname = std::ffi::CString::new(name).ok()?;

    #[cfg(windows)]
    unsafe {
        let handles = &*std::ptr::addr_of!(CLIB_DEF_HANDLES);
        for &h in handles.iter() {
            if h != 0 {
                let p = GetProcAddress(h, cname.as_ptr() as *const u8);
                if !p.is_null() { return Some(p as usize); }
            }
        }
    }
    #[cfg(unix)]
    unsafe {
        let p = dlsym(std::ptr::null_mut(), cname.as_ptr());
        if !p.is_null() { return Some(p as usize); }
    }
    None
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}
