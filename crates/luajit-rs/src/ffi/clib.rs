//! C library loading — `ffi.C` namespace and `ffi.load`.
//! Port of LuaJIT's `lj_clib.h/c`.
//!
//! Provides cross-platform dynamic library loading (dlopen/LoadLibrary)
//! and symbol resolution (dlsym/GetProcAddress), both from the default
//! namespace (`ffi.C`) and from named libraries (`ffi.load`).

/// A loaded C library (or the default namespace, handle 0).
pub struct CLib {
    pub name: String,
    /// dlopen/LoadLibrary handle. 0 = default namespace (`ffi.C`).
    pub handle: usize,
    /// Outstanding `ffi.load` references. Released by `cdata:free()` or
    /// GC finalization.
    pub refcount: u32,
    /// Symbol name → resolved address (per-library resolution cache).
    pub cache: std::collections::HashMap<String, usize>,
}

impl CLib {
    /// Open a named library (or the default namespace for ""/"C").
    /// Returns the error message string on failure.
    pub fn load(name: &str, global: bool) -> Result<CLib, String> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (name, global);
            return Err("dynamic libraries not supported on this platform".to_string());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let name = normalize_name(name);
            if name == "C" {
                // Default namespace: the process handle + system libraries.
                return Ok(CLib {
                    name,
                    handle: 0,
                    refcount: 0,
                    cache: std::collections::HashMap::new(),
                });
            }
            let handle = load_library(&name, global)
                .map_err(|e| format!("cannot load library '{}': {}", name, e))?;
            Ok(CLib {
                name,
                handle,
                refcount: 0,
                cache: std::collections::HashMap::new(),
            })
        }
    }

    /// Resolve a symbol within this library (or the default namespace
    /// when `handle == 0`). `None` when the symbol does not exist.
    pub fn resolve(&self, name: &str) -> Option<usize> {
        if self.handle == 0 {
            return resolve_symbol(name);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let cname = std::ffi::CString::new(name).ok()?;
        #[cfg(target_arch = "wasm32")]
        let _ = name;
        #[cfg(windows)]
        unsafe {
            let p = GetProcAddress(self.handle as isize, cname.as_ptr() as *const u8);
            if !p.is_null() {
                return Some(p as usize);
            }
        }
        #[cfg(unix)]
        unsafe {
            let p = dlsym(self.handle as *mut std::ffi::c_void, cname.as_ptr());
            if !p.is_null() {
                return Some(p as usize);
            }
        }
        None
    }

    /// Close the library (no-op for the default namespace). Only called
    /// when the refcount reaches zero.
    pub fn close(&mut self) {
        if self.handle == 0 {
            return;
        }
        #[cfg(windows)]
        unsafe {
            FreeLibrary(self.handle as isize);
        }
        #[cfg(unix)]
        unsafe {
            dlclose(self.handle as *mut std::ffi::c_void);
        }
        self.handle = 0;
    }
}

/// Normalize a `ffi.load` name: empty/"C"/"c" → default namespace; plain
/// names get platform suffix candidates appended by `load_library`.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub fn normalize_name(name: &str) -> String {
    if name.is_empty() || name == "C" || name == "c" {
        return "C".to_string();
    }
    name.to_string()
}

/// Release a CLib reference when a CLibrary cdata dies (GC finalization).
/// Unloads the library when the refcount reaches zero. No-op for stale or
/// already-released entries.
pub(crate) fn gc_release(g: &mut crate::state::GlobalState, idx: usize) {
    let Some(cts) = g.cts.as_mut() else { return };
    let Some(cl) = cts.clibs.get_mut(idx) else { return };
    if cl.refcount == 0 {
        return;
    }
    cl.refcount -= 1;
    if cl.refcount == 0 {
        let name = cl.name.clone();
        cl.close();
        if name != "C" {
            cts.clib_names.remove(&name);
        }
    }
}

/// Open a library by name, trying platform suffix candidates:
/// - Windows: `name`, then `name.dll`.
/// - Linux: `name`, then `name.so`.
/// - macOS: `name`, then `name.dylib`.
#[cfg_attr(target_arch = "wasm32", allow(unused))]
#[cfg_attr(windows, allow(unused_variables))]
fn load_library(name: &str, global: bool) -> Result<usize, String> {
    let mut candidates: Vec<String> = vec![name.to_string()];
    if !name.contains('.') {
        #[cfg(windows)]
        candidates.push(format!("{}.dll", name));
        #[cfg(all(unix, not(target_os = "macos")))]
        candidates.push(format!("{}.so", name));
        #[cfg(target_os = "macos")]
        candidates.push(format!("{}.dylib", name));
    }
    let mut last_err = String::new();
    for cand in &candidates {
        #[cfg(windows)]
        {
            let c = match std::ffi::CString::new(cand.as_str()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            unsafe {
                let h = LoadLibraryA(c.as_ptr() as *const u8);
                if h != 0 {
                    return Ok(h as usize);
                }
                last_err = format!("{} (error {})", cand, std::io::Error::last_os_error());
            }
        }
        #[cfg(unix)]
        {
            let c = match std::ffi::CString::new(cand.as_str()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let flags = if global { RTLD_GLOBAL | RTLD_NOW } else { RTLD_NOW };
            unsafe {
                let h = dlopen(c.as_ptr(), flags);
                if !h.is_null() {
                    return Ok(h as usize);
                }
                let err = dlerror();
                last_err = if err.is_null() {
                    format!("cannot open '{}'", cand)
                } else {
                    let s = std::ffi::CStr::from_ptr(err).to_string_lossy();
                    format!("{} ({})", cand, s)
                };
            }
        }
    }
    Err(last_err)
}

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
    fn FreeLibrary(h: isize) -> i32;
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(name: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlerror() -> *mut std::ffi::c_char;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
    fn dlsym(
        handle: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
}

#[cfg(unix)]
const RTLD_NOW: i32 = 2;
#[cfg(all(unix, not(target_os = "macos")))]
const RTLD_GLOBAL: i32 = 0x100;
#[cfg(target_os = "macos")]
const RTLD_GLOBAL: i32 = 0x8;

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
        GetModuleHandleExA(
            6,
            init_default_libs as *const u8,
            &mut (*handles)[CLIB_HANDLE_DLL],
        );
        let msvcrt = cstr("msvcrt.dll");
        (*handles)[CLIB_HANDLE_CRT] = LoadLibraryA(msvcrt.as_ptr() as *const u8);
        let k32 = cstr("kernel32.dll");
        (*handles)[CLIB_HANDLE_KERNEL32] = LoadLibraryA(k32.as_ptr() as *const u8);
        let u32 = cstr("user32.dll");
        (*handles)[CLIB_HANDLE_USER32] = LoadLibraryA(u32.as_ptr() as *const u8);
        let g32 = cstr("gdi32.dll");
        (*handles)[CLIB_HANDLE_GDI32] = LoadLibraryA(g32.as_ptr() as *const u8);
    }
}

#[cfg(windows)]
fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap()
}

/// Resolve a symbol from the default C library.
/// Cross-platform: Windows searches all default handles; Unix uses RTLD_DEFAULT.
#[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
pub fn resolve_symbol(name: &str) -> Option<usize> {
    #[cfg(not(target_arch = "wasm32"))]
    let cname = std::ffi::CString::new(name).ok()?;
    #[cfg(target_arch = "wasm32")]
    {
        let _ = name;
        return None;
    }

    #[cfg(windows)]
    unsafe {
        let handles = &*std::ptr::addr_of!(CLIB_DEF_HANDLES);
        for &h in handles.iter() {
            if h != 0 {
                let p = GetProcAddress(h, cname.as_ptr() as *const u8);
                if !p.is_null() {
                    return Some(p as usize);
                }
            }
        }
    }
    #[cfg(unix)]
    unsafe {
        let p = dlsym(rtld_default(), cname.as_ptr());
        if !p.is_null() {
            return Some(p as usize);
        }
    }
    None
}

/// `RTLD_DEFAULT`: 0 on Linux/FreeBSD, but `(void *)-2` on macOS — the
/// loader treats 0 as an invalid handle there, so dlsym would fail to
/// resolve any libc symbol (strlen, strchr, ...).
#[cfg(unix)]
fn rtld_default() -> *mut std::ffi::c_void {
    #[cfg(target_os = "macos")]
    {
        -2isize as *mut std::ffi::c_void
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::ptr::null_mut()
    }
}
