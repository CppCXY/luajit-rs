//! Machine code area management, ported from lj_mcode.c.
//!
//! Cross-platform W^X executable memory: VirtualAlloc/VirtualProtect on
//! Windows, mmap/mprotect on Unix — declared directly (this crate has no
//! external dependencies). The native assembler backends (jit/asm/<arch>)
//! emit into an RW area, which is flipped to RX before execution and back
//! for patching (`lj_mcode_patch`).
//!
//! Note for macOS on Apple Silicon: a hardened runtime would additionally
//! require MAP_JIT + pthread_jit_write_protect_np; plain mprotect flips
//! work for the normal (non-hardened) case and are what we use here.

/// One executable memory area (a simplified `MCLink`ed area).
pub struct McodeArea {
    ptr: *mut u8,
    len: usize,
    exec: bool,
}

impl McodeArea {
    /// Allocate a read-write area of at least `size` bytes (page rounded).
    pub fn alloc(size: usize) -> Option<McodeArea> {
        let len = size.max(1).next_multiple_of(sys::page_size());
        let ptr = sys::alloc_rw(len)?;
        Some(McodeArea {
            ptr,
            len,
            exec: false,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Entry pointer of the area.
    #[inline]
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Writable view. Only valid while the area is not executable.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        assert!(!self.exec, "mcode area is executable");
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Flip the area to executable (W^X) and flush the icache.
    pub fn protect_exec(&mut self) -> bool {
        if !self.exec && !sys::protect(self.ptr, self.len, true) {
            return false;
        }
        sys::flush_icache(self.ptr, self.len);
        self.exec = true;
        true
    }

    /// Flip the area back to writable (for exit-branch patching).
    pub fn protect_rw(&mut self) -> bool {
        if self.exec && !sys::protect(self.ptr, self.len, false) {
            return false;
        }
        self.exec = false;
        true
    }
}

impl Drop for McodeArea {
    fn drop(&mut self) {
        sys::free(self.ptr, self.len);
    }
}

// SAFETY: the area is plain memory owned by this handle.
unsafe impl Send for McodeArea {}

#[cfg(windows)]
mod sys {
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;
    const PAGE_EXECUTE_READ: u32 = 0x20;

    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, ty: u32, prot: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, size: usize, ty: u32) -> i32;
        fn VirtualProtect(addr: *mut u8, size: usize, prot: u32, old: *mut u32) -> i32;
        fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
        fn GetCurrentProcess() -> isize;
    }

    pub fn page_size() -> usize {
        4096
    }

    pub fn alloc_rw(len: usize) -> Option<*mut u8> {
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if p.is_null() { None } else { Some(p) }
    }

    pub fn protect(ptr: *mut u8, len: usize, exec: bool) -> bool {
        let prot = if exec {
            PAGE_EXECUTE_READ
        } else {
            PAGE_READWRITE
        };
        let mut old = 0u32;
        unsafe { VirtualProtect(ptr, len, prot, &mut old) != 0 }
    }

    pub fn flush_icache(ptr: *const u8, len: usize) {
        // Required on ARM64 Windows; a cheap no-op call on x64.
        unsafe {
            FlushInstructionCache(GetCurrentProcess(), ptr, len);
        }
    }

    pub fn free(ptr: *mut u8, _len: usize) {
        unsafe {
            VirtualFree(ptr, 0, MEM_RELEASE);
        }
    }
}

#[cfg(unix)]
mod sys {
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    const MAP_PRIVATE: i32 = 0x02;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MAP_ANON: i32 = 0x20;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    const MAP_ANON: i32 = 0x1000;

    unsafe extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
        fn mprotect(addr: *mut u8, len: usize, prot: i32) -> i32;
    }

    pub fn page_size() -> usize {
        16384
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    pub fn alloc_rw(len: usize) -> Option<*mut u8> {
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        if p as isize == -1 || p.is_null() { None } else { Some(p) }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn alloc_rw(len: usize) -> Option<*mut u8> {
        // macOS ARM64: MAP_JIT is required even when using plain
        // mprotect — without it the kernel may refuse RW→RX
        // transitions on Apple Silicon.
        const MAP_JIT: i32 = 0x800;
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON | MAP_JIT,
                -1,
                0,
            )
        };
        if p as isize == -1 || p.is_null() { None } else { Some(p) }
    }

    pub fn protect(ptr: *mut u8, len: usize, exec: bool) -> bool {
        let prot = if exec {
            PROT_READ | PROT_EXEC
        } else {
            PROT_READ | PROT_WRITE
        };
        unsafe { mprotect(ptr, len, prot) == 0 }
    }

    pub fn flush_icache(ptr: *const u8, len: usize) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe extern "C" {
                fn sys_icache_invalidate(addr: *const u8, size: usize);
            }
            unsafe { sys_icache_invalidate(ptr, len) };
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64",
                      all(target_os = "macos", target_arch = "aarch64"))))]
        {
            unsafe extern "C" {
                fn __clear_cache(start: *const u8, end: *const u8);
            }
            unsafe { __clear_cache(ptr, ptr.add(len)) };
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let _ = (ptr, len);
        }
    }

    pub fn free(ptr: *mut u8, len: usize) {
        unsafe {
            munmap(ptr, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rw_write_and_read_back() {
        let mut area = McodeArea::alloc(64).expect("mcode alloc");
        assert!(area.len() >= 64);
        let s = area.as_mut_slice();
        s[0] = 0xAA;
        s[63] = 0x55;
        assert_eq!((s[0], s[63]), (0xAA, 0x55));
        assert!(area.protect_exec());
        assert!(area.protect_rw());
        assert_eq!(area.as_mut_slice()[0], 0xAA);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn execute_generated_code() {
        // return 42;
        #[cfg(target_arch = "x86_64")]
        const CODE: &[u8] = &[0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3]; // mov eax,42; ret
        #[cfg(target_arch = "aarch64")]
        const CODE: &[u8] = &[
            0x40, 0x05, 0x80, 0x52, // mov w0, #42
            0xC0, 0x03, 0x5F, 0xD6, // ret
        ];
        let mut area = McodeArea::alloc(CODE.len()).expect("mcode alloc");
        area.as_mut_slice()[..CODE.len()].copy_from_slice(CODE);
        assert!(area.protect_exec());
        let f: extern "C" fn() -> u32 = unsafe { std::mem::transmute(area.ptr()) };
        assert_eq!(f(), 42);
    }
}
