//! FFI callbacks — Lua closures exposed to C as function pointers.
//!
//! When a Lua function is cast to a function-pointer ctype, a small
//! machine-code trampoline is emitted whose address becomes the C function
//! pointer. C calling the trampoline saves the ABI argument registers into a
//! [`CCallbackFrame`] and jumps to [`ffi_cb_handler`], which marshals the C
//! arguments into Lua values, runs the closure on a dedicated callback
//! thread, and marshals the first result back. The trampoline then loads the
//! return value into both `rax` and `xmm0`, so integer and floating-point
//! returns both work.
//!
//! The trampoline layout (x86-64) mirrors LuaJIT's `lj_ccallback.c`:
//! pointer ops of a function-pointer declarator are applied *outside* the
//! function suffix, and a callback is stored in a registry rooted for GC.

use crate::ffi::{ctinfo, ctype_isfunc, ctype_type, CT, CTState, CTypeID};
use crate::jit::mcode::McodeArea;
use crate::runtime::cdata::CData;
use crate::state::LuaState;
use crate::value::LuaValue;
use crate::{LuaError, LuaResult};
use crate::vm;

use super::lib::{ccall_param_types, ccall_ret_type};

/// The argument frame saved by a callback trampoline. The generated machine
/// code writes these exact offsets (`repr(C)`), so the layout must match the
/// emitter in [`emit_trampoline`].
#[repr(C)]
pub struct CCallbackFrame {
    pub gpr: [u64; 8],
    pub fpr: [u64; 8],
    pub entry_rsp: usize,
    pub cb_id: u32,
    _pad: u32,
    pub cts: usize,
}

/// A registered callback: the Lua closure, its function ctype (for argument
/// marshalling), and the executable trampoline.
pub struct Callback {
    pub func: LuaValue,
    /// The `CT::Func` ctype id (resolved from the cast target).
    pub func_ctype: u32,
    /// The trampoline memory (kept alive for the lifetime of the callback).
    pub mcode: McodeArea,
    /// The trampoline entry address.
    pub addr: usize,
    /// Set by `cdata:free()`; the trampoline stays alive (calling it raises).
    pub freed: bool,
}

/// Resolve the `CT::Func` ctype id behind a (possibly pointer-to-function)
/// ctype id. Returns `None` when `ctype_id` is not a function type.
pub fn func_ctype_of(cts: &CTState, ctype_id: u32) -> Option<u32> {
    let raw = cts.raw(ctype_id);
    match ctype_type(raw.info) {
        CT::Func => Some(ctype_id),
        CT::Ptr => {
            let child = raw.info & ctinfo::MASK_CID;
            let rc = cts.raw(child);
            if ctype_isfunc(rc.info) {
                Some(child)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Is `ctype_id` a function pointer (or function) type?
pub fn is_func_ptr_type(cts: &CTState, ctype_id: u32) -> bool {
    func_ctype_of(cts, ctype_id).is_some()
}

/// The machine-code entry point called by the C side.
///
/// The trampoline passes the address of a [`CCallbackFrame`] (first arg on
/// every ABI). This runs the registered Lua closure with the marshalled
/// arguments and returns the raw result bits in `rax` (the trampoline copies
/// them to `xmm0`).
pub extern "C" fn ffi_cb_handler(frame: *mut CCallbackFrame) -> u64 {
    let f = unsafe { &*frame };
    let cts = unsafe { &mut *(f.cts as *mut CTState) };
    dispatch(cts, f.cb_id as usize, f)
}

fn dispatch(cts: &mut CTState, cb_id: usize, frame: &CCallbackFrame) -> u64 {
    let Some(thread) = cts.callback_thread else {
        return 0;
    };
    let Some(cb) = cts.callbacks.get(cb_id) else {
        return 0;
    };
    if cb.freed {
        // Freed callback: surface an error at the enclosing ffi.C call.
        let l = thread.as_mut();
        l.runtime_error(b"attempt to call a freed callback");
        l.global().ffi_cb_error = Some(l.errval);
        return 0;
    }
    let func = cb.func;
    let fid = cb.func_ctype;
    let params = ccall_param_types(cts, fid);
    let ret_tid = ccall_ret_type(cts, fid);
    // Remember the closure's proto for result-conversion error locations.
    let cb_proto = func.as_func().and_then(|fv| match fv.as_ref() {
        crate::func::GcFunc::Lua(cl) => Some(cl.proto),
        _ => None,
    });

    let l = thread.as_mut();
    let mut args: Vec<LuaValue> = Vec::with_capacity(params.len());
    let mut int_reg = 0usize;
    let mut fp_reg = 0usize;
    let mut stack = 0usize;
    for (i, &p) in params.iter().enumerate() {
        let p = p & 0xFFFF;
        let raw = cts.raw(p);
        let t = ctype_type(raw.info);
        let is_fp = crate::ffi::ctype_isfp(raw.info)
            || (t == CT::Array && crate::ffi::ctype_iscomplex(raw.info));
        let bits = read_arg(frame, i, is_fp, &mut int_reg, &mut fp_reg, &mut stack);
        args.push(c_arg_to_lua(cts, l, p, bits, is_fp));
    }

    match vm::call(l, func, &args) {
        Ok(results) => match lua_to_c_ret(cts, ret_tid, &results) {
            Some(bits) => bits,
            None => {
                // Result conversion failed (e.g. nil for a non-void return).
                if let Some(pt) = cb_proto {
                    callback_error(l, pt.as_ref(), "ffi: callback result conversion failed");
                } else {
                    l.runtime_error(b"ffi: callback result conversion failed");
                }
                l.global().ffi_cb_error = Some(l.errval);
                0
            }
        },
        Err(LuaError::Yield) => {
            l.runtime_error(b"callback: attempt to yield from C callback");
            l.global().ffi_cb_error = Some(l.errval);
            0
        }
        Err(LuaError::Runtime) => {
            // Store the error object for the enclosing ffi.C call site to
            // re-raise (call_c / jit_ffi_call check ffi_cb_error afterwards).
            l.global().ffi_cb_error = Some(l.errval);
            0
        }
    }
}

/// The address of the `slot`-th stack-passed argument. The layout differs
/// per ABI: Win64 reserves a 32-byte shadow space (and pushes the return
/// address), SysV pushes the return address but reserves no shadow space,
/// and ARM64 keeps the return address in `x30` (no stack slot).
#[inline]
fn stack_arg_addr(frame: &CCallbackFrame, slot: usize) -> *const u64 {
    #[cfg(all(target_arch = "x86_64", windows))]
    {
        (frame.entry_rsp + 40 + slot * 8) as *const u64
    }
    #[cfg(all(target_arch = "x86_64", not(windows)))]
    {
        (frame.entry_rsp + 8 + slot * 8) as *const u64
    }
    #[cfg(target_arch = "aarch64")]
    {
        (frame.entry_rsp + slot * 8) as *const u64
    }
}

/// Read the raw bits of the `i`-th argument from the saved frame, applying
/// the ABI's register/stack assignment.
#[inline]
#[cfg_attr(windows, allow(unused_variables))]
fn read_arg(
    frame: &CCallbackFrame,
    i: usize,
    is_fp: bool,
    int_reg: &mut usize,
    fp_reg: &mut usize,
    stack: &mut usize,
) -> u64 {
    #[cfg(windows)]
    {
        // Win64: positional — args 0..4 in gpr[i]/fpr[i], the rest on stack.
        if i < 4 {
            return if is_fp { frame.fpr[i] } else { frame.gpr[i] };
        }
        let slot = *stack;
        *stack += 1;
        unsafe { *stack_arg_addr(frame, slot) }
    }
    #[cfg(not(windows))]
    {
        // SysV / ARM64: separate integer/FP register banks.
        let int_n = if cfg!(target_arch = "aarch64") { 8 } else { 6 };
        if is_fp {
            if *fp_reg < 8 {
                let r = frame.fpr[*fp_reg];
                *fp_reg += 1;
                r
            } else {
                let slot = *stack;
                *stack += 1;
                unsafe { *stack_arg_addr(frame, slot) }
            }
        } else if *int_reg < int_n {
            let r = frame.gpr[*int_reg];
            *int_reg += 1;
            r
        } else {
            let slot = *stack;
            *stack += 1;
            unsafe { *stack_arg_addr(frame, slot) }
        }
    }
}

/// Convert a raw C argument to a Lua value.
fn c_arg_to_lua(cts: &CTState, l: &mut LuaState, tid: u32, bits: u64, is_fp: bool) -> LuaValue {
    let raw = cts.raw(tid);
    let t = ctype_type(raw.info);
    if is_fp {
        // float is passed as the low 32 bits of the register; double as 64.
        return if raw.size == 4 {
            LuaValue::number(f32::from_bits(bits as u32) as f64)
        } else {
            LuaValue::number(f64::from_bits(bits))
        };
    }
    if raw.info & ctinfo::BOOL != 0 {
        return LuaValue::boolean(bits != 0);
    }
    if t == CT::Ptr || t == CT::Func {
        let mut cd = CData::new(tid, 8);
        cd.set_ptr(bits as usize);
        return LuaValue::cdata(l.global().heap.alloc_cdata(cd));
    }
    // Integer scalar (Num or Enum).
    let sz = raw.size;
    let uns = raw.info & ctinfo::UNSIGNED != 0;
    match sz {
        8 => {
            // 64-bit integers cannot be represented exactly as doubles.
            let ctypeid = if uns {
                CTypeID::UInt64 as u32
            } else {
                CTypeID::Int64 as u32
            };
            let mut cd = CData::new(ctypeid, 8);
            cd.data[..8].copy_from_slice(&bits.to_le_bytes());
            LuaValue::cdata(l.global().heap.alloc_cdata(cd))
        }
        4 => {
            let v = (bits & 0xFFFF_FFFF) as u32;
            if uns {
                LuaValue::number(v as f64)
            } else {
                LuaValue::number(v as i32 as f64)
            }
        }
        2 => {
            let v = (bits & 0xFFFF) as u16;
            if uns {
                LuaValue::number(v as f64)
            } else {
                LuaValue::number(v as i16 as f64)
            }
        }
        1 => {
            let v = (bits & 0xFF) as u8;
            if uns {
                LuaValue::number(v as f64)
            } else {
                LuaValue::number(v as i8 as f64)
            }
        }
        _ => LuaValue::NIL,
    }
}

/// The numeric value of a Lua value, treating booleans as 0/1.
fn lua_num(v: LuaValue) -> Option<f64> {
    v.as_number().or_else(|| {
        if v.is_true() {
            Some(1.0)
        } else if v.is_false() {
            Some(0.0)
        } else {
            None
        }
    })
}

/// Marshal the Lua results back to the C return value (raw bits). Returns
/// `None` when the result cannot be converted (e.g. `nil` for a non-void
/// return).
fn lua_to_c_ret(cts: &CTState, tid: u32, results: &[LuaValue]) -> Option<u64> {
    let raw = cts.raw(tid);
    let t = ctype_type(raw.info);
    if t == CT::Void {
        return Some(0);
    }
    let v = results.first().copied().unwrap_or(LuaValue::NIL);
    let is_fp = crate::ffi::ctype_isfp(raw.info);
    if is_fp {
        let n = lua_num(v)?;
        return Some(if raw.size == 4 {
            (n as f32).to_bits() as u64
        } else {
            n.to_bits()
        });
    }
    if raw.info & ctinfo::BOOL != 0 {
        return Some((lua_num(v).unwrap_or(0.0) != 0.0) as u64);
    }
    if t == CT::Ptr || t == CT::Func {
        if let Some(cd) = v.as_cdata() {
            return Some(cd.as_ref().get_ptr() as u64);
        }
        if let Some(n) = v.as_number() {
            return Some((n as i64) as u64);
        }
        if v.is_nil() {
            return Some(0);
        }
        return None;
    }
    if let Some(cd) = v.as_cdata() {
        let d = &cd.as_ref().data;
        let mut buf = [0u8; 8];
        let n = d.len().min(8);
        buf[..n].copy_from_slice(&d[..n]);
        return Some(u64::from_le_bytes(buf));
    }
    let n = lua_num(v)?;
    Some(match raw.size {
        1 => (n as i64 as u8) as u64,
        2 => (n as i64 as u16) as u64,
        4 => (n as i64 as u32) as u64,
        _ => (n as i64) as u64,
    })
}

/// Create a callback for `func` cast to `ctype_id` (a function-pointer or
/// function type) and return a cdata holding the trampoline address.
pub fn ffi_callback_new(l: &mut LuaState, ctype_id: u32, func: LuaValue) -> LuaResult<LuaValue> {
    if l.global().cts.is_none() {
        l.global().cts = Some(CTState::new());
    }

    // Resolve the function ctype and validate it has no by-value struct
    // params or returns (those need an sret pointer / struct copy we do not
    // implement yet).
    let fid = {
        let g = l.global();
        let cts = g.cts.as_ref().unwrap();
        func_ctype_of(cts, ctype_id)
            .ok_or_else(|| l.runtime_error(b"ffi: cannot create a callback for a non-function type"))?
    };
    {
        let cts = l.global().cts.as_ref().unwrap();
        validate_callback_ctype(cts, fid)
            .map_err(|m| l.runtime_error(m.as_bytes()))?;
    }

    // Ensure the dedicated callback thread exists.
    if l.global().cts.as_ref().unwrap().callback_thread.is_none() {
        let th = crate::state::new_thread(l);
        l.global().cts.as_mut().unwrap().callback_thread = Some(th);
    }

    // Emit the trampoline.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let code = {
        let cts_addr = l.global().cts.as_mut().unwrap() as *mut CTState as usize;
        let cb_id = l.global().cts.as_ref().unwrap().callbacks.len() as u32;
        let handler = ffi_cb_handler as *const () as usize;
        emit_trampoline(handler, cts_addr, cb_id)
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ctype_id;
        return Err(l.runtime_error(b"ffi: callbacks are not supported on this architecture"));
    }

    let mut area = McodeArea::alloc(code.bytes.len().max(256))
        .ok_or_else(|| l.runtime_error(b"ffi: out of memory for callback"))?;
    let entry = area.ptr() as usize;
    area.as_mut_slice()[..code.bytes.len()].copy_from_slice(&code.bytes);
    area.protect_exec();

    let cb_id = l.global().cts.as_ref().unwrap().callbacks.len() as u32;
    l.global()
        .cts
        .as_mut()
        .unwrap()
        .callbacks
        .push(Callback {
            func,
            func_ctype: fid,
            mcode: area,
            addr: entry,
            freed: false,
        });
    l.global()
        .cts
        .as_mut()
        .unwrap()
        .callback_by_addr
        .insert(entry, cb_id);

    // The cdata holds the trampoline address; call_c marshals function
    // pointers via get_ptr().
    let mut cd = CData::new(ctype_id, 8);
    cd.set_ptr(entry);
    let ptr = l.global().heap.alloc_cdata(cd);
    Ok(LuaValue::cdata(ptr))
}

/// Reject callback signatures with by-value struct/union/array params or a
/// struct/array return (those need sret/struct-copy support).
fn validate_callback_ctype(cts: &CTState, fid: u32) -> Result<(), String> {
    for p in ccall_param_types(cts, fid) {
        let raw = cts.raw(p & 0xFFFF);
        match ctype_type(raw.info) {
            CT::Struct => return Err("ffi: callback with a by-value struct parameter is unsupported".into()),
            CT::Array if !crate::ffi::ctype_iscomplex(raw.info) && !crate::ffi::ctype_isvector(raw.info) => {
                return Err("ffi: callback with a by-value array parameter is unsupported".into());
            }
            _ => {}
        }
    }
    let ret = cts.raw(ccall_ret_type(cts, fid));
    match ctype_type(ret.info) {
        CT::Struct => Err("ffi: callback with a struct return is unsupported".into()),
        CT::Array if !crate::ffi::ctype_iscomplex(ret.info) && !crate::ffi::ctype_isvector(ret.info) => {
            Err("ffi: callback with an array return is unsupported".into())
        }
        _ => Ok(()),
    }
}

/// Raise a callback result-conversion error, located at the closure's last
/// line (LuaJIT reports result-conversion errors there).
fn callback_error(l: &mut LuaState, pt: &crate::proto::Proto, msg: &str) -> LuaError {
    let line = pt.firstline + pt.numline;
    let src = pt
        .source
        .map(|sid| {
            let bytes = l.str_static(sid);
            let s = if bytes.starts_with(b"@") || bytes.starts_with(b"=") {
                &bytes[1..]
            } else {
                bytes
            };
            String::from_utf8_lossy(s).into_owned()
        })
        .unwrap_or_else(|| "=?".to_string());
    let full = format!("{}:{}: {}", src, line, msg);
    let sid = l.heap().intern(full.as_bytes());
    l.errval = l.heap().str_value(sid);
    LuaError::Runtime
}

// ---------------------------------------------------------------------------
// Trampoline code generation
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) struct TrampolineCode {
    pub bytes: Vec<u8>,
}

#[cfg(target_arch = "x86_64")]
fn emit_trampoline(handler: usize, cts: usize, cb_id: u32) -> TrampolineCode {
    let mut b: Vec<u8> = Vec::with_capacity(128);

    // Stack frame: shadow space (0x20) + CCallbackFrame (0x98) = 0xB8.
    // Entered with rsp%16 == 8 (after the caller's `call`), so subtracting
    // 0xB8 (== 8 mod 16) leaves rsp 16-aligned before the `call handler`.
    let stack_size: i32 = 0xB8;

    // sub rsp, stack_size
    b.extend_from_slice(&[0x48, 0x81, 0xEC]);
    b.extend_from_slice(&stack_size.to_le_bytes());

    // Save integer argument registers.
    #[cfg(windows)]
    let gpr_save: [(u8, u8); 6] = [
        (0x4C, 0x20), // rcx @ 0x20
        (0x54, 0x28), // rdx @ 0x28
        (0x44, 0x30), // r8  @ 0x30
        (0x4C, 0x38), // r9  @ 0x38
        (0x7C, 0x40), // rdi @ 0x40
        (0x74, 0x48), // rsi @ 0x48
    ];
    #[cfg(not(windows))]
    let gpr_save: [(u8, u8); 6] = [
        (0x7C, 0x20), // rdi @ 0x20
        (0x74, 0x28), // rsi @ 0x28
        (0x54, 0x30), // rdx @ 0x30
        (0x4C, 0x38), // rcx @ 0x38
        (0x44, 0x40), // r8  @ 0x40
        (0x4C, 0x48), // r9  @ 0x48
    ];
    for (i, (modrm, disp)) in gpr_save.iter().enumerate() {
        // mov [rsp+disp], r64 — the extended regs (r8/r9) need REX.R.
        if i >= 2 && i <= 3 && cfg!(windows) {
            b.push(0x4C); // REX.W+R
        } else if i >= 4 && i <= 5 && !cfg!(windows) {
            b.push(0x4C); // REX.W+R (r8/r9 in sysv positions 4,5)
        } else {
            b.push(0x48); // REX.W
        }
        b.extend_from_slice(&[0x89, *modrm, 0x24, *disp]);
    }

    // Save FP argument registers (xmm0..xmm7) at 0x60..0x98.
    for i in 0..8u8 {
        let disp = 0x60 + i * 8;
        b.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x44 + (i << 3), 0x24, disp]);
    }

    // movabs rax, cts ; mov [rsp+0xB0], rax
    b.push(0x48);
    b.push(0xB8);
    b.extend_from_slice(&(cts as u64).to_le_bytes());
    b.extend_from_slice(&[0x48, 0x89, 0x84, 0x24, 0xB0, 0x00, 0x00, 0x00]);

    // lea rax, [rsp+stack_size] ; mov [rsp+0xA0], rax   (entry_rsp)
    b.extend_from_slice(&[0x48, 0x8D, 0x84, 0x24]);
    b.extend_from_slice(&(stack_size as u32).to_le_bytes());
    b.extend_from_slice(&[0x48, 0x89, 0x84, 0x24, 0xA0, 0x00, 0x00, 0x00]);

    // mov dword [rsp+0xA8], cb_id
    b.extend_from_slice(&[0xC7, 0x84, 0x24, 0xA8, 0x00, 0x00, 0x00]);
    b.extend_from_slice(&cb_id.to_le_bytes());

    // lea rcx/rdi, [rsp+0x20]  (frame pointer = handler's first argument)
    #[cfg(windows)]
    b.extend_from_slice(&[0x48, 0x8D, 0x4C, 0x24, 0x20]);
    #[cfg(not(windows))]
    b.extend_from_slice(&[0x48, 0x8D, 0x7C, 0x24, 0x20]);

    // movabs rax, handler ; call rax  (absolute call — the trampoline and
    // the handler may be more than ±2GB apart, so a rel32 call cannot reach)
    b.push(0x48);
    b.push(0xB8);
    b.extend_from_slice(&(handler as u64).to_le_bytes());
    b.extend_from_slice(&[0xFF, 0xD0]);

    // movq xmm0, rax
    b.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC0]);

    // add rsp, stack_size ; ret
    b.extend_from_slice(&[0x48, 0x81, 0xC4]);
    b.extend_from_slice(&stack_size.to_le_bytes());
    b.push(0xC3);

    TrampolineCode {
        bytes: b,
    }
}

// ---------------------------------------------------------------------------
// ARM64 trampoline
// ---------------------------------------------------------------------------

/// Emit `movz` + three `movk`s loading a 64-bit immediate into `x{reg}`.
#[cfg(target_arch = "aarch64")]
fn arm64_mov_imm(b: &mut Vec<u8>, reg: u32, val: u64) {
    let imm = |i: u32| ((val >> (16 * i)) & 0xFFFF) as u32;
    b.extend_from_slice(&(0xD2800000 | (imm(0) << 5) | reg).to_le_bytes()); // movz xN, #lo
    b.extend_from_slice(&(0xF2A00000 | (imm(1) << 5) | reg).to_le_bytes()); // movk lsl #16
    b.extend_from_slice(&(0xF2C00000 | (imm(2) << 5) | reg).to_le_bytes()); // movk lsl #32
    b.extend_from_slice(&(0xF2E00000 | (imm(3) << 5) | reg).to_le_bytes()); // movk lsl #48
}

#[cfg(target_arch = "aarch64")]
fn emit_trampoline(handler: usize, cts: usize, cb_id: u32) -> TrampolineCode {
    let mut b: Vec<u8> = Vec::with_capacity(256);
    macro_rules! w {
        ($ins:expr) => {
            b.extend_from_slice(&($ins as u32).to_le_bytes());
        };
    }

    const SP: u32 = 31;
    const FRAME: u32 = 0xC0; // 0x98 frame + 16 bytes for the saved lr + padding.

    // sub sp, sp, #FRAME
    w!(0xD10003FF | (FRAME << 10));

    // Save integer argument registers x0..x7 (gpr[0..8] at 0x00..0x40).
    for i in 0..8u32 {
        w!(0xF9000000 | (i << 10) | (SP << 5) | i);
    }
    // Save FP argument registers d0..d7 (fpr[0..8] at 0x40..0x80).
    for i in 0..8u32 {
        w!(0xFD000000 | ((8 + i) << 10) | (SP << 5) | i);
    }

    // entry_sp = sp + FRAME, stored at 0x80.
    w!(0x91000000 | (FRAME << 10) | (SP << 5) | 9); // add x9, sp, #FRAME
    w!(0xF9000000 | (16 << 10) | (SP << 5) | 9); // str x9, [sp, #0x80]

    // cb_id (u32) at 0x88.
    w!(0x52800000 | ((cb_id & 0xFFFF) << 5) | 10); // movz w10, #lo
    w!(0x72A00000 | (((cb_id >> 16) & 0xFFFF) << 5) | 10); // movk w10, #hi, lsl #16
    w!(0xB9000000 | (34 << 10) | (SP << 5) | 10); // str w10, [sp, #0x88]

    // cts at 0x90.
    arm64_mov_imm(&mut b, 10, cts as u64);
    w!(0xF9000000 | (18 << 10) | (SP << 5) | 10); // str x10, [sp, #0x90]

    // Save the link register (clobbered by blr) at 0xA0.
    w!(0xF9000000 | (20 << 10) | (SP << 5) | 30); // str x30, [sp, #0xA0]

    // frame pointer → x0 (the handler's first argument).
    w!(0x91000000 | (0 << 10) | (SP << 5) | 0); // mov x0, sp

    // Absolute call to the handler.
    arm64_mov_imm(&mut b, 16, handler as u64);
    w!(0xD63F0200); // blr x16

    // Restore lr and copy the result to d0 (FP return).
    w!(0xF9400000 | (20 << 10) | (SP << 5) | 30); // ldr x30, [sp, #0xA0]
    w!(0x9E670000); // fmov d0, x0

    w!(0x910003FF | (FRAME << 10)); // add sp, sp, #FRAME
    w!(0xD65F03C0); // ret

    TrampolineCode { bytes: b }
}
