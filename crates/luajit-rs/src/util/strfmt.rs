//! Number and `string.format` formatting, mirroring the pieces of
//! `lj_strfmt*` the runtime needs.

use std::fmt::Write;

/// Stack buffer for `core::fmt::Write` — zero-allocation formatting.
struct BufWriter<'a> { buf: &'a mut [u8], pos: usize }
impl<'a> BufWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self { BufWriter { buf, pos: 0 } }
    fn as_slice(&self) -> &[u8] { &self.buf[..self.pos] }
    fn len(&self) -> usize { self.pos }
}
impl<'a> Write for BufWriter<'a> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        let end = (self.pos + b.len()).min(self.buf.len());
        self.buf[self.pos..end].copy_from_slice(&b[..end - self.pos]);
        self.pos = end;
        Ok(())
    }
}

/// Format a double like LuaJIT's `STRFMT_G14` (`%.14g`, with integral values
/// printed without a decimal point and `inf`/`nan` spellings).
pub fn g14(n: f64) -> String {
    let mut buf = [0u8; 64];
    let len = g14_to_buf(n, &mut buf);
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

/// Like `g14()` but writes into a pre-allocated stack buffer and returns
/// the byte count. Exact integers take a pure-itoa fast path.
pub fn g14_to_buf(n: f64, buf: &mut [u8; 64]) -> usize {
    // Special values
    if n == 0.0 {
        if n.is_sign_negative() { buf[0] = b'-'; buf[1] = b'0'; return 2; }
        buf[0] = b'0'; return 1;
    }
    if n.is_nan() { buf[..3].copy_from_slice(b"nan"); return 3; }
    if n.is_infinite() {
        if n < 0.0 { buf[..4].copy_from_slice(b"-inf"); return 4; }
        buf[..3].copy_from_slice(b"inf"); return 3;
    }
    // Integer fast path (|n| < 2^53).
    let i = n as i64;
    if i as f64 == n && i.unsigned_abs() < (1u64 << 53) {
        return itoa_i64(i, buf);
    }
    // General float — zero-alloc via stack buffer + write!.
    let mut tmp = [0u8; 64];
    let mut w = BufWriter::new(&mut tmp);
    let _ = write!(w, "{:.13e}", n);
    let mant_str = std::str::from_utf8(w.as_slice()).unwrap();
    let (m, e) = mant_str.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();

    if !(-4..14).contains(&exp) {
        // Scientific notation.
        let m2 = m.trim_end_matches('0').trim_end_matches('.');
        let mut w2 = BufWriter::new(buf);
        let _ = write!(w2, "{}e{}{:02}", m2, if exp < 0 { '-' } else { '+' }, exp.abs());
        return w2.len();
    }
    // Decimal notation.
    let prec = (13 - exp).max(0) as usize;
    let out = {
        let mut w3 = BufWriter::new(buf);
        let _ = write!(w3, "{:.*}", prec, n);
        let s = std::str::from_utf8(w3.as_slice()).unwrap();
        if s.contains('.') {
            let t = s.trim_end_matches('0').trim_end_matches('.');
            let blen = t.len().min(64);
            let mut tmp = [0u8; 64];
            tmp[..blen].copy_from_slice(t.as_bytes());
            // Need to write back into buf. Drop w3 first.
            drop(w3);
            buf[..blen].copy_from_slice(&tmp[..blen]);
            blen
        } else {
            let blen = s.len().min(64);
            drop(w3);
            blen
        }
    };
    out
}

/// Minimal signed integer-to-ASCII, returns byte count.
#[inline]
fn itoa_i64(mut v: i64, buf: &mut [u8; 64]) -> usize {
    let neg = v < 0;
    let mut tmp = [0u8; 20];
    let mut t = 20;
    if neg {
        let mut u = (v as u64).wrapping_neg();
        while u >= 10 { t -= 1; tmp[t] = b'0' + (u % 10) as u8; u /= 10; }
        t -= 1; tmp[t] = b'0' + u as u8;
    } else {
        while v >= 10 { t -= 1; tmp[t] = b'0' + (v % 10) as u8; v /= 10; }
        t -= 1; tmp[t] = b'0' + v as u8;
    }
    let digits = 20 - t;
    let mut o = 0;
    if neg { buf[0] = b'-'; o = 1; }
    buf[o..o + digits].copy_from_slice(&tmp[t..]);
    o + digits
}

/// A single format argument for `string.format`.
pub enum FmtArg<'a> {
    Num(f64),
    Str(&'a [u8]),
}

/// A minimal `string.format`, covering the conversions used so far:
/// `%%`, `%d/%i/%u`, `%c`, `%x/%X`, `%o`, `%f/%F`, `%e/%E`, `%g/%G`, `%s`,
/// with optional flags, width and precision. Returns an error message on a
/// malformed spec or argument-type mismatch.
pub fn format(fmt: &[u8], args: &[FmtArg]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut ai = 0usize;
    let mut i = 0usize;
    while i < fmt.len() {
        let c = fmt[i];
        if c != b'%' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        if i < fmt.len() && fmt[i] == b'%' {
            out.push(b'%');
            i += 1;
            continue;
        }
        let start = i;
        // flags
        while i < fmt.len() && matches!(fmt[i], b'-' | b'+' | b' ' | b'#' | b'0') {
            i += 1;
        }
        // width
        while i < fmt.len() && fmt[i].is_ascii_digit() {
            i += 1;
        }
        // precision
        if i < fmt.len() && fmt[i] == b'.' {
            i += 1;
            while i < fmt.len() && fmt[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i >= fmt.len() {
            return Err("invalid conversion to 'format'".into());
        }
        let conv = fmt[i];
        let spec = std::str::from_utf8(&fmt[start..i]).map_err(|_| "invalid format".to_string())?;
        i += 1;

        let next_num = |ai: &mut usize| -> Result<f64, String> {
            let a = args
                .get(*ai)
                .ok_or_else(|| "bad argument to 'format'".to_string())?;
            *ai += 1;
            match a {
                FmtArg::Num(n) => Ok(*n),
                FmtArg::Str(s) => std::str::from_utf8(s)
                    .ok()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .ok_or_else(|| "bad argument to 'format' (number expected)".to_string()),
            }
        };

        match conv {
            b'd' | b'i' => {
                let n = next_num(&mut ai)? as i64;
                out.extend_from_slice(pad_int(spec, &n.to_string(), n < 0).as_bytes());
            }
            b'u' => {
                let n = next_num(&mut ai)? as i64 as u64;
                out.extend_from_slice(pad_int(spec, &n.to_string(), false).as_bytes());
            }
            b'c' => {
                let n = next_num(&mut ai)? as i64 as u8;
                out.push(n);
            }
            b'x' => {
                let n = next_num(&mut ai)? as i64 as u64;
                out.extend_from_slice(pad_int(spec, &format!("{:x}", n), false).as_bytes());
            }
            b'X' => {
                let n = next_num(&mut ai)? as i64 as u64;
                out.extend_from_slice(pad_int(spec, &format!("{:X}", n), false).as_bytes());
            }
            b'o' => {
                let n = next_num(&mut ai)? as i64 as u64;
                out.extend_from_slice(pad_int(spec, &format!("{:o}", n), false).as_bytes());
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let n = next_num(&mut ai)?;
                out.extend_from_slice(fmt_float(spec, conv, n).as_bytes());
            }
            b's' => {
                let a = args
                    .get(ai)
                    .ok_or_else(|| "bad argument to 'format'".to_string())?;
                ai += 1;
                let s: Vec<u8> = match a {
                    FmtArg::Str(s) => s.to_vec(),
                    FmtArg::Num(n) => g14(*n).into_bytes(),
                };
                out.extend_from_slice(&pad_str(spec, &s));
            }
            b'q' => {
                let a = args
                    .get(ai)
                    .ok_or_else(|| "bad argument to 'format'".to_string())?;
                ai += 1;
                out.push(b'"');
                if let FmtArg::Str(s) = a {
                    for &b in *s {
                        match b {
                            b'"' | b'\\' => {
                                out.push(b'\\');
                                out.push(b);
                            }
                            b'\n' => out.extend_from_slice(b"\\n"),
                            b'\r' => out.extend_from_slice(b"\\r"),
                            0 => out.extend_from_slice(b"\\0"),
                            _ => out.push(b),
                        }
                    }
                }
                out.push(b'"');
            }
            _ => return Err(format!("invalid conversion '%{}'", conv as char)),
        }
    }
    Ok(out)
}

fn parse_spec(spec: &str) -> (bool, bool, Option<usize>, Option<usize>) {
    let b = spec.as_bytes();
    let mut i = 0;
    let mut left = false;
    let mut zero = false;
    while i < b.len() && matches!(b[i], b'-' | b'+' | b' ' | b'#' | b'0') {
        if b[i] == b'-' {
            left = true;
        }
        if b[i] == b'0' {
            zero = true;
        }
        i += 1;
    }
    let ws = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let width = if i > ws {
        spec[ws..i].parse().ok()
    } else {
        None
    };
    let mut prec = None;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let ps = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        prec = Some(spec[ps..i].parse().unwrap_or(0));
    }
    (left, zero, width, prec)
}

fn pad(s: Vec<u8>, width: Option<usize>, left: bool) -> Vec<u8> {
    match width {
        Some(w) if s.len() < w => {
            let padn = w - s.len();
            let mut out = Vec::with_capacity(w);
            if left {
                out.extend_from_slice(&s);
                out.extend(std::iter::repeat_n(b' ', padn));
            } else {
                out.extend(std::iter::repeat_n(b' ', padn));
                out.extend_from_slice(&s);
            }
            out
        }
        _ => s,
    }
}

fn pad_int(spec: &str, digits: &str, negative: bool) -> String {
    let (left, zero, width, prec) = parse_spec(spec);
    let mut body = digits.trim_start_matches('-').to_string();
    if let Some(p) = prec {
        while body.len() < p {
            body.insert(0, '0');
        }
    }
    let sign = if negative { "-" } else { "" };
    if zero
        && !left
        && prec.is_none()
        && let Some(w) = width
    {
        let total = sign.len() + body.len();
        if total < w {
            let mut s = String::new();
            s.push_str(sign);
            for _ in 0..(w - total) {
                s.push('0');
            }
            s.push_str(&body);
            return s;
        }
    }
    let s = format!("{}{}", sign, body).into_bytes();
    String::from_utf8(pad(s, width, left)).unwrap()
}

fn pad_str(spec: &str, s: &[u8]) -> Vec<u8> {
    let (left, _zero, width, prec) = parse_spec(spec);
    let s = match prec {
        Some(p) if p < s.len() => &s[..p],
        _ => s,
    };
    pad(s.to_vec(), width, left)
}

fn fmt_float(spec: &str, conv: u8, n: f64) -> String {
    let (left, zero, width, prec) = parse_spec(spec);
    let p = prec.unwrap_or(6);
    let mut body = match conv {
        b'f' | b'F' => format!("{:.*}", p, n.abs()),
        b'e' => fmt_e(n.abs(), p, false),
        b'E' => fmt_e(n.abs(), p, true),
        b'g' | b'G' => {
            let s = fmt_g(n.abs(), if prec.is_some() { p.max(1) } else { 6 });
            if conv == b'G' { s.to_uppercase() } else { s }
        }
        _ => unreachable!(),
    };
    let sign = if n.is_sign_negative() { "-" } else { "" };
    if zero
        && !left
        && let Some(w) = width
    {
        let total = sign.len() + body.len();
        if total < w {
            let mut s = String::new();
            s.push_str(sign);
            for _ in 0..(w - total) {
                s.push('0');
            }
            s.push_str(&body);
            return s;
        }
    }
    body.insert_str(0, sign);
    String::from_utf8(pad(body.into_bytes(), width, left)).unwrap()
}

fn fmt_e(n: f64, prec: usize, upper: bool) -> String {
    let s = format!("{:.*e}", prec, n);
    let (m, e) = s.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();
    format!(
        "{}{}{}{:02}",
        m,
        if upper { 'E' } else { 'e' },
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

fn fmt_g(n: f64, prec: usize) -> String {
    if n == 0.0 {
        return "0".to_string();
    }
    let exp = n.abs().log10().floor() as i32;
    if exp < -4 || exp >= prec as i32 {
        let m = format!("{:.*e}", prec.saturating_sub(1), n);
        let (mant, e) = m.split_once('e').unwrap();
        let ex: i32 = e.parse().unwrap();
        let mant = mant.trim_end_matches('0').trim_end_matches('.');
        format!("{}e{}{:02}", mant, if ex < 0 { '-' } else { '+' }, ex.abs())
    } else {
        let decimals = (prec as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, n);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}
