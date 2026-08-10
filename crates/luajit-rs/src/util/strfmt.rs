//! Number and `string.format` formatting, mirroring the pieces of
//! `lj_strfmt*` the runtime needs.

use std::fmt::Write;

/// Stack buffer for `core::fmt::Write` — zero-allocation formatting.
struct BufWriter<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> BufWriter<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        BufWriter { buf, pos: 0 }
    }
    fn as_slice(&self) -> &[u8] {
        &self.buf[..self.pos]
    }
    fn len(&self) -> usize {
        self.pos
    }
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
        if n.is_sign_negative() {
            buf[0] = b'-';
            buf[1] = b'0';
            return 2;
        }
        buf[0] = b'0';
        return 1;
    }
    if n.is_nan() {
        buf[..3].copy_from_slice(b"nan");
        return 3;
    }
    if n.is_infinite() {
        if n < 0.0 {
            buf[..4].copy_from_slice(b"-inf");
            return 4;
        }
        buf[..3].copy_from_slice(b"inf");
        return 3;
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
        let _ = write!(
            w2,
            "{}e{}{:02}",
            m2,
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        );
        return w2.len();
    }
    // Decimal notation.
    let prec = (13 - exp).max(0) as usize;

    {
        let mut w3 = BufWriter::new(buf);
        let _ = write!(w3, "{:.*}", prec, n);
        let s = std::str::from_utf8(w3.as_slice()).unwrap();
        if s.contains('.') {
            let t = s.trim_end_matches('0').trim_end_matches('.');
            let blen = t.len().min(64);
            let mut tmp = [0u8; 64];
            tmp[..blen].copy_from_slice(t.as_bytes());
            // Need to write back into buf. Drop w3 first (releases the
            // mutable borrow so buf can be accessed again).
            #[allow(clippy::drop_non_drop)]
            drop(w3);
            buf[..blen].copy_from_slice(&tmp[..blen]);
            blen
        } else {
            let blen = s.len().min(64);
            #[allow(clippy::drop_non_drop)]
            drop(w3);
            blen
        }
    }
}

/// Minimal signed integer-to-ASCII, returns byte count.
#[inline]
fn itoa_i64(mut v: i64, buf: &mut [u8; 64]) -> usize {
    let neg = v < 0;
    let mut tmp = [0u8; 20];
    let mut t = 20;
    if neg {
        let mut u = (v as u64).wrapping_neg();
        while u >= 10 {
            t -= 1;
            tmp[t] = b'0' + (u % 10) as u8;
            u /= 10;
        }
        t -= 1;
        tmp[t] = b'0' + u as u8;
    } else {
        while v >= 10 {
            t -= 1;
            tmp[t] = b'0' + (v % 10) as u8;
            v /= 10;
        }
        t -= 1;
        tmp[t] = b'0' + v as u8;
    }
    let digits = 20 - t;
    let mut o = 0;
    if neg {
        buf[0] = b'-';
        o = 1;
    }
    buf[o..o + digits].copy_from_slice(&tmp[t..]);
    o + digits
}

/// A single format argument for `string.format`.
pub enum FmtArg<'a> {
    Num(f64),
    Str(&'a [u8]),
}

/// Parsed flags/width/precision of one conversion spec.
#[derive(Clone, Copy, Default)]
struct Spec {
    left: bool,
    zero: bool,
    space: bool,
    plus: bool,
    hash: bool,
    width: Option<usize>,
    prec: Option<usize>,
}

/// Parse a spec from raw bytes (no UTF-8 round trip).
fn parse_spec_bytes(fmt: &[u8], mut i: usize, end: usize) -> Spec {
    let mut s = Spec::default();
    while i < end && matches!(fmt[i], b'-' | b'+' | b' ' | b'#' | b'0') {
        match fmt[i] {
            b'-' => s.left = true,
            b'+' => s.plus = true,
            b' ' => s.space = true,
            b'#' => s.hash = true,
            _ => s.zero = true,
        }
        i += 1;
    }
    let ws = i;
    while i < end && fmt[i].is_ascii_digit() {
        i += 1;
    }
    if i > ws {
        s.width = std::str::from_utf8(&fmt[ws..i])
            .ok()
            .and_then(|x| x.parse().ok());
    }
    if i < end && fmt[i] == b'.' {
        i += 1;
        let ps = i;
        while i < end && fmt[i].is_ascii_digit() {
            i += 1;
        }
        s.prec = std::str::from_utf8(&fmt[ps..i])
            .ok()
            .and_then(|x| x.parse().ok());
    }
    s
}

/// Append `body` (display digits, no sign) to `out` with the sign and
/// the spec's padding rules — a single pass, no intermediate strings.
/// `zero_with_prec` allows zero-padding alongside a precision (floats:
/// `%010.5g`); integers keep the old rule where precision wins.
fn push_padded(out: &mut Vec<u8>, s: &Spec, body: &[u8], sign: u8, zero_with_prec: bool) {
    let total = (sign != 0) as usize + body.len();
    let w = s.width.unwrap_or(0);
    if s.zero && !s.left && (zero_with_prec || s.prec.is_none()) && w > total {
        if sign != 0 {
            out.push(sign);
        }
        out.resize(out.len() + (w - total), b'0');
        out.extend_from_slice(body);
        return;
    }
    let padn = w.saturating_sub(total);
    if s.left {
        if sign != 0 {
            out.push(sign);
        }
        out.extend_from_slice(body);
        out.resize(out.len() + padn, b' ');
    } else {
        out.resize(out.len() + padn, b' ');
        if sign != 0 {
            out.push(sign);
        }
        out.extend_from_slice(body);
    }
}

/// i64 -> decimal digits into `buf` (returns the digit slice). Uses
/// truncating division: `div_euclid` never reaches 0 for negatives
/// (`-1.div_euclid(10) == -1`).
fn itoa(mut n: i64, buf: &mut [u8; 32]) -> &[u8] {
    let mut i = buf.len();
    if n == 0 {
        buf[31] = b'0';
        return &buf[31..];
    }
    while n != 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10).unsigned_abs() as u8;
        n /= 10;
    }
    &buf[i..]
}

/// u64 -> hex digits into `buf`.
fn utohex(mut n: u64, buf: &mut [u8; 32], upper: bool) -> &[u8] {
    let mut i = buf.len();
    if n == 0 {
        buf[31] = b'0';
        return &buf[31..];
    }
    while n != 0 {
        i -= 1;
        let d = (n & 0xF) as u8;
        buf[i] = if d < 10 {
            b'0' + d
        } else if upper {
            b'A' + d - 10
        } else {
            b'a' + d - 10
        };
        n >>= 4;
    }
    &buf[i..]
}

/// u64 -> octal digits into `buf`.
fn utooct(mut n: u64, buf: &mut [u8; 32]) -> &[u8] {
    let mut i = buf.len();
    if n == 0 {
        buf[31] = b'0';
        return &buf[31..];
    }
    while n != 0 {
        i -= 1;
        buf[i] = b'0' + (n & 7) as u8;
        n >>= 3;
    }
    &buf[i..]
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
        let spec = parse_spec_bytes(fmt, start, i);
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
                let mut buf = [0u8; 32];
                let digits = itoa(n, &mut buf);
                let sign = if n < 0 {
                    b'-'
                } else if spec.plus {
                    b'+'
                } else if spec.space {
                    b' '
                } else {
                    0
                };
                push_padded(&mut out, &spec, digits, sign, false);
            }
            b'u' => {
                let n = next_num(&mut ai)? as i64 as u64;
                let mut buf = [0u8; 32];
                let digits = itoa(n as i64, &mut buf);
                push_padded(&mut out, &spec, digits, 0, false);
            }
            b'c' => {
                let n = next_num(&mut ai)? as i64 as u8;
                out.push(n);
            }
            b'x' => {
                let n = next_num(&mut ai)? as i64 as u64;
                let mut buf = [0u8; 32];
                let digits = utohex(n, &mut buf, false);
                push_padded(&mut out, &spec, digits, 0, false);
            }
            b'X' => {
                let n = next_num(&mut ai)? as i64 as u64;
                let mut buf = [0u8; 32];
                let digits = utohex(n, &mut buf, true);
                push_padded(&mut out, &spec, digits, 0, false);
            }
            b'o' => {
                let n = next_num(&mut ai)? as i64 as u64;
                let mut buf = [0u8; 32];
                let digits = utooct(n, &mut buf);
                push_padded(&mut out, &spec, digits, 0, false);
            }
            b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
                let n = next_num(&mut ai)?;
                let body = fmt_float_body(&spec, conv, n);
                let sign = if n.is_sign_negative() {
                    b'-'
                } else if spec.plus {
                    b'+'
                } else if spec.space {
                    b' '
                } else {
                    0
                };
                push_padded(&mut out, &spec, body.as_bytes(), sign, true);
            }
            b'a' | b'A' => {
                let n = next_num(&mut ai)?;
                let prec = spec.prec.unwrap_or(13);
                let mut s = fmt_a(n, prec, conv == b'A');
                if n.is_sign_positive() && spec.plus {
                    s.insert(0, '+');
                } else if n.is_sign_positive() && spec.space && !s.starts_with(' ') {
                    s.insert(0, ' ');
                }
                out.extend_from_slice(&pad(s.into_bytes(), spec.width, spec.left));
            }
            b's' => {
                let a = args
                    .get(ai)
                    .ok_or_else(|| "bad argument to 'format'".to_string())?;
                ai += 1;
                let body: &[u8] = match a {
                    FmtArg::Str(s) => s,
                    FmtArg::Num(n) => {
                        let t = g14(*n);
                        out.extend_from_slice(t.as_bytes());
                        &[]
                    }
                };
                if !body.is_empty() {
                    // Precision truncates; otherwise write through with
                    // the padding rules (no intermediate clone).
                    let body = match spec.prec {
                        Some(p) if p < body.len() => &body[..p],
                        _ => body,
                    };
                    push_padded(&mut out, &spec, body, 0, false);
                }
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
                            // LuaJIT quotes newline/CR with a backslash
                            // followed by the *literal* byte.
                            b'\n' | b'\r' => {
                                out.push(b'\\');
                                out.push(b);
                            }
                            b'\t' => out.extend_from_slice(b"\\t"),
                            b'\x07' => out.extend_from_slice(b"\\a"),
                            b'\x08' => out.extend_from_slice(b"\\b"),
                            b'\x0c' => out.extend_from_slice(b"\\f"),
                            b'\x0b' => out.extend_from_slice(b"\\v"),
                            // Other control bytes: 3-digit octal ("\000").
                            b if b < 32 || b == 127 => {
                                let oct = format!("\\{:03o}", b);
                                out.extend_from_slice(oct.as_bytes());
                            }
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

fn fmt_float_body(spec: &Spec, conv: u8, n: f64) -> String {
    let hash = spec.hash;
    let p = spec.prec.unwrap_or(6);
    match conv {
        b'f' | b'F' => {
            let s = format!("{:.*}", p, n.abs());
            if hash && !s.contains('.') {
                format!("{}.", s)
            } else {
                s
            }
        }
        b'e' | b'E' => fmt_e(n.abs(), p, conv == b'E', hash),
        b'g' | b'G' => {
            let s = fmt_g(
                n.abs(),
                if spec.prec.is_some() { p.max(1) } else { 6 },
                hash,
            );
            if conv == b'G' { s.to_uppercase() } else { s }
        }
        _ => unreachable!(),
    }
}

fn fmt_e(n: f64, prec: usize, upper: bool, hash: bool) -> String {
    let s = format!("{:.*e}", prec, n);
    let (m, e) = s.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();
    let mant = if hash && !m.contains('.') {
        format!("{}.", m)
    } else {
        m.to_string()
    };
    format!(
        "{}{}{}{:02}",
        mant,
        if upper { 'E' } else { 'e' },
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    )
}

fn fmt_g(n: f64, prec: usize, hash: bool) -> String {
    if n == 0.0 {
        return "0".to_string();
    }
    // The exponent of the *rounded* value (from the e-form) decides the
    // style, like the C library.
    let e_s = format!("{:.*e}", prec - 1, n);
    let exp: i32 = e_s.split_once('e').unwrap().1.parse().unwrap();
    if exp < -4 || exp >= prec as i32 {
        // e-style with the mantissa trimmed (kept with #).
        let (mant, _e) = e_s.split_once('e').unwrap();
        let mant = if hash {
            mant.to_string()
        } else {
            mant.trim_end_matches('0').trim_end_matches('.').to_string()
        };
        format!(
            "{}e{}{:02}",
            mant,
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        let decimals = (prec as i32 - 1 - exp).max(0) as usize;
        let s = format!("{:.*}", decimals, n);
        if hash {
            s
        } else if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
}

/// Hex float formatting (`%a`/`%A`): `[-]0x1.HEXP±N`.
fn fmt_a(n: f64, prec: usize, upper: bool) -> String {
    let (prefix, hexes) = if upper {
        ("0X", "0123456789ABCDEF")
    } else {
        ("0x", "0123456789abcdef")
    };
    let hex_digit = |v: usize| hexes.as_bytes()[v] as char;
    let mut v = n;
    let sign = if v.is_sign_negative() {
        v = -v;
        "-"
    } else {
        ""
    };
    if v == 0.0 {
        return format!("{}{}0p+0", sign, prefix);
    }
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (e2, mant) = if exp == 0 {
        // Subnormal: normalize by finding the highest set bit.
        let lz = frac.leading_zeros() as i32;
        let shift = lz - 11;
        let m = frac << shift;
        (-1023 - shift, m)
    } else {
        (exp - 1023, (1u64 << 52) | frac)
    };
    // The mantissa's hex digits after "1.": 13 digits from the 52-bit
    // fraction.
    let mut digits = String::with_capacity(13);
    let mut m = mant;
    for _ in 0..13 {
        digits.push(hex_digit(((m >> 48) & 0xF) as usize));
        m <<= 4;
    }
    if prec == 0 {
        // Round to the nearest whole power of two: a first fraction
        // digit >= 8 carries into the integer part ("0x2p+...").
        let first = ((mant >> 48) & 0xF) as usize;
        let (whole, e2) = if first >= 8 { (2, e2) } else { (1, e2) };
        return format!(
            "{}{}{}p{}{}",
            sign,
            prefix,
            whole,
            if e2 < 0 { '-' } else { '+' },
            e2.abs()
        );
    }
    let (frac_s, e2) = if prec <= 13 {
        // The first `prec` digits (trailing zeros kept), rounded.
        let keep = prec.min(13);
        let mut ds: Vec<char> = digits.chars().take(keep).collect();
        while ds.len() < keep {
            ds.push('0');
        }
        let next = digits
            .chars()
            .nth(keep)
            .map(|c| c.to_digit(16).unwrap() as i32)
            .unwrap_or(0);
        let mut e2 = e2;
        if next >= 8 {
            // Round up with carry.
            let mut i = keep as i32 - 1;
            loop {
                if i < 0 {
                    ds.insert(0, '1');
                    ds.pop();
                    e2 += 1;
                    break;
                }
                let d = ds[i as usize].to_digit(16).unwrap() as i32 + 1;
                if d < 16 {
                    ds[i as usize] = hex_digit(d as usize);
                    break;
                }
                ds[i as usize] = '0';
                i -= 1;
            }
        }
        (ds.into_iter().collect(), e2)
    } else {
        // No precision: the 13 significant digits, trimmed.
        let s: String = digits.trim_end_matches('0').to_string();
        (if s.is_empty() { "0".to_string() } else { s }, e2)
    };
    let exp_letter = if upper { 'P' } else { 'p' };
    format!(
        "{}{}1.{}p{}{}",
        sign,
        prefix,
        frac_s,
        if e2 < 0 { '-' } else { '+' },
        e2.abs()
    )
    .replace('p', &exp_letter.to_string())
}
