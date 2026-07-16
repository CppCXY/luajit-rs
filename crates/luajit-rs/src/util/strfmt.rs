//! Number and `string.format` formatting, mirroring the pieces of
//! `lj_strfmt*` the runtime needs.

/// Format a double like LuaJIT's `STRFMT_G14` (`%.14g`, with integral values
/// printed without a decimal point and `inf`/`nan` spellings).
pub fn g14(n: f64) -> String {
    if n == 0.0 {
        return if n.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    if n.is_nan() {
        return "nan".to_string();
    }
    if n.is_infinite() {
        return if n < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let mant = format!("{:.13e}", n);
    let (m, e) = mant.split_once('e').unwrap();
    let exp: i32 = e.parse().unwrap();
    if !(-4..14).contains(&exp) {
        let m = m.trim_end_matches('0').trim_end_matches('.');
        format!("{}e{}{:02}", m, if exp < 0 { '-' } else { '+' }, exp.abs())
    } else {
        let prec = (13 - exp).max(0) as usize;
        let s = format!("{:.*}", prec, n);
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    }
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
