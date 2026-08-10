use std::num::Wrapping;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumSuffix {
    None,
    Ll,
    Ull,
    Imag,
}

pub struct ScanNumberResult {
    pub n: f64,
    pub u: u64,
    pub suffix: NumSuffix,
}

pub fn scan_number_full(s: &[u8]) -> Option<ScanNumberResult> {
    if s.is_empty() {
        return None;
    }
    let (body, suffix) = split_suffix(s);
    let (u, is_float) = if body.len() >= 2 && body[0] == b'0' && (body[1] | 0x20) == b'x' {
        scan_hex_to_u64(&body[2..])?
    } else if body.len() >= 2 && body[0] == b'0' && (body[1] | 0x20) == b'b' {
        (scan_bin_to_u64(&body[2..])?, true)
    } else {
        scan_dec_to_f64(body)?
    };
    // For float numbers with no suffix, n is the parsed f64 from body
    // For imaginary suffix, also parse as f64 (n is the numeric value).
    // If suffix is LL/ULL, n is the u64 cast to f64 (for display/tostring)
    let n = if is_float && (suffix == NumSuffix::None || suffix == NumSuffix::Imag) {
        std::str::from_utf8(body).ok()?.parse::<f64>().ok()?
    } else {
        u as f64
    };
    // For imaginary suffix, store the f64 bits in u for the cdata representation.
    let u = if suffix == NumSuffix::Imag {
        n.to_bits()
    } else {
        u
    };
    Some(ScanNumberResult { n, u, suffix })
}

pub fn scan_number(s: &[u8]) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    // Lua 5.1 `luaO_str2d`: skip leading whitespace, parse the numeric
    // token, skip trailing whitespace, then require the end of string
    // (math.lua: `"2" + " 3e0 "` works, but `tonumber("1  a")` is nil).
    let start = skip_leading_space(s);
    if start >= s.len() {
        return None;
    }
    let rest = &s[start..];
    let hex = rest.len() >= 2 && rest[0] == b'0' && (rest[1] | 0x20) == b'x';
    let end = if hex {
        let e = hex_token_len(rest);
        if e <= 2 {
            return None;
        }
        e
    } else {
        let e = dec_token_len(rest);
        if e == 0 {
            return None;
        }
        e
    };
    // Trailing whitespace is allowed; anything else is invalid.
    let mut after = end;
    while after < rest.len() && (rest[after] as char).is_ascii_whitespace() {
        after += 1;
    }
    if after != rest.len() {
        return None;
    }
    if hex {
        scan_hex(&rest[..end])
    } else {
        scan_dec(&rest[..end])
    }
}

/// Length of the leading run of ASCII whitespace.
fn skip_leading_space(s: &[u8]) -> usize {
    let mut i = 0;
    while i < s.len() && (s[i] as char).is_ascii_whitespace() {
        i += 1;
    }
    i
}

/// Length of the longest decimal-number prefix of `s` (sign, digits,
/// optional fraction and exponent). `0` when it cannot start a number.
fn dec_token_len(s: &[u8]) -> usize {
    let mut i = 0;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }
    let mut seen_any = false;
    let mut seen_exp = false;
    while i < s.len() {
        match s[i] {
            b'0'..=b'9' => {
                seen_any = true;
                i += 1;
            }
            b'.' => {
                if seen_exp {
                    break;
                }
                seen_any = true; // ".5" and "3." are valid.
                i += 1;
            }
            b'e' | b'E' => {
                if seen_exp {
                    break;
                }
                seen_exp = true;
                i += 1;
                if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
                    i += 1;
                }
            }
            _ => break,
        }
    }
    if seen_any { i } else { 0 }
}

/// Length of the longest hex-number prefix (0x..., optional `p` exponent).
fn hex_token_len(s: &[u8]) -> usize {
    let mut i = 2; // past "0x"
    while i < s.len() {
        let c = s[i];
        if c.is_ascii_hexdigit() {
            i += 1;
        } else if c == b'.' {
            i += 1;
        } else {
            break;
        }
    }
    i
}

fn scan_dec(s: &[u8]) -> Option<f64> {
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    let mut exp_digits = false;
    let mut i = 0;
    // Leading sign (LuaJIT's strscan accepts "-1.5" and "+1e2").
    let neg = if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        let n = s[i] == b'-';
        i += 1;
        n
    } else {
        false
    };
    let mag_start = i;
    while i < s.len() {
        let c = s[i];
        match c {
            b'0'..=b'9' => {
                if seen_exp {
                    exp_digits = true;
                } else {
                    seen_digit = true;
                }
            }
            b'.' => {
                if seen_dot || seen_exp {
                    return None;
                }
                seen_dot = true;
            }
            b'e' | b'E' => {
                if seen_exp || !seen_digit {
                    return None;
                }
                seen_exp = true;
                if i + 1 < s.len() && (s[i + 1] == b'+' || s[i + 1] == b'-') {
                    i += 1;
                }
            }
            _ => return None,
        }
        i += 1;
    }
    if (!seen_digit && !seen_dot) || (seen_exp && !exp_digits) {
        return None;
    }
    // Parse the magnitude (without the leading sign, which the full
    // string's parse would already apply) and negate explicitly.
    let mut v = std::str::from_utf8(&s[mag_start..])
        .ok()?
        .parse::<f64>()
        .ok()?;
    if neg {
        v = -v;
    }
    Some(v)
}

fn scan_hex(s: &[u8]) -> Option<f64> {
    let mut mant: u64 = 0;
    let mut exp: i32 = 0;
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u64,
            b'a'..=b'f' => (c - b'a' + 10) as u64,
            b'A'..=b'F' => (c - b'A' + 10) as u64,
            b'.' => {
                if seen_dot {
                    return None;
                }
                seen_dot = true;
                i += 1;
                continue;
            }
            b'p' | b'P' => break,
            _ => return None,
        };
        seen_digit = true;
        if mant < (1u64 << 60) {
            mant = mant * 16 + d;
            if seen_dot {
                exp -= 4;
            }
        } else if !seen_dot {
            exp += 4;
        }
        i += 1;
    }
    if !seen_digit {
        return None;
    }
    if i < s.len() {
        i += 1;
        let mut neg = false;
        if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
            neg = s[i] == b'-';
            i += 1;
        }
        if i >= s.len() {
            return None;
        }
        let mut e: i32 = 0;
        while i < s.len() {
            let c = s[i];
            if !c.is_ascii_digit() {
                return None;
            }
            e = e.saturating_mul(10).saturating_add((c - b'0') as i32);
            i += 1;
        }
        exp += if neg { -e } else { e };
    }
    Some((mant as f64) * (exp as f64).exp2())
}

fn split_suffix(s: &[u8]) -> (&[u8], NumSuffix) {
    let len = s.len();
    // Handle 'i' / 'I' suffix (imaginary / complex)
    if let Some((last, rest)) = s.split_last()
        && (*last == b'i' || *last == b'I')
    {
        return (rest, NumSuffix::Imag);
    }
    if len >= 3 {
        let last3 = &s[len - 3..];
        if (last3[0] == b'U' || last3[0] == b'u')
            && (last3[1] == b'L' || last3[1] == b'l')
            && (last3[2] == b'L' || last3[2] == b'l')
        {
            return (&s[..len - 3], NumSuffix::Ull);
        }
    }
    if len >= 2 {
        let last2 = &s[len - 2..];
        if (last2[0] == b'L' || last2[0] == b'l') && (last2[1] == b'L' || last2[1] == b'l') {
            if len >= 3 && (s[len - 3] == b'U' || s[len - 3] == b'u') {
                return (&s[..len - 3], NumSuffix::Ull);
            }
            return (&s[..len - 2], NumSuffix::Ll);
        }
    }
    (s, NumSuffix::None)
}

fn scan_hex_to_u64(s: &[u8]) -> Option<(u64, bool)> {
    let mut mant: Wrapping<u64> = Wrapping(0);
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut i = 0;
    while i < s.len() {
        let c = s[i];
        let d = match c {
            b'0'..=b'9' => (c - b'0') as u64,
            b'a'..=b'f' => (c - b'a' + 10) as u64,
            b'A'..=b'F' => (c - b'A' + 10) as u64,
            b'.' => {
                if seen_dot {
                    return None;
                }
                seen_dot = true;
                i += 1;
                continue;
            }
            b'p' | b'P' => break,
            b'_' => {
                i += 1;
                continue;
            }
            _ => return None,
        };
        seen_digit = true;
        mant = mant * Wrapping(16) + Wrapping(d);
        i += 1;
    }
    if !seen_digit {
        return None;
    }
    let mut exp: i32 = 0;
    if i < s.len() && (s[i] == b'p' || s[i] == b'P') {
        i += 1;
        let mut neg = false;
        if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
            neg = s[i] == b'-';
            i += 1;
        }
        let mut e: i32 = 0;
        while i < s.len() {
            let c = s[i];
            if c == b'_' {
                i += 1;
                continue;
            }
            if !c.is_ascii_digit() {
                return None;
            }
            e = e.saturating_mul(10).saturating_add((c - b'0') as i32);
            i += 1;
        }
        exp = if neg { -e } else { e };
    }
    if seen_dot || exp != 0 {
        let mut bits = mant.0 as f64;
        bits *= (exp as f64).exp2();
        let u = bits as u64;
        Some((u, true))
    } else {
        Some((mant.0, false))
    }
}

pub fn scan_bin_to_u64(s: &[u8]) -> Option<u64> {
    let mut val: Wrapping<u64> = Wrapping(0);
    for &c in s {
        match c {
            b'0' => val *= Wrapping(2),
            b'1' => val = val * Wrapping(2) + Wrapping(1),
            b'_' => continue,
            _ => return None,
        }
    }
    Some(val.0)
}

fn scan_dec_to_f64(s: &[u8]) -> Option<(u64, bool)> {
    let mut i = 0;
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    let mut exp_digits = false;
    while i < s.len() {
        let c = s[i];
        match c {
            b'0'..=b'9' => {
                if seen_exp {
                    exp_digits = true;
                } else {
                    seen_digit = true;
                }
            }
            b'.' => {
                if seen_dot || seen_exp {
                    return None;
                }
                seen_dot = true;
            }
            b'e' | b'E' => {
                if seen_exp || !seen_digit {
                    return None;
                }
                seen_exp = true;
                if i + 1 < s.len() && (s[i + 1] == b'+' || s[i + 1] == b'-') {
                    i += 1;
                }
            }
            b'_' => {
                i += 1;
                continue;
            }
            _ => return None,
        }
        i += 1;
    }
    if !seen_digit || (seen_exp && !exp_digits) {
        return None;
    }
    if seen_dot || seen_exp {
        let s_clean: Vec<u8> = s.iter().filter(|&&c| c != b'_').copied().collect();
        let n = std::str::from_utf8(&s_clean).ok()?.parse::<f64>().ok()?;
        Some((n.to_bits(), true))
    } else {
        // Pure integer — parse directly to u64 to avoid f64 precision loss
        let mut val: Wrapping<u64> = Wrapping(0);
        for &c in s {
            if c == b'_' {
                continue;
            }
            let d = (c - b'0') as u64;
            val = val * Wrapping(10) + Wrapping(d);
        }
        Some((val.0, false))
    }
}
