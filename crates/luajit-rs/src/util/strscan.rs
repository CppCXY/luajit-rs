pub fn scan_number(s: &[u8]) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    if s.len() >= 2 && s[0] == b'0' && (s[1] | 0x20) == b'x' {
        return scan_hex(&s[2..]);
    }
    scan_dec(s)
}

fn scan_dec(s: &[u8]) -> Option<f64> {
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exp = false;
    let mut exp_digits = false;
    let mut i = 0;
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
    if !seen_digit || (seen_exp && !exp_digits) {
        return None;
    }
    std::str::from_utf8(s).ok()?.parse::<f64>().ok()
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
