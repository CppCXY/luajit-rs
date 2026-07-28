//! OS library: `os.clock`, `os.date`, `os.difftime`, `os.execute`,
//! `os.exit`, `os.getenv`, `os.remove`, `os.rename`, `os.setlocale`,
//! `os.time`, `os.tmpname`.
//!
//! Uses the crate-private `time` module for WASM-compatible clock / time.

use crate::state::LuaState;
use crate::value::LuaValue;
use crate::{err::LuaResult, stdlib::time};

use super::{LibTarget, arg, err_bad_arg, push};
use crate::lual_reg;

fn os_clock(l: &mut LuaState) -> LuaResult<i32> {
    let elapsed = l.global().boot_time.elapsed_secs_f64();
    push(l, LuaValue::number(elapsed));
    Ok(1)
}

fn os_time(l: &mut LuaState) -> LuaResult<i32> {
    let v = arg(l, 0);
    if let Some(tbl) = v.as_table() {
        let get = |name: &str| -> Option<i64> {
            let sid = l.heap().intern(name.as_bytes());
            let key = l.heap().str_value(sid);
            tbl.as_ref().get(key).as_number().map(|n| n as i64)
        };
        let year = get("year").unwrap_or(0);
        let month = get("month").unwrap_or(0);
        let day = get("day").unwrap_or(0);
        let hour = get("hour").unwrap_or(12);
        let min = get("min").unwrap_or(0);
        let sec = get("sec").unwrap_or(0);
        if year == 0 || month == 0 || day == 0 {
            return Err(l.runtime_error(b"field missing in date table"));
        }
        let ts = civil_to_timestamp(year, month, day, hour, min, sec);
        push(l, LuaValue::number(ts as f64));
    } else if v.is_nil() {
        push(l, LuaValue::number(time::unix_secs() as f64));
    } else {
        return Err(l.runtime_error(b"table expected"));
    }
    Ok(1)
}

fn civil_to_timestamp(y: i64, mo: i64, d: i64, h: i64, m: i64, s: i64) -> i64 {
    let month = mo;
    let year = if month <= 2 { y - 1 } else { y };
    let era = if year >= 0 {
        year / 400
    } else {
        (year - 399) / 400
    };
    let yoe = (year - era * 400) as u64;
    let mp = if month <= 2 { month + 9 } else { month - 3 };
    let doy = (153 * mp as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = (era * 146097) + doe as i64 - 719468;
    days * 86400 + h * 3600 + m * 60 + s
}

fn os_date(l: &mut LuaState) -> LuaResult<i32> {
    let fmt = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => b"%c".to_vec(),
    };
    let ts = arg(l, 1)
        .as_number()
        .map_or_else(|| time::unix_secs() as i64, |n| n as i64);

    let days = ts.div_euclid(86400);
    let time_of_day = ts.rem_euclid(86400);
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    let (y, mo, d) = civil_from_days(days);
    let wday = ((days + 4).rem_euclid(7)) as u32;
    let yday = days - civil_to_days(y, 1, 1);

    if fmt == b"*t" {
        let t = l.heap().alloc_table(crate::table::LuaTable::new(0, 4));
        let set_int = |k: &str, v: i64| {
            let sid = l.heap().intern(k.as_bytes());
            t.as_mut()
                .set(l.heap().str_value(sid), LuaValue::number(v as f64));
        };
        set_int("year", y);
        set_int("month", mo);
        set_int("day", d);
        set_int("hour", h);
        set_int("min", m);
        set_int("sec", s);
        set_int("wday", wday as i64 + 1);
        set_int("yday", yday + 1);
        let sid = l.heap().intern(b"isdst");
        let dsid = l.heap().intern(b"false");
        t.as_mut()
            .set(l.heap().str_value(sid), l.heap().str_value(dsid));
        push(l, LuaValue::table(t));
    } else {
        let out = format_fmt(&fmt, y, mo, d, h as u64, m as u64, s as u64, wday, yday);
        let sid = l.heap().intern(out.as_bytes());
        push(l, l.heap().str_value(sid));
    }
    Ok(1)
}

fn civil_from_days(mut d: i64) -> (i64, i64, i64) {
    d += 719468;
    let era = (if d >= 0 { d } else { d - 146096 }) / 146097;
    let doe = (d - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as i64, d as i64)
}

fn civil_to_days(y: i64, m: i64, d: i64) -> i64 {
    let month = m;
    let year = if month <= 2 { y - 1 } else { y };
    let era = if year >= 0 {
        year / 400
    } else {
        (year - 399) / 400
    };
    let yoe = (year - era * 400) as u64;
    let mp = if month <= 2 { month + 9 } else { month - 3 };
    let doy = (153 * mp as u64 + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097) + doe as i64 - 719468
}

#[allow(clippy::too_many_arguments)]
fn format_fmt(
    fmt: &[u8],
    y: i64,
    mo: i64,
    d: i64,
    h: u64,
    m: u64,
    s: u64,
    wday: u32,
    yday: i64,
) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < fmt.len() {
        if fmt[i] == b'%' && i + 1 < fmt.len() {
            i += 1;
            let c = fmt[i];
            match c {
                b'Y' => out.push_str(&format!("{:04}", y)),
                b'm' => out.push_str(&format!("{:02}", mo)),
                b'd' => out.push_str(&format!("{:02}", d)),
                b'H' => out.push_str(&format!("{:02}", h)),
                b'M' => out.push_str(&format!("{:02}", m)),
                b'S' => out.push_str(&format!("{:02}", s)),
                b'w' => out.push_str(&format!("{}", wday)),
                b'a' => {
                    out.push_str(["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][wday as usize])
                }
                b'A' => out.push_str(
                    [
                        "Sunday",
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                    ][wday as usize],
                ),
                b'b' | b'h' => out.push_str(
                    [
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec",
                    ][(mo as usize).saturating_sub(1).min(11)],
                ),
                b'B' => out.push_str(
                    [
                        "January",
                        "February",
                        "March",
                        "April",
                        "May",
                        "June",
                        "July",
                        "August",
                        "September",
                        "October",
                        "November",
                        "December",
                    ][(mo as usize).saturating_sub(1).min(11)],
                ),
                b'c' => out.push_str(&format!(
                    "{} {:02} {:02}:{:02}:{:02} {}",
                    [
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"
                    ][(mo as usize).saturating_sub(1).min(11)],
                    d,
                    h,
                    m,
                    s,
                    y
                )),
                b'x' => out.push_str(&format!("{:02}/{:02}/{:02}", mo, d, y % 100)),
                b'X' => out.push_str(&format!("{:02}:{:02}:{:02}", h, m, s)),
                b'y' => out.push_str(&format!("{:02}", y % 100)),
                b'j' => out.push_str(&format!("{:03}", yday + 1)),
                b'%' => out.push('%'),
                _ => {
                    out.push('%');
                    out.push(c as char);
                }
            }
        } else {
            out.push(fmt[i] as char);
        }
        i += 1;
    }
    out
}

fn os_difftime(l: &mut LuaState) -> LuaResult<i32> {
    let t2 = arg(l, 0).as_number().unwrap_or(0.0);
    let t1 = arg(l, 1).as_number().unwrap_or(0.0);
    push(l, LuaValue::number(t2 - t1));
    Ok(1)
}

fn os_exit(l: &mut LuaState) -> LuaResult<i32> {
    #[cfg(target_arch = "wasm32")]
    {
        return Err(l.runtime_error(b"os.exit not available in WASM"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let code = arg(l, 0).as_number().unwrap_or(0.0) as i32;
        std::process::exit(code);
    }
}

fn os_execute(l: &mut LuaState) -> LuaResult<i32> {
    #[cfg(target_arch = "wasm32")]
    {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cmd = arg(l, 0);
        if cmd.is_nil() {
            push(l, LuaValue::boolean(true));
            return Ok(1);
        }
        let cmd_str = match cmd.as_string_id() {
            Some(sid) => String::from_utf8_lossy(l.str_static(sid)),
            None => return Err(err_bad_arg(l, 1, "os.execute", "string", "")),
        };
        let status = if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/C", &cmd_str])
                .status()
        } else {
            std::process::Command::new("sh")
                .arg("-c")
                .arg(&*cmd_str)
                .status()
        };
        match status {
            Ok(s) if s.success() => push(l, LuaValue::boolean(true)),
            Ok(_) => push(l, LuaValue::NIL),
            Err(_) => push(l, LuaValue::NIL),
        }
        Ok(1)
    }
}

fn os_getenv(l: &mut LuaState) -> LuaResult<i32> {
    #[cfg(target_arch = "wasm32")]
    {
        push(l, LuaValue::NIL);
        return Ok(1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let name = match arg(l, 0).as_string_id() {
            Some(sid) => l.str_static(sid),
            None => return Err(err_bad_arg(l, 1, "os.getenv", "string", "")),
        };
        let name_str = std::str::from_utf8(name).unwrap_or("");
        match std::env::var(name_str) {
            Ok(val) => {
                let sid = l.heap().intern(val.as_bytes());
                push(l, l.heap().str_value(sid));
            }
            Err(_) => push(l, LuaValue::NIL),
        }
        Ok(1)
    }
}

fn os_remove(l: &mut LuaState) -> LuaResult<i32> {
    #[cfg(target_arch = "wasm32")]
    {
        return Err(l.runtime_error(b"os.remove not available in WASM"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let name = match arg(l, 0).as_string_id() {
            Some(sid) => String::from_utf8_lossy(l.str_static(sid)),
            None => return Err(err_bad_arg(l, 1, "os.remove", "string", "")),
        };
        match std::fs::remove_file(name.as_ref()) {
            Ok(()) => push(l, LuaValue::boolean(true)),
            Err(e) => {
                let msg = format!("{}", e);
                let sid = l.heap().intern(msg.as_bytes());
                push(l, LuaValue::NIL);
                push(l, l.heap().str_value(sid));
                return Ok(2);
            }
        }
        Ok(1)
    }
}

fn os_rename(l: &mut LuaState) -> LuaResult<i32> {
    #[cfg(target_arch = "wasm32")]
    {
        return Err(l.runtime_error(b"os.rename not available in WASM"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let old = match arg(l, 0).as_string_id() {
            Some(sid) => String::from_utf8_lossy(l.str_static(sid)),
            None => return Err(err_bad_arg(l, 1, "os.rename", "string", "")),
        };
        let new = match arg(l, 1).as_string_id() {
            Some(sid) => String::from_utf8_lossy(l.str_static(sid)),
            None => return Err(err_bad_arg(l, 2, "os.rename", "string", "")),
        };
        match std::fs::rename(old.as_ref(), new.as_ref()) {
            Ok(()) => push(l, LuaValue::boolean(true)),
            Err(e) => {
                let msg = format!("{}", e);
                let sid = l.heap().intern(msg.as_bytes());
                push(l, LuaValue::NIL);
                push(l, l.heap().str_value(sid));
                return Ok(2);
            }
        }
        Ok(1)
    }
}

fn os_setlocale(l: &mut LuaState) -> LuaResult<i32> {
    let locale = match arg(l, 0).as_string_id() {
        Some(sid) => String::from_utf8_lossy(l.str_static(sid)),
        None => "C".to_string().into(),
    };
    let sid = l.heap().intern(locale.as_bytes());
    push(l, l.heap().str_value(sid));
    Ok(1)
}

fn os_tmpname(l: &mut LuaState) -> LuaResult<i32> {
    let ts = super::time::unix_secs();
    let name = format!("/tmp/lua_{}", ts);
    let sid = l.heap().intern(name.as_bytes());
    push(l, l.heap().str_value(sid));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"os", LibTarget::Global)
        .func(b"clock", os_clock)
        .func(b"date", os_date)
        .func(b"difftime", os_difftime)
        .func(b"execute", os_execute)
        .func(b"exit", os_exit)
        .func(b"getenv", os_getenv)
        .func(b"remove", os_remove)
        .func(b"rename", os_rename)
        .func(b"setlocale", os_setlocale)
        .func(b"time", os_time)
        .func(b"tmpname", os_tmpname)
        .build();
}
