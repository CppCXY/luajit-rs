//! OS library: `os.clock`, `os.date`, `os.difftime`, `os.execute`,
//! `os.exit`, `os.getenv`, `os.remove`, `os.rename`, `os.time`,
//! `os.tmpname`.

use crate::err::LuaResult;
use crate::state::LuaState;
use crate::value::LuaValue;

use super::{LibTarget, arg, err_bad_arg, push};
use crate::lual_reg;

fn os_clock(l: &mut LuaState) -> LuaResult<i32> {
    use std::time::UNIX_EPOCH;
    let now = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    push(l, LuaValue::number(now - l.global().boot_time));
    Ok(1)
}

fn os_date(l: &mut LuaState) -> LuaResult<i32> {
    let fmt = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid).to_vec(),
        None => b"%c".to_vec(),
    };
    let time = arg(l, 1).as_number().map(|t| t as i64);
    use std::time::UNIX_EPOCH;
    let dur = if let Some(ts) = time {
        if ts >= 0 {
            std::time::Duration::from_secs(ts as u64)
        } else {
            std::time::Duration::from_secs(0)
        }
    } else {
        std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    };
    let secs = dur.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    let (y, mo, d) = civil_from_days(days as i64);
    let wday = ((days + 4) % 7) as u32;

    if fmt == b"*t" {
        let t = l.heap().alloc_table(crate::table::LuaTable::new(0, 4));
        let set_int = |k: &str, v: i64| {
            let sid = l.heap().intern(k.as_bytes());
            t.as_mut()
                .set(l.heap().str_value(sid), LuaValue::number(v as f64));
        };
        let is_dst = false;
        set_int("year", y);
        set_int("month", mo);
        set_int("day", d);
        set_int("hour", h as i64);
        set_int("min", m as i64);
        set_int("sec", s as i64);
        set_int("wday", wday as i64 + 1);
        set_int("yday", 0);
        let dsid = l.heap().intern(if is_dst { b"true" } else { b"false" });
        t.as_mut().set_str(
            l.heap().str_value(l.heap().intern(b"isdst")),
            l.heap().str_value(dsid),
        );
        push(l, LuaValue::table(t));
    } else {
        let out = format_fmt(&fmt, y, mo, d, h, m, s, wday);
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

fn format_fmt(fmt: &[u8], y: i64, mo: i64, d: i64, h: u64, m: u64, s: u64, wday: u32) -> String {
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
                b'b' => out.push_str(
                    [
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec",
                    ][mo as usize - 1],
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
                    ][mo as usize - 1],
                ),
                b'c' => out.push_str(&format!(
                    "{} {:02} {:02}:{:02}:{:02} {}",
                    [
                        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                        "Nov", "Dec"
                    ][mo as usize - 1],
                    d,
                    h,
                    m,
                    s,
                    y
                )),
                b'x' => out.push_str(&format!("{:02}/{:02}/{:02}", mo, d, y % 100)),
                b'X' => out.push_str(&format!("{:02}:{:02}:{:02}", h, m, s)),
                b'y' => out.push_str(&format!("{:02}", y % 100)),
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
    let code = arg(l, 0).as_number().unwrap_or(0.0) as i32;
    std::process::exit(code);
}

fn os_getenv(l: &mut LuaState) -> LuaResult<i32> {
    let name = match arg(l, 0).as_string_id() {
        Some(sid) => l.str_static(sid),
        None => return Err(err_bad_arg(l, 1, "os.getenv", "string", "")),
    };
    let name_str = std::str::from_utf8(name).unwrap_or("");
    match std::env::var(name_str) {
        Ok(val) => {
            let sid = l.heap().intern(val.as_bytes());
            push(l, l.heap().str_value(sid));
            Ok(1)
        }
        Err(_) => {
            push(l, LuaValue::NIL);
            Ok(1)
        }
    }
}

fn os_time(l: &mut LuaState) -> LuaResult<i32> {
    use std::time::UNIX_EPOCH;
    let secs = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    push(l, LuaValue::number(secs as f64));
    Ok(1)
}

pub fn open(l: &mut LuaState) {
    lual_reg!(l, b"os", LibTarget::Global)
        .func(b"clock", os_clock)
        .func(b"date", os_date)
        .func(b"difftime", os_difftime)
        .func(b"exit", os_exit)
        .func(b"getenv", os_getenv)
        .func(b"time", os_time)
        .build();
}
