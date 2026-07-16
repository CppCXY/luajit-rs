use crate::bc::*;
use crate::lex::Interner;
use crate::proto::{KGc, Proto};

fn ctlsub(out: &mut Vec<u8>, b: u8) {
    match b {
        b'\n' => out.extend_from_slice(b"\\n"),
        b'\r' => out.extend_from_slice(b"\\r"),
        b'\t' => out.extend_from_slice(b"\\t"),
        c if c < 0x20 || c == 0x7f => {
            out.extend_from_slice(format!("\\{:03}", c).as_bytes());
        }
        c => out.push(c),
    }
}

fn str_disp(s: &[u8]) -> Vec<u8> {
    let mut esc = Vec::new();
    for &b in s {
        ctlsub(&mut esc, b);
    }
    let mut out = Vec::new();
    out.push(b'"');
    if s.len() > 40 {
        esc.truncate(40);
        out.extend_from_slice(&esc);
        out.push(b'"');
        out.push(b'~');
    } else {
        out.extend_from_slice(&esc);
        out.push(b'"');
    }
    out
}

fn num_disp(n: f64) -> Vec<u8> {
    g14(n).into_bytes()
}

fn g14(n: f64) -> String {
    if n == 0.0 {
        return if n.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
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

pub fn dump(pt: &Proto, strs: &Interner, chunk: &str, out: &mut Vec<u8>) {
    for k in pt.kgc.iter() {
        if let KGc::Proto(child) = k {
            dump(child, strs, chunk, out);
        }
    }

    let lastline = pt.firstline + pt.numline;
    out.extend_from_slice(
        format!("-- BYTECODE -- {}:{}-{}\n", chunk, pt.firstline, lastline).as_bytes(),
    );

    let n = pt.bc.len();
    let mut targets = vec![false; n + 1];
    for i in 1..n {
        let ins = pt.bc[i];
        let op = bc_op(ins);
        if bcmode_c(op) == BCMode::Jump as u32 {
            let t = (i as i64 + bc_d(ins) as i64 - 0x7fff) as usize;
            if t < targets.len() {
                targets[t] = true;
            }
        }
    }

    for (i, &ins) in pt.bc.iter().enumerate().skip(1) {
        let op = bc_op(ins);
        let ma = bcmode_a(op);
        let mb = bcmode_b(op);
        let mc = bcmode_c(op);
        let a = bc_a(ins);
        let name = BC_NAMES[op as usize];
        let prefix = if targets[i] { "=>" } else { "  " };
        let astr = if ma == 0 {
            String::new()
        } else {
            format!("{}", a)
        };
        let line = format!("{:04} {} {:<6} {:>3} ", i, prefix, name, astr);

        let mut d = bc_d(ins);

        if mc == BCMode::Jump as u32 {
            let t = i as i64 + d as i64 - 0x7fff;
            out.extend_from_slice(format!("{}=> {:04}\n", line, t).as_bytes());
            continue;
        }

        if mb != 0 {
            d &= 0xff;
        } else if mc == 0 {
            out.extend_from_slice(line.as_bytes());
            out.push(b'\n');
            continue;
        }

        let mut kc: Option<Vec<u8>> = None;
        if mc == BCMode::Str as u32 {
            if let KGc::Str(sid) = &pt.kgc[d as usize] {
                kc = Some(str_disp(strs.get(*sid)));
            }
        } else if mc == BCMode::Num as u32 {
            let mut v = pt.kn[d as usize];
            if op == BCOp::TSETM {
                v -= 2f64.powi(52);
            }
            kc = Some(num_disp(v));
        } else if mc == BCMode::Func as u32 {
            if let KGc::Proto(child) = &pt.kgc[d as usize] {
                kc = Some(format!("{}:{}", chunk, child.firstline).into_bytes());
            }
        } else if mc == BCMode::Uv as u32 {
            kc = Some(uvname(pt, d as usize));
        }

        if ma == BCMode::Uv as u32 {
            let ka = uvname(pt, a as usize);
            kc = Some(match kc {
                Some(c) => {
                    let mut v = ka;
                    v.extend_from_slice(b" ; ");
                    v.extend_from_slice(&c);
                    v
                }
                None => ka,
            });
        }

        if mb != 0 {
            let b = bc_b(ins);
            match kc {
                Some(c) => {
                    out.extend_from_slice(format!("{}{:>3} {:>3}  ; ", line, b, d).as_bytes());
                    out.extend_from_slice(&c);
                    out.push(b'\n');
                }
                None => {
                    out.extend_from_slice(format!("{}{:>3} {:>3}\n", line, b, d).as_bytes());
                }
            }
            continue;
        }

        if let Some(c) = kc {
            out.extend_from_slice(format!("{}{:>3}      ; ", line, d).as_bytes());
            out.extend_from_slice(&c);
            out.push(b'\n');
            continue;
        }

        if mc == BCMode::Lits as u32 && d > 32767 {
            let sd = d as i32 - 65536;
            out.extend_from_slice(format!("{}{:>3}\n", line, sd).as_bytes());
            continue;
        }
        out.extend_from_slice(format!("{}{:>3}\n", line, d).as_bytes());
    }
    out.push(b'\n');
}

fn uvname(pt: &Proto, idx: usize) -> Vec<u8> {
    pt.uvnames
        .get(idx)
        .map(|s| s.clone().into_bytes())
        .unwrap_or_default()
}
