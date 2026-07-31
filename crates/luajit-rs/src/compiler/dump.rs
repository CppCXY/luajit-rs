use crate::lex::Interner;
use crate::proto::{KGc, Proto};

const MAGIC: &[u8] = b"\x1bLJ";
const VERSION: u8 = 1;

struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { out: Vec::new() }
    }

    fn w_u8(&mut self, v: u8) {
        self.out.push(v);
    }

    fn w_u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn w_u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn w_f64(&mut self, v: f64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    fn w_bytes(&mut self, b: &[u8]) {
        self.out.extend_from_slice(b);
    }
}

fn write_kgc(w: &mut Writer, kgc: &KGc) {
    match kgc {
        KGc::Str(sid) => {
            w.w_u8(0);
            w.w_u32(*sid);
        }
        KGc::Proto(child) => {
            w.w_u8(1);
            write_proto(w, child);
        }
        KGc::ProtoRef(r) => {
            w.w_u8(1);
            write_proto(w, r.as_ref());
        }
        KGc::Table(_) | KGc::TableRef(_) => {
            w.w_u8(2);
        }
        KGc::CData(_) => {
            w.w_u8(3);
        }
    }
}

fn write_proto(w: &mut Writer, pt: &Proto) {
    w.w_u8(pt.flags);
    w.w_u8(pt.numparams);
    w.w_u8(pt.framesize);
    w.w_u32(pt.bc.len() as u32);
    w.w_u32(pt.kgc.len() as u32);
    w.w_u32(pt.kn.len() as u32);
    w.w_u32(pt.kstrv.len() as u32);
    w.w_u32(pt.uv.len() as u32);
    w.w_u32(pt.uvnames.len() as u32);
    w.w_u32(pt.firstline);
    w.w_u32(pt.numline);
    if let Some(sid) = pt.source {
        w.w_u8(1);
        w.w_u32(sid);
    } else {
        w.w_u8(0);
    }

    for &ins in &pt.bc {
        w.w_u32(ins);
    }
    for &ln in &pt.lines {
        w.w_u32(ln);
    }
    for &n in &pt.kn {
        w.w_f64(n);
    }
    for &sv in &pt.kstrv {
        w.w_u32(sv.to_bits() as u32);
    }
    for &u in &pt.uv {
        w.w_u16(u);
    }
    for name in &pt.uvnames {
        let b = name.as_bytes();
        w.w_u16(b.len() as u16);
        w.w_bytes(b);
    }
    for kgc in &pt.kgc {
        write_kgc(w, kgc);
    }
}

pub fn dump(pt: &Proto, _strs: &Interner, _chunk: &str, out: &mut Vec<u8>) {
    let mut w = Writer::new();
    w.w_bytes(MAGIC);
    w.w_u8(VERSION);
    let flags: u8 = 0;
    w.w_u8(flags);
    write_proto(&mut w, pt);
    *out = w.out;
}
