use crate::proto::{KGc, Proto};
use crate::runtime::cdata::CData;
use crate::string::Interner;
use crate::table::LuaTable;
use crate::value::LuaValue;

const MAGIC: &[u8] = b"\x1bLJ";
const VERSION: u8 = 1;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    fn read_u16(&mut self) -> u16 {
        let b = &self.data[self.pos..self.pos + 2];
        self.pos += 2;
        u16::from_le_bytes([b[0], b[1]])
    }

    fn read_u32(&mut self) -> u32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn read_f64(&mut self) -> f64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    fn read_bytes(&mut self, len: usize) -> &[u8] {
        let s = &self.data[self.pos..self.pos + len];
        self.pos += len;
        s
    }
}

fn read_kgc(r: &mut Reader, strs: &mut Interner) -> KGc {
    let tag = r.read_u8();
    match tag {
        0 => {
            let sid: u32 = r.read_u32();
            KGc::Str(sid)
        }
        1 => KGc::Proto(Box::new(read_proto(r, strs))),
        2 => KGc::Table(Box::new(LuaTable::new(0, 0))),
        3 => KGc::CData(Box::new(CData {
            ctypeid: 0,
            data: Box::new([]),
            base: None,
            offset: 0,
        })),
        _ => panic!("bad kgc tag {tag}"),
    }
}

fn read_proto(r: &mut Reader, strs: &mut Interner) -> Proto {
    let flags = r.read_u16();
    let numparams = r.read_u8();
    let framesize = r.read_u8();
    let bc_len = r.read_u32() as usize;
    let kgc_len = r.read_u32() as usize;
    let kn_len = r.read_u32() as usize;
    let kstrv_len = r.read_u32() as usize;
    let uv_len = r.read_u32() as usize;
    let uvnames_len = r.read_u32() as usize;
    let firstline = r.read_u32();
    let numline = r.read_u32();
    let has_source = r.read_u8();
    let source = if has_source != 0 {
        let len = r.read_u32() as usize;
        let bytes = r.read_bytes(len);
        Some(strs.intern(bytes))
    } else {
        None
    };

    let mut bc = Vec::with_capacity(bc_len);
    for _ in 0..bc_len {
        bc.push(r.read_u32());
    }

    let mut lines = Vec::with_capacity(bc_len);
    for _ in 0..bc_len {
        lines.push(r.read_u32());
    }

    let mut kn = Vec::with_capacity(kn_len);
    for _ in 0..kn_len {
        kn.push(r.read_f64());
    }

    let mut kstrv = Vec::with_capacity(kstrv_len);
    for _ in 0..kstrv_len {
        let raw = r.read_u32();
        kstrv.push(if raw == 0xFFFF_FFFF {
            LuaValue::NIL
        } else {
            LuaValue::from_bits(raw as u64)
        });
    }

    let mut uv = Vec::with_capacity(uv_len);
    for _ in 0..uv_len {
        uv.push(r.read_u16());
    }

    let mut uvnames = Vec::with_capacity(uvnames_len);
    for _ in 0..uvnames_len {
        let len = r.read_u16() as usize;
        let name_bytes = r.read_bytes(len);
        uvnames.push(String::from_utf8_lossy(name_bytes).into_owned());
    }

    let mut kgc = Vec::with_capacity(kgc_len);
    for _ in 0..kgc_len {
        kgc.push(read_kgc(r, strs));
    }

    Proto {
        bc,
        lines,
        kgc,
        kn,
        kstrv,
        uv,
        uvnames,
        varnames: Vec::new(),
        flags,
        numparams,
        framesize,
        firstline,
        numline,
        source,
    }
}

pub fn undump(data: &[u8], strs: &mut Interner) -> Result<Proto, String> {
    if data.len() < 5 || &data[..3] != MAGIC {
        return Err("not a valid bytecode chunk".to_string());
    }
    let mut r = Reader::new(data);
    r.pos = 3;
    let version = r.read_u8();
    if version != VERSION {
        return Err(format!(
            "bytecode version mismatch: got {version}, expected {VERSION}"
        ));
    }
    let _flags = r.read_u8();
    // Need to recursively read child protos first, then the main proto.
    // The dump writes main proto last (with children first in kgc).
    // The undump reads main proto and kgc children are read recursively.
    Ok(read_proto(&mut r, strs))
}
