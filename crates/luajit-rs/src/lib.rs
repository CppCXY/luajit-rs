pub mod bc;
pub mod dump;
pub mod lex;
pub mod parse;
pub mod strscan;

use std::panic::{catch_unwind, AssertUnwindSafe};

pub fn compile(src: Vec<u8>, chunkname: &str) -> Result<(parse::Proto, lex::Interner), String> {
    let name = chunkname.to_string();
    let result = catch_unwind(AssertUnwindSafe(move || {
        let parser = parse::Parser::new(src, name);
        parser.parse()
    }));
    match result {
        Ok(out) => Ok(out),
        Err(e) => {
            if let Some(ce) = e.downcast_ref::<lex::CompileError>() {
                Err(ce.0.clone())
            } else if let Some(s) = e.downcast_ref::<String>() {
                Err(s.clone())
            } else if let Some(s) = e.downcast_ref::<&str>() {
                Err((*s).to_string())
            } else {
                Err("unknown compile error".to_string())
            }
        }
    }
}

pub fn list_bytecode(src: Vec<u8>, chunkname: &str) -> Result<Vec<u8>, String> {
    let (pt, strs) = compile(src, chunkname)?;
    let mut out = Vec::new();
    dump::dump(&pt, &strs, chunkname, &mut out);
    Ok(out)
}
