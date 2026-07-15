use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut list = false;
    let mut file: Option<&str> = None;
    for a in &args[1..] {
        if a == "-bl" || a == "--list" {
            list = true;
        } else if file.is_none() {
            file = Some(a);
        } else {
            eprintln!("usage: luajit-rs -bl <file.lua>");
            exit(1);
        }
    }
    let Some(file) = file else {
        eprintln!("usage: luajit-rs -bl <file.lua>");
        exit(1);
    };
    let src = match std::fs::read(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open {}: {}", file, e);
            exit(1);
        }
    };
    let chunkname = std::path::Path::new(file)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| file.to_string());
    if !list {
        eprintln!("note: only -bl (bytecode listing) is implemented so far");
    }
    match luajit_rs::list_bytecode(src, &chunkname) {
        Ok(out) => {
            use std::io::Write;
            std::io::stdout().write_all(&out).unwrap();
        }
        Err(e) => {
            eprintln!("luajit-rs: {}", e);
            exit(1);
        }
    }
}
