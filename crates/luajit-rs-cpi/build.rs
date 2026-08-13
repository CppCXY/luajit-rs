fn main() {
    println!("cargo:rerun-if-changed=c/ljrs_shim.c");
    cc::Build::new()
        .file("c/ljrs_shim.c")
        .flag_if_supported("-std=c11")
        .warnings(false)
        .compile("ljrs_shim");
}
