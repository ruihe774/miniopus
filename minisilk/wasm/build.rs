// Linker settings of the cdylib.
//
// The shadow stack (`__stack_pointer`, which rustc/lld place at the start of linear memory, 1 MiB by
// default) only holds the locals of the Rust frames: all the scratch buffers are in the static
// pseudostack (`minisilk::stack`). Its measured peak is about 36 KiB (see PORTING.md, "Memory"), so
// 48 KiB leaves a margin.
// `MINISILK_SHADOW_STACK=<bytes>` overrides the size (to measure the peak, see PORTING.md).
fn main() {
    println!("cargo:rerun-if-env-changed=MINISILK_SHADOW_STACK");
    let size = std::env::var("MINISILK_SHADOW_STACK").unwrap_or_else(|_| "49152".into());
    println!("cargo:rustc-link-arg-cdylib=-zstack-size={size}");
}
