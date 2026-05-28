//! Source of truth for the ferrum-gpu-fft kernels.
//!
//! This crate hosts `src/kernels_body.rs`, which contains the full
//! `#[cuda_module] mod kernels { ... }` block for the in-tree FFT kernels.
//! Consumer binaries (`crates/ferrum-gpu-bench`, `examples/fft-1d-c2c`,
//! `crates/ferrum-gpu-py`) attach the kernel module via
//! `include!(".../kernels_body.rs");` at the top of their crate root.
//!
//! Why no `#[cuda_module]` here, and no `pub mod kernels`: cuda-oxide's
//! `#[cuda_module]` proc macro requires an INLINE module (it rejects
//! `#[path]` file modules) AND embeds PTX into the binary crate that owns
//! the include site (it does not propagate through library rlibs). So we
//! cannot host a usable `LoadedModule` in a library; the consuming binary
//! must own its own copy of the module via `include!`.
//!
//! See `kernels_body.rs` for the header that documents the include
//! contract from the consumer side.

/// Absolute filesystem path to `kernels_body.rs`, useful for diagnostics
/// and tooling.
pub const KERNELS_BODY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels_body.rs");
