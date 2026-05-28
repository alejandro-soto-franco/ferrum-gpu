//! Single-source-of-truth for ferrum-gpu-fft kernels.
//!
//! cuda-oxide's `#[cuda_module]` embeds PTX into the building binary crate only,
//! not propagated through library-crate rlibs. So each consumer
//! (`examples/fft-1d-c2c`, `crates/ferrum-gpu-bench`, `crates/ferrum-gpu-py`)
//! must wrap its own `#[cuda_module] mod kernels` declaration. To keep ONE
//! source of truth for kernel definitions, the kernel bodies live in
//! `kernels_body.rs` and each consumer attaches that file as its `kernels`
//! module via `#[path = ".../kernels_body.rs"] mod kernels;`.
//!
//! This crate's own `#[cuda_module] mod kernels` exists for hosting
//! kernel-cross-check tests under `tests/`. Downstream binaries do not depend
//! on this crate; they attach `kernels_body.rs` directly via `#[path]`.

#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

use cuda_host::cuda_module;

#[cuda_module]
#[path = "kernels_body.rs"]
pub mod kernels;

pub use kernels::LoadedModule;

/// Absolute filesystem path to `kernels_body.rs`. Useful for diagnostics.
pub const KERNELS_BODY_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/kernels_body.rs");
