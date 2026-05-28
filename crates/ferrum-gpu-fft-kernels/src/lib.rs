//! cuda-oxide-compiled FFT kernels for ferrum-gpu-fft.
//!
//! Hosts the v0.1 generic Stockham fallback plus three per-size specialised
//! kernels for N in {256, 1024, 4096}. The `#[cuda_module]` embeds PTX into
//! this crate so the three consumers (`examples/fft-1d-c2c`,
//! `crates/ferrum-gpu-bench`, `crates/ferrum-gpu-py`) share one copy.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Placeholder kernel; will be replaced in Task 1.2.
    #[kernel]
    pub(crate) fn _placeholder() {
        let _ = thread::threadIdx_x();
    }
}

pub use kernels::LoadedModule;

/// Load all kernels into the given CUDA context.
pub fn load(ctx: &std::sync::Arc<cuda_core::CudaContext>) -> anyhow::Result<LoadedModule> {
    kernels::load(ctx).map_err(Into::into)
}
