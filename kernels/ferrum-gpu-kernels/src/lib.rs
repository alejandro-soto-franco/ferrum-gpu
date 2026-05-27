//! Rust kernels for ferrum-gpu, compiled to PTX by cuda-oxide.
//!
//! Not `no_std`: the `#[cuda_module]` macro generates host-side launch
//! glue that references `std` and `cuda_host`.

use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

/// Elementwise add kernel.
///
/// `cuda_module` embeds the device artifact into the host binary and
/// generates a typed `module.vector_add(stream, cfg, &a, &b, &mut c)`
/// launch method consumed by `examples/vector-add-cuda-oxide`.
#[cuda_module]
pub mod vector_add_kernel {
    use super::*;

    /// `out[i] = a[i] + b[i]` for `i in 0..out.len()`.
    #[kernel]
    pub fn vector_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = a[i] + b[i];
        }
    }
}
