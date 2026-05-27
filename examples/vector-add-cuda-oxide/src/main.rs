//! Plan 2 demo: vector_add written in Rust, compiled to PTX by cuda-oxide,
//! dispatched on the user's GPU.
//!
//! cuda-oxide's `#[cuda_module]` embeds the PTX into the binary crate at
//! link time, so the kernel lives inline here (same pattern as
//! cuda-oxide's own examples). Plan 3 will explore extracting the PTX
//! bytes through `ferrum-gpu`'s `Backend::load_module` once cuda-oxide
//! exposes a multi-crate kernel-bundling path.

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = a[i] + b[i]` for `i in 0..out.len()`.
    #[kernel]
    pub(crate) fn vector_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = a[i] + b[i];
        }
    }
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let n: u32 = 1 << 20;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n as f32) - (i as f32)).collect();

    let da = DeviceBuffer::from_host(&stream, &a)?;
    let db = DeviceBuffer::from_host(&stream, &b)?;
    let mut dc = DeviceBuffer::<f32>::zeroed(&stream, n as usize)?;

    println!("loading cuda-oxide-compiled vector_add module");
    let module = kernels::load(&ctx)?;

    println!("launching kernel");
    module.vector_add(&stream, LaunchConfig::for_num_elems(n), &da, &db, &mut dc)?;

    let c = dc.to_host_vec(&stream)?;
    for i in 0..(n as usize) {
        let expected = a[i] + b[i];
        if (c[i] - expected).abs() > 1e-6 {
            anyhow::bail!("mismatch at {}: got {}, expected {}", i, c[i], expected);
        }
    }
    println!("vector_add (cuda-oxide): {} elements verified", n);
    Ok(())
}
