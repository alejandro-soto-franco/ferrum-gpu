//! Minimal kernel that derefs a CuSimd<f32, 4> read from global memory.
//! Used to verify cuda-oxide lowers the load to `ld.global.v4.f32`.
//!
//! Inspect the PTX with: `make ptx-cusimd-dump` or
//! `find /home/cargo-targets/ferrum-gpu -name 'cusimd_ptx_dump*.ptx' -exec cat {} \;`.

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_device::cusimd::CuSimd;
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub(crate) fn cusimd_v4_load(
        in_data: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let base = tid * 4;
        let v: CuSimd<f32, 4> = unsafe {
            *(in_data.as_ptr().add(base) as *const CuSimd<f32, 4>)
        };
        unsafe {
            *out_data.get_unchecked_mut(base + 0) = v[0];
            *out_data.get_unchecked_mut(base + 1) = v[1];
            *out_data.get_unchecked_mut(base + 2) = v[2];
            *out_data.get_unchecked_mut(base + 3) = v[3];
        }
    }
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let stream = ctx.default_stream();
    let input: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let dbuf_in = DeviceBuffer::from_host(&stream, &input)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    module.cusimd_v4_load(stream.as_ref(), cfg, &dbuf_in, &mut dbuf_out)?;
    let out = dbuf_out.to_host_vec(&stream)?;
    assert_eq!(&out[..16], &input[..16]);
    println!("cusimd_v4_load: ok");
    Ok(())
}
