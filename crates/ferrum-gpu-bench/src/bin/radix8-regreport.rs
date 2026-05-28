//! Prototype radix-8 butterfly kernel used to measure register/spill cost
//! through the cuda-oxide codegen path. Not functionally correct as an FFT,
//! just enough live-set pressure to be representative.

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn radix8_butterfly_inplace(x: &mut [f32; 16], tw: &[f32; 14]) {
        // 8 complex inputs as 16 floats: [re0, im0, re1, im1, ..., re7, im7].
        // 7 twiddle constants as 14 floats: w_k = (cos, sin) for k = 1..7.
        let mut t = [0.0f32; 16];
        for k in 0..8 {
            let mut sr = 0.0f32;
            let mut si = 0.0f32;
            for n in 0..8 {
                let (cw, sw) = if n == 0 {
                    (1.0, 0.0)
                } else {
                    let idx = ((k * n) % 8) - 1;
                    if idx == 0 { (1.0, 0.0) } else { (tw[2 * (idx % 7)], tw[2 * (idx % 7) + 1]) }
                };
                let xr = x[2 * n];
                let xi = x[2 * n + 1];
                sr += xr * cw - xi * sw;
                si += xr * sw + xi * cw;
            }
            t[2 * k] = sr;
            t[2 * k + 1] = si;
        }
        *x = t;
    }

    #[kernel]
    pub(crate) fn radix8_pressure(
        in_data: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let base = tid * 16;
        let mut x = [0.0f32; 16];
        for i in 0..16 {
            x[i] = in_data[base + i];
        }
        // Static twiddle constants approximating the radix-8 inner table.
        let tw: [f32; 14] = [
            0.7071068,  -0.7071068,
            0.0,        -1.0,
           -0.7071068,  -0.7071068,
           -1.0,         0.0,
           -0.7071068,   0.7071068,
            0.0,         1.0,
            0.7071068,   0.7071068,
        ];
        for _ in 0..4 {
            radix8_butterfly_inplace(&mut x, &tw);
        }
        for i in 0..16 {
            unsafe { *out_data.get_unchecked_mut(base + i) = x[i]; }
        }
    }
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let stream = ctx.default_stream();
    let input: Vec<f32> = (0..512 * 16).map(|i| (i as f32 * 0.001).sin()).collect();
    let dbuf_in = DeviceBuffer::from_host(&stream, &input)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, input.len())?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (512, 1, 1), shared_mem_bytes: 0 };
    module.radix8_pressure(stream.as_ref(), cfg, &dbuf_in, &mut dbuf_out)?;
    let _ = dbuf_out.to_host_vec(&stream)?;
    println!("radix8_pressure: ok");
    Ok(())
}
