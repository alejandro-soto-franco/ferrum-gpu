//! Operator harness for `ncu` to capture cuFFT kernel names at the three target sizes.
//!
//! Run as: `ncu --set full --csv target/release/cufft-ncu-trace > tools/ncu/cufft-blackwell-<date>.txt`
//! Then commit the output.

use anyhow::{Result, anyhow};
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext;

const LOG_NS: &[u32] = &[8, 10, 12];
const BATCH: usize = 256;
const REPS: usize = 5;

fn main() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("CudaContext::new: {e}"))?;
    let stream = ctx.default_stream();

    for &log_n in LOG_NS {
        let n = 1usize << log_n;
        let total = n * BATCH;
        let input: Vec<cufft_sys::float2> = (0..total)
            .map(|i| cufft_sys::float2 {
                x: (i as f32 * 0.001).sin(),
                y: 0.0,
            })
            .collect();
        let mut d_in = stream
            .clone_htod(&input)
            .map_err(|e| anyhow!("htod: {e}"))?;
        let mut d_out = stream
            .alloc_zeros::<cufft_sys::float2>(total)
            .map_err(|e| anyhow!("alloc: {e}"))?;
        let plan = CudaFft::plan_1d(
            n as i32,
            cufft_sys::cufftType::CUFFT_C2C,
            BATCH as i32,
            stream.clone(),
        )
        .map_err(|e| anyhow!("plan_1d: {e:?}"))?;
        for _ in 0..REPS {
            plan.exec_c2c(&mut d_in, &mut d_out, FftDirection::Forward)
                .map_err(|e| anyhow!("exec: {e:?}"))?;
        }
        stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        eprintln!("ran cuFFT N={n} batch={BATCH} x{REPS}");
    }
    Ok(())
}
