//! Focused alternating perf-gate for the N=256 warp-per-FFT kernel vs cuFFT.
//!
//! Reuses `alternating_bench` (same event-time, locked-clock methodology as
//! `perf-gate`) but dispatches `fft_c2c_256_warp` with its own w32/w256 twiddle
//! buffers (the shared harness supplies only one twiddle buffer, which this
//! closure ignores). Lets P3/P4 measure the real ferrum/cuFFT ratio and sweep
//! K = warps/block via the `WARP_BLOCK` env var WITHOUT touching the plan or
//! the production perf-gate. Run under `tools/bench-gpu-lock.sh` for a fair
//! locked-clock head-to-head.
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin perf-gate-warp256`
//!   env: WARP_BLOCK (threads/block, multiple of 32; default 256 => K=8).

use std::f32::consts::PI;

use anyhow::Result;
use cuda_core::{DeviceBuffer, LaunchConfig};
use ferrum_gpu_bench::{BATCH, alternating_bench, init_cuda_contexts};

include!("../../../ferrum-gpu-fft-kernels/src/fft256_warp_body.rs");

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    let module = fft256_warp::load(&core_ctx)?;
    let stream = core_ctx.default_stream();

    let block: u32 = std::env::var("WARP_BLOCK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let k = block / 32; // FFTs per block

    // w32: W_32^e e in 0..16; w256: W_256^e e in 0..256.
    let w32: Vec<f32> = (0..16)
        .flat_map(|e| {
            let t = -2.0 * PI * e as f32 / 32.0;
            [t.cos(), t.sin()]
        })
        .collect();
    let w256: Vec<f32> = (0..256)
        .flat_map(|e| {
            let t = -2.0 * PI * e as f32 / 256.0;
            [t.cos(), t.sin()]
        })
        .collect();
    let d_w32 = DeviceBuffer::from_host(&stream, &w32)?;
    let d_w256 = DeviceBuffer::from_host(&stream, &w256)?;

    let cfg = LaunchConfig {
        grid_dim: ((BATCH as u32 * 32) / block, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    let launch = |din: &DeviceBuffer<f32>,
                  _tw: &DeviceBuffer<f32>,
                  dout: &mut DeviceBuffer<f32>|
     -> Result<()> {
        module.fft_c2c_256_warp(stream.as_ref(), cfg, din, &d_w32, &d_w256, dout)?;
        Ok(())
    };

    let (ferr, cu) = alternating_bench(&core_ctx, &cudarc_ctx, 8, launch)?;
    let fe = ferr.event_med_s * 1.0e6 / BATCH as f64;
    let cu_us = cu.event_med_s * 1.0e6 / BATCH as f64;
    let ratio = fe / cu_us;
    println!(
        "N=256 warp (K={k} warps/block): ferrum {fe:.4} us  cufft {cu_us:.4} us  ratio {ratio:.3}  {}",
        if ratio < 1.0 { "BEATS cuFFT" } else if ratio <= 0.9 { "PASS gate" } else { "" }
    );
    Ok(())
}
