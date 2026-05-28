//! cuFFT-comparison benchmark for ferrum-gpu-fft.
//!
//! Times the `ferrum-gpu-fft-kernels` Stockham radix-2 power-of-2 C2C
//! fallback against cuFFT (via cudarc's `cufft` feature) for batched 1D
//! complex-to-complex transforms at N in {256, 1024, 4096}. Both paths see
//! the same input data; cuFFT uses its own device buffers since cudarc's
//! `CudaSlice<float2>` and cuda-core's `DeviceBuffer<f32>` are not
//! interchangeable.
//!
//! Each measurement runs `WARMUP` warmup launches followed by `TRIALS`
//! timed launches, with a single `stream.synchronize()` bracketing the
//! timed window. Reported microseconds are per-batch averages.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext as CudarcContext;

use ferrum_gpu_fft::Plan;
use ferrum_gpu_fft_kernels::kernels;

const LOG_NS: &[u32] = &[8, 10, 12]; // N in {256, 1024, 4096}
const BATCH: usize = 256;
const WARMUP: usize = 10;
const TRIALS: usize = 100;

fn bench_ferrum(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    log_n: u32,
) -> Result<f64> {
    let n = 1usize << log_n;
    let total = n * BATCH;
    let plan = Plan::new(log_n, BATCH, false);
    let stream = ctx.default_stream();

    let input_flat: Vec<f32> = (0..total * 2).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut twiddles_flat: Vec<f32> = Vec::with_capacity((n - 1) * 2);
    for c in plan.twiddles() {
        twiddles_flat.push(c.re);
        twiddles_flat.push(c.im);
    }

    let dbuf_in = DeviceBuffer::from_host(&stream, &input_flat)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (BATCH as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    for _ in 0..WARMUP {
        module.fft_radix2_c2c_pow2_1d_fallback(
            stream.as_ref(),
            cfg,
            &dbuf_in,
            &dbuf_tw,
            &mut dbuf_out,
            log_n,
        )?;
    }
    stream.synchronize()?;

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        module.fft_radix2_c2c_pow2_1d_fallback(
            stream.as_ref(),
            cfg,
            &dbuf_in,
            &dbuf_tw,
            &mut dbuf_out,
            log_n,
        )?;
    }
    stream.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    Ok(dt / (TRIALS as f64))
}

fn bench_cufft(cudarc_ctx: &Arc<CudarcContext>, log_n: u32) -> Result<f64> {
    let n = 1usize << log_n;
    let total = n * BATCH;
    let stream = cudarc_ctx.default_stream();

    // Build the input as interleaved float2 = (re, im).
    let input_data: Vec<cufft_sys::float2> = (0..total)
        .map(|i| {
            let re = ((2 * i) as f32 * 0.001).sin();
            let im = ((2 * i + 1) as f32 * 0.001).sin();
            cufft_sys::float2 { x: re, y: im }
        })
        .collect();

    let mut d_in = stream
        .clone_htod(&input_data)
        .map_err(|e| anyhow!("htod input: {e}"))?;
    let mut d_out = stream
        .alloc_zeros::<cufft_sys::float2>(total)
        .map_err(|e| anyhow!("alloc out: {e}"))?;

    let plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        BATCH as i32,
        stream.clone(),
    )
    .map_err(|e| anyhow!("cufft plan_1d: {e:?}"))?;

    for _ in 0..WARMUP {
        plan.exec_c2c(&mut d_in, &mut d_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft exec: {e:?}"))?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        plan.exec_c2c(&mut d_in, &mut d_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft exec: {e:?}"))?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    Ok(dt / (TRIALS as f64))
}

fn main() -> Result<()> {
    // cuda-oxide CudaContext: hosts the in-tree Stockham kernel module.
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;

    // cudarc CudaContext: hosts cuFFT. Both point at GPU 0 of the same
    // driver; cuFFT and our kernel each operate over their own device
    // buffers so the two contexts don't have to share allocations.
    let cudarc_ctx = CudarcContext::new(0).map_err(|e| anyhow!("cudarc CudaContext::new: {e}"))?;

    println!(
        "ferrum-gpu-bench: 1D radix-2 C2C FFT, batch={}, warmup={}, trials={}",
        BATCH, WARMUP, TRIALS
    );
    println!(
        "{:<8} {:<14} {:<14} {:<10}",
        "N", "ferrum_us", "cufft_us", "ratio"
    );
    for &log_n in LOG_NS {
        let n = 1usize << log_n;
        let ferrum_s = bench_ferrum(&ctx, &module, log_n)?;
        let cufft_s = bench_cufft(&cudarc_ctx, log_n)?;
        let ferrum_us = ferrum_s * 1.0e6 / (BATCH as f64);
        let cufft_us = cufft_s * 1.0e6 / (BATCH as f64);
        let ratio = if cufft_us > 0.0 {
            ferrum_us / cufft_us
        } else {
            f64::NAN
        };
        println!(
            "{:<8} {:<14.3} {:<14.3} {:<10.2}",
            n, ferrum_us, cufft_us, ratio
        );
    }
    Ok(())
}
