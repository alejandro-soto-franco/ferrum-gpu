//! cuFFT-comparison benchmark for ferrum-gpu-fft.
//!
//! Times the in-tree cuda-oxide-compiled Stockham radix-2 power-of-2 C2C
//! kernel against cuFFT (via cudarc's `cufft` feature) for batched 1D
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
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext as CudarcContext;

use ferrum_gpu_fft::Plan;

#[cuda_module]
mod kernels {
    use super::*;

    /// Single-block-per-lane radix-2 Stockham FFT. Body matches the kernel
    /// in `examples/fft-1d-c2c/src/main.rs` and `crates/ferrum-gpu-py/src/lib.rs`.
    #[kernel]
    pub(crate) fn fft_radix2_c2c_pow2_1d(
        in_data: &[f32],
        twiddles: &[f32],
        mut out_data: DisjointSlice<f32>,
        log_n: u32,
    ) {
        static mut PING: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut PONG: SharedArray<f32, 8192> = SharedArray::UNINIT;

        let n = 1usize << log_n;
        let half_n = n >> 1;

        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x() as usize;
        let lane_off = block * n * 2;

        {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    PING[i0] = in_data[lane_off + i0];
                    PING[i0 + 1] = in_data[lane_off + i0 + 1];
                    PING[i1] = in_data[lane_off + i1];
                    PING[i1 + 1] = in_data[lane_off + i1 + 1];
                }
                t += block_dim;
            }
        }
        thread::sync_threads();

        let mut s = 1u32;
        let mut src_is_ping = true;
        while s <= log_n {
            let m_half = 1usize << (s - 1);
            let m = 1usize << s;
            let stage_off = n - m;

            let mut b = tid;
            while b < half_n {
                let j = b / m_half;
                let k = b - j * m_half;
                let tw_i = 2 * (stage_off + k);
                let w_re = twiddles[tw_i];
                let w_im = twiddles[tw_i + 1];

                let src_a = 2 * (j * m_half + k);
                let src_b = 2 * (j * m_half + k + half_n);
                let dst_a = 2 * (j * m + k);
                let dst_b = 2 * (j * m + k + m_half);

                if src_is_ping {
                    let a_re = unsafe { PING[src_a] };
                    let a_im = unsafe { PING[src_a + 1] };
                    let b_re = unsafe { PING[src_b] };
                    let b_im = unsafe { PING[src_b + 1] };
                    let wb_re = w_re * b_re - w_im * b_im;
                    let wb_im = w_re * b_im + w_im * b_re;
                    unsafe {
                        PONG[dst_a] = a_re + wb_re;
                        PONG[dst_a + 1] = a_im + wb_im;
                        PONG[dst_b] = a_re - wb_re;
                        PONG[dst_b + 1] = a_im - wb_im;
                    }
                } else {
                    let a_re = unsafe { PONG[src_a] };
                    let a_im = unsafe { PONG[src_a + 1] };
                    let b_re = unsafe { PONG[src_b] };
                    let b_im = unsafe { PONG[src_b + 1] };
                    let wb_re = w_re * b_re - w_im * b_im;
                    let wb_im = w_re * b_im + w_im * b_re;
                    unsafe {
                        PING[dst_a] = a_re + wb_re;
                        PING[dst_a + 1] = a_im + wb_im;
                        PING[dst_b] = a_re - wb_re;
                        PING[dst_b + 1] = a_im - wb_im;
                    }
                }
                b += block_dim;
            }
            thread::sync_threads();
            src_is_ping = !src_is_ping;
            s += 1;
        }

        if src_is_ping {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    *out_data.get_unchecked_mut(lane_off + i0) = PING[i0];
                    *out_data.get_unchecked_mut(lane_off + i0 + 1) = PING[i0 + 1];
                    *out_data.get_unchecked_mut(lane_off + i1) = PING[i1];
                    *out_data.get_unchecked_mut(lane_off + i1 + 1) = PING[i1 + 1];
                }
                t += block_dim;
            }
        } else {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    *out_data.get_unchecked_mut(lane_off + i0) = PONG[i0];
                    *out_data.get_unchecked_mut(lane_off + i0 + 1) = PONG[i0 + 1];
                    *out_data.get_unchecked_mut(lane_off + i1) = PONG[i1];
                    *out_data.get_unchecked_mut(lane_off + i1 + 1) = PONG[i1 + 1];
                }
                t += block_dim;
            }
        }
    }
}

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
        module.fft_radix2_c2c_pow2_1d(
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
        module.fft_radix2_c2c_pow2_1d(
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
