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
    pub(crate) fn fft_radix2_c2c_pow2_1d_fallback(
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

/// Per-launch timing report (seconds), averaged over TRIALS.
#[derive(Debug, Clone, Copy)]
pub struct BenchSample {
    /// Median per-launch GPU time (CUDA events bracketing each kernel).
    pub event_med_s: f64,
    /// Mean per-launch wall-clock time (Instant around the TRIALS loop / TRIALS).
    pub wall_mean_s: f64,
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Alternating ferrum + cuFFT measurement at a single size.
///
/// Each timed iteration runs one ferrum launch (cuda-core stream) immediately
/// followed by one cuFFT launch (cudarc stream), with separate event pairs
/// bracketing each. Interleaving reduces per-size variance from background
/// activity (DVFS, contention) that would otherwise affect the two backends
/// asymmetrically when they run in sequential blocks.
fn alternating_bench(
    core_ctx: &Arc<CudaContext>,
    cudarc_ctx: &Arc<CudarcContext>,
    module: &kernels::LoadedModule,
    log_n: u32,
) -> Result<(BenchSample, BenchSample)> {
    let n = 1usize << log_n;
    let total = n * BATCH;
    let plan = Plan::new(log_n, BATCH, false);

    // --- ferrum (cuda-core) setup ---
    let core_stream = core_ctx.default_stream();
    let input_flat: Vec<f32> = (0..total * 2).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut twiddles_flat: Vec<f32> = Vec::with_capacity((n - 1) * 2);
    for c in plan.twiddles() {
        twiddles_flat.push(c.re);
        twiddles_flat.push(c.im);
    }
    let f_dbuf_in = DeviceBuffer::from_host(&core_stream, &input_flat)?;
    let f_dbuf_tw = DeviceBuffer::from_host(&core_stream, &twiddles_flat)?;
    let mut f_dbuf_out = DeviceBuffer::<f32>::zeroed(&core_stream, total * 2)?;
    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (BATCH as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch_ferrum = |dbuf_in: &DeviceBuffer<f32>,
                         dbuf_tw: &DeviceBuffer<f32>,
                         dbuf_out: &mut DeviceBuffer<f32>|
     -> Result<()> {
        module.fft_radix2_c2c_pow2_1d_fallback(
            core_stream.as_ref(),
            cfg,
            dbuf_in,
            dbuf_tw,
            dbuf_out,
            log_n,
        )?;
        Ok(())
    };

    // --- cuFFT (cudarc) setup ---
    let cu_stream = cudarc_ctx.default_stream();
    let cu_input_data: Vec<cufft_sys::float2> = (0..total)
        .map(|i| {
            let re = ((2 * i) as f32 * 0.001).sin();
            let im = ((2 * i + 1) as f32 * 0.001).sin();
            cufft_sys::float2 { x: re, y: im }
        })
        .collect();
    let mut c_d_in = cu_stream
        .clone_htod(&cu_input_data)
        .map_err(|e| anyhow!("htod input: {e}"))?;
    let mut c_d_out = cu_stream
        .alloc_zeros::<cufft_sys::float2>(total)
        .map_err(|e| anyhow!("alloc out: {e}"))?;
    let cu_plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        BATCH as i32,
        cu_stream.clone(),
    )
    .map_err(|e| anyhow!("cufft plan_1d: {e:?}"))?;

    // --- Warmup: interleave both backends, no events ---
    for _ in 0..WARMUP {
        launch_ferrum(&f_dbuf_in, &f_dbuf_tw, &mut f_dbuf_out)?;
        cu_plan
            .exec_c2c(&mut c_d_in, &mut c_d_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft exec: {e:?}"))?;
    }
    core_stream.synchronize()?;
    cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    // --- Timed loop: per trial, ferrum + cuFFT with separate event pairs ---
    let f_timing_flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let c_timing_flag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut f_event_samples_s: Vec<f64> = Vec::with_capacity(TRIALS);
    let mut c_event_samples_s: Vec<f64> = Vec::with_capacity(TRIALS);

    let t0 = Instant::now();
    for _ in 0..TRIALS {
        // ferrum half
        let f_start = core_ctx.new_event(f_timing_flag)?;
        let f_stop = core_ctx.new_event(f_timing_flag)?;
        f_start.record(&core_stream)?;
        launch_ferrum(&f_dbuf_in, &f_dbuf_tw, &mut f_dbuf_out)?;
        f_stop.record(&core_stream)?;
        f_stop.synchronize()?;
        let f_ms = f_start.elapsed_ms(&f_stop)?;
        f_event_samples_s.push(f_ms as f64 * 1.0e-3);

        // cuFFT half
        let c_start = cudarc_ctx
            .new_event(c_timing_flag)
            .map_err(|e| anyhow!("new_event: {e}"))?;
        let c_stop = cudarc_ctx
            .new_event(c_timing_flag)
            .map_err(|e| anyhow!("new_event: {e}"))?;
        c_start
            .record(&cu_stream)
            .map_err(|e| anyhow!("record start: {e}"))?;
        cu_plan
            .exec_c2c(&mut c_d_in, &mut c_d_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft exec: {e:?}"))?;
        c_stop
            .record(&cu_stream)
            .map_err(|e| anyhow!("record stop: {e}"))?;
        c_stop
            .synchronize()
            .map_err(|e| anyhow!("event sync: {e}"))?;
        let c_ms = c_start
            .elapsed_ms(&c_stop)
            .map_err(|e| anyhow!("elapsed: {e}"))?;
        c_event_samples_s.push(c_ms as f64 * 1.0e-3);
    }
    core_stream.synchronize()?;
    cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    // Per-launch share of total wall: each trial does one ferrum + one cuFFT
    // launch, so wall_mean_s is the average wall time per launch.
    let wall_mean_s = t0.elapsed().as_secs_f64() / (2.0 * TRIALS as f64);

    Ok((
        BenchSample { event_med_s: median(f_event_samples_s), wall_mean_s },
        BenchSample { event_med_s: median(c_event_samples_s), wall_mean_s },
    ))
}

fn main() -> Result<()> {
    // cuda-oxide CudaContext: hosts the in-tree Stockham kernel module.
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;

    // cudarc CudaContext: hosts cuFFT. Both point at GPU 0 of the same
    // driver; cuFFT and our kernel each operate over their own device
    // buffers so the two contexts don't have to share allocations.
    let cudarc_ctx = CudarcContext::new(0).map_err(|e| anyhow!("cudarc CudaContext::new: {e}"))?;

    let gate_mode = std::env::args().any(|a| a == "--gate");
    let gate_ratio = 0.9_f64;
    let mut misses: Vec<(usize, f64)> = Vec::new();

    println!(
        "ferrum-gpu-bench: 1D radix-2 C2C FFT, batch={}, warmup={}, trials={}",
        BATCH, WARMUP, TRIALS
    );
    println!(
        "{:<8} {:<11} {:<11} {:<11} {:<11} {:<9}",
        "N", "fe_ev_us", "cu_ev_us", "fe_wl_us", "cu_wl_us", "sp_event"
    );
    for &log_n in LOG_NS {
        let n = 1usize << log_n;
        let (ferr, cu) = alternating_bench(&ctx, &cudarc_ctx, &module, log_n)?;
        let fe_ev_us = ferr.event_med_s * 1.0e6 / (BATCH as f64);
        let cu_ev_us = cu.event_med_s * 1.0e6 / (BATCH as f64);
        let fe_wl_us = ferr.wall_mean_s * 1.0e6 / (BATCH as f64);
        let cu_wl_us = cu.wall_mean_s * 1.0e6 / (BATCH as f64);
        let sp_event = if fe_ev_us > 0.0 { cu_ev_us / fe_ev_us } else { f64::NAN };
        println!(
            "{:<8} {:<11.3} {:<11.3} {:<11.3} {:<11.3} {:<9.2}",
            n, fe_ev_us, cu_ev_us, fe_wl_us, cu_wl_us, sp_event
        );
        if gate_mode {
            let event_ratio = fe_ev_us / cu_ev_us;
            if event_ratio > gate_ratio {
                misses.push((n, event_ratio));
            }
        }
    }
    if gate_mode {
        if misses.is_empty() {
            println!("\nperf-gate: PASS (ferrum_event_us <= {} * cufft_event_us on all sizes)", gate_ratio);
        } else {
            eprintln!("\nperf-gate: MISS on:");
            for (n, r) in &misses {
                eprintln!("  N = {n}: ratio = {:.3} (need <= {})", r, gate_ratio);
            }
            std::process::exit(1);
        }
    }
    Ok(())
}
