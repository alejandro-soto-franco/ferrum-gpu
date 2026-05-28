//! Shared host-side runtime for `ferrum-gpu-bench` and `perf-gate`.
//!
//! Owns the alternating ferrum/cuFFT timed-bench helper plus its sample
//! type and the bench-wide constants, so both the table bench and the
//! CI-style gate share one measurement loop.
//!
//! cuda-oxide keys embedded PTX to the binary crate, so the `#[cuda_module]`
//! block itself cannot be a library item. Instead we expose
//! [`define_fft_radix2_kernels`], a declarative macro whose expansion is the
//! full `#[cuda_module] mod kernels { ... }` block (declarative macros expand
//! before proc-macro attribute processing, so `#[cuda_module]` sees the
//! `#[kernel]` functions normally). Each binary invokes the macro once at
//! its top level and then passes a launch closure to [`alternating_bench`]
//! rather than a module reference, so the bench library never depends on a
//! particular `LoadedModule` type.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext as CudarcContext;

use ferrum_gpu_fft::Plan;

/// Expand to the full `#[cuda_module] mod kernels { ... }` block hosting the
/// v0.1 radix-2 Stockham power-of-2 C2C fallback kernel.
///
/// Invoke at the top level of a binary crate (not from inside a function).
/// After expansion, `kernels::load(&core_ctx)?` returns a `LoadedModule` with
/// the `fft_radix2_c2c_pow2_1d_fallback` launcher.
#[macro_export]
macro_rules! define_fft_radix2_kernels {
    () => {
        #[::cuda_host::cuda_module]
        mod kernels {
            use ::cuda_device::{DisjointSlice, SharedArray, kernel, thread};

            /// Single-block-per-lane radix-2 Stockham FFT. Body matches the
            /// kernel in `examples/fft-1d-c2c/src/main.rs` and
            /// `crates/ferrum-gpu-py/src/lib.rs`.
            #[kernel]
            pub fn fft_radix2_c2c_pow2_1d_fallback(
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
    };
}

/// Target log2(N) for the three bench sizes (256, 1024, 4096).
pub const LOG_NS: &[u32] = &[8, 10, 12];
/// Batch size used at every bench size.
pub const BATCH: usize = 256;
/// Untimed warmup iterations before measurement begins.
pub const WARMUP: usize = 10;
/// Timed iterations per size.
pub const TRIALS: usize = 100;

/// Per-launch timing report (seconds), averaged over `TRIALS`.
#[derive(Debug, Clone, Copy)]
pub struct BenchSample {
    /// Median per-launch GPU time (CUDA events bracketing each kernel).
    pub event_med_s: f64,
    /// Mean per-launch wall-clock time (per-launch share of the loop's total
    /// Instant elapsed; identical for ferrum and cuFFT in alternating mode).
    pub wall_mean_s: f64,
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Initialise both CUDA stacks at GPU 0.
///
/// Returns: cuda-core context (host for the in-tree kernel) and cudarc
/// context (host for cuFFT). The kernel module is loaded by the caller
/// because the `#[cuda_module]` block must live in the binary crate
/// (cuda-oxide PTX embedding constraint).
pub fn init_cuda_contexts() -> Result<(Arc<CudaContext>, Arc<CudarcContext>)> {
    let core_ctx = CudaContext::new(0)?;
    let cudarc_ctx =
        CudarcContext::new(0).map_err(|e| anyhow!("cudarc CudaContext::new: {e}"))?;
    Ok((core_ctx, cudarc_ctx))
}

/// Alternating ferrum + cuFFT measurement at a single size.
///
/// Each timed iteration runs one ferrum launch (cuda-core stream) immediately
/// followed by one cuFFT launch (cudarc stream), with separate event pairs
/// bracketing each. Interleaving reduces per-size variance from background
/// activity (DVFS, contention) that would otherwise affect the two backends
/// asymmetrically when they run in sequential blocks.
///
/// `launch_ferrum` is the caller-supplied closure that performs one ferrum
/// kernel launch on the bench-owned input, twiddles, and output buffers.
/// The closure captures the caller's `kernels::LoadedModule`, the stream,
/// the launch config, and `log_n`.
pub fn alternating_bench(
    core_ctx: &Arc<CudaContext>,
    cudarc_ctx: &Arc<CudarcContext>,
    log_n: u32,
    mut launch_ferrum: impl FnMut(
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> Result<()>,
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

    let wall_mean_s = t0.elapsed().as_secs_f64() / (2.0 * TRIALS as f64);

    Ok((
        BenchSample { event_med_s: median(f_event_samples_s), wall_mean_s },
        BenchSample { event_med_s: median(c_event_samples_s), wall_mean_s },
    ))
}

/// Build the standard `LaunchConfig` for the radix-2 fallback kernel at this
/// `log_n`: one block per batch lane, `min(N/2, 1024)` threads per block.
pub fn fallback_launch_cfg(log_n: u32) -> LaunchConfig {
    let n = 1usize << log_n;
    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    LaunchConfig {
        grid_dim: (BATCH as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
