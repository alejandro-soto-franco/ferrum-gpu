//! cuda-oxide-compiled FFT kernels for ferrum-gpu-fft.
//!
//! Hosts the v0.1 generic Stockham fallback plus three per-size specialised
//! kernels for N in {256, 1024, 4096}. The `#[cuda_module]` embeds PTX into
//! this crate so the three consumers (`examples/fft-1d-c2c`,
//! `crates/ferrum-gpu-bench`, `crates/ferrum-gpu-py`) share one copy.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Single-block-per-lane radix-2 Stockham FFT (v0.1 fallback).
    ///
    /// Layout assumptions, matching the CPU reference in
    /// `ferrum-gpu-fft/src/cpu.rs`:
    ///   * `in_data` / `out_data` are interleaved `[re0, im0, re1, im1, ...]`
    ///     of length `2 * N * batch`.
    ///   * `twiddles` is interleaved `[re, im, ...]` of length `2 * (N - 1)`,
    ///     laid out stages descending (largest first): stage `s` occupies
    ///     offset `N - (1 << s)` ... `N - (1 << s) + (1 << (s - 1))`.
    ///   * `grid_dim = (batch, 1, 1)`, `block_dim = (N / 2, 1, 1)`.
    ///   * `log_n` in [2, 12], so `N` in {4, 8, ..., 4096}.
    ///
    /// Each thread loads two complex elements into shared memory (stride
    /// N/2 apart), runs `log_n` Stockham stages with a per-stage barrier,
    /// then stores the result back to global memory.
    #[kernel]
    pub fn fft_radix2_c2c_pow2_1d_fallback(
        in_data: &[f32],
        twiddles: &[f32],
        mut out_data: DisjointSlice<f32>,
        log_n: u32,
    ) {
        // N_MAX = 4096 complex elements = 8192 f32 slots per ping-pong slab.
        static mut PING: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut PONG: SharedArray<f32, 8192> = SharedArray::UNINIT;

        let n = 1usize << log_n;
        let half_n = n >> 1;

        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x() as usize;
        let lane_off = block * n * 2;

        // ---------------------------------------------------------------
        // Load: each thread copies multiple complex elements into PING via
        // a strided loop, so block_dim can be capped below half_n for
        // large N. For N <= 2 * block_dim each thread does one outer
        // iteration; for N = 4096 with block_dim = 1024 each thread does
        // two outer iterations.
        // ---------------------------------------------------------------
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

        // ---------------------------------------------------------------
        // Stockham stages. Source/destination ping-pongs each stage. We
        // can't conditionally bind a `&mut SharedArray` reference in Rust
        // without aliasing trouble, so we duplicate the stage body per
        // direction of the ping-pong flag. Each thread handles multiple
        // butterflies via a strided loop over (b in 0..half_n).
        // ---------------------------------------------------------------
        let mut s = 1u32;
        let mut src_is_ping = true;
        while s <= log_n {
            let m_half = 1usize << (s - 1);
            let m = 1usize << s;
            // Stage offset in the descending-layout twiddle table.
            // Stage s lives at offset `N - (1 << s)` with length `2^(s-1)`.
            let stage_off = n - m;

            let mut b = tid;
            while b < half_n {
                // Map butterfly index b in 0..half_n to (j, k):
                // j = b / m_half, k = b % m_half.
                let j = b / m_half;
                let k = b - j * m_half;
                let tw_i = 2 * (stage_off + k);
                let w_re = twiddles[tw_i];
                let w_im = twiddles[tw_i + 1];

                // Source / destination indices (in f32 slots).
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

        // ---------------------------------------------------------------
        // Store: after log_n stages, the result lives in whichever slab
        // is now the "source" (we toggled after the last write).
        // src_is_ping == true => result in PING; false => result in PONG.
        // ---------------------------------------------------------------
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

pub use kernels::LoadedModule;

/// Load all kernels into the given CUDA context.
pub fn load(ctx: &std::sync::Arc<cuda_core::CudaContext>) -> anyhow::Result<LoadedModule> {
    kernels::load(ctx).map_err(Into::into)
}
