// Kernel module body, loaded via `#[path = ".../kernels_body.rs"] mod kernels;`
// from each binary crate's `lib.rs`/`main.rs`. The consumer wraps the `mod`
// declaration with `#[cuda_module]` so PTX gets embedded into THAT binary
// (cuda-oxide's `#[cuda_module]` is binary-crate-only, not propagated through
// rlibs). One source file, many embedded copies.

use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

/// Single-block-per-lane radix-2 Stockham FFT (v0.1 fallback).
///
/// Layout assumptions match `ferrum-gpu-fft/src/cpu.rs`:
///   * `in_data` / `out_data` are interleaved `[re, im, ...]` of length `2 * N * batch`.
///   * `twiddles` are interleaved `[re, im, ...]`, stages descending.
///   * `grid_dim = (batch, 1, 1)`, `block_dim = (min(N/2, 1024), 1, 1)`.
///   * `log_n` in `[2, 12]` (N in {4..4096}).
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
