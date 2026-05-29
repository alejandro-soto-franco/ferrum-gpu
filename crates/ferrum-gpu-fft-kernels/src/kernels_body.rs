// Single source of truth for the ferrum-gpu-fft kernels.
//
// Consumed by binary crates via `include!("...kernels_body.rs");` at the
// top level (NOT inside an existing `mod` block; the `#[cuda_module]`
// proc macro rejects `#[path]` mod declarations and `include!` invocations
// nested inside its input). Absolute paths (`::cuda_host`, `::cuda_device`)
// are used so the consumer does not need to bring those crates into scope.
//
// cuda-oxide's `#[cuda_module]` embeds PTX into the binary crate that owns
// the include site (it does not propagate through rlibs). One file here,
// many embedded copies in the binaries that include it.

#[::cuda_host::cuda_module]
mod kernels {
    use ::cuda_device::{DisjointSlice, SharedArray, kernel, thread};

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

    /// In-register 8-point forward DFT on interleaved `[re, im, ...]` input.
    ///
    /// Mirrors `ferrum_gpu_fft::cpu_radix8::dft8_inplace` (which is unit-tested
    /// against a brute-force DFT and against the radix-2 reference). DIT
    /// radix-2 flow graph with the `W_8` factors constant-folded;
    /// `C = cos(pi/4) = sin(pi/4)`.
    ///
    /// Takes/returns the 16 floats by value: cuda-oxide's codegen does not yet
    /// support assigning through a `&mut [f32; 16]` (a `Deref -> ConstantIndex`
    /// projection), so the host-side in-place variant cannot be reused here.
    #[inline(always)]
    fn dft8(x: [f32; 16]) -> [f32; 16] {
        const C: f32 = 0.70710678_f32;

        let (x0r, x0i) = (x[0], x[1]);
        let (x1r, x1i) = (x[2], x[3]);
        let (x2r, x2i) = (x[4], x[5]);
        let (x3r, x3i) = (x[6], x[7]);
        let (x4r, x4i) = (x[8], x[9]);
        let (x5r, x5i) = (x[10], x[11]);
        let (x6r, x6i) = (x[12], x[13]);
        let (x7r, x7i) = (x[14], x[15]);

        // Even half: 4-point DFT of {x0, x2, x4, x6}.
        let (a0r, a0i) = (x0r + x4r, x0i + x4i);
        let (a1r, a1i) = (x0r - x4r, x0i - x4i);
        let (b0r, b0i) = (x2r + x6r, x2i + x6i);
        let (b1r, b1i) = (x2r - x6r, x2i - x6i);
        let (e0r, e0i) = (a0r + b0r, a0i + b0i);
        let (e2r, e2i) = (a0r - b0r, a0i - b0i);
        let (e1r, e1i) = (a1r + b1i, a1i - b1r); // a1 - i*b1
        let (e3r, e3i) = (a1r - b1i, a1i + b1r); // a1 + i*b1

        // Odd half: 4-point DFT of {x1, x3, x5, x7}.
        let (c0r, c0i) = (x1r + x5r, x1i + x5i);
        let (c1r, c1i) = (x1r - x5r, x1i - x5i);
        let (d0r, d0i) = (x3r + x7r, x3i + x7i);
        let (d1r, d1i) = (x3r - x7r, x3i - x7i);
        let (o0r, o0i) = (c0r + d0r, c0i + d0i);
        let (o2r, o2i) = (c0r - d0r, c0i - d0i);
        let (o1r, o1i) = (c1r + d1i, c1i - d1r); // c1 - i*d1
        let (o3r, o3i) = (c1r - d1i, c1i + d1r); // c1 + i*d1

        // X_q = E_q + W_8^q O_q, X_{q+4} = E_q - W_8^q O_q.
        let (w0r, w0i) = (o0r, o0i); // W_8^0 = 1
        let (w1r, w1i) = (C * (o1r + o1i), C * (o1i - o1r)); // W_8^1 = (C, -C)
        let (w2r, w2i) = (o2i, -o2r); // W_8^2 = -i
        let (w3r, w3i) = (C * (o3i - o3r), -C * (o3r + o3i)); // W_8^3 = (-C, -C)

        [
            e0r + w0r,
            e0i + w0i,
            e1r + w1r,
            e1i + w1i,
            e2r + w2r,
            e2i + w2i,
            e3r + w3r,
            e3i + w3i,
            e0r - w0r,
            e0i - w0i,
            e1r - w1r,
            e1i - w1i,
            e2r - w2r,
            e2i - w2i,
            e3r - w3r,
            e3i - w3i,
        ]
    }

    /// 1D forward C2C FFT for N = 4096, batch arbitrary. One block per FFT,
    /// 512 threads, 4 radix-8 Stockham stages.
    ///
    /// Same algorithm as `ferrum_gpu_fft::cpu_radix8::radix8_forward_lane`, but
    /// register-resident across each stage so a SINGLE 32 KiB shared buffer
    /// suffices instead of a 2x32 KiB ping-pong. Halving shared usage lifts the
    /// 1-block-per-SM ceiling the dual-buffer version hit on sm_120. Each stage
    /// is read-all-to-registers -> sync -> write-all-to-shared -> sync, which
    /// keeps the single buffer race-free (every thread finishes reading before
    /// any thread overwrites).
    ///
    /// `in_data` / `out_data` are interleaved `[re, im, ...]` of length
    /// `2 * 4096 * batch`; `twiddles` is `ferrum_gpu_fft::twiddles_radix8(12)`
    /// flattened (the per-stage `[k][p]` input-twiddle table).
    ///
    /// `grid_dim = (batch, 1, 1)`, `block_dim = (512, 1, 1)`,
    /// `shared_mem_bytes = 0` (static `SharedArray`).
    #[kernel]
    pub fn fft_c2c_4096(
        in_data: &[f32],
        twiddles: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        static mut BUF: SharedArray<f32, 8192> = SharedArray::UNINIT;

        const N: usize = 4096;
        const STAGES: usize = 4; // 8^4 = 4096
        const THREADS: usize = 512;
        const NR: usize = N / 8; // 512: lane stride between the 8 gathered inputs
        // BUTTERFLIES == THREADS == 512: exactly one radix-8 butterfly / thread.

        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane_off = block * N * 2;

        // Load this lane into BUF.
        {
            let mut t = tid;
            while t < N {
                unsafe {
                    BUF[2 * t] = in_data[lane_off + 2 * t];
                    BUF[2 * t + 1] = in_data[lane_off + 2 * t + 1];
                }
                t += THREADS;
            }
        }
        thread::sync_threads();

        // One butterfly per thread (BUTTERFLIES == THREADS == 512).
        let b = tid;
        let mut m_r: usize = 1; // 8^(s-1)
        let mut stage_off: usize = 0;
        let mut stage = 0;
        while stage < STAGES {
            let m = m_r * 8; // 8^s
            let j = b / m_r;
            let k = b - j * m_r;

            // Phase 1: read 8 strided inputs, apply input twiddles, radix-8
            // DFT, hold the 8 outputs in registers across the barrier.
            //
            // Fully unrolled into named scalars on purpose: cuda-oxide places
            // any `[f32; N]` that is indexed by a runtime value in LOCAL (DRAM)
            // memory, so the looped form spilled all 16 floats every butterfly.
            // Constant-index scalars stay in registers. The twiddle factor for
            // input p is `W_m^(p*k)` at table slot `stage_off + 8*k + p`.
            let src_base = j * m_r + k;
            let tw_base = 2 * (stage_off + 8 * k);

            // gather input p: scalar (re, im) at shared offset 2*(src_base+p*NR).
            macro_rules! ld {
                ($p:expr) => {{
                    let si = 2 * (src_base + ($p) * NR);
                    unsafe { (BUF[si], BUF[si + 1]) }
                }};
            }
            // twiddle-multiply input p by W_m^(p*k).
            macro_rules! tw {
                ($p:expr, $re:expr, $im:expr) => {{
                    let wi_off = tw_base + 2 * ($p);
                    let wr = twiddles[wi_off];
                    let wi = twiddles[wi_off + 1];
                    ($re * wr - $im * wi, $re * wi + $im * wr)
                }};
            }

            let (g0r, g0i) = ld!(0);
            let (r1, i1) = ld!(1);
            let (r2, i2) = ld!(2);
            let (r3, i3) = ld!(3);
            let (r4, i4) = ld!(4);
            let (r5, i5) = ld!(5);
            let (r6, i6) = ld!(6);
            let (r7, i7) = ld!(7);
            let (t1r, t1i) = tw!(1, r1, i1);
            let (t2r, t2i) = tw!(2, r2, i2);
            let (t3r, t3i) = tw!(3, r3, i3);
            let (t4r, t4i) = tw!(4, r4, i4);
            let (t5r, t5i) = tw!(5, r5, i5);
            let (t6r, t6i) = tw!(6, r6, i6);
            let (t7r, t7i) = tw!(7, r7, i7);

            let [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15] = dft8([
                g0r, g0i, t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i, t5r, t5i, t6r, t6i, t7r, t7i,
            ]);

            // Every thread has finished reading; safe to overwrite BUF.
            thread::sync_threads();

            // Phase 2: scatter the 8 outputs to their Stockham positions
            // (output q at dst_base + q*m_r). Unrolled for the same reason.
            let dst_base = j * m + k;
            macro_rules! st {
                ($q:expr, $re:expr, $im:expr) => {{
                    let di = 2 * (dst_base + ($q) * m_r);
                    unsafe {
                        BUF[di] = $re;
                        BUF[di + 1] = $im;
                    }
                }};
            }
            st!(0, o0, o1);
            st!(1, o2, o3);
            st!(2, o4, o5);
            st!(3, o6, o7);
            st!(4, o8, o9);
            st!(5, o10, o11);
            st!(6, o12, o13);
            st!(7, o14, o15);
            thread::sync_threads();

            stage_off += 8 * m_r;
            m_r = m;
            stage += 1;
        }

        // Store the result back to global.
        {
            let mut t = tid;
            while t < N {
                let (re, im) = unsafe { (BUF[2 * t], BUF[2 * t + 1]) };
                unsafe {
                    *out_data.get_unchecked_mut(lane_off + 2 * t) = re;
                    *out_data.get_unchecked_mut(lane_off + 2 * t + 1) = im;
                }
                t += THREADS;
            }
        }
    }

    /// Tiled square transpose of a complex matrix laid out as interleaved
    /// f32 (re, im). Both `src` and `dst` are length `2*N*N` where
    /// `N = 1 << log_n`.
    ///
    /// Tile size: 16x16 complex (32x16 f32). One thread per complex element,
    /// using shared memory to coalesce both reads and writes. Row stride 17
    /// (one f32-pair pad) eliminates shared-memory bank conflicts.
    ///
    /// Launch config: grid_dim = (N/TILE, N/TILE, 1), block_dim =
    /// (TILE, TILE, 1).
    #[kernel]
    pub fn transpose_complex_pow2(
        src: &[f32],
        mut dst: DisjointSlice<f32>,
        log_n: u32,
    ) {
        // 16 * 17 * 2 = 544 f32 = 2.1 KiB shared.
        static mut TILE_BUF: SharedArray<f32, 544> = SharedArray::UNINIT;
        const TILE: u32 = 16;

        let n = 1u32 << log_n;
        let bx = thread::blockIdx_x();
        let by = thread::blockIdx_y();
        let tx = thread::threadIdx_x();
        let ty = thread::threadIdx_y();

        // Source coords in the input matrix.
        let src_i = by * TILE + ty;
        let src_j = bx * TILE + tx;
        if src_i < n && src_j < n {
            let src_idx = 2 * ((src_i * n + src_j) as usize);
            let sh_idx = 2 * ((ty * 17 + tx) as usize);
            unsafe {
                TILE_BUF[sh_idx] = src[src_idx];
                TILE_BUF[sh_idx + 1] = src[src_idx + 1];
            }
        }
        thread::sync_threads();

        // Destination coords: dst[j][i] = src[i][j]. Block-level swap
        // (bx,by) -> (by,bx); thread-level swap (tx,ty) inside the tile.
        let dst_i = bx * TILE + ty;
        let dst_j = by * TILE + tx;
        if dst_i < n && dst_j < n {
            let dst_idx = 2 * ((dst_i * n + dst_j) as usize);
            let sh_idx = 2 * ((tx * 17 + ty) as usize);
            unsafe {
                let re = TILE_BUF[sh_idx];
                let im = TILE_BUF[sh_idx + 1];
                *dst.get_unchecked_mut(dst_idx) = re;
                *dst.get_unchecked_mut(dst_idx + 1) = im;
            }
        }
    }
}
