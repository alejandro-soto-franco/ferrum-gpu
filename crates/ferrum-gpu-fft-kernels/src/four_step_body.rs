// Four-step (64x64) FFT for N=4096 — the warp-shuffle redesign aiming to close
// the cuFFT gap. Included at crate-root scope by the `four-step-spike` bin.
//
// Mirrors `ferrum_gpu_fft::warp_fft::four_step_model` (verified == radix-2):
//   step 1: 64 column FFTs over n2 (one warp each), register-resident via
//           shfl_xor, then twiddle C[n1][k2] = B[n1][k2] * W_4096^(n1*k2);
//   step 3: 64 row FFTs over n1 (one warp each).
// The inter-step "transpose" is implicit: step 1 writes C[n1][k2] to shared
// slot n1 + 64*k2 (stride-64 per column); step 3 reads slot a + 64*k2 over
// a=0..63 (the contiguous block at 64*k2). No explicit transpose pass.
//
// The 64-pt warp FFT (`warp_fft64!`) mirrors `warp_dif_model(_, 6)`: lane L
// holds elements (e0 at logical index L, e1 at L+32); stage 0 (d=32) is a
// within-lane butterfly, stages 1..5 (d=16..1) shuffle both elements with
// `shfl_xor`. Output: e0 = X[2*bitrev5(L)], e1 = X[2*bitrev5(L)+1].

#[::cuda_host::cuda_module]
mod four_step {
    use ::cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
    use ::cuda_device::{DisjointSlice, SharedArray, kernel, thread};

    const NW: usize = 8; // warps per block (block_dim = 256)

    /// 1D forward C2C FFT, N=4096, one block per transform, 256 threads.
    ///
    /// `w64`: `W_64^e`, e in 0..32 (32 complex, interleaved).
    /// `w4096`: `W_4096^e`, e in 0..4096 (4096 complex, interleaved).
    /// `in_data`/`out_data`: interleaved `[re, im, ...]`, length `2*4096*batch`.
    /// Launch: `grid_dim=(batch,1,1)`, `block_dim=(256,1,1)`.
    #[kernel]
    pub fn fft_c2c_4096_4step(
        in_data: &[f32],
        w64: &[f32],
        w4096: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        static mut BUF: SharedArray<f32, 8192> = SharedArray::UNINIT; // 4096 complex

        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank() as usize;
        let wib = (thread::threadIdx_x() / 32) as usize; // warp index in block
        let blk = thread::blockIdx_x() as usize;
        let base = blk * 4096 * 2;

        // 5-bit reversal of the lane.
        let br = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) | ((lane & 16) >> 4);

        // Load 4096 complex into shared.
        {
            let mut t = thread::threadIdx_x() as usize;
            while t < 4096 {
                unsafe {
                    BUF[2 * t] = in_data[base + 2 * t];
                    BUF[2 * t + 1] = in_data[base + 2 * t + 1];
                }
                t += 256;
            }
        }
        thread::sync_threads();

        // 64-pt warp FFT on (e0r,e0i,e1r,e1i); needs `lane`, `warp`, `w64` in scope.
        macro_rules! warp_fft64 {
            ($e0r:ident, $e0i:ident, $e1r:ident, $e1i:ident) => {{
                // Stage 0 (d=32, within-lane): e0 lower, e1 upper, tw = W_64^lane.
                {
                    let sr = $e0r + $e1r;
                    let si = $e0i + $e1i;
                    let dr = $e0r - $e1r;
                    let di = $e0i - $e1i;
                    let wr = w64[2 * lane];
                    let wi = w64[2 * lane + 1];
                    $e0r = sr;
                    $e0i = si;
                    $e1r = dr * wr - di * wi;
                    $e1i = dr * wi + di * wr;
                }
                // Stages 1..5 (d = 16,8,4,2,1; cross-lane shuffle, both elements).
                let mut s = 1u32;
                while s < 6 {
                    let d = 32usize >> s;
                    let p0r = warp.shfl_xor_f32($e0r, d as u32);
                    let p0i = warp.shfl_xor_f32($e0i, d as u32);
                    let p1r = warp.shfl_xor_f32($e1r, d as u32);
                    let p1i = warp.shfl_xor_f32($e1i, d as u32);
                    if lane & d == 0 {
                        $e0r = $e0r + p0r;
                        $e0i = $e0i + p0i;
                        $e1r = $e1r + p1r;
                        $e1i = $e1i + p1i;
                    } else {
                        let exp = (lane & (d - 1)) * (1usize << s);
                        let wr = w64[2 * exp];
                        let wi = w64[2 * exp + 1];
                        let d0r = p0r - $e0r;
                        let d0i = p0i - $e0i;
                        let d1r = p1r - $e1r;
                        let d1i = p1i - $e1i;
                        $e0r = d0r * wr - d0i * wi;
                        $e0i = d0r * wi + d0i * wr;
                        $e1r = d1r * wr - d1i * wi;
                        $e1i = d1r * wi + d1i * wr;
                    }
                    s += 1;
                }
            }};
        }

        // STEP 1: column FFTs over n2 for each column n1, + twiddle, write back.
        {
            let mut col = wib;
            while col < 64 {
                let i0 = 2 * (col + 64 * lane);
                let i1 = 2 * (col + 64 * (lane + 32));
                let mut e0r = unsafe { BUF[i0] };
                let mut e0i = unsafe { BUF[i0 + 1] };
                let mut e1r = unsafe { BUF[i1] };
                let mut e1i = unsafe { BUF[i1 + 1] };

                warp_fft64!(e0r, e0i, e1r, e1i);

                // e0 -> k2 = 2*br, e1 -> k2 = 2*br+1; twiddle by W_4096^(col*k2).
                let k2a = 2 * br;
                let k2b = 2 * br + 1;
                let ea = 2 * ((col * k2a) & 4095);
                let eb = 2 * ((col * k2b) & 4095);
                let war = w4096[ea];
                let wai = w4096[ea + 1];
                let wbr = w4096[eb];
                let wbi = w4096[eb + 1];
                let c0r = e0r * war - e0i * wai;
                let c0i = e0r * wai + e0i * war;
                let c1r = e1r * wbr - e1i * wbi;
                let c1i = e1r * wbi + e1i * wbr;

                let o0 = 2 * (col + 64 * k2a);
                let o1 = 2 * (col + 64 * k2b);
                unsafe {
                    BUF[o0] = c0r;
                    BUF[o0 + 1] = c0i;
                    BUF[o1] = c1r;
                    BUF[o1 + 1] = c1i;
                }
                col += NW;
            }
        }
        thread::sync_threads();

        // STEP 3: row FFTs over n1 for each k2; output X[64*k1 + k2].
        {
            let mut row = wib;
            while row < 64 {
                let i0 = 2 * (lane + 64 * row);
                let i1 = 2 * ((lane + 32) + 64 * row);
                let mut e0r = unsafe { BUF[i0] };
                let mut e0i = unsafe { BUF[i0 + 1] };
                let mut e1r = unsafe { BUF[i1] };
                let mut e1i = unsafe { BUF[i1 + 1] };

                warp_fft64!(e0r, e0i, e1r, e1i);

                // e0 -> k1 = 2*br, e1 -> k1 = 2*br+1; X[64*k1 + row].
                let k1a = 2 * br;
                let k1b = 2 * br + 1;
                let oa = base + 2 * (64 * k1a + row);
                let ob = base + 2 * (64 * k1b + row);
                unsafe {
                    *out_data.get_unchecked_mut(oa) = e0r;
                    *out_data.get_unchecked_mut(oa + 1) = e0i;
                    *out_data.get_unchecked_mut(ob) = e1r;
                    *out_data.get_unchecked_mut(ob + 1) = e1i;
                }
                row += NW;
            }
        }
    }
}
