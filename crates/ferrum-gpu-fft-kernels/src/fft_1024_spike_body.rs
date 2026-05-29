// Phase 4 spike: four-step (32x32) warp-shuffle FFT for N=1024.
// Included at crate-root scope by the `fft-1024-spike` bin.
//
// 1024 = 32 x 32. Each 32-pt sub-FFT is one warp (1 element/lane, 5 shfl_xor
// stages, NO within-lane stage and NO wave loop: 32 warps == 32 sub-FFTs).
// Mirrors `ferrum_gpu_fft::warp_fft::four_step_model(_, 5, 5)` (verified ==
// radix-2). Implicit transpose via stride-33 / stride-1 shared layout; the +1
// pad turns step-1's column reads from a 32-way bank conflict into 2-way.

#[::cuda_host::cuda_module]
mod fft1024 {
    use ::cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
    use ::cuda_device::{DisjointSlice, SharedArray, kernel, thread};

    /// 1D forward C2C FFT, N=1024, one block (32 warps = 1024 threads) per
    /// transform.
    ///
    /// `w32`: `W_32^e`, e in 0..16 (16 complex). `w1024`: `W_1024^e`, e in
    /// 0..1024 (1024 complex). `in_data`/`out_data`: interleaved, len
    /// `2*1024*batch`. Launch: `grid_dim=(batch,1,1)`, `block_dim=(1024,1,1)`.
    #[kernel]
    pub fn fft_c2c_1024_4step(
        in_data: &[f32],
        w32: &[f32],
        w1024: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        // 32 rows x 33 complex (padded) = 2112 f32 = 8.25 KiB.
        static mut BUF: SharedArray<f32, 2112> = SharedArray::UNINIT;
        const STRIDE: usize = 33;

        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank() as usize;
        let wib = (thread::threadIdx_x() / 32) as usize; // warp = column (step1) / row (step3)
        let blk = thread::blockIdx_x() as usize;
        let base = blk * 1024 * 2;

        // 5-bit reversal of the lane.
        let br = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) | ((lane & 16) >> 4);

        // Load 1024 complex at physical (n2*STRIDE + n1) for n = n1 + 32*n2.
        {
            let mut t = thread::threadIdx_x() as usize;
            while t < 1024 {
                let n1 = t & 31;
                let n2 = t >> 5;
                let p = 2 * (n2 * STRIDE + n1);
                unsafe {
                    BUF[p] = in_data[base + 2 * t];
                    BUF[p + 1] = in_data[base + 2 * t + 1];
                }
                t += 1024;
            }
        }
        thread::sync_threads();

        // 32-pt warp FFT on (er, ei): 5 branchless shfl_xor stages, output
        // lane L -> X[bitrev5(L)]. Needs `lane`, `warp`, `w32` in scope.
        macro_rules! warp_fft32 {
            ($er:ident, $ei:ident) => {{
                let mut s = 0u32;
                while s < 5 {
                    let d = 16usize >> s;
                    let pr = warp.shfl_xor_f32($er, d as u32);
                    let pi = warp.shfl_xor_f32($ei, d as u32);
                    let upper = lane & d != 0;
                    let sign = if upper { -1.0f32 } else { 1.0f32 };
                    let exp = (lane & (d - 1)) * (1usize << s);
                    let tr = w32[2 * exp];
                    let ti = w32[2 * exp + 1];
                    let twr = if upper { tr } else { 1.0 };
                    let twi = if upper { ti } else { 0.0 };
                    let ar = sign * $er + pr;
                    let ai = sign * $ei + pi;
                    $er = ar * twr - ai * twi;
                    $ei = ar * twi + ai * twr;
                    s += 1;
                }
            }};
        }

        // STEP 1: column FFT over n2 for column n1 = wib; twiddle; write back.
        {
            let col = wib; // 0..32
            let i = 2 * (lane * STRIDE + col);
            let mut er = unsafe { BUF[i] };
            let mut ei = unsafe { BUF[i + 1] };
            warp_fft32!(er, ei);
            // lane L -> k2 = br; twiddle W_1024^(col*k2).
            let k2 = br;
            let e = 2 * ((col * k2) & 1023);
            let wr = w1024[e];
            let wi = w1024[e + 1];
            let cr = er * wr - ei * wi;
            let ci = er * wi + ei * wr;
            let o = 2 * (k2 * STRIDE + col);
            unsafe {
                BUF[o] = cr;
                BUF[o + 1] = ci;
            }
        }
        thread::sync_threads();

        // STEP 3: row FFT over n1 for row k2 = wib; output X[32*k1 + k2].
        {
            let row = wib; // = k2
            let i = 2 * (row * STRIDE + lane);
            let mut er = unsafe { BUF[i] };
            let mut ei = unsafe { BUF[i + 1] };
            warp_fft32!(er, ei);
            // lane L -> k1 = br; X[32*k1 + k2].
            let k1 = br;
            let o = base + 2 * (32 * k1 + row);
            unsafe {
                *out_data.get_unchecked_mut(o) = er;
                *out_data.get_unchecked_mut(o + 1) = ei;
            }
        }
    }
}
