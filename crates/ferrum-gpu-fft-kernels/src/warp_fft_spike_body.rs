// Spike kernel for the warp-shuffle FFT redesign (Task: beat cuFFT).
//
// Included at crate-root scope by the `warp-fft-spike` bench binary (same
// `include!` contract as kernels_body.rs; cuda-oxide embeds PTX per binary).
//
// Proves the warp-resident register FFT works through cuda-oxide: one warp
// computes a 32-point C2C FFT entirely in registers, exchanging butterfly
// partners with `shfl_xor_f32` and never touching shared memory. This is the
// building block of the four-step (64x64) kernel that aims to close the
// cuFFT gap. Mirrors `ferrum_gpu_fft::warp_fft::warp_fft32_model` (DIF,
// natural in, bit-reversed register output written to natural position).

#[::cuda_host::cuda_module]
mod warp_kernels {
    use ::cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
    use ::cuda_device::{DisjointSlice, kernel, thread};

    /// Batched 32-point forward C2C FFT, one warp per transform.
    ///
    /// `in_data` / `out_data`: interleaved `[re, im, ...]`, length `2*32*warps`.
    /// `tw`: `W_32^e` for `e in 0..16` interleaved (16 complex = 32 f32).
    /// Launch: `block_dim` a multiple of 32; `grid_dim.x * block_dim.x / 32`
    /// warps total, one per FFT.
    #[kernel]
    pub fn warp_fft32(in_data: &[f32], tw: &[f32], mut out_data: DisjointSlice<f32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank() as usize; // 0..31

        let gtid = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let warp_id = gtid / 32; // global FFT index
        let base = warp_id * 32 * 2;

        let mut re = in_data[base + 2 * lane];
        let mut im = in_data[base + 2 * lane + 1];

        // DIF stages s = 0..5, distance d = 16, 8, 4, 2, 1.
        let mut s = 0u32;
        while s < 5 {
            let d = 16usize >> s;
            // Collective: every lane must call shfl before the divergent branch.
            let p_re = warp.shfl_xor_f32(re, d as u32);
            let p_im = warp.shfl_xor_f32(im, d as u32);
            if lane & d == 0 {
                // Lower lane: a + b.
                re = re + p_re;
                im = im + p_im;
            } else {
                // Upper lane: (a - b) * W_32^((lane mod d) * 2^s), a = partner.
                let dr = p_re - re;
                let di = p_im - im;
                let exp = (lane & (d - 1)) * (1usize << s); // 0..15
                let wr = tw[2 * exp];
                let wi = tw[2 * exp + 1];
                re = dr * wr - di * wi;
                im = dr * wi + di * wr;
            }
            s += 1;
        }

        // DIF leaves lane L holding X[bitrev5(L)]; write to natural position.
        let br = ((lane & 1) << 4)
            | ((lane & 2) << 2)
            | (lane & 4)
            | ((lane & 8) >> 2)
            | ((lane & 16) >> 4);
        unsafe {
            *out_data.get_unchecked_mut(base + 2 * br) = re;
            *out_data.get_unchecked_mut(base + 2 * br + 1) = im;
        }
    }
}
