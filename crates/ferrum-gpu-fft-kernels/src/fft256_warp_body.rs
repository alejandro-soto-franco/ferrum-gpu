// N=256 warp-per-FFT kernel — the multi-FFT-per-block redesign aiming to beat
// cuFFT at N=256. Included at crate-root scope by the `fft256-warp-spike` bin
// (same `include!` contract as kernels_body.rs; cuda-oxide embeds PTX per
// binary). Mirrors `ferrum_gpu_fft::warp_fft::warp256_model` (= four_step_model
// with N1=32, N2=8), itself verified against the radix-2 reference.
//
// 256 = 32 x 8. One WARP computes one entire 256-pt FFT in registers — no
// shared memory, no __syncthreads. A block runs K warps = K FFTs. Lane L holds
// 8 complex register slots; slot c starts as x[L + 32*c]:
//   step 1: in-register 8-pt DFT over c (the n2 dimension)            -> B[L][k2]
//   step 2: twiddle B[L][k2] *= W_256^(L*k2)
//   step 3: 32-pt warp DFT over the lane dimension, run 8-wide (one per k2 slot)
//   store : X[8*k1 + k2], k1 = bitrev5(lane) (DIF leaves bit-reversed lanes)
//
// FMA: the step-2 and step-3 complex multiplies use `f32::mul_add` (lowers to
// libdevice `__nv_fmaf` -> a single `fma.rn.f32` via cuda-oxide's NVVM path;
// verified working at rev 6ed9938 — see docs spec / phase0-findings). The
// in-register dft8 is left as the proven plain-op flow graph from kernels_body.

#[::cuda_host::cuda_module]
mod fft256_warp {
    use ::cuda_device::cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block};
    use ::cuda_device::{DisjointSlice, kernel, thread};

    /// 1D forward C2C FFT, N=256, one warp per transform, register-resident.
    ///
    /// `in_data` / `out_data`: interleaved `[re, im, ...]`, length `2*256*batch`.
    /// `w32`:  `W_32^e`,  e in 0..16  (16 complex = 32 f32) — warp butterfly.
    /// `w256`: `W_256^e`, e in 0..256 (256 complex = 512 f32) — step-2 twiddle.
    /// Launch: `block_dim` a multiple of 32; `grid.x*block.x/32` warps total,
    /// one per FFT. `shared_mem_bytes = 0`.
    #[kernel]
    pub fn fft_c2c_256_warp(
        in_data: &[f32],
        w32: &[f32],
        w256: &[f32],
        mut out_data: DisjointSlice<f32>,
    ) {
        let warp = this_thread_block().tiled_partition::<32>();
        let lane = warp.thread_rank() as usize; // 0..31 = n1 = k1 dimension
        let gtid =
            (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let fft = gtid / 32; // global FFT index
        let base = fft * 256 * 2;

        // Load 8 register slots: slot c = x[lane + 32*c], c = n2 in 0..8.
        // Constant-index named scalars (cuda-oxide puts runtime-indexed [f32;N]
        // in local DRAM; named scalars stay in registers — see kernels_body).
        let r0 = in_data[base + 2 * (lane)];
        let i0 = in_data[base + 2 * (lane) + 1];
        let r1 = in_data[base + 2 * (lane + 32)];
        let i1 = in_data[base + 2 * (lane + 32) + 1];
        let r2 = in_data[base + 2 * (lane + 64)];
        let i2 = in_data[base + 2 * (lane + 64) + 1];
        let r3 = in_data[base + 2 * (lane + 96)];
        let i3 = in_data[base + 2 * (lane + 96) + 1];
        let r4 = in_data[base + 2 * (lane + 128)];
        let i4 = in_data[base + 2 * (lane + 128) + 1];
        let r5 = in_data[base + 2 * (lane + 160)];
        let i5 = in_data[base + 2 * (lane + 160) + 1];
        let r6 = in_data[base + 2 * (lane + 192)];
        let i6 = in_data[base + 2 * (lane + 192) + 1];
        let r7 = in_data[base + 2 * (lane + 224)];
        let i7 = in_data[base + 2 * (lane + 224) + 1];

        // --- Step 1: in-register 8-pt forward DFT over the slots (n2 -> k2).
        // Natural-order radix-2 DIT flow graph, W_8 constant-folded; copied
        // verbatim from kernels_body::fft_c2c_4096's `dft8!` (unit-tested via
        // cpu_radix8::dft8_inplace). Evaluates to (X0.re,X0.im,...,X7.re,X7.im).
        macro_rules! dft8 {
            ($x0:expr,$x1:expr,$x2:expr,$x3:expr,$x4:expr,$x5:expr,$x6:expr,$x7:expr,
             $x8:expr,$x9:expr,$x10:expr,$x11:expr,$x12:expr,$x13:expr,$x14:expr,$x15:expr) => {{
                const C: f32 = 0.70710678_f32;
                let (x0r, x0i) = ($x0, $x1);
                let (x1r, x1i) = ($x2, $x3);
                let (x2r, x2i) = ($x4, $x5);
                let (x3r, x3i) = ($x6, $x7);
                let (x4r, x4i) = ($x8, $x9);
                let (x5r, x5i) = ($x10, $x11);
                let (x6r, x6i) = ($x12, $x13);
                let (x7r, x7i) = ($x14, $x15);
                let (a0r, a0i) = (x0r + x4r, x0i + x4i);
                let (a1r, a1i) = (x0r - x4r, x0i - x4i);
                let (b0r, b0i) = (x2r + x6r, x2i + x6i);
                let (b1r, b1i) = (x2r - x6r, x2i - x6i);
                let (e0r, e0i) = (a0r + b0r, a0i + b0i);
                let (e2r, e2i) = (a0r - b0r, a0i - b0i);
                let (e1r, e1i) = (a1r + b1i, a1i - b1r);
                let (e3r, e3i) = (a1r - b1i, a1i + b1r);
                let (c0r, c0i) = (x1r + x5r, x1i + x5i);
                let (c1r, c1i) = (x1r - x5r, x1i - x5i);
                let (d0r, d0i) = (x3r + x7r, x3i + x7i);
                let (d1r, d1i) = (x3r - x7r, x3i - x7i);
                let (o0r, o0i) = (c0r + d0r, c0i + d0i);
                let (o2r, o2i) = (c0r - d0r, c0i - d0i);
                let (o1r, o1i) = (c1r + d1i, c1i - d1r);
                let (o3r, o3i) = (c1r - d1i, c1i + d1r);
                let (w1r, w1i) = (C * (o1r + o1i), C * (o1i - o1r));
                let (w3r, w3i) = (C * (o3i - o3r), -C * (o3r + o3i));
                (
                    e0r + o0r, e0i + o0i,
                    e1r + w1r, e1i + w1i,
                    e2r + o2i, e2i - o2r,
                    e3r + w3r, e3i + w3i,
                    e0r - o0r, e0i - o0i,
                    e1r - w1r, e1i - w1i,
                    e2r - o2i, e2i + o2r,
                    e3r - w3r, e3i - w3i,
                )
            }};
        }

        let (mut s0r, mut s0i, mut s1r, mut s1i, mut s2r, mut s2i, mut s3r, mut s3i,
             mut s4r, mut s4i, mut s5r, mut s5i, mut s6r, mut s6i, mut s7r, mut s7i) =
            dft8!(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4, r5, i5, r6, i6, r7, i7);

        // --- Step 2: twiddle slot k2 by W_256^(lane*k2) (k2=0 is W^0=1, skip).
        // Complex multiply via FMA: (re,im)*(wr,wi).
        macro_rules! tw256 {
            ($k2:expr, $re:ident, $im:ident) => {{
                let e = 2 * ((lane * $k2) & 255);
                let wr = w256[e];
                let wi = w256[e + 1];
                let nr = $re.mul_add(wr, -($im * wi));
                let ni = $re.mul_add(wi, $im * wr);
                $re = nr;
                $im = ni;
            }};
        }
        tw256!(1, s1r, s1i);
        tw256!(2, s2r, s2i);
        tw256!(3, s3r, s3i);
        tw256!(4, s4r, s4i);
        tw256!(5, s5r, s5i);
        tw256!(6, s6r, s6i);
        tw256!(7, s7r, s7i);

        // --- Step 3: 32-pt warp DFT over the lane dimension, applied to all 8
        // k2 slots (8-wide). DIF radix-2, distance d = 16,8,4,2,1; partner via
        // shfl_xor. d/upper/exp/W_32 depend only on (lane, stage), computed once
        // per stage and shared across slots. Upper-lane twiddle multiply via FMA.
        // (Divergent `if upper` here is the simple correct form; P3 makes it
        // branchless per the four-step notes.)
        macro_rules! warp_bf {
            ($re:ident, $im:ident, $d:expr, $upper:expr, $wr:expr, $wi:expr) => {{
                let pr = warp.shfl_xor_f32($re, $d as u32);
                let pi = warp.shfl_xor_f32($im, $d as u32);
                if $upper {
                    let dr = pr - $re;
                    let di = pi - $im;
                    $re = dr.mul_add($wr, -(di * $wi));
                    $im = dr.mul_add($wi, di * $wr);
                } else {
                    $re = $re + pr;
                    $im = $im + pi;
                }
            }};
        }
        let mut s = 0u32;
        while s < 5 {
            let d = 16usize >> s;
            let upper = lane & d != 0;
            let exp = (lane & (d - 1)) * (1usize << s); // 0..15
            let wr = w32[2 * exp];
            let wi = w32[2 * exp + 1];
            warp_bf!(s0r, s0i, d, upper, wr, wi);
            warp_bf!(s1r, s1i, d, upper, wr, wi);
            warp_bf!(s2r, s2i, d, upper, wr, wi);
            warp_bf!(s3r, s3i, d, upper, wr, wi);
            warp_bf!(s4r, s4i, d, upper, wr, wi);
            warp_bf!(s5r, s5i, d, upper, wr, wi);
            warp_bf!(s6r, s6i, d, upper, wr, wi);
            warp_bf!(s7r, s7i, d, upper, wr, wi);
            s += 1;
        }

        // --- Store: DIF leaves lane L holding X[bitrev5(L)] for each k2 slot.
        // Output natural index = 8*k1 + k2 with k1 = bitrev5(lane).
        let br = ((lane & 1) << 4)
            | ((lane & 2) << 2)
            | (lane & 4)
            | ((lane & 8) >> 2)
            | ((lane & 16) >> 4);
        let obase = base + 2 * (8 * br);
        macro_rules! st {
            ($k2:expr, $re:expr, $im:expr) => {{
                let o = obase + 2 * ($k2);
                unsafe {
                    *out_data.get_unchecked_mut(o) = $re;
                    *out_data.get_unchecked_mut(o + 1) = $im;
                }
            }};
        }
        st!(0, s0r, s0i);
        st!(1, s1r, s1i);
        st!(2, s2r, s2i);
        st!(3, s3r, s3i);
        st!(4, s4r, s4i);
        st!(5, s5r, s5i);
        st!(6, s6r, s6i);
        st!(7, s7r, s7i);
    }
}
