//! CPU interleaved-vs-split layout harness. Split-radix decimation-in-time FFT
//! in both layouts (layout is the only variable on identical math), plus
//! elementwise complex multiply in both layouts. Mirrors the storage question
//! the burn complex backend (burn-flex, CPU SIMD) must answer.

use wide::f32x8;

/// Recursive split-radix DIT forward FFT, interleaved `[re, im, ...]`.
/// Out-of-place into `out`. N = x.len()/2 must be a power of two.
pub fn split_radix_interleaved(x: &[f32], out: &mut [f32]) {
    let n = x.len() / 2;
    match n {
        1 => {
            out[0] = x[0];
            out[1] = x[1];
        }
        2 => {
            let (a0, a1) = (x[0], x[1]);
            let (b0, b1) = (x[2], x[3]);
            out[0] = a0 + b0;
            out[1] = a1 + b1;
            out[2] = a0 - b0;
            out[3] = a1 - b1;
        }
        _ => {
            let (n2, n4) = (n / 2, n / 4);
            let mut even = vec![0.0f32; 2 * n2];
            let mut o1 = vec![0.0f32; 2 * n4];
            let mut o3 = vec![0.0f32; 2 * n4];
            for k in 0..n2 {
                even[2 * k] = x[4 * k];
                even[2 * k + 1] = x[4 * k + 1];
            }
            for k in 0..n4 {
                o1[2 * k] = x[2 * (4 * k + 1)];
                o1[2 * k + 1] = x[2 * (4 * k + 1) + 1];
                o3[2 * k] = x[2 * (4 * k + 3)];
                o3[2 * k + 1] = x[2 * (4 * k + 3) + 1];
            }
            let mut s_even = vec![0.0f32; 2 * n2];
            let mut s_o1 = vec![0.0f32; 2 * n4];
            let mut s_o3 = vec![0.0f32; 2 * n4];
            split_radix_interleaved(&even, &mut s_even);
            split_radix_interleaved(&o1, &mut s_o1);
            split_radix_interleaved(&o3, &mut s_o3);
            for k in 0..n4 {
                let ang1 = -2.0 * core::f64::consts::PI * (k as f64) / (n as f64);
                let ang3 = -2.0 * core::f64::consts::PI * 3.0 * (k as f64) / (n as f64);
                let (w1r, w1i) = (ang1.cos() as f32, ang1.sin() as f32);
                let (w3r, w3i) = (ang3.cos() as f32, ang3.sin() as f32);
                let (z1r, z1i) = (s_o1[2 * k], s_o1[2 * k + 1]);
                let (z3r, z3i) = (s_o3[2 * k], s_o3[2 * k + 1]);
                let ar = w1r * z1r - w1i * z1i;
                let ai = w1r * z1i + w1i * z1r;
                let br = w3r * z3r - w3i * z3i;
                let bi = w3r * z3i + w3i * z3r;
                let (sr, si) = (ar + br, ai + bi);
                let (dr, di) = (ar - br, ai - bi);
                let (er, ei) = (s_even[2 * k], s_even[2 * k + 1]);
                let (er2, ei2) = (s_even[2 * (k + n4)], s_even[2 * (k + n4) + 1]);
                out[2 * k] = er + sr;
                out[2 * k + 1] = ei + si;
                out[2 * (k + n2)] = er - sr;
                out[2 * (k + n2) + 1] = ei - si;
                out[2 * (k + n4)] = er2 + di;
                out[2 * (k + n4) + 1] = ei2 - dr;
                out[2 * (k + n4 + n2)] = er2 - di;
                out[2 * (k + n4 + n2) + 1] = ei2 + dr;
            }
        }
    }
}

/// Identical split-radix recursion, split / SoA layout: real and imaginary in
/// separate arrays. Only indexing differs from `split_radix_interleaved`.
pub fn split_radix_soa(re: &[f32], im: &[f32], ore: &mut [f32], oim: &mut [f32]) {
    let n = re.len();
    match n {
        1 => {
            ore[0] = re[0];
            oim[0] = im[0];
        }
        2 => {
            ore[0] = re[0] + re[1];
            oim[0] = im[0] + im[1];
            ore[1] = re[0] - re[1];
            oim[1] = im[0] - im[1];
        }
        _ => {
            let (n2, n4) = (n / 2, n / 4);
            let mut e_re = vec![0.0f32; n2];
            let mut e_im = vec![0.0f32; n2];
            let mut o1_re = vec![0.0f32; n4];
            let mut o1_im = vec![0.0f32; n4];
            let mut o3_re = vec![0.0f32; n4];
            let mut o3_im = vec![0.0f32; n4];
            for k in 0..n2 {
                e_re[k] = re[2 * k];
                e_im[k] = im[2 * k];
            }
            for k in 0..n4 {
                o1_re[k] = re[4 * k + 1];
                o1_im[k] = im[4 * k + 1];
                o3_re[k] = re[4 * k + 3];
                o3_im[k] = im[4 * k + 3];
            }
            let mut se_re = vec![0.0f32; n2];
            let mut se_im = vec![0.0f32; n2];
            let mut s1_re = vec![0.0f32; n4];
            let mut s1_im = vec![0.0f32; n4];
            let mut s3_re = vec![0.0f32; n4];
            let mut s3_im = vec![0.0f32; n4];
            split_radix_soa(&e_re, &e_im, &mut se_re, &mut se_im);
            split_radix_soa(&o1_re, &o1_im, &mut s1_re, &mut s1_im);
            split_radix_soa(&o3_re, &o3_im, &mut s3_re, &mut s3_im);
            for k in 0..n4 {
                let ang1 = -2.0 * core::f64::consts::PI * (k as f64) / (n as f64);
                let ang3 = -2.0 * core::f64::consts::PI * 3.0 * (k as f64) / (n as f64);
                let (w1r, w1i) = (ang1.cos() as f32, ang1.sin() as f32);
                let (w3r, w3i) = (ang3.cos() as f32, ang3.sin() as f32);
                let ar = w1r * s1_re[k] - w1i * s1_im[k];
                let ai = w1r * s1_im[k] + w1i * s1_re[k];
                let br = w3r * s3_re[k] - w3i * s3_im[k];
                let bi = w3r * s3_im[k] + w3i * s3_re[k];
                let (sr, si) = (ar + br, ai + bi);
                let (dr, di) = (ar - br, ai - bi);
                ore[k] = se_re[k] + sr;
                oim[k] = se_im[k] + si;
                ore[k + n2] = se_re[k] - sr;
                oim[k + n2] = se_im[k] - si;
                ore[k + n4] = se_re[k + n4] + di;
                oim[k + n4] = se_im[k + n4] - dr;
                ore[k + n4 + n2] = se_re[k + n4] - di;
                oim[k + n4 + n2] = se_im[k + n4] + dr;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Allocation-free iterative radix-2 FFT, both layouts.
//
// The recursive `split_radix_*` pair above allocates scratch vectors at every
// recursion level, and the SoA arm allocates twice as many (re and im split),
// so a slice of its measured interleaved-vs-SoA gap is malloc traffic, not
// layout. The pair below removes allocation entirely: the bit-reversal table
// and twiddle table are precomputed once (outside any timed region) and the
// transform runs in a caller-owned, preallocated buffer. Identical arithmetic
// in both layouts; only the data indexing differs (one (re,im) pair per index
// vs two parallel arrays). This isolates layout = cache-adjacency, nothing else.

/// Bit-reversal permutation table for an N-point transform (N a power of two).
/// Precompute once, reuse across calls and batches.
pub fn bit_reverse_table(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros();
    (0..n)
        .map(|i| (i as u32).reverse_bits() >> (32 - bits))
        .map(|r| r as usize)
        .collect()
}

/// Twiddle table `[cos, sin, ...]` of `exp(-2*pi*i*j/N)` for j in 0..N/2.
/// Precompute once, reuse across calls and batches.
pub fn twiddle_table(n: usize) -> Vec<f32> {
    (0..n / 2)
        .flat_map(|j| {
            let th = -2.0 * core::f64::consts::PI * (j as f64) / (n as f64);
            [th.cos() as f32, th.sin() as f32]
        })
        .collect()
}

/// Allocation-free iterative radix-2 DIT FFT, interleaved `[re, im, ...]`.
/// Gathers `src` into `dst` by bit-reversal, then runs in place in `dst`.
/// `brev` and `tw` come from `bit_reverse_table` / `twiddle_table`.
pub fn fft_iter_interleaved(src: &[f32], dst: &mut [f32], brev: &[usize], tw: &[f32]) {
    let n = src.len() / 2;
    for i in 0..n {
        let j = brev[i];
        dst[2 * i] = src[2 * j];
        dst[2 * i + 1] = src[2 * j + 1];
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        let mut base = 0;
        while base < n {
            for k in 0..half {
                let (wr, wi) = (tw[2 * k * step], tw[2 * k * step + 1]);
                let (ar, ai) = (dst[2 * (base + k + half)], dst[2 * (base + k + half) + 1]);
                let vr = wr * ar - wi * ai;
                let vi = wr * ai + wi * ar;
                let (ur, ui) = (dst[2 * (base + k)], dst[2 * (base + k) + 1]);
                dst[2 * (base + k)] = ur + vr;
                dst[2 * (base + k) + 1] = ui + vi;
                dst[2 * (base + k + half)] = ur - vr;
                dst[2 * (base + k + half) + 1] = ui - vi;
            }
            base += len;
        }
        len <<= 1;
    }
}

/// Allocation-free iterative radix-2 DIT FFT, split / SoA (`re`, `im` arrays).
/// Identical arithmetic to `fft_iter_interleaved`; only indexing differs.
pub fn fft_iter_soa(
    sre: &[f32],
    sim: &[f32],
    dre: &mut [f32],
    dim: &mut [f32],
    brev: &[usize],
    tw: &[f32],
) {
    let n = sre.len();
    for i in 0..n {
        let j = brev[i];
        dre[i] = sre[j];
        dim[i] = sim[j];
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        let mut base = 0;
        while base < n {
            for k in 0..half {
                let (wr, wi) = (tw[2 * k * step], tw[2 * k * step + 1]);
                let (ar, ai) = (dre[base + k + half], dim[base + k + half]);
                let vr = wr * ar - wi * ai;
                let vi = wr * ai + wi * ar;
                let (ur, ui) = (dre[base + k], dim[base + k]);
                dre[base + k] = ur + vr;
                dim[base + k] = ui + vi;
                dre[base + k + half] = ur - vr;
                dim[base + k + half] = ui - vi;
            }
            base += len;
        }
        len <<= 1;
    }
}

/// Split / SoA complex multiply: clean vertical SIMD, no deinterleave.
pub fn cmul_soa(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32], or: &mut [f32], oi: &mut [f32]) {
    let n = ar.len();
    let mut i = 0;
    while i + 8 <= n {
        let a_r = f32x8::from(<[f32; 8]>::try_from(&ar[i..i + 8]).unwrap());
        let a_i = f32x8::from(<[f32; 8]>::try_from(&ai[i..i + 8]).unwrap());
        let b_r = f32x8::from(<[f32; 8]>::try_from(&br[i..i + 8]).unwrap());
        let b_i = f32x8::from(<[f32; 8]>::try_from(&bi[i..i + 8]).unwrap());
        or[i..i + 8].copy_from_slice(&(a_r * b_r - a_i * b_i).to_array());
        oi[i..i + 8].copy_from_slice(&(a_r * b_i + a_i * b_r).to_array());
        i += 8;
    }
    while i < n {
        or[i] = ar[i] * br[i] - ai[i] * bi[i];
        oi[i] = ar[i] * bi[i] + ai[i] * br[i];
        i += 1;
    }
}

/// Interleaved complex multiply: the AoS path. Vectorizing it needs a
/// deinterleave/shuffle the SoA path avoids; left scalar to expose that cost.
pub fn cmul_interleaved(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len() / 2;
    for i in 0..n {
        let (ar, ai, br, bi) = (a[2 * i], a[2 * i + 1], b[2 * i], b[2 * i + 1]);
        out[2 * i] = ar * br - ai * bi;
        out[2 * i + 1] = ar * bi + ai * br;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfft::{FftPlanner, num_complex::Complex};

    fn rustfft_ref(x: &[f32]) -> Vec<f32> {
        let n = x.len() / 2;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut buf: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new(x[2 * i], x[2 * i + 1]))
            .collect();
        fft.process(&mut buf);
        buf.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    #[test]
    fn split_radix_interleaved_matches_rustfft() {
        for &n in &[256usize, 1024, 4096] {
            let x: Vec<f32> = (0..2 * n)
                .map(|i| ((i * 7 + 3) % 17) as f32 - 8.0)
                .collect();
            let mut out = vec![0.0f32; 2 * n];
            split_radix_interleaved(&x, &mut out);
            let r = rustfft_ref(&x);
            let err = out
                .iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1.0, "N={n} max abs err {err}");
        }
    }

    #[test]
    fn split_radix_soa_matches_interleaved() {
        for &n in &[256usize, 1024, 4096] {
            let x: Vec<f32> = (0..2 * n)
                .map(|i| ((i * 11 + 1) % 13) as f32 - 6.0)
                .collect();
            let mut oi = vec![0.0f32; 2 * n];
            split_radix_interleaved(&x, &mut oi);
            let (re, im): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (x[2 * i], x[2 * i + 1])).unzip();
            let (mut orr, mut oii) = (vec![0.0f32; n], vec![0.0f32; n]);
            split_radix_soa(&re, &im, &mut orr, &mut oii);
            for k in 0..n {
                assert!((orr[k] - oi[2 * k]).abs() < 1e-2, "N={n} re k={k}");
                assert!((oii[k] - oi[2 * k + 1]).abs() < 1e-2, "N={n} im k={k}");
            }
        }
    }

    #[test]
    fn fft_iter_interleaved_matches_rustfft() {
        for &n in &[256usize, 1024, 4096] {
            let x: Vec<f32> = (0..2 * n)
                .map(|i| ((i * 7 + 3) % 17) as f32 - 8.0)
                .collect();
            let brev = bit_reverse_table(n);
            let tw = twiddle_table(n);
            let mut out = vec![0.0f32; 2 * n];
            fft_iter_interleaved(&x, &mut out, &brev, &tw);
            let r = rustfft_ref(&x);
            let err = out
                .iter()
                .zip(r.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err < 1.0, "N={n} max abs err {err}");
        }
    }

    #[test]
    fn fft_iter_soa_matches_interleaved() {
        for &n in &[256usize, 1024, 4096] {
            let x: Vec<f32> = (0..2 * n)
                .map(|i| ((i * 11 + 1) % 13) as f32 - 6.0)
                .collect();
            let brev = bit_reverse_table(n);
            let tw = twiddle_table(n);
            let mut oi = vec![0.0f32; 2 * n];
            fft_iter_interleaved(&x, &mut oi, &brev, &tw);
            let (re, im): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (x[2 * i], x[2 * i + 1])).unzip();
            let (mut orr, mut oii) = (vec![0.0f32; n], vec![0.0f32; n]);
            fft_iter_soa(&re, &im, &mut orr, &mut oii, &brev, &tw);
            for k in 0..n {
                assert!((orr[k] - oi[2 * k]).abs() < 1e-3, "N={n} re k={k}");
                assert!((oii[k] - oi[2 * k + 1]).abs() < 1e-3, "N={n} im k={k}");
            }
        }
    }

    #[test]
    fn cmul_layouts_agree() {
        let n = 1024;
        let a: Vec<f32> = (0..2 * n).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..2 * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let mut oi = vec![0.0f32; 2 * n];
        cmul_interleaved(&a, &b, &mut oi);
        let (ar, ai): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (a[2 * i], a[2 * i + 1])).unzip();
        let (br, bi): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (b[2 * i], b[2 * i + 1])).unzip();
        let (mut orr, mut oii) = (vec![0.0f32; n], vec![0.0f32; n]);
        cmul_soa(&ar, &ai, &br, &bi, &mut orr, &mut oii);
        for k in 0..n {
            assert!((orr[k] - oi[2 * k]).abs() < 1e-4 && (oii[k] - oi[2 * k + 1]).abs() < 1e-4);
        }
    }
}
