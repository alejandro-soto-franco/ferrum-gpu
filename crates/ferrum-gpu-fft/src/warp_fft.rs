//! CPU model of a warp-distributed radix-2 FFT, used to nail the lane layout,
//! twiddle factors, shuffle masks, and output ordering *before* porting to a
//! GPU `shfl_xor` kernel (the four-step / register-blocked redesign that aims
//! to close the cuFFT gap; see `ferrum-gpu-planning`).
//!
//! Model: a warp of 32 lanes, one complex element per lane (lane `L` starts
//! with `x[L]`). The transform is decimation-in-frequency (DIF): natural-order
//! input, bit-reversed-order intermediate output, reordered to natural at the
//! end. Each stage is a cross-lane radix-2 butterfly whose partner is
//! `L XOR d` — exactly the `shfl_xor(., d)` the GPU kernel will issue.
//!
//! [`warp_fft32_model`] is intentionally written in lock-step over all 32
//! lanes (computing every lane's next register from the current registers via
//! a simulated `shfl_xor`) so the control flow matches the SIMT kernel. The
//! test checks it against a direct DFT.

use core::f32::consts::PI;

use crate::complex::Complex32;

/// `W_N^e = exp(-2 pi i e / N)`.
fn twiddle(e: usize, n: usize) -> Complex32 {
    let theta = -2.0 * PI * (e as f32) / (n as f32);
    Complex32::new(theta.cos(), theta.sin())
}

/// `W_32^e = exp(-2 pi i e / 32)`.
fn twiddle32(e: usize) -> Complex32 {
    twiddle(e, 32)
}

/// `log_n`-bit reversal.
fn bitrev(i: usize, log_n: u32) -> usize {
    let mut r = 0;
    for b in 0..log_n {
        if i & (1 << b) != 0 {
            r |= 1 << (log_n - 1 - b);
        }
    }
    r
}

/// 5-bit reversal (for N = 32).
fn bitrev5(i: usize) -> usize {
    bitrev(i, 5)
}

/// In-place forward DFT of `n = 1 << log_n` points via the warp-distributed
/// DIF schedule, general over `n`. Same butterfly as [`warp_fft32_model`]:
/// stage `s` has distance `d = (n/2) >> s`, partner `g XOR d` (a `shfl_xor`
/// when `d < 32`, a within-lane swap when `d == 32`), twiddle
/// `W_n^((g mod d) * 2^s)` on the upper element; output is unscrambled from
/// bit-reversed order at the end. This is the executable spec for the
/// multi-element-per-lane GPU warp kernels (e.g. the 64-pt building block of
/// the four-step 4096 FFT).
pub fn warp_dif_model(x: &mut [Complex32], log_n: u32) {
    let n = 1usize << log_n;
    debug_assert_eq!(x.len(), n);
    for s in 0..log_n {
        let d = (n >> 1) >> s;
        let cur = x.to_vec();
        for g in 0..n {
            let partner = cur[g ^ d];
            if g & d == 0 {
                x[g] = cur[g].add(partner);
            } else {
                let diff = partner.sub(cur[g]);
                let exp = (g & (d - 1)) * (1usize << s);
                x[g] = diff.mul(twiddle(exp, n));
            }
        }
    }
    let scrambled = x.to_vec();
    for g in 0..n {
        x[bitrev(g, log_n)] = scrambled[g];
    }
}

/// In-place forward DFT of 32 complex points via the warp-distributed DIF
/// schedule. `x[L]` is "lane L's register". Mirrors the GPU kernel:
/// each stage does `partner = shfl_xor(x, d)` then a lower/upper butterfly.
pub fn warp_fft32_model(x: &mut [Complex32; 32]) {
    // DIF stages s = 0..5, butterfly distance d = 16, 8, 4, 2, 1.
    for s in 0..5u32 {
        let d = 16usize >> s;
        let cur = *x;
        let mut next = cur;
        for l in 0..32usize {
            let partner = cur[l ^ d]; // simulated shfl_xor(x, d)
            if l & d == 0 {
                // Lower lane (index i): X[i] = a + b, a = x[l], b = x[l+d].
                next[l] = cur[l].add(partner);
            } else {
                // Upper lane (index i+d): X[i+d] = (a - b) * W_32^(j * 2^s),
                // a = x[l-d] = partner, b = x[l], j = l mod d.
                let diff = partner.sub(cur[l]);
                let exp = (l % d) * (1usize << s);
                next[l] = diff.mul(twiddle32(exp));
            }
        }
        *x = next;
    }
    // DIF leaves X[k] in bit-reversed position: register l holds X[bitrev5(l)].
    let scrambled = *x;
    let mut natural = scrambled;
    for l in 0..32usize {
        natural[bitrev5(l)] = scrambled[l];
    }
    *x = natural;
}

/// CPU model of the four-step (Cooley-Tukey `N = N1 * N2`) FFT that the GPU
/// `fft_c2c_4096_4step` kernel will implement. Decomposition `n = n1 + N1*n2`,
/// `k = N2*k1 + k2`:
///   1. inner DFTs of size `N2` over `n2` for each `n1` -> `B[n1][k2]`
///   2. twiddle `C[n1][k2] = B[n1][k2] * W_N^(n1*k2)`
///   3. outer DFTs of size `N1` over `n1` for each `k2` -> `X[N2*k1 + k2]`
///
/// The size-`N1`/`N2` sub-transforms use [`warp_dif_model`] (the warp FFT), so
/// this also exercises the warp kernel in composition. Verified against the
/// radix-2 reference for N = 4096 (N1 = N2 = 64).
pub fn four_step_model(input: &[Complex32], log_n1: u32, log_n2: u32) -> Vec<Complex32> {
    let n1 = 1usize << log_n1;
    let n2 = 1usize << log_n2;
    let n = n1 * n2;
    debug_assert_eq!(input.len(), n);

    // Step 1: inner DFT over n2 for each n1; store B[n1][k2] row-major in n2.
    let mut b = vec![Complex32::zero(); n];
    for a in 0..n1 {
        let mut col: Vec<Complex32> = (0..n2).map(|c| input[a + n1 * c]).collect();
        warp_dif_model(&mut col, log_n2);
        for k2 in 0..n2 {
            b[a * n2 + k2] = col[k2];
        }
    }

    // Step 2: twiddle.
    for a in 0..n1 {
        for k2 in 0..n2 {
            b[a * n2 + k2] = b[a * n2 + k2].mul(twiddle(a * k2, n));
        }
    }

    // Step 3: outer DFT over n1 for each k2; scatter to X[N2*k1 + k2].
    let mut out = vec![Complex32::zero(); n];
    for k2 in 0..n2 {
        let mut coln: Vec<Complex32> = (0..n1).map(|a| b[a * n2 + k2]).collect();
        warp_dif_model(&mut coln, log_n1);
        for k1 in 0..n1 {
            out[n2 * k1 + k2] = coln[k1];
        }
    }
    out
}

/// CPU model of the N=256 warp-per-FFT kernel (`fft_c2c_256_warp`): the
/// four-step decomposition `256 = 32 x 8` (N1=32 the warp/lane dimension,
/// N2=8 the in-register dimension). Executable spec for the GPU kernel's lane
/// layout and output ordering:
///
/// * lane L holds 8 inputs `x[L + 32*n2]`, `n2 in 0..8`;
/// * step 1: in-register 8-pt DFT over n2 -> `B[L][k2]`;
/// * step 2: twiddle `B[L][k2] *= W_256^(L*k2)`;
/// * step 3: 32-pt warp DFT over the lane dimension for each k2 -> X;
/// * output `X[8*k1 + k2]` (k1 = lane after the warp transform).
///
/// Verified against the radix-2 reference in tests.
pub fn warp256_model(input: &[Complex32]) -> Vec<Complex32> {
    four_step_model(input, 5, 3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Direction, Plan};

    #[allow(clippy::needless_range_loop)]
    fn direct_dft32(x: &[Complex32; 32]) -> [Complex32; 32] {
        let mut out = [Complex32::zero(); 32];
        for k in 0..32usize {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for n in 0..32usize {
                let ang = -2.0 * std::f64::consts::PI * (k * n) as f64 / 32.0;
                let (s, c) = ang.sin_cos();
                let (xr, xi) = (x[n].re as f64, x[n].im as f64);
                re += xr * c - xi * s;
                im += xr * s + xi * c;
            }
            out[k] = Complex32::new(re as f32, im as f32);
        }
        out
    }

    fn direct_dft(x: &[Complex32]) -> Vec<Complex32> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (nn, c) in x.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * (k * nn) as f64 / n as f64;
                    let (s, cs) = ang.sin_cos();
                    re += c.re as f64 * cs - c.im as f64 * s;
                    im += c.re as f64 * s + c.im as f64 * cs;
                }
                Complex32::new(re as f32, im as f32)
            })
            .collect()
    }

    #[test]
    fn warp_dif_matches_direct_32_and_64() {
        for &log_n in &[5u32, 6] {
            let n = 1usize << log_n;
            let mut x: Vec<Complex32> = (0..n)
                .map(|i| Complex32::new((i as f32 * 0.29).sin(), (i as f32 * 0.13).cos()))
                .collect();
            let want = direct_dft(&x);
            warp_dif_model(&mut x, log_n);
            for k in 0..n {
                let err = ((x[k].re - want[k].re).powi(2) + (x[k].im - want[k].im).powi(2)).sqrt();
                let scale = (want[k].re * want[k].re + want[k].im * want[k].im)
                    .sqrt()
                    .max(1.0);
                assert!(
                    err / scale < 1e-4,
                    "n={n} bin {k}: got {:?} want {:?}",
                    x[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn four_step_4096_matches_radix2_reference() {
        let n = 4096usize;
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.021).cos()))
            .collect();
        let mut reference = input.clone();
        Plan::new(12, 1, false).cpu_execute(&mut reference, Direction::Forward);

        let got = four_step_model(&input, 6, 6); // 64 x 64
        for k in 0..n {
            let err = ((got[k].re - reference[k].re).powi(2)
                + (got[k].im - reference[k].im).powi(2))
            .sqrt();
            let scale = (reference[k].re * reference[k].re + reference[k].im * reference[k].im)
                .sqrt()
                .max(1.0);
            assert!(
                err / scale < 1e-3,
                "bin {k}: four-step {:?} vs radix-2 {:?}",
                got[k],
                reference[k]
            );
        }
    }

    #[test]
    fn warp256_matches_radix2_reference() {
        // Asymmetric four-step (N1=32 != N2=8) — the layout the GPU
        // fft_c2c_256_warp kernel implements. Not covered by the 64x64 test.
        let n = 256usize;
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.021).cos()))
            .collect();
        let mut reference = input.clone();
        Plan::new(8, 1, false).cpu_execute(&mut reference, Direction::Forward);

        let got = warp256_model(&input);
        for k in 0..n {
            let err = ((got[k].re - reference[k].re).powi(2)
                + (got[k].im - reference[k].im).powi(2))
            .sqrt();
            let scale = (reference[k].re * reference[k].re + reference[k].im * reference[k].im)
                .sqrt()
                .max(1.0);
            assert!(
                err / scale < 1e-3,
                "bin {k}: warp256 {:?} vs radix-2 {:?}",
                got[k],
                reference[k]
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn warp_fft32_matches_direct() {
        let mut x = [Complex32::zero(); 32];
        for n in 0..32usize {
            x[n] = Complex32::new((n as f32 * 0.31).sin(), (n as f32 * 0.17).cos());
        }
        let want = direct_dft32(&x);
        warp_fft32_model(&mut x);
        for k in 0..32usize {
            let err = ((x[k].re - want[k].re).powi(2) + (x[k].im - want[k].im).powi(2)).sqrt();
            let scale = (want[k].re * want[k].re + want[k].im * want[k].im)
                .sqrt()
                .max(1.0);
            assert!(
                err / scale < 1e-4,
                "bin {k}: got {:?}, want {:?}, relerr {}",
                x[k],
                want[k],
                err / scale
            );
        }
    }
}
