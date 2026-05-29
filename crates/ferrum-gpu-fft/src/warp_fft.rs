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

/// `W_32^e = exp(-2 pi i e / 32)`.
fn twiddle32(e: usize) -> Complex32 {
    let theta = -2.0 * PI * (e as f32) / 32.0;
    Complex32::new(theta.cos(), theta.sin())
}

/// 5-bit reversal (for N = 32).
fn bitrev5(i: usize) -> usize {
    let mut r = 0;
    for b in 0..5 {
        if i & (1 << b) != 0 {
            r |= 1 << (4 - b);
        }
    }
    r
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn warp_fft32_matches_direct() {
        let mut x = [Complex32::zero(); 32];
        for n in 0..32usize {
            x[n] = Complex32::new((n as f32 * 0.31).sin(), (n as f32 * 0.17).cos());
        }
        let want = direct_dft32(&x);
        warp_fft32_model(&mut x);
        for k in 0..32usize {
            let err = ((x[k].re - want[k].re).powi(2) + (x[k].im - want[k].im).powi(2)).sqrt();
            let scale = (want[k].re * want[k].re + want[k].im * want[k].im).sqrt().max(1.0);
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
