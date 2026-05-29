//! CPU radix-8 Stockham auto-sort FFT reference.
//!
//! This module is the executable specification for the GPU `fft_c2c_4096`
//! kernel: it mirrors the kernel's gather/twiddle/butterfly/scatter exactly,
//! on interleaved `[re, im, ...]` `f32` buffers and the
//! [`crate::twiddles::twiddles_radix8`] table layout. The cross-check test
//! verifies it against the numpy-validated radix-2 [`crate::cpu`] reference, so
//! a bug in the algorithm shows up here (where iteration is instant) rather
//! than on the GPU.
//!
//! Stage `s` (1-based, `m = 8^s`, `m_r = 8^(s-1)`, `n/8` lanes apart):
//!   * gather   `src_p = src[j*m_r + k + p*(n/8)]`, `p in 0..8`
//!   * twiddle  `t_p = W_m^(p*k) * src_p`  (`t_0 = src_0`)
//!   * butterfly `X = DFT8(t)`             ([`dft8_inplace`])
//!   * scatter  `dst[j*m + k + q*m_r] = X_q`, `q in 0..8`

use crate::complex::Complex32;

/// In-register 8-point forward DFT on interleaved `[re, im, ...]` input.
///
/// `x` holds 8 complex values (16 `f32`). On return it holds
/// `X_k = sum_n x_n * exp(-2 pi i n k / 8)`. Decimation-in-time radix-2 flow
/// graph with the `W_8` factors constant-folded; `C = cos(pi/4) = sin(pi/4)`.
#[inline(always)]
pub fn dft8_inplace(x: &mut [f32; 16]) {
    const C: f32 = core::f32::consts::FRAC_1_SQRT_2;

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

    // Combine: X_q = E_q + W_8^q O_q, X_{q+4} = E_q - W_8^q O_q.
    let (w0r, w0i) = (o0r, o0i); // W_8^0 = 1
    let (w1r, w1i) = (C * (o1r + o1i), C * (o1i - o1r)); // W_8^1 = (C, -C)
    let (w2r, w2i) = (o2i, -o2r); // W_8^2 = -i
    let (w3r, w3i) = (C * (o3i - o3r), -C * (o3r + o3i)); // W_8^3 = (-C, -C)

    x[0] = e0r + w0r;
    x[1] = e0i + w0i;
    x[2] = e1r + w1r;
    x[3] = e1i + w1i;
    x[4] = e2r + w2r;
    x[5] = e2i + w2i;
    x[6] = e3r + w3r;
    x[7] = e3i + w3i;
    x[8] = e0r - w0r;
    x[9] = e0i - w0i;
    x[10] = e1r - w1r;
    x[11] = e1i - w1i;
    x[12] = e2r - w2r;
    x[13] = e2i - w2i;
    x[14] = e3r - w3r;
    x[15] = e3i - w3i;
}

/// Run one length-`n` (`n = 1 << log_n`, `log_n` a multiple of 3) forward
/// radix-8 Stockham FFT in place.
///
/// `lane` is interleaved `[re, im, ...]` of length `2 * n`. `scratch` must have
/// length `2 * n`. `tw` is the full [`crate::twiddles::twiddles_radix8`] table.
pub fn radix8_forward_lane(lane: &mut [f32], scratch: &mut [f32], log_n: u32, tw: &[Complex32]) {
    let n = 1usize << log_n;
    let stages = log_n / 3;
    let nr = n / 8;
    let butterflies = n / 8;

    let mut src_is_lane = true;
    let mut m_r = 1usize; // 8^(s-1)
    let mut stage_off = 0usize;

    for _s in 1..=stages {
        let m = m_r * 8; // 8^s
        for b in 0..butterflies {
            let j = b / m_r;
            let k = b - j * m_r;
            let src_base = j * m_r + k;

            let mut x = [0.0f32; 16];
            for p in 0..8usize {
                let si = 2 * (src_base + p * nr);
                let (re, im) = if src_is_lane {
                    (lane[si], lane[si + 1])
                } else {
                    (scratch[si], scratch[si + 1])
                };
                if p == 0 {
                    x[0] = re;
                    x[1] = im;
                } else {
                    let w = tw[stage_off + 8 * k + p];
                    x[2 * p] = re * w.re - im * w.im;
                    x[2 * p + 1] = re * w.im + im * w.re;
                }
            }

            dft8_inplace(&mut x);

            let dst_base = j * m + k;
            for q in 0..8usize {
                let di = 2 * (dst_base + q * m_r);
                if src_is_lane {
                    scratch[di] = x[2 * q];
                    scratch[di + 1] = x[2 * q + 1];
                } else {
                    lane[di] = x[2 * q];
                    lane[di + 1] = x[2 * q + 1];
                }
            }
        }
        stage_off += 8 * m_r;
        m_r = m;
        src_is_lane = !src_is_lane;
    }

    // After `stages` stages the result is in `lane` when `stages` is even,
    // in `scratch` when odd.
    if stages % 2 == 1 {
        lane.copy_from_slice(&scratch[..2 * n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Direction, Plan};
    use crate::twiddles::twiddles_radix8;

    #[test]
    fn dft8_matches_direct() {
        let inp: [f32; 16] = core::array::from_fn(|i| (i as f32 * 0.37).sin());
        let mut x = inp;
        dft8_inplace(&mut x);
        for k in 0..8usize {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for nn in 0..8usize {
                let ang = -2.0 * std::f64::consts::PI * (k * nn) as f64 / 8.0;
                let (s, c) = ang.sin_cos();
                let (xr, xi) = (inp[2 * nn] as f64, inp[2 * nn + 1] as f64);
                re += xr * c - xi * s;
                im += xr * s + xi * c;
            }
            let (gr, gi) = (x[2 * k] as f64, x[2 * k + 1] as f64);
            assert!(
                (gr - re).abs() < 1e-4 && (gi - im).abs() < 1e-4,
                "k={k}: got ({gr}, {gi}), want ({re}, {im})"
            );
        }
    }

    #[test]
    fn radix8_matches_radix2_reference() {
        for &log_n in &[3u32, 6, 9, 12] {
            let n = 1usize << log_n;
            let batch = 3usize;
            let tw = twiddles_radix8(log_n);

            let input: Vec<Complex32> = (0..n * batch)
                .map(|i| {
                    let t = (i % n) as f32;
                    Complex32::new((t * 0.013).sin(), (t * 0.021).cos())
                })
                .collect();

            // Ground truth: radix-2 Stockham (cross-checked against numpy).
            let mut reference = input.clone();
            Plan::new(log_n, batch, false).cpu_execute(&mut reference, Direction::Forward);

            // Radix-8 under test.
            let mut scratch = vec![0.0f32; 2 * n];
            let mut got = vec![Complex32::zero(); n * batch];
            for (lane_idx, lane) in input.chunks(n).enumerate() {
                let mut flat: Vec<f32> =
                    lane.iter().flat_map(|c| [c.re, c.im]).collect();
                radix8_forward_lane(&mut flat, &mut scratch, log_n, &tw);
                for i in 0..n {
                    got[lane_idx * n + i] = Complex32::new(flat[2 * i], flat[2 * i + 1]);
                }
            }

            for i in 0..n * batch {
                let (a, b) = (reference[i], got[i]);
                let err = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
                let scale = (a.re * a.re + a.im * a.im).sqrt().max(1.0);
                assert!(
                    err / scale < 1e-3,
                    "log_n={log_n} i={i}: ref={a:?} got={b:?} relerr={}",
                    err / scale
                );
            }
        }
    }
}
