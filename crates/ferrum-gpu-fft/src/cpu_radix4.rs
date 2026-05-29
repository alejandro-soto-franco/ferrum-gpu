//! CPU radix-4 Stockham auto-sort FFT reference — executable spec for the GPU
//! `fft_c2c_1024` kernel. Mirrors [`crate::cpu_radix8`] at radix 4 and the
//! [`crate::twiddles::twiddles_radix4`] layout; verified against the radix-2
//! reference.
//!
//! Stage `s` (1-based, `m = 4^s`, `m_r = 4^(s-1)`, `n/4` lanes apart):
//!   gather `src_p = src[j*m_r + k + p*(n/4)]` (p in 0..4); twiddle
//!   `t_p = W_m^(p*k) * src_p`; butterfly `X = DFT4(t)`; scatter
//!   `dst[j*m + k + q*m_r] = X_q`.

use crate::complex::Complex32;

/// In-register 4-point forward DFT on interleaved `[re, im, ...]` (8 floats).
/// `X_k = sum_n x_n exp(-2 pi i n k / 4)`; radix-2 split, `W_4 = ±i` folded.
#[inline(always)]
pub fn dft4_inplace(x: &mut [f32; 8]) {
    let (x0r, x0i) = (x[0], x[1]);
    let (x1r, x1i) = (x[2], x[3]);
    let (x2r, x2i) = (x[4], x[5]);
    let (x3r, x3i) = (x[6], x[7]);
    let (ar, ai) = (x0r + x2r, x0i + x2i);
    let (br, bi) = (x0r - x2r, x0i - x2i);
    let (cr, ci) = (x1r + x3r, x1i + x3i);
    let (dr, di) = (x1r - x3r, x1i - x3i);
    // X0 = a + c; X2 = a - c; X1 = b - i*d; X3 = b + i*d.
    // -i*d = (di, -dr); +i*d = (-di, dr).
    x[0] = ar + cr;
    x[1] = ai + ci;
    x[2] = br + di;
    x[3] = bi - dr;
    x[4] = ar - cr;
    x[5] = ai - ci;
    x[6] = br - di;
    x[7] = bi + dr;
}

/// Run one length-`n` (`n = 1 << log_n`, `log_n` even) forward radix-4 Stockham
/// FFT in place. `lane`/`scratch` are interleaved `[re, im, ...]` of length
/// `2*n`; `tw` is the full [`crate::twiddles::twiddles_radix4`] table.
pub fn radix4_forward_lane(lane: &mut [f32], scratch: &mut [f32], log_n: u32, tw: &[Complex32]) {
    let n = 1usize << log_n;
    let stages = log_n / 2;
    let nr = n / 4;
    let butterflies = n / 4;

    let mut src_is_lane = true;
    let mut m_r = 1usize; // 4^(s-1)
    let mut stage_off = 0usize;

    for _s in 1..=stages {
        let m = m_r * 4;
        for b in 0..butterflies {
            let j = b / m_r;
            let k = b - j * m_r;
            let src_base = j * m_r + k;

            let mut x = [0.0f32; 8];
            for p in 0..4usize {
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
                    let w = tw[stage_off + 4 * k + p];
                    x[2 * p] = re * w.re - im * w.im;
                    x[2 * p + 1] = re * w.im + im * w.re;
                }
            }

            dft4_inplace(&mut x);

            let dst_base = j * m + k;
            for q in 0..4usize {
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
        stage_off += 4 * m_r;
        m_r = m;
        src_is_lane = !src_is_lane;
    }

    if stages % 2 == 1 {
        lane.copy_from_slice(&scratch[..2 * n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Direction, Plan};
    use crate::twiddles::twiddles_radix4;

    #[test]
    fn dft4_matches_direct() {
        let inp: [f32; 8] = core::array::from_fn(|i| (i as f32 * 0.41).sin());
        let mut x = inp;
        dft4_inplace(&mut x);
        for k in 0..4usize {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for nn in 0..4usize {
                let ang = -2.0 * std::f64::consts::PI * (k * nn) as f64 / 4.0;
                let (s, c) = ang.sin_cos();
                re += inp[2 * nn] as f64 * c - inp[2 * nn + 1] as f64 * s;
                im += inp[2 * nn] as f64 * s + inp[2 * nn + 1] as f64 * c;
            }
            assert!(
                (x[2 * k] as f64 - re).abs() < 1e-4 && (x[2 * k + 1] as f64 - im).abs() < 1e-4,
                "k={k}"
            );
        }
    }

    #[test]
    fn radix4_matches_radix2_reference() {
        for &log_n in &[2u32, 4, 6, 8, 10] {
            let n = 1usize << log_n;
            let batch = 3usize;
            let tw = twiddles_radix4(log_n);
            let input: Vec<Complex32> = (0..n * batch)
                .map(|i| {
                    let t = (i % n) as f32;
                    Complex32::new((t * 0.013).sin(), (t * 0.021).cos())
                })
                .collect();
            let mut reference = input.clone();
            Plan::new(log_n, batch, false).cpu_execute(&mut reference, Direction::Forward);

            let mut scratch = vec![0.0f32; 2 * n];
            let mut got = vec![Complex32::zero(); n * batch];
            for (li, lane) in input.chunks(n).enumerate() {
                let mut flat: Vec<f32> = lane.iter().flat_map(|c| [c.re, c.im]).collect();
                radix4_forward_lane(&mut flat, &mut scratch, log_n, &tw);
                for i in 0..n {
                    got[li * n + i] = Complex32::new(flat[2 * i], flat[2 * i + 1]);
                }
            }
            for i in 0..n * batch {
                let (a, b) = (reference[i], got[i]);
                let err = ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt();
                let scale = (a.re * a.re + a.im * a.im).sqrt().max(1.0);
                assert!(err / scale < 1e-3, "log_n={log_n} i={i}: ref={a:?} got={b:?}");
            }
        }
    }
}
