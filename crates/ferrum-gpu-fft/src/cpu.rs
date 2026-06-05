//! CPU radix-2 Stockham auto-sort FFT, used as the reference for the GPU
//! kernel and for the rustfft cross-check.
//!
//! Reads the per-stage twiddle table laid out by `crate::twiddles::twiddles`:
//! stages descending (largest first), then by k.

use crate::complex::Complex32;
use crate::plan::{Direction, Plan};

/// Per-stage start offset into the twiddle table for the layout produced by
/// `twiddles::twiddles(log_n)`. Stage 1 occupies a single twiddle, stage 2
/// occupies two, and so on, but they appear in the table with stage `log_n`
/// first. This routine returns a length-`log_n + 1` table where
/// `stage_offsets[s]` is the start of stage `s`'s block (s in 1..=log_n).
fn stage_offsets(log_n: u32) -> Vec<usize> {
    let mut offs = vec![0usize; (log_n + 1) as usize];
    let mut off = 0usize;
    for s in (1..=log_n).rev() {
        offs[s as usize] = off;
        off += 1usize << (s - 1);
    }
    offs
}

/// One radix-2 Stockham stage.
///
/// Reads from `src` and writes to `dst`. `n` is the transform length.
/// `s` is the 1-based stage index. `tw` is the contiguous twiddle slice for
/// this stage (length `2^(s-1)`).
fn stockham_stage(src: &[Complex32], dst: &mut [Complex32], n: usize, s: u32, tw: &[Complex32]) {
    let m = 1usize << s;
    let m_half = 1usize << (s - 1);
    let l = n / m;
    let half_n = n / 2;
    for j in 0..l {
        for k in 0..m_half {
            let w = tw[k];
            let a = src[j * m_half + k];
            let b = src[j * m_half + k + half_n];
            let wb = w.mul(b);
            dst[j * m + k] = a.add(wb);
            dst[j * m + k + m_half] = a.sub(wb);
        }
    }
}

/// Run the forward Stockham FFT on one length-`n` lane.
///
/// `buf` is the input + output. `scratch` must have length `n`. The twiddles
/// slice is the full plan twiddle table (stages descending).
fn forward_lane(
    buf: &mut [Complex32],
    scratch: &mut [Complex32],
    log_n: u32,
    twiddles: &[Complex32],
    offs: &[usize],
) {
    let n = 1usize << log_n;
    // ping-pong: a/b are logical pointers selecting buf vs scratch.
    // After each stage we flip; after log_n stages, result is in buf if log_n
    // is even, else in scratch. We copy back at the end if needed.
    let mut from_buf_to_scratch = true; // first stage: read buf, write scratch
    for s in 1..=log_n {
        let m_half = 1usize << (s - 1);
        let tw = &twiddles[offs[s as usize]..offs[s as usize] + m_half];
        if from_buf_to_scratch {
            stockham_stage(buf, scratch, n, s, tw);
        } else {
            stockham_stage(scratch, buf, n, s, tw);
        }
        from_buf_to_scratch = !from_buf_to_scratch;
    }
    // If the last stage wrote into scratch (i.e. from_buf_to_scratch is now
    // false after toggling and result lives in scratch), copy back.
    // Tracking: initial from_buf_to_scratch=true means stage 1 wrote to scratch.
    // After log_n stages, the result is in `scratch` when log_n is odd, in
    // `buf` when log_n is even.
    if log_n % 2 == 1 {
        buf.copy_from_slice(scratch);
    }
}

impl Plan {
    /// CPU reference execution. Processes `buf` in place, lane by lane
    /// (`buf.len()` must be a multiple of `self.n()`).
    ///
    /// `Direction::Forward` computes the unnormalised DFT. `Direction::Inverse`
    /// computes the IDFT using the conjugate-trick: `IFFT(x) = conj(FFT(conj(x)))`,
    /// scaled by `1/N` when `self.normalize` is true.
    pub fn cpu_execute(&self, buf: &mut [Complex32], dir: Direction) {
        let n = self.n();
        assert!(
            !buf.is_empty() && buf.len() % n == 0,
            "buf.len() ({}) must be a positive multiple of n ({})",
            buf.len(),
            n
        );
        let offs = stage_offsets(self.log_n);
        let mut scratch = vec![Complex32::zero(); n];

        for lane in buf.chunks_mut(n) {
            match dir {
                Direction::Forward => {
                    forward_lane(lane, &mut scratch, self.log_n, &self.twiddles_host, &offs);
                }
                Direction::Inverse => {
                    for c in lane.iter_mut() {
                        *c = c.conj();
                    }
                    forward_lane(lane, &mut scratch, self.log_n, &self.twiddles_host, &offs);
                    for c in lane.iter_mut() {
                        *c = c.conj();
                    }
                    if self.normalize {
                        let inv_n = 1.0f32 / n as f32;
                        for c in lane.iter_mut() {
                            c.re *= inv_n;
                            c.im *= inv_n;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_offsets_log_n_3() {
        // Layout for log_n=3 is [W_8^0..3, W_4^0..1, W_2^0]; stage 3 at 0,
        // stage 2 at 4, stage 1 at 6.
        let offs = stage_offsets(3);
        assert_eq!(offs[3], 0);
        assert_eq!(offs[2], 4);
        assert_eq!(offs[1], 6);
    }

    #[test]
    fn forward_n4_matches_hand_dft() {
        // x = [1, 2, 3, 4]; expected DFT (forward, exp(-2 pi i k n / N)):
        //   X[0] = 10
        //   X[1] = -2 + 2i
        //   X[2] = -2
        //   X[3] = -2 - 2i
        let plan = Plan::new(2, 1, false);
        let mut buf = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        plan.cpu_execute(&mut buf, Direction::Forward);
        let expected = [
            Complex32::new(10.0, 0.0),
            Complex32::new(-2.0, 2.0),
            Complex32::new(-2.0, 0.0),
            Complex32::new(-2.0, -2.0),
        ];
        for (i, (a, b)) in buf.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a.re - b.re).abs() < 1e-5 && (a.im - b.im).abs() < 1e-5,
                "bin {i}: got {:?}, expected {:?}",
                a,
                b
            );
        }
    }
}
