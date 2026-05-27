//! Twiddle factor generation for radix-2 Stockham FFT.

use core::f32::consts::PI;

use crate::complex::Complex32;

/// Returns the twiddle table laid out by stage (largest first), then by k:
/// `[W_{2^log_n}^0, ..., W_{2^log_n}^{half-1}, ..., W_2^0]`
/// of total length `2^log_n - 1`.
pub fn twiddles(log_n: u32) -> Vec<Complex32> {
    assert!(log_n >= 1, "twiddles requires log_n >= 1");
    let mut out = Vec::with_capacity((1usize << log_n).saturating_sub(1));
    for stage in (1..=log_n).rev() {
        let n_stage = 1u64 << stage;
        let half = (n_stage / 2) as usize;
        let scale = -2.0 * PI / n_stage as f32;
        for k in 0..half {
            let theta = scale * k as f32;
            out.push(Complex32::new(theta.cos(), theta.sin()));
        }
    }
    out
}
