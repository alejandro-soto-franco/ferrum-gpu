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

/// Input-twiddle table for a radix-4 Stockham auto-sort FFT of length
/// `1 << log_n` (`log_n` must be a positive multiple of 2).
///
/// Layout, by ascending stage `s = 1..=log_n/2`: for `k in 0..m_r`, for
/// `p in 0..4`: `W_m^((p*k) mod m)`, where `m = 4^s`, `m_r = 4^(s-1)`. Each
/// stage block has length `4 * m_r`; the `p == 0` entries are `(1, 0)` so the
/// kernel can index `stage_off + 4*k + p` branch-free. The radix-4 butterfly's
/// internal `W_4` factors (just `±i`) are folded into the butterfly itself.
pub fn twiddles_radix4(log_n: u32) -> Vec<Complex32> {
    assert!(
        log_n >= 2 && log_n % 2 == 0,
        "twiddles_radix4 requires log_n a positive multiple of 2"
    );
    let stages = log_n / 2;
    let mut out = Vec::new();
    let mut m_r = 1usize; // 4^(s-1)
    for _s in 1..=stages {
        let m = m_r * 4; // 4^s
        let scale = -2.0 * PI / m as f32;
        for k in 0..m_r {
            for p in 0..4usize {
                let e = (p * k) % m;
                let theta = scale * e as f32;
                out.push(Complex32::new(theta.cos(), theta.sin()));
            }
        }
        m_r = m;
    }
    out
}

/// Input-twiddle table for a radix-8 Stockham auto-sort FFT of length
/// `1 << log_n` (`log_n` must be a positive multiple of 3).
///
/// Layout, by ascending stage `s = 1..=log_n/3`:
/// for `k in 0..m_r`, for `p in 0..8`: `W_m^((p*k) mod m)`, where
/// `m = 8^s`, `m_r = 8^(s-1)`, and `W_m = exp(-2 pi i / m)`. Each stage block
/// has length `8 * m_r`; the `p == 0` entries are `(1, 0)` and are kept so the
/// kernel can index `stage_off + 8*k + p` branch-free.
///
/// The radix-8 butterfly's internal `W_8^(p*q)` factors are constant-folded
/// into the butterfly itself (see `cpu_radix8::dft8_inplace` and the GPU
/// `fft_c2c_4096` kernel); this table holds only the per-stage input twiddles.
pub fn twiddles_radix8(log_n: u32) -> Vec<Complex32> {
    assert!(
        log_n >= 3 && log_n % 3 == 0,
        "twiddles_radix8 requires log_n a positive multiple of 3"
    );
    let stages = log_n / 3;
    let mut out = Vec::new();
    let mut m_r = 1usize; // 8^(s-1)
    for _s in 1..=stages {
        let m = m_r * 8; // 8^s
        let scale = -2.0 * PI / m as f32;
        for k in 0..m_r {
            for p in 0..8usize {
                let e = (p * k) % m;
                let theta = scale * e as f32;
                out.push(Complex32::new(theta.cos(), theta.sin()));
            }
        }
        m_r = m;
    }
    out
}
