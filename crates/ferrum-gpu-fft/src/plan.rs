//! Plan: book-keeping for a 1D radix-2 power-of-2 C2C FFT.

use crate::complex::Complex32;
use crate::twiddles;

/// Transform direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Forward: sum_n x[n] exp(-2 pi i n k / N).
    Forward,
    /// Inverse: sum_n X[k] exp(+2 pi i n k / N), optionally divided by N.
    Inverse,
}

/// A 1D radix-2 power-of-2 C2C plan.
#[derive(Clone, Debug)]
pub struct Plan {
    /// log2 of transform length. Must be in [2, 12] for v0.0.3 (N = 4 through 4096).
    pub log_n: u32,
    /// Number of independent transforms processed per execute() call.
    pub batch: usize,
    /// Whether to divide the inverse-direction output by N.
    pub normalize: bool,
    /// Twiddle table laid out per stage (see `twiddles::twiddles`).
    pub twiddles_host: Vec<Complex32>,
}

impl Plan {
    /// Construct a new plan and precompute twiddles.
    pub fn new(log_n: u32, batch: usize, normalize: bool) -> Self {
        assert!((2..=12).contains(&log_n), "log_n must be in [2, 12]");
        let twiddles_host = twiddles::twiddles(log_n);
        Self { log_n, batch, normalize, twiddles_host }
    }

    /// Transform length: `1 << log_n`.
    pub fn n(&self) -> usize {
        1usize << self.log_n
    }

    /// Borrow the twiddle table.
    pub fn twiddles(&self) -> &[Complex32] {
        &self.twiddles_host
    }
}
