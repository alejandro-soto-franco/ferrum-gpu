//! Complex32 type used by the FFT kernels and host code.
//!
//! Defined as `[f32; 2]`-shaped (real, imag interleaved) so it is trivially
//! `Pod + Zeroable + Send + Sync` without dragging in num_complex on the
//! device side. The host code converts to `num_complex::Complex32` when
//! needed for arithmetic.

use bytemuck::{Pod, Zeroable};

/// 32-bit interleaved complex number. Layout matches CUDA's `cuComplex`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Complex32 {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

#[allow(clippy::should_implement_trait)]
impl Complex32 {
    /// Construct from real + imag.
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Zero element.
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Multiplication. Host-side only; kernel-side uses raw f32 ops.
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Addition.
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Subtraction.
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Conjugate (used to switch direction without rebuilding twiddles).
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Convert to num_complex.
    pub fn to_num(self) -> num_complex::Complex<f32> {
        num_complex::Complex::new(self.re, self.im)
    }

    /// Convert from num_complex.
    pub fn from_num(c: num_complex::Complex<f32>) -> Self {
        Self::new(c.re, c.im)
    }
}
