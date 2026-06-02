//! FFT application built on the ferrum-gpu substrate.
//!
//! v0.0.3 scope: 1D radix-2 power-of-2 C2C, sizes 4 through 4096, batched,
//! in-place. The GPU kernel lives in the consuming binary's source (see
//! `examples/fft-1d-c2c/src/main.rs`) because cuda-oxide's `#[cuda_module]`
//! embeds PTX into the binary crate only. This crate is host-only: twiddle
//! generation, the CPU reference, and the `Plan` book-keeping type.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

pub mod complex;
pub mod cpu;
pub mod cpu_2d;
pub mod cpu_radix4;
pub mod cpu_radix8;
pub mod plan;
pub mod twiddles;
pub mod warp_fft;

pub use complex::Complex32;
pub use plan::{Direction, KernelKind, Plan, Plan2D};
pub use twiddles::{twiddles, twiddles_full_roots, twiddles_radix4, twiddles_radix8};

/// Minimum batch at which the host runner routes N=256 and N=4096 forward
/// transforms to the scalarised radix-16 + u64-coalesced kernels
/// (`fft_c2c_256_r16s` / `fft_c2c_4096_r16s`) instead of the latency-tuned
/// `fft_c2c_256` / `fft_c2c_4096`. Below this the GPU is grid-starved and the
/// register/shared kernels win on launch latency; at or above it the radix-16
/// kernels reach cuFFT parity (N=256) and a 1.74x improvement (N=4096). See
/// `tools/large-batch-findings.md`. 4096 is the smallest batch the sweep
/// measured the radix-16 kernels strictly ahead at both sizes.
pub const R16S_MIN_BATCH: usize = 4096;
