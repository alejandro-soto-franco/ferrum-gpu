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
pub mod plan;
pub mod twiddles;

pub use complex::Complex32;
pub use plan::{Direction, KernelKind, Plan, Plan2D};
pub use twiddles::twiddles;
