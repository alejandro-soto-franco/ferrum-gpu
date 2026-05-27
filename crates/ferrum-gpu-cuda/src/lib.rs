//! CUDA backend for `ferrum-gpu`.
//!
//! Realises `ferrum_gpu_core::Backend` over `cudarc`. The marker type
//! [`Cuda`] is the type parameter `B` users supply to
//! `ferrum_gpu::Device<Cuda>`, `Buffer<T, Cuda>`, etc.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

pub mod backend_impl;
pub mod buffer;
pub mod error;

pub use buffer::CudaBuffer;
pub use error::CudaBackendError;

/// Zero-sized marker selecting the CUDA backend.
#[derive(Clone, Copy, Debug)]
pub struct Cuda;
