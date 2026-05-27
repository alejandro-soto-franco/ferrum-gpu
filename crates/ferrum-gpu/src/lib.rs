//! `ferrum-gpu`: pure-Rust GPU compute substrate.
//!
//! Backend-agnostic facade. Backend implementations live in
//! `ferrum-gpu-cuda` and (post-SP2b) `ferrum-gpu-vulkan`.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

pub mod buffer;
pub mod device;

pub use buffer::Buffer;
pub use device::Device;
pub use ferrum_gpu_core::{
    AnyBufferHandle, ArgKind, Backend, BackendCaps, BackendId, BufferArg, Dim3, Direction,
    FerrumGpuError, KernelArtifact, KernelMeta, LaunchArg, LaunchArgs, Result, TargetInfo,
};

/// CUDA backend submodule. Re-exports the marker, the device constructor,
/// and type aliases.
#[cfg(feature = "cuda")]
pub mod cuda {
    pub use ferrum_gpu_cuda::{Cuda, CudaBackendError};

    /// Convenience alias for `Device<Cuda>`.
    pub type Device = crate::Device<Cuda>;

    /// Convenience alias for `Buffer<T, Cuda>`.
    pub type Buffer<T> = crate::Buffer<T, Cuda>;

    impl crate::Device<Cuda> {
        /// Open CUDA device `ordinal` and return a `Device<Cuda>` with a
        /// default stream and an empty module cache.
        pub fn new(ordinal: usize) -> std::result::Result<Self, CudaBackendError> {
            let ctx = cudarc::driver::CudaContext::new(ordinal)?;
            Ok(Self::from_handle(ctx))
        }
    }
}
