//! CUDA-specific error type. Wraps `cudarc::DriverError` and adds the
//! backend-mismatch / kernel-not-found cases at this layer.

use thiserror::Error;

/// Errors raised by the CUDA backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CudaBackendError {
    /// CUDA driver error from `cudarc`.
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// nvrtc compile error.
    #[error("nvrtc compile error: {0}")]
    Nvrtc(#[from] cudarc::nvrtc::CompileError),

    /// The artifact's backend tag does not match `BackendId::Cuda`.
    #[error("backend mismatch: expected Cuda (id=1), got id={0:?}")]
    BackendMismatch(ferrum_gpu_core::BackendId),

    /// A kernel name was not found in the loaded module.
    #[error("kernel not found in module: {0}")]
    KernelNotFound(String),

    /// The requested target architecture is incompatible with the loaded device.
    #[error("target mismatch: artifact targets {artifact}, device is {device}")]
    TargetMismatch {
        /// Artifact's target description.
        artifact: String,
        /// Device's capability description.
        device: String,
    },
}
