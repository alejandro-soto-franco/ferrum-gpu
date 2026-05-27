//! Core error type. Backends extend with their own typed errors.

use alloc::string::String;
use thiserror::Error;

/// Errors raised by core types. Backend implementations carry their own
/// error type (`Backend::Error`) and convert into this only when crossing
/// the substrate boundary into user code that wants a unified type.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FerrumGpuError {
    /// A `KernelArtifact` tagged with a different backend was passed to a
    /// backend that cannot consume it.
    #[error("backend mismatch: expected {expected}, got {got}")]
    BackendMismatch {
        /// The backend that received the artifact.
        expected: u8,
        /// The backend that produced the artifact.
        got: u8,
    },

    /// A kernel name was requested from a module that does not contain it.
    #[error("kernel not found: {0}")]
    KernelNotFound(String),

    /// The toolchain configuration on disk does not match what the builder
    /// expected (e.g. wrong rustc nightly).
    #[error("toolchain mismatch: {0}")]
    ToolchainMismatch(String),

    /// A backend-specific error promoted to the core type.
    #[error("backend error: {0}")]
    Backend(String),
}

/// Convenience alias for results in the core crate.
pub type Result<T> = core::result::Result<T, FerrumGpuError>;
