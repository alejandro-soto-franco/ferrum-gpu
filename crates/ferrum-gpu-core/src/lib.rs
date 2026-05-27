//! Core substrate types for `ferrum-gpu`.
//!
//! Backend-agnostic contract surface. `no_std + alloc`. Backend
//! implementations like `ferrum-gpu-cuda` (and future `ferrum-gpu-vulkan`)
//! implement [`Backend`].

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

pub mod artifact;
pub mod dim;
pub mod error;
pub mod backend;
pub mod launch;

pub use artifact::{
    ArgKind, BackendCaps, BackendId, KernelArtifact, KernelMeta, TargetInfo,
};
pub use dim::{Dim3, Direction};
pub use error::{FerrumGpuError, Result};
pub use backend::Backend;
pub use launch::{AnyBufferHandle, BufferArg, LaunchArg, LaunchArgs};
