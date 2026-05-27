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

pub mod dim;

pub use dim::{Dim3, Direction};
