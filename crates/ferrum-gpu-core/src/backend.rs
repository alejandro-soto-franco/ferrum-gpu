//! Backend contract.

use bytemuck::Pod;

use crate::artifact::{BackendId, KernelArtifact};
use crate::dim::Dim3;
use crate::launch::LaunchArgs;

/// Contract every backend implementation must satisfy. Each method is
/// async-free in v0.1.0; streams are present in the signature so future
/// async APIs can layer on without trait breakage.
pub trait Backend: 'static + Send + Sync + Sized {
    /// Opaque device handle owned by `Device<Self>`.
    type DeviceHandle: Clone + Send + Sync;
    /// Backend-owned buffer of `T` on the device.
    type BufferHandle<T: Pod>: Send + Sync;
    /// Execution stream / queue handle.
    type Stream: Clone + Send + Sync;
    /// A loaded module (collection of kernels).
    type Module: Clone + Send + Sync;
    /// A handle to one kernel inside a module.
    type KernelHandle: Clone + Send + Sync;
    /// Backend-typed error. Wraps the underlying driver error.
    type Error: core::error::Error + Send + Sync + 'static;

    /// Compile-time tag identifying this backend.
    fn id() -> BackendId;

    /// Allocate an `n`-element buffer zero-initialised on the device.
    fn alloc_zeros<T: Pod>(
        dev: &Self::DeviceHandle,
        n: usize,
    ) -> Result<Self::BufferHandle<T>, Self::Error>;

    /// Synchronously copy host slice into device buffer.
    fn copy_h2d<T: Pod>(
        dev: &Self::DeviceHandle,
        buf: &mut Self::BufferHandle<T>,
        src: &[T],
    ) -> Result<(), Self::Error>;

    /// Synchronously copy device buffer into host slice.
    fn copy_d2h<T: Pod>(
        dev: &Self::DeviceHandle,
        buf: &Self::BufferHandle<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error>;

    /// Load a compiled artifact into a module on the device.
    fn load_module(
        dev: &Self::DeviceHandle,
        artifact: &KernelArtifact,
    ) -> Result<Self::Module, Self::Error>;

    /// Look up a kernel by its mangled name.
    fn get_kernel(
        module: &Self::Module,
        mangled_name: &str,
    ) -> Result<Self::KernelHandle, Self::Error>;

    /// Launch a kernel on the given stream.
    fn launch(
        dev: &Self::DeviceHandle,
        stream: &Self::Stream,
        kernel: &Self::KernelHandle,
        grid: Dim3,
        block: Dim3,
        shared_bytes: u32,
        args: LaunchArgs<'_, Self>,
    ) -> Result<(), Self::Error>;

    /// The default stream associated with a device.
    fn default_stream(dev: &Self::DeviceHandle) -> Self::Stream;

    /// Block the calling thread until all work on `stream` completes.
    fn sync_stream(
        dev: &Self::DeviceHandle,
        stream: &Self::Stream,
    ) -> Result<(), Self::Error>;
}
