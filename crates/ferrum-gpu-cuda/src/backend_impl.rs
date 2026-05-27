//! `impl Backend for Cuda` over `cudarc::driver`.
//!
//! Only `alloc_zeros`, `copy_h2d`, `copy_d2h`, `default_stream`, and
//! `sync_stream` are implemented here. `load_module`, `get_kernel`, and
//! `launch` are stubbed with `unimplemented!`; Task 10 fills them in.

use core::any::TypeId;

use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use ferrum_gpu_core::{AnyBufferHandle, Backend, BackendId, Dim3, KernelArtifact, LaunchArgs};

use crate::buffer::CudaBuffer;
use crate::error::CudaBackendError;
use crate::Cuda;

impl Backend for Cuda {
    /// The device handle is the `Arc<CudaContext>` returned by `CudaContext::new`.
    type DeviceHandle = Arc<CudaContext>;
    /// An owned device allocation; see `crate::buffer::CudaBuffer`.
    type BufferHandle<T: Pod> = CudaBuffer<T>;
    /// A CUDA stream.
    type Stream = Arc<CudaStream>;
    /// A loaded PTX module.
    type Module = Arc<CudaModule>;
    /// A handle to one kernel function inside a module.
    type KernelHandle = CudaFunction;
    /// Error type wrapping `cudarc::driver::DriverError` plus backend-specific variants.
    type Error = CudaBackendError;

    fn id() -> BackendId {
        BackendId::Cuda
    }

    /// Allocate an `n`-element buffer, zero-initialised on the device.
    ///
    /// Uses the device's default stream for the allocation and the subsequent
    /// memset, synchronising before returning so the caller sees zeroed memory.
    fn alloc_zeros<T: Pod>(
        dev: &Arc<CudaContext>,
        n: usize,
    ) -> Result<CudaBuffer<T>, CudaBackendError> {
        let stream = dev.default_stream();
        let buf = CudaBuffer::<T>::alloc_zeros(dev, &stream, n)?;
        Ok(buf)
    }

    /// Synchronously copy a host slice into a device buffer.
    fn copy_h2d<T: Pod>(
        dev: &Arc<CudaContext>,
        buf: &mut CudaBuffer<T>,
        src: &[T],
    ) -> Result<(), CudaBackendError> {
        buf.copy_from_host(dev, src)?;
        Ok(())
    }

    /// Synchronously copy a device buffer into a host slice.
    fn copy_d2h<T: Pod>(
        dev: &Arc<CudaContext>,
        buf: &CudaBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), CudaBackendError> {
        buf.copy_to_host(dev, dst)?;
        Ok(())
    }

    // --- Stubs: Task 10 lands these ---

    fn load_module(
        _dev: &Arc<CudaContext>,
        _art: &KernelArtifact,
    ) -> Result<Arc<CudaModule>, CudaBackendError> {
        unimplemented!("load_module lands in Task 10")
    }

    fn get_kernel(
        _module: &Arc<CudaModule>,
        _name: &str,
    ) -> Result<CudaFunction, CudaBackendError> {
        unimplemented!("get_kernel lands in Task 10")
    }

    fn launch(
        _dev: &Arc<CudaContext>,
        _stream: &Arc<CudaStream>,
        _kernel: &CudaFunction,
        _grid: Dim3,
        _block: Dim3,
        _shared: u32,
        _args: LaunchArgs<'_, Self>,
    ) -> Result<(), CudaBackendError> {
        unimplemented!("launch lands in Task 10")
    }

    // --- Stream helpers ---

    fn default_stream(dev: &Arc<CudaContext>) -> Arc<CudaStream> {
        dev.default_stream()
    }

    fn sync_stream(
        _dev: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), CudaBackendError> {
        stream.synchronize()?;
        Ok(())
    }
}

/// Allow `CudaBuffer<T>` to be packed as a `LaunchArg::Buffer` in Task 10.
impl<T: Pod + 'static> AnyBufferHandle<Cuda> for CudaBuffer<T> {
    fn elem_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}
