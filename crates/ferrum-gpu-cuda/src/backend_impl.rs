//! `impl Backend for Cuda` over `cudarc::driver`.

use core::any::TypeId;

use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use ferrum_gpu_core::{
    AnyBufferHandle, Backend, BackendId, Dim3, KernelArtifact, LaunchArg, LaunchArgs,
};

use crate::Cuda;
use crate::buffer::CudaBuffer;
use crate::error::CudaBackendError;

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

    /// Load a compiled PTX artifact into a `CudaModule` on the device.
    ///
    /// The artifact's `blob` is interpreted as UTF-8 PTX source text. CUBIN
    /// blobs are not supported at v0.1.0.
    fn load_module(
        dev: &Arc<CudaContext>,
        artifact: &KernelArtifact,
    ) -> Result<Arc<CudaModule>, CudaBackendError> {
        if artifact.backend != BackendId::Cuda {
            return Err(CudaBackendError::BackendMismatch(artifact.backend));
        }
        let ptx_str = core::str::from_utf8(artifact.blob.as_ref()).map_err(|_| {
            CudaBackendError::KernelNotFound("artifact blob is not UTF-8 PTX".into())
        })?;
        let module = dev.load_module(Ptx::from_src(ptx_str.to_owned()))?;
        Ok(module)
    }

    /// Look up a kernel function by name inside a loaded module.
    fn get_kernel(module: &Arc<CudaModule>, name: &str) -> Result<CudaFunction, CudaBackendError> {
        module
            .load_function(name)
            .map_err(|_| CudaBackendError::KernelNotFound(name.into()))
    }

    /// Launch a kernel on `stream`.
    ///
    /// Routes through cudarc's safe `launch_builder` / `PushKernelArg` path
    /// to drive `cuLaunchKernel`. Each `LaunchArg::Buffer` is materialised as
    /// a stable `CUdeviceptr` slot in a per-launch `Vec`; pushing `&slot`
    /// into the builder pushes `&slot as *const u64 as *mut c_void` into the
    /// kernel-param array (which is what CUDA reads as the buffer base
    /// address). Each `LaunchArg::Scalar(bytes)` pushes `&bytes[0]`, whose
    /// address is the start of the scalar value's representation.
    fn launch(
        dev: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        kernel: &CudaFunction,
        grid: Dim3,
        block: Dim3,
        shared_bytes: u32,
        args: LaunchArgs<'_, Self>,
    ) -> Result<(), CudaBackendError> {
        dev.bind_to_thread()?;

        // Stable storage: one `CUdeviceptr` per buffer arg. Pre-sized so no
        // reallocation invalidates the references handed to the builder.
        let buf_count = args
            .iter()
            .filter(|a| matches!(a, LaunchArg::Buffer(_)))
            .count();
        let mut buf_ptrs: Vec<sys::CUdeviceptr> = Vec::with_capacity(buf_count);
        for a in args.iter() {
            if let LaunchArg::Buffer(b) = a {
                buf_ptrs.push(cuda_buffer_device_ptr(b.handle));
            }
        }

        let mut builder = stream.launch_builder(kernel);
        let cfg = LaunchConfig {
            grid_dim: (grid.x, grid.y, grid.z),
            block_dim: (block.x, block.y, block.z),
            shared_mem_bytes: shared_bytes,
        };

        let mut buf_idx = 0usize;
        for a in args.iter() {
            match a {
                LaunchArg::Buffer(_) => {
                    let ptr_ref: &sys::CUdeviceptr = &buf_ptrs[buf_idx];
                    builder.arg(ptr_ref);
                    buf_idx += 1;
                }
                LaunchArg::Scalar(bytes) => {
                    // `bytes.is_empty()` would be a caller bug; reject up
                    // front so we never deref past the slice.
                    assert!(
                        !bytes.is_empty(),
                        "ferrum-gpu-cuda: LaunchArg::Scalar got an empty byte slice"
                    );
                    let first: &u8 = &bytes[0];
                    builder.arg(first);
                }
            }
        }

        // SAFETY: the kernel-arg types, count, and order must match the
        // kernel's signature. This is the standard trust boundary for any
        // CUDA kernel launch and is the responsibility of the caller (in
        // practice, the `#[kernel]` macro's host stub in Plan 2). The
        // backing storage for both buffer pointers (`buf_ptrs`) and scalar
        // bytes (held by the caller for the lifetime of `args`) outlives
        // the synchronous-from-host-perspective launch call.
        unsafe { builder.launch(cfg)? };

        // Keep buffer-pointer storage alive across the launch call.
        drop(buf_ptrs);
        Ok(())
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

/// Allow `CudaBuffer<T>` to be packed as a `LaunchArg::Buffer`.
impl<T: Pod + 'static> AnyBufferHandle<Cuda> for CudaBuffer<T> {
    fn elem_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

/// Recover the raw device pointer from a type-erased `AnyBufferHandle<Cuda>`.
///
/// Inside this crate, the only type that implements `AnyBufferHandle<Cuda>` is
/// `CudaBuffer<T>`, which is `#[repr(C)]` with `CUdeviceptr` as its first
/// field. The trait object's data pointer therefore points at the start of a
/// `CudaBuffer<T>`, and reading the first field yields the device pointer
/// regardless of `T` (since `PhantomData<T>` is zero-sized and lives at the
/// tail of the struct).
fn cuda_buffer_device_ptr(handle: &dyn AnyBufferHandle<Cuda>) -> sys::CUdeviceptr {
    // Convert the fat `&dyn` pointer to a thin data pointer.
    let thin: *const () = core::ptr::from_ref(handle).cast::<()>();
    let ptr_field: *const sys::CUdeviceptr = thin.cast::<sys::CUdeviceptr>();
    // SAFETY: the only implementor of `AnyBufferHandle<Cuda>` in this crate
    // is `CudaBuffer<T>`, which is `#[repr(C)]` with `ptr: CUdeviceptr` as
    // its first field. The data pointer is properly aligned for
    // `CUdeviceptr` (u64) because `CudaBuffer<T>` includes a `CUdeviceptr`
    // as its first field, forcing the struct's alignment to be at least
    // that of u64.
    unsafe { *ptr_field }
}
