//! `CudaBuffer<T>` — an owned device allocation backed by raw cudarc driver calls.
//!
//! Using the low-level `cudarc::driver::result` layer lets us work with any
//! `T: bytemuck::Pod` without requiring `cudarc`'s `DeviceRepr + ValidAsZeroBits`
//! bounds, which are not implied by `Pod`.

use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use cudarc::driver::{CudaContext, CudaStream, result};

/// Owned device buffer of `n` elements of type `T`.
///
/// Drop frees the underlying device allocation synchronously via the stream
/// the buffer was created on.
///
/// `#[repr(C)]` is load-bearing: the kernel-launch path in `backend_impl.rs`
/// reads `ptr` at offset 0 of the struct via a type-erased view in order to
/// pack `LaunchArg::Buffer` handles into the raw `cuLaunchKernel` arg vector.
/// `PhantomData<T>` is zero-sized so the prefix layout (CUdeviceptr, usize,
/// Arc<CudaStream>) is identical for every `T`.
#[repr(C)]
pub struct CudaBuffer<T: Pod> {
    /// Raw device pointer.
    pub(crate) ptr: cudarc::driver::sys::CUdeviceptr,
    /// Number of elements (not bytes).
    pub(crate) len: usize,
    /// Stream the buffer was allocated on; kept alive so drop can synchronize.
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) _marker: PhantomData<T>,
}

// SAFETY: `CUdeviceptr` is an integer alias for a device-side address. The
// buffer does not hold any host-side pointer to `T`; the `PhantomData<T>` is
// purely for type-level tracking. Device memory is not accessible from the
// host without an explicit copy call, so sharing or sending the handle across
// threads is safe regardless of whether `T` itself is `Send`/`Sync`.
unsafe impl<T: Pod> Send for CudaBuffer<T> {}
// SAFETY: same reasoning as the preceding Send impl. Sharing &CudaBuffer<T>
// across threads only exposes the device pointer (an integer) and length,
// neither of which is a `T`-typed host-side value.
unsafe impl<T: Pod> Sync for CudaBuffer<T> {}

impl<T: Pod> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 || self.len == 0 {
            return;
        }
        // Synchronize before freeing to ensure in-flight async copies are done.
        let _ = self.stream.synchronize();
        // SAFETY: ptr was allocated with malloc_sync or malloc_async on this
        // stream; we only free once (this is the only Drop impl).
        let _ = unsafe { result::free_sync(self.ptr) };
    }
}

impl<T: Pod> CudaBuffer<T> {
    /// Allocate `n` elements of `T`, zero-initialised.
    pub(crate) fn alloc_zeros(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        n: usize,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let num_bytes = n * std::mem::size_of::<T>();
        ctx.bind_to_thread()?;
        let ptr = if num_bytes == 0 {
            0
        } else {
            // SAFETY: malloc_sync does not depend on element type; we treat the
            // memory as a byte array for the subsequent memset.
            let p = unsafe { result::malloc_sync(num_bytes) }?;
            // Zero the allocation synchronously.
            // SAFETY: p is freshly allocated, num_bytes matches, stream is valid.
            unsafe {
                result::memset_d8_async(p, 0, num_bytes, stream.cu_stream())?;
            }
            stream.synchronize()?;
            p
        };
        Ok(CudaBuffer {
            ptr,
            len: n,
            stream: stream.clone(),
            _marker: PhantomData,
        })
    }

    /// Copy a host slice into this buffer (synchronous from caller's perspective).
    pub(crate) fn copy_from_host(
        &mut self,
        ctx: &Arc<CudaContext>,
        src: &[T],
    ) -> Result<(), cudarc::driver::DriverError> {
        assert!(
            src.len() <= self.len,
            "copy_h2d: src.len() ({}) > buf.len() ({})",
            src.len(),
            self.len
        );
        if src.is_empty() {
            return Ok(());
        }
        ctx.bind_to_thread()?;
        // SAFETY: ptr is valid, src lives for the duration of the async call,
        // and we synchronize before returning.
        unsafe {
            result::memcpy_htod_async(self.ptr, src, self.stream.cu_stream())?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Copy this buffer into a host slice (synchronous from caller's perspective).
    pub(crate) fn copy_to_host(
        &self,
        ctx: &Arc<CudaContext>,
        dst: &mut [T],
    ) -> Result<(), cudarc::driver::DriverError> {
        assert!(
            dst.len() >= self.len,
            "copy_d2h: dst.len() ({}) < buf.len() ({})",
            dst.len(),
            self.len
        );
        if self.len == 0 {
            return Ok(());
        }
        ctx.bind_to_thread()?;
        // SAFETY: ptr is valid for self.len elements of T; dst is long enough;
        // we synchronize before returning so dst is populated before we exit.
        unsafe {
            result::memcpy_dtoh_async(&mut dst[..self.len], self.ptr, self.stream.cu_stream())?;
        }
        self.stream.synchronize()?;
        Ok(())
    }
}
