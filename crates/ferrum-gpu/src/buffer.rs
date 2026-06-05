//! Ergonomic `Buffer<T, B>` wrapper.

use bytemuck::Pod;
use ferrum_gpu_core::Backend;

/// Typed device buffer. Holds the backend handle plus a clone of the
/// device handle for sync copies.
pub struct Buffer<T: Pod, B: Backend> {
    pub(crate) handle: B::BufferHandle<T>,
    pub(crate) dev: B::DeviceHandle,
    pub(crate) len: usize,
}

impl<T: Pod, B: Backend> Buffer<T, B> {
    pub(crate) fn from_handle(
        handle: B::BufferHandle<T>,
        dev: B::DeviceHandle,
        len: usize,
    ) -> Self {
        Self { handle, dev, len }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if zero-length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy host slice into device buffer (synchronous).
    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), B::Error> {
        B::copy_h2d(&self.dev, &mut self.handle, src)
    }

    /// Copy device buffer into host slice (synchronous).
    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), B::Error> {
        B::copy_d2h(&self.dev, &self.handle, dst)
    }

    /// Borrow the underlying backend handle (for launch packing).
    pub fn handle(&self) -> &B::BufferHandle<T> {
        &self.handle
    }
}
