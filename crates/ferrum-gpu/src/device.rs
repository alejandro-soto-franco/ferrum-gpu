//! Ergonomic `Device<B>` wrapper around a `Backend::DeviceHandle`.

use std::collections::HashMap;

use bytemuck::Pod;
use ferrum_gpu_core::{Backend, KernelArtifact};
use parking_lot::Mutex;

use crate::buffer::Buffer;

/// Backend-typed device handle with a module cache and a default stream.
pub struct Device<B: Backend> {
    pub(crate) inner: B::DeviceHandle,
    pub(crate) default_stream: B::Stream,
    pub(crate) modules: Mutex<HashMap<&'static str, B::Module>>,
}

impl<B: Backend> Device<B> {
    /// Construct from a backend-supplied handle.
    pub fn from_handle(handle: B::DeviceHandle) -> Self {
        let default_stream = B::default_stream(&handle);
        Self {
            inner: handle,
            default_stream,
            modules: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate an `n`-element zero-initialised buffer.
    pub fn alloc<T: Pod>(&self, n: usize) -> Result<Buffer<T, B>, B::Error> {
        let handle = B::alloc_zeros::<T>(&self.inner, n)?;
        Ok(Buffer::from_handle(handle, self.inner.clone(), n))
    }

    /// Synchronise the default stream.
    pub fn sync(&self) -> Result<(), B::Error> {
        B::sync_stream(&self.inner, &self.default_stream)
    }

    /// Load (or return cached) module for the given crate identifier.
    pub fn load_or_cache(
        &self,
        crate_name: &'static str,
        art: &KernelArtifact,
    ) -> Result<B::Module, B::Error> {
        let mut cache = self.modules.lock();
        if let Some(m) = cache.get(crate_name) {
            return Ok(m.clone());
        }
        let module = B::load_module(&self.inner, art)?;
        cache.insert(crate_name, module.clone());
        Ok(module)
    }

    /// Borrow the underlying backend handle (for backend-specific helpers).
    pub fn handle(&self) -> &B::DeviceHandle {
        &self.inner
    }

    /// Borrow the default stream.
    pub fn default_stream(&self) -> &B::Stream {
        &self.default_stream
    }
}
