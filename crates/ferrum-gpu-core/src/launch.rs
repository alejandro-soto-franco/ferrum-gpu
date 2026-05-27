//! Backend-uniform launch-argument packing.
//!
//! A `LaunchArgs<'a, B>` is a slice of `LaunchArg<'a, B>` variants. Each
//! backend's `Backend::launch` impl iterates this slice and packs into its
//! own native representation (CUDA: `*const c_void` array for
//! `cuLaunchKernel`; Vulkan: descriptor sets + push constants).

use core::any::TypeId;

use crate::backend::Backend;

/// Object-safe view of a backend buffer handle, used for runtime-typed
/// argument packing.
pub trait AnyBufferHandle<B: Backend>: Send + Sync {
    /// `TypeId` of the buffer's element type, used by the backend to
    /// validate kernel-arg layouts at launch time.
    fn elem_type_id(&self) -> TypeId;
}

/// A single launch argument.
pub enum LaunchArg<'a, B: Backend> {
    /// A buffer argument (passed by reference to the backend handle).
    Buffer(BufferArg<'a, B>),
    /// A scalar argument (the raw bytes of a `Pod` value).
    Scalar(&'a [u8]),
}

impl<'a, B: Backend> LaunchArg<'a, B> {
    /// Construct a buffer argument from any handle that implements
    /// `AnyBufferHandle<B>`.
    pub fn buffer<H: AnyBufferHandle<B> + 'a>(h: &'a H) -> Self {
        Self::Buffer(BufferArg { handle: h })
    }

    /// Construct a scalar argument from raw `Pod` bytes (use
    /// `bytemuck::bytes_of(&x)`).
    pub fn scalar(bytes: &'a [u8]) -> Self {
        Self::Scalar(bytes)
    }
}

/// A buffer reference inside a `LaunchArg`.
pub struct BufferArg<'a, B: Backend> {
    /// The buffer handle, type-erased through `AnyBufferHandle`.
    pub handle: &'a (dyn AnyBufferHandle<B> + 'a),
}

/// Packed argument list passed to `Backend::launch`.
pub struct LaunchArgs<'a, B: Backend> {
    args: &'a [LaunchArg<'a, B>],
}

impl<'a, B: Backend> LaunchArgs<'a, B> {
    /// Construct from a slice of args.
    pub fn new(args: &'a [LaunchArg<'a, B>]) -> Self {
        Self { args }
    }

    /// Number of arguments.
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// `true` when empty.
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Iterate the args in order.
    pub fn iter(&self) -> core::slice::Iter<'_, LaunchArg<'a, B>> {
        self.args.iter()
    }
}
