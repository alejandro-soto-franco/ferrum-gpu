//! Launch-time argument packing. Filled in in Task 7.

use core::marker::PhantomData;

use crate::backend::Backend;

/// Placeholder argument-pack type. Replaced with the full implementation in Task 7.
pub struct LaunchArgs<'a, B: Backend> {
    _p: PhantomData<&'a B>,
}

/// Placeholder enum.
pub enum LaunchArg<'a, B: Backend> {
    /// To be filled.
    _Buffer(BufferArg<'a, B>),
    /// To be filled.
    _Scalar(&'a [u8]),
}

/// Placeholder struct.
pub struct BufferArg<'a, B: Backend> {
    _p: PhantomData<&'a B>,
}

/// Placeholder trait.
pub trait AnyBufferHandle<B: Backend> {}
