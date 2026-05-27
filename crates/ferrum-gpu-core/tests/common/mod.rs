//! Shared test helpers across integration tests.

use core::marker::PhantomData;
use ferrum_gpu_core::{AnyBufferHandle, Backend, BackendId, Dim3, KernelArtifact, LaunchArgs};

#[derive(Clone)]
pub struct MockDevice;

#[derive(Clone, Default)]
pub struct MockBuf<T>(pub PhantomData<T>);
unsafe impl<T> Send for MockBuf<T> {}
unsafe impl<T> Sync for MockBuf<T> {}
impl<T: 'static> AnyBufferHandle<MockBackend> for MockBuf<T> {
    fn elem_type_id(&self) -> core::any::TypeId {
        core::any::TypeId::of::<T>()
    }
}

#[derive(Clone)]
pub struct MockStream;

#[derive(Clone)]
pub struct MockModule;

#[derive(Clone)]
pub struct MockKernel;

#[derive(Debug, thiserror::Error)]
#[error("mock error")]
pub struct MockError;

pub struct MockBackend;

impl Backend for MockBackend {
    type DeviceHandle = MockDevice;
    type BufferHandle<T: bytemuck::Pod> = MockBuf<T>;
    type Stream = MockStream;
    type Module = MockModule;
    type KernelHandle = MockKernel;
    type Error = MockError;

    fn id() -> BackendId { BackendId::Cuda }
    fn alloc_zeros<T: bytemuck::Pod>(_: &MockDevice, _: usize) -> Result<MockBuf<T>, MockError> {
        Ok(MockBuf(PhantomData))
    }
    fn copy_h2d<T: bytemuck::Pod>(_: &MockDevice, _: &mut MockBuf<T>, _: &[T]) -> Result<(), MockError> { Ok(()) }
    fn copy_d2h<T: bytemuck::Pod>(_: &MockDevice, _: &MockBuf<T>, _: &mut [T]) -> Result<(), MockError> { Ok(()) }
    fn load_module(_: &MockDevice, _: &KernelArtifact) -> Result<MockModule, MockError> { Ok(MockModule) }
    fn get_kernel(_: &MockModule, _: &str) -> Result<MockKernel, MockError> { Ok(MockKernel) }
    fn launch(
        _: &MockDevice, _: &MockStream, _: &MockKernel,
        _: Dim3, _: Dim3, _: u32, _: LaunchArgs<'_, Self>,
    ) -> Result<(), MockError> { Ok(()) }
    fn default_stream(_: &MockDevice) -> MockStream { MockStream }
    fn sync_stream(_: &MockDevice, _: &MockStream) -> Result<(), MockError> { Ok(()) }
}
