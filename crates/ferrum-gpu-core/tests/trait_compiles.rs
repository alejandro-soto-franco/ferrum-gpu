use core::marker::PhantomData;
use ferrum_gpu_core::{Backend, BackendId, Dim3, KernelArtifact};

#[derive(Clone)]
struct MockDevice;

#[derive(Clone)]
struct MockBuf<T>(PhantomData<T>);
unsafe impl<T> Send for MockBuf<T> {}
unsafe impl<T> Sync for MockBuf<T> {}

#[derive(Clone)]
struct MockStream;

#[derive(Clone)]
struct MockModule;

#[derive(Clone)]
struct MockKernel;

#[derive(Debug, thiserror::Error)]
#[error("mock error")]
struct MockError;

struct MockBackend;

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
        _: Dim3, _: Dim3, _: u32, _: ferrum_gpu_core::LaunchArgs<'_, Self>,
    ) -> Result<(), MockError> { Ok(()) }
    fn default_stream(_: &MockDevice) -> MockStream { MockStream }
    fn sync_stream(_: &MockDevice, _: &MockStream) -> Result<(), MockError> { Ok(()) }
}

#[test]
fn mock_backend_implements_trait() {
    let dev = MockDevice;
    let mut buf: MockBuf<f32> = MockBackend::alloc_zeros(&dev, 16).unwrap();
    MockBackend::copy_h2d(&dev, &mut buf, &[1.0_f32; 16]).unwrap();
    let mut out = [0.0_f32; 16];
    MockBackend::copy_d2h(&dev, &buf, &mut out).unwrap();
}
