//! Integration test crate.

mod common;
use common::MockBackend;
use ferrum_gpu_core::Backend;

#[test]
fn mock_backend_implements_trait() {
    let dev = common::MockDevice;
    let mut buf: common::MockBuf<f32> = MockBackend::alloc_zeros(&dev, 16).unwrap();
    MockBackend::copy_h2d(&dev, &mut buf, &[1.0_f32; 16]).unwrap();
    let mut out = [0.0_f32; 16];
    MockBackend::copy_d2h(&dev, &buf, &mut out).unwrap();
}
