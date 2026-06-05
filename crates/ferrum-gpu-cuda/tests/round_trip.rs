//! Host-to-device-to-host round-trip on a real GPU. Requires
//! `FERRUM_GPU_HAS_CUDA=1` plus a working NVIDIA driver + CUDA 12.x.

use cudarc::driver::CudaContext;
use ferrum_gpu_core::Backend;
use ferrum_gpu_cuda::Cuda;

fn gpu_available() -> bool {
    std::env::var("FERRUM_GPU_HAS_CUDA")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[test]
fn round_trip_1024_f32() {
    if !gpu_available() {
        eprintln!("skipping: set FERRUM_GPU_HAS_CUDA=1 to run");
        return;
    }
    let ctx = CudaContext::new(0).expect("CUDA context");
    let mut buf = Cuda::alloc_zeros::<f32>(&ctx, 1024).expect("alloc");
    let src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    Cuda::copy_h2d(&ctx, &mut buf, &src).expect("h2d");
    let mut dst = vec![0.0_f32; 1024];
    Cuda::copy_d2h(&ctx, &buf, &mut dst).expect("d2h");
    assert_eq!(src, dst);
}
