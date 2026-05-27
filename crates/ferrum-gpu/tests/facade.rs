use ferrum_gpu::{cuda, Device};

fn gpu_available() -> bool {
    std::env::var("FERRUM_GPU_HAS_CUDA").map(|v| v == "1").unwrap_or(false)
}

#[test]
fn facade_round_trip() {
    if !gpu_available() { return; }
    let dev: Device<cuda::Cuda> = cuda::Device::new(0).expect("cuda device 0");
    let n = 256;
    let mut buf = dev.alloc::<f32>(n).expect("alloc");
    let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
    buf.copy_from_host(&src).expect("h2d");
    let mut dst = vec![0.0; n];
    buf.copy_to_host(&mut dst).expect("d2h");
    assert_eq!(src, dst);
}
