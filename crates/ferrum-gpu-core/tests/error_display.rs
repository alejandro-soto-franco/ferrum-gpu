use ferrum_gpu_core::FerrumGpuError;

#[test]
fn backend_mismatch_displays() {
    let e = FerrumGpuError::BackendMismatch {
        expected: 1,
        got: 2,
    };
    let s = std::format!("{e}");
    assert!(s.contains("backend mismatch"));
    assert!(s.contains("expected") && s.contains("got"));
}

#[test]
fn kernel_not_found_carries_name() {
    let e = FerrumGpuError::KernelNotFound("foo".into());
    assert!(std::format!("{e}").contains("foo"));
}
