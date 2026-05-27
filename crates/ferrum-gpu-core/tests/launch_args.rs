use ferrum_gpu_core::{LaunchArg, LaunchArgs};

// Use the MockBackend from Task 6's test by re-declaring it here. (Rust
// integration tests cannot share helpers across files unless you create a
// `common/mod.rs` module.)
mod common;
use common::{MockBackend, MockBuf};

#[test]
fn launch_args_zero_args_constructs() {
    let args: LaunchArgs<'_, MockBackend> = LaunchArgs::new(&[]);
    assert_eq!(args.len(), 0);
}

#[test]
fn launch_args_scalar_arg_round_trips() {
    let x: u32 = 42;
    let bytes = bytemuck::bytes_of(&x);
    let args = [LaunchArg::<MockBackend>::scalar(bytes)];
    let pack = LaunchArgs::new(&args);
    assert_eq!(pack.len(), 1);
}

#[test]
fn launch_args_buffer_arg_round_trips() {
    let buf: MockBuf<f32> = MockBuf::default();
    let args = [LaunchArg::<MockBackend>::buffer(&buf)];
    let pack = LaunchArgs::new(&args);
    assert_eq!(pack.len(), 1);
}
