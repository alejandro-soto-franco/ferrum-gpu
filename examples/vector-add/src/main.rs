//! End-to-end Plan 1 demo: load a hand-written PTX vector_add kernel
//! through the `ferrum-gpu` substrate, dispatch on the user's GPU,
//! verify the result on host.
//!
//! The PTX targets PTX 7.0 / sm_70 (broadly compatible). Plan 2 routes
//! kernels through cuda-oxide for Rust source compilation.

use std::borrow::Cow;

use ferrum_gpu::cuda;
use ferrum_gpu_core::{
    ArgKind, Backend, BackendCaps, BackendId, Dim3, KernelArtifact, KernelMeta, LaunchArg,
    LaunchArgs, TargetInfo,
};
use ferrum_gpu_cuda::Cuda;

const VECTOR_ADD_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry vector_add(
    .param .u64 param_0,
    .param .u64 param_1,
    .param .u64 param_2,
    .param .u32 param_3
)
{
    .reg .pred  %p<2>;
    .reg .f32   %f<4>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<11>;

    ld.param.u64 %rd1, [param_0];
    ld.param.u64 %rd2, [param_1];
    ld.param.u64 %rd3, [param_2];
    ld.param.u32 %r2, [param_3];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;
    setp.ge.s32 %p1, %r1, %r2;
    @%p1 bra LBB0_2;

    cvta.to.global.u64 %rd4, %rd1;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    cvta.to.global.u64 %rd7, %rd2;
    add.s64 %rd8, %rd7, %rd5;
    cvta.to.global.u64 %rd9, %rd3;
    add.s64 %rd10, %rd9, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd8];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd10], %f3;

LBB0_2:
    ret;
}
"#;

fn main() -> anyhow::Result<()> {
    let dev = cuda::Device::new(0)?;
    let n: i32 = 1 << 20;
    println!("device opened; preparing {} elements", n);

    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();

    let mut da = dev.alloc::<f32>(n as usize)?;
    let mut db = dev.alloc::<f32>(n as usize)?;
    let dc = dev.alloc::<f32>(n as usize)?;
    da.copy_from_host(&a)?;
    db.copy_from_host(&b)?;

    println!("loading hand-written PTX (target=sm_70)");
    let artifact = KernelArtifact {
        backend: BackendId::Cuda,
        target: TargetInfo::Cuda { sm_major: 7, sm_minor: 0 },
        blob: Cow::Borrowed(VECTOR_ADD_PTX.as_bytes()),
        manifest: vec![KernelMeta {
            mangled_name: "vector_add".into(),
            arg_kinds: vec![
                ArgKind::BufferF32,
                ArgKind::BufferF32,
                ArgKind::BufferF32,
                ArgKind::ScalarI32,
            ],
            shared_memory_bytes: 0,
            recommended_block: Some(Dim3::new(256, 1, 1)),
            required_caps: BackendCaps::Cuda { min_sm_major: 5, min_sm_minor: 0 },
        }],
    };
    let module = dev.load_or_cache("vector-add", &artifact)?;
    let kernel = Cuda::get_kernel(&module, "vector_add")?;

    println!("launching kernel");
    let n_bytes = bytemuck::bytes_of(&n);
    let args = [
        LaunchArg::<Cuda>::buffer(da.handle()),
        LaunchArg::<Cuda>::buffer(db.handle()),
        LaunchArg::<Cuda>::buffer(dc.handle()),
        LaunchArg::<Cuda>::scalar(n_bytes),
    ];
    let pack = LaunchArgs::new(&args);
    let grid = Dim3::new((n as u32 + 255) / 256, 1, 1);
    let block = Dim3::new(256, 1, 1);
    Cuda::launch(dev.handle(), dev.default_stream(), &kernel, grid, block, 0, pack)?;
    dev.sync()?;

    let mut c = vec![0.0_f32; n as usize];
    dc.copy_to_host(&mut c)?;
    for i in 0..n as usize {
        let expected = a[i] + b[i];
        if (c[i] - expected).abs() > 1e-6 {
            anyhow::bail!("mismatch at {}: got {}, expected {}", i, c[i], expected);
        }
    }
    println!("vector_add: {} elements verified", n);
    Ok(())
}
