//! Correctness gate for the specialised FFT kernels.
//!
//! cuda-oxide keys embedded PTX to the building binary crate, so the
//! `#[cuda_module]` block cannot be a library item and integration tests
//! (`cargo test`) cannot reach it (`cargo oxide test` does not exist). This
//! binary is the stand-in: it `include!`s the kernel source the same way the
//! bench does, runs each specialised kernel on the GPU, and compares the
//! result against the numpy-validated CPU reference (`Plan::cpu_execute`).
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin kernel-cross-check`.
//! Exits 0 on pass, 1 on any mismatch.

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrum_gpu_fft::{Complex32, Direction, KernelKind, Plan};

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

/// Relative-error tolerance for the GPU-vs-CPU comparison (fp32, matches the
/// example's verification gate).
const REL_TOL: f32 = 1e-3;

fn flatten(c: &[Complex32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(c.len() * 2);
    for x in c {
        v.push(x.re);
        v.push(x.im);
    }
    v
}

fn test_input(n: usize, batch: usize) -> Vec<Complex32> {
    (0..n * batch)
        .map(|i| {
            let t = (i % n) as f32;
            Complex32::new((t * 0.013).sin(), (t * 0.021).cos())
        })
        .collect()
}

/// Largest relative error between a GPU result (interleaved flat) and the CPU
/// reference (complex), normalised by the reference magnitude (floored at 1).
fn max_rel_err(gpu_flat: &[f32], cpu: &[Complex32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, c) in cpu.iter().enumerate() {
        let (gr, gi) = (gpu_flat[2 * i], gpu_flat[2 * i + 1]);
        let err = ((gr - c.re).powi(2) + (gi - c.im).powi(2)).sqrt();
        let scale = (c.re * c.re + c.im * c.im).sqrt().max(1.0);
        let rel = err / scale;
        if rel > worst {
            worst = rel;
            worst_i = i;
        }
    }
    (worst, worst_i)
}

fn check_specialised_4096(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
) -> Result<bool> {
    let log_n = 12u32;
    let n = 1usize << log_n;
    let batch = 4usize;
    let total = n * batch;

    let plan = Plan::new(log_n, batch, false);
    assert_eq!(plan.kernel_kind, KernelKind::Specialised4096);

    let input = test_input(n, batch);

    // CPU ground truth.
    let mut cpu = input.clone();
    plan.cpu_execute(&mut cpu, Direction::Forward);

    // GPU.
    let stream = ctx.default_stream();
    let dbuf_in = DeviceBuffer::from_host(&stream, &flatten(&input))?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &flatten(&plan.kernel_twiddles()))?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    };
    module.fft_c2c_4096(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
    let gpu = dbuf_out.to_host_vec(&stream)?;

    let (rel, idx) = max_rel_err(&gpu, &cpu);
    let pass = rel <= REL_TOL;
    println!(
        "fft_c2c_4096 vs CPU (N={n}, batch={batch}): max_rel_err={rel:.2e} at bin {idx} -> {}",
        if pass { "PASS" } else { "FAIL" }
    );
    if !pass {
        let c = cpu[idx];
        eprintln!(
            "  worst bin {idx}: cpu=({:.5}, {:.5}) gpu=({:.5}, {:.5})",
            c.re, c.im, gpu[2 * idx], gpu[2 * idx + 1]
        );
    }
    Ok(pass)
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("CudaContext::new: {e}"))?;
    let module = kernels::load(&ctx)?;

    let mut all_pass = true;
    all_pass &= check_specialised_4096(&ctx, &module)?;

    if all_pass {
        println!("\nkernel-cross-check: PASS");
        Ok(())
    } else {
        eprintln!("\nkernel-cross-check: FAIL");
        std::process::exit(1);
    }
}
