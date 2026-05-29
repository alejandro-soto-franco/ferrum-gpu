//! Plan 3 vertical integration test: 1D radix-2 power-of-2 C2C FFT.
//!
//! The kernel is a single-block-per-FFT Stockham auto-sort implementation
//! that pings between two `SharedArray<f32, 8192>` slots. For each stage,
//! one thread handles one butterfly; the block writes the final ping-pong
//! slot back to global memory. Sizes 4 through 4096 are supported (N_MAX
//! = 4096; shared memory = 2 * 8192 * 4 = 65536 bytes per block, which
//! fits within the RTX 5060's per-block shared-memory ceiling on sm_120).
//!
//! The release gate is the 8-case verification table at the bottom of the
//! binary: each case constructs a `ferrum_gpu_fft::Plan`, builds a CPU
//! reference via `Plan::cpu_execute`, runs the GPU kernel against the same
//! input, and asserts elementwise relative error below 1e-4.
//!
//! Inverse direction is implemented on the host via the conjugate trick
//! (conj input → forward FFT → conj output, optionally / N). The kernel
//! itself does forward only and ignores `direction`.

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrum_gpu_fft::{Complex32, Direction, KernelKind, Plan};

include!("../../../crates/ferrum-gpu-fft-kernels/src/kernels_body.rs");

fn flatten(c: &[Complex32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(c.len() * 2);
    for x in c {
        v.push(x.re);
        v.push(x.im);
    }
    v
}

fn unflatten(f: &[f32]) -> Vec<Complex32> {
    f.chunks_exact(2)
        .map(|c| Complex32::new(c[0], c[1]))
        .collect()
}

struct Case {
    log_n: u32,
    batch: usize,
    dir: Direction,
    normalize: bool,
    /// Relative-error gate. The radix-2 GPU kernels share the radix-2 CPU
    /// reference's summation order so they match to ~1e-4; the N=4096 radix-8
    /// kernel uses a different order and diverges at the fp32 4th digit, so its
    /// gate is looser (the algorithm is proven exact-to-1e-3 on CPU by
    /// `cpu_radix8::tests::radix8_matches_radix2_reference`).
    rel_tol: f32,
    label: &'static str,
}

const CASES: &[Case] = &[
    Case { log_n: 2,  batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=4 fwd" },
    Case { log_n: 3,  batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=8 fwd" },
    Case { log_n: 6,  batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=64 fwd" },
    Case { log_n: 8,  batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=256 fwd" },
    Case { log_n: 10, batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=1024 fwd" },
    Case { log_n: 12, batch: 1, dir: Direction::Forward, normalize: false, rel_tol: 5e-4, label: "N=4096 fwd" },
    Case { log_n: 8,  batch: 8, dir: Direction::Forward, normalize: false, rel_tol: 1e-4, label: "N=256 fwd batch=8" },
    Case { log_n: 8,  batch: 1, dir: Direction::Inverse, normalize: true,  rel_tol: 1e-4, label: "N=256 inv normalize" },
];

fn run_case(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    case: &Case,
) -> Result<()> {
    let plan = Plan::new(case.log_n, case.batch, case.normalize);
    let n = plan.n();
    let total = n * case.batch;
    let input: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new(((i % n) as f32).sin(), ((i % n) as f32).cos()))
        .collect();

    // CPU reference (acts on a copy).
    let mut cpu = input.clone();
    plan.cpu_execute(&mut cpu, case.dir);

    // GPU input: for inverse, conjugate on host so the forward-only kernel
    // produces the inverse DFT (up to the final conjugate + 1/N scale).
    let mut input_for_gpu = input.clone();
    if case.dir == Direction::Inverse {
        for c in input_for_gpu.iter_mut() {
            *c = c.conj();
        }
    }

    let stream = ctx.default_stream();
    let input_flat = flatten(&input_for_gpu);
    // Twiddles match the plan's specialised kernel (radix-8 for N=4096,
    // radix-2 otherwise). Inverse uses the conjugate trick on the host, so
    // the forward-only kernel choice depends only on log_n.
    let twiddles_flat = flatten(&plan.kernel_twiddles());
    let dbuf_in = DeviceBuffer::from_host(&stream, &input_flat)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    match plan.kernel_kind {
        KernelKind::Specialised4096 => {
            // One block per lane, 512 threads, 4 radix-8 Stockham stages.
            let cfg = LaunchConfig {
                grid_dim: (case.batch as u32, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_4096(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        KernelKind::Specialised1024 => {
            // One block per lane, 256 threads, 5 radix-4 Stockham stages.
            let cfg = LaunchConfig {
                grid_dim: (case.batch as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_1024(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        KernelKind::Specialised256 => {
            // One block per lane, 64 threads, 4 radix-4 Stockham stages.
            let cfg = LaunchConfig {
                grid_dim: (case.batch as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_256(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        _ => {
            // CUDA caps block_dim at 1024 threads per block; one thread per
            // butterfly (min(N/2, 1024)).
            let block_threads = core::cmp::min(n / 2, 1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (case.batch as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_radix2_c2c_pow2_1d_fallback(
                stream.as_ref(),
                cfg,
                &dbuf_in,
                &dbuf_tw,
                &mut dbuf_out,
                case.log_n,
            )?;
        }
    }

    let mut host_out_flat = dbuf_out.to_host_vec(&stream)?;

    // Inverse: conjugate output, then scale by 1/N if normalize.
    if case.dir == Direction::Inverse {
        // host_out_flat is [re0, im0, re1, im1, ...]; conjugate by negating
        // every odd index.
        for chunk in host_out_flat.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
        if case.normalize {
            let inv_n = 1.0f32 / n as f32;
            for v in host_out_flat.iter_mut() {
                *v *= inv_n;
            }
        }
    }

    let host_out = unflatten(&host_out_flat);
    for i in 0..total {
        let g = host_out[i];
        let c = cpu[i];
        let dre = (g.re - c.re).abs();
        let dim = (g.im - c.im).abs();
        let norm = c.re.abs() + c.im.abs() + 1.0;
        if (dre + dim) / norm > case.rel_tol {
            return Err(anyhow!(
                "{}: mismatch at {i}: gpu=({}, {}), cpu=({}, {}), |dre|+|dim|/norm={}",
                case.label,
                g.re,
                g.im,
                c.re,
                c.im,
                (dre + dim) / norm,
            ));
        }
    }
    println!("ok  {} (N={}, batch={})", case.label, n, case.batch);
    Ok(())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let mut fails = 0usize;
    for case in CASES {
        if let Err(e) = run_case(&ctx, &module, case) {
            eprintln!("FAIL {}: {e}", case.label);
            fails += 1;
        }
    }
    if fails > 0 {
        anyhow::bail!("{} of {} cases failed", fails, CASES.len());
    }
    println!("\nfft-1d-c2c: {}/{} cases verified", CASES.len(), CASES.len());
    Ok(())
}
