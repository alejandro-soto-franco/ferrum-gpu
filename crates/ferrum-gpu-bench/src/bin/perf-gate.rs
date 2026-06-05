//! Asserts ferrum_event_us <= 0.9 * cufft_event_us on all three target sizes.
//! Exits 0 on pass, 1 on miss with a diagnostic summary.
//!
//! Designed to be wrapped by `make perf-gate` (which runs it under
//! `tools/bench-gpu-lock.sh`). Re-uses `alternating_bench` from the bench
//! lib; the `#[cuda_module]` block is duplicated here (kernel body shared
//! via `include!`) because cuda-oxide PTX embedding is keyed to the binary.

use anyhow::Result;

use ferrum_gpu_bench::{
    BATCH, alternating_bench, fallback_launch_cfg, init_cuda_contexts, spec256_launch_cfg,
    spec1024_launch_cfg, spec4096_launch_cfg,
};
use ferrum_gpu_fft::KernelKind;

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

const TARGETS: &[u32] = &[8, 10, 12];
const GATE_RATIO: f64 = 0.9;

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    let module = kernels::load(&core_ctx)?;
    let core_stream = core_ctx.default_stream();

    let mut misses: Vec<(usize, f64)> = Vec::new();
    println!(
        "perf-gate (gate: ferrum_event_us <= {} * cufft_event_us)",
        GATE_RATIO
    );
    println!(
        "{:<8} {:<12} {:<12} {:<8}",
        "N", "ferrum_us", "cufft_us", "ratio"
    );
    for &log_n in TARGETS {
        let kind = KernelKind::for_forward_pow2(log_n);
        let fallback_cfg = fallback_launch_cfg(log_n);
        let spec4096_cfg = spec4096_launch_cfg();
        let spec1024_cfg = spec1024_launch_cfg();
        let spec256_cfg = spec256_launch_cfg();
        let launch_ferrum = |dbuf_in: &cuda_core::DeviceBuffer<f32>,
                             dbuf_tw: &cuda_core::DeviceBuffer<f32>,
                             dbuf_out: &mut cuda_core::DeviceBuffer<f32>|
         -> Result<()> {
            match kind {
                KernelKind::Specialised4096 => {
                    module.fft_c2c_4096(
                        core_stream.as_ref(),
                        spec4096_cfg,
                        dbuf_in,
                        dbuf_tw,
                        dbuf_out,
                    )?;
                }
                KernelKind::Specialised1024 => {
                    module.fft_c2c_1024(
                        core_stream.as_ref(),
                        spec1024_cfg,
                        dbuf_in,
                        dbuf_tw,
                        dbuf_out,
                    )?;
                }
                KernelKind::Specialised256 => {
                    module.fft_c2c_256(
                        core_stream.as_ref(),
                        spec256_cfg,
                        dbuf_in,
                        dbuf_tw,
                        dbuf_out,
                    )?;
                }
                _ => {
                    module.fft_radix2_c2c_pow2_1d_fallback(
                        core_stream.as_ref(),
                        fallback_cfg,
                        dbuf_in,
                        dbuf_tw,
                        dbuf_out,
                        log_n,
                    )?;
                }
            }
            Ok(())
        };
        let (ferr, cu) = alternating_bench(&core_ctx, &cudarc_ctx, log_n, launch_ferrum)?;
        let fe = ferr.event_med_s * 1.0e6 / (BATCH as f64);
        let cu_us = cu.event_med_s * 1.0e6 / (BATCH as f64);
        let ratio = fe / cu_us;
        let pass = ratio <= GATE_RATIO;
        let n = 1usize << log_n;
        println!(
            "{:<8} {:<12.3} {:<12.3} {:<8.3} {}",
            n,
            fe,
            cu_us,
            ratio,
            if pass { "PASS" } else { "MISS" }
        );
        if !pass {
            misses.push((n, ratio));
        }
    }
    if misses.is_empty() {
        println!("\nperf-gate: PASS on all sizes");
        Ok(())
    } else {
        eprintln!("\nperf-gate: MISS on:");
        for (n, r) in &misses {
            eprintln!("  N = {n}: ratio = {:.3} (need <= {})", r, GATE_RATIO);
        }
        std::process::exit(1);
    }
}
