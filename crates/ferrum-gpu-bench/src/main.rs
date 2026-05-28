//! cuFFT-comparison benchmark for ferrum-gpu-fft.
//!
//! Prints a per-size timing table at N in {256, 1024, 4096}, batch = 256,
//! comparing the in-tree cuda-oxide Stockham radix-2 power-of-2 C2C kernel
//! against cuFFT. Pass `--gate` to additionally fail (exit 1) on any size
//! where `ferrum_event_us > 0.9 * cufft_event_us`. For a dedicated gate
//! binary, use `perf-gate` instead; this binary is for human inspection of
//! the full table.

use anyhow::Result;

use ferrum_gpu_bench::{
    BATCH, LOG_NS, TRIALS, WARMUP, alternating_bench, fallback_launch_cfg, init_cuda_contexts,
};

include!("../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    let module = kernels::load(&core_ctx)?;
    let core_stream = core_ctx.default_stream();

    let gate_mode = std::env::args().any(|a| a == "--gate");
    let gate_ratio = 0.9_f64;
    let mut misses: Vec<(usize, f64)> = Vec::new();

    println!(
        "ferrum-gpu-bench: 1D radix-2 C2C FFT, batch={}, warmup={}, trials={}",
        BATCH, WARMUP, TRIALS
    );
    println!(
        "{:<8} {:<11} {:<11} {:<11} {:<11} {:<9}",
        "N", "fe_ev_us", "cu_ev_us", "fe_wl_us", "cu_wl_us", "sp_event"
    );
    for &log_n in LOG_NS {
        let n = 1usize << log_n;
        let cfg = fallback_launch_cfg(log_n);
        let launch_ferrum = |dbuf_in: &cuda_core::DeviceBuffer<f32>,
                             dbuf_tw: &cuda_core::DeviceBuffer<f32>,
                             dbuf_out: &mut cuda_core::DeviceBuffer<f32>|
         -> Result<()> {
            module.fft_radix2_c2c_pow2_1d_fallback(
                core_stream.as_ref(),
                cfg,
                dbuf_in,
                dbuf_tw,
                dbuf_out,
                log_n,
            )?;
            Ok(())
        };
        let (ferr, cu) = alternating_bench(&core_ctx, &cudarc_ctx, log_n, launch_ferrum)?;
        let fe_ev_us = ferr.event_med_s * 1.0e6 / (BATCH as f64);
        let cu_ev_us = cu.event_med_s * 1.0e6 / (BATCH as f64);
        let fe_wl_us = ferr.wall_mean_s * 1.0e6 / (BATCH as f64);
        let cu_wl_us = cu.wall_mean_s * 1.0e6 / (BATCH as f64);
        let sp_event = if fe_ev_us > 0.0 { cu_ev_us / fe_ev_us } else { f64::NAN };
        println!(
            "{:<8} {:<11.3} {:<11.3} {:<11.3} {:<11.3} {:<9.2}",
            n, fe_ev_us, cu_ev_us, fe_wl_us, cu_wl_us, sp_event
        );
        if gate_mode {
            let event_ratio = fe_ev_us / cu_ev_us;
            if event_ratio > gate_ratio {
                misses.push((n, event_ratio));
            }
        }
    }
    if gate_mode {
        if misses.is_empty() {
            println!(
                "\nperf-gate: PASS (ferrum_event_us <= {} * cufft_event_us on all sizes)",
                gate_ratio
            );
        } else {
            eprintln!("\nperf-gate: MISS on:");
            for (n, r) in &misses {
                eprintln!("  N = {n}: ratio = {:.3} (need <= {})", r, gate_ratio);
            }
            std::process::exit(1);
        }
    }
    Ok(())
}
