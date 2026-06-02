//! Batch sweep at N=256: warp, radix-4, and radix-4+float4-IO (v4) vs cuFFT,
//! across batch sizes. At large batch the GPU saturates and the bottleneck
//! flips to memory bandwidth; vectorized (float4) global IO is the lever to
//! beat cuFFT (scalar loads were 18.78% sector-efficient at batch=32k).
//!
//! Run under `tools/bench-gpu-lock.sh`. Run: `cargo oxide run ferrum-gpu-bench --bin batch-sweep`
//!   env: WARP_BLOCK (default 256), BATCHES (default "4096,16384,65536,262144").

use std::f32::consts::PI;

use anyhow::{Result, anyhow};
use cuda_core::{DeviceBuffer, LaunchConfig};
use ferrum_gpu_bench::{alternating_bench_batch, init_cuda_contexts};
use ferrum_gpu_fft::{Complex32, Direction, Plan};

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");
include!("../../../ferrum-gpu-fft-kernels/src/fft256_warp_body.rs");

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    // fft_c2c_256_r16s now lives in `mod kernels` (shipped via the wheel);
    // exercise that exact kernel here rather than a duplicate standalone copy.
    let mod_k = kernels::load(&core_ctx)?;
    let mod_w = fft256_warp::load(&core_ctx)?;
    let mod_r = &mod_k;
    let stream = core_ctx.default_stream();

    let wblock: u32 = std::env::var("WARP_BLOCK").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let batches: Vec<usize> = std::env::var("BATCHES")
        .unwrap_or_else(|_| "4096,16384,65536,262144".into())
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();

    let w32: Vec<f32> = (0..16).flat_map(|e| { let t = -2.0*PI*e as f32/32.0; [t.cos(), t.sin()] }).collect();
    let w256: Vec<f32> = (0..256).flat_map(|e| { let t = -2.0*PI*e as f32/256.0; [t.cos(), t.sin()] }).collect();
    let d_w32 = DeviceBuffer::from_host(&stream, &w32)?;
    let d_w256 = DeviceBuffer::from_host(&stream, &w256)?;
    // --- r16s correctness vs radix-2 (batch 4 so we exercise multiple blocks). ---
    {
        let bb = 4usize;
        let n = 256usize;
        let input: Vec<Complex32> = (0..bb * n)
            .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.021).cos())).collect();
        let flat: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        let d_in = DeviceBuffer::from_host(&stream, &flat)?;
        let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, bb * n * 2)?;
        let cfg = LaunchConfig { grid_dim: (bb as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        mod_r.fft_c2c_256_r16s(stream.as_ref(), cfg, &d_in, &d_w256, &mut d_out)?;
        let gpu = d_out.to_host_vec(&stream)?;
        let mut worst = 0.0f32;
        for f in 0..bb {
            let mut r = input[f*n..(f+1)*n].to_vec();
            Plan::new(8, 1, false).cpu_execute(&mut r, Direction::Forward);
            for k in 0..n {
                let e = ((gpu[(f*n+k)*2]-r[k].re).powi(2)+(gpu[(f*n+k)*2+1]-r[k].im).powi(2)).sqrt();
                let s = (r[k].re.powi(2)+r[k].im.powi(2)).sqrt().max(1.0);
                worst = worst.max(e/s);
            }
        }
        println!("fft_c2c_256_r16s vs radix-2: max_rel_err = {worst:.2e} -> {}",
            if worst <= 1e-3 { "PASS" } else { "FAIL" });
        if worst > 1e-3 { return Err(anyhow!("r16s correctness FAIL")); }
    }

    println!("N=256 batch sweep  (ratio = ferrum/cufft, <1.0 BEATS cuFFT)  warp K={}", wblock/32);
    println!("{:<9} {:<9} {:<9} {:<9} {:<10} {:<12}", "batch", "warp", "radix4", "r16s", "cufft_us", "best");

    for &b in &batches {
        let wcfg = LaunchConfig { grid_dim: ((b as u32*32)/wblock,1,1), block_dim: (wblock,1,1), shared_mem_bytes: 0 };
        let warp = |din: &DeviceBuffer<f32>, _t: &DeviceBuffer<f32>, dout: &mut DeviceBuffer<f32>| -> Result<()> {
            mod_w.fft_c2c_256_warp(stream.as_ref(), wcfg, din, &d_w32, &d_w256, dout)?; Ok(()) };
        let (fw, cu) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 8, b, warp)?;
        let warp_r = (fw.event_med_s*1e6/b as f64) / (cu.event_med_s*1e6/b as f64);
        let cu_us = cu.event_med_s*1e6/b as f64;

        let rcfg = LaunchConfig { grid_dim: (b as u32,1,1), block_dim: (64,1,1), shared_mem_bytes: 0 };
        let r4 = |din: &DeviceBuffer<f32>, t: &DeviceBuffer<f32>, dout: &mut DeviceBuffer<f32>| -> Result<()> {
            mod_k.fft_c2c_256(stream.as_ref(), rcfg, din, t, dout)?; Ok(()) };
        let (fr, _) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 8, b, r4)?;
        let r4_r = (fr.event_med_s*1e6/b as f64) / cu_us;

        let rscfg = LaunchConfig { grid_dim: (b as u32,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 };
        let r16s = |din: &DeviceBuffer<f32>, _t: &DeviceBuffer<f32>, dout: &mut DeviceBuffer<f32>| -> Result<()> {
            mod_r.fft_c2c_256_r16s(stream.as_ref(), rscfg, din, &d_w256, dout)?; Ok(()) };
        let (fv, _) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 8, b, r16s)?;
        let r16s_r = (fv.event_med_s*1e6/b as f64) / cu_us;

        let best = warp_r.min(r4_r).min(r16s_r);
        println!("{:<9} {:<9.3} {:<9.3} {:<9.3} {:<10.4} {:<12}",
            b, warp_r, r4_r, r16s_r, cu_us,
            if best < 1.0 { format!("{:.3} WIN", best) } else { format!("{:.3}", best) });
    }
    Ok(())
}
