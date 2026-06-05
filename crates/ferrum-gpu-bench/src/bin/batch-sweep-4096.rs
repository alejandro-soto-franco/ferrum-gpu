//! N=4096 batch sweep: radix-16 (u64 IO) vs the existing radix-8 vs cuFFT.
//! Tests whether the N=256 recipe (higher radix + u64 coalescing) generalises
//! to the size where ferrum was WORST (radix-8, ~3.6x at large batch).
//! Run under `tools/bench-gpu-lock.sh`. env: BATCHES (default "1024,4096,16384,65536").

use anyhow::{Result, anyhow};
use cuda_core::{DeviceBuffer, LaunchConfig};
use ferrum_gpu_bench::{alternating_bench_batch, init_cuda_contexts};
use ferrum_gpu_fft::{Complex32, Direction, Plan};
use std::f32::consts::PI;

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    // fft_c2c_4096_r16s now lives in `mod kernels` (shipped via the wheel);
    // exercise that exact kernel here rather than a duplicate standalone copy.
    let mod_k = kernels::load(&core_ctx)?;
    let mod_r = &mod_k;
    let stream = core_ctx.default_stream();
    let batches: Vec<usize> = std::env::var("BATCHES")
        .unwrap_or_else(|_| "1024,4096,16384,65536".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let w4096: Vec<f32> = (0..4096)
        .flat_map(|e| {
            let t = -2.0 * PI * e as f32 / 4096.0;
            [t.cos(), t.sin()]
        })
        .collect();
    let d_w4096 = DeviceBuffer::from_host(&stream, &w4096)?;

    // r16s-4096 correctness vs radix-2. Use a multi-block batch (512) so
    // co-resident blocks on each SM exercise the in-place shared-buffer
    // gather/scatter: the kernel's intra-stage barrier is what keeps that
    // race-free, and a batch of 2 was too small to expose a missing one.
    {
        let bb = 512usize;
        let n = 4096usize;
        let input: Vec<Complex32> = (0..bb * n)
            .map(|i| Complex32::new((i as f32 * 0.0007).sin(), (i as f32 * 0.0011).cos()))
            .collect();
        let flat: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        let d_in = DeviceBuffer::from_host(&stream, &flat)?;
        let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, bb * n * 2)?;
        let cfg = LaunchConfig {
            grid_dim: (bb as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        mod_r.fft_c2c_4096_r16s(stream.as_ref(), cfg, &d_in, &d_w4096, &mut d_out)?;
        let gpu = d_out.to_host_vec(&stream)?;
        let mut worst = 0.0f32;
        for f in 0..bb {
            let mut r = input[f * n..(f + 1) * n].to_vec();
            Plan::new(12, 1, false).cpu_execute(&mut r, Direction::Forward);
            for k in 0..n {
                let e = ((gpu[(f * n + k) * 2] - r[k].re).powi(2)
                    + (gpu[(f * n + k) * 2 + 1] - r[k].im).powi(2))
                .sqrt();
                let s = (r[k].re.powi(2) + r[k].im.powi(2)).sqrt().max(1.0);
                worst = worst.max(e / s);
            }
        }
        println!(
            "fft_c2c_4096_r16s vs radix-2: max_rel_err = {worst:.2e} -> {}",
            if worst <= 1e-3 { "PASS" } else { "FAIL" }
        );
        if worst > 1e-3 {
            return Err(anyhow!("r16s-4096 correctness FAIL"));
        }
    }

    // Same many-block correctness gate for the register four-step kernel.
    {
        let bb = 512usize;
        let n = 4096usize;
        let input: Vec<Complex32> = (0..bb * n)
            .map(|i| Complex32::new((i as f32 * 0.0007).sin(), (i as f32 * 0.0011).cos()))
            .collect();
        let flat: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        let d_in = DeviceBuffer::from_host(&stream, &flat)?;
        let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, bb * n * 2)?;
        let cfg = LaunchConfig {
            grid_dim: (bb as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        mod_r.fft_c2c_4096_4step_reg(stream.as_ref(), cfg, &d_in, &d_w4096, &mut d_out)?;
        let gpu = d_out.to_host_vec(&stream)?;
        // Optional accuracy dump (ACC_DUMP=path): first FFT's input + four-step
        // output, scored offline against an f64 reference. Rows:
        // size,index,in_re,in_im,kernel_re,kernel_im.
        if let Ok(path) = std::env::var("ACC_DUMP") {
            use std::io::Write;
            let mut fh = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            for k in 0..n {
                writeln!(
                    fh,
                    "{n},{k},{:.9},{:.9},{:.9},{:.9}",
                    flat[2 * k],
                    flat[2 * k + 1],
                    gpu[2 * k],
                    gpu[2 * k + 1]
                )?;
            }
        }
        let mut worst = 0.0f32;
        for f in 0..bb {
            let mut r = input[f * n..(f + 1) * n].to_vec();
            Plan::new(12, 1, false).cpu_execute(&mut r, Direction::Forward);
            for k in 0..n {
                let e = ((gpu[(f * n + k) * 2] - r[k].re).powi(2)
                    + (gpu[(f * n + k) * 2 + 1] - r[k].im).powi(2))
                .sqrt();
                let s = (r[k].re.powi(2) + r[k].im.powi(2)).sqrt().max(1.0);
                worst = worst.max(e / s);
            }
        }
        println!(
            "fft_c2c_4096_4step_reg vs radix-2: max_rel_err = {worst:.2e} -> {}",
            if worst <= 1e-3 { "PASS" } else { "FAIL" }
        );
        if worst > 1e-3 {
            return Err(anyhow!("4step_reg-4096 correctness FAIL"));
        }
    }

    println!("N=4096 batch sweep (ratio=ferrum/cufft, <1.0 BEATS cuFFT)");
    println!(
        "{:<9} {:<9} {:<9} {:<9} {:<10} {:<12}",
        "batch", "radix8", "r16s", "4step", "cufft_us", "best"
    );
    for &b in &batches {
        let cfg8 = LaunchConfig {
            grid_dim: (b as u32, 1, 1),
            block_dim: (512, 1, 1),
            shared_mem_bytes: 0,
        };
        let r8 = |din: &DeviceBuffer<f32>,
                  t: &DeviceBuffer<f32>,
                  dout: &mut DeviceBuffer<f32>|
         -> Result<()> {
            mod_k.fft_c2c_4096(stream.as_ref(), cfg8, din, t, dout)?;
            Ok(())
        };
        let (f8, cu) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 12, b, r8)?;
        let cu_us = cu.event_med_s * 1e6 / b as f64;
        let r8_r = (f8.event_med_s * 1e6 / b as f64) / cu_us;

        let cfgr = LaunchConfig {
            grid_dim: (b as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rr = |din: &DeviceBuffer<f32>,
                  _t: &DeviceBuffer<f32>,
                  dout: &mut DeviceBuffer<f32>|
         -> Result<()> {
            mod_r.fft_c2c_4096_r16s(stream.as_ref(), cfgr, din, &d_w4096, dout)?;
            Ok(())
        };
        let (fr, _) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 12, b, rr)?;
        let r16_r = (fr.event_med_s * 1e6 / b as f64) / cu_us;

        let rs = |din: &DeviceBuffer<f32>,
                  _t: &DeviceBuffer<f32>,
                  dout: &mut DeviceBuffer<f32>|
         -> Result<()> {
            mod_r.fft_c2c_4096_4step_reg(stream.as_ref(), cfgr, din, &d_w4096, dout)?;
            Ok(())
        };
        let (fs, _) = alternating_bench_batch(&core_ctx, &cudarc_ctx, 12, b, rs)?;
        let r4s_r = (fs.event_med_s * 1e6 / b as f64) / cu_us;

        let best = r8_r.min(r16_r).min(r4s_r);
        println!(
            "{:<9} {:<9.3} {:<9.3} {:<9.3} {:<10.4} {:<12}",
            b,
            r8_r,
            r16_r,
            r4s_r,
            cu_us,
            if best < 1.0 {
                format!("{:.3} WIN", best)
            } else {
                format!("{:.3}", best)
            }
        );
    }
    Ok(())
}
