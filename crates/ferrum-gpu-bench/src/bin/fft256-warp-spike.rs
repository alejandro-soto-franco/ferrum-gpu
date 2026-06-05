//! Spike: N=256 warp-per-FFT kernel through cuda-oxide (Task: beat cuFFT @256).
//!
//! Cross-checks `fft_c2c_256_warp` against (a) the CPU layout oracle
//! `ferrum_gpu_fft::warp_fft::warp256_model` and (b) the radix-2 ground-truth
//! reference (`Plan::cpu_execute`), then reports rough event-timed throughput.
//! The authoritative head-to-head vs cuFFT is `perf-gate` (P5 integration).
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin fft256-warp-spike`.

use std::f32::consts::PI;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrum_gpu_fft::warp_fft::warp256_model;
use ferrum_gpu_fft::{Complex32, Direction, Plan};

include!("../../../ferrum-gpu-fft-kernels/src/fft256_warp_body.rs");

const N: usize = 256;
const BATCH: usize = 256;
const BLOCK: u32 = 256; // 8 warps/block = 8 FFTs/block (K=8 start point)
const WARMUP: usize = 20;
const TRIALS: usize = 100;

fn flatten(c: &[Complex32]) -> Vec<f32> {
    c.iter().flat_map(|x| [x.re, x.im]).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("ctx: {e}"))?;
    let module = fft256_warp::load(&ctx)?;
    let stream = ctx.default_stream();

    // Twiddle tables. w32: W_32^e, e in 0..16. w256: W_256^e, e in 0..256.
    let w32: Vec<f32> = (0..16)
        .flat_map(|e| {
            let th = -2.0 * PI * e as f32 / 32.0;
            [th.cos(), th.sin()]
        })
        .collect();
    let w256: Vec<f32> = (0..256)
        .flat_map(|e| {
            let th = -2.0 * PI * e as f32 / 256.0;
            [th.cos(), th.sin()]
        })
        .collect();

    // Input: BATCH independent 256-pt signals.
    let input: Vec<Complex32> = (0..BATCH * N)
        .map(|idx| {
            let n = (idx % N) as f32;
            let f = (idx / N) as f32;
            Complex32::new((n * 0.013 + f * 0.001).sin(), (n * 0.021).cos())
        })
        .collect();

    let d_in = DeviceBuffer::from_host(&stream, &flatten(&input))?;
    let d_w32 = DeviceBuffer::from_host(&stream, &w32)?;
    let d_w256 = DeviceBuffer::from_host(&stream, &w256)?;
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, BATCH * N * 2)?;

    let grid = (BATCH as u32 * 32) / BLOCK; // one warp per FFT
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    module.fft_c2c_256_warp(stream.as_ref(), cfg, &d_in, &d_w32, &d_w256, &mut d_out)?;
    let gpu = d_out.to_host_vec(&stream)?;

    // Correctness vs BOTH oracles, over all BATCH FFTs.
    let mut worst_model = 0.0f32;
    let mut worst_ref = 0.0f32;
    for f in 0..BATCH {
        let sig: Vec<Complex32> = input[f * N..(f + 1) * N].to_vec();

        let model = warp256_model(&sig);
        let mut reference = sig.clone();
        Plan::new(8, 1, false).cpu_execute(&mut reference, Direction::Forward);

        for k in 0..N {
            let gr = gpu[(f * N + k) * 2];
            let gi = gpu[(f * N + k) * 2 + 1];
            let em = ((gr - model[k].re).powi(2) + (gi - model[k].im).powi(2)).sqrt();
            let sm = (model[k].re * model[k].re + model[k].im * model[k].im)
                .sqrt()
                .max(1.0);
            worst_model = worst_model.max(em / sm);
            let er = ((gr - reference[k].re).powi(2) + (gi - reference[k].im).powi(2)).sqrt();
            let sr = (reference[k].re * reference[k].re + reference[k].im * reference[k].im)
                .sqrt()
                .max(1.0);
            worst_ref = worst_ref.max(er / sr);
        }
    }
    let ok = worst_ref <= 1e-3;
    println!("fft_c2c_256_warp vs CPU model:  max_rel_err = {worst_model:.2e}");
    println!(
        "fft_c2c_256_warp vs radix-2 ref: max_rel_err = {worst_ref:.2e} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        return Err(anyhow!("correctness FAIL"));
    }

    // Rough event-timed throughput (NOT the cuFFT head-to-head; see perf-gate).
    let flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    for _ in 0..WARMUP {
        module.fft_c2c_256_warp(stream.as_ref(), cfg, &d_in, &d_w32, &d_w256, &mut d_out)?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let s = ctx.new_event(flag)?;
        let e = ctx.new_event(flag)?;
        s.record(&stream)?;
        module.fft_c2c_256_warp(stream.as_ref(), cfg, &d_in, &d_w32, &d_w256, &mut d_out)?;
        e.record(&stream)?;
        e.synchronize()?;
        samples.push(s.elapsed_ms(&e)? as f64 * 1.0e-3);
    }
    let med_s = median(samples);
    let per_fft_us = med_s * 1.0e6 / BATCH as f64;
    let gpts = (BATCH * N) as f64 / med_s / 1.0e9;
    println!(
        "fft_c2c_256_warp: {BATCH} FFTs of {N} (K={} warps/block) in {:.2} us -> {:.5} us/FFT, {:.1} Gpt/s",
        BLOCK / 32,
        med_s * 1.0e6,
        per_fft_us,
        gpts
    );
    Ok(())
}
