//! Spike: warp-shuffle 32-point FFT through cuda-oxide (Task: beat cuFFT).
//!
//! De-risks the four-step redesign by proving a warp-resident register FFT
//! (no shared memory, `shfl_xor_f32` butterflies) is correct and fast through
//! cuda-oxide's codegen. Cross-checks the GPU kernel against the CPU model
//! `ferrum_gpu_fft::warp_fft::warp_fft32_model` (itself verified vs a direct
//! DFT) and reports throughput.
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin warp-fft-spike`.

use std::f32::consts::PI;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrum_gpu_fft::Complex32;
use ferrum_gpu_fft::warp_fft::warp_fft32_model;

include!("../../../ferrum-gpu-fft-kernels/src/warp_fft_spike_body.rs");

const N: usize = 32;
const WARPS: usize = 1 << 16; // 65536 independent 32-pt FFTs
const BLOCK: u32 = 256; // 8 warps/block
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
    let module = warp_kernels::load(&ctx)?;
    let stream = ctx.default_stream();

    // W_32^e for e in 0..16.
    let tw: Vec<f32> = (0..16)
        .flat_map(|e| {
            let th = -2.0 * PI * e as f32 / 32.0;
            [th.cos(), th.sin()]
        })
        .collect();

    // Input: WARPS independent 32-pt signals.
    let input: Vec<Complex32> = (0..WARPS * N)
        .map(|i| {
            let n = (i % N) as f32;
            let w = (i / N) as f32;
            Complex32::new((n * 0.31 + w * 0.001).sin(), (n * 0.17).cos())
        })
        .collect();

    let d_in = DeviceBuffer::from_host(&stream, &flatten(&input))?;
    let d_tw = DeviceBuffer::from_host(&stream, &tw)?;
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, WARPS * N * 2)?;

    let grid = (WARPS as u32 * 32) / BLOCK; // total threads / block size
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    module.warp_fft32(stream.as_ref(), cfg, &d_in, &d_tw, &mut d_out)?;
    let gpu = d_out.to_host_vec(&stream)?;

    // Correctness: compare the first few FFTs against the CPU model.
    let mut worst = 0.0f32;
    for w in 0..8usize {
        let mut lane = [Complex32::zero(); 32];
        for n in 0..N {
            lane[n] = input[w * N + n];
        }
        warp_fft32_model(&mut lane);
        for k in 0..N {
            let (gr, gi) = (gpu[(w * N + k) * 2], gpu[(w * N + k) * 2 + 1]);
            let err = ((gr - lane[k].re).powi(2) + (gi - lane[k].im).powi(2)).sqrt();
            let scale = (lane[k].re * lane[k].re + lane[k].im * lane[k].im)
                .sqrt()
                .max(1.0);
            worst = worst.max(err / scale);
        }
    }
    let ok = worst <= 1e-4;
    println!(
        "warp_fft32 vs CPU model: max_rel_err = {worst:.2e} -> {}",
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        return Err(anyhow!("correctness FAIL"));
    }

    // Timing (CUDA events).
    let flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    for _ in 0..WARMUP {
        module.warp_fft32(stream.as_ref(), cfg, &d_in, &d_tw, &mut d_out)?;
    }
    stream.synchronize()?;
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let s = ctx.new_event(flag)?;
        let e = ctx.new_event(flag)?;
        s.record(&stream)?;
        module.warp_fft32(stream.as_ref(), cfg, &d_in, &d_tw, &mut d_out)?;
        e.record(&stream)?;
        e.synchronize()?;
        samples.push(s.elapsed_ms(&e)? as f64 * 1.0e-3);
    }
    let med_s = median(samples);
    let per_fft_us = med_s * 1.0e6 / WARPS as f64;
    let gpts = (WARPS * N) as f64 / med_s / 1.0e9;
    println!(
        "warp_fft32: {WARPS} FFTs of {N} in {:.1} us  ->  {:.5} us/FFT, {:.1} Gpt/s",
        med_s * 1.0e6,
        per_fft_us,
        gpts
    );
    Ok(())
}
