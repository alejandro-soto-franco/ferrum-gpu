//! Spike: four-step (64x64) warp-shuffle FFT for N=4096 vs cuFFT.
//!
//! Correctness vs the numpy-validated radix-2 CPU reference (`Plan::cpu_execute`),
//! then timing vs cuFFT (alternating CUDA-event scheme). This is the redesign
//! aiming to close the 3.4x gap the shared-memory radix-8 kernel left.
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin four-step-spike`.

use std::f32::consts::PI;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext as CudarcContext;
use ferrum_gpu_fft::{Complex32, Direction, Plan};

include!("../../../ferrum-gpu-fft-kernels/src/four_step_body.rs");

const N: usize = 4096;
const WARMUP: usize = 10;
const TRIALS: usize = 100;

fn flatten(c: &[Complex32]) -> Vec<f32> {
    c.iter().flat_map(|x| [x.re, x.im]).collect()
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn w_table(n: usize, count: usize) -> Vec<f32> {
    (0..count)
        .flat_map(|e| {
            let th = -2.0 * PI * e as f32 / n as f32;
            [th.cos(), th.sin()]
        })
        .collect()
}

fn launch(
    module: &four_step::LoadedModule,
    stream: &cuda_core::CudaStream,
    batch: usize,
    d_in: &DeviceBuffer<f32>,
    d_w64: &DeviceBuffer<f32>,
    d_w4096: &DeviceBuffer<f32>,
    d_out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let block: u32 = std::env::var("FOURSTEP_BLOCK").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let cfg = LaunchConfig { grid_dim: (batch as u32, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };
    module.fft_c2c_4096_4step(stream, cfg, d_in, d_w64, d_w4096, d_out)?;
    Ok(())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("ctx: {e}"))?;
    let module = four_step::load(&ctx)?;
    let stream = ctx.default_stream();

    let w64 = w_table(64, 32);
    let w4096 = w_table(4096, 4096);
    let d_w64 = DeviceBuffer::from_host(&stream, &w64)?;
    let d_w4096 = DeviceBuffer::from_host(&stream, &w4096)?;

    // --- Correctness at batch=4 vs radix-2 CPU reference ---
    {
        let batch = 4;
        let input: Vec<Complex32> = (0..N * batch)
            .map(|i| {
                let t = (i % N) as f32;
                Complex32::new((t * 0.013).sin(), (t * 0.021).cos())
            })
            .collect();
        let mut cpu = input.clone();
        Plan::new(12, batch, false).cpu_execute(&mut cpu, Direction::Forward);

        let d_in = DeviceBuffer::from_host(&stream, &flatten(&input))?;
        let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, N * batch * 2)?;
        launch(&module, stream.as_ref(), batch, &d_in, &d_w64, &d_w4096, &mut d_out)?;
        let gpu = d_out.to_host_vec(&stream)?;

        let (mut worst, mut wi) = (0.0f32, 0usize);
        for i in 0..N * batch {
            let (gr, gi) = (gpu[2 * i], gpu[2 * i + 1]);
            let c = cpu[i];
            let err = ((gr - c.re).powi(2) + (gi - c.im).powi(2)).sqrt();
            let scale = (c.re * c.re + c.im * c.im).sqrt().max(1.0);
            if err / scale > worst {
                worst = err / scale;
                wi = i;
            }
        }
        let ok = worst <= 1e-3;
        println!("fft_c2c_4096_4step vs CPU (batch={batch}): max_rel_err={worst:.2e} at {wi} -> {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            let c = cpu[wi];
            eprintln!("  worst {wi}: cpu=({:.5},{:.5}) gpu=({:.5},{:.5})", c.re, c.im, gpu[2 * wi], gpu[2 * wi + 1]);
            return Err(anyhow!("correctness FAIL"));
        }
    }

    // --- Timing (batch from FOURSTEP_BATCH, default 256), alternating vs cuFFT ---
    let batch: usize = std::env::var("FOURSTEP_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let total = N * batch;
    let input: Vec<f32> = (0..total * 2).map(|i| (i as f32 * 0.001).sin()).collect();
    let d_in = DeviceBuffer::from_host(&stream, &input)?;
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    let cudarc_ctx = CudarcContext::new(0).map_err(|e| anyhow!("cudarc: {e}"))?;
    let cu_stream = cudarc_ctx.default_stream();
    let cu_in: Vec<cufft_sys::float2> = (0..total)
        .map(|i| cufft_sys::float2 { x: (2.0 * i as f32 * 0.001).sin(), y: 0.0 })
        .collect();
    let mut c_in = cu_stream.clone_htod(&cu_in).map_err(|e| anyhow!("htod: {e}"))?;
    let mut c_out = cu_stream.alloc_zeros::<cufft_sys::float2>(total).map_err(|e| anyhow!("alloc: {e}"))?;
    let cu_plan = CudaFft::plan_1d(N as i32, cufft_sys::cufftType::CUFFT_C2C, batch as i32, cu_stream.clone())
        .map_err(|e| anyhow!("plan: {e:?}"))?;

    for _ in 0..WARMUP {
        launch(&module, stream.as_ref(), batch, &d_in, &d_w64, &d_w4096, &mut d_out)?;
        cu_plan.exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward).map_err(|e| anyhow!("exec: {e:?}"))?;
    }
    stream.synchronize()?;
    cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    let flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let cflag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let mut ours = Vec::with_capacity(TRIALS);
    let mut cu = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let s = ctx.new_event(flag)?;
        let e = ctx.new_event(flag)?;
        s.record(&stream)?;
        launch(&module, stream.as_ref(), batch, &d_in, &d_w64, &d_w4096, &mut d_out)?;
        e.record(&stream)?;
        e.synchronize()?;
        ours.push(s.elapsed_ms(&e)? as f64 * 1.0e-3);

        let cs = cudarc_ctx.new_event(cflag).map_err(|e| anyhow!("ev: {e}"))?;
        let ce = cudarc_ctx.new_event(cflag).map_err(|e| anyhow!("ev: {e}"))?;
        cs.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        cu_plan.exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward).map_err(|e| anyhow!("exec: {e:?}"))?;
        ce.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        ce.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        cu.push(cs.elapsed_ms(&ce).map_err(|e| anyhow!("el: {e}"))? as f64 * 1.0e-3);
    }
    let o_us = median(ours) * 1.0e6 / batch as f64;
    let c_us = median(cu) * 1.0e6 / batch as f64;
    println!("four-step : {o_us:.4} us/FFT");
    println!("cuFFT     : {c_us:.4} us/FFT");
    println!("ratio (ours/cuFFT): {:.3}  (gate wants <= 0.9; radix-8 was ~3.8)", o_us / c_us);
    Ok(())
}
