//! Interleaved vs split complex-layout FFT sweep (N x batch), cuFFT-anchored.
//!
//! For each (N in {256,1024,4096}) x batch, runs the interleaved kernel and its
//! split-layout twin on the same data, gates correctness (split reassembled ==
//! interleaved, and vs naive DFT at the smallest point), then times both plus
//! cuFFT. Emits CSV: N,batch,layout,us_per_fft,cufft_us_per_fft,gpoints_per_s.
//!
//! Buffers are capped at 2^26 complex elements per FFT-batch to fit 8 GiB; any
//! (N,batch) above that is logged as SKIP, never silently dropped.
//!
//! Usage:
//!   layout-sweep                 full sweep, CSV to stdout
//!   layout-sweep <N> <batch>     single point, launches both kernels once
//!                                (for ncu: -k "regex:fft_c2c_<N>.*(_split)?")
//!
//! Run under tools/bench-gpu-lock.sh.

use std::f32::consts::PI;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::CudaContext as CudarcContext;
use ferrum_gpu_fft::Plan;
use ferrum_gpu_fft::layout_study::{dft_naive_interleaved, interleaved_to_split};

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

const WARMUP: usize = 5;
const TRIALS: usize = 30;
const CAP: usize = 1 << 26; // max complex elements per (N*batch)

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

/// Deterministic interleaved input of `total` complex (2*total f32).
fn make_input(total: usize) -> Vec<f32> {
    (0..total * 2).map(|i| (i as f32 * 0.001).sin()).collect()
}

fn cfg_for(n: usize, batch: usize) -> LaunchConfig {
    let block = if n == 256 { 32 } else { 256 };
    LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Twiddle device buffer for the given N. 256 and 4096 use the unit-root table;
/// 1024 uses the radix-4 plan twiddles the `fft_c2c_1024` kernel expects.
fn twiddles_for(n: usize, stream: &cuda_core::CudaStream) -> Result<DeviceBuffer<f32>> {
    let host: Vec<f32> = match n {
        256 => w_table(256, 256),
        4096 => w_table(4096, 4096),
        1024 => {
            let plan = Plan::new(10, 1, false);
            plan.kernel_twiddles()
                .iter()
                .flat_map(|c| [c.re, c.im])
                .collect()
        }
        _ => return Err(anyhow!("unsupported N {n}")),
    };
    Ok(DeviceBuffer::from_host(stream, &host)?)
}

/// Launch the interleaved kernel for N.
fn launch_inter(
    m: &kernels::LoadedModule,
    s: &cuda_core::CudaStream,
    n: usize,
    cfg: LaunchConfig,
    din: &DeviceBuffer<f32>,
    tw: &DeviceBuffer<f32>,
    dout: &mut DeviceBuffer<f32>,
) -> Result<()> {
    match n {
        256 => m.fft_c2c_256_r16s(s, cfg, din, tw, dout)?,
        1024 => m.fft_c2c_1024(s, cfg, din, tw, dout)?,
        4096 => m.fft_c2c_4096_4step_reg(s, cfg, din, tw, dout)?,
        _ => return Err(anyhow!("unsupported N {n}")),
    }
    Ok(())
}

/// Launch the split kernel for N.
#[allow(clippy::too_many_arguments)]
fn launch_split(
    m: &kernels::LoadedModule,
    s: &cuda_core::CudaStream,
    n: usize,
    cfg: LaunchConfig,
    re: &DeviceBuffer<f32>,
    im: &DeviceBuffer<f32>,
    tw: &DeviceBuffer<f32>,
    ore: &mut DeviceBuffer<f32>,
    oim: &mut DeviceBuffer<f32>,
) -> Result<()> {
    match n {
        256 => m.fft_c2c_256_r16s_split(s, cfg, re, im, tw, ore, oim)?,
        1024 => m.fft_c2c_1024_split(s, cfg, re, im, tw, ore, oim)?,
        4096 => m.fft_c2c_4096_4step_reg_split(s, cfg, re, im, tw, ore, oim)?,
        _ => return Err(anyhow!("unsupported N {n}")),
    }
    Ok(())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0).map_err(|e| anyhow!("ctx: {e}"))?;
    let module = kernels::load(&ctx)?;
    let stream = ctx.default_stream();
    let flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);

    let args: Vec<String> = std::env::args().collect();
    let single: Option<(usize, usize)> = if args.len() >= 3 {
        Some((args[1].parse()?, args[2].parse()?))
    } else {
        None
    };

    let event_time = |launch: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        for _ in 0..WARMUP {
            launch()?;
        }
        stream.synchronize()?;
        let mut s = Vec::with_capacity(TRIALS);
        for _ in 0..TRIALS {
            let a = ctx.new_event(flag)?;
            let b = ctx.new_event(flag)?;
            a.record(&stream)?;
            launch()?;
            b.record(&stream)?;
            b.synchronize()?;
            s.push(a.elapsed_ms(&b)? as f64 * 1.0e-3);
        }
        Ok(median(s))
    };

    let ns = [256usize, 1024, 4096];
    let batches = [1usize, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144];

    if single.is_none() {
        println!("N,batch,layout,us_per_fft,cufft_us_per_fft,gpoints_per_s");
    }

    for &n in &ns {
        let tw = twiddles_for(n, stream.as_ref())?;
        for &batch in &batches {
            if let Some((sn, sb)) = single {
                if n != sn || batch != sb {
                    continue;
                }
            }
            let total = n * batch;
            if total > CAP {
                if single.is_none() {
                    eprintln!("SKIP N={n} batch={batch} (total {total} > cap {CAP}, 8GiB limit)");
                }
                continue;
            }

            // --- host data + split ---
            let inp = make_input(total);
            let (re_h, im_h) = interleaved_to_split(&inp);

            let d_in = DeviceBuffer::from_host(&stream, &inp)?;
            let d_re = DeviceBuffer::from_host(&stream, &re_h)?;
            let d_im = DeviceBuffer::from_host(&stream, &im_h)?;
            let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;
            let mut d_ore = DeviceBuffer::<f32>::zeroed(&stream, total)?;
            let mut d_oim = DeviceBuffer::<f32>::zeroed(&stream, total)?;
            let cfg = cfg_for(n, batch);

            // --- run both once for correctness ---
            launch_inter(&module, stream.as_ref(), n, cfg, &d_in, &tw, &mut d_out)?;
            launch_split(
                &module,
                stream.as_ref(),
                n,
                cfg,
                &d_re,
                &d_im,
                &tw,
                &mut d_ore,
                &mut d_oim,
            )?;
            stream.synchronize()?;
            let out_i = d_out.to_host_vec(&stream)?;
            let out_re = d_ore.to_host_vec(&stream)?;
            let out_im = d_oim.to_host_vec(&stream)?;

            // gate: split reassembled == interleaved
            let mut worst = 0.0f32;
            for k in 0..total {
                let dr = (out_re[k] - out_i[2 * k]).abs();
                let di = (out_im[k] - out_i[2 * k + 1]).abs();
                let scale = (out_i[2 * k].powi(2) + out_i[2 * k + 1].powi(2))
                    .sqrt()
                    .max(1.0);
                worst = worst.max((dr.max(di)) / scale);
            }
            if worst > 1e-3 {
                eprintln!("MISMATCH N={n} batch={batch} split-vs-interleaved rel={worst:.2e}");
                std::process::exit(1);
            }
            // extra oracle at the smallest N=256 point
            if n == 256 && batch == 1 {
                let oracle = dft_naive_interleaved(&inp[..2 * n]);
                let mut wo = 0.0f32;
                for k in 0..n {
                    let e = ((out_i[2 * k] - oracle[2 * k]).powi(2)
                        + (out_i[2 * k + 1] - oracle[2 * k + 1]).powi(2))
                    .sqrt();
                    let sc = (oracle[2 * k].powi(2) + oracle[2 * k + 1].powi(2))
                        .sqrt()
                        .max(1.0);
                    wo = wo.max(e / sc);
                }
                if wo > 1e-2 {
                    eprintln!("MISMATCH N=256 vs naive DFT rel={wo:.2e}");
                    std::process::exit(1);
                }
            }

            if single.is_some() {
                // single-point mode: kernels already launched once above for ncu.
                println!("single-point N={n} batch={batch} OK (rel={worst:.2e})");
                return Ok(());
            }

            // --- timing: interleaved, split ---
            let mut li = || launch_inter(&module, stream.as_ref(), n, cfg, &d_in, &tw, &mut d_out);
            let t_i = event_time(&mut li)?;
            let mut ls = || {
                launch_split(
                    &module,
                    stream.as_ref(),
                    n,
                    cfg,
                    &d_re,
                    &d_im,
                    &tw,
                    &mut d_ore,
                    &mut d_oim,
                )
            };
            let t_s = event_time(&mut ls)?;

            // --- cuFFT anchor (interleaved input) ---
            let cudarc_ctx = CudarcContext::new(0).map_err(|e| anyhow!("cudarc: {e}"))?;
            let cu_stream = cudarc_ctx.default_stream();
            let cu_in: Vec<cufft_sys::float2> = (0..total)
                .map(|i| cufft_sys::float2 {
                    x: inp[2 * i],
                    y: inp[2 * i + 1],
                })
                .collect();
            let mut c_in = cu_stream
                .clone_htod(&cu_in)
                .map_err(|e| anyhow!("htod: {e}"))?;
            let mut c_out = cu_stream
                .alloc_zeros::<cufft_sys::float2>(total)
                .map_err(|e| anyhow!("alloc: {e}"))?;
            let cu_plan = CudaFft::plan_1d(
                n as i32,
                cufft_sys::cufftType::CUFFT_C2C,
                batch as i32,
                cu_stream.clone(),
            )
            .map_err(|e| anyhow!("plan: {e:?}"))?;
            let cflag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            for _ in 0..WARMUP {
                cu_plan
                    .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
                    .map_err(|e| anyhow!("exec: {e:?}"))?;
            }
            cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
            let mut cu = Vec::with_capacity(TRIALS);
            for _ in 0..TRIALS {
                let cs = cudarc_ctx
                    .new_event(cflag)
                    .map_err(|e| anyhow!("ev: {e}"))?;
                let ce = cudarc_ctx
                    .new_event(cflag)
                    .map_err(|e| anyhow!("ev: {e}"))?;
                cs.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
                cu_plan
                    .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
                    .map_err(|e| anyhow!("exec: {e:?}"))?;
                ce.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
                ce.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
                cu.push(cs.elapsed_ms(&ce).map_err(|e| anyhow!("el: {e}"))? as f64 * 1.0e-3);
            }
            let cu_med = median(cu);

            let us = |t: f64| t * 1.0e6 / batch as f64;
            let gpts = |t: f64| (total as f64) / t / 1.0e9;
            let cu_us = us(cu_med);
            println!(
                "{n},{batch},interleaved,{:.5},{:.5},{:.4}",
                us(t_i),
                cu_us,
                gpts(t_i)
            );
            println!(
                "{n},{batch},split,{:.5},{:.5},{:.4}",
                us(t_s),
                cu_us,
                gpts(t_s)
            );
        }
    }
    Ok(())
}
