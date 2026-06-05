//! Measures wall-clock per-launch overhead through both cuda-core and cudarc.
//! Launches an empty kernel TRIALS times in a single stream, divides total
//! wall time by TRIALS. Reports both paths and their gap.

use std::time::Instant;

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext as CoreCtx, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
use cudarc::driver::{CudaContext as CudarcCtx, LaunchConfig as CudarcCfg};

const TRIALS: usize = 10_000;
const WARMUP: usize = 1_000;

#[cuda_module]
mod core_kern {
    use super::*;
    #[kernel]
    pub(crate) fn noop() {
        let _ = thread::threadIdx_x();
    }
}

fn bench_core() -> Result<f64> {
    let ctx = CoreCtx::new(0)?;
    let module = core_kern::load(&ctx)?;
    let stream = ctx.default_stream();
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..WARMUP {
        module.noop(stream.as_ref(), cfg)?;
    }
    stream.synchronize()?;
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        module.noop(stream.as_ref(), cfg)?;
    }
    stream.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() / TRIALS as f64)
}

fn bench_cudarc() -> Result<f64> {
    // cudarc has no clean empty-kernel path; compile a tiny PTX module.
    let ptx_src = r#"
.version 7.0
.target sm_50
.address_size 64
.visible .entry noop() { ret; }
"#;
    let ctx = CudarcCtx::new(0).map_err(|e| anyhow!("cudarc ctx: {e}"))?;
    let module = ctx
        .load_module(cudarc::nvrtc::Ptx::from_src(ptx_src))
        .map_err(|e| anyhow!("load_module: {e}"))?;
    let func = module
        .load_function("noop")
        .map_err(|e| anyhow!("load_function: {e}"))?;
    let stream = ctx.default_stream();
    let cfg = CudarcCfg {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..WARMUP {
        let mut builder = stream.launch_builder(&func);
        unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        let mut builder = stream.launch_builder(&func);
        unsafe { builder.launch(cfg) }.map_err(|e| anyhow!("launch: {e}"))?;
    }
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
    Ok(t0.elapsed().as_secs_f64() / TRIALS as f64)
}

fn main() -> Result<()> {
    let core_s = bench_core()?;
    let cudarc_s = bench_cudarc()?;
    let core_us = core_s * 1.0e6;
    let cudarc_us = cudarc_s * 1.0e6;
    let gap = core_us - cudarc_us;
    println!("cuda-core empty-launch: {:.3} us", core_us);
    println!("cudarc    empty-launch: {:.3} us", cudarc_us);
    println!("gap (cuda-core - cudarc): {:+.3} us", gap);
    Ok(())
}
