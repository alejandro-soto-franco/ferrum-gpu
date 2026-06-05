//! Probes shared-memory bank-conflict behaviour on sm_120.
//! Two kernels: stride-256 reads with +0 padding (conflict-prone) and with +1 padding
//! (conflict-free). Times each over many trials. Ratio tells us if Blackwell
//! still has the 32-bank x 4-byte layout the tuning guide implies.

use std::time::Instant;

use anyhow::Result;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

const TRIALS: usize = 1_000;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub(crate) fn stride_read_pad0(mut out: DisjointSlice<f32>) {
        static mut BUF: SharedArray<f32, 8192> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        // Initialize.
        let mut i = tid;
        while i < 8192 {
            unsafe {
                BUF[i] = i as f32;
            }
            i += 1024;
        }
        thread::sync_threads();
        // Stride-256 reads (conflict-prone if 256 % 32 == 0 which it is).
        let mut acc = 0.0f32;
        let mut k = tid;
        while k < 32 {
            acc += unsafe { BUF[k * 256] };
            k += 1024;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = acc;
            }
        }
    }

    #[kernel]
    pub(crate) fn stride_read_pad1(mut out: DisjointSlice<f32>) {
        // +1 word every 256 elements. Buffer size = 8192 + 32 = 8224.
        static mut BUF: SharedArray<f32, 8224> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let mut i = tid;
        while i < 8192 {
            let pad = i / 256;
            unsafe {
                BUF[i + pad] = i as f32;
            }
            i += 1024;
        }
        thread::sync_threads();
        let mut acc = 0.0f32;
        let mut k = tid;
        while k < 32 {
            let i = k * 256;
            let pad = i / 256;
            acc += unsafe { BUF[i + pad] };
            k += 1024;
        }
        if tid == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = acc;
            }
        }
    }
}

fn time<F: FnMut() -> Result<()>>(mut f: F) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    let t0 = Instant::now();
    for _ in 0..TRIALS {
        f()?;
    }
    Ok(t0.elapsed().as_secs_f64() / TRIALS as f64)
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let stream = ctx.default_stream();
    let mut out_pad0 = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let mut out_pad1 = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    };
    let s_pad0 = time(|| {
        module.stride_read_pad0(stream.as_ref(), cfg, &mut out_pad0)?;
        stream.synchronize()?;
        Ok(())
    })?;
    let s_pad1 = time(|| {
        module.stride_read_pad1(stream.as_ref(), cfg, &mut out_pad1)?;
        stream.synchronize()?;
        Ok(())
    })?;
    let us_pad0 = s_pad0 * 1.0e6;
    let us_pad1 = s_pad1 * 1.0e6;
    println!("stride-256 read, pad +0: {:.3} us", us_pad0);
    println!("stride-256 read, pad +1: {:.3} us", us_pad1);
    println!("ratio pad0/pad1: {:.2}x", us_pad0 / us_pad1);
    Ok(())
}
