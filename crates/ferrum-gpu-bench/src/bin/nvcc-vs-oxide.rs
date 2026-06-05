//! Codegen head-to-head: cuda-oxide vs nvcc on the IDENTICAL N=256 radix-16
//! kernel (`fft_c2c_256_r16s`), plus cuFFT as a reference.
//!
//! This isolates CODEGEN quality from algorithm: the Rust kernel (compiled by
//! the cuda-oxide fork backend) and the CUDA C kernel below are the same
//! algorithm, same shared layout, same dft8, same u64-coalesced IO. nvcc (via
//! NVRTC) compiles the C; we compile the Rust. Whoever is faster wins on
//! codegen. cuFFT is a different (tuned) algorithm and only a yardstick.
//!
//! Run: `cargo oxide run ferrum-gpu-bench --bin nvcc-vs-oxide`.
//! Env: NVCC_BATCH (default 4096).

use std::time::Instant;

use anyhow::{Result, anyhow};
use cuda_core::{DeviceBuffer, LaunchConfig};
use cudarc::cufft::{CudaFft, FftDirection, sys as cufft_sys};
use cudarc::driver::{LaunchConfig as CuLaunchConfig, PushKernelArg};
use ferrum_gpu_bench::init_cuda_contexts;
use ferrum_gpu_fft::{Complex32, Direction, Plan, twiddles_full_roots};

include!("../../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

const WARMUP_DEFAULT: usize = 50;
const TRIALS_DEFAULT: usize = 200;

// CUDA C port of `fft_c2c_256_r16s` — bit-faithful to the Rust kernel:
// 32 threads, __shared__ float BUF[512], u64-coalesced load, two radix-16
// Stockham stages (16 active lanes), u64-coalesced store. Same dft8 butterfly,
// same per-stage twiddle indexing, same combine table.
const CUDA_SRC: &str = r#"
typedef unsigned long long u64;
typedef unsigned int u32;

struct d8 { float v[16]; };

__device__ __forceinline__ d8 dft8(
    float x0,float x1,float x2,float x3,float x4,float x5,float x6,float x7,
    float x8,float x9,float x10,float x11,float x12,float x13,float x14,float x15) {
    const float C = 0.70710678f;
    float x0r=x0,x0i=x1, x1r=x2,x1i=x3, x2r=x4,x2i=x5, x3r=x6,x3i=x7;
    float x4r=x8,x4i=x9, x5r=x10,x5i=x11, x6r=x12,x6i=x13, x7r=x14,x7i=x15;
    float a0r=x0r+x4r,a0i=x0i+x4i, a1r=x0r-x4r,a1i=x0i-x4i;
    float b0r=x2r+x6r,b0i=x2i+x6i, b1r=x2r-x6r,b1i=x2i-x6i;
    float e0r=a0r+b0r,e0i=a0i+b0i, e2r=a0r-b0r,e2i=a0i-b0i;
    float e1r=a1r+b1i,e1i=a1i-b1r, e3r=a1r-b1i,e3i=a1i+b1r;
    float c0r=x1r+x5r,c0i=x1i+x5i, c1r=x1r-x5r,c1i=x1i-x5i;
    float d0r=x3r+x7r,d0i=x3i+x7i, d1r=x3r-x7r,d1i=x3i-x7i;
    float o0r=c0r+d0r,o0i=c0i+d0i, o2r=c0r-d0r,o2i=c0i-d0i;
    float o1r=c1r+d1i,o1i=c1i-d1r, o3r=c1r-d1i,o3i=c1i+d1r;
    float w1r=C*(o1r+o1i), w1i=C*(o1i-o1r);
    float w3r=C*(o3i-o3r), w3i=-C*(o3r+o3i);
    d8 o;
    o.v[0]=e0r+o0r; o.v[1]=e0i+o0i; o.v[2]=e1r+w1r; o.v[3]=e1i+w1i;
    o.v[4]=e2r+o2i; o.v[5]=e2i-o2r; o.v[6]=e3r+w3r; o.v[7]=e3i+w3i;
    o.v[8]=e0r-o0r; o.v[9]=e0i-o0i; o.v[10]=e1r-w1r; o.v[11]=e1i-w1i;
    o.v[12]=e2r-o2i; o.v[13]=e2i+o2r; o.v[14]=e3r-w3r; o.v[15]=e3i-w3i;
    return o;
}

__device__ void stage256(int b, int m_r, int do_tw, float* BUF, const float* w256) {
    int m = m_r * 16;
    int j = b / m_r;
    int k = b - j * m_r;
    int src_base = j * m_r + k;
    float p[32];
    #pragma unroll
    for (int q = 0; q < 16; q++) {
        int s = 2 * (src_base + q * 16);
        float re = BUF[s], im = BUF[s + 1];
        if (do_tw && q != 0) {
            int e = 2 * ((q * k) & 255);
            float wr = w256[e], wi = w256[e + 1];
            float nr = re * wr - im * wi;
            float ni = re * wi + im * wr;
            re = nr; im = ni;
        }
        p[2 * q] = re; p[2 * q + 1] = im;
    }
    d8 ev = dft8(p[0],p[1], p[4],p[5], p[8],p[9], p[12],p[13],
                 p[16],p[17], p[20],p[21], p[24],p[25], p[28],p[29]);
    d8 od = dft8(p[2],p[3], p[6],p[7], p[10],p[11], p[14],p[15],
                 p[18],p[19], p[22],p[23], p[26],p[27], p[30],p[31]);
    const float Wt[16] = {
        1.0f,0.0f, 0.92387953f,-0.38268343f, 0.70710678f,-0.70710678f,
        0.38268343f,-0.92387953f, 0.0f,-1.0f, -0.38268343f,-0.92387953f,
        -0.70710678f,-0.70710678f, -0.92387953f,-0.38268343f };
    float outr[16], outi[16];
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        float wr = Wt[2*c], wi = Wt[2*c+1];
        float er = ev.v[2*c], ei = ev.v[2*c+1];
        float orr = od.v[2*c], oii = od.v[2*c+1];
        float tr = orr*wr - oii*wi, ti = orr*wi + oii*wr;
        outr[c] = er + tr; outi[c] = ei + ti;
        outr[c+8] = er - tr; outi[c+8] = ei - ti;
    }
    int dst_base = j * m + k;
    #pragma unroll
    for (int q = 0; q < 16; q++) {
        int d = 2 * (dst_base + q * m_r);
        BUF[d] = outr[q]; BUF[d + 1] = outi[q];
    }
}

extern "C" __global__ void fft256_r16s(const float* in_data, const float* w256, float* out_data) {
    __shared__ float BUF[512];
    const int N = 256;
    const int THREADS = 32;
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    int lane_off = blk * N * 2;
    for (int t = tid; t < N; t += THREADS) {
        u64 c = *(const u64*)(in_data + lane_off + 2 * t);
        BUF[2*t]   = __uint_as_float((u32)c);
        BUF[2*t+1] = __uint_as_float((u32)(c >> 32));
    }
    __syncthreads();
    if (tid < 16) stage256(tid, 1, 0, BUF, w256);
    __syncthreads();
    if (tid < 16) stage256(tid, 16, 1, BUF, w256);
    __syncthreads();
    for (int t = tid; t < N; t += THREADS) {
        u32 re = __float_as_uint(BUF[2*t]);
        u32 im = __float_as_uint(BUF[2*t+1]);
        u64 packed = (u64)re | ((u64)im << 32);
        *(u64*)(out_data + lane_off + 2 * t) = packed;
    }
}
"#;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let (core_ctx, cudarc_ctx) = init_cuda_contexts()?;
    let batch: usize = std::env::var("NVCC_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    // WARMUP/TRIALS are env-overridable so a profiler run (ncu) can do just the
    // correctness gate plus a minimal tail instead of 50 + 400 timed launches.
    let warmup: usize = std::env::var("WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(WARMUP_DEFAULT);
    let trials: usize = std::env::var("TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(TRIALS_DEFAULT);
    let n = 256usize;
    let total = n * batch;

    // Twiddles: full roots W_256^e, e in 0..256 (matches the Rust kernel).
    let w: Vec<Complex32> = twiddles_full_roots(8);
    let w_flat: Vec<f32> = w.iter().flat_map(|c| [c.re, c.im]).collect();

    // Host input (shared by all three paths).
    let input_flat: Vec<f32> = (0..total * 2).map(|i| (i as f32 * 0.001).sin()).collect();

    // ---- oxide (cuda-core) ----
    let core_stream = core_ctx.default_stream();
    let mod_k = kernels::load(&core_ctx)?;
    let o_in = DeviceBuffer::from_host(&core_stream, &input_flat)?;
    let o_w = DeviceBuffer::from_host(&core_stream, &w_flat)?;
    let mut o_out = DeviceBuffer::<f32>::zeroed(&core_stream, total * 2)?;
    let o_cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let run_oxide = |din: &DeviceBuffer<f32>,
                     dw: &DeviceBuffer<f32>,
                     dout: &mut DeviceBuffer<f32>|
     -> Result<()> {
        mod_k.fft_c2c_256_r16s(core_stream.as_ref(), o_cfg, din, dw, dout)?;
        Ok(())
    };

    // ---- nvcc (cudarc + NVRTC) ----
    let opts = cudarc::nvrtc::CompileOptions {
        arch: Some("compute_120"),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(CUDA_SRC, opts)
        .map_err(|e| anyhow!("nvrtc compile: {e}"))?;
    let cu_stream = cudarc_ctx.default_stream();
    let nv_module = cudarc_ctx
        .load_module(ptx)
        .map_err(|e| anyhow!("load_module: {e}"))?;
    let nv_func = nv_module
        .load_function("fft256_r16s")
        .map_err(|e| anyhow!("load_function: {e}"))?;
    let n_in = cu_stream
        .clone_htod(&input_flat)
        .map_err(|e| anyhow!("htod in: {e}"))?;
    let n_w = cu_stream
        .clone_htod(&w_flat)
        .map_err(|e| anyhow!("htod w: {e}"))?;
    let mut n_out = cu_stream
        .alloc_zeros::<f32>(total * 2)
        .map_err(|e| anyhow!("alloc out: {e}"))?;
    let nv_cfg = CuLaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let run_nvcc = |din: &cudarc::driver::CudaSlice<f32>,
                    dw: &cudarc::driver::CudaSlice<f32>,
                    dout: &mut cudarc::driver::CudaSlice<f32>|
     -> Result<()> {
        let mut lb = cu_stream.launch_builder(&nv_func);
        lb.arg(din).arg(dw).arg(dout);
        unsafe { lb.launch(nv_cfg) }.map_err(|e| anyhow!("nvcc launch: {e}"))?;
        Ok(())
    };

    // ---- cuFFT reference ----
    let cu_in_data: Vec<cufft_sys::float2> = (0..total)
        .map(|i| cufft_sys::float2 {
            x: ((2 * i) as f32 * 0.001).sin(),
            y: ((2 * i + 1) as f32 * 0.001).sin(),
        })
        .collect();
    let mut c_in = cu_stream
        .clone_htod(&cu_in_data)
        .map_err(|e| anyhow!("htod cufft: {e}"))?;
    let mut c_out = cu_stream
        .alloc_zeros::<cufft_sys::float2>(total)
        .map_err(|e| anyhow!("alloc cufft: {e}"))?;
    let c_plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        batch as i32,
        cu_stream.clone(),
    )
    .map_err(|e| anyhow!("cufft plan: {e:?}"))?;

    // ---- Correctness: oxide and nvcc vs the radix-2 CPU reference ----
    run_oxide(&o_in, &o_w, &mut o_out)?;
    run_nvcc(&n_in, &n_w, &mut n_out)?;
    core_stream.synchronize()?;
    cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
    let o_host = o_out.to_host_vec(&core_stream)?;
    let n_host = cu_stream
        .clone_dtoh(&n_out)
        .map_err(|e| anyhow!("dtoh: {e}"))?;
    let mut worst_o = 0.0f32;
    let mut worst_n = 0.0f32;
    for f in 0..batch.min(64) {
        let mut r: Vec<Complex32> = (0..n)
            .map(|t| Complex32::new(input_flat[(f * n + t) * 2], input_flat[(f * n + t) * 2 + 1]))
            .collect();
        Plan::new(8, 1, false).cpu_execute(&mut r, Direction::Forward);
        for t in 0..n {
            let mag = (r[t].re.powi(2) + r[t].im.powi(2)).sqrt().max(1.0);
            let eo = ((o_host[(f * n + t) * 2] - r[t].re).powi(2)
                + (o_host[(f * n + t) * 2 + 1] - r[t].im).powi(2))
            .sqrt();
            let en = ((n_host[(f * n + t) * 2] - r[t].re).powi(2)
                + (n_host[(f * n + t) * 2 + 1] - r[t].im).powi(2))
            .sqrt();
            worst_o = worst_o.max(eo / mag);
            worst_n = worst_n.max(en / mag);
        }
    }
    // Optional accuracy dump (ACC_DUMP=path): write the first FFT's input and
    // all three outputs so an offline f64 reference (numpy) can score max / RMS
    // / ULP error per kernel. Runs cuFFT once to capture its output.
    if let Ok(path) = std::env::var("ACC_DUMP") {
        use std::io::Write;
        c_plan
            .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft acc: {e:?}"))?;
        cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let c_host = cu_stream
            .clone_dtoh(&c_out)
            .map_err(|e| anyhow!("dtoh cufft: {e}"))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        for t in 0..n {
            writeln!(
                f,
                "{n},{t},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
                input_flat[2 * t],
                input_flat[2 * t + 1],
                o_host[2 * t],
                o_host[2 * t + 1],
                n_host[2 * t],
                n_host[2 * t + 1],
                c_host[t].x,
                c_host[t].y
            )?;
        }
    }
    println!("correctness vs radix-2 CPU: oxide={worst_o:.2e}  nvcc={worst_n:.2e}");
    if worst_o > 1e-3 || worst_n > 1e-3 {
        return Err(anyhow!(
            "correctness FAIL (oxide={worst_o:.2e} nvcc={worst_n:.2e})"
        ));
    }

    // ---- Timing (median per-FFT us over TRIALS) ----
    for _ in 0..warmup {
        run_oxide(&o_in, &o_w, &mut o_out)?;
        run_nvcc(&n_in, &n_w, &mut n_out)?;
        c_plan
            .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft exec: {e:?}"))?;
    }
    core_stream.synchronize()?;
    cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    let core_flag = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let cu_flag = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let (mut ts_o, mut ts_n, mut ts_c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..trials {
        // oxide (cuda-core events)
        let a = core_ctx.new_event(core_flag)?;
        let b = core_ctx.new_event(core_flag)?;
        a.record(&core_stream)?;
        run_oxide(&o_in, &o_w, &mut o_out)?;
        b.record(&core_stream)?;
        b.synchronize()?;
        ts_o.push(a.elapsed_ms(&b)? as f64 * 1e-3);

        // nvcc (cudarc events)
        let a = cudarc_ctx
            .new_event(cu_flag)
            .map_err(|e| anyhow!("ev: {e}"))?;
        let b = cudarc_ctx
            .new_event(cu_flag)
            .map_err(|e| anyhow!("ev: {e}"))?;
        a.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        run_nvcc(&n_in, &n_w, &mut n_out)?;
        b.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        b.synchronize().map_err(|e| anyhow!("sync ev: {e}"))?;
        ts_n.push(a.elapsed_ms(&b).map_err(|e| anyhow!("el: {e}"))? as f64 * 1e-3);

        // cufft (cudarc events)
        let a = cudarc_ctx
            .new_event(cu_flag)
            .map_err(|e| anyhow!("ev: {e}"))?;
        let b = cudarc_ctx
            .new_event(cu_flag)
            .map_err(|e| anyhow!("ev: {e}"))?;
        a.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        c_plan
            .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft: {e:?}"))?;
        b.record(&cu_stream).map_err(|e| anyhow!("rec: {e}"))?;
        b.synchronize().map_err(|e| anyhow!("sync ev: {e}"))?;
        ts_c.push(a.elapsed_ms(&b).map_err(|e| anyhow!("el: {e}"))? as f64 * 1e-3);
    }
    // ---- Unified host wall-clock timing (ONE std::time::Instant for all
    // three) -- eliminates the cuda-core-vs-cudarc EVENT-API confound that the
    // event loop above carries (oxide is timed on cuda-core events, nvcc/cuFFT
    // on cudarc events, in separate contexts). Here every kernel is timed by
    // the same host clock: sync the stream, start the timer, launch, sync, stop.
    // At large batch the kernel time dwarfs launch+sync overhead, so this ratio
    // is essentially pure GPU time measured identically across all three. ----
    let (mut hs_o, mut hs_n, mut hs_c) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..trials {
        core_stream.synchronize()?;
        let t = Instant::now();
        run_oxide(&o_in, &o_w, &mut o_out)?;
        core_stream.synchronize()?;
        hs_o.push(t.elapsed().as_secs_f64());

        cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let t = Instant::now();
        run_nvcc(&n_in, &n_w, &mut n_out)?;
        cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        hs_n.push(t.elapsed().as_secs_f64());

        cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        let t = Instant::now();
        c_plan
            .exec_c2c(&mut c_in, &mut c_out, FftDirection::Forward)
            .map_err(|e| anyhow!("cufft: {e:?}"))?;
        cu_stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;
        hs_c.push(t.elapsed().as_secs_f64());
    }

    // Optional per-trial dump (DUMP_CSV=path, REP tags the repeat) for offline
    // statistics: paired event and host samples for all three kernels, per-FFT
    // us. Rows: rep,batch,trial,timer,oxide_us,nvcc_us,cufft_us.
    if let Ok(path) = std::env::var("DUMP_CSV") {
        use std::io::Write;
        let rep: usize = std::env::var("REP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let pf = 1e6 / batch as f64; // total seconds -> per-FFT microseconds
        for i in 0..ts_o.len() {
            writeln!(
                f,
                "{rep},{batch},{i},event,{:.6},{:.6},{:.6}",
                ts_o[i] * pf,
                ts_n[i] * pf,
                ts_c[i] * pf
            )?;
        }
        for i in 0..hs_o.len() {
            writeln!(
                f,
                "{rep},{batch},{i},host,{:.6},{:.6},{:.6}",
                hs_o[i] * pf,
                hs_n[i] * pf,
                hs_c[i] * pf
            )?;
        }
    }

    let o_us = median(ts_o) * 1e6 / batch as f64;
    let n_us = median(ts_n) * 1e6 / batch as f64;
    let c_us = median(ts_c) * 1e6 / batch as f64;
    let ho_us = median(hs_o) * 1e6 / batch as f64;
    let hn_us = median(hs_n) * 1e6 / batch as f64;
    let hc_us = median(hs_c) * 1e6 / batch as f64;
    println!("N=256 batch={batch}  (per-FFT us; lower is better)");
    println!("  cuda-oxide : {o_us:.5} us");
    println!("  nvcc       : {n_us:.5} us");
    println!("  cuFFT      : {c_us:.5} us");
    println!(
        "  oxide/nvcc : {:.3}  ({})",
        o_us / n_us,
        if o_us < n_us {
            "OXIDE BEATS NVCC"
        } else {
            "nvcc faster"
        }
    );
    println!("  oxide/cuFFT: {:.3}", o_us / c_us);
    println!("  nvcc/cuFFT : {:.3}", n_us / c_us);
    println!("  -- unified host clock (one std::Instant, all three) --");
    println!("  host cuda-oxide : {ho_us:.5} us");
    println!("  host nvcc       : {hn_us:.5} us");
    println!("  host cuFFT      : {hc_us:.5} us");
    println!(
        "  host oxide/nvcc : {:.3}  ({})",
        ho_us / hn_us,
        if ho_us < hn_us {
            "OXIDE BEATS NVCC"
        } else {
            "nvcc faster"
        }
    );
    println!("  host oxide/cuFFT: {:.3}", ho_us / hc_us);
    println!("  host nvcc/cuFFT : {:.3}", hn_us / hc_us);
    Ok(())
}
