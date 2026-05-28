//! Plan 3 vertical integration test: 1D radix-2 power-of-2 C2C FFT.
//!
//! The kernel is a single-block-per-FFT Stockham auto-sort implementation
//! that pings between two `SharedArray<f32, 8192>` slots. For each stage,
//! one thread handles one butterfly; the block writes the final ping-pong
//! slot back to global memory. Sizes 4 through 4096 are supported (N_MAX
//! = 4096; shared memory = 2 * 8192 * 4 = 65536 bytes per block, which
//! fits within the RTX 5060's per-block shared-memory ceiling on sm_120).
//!
//! The release gate is the 8-case verification table at the bottom of the
//! binary: each case constructs a `ferrum_gpu_fft::Plan`, builds a CPU
//! reference via `Plan::cpu_execute`, runs the GPU kernel against the same
//! input, and asserts elementwise relative error below 1e-4.
//!
//! Inverse direction is implemented on the host via the conjugate trick
//! (conj input → forward FFT → conj output, optionally / N). The kernel
//! itself does forward only and ignores `direction`.

use anyhow::{Result, anyhow};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

use ferrum_gpu_fft::{Complex32, Direction, Plan};

#[cuda_module]
mod kernels {
    use super::*;

    /// Single-block-per-lane radix-2 Stockham FFT.
    ///
    /// Layout assumptions, matching the CPU reference in
    /// `ferrum-gpu-fft/src/cpu.rs`:
    ///   * `in_data` / `out_data` are interleaved `[re0, im0, re1, im1, ...]`
    ///     of length `2 * N * batch`.
    ///   * `twiddles` is interleaved `[re, im, ...]` of length `2 * (N - 1)`,
    ///     laid out stages descending (largest first): stage `s` occupies
    ///     offset `N - (1 << s)` ... `N - (1 << s) + (1 << (s - 1))`.
    ///   * `grid_dim = (batch, 1, 1)`, `block_dim = (N / 2, 1, 1)`.
    ///   * `log_n` in [2, 12], so `N` in {4, 8, ..., 4096}.
    ///
    /// Each thread loads two complex elements into shared memory (stride
    /// N/2 apart), runs `log_n` Stockham stages with a per-stage barrier,
    /// then stores the result back to global memory.
    #[kernel]
    pub(crate) fn fft_radix2_c2c_pow2_1d_fallback(
        in_data: &[f32],
        twiddles: &[f32],
        mut out_data: DisjointSlice<f32>,
        log_n: u32,
    ) {
        // N_MAX = 4096 complex elements = 8192 f32 slots per ping-pong slab.
        static mut PING: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut PONG: SharedArray<f32, 8192> = SharedArray::UNINIT;

        let n = 1usize << log_n;
        let half_n = n >> 1;

        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x() as usize;
        let lane_off = block * n * 2;

        // ---------------------------------------------------------------
        // Load: each thread copies multiple complex elements into PING via
        // a strided loop, so block_dim can be capped below half_n for
        // large N. For N <= 2 * block_dim each thread does one outer
        // iteration; for N = 4096 with block_dim = 1024 each thread does
        // two outer iterations.
        // ---------------------------------------------------------------
        {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    PING[i0] = in_data[lane_off + i0];
                    PING[i0 + 1] = in_data[lane_off + i0 + 1];
                    PING[i1] = in_data[lane_off + i1];
                    PING[i1 + 1] = in_data[lane_off + i1 + 1];
                }
                t += block_dim;
            }
        }
        thread::sync_threads();

        // ---------------------------------------------------------------
        // Stockham stages. Source/destination ping-pongs each stage. We
        // can't conditionally bind a `&mut SharedArray` reference in Rust
        // without aliasing trouble, so we duplicate the stage body per
        // direction of the ping-pong flag. Each thread handles multiple
        // butterflies via a strided loop over (b in 0..half_n).
        // ---------------------------------------------------------------
        let mut s = 1u32;
        let mut src_is_ping = true;
        while s <= log_n {
            let m_half = 1usize << (s - 1);
            let m = 1usize << s;
            // Stage offset in the descending-layout twiddle table.
            // Stage s lives at offset `N - (1 << s)` with length `2^(s-1)`.
            let stage_off = n - m;

            let mut b = tid;
            while b < half_n {
                // Map butterfly index b in 0..half_n to (j, k):
                // j = b / m_half, k = b % m_half.
                let j = b / m_half;
                let k = b - j * m_half;
                let tw_i = 2 * (stage_off + k);
                let w_re = twiddles[tw_i];
                let w_im = twiddles[tw_i + 1];

                // Source / destination indices (in f32 slots).
                let src_a = 2 * (j * m_half + k);
                let src_b = 2 * (j * m_half + k + half_n);
                let dst_a = 2 * (j * m + k);
                let dst_b = 2 * (j * m + k + m_half);

                if src_is_ping {
                    let a_re = unsafe { PING[src_a] };
                    let a_im = unsafe { PING[src_a + 1] };
                    let b_re = unsafe { PING[src_b] };
                    let b_im = unsafe { PING[src_b + 1] };
                    let wb_re = w_re * b_re - w_im * b_im;
                    let wb_im = w_re * b_im + w_im * b_re;
                    unsafe {
                        PONG[dst_a] = a_re + wb_re;
                        PONG[dst_a + 1] = a_im + wb_im;
                        PONG[dst_b] = a_re - wb_re;
                        PONG[dst_b + 1] = a_im - wb_im;
                    }
                } else {
                    let a_re = unsafe { PONG[src_a] };
                    let a_im = unsafe { PONG[src_a + 1] };
                    let b_re = unsafe { PONG[src_b] };
                    let b_im = unsafe { PONG[src_b + 1] };
                    let wb_re = w_re * b_re - w_im * b_im;
                    let wb_im = w_re * b_im + w_im * b_re;
                    unsafe {
                        PING[dst_a] = a_re + wb_re;
                        PING[dst_a + 1] = a_im + wb_im;
                        PING[dst_b] = a_re - wb_re;
                        PING[dst_b + 1] = a_im - wb_im;
                    }
                }
                b += block_dim;
            }
            thread::sync_threads();
            src_is_ping = !src_is_ping;
            s += 1;
        }

        // ---------------------------------------------------------------
        // Store: after log_n stages, the result lives in whichever slab
        // is now the "source" (we toggled after the last write).
        // src_is_ping == true => result in PING; false => result in PONG.
        // ---------------------------------------------------------------
        if src_is_ping {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    *out_data.get_unchecked_mut(lane_off + i0) = PING[i0];
                    *out_data.get_unchecked_mut(lane_off + i0 + 1) = PING[i0 + 1];
                    *out_data.get_unchecked_mut(lane_off + i1) = PING[i1];
                    *out_data.get_unchecked_mut(lane_off + i1 + 1) = PING[i1 + 1];
                }
                t += block_dim;
            }
        } else {
            let mut t = tid;
            while t < half_n {
                let i0 = 2 * t;
                let i1 = 2 * (t + half_n);
                unsafe {
                    *out_data.get_unchecked_mut(lane_off + i0) = PONG[i0];
                    *out_data.get_unchecked_mut(lane_off + i0 + 1) = PONG[i0 + 1];
                    *out_data.get_unchecked_mut(lane_off + i1) = PONG[i1];
                    *out_data.get_unchecked_mut(lane_off + i1 + 1) = PONG[i1 + 1];
                }
                t += block_dim;
            }
        }
    }
}

fn flatten(c: &[Complex32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(c.len() * 2);
    for x in c {
        v.push(x.re);
        v.push(x.im);
    }
    v
}

fn unflatten(f: &[f32]) -> Vec<Complex32> {
    f.chunks_exact(2)
        .map(|c| Complex32::new(c[0], c[1]))
        .collect()
}

struct Case {
    log_n: u32,
    batch: usize,
    dir: Direction,
    normalize: bool,
    label: &'static str,
}

const CASES: &[Case] = &[
    Case { log_n: 2,  batch: 1, dir: Direction::Forward, normalize: false, label: "N=4 fwd" },
    Case { log_n: 3,  batch: 1, dir: Direction::Forward, normalize: false, label: "N=8 fwd" },
    Case { log_n: 6,  batch: 1, dir: Direction::Forward, normalize: false, label: "N=64 fwd" },
    Case { log_n: 8,  batch: 1, dir: Direction::Forward, normalize: false, label: "N=256 fwd" },
    Case { log_n: 10, batch: 1, dir: Direction::Forward, normalize: false, label: "N=1024 fwd" },
    Case { log_n: 12, batch: 1, dir: Direction::Forward, normalize: false, label: "N=4096 fwd" },
    Case { log_n: 8,  batch: 8, dir: Direction::Forward, normalize: false, label: "N=256 fwd batch=8" },
    Case { log_n: 8,  batch: 1, dir: Direction::Inverse, normalize: true,  label: "N=256 inv normalize" },
];

fn run_case(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    case: &Case,
) -> Result<()> {
    let plan = Plan::new(case.log_n, case.batch, case.normalize);
    let n = plan.n();
    let total = n * case.batch;
    let input: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new(((i % n) as f32).sin(), ((i % n) as f32).cos()))
        .collect();

    // CPU reference (acts on a copy).
    let mut cpu = input.clone();
    plan.cpu_execute(&mut cpu, case.dir);

    // GPU input: for inverse, conjugate on host so the forward-only kernel
    // produces the inverse DFT (up to the final conjugate + 1/N scale).
    let mut input_for_gpu = input.clone();
    if case.dir == Direction::Inverse {
        for c in input_for_gpu.iter_mut() {
            *c = c.conj();
        }
    }

    let stream = ctx.default_stream();
    let input_flat = flatten(&input_for_gpu);
    let twiddles_flat = flatten(plan.twiddles());
    let dbuf_in = DeviceBuffer::from_host(&stream, &input_flat)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    // CUDA caps block_dim at 1024 threads per block. For N <= 2048, one
    // thread per butterfly. For N = 4096 (half_n = 2048), each thread does
    // two butterflies via the kernel's strided loop.
    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (case.batch as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    module.fft_radix2_c2c_pow2_1d_fallback(
        stream.as_ref(),
        cfg,
        &dbuf_in,
        &dbuf_tw,
        &mut dbuf_out,
        case.log_n,
    )?;

    let mut host_out_flat = dbuf_out.to_host_vec(&stream)?;

    // Inverse: conjugate output, then scale by 1/N if normalize.
    if case.dir == Direction::Inverse {
        // host_out_flat is [re0, im0, re1, im1, ...]; conjugate by negating
        // every odd index.
        for chunk in host_out_flat.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
        if case.normalize {
            let inv_n = 1.0f32 / n as f32;
            for v in host_out_flat.iter_mut() {
                *v *= inv_n;
            }
        }
    }

    let host_out = unflatten(&host_out_flat);
    for i in 0..total {
        let g = host_out[i];
        let c = cpu[i];
        let dre = (g.re - c.re).abs();
        let dim = (g.im - c.im).abs();
        let norm = c.re.abs() + c.im.abs() + 1.0;
        if (dre + dim) / norm > 1e-4 {
            return Err(anyhow!(
                "{}: mismatch at {i}: gpu=({}, {}), cpu=({}, {}), |dre|+|dim|/norm={}",
                case.label,
                g.re,
                g.im,
                c.re,
                c.im,
                (dre + dim) / norm,
            ));
        }
    }
    println!("ok  {} (N={}, batch={})", case.label, n, case.batch);
    Ok(())
}

fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;
    let mut fails = 0usize;
    for case in CASES {
        if let Err(e) = run_case(&ctx, &module, case) {
            eprintln!("FAIL {}: {e}", case.label);
            fails += 1;
        }
    }
    if fails > 0 {
        anyhow::bail!("{} of {} cases failed", fails, CASES.len());
    }
    println!("\nfft-1d-c2c: {}/{} cases verified", CASES.len(), CASES.len());
    Ok(())
}
