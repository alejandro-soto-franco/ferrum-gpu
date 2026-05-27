//! Python bindings for ferrum-gpu.
//!
//! The cuda-oxide FFT kernel lives inline in this cdylib (kernels must live
//! in the binary crate so cuda-oxide's `#[cuda_module]` macro can embed the
//! PTX into the link section). The Python surface is
//! `ferrum_gpu.fft.fft_1d_c2c_pow2(arr, log_n, direction, normalize)`.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use cuda_core::embedded::{ArtifactPayloadKind, artifact_bundles_from_binary_path};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

use ferrum_gpu_fft::Plan;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cuda_module]
mod kernels {
    use super::*;

    /// Single-block-per-lane radix-2 Stockham FFT. Same body as
    /// `examples/fft-1d-c2c/src/main.rs::kernels::fft_radix2_c2c_pow2_1d`.
    #[kernel]
    pub(crate) fn fft_radix2_c2c_pow2_1d(
        in_data: &[f32],
        twiddles: &[f32],
        mut out_data: DisjointSlice<f32>,
        log_n: u32,
    ) {
        static mut PING: SharedArray<f32, 8192> = SharedArray::UNINIT;
        static mut PONG: SharedArray<f32, 8192> = SharedArray::UNINIT;

        let n = 1usize << log_n;
        let half_n = n >> 1;

        let block = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let block_dim = thread::blockDim_x() as usize;
        let lane_off = block * n * 2;

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

        let mut s = 1u32;
        let mut src_is_ping = true;
        while s <= log_n {
            let m_half = 1usize << (s - 1);
            let m = 1usize << s;
            let stage_off = n - m;

            let mut b = tid;
            while b < half_n {
                let j = b / m_half;
                let k = b - j * m_half;
                let tw_i = 2 * (stage_off + k);
                let w_re = twiddles[tw_i];
                let w_im = twiddles[tw_i + 1];

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

/// Locate the path to our own cdylib at runtime. The macro-generated
/// `kernels::load()` calls `std::env::current_exe()` to find the embedded
/// CUDA module, but in a Python extension the current exe is the Python
/// interpreter (which carries no PTX). We use `libc::dladdr` on a function
/// inside this crate to recover the path of the shared library that
/// contains it.
fn our_cdylib_path() -> Result<PathBuf> {
    let mut info: libc::Dl_info = unsafe { core::mem::zeroed() };
    let func_ptr = our_cdylib_path as *const libc::c_void;
    let ret = unsafe { libc::dladdr(func_ptr, &mut info) };
    if ret == 0 || info.dli_fname.is_null() {
        return Err(anyhow!("dladdr failed to locate the ferrum-gpu .so"));
    }
    let c_str = unsafe { CStr::from_ptr(info.dli_fname) };
    Ok(PathBuf::from(c_str.to_string_lossy().into_owned()))
}

/// Bypass `kernels::load()`: load the embedded PTX bundle from THIS cdylib
/// (not from the Python interpreter), build a `CudaModule`, wrap it into
/// the macro-generated `LoadedModule`.
fn load_kernels(ctx: &Arc<CudaContext>) -> Result<kernels::LoadedModule> {
    let so_path = our_cdylib_path()?;
    let bundles = artifact_bundles_from_binary_path(&so_path)
        .map_err(|e| anyhow!("artifact bundles read failed for {}: {e}", so_path.display()))?;
    let bundle = bundles
        .into_iter()
        .find(|b| b.name == env!("CARGO_PKG_NAME"))
        .ok_or_else(|| {
            anyhow!(
                "no embedded CUDA bundle named {:?} in {}",
                env!("CARGO_PKG_NAME"),
                so_path.display()
            )
        })?;
    let ptx = bundle
        .payload(ArtifactPayloadKind::Ptx)
        .ok_or_else(|| anyhow!("bundle has no Ptx payload"))?;
    let cuda_module = ctx
        .load_module_from_image(ptx)
        .map_err(|e| anyhow!("load_module_from_image: {e}"))?;
    kernels::from_module(cuda_module)
        .map_err(|e| anyhow!("kernels::from_module: {e}"))
}

/// Host-side runner. Input/output are flat f32 slices interleaved (re, im).
/// `dir`: +1 forward, -1 inverse. `normalize`: divide inverse output by N.
///
/// Takes a borrowed `Arc<CudaContext>` + a borrowed `LoadedModule` so callers
/// can reuse a persistent device across many calls.
fn run_fft_flat_with_device(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    input_flat: &[f32],
    log_n: u32,
    batch: usize,
    dir: i32,
    normalize: bool,
) -> Result<Vec<f32>> {
    let n = 1usize << log_n;
    let total = n * batch;
    if input_flat.len() != total * 2 {
        return Err(anyhow!(
            "input_flat len {} != expected {} (2*N*batch)",
            input_flat.len(),
            total * 2
        ));
    }

    let plan = Plan::new(log_n, batch, normalize);
    let stream = ctx.default_stream();

    // For inverse: conjugate input on host (forward kernel + conjugate output
    // = inverse DFT, optionally / N).
    let mut input_for_gpu: Vec<f32> = input_flat.to_vec();
    if dir < 0 {
        for chunk in input_for_gpu.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
    }

    let mut twiddles_flat: Vec<f32> = Vec::with_capacity((n - 1) * 2);
    for c in plan.twiddles() {
        twiddles_flat.push(c.re);
        twiddles_flat.push(c.im);
    }

    let dbuf_in = DeviceBuffer::from_host(&stream, &input_for_gpu)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    let cfg = LaunchConfig {
        grid_dim: (batch as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    module.fft_radix2_c2c_pow2_1d(
        stream.as_ref(),
        cfg,
        &dbuf_in,
        &dbuf_tw,
        &mut dbuf_out,
        log_n,
    )?;

    let mut host_out_flat = dbuf_out.to_host_vec(&stream)?;

    if dir < 0 {
        for chunk in host_out_flat.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
        if normalize {
            let inv_n = 1.0f32 / n as f32;
            for v in host_out_flat.iter_mut() {
                *v *= inv_n;
            }
        }
    }

    Ok(host_out_flat)
}

/// One-shot variant: build a transient context + module, run, drop. Backward
/// compatible path for callers that don't pass an explicit `device=`.
fn run_fft_flat_oneshot(
    input_flat: &[f32],
    log_n: u32,
    batch: usize,
    dir: i32,
    normalize: bool,
) -> Result<Vec<f32>> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let module = load_kernels(&ctx)?;
    run_fft_flat_with_device(&ctx, &module, input_flat, log_n, batch, dir, normalize)
}

/// Persistent CUDA device + loaded module. Construct once, reuse across
/// many `fft_1d_c2c_pow2` calls. Avoids ~200 ms of context-creation +
/// module-load overhead per call.
#[pyclass(module = "ferrum_gpu._native")]
pub struct Device {
    inner: Arc<CudaContext>,
    module: kernels::LoadedModule,
}

#[pymethods]
impl Device {
    #[new]
    #[pyo3(signature = (ordinal = 0))]
    fn new(ordinal: usize) -> PyResult<Self> {
        let inner = CudaContext::new(ordinal).map_err(|e| {
            PyValueError::new_err(format!("CudaContext::new({ordinal}): {e}"))
        })?;
        let module = load_kernels(&inner)
            .map_err(|e| PyValueError::new_err(format!("load_kernels: {e}")))?;
        Ok(Self { inner, module })
    }

    /// Synchronise the default stream.
    fn sync(&self) -> PyResult<()> {
        self.inner
            .default_stream()
            .synchronize()
            .map_err(|e| PyValueError::new_err(format!("sync: {e}")))
    }
}

/// Returns the crate version (smoke).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// FFT of a 1D `complex64` numpy array.
///
/// Arguments
/// ---------
/// arr : numpy.ndarray
///     `complex64` array of length `batch * N` where `N = 1 << log_n` and
///     `log_n in [2, 12]`.
/// log_n : int
///     log2(N).
/// direction : str
///     "forward" (default) or "inverse".
/// normalize : bool
///     Scale inverse output by 1/N when True.
/// device : ferrum_gpu.cuda.Device, optional
///     Persistent device handle. When supplied, the FFT reuses its
///     CudaContext + loaded module instead of building transient ones.
///
/// Returns
/// -------
/// numpy.ndarray of `complex64`, same length as input.
#[pyfunction]
#[pyo3(signature = (arr, log_n, direction = "forward", normalize = false, device = None))]
fn fft_1d_c2c_pow2<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, num_complex::Complex32>,
    log_n: u32,
    direction: &str,
    normalize: bool,
    device: Option<PyRef<'_, Device>>,
) -> PyResult<Bound<'py, PyArray1<num_complex::Complex32>>> {
    if !(2..=12).contains(&log_n) {
        return Err(PyValueError::new_err(format!(
            "log_n must be in [2, 12], got {log_n}"
        )));
    }
    let n = 1usize << log_n;
    let arr_view = arr.as_slice()?;
    let total = arr_view.len();
    if total == 0 || total % n != 0 {
        return Err(PyValueError::new_err(format!(
            "arr len {total} must be a positive multiple of N = {n}"
        )));
    }
    let batch = total / n;
    let dir_i = match direction {
        "forward" => 1i32,
        "inverse" => -1i32,
        other => {
            return Err(PyValueError::new_err(format!(
                "direction must be 'forward' or 'inverse', got {other:?}"
            )));
        }
    };

    let mut input_flat: Vec<f32> = Vec::with_capacity(total * 2);
    for c in arr_view {
        input_flat.push(c.re);
        input_flat.push(c.im);
    }

    // Pull Send-able handles out of the (potentially borrowed) Device BEFORE
    // entering allow_threads (PyRef<'_, Device> is !Send).
    let device_handles: Option<(Arc<CudaContext>, kernels::LoadedModule)> = device
        .as_ref()
        .map(|d| (d.inner.clone(), d.module.clone()));

    let output_flat = py
        .allow_threads(|| match device_handles {
            Some((ctx, module)) => run_fft_flat_with_device(
                &ctx, &module, &input_flat, log_n, batch, dir_i, normalize,
            ),
            None => run_fft_flat_oneshot(&input_flat, log_n, batch, dir_i, normalize),
        })
        .map_err(|e| PyValueError::new_err(format!("ferrum-gpu fft error: {e}")))?;

    let mut out: Vec<num_complex::Complex32> = Vec::with_capacity(total);
    for chunk in output_flat.chunks_exact(2) {
        out.push(num_complex::Complex::new(chunk[0], chunk[1]));
    }
    Ok(out.into_pyarray(py))
}

/// Native extension module entry point. Exposed under `ferrum_gpu._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;

    let cuda = PyModule::new(m.py(), "cuda")?;
    cuda.add_class::<Device>()?;
    m.add_submodule(&cuda)?;

    let fft = PyModule::new(m.py(), "fft")?;
    fft.add_function(wrap_pyfunction!(fft_1d_c2c_pow2, &fft)?)?;
    m.add_submodule(&fft)?;
    Ok(())
}
