//! Python bindings for ferrum-gpu.
//!
//! The cuda-oxide FFT kernel is included via
//! `include!(".../ferrum-gpu-fft-kernels/src/kernels_body.rs")` so the
//! `#[cuda_module] mod kernels { ... }` block ends up inline in this
//! cdylib (cuda-oxide embeds PTX into the link section of the crate that
//! owns the inline module). The Python surface is
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

use ferrum_gpu_fft::{KernelKind, Plan};

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

include!("../../ferrum-gpu-fft-kernels/src/kernels_body.rs");

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

    // Twiddles match the plan's specialised kernel (radix-8 for N=4096,
    // radix-2 otherwise). Inverse uses the host conjugate trick, so the
    // forward-only kernel choice depends only on log_n.
    let kernel_tw = plan.kernel_twiddles();
    let mut twiddles_flat: Vec<f32> = Vec::with_capacity(kernel_tw.len() * 2);
    for c in &kernel_tw {
        twiddles_flat.push(c.re);
        twiddles_flat.push(c.im);
    }

    let dbuf_in = DeviceBuffer::from_host(&stream, &input_for_gpu)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;
    let mut dbuf_out = DeviceBuffer::<f32>::zeroed(&stream, total * 2)?;

    match plan.kernel_kind {
        KernelKind::Specialised4096 => {
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_4096(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        KernelKind::Specialised1024 => {
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_1024(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        KernelKind::Specialised256 => {
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (64, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_c2c_256(stream.as_ref(), cfg, &dbuf_in, &dbuf_tw, &mut dbuf_out)?;
        }
        _ => {
            let block_threads = core::cmp::min(n / 2, 1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: 0,
            };
            module.fft_radix2_c2c_pow2_1d_fallback(
                stream.as_ref(),
                cfg,
                &dbuf_in,
                &dbuf_tw,
                &mut dbuf_out,
                log_n,
            )?;
        }
    }

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

/// 2D FFT runner: row-FFT, transpose, row-FFT, transpose-back. Four GPU
/// launches, two ping-pong device buffers.
///
/// Input/output are flat f32 slices of length `2*N*N` (re, im interleaved,
/// row-major). `dir`: +1 forward, -1 inverse. `normalize`: divide inverse
/// output by N*N. Requires `log_n in [4, 12]` so that N is a multiple of
/// the transpose tile size (16).
fn run_fft_2d_flat(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    input_flat: &[f32],
    log_n: u32,
    dir: i32,
    normalize: bool,
) -> Result<Vec<f32>> {
    let n = 1usize << log_n;
    if input_flat.len() != n * n * 2 {
        return Err(anyhow!(
            "2D input_flat len {} != {} (2*N*N)",
            input_flat.len(),
            n * n * 2
        ));
    }
    let stream = ctx.default_stream();

    // Inverse: conjugate input on host, reuse forward kernel.
    let mut in_for_gpu: Vec<f32> = input_flat.to_vec();
    if dir < 0 {
        for chunk in in_for_gpu.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
    }

    // Twiddles for a single row FFT (length N). Reused for both passes.
    let plan_1d = Plan::new(log_n, n, false);
    let mut twiddles_flat: Vec<f32> = Vec::with_capacity((n - 1) * 2);
    for c in plan_1d.twiddles() {
        twiddles_flat.push(c.re);
        twiddles_flat.push(c.im);
    }

    let mut buf_a = DeviceBuffer::from_host(&stream, &in_for_gpu)?;
    let mut buf_b = DeviceBuffer::<f32>::zeroed(&stream, n * n * 2)?;
    let dbuf_tw = DeviceBuffer::from_host(&stream, &twiddles_flat)?;

    let block_threads = core::cmp::min(n / 2, 1024) as u32;
    let fft_cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    const TILE: u32 = 16;
    let n_u32 = n as u32;
    let trans_cfg = LaunchConfig {
        grid_dim: (n_u32 / TILE, n_u32 / TILE, 1),
        block_dim: (TILE, TILE, 1),
        shared_mem_bytes: 0,
    };

    // Pass 1: row FFT into buf_b.
    module.fft_radix2_c2c_pow2_1d_fallback(
        stream.as_ref(),
        fft_cfg,
        &buf_a,
        &dbuf_tw,
        &mut buf_b,
        log_n,
    )?;
    // Transpose buf_b -> buf_a.
    module.transpose_complex_pow2(stream.as_ref(), trans_cfg, &buf_b, &mut buf_a, log_n)?;
    // Pass 2: row FFT (column pass) buf_a -> buf_b.
    module.fft_radix2_c2c_pow2_1d_fallback(
        stream.as_ref(),
        fft_cfg,
        &buf_a,
        &dbuf_tw,
        &mut buf_b,
        log_n,
    )?;
    // Transpose back buf_b -> buf_a.
    module.transpose_complex_pow2(stream.as_ref(), trans_cfg, &buf_b, &mut buf_a, log_n)?;

    let mut host_out_flat = buf_a.to_host_vec(&stream)?;

    if dir < 0 {
        for chunk in host_out_flat.chunks_exact_mut(2) {
            chunk[1] = -chunk[1];
        }
        if normalize {
            let inv = 1.0f32 / (n as f32 * n as f32);
            for v in host_out_flat.iter_mut() {
                *v *= inv;
            }
        }
    }

    Ok(host_out_flat)
}

/// One-shot 2D variant: transient context + module.
fn run_fft_2d_flat_oneshot(
    input_flat: &[f32],
    log_n: u32,
    dir: i32,
    normalize: bool,
) -> Result<Vec<f32>> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let module = load_kernels(&ctx)?;
    run_fft_2d_flat(&ctx, &module, input_flat, log_n, dir, normalize)
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

/// 2D FFT of a square `complex64` numpy array of shape `(N, N)`.
///
/// Arguments
/// ---------
/// arr : numpy.ndarray
///     `complex64` 2D array of shape `(N, N)` where `N = 1 << log_n` and
///     `log_n in [4, 12]`. The lower bound is set by the 16-wide transpose
///     tile (N must be a multiple of 16).
/// log_n : int
///     log2(N).
/// direction : str
///     "forward" (default) or "inverse".
/// normalize : bool
///     Scale inverse output by 1/(N*N) when True.
/// device : ferrum_gpu.cuda.Device, optional
///     Persistent device handle. When supplied, the FFT reuses its
///     CudaContext + loaded module instead of building transient ones.
///
/// Returns
/// -------
/// numpy.ndarray of `complex64`, shape `(N, N)`.
#[pyfunction]
#[pyo3(signature = (arr, log_n, direction = "forward", normalize = false, device = None))]
fn fft_2d_c2c_pow2<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, num_complex::Complex32>,
    log_n: u32,
    direction: &str,
    normalize: bool,
    device: Option<PyRef<'_, Device>>,
) -> PyResult<Bound<'py, PyArray2<num_complex::Complex32>>> {
    if !(4..=12).contains(&log_n) {
        return Err(PyValueError::new_err(format!(
            "2D requires log_n in [4, 12]; got {log_n}"
        )));
    }
    let n = 1usize << log_n;
    let dims = arr.shape();
    if dims.len() != 2 || dims[0] != n || dims[1] != n {
        return Err(PyValueError::new_err(format!(
            "arr shape {:?} != ({}, {})",
            dims, n, n
        )));
    }
    let dir_i = match direction {
        "forward" => 1i32,
        "inverse" => -1i32,
        other => {
            return Err(PyValueError::new_err(format!(
                "direction must be 'forward' or 'inverse', got {other:?}"
            )));
        }
    };

    let arr_view = arr.as_slice()?;
    let mut input_flat: Vec<f32> = Vec::with_capacity(n * n * 2);
    for c in arr_view {
        input_flat.push(c.re);
        input_flat.push(c.im);
    }

    let device_handles: Option<(Arc<CudaContext>, kernels::LoadedModule)> = device
        .as_ref()
        .map(|d| (d.inner.clone(), d.module.clone()));

    let output_flat = py
        .allow_threads(|| match device_handles {
            Some((ctx, module)) => {
                run_fft_2d_flat(&ctx, &module, &input_flat, log_n, dir_i, normalize)
            }
            None => run_fft_2d_flat_oneshot(&input_flat, log_n, dir_i, normalize),
        })
        .map_err(|e| PyValueError::new_err(format!("ferrum-gpu fft 2d error: {e}")))?;

    let mut out: Vec<num_complex::Complex32> = Vec::with_capacity(n * n);
    for chunk in output_flat.chunks_exact(2) {
        out.push(num_complex::Complex::new(chunk[0], chunk[1]));
    }
    let flat = out.into_pyarray(py);
    let reshaped = flat
        .reshape([n, n])
        .map_err(|e| PyValueError::new_err(format!("reshape to (N,N): {e}")))?;
    Ok(reshaped)
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
    fft.add_function(wrap_pyfunction!(fft_2d_c2c_pow2, &fft)?)?;
    m.add_submodule(&fft)?;
    Ok(())
}
