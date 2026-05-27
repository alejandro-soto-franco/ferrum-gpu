//! Python bindings for ferrum-gpu.

#![warn(missing_docs)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

use pyo3::prelude::*;

/// Returns the crate version (smoke test).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Native extension module entry point. Exposed under `ferrum_gpu._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
