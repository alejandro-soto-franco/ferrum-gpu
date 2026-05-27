"""ferrum-gpu Python bindings.

Re-exports the native extension module for ergonomic access. The `fft`
submodule lands in P4T3.
"""

from ferrum_gpu._native import version

__all__ = ["version"]
