"""ferrum_gpu.fft submodule.

The native FFT functions are exposed as attributes of the `fft` submodule
inside the `_native` PyO3 cdylib. PyO3 submodules are not importable via
`from ferrum_gpu._native.fft import ...` (they are not Python packages on
disk), so this wrapper pulls them via attribute access.
"""

from ferrum_gpu._native import fft as _fft_native

fft_1d_c2c_pow2 = _fft_native.fft_1d_c2c_pow2

__all__ = ["fft_1d_c2c_pow2"]
