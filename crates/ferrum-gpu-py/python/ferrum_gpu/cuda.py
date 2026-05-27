"""ferrum_gpu.cuda submodule.

The native `Device` class is exposed as an attribute of the `cuda` submodule
inside the `_native` PyO3 cdylib. PyO3 submodules are not importable via
`from ferrum_gpu._native.cuda import ...` (they are not Python packages on
disk), so this wrapper pulls the class via attribute access.
"""

from ferrum_gpu._native import cuda as _cuda_native

Device = _cuda_native.Device

__all__ = ["Device"]
