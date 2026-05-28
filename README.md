# ferrum-gpu

Pure-Rust GPU compute substrate with Python bindings. FFT kernels run on NVIDIA GPUs today via [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Rust source compiled to PTX, no CUDA C). Cross-vendor support via spirv-oxide → Vulkan is the v0.2 roadmap.

`v0.1.0` is live on PyPI:

```bash
pip install ferrum-gpu
# or, faster:
uv pip install ferrum-gpu
```

```python
import numpy as np
import ferrum_gpu as fgpu

dev = fgpu.cuda.Device(0)
arr = np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype=np.complex64)
print(fgpu.fft.fft_1d_c2c_pow2(arr, log_n=2, device=dev))
# [10+0j, -2+2j, -2+0j, -2-2j]
```

29 GPU integration tests verify the 1D and 2D Stockham FFTs against `numpy.fft.fft` / `numpy.fft.fft2` (1D: 16 cases, 2D: 13 cases) within 1e-3 to 1e-4 relative error.

The workspace ships:

- `ferrum-gpu-core`: `Backend` trait, `KernelArtifact`, errors. `no_std + alloc`.
- `ferrum-gpu-cuda`: `impl Backend for Cuda` over `cudarc` 0.19.
- `ferrum-gpu`: facade with `Device<B>` and `Buffer<T, B>`.
- `ferrum-gpu-fft`: 1D + 2D radix-2 power-of-2 C2C FFT host scaffolding + CPU Stockham reference.
- `ferrum-gpu-py`: Python bindings via PyO3 + maturin. `ferrum_gpu.cuda.Device(0)` persistent handle + `ferrum_gpu.fft.fft_1d_c2c_pow2` + `ferrum_gpu.fft.fft_2d_c2c_pow2`.
- `ferrum-gpu-bench`: cuFFT comparison binary (1D, batched).
- `examples/vector-add`: end-to-end demo using hand-written PTX through the substrate.
- `examples/vector-add-cuda-oxide`: same kernel in Rust, compiled to PTX by cuda-oxide.
- `examples/fft-1d-c2c`: 1D Stockham FFT in Rust, GPU-vs-CPU on 8 cases (N from 4 to 4096, batched, forward + inverse).

## Runtime requirements (wheel users)

- Linux x86_64 with glibc >= 2.34 (Ubuntu 22.04+, RHEL 9+, Fedora 36+, etc.)
- NVIDIA driver supporting CUDA 13.x (driver 580+)
- Python 3.10+

The wheel does not bundle `libcuda`; users need the NVIDIA driver installed.

## Source build requirements (developers)

- Linux x86_64
- CUDA Toolkit 13.x
- NVIDIA driver compatible with the installed Toolkit
- Rust nightly `2026-04-03` (pinned via `rust-toolchain.toml`)
- `cargo-oxide`: `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`
- For the Python bindings: Python 3.10+ with `maturin` + `pytest` + `numpy`

## Quick start: vector-add via hand-written PTX

```bash
git clone https://github.com/alejandro-soto-franco/ferrum-gpu
cd ferrum-gpu
make example-vector-add
```

Expected:
```
vector_add: 1048576 elements verified
```

## Quick start: vector-add via Rust source + cuda-oxide

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo oxide doctor       # one-time codegen-backend bootstrap
make example-vector-add-oxide
```

Expected:
```
vector_add (cuda-oxide): 1048576 elements verified
```

## Quick start: 1D Stockham FFT

```bash
make example-fft
```

Runs 8 cases (N=4 through N=4096, batched, forward + inverse), each verified against a CPU Stockham reference within 1e-4 relative error.

## Python development

[`uv`](https://github.com/astral-sh/uv) is the Python package manager written in Rust by Astral; it is a drop-in faster replacement for `pip` + `venv`. The Makefile targets and the wheel install path work the same on `pip` for users who prefer it.

```bash
uv venv ~/.venvs/ferrum-gpu
source ~/.venvs/ferrum-gpu/bin/activate
uv pip install maturin pytest numpy
make develop                       # builds the cdylib + installs into the venv
python3 -c "
import numpy as np, ferrum_gpu as fgpu
arr = np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype=np.complex64)
print(fgpu.fft.fft_1d_c2c_pow2(arr, log_n=2))
"
```

Pip equivalent:

```bash
python3 -m venv ~/.venvs/ferrum-gpu
source ~/.venvs/ferrum-gpu/bin/activate
pip install maturin pytest numpy
make develop
```

Run the pytest matrix:

```bash
make pytest
```

29 cases (16 1D + 13 2D), each compared against `numpy.fft` within 1e-3 to 1e-4 relative error.

## Performance

`make bench` runs `ferrum-gpu-bench`, which times the in-tree
cuda-oxide-compiled Stockham radix-2 power-of-2 C2C kernel against cuFFT
(via cudarc 0.19's `cufft` feature) for batched 1D transforms at
N in {256, 1024, 4096}, batch = 256, 100 trials per size + 10-trial
warmup. Per-batch microseconds, measured on an RTX 5060 Laptop (sm_120):

| N    | ferrum_us | cufft_us | ratio |
| ---- | --------- | -------- | ----- |
| 256  | 0.089     | 0.016    | 5.52  |
| 1024 | 0.162     | 0.059    | 2.72  |
| 4096 | 0.548     | 0.080    | 6.86  |

The Stockham kernel is a single-block-per-FFT reference implementation
with no radix-4 or warp-specialised stages, so cuFFT's vendor-tuned plan
wins outright at these sizes. Closing the gap is on the v0.2 roadmap.

## Testing

CPU-only tests: `make test`.

GPU tests + all examples + pytest (requires CUDA + NVIDIA GPU): `make verify-all`.

## Publishing a new release

Maintainers only. The public wheel is built locally via maturin (manylinux_2_34 tag forced via `--compatibility`, `libcuda` excluded via `--auditwheel skip`) then attached to a GitHub Release. The `release.yml` workflow downloads the asset and uploads to PyPI via OIDC trusted publishing.

```bash
make wheel             # builds dist/ferrum_gpu-*-manylinux_2_34_x86_64.whl
gh release create vX.Y.Z dist/*.whl --title "vX.Y.Z" --notes "..."
gh workflow run release.yml --field release_tag=vX.Y.Z --field target_index=pypi
```

For a glibc 2.28 baseline (RHEL 8, Ubuntu 20.04), use `make wheel-manylinux` instead; it runs the build inside a manylinux_2_28 Docker image.

## License

Apache-2.0.
