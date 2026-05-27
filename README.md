# ferrum-gpu

Pure-Rust GPU compute substrate with Python bindings.

This is `v0.0.4` (Plans 1-4 of 5 toward `v0.1.0`). Today, the workspace ships:

- `ferrum-gpu-core`: `Backend` trait, `KernelArtifact`, errors. `no_std + alloc`.
- `ferrum-gpu-cuda`: `impl Backend for Cuda` over `cudarc` 0.19.
- `ferrum-gpu`: facade with `Device<B>` and `Buffer<T, B>`.
- `ferrum-gpu-fft`: 1D radix-2 power-of-2 C2C FFT host scaffolding + CPU Stockham reference.
- `ferrum-gpu-py`: Python bindings (`ferrum_gpu.fft.fft_1d_c2c_pow2`) via PyO3 + maturin.
- `examples/vector-add`: end-to-end demo using hand-written PTX dispatched through the substrate.
- `examples/vector-add-cuda-oxide`: same kernel written in Rust, compiled to PTX by [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
- `examples/fft-1d-c2c`: 1D Stockham FFT kernel in Rust, verified GPU-vs-CPU across 8 cases (N from 4 to 4096, batched, forward + inverse).

Plan 5 polishes for the public release: docs, wheel distribution, additional FFT shapes.

## Requirements

- Linux x86_64
- CUDA Toolkit 13.x
- NVIDIA driver compatible with the installed Toolkit
- Rust nightly `2026-04-03` (pinned via `rust-toolchain.toml`)
- `cargo-oxide`: `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`
- Python 3.10+ with maturin + numpy + pytest (Plan 4 only)

## Quick start (Plan 1: hand-written PTX)

```bash
git clone https://github.com/alejandro-soto-franco/ferrum-gpu
cd ferrum-gpu
make example-vector-add
```

Expected:
```
vector_add: 1048576 elements verified
```

## cuda-oxide quick start (Plan 2: Rust-source vector_add)

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo oxide doctor       # one-time codegen-backend bootstrap
make example-vector-add-oxide
```

Expected:
```
vector_add (cuda-oxide): 1048576 elements verified
```

## FFT quick start (Plan 3: Stockham 1D C2C)

```bash
make example-fft
```

Runs 8 cases (N=4 through N=4096, batched, forward + inverse), each verified against a CPU Stockham reference within 1e-4 relative error.

## Python quick start (Plan 4)

```bash
python3 -m venv ~/.venvs/ferrum-gpu
source ~/.venvs/ferrum-gpu/bin/activate
pip install maturin pytest numpy
make develop                       # builds the cdylib + installs into the venv
python3 -c "
import numpy as np, ferrum_gpu as fg
arr = np.array([1+0j, 2+0j, 3+0j, 4+0j], dtype=np.complex64)
print(fg.fft.fft_1d_c2c_pow2(arr, log_n=2))
"
```

Run the pytest matrix:

```bash
make pytest
```

8 cases, each compared against `numpy.fft.fft` within 1e-4 relative error. All pass on the user's RTX 5060 Laptop.

## Testing

CPU-only tests: `make test`.

GPU tests + all examples + pytest (requires CUDA + NVIDIA GPU): `make verify-all`.

## License

Apache-2.0.
