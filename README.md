# ferrum-gpu

Pure-Rust GPU compute substrate with Python bindings.

This is `v0.0.3` (Plans 1-3 of 5 toward `v0.1.0`). Today, the workspace ships:

- `ferrum-gpu-core`: `Backend` trait, `KernelArtifact`, errors. `no_std + alloc`.
- `ferrum-gpu-cuda`: `impl Backend for Cuda` over `cudarc` 0.19.
- `ferrum-gpu`: facade with `Device<B>` and `Buffer<T, B>`.
- `ferrum-gpu-fft`: 1D radix-2 power-of-2 C2C FFT host scaffolding + CPU Stockham reference.
- `examples/vector-add`: end-to-end demo using hand-written PTX dispatched through the substrate.
- `examples/vector-add-cuda-oxide`: same kernel written in Rust, compiled to PTX by [cuda-oxide](https://github.com/NVlabs/cuda-oxide).
- `examples/fft-1d-c2c`: 1D Stockham FFT kernel written in Rust, verified GPU-vs-CPU on 8 size/direction cases (N from 4 to 4096, batched, forward + inverse).

The Python bindings (`ferrum-gpu` on PyPI) land in Plan 4.

## Requirements

- Linux x86_64
- CUDA Toolkit 13.x (the cuda-oxide examples expect 13.x; the hand-written-PTX example works with 12.x or 13.x)
- NVIDIA driver compatible with the installed Toolkit
- Rust nightly `2026-04-03` (pinned via `rust-toolchain.toml`)
- For the cuda-oxide and FFT examples: install `cargo-oxide` via `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`

## Quick start (Plan 1: hand-written PTX)

```bash
git clone https://github.com/alejandro-soto-franco/ferrum-gpu
cd ferrum-gpu
make example-vector-add
```

Expected output:
```
device opened; preparing 1048576 elements
loading hand-written PTX (target=sm_70)
launching kernel
vector_add: 1048576 elements verified
```

## cuda-oxide quick start (Plan 2: Rust-source vector_add)

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo oxide doctor       # one-time codegen-backend bootstrap
make example-vector-add-oxide
```

Expected output:
```
loading cuda-oxide-compiled vector_add module
launching kernel
vector_add (cuda-oxide): 1048576 elements verified
```

## FFT quick start (Plan 3: Stockham 1D C2C)

```bash
make example-fft
```

Runs 8 sizes from N=4 through N=4096 plus a batched run plus an inverse round-trip, comparing each GPU result against a CPU Stockham reference within `1e-4` relative error. Any mismatch exits non-zero.

Expected output:
```
ok  N=4 fwd (N=4, batch=1)
ok  N=8 fwd (N=8, batch=1)
ok  N=64 fwd (N=64, batch=1)
ok  N=256 fwd (N=256, batch=1)
ok  N=1024 fwd (N=1024, batch=1)
ok  N=4096 fwd (N=4096, batch=1)
ok  N=256 fwd batch=8 (N=256, batch=8)
ok  N=256 inv normalize (N=256, batch=1)

fft-1d-c2c: 8/8 cases verified
```

## Testing

CPU-only tests: `make test`.

GPU tests + all examples (requires CUDA + NVIDIA GPU): `make verify-all`.

## License

Apache-2.0.
