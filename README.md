# ferrum-gpu

Pure-Rust GPU compute substrate with Python bindings.

This is `v0.0.2` (Plans 1-2 of 5 toward `v0.1.0`). Today, the workspace ships:

- `ferrum-gpu-core`: `Backend` trait, `KernelArtifact`, errors. `no_std + alloc`.
- `ferrum-gpu-cuda`: `impl Backend for Cuda` over `cudarc` 0.19.
- `ferrum-gpu`: facade with `Device<B>` and `Buffer<T, B>`.
- `examples/vector-add`: end-to-end demo using hand-written PTX dispatched through the substrate.
- `examples/vector-add-cuda-oxide`: same kernel written in Rust, compiled to PTX by [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

The FFT application (`ferrum-gpu-fft`) lands in Plan 3. The Python bindings (`ferrum-gpu` on PyPI) land in Plan 4.

## Requirements

- Linux x86_64
- CUDA Toolkit 13.x (the cuda-oxide example expects 13.x; the hand-written-PTX example works with 12.x or 13.x)
- NVIDIA driver compatible with the installed Toolkit
- Rust nightly `2026-04-03` (pinned via `rust-toolchain.toml`)
- For the cuda-oxide example: install `cargo-oxide` via `cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide`

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

## cuda-oxide quick start (Plan 2: Rust-source kernel)

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

## Testing

CPU-only tests: `make test`.

GPU tests (requires CUDA + NVIDIA GPU): `make test-gpu`.

## License

Apache-2.0.
