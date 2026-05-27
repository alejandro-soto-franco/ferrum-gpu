# ferrum-gpu

Pure-Rust GPU compute substrate with Python bindings.

This is `v0.0.1` (Plan 1 of 5 toward `v0.1.0`). Today, the workspace ships:

- `ferrum-gpu-core`: `Backend` trait, `KernelArtifact`, errors. `no_std + alloc`.
- `ferrum-gpu-cuda`: `impl Backend for Cuda` over `cudarc` 0.19.
- `ferrum-gpu`: facade with `Device<B>` and `Buffer<T, B>`.
- `examples/vector-add`: end-to-end demo. Hand-written PTX dispatched through the substrate.

The Rust-source-to-PTX path through [cuda-oxide](https://github.com/NVlabs/cuda-oxide) lands in Plan 2. The FFT application (`ferrum-gpu-fft`) lands in Plan 3. The Python bindings (`ferrum-gpu` on PyPI) land in Plan 4.

## Requirements

- Linux x86_64
- CUDA Toolkit 12.x or 13.x (matching driver capability)
- NVIDIA driver compatible with the installed Toolkit
- Rust stable (`rust-toolchain.toml` pins it)

## Quick start

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

## Testing

CPU-only tests: `make test`.

GPU tests (requires CUDA + NVIDIA GPU): `make test-gpu`.

## License

Apache-2.0.
