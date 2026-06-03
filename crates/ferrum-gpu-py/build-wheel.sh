#!/usr/bin/env bash
# Build the ferrum-gpu manylinux_2_28_x86_64 wheel inside the docker
# image produced by Dockerfile.manylinux.
#
# Assumes /work is mounted to the repo root.

set -euo pipefail

cd /work/crates/ferrum-gpu-py

# Target sm_80 (Ampere). cuda-oxide emits PTX ISA 7.0 for sm_80, which the
# NVIDIA driver JIT-compiles to native SASS on every Ampere-or-newer GPU
# (sm_80 .. sm_120+). Without this override, cuda-oxide auto-detects the
# build host's GPU (or fails if none is present), producing a wheel that only
# loads on that exact architecture.
#
# Do NOT use a compute_* virtual target here: cuda-oxide writes a stale
# `.version 3.2` PTX directive for compute_* targets while honouring the
# requested `.target`, yielding invalid PTX that ptxas refuses ("PTX
# .version 3.2 does not support .target ...") so the wheel fails its first
# FFT with DriverError(218). sm_* targets map the ISA version correctly.
export CUDA_OXIDE_TARGET="${CUDA_OXIDE_TARGET:-sm_80}"

# The cdylib links libcuda.so.1 (NEEDED) but the build container has no NVIDIA
# driver. The CUDA Toolkit ships a stub libcuda for link purposes; expose it as
# libcuda.so.1 so maturin's link step resolves it. No GPU is needed to BUILD
# (the runtime driver provides the real libcuda when the wheel is used).
STUBS=/usr/local/cuda/lib64/stubs
[ -e "$STUBS/libcuda.so.1" ] || ln -sf "$STUBS/libcuda.so" "$STUBS/libcuda.so.1"
export LD_LIBRARY_PATH="$STUBS:${LD_LIBRARY_PATH:-}"

# The personal fork's codegen backend is pre-built into the image (see
# Dockerfile.manylinux, which clones github.com/alejandro-soto-franco/cuda-oxide
# and builds crates/rustc-codegen-cuda). Fail loudly if it is missing rather
# than silently falling back to a different backend.
BACKEND=/root/.cargo/cuda-oxide/librustc_codegen_cuda.so
if [ ! -f "$BACKEND" ]; then
    echo "error: fork codegen backend missing at $BACKEND" >&2
    echo "       rebuild the image from Dockerfile.manylinux." >&2
    exit 1
fi

# RUSTFLAGS that drive maturin through the fork's codegen backend. Identical to
# what `cargo oxide run` sets internally, and to the local Makefile path.
export RUSTFLAGS="-Z codegen-backend=$BACKEND -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"

echo "Building wheel with CUDA_OXIDE_TARGET=$CUDA_OXIDE_TARGET"

# Build wheel into /work/dist/. abi3-py310 means one wheel works on 3.10+.
# CARGO_TARGET_DIR points at a real in-container dir because the workspace's
# /work/target is the dangling host cargo-targets symlink (ENOTDIR otherwise).
#
# `--auditwheel skip`: do NOT run auditwheel repair. The cdylib has a NEEDED
# libcuda.so.1 that the end user's NVIDIA driver provides at runtime; repair
# would instead bundle the build container's *stub* libcuda into the wheel,
# which has no real entry points and breaks every CUDA call. The wheel is
# still tagged manylinux_2_28 (it is built against the image's glibc 2.28, so
# the tag is truthful). Mirrors the local `make wheel` path.
CARGO_TARGET_DIR=/tmp/ferrum-gpu-target "$MATURIN" build --release \
    --compatibility manylinux_2_28 \
    --auditwheel skip \
    --out /work/dist

echo
echo "Wheel(s) produced:"
ls -la /work/dist/
echo
echo "auditwheel show:"
auditwheel show /work/dist/ferrum_gpu-*manylinux_2_28*.whl || true
