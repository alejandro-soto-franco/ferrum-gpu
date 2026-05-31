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

# Bootstrap the cuda-oxide codegen backend here, at `docker run` time, rather
# than in the Dockerfile. cargo-oxide is dynamically linked against
# libcuda.so.1, which is NOT present at image-build time (the build container
# has no NVIDIA driver), so a `cargo oxide setup` baked into the image fails to
# even start the binary. The CUDA Toolkit ships a stub libcuda for link/load
# purposes; pointing the loader at it lets cargo-oxide run and compile the
# backend (.so lands in /root/.cargo/cuda-oxide/). No GPU is needed for the
# compile. Crucially this runs BEFORE RUSTFLAGS is set: the backend must be
# built with the default codegen backend, not with a `-Z codegen-backend=`
# flag pointing at the .so we are about to create.
# `cargo oxide setup` must run from inside a cuda-oxide project, and it both
# builds and discovers the backend .so at cwd/target. It cannot run in /work
# (the workspace's /work/target is the dangling host cargo-targets symlink,
# ENOTDIR) and CARGO_TARGET_DIR does not help (setup discovers at cwd/target
# regardless). So scaffold a throwaway project under /tmp, which has a real,
# writable target, and run setup there; it installs the backend globally to
# /root/.cargo/cuda-oxide/librustc_codegen_cuda.so.
if [ ! -f /root/.cargo/cuda-oxide/librustc_codegen_cuda.so ]; then
    STUBS=/usr/local/cuda/lib64/stubs
    [ -e "$STUBS/libcuda.so.1" ] || ln -sf "$STUBS/libcuda.so" "$STUBS/libcuda.so.1"
    echo "Bootstrapping cuda-oxide codegen backend (stub libcuda)…"
    rm -rf /tmp/oxide-bootstrap
    ( cd /tmp && LD_LIBRARY_PATH="$STUBS:${LD_LIBRARY_PATH:-}" cargo oxide new oxide-bootstrap )
    ( cd /tmp/oxide-bootstrap && LD_LIBRARY_PATH="$STUBS:${LD_LIBRARY_PATH:-}" cargo oxide setup )
fi

# RUSTFLAGS that cargo-oxide sets internally for builds that drive its
# codegen backend. Identical to what cargo oxide run sets. Set AFTER the
# backend bootstrap above.
export RUSTFLAGS="-Z codegen-backend=/root/.cargo/cuda-oxide/librustc_codegen_cuda.so -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"

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
