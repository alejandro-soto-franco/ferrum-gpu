#!/usr/bin/env bash
# Build the ferrum-gpu manylinux_2_28_x86_64 wheel inside the docker
# image produced by Dockerfile.manylinux.
#
# Assumes /work is mounted to the repo root.

set -euo pipefail

cd /work/crates/ferrum-gpu-py

# RUSTFLAGS that cargo-oxide sets internally for builds that drive its
# codegen backend. Identical to what cargo oxide run sets.
export RUSTFLAGS="-Z codegen-backend=/root/.cargo/cuda-oxide/librustc_codegen_cuda.so -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"

# Build wheel into /work/dist/. abi3-py310 means one wheel works on 3.10+.
"$MATURIN" build --release \
    --manylinux 2_28 \
    --out /work/dist

echo
echo "Wheel(s) produced:"
ls -la /work/dist/
echo
echo "auditwheel show:"
auditwheel show /work/dist/ferrum_gpu-*.whl || true
