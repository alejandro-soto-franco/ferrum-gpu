#!/usr/bin/env bash
# Re-establish a working nvcc on Fedora (glibc) + CUDA 13.x.
#
# CUDA 13.1's `crt/math_functions.h` declares `rsqrt`/`rsqrtf` WITHOUT
# `noexcept`, while modern glibc's `bits/mathcalls.h` declares them WITH it.
# Under C++17 (nvcc's default front-end, EDG/cudafe) that mismatch is a hard
# error -> "exception specification is incompatible". nvcc then cannot compile
# ANY CUDA translation unit that pulls the device math headers.
#
# This adds `noexcept` to the two CUDA declarations so they match glibc.
# Idempotent (safe to re-run) and keeps a one-time backup.
#
# WHEN TO RUN: after any CUDA toolkit (re)install, including the one triggered
# by a Fedora major bump (e.g. F43 -> F44) that rebuilds the NVIDIA kernel
# module. The reinstall overwrites the header and reverts this fix.
#
# Needs root (writes under /usr/local/cuda). Run:  sudo tools/fix-cuda-glibc-headers.sh
# (or with a GUI askpass:  SUDO_ASKPASS=/path/to/askpass sudo -A tools/fix-cuda-glibc-headers.sh)
set -euo pipefail

# Locate the header across common CUDA install layouts.
CANDIDATES=(
    "${CUDA_HOME:-}/targets/x86_64-linux/include/crt/math_functions.h"
    "${CUDA_PATH:-}/targets/x86_64-linux/include/crt/math_functions.h"
    "/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h"
    "/opt/cuda/targets/x86_64-linux/include/crt/math_functions.h"
)
HDR=""
for c in "${CANDIDATES[@]}"; do
    [ -n "$c" ] && [ -f "$c" ] && { HDR="$c"; break; }
done
if [ -z "$HDR" ]; then
    echo "error: could not find crt/math_functions.h; set CUDA_HOME and re-run." >&2
    exit 1
fi
echo "Header: $HDR"

if [ ! -f "$HDR.bak-precudafix" ]; then
    cp -p "$HDR" "$HDR.bak-precudafix"
    echo "Backed up -> $HDR.bak-precudafix"
fi

changed=0
patch_one() {
    # $1 = the function signature substring up to the closing paren, e.g. 'rsqrt(double x)'
    if grep -qE "$1 noexcept;" "$HDR"; then
        echo "already patched: $1"
    elif grep -qE "$1;" "$HDR"; then
        sed -i -E "s/($1);/\1 noexcept;/" "$HDR"
        echo "patched: $1 -> noexcept"
        changed=1
    else
        echo "note: '$1;' not found (header layout differs?) — skipping"
    fi
}

patch_one 'rsqrt\(double x\)'
patch_one 'rsqrtf\(float x\)'

echo "---"
grep -nE 'rsqrtf?\(.*\) noexcept' "$HDR" || true
[ "$changed" -eq 1 ] && echo "done (modified)." || echo "done (no changes needed)."
