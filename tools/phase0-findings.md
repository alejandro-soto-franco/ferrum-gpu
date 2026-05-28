# Phase 0 findings

Run on: <date>
GPU: RTX 5060 Laptop (sm_120)
Driver: <nvidia-smi version>
CUDA Toolkit: <nvcc --version>

## Task 0.1: cuFFT kernel-name trace

Output: `tools/ncu/cufft-blackwell-<date>.txt`

| N    | Kernel name(s)     | Notes |
| ---- | ------------------ | ----- |
| 256  | ?                  |       |
| 1024 | ?                  |       |
| 4096 | ?                  |       |

## Task 0.2: CuSimd<f32, 4> PTX lowering

Generated PTX file: `crates/ferrum-gpu-bench/cusimd_ptx_dump.ptx`
Expected `ld.global.v4.f32` instruction: ?

## Task 0.3: radix-8 register report

`-Xptxas -v` output:
- Registers per thread: ?
- Spill stores: ?
- Spill loads: ?
- Decision on `__launch_bounds__(512, 1)`: ?

## Task 0.4: Launch overhead

| Stack    | Median empty-launch (us) |
| -------- | ------------------------ |
| cuda-core | ?                       |
| cudarc    | ?                       |

Gap: ? us. Action: ?

## Task 0.5: Shared-mem bank conflicts

Stride-256 read with +0 pad: ? cycles/iter
Stride-256 read with +1 pad: ? cycles/iter
Conclusion: ?
