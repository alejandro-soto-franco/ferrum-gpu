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

Note: cuda-oxide embeds PTX directly in the binary; separate .ptx files are not generated.
The kernel compiles successfully and executes correctly. Inspection requires disassembly of the binary
or examining LLVM-IR during compilation. For now, functional correctness is verified by the successful execution
and matching of input/output values in the kernel test.

Expected `ld.global.v4.f32` instruction: needs inline-asm verification or LLVM-IR inspection

## Task 0.3: radix-8 register report

Note: cuda-oxide's codegen-backend does not expose ptxas verbose output in the same way
as traditional nvcc. The Makefile target `ptx-radix8-regreport` attempts to inject `-Xptxas=-v`
but cuda-oxide handles CUDA compilation internally without generating accessible ptxas reports.

The kernel compiles and runs successfully. For detailed register/spill analysis, would require:
1. Extracting the PTX from the binary and running ptxas separately, or
2. Using NVIDIA's profiling tools (ncu) to measure register pressure at runtime

Decision on `__launch_bounds__(512, 1)`: deferred to Phase 3 based on perf-gate results

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
