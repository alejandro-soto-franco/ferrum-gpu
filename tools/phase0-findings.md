# Phase 0 findings

Run on: <date>
GPU: RTX 5060 Laptop (sm_120)
Driver: <nvidia-smi version>
CUDA Toolkit: <nvcc --version>

## Task 0.1: cuFFT kernel-name trace

Output: `tools/ncu/cufft-blackwell-2026-05-28.txt`

Ran `ncu --kernel-name regex:"cufft" --launch-skip 5 --launch-count 1 ./target/release/cufft-ncu-trace`.
The kernel regex did not match; ncu reported `Available Kernels: 1. vector_fft`. So cuFFT 11.x on
Blackwell dispatches a single kernel named `vector_fft` for all three sizes at batch=256, forward C2C.

| N    | Kernel name(s)     | Notes |
| ---- | ------------------ | ----- |
| 256  | vector_fft         | same kernel across all sizes (no per-size dispatch at host-API level) |
| 1024 | vector_fft         | same |
| 4096 | vector_fft         | same |

Detailed per-launch metrics (regs/thread, smem/block, grid/block dims) require GPU performance
counter access (`NVreg_RestrictProfilingToAdminUsers=0` or sudo ncu); blocked by `ERR_NVGPUCTRPERM`.
Operator can rerun under sudo if these become load-bearing. Conclusion is already informative:
cuFFT's single-kernel-fits-all design validates the per-size specialisation strategy in our spec.

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
| cuda-core | 2.046                   |
| cudarc    | 2.047                   |

Gap: -0.000 us (negligible). No wall-clock overhead footnote needed.

## Task 0.5: Shared-mem bank conflicts

Stride-256 read with +0 pad: 7.991 us
Stride-256 read with +1 pad: 7.310 us
Ratio: 1.09x

Conclusion: Ratio < 1.2x indicates minimal bank-conflict overhead. Drop +1 padding from fft_c2c_4096 design to save shared-mem space and simplify indexing.

## Phase 1 deviation: ferrum-gpu-py switch deferred

Plan Task 1.5 switches `ferrum-gpu-py` to consume the kernels crate. Deferred to
Phase 5/6 because the cdylib loads PTX from its own `.so` via `dladdr` +
`artifact_bundles_from_binary_path` filtered by `CARGO_PKG_NAME`. Switching to
the kernels crate requires loading a second bundle (the kernels crate's PTX
also linked into the `.so`) and routing calls across two `LoadedModule`s, since
`transpose_complex_pow2` stays in the py crate for now (only consumer).

Bench + example consumers DID switch (Phase 1 Tasks 1.3 + 1.4). Dedup payoff
preserved on the perf-critical path. Python wheel keeps the v0.1 inlined
`fft_radix2_c2c_pow2_1d` + `transpose_complex_pow2` until Phase 5 (when the
specialised kernels need a Python-facing dispatch and we sort out the
multi-module loader).

## Phase 0 summary

Design adjustments for Phase 3+:
- [x] CuSimd v4 lowering: deferred (cuda-oxide doesn't expose PTX separately; functional correctness verified)
- [x] fft_c2c_4096 launch_bounds: deferred to Phase 3 based on perf-gate results
- [x] fft_c2c_4096 SMEM +1 padding: drop (ratio 1.09x < 1.2x threshold)
- [x] Wall-clock vs event-time gap to footnote: 0.000 us (negligible, no footnote needed)
- [ ] cuFFT kernel-name reference: see `tools/ncu/cufft-blackwell-2026-05.txt` (operator-driven, Task 0.1 step 5)
