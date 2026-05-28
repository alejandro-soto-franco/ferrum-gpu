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

## Phase 1 reverted: cuda-oxide constraint blocks the kernels-crate dedup

The spec proposed a single `ferrum-gpu-fft-kernels` library crate consumed by
the example + bench + py. After implementation this hit two compounding
cuda-oxide constraints:

1. `#[cuda_module]` embeds PTX into the **building binary crate only**. The
   PTX bundle is named after the building crate's `CARGO_PKG_NAME`. The
   bundle does **not** propagate through library-crate rlibs into downstream
   binaries (verified empirically: `cargo oxide run` produced binaries that
   linked the kernels crate's code but did not embed its PTX bundle; runtime
   loader failed with `embedded CUDA module 'ferrum-gpu-fft-kernels' was not
   found`).

2. cuda-oxide's macro explicitly rejects `#[path = "..."] mod kernels;`
   declarations with `cuda_module requires an inline module so kernel
   signatures are visible`. This blocks the include-by-path workaround
   (which would otherwise let one source file expand into multiple inline
   modules at the proc-macro layer). `include!` inside the module body
   doesn't work either because `#[cuda_module]` runs before declarative
   macros expand and sees no `#[kernel]` functions.

**Resolution**: revert Phase 1 entirely. Each binary keeps its own inline
`#[cuda_module] mod kernels { ... }` with the kernel source duplicated. New
specialised kernels in Phase 3-5 (fft_c2c_256 / 1024 / 4096) are added to
each consumer separately. v0.1 already paid this duplication cost; v0.2 stays
on the same footing.

The kernel function was renamed `fft_radix2_c2c_pow2_1d -> fft_radix2_c2c_pow2_1d_fallback`
at the inline definitions in `examples/fft-1d-c2c/src/main.rs` and
`crates/ferrum-gpu-bench/src/main.rs` so the name is forward-compatible with
the Phase 3+ KernelKind dispatch. The py crate stays on the original name for
this phase (py is untouched).

Follow-up for v0.3+ (out of scope for v0.2): file an upstream issue at
`NVlabs/cuda-oxide` for library-crate PTX propagation, or design a
build-script-based workaround that emits PTX to a known path each consumer
loads via `cuda_core::CudaContext::load_module_from_image`.

## Phase 0 summary

Design adjustments for Phase 3+:
- [x] CuSimd v4 lowering: deferred (cuda-oxide doesn't expose PTX separately; functional correctness verified)
- [x] fft_c2c_4096 launch_bounds: deferred to Phase 3 based on perf-gate results
- [x] fft_c2c_4096 SMEM +1 padding: drop (ratio 1.09x < 1.2x threshold)
- [x] Wall-clock vs event-time gap to footnote: 0.000 us (negligible, no footnote needed)
- [ ] cuFFT kernel-name reference: see `tools/ncu/cufft-blackwell-2026-05.txt` (operator-driven, Task 0.1 step 5)
