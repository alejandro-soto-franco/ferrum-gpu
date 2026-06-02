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

## fft_c2c_4096 iteration log (Phase 3 / Task 3.5)

Gate target: `ferrum_event_us <= 0.9 * cufft_event_us` at N=4096, batch=256.
Measured with `perf-gate` (alternating ferrum/cuFFT trials). Clock lock
unavailable (needs root; `clocks.base.graphics` not a queryable field on
driver 580.159.03), so numbers carry DVFS noise — the alternating design
controls for the asymmetric part. N=256/1024 stay on the radix-2 fallback
(specialised kernels are Phases 4-5) and are expected MISSes here.

| Iter | Change | N=4096 ferrum_us | cufft_us | ratio | keep? |
| ---- | ------ | ---------------- | -------- | ----- | ----- |
| 0 (baseline) | radix-8, dual 64KB ping-pong SMEM | 0.776 | 0.108 | 7.15 | n/a |
| 1 | single 32KB SMEM buffer, register-resident butterflies (read->sync->write->sync) | 0.690 | 0.127 | 5.45 | keep (ferrum -11%; halves SMEM, lifts 1-block/SM cap) |
| 2 | unroll 8-pt gather/scatter to named scalars (kill runtime array indexing) | 0.457 | 0.116 | 3.92 | keep (ferrum -34%; cumulative -41%) |
| 3 | inline dft8 as a tuple macro (was a non-inlined `.func` w/ by-ptr [f32;16]) | 0.292 | 0.077 | 3.80 | keep (st.local 16->0, all spills gone; ratio -3%, abs -36% is mostly clock boost) |

The ratio is the clock-invariant metric (no clock lock available); absolute
us swing with DVFS. Ratio history 7.15 -> 5.45 -> 3.92 -> 3.80. The big wins
were the SMEM-halving (iter 1) and spill-elimination (iter 2); iter 3's
remaining spills were off the critical path so it bought little ratio, but the
kernel now has ZERO local-memory traffic and dft8 fully inlined.

Failed attempt (not kept): `f32::mul_add` to force `fma.rn.f32`. cuda-oxide's
codegen does not lower `llvm.fma.f32` (only its own `cuda_device` intrinsics),
leaving an unresolved extern -> `nvJitLink error 4`. FMA contraction is
therefore unreachable from Rust source on this backend; ~10-20% left on the
table. Tracked as an upstream codegen gap alongside the no-auto-FMA and
array-in-local-memory issues.

> **CORRECTION (2026-06-02): the above is STALE — `f32::mul_add` works.** At the
> pinned rev `6ed9938`, cuda-oxide lowers the Rust FMA intrinsics to libdevice:
> `mir-lower/src/convert/ops/call.rs:271` maps `FmaF32 | FmuladdF32 ->
> "__nv_fmaf"` (and `FmaF64 | FmuladdF64 -> "__nv_fma"`). Chain:
> `f32::mul_add` -> `core::intrinsics::fmaf32` -> `__nv_fmaf`, which NVVM links +
> inlines to a single `fma.rn.f32` (cuda-oxide compiles via NVVM/libdevice).
> The original failure was almost certainly `__nv_fmaf` going unlinked in a
> kernel that used no other libdevice function, not a missing lowering.
> VERIFIED: a `mul_add` probe in the `vector-add-cuda-oxide` kernel compiled,
> JIT-linked, and verified 1,048,576 elements (`a*2+b`) correctly. So FMA is
> reachable in 100% pure compiler-lowered Rust (`re*wr - im*wi` ->
> `re.mul_add(wr, -(im*wi))`); the ~10-20% is NOT codegen-blocked. The existing
> 256/1024/4096 butterflies can take this win directly. No `asm!`, no fork.

Status: codegen-level spill recovery is done (the cuda-oxide-is-bad-at-PTX
hypothesis is confirmed and largely mitigated at source level). Remaining
~3.8x is FMA (codegen-blocked) + structural (1 FFT/block, 3 barriers/run,
occupancy). Closing it needs the "bridge": hand-PTX or nvcc-compiled PTX for
this kernel loaded via cuda-core, OR upstream codegen FMA/optimisation support.

### Task 3.5c: nvcc ceiling experiment — BLOCKED by toolchain

Wrote `tools/radix8_ceiling.cu` (identical radix-8 algorithm in CUDA-C, to
compile with nvcc -O3 and time vs cuFFT) to decide codegen-gap vs algorithmic-
gap. It does not compile on this box:

- nvcc 13.1 + Fedora 43 glibc clash: `bits/mathcalls.h` declares
  `rsqrt`/`rsqrtf` `noexcept(true)`, CUDA's `crt/math_functions.h` declares
  them without -> "exception specification is incompatible" (a hard EDG error,
  no suppressible number; independent of `-std`, `-ccbin gcc/clang`).
- clang 21 `-x cuda` can't use CUDA 13's headers (`texture_fetch_functions.h`
  removed in 13).
- Device-only `nvcc -ptx` clears it for a trivial kernel but the real kernel
  (`__shared__` etc.) pulls the conflicting headers again.

Implication: nvcc cannot build CUDA on this host without patching system CUDA
headers, so a *nvcc-compiled-PTX* bridge cannot be produced here either.
cuda-oxide works only because it bypasses nvcc/cudafe with its own LLVM +
libdevice pipeline.

UPDATE: unblocked nvcc by patching `crt/math_functions.h` to add `noexcept`
to `rsqrt`/`rsqrtf` (backup at `*.bak-precudafix`; a Fedora 44 + CUDA reinstall
will reset it — re-apply or update to a CUDA toolkit that ships the fix).

### DECISIVE RESULT: the gap is ALGORITHMIC, not codegen

Built and ran `tools/radix8_ceiling.cu` (nvcc -O3, same algorithm). Stable
over 3 runs:

| Build                         | us/FFT | ratio vs cuFFT |
| ----------------------------- | ------ | -------------- |
| cuda-oxide (ours, iter 3)     | ~      | 3.80           |
| nvcc -O3 (same algorithm)     | ~0.284 | **3.45**       |
| cuFFT `vector_fft`            | ~0.082 | 1.00           |

nvcc's PTX is fully optimised: 7 `fma.rn.f32`, 0 leftover mul/add/sub, 0
spills, 3 barriers. Yet the *same algorithm* compiled perfectly is still
**3.45x slower than cuFFT**. Our cuda-oxide kernel (3.80) is within ~10% of
that ceiling.

Conclusions:
1. cuda-oxide is NOT the bottleneck anymore. After the spill fixes (iters
   1-3) it trails optimal codegen by only ~10% (the FMA + minor residue). The
   substrate strategy is sound; a nvcc/hand-PTX bridge would buy ~10%, not
   close the gate. NOT worth building for perf.
2. cuFFT's 3.4x advantage is its KERNEL DESIGN, not its compiler: mixed-radix,
   multiple FFTs per block, register blocking, far fewer shared-memory round
   trips / barriers than one-4096-FFT-per-512-thread-block with 3 barriers.
3. The 0.9x gate at N=4096 is unreachable without a fundamental algorithmic
   redesign (and beating cuFFT even then is research-grade). Ship Phase 3 with
   the gate documented as a tracked target per spec Section 5.3.

Further pure-source micro-opts (Task 3.5a: launch_bounds, dual-buffer, SMEM
twiddles) are bounded above by the nvcc 3.45 ceiling, so they cannot reach
0.9x either. Stop the perf loop here; correctness + the ~2x codegen recovery
are the shipped wins.

### Redesign spike: warp-shuffle FFT (the path to actually beat cuFFT)

The 3.4x is algorithmic, so the redesign targets the algorithm: a four-step
(64x64) FFT with warp-resident register sub-FFTs (shuffle butterflies) and a
single shared-memory transpose, instead of 4 shared-memory round trips.

Building block proven (`warp_fft32` kernel + `warp-fft-spike` bin, CPU model
`ferrum_gpu_fft::warp_fft::warp_fft32_model`):
- One warp computes a 32-pt C2C FFT entirely in registers, exchanging radix-2
  DIF butterfly partners via `shfl_xor_f32` (cuda-device cooperative_groups),
  zero shared memory. Output written to bit-reversed position.
- Correctness: **max_rel_err 0.00e0** vs CPU model (bit-exact).
- Throughput: **19.7 Gpt/s** (0.00163 us/FFT, 65536 FFTs), vs ~6 Gpt/s for the
  shared-memory radix-8 4096 kernel and ~38 Gpt/s for cuFFT. ~3x the shared
  approach, with no tuning and no four-step composition yet.

Confirms: `shfl_xor_f32` lowers and runs correctly through cuda-oxide; the
warp-register FFT is the right structure.

### Four-step 64x64 GPU kernel: CORRECT but not yet competitive

Built `fft_c2c_4096_4step` (`four_step_body.rs` + `four-step-spike` bin),
CPU-modelled first (`warp_fft::four_step_model`, verified == radix-2). On GPU:

- Correct: max_rel_err 1.36e-4 vs radix-2 CPU reference.
- Perf (ratio vs cuFFT, batch=256): 256 thr/block 10.2 -> +65-stride SMEM pad
  9.97 -> 1024 thr/block (32 warps, 2 waves) 8.30 -> branchless butterfly
  (no warp divergence) 8.01. **Still worse than the tuned shared-memory
  radix-8 kernel (3.80).** Each lever (bank conflicts, block parallelism,
  divergence) gave only a small gain -> none is the dominant cost; profiling
  is required to find it.

So the four-step as composed is a regression, not a win. The 32-pt spike was
fast (19.7 Gpt/s) because of massive parallelism (65536 warps) and no
shared/sync/twiddle-table overhead; the single-block 4096 four-step has the
opposite profile: only 256 heavy blocks, a full 4096-complex shared load, two
1024-thread barriers, a 4096-entry global W_4096 table read scattered per
butterfly, and DIVERGENT warp butterflies (`if lane & d`). Diagnosing which of
these dominates needs `ncu` perf counters (root-gated here:
`ERR_NVGPUCTRPERM`), so further blind iteration is low-value.

DECISION: keep the radix-8 kernel as the production fft_c2c_4096. The
four-step is committed as a documented research artifact (NOT wired into any
consumer). Closing the cuFFT gap from here is a profiler-guided,
multi-session effort. Concrete next levers when resumed:
- profile under `ncu` (set `NVreg_RestrictProfilingToAdminUsers=0` or sudo) to
  find the actual bottleneck before changing anything;
- branchless warp butterfly (arithmetic select instead of `if lane & d`) to
  kill warp divergence;
- stage the W_4096 twiddles in shared or compute on-chip (sincos) instead of a
  scattered 4096-entry global table;
- consider the two-kernel four-step (full batch parallelism per pass) vs the
  single-kernel on-chip tradeoff (extra global round-trip ~16 MB/batch);
- `asm!` `fma.rn.f32` for the inner butterflies (cuda-oxide emits none).

### PTX-quality probe (Task 3.5b) — cuda-oxide IS a major bottleneck

Dumped the emitted PTX (repo-root `<binary>.ptx`, e.g. `kernel_cross_check.ptx`)
and inspected the `fft_c2c_4096` entry.

Baseline (iter 1) codegen pathologies:
- **`[f32; 16]` arrays indexed by a runtime variable were placed in LOCAL
  (DRAM) memory.** The kernel round-tripped all 16 floats every butterfly:
  36 `st.local` + 18 `ld.local`. Iter-2 fix (unroll to constant-index named
  scalars) cut this to 16 write-only `st.local` + **0 `ld.local`** — no spill
  reloads in the hot path. This alone bought the 34% in iter 2.
- **No FMA contraction.** `re*wr - im*wi` emits `mul.rn` + `sub.rn` instead of
  `mul` + `fma`. Still 28 `mul.rn` + 7 `add.rn` + 7 `sub.rn`, 0 `fma` after
  iter 2. Candidate for iter 3 (try `f32::mul_add`).

Upstream check (requested): pinned `6ed9938`; `origin/main` is `396c76a`, only
ONE commit ahead — *"catch device-codegen panics, emit our own diagnostic"*,
a diagnostics change, not a codegen-quality fix. No upstream work on FMA /
array-promotion / register allocation. So source-level workarounds (constant
indexing, scalarisation) are the right lever; the pin stays at `6ed9938`.

## Phase 4: fft_c2c_1024

Tried 1024 = 32x32 four-step first (reuses the fast 32-pt warp FFT, 32 warps =
32 sub-FFTs so no wave loop, 8 KiB shared). Correct (max_rel_err 3.08e-5).
Perf under LOCKED clock (1500 MHz, fair head-to-head): four-step 1024 = 7.1x
vs cuFFT, but the radix-2 **fallback 1024 = 6.34x** — so the four-step LOSES
to the simple fallback (the earlier unlocked "4.96 < 5.88 win" was clock
noise; locking the clock via the new sudo/zenity path settled it). Same
verdict as the 4096 four-step: single-block-per-FFT four-step underperforms.

Locked-clock baseline (perf-gate): fallback 256 = 6.34x, fallback 1024 =
6.34x, radix-8 4096 = 3.78x. The radix-8 4096 (shared-memory Stockham,
scalarized gather/scatter, inline dft8 macro, single buffer) is the best
ferrum kernel and the pattern to replicate: its win over the fallback is the
4 stages (fewer barriers) vs the fallback's 10-12 radix-2 stages.

DECISION: ship Phase 4 as a radix-4 shared-memory Stockham kernel (1024 = 4^5,
5 stages) mirroring the radix-8 structure, not the warp four-step. The 1024
four-step is kept as a documented artifact (NOT wired); radix-4 is the
production Specialised1024.

RESULT: `fft_c2c_1024` (radix-4, single 8 KiB buffer, scalarized gather/
scatter, inline dft4) under LOCKED clock = **2.15x** vs cuFFT, vs the fallback's
6.34x and even better than the radix-8 4096 (3.74x) — the smaller transform's
8 KiB shared gives much better occupancy than 4096's 32 KiB. CPU-verified
(`cpu_radix4`, radix-2 ref, log_n 2..10), GPU cross-check 3.41e-5, example
8/8, pytest 29/29. Wired into all consumers via `KernelKind::Specialised1024`
+ `Plan::kernel_twiddles` (twiddles_radix4). This is the best ferrum kernel
and the closest to the 0.9x gate.

Locked-clock scoreboard (ratio vs cuFFT): N=256 fallback 6.34, N=1024 radix-4
**2.15**, N=4096 radix-8 3.74. Phase 5 (fft_c2c_256) should use the radix-4
pattern (256 = 4^4, 4 stages) — likely the best size yet given even smaller
shared.

## Phase 5: fft_c2c_256

`fft_c2c_256` (radix-4, 256 = 4^4, 4 stages, 2 KiB shared, 64 threads) — same
template as fft_c2c_1024. CPU side already covered by cpu_radix4/twiddles_radix4
(verified log_n=8). GPU cross-check 1.22e-5; example 8/8; pytest 29/29. Wired
via KernelKind::Specialised256.

ALL THREE SIZES NOW SPECIALISED. Full perf-gate under LOCKED clock (1500 MHz),
ratio vs cuFFT:

| N    | kernel  | ratio | fallback was |
| ---- | ------- | ----- | ------------ |
| 256  | radix-4 | 1.32  | 6.34         |
| 1024 | radix-4 | 2.13  | 6.34         |
| 4096 | radix-8 | 3.69  | (n/a)        |

Clear trend: smaller N -> closer to cuFFT (less shared, better occupancy,
fewer stages). N=256 is at **1.32x** — within ~30% of cuFFT and the closest to
the 0.9x gate. None passes 0.9x yet (that needs the multi-FFT-per-block /
profiler-guided redesign tracked in the four-step notes), but every
specialised size now beats its radix-2 fallback by 2-5x. The shared-memory
radix-R Stockham family (scalarized, inline butterfly macro, single buffer) is
the established winning pattern.

## N=256 warp-per-FFT redesign (2026-06-02): profiler REFUTES the approach

Built `fft_c2c_256_warp` (`fft256_warp_body.rs`): one warp = one 256-pt FFT,
256 = 32x8, register-resident (in-register dft8 + W_256 twiddle + 8-wide 32-pt
warp shfl), zero shared / zero syncthreads, pure-Rust `mul_add` FMA. Correct:
max_rel_err 3.08e-5 vs radix-2 (CPU oracle `warp256_model` = four_step_model
N1=32,N2=8). Spike `fft256-warp-spike`; alternating gate `perf-gate-warp256`
(env `WARP_BLOCK` sweeps K = warps/block).

Result (unlocked, alternating vs cuFFT): ratio is BEST at K=1 (2.15x) and gets
monotonically WORSE with more warps/block (K=8 -> 2.66x, K=32 -> 5.29x). So the
"multiple FFTs per block" lever HURTS here. Worse than the existing radix-4
(1.32x).

ncu (K=1, sudo via zenity askpass): **40 reg/thread** (not register-bound),
**17% achieved occupancy**, **0.41 waves/SM**, 10.69 us. The bottleneck is
GRID STARVATION, not register pressure: at batch=256 a warp-per-FFT launches
only 256 warps (32 threads/FFT) = 0.41 of one wave -> the grid does not fill
the GPU once. More K -> fewer blocks -> even smaller grid -> worse.

DECISIVE INVERSION: at this tiny, latency-bound batch the spec's premise is
backwards. You want MORE threads/FFT (more warps in flight), not fewer. radix-4
wins (1.32x) precisely because it uses 64 threads/FFT (2 warps -> 512 warps,
2x the parallelism of warp-per-FFT's 256). cuFFT's vector_fft likewise.

DECISION: warp-per-FFT is NOT the path to beat cuFFT at N=256 (kept as a
documented research artifact; correct, not wired). The profiler-indicated
levers instead:
  1. FMA the EXISTING radix-4 fft_c2c_256 butterflies (`mul_add`, now proven to
     work) — certain ~10-20%, brings 1.32x toward ~1.1-1.2x. Lowest risk.
  2. radix-16 shared kernel (256 = 16x16, 2 stages = 1 barrier vs radix-4's 4
     stages/3 barriers), MORE threads/FFT, FMA'd — attacks barriers AND raises
     warps-in-flight, the two things the profiler says matter here.
  3. Caveat (per the earlier nvcc-ceiling note): beating cuFFT at a latency-
     bound tiny batch is research-grade and may be infeasible.

### FMA lever measured (2026-06-02): perf-NEUTRAL at batch=256

Applied `mul_add` FMA to the radix-4 `tw!` twiddle multiplies in
`fft_c2c_256/1024/4096` (the only complex multiplies; the radix-4/8 DFT bodies
are add/sub). Correct (example 8/8). Also fixed a latent bug it surfaced: the
radix-2 fallback's two `SharedArray<f32, 8192>` = 64 KiB static shared exceeds
the 48 KiB sm_120 limit; it had been linking only on a lucky nvJitLink LTO
carveout, and FMA-ing the specialised kernels shifted that carveout to 48 KiB.
Resized to `<f32, 4096>` (32 KiB; fallback max is N=2048 since 256/1024/4096 are
always specialised).

A/B under locked 1500 MHz (median of 4, ratio vs cuFFT), driver 595:

| N    | non-FMA | FMA  | verdict |
| ---- | ------- | ---- | ------- |
| 256  | ~1.55   | ~1.51 | within noise |
| 1024 | ~1.9    | ~2.06 | within noise |
| 4096 | ~3.60   | ~3.68 | within noise |

`ferrum_us` at 256 is unchanged (~0.035). FMA buys ~nothing here: at batch=256
these kernels are latency/occupancy/memory-bound (0.41 waves/SM; cf. the warp
finding above), NOT ALU-bound, so the saved twiddle multiplies hide behind
memory+sync latency. The phase0 "~10-20% left on the table" estimate was for an
ALU-saturated regime, not this tiny-batch latency-bound one. FMA kept anyway
(canonical form, numerically equal-or-better, and it concretely exercises the
P0 `mul_add`-works fix); the fallback resize is a genuine bug fix.

NOTE: driver 595 moved the N=256 baseline from phase0's 1.32x (driver 580,
1500 MHz) to ~1.55x — cuFFT got relatively faster; always A/B within one driver.

### Standing conclusion (2026-06-02)

Two of the three pivot levers tested empirically, both REFUTED for beating
cuFFT at N=256: (1) warp-per-FFT redesign — grid-starved; (2) FMA — perf-neutral
(latency-bound). The benchmark regime (batch=256, N=256) is latency/occupancy-
bound with the GPU <half-occupied (0.41 waves); cuFFT's `vector_fft` is a
tighter latency-optimised kernel for exactly this regime. Remaining untried
lever aligned with the profiler insight (more threads/FFT + fewer barriers):
radix-16 shared (256 = 16x16, 1 barrier). Beating cuFFT here remains
research-grade and may be infeasible.

## Phase 0 summary

Design adjustments for Phase 3+:
- [x] CuSimd v4 lowering: deferred (cuda-oxide doesn't expose PTX separately; functional correctness verified)
- [x] fft_c2c_4096 launch_bounds: deferred to Phase 3 based on perf-gate results
- [x] fft_c2c_4096 SMEM +1 padding: drop (ratio 1.09x < 1.2x threshold)
- [x] Wall-clock vs event-time gap to footnote: 0.000 us (negligible, no footnote needed)
- [ ] cuFFT kernel-name reference: see `tools/ncu/cufft-blackwell-2026-05.txt` (operator-driven, Task 0.1 step 5)
