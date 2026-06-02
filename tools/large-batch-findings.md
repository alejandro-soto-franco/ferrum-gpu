# Large-batch (throughput regime): how pure-Rust reaches cuFFT parity

Date: 2026-06-02. GPU: RTX 5060 Laptop (sm_120). Clock locked 1500 MHz.
Metric: `alternating_bench_batch` event-time, ratio = ferrum / cuFFT (lower is
better; <1.0 BEATS cuFFT). Bins: `batch-sweep` (N=256), `batch-sweep-4096`.

## TL;DR

The `phase0-findings.md` conclusion ("cuFFT unbeatable at N=256") was correct
ONLY for the small-batch (latency-bound) regime. At **large batch** the GPU
saturates, the bottleneck flips to memory/kernel efficiency, and a pure-Rust
kernel reaches **cuFFT parity at N=256** (ratio ~1.0, runs 0.95–1.01 — beats
cuFFT on favorable runs) and a **27% improvement at N=4096** (1.74x vs the old
radix-8's 2.38x). cuda-oxide (Rust→PTX, no CUDA C) is the substrate.

## The recipe (3 ingredients, each profiler-driven)

1. **Higher radix = fewer shared round-trips.** The shared-memory Stockham
   bottleneck at large batch is l1tex throughput, dominated by per-stage shared
   read/write. radix-R has log_R(N) stages. Going radix-4 (256=4^4, 4 stages) ->
   radix-16 (256=16^2, 2 stages) halved the shared traffic and dropped l1tex
   from 93% to 72%. radix-16 tiles 256=16^2 and 4096=16^3 exactly.

2. **u64-coalesced global IO.** THE decisive lever. cuda-oxide emits scalar
   GENERIC-space `ld.b32` for `&[f32]` indexing — never `ld.global`, never
   vectorized (CuSimd<f32,4> also lowered to 4 `ld.b32`, NOT `ld.global.v4`).
   On the strided re/im loads this gave only ~18–22% memory-sector utilisation
   (fetch 32 B, use ~6). Loading each complex as ONE `u64` (re|im, 8 B) makes
   consecutive threads read contiguous 8 B -> ~100% coalescing. Recover the two
   f32 with `f32::from_bits` (interleaved [re,im,…] is little-endian: re low):
       let c: u64 = *(in_ptr.add(2*t) as *const u64);
       let re = f32::from_bits(c as u32);
       let im = f32::from_bits((c >> 32) as u32);
   Store: `(re.to_bits() as u64) | ((im.to_bits() as u64) << 32)` via a
   `*mut u64` cast of the DisjointSlice element pointer. This single change
   took N=256 from ratio 1.009 -> ~0.95–1.0 (the last 5%).

3. **One FFT per block (contiguous warp).** Each warp must stay on ONE
   contiguous FFT region in global memory. Packing 2 FFTs/warp to use the idle
   butterfly lanes REGRESSED (1.009 -> 1.21): the warp then straddled two
   non-contiguous 2 KiB regions, wrecking coalescing. Idle butterfly lanes
   don't matter (DRAM-bound); coalescing does. So: 1 FFT/block, all threads do
   the (coalesced) load/store, a subset run the butterflies.

Plus: **full scalarisation** (named registers, no `[f32;N]`/loops) — cuda-oxide
rejects runtime-indexed arrays ("invalid input program") and spills
constant-indexed ones to local DRAM. dft16 = two scalar dft8 + 8 W_16 combines.

## Results (ratio vs cuFFT, locked 1500 MHz)

N=256 (`batch-sweep`):
| batch  | warp | radix4 | radix16+u64 |
| ------ | ---- | ------ | ----------- |
| 4096   | 4.5  | 2.9    | 2.1         |
| 16384  | 2.0  | 1.40   | 1.08        |
| 65536  | 1.76 | 1.24   | 1.03        |
| 262144 | 1.70 | 1.20   | **0.95–1.01 (PARITY / WIN)** |

N=4096 (`batch-sweep-4096`, OOM beyond 16384):
| batch | radix8 (old) | radix16+u64 |
| ----- | ------------ | ----------- |
| 4096  | 2.46         | 1.86        |
| 16384 | 2.38         | **1.74**    |

Trend: ratio falls monotonically with batch (the GPU fills: occupancy ~95%).
N=256 is DRAM-bound at parity; N=4096 is held back by its 32 KiB shared/block
(occupancy), so it improves a lot but doesn't reach parity.

## cuda-oxide codegen limitations found (the real ceilings)

- **No `ld.global`**: all global access is generic `ld.b32`. Worked around by
  u64-packing for coalescing; a true fix (mark loads global, emit `ld.global`)
  is upstream codegen work and would likely push N=4096 to parity too.
- **No load/store vectorisation**: `CuSimd<f32,4>` -> 4 `ld.b32`, not
  `ld.global.v4.f32`. (The Task 0.2 "expected v4" note was never actually
  verified; PTX disproves it.) u64 sidesteps this for the 8 B case; v4 (16 B)
  is still unreachable.
- **No runtime-indexed register arrays**: `[f32;N]`[i] -> local DRAM or a hard
  "invalid input program". Everything must be hand-scalarised.
- **FMA (`mul_add`)**: works in the `cargo oxide` git-dep backend (6ed9938)
  but NOT the wheel's older standalone backend — see phase0-findings. Also
  perf-neutral here (memory-bound). Kept out of shipped kernels.

## Not done / next (to reverse-engineer from)

- **Ship it**: these kernels are `cargo oxide`-only. The u64-cast and the
  `f32::from_bits`/`to_bits` patterns must be verified on the wheel's standalone
  maturin backend before going into kernels_body (same two-backend caveat that
  bit FMA). If the backend supports them, the wheel gets the win directly.
- **N=4096 to parity**: it's occupancy-bound on 32 KiB shared. Levers: stage
  the W_4096 twiddles in shared vs the current runtime-indexed global table;
  try fewer-stage factorisations; or a two-kernel four-step.
- **N=1024** (= 2^10, not a power of 16): needs mixed radix (16x16x4) or
  radix-32. Not yet built.
- **Upstream `ld.global` + vectorisation in cuda-oxide**: the highest-leverage
  fix — it would lift coalescing for every kernel and likely beat cuFFT
  outright at every size. A contributor PR (the author is a cuda-oxide
  contributor).
- **Integrate into perf-gate / KernelKind** once shipped, with cross-check +
  pytest, and a batch-aware kernel selection (warp/radix4 small-batch latency
  vs radix16 large-batch throughput).
