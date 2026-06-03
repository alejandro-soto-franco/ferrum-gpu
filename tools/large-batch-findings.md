# Large-batch (throughput regime): how pure-Rust reaches cuFFT parity

> UPDATE 2026-06-02 (ship + correctness): the radix-16 kernels now live in
> `crates/ferrum-gpu-fft-kernels/src/kernels_body.rs` (the single module the
> wheel, bench, and examples all `include!`), NOT the old standalone
> `fft256_r16s_body.rs` / `fft4096_r16s_body.rs` files (deleted). The Python
> wheel ships them: `run_fft_flat_with_device` routes N=256/4096 at
> `batch >= ferrum_gpu_fft::R16S_MIN_BATCH` (4096) to the radix-16 kernels.
> The wheel is built through the personal cuda-oxide fork backend
> (Makefile `FERRUM_GPU_BACKEND`), pytest green (35 cases incl. large-batch
> r16s vs numpy). **The earlier "N=4096 = 1.74x, bit-exact" claim was WRONG:
> that kernel had a data race** (in-place shared gather/scatter with 8 warps
> and no intra-stage barrier) that only manifested at large batch /
> many-co-resident-blocks, which the batch=2 bench check never exposed. Adding
> one `sync_threads()` between the per-stage gather and scatter fixes it; the
> honest, race-free number is **~1.37x (batch 4096) to 1.44x (batch 16384)**
> vs cuFFT (still ~18-22% better than radix-8). The N=256 kernel is
> single-warp/lockstep and was always correct. See the closing section.

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

N=4096 (`batch-sweep-4096`, OOM beyond 16384). The radix16+u64 column here is
the ORIGINAL racy kernel (no intra-stage barrier); see the closing section for
the corrected, race-free numbers (~1.37-1.44x):
| batch | radix8 (old) | radix16+u64 (racy, superseded) |
| ----- | ------------ | ------------------------------ |
| 4096  | 2.46         | 1.86                           |
| 16384 | 2.38         | ~~1.74~~ (raced)               |

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
- **`ld.global.v4` vectorisation in the cuda-oxide FORK** (`~/cuda-oxide`, the
  owner controls it). Scoped 2026-06-02. cuda-oxide is an MLIR/pliron compiler;
  `convert_load` (mir-lower/convert/ops/memory.rs) emits ONE `llvm::LoadOp` of
  the converted result type, and a `CuSimd<f32,4>` converts to an LLVM AGGREGATE
  -> legalised to 4 scalar `ld.b32` (PTX-confirmed; the Task-0.2 "expected v4"
  was never real). There IS an LLVM-dialect `VectorType` (-> NVVM `ld.v4`), and
  `tcgen05.ld`/`shfl` are the precedent: a `cuda_device` intrinsic -> `nvvm`
  dialect op -> INLINE PTX. So the fix is a 4-change intrinsic:
    1. cuda-device: `ld_global_v4(ptr)->CuSimd<f32,4>` (+ v2, + stores).
    2. dialect-nvvm: a vector global-load op.
    3. mir-importer: recognise the call -> that op.
    4. mir-lower: lower to inline PTX `ld.global.v4.f32 {%0..%3},[%4]`.
  Build: `cd ~/cuda-oxide/crates/rustc-codegen-cuda && cargo build` produces the
  backend .so; ferrum-gpu picks it up via `CUDA_OXIDE_BACKEND=<that .so>`
  (backend.rs discovery order: env > local repo > ~/.cargo/cuda-oxide cache).
  CAVEAT: marginal at N=256 (DRAM roofline) — value is instruction-bound kernels
  + clean &[f32] code + reusability. A focused dedicated session (long backend
  builds). Merged fork branch fix/fail-loud-silent-miscompiles -> main (c8b3103).

## N=4096 bank-conflict + occupancy (2026-06-02): lever identified, swizzle reverted

Profiled r16s-4096 @ batch 16384: **DRAM only 57%** (NOT at roofline — cuFFT-4096
IS at roofline ~0.21us), **occupancy 49.5% limited to 3 blocks/SM by the 32 KiB
shared**, **5.6M ld.shared bank conflicts** (the stride-256 radix-16 gather is
bank-aligned -> 16-way conflict), 48 reg/thread. So 4096 (1.74x) has real
headroom to parity if the conflicts + occupancy are fixed.

Tried: a uniform shared swizzle `phys(i)=i+i/256` (1 pad slot per 256 complex)
to make the 16 gathered elements land in banks {0,2,..,30}. It's a correct
bijection AND race-free (with an added intra-stage barrier), yet produced WRONG
output that ncu's kernel-replay caught (proven kernel passes 9 replays; swizzled
fails). Root cause not found quickly -> REVERTED to the proven 1.74x kernel.
Shipping a miscompiled FFT to save microseconds is the wrong trade. NOTE: ncu
replay is a good race/correctness oracle — use it to vet kernels. A real 4096
conflict fix needs per-stage swizzle analysis (the gather stride is 256 every
stage, but the scatter pattern differs), not one uniform pad.

## SHIP + the N=4096 data race (2026-06-02)

Integrating the radix-16 kernels into the Python wheel exposed a real bug the
benchmark had hidden. The wheel routes N=256/4096 at batch >= `R16S_MIN_BATCH`
(4096) to these kernels; a new pytest case (N=4096 fwd, batch 4096, vs numpy)
failed with **rel err 0.18, non-deterministic** (1773 / 1765 / 1880 bad lanes
across three runs, different lanes each time) even though the input was the
SAME signal tiled across all lanes (so a correct kernel MUST emit identical
output per lane). Non-determinism + identical input = a data race.

Root cause: the N=4096 kernel does an **in-place** radix-16 stage on one shared
buffer with **256 threads = 8 warps**, and had **no barrier between the per-stage
gather and scatter**. A warp that reaches the scatter first overwrites BUF
entries a slower warp has not yet read. The N=256 kernel is a SINGLE warp (32
threads, 16 active) running in lockstep, so its 16 reads always precede its 16
writes, so it never needed the barrier and was always correct. With 8 warps the
hazard is real and shows up only at large batch (many co-resident blocks per SM),
which is why the bench's batch=2 correctness check never caught it, and why the
"1.74x bit-exact" claim was false.

Fix: one `thread::sync_threads()` between the gather/compute and the scatter in
the N=4096 stage (all 256 threads run it uniformly, no divergence -> no deadlock).
After the fix: pytest green, and the determinism probe shows 0 bad lanes across
3 runs. The bench's correctness check was hardened from batch=2 to batch=512 so
this race class is caught in future (it is now a genuine race oracle, cheaper
than ncu replay).

Corrected, race-free perf (locked 1500 MHz, driver 595, ratio vs cuFFT):
| batch | radix8 | radix16+u64 (barrier) |
| ----- | ------ | --------------------- |
| 4096  | 1.604  | **1.371**             |
| 16384 | 1.766  | **1.440**             |
Still ~18-22% faster than radix-8; ~1.37-1.44x of cuFFT (the barrier costs a
little vs the racy timing, but that timing measured a wrong kernel).

## Shipping the wheel through the fork (2026-06-02)

The wheel now builds through the PERSONAL cuda-oxide fork backend, not the stale
standalone at `~/.cargo/cuda-oxide`. Mechanics:
- `Makefile`: `FERRUM_GPU_BACKEND ?= ~/cuda-oxide/crates/rustc-codegen-cuda/target/debug/librustc_codegen_cuda.so`
  feeds `-Z codegen-backend=` for `make develop` / `make wheel`. Build the
  backend with `cd ~/cuda-oxide/crates/rustc-codegen-cuda && cargo build` (a
  NESTED workspace; its `target/` is symlinked to /home/cargo-targets per the
  btrfs rule).
- `Dockerfile.manylinux` + `build-wheel.sh`: the PyPI image clones the fork,
  builds `rustc-codegen-cuda`, and points maturin's RUSTFLAGS at it (untested in
  the integration session: validate with `make wheel-manylinux`).
- Kernels are a SINGLE source of truth in `kernels_body.rs`'s `mod kernels`;
  the standalone `fft256_r16s_body.rs` / `fft4096_r16s_body.rs` were DELETED
  (the kernel export symbol is `cuda_oxide_kernel_<hash>_<fn>` keyed by fn name
  + content, NOT the Rust module, so the same kernel in two modules is a hard
  "symbol already defined" error under the fork's fail-loud codegen). The bench
  bins now call `kernels::fft_c2c_*_r16s`.

Known limitation: the fork backend does not load a binary with TWO+
`#[cuda_module]` blocks (`batch-sweep`, which also has `mod fft256_warp`, fails
at runtime with "embedded CUDA module 'ferrum-gpu-bench' was not found"); single
-module binaries (the wheel, `batch-sweep-4096`) are fine. The wheel is
single-module, so this does not affect shipping.
- **Integrate into perf-gate / KernelKind** once shipped, with cross-check +
  pytest, and a batch-aware kernel selection (warp/radix4 small-batch latency
  vs radix16 large-batch throughput).
