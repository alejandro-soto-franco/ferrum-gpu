# N=256 warp-per-FFT (multi-FFT-per-block) redesign — design spec

Date: 2026-06-02
Branch: `feat/fft-256-warp-per-block`
Status: approved (brainstorming), implemented + investigated.

> OUTCOME (2026-06-02, see `tools/phase0-findings.md` for the full log): the
> warp-per-FFT kernel was built and is correct but does NOT beat cuFFT at
> N=256 — it's grid-starved (0.41 waves/SM, batch=256 under-fills the GPU).
> FMA (P0's "free win") works under `cargo oxide` but BREAKS the wheel (older
> standalone backend) and is perf-neutral here anyway, so it was reverted from
> the shipped kernels. radix-16 (fewer barriers) was codegen-rejected and is
> predicted to lose on the same occupancy ceiling. Net: at batch=256 the regime
> is latency/occupancy-bound and cuFFT's vector_fft wins; beating it needs a
> larger batch where parallelism exists. Kept: correct warp artifact, CPU
> oracle, ncu harness, the fallback 64->32 KiB bug-fix, and the corrected
> FMA-works-but-only-in-the-dev-backend finding.

## 1. Goal & success criteria

Push the forward C2C FFT at **N=256** from its current **1.32×** cuFFT down to
**< 1.0× cuFFT** (i.e. ferrum strictly faster than cuFFT) on the production
metric: `make perf-gate` alternating event-time, batch=256, graphics clock
locked to 1500 MHz, RTX 5060 Laptop (sm_120, Blackwell).

`ratio = ferrum_event_us / cufft_event_us`. Lower is better; **1.0 = parity**.
The existing `GATE_RATIO = 0.9` in `perf-gate.rs` is the stretch target;
crossing **1.0** is the headline ("pure-Rust beats cuFFT").

Hard constraints (non-negotiable, all must still hold at ship):
- **Correctness**: new kernel cross-checks against a CPU model AND the radix-2
  reference at `max_rel_err <= 1e-3`; `pytest` stays **29/29**; the
  `fft-1d-c2c` example stays **8/8**.
- **Pure-Rust source**: the kernel's numerical body is Rust, fully
  compiler-lowered. FMA is `f32::mul_add` (lowers to libdevice `__nv_fmaf` →
  `fma.rn.f32`); no `asm!`, no CUDA C, no linked PTX blob (see §6).
- **Reuse-ready**: the kernel is built from a `local_radixR! + warp32_shuffle!
  + twiddle` macro template structured so N=1024 (32×32) and N=4096 can adopt
  it in a fast-follow without redesign. (1024/4096 implementation is OUT of
  scope for this spec — see §8.)

## 2. Background (why this, why now)

`tools/phase0-findings.md` established the decisive result: the gap to cuFFT is
**algorithmic, not codegen**. The same radix-8 algorithm compiled by `nvcc -O3`
(optimal PTX, 7 FMA, 0 spills) is still 3.45× slower than cuFFT; our cuda-oxide
kernel (3.80×) is within ~10% of that nvcc ceiling. So micro-opt is exhausted.

cuFFT's advantage is **kernel design**: mixed-radix, **multiple FFTs per
block**, register blocking, and far fewer shared-memory round-trips / barriers
than our current one-FFT-per-block shared-memory Stockham kernels.

The current `fft_c2c_256` (kernels_body.rs) is radix-4 shared-memory Stockham:
4 stages, 4 `sync_threads` barriers, a single 2 KiB shared buffer, **64
threads/block, one FFT/block** — low occupancy. It is the closest size to
parity precisely because its shared footprint is smallest. Attacking the
barrier/round-trip cost directly (not amortizing it) is what can flip it.

The proven building block already exists: the `warp_fft32` spike
(`warp_fft_spike_body.rs`, CPU model `ferrum_gpu_fft::warp_fft::warp_fft32_model`)
computes a 32-pt C2C FFT entirely in registers via `shfl_xor_f32` through
cuda-oxide — **bit-exact**, 19.7 Gpt/s, zero shared memory. This redesign
composes that primitive into a full 256-pt transform.

## 3. Approach: warp-per-FFT, register-resident, multi-warp-per-block

Cooley–Tukey factor **256 = 32 × 8** (N1 = 32 the warp dimension, N2 = 8 the
in-register dimension). One **warp computes one entire 256-pt FFT in registers**
with **zero shared memory and zero `sync_threads`**; a block runs **K warps =
K FFTs** (K tuned in §5, e.g. 4/8/16).

Index maps (decimation-in-frequency convention; the verified CPU model in P1 is
the source of truth and may pick the dual DIT map if cleaner):
- input  `n = n1 + 32 * n2`, `n1 = lane ∈ [0,32)`, `n2 = p ∈ [0,8)`.
- output `k = k1 + 32 * k2`, `k1 = lane ∈ [0,32)`, `k2 ∈ [0,8)`.

Per warp, per FFT:
1. **Load** 8 elements per lane: lane L loads `e[p] = x[L + 32*p]`, `p=0..7`.
   Across lanes at fixed `p` this is stride-1 → **coalesced**.
2. **Step 1 — 32-pt warp FFT, 8-wide**: run the proven `warp_fft32` butterfly
   (radix-2 DIF, 5 `shfl_xor` stages) over the lane dimension, applied to all
   8 register elements simultaneously (8 independent 32-pt transforms, one per
   `p`). Produces `B[k1][p]` held in lane `k1`.
3. **Step 2 — twiddle**: `B[k1][p] *= W_256^(k1 * p)`. Twiddle source is a
   tuning knob (§5): small constant table in registers/shared vs on-chip
   `sincos`. 8 twiddles per lane.
4. **Step 3 — 8-pt in-register DFT**: run the proven `dft8!` macro (from
   kernels_body.rs) over the `p` dimension within each lane → the 8 outputs
   `X[k1 + 32*k2]`, `k2=0..7`, held in registers.
5. **Store** 8 outputs per lane to global at `k1 + 32*k2` (stride-32 across
   lanes → coalesced). Handle the warp-FFT bit-reversed output ordering in the
   index map (as `warp_fft32` already does on write).

All butterfly arithmetic (steps 2–4) is written as explicit FMA via the §6
`fma!` helper so the multiply-adds lower to `fma.rn.f32`.

Register budget: 8 complex = 16 live f32 + shuffle partners + twiddles per
lane. Whether this caps occupancy is the **central question profiling answers**
(§5); if it proves fatal, fall back to Approach B (§7).

## 4. Components & files

New / changed, smallest-purpose units:

- `crates/ferrum-gpu-fft/src/warp_fft.rs` (existing): add `warp256_model`
  (CPU, 32×8 Cooley–Tukey) + unit tests vs the radix-2 / `rustfft` reference.
  This is the correctness oracle the GPU kernel is checked against.
- `crates/ferrum-gpu-fft-kernels/src/fft256_warp_body.rs` (new): the
  `#[cuda_module]` block with `fft_c2c_256_warp`, plus the shared
  `warp_fft32!` / `dft8!` / `fma!` macros. Single source of truth, included by
  consumers per the existing `include!` contract.
- FMA: no helper crate needed — butterflies use `f32::mul_add` directly (§6).
- `crates/ferrum-gpu-bench/src/bin/fft256-warp-spike.rs` (new): GPU vs CPU-model
  + radix-2 cross-check and standalone timing for profiler runs.
- `crates/ferrum-gpu-bench/src/lib.rs`: add `spec256_warp_launch_cfg()` (K
  warps/block sweep) alongside the existing `spec256_launch_cfg`.
- Integration (P5): `fft256_warp_body.rs` included into `perf-gate.rs`,
  `main.rs`, `example/fft-1d-c2c`, `ferrum-gpu-py`; `KernelKind::Specialised256`
  dispatch repointed to `fft_c2c_256_warp`; `Plan::kernel_twiddles` supplies the
  warp twiddle layout.

Twiddles: a new `twiddles_warp256()` in `ferrum-gpu-fft` producing exactly the
`W_32` (warp butterfly) + `W_256^(k1*p)` (step-2) tables the kernel reads,
unit-tested for layout.

## 5. Profiler-guided tuning loop (ncu enabled via zenity/sudo)

Profiling is authorised on this box; the agent drives the existing sudo/zenity
path (referenced in phase0 Phase-4 clock-lock notes) to (a) lock the clock for
fair head-to-head and (b) run `ncu`. **Measure before changing anything.**

Per-iteration:
1. `ncu` the spike: `achieved_occupancy`, `registers_per_thread`,
   `smsp__warp_issue_stall*` (find the dominant stall), memory throughput,
   `launch__registers_per_thread` vs the occupancy cliff.
2. Change ONE knob, re-measure ratio under locked clock:
   - **K = warps/block** (4/8/16): the core multi-FFT-per-block lever.
   - `__launch_bounds__` / occupancy hints if cuda-oxide exposes them.
   - twiddle source: register-const vs shared vs on-chip `sincos`.
   - FMA on/off (quantify the §6 lever in situ).
3. Log every iter in `phase0-findings.md` (ratio, the knob, keep/drop), exactly
   like the Phase-3/4/5 tables.

Stop when ratio < 1.0 (success) or profiling shows a hard occupancy/IO wall that
Approach B addresses better (fall back, §7).

## 6. FMA: `f32::mul_add` works today (RESOLVED — no asm!, no fork)

**Empirically settled 2026-06-02 (P0).** The `phase0-findings.md` claim that
"cuda-oxide does not lower `llvm.fma.f32` → unresolved extern → nvJitLink error
4" is **stale**. At the pinned rev `6ed9938`, cuda-oxide lowers the Rust FMA
intrinsics to libdevice:

```rust
// mir-lower/src/convert/ops/call.rs:271
Self::FmaF32 | Self::FmuladdF32 => Ok("__nv_fmaf"),
Self::FmaF64 | Self::FmuladdF64 => Ok("__nv_fma"),
```

Chain: `f32::mul_add` → `core::intrinsics::fmaf32` → libdevice `__nv_fmaf`,
which NVVM links + inlines to a single `fma.rn.f32` before ptxas (cuda-oxide
compiles via NVVM/libdevice — `libnvvm-sys`, `mir-importer/pipeline.rs`). The
old failure was almost certainly `__nv_fmaf` not being linked in a kernel that
used no other libdevice function — not a missing lowering.

**Verified:** a `mul_add` probe in the `vector-add-cuda-oxide` kernel compiled,
JIT-linked, and verified 1,048,576 elements (`a*2+b`) correctly.

**Consequence — both the `asm!` lever and the cuda-oxide fork are DROPPED from
this plan.** FMA is reached in 100% pure, compiler-lowered Rust today. The
butterflies are written with explicit `mul_add` (complex multiply
`re*wr - im*wi` → `re.mul_add(wr, -(im*wi))` = 1 mul + 1 fma). FMA stays a
perf multiplier applied at tuning time (P3), not a correctness dependency.

**Bonus (separate, tracked):** the same `mul_add` rewrite applies to the
EXISTING `fft_c2c_256/1024/4096` butterflies — a quick FMA win on all three
production kernels independent of this redesign (the "~10-20% left on the
table" the stale note assumed was unreachable). Captured as a fast-follow.

## 7. Fallback: Approach B (packed-shared multi-FFT)

If P3 profiling shows register pressure from 8 complex/lane caps occupancy so
hard that warp-per-FFT cannot beat the shared kernel: keep the proven radix-4
shared Stockham kernel but **pack K FFTs per block** (e.g. 8 × 256 complex in
shared) to raise occupancy from the current 64-thread/block design. Lower
ceiling (does not cut per-FFT barriers) but reuses working code. Documented as
the safety net; chosen only on profiler evidence, not speculation.

## 8. Out of scope (YAGNI)

- N=1024 / N=4096 implementation (this spec only *designs the template for
  reuse*; a fast-follow spec implements them).
- Inverse FFT (forward C2C only, matching every current specialised kernel).
- cuda-oxide **auto**-FMA-contraction (mul+add fused without explicit
  `mul_add`) — separate, harder fast-math work; only the explicit intrinsic
  path is in scope.
- Non-power-of-two / arbitrary N.

## 9. Risks

| Risk | Mitigation |
| ---- | ---------- |
| ~~cuda-oxide doesn't forward inline `asm!`~~ | RESOLVED: `f32::mul_add` works today (§6); asm!/fork dropped. |
| 8-complex/lane register pressure caps occupancy | Profiler-measured (§5); Approach B fallback (§7). |
| `shfl_xor` correctness through cuda-oxide | Already proven by the `warp_fft32` spike (bit-exact). Low risk. |
| Driver upgraded 580 → 595.71.05 since phase0 | Verify the zenity/sudo clock-lock path + `ncu` permission still work before trusting ratios; re-baseline cuFFT first. |
| Bit-reversed warp output index errors | CPU model (P1) is the oracle; GPU cross-check gates every change. |

## 10. Phased implementation order

- **P0 — FMA probe** (DONE 2026-06-02): verified `f32::mul_add` compiles,
  JIT-links, and runs correctly through cuda-oxide → libdevice `__nv_fmaf`.
  asm!/fork dropped (§6). Pure-Rust FMA confirmed available.
- **P1 — CPU model**: `warp256_model` (32×8) in `warp_fft.rs`, unit-tested
  `== radix-2 / rustfft` for log_n=8. Correctness oracle.
- **P2 — GPU spike kernel**: `fft_c2c_256_warp` (single warp/FFT first, then K
  warps/block) + `fft256-warp-spike` cross-check vs CPU model + radix-2.
- **P3 — profiler-guided tuning**: §5 loop under locked clock; iterate ratio.
- **P4 — converge < 1.0×**: perf-gate at N=256; record iter log.
- **P5 — integrate**: repoint `KernelKind::Specialised256` across all consumers;
  cross-check + example 8/8 + pytest 29/29 green.
- **P6 — reuse hooks**: document the macro-template extension to 1024 (32×32) /
  4096 in `phase0-findings.md`; no implementation.
- **Tracked follow-up**: apply the `mul_add` FMA rewrite to the existing
  `fft_c2c_256/1024/4096` butterflies (separate quick win on all three; §6).

## 11. Testing strategy

- CPU model unit tests (`warp256_model == reference`) — `cargo test`.
- GPU cross-check binary (`max_rel_err` vs CPU model AND radix-2) — gates every
  kernel change.
- `fft-1d-c2c` example 8/8; `pytest` 29/29 (no regression at integration).
- `perf-gate` ratio under locked clock — the success metric.
- Every profiler iteration logged in `phase0-findings.md`.
