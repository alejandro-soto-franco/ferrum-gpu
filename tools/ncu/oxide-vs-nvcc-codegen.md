# Do we beat nvcc? cuda-oxide vs nvcc on the identical kernel (sm_120)

Victor's bar is "beat nvcc" — a CODEGEN claim (cuda-oxide vs nvcc on the same
algorithm), distinct from cuFFT (a better algorithm). Measured 2026-06-03 with
`nvcc-vs-oxide` (bench bin): our `fft_c2c_256_r16s` compiled by the cuda-oxide
fork backend vs the bit-identical CUDA C `fft256_r16s` compiled by nvcc 13.1
(via NVRTC, `compute_120`), cuFFT as a yardstick. Both bit-correct vs the
radix-2 CPU reference.

## Verdict: not yet — nvcc is 3 to 13% faster on runtime

per-FFT us, N=256, oxide/nvcc ratio (>1 = nvcc faster):

| batch | oxide/nvcc | oxide/cuFFT | nvcc/cuFFT |
|---|---|---|---|
| 256    | 1.13 | 1.52 | 1.33 |
| 1024   | 1.04 | 2.01 | — |
| 16384  | 1.07 | 1.26 | 1.18 |
| 65536  | 1.04 | 1.10 | 1.06 |
| 262144 | 1.03 | 1.13 | 1.09 |

The gap narrows with batch (3% at max throughput, up to 13% at small batch),
and nvcc itself is only ~6 to 9% off cuFFT at large batch — so most of "our 2x
vs cuFFT" at small batch is the regime, not the algorithm.

## Root cause: cuda-oxide emitted MORE instructions for the same kernel

ncu, batch 65536, identical algorithm (memory ops byte-identical: same shared
ld/st, same global ld):

| | oxide (before fix) | nvcc |
|---|---|---|
| total instructions | 60.2M | 44.0M |
| registers/thread | 56 | 44 |
| fma.rn.f32 in PTX | **0** | (many) |

The big one: **cuda-oxide did zero FMA contraction.** Every multiply-add
lowered to a separate `mul.rn.f32` + `add.rn.f32`; nvcc fuses them
(`--fmad=true` by default). On the 256 kernel the PTX had 126 mul + 177 add +
157 sub and 0 fma.

## Fix (fork commit 6b605e9): contract fma, matching nvcc's default

- mir-lower: set the LLVM `contract` fast-math flag (contract only, no
  reassoc/nnan/...) on every fadd/fsub/fmul.
- pipeline: pass `-fp-contract=fast` to llc (this is what actually forms the
  fma in the NVPTX backend; the per-op flag alone did not).

Effect on the 256 kernel: **fma.rn.f32 0 -> 94, instructions 60.2M -> 55.2M,
registers 56 -> 54.** Correctness shifted 4.05e-6 -> 3.30e-6 (fused rounding,
now matching nvcc's 3.20e-6). All 35 ferrum wheel tests still pass.

## Why the fix did not move the runtime (yet)

The 256 kernel is **DRAM-bound at ~80% of peak** at large batch, so the extra
ALU was already hidden behind memory latency; cutting it does not speed up a
memory-bound kernel. At small batch it is launch/occupancy-bound, where the
remaining register gap (54 vs 44 -> lower occupancy) dominates.

So FMA is a real, broad codegen win (every fp kernel, fewer instructions, nvcc-
matching numerics) but not sufficient to beat nvcc on THIS kernel's runtime.

## What is left to actually beat nvcc

Two remaining codegen gaps, both in cuda-oxide (the fork):
1. **Integer/address arithmetic**: still ~25% more instructions than nvcc after
   FMA (55.2M vs 44.0M). Likely weaker CSE / strength-reduction of the shared
   address math (`2*(src_base + p*16)` recomputed) and redundant moves.
2. **Register allocation**: 54 vs 44 regs/thread -> lower occupancy, which is
   what costs us at small batch.

Both are compiler work in the fork. And note the ceiling: this kernel is DRAM-
bound, so even a perfect codegen match caps the win at ~3-6% here. The clean
"beat nvcc" demonstration wants a more compute-bound kernel (e.g. a larger
register FFT) where instruction count actually gates runtime.
