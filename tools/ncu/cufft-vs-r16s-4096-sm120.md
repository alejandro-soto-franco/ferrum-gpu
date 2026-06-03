# What cuFFT does better at N=4096 (sm_120, RTX 5060 Laptop)

Measured 2026-06-02 with Nsight Compute 2025.4.1, `batch-sweep-4096` built
through the cuda-oxide fork backend. Our kernel profiled at grid 512 (batch
512), cuFFT at grid 4096; percentages and per-FFT counts are the comparable
quantities.

## cuFFT's kernel

cuFFT picks **`void vector_fft<4096, EPT<32>, 1, 4, 70, 0, 2, 0, unsigned int,
float, HostConfigPlaceholder>`**, launched with **128 threads/block**,
**EPT (elements per thread) = 32**, one block per FFT. So each thread owns 32
of the 4096 points in registers (116 registers/thread) and the whole transform
runs mostly in registers with minimal shared traffic.

## Side-by-side (per single 4096-point FFT)

| metric | ours `fft_c2c_4096_r16s` | cuFFT `vector_fft` |
|---|---|---|
| threads / EPT | 256 / 16 | **128 / 32** |
| registers/thread | 48 | **116** |
| static shared/block | 32 KB | 0 (minimal dynamic) |
| **DRAM throughput** | **35.5 %** | **90.4 %** (≈ roofline) |
| **L1TEX (shared) SOL** | **69.6 %** | 29.4 % |
| SM (compute) SOL | 25.4 % | 29.4 % |
| bank conflicts / FFT | **2456** | **38** |
| shared ld+st / FFT | ~1216 | ~1024 |
| achieved occupancy | 47.4 % | 32.8 % |

## Conclusion

**cuFFT is DRAM-bandwidth-bound at 90 % of peak, which is essentially optimal:
the only unavoidable cost of an out-of-place FFT is reading the input and
writing the output once. Our kernel is shared-memory-bound (L1TEX 69.6 %, the
single highest SOL component) and never gets past 35 % DRAM.**

The root cause is structural, not a tuning knob:

- We use **256 threads with EPT 16 and three full in-place shared round-trips**
  (radix-16 stages at m_r = 1, 16, 256), each a gather-16 + scatter-16 with two
  `sync_threads`. That is three passes of 8192 f32 through shared plus six
  barriers, and the stride-256 radix-16 access pattern costs 2456 bank
  conflicts per FFT.
- cuFFT uses **128 threads with EPT 32 and ~116 registers**, doing most of the
  butterflies in registers and touching shared roughly once (≈38 conflicts,
  L1TEX only 29 %). It is therefore limited only by DRAM.

This also explains why the `i + i/32` bank-conflict padding made us *slower*
(1.382 -> 1.588): it cut conflicts but did nothing about the three round-trips,
added index ALU, and grew shared 32 KB -> 33 KB, which crossed a blocks/SM
occupancy threshold. You cannot fix this by deconflicting the three passes; you
have to *remove* passes.

## The lever (concrete)

Restructure 4096 toward cuFFT's shape: **higher elements-per-thread, fewer
threads, far less shared.** Cooley-Tukey 4096 = N1 x N2 (e.g. 64 x 64 or
128 x 32):

1. Load; each thread holds EPT = 32 elements in registers (128 threads/block).
2. Register FFT along the first factor (no shared).
3. Twiddle multiply (registers).
4. **One** shared transpose (the only shared round-trip).
5. Register FFT along the second factor.
6. Store.

Target: one shared pass instead of three, occupancy back up from the small
shared footprint, and the kernel becomes DRAM-bound like cuFFT (~0.20 us/FFT
at batch 4096). This is a from-scratch kernel, not an edit to the radix-16
stage loop, and carries the usual in-place/barrier correctness risk: validate
at many-block batch (the batch-512 oracle), not batch 2.

## Update: the existing four-step (64x64 warp-shuffle) redesign is WORSE, not better

`fft_c2c_4096_4step` (in `four_step_body.rs`, run by `four-step-spike`) already
implements the "register FFT + one shared transpose" idea, but via WARP SHUFFLE
(each lane holds 2 of the 64 elements, EPT=2, exchanged with `shfl_xor`). It is
correct (1.36e-4 vs the radix-2 CPU reference) but slow.

Measured at batch 4096, fork backend, sm_120:

| block (threads) | ratio ours/cuFFT |
|---|---|
| 128 | 6.29 |
| 256 | 4.58 |
| 512 | **3.79** |
| 1024 | 3.85 |

Best ~3.8x (LARGER blocks better: more warps = fewer waves over the 64
columns, so it is wave-serialization, not occupancy). That is far worse than
the radix-16 shared kernel `fft_c2c_4096_r16s` at **1.38-1.40x**.

ncu of the four-step at block 512 / batch 4096 (grid 4096):

| metric | value |
|---|---|
| DRAM throughput | 21.5 % |
| L1TEX (shared) | 53.0 % |
| SM (compute) | 39.5 % |
| inst issued | 39.7 % |
| occupancy | 64.5 % |
| registers/thread | 30 |
| bank conflicts / FFT | 1919 (the +1 stride pad only halved it) |

**Nothing is saturated -> the four-step is LATENCY-bound.** The cause is the
long serial `shfl_xor` dependency chains (5 sequential cross-lane stages per
64-pt FFT, each depending on the previous) plus per-stage global reads of the
`w64`/`w4096` twiddle tables, and residual bank conflicts in the transpose. The
warp-shuffle scheme (EPT=2, 30 registers) is the OPPOSITE of cuFFT's recipe
(EPT=32, 116 registers, intra-thread register butterflies with no shuffles).

## Standing conclusion

- **`fft_c2c_4096_r16s` (radix-16 shared, 1.38-1.40x) remains the best 4096
  kernel on sm_120.** It is shared-memory-bound.
- The warp-shuffle four-step (3.8x) and the bank-conflict padding (1.59x) are
  both worse; do not pursue either.
- cuFFT's edge is a genuine register-resident EPT~32 design that lands on the
  DRAM roofline. Replicating it is blocked by a register-budget wall: a 2-pass
  64x64 needs a 64-pt register FFT (~128+ regs, spills), while staying inside
  ~116 regs (32-pt register FFT) forces 3 passes (4096 = 16^3), which is what
  r16s already does and why it is shared-bound. cuFFT threads this needle with a
  non-obvious multi-factor decomposition; matching it is a from-scratch
  register-kernel research effort, not a tuning pass.

## RESULT: register four-step reaches cuFFT near-parity (and ships)

`fft_c2c_4096_4step_reg` (kernels_body.rs) implements the lever above WITHOUT
the spill: same radix-16 Stockham math and `dft8` butterfly as r16s, but
register-resident across stages so shared traffic halves. The three stages move
register -> register, exchanging only via shared between them:

  - stage 0 gathers 16 inputs straight from GLOBAL into registers, butterflies,
    scatters to shared (first transpose). No load-to-shared pass.
  - stage 1 gathers shared, butterflies, scatters shared (second transpose) --
    the only in-place shared stage, keeps the intra-stage race barrier.
  - stage 2 gathers shared, butterflies, scatters straight to GLOBAL. No store
    pass.

Shared half-trips: 8 (r16s: load + 3x(gather+scatter) + store) -> 4. EPT stays
16 (48 reg/thread, no spill).

Measured (sm_120, batch 4096-32768, unlocked laptop clocks so +/-10% noise):

| kernel | ratio vs cuFFT |
|---|---|
| radix8 | 1.6 - 2.0 |
| r16s | 1.38 - 1.49 |
| **4step_reg** | **1.07 - 1.25 (occasionally <1.0 = WIN)** |

Correct (4.88e-4 vs radix-2 CPU, race-free at batch 512). Shared ops/FFT: 448
(256 ld + 192 st) vs cuFFT's 1024 and r16s's ~1216 -- we now touch shared LESS
than cuFFT. Shipped in the wheel's 4096 dispatch (ferrum-gpu-py); 35/35 pytest
green. This closes most of the gap the earlier "from-scratch register kernel"
note called for, reusing the verified butterfly instead of a 64-pt rewrite.

Remaining headroom: at small batch the kernel is occupancy-bound (47%); stage-0
global gather is scalar ld (could go u64/v4); residual bank conflicts in the two
transposes (~1850/block). But the headline holds: near-parity, occasionally
winning, vs r16s's 1.4x.
