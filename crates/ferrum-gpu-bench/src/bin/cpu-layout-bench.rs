//! CPU interleaved-vs-split layout bench (x86 AVX). Mirrors the burn-flex
//! (CPU SIMD) layout decision. Three arms:
//!   1. hand-rolled split-radix FFT: interleaved vs SoA (layout isolated).
//!   2. rustfft native interleaved vs split-tax (interleave + fft + deinterleave).
//!   3. elementwise complex multiply: interleaved (AoS) vs SoA (vertical SIMD).
//!
//! CSV: arm,N,batch,layout,ns_per_unit,gpoints_per_s
//! (FFT unit = one N-point transform; cmul unit = one complex element.)

use std::time::Instant;

use ferrum_gpu_bench::cpu_layout::{
    cmul_interleaved, cmul_soa, split_radix_interleaved, split_radix_soa,
};
use rustfft::{num_complex::Complex, FftPlanner};

const REPS: usize = 11;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median wall time (seconds) of `f` over REPS runs.
fn time<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut s = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        s.push(t.elapsed().as_secs_f64());
    }
    median(s)
}

fn main() {
    println!("arm,N,batch,layout,ns_per_unit,gpoints_per_s");
    let ns = [256usize, 1024, 4096];
    let batches = [1usize, 16, 256, 4096];

    for &n in &ns {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);

        for &batch in &batches {
            // ---- data ----
            let inter: Vec<Vec<f32>> = (0..batch)
                .map(|b| {
                    (0..2 * n)
                        .map(|i| (((i + 13 * b) * 7 + 3) % 17) as f32 - 8.0)
                        .collect()
                })
                .collect();
            let split: Vec<(Vec<f32>, Vec<f32>)> = inter
                .iter()
                .map(|x| {
                    let re = (0..n).map(|i| x[2 * i]).collect();
                    let im = (0..n).map(|i| x[2 * i + 1]).collect();
                    (re, im)
                })
                .collect();

            let total = (n * batch) as f64;
            let row = |arm: &str, layout: &str, t: f64, units: f64| {
                let ns_per = t * 1e9 / units;
                let gpts = total / t / 1e9;
                println!("{arm},{n},{batch},{layout},{ns_per:.3},{gpts:.4}");
            };

            // ---- arm 1: hand-rolled split-radix, layout isolated ----
            let mut out_i = vec![0.0f32; 2 * n];
            let t_inter = time(|| {
                for x in &inter {
                    split_radix_interleaved(x, &mut out_i);
                }
            });
            let mut ore = vec![0.0f32; n];
            let mut oim = vec![0.0f32; n];
            let t_soa = time(|| {
                for (re, im) in &split {
                    split_radix_soa(re, im, &mut ore, &mut oim);
                }
            });
            row("fft_handroll", "interleaved", t_inter, batch as f64);
            row("fft_handroll", "soa", t_soa, batch as f64);

            // ---- arm 2: rustfft native vs split-tax ----
            let mut bufs: Vec<Vec<Complex<f32>>> = inter
                .iter()
                .map(|x| (0..n).map(|i| Complex::new(x[2 * i], x[2 * i + 1])).collect())
                .collect();
            let t_native = time(|| {
                for buf in bufs.iter_mut() {
                    fft.process(buf);
                }
            });
            let mut scratch: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n];
            let t_tax = time(|| {
                for (re, im) in &split {
                    // interleave split -> Complex
                    for i in 0..n {
                        scratch[i] = Complex::new(re[i], im[i]);
                    }
                    fft.process(&mut scratch);
                    // deinterleave Complex -> split (write-back)
                    for i in 0..n {
                        ore[i] = scratch[i].re;
                        oim[i] = scratch[i].im;
                    }
                }
            });
            row("fft_rustfft", "native_interleaved", t_native, batch as f64);
            row("fft_rustfft", "split_tax", t_tax, batch as f64);

            // ---- arm 3: elementwise complex multiply ----
            let m = n * batch;
            let a: Vec<f32> = (0..2 * m).map(|i| (i % 5) as f32).collect();
            let b: Vec<f32> = (0..2 * m).map(|i| (i % 7) as f32 - 3.0).collect();
            let mut oint = vec![0.0f32; 2 * m];
            let t_ci = time(|| cmul_interleaved(&a, &b, &mut oint));
            let (ar, ai): (Vec<f32>, Vec<f32>) = (0..m).map(|i| (a[2 * i], a[2 * i + 1])).unzip();
            let (br, bi): (Vec<f32>, Vec<f32>) = (0..m).map(|i| (b[2 * i], b[2 * i + 1])).unzip();
            let (mut crr, mut cii) = (vec![0.0f32; m], vec![0.0f32; m]);
            let t_cs = time(|| cmul_soa(&ar, &ai, &br, &bi, &mut crr, &mut cii));
            row("cmul", "interleaved", t_ci, m as f64);
            row("cmul", "soa", t_cs, m as f64);
        }
    }
}
