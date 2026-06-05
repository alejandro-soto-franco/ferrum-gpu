//! Paired, frequency-locked CPU layout microbench for tight confidence intervals.
//! Times the interleaved and split (SoA) variant of each op BACK-TO-BACK within
//! every trial, so residual drift cancels in the per-trial ratio, and emits raw
//! per-trial samples for offline bootstrap CIs and a paired significance test.
//!
//! Covers the allocation-free iterative FFT and the elementwise complex multiply,
//! the two CPU arms whose layout effect is small enough to need statistics.
//!
//! CSV: op,n,trial,ns_inter,ns_split  (ns per element; ratio is what matters).
//! Run frequency-locked and pinned, e.g.
//!   RUSTFLAGS="-C target-cpu=native" cargo build --release -p ferrum-gpu-bench --bin cpu-paired
//!   taskset -c 0 ./target/release/cpu-paired

use std::time::Instant;

use ferrum_gpu_bench::cpu_layout::{
    bit_reverse_table, cmul_interleaved, cmul_soa, fft_iter_interleaved, fft_iter_soa,
    twiddle_table,
};

const WARMUP: usize = 20;
const TRIALS: usize = 200;
const INNER: usize = 16;

fn block<F: FnMut()>(mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..INNER {
        f();
    }
    t.elapsed().as_secs_f64()
}

fn main() {
    println!("op,n,trial,ns_inter,ns_split");

    // ---- allocation-free iterative FFT, batched, layout isolated ----
    for &nn in &[256usize, 1024, 4096] {
        let batch = 4096usize;
        let brev = bit_reverse_table(nn);
        let tw = twiddle_table(nn);
        let inter: Vec<Vec<f32>> = (0..batch)
            .map(|b| {
                (0..2 * nn)
                    .map(|i| (((i + 13 * b) * 7 + 3) % 17) as f32 - 8.0)
                    .collect()
            })
            .collect();
        let split: Vec<(Vec<f32>, Vec<f32>)> = inter
            .iter()
            .map(|x| {
                (
                    (0..nn).map(|i| x[2 * i]).collect(),
                    (0..nn).map(|i| x[2 * i + 1]).collect(),
                )
            })
            .collect();
        let mut out_i = vec![0.0f32; 2 * nn];
        let mut ore = vec![0.0f32; nn];
        let mut oim = vec![0.0f32; nn];

        let mut fi = || {
            for x in &inter {
                fft_iter_interleaved(x, &mut out_i, &brev, &tw);
            }
        };
        let mut fs = || {
            for (re, im) in &split {
                fft_iter_soa(re, im, &mut ore, &mut oim, &brev, &tw);
            }
        };
        for _ in 0..WARMUP {
            fi();
            fs();
        }
        let scale = 1e9 / (INNER as f64 * (nn * batch) as f64);
        for t in 0..TRIALS {
            let ti = block(&mut fi);
            let ts = block(&mut fs);
            println!("fft_iter,{nn},{t},{:.5},{:.5}", ti * scale, ts * scale);
        }
    }

    // ---- elementwise complex multiply, layout isolated ----
    for &n in &[262144usize, 1048576] {
        let a: Vec<f32> = (0..2 * n).map(|i| (i % 5) as f32).collect();
        let b: Vec<f32> = (0..2 * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let mut oint = vec![0.0f32; 2 * n];
        let (ar, ai): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (a[2 * i], a[2 * i + 1])).unzip();
        let (br, bi): (Vec<f32>, Vec<f32>) = (0..n).map(|i| (b[2 * i], b[2 * i + 1])).unzip();
        let (mut crr, mut cii) = (vec![0.0f32; n], vec![0.0f32; n]);

        let mut fi = || cmul_interleaved(&a, &b, &mut oint);
        let mut fs = || cmul_soa(&ar, &ai, &br, &bi, &mut crr, &mut cii);
        for _ in 0..WARMUP {
            fi();
            fs();
        }
        let scale = 1e9 / (INNER as f64 * n as f64);
        for t in 0..TRIALS {
            let ti = block(&mut fi);
            let ts = block(&mut fs);
            println!("cmul,{n},{t},{:.5},{:.5}", ti * scale, ts * scale);
        }
    }
}
