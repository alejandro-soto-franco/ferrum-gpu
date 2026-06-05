//! 2D CPU FFT vs rustfft (row + column 1D).

use ferrum_gpu_fft::{Complex32, Direction, Plan2D};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32 as RustComplex32;

fn rustfft_2d_forward(input: &[Complex32], n: usize) -> Vec<Complex32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    // Row pass.
    let mut row_major: Vec<RustComplex32> = input
        .iter()
        .map(|c| RustComplex32::new(c.re, c.im))
        .collect();
    for row in row_major.chunks_exact_mut(n) {
        fft.process(row);
    }
    // Transpose.
    let mut col_major = vec![RustComplex32::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            col_major[j * n + i] = row_major[i * n + j];
        }
    }
    // Column pass (now row pass over the transposed buffer).
    for row in col_major.chunks_exact_mut(n) {
        fft.process(row);
    }
    // Transpose back so output is in row-major.
    let mut out = vec![RustComplex32::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = col_major[i * n + j];
        }
    }
    out.into_iter()
        .map(|c| Complex32::new(c.re, c.im))
        .collect()
}

#[test]
fn cross_check_n8_n64() {
    for log_n in [3u32, 6u32] {
        let plan = Plan2D::new(log_n, false);
        let n = plan.n_per_side();
        let total = n * n;
        let input: Vec<Complex32> = (0..total)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.07).cos()))
            .collect();
        let expected = rustfft_2d_forward(&input, n);
        let mut buf = input.clone();
        plan.cpu_execute(&mut buf, Direction::Forward);
        for i in 0..total {
            let dre = (buf[i].re - expected[i].re).abs();
            let dim = (buf[i].im - expected[i].im).abs();
            let rel = (dre + dim) / (expected[i].re.abs() + expected[i].im.abs() + 1.0);
            assert!(
                rel < 1e-4,
                "log_n={log_n} 2D mismatch at {i}: got ({}, {}), expected ({}, {}), rel={rel}",
                buf[i].re,
                buf[i].im,
                expected[i].re,
                expected[i].im
            );
        }
    }
}
