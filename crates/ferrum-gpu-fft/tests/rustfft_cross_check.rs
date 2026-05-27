//! Integration test crate.

use ferrum_gpu_fft::{Complex32, Direction, Plan};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32 as RustComplex32;

fn run_rustfft_forward(input: &[Complex32]) -> Vec<Complex32> {
    let mut planner = FftPlanner::new();
    let n = input.len();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<RustComplex32> = input.iter().map(|c| RustComplex32::new(c.re, c.im)).collect();
    fft.process(&mut buf);
    buf.into_iter().map(|c| Complex32::new(c.re, c.im)).collect()
}

#[test]
fn cross_check_n4_n8_n64() {
    for log_n in [2u32, 3, 6] {
        let plan = Plan::new(log_n, 1, false);
        let n = plan.n();
        let input: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
            .collect();
        let expected = run_rustfft_forward(&input);
        let mut buf = input.clone();
        plan.cpu_execute(&mut buf, Direction::Forward);
        for i in 0..n {
            let dre = (buf[i].re - expected[i].re).abs();
            let dim = (buf[i].im - expected[i].im).abs();
            let rel = (dre + dim) / (expected[i].re.abs() + expected[i].im.abs() + 1.0);
            assert!(
                rel < 1e-4,
                "log_n={log_n} mismatch at {i}: got ({}, {}), expected ({}, {}), rel={}",
                buf[i].re, buf[i].im, expected[i].re, expected[i].im, rel
            );
        }
    }
}
