//! Forward-then-inverse identity for 2D CPU FFT.

use ferrum_gpu_fft::{Complex32, Direction, Plan2D};

#[test]
fn cpu_2d_roundtrip_n4() {
    let plan = Plan2D::new(2, true);
    let n = plan.n_per_side();
    let total = n * n;
    let input: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new(i as f32 + 1.0, 2.0 * i as f32 - 0.5))
        .collect();
    let mut buf = input.clone();
    plan.cpu_execute(&mut buf, Direction::Forward);
    plan.cpu_execute(&mut buf, Direction::Inverse);
    for (i, (a, b)) in input.iter().zip(buf.iter()).enumerate() {
        let dre = a.re - b.re;
        let dim = a.im - b.im;
        assert!(
            dre.abs() < 1e-4 && dim.abs() < 1e-4,
            "2D N=4 roundtrip mismatch at {i}: expected {:?}, got {:?}", a, b
        );
    }
}

#[test]
fn cpu_2d_roundtrip_n64() {
    let plan = Plan2D::new(6, true);
    let n = plan.n_per_side();
    let total = n * n;
    let input: Vec<Complex32> = (0..total)
        .map(|i| Complex32::new(((i % n) as f32).sin(), ((i / n) as f32).cos()))
        .collect();
    let mut buf = input.clone();
    plan.cpu_execute(&mut buf, Direction::Forward);
    plan.cpu_execute(&mut buf, Direction::Inverse);
    for (i, (a, b)) in input.iter().zip(buf.iter()).enumerate() {
        let dre = a.re - b.re;
        let dim = a.im - b.im;
        assert!(
            dre.abs() < 1e-3 && dim.abs() < 1e-3,
            "2D N=64 roundtrip mismatch at {i}: expected {:?}, got {:?}", a, b
        );
    }
}
