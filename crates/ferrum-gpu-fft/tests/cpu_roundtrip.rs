//! Integration test crate.

use ferrum_gpu_fft::{Complex32, Direction, Plan};

#[test]
fn cpu_roundtrip_n4() {
    let plan = Plan::new(2, 1, true);
    let n = plan.n();
    let input: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32 + 1.0, 2.0 * i as f32 - 1.0))
        .collect();
    let mut buf = input.clone();
    plan.cpu_execute(&mut buf, Direction::Forward);
    plan.cpu_execute(&mut buf, Direction::Inverse);
    for (i, (a, b)) in input.iter().zip(buf.iter()).enumerate() {
        let dre = a.re - b.re;
        let dim = a.im - b.im;
        assert!(
            dre.abs() < 1e-4 && dim.abs() < 1e-4,
            "roundtrip mismatch at {i}: expected {:?}, got {:?}",
            a,
            b
        );
    }
}

#[test]
fn cpu_roundtrip_n8() {
    let plan = Plan::new(3, 1, true);
    let n = plan.n();
    let input: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32, (n - i) as f32))
        .collect();
    let mut buf = input.clone();
    plan.cpu_execute(&mut buf, Direction::Forward);
    plan.cpu_execute(&mut buf, Direction::Inverse);
    for (i, (a, b)) in input.iter().zip(buf.iter()).enumerate() {
        let dre = a.re - b.re;
        let dim = a.im - b.im;
        assert!(
            dre.abs() < 1e-4 && dim.abs() < 1e-4,
            "n=8 roundtrip mismatch at {i}: expected {:?}, got {:?}",
            a,
            b
        );
    }
}

#[test]
fn cpu_roundtrip_n256() {
    let plan = Plan::new(8, 1, true);
    let n = plan.n();
    let input: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32, (n - i) as f32))
        .collect();
    let mut buf = input.clone();
    plan.cpu_execute(&mut buf, Direction::Forward);
    plan.cpu_execute(&mut buf, Direction::Inverse);
    for (i, (a, b)) in input.iter().zip(buf.iter()).enumerate() {
        let dre = a.re - b.re;
        let dim = a.im - b.im;
        assert!(
            dre.abs() < 1e-3 && dim.abs() < 1e-3,
            "n=256 roundtrip mismatch at {i}: expected {:?}, got {:?}",
            a,
            b
        );
    }
}
