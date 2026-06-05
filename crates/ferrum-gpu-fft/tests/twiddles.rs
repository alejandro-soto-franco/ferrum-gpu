//! Integration test crate.

use ferrum_gpu_fft::twiddles;

#[test]
fn twiddles_n4_outermost_value() {
    // For N=4, twiddles are W_4^k = exp(-2 pi i k / 4), k = 0..N/2 = 2.
    let tw = twiddles(2);
    assert!(!tw.is_empty(), "twiddles(2) must return at least one stage");
    let last = tw.last().unwrap();
    assert!((last.re - 1.0).abs() < 1e-6, "got re = {}", last.re);
    assert!(last.im.abs() < 1e-6, "got im = {}", last.im);
}

#[test]
fn twiddles_first_stage_is_one() {
    let tw = twiddles(3);
    assert!(!tw.is_empty());
    let first = tw[0];
    assert!((first.re - 1.0).abs() < 1e-6);
    assert!(first.im.abs() < 1e-6);
}

#[test]
fn twiddles_total_count_is_n_minus_one() {
    for log_n in [2u32, 3, 6, 8, 10, 12] {
        let n = 1usize << log_n;
        let tw = twiddles(log_n);
        assert_eq!(
            tw.len(),
            n - 1,
            "log_n={log_n}: expected {} twiddles, got {}",
            n - 1,
            tw.len()
        );
    }
}
