//! Helpers for the interleaved-vs-split layout study: layout conversion and a
//! naive DFT used only as a correctness oracle for the fast kernels.

/// Naive O(N^2) forward DFT on interleaved `[re, im, ...]`. Oracle only.
pub fn dft_naive_interleaved(input: &[f32]) -> Vec<f32> {
    let n = input.len() / 2;
    let mut out = vec![0.0f32; 2 * n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for t in 0..n {
            let ang = -2.0 * core::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            let (c, s) = (ang.cos(), ang.sin());
            let (xr, xi) = (input[2 * t] as f64, input[2 * t + 1] as f64);
            sr += xr * c - xi * s;
            si += xr * s + xi * c;
        }
        out[2 * k] = sr as f32;
        out[2 * k + 1] = si as f32;
    }
    out
}

/// `[re, im, ...]` -> (`[re, ...]`, `[im, ...]`).
pub fn interleaved_to_split(inter: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = inter.len() / 2;
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    for t in 0..n {
        re[t] = inter[2 * t];
        im[t] = inter[2 * t + 1];
    }
    (re, im)
}

/// (`[re, ...]`, `[im, ...]`) -> `[re, im, ...]`.
pub fn split_to_interleaved(re: &[f32], im: &[f32]) -> Vec<f32> {
    let n = re.len();
    let mut out = vec![0.0f32; 2 * n];
    for t in 0..n {
        out[2 * t] = re[t];
        out[2 * t + 1] = im[t];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_roundtrip_is_identity() {
        let inter: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (re, im) = interleaved_to_split(&inter);
        assert_eq!(split_to_interleaved(&re, &im), inter);
    }

    #[test]
    fn dft_naive_matches_known_4pt() {
        // x = [1,2,3,4] real -> DFT = [10, -2+2i, -2, -2-2i]
        let x = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let y = dft_naive_interleaved(&x);
        let expect = [10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
        for (a, b) in y.iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}
