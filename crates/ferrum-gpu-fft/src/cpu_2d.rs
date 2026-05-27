//! CPU 2D radix-2 power-of-2 C2C FFT via row-then-column separable
//! application of the 1D Stockham.

use crate::complex::Complex32;
use crate::plan::{Direction, Plan, Plan2D};

impl Plan2D {
    /// Run a 2D FFT in place. `buf.len() == n_per_side^2`.
    pub fn cpu_execute(&self, buf: &mut [Complex32], dir: Direction) {
        let n = self.n_per_side();
        assert_eq!(
            buf.len(),
            n * n,
            "2D buf.len() {} != n_per_side^2 {}",
            buf.len(),
            n * n
        );

        // Row pass: n batched 1D transforms (each row is length n).
        // We construct a transient 1D Plan with batch = n and run it
        // forward. Inverse is handled at the end by conjugating + scaling.
        // To avoid double-scaling, do BOTH passes with normalize=false at
        // the 1D level and apply 1/(n*n) once at the end.
        let row_plan = Plan::new(self.log_n_per_side, n, /* normalize */ false);

        // Forward direction (inverse uses conjugate trick at the wrapper level).
        let inner_dir = match dir {
            Direction::Forward => Direction::Forward,
            // For inverse: conjugate the input, run forward, conjugate output,
            // optionally divide by n*n.
            Direction::Inverse => {
                for c in buf.iter_mut() {
                    *c = c.conj();
                }
                Direction::Forward
            }
        };

        // Row pass.
        row_plan.cpu_execute(buf, inner_dir);

        // Transpose in place.
        transpose_square(buf, n);

        // Column pass (which is a row pass on the transposed buffer).
        row_plan.cpu_execute(buf, inner_dir);

        // Transpose back to row-major.
        transpose_square(buf, n);

        // Post for inverse.
        if dir == Direction::Inverse {
            for c in buf.iter_mut() {
                *c = c.conj();
            }
            if self.normalize {
                let scale = 1.0f32 / (n as f32 * n as f32);
                for c in buf.iter_mut() {
                    c.re *= scale;
                    c.im *= scale;
                }
            }
        }
    }
}

fn transpose_square(buf: &mut [Complex32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            buf.swap(i * n + j, j * n + i);
        }
    }
}
