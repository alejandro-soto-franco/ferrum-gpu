//! Plan struct. Real impl lands in P3T2.

/// Placeholder. Filled in in P3T2.
pub struct Plan;

/// Transform direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Forward transform.
    Forward,
    /// Inverse transform.
    Inverse,
}
