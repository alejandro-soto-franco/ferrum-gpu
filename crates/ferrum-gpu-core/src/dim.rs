//! Dimension and direction primitives.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// 3D grid or block dimension. Matches CUDA's `dim3` and Vulkan's
/// `WorkGroupSize`/`NumWorkGroups` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable, Serialize, Deserialize)]
pub struct Dim3 {
    /// X component.
    pub x: u32,
    /// Y component.
    pub y: u32,
    /// Z component.
    pub z: u32,
}

impl Dim3 {
    /// Construct a `Dim3` from three components.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of threads (or workgroup invocations) when used as a block dim.
    pub const fn product(self) -> u64 {
        (self.x as u64) * (self.y as u64) * (self.z as u64)
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self::new(x, 1, 1)
    }
}

/// FFT (or any signed-transform) direction.
#[repr(i8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Forward transform, sign = +1.
    Forward = 1,
    /// Inverse transform, sign = -1.
    Inverse = -1,
}

impl Direction {
    /// Sign convention as a small int. Useful for kernel scalar args.
    pub const fn sign(self) -> i32 {
        self as i32
    }
}
