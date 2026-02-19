//! Camera math: perspective projection and look-at view matrix.
//!
//! All matrices use Vulkan clip-space conventions:
//! - Y is flipped (negative in projection)
//! - Depth range [0, 1]
//! - Column-major storage (glam default)

use glam::{Mat4, Vec3};

/// Create a perspective projection matrix for Vulkan clip space.
///
/// The Y axis is flipped (m[1][1] = -f) to match Vulkan's top-down convention.
/// Depth maps to [0, 1] (not [-1, 1] like OpenGL).
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y / 2.0).tan();

    // Build column-major matrix matching the Python playground exactly.
    // Column 0: (f/aspect, 0, 0, 0)
    // Column 1: (0, -f, 0, 0)        <- Y-flip for Vulkan
    // Column 2: (0, 0, far/(near-far), -1)
    // Column 3: (0, 0, near*far/(near-far), 0)
    Mat4::from_cols(
        glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -f, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, far / (near - far), -1.0),
        glam::Vec4::new(0.0, 0.0, (near * far) / (near - far), 0.0),
    )
}

/// Create a look-at view matrix (column-major).
///
/// Matches the Python playground's `look_at()` function.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    // Column-major storage matching Python numpy layout:
    // Column 0: (s.x, u.x, -f.x, 0)
    // Column 1: (s.y, u.y, -f.y, 0)
    // Column 2: (s.z, u.z, -f.z, 0)
    // Column 3: (-dot(s,eye), -dot(u,eye), dot(f,eye), 1)
    Mat4::from_cols(
        glam::Vec4::new(s.x, u.x, -f.x, 0.0),
        glam::Vec4::new(s.y, u.y, -f.y, 0.0),
        glam::Vec4::new(s.z, u.z, -f.z, 0.0),
        glam::Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Default camera parameters matching the Python playground.
pub struct DefaultCamera;

impl DefaultCamera {
    pub const EYE: Vec3 = Vec3::new(0.0, 0.0, 3.0);
    pub const TARGET: Vec3 = Vec3::ZERO;
    pub const UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const FOV_Y_DEG: f32 = 45.0;
    pub const NEAR: f32 = 0.1;
    pub const FAR: f32 = 100.0;

    pub fn view() -> Mat4 {
        look_at(Self::EYE, Self::TARGET, Self::UP)
    }

    pub fn projection(aspect: f32) -> Mat4 {
        perspective(Self::FOV_Y_DEG.to_radians(), aspect, Self::NEAR, Self::FAR)
    }

    pub fn model() -> Mat4 {
        Mat4::IDENTITY
    }
}
