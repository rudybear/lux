//! Scene geometry: vertex types, sphere mesh, triangle, procedural texture.

use bytemuck::{Pod, Zeroable};

/// PBR vertex: position + normal + UV = 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PbrVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Simple colored triangle vertex: position + color = 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TriangleVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

/// Generate a UV sphere with the given number of stacks and slices.
///
/// Returns (vertices, indices) matching the Python playground layout exactly.
pub fn generate_sphere(stacks: u32, slices: u32) -> (Vec<PbrVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        let phi = std::f32::consts::PI * i as f32 / stacks as f32;
        let v_coord = i as f32 / stacks as f32;
        for j in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / slices as f32;
            let u_coord = j as f32 / slices as f32;

            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();

            vertices.push(PbrVertex {
                position: [x, y, z],
                normal: [x, y, z],
                uv: [u_coord, v_coord],
            });
        }
    }

    for i in 0..stacks {
        for j in 0..slices {
            let a = i * (slices + 1) + j;
            let b = a + slices + 1;
            indices.push(a);
            indices.push(b);
            indices.push(a + 1);
            indices.push(b);
            indices.push(b + 1);
            indices.push(a + 1);
        }
    }

    (vertices, indices)
}

/// Generate a colored triangle (3 vertices) matching the Python playground.
///
/// - Top:          (0, 0.5, 0) — red
/// - Bottom-left:  (-0.5, -0.5, 0) — green
/// - Bottom-right: (0.5, -0.5, 0) — blue
pub fn generate_triangle() -> Vec<TriangleVertex> {
    vec![
        TriangleVertex {
            position: [0.0, 0.5, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        TriangleVertex {
            position: [-0.5, -0.5, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        TriangleVertex {
            position: [0.5, -0.5, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ]
}

/// Generate a procedural RGBA8 texture (checker pattern with sinusoidal color variation).
///
/// Matches the Python playground's `generate_procedural_texture()` exactly.
pub fn generate_procedural_texture(size: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32;
            let v = y as f32 / size as f32;

            let cx = (u * 8.0) as i32 % 2;
            let cy = (v * 8.0) as i32 % 2;
            let checker = cx ^ cy;

            let (r, g, b) = if checker != 0 {
                // Warm: terracotta / orange tones
                let r = 200.0 + 40.0 * (u * 12.0).sin();
                let g = 120.0 + 30.0 * (v * 10.0).sin();
                let b = 80.0 + 20.0 * (u * 8.0 + v * 6.0).cos();
                (r, g, b)
            } else {
                // Cool: teal / blue-green tones
                let r = 60.0 + 30.0 * (u * 10.0 + v * 4.0).sin();
                let g = 150.0 + 40.0 * (v * 8.0).cos();
                let b = 170.0 + 50.0 * (u * 6.0).sin();
                (r, g, b)
            };

            data.push(r.clamp(0.0, 255.0) as u8);
            data.push(g.clamp(0.0, 255.0) as u8);
            data.push(b.clamp(0.0, 255.0) as u8);
            data.push(255);
        }
    }

    data
}
