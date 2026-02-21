//! CPU-side meshlet building from indexed triangle meshes.
//!
//! Produces meshlet descriptors, per-meshlet vertex index lists, and
//! per-meshlet local triangle index lists suitable for GPU mesh shader
//! consumption.

/// GPU-side meshlet descriptor (matches shader uvec4 layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshletDescriptor {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub triangle_offset: u32,
    pub triangle_count: u32,
}

// Safety: MeshletDescriptor is repr(C) with only u32 fields.
unsafe impl bytemuck::Pod for MeshletDescriptor {}
unsafe impl bytemuck::Zeroable for MeshletDescriptor {}

/// Result of meshlet building: descriptors + flattened vertex/triangle index arrays.
pub struct MeshletBuildResult {
    pub meshlets: Vec<MeshletDescriptor>,
    /// Global vertex indices per meshlet (concatenated).
    pub meshlet_vertices: Vec<u32>,
    /// Local triangle indices per meshlet (3 per triangle, concatenated).
    pub meshlet_triangles: Vec<u32>,
}

/// Build meshlets from an indexed triangle mesh using greedy partitioning.
///
/// Each meshlet will contain at most `max_vertices` unique vertices and
/// `max_primitives` triangles.
pub fn build_meshlets(
    indices: &[u32],
    _vertex_count: u32,
    max_vertices: u32,
    max_primitives: u32,
) -> MeshletBuildResult {
    let mut result = MeshletBuildResult {
        meshlets: Vec::new(),
        meshlet_vertices: Vec::new(),
        meshlet_triangles: Vec::new(),
    };

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return result;
    }

    let mut current_vertices: Vec<u32> = Vec::new();
    let mut current_triangles: Vec<u32> = Vec::new();

    let finalize = |verts: &mut Vec<u32>,
                        tris: &mut Vec<u32>,
                        result: &mut MeshletBuildResult| {
        if tris.is_empty() {
            return;
        }

        let desc = MeshletDescriptor {
            vertex_offset: result.meshlet_vertices.len() as u32,
            vertex_count: verts.len() as u32,
            triangle_offset: result.meshlet_triangles.len() as u32,
            triangle_count: (tris.len() / 3) as u32,
        };

        result.meshlets.push(desc);
        result.meshlet_vertices.extend_from_slice(verts);
        result.meshlet_triangles.extend_from_slice(tris);

        verts.clear();
        tris.clear();
    };

    for t in 0..triangle_count {
        let i0 = indices[t * 3];
        let i1 = indices[t * 3 + 1];
        let i2 = indices[t * 3 + 2];

        // Count new vertices this triangle would add
        let new_verts = [i0, i1, i2]
            .iter()
            .filter(|&&idx| !current_vertices.contains(&idx))
            .count() as u32;

        // Check limits — flush current meshlet if this triangle would exceed them
        if current_vertices.len() as u32 + new_verts > max_vertices
            || current_triangles.len() as u32 / 3 + 1 > max_primitives
        {
            finalize(&mut current_vertices, &mut current_triangles, &mut result);
        }

        // Add vertices, get local indices
        let get_or_add = |v: u32, verts: &mut Vec<u32>| -> u32 {
            if let Some(pos) = verts.iter().position(|&x| x == v) {
                pos as u32
            } else {
                let idx = verts.len() as u32;
                verts.push(v);
                idx
            }
        };

        let l0 = get_or_add(i0, &mut current_vertices);
        let l1 = get_or_add(i1, &mut current_vertices);
        let l2 = get_or_add(i2, &mut current_vertices);

        current_triangles.push(l0);
        current_triangles.push(l1);
        current_triangles.push(l2);
    }

    // Finalize last meshlet
    finalize(&mut current_vertices, &mut current_triangles, &mut result);

    result
}
