//! Rasterization renderer: scene/pipeline architecture.
//!
//! Two public entry points:
//! - `render_raster()` — render a raster scene (sphere, triangle, or glTF) through a pipeline
//! - `render_fullscreen()` — render a fullscreen fragment shader

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::{Allocation, Allocator};
use log::info;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;
use gpu_allocator::MemoryLocation;

use crate::camera::DefaultCamera;
use crate::reflected_pipeline;
use crate::scene;
use crate::scene_manager::{self, GpuBuffer, GpuImage, IblAssets};
use crate::screenshot;
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

// ===========================================================================
// Shadow map constants and embedded SPIR-V
// ===========================================================================

/// Maximum number of shadow-casting lights.
const MAX_SHADOW_MAPS: usize = scene_manager::MAX_SHADOW_MAPS;

/// Shadow map resolution (width and height).
const SHADOW_MAP_RESOLUTION: u32 = scene_manager::SHADOW_MAP_RESOLUTION;

/// Embedded SPIR-V for the shadow depth vertex shader.
///
/// This is the compiled bytecode for:
/// ```glsl
/// #version 450
/// layout(push_constant) uniform PC { mat4 mvp; };
/// layout(location = 0) in vec3 inPosition;
/// void main() { gl_Position = mvp * vec4(inPosition, 1.0); }
/// ```
#[rustfmt::skip]
const SHADOW_VERT_SPV: &[u8] = &[
    0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0b, 0x00, 0x08, 0x00,
    0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
    0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
    0x0d, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00,
    0x02, 0x00, 0x00, 0x00, 0xc2, 0x01, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
    0x05, 0x00, 0x06, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x67, 0x6c, 0x5f, 0x50,
    0x65, 0x72, 0x56, 0x65, 0x72, 0x74, 0x65, 0x78, 0x00, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x06, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x67, 0x6c, 0x5f, 0x50, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x00,
    0x06, 0x00, 0x07, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x67, 0x6c, 0x5f, 0x50, 0x6f, 0x69, 0x6e, 0x74, 0x53, 0x69, 0x7a, 0x65,
    0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x07, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x67, 0x6c, 0x5f, 0x43, 0x6c, 0x69, 0x70, 0x44,
    0x69, 0x73, 0x74, 0x61, 0x6e, 0x63, 0x65, 0x00, 0x06, 0x00, 0x07, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x67, 0x6c, 0x5f, 0x43,
    0x75, 0x6c, 0x6c, 0x44, 0x69, 0x73, 0x74, 0x61, 0x6e, 0x63, 0x65, 0x00,
    0x05, 0x00, 0x03, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x05, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00, 0x50, 0x43, 0x00, 0x00,
    0x06, 0x00, 0x04, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x6d, 0x76, 0x70, 0x00, 0x05, 0x00, 0x03, 0x00, 0x13, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00, 0x19, 0x00, 0x00, 0x00,
    0x69, 0x6e, 0x50, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x00, 0x00,
    0x47, 0x00, 0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x48, 0x00, 0x04, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
    0x48, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
    0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x19, 0x00, 0x00, 0x00,
    0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00,
    0x08, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x2b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x06, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
    0x0c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    0x3b, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
    0x0e, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x18, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00,
    0x09, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
    0x12, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
    0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
    0x3b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x20, 0x00, 0x04, 0x00,
    0x21, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00,
    0x15, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
    0x3d, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
    0x15, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
    0x1a, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
    0x06, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
    0x1d, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
    0x1a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
    0x1d, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00,
    0x91, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x16, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00,
    0x21, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x22, 0x00, 0x00, 0x00,
    0x20, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00,
];

/// Material UBO data matching `properties Material { ... }` in gltf_pbr_layered.lux (std140 layout, 80 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MaterialUboData {
    base_color_factor: [f32; 4],          // offset  0, size 16
    emissive_factor: [f32; 3],            // offset 16, size 12
    metallic_factor: f32,                 // offset 28, size  4
    roughness_factor: f32,                // offset 32, size  4
    emissive_strength: f32,               // offset 36, size  4
    ior: f32,                             // offset 40, size  4
    clearcoat_factor: f32,                // offset 44, size  4
    clearcoat_roughness_factor: f32,      // offset 48, size  4
    sheen_roughness_factor: f32,          // offset 52, size  4
    transmission_factor: f32,             // offset 56, size  4
    _pad_before_sheen: f32,              // offset 60, size  4 (padding for vec3 alignment)
    sheen_color_factor: [f32; 3],         // offset 64, size 12
    _pad0: f32,                           // offset 76, size  4
    // KHR_texture_transform (offset 80 -> 144)
    base_color_uv_st: [f32; 4],          // offset 80: [offset.x, offset.y, scale.x, scale.y]
    normal_uv_st: [f32; 4],              // offset 96
    mr_uv_st: [f32; 4],                  // offset 112
    base_color_uv_rot: f32,              // offset 128
    normal_uv_rot: f32,                  // offset 132
    mr_uv_rot: f32,                      // offset 136
    _pad1: f32,                          // offset 140
}
unsafe impl bytemuck::Pod for MaterialUboData {}
unsafe impl bytemuck::Zeroable for MaterialUboData {}

impl MaterialUboData {
    fn default_values() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_strength: 1.0,
            ior: 1.5,
            clearcoat_factor: 0.0,
            clearcoat_roughness_factor: 0.0,
            sheen_roughness_factor: 0.0,
            transmission_factor: 0.0,
            _pad_before_sheen: 0.0,
            sheen_color_factor: [0.0, 0.0, 0.0],
            _pad0: 0.0,
            base_color_uv_st: [0.0, 0.0, 1.0, 1.0],
            normal_uv_st: [0.0, 0.0, 1.0, 1.0],
            mr_uv_st: [0.0, 0.0, 1.0, 1.0],
            base_color_uv_rot: 0.0,
            normal_uv_rot: 0.0,
            mr_uv_rot: 0.0,
            _pad1: 0.0,
        }
    }

    fn from_gltf_material(mat: &crate::gltf_loader::GltfMaterial) -> Self {
        Self {
            base_color_factor: mat.base_color,
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            emissive_factor: mat.emissive,
            emissive_strength: mat.emissive_strength,
            ior: mat.ior,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness_factor: mat.clearcoat_roughness_factor,
            sheen_color_factor: mat.sheen_color_factor,
            sheen_roughness_factor: mat.sheen_roughness_factor,
            transmission_factor: mat.transmission_factor,
            _pad_before_sheen: 0.0,
            _pad0: 0.0,
            base_color_uv_st: [
                mat.base_color_uv_xform.offset[0], mat.base_color_uv_xform.offset[1],
                mat.base_color_uv_xform.scale[0], mat.base_color_uv_xform.scale[1],
            ],
            normal_uv_st: [
                mat.normal_uv_xform.offset[0], mat.normal_uv_xform.offset[1],
                mat.normal_uv_xform.scale[0], mat.normal_uv_xform.scale[1],
            ],
            mr_uv_st: [
                mat.metallic_roughness_uv_xform.offset[0], mat.metallic_roughness_uv_xform.offset[1],
                mat.metallic_roughness_uv_xform.scale[0], mat.metallic_roughness_uv_xform.scale[1],
            ],
            base_color_uv_rot: mat.base_color_uv_xform.rotation,
            normal_uv_rot: mat.normal_uv_xform.rotation,
            mr_uv_rot: mat.metallic_roughness_uv_xform.rotation,
            _pad1: 0.0,
        }
    }
}

// ===========================================================================
// Shader manifest: permutation selection
// ===========================================================================

/// A single permutation entry from the manifest.
#[derive(Debug, Clone, serde::Deserialize)]
struct ManifestPermutation {
    suffix: String,
    features: HashMap<String, bool>,
}

/// Parsed shader manifest describing available permutations.
#[derive(Debug, Clone, serde::Deserialize)]
struct ShaderManifest {
    #[serde(default)]
    pipeline: String,
    #[serde(default)]
    features: Vec<String>,
    #[serde(default)]
    permutations: Vec<ManifestPermutation>,
}

impl Default for ShaderManifest {
    fn default() -> Self {
        Self {
            pipeline: String::new(),
            features: Vec::new(),
            permutations: Vec::new(),
        }
    }
}

/// Parse a .manifest.json file into a ShaderManifest struct.
fn parse_manifest_json(path: &Path) -> Option<ShaderManifest> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Try to load manifest from a pipeline base path.
/// Checks for basePath + ".manifest.json" and the legacy gltf_pbr/ subdirectory.
fn try_load_manifest(pipeline_base: &str) -> Option<ShaderManifest> {
    // Try direct: pipelineBase + ".manifest.json"
    let path1 = format!("{}.manifest.json", pipeline_base);
    if Path::new(&path1).exists() {
        info!("Loading shader manifest: {}", path1);
        if let Some(manifest) = parse_manifest_json(Path::new(&path1)) {
            if !manifest.permutations.is_empty() {
                return Some(manifest);
            }
        }
    }

    // Try legacy subdirectory format: shadercache/gltf_pbr/gltf_pbr_layered.manifest.json
    let p = std::path::Path::new(pipeline_base);
    if let (Some(dir), Some(filename)) = (p.parent(), p.file_name()) {
        let path2 = dir.join("gltf_pbr").join(format!("{}.manifest.json", filename.to_string_lossy()));
        if path2.exists() {
            info!("Loading shader manifest: {}", path2.display());
            if let Some(manifest) = parse_manifest_json(&path2) {
                if !manifest.permutations.is_empty() {
                    return Some(manifest);
                }
            }
        }
    }

    None // no manifest found
}

/// Find the best matching permutation suffix for a set of material features.
/// Returns "" (base) if no exact match found.
fn find_permutation_suffix(manifest: &ShaderManifest, features: &BTreeSet<String>) -> String {
    // Build feature map from the material's features
    let mut wanted: HashMap<String, bool> = HashMap::new();
    for fname in &manifest.features {
        wanted.insert(fname.clone(), features.contains(fname));
    }

    // Find exact match in manifest permutations
    for perm in &manifest.permutations {
        let mut is_match = true;
        for fname in &manifest.features {
            let perm_has = perm.features.get(fname).copied().unwrap_or(false);
            let want = *wanted.get(fname).unwrap_or(&false);
            if perm_has != want {
                is_match = false;
                break;
            }
        }
        if is_match {
            return perm.suffix.clone();
        }
    }

    // Fallback: return base (empty suffix)
    String::new()
}

/// Per-permutation pipeline data for multi-material rendering.
struct PermutationPipeline {
    suffix: String,
    base_path: String,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    #[allow(dead_code)]
    vert_refl: reflected_pipeline::ReflectionData,
    #[allow(dead_code)]
    frag_refl: reflected_pipeline::ReflectionData,
    material_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    per_material_desc_sets: Vec<vk::DescriptorSet>,
    per_material_ubos: Vec<GpuBuffer>,
    material_indices: Vec<usize>,
}

/// Built-in fullscreen vertex shader as pre-compiled SPIR-V.
///
/// This is the SPIR-V bytecode for:
/// ```glsl
/// #version 450
/// layout(location = 0) out vec2 uv;
/// void main() {
///     vec2 positions[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
///     gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
///     vec2 p = positions[gl_VertexIndex];
///     uv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
/// }
/// ```
///
/// We embed the SPIR-V inline so no external file is needed.
/// This was assembled from the GLSL above; if you need to regenerate it,
/// use: glslangValidator -V fullscreen.vert -o fullscreen.vert.spv
///
/// For portability, we define a minimal SPIR-V module directly.
fn fullscreen_vert_spirv() -> Vec<u32> {
    // Instead of embedding raw bytes, we compile the fullscreen shader at build time
    // by writing the GLSL to a temp file and invoking glslangValidator.
    // However, for maximum portability, we embed the known-good SPIR-V directly.
    //
    // This SPIR-V was generated from the GLSL above and is architecture-independent.
    vec![
        // SPIR-V magic
        0x07230203,
        // Version 1.0
        0x00010000,
        // Generator (Khronos glslang)
        0x000d000b,
        // Bound
        0x00000030,
        // Schema
        0x00000000,
        // OpCapability Shader
        0x00020011, 0x00000001,
        // OpMemoryModel Logical GLSL450
        0x0003000e, 0x00000000, 0x00000001,
        // OpEntryPoint Vertex %main "main" %gl_VertexIndex %_ %uv
        0x0008000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000,
        0x0000000d, 0x00000012, 0x0000002b,
        // OpSource GLSL 450
        0x00030003, 0x00000002, 0x000001c2,
        // OpName %main "main"
        0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
        // OpName %positions "positions"
        0x00050005, 0x00000009, 0x69736f70, 0x6e6f6974, 0x00000073,
        // OpName %gl_VertexIndex "gl_VertexIndex"
        0x00060005, 0x0000000d, 0x565f6c67, 0x65747265, 0x646e4978, 0x00007865,
        // OpName %gl_PerVertex "gl_PerVertex"
        0x00060005, 0x00000010, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
        // OpMemberName %gl_PerVertex 0 "gl_Position"
        0x00060006, 0x00000010, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
        // OpMemberName %gl_PerVertex 1 "gl_PointSize"
        0x00070006, 0x00000010, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
        0x00000000,
        // OpMemberName %gl_PerVertex 2 "gl_ClipDistance"
        0x00070006, 0x00000010, 0x00000002, 0x435f6c67, 0x4470696c, 0x61747369,
        0x0065636e,
        // OpMemberName %gl_PerVertex 3 "gl_CullDistance"
        0x00070006, 0x00000010, 0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
        0x0065636e,
        // OpName %_ ""
        0x00030005, 0x00000012, 0x00000000,
        // OpName %uv "uv"
        0x00030005, 0x0000002b, 0x00007675,
        // OpDecorate %gl_VertexIndex BuiltIn VertexIndex
        0x00040047, 0x0000000d, 0x0000000b, 0x0000002a,
        // OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
        0x00050048, 0x00000010, 0x00000000, 0x0000000b, 0x00000000,
        // OpMemberDecorate %gl_PerVertex 1 BuiltIn PointSize
        0x00050048, 0x00000010, 0x00000001, 0x0000000b, 0x00000001,
        // OpMemberDecorate %gl_PerVertex 2 BuiltIn ClipDistance
        0x00050048, 0x00000010, 0x00000002, 0x0000000b, 0x00000003,
        // OpMemberDecorate %gl_PerVertex 3 BuiltIn CullDistance
        0x00050048, 0x00000010, 0x00000003, 0x0000000b, 0x00000004,
        // OpDecorate %gl_PerVertex Block
        0x00030047, 0x00000010, 0x00000002,
        // OpDecorate %uv Location 0
        0x00040047, 0x0000002b, 0x0000001e, 0x00000000,
    ]
}

/// Delegate to scene_manager::create_buffer_with_data.
fn create_buffer_with_data(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    scene_manager::create_buffer_with_data(device, allocator, data, usage, name)
}

/// Delegate to scene_manager::create_offscreen_image.
fn create_offscreen_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_offscreen_image(device, allocator, width, height, format, usage, aspect, name)
}

/// Create a texture image from pixel data and upload it via a staging buffer.
/// Delegates to scene_manager::create_texture_image with FRAGMENT_SHADER dst_stage.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image(ctx, width, height, pixels, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Render a raster scene (sphere, triangle, or glTF) through a pipeline.
///
/// Three-phase architecture:
/// 1. Upload scene geometry to GPU
/// 2. Create pipeline from shaders
/// 3. Bind scene to pipeline and execute render
pub fn render_raster(
    ctx: &mut VulkanContext,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
    demo_lights: bool,
) -> Result<(), String> {
    info!(
        "Raster render: scene='{}', pipeline='{}', {}x{}",
        scene_source, pipeline_base, width, height
    );

    match scene_source {
        "triangle" => {
            let vert_path = format!("{}.vert.spv", pipeline_base);
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            render_triangle_scene(ctx, vert_module, frag_module, width, height, output_path)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
        source if source.ends_with(".glb") || source.ends_with(".gltf") => {
            // glTF scene: load mesh, use PBR pipeline
            // Resolve permutation suffix from manifest + scene features
            let resolved_base = if let Some(manifest) = try_load_manifest(pipeline_base) {
                let gltf_scene = crate::gltf_loader::load_gltf(Path::new(source))?;
                let demo_light_vec = if demo_lights { crate::setup_demo_lights() } else { vec![] };
                let scene_features = scene_manager::detect_scene_features(&gltf_scene, &demo_light_vec);
                let suffix = find_permutation_suffix(&manifest, &scene_features);
                if !suffix.is_empty() {
                    let candidate = format!("{}{}", pipeline_base, suffix);
                    if Path::new(&format!("{}.vert.spv", candidate)).exists()
                        && Path::new(&format!("{}.frag.spv", candidate)).exists()
                    {
                        info!("Resolved headless permutation: {}", suffix);
                        candidate
                    } else {
                        pipeline_base.to_string()
                    }
                } else {
                    pipeline_base.to_string()
                }
            } else {
                pipeline_base.to_string()
            };
            let vert_path = format!("{}.vert.spv", resolved_base);
            let frag_path = format!("{}.frag.spv", resolved_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            // For glTF, use the PBR rendering path with glTF mesh data
            render_pbr_scene(ctx, vert_module, frag_module, &resolved_base, scene_source, width, height, output_path, ibl_name, demo_lights)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
        _ => {
            // Default: sphere or other procedural scenes use PBR pipeline
            let vert_path = format!("{}.vert.spv", pipeline_base);
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let vert_code = spv_loader::load_spirv(Path::new(&vert_path))?;
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let vert_module = spv_loader::create_shader_module(&ctx.device, &vert_code)?;
            let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

            render_pbr_scene(ctx, vert_module, frag_module, pipeline_base, scene_source, width, height, output_path, ibl_name, demo_lights)?;

            unsafe {
                ctx.device.destroy_shader_module(frag_module, None);
                ctx.device.destroy_shader_module(vert_module, None);
            }
        }
    }

    Ok(())
}

/// Render a fullscreen fragment shader.
pub fn render_fullscreen(
    ctx: &mut VulkanContext,
    pipeline_base: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    _ibl_name: &str,
) -> Result<(), String> {
    info!(
        "Fullscreen render: pipeline='{}', {}x{}",
        pipeline_base, width, height
    );

    let frag_path = format!("{}.frag.spv", pipeline_base);
    let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
    let frag_module = spv_loader::create_shader_module(&ctx.device, &frag_code)?;

    render_fullscreen_scene(ctx, frag_module, width, height, output_path)?;

    unsafe {
        ctx.device.destroy_shader_module(frag_module, None);
    }

    Ok(())
}

// ---- Triangle scene rendering ----

fn render_triangle_scene(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Create vertex buffer
    let vertices = scene::generate_triangle();
    let vertex_data: &[u8] = bytemuck::cast_slice(&vertices);
    let mut vbo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        vertex_data,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        "triangle_vbo",
    )?;

    // Create offscreen color image
    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "triangle_color",
    )?;

    // Create render pass
    let color_attachment = vk::AttachmentDescription::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));

    let tri_dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&tri_dependencies);

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?
    };

    // Create framebuffer
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&color_image.view))
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
    };

    // Pipeline layout (no descriptors for triangle)
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?
    };

    // Graphics pipeline
    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let binding_desc = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(24) // sizeof(TriangleVertex)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attr_descs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .x(0.0)
        .y(0.0)
        .width(width as f32)
        .height(height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?[0]
    };

    // Record commands
    let cmd = ctx.begin_single_commands()?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    ctx.cmd_begin_label(cmd, "Raster Pass", [0.2, 0.8, 0.2, 1.0]);
    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_bind_vertex_buffers(cmd, 0, &[vbo.buffer], &[0]);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);
    }
    ctx.cmd_end_label(cmd);

    // Copy to staging and read back
    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved triangle render to {:?}", output_path);

    // Cleanup
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    vbo.destroy(device, ctx.allocator_mut());

    Ok(())
}

// ---- Fullscreen scene rendering ----

fn render_fullscreen_scene(
    ctx: &mut VulkanContext,
    frag_module: vk::ShaderModule,
    width: u32,
    height: u32,
    output_path: &Path,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Create built-in fullscreen vertex shader
    // Since embedding raw SPIR-V is fragile, we use a minimal hand-assembled approach.
    // For production, the SPIR-V should be generated from glslangValidator.
    // Here we use the fullscreen triangle approach with a hardcoded vertex shader.

    // We'll load a pre-compiled fullscreen.vert.spv if available, otherwise error.
    // Actually, let's generate a proper SPIR-V module programmatically.
    // The simplest approach: include the SPIR-V bytes directly.

    // For now, attempt to load from a known path, or use the assembly.
    let vert_module = create_builtin_fullscreen_vert_module(device)?;

    // Create offscreen color image
    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "fullscreen_color",
    )?;

    // Create render pass
    let color_attachment = vk::AttachmentDescription::default()
        .format(vk::Format::R8G8B8A8_UNORM)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));

    let fs_dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&fs_dependencies);

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create render pass: {:?}", e))?
    };

    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(std::slice::from_ref(&color_image.view))
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
    };

    // Pipeline (no vertex input, no descriptors)
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))?
    };

    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .width(width as f32)
        .height(height as f32)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create graphics pipeline: {:?}", e))?[0]
    };

    // Record commands
    let cmd = ctx.begin_single_commands()?;

    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    }];

    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    ctx.cmd_begin_label(cmd, "Raster Pass", [0.2, 0.8, 0.2, 1.0]);
    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(cmd);
    }
    ctx.cmd_end_label(cmd);

    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved fullscreen render to {:?}", output_path);

    // Cleanup
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_shader_module(vert_module, None);
    }
    color_image.destroy(device, ctx.allocator_mut());

    Ok(())
}

/// Create the built-in fullscreen triangle vertex shader module.
///
/// This attempts to load a pre-compiled fullscreen.vert.spv from common paths,
/// or generates SPIR-V at runtime by invoking glslangValidator.
fn create_builtin_fullscreen_vert_module(
    device: &ash::Device,
) -> Result<vk::ShaderModule, String> {
    // Try to find a pre-compiled fullscreen.vert.spv
    let search_paths = [
        "fullscreen.vert.spv",
        "shaders/fullscreen.vert.spv",
        "../shaders/fullscreen.vert.spv",
    ];

    for path in &search_paths {
        if let Ok(code) = spv_loader::load_spirv(Path::new(path)) {
            return spv_loader::create_shader_module(device, &code);
        }
    }

    // Try runtime compilation via glslangValidator
    let glsl_source = r#"#version 450
layout(location = 0) out vec2 uv;
void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    vec2 p = positions[gl_VertexIndex];
    uv = vec2(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
}
"#;

    // Write to temp file and compile
    let temp_dir = std::env::temp_dir();
    let glsl_path = temp_dir.join("lux_fullscreen.vert");
    let spv_path = temp_dir.join("lux_fullscreen.vert.spv");

    std::fs::write(&glsl_path, glsl_source)
        .map_err(|e| format!("Failed to write temp GLSL: {}", e))?;

    let result = std::process::Command::new("glslangValidator")
        .args([
            "-V",
            glsl_path.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ])
        .output();

    match result {
        Ok(output) if output.status.success() => {
            let code = spv_loader::load_spirv(&spv_path)?;
            let module = spv_loader::create_shader_module(device, &code)?;
            let _ = std::fs::remove_file(&glsl_path);
            let _ = std::fs::remove_file(&spv_path);
            Ok(module)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "glslangValidator failed to compile fullscreen vertex shader: {}",
                stderr
            ))
        }
        Err(_) => Err(
            "No pre-compiled fullscreen.vert.spv found and glslangValidator is not on PATH. \
             Please compile the fullscreen vertex shader manually:\n\
             glslangValidator -V fullscreen.vert -o fullscreen.vert.spv"
                .to_string(),
        ),
    }
}

// ---- PBR scene rendering (sphere, glTF, or other) ----
//
// Reflection-driven: reads .vert.json and .frag.json to determine descriptor
// set layouts, pool sizes, and binding assignments. No hardcoded layouts.

/// Create a 1x1 white RGBA default texture image (used when a texture slot
/// is declared in the reflection but no actual image data is available).
fn create_default_white_texture(ctx: &mut VulkanContext) -> Result<GpuImage, String> {
    let white_pixel: [u8; 4] = [255, 255, 255, 255];
    create_texture_image(ctx, 1, 1, &white_pixel, "default_white_1x1")
}

/// Delegate to scene_manager::create_cubemap_image with FRAGMENT_SHADER dst_stage.
#[allow(dead_code)]
fn create_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    format: vk::Format,
    pixel_data: &[u8],
    bytes_per_pixel: u32,
    name: &str,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    scene_manager::create_cubemap_image(ctx, face_size, mip_count, format, pixel_data, bytes_per_pixel, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Delegate to scene_manager::create_texture_image_raw with FRAGMENT_SHADER dst_stage.
#[allow(dead_code)]
fn create_texture_image_raw(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixel_data: &[u8],
    format: vk::Format,
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image_raw(ctx, width, height, pixel_data, format, name, vk::PipelineStageFlags::FRAGMENT_SHADER)
}

/// Delegate to scene_manager::load_ibl_assets with FRAGMENT_SHADER dst_stage.
fn load_ibl_assets(ctx: &mut VulkanContext, ibl_name: &str) -> IblAssets {
    scene_manager::load_ibl_assets(ctx, vk::PipelineStageFlags::FRAGMENT_SHADER, ibl_name)
}

/// Bindless raster rendering path: single pipeline, materials SSBO, texture array.
///
/// Uses push constants {model: mat4, material_index: uint} per draw range
/// and bindless descriptor sets with UPDATE_AFTER_BIND for the texture array.
#[allow(clippy::too_many_arguments)]
fn render_pbr_scene_bindless(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    _pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
    vert_refl: reflected_pipeline::ReflectionData,
    frag_refl: reflected_pipeline::ReflectionData,
    demo_lights: bool,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // Merge descriptor sets
    let merged = reflected_pipeline::merge_descriptor_sets(&[&vert_refl, &frag_refl]);

    // Use bindless layout creation
    let ds_layouts = unsafe {
        reflected_pipeline::create_descriptor_set_layouts_bindless(device, &merged)?
    };

    // Load glTF scene
    let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");
    let mut gltf_scene = if is_gltf {
        Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
    } else {
        None
    };

    // Build push constant ranges from reflection
    let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
    for pc in &frag_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }
    // Merge vertex push constants if present
    for pc in &vert_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }

    // Pipeline layout
    let pipeline_layout = unsafe {
        reflected_pipeline::create_pipeline_layout_from_merged(device, &ds_layouts, &push_ranges)?
    };

    // --- Geometry ---
    let pipeline_vertex_stride = vert_refl.vertex_stride;
    let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
    let mut auto_target = glam::Vec3::ZERO;
    let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
    let mut auto_far = 100.0f32;
    let mut has_scene_bounds = false;

    let (vertex_data_owned, pbr_indices, draw_ranges) = if let Some(ref mut gltf_s) = gltf_scene {
        if gltf_s.meshes.is_empty() {
            return Err(format!("No meshes found in glTF file: {}", scene_source));
        }
        let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
        if draw_items.is_empty() {
            return Err(format!("No draw items in glTF scene: {}", scene_source));
        }

        let mut vdata: Vec<f32> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut vertex_offset: u32 = 0;
        let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
        let mut draw_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();
        let mut index_offset: u32 = 0;

        for item in &draw_items {
            let gmesh = &gltf_s.meshes[item.mesh_index];
            let world = item.world_transform;
            let upper3x3 = glam::Mat3::from_mat4(world);
            let normal_mat = upper3x3.inverse().transpose();

            for gv in &gmesh.vertices {
                let pos = world.transform_point3(glam::Vec3::from(gv.position));
                let norm = (normal_mat * glam::Vec3::from(gv.normal)).normalize();
                vertex_positions.push([pos.x, pos.y, pos.z]);
                vdata.push(pos.x); vdata.push(pos.y); vdata.push(pos.z);
                vdata.push(norm.x); vdata.push(norm.y); vdata.push(norm.z);
                vdata.push(gv.uv[0]); vdata.push(gv.uv[1]);

                if pipeline_vertex_stride >= 48 {
                    if gmesh.has_tangents {
                        let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                        vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                        vdata.push(gv.tangent[3]);
                    } else {
                        vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                    }
                }
            }

            for &idx in &gmesh.indices {
                all_indices.push(idx + vertex_offset);
            }
            vertex_offset += gmesh.vertices.len() as u32;

            draw_ranges.push(crate::gltf_loader::DrawRange {
                index_offset,
                index_count: gmesh.indices.len() as u32,
                material_index: item.material_index,
            });
            index_offset += gmesh.indices.len() as u32;
        }

        has_scene_bounds = true;
        let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(
            &vertex_positions, &draw_items,
        );
        auto_eye = eye; auto_target = target; auto_up = up; auto_far = far;

        let data: Vec<u8> = bytemuck::cast_slice(&vdata).to_vec();
        (data, all_indices, draw_ranges)
    } else {
        let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);
        let data: Vec<u8> = bytemuck::cast_slice(&sphere_verts).to_vec();
        (data, sphere_indices, Vec::new())
    };

    let vertex_data: &[u8] = &vertex_data_owned;
    let index_data: &[u8] = bytemuck::cast_slice(&pbr_indices);
    let num_indices = pbr_indices.len() as u32;

    let mut vbo = create_buffer_with_data(device, ctx.allocator_mut(), vertex_data, vk::BufferUsageFlags::VERTEX_BUFFER, "bindless_vbo")?;
    let mut ibo = create_buffer_with_data(device, ctx.allocator_mut(), index_data, vk::BufferUsageFlags::INDEX_BUFFER, "bindless_ibo")?;

    // --- Uniforms ---
    let aspect = width as f32 / height as f32;
    let (model, view_mat, proj) = if has_scene_bounds {
        (glam::Mat4::IDENTITY, crate::camera::look_at(auto_eye, auto_target, auto_up),
         crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far))
    } else {
        (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
    };

    let mut mvp_data = [0u8; 192];
    mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
    mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view_mat.as_ref()));
    mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));
    let mut mvp_buffer = create_buffer_with_data(device, ctx.allocator_mut(), &mvp_data, vk::BufferUsageFlags::UNIFORM_BUFFER, "bindless_mvp")?;

    let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
    let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
    let mut light_data = [0u8; 32];
    light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
    light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
    light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
    light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());
    let mut light_buffer = create_buffer_with_data(device, ctx.allocator_mut(), &light_data, vk::BufferUsageFlags::UNIFORM_BUFFER, "bindless_light")?;

    // --- Multi-light SSBO (Phase E) ---
    // Build a SceneManager and populate lights from glTF scene data.
    let mut sm = scene_manager::SceneManager::new();
    if demo_lights {
        sm.override_lights(crate::setup_demo_lights());
    } else if let Some(ref gs) = gltf_scene {
        sm.populate_lights_from_gltf(&gs.lights);
    } else {
        // Ensure at least a default directional light
        sm.populate_lights_from_gltf(&[]);
    }

    // Check if shader uses multi-light SSBO ("lights" storage_buffer binding)
    let has_multi_light = merged.values().any(|bindings| {
        bindings.iter().any(|b| b.name == "lights" && b.binding_type == "storage_buffer")
    });

    let mut lights_ssbo: Option<scene_manager::GpuBuffer> = None;
    if has_multi_light {
        let light_floats = sm.pack_lights_buffer();
        let data: &[u8] = if light_floats.is_empty() {
            &[0u8; 64] // dummy light (one vec4x4 = 64 bytes)
        } else {
            bytemuck::cast_slice(&light_floats)
        };
        let buf = create_buffer_with_data(
            device, ctx.allocator_mut(), data,
            vk::BufferUsageFlags::STORAGE_BUFFER, "lights_ssbo",
        )?;
        info!("Multi-light SSBO: {} lights, {} bytes", sm.lights.len(), data.len());
        lights_ssbo = Some(buf);
    }

    // Build SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
    let mut scene_light_ubo: Option<scene_manager::GpuBuffer> = None;
    if has_multi_light {
        let mut scene_light_data = [0u8; 16];
        scene_light_data[0..12].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        let light_count = sm.lights.len() as i32;
        scene_light_data[12..16].copy_from_slice(&light_count.to_le_bytes());
        let buf = create_buffer_with_data(
            device, ctx.allocator_mut(), &scene_light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "scene_light_ubo",
        )?;
        scene_light_ubo = Some(buf);
    }

    // --- Textures and materials ---
    let sampler_create = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);
    let sampler = unsafe { device.create_sampler(&sampler_create, None).map_err(|e| format!("Failed: {:?}", e))? };

    let mut default_texture = create_default_white_texture(ctx)?;
    let mut default_black_texture = create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "bindless_default_black")?;
    let mut default_normal_texture = create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "bindless_default_normal")?;

    // Build per-material texture maps (GpuImage)
    let mut per_material_textures: Vec<HashMap<String, scene_manager::GpuImage>> = Vec::new();
    if let Some(ref gs) = gltf_scene {
        for (mat_idx, mat) in gs.materials.iter().enumerate() {
            let mut tex_map: HashMap<String, scene_manager::GpuImage> = HashMap::new();
            let tex_slots: Vec<(&str, Option<&crate::gltf_loader::TextureImage>)> = vec![
                ("base_color_tex", mat.base_color_image.as_ref()),
                ("normal_tex", mat.normal_image.as_ref()),
                ("metallic_roughness_tex", mat.metallic_roughness_image.as_ref()),
                ("occlusion_tex", mat.occlusion_image.as_ref()),
                ("emissive_tex", mat.emissive_image.as_ref()),
                ("clearcoat_tex", mat.clearcoat_image.as_ref()),
                ("clearcoat_roughness_tex", mat.clearcoat_roughness_image.as_ref()),
                ("sheen_color_tex", mat.sheen_color_image.as_ref()),
                ("transmission_tex", mat.transmission_image.as_ref()),
            ];
            for (name, img_opt) in tex_slots {
                if let Some(tex_data) = img_opt {
                    let label = format!("bindless_{}[{}]", name, mat_idx);
                    let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                    tex_map.insert(name.to_string(), gpu_img);
                }
            }
            per_material_textures.push(tex_map);
        }
    }

    // Build bindless texture array
    let (tex_array, dedup) = scene_manager::build_bindless_texture_array_from_gpu_images(
        &per_material_textures, sampler, default_texture.view, default_black_texture.view, default_normal_texture.view,
    );

    // Build materials SSBO
    let mut materials_ssbo = if let Some(ref gs) = gltf_scene {
        scene_manager::build_materials_ssbo(
            device, ctx.allocator_mut(), gs, &dedup, &per_material_textures,
            default_texture.view, default_black_texture.view, default_normal_texture.view,
        )?
    } else {
        // Create a default single-material SSBO
        let default_scene = crate::gltf_loader::GltfScene {
            meshes: Vec::new(), materials: Vec::new(), nodes: Vec::new(),
            cameras: Vec::new(), lights: Vec::new(), root_nodes: Vec::new(),
        };
        scene_manager::build_materials_ssbo(
            device, ctx.allocator_mut(), &default_scene, &dedup, &per_material_textures,
            default_texture.view, default_black_texture.view, default_normal_texture.view,
        )?
    };

    // Load IBL assets
    let mut ibl_assets = load_ibl_assets(ctx, ibl_name);

    // --- Descriptor pool (with UPDATE_AFTER_BIND for bindless set) ---
    let (pool_sizes, total_sets) = reflected_pipeline::compute_pool_sizes_bindless(&merged);
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(total_sets)
        .pool_sizes(&pool_sizes)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(&pool_info, None).map_err(|e| format!("Failed: {:?}", e))?
    };

    // Allocate descriptor sets
    let max_set_idx = ds_layouts.keys().copied().max().unwrap_or(0);
    let mut ordered_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
    for i in 0..=max_set_idx {
        if let Some(&layout) = ds_layouts.get(&i) {
            ordered_layouts.push(layout);
        }
    }

    // For the bindless set (set 2), use variable descriptor count allocation
    let bindless_info = frag_refl.bindless.as_ref().unwrap();
    let tex_info = bindless_info.texture_array.as_ref().unwrap();
    let actual_tex_count = tex_array.texture_count;
    let variable_counts: Vec<u32> = (0..=max_set_idx)
        .filter(|i| ds_layouts.contains_key(i))
        .map(|i| if i == tex_info.set { actual_tex_count } else { 0 })
        .collect();

    let mut variable_count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(&variable_counts);

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&ordered_layouts)
        .push_next(&mut variable_count_info);

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(&alloc_info).map_err(|e| format!("Failed to alloc bindless DS: {:?}", e))?
    };

    // Build ds_map
    let mut ds_map: HashMap<u32, vk::DescriptorSet> = HashMap::new();
    {
        let mut ds_idx = 0;
        for i in 0..=max_set_idx {
            if ds_layouts.contains_key(&i) {
                ds_map.insert(i, descriptor_sets[ds_idx]);
                ds_idx += 1;
            }
        }
    }

    // --- Write descriptors ---
    // Set 0: MVP UBO
    if let Some(&ds0) = ds_map.get(&0) {
        if let Some(bindings) = merged.get(&0) {
            let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
            for binding in bindings {
                let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                if vk_type == vk::DescriptorType::UNIFORM_BUFFER && binding.name == "MVP" {
                    let idx = buf_infos.len();
                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(mvp_buffer.buffer).offset(0).range(binding.size as u64));
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds0).dst_binding(binding.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                    );
                }
            }
            if !writes.is_empty() {
                unsafe { device.update_descriptor_sets(&writes, &[]); }
            }
        }
    }

    // Set 1: Light UBO, IBL samplers/images, materials SSBO
    if let Some(&ds1) = ds_map.get(&1) {
        if let Some(bindings) = merged.get(&1) {
            let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

            for binding in bindings {
                let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                match vk_type {
                    vk::DescriptorType::UNIFORM_BUFFER => {
                        if binding.name == "Light" {
                            let idx = buf_infos.len();
                            buf_infos.push(vk::DescriptorBufferInfo::default().buffer(light_buffer.buffer).offset(0).range(binding.size as u64));
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(ds1).dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                            );
                        } else if binding.name == "SceneLight" {
                            // Multi-light mode: bind the SceneLight UBO (camera_pos + light_count)
                            if let Some(ref sl_buf) = scene_light_ubo {
                                let idx = buf_infos.len();
                                buf_infos.push(vk::DescriptorBufferInfo::default().buffer(sl_buf.buffer).offset(0).range(binding.size as u64));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds1).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                );
                            }
                        }
                    }
                    vk::DescriptorType::SAMPLER => {
                        let actual_sampler = ibl_assets.sampler_for_binding(&binding.name).unwrap_or(sampler);
                        let idx = img_infos.len();
                        img_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds1).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                        );
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        let view = ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view);
                        let idx = img_infos.len();
                        img_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds1).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                        );
                    }
                    vk::DescriptorType::STORAGE_BUFFER => {
                        if binding.name == "materials" {
                            let idx = buf_infos.len();
                            let ssbo_size = (materials_ssbo.material_count as u64) * std::mem::size_of::<scene_manager::BindlessMaterialData>() as u64;
                            buf_infos.push(vk::DescriptorBufferInfo::default().buffer(materials_ssbo.buffer.buffer).offset(0).range(ssbo_size));
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(ds1).dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                            );
                        } else if binding.name == "lights" {
                            if let Some(ref lights_buf) = lights_ssbo {
                                let idx = buf_infos.len();
                                let ssbo_size = (sm.lights.len() as u64) * 64; // 16 floats * 4 bytes per light
                                let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds1).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !writes.is_empty() {
                unsafe { device.update_descriptor_sets(&writes, &[]); }
            }
        }
    }

    // Set 2: Bindless texture array (COMBINED_IMAGE_SAMPLER array)
    if let Some(&ds2) = ds_map.get(&tex_info.set) {
        if actual_tex_count > 0 {
            let img_infos: Vec<vk::DescriptorImageInfo> = (0..actual_tex_count as usize)
                .map(|i| {
                    vk::DescriptorImageInfo::default()
                        .sampler(tex_array.samplers[i])
                        .image_view(tex_array.image_views[i])
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();

            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds2)
                .dst_binding(tex_info.binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&img_infos);

            unsafe { device.update_descriptor_sets(&[write], &[]); }
        }
    }

    // --- Render pass, framebuffer, pipeline ---
    let mut color_image = create_offscreen_image(device, ctx.allocator_mut(), width, height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR, "bindless_color")?;

    let mut depth_image = create_offscreen_image(device, ctx.allocator_mut(), width, height,
        vk::Format::D32_SFLOAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH, "bindless_depth")?;

    let attachments = [
        vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM).samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE).stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED).final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT).samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE).stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED).final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];
    let color_ref = vk::AttachmentReference::default().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default().attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);
    let dependencies = [
        vk::SubpassDependency::default().src_subpass(vk::SUBPASS_EXTERNAL).dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        vk::SubpassDependency::default().src_subpass(0).dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];

    let render_pass = unsafe {
        device.create_render_pass(&vk::RenderPassCreateInfo::default().attachments(&attachments).subpasses(std::slice::from_ref(&subpass)).dependencies(&dependencies), None)
            .map_err(|e| format!("Failed to create bindless render pass: {:?}", e))?
    };

    let fb_attachments = [color_image.view, depth_image.view];
    let framebuffer = unsafe {
        device.create_framebuffer(&vk::FramebufferCreateInfo::default().render_pass(render_pass).attachments(&fb_attachments).width(width).height(height).layers(1), None)
            .map_err(|e| format!("Failed to create bindless framebuffer: {:?}", e))?
    };

    // Graphics pipeline
    let entry_name = c"main";
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(entry_name),
        vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(entry_name),
    ];
    let (binding_desc, attr_descs) = reflected_pipeline::create_reflected_vertex_input(&vert_refl);
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&binding_desc).vertex_attribute_descriptions(&attr_descs);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::Viewport::default().width(width as f32).height(height as f32).max_depth(1.0);
    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
    let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::LESS);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
    let color_blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[
            vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages).vertex_input_state(&vertex_input).input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state).rasterization_state(&rasterizer).multisample_state(&multisampling)
                .depth_stencil_state(&depth_stencil).color_blend_state(&color_blending)
                .layout(pipeline_layout).render_pass(render_pass).subpass(0)
        ], None).map_err(|e| format!("Failed to create bindless pipeline: {:?}", e))?[0]
    };

    // --- Record commands ---
    let cmd = ctx.begin_single_commands()?;
    let clear_values = [
        vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.05, 0.08, 1.0] } },
        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
    ];
    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass).framebuffer(framebuffer)
        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width, height } })
        .clear_values(&clear_values);

    ctx.cmd_begin_label(cmd, "Raster Pass", [0.2, 0.8, 0.2, 1.0]);
    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

        // Bind all descriptor sets once
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, 0, &descriptor_sets, &[]);
        device.cmd_bind_vertex_buffers(cmd, 0, &[vbo.buffer], &[0]);
        device.cmd_bind_index_buffer(cmd, ibo.buffer, 0, vk::IndexType::UINT32);

        // Push constant: model (mat4, offset 0) + material_index (uint, offset 64) = 80 bytes
        // The push constant size is from frag reflection
        let pc_size = frag_refl.push_constants.first().map(|pc| pc.size as usize).unwrap_or(80);
        let mut push_data = vec![0u8; pc_size];
        // Write model matrix (identity since transforms are baked into vertices)
        let model_bytes = bytemuck::cast_slice::<f32, u8>(glam::Mat4::IDENTITY.as_ref());
        if push_data.len() >= 64 {
            push_data[0..64].copy_from_slice(model_bytes);
        }

        let push_stage = if !frag_refl.push_constants.is_empty() {
            reflected_pipeline::stage_flags_from_strings_public(&frag_refl.push_constants[0].stage_flags)
        } else {
            vk::ShaderStageFlags::FRAGMENT
        };

        if !draw_ranges.is_empty() {
            for range in &draw_ranges {
                // Write material_index at offset 64
                if push_data.len() >= 68 {
                    push_data[64..68].copy_from_slice(&(range.material_index as u32).to_le_bytes());
                }
                device.cmd_push_constants(cmd, pipeline_layout, push_stage, 0, &push_data);
                device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
            }
        } else {
            // Single draw, material_index = 0
            device.cmd_push_constants(cmd, pipeline_layout, push_stage, 0, &push_data);
            device.cmd_draw_indexed(cmd, num_indices, 1, 0, 0, 0);
        }

        device.cmd_end_render_pass(cmd);
    }
    ctx.cmd_end_label(cmd);

    let mut staging = screenshot::StagingBuffer::new(device, ctx.allocator_mut(), width, height)?;
    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);
    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;
    info!("Saved bindless raster render to {:?}", output_path);

    // --- Cleanup ---
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        for (_, layout) in &ds_layouts { device.destroy_descriptor_set_layout(*layout, None); }
        device.destroy_sampler(sampler, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    depth_image.destroy(device, ctx.allocator_mut());
    default_texture.destroy(device, ctx.allocator_mut());
    default_black_texture.destroy(device, ctx.allocator_mut());
    default_normal_texture.destroy(device, ctx.allocator_mut());
    materials_ssbo.buffer.destroy(device, ctx.allocator_mut());
    for tex_map in per_material_textures {
        for (_, mut tex) in tex_map { tex.destroy(device, ctx.allocator_mut()); }
    }
    ibl_assets.destroy(device, ctx.allocator_mut());
    mvp_buffer.destroy(device, ctx.allocator_mut());
    light_buffer.destroy(device, ctx.allocator_mut());
    if let Some(mut buf) = lights_ssbo {
        buf.destroy(device, ctx.allocator_mut());
    }
    if let Some(mut buf) = scene_light_ubo {
        buf.destroy(device, ctx.allocator_mut());
    }
    vbo.destroy(device, ctx.allocator_mut());
    ibo.destroy(device, ctx.allocator_mut());

    Ok(())
}

fn render_pbr_scene(
    ctx: &mut VulkanContext,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
    demo_lights: bool,
) -> Result<(), String> {
    let device_owned = ctx.device.clone();
    let device = &device_owned;

    // =====================================================================
    // Phase 0: Load reflection JSON and build descriptor layouts
    // =====================================================================
    let vert_json_path = format!("{}.vert.json", pipeline_base);
    let frag_json_path = format!("{}.frag.json", pipeline_base);

    let vert_refl = reflected_pipeline::load_reflection(Path::new(&vert_json_path))?;
    let frag_refl = reflected_pipeline::load_reflection(Path::new(&frag_json_path))?;

    info!(
        "Reflection loaded: vert={} frag={} descriptor_sets: vert={:?}, frag={:?}",
        vert_json_path,
        frag_json_path,
        vert_refl.descriptor_sets.keys().collect::<Vec<_>>(),
        frag_refl.descriptor_sets.keys().collect::<Vec<_>>(),
    );

    // --- Check for bindless mode ---
    let bindless_enabled = frag_refl.bindless.as_ref().map_or(false, |b| b.enabled)
        && ctx.bindless_supported;
    if bindless_enabled {
        info!("Bindless mode detected and supported -- using bindless raster path");
        return render_pbr_scene_bindless(
            ctx, vert_module, frag_module, pipeline_base, scene_source,
            width, height, output_path, ibl_name, vert_refl, frag_refl,
            demo_lights,
        );
    }

    // Merge descriptor sets from both stages
    let merged = reflected_pipeline::merge_descriptor_sets(&[&vert_refl, &frag_refl]);

    // Create VkDescriptorSetLayouts from reflection
    let ds_layouts = unsafe {
        reflected_pipeline::create_descriptor_set_layouts_from_merged(device, &merged)?
    };

    // Load glTF scene early so we know how many materials we need
    let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");
    let mut gltf_scene = if is_gltf {
        Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
    } else {
        None
    };

    // Identify vertex vs fragment set indices
    let mut frag_set_idx: Option<u32> = None;
    let mut vert_set_idx: Option<u32> = None;
    for (&set_idx, bindings) in &merged {
        for binding in bindings {
            if binding.name == "Material" { frag_set_idx = Some(set_idx); }
            if binding.name == "MVP" { vert_set_idx = Some(set_idx); }
        }
    }
    let frag_set_idx = frag_set_idx.unwrap_or(1);
    let vert_set_idx = vert_set_idx.unwrap_or(0);

    let num_materials = gltf_scene.as_ref().map(|s| s.materials.len().max(1)).unwrap_or(1);

    // Compute pool sizes with fragment bindings multiplied by num_materials
    let (mut pool_sizes, _) = reflected_pipeline::compute_pool_sizes(&merged);
    for ps in &mut pool_sizes {
        let frag_count: u32 = merged.get(&frag_set_idx).map(|bindings| {
            bindings.iter().filter(|b| {
                reflected_pipeline::binding_type_to_vk_public(&b.binding_type) == ps.ty
            }).count() as u32
        }).unwrap_or(0);
        if frag_count > 0 {
            ps.descriptor_count += frag_count * (num_materials as u32 - 1);
        }
    }
    let max_sets = 1 + num_materials as u32; // 1 vertex set + N fragment sets

    info!(
        "Reflection-driven descriptors: {} sets ({} materials), {} pool size entries",
        max_sets, num_materials, pool_sizes.len()
    );

    // Create descriptor pool
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(max_sets)
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
    };

    // Allocate vertex set (1 copy)
    let vert_ds = if let Some(&layout) = ds_layouts.get(&vert_set_idx) {
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&layout));
        unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?[0] }
    } else {
        vk::DescriptorSet::null()
    };

    // Allocate N fragment sets
    let mut per_material_desc_sets: Vec<vk::DescriptorSet> = Vec::new();
    let mut per_material_buffers: Vec<GpuBuffer> = Vec::new();

    if let Some(&frag_layout) = ds_layouts.get(&frag_set_idx) {
        let layouts_vec: Vec<vk::DescriptorSetLayout> = vec![frag_layout; num_materials];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts_vec);
        per_material_desc_sets = unsafe {
            device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?
        };
    }

    // Build push constant ranges from reflection data
    let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
    for pc in &vert_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }
    for pc in &frag_refl.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }

    // Build push constant data buffer from reflection fields
    let mut push_constant_data: Vec<u8> = Vec::new();
    let mut push_constant_stage_flags = vk::ShaderStageFlags::empty();
    {
        let all_pcs: Vec<&reflected_pipeline::PushConstantInfo> = vert_refl
            .push_constants
            .iter()
            .chain(frag_refl.push_constants.iter())
            .collect();
        if !all_pcs.is_empty() {
            // Use the max size across all push constant blocks
            let total_size = all_pcs.iter().map(|pc| pc.size as usize).max().unwrap_or(0);
            push_constant_data.resize(total_size, 0u8);
            for pc in &all_pcs {
                push_constant_stage_flags |=
                    reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
                for field in &pc.fields {
                    let offset = field.offset as usize;
                    match field.name.as_str() {
                        "light_dir" => {
                            let v = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
                            let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                            push_constant_data[offset..offset + bytes.len()]
                                .copy_from_slice(bytes);
                        }
                        "view_pos" => {
                            let v = glam::Vec3::new(0.0, 0.0, 3.0);
                            let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                            push_constant_data[offset..offset + bytes.len()]
                                .copy_from_slice(bytes);
                        }
                        _ => {
                            info!(
                                "Unknown push constant field '{}', leaving zeroed",
                                field.name
                            );
                        }
                    }
                }
            }
        }
    }

    // Pipeline layout
    let pipeline_layout = unsafe {
        reflected_pipeline::create_pipeline_layout_from_merged(device, &ds_layouts, &push_ranges)?
    };

    // =====================================================================
    // Phase 1: Load scene geometry
    // =====================================================================

    // Determine if we need 48-byte vertices (with tangent) based on reflection stride
    let pipeline_vertex_stride = vert_refl.vertex_stride;

    // Auto-camera data computed from scene bounds
    let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
    let mut auto_target = glam::Vec3::ZERO;
    let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
    let mut auto_far = 100.0f32;
    let mut has_scene_bounds = false;

    let (vertex_data_owned, pbr_indices, draw_ranges) = if let Some(ref mut gltf_s) = gltf_scene {
        if gltf_s.meshes.is_empty() {
            return Err(format!("No meshes found in glTF file: {}", scene_source));
        }
        let draw_items = crate::gltf_loader::flatten_scene(gltf_s);

        if draw_items.is_empty() {
            return Err(format!("No draw items in glTF scene: {}", scene_source));
        }

        // Pack ALL meshes with world transforms
        let mut vdata: Vec<f32> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut vertex_offset: u32 = 0;
        let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
        let mut draw_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();
        let mut index_offset: u32 = 0;

        for item in &draw_items {
            let gmesh = &gltf_s.meshes[item.mesh_index];
            let world = item.world_transform;
            let upper3x3 = glam::Mat3::from_mat4(world);
            let normal_mat = upper3x3.inverse().transpose();

            for gv in &gmesh.vertices {
                let pos = world.transform_point3(glam::Vec3::from(gv.position));
                let norm = (normal_mat * glam::Vec3::from(gv.normal)).normalize();
                vertex_positions.push([pos.x, pos.y, pos.z]);

                vdata.push(pos.x); vdata.push(pos.y); vdata.push(pos.z);
                vdata.push(norm.x); vdata.push(norm.y); vdata.push(norm.z);
                vdata.push(gv.uv[0]); vdata.push(gv.uv[1]);

                // Only pack tangent data when the shader expects 48-byte vertices
                if pipeline_vertex_stride >= 48 {
                    if gmesh.has_tangents {
                        let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                        vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                        vdata.push(gv.tangent[3]);
                    } else {
                        vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                    }
                }
            }

            for &idx in &gmesh.indices {
                all_indices.push(idx + vertex_offset);
            }
            vertex_offset += gmesh.vertices.len() as u32;

            draw_ranges.push(crate::gltf_loader::DrawRange {
                index_offset,
                index_count: gmesh.indices.len() as u32,
                material_index: item.material_index,
            });
            index_offset += gmesh.indices.len() as u32;
        }

        // Compute auto-camera using shared function (matches all engines)
        has_scene_bounds = true;
        {
            let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(
                &vertex_positions,
                &draw_items,
            );
            auto_eye = eye;
            auto_target = target;
            auto_up = up;
            auto_far = far;
        }

        let data: Vec<u8> = bytemuck::cast_slice(&vdata).to_vec();
        (data, all_indices, draw_ranges)
    } else {
        let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);

        if pipeline_vertex_stride == 48 {
            // Pad sphere PbrVertex (32 bytes) to 48 bytes with default tangent
            let default_tangent: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
            let mut padded_data: Vec<u8> = Vec::with_capacity(sphere_verts.len() * 48);
            for v in &sphere_verts {
                padded_data.extend_from_slice(bytemuck::cast_slice(&[*v]));
                padded_data.extend_from_slice(bytemuck::cast_slice(&default_tangent));
            }
            (padded_data, sphere_indices, Vec::new())
        } else {
            let data: Vec<u8> = bytemuck::cast_slice(&sphere_verts).to_vec();
            (data, sphere_indices, Vec::new())
        }
    };

    let vertex_data: &[u8] = &vertex_data_owned;
    let index_data: &[u8] = bytemuck::cast_slice(&pbr_indices);
    let num_indices = pbr_indices.len() as u32;

    let mut vbo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        vertex_data,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        "pbr_vbo",
    )?;

    let mut ibo = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        index_data,
        vk::BufferUsageFlags::INDEX_BUFFER,
        "pbr_ibo",
    )?;

    // =====================================================================
    // Phase 2: Create uniform buffers and textures based on reflection
    // =====================================================================

    // MVP uniform buffer (192 bytes)
    let aspect = width as f32 / height as f32;
    let (model, view, proj): (glam::Mat4, glam::Mat4, glam::Mat4) = if has_scene_bounds {
        let m = glam::Mat4::IDENTITY; // transforms baked into vertices
        let v = crate::camera::look_at(auto_eye, auto_target, auto_up);
        let p = crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far);
        (m, v, p)
    } else {
        (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
    };

    let mut mvp_data = [0u8; 192];
    mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
    mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
    mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

    let mut mvp_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        &mvp_data,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_mvp",
    )?;

    // Light uniform buffer (32 bytes)
    let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
    let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
    let mut light_data = [0u8; 32];
    light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
    light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
    light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
    light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

    let mut light_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        &light_data,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_light",
    )?;

    // --- Multi-light SSBO (Phase E) ---
    let mut sm = scene_manager::SceneManager::new();
    if demo_lights {
        sm.override_lights(crate::setup_demo_lights());
    } else if let Some(ref gs) = gltf_scene {
        sm.populate_lights_from_gltf(&gs.lights);
    } else {
        sm.populate_lights_from_gltf(&[]);
    }

    let has_multi_light = merged.values().any(|bindings| {
        bindings.iter().any(|b| b.name == "lights" && b.binding_type == "storage_buffer")
    });

    let mut lights_ssbo: Option<scene_manager::GpuBuffer> = None;
    if has_multi_light {
        let light_floats = sm.pack_lights_buffer();
        let data: &[u8] = if light_floats.is_empty() {
            &[0u8; 64]
        } else {
            bytemuck::cast_slice(&light_floats)
        };
        let buf = create_buffer_with_data(
            device, ctx.allocator_mut(), data,
            vk::BufferUsageFlags::STORAGE_BUFFER, "lights_ssbo",
        )?;
        lights_ssbo = Some(buf);
    }

    // SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
    let mut scene_light_ubo: Option<scene_manager::GpuBuffer> = None;
    if has_multi_light {
        let mut scene_light_data = [0u8; 16];
        scene_light_data[0..12].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        let light_count = sm.lights.len() as i32;
        scene_light_data[12..16].copy_from_slice(&light_count.to_le_bytes());
        let buf = create_buffer_with_data(
            device, ctx.allocator_mut(), &scene_light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "scene_light_ubo",
        )?;
        scene_light_ubo = Some(buf);
    }

    // Material uniform buffer (80 bytes)
    let material_data = if let Some(ref gs) = gltf_scene {
        if !gs.materials.is_empty() {
            MaterialUboData::from_gltf_material(&gs.materials[0])
        } else {
            MaterialUboData::default_values()
        }
    } else {
        MaterialUboData::default_values()
    };

    let mut material_buffer = create_buffer_with_data(
        device,
        ctx.allocator_mut(),
        bytemuck::bytes_of(&material_data),
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        "pbr_material",
    )?;

    // Sampler (shared by all texture bindings)
    let sampler_create = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);

    let sampler = unsafe {
        device
            .create_sampler(&sampler_create, None)
            .map_err(|e| format!("Failed to create sampler: {:?}", e))?
    };

    // Create default textures for missing texture slots
    let mut default_texture = create_default_white_texture(ctx)?;
    // Black texture for emissive (no emission)
    let black_pixel = vec![0u8, 0, 0, 255];
    let mut default_black_texture = create_texture_image(ctx, 1, 1, &black_pixel, "default_black")?;
    // Flat normal texture (128,128,255) = tangent-space (0,0,1)
    let normal_pixel = vec![128u8, 128, 255, 255];
    let mut default_normal_texture = create_texture_image(ctx, 1, 1, &normal_pixel, "default_normal")?;

    // Build per-material texture maps
    let mut per_material_textures: Vec<HashMap<String, GpuImage>> = Vec::new();

    if let Some(ref gs) = gltf_scene {
        for (mat_idx, mat) in gs.materials.iter().enumerate() {
            let mut tex_map: HashMap<String, GpuImage> = HashMap::new();

            let tex_slots: Vec<(&str, Option<&crate::gltf_loader::TextureImage>)> = vec![
                ("base_color_tex", mat.base_color_image.as_ref()),
                ("normal_tex", mat.normal_image.as_ref()),
                ("metallic_roughness_tex", mat.metallic_roughness_image.as_ref()),
                ("occlusion_tex", mat.occlusion_image.as_ref()),
                ("emissive_tex", mat.emissive_image.as_ref()),
                ("clearcoat_tex", mat.clearcoat_image.as_ref()),
                ("clearcoat_roughness_tex", mat.clearcoat_roughness_image.as_ref()),
                ("sheen_color_tex", mat.sheen_color_image.as_ref()),
                ("transmission_tex", mat.transmission_image.as_ref()),
            ];

            for (name, img_opt) in tex_slots {
                if let Some(tex_data) = img_opt {
                    let label = format!("{}[{}]", name, mat_idx);
                    info!("Uploading texture '{}': {}x{}", label, tex_data.width, tex_data.height);
                    let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                    tex_map.insert(name.to_string(), gpu_img);
                }
            }

            // Also map albedo_tex to base_color_image for backward compat
            if mat.base_color_image.is_some() && !tex_map.contains_key("albedo_tex") {
                if let Some(tex_data) = mat.base_color_image.as_ref() {
                    let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &format!("albedo_tex[{}]", mat_idx))?;
                    tex_map.insert("albedo_tex".to_string(), gpu_img);
                }
            }

            per_material_textures.push(tex_map);
        }
    }

    // Backward compat: also build texture_images for non-glTF or fallback
    let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
    // Keep texture_images for non-glTF or fallback
    if gltf_scene.is_none() {
        info!("Generating procedural texture for non-glTF scene...");
        let tex_pixels = scene::generate_procedural_texture(512);
        let proc_texture = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
        texture_images.insert("albedo_tex".to_string(), proc_texture);
        let proc_texture2 = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
        texture_images.insert("base_color_tex".to_string(), proc_texture2);
    }

    // Load IBL assets (specular cubemap, irradiance cubemap, BRDF LUT)
    let mut ibl_assets = load_ibl_assets(ctx, ibl_name);

    // =====================================================================
    // Phase 3: Write descriptor sets from reflection data
    // =====================================================================

    // Write vertex set (set 0): MVP + Light (+ SceneLight UBO + lights SSBO if present)
    {
        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

        if let Some(bindings) = merged.get(&vert_set_idx) {
            for binding in bindings {
                let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                if vk_type == vk::DescriptorType::UNIFORM_BUFFER {
                    let (buffer, range) = match binding.name.as_str() {
                        "MVP" => (mvp_buffer.buffer, binding.size as u64),
                        "Light" => (light_buffer.buffer, binding.size as u64),
                        "SceneLight" => {
                            if let Some(ref sl_buf) = scene_light_ubo {
                                (sl_buf.buffer, binding.size as u64)
                            } else {
                                continue;
                            }
                        }
                        _ => continue,
                    };
                    let idx = buf_infos.len();
                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(range));
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(vert_ds)
                            .dst_binding(binding.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                    );
                } else if vk_type == vk::DescriptorType::STORAGE_BUFFER && binding.name == "lights" {
                    if let Some(ref lights_buf) = lights_ssbo {
                        let idx = buf_infos.len();
                        let ssbo_size = (sm.lights.len() as u64) * 64;
                        let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                        buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(vert_ds)
                                .dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                        );
                    }
                }
            }
        }
        if !writes.is_empty() {
            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
    }

    // Create per-material UBO buffers and write per-material fragment sets
    for mi in 0..num_materials {
        let mat_data = if let Some(ref gs) = gltf_scene {
            if mi < gs.materials.len() {
                MaterialUboData::from_gltf_material(&gs.materials[mi])
            } else {
                MaterialUboData::default_values()
            }
        } else {
            MaterialUboData::default_values()
        };

        let mat_buf = create_buffer_with_data(
            device, ctx.allocator_mut(),
            bytemuck::bytes_of(&mat_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &format!("pbr_material_{}", mi),
        )?;
        per_material_buffers.push(mat_buf);
    }

    // Write each per-material descriptor set
    for mi in 0..num_materials {
        let ds = per_material_desc_sets[mi];
        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

        if let Some(bindings) = merged.get(&frag_set_idx) {
            for binding in bindings {
                let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                match vk_type {
                    vk::DescriptorType::UNIFORM_BUFFER => {
                        let (buffer, range) = if binding.name == "Material" {
                            (per_material_buffers[mi].buffer, std::mem::size_of::<MaterialUboData>() as u64)
                        } else if binding.name == "Light" {
                            (light_buffer.buffer, binding.size as u64)
                        } else if binding.name == "SceneLight" {
                            if let Some(ref sl_buf) = scene_light_ubo {
                                (sl_buf.buffer, binding.size as u64)
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        };
                        let idx = buf_infos.len();
                        buf_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer)
                                .offset(0)
                                .range(range),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                        );
                    }
                    vk::DescriptorType::STORAGE_BUFFER => {
                        if binding.name == "lights" {
                            if let Some(ref lights_buf) = lights_ssbo {
                                let idx = buf_infos.len();
                                let ssbo_size = (sm.lights.len() as u64) * 64;
                                let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                );
                            }
                        }
                    }
                    vk::DescriptorType::SAMPLER => {
                        let actual_sampler = ibl_assets.sampler_for_binding(&binding.name).unwrap_or(sampler);
                        let idx = img_infos.len();
                        img_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                        );
                    }
                    vk::DescriptorType::SAMPLED_IMAGE => {
                        let is_cube = reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                        let view = if is_cube || binding.name == "brdf_lut" {
                            ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                        } else {
                            // Per-material texture lookup
                            let per_mat_view = if mi < per_material_textures.len() {
                                per_material_textures[mi].get(&binding.name).map(|img| img.view)
                            } else {
                                None
                            };
                            per_mat_view.unwrap_or_else(|| {
                                let fallback = match binding.name.as_str() {
                                    "emissive_tex" => default_black_texture.view,
                                    "normal_tex" => default_normal_texture.view,
                                    _ => default_texture.view,
                                };
                                texture_images.get(&binding.name).map(|img| img.view).unwrap_or(fallback)
                            })
                        };
                        let idx = img_infos.len();
                        img_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds).dst_binding(binding.binding)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                        );
                    }
                    _ => {}
                }
            }
        }
        if !writes.is_empty() {
            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
    }

    // =====================================================================
    // Phase 4: Create render pass, framebuffer, graphics pipeline
    // =====================================================================

    let mut color_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        "pbr_color",
    )?;

    let mut depth_image = create_offscreen_image(
        device,
        ctx.allocator_mut(),
        width,
        height,
        vk::Format::D32_SFLOAT,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        "pbr_depth",
    )?;

    let attachments = [
        vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref))
        .depth_stencil_attachment(&depth_ref);

    let dependencies = [
        // EXTERNAL -> 0: ensure prior commands complete before rendering
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
        // 0 -> EXTERNAL: ensure color writes are visible to subsequent blit/transfer
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];

    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&dependencies);

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .map_err(|e| format!("Failed to create PBR render pass: {:?}", e))?
    };

    let fb_attachments = [color_image.view, depth_image.view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&fb_attachments)
        .width(width)
        .height(height)
        .layers(1);

    let framebuffer = unsafe {
        device
            .create_framebuffer(&framebuffer_info, None)
            .map_err(|e| format!("Failed to create PBR framebuffer: {:?}", e))?
    };

    // Graphics pipeline
    let entry_name = c"main";

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(entry_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(entry_name),
    ];

    // Use reflection-driven vertex input (stride and attributes from vert_refl)
    let (binding_desc, attr_descs) =
        reflected_pipeline::create_reflected_vertex_input(&vert_refl);

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .width(width as f32)
        .height(height as f32)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&color_blend_attachment));

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|e| format!("Failed to create PBR pipeline: {:?}", e))?[0]
    };

    // =====================================================================
    // Phase 5: Record commands and render
    // =====================================================================

    let cmd = ctx.begin_single_commands()?;

    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.05, 0.05, 0.08, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];

    let render_pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D { width, height },
        })
        .clear_values(&clear_values);

    ctx.cmd_begin_label(cmd, "Raster Pass", [0.2, 0.8, 0.2, 1.0]);
    unsafe {
        device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

        if !push_constant_data.is_empty() {
            device.cmd_push_constants(
                cmd,
                pipeline_layout,
                push_constant_stage_flags,
                0,
                &push_constant_data,
            );
        }

        // Bind vertex set (shared)
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, vert_set_idx, std::slice::from_ref(&vert_ds), &[]);

        device.cmd_bind_vertex_buffers(cmd, 0, &[vbo.buffer], &[0]);
        device.cmd_bind_index_buffer(cmd, ibo.buffer, 0, vk::IndexType::UINT32);

        if !draw_ranges.is_empty() && !per_material_desc_sets.is_empty() {
            let mut current_mat: i32 = -1;
            for range in &draw_ranges {
                if range.material_index as i32 != current_mat {
                    current_mat = range.material_index as i32;
                    let mat_idx = (current_mat as usize).min(per_material_desc_sets.len() - 1);
                    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, frag_set_idx, std::slice::from_ref(&per_material_desc_sets[mat_idx]), &[]);
                }
                device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
            }
        } else {
            // Fallback: single draw
            if !per_material_desc_sets.is_empty() {
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_layout, frag_set_idx, std::slice::from_ref(&per_material_desc_sets[0]), &[]);
            }
            device.cmd_draw_indexed(cmd, num_indices, 1, 0, 0, 0);
        }

        device.cmd_end_render_pass(cmd);
    }
    ctx.cmd_end_label(cmd);

    let mut staging = screenshot::StagingBuffer::new(
        device,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(device, cmd, color_image.image, staging.buffer, width, height);

    ctx.end_single_commands(cmd)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved raster scene render to {:?}", output_path);

    // =====================================================================
    // Cleanup
    // =====================================================================
    staging.destroy(device, ctx.allocator_mut());
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        for (_, layout) in &ds_layouts {
            device.destroy_descriptor_set_layout(*layout, None);
        }
        device.destroy_sampler(sampler, None);
    }
    color_image.destroy(device, ctx.allocator_mut());
    depth_image.destroy(device, ctx.allocator_mut());
    default_texture.destroy(device, ctx.allocator_mut());
    default_black_texture.destroy(device, ctx.allocator_mut());
    default_normal_texture.destroy(device, ctx.allocator_mut());
    for (_, mut tex) in texture_images {
        tex.destroy(device, ctx.allocator_mut());
    }
    ibl_assets.destroy(device, ctx.allocator_mut());
    mvp_buffer.destroy(device, ctx.allocator_mut());
    light_buffer.destroy(device, ctx.allocator_mut());
    if let Some(mut buf) = lights_ssbo {
        buf.destroy(device, ctx.allocator_mut());
    }
    if let Some(mut buf) = scene_light_ubo {
        buf.destroy(device, ctx.allocator_mut());
    }
    material_buffer.destroy(device, ctx.allocator_mut());
    for mut buf in per_material_buffers {
        buf.destroy(device, ctx.allocator_mut());
    }
    for (_, tex_map) in per_material_textures.into_iter().enumerate() {
        for (_, mut tex) in tex_map {
            tex.destroy(device, ctx.allocator_mut());
        }
    }
    vbo.destroy(device, ctx.allocator_mut());
    ibo.destroy(device, ctx.allocator_mut());

    Ok(())
}

// =========================================================================
// Shadow map infrastructure
// =========================================================================

/// Holds all shadow map GPU resources. Created when the shader
/// requires "shadow_maps" and "shadow_matrices" bindings.
struct ShadowResources {
    image: vk::Image,
    image_alloc: gpu_allocator::vulkan::Allocation,
    array_view: vk::ImageView,
    per_layer_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    render_pass: vk::RenderPass,
    comparison_sampler: vk::Sampler,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vert_module: vk::ShaderModule,
}

/// Create all shadow map GPU resources: depth image array, views, framebuffers,
/// render pass, comparison sampler, and shadow depth pipeline.
fn create_shadow_resources(
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    layer_count: u32,
    gltf_vertex_stride: u32,
) -> Result<ShadowResources, String> {
    let res = SHADOW_MAP_RESOLUTION;

    // 1. Create 2D array depth image (D32_SFLOAT, res x res, up to 8 layers)
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::D32_SFLOAT)
        .extent(vk::Extent3D { width: res, height: res, depth: 1 })
        .mip_levels(1)
        .array_layers(layer_count)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create shadow depth image: {:?}", e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let image_alloc = allocator
        .allocate(&AllocationCreateDesc {
            name: "shadow_depth_array",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate shadow depth image memory: {:?}", e))?;

    unsafe {
        device
            .bind_image_memory(image, image_alloc.memory(), image_alloc.offset())
            .map_err(|e| format!("Failed to bind shadow depth image memory: {:?}", e))?;
    }

    // 2. Create array image view (for sampler2DArray binding in main pass)
    let array_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(vk::Format::D32_SFLOAT)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(layer_count),
        );

    let array_view = unsafe {
        device
            .create_image_view(&array_view_info, None)
            .map_err(|e| format!("Failed to create shadow array image view: {:?}", e))?
    };

    // 3. Create per-layer views and framebuffers
    // First create the depth-only render pass
    let depth_attachment = vk::AttachmentDescription::default()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref);

    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ),
    ];

    let render_pass = unsafe {
        device
            .create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&depth_attachment))
                    .subpasses(std::slice::from_ref(&subpass))
                    .dependencies(&dependencies),
                None,
            )
            .map_err(|e| format!("Failed to create shadow render pass: {:?}", e))?
    };

    let mut per_layer_views = Vec::with_capacity(layer_count as usize);
    let mut framebuffers = Vec::with_capacity(layer_count as usize);

    for layer in 0..layer_count {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .components(vk::ComponentMapping::default())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(layer)
                    .layer_count(1),
            );

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .map_err(|e| format!("Failed to create shadow layer {} view: {:?}", layer, e))?
        };

        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(std::slice::from_ref(&view))
            .width(res)
            .height(res)
            .layers(1);

        let fb = unsafe {
            device
                .create_framebuffer(&fb_info, None)
                .map_err(|e| format!("Failed to create shadow layer {} framebuffer: {:?}", layer, e))?
        };

        per_layer_views.push(view);
        framebuffers.push(fb);
    }

    // 5. Create comparison sampler (LESS_OR_EQUAL)
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .max_lod(1.0);

    let comparison_sampler = unsafe {
        device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create shadow comparison sampler: {:?}", e))?
    };

    // 6. Create shadow pipeline with push constants (mat4 MVP = 64 bytes)
    // Create shader module from embedded SPIR-V
    let spv_u32: Vec<u32> = SHADOW_VERT_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let vert_module = unsafe {
        device
            .create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&spv_u32),
                None,
            )
            .map_err(|e| format!("Failed to create shadow vertex shader module: {:?}", e))?
    };

    // Pipeline layout: push constant mat4 (64 bytes) for vertex stage
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(64);

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
            .map_err(|e| format!("Failed to create shadow pipeline layout: {:?}", e))?
    };

    // Vertex input: only position (vec3, 12 bytes at offset 0), but stride matches the glTF VBO
    let entry_name = c"main";
    let shader_stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(entry_name)];

    // The shadow shader reads only location 0 (vec3 position) but the VBO
    // has a stride of gltf_vertex_stride (48 for glTF with tangents, or 32).
    let stride = if gltf_vertex_stride > 0 { gltf_vertex_stride } else { 32 };
    let binding_desc = [vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(stride)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attr_descs = [vk::VertexInputAttributeDescription::default()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_desc)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .width(res as f32)
        .height(res as f32)
        .max_depth(1.0);
    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: res, height: res });
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(std::slice::from_ref(&viewport))
        .scissors(std::slice::from_ref(&scissor));

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE) // No culling for shadow maps (avoids peter-panning)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(1.25)
        .depth_bias_slope_factor(1.75);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    // No color attachments for depth-only pass
    let color_blending = vk::PipelineColorBlendStateCreateInfo::default();

    let pipeline = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages)
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_assembly)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterizer)
                    .multisample_state(&multisampling)
                    .depth_stencil_state(&depth_stencil)
                    .color_blend_state(&color_blending)
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)],
                None,
            )
            .map_err(|e| format!("Failed to create shadow pipeline: {:?}", e))?[0]
    };

    info!(
        "Shadow resources created: {} layers, {}x{}, D32_SFLOAT",
        layer_count, res, res
    );

    Ok(ShadowResources {
        image,
        image_alloc,
        array_view,
        per_layer_views,
        framebuffers,
        render_pass,
        comparison_sampler,
        pipeline,
        pipeline_layout,
        vert_module,
    })
}

// =========================================================================
// PersistentRenderer — holds GPU resources for interactive multi-frame rendering
// =========================================================================

/// Persistent PBR renderer for interactive mode.
///
/// Holds all GPU resources (pipeline, descriptors, textures, mesh buffers)
/// across frames. Supports camera updates and re-rendering.
pub struct PersistentRenderer {
    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    ds_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Shader modules
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,

    // Render target (offscreen)
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    color_image: GpuImage,
    depth_image: GpuImage,
    width: u32,
    height: u32,

    // Geometry
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    num_indices: u32,

    // Multi-material draw
    draw_ranges: Vec<crate::gltf_loader::DrawRange>,
    per_material_desc_sets: Vec<vk::DescriptorSet>,
    per_material_buffers: Vec<GpuBuffer>,
    per_material_textures: Vec<HashMap<String, GpuImage>>,
    vert_set_idx: u32,
    frag_set_idx: u32,
    vert_ds: vk::DescriptorSet,

    // Uniforms
    mvp_buffer: GpuBuffer,
    light_buffer: GpuBuffer,
    material_buffer: GpuBuffer,

    // Multi-light (Phase E)
    has_multi_light: bool,
    lights_ssbo: Option<GpuBuffer>,
    scene_light_ubo: Option<GpuBuffer>,

    // Shadow maps (Phase F)
    shadow_enabled: bool,
    shadow_image: vk::Image,
    shadow_image_alloc: Option<gpu_allocator::vulkan::Allocation>,
    shadow_array_view: vk::ImageView,
    shadow_per_layer_views: Vec<vk::ImageView>,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    shadow_render_pass: vk::RenderPass,
    shadow_sampler: vk::Sampler,
    shadow_pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_vert_module: vk::ShaderModule,
    shadow_matrices_ssbo: Option<GpuBuffer>,
    shadow_count: u32,
    scene_manager: scene_manager::SceneManager,

    // Push constants
    push_constant_data: Vec<u8>,
    push_constant_stage_flags: vk::ShaderStageFlags,

    // Textures
    sampler: vk::Sampler,
    default_texture: GpuImage,
    default_black_texture: GpuImage,
    default_normal_texture: GpuImage,
    texture_images: HashMap<String, GpuImage>,
    ibl_assets: IblAssets,

    // Multi-pipeline mode
    multi_pipeline: bool,
    permutations: Vec<PermutationPipeline>,
    material_to_permutation: Vec<usize>,
    shared_set0_layout: vk::DescriptorSetLayout,

    // Auto-camera
    pub auto_eye: glam::Vec3,
    pub auto_target: glam::Vec3,
    pub auto_up: glam::Vec3,
    pub auto_far: f32,
    pub has_scene_bounds: bool,
}

impl PersistentRenderer {
    /// Initialize the persistent renderer with all GPU resources.
    ///
    /// Supports both single-pipeline mode (original) and multi-pipeline mode
    /// (when a .manifest.json exists AND the scene has multiple materials).
    pub fn init(
        ctx: &mut VulkanContext,
        pipeline_base: &str,
        scene_source: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
        demo_lights: bool,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();

        // Load glTF scene early so we can decide single vs multi pipeline
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");
        let mut gltf_scene = if is_gltf {
            Some(crate::gltf_loader::load_gltf(std::path::Path::new(scene_source))?)
        } else {
            None
        };

        // Check for manifest to decide multi-pipeline mode
        let manifest = try_load_manifest(pipeline_base);
        let has_multiple_materials = gltf_scene.as_ref().map(|s| s.materials.len() > 1).unwrap_or(false);
        let use_multi_pipeline = manifest.is_some() && has_multiple_materials;

        if use_multi_pipeline {
            info!("Multi-pipeline mode enabled: manifest found and {} materials",
                  gltf_scene.as_ref().map(|s| s.materials.len()).unwrap_or(0));
        }

        // Resolve single-material permutation from manifest (if not multi-pipeline).
        // In multi-pipeline mode, each permutation loads its own shaders.
        // In single-pipeline mode, we need to resolve the best permutation for the scene.
        let resolved_base = if !use_multi_pipeline {
            if let Some(ref mf) = manifest {
                if let Some(ref gs) = gltf_scene {
                    let demo_light_vec = if demo_lights { crate::setup_demo_lights() } else { vec![] };
                    let scene_features = scene_manager::detect_scene_features(gs, &demo_light_vec);
                    let suffix = find_permutation_suffix(mf, &scene_features);
                    if !suffix.is_empty() {
                        let candidate = format!("{}{}", pipeline_base, suffix);
                        if Path::new(&format!("{}.vert.spv", candidate)).exists()
                            && Path::new(&format!("{}.frag.spv", candidate)).exists()
                        {
                            info!("Resolved single-material permutation: {}", suffix);
                            candidate
                        } else {
                            pipeline_base.to_string()
                        }
                    } else {
                        pipeline_base.to_string()
                    }
                } else {
                    pipeline_base.to_string()
                }
            } else {
                pipeline_base.to_string()
            }
        } else {
            pipeline_base.to_string()
        };

        // Load shaders (resolved permutation for single-pipeline, base for multi-pipeline)
        let vert_path = format!("{}.vert.spv", resolved_base);
        let frag_path = format!("{}.frag.spv", resolved_base);
        let vert_code = crate::spv_loader::load_spirv(std::path::Path::new(&vert_path))?;
        let frag_code = crate::spv_loader::load_spirv(std::path::Path::new(&frag_path))?;
        let vert_module = crate::spv_loader::create_shader_module(&device, &vert_code)?;
        let frag_module = crate::spv_loader::create_shader_module(&device, &frag_code)?;

        // Phase 0: Reflection (resolved pipeline)
        let vert_json_path = format!("{}.vert.json", resolved_base);
        let frag_json_path = format!("{}.frag.json", resolved_base);
        let vert_refl = crate::reflected_pipeline::load_reflection(std::path::Path::new(&vert_json_path))?;
        let frag_refl = crate::reflected_pipeline::load_reflection(std::path::Path::new(&frag_json_path))?;

        let merged = crate::reflected_pipeline::merge_descriptor_sets(&[&vert_refl, &frag_refl]);

        // Identify vertex vs fragment set indices
        let mut frag_set_idx_opt: Option<u32> = None;
        let mut vert_set_idx_opt: Option<u32> = None;
        for (&set_idx, bindings) in &merged {
            for binding in bindings {
                if binding.name == "Material" { frag_set_idx_opt = Some(set_idx); }
                if binding.name == "MVP" { vert_set_idx_opt = Some(set_idx); }
            }
        }
        let frag_set_idx = frag_set_idx_opt.unwrap_or(1);
        let vert_set_idx = vert_set_idx_opt.unwrap_or(0);

        let num_materials = gltf_scene.as_ref().map(|s| s.materials.len().max(1)).unwrap_or(1);

        // Vertex stride: always 48 for glTF (with tangent)
        let pipeline_vertex_stride = vert_refl.vertex_stride;

        // -------------------------------------------------------------------
        // Scene geometry (shared between single and multi pipeline modes)
        // -------------------------------------------------------------------
        let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
        let mut auto_target = glam::Vec3::ZERO;
        let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
        let mut auto_far = 100.0f32;
        let mut has_scene_bounds = false;

        // Always pack vertices with 48-byte stride for glTF (tangent included)
        // This is needed for multi-pipeline mode where some permutations expect
        // tangent and some don't -- the buffer must have a consistent stride.
        let gltf_vertex_stride: u32 = if is_gltf { 48 } else { pipeline_vertex_stride };

        let (vertex_data_owned, pbr_indices, draw_ranges) = if let Some(ref mut gltf_s) = gltf_scene {
            if gltf_s.meshes.is_empty() {
                return Err(format!("No meshes found in glTF file: {}", scene_source));
            }
            let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
            if draw_items.is_empty() {
                return Err(format!("No draw items in glTF scene: {}", scene_source));
            }

            let mut vdata: Vec<f32> = Vec::new();
            let mut all_indices: Vec<u32> = Vec::new();
            let mut vertex_offset: u32 = 0;
            let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
            let mut draw_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();
            let mut index_offset: u32 = 0;

            for item in &draw_items {
                let gmesh = &gltf_s.meshes[item.mesh_index];
                let world = item.world_transform;
                let upper3x3 = glam::Mat3::from_mat4(world);
                let normal_mat = upper3x3.inverse().transpose();

                for gv in &gmesh.vertices {
                    let pos = world.transform_point3(glam::Vec3::from(gv.position));
                    let norm = (normal_mat * glam::Vec3::from(gv.normal)).normalize();
                    vertex_positions.push([pos.x, pos.y, pos.z]);

                    vdata.push(pos.x); vdata.push(pos.y); vdata.push(pos.z);
                    vdata.push(norm.x); vdata.push(norm.y); vdata.push(norm.z);
                    vdata.push(gv.uv[0]); vdata.push(gv.uv[1]);

                    // Always pack tangent data for glTF (48-byte stride)
                    if gltf_vertex_stride >= 48 {
                        if gmesh.has_tangents {
                            let tang3 = (upper3x3 * glam::Vec3::new(gv.tangent[0], gv.tangent[1], gv.tangent[2])).normalize();
                            vdata.push(tang3.x); vdata.push(tang3.y); vdata.push(tang3.z);
                            vdata.push(gv.tangent[3]);
                        } else {
                            vdata.push(1.0); vdata.push(0.0); vdata.push(0.0); vdata.push(1.0);
                        }
                    }
                }

                for &idx in &gmesh.indices {
                    all_indices.push(idx + vertex_offset);
                }
                vertex_offset += gmesh.vertices.len() as u32;

                draw_ranges.push(crate::gltf_loader::DrawRange {
                    index_offset,
                    index_count: gmesh.indices.len() as u32,
                    material_index: item.material_index,
                });
                index_offset += gmesh.indices.len() as u32;
            }

            has_scene_bounds = true;
            {
                let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(
                    &vertex_positions,
                    &draw_items,
                );
                auto_eye = eye;
                auto_target = target;
                auto_up = up;
                auto_far = far;
            }

            let data: Vec<u8> = bytemuck::cast_slice(&vdata).to_vec();
            (data, all_indices, draw_ranges)
        } else {
            let (sphere_verts, sphere_indices) = scene::generate_sphere(32, 32);
            if pipeline_vertex_stride == 48 {
                let default_tangent: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
                let mut padded_data: Vec<u8> = Vec::with_capacity(sphere_verts.len() * 48);
                for v in &sphere_verts {
                    padded_data.extend_from_slice(bytemuck::cast_slice(&[*v]));
                    padded_data.extend_from_slice(bytemuck::cast_slice(&default_tangent));
                }
                (padded_data, sphere_indices, Vec::new())
            } else {
                let data: Vec<u8> = bytemuck::cast_slice(&sphere_verts).to_vec();
                (data, sphere_indices, Vec::new())
            }
        };

        let num_indices = pbr_indices.len() as u32;

        let vbo = create_buffer_with_data(
            &device, ctx.allocator_mut(), &vertex_data_owned,
            vk::BufferUsageFlags::VERTEX_BUFFER, "pbr_vbo",
        )?;
        let ibo = create_buffer_with_data(
            &device, ctx.allocator_mut(), bytemuck::cast_slice(&pbr_indices),
            vk::BufferUsageFlags::INDEX_BUFFER, "pbr_ibo",
        )?;

        // -------------------------------------------------------------------
        // Uniforms (shared between single and multi pipeline modes)
        // -------------------------------------------------------------------
        let aspect = width as f32 / height as f32;
        let (model, view, proj) = if has_scene_bounds {
            (
                glam::Mat4::IDENTITY,
                crate::camera::look_at(auto_eye, auto_target, auto_up),
                crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far),
            )
        } else {
            (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
        };

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        let mvp_buffer = create_buffer_with_data(
            &device, ctx.allocator_mut(), &mvp_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "pbr_mvp",
        )?;

        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
        let mut light_data = [0u8; 32];
        light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
        light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
        light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

        let light_buffer = create_buffer_with_data(
            &device, ctx.allocator_mut(), &light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER, "pbr_light",
        )?;

        // --- Multi-light SSBO (Phase E) ---
        let mut sm = scene_manager::SceneManager::new();
        if demo_lights {
            sm.override_lights(crate::setup_demo_lights());
        } else if let Some(ref gs) = gltf_scene {
            sm.populate_lights_from_gltf(&gs.lights);
        } else {
            sm.populate_lights_from_gltf(&[]);
        }

        // Check if any shader in the merged reflection uses a "lights" SSBO
        let has_multi_light = merged.values().any(|bindings| {
            bindings.iter().any(|b| b.name == "lights" && b.binding_type == "storage_buffer")
        });

        let mut lights_ssbo: Option<scene_manager::GpuBuffer> = None;
        if has_multi_light {
            let light_floats = sm.pack_lights_buffer();
            let data: &[u8] = if light_floats.is_empty() {
                &[0u8; 64]
            } else {
                bytemuck::cast_slice(&light_floats)
            };
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), data,
                vk::BufferUsageFlags::STORAGE_BUFFER, "lights_ssbo",
            )?;
            info!("PersistentRenderer multi-light SSBO: {} lights, {} bytes", sm.lights.len(), data.len());
            lights_ssbo = Some(buf);
        }

        // SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
        let mut scene_light_ubo: Option<scene_manager::GpuBuffer> = None;
        if has_multi_light {
            let mut scene_light_data = [0u8; 16];
            scene_light_data[0..12].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
            let light_count = sm.lights.len() as i32;
            scene_light_data[12..16].copy_from_slice(&light_count.to_le_bytes());
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), &scene_light_data,
                vk::BufferUsageFlags::UNIFORM_BUFFER, "scene_light_ubo",
            )?;
            scene_light_ubo = Some(buf);
        }

        // --- Shadow maps (Phase F) ---
        // Detect from reflection: look for "shadow_maps" sampler and "shadow_matrices" storage_buffer
        let has_shadow_maps = merged.values().any(|bindings| {
            bindings.iter().any(|b| b.name == "shadow_maps")
        });
        let has_shadow_matrices = merged.values().any(|bindings| {
            bindings.iter().any(|b| b.name == "shadow_matrices" && b.binding_type == "storage_buffer")
        });
        let shadow_needed = has_shadow_maps && has_shadow_matrices;

        let mut shadow_enabled = false;
        let mut shadow_image = vk::Image::null();
        let mut shadow_image_alloc: Option<gpu_allocator::vulkan::Allocation> = None;
        let mut shadow_array_view = vk::ImageView::null();
        let mut shadow_per_layer_views: Vec<vk::ImageView> = Vec::new();
        let mut shadow_framebuffers: Vec<vk::Framebuffer> = Vec::new();
        let mut shadow_render_pass = vk::RenderPass::null();
        let mut shadow_sampler = vk::Sampler::null();
        let mut shadow_pipeline = vk::Pipeline::null();
        let mut shadow_pipeline_layout = vk::PipelineLayout::null();
        let mut shadow_vert_module = vk::ShaderModule::null();
        let mut shadow_matrices_ssbo: Option<GpuBuffer> = None;
        let mut shadow_count_val: u32 = 0;

        if shadow_needed {
            // Compute shadow data from camera matrices
            let view_mat = if has_scene_bounds {
                crate::camera::look_at(auto_eye, auto_target, auto_up)
            } else {
                DefaultCamera::view()
            };
            let proj_mat = if has_scene_bounds {
                crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far)
            } else {
                DefaultCamera::projection(aspect)
            };
            let view_arr: [f32; 16] = *view_mat.as_ref();
            let proj_arr: [f32; 16] = *proj_mat.as_ref();
            let near_val = 0.1f32;
            let far_val = if has_scene_bounds { auto_far } else { DefaultCamera::FAR };

            sm.compute_shadow_data(&view_arr, &proj_arr, near_val, far_val);
            shadow_count_val = sm.shadow_count() as u32;

            if shadow_count_val > 0 {
                let layer_count = (shadow_count_val as usize).min(MAX_SHADOW_MAPS) as u32;
                let res = create_shadow_resources(
                    &device,
                    ctx.allocator_mut(),
                    layer_count.max(1), // at least 1 layer for valid image
                    gltf_vertex_stride,
                )?;

                shadow_enabled = true;
                shadow_image = res.image;
                shadow_image_alloc = Some(res.image_alloc);
                shadow_array_view = res.array_view;
                shadow_per_layer_views = res.per_layer_views;
                shadow_framebuffers = res.framebuffers;
                shadow_render_pass = res.render_pass;
                shadow_sampler = res.comparison_sampler;
                shadow_pipeline = res.pipeline;
                shadow_pipeline_layout = res.pipeline_layout;
                shadow_vert_module = res.vert_module;

                // Create shadow matrices SSBO
                let shadow_floats = sm.pack_shadow_buffer();
                let data: &[u8] = if shadow_floats.is_empty() {
                    &[0u8; 80] // 20 floats * 4 bytes for at least one entry
                } else {
                    bytemuck::cast_slice(&shadow_floats)
                };
                let buf = create_buffer_with_data(
                    &device, ctx.allocator_mut(), data,
                    vk::BufferUsageFlags::STORAGE_BUFFER, "shadow_matrices_ssbo",
                )?;
                info!("Shadow matrices SSBO: {} entries, {} bytes", sm.shadow_count(), data.len());
                shadow_matrices_ssbo = Some(buf);

                // Re-pack lights buffer to include updated shadow_index values
                if let Some(ref mut lights_buf) = lights_ssbo {
                    let light_floats = sm.pack_lights_buffer();
                    if !light_floats.is_empty() {
                        let light_data: &[u8] = bytemuck::cast_slice(&light_floats);
                        if let Some(ref mut alloc) = lights_buf.allocation {
                            if let Some(mapped) = alloc.mapped_slice_mut() {
                                let copy_len = light_data.len().min(mapped.len());
                                mapped[..copy_len].copy_from_slice(&light_data[..copy_len]);
                            }
                        }
                    }
                }

                info!("Shadow maps enabled: {} shadow-casting light(s)", shadow_count_val);
            } else {
                info!("Shadow maps: no shadow-casting lights found");
            }
        }

        let material_data = if let Some(ref gs) = gltf_scene {
            if !gs.materials.is_empty() {
                MaterialUboData::from_gltf_material(&gs.materials[0])
            } else {
                MaterialUboData::default_values()
            }
        } else {
            MaterialUboData::default_values()
        };

        let material_buffer = create_buffer_with_data(
            &device, ctx.allocator_mut(),
            bytemuck::bytes_of(&material_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER, "pbr_material",
        )?;

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT),
                    None,
                )
                .map_err(|e| format!("Failed to create sampler: {:?}", e))?
        };

        let default_texture = create_default_white_texture(ctx)?;
        let default_black_texture = create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "default_black")?;
        let default_normal_texture = create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "default_normal")?;

        // Build per-material texture maps
        let mut per_material_textures: Vec<HashMap<String, GpuImage>> = Vec::new();

        if let Some(ref gs) = gltf_scene {
            for (mat_idx, mat) in gs.materials.iter().enumerate() {
                let mut tex_map: HashMap<String, GpuImage> = HashMap::new();

                let tex_slots: Vec<(&str, Option<&crate::gltf_loader::TextureImage>)> = vec![
                    ("base_color_tex", mat.base_color_image.as_ref()),
                    ("normal_tex", mat.normal_image.as_ref()),
                    ("metallic_roughness_tex", mat.metallic_roughness_image.as_ref()),
                    ("occlusion_tex", mat.occlusion_image.as_ref()),
                    ("emissive_tex", mat.emissive_image.as_ref()),
                    ("clearcoat_tex", mat.clearcoat_image.as_ref()),
                    ("clearcoat_roughness_tex", mat.clearcoat_roughness_image.as_ref()),
                    ("sheen_color_tex", mat.sheen_color_image.as_ref()),
                    ("transmission_tex", mat.transmission_image.as_ref()),
                ];

                for (name, img_opt) in tex_slots {
                    if let Some(tex_data) = img_opt {
                        let label = format!("{}[{}]", name, mat_idx);
                        info!("Uploading texture '{}': {}x{}", label, tex_data.width, tex_data.height);
                        let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                        tex_map.insert(name.to_string(), gpu_img);
                    }
                }

                if mat.base_color_image.is_some() && !tex_map.contains_key("albedo_tex") {
                    if let Some(tex_data) = mat.base_color_image.as_ref() {
                        let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &format!("albedo_tex[{}]", mat_idx))?;
                        tex_map.insert("albedo_tex".to_string(), gpu_img);
                    }
                }

                per_material_textures.push(tex_map);
            }
        }

        let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
        if gltf_scene.is_none() {
            let tex_pixels = scene::generate_procedural_texture(512);
            let proc_texture = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
            texture_images.insert("albedo_tex".to_string(), proc_texture);
            let proc_texture2 = create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
            texture_images.insert("base_color_tex".to_string(), proc_texture2);
        }

        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // -------------------------------------------------------------------
        // Render pass + framebuffer (shared between single and multi)
        // -------------------------------------------------------------------
        let color_image = create_offscreen_image(
            &device, ctx.allocator_mut(), width, height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR, "pbr_color",
        )?;

        let depth_image = create_offscreen_image(
            &device, ctx.allocator_mut(), width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH, "pbr_depth",
        )?;

        let attachments = [
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let color_ref = vk::AttachmentReference::default()
            .attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(1).layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref))
            .depth_stencil_attachment(&depth_ref);

        let mp_dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let render_pass = unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(std::slice::from_ref(&subpass))
                        .dependencies(&mp_dependencies),
                    None,
                )
                .map_err(|e| format!("Failed to create render pass: {:?}", e))?
        };

        let fb_attachments = [color_image.view, depth_image.view];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&fb_attachments)
                        .width(width).height(height).layers(1),
                    None,
                )
                .map_err(|e| format!("Failed to create framebuffer: {:?}", e))?
        };

        // ===================================================================
        // Branch: multi-pipeline vs single-pipeline
        // ===================================================================
        if use_multi_pipeline {
            // ---------------------------------------------------------------
            // MULTI-PIPELINE MODE
            // ---------------------------------------------------------------
            let gltf_s = gltf_scene.as_ref().unwrap();
            let groups = scene_manager::group_materials_by_features(gltf_s);
            let total_materials = gltf_s.materials.len();

            let mut material_to_permutation: Vec<usize> = vec![0; total_materials];
            let mut permutations: Vec<PermutationPipeline> = Vec::new();

            // Create shared set 0 layout (MVP + Light) from base vertex reflection
            let shared_set0_layout = {
                let set0_merged = crate::reflected_pipeline::merge_descriptor_sets(&[&vert_refl]);
                let set0_layouts = unsafe {
                    crate::reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &set0_merged)?
                };
                let layout = set0_layouts.get(&0).copied().unwrap_or(vk::DescriptorSetLayout::null());
                // Destroy non-set-0 layouts
                for (&idx, &l) in &set0_layouts {
                    if idx != 0 {
                        unsafe { device.destroy_descriptor_set_layout(l, None); }
                    }
                }
                layout
            };

            // Count total descriptor pool requirements
            let mut type_counts: HashMap<vk::DescriptorType, u32> = HashMap::new();
            let mut max_sets: u32 = 1; // set 0

            // Set 0 bindings (shared) - count from base vertex reflection
            if let Some(bindings) = merged.get(&0) {
                for b in bindings {
                    let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&b.binding_type);
                    *type_counts.entry(vk_type).or_insert(0) += 1;
                }
            }

            // Pre-pass: load per-permutation reflections and count descriptors
            struct PermPrep {
                suffix: String,
                perm_base: String,
                material_indices: Vec<usize>,
                vert_refl: crate::reflected_pipeline::ReflectionData,
                frag_refl: crate::reflected_pipeline::ReflectionData,
                perm_merged: HashMap<u32, Vec<crate::reflected_pipeline::BindingInfo>>,
                frag_set_idx: u32,
            }
            let mut perm_preps: Vec<PermPrep> = Vec::new();

            for (suffix, material_indices) in &groups {
                let mut resolved_suffix = suffix.clone();
                let mut perm_base = format!("{}{}", pipeline_base, resolved_suffix);
                let perm_vert_path = format!("{}.vert.spv", perm_base);
                let perm_frag_path = format!("{}.frag.spv", perm_base);

                if !Path::new(&perm_vert_path).exists() || !Path::new(&perm_frag_path).exists() {
                    info!("Missing shader for permutation '{}', falling back to base", resolved_suffix);
                    perm_base = pipeline_base.to_string();
                    resolved_suffix = String::new();
                }

                let perm_vert_json = format!("{}.vert.json", perm_base);
                let perm_frag_json = format!("{}.frag.json", perm_base);
                let p_vert_refl = crate::reflected_pipeline::load_reflection(Path::new(&perm_vert_json))?;
                let p_frag_refl = crate::reflected_pipeline::load_reflection(Path::new(&perm_frag_json))?;
                let p_merged = crate::reflected_pipeline::merge_descriptor_sets(&[&p_vert_refl, &p_frag_refl]);

                // Find frag set idx for this permutation
                let mut p_frag_set_idx: u32 = 1;
                for (&set_idx, bindings) in &p_merged {
                    for b in bindings {
                        if b.name == "Material" { p_frag_set_idx = set_idx; }
                    }
                }

                // Count descriptors needed for per-material sets
                if let Some(bindings) = p_merged.get(&p_frag_set_idx) {
                    for b in bindings {
                        let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&b.binding_type);
                        *type_counts.entry(vk_type).or_insert(0) += material_indices.len() as u32;
                    }
                }
                max_sets += material_indices.len() as u32;

                perm_preps.push(PermPrep {
                    suffix: resolved_suffix,
                    perm_base,
                    material_indices: material_indices.clone(),
                    vert_refl: p_vert_refl,
                    frag_refl: p_frag_refl,
                    perm_merged: p_merged,
                    frag_set_idx: p_frag_set_idx,
                });
            }

            // Create descriptor pool
            let pool_sizes: Vec<vk::DescriptorPoolSize> = type_counts.iter()
                .map(|(&ty, &count)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(count))
                .collect();

            let descriptor_pool = unsafe {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(max_sets)
                            .pool_sizes(&pool_sizes),
                        None,
                    )
                    .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
            };

            // Allocate and write shared set 0 (MVP + Light)
            let vert_ds = if shared_set0_layout != vk::DescriptorSetLayout::null() {
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&shared_set0_layout));
                unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?[0] }
            } else {
                vk::DescriptorSet::null()
            };

            // Write set 0: MVP + Light (+ SceneLight UBO + lights SSBO + shadow bindings if present)
            {
                let set0_merged = crate::reflected_pipeline::merge_descriptor_sets(&[&vert_refl]);
                let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                if let Some(bindings) = set0_merged.get(&0) {
                    for binding in bindings {
                        let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                        if vk_type == vk::DescriptorType::UNIFORM_BUFFER {
                            let (buffer, range) = match binding.name.as_str() {
                                "MVP" => (mvp_buffer.buffer, binding.size as u64),
                                "Light" => (light_buffer.buffer, binding.size as u64),
                                "SceneLight" => {
                                    if let Some(ref sl_buf) = scene_light_ubo {
                                        (sl_buf.buffer, binding.size as u64)
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            };
                            let idx = buf_infos.len();
                            buf_infos.push(vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(range));
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(vert_ds)
                                    .dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                            );
                        } else if vk_type == vk::DescriptorType::STORAGE_BUFFER {
                            if binding.name == "lights" {
                                if let Some(ref lights_buf) = lights_ssbo {
                                    let idx = buf_infos.len();
                                    let ssbo_size = (sm.lights.len() as u64) * 64;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(vert_ds)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                    );
                                }
                            } else if binding.name == "shadow_matrices" {
                                if let Some(ref sm_buf) = shadow_matrices_ssbo {
                                    let idx = buf_infos.len();
                                    let ssbo_size = (shadow_count_val as u64) * 80;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 80 };
                                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(sm_buf.buffer).offset(0).range(range));
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(vert_ds)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                    );
                                }
                            }
                        } else if vk_type == vk::DescriptorType::SAMPLED_IMAGE && binding.name == "shadow_maps" {
                            if shadow_enabled {
                                let idx = img_infos.len();
                                img_infos.push(
                                    vk::DescriptorImageInfo::default()
                                        .image_view(shadow_array_view)
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                                );
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(vert_ds)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                        } else if vk_type == vk::DescriptorType::SAMPLER && binding.name == "shadow_maps" {
                            if shadow_enabled {
                                let idx = img_infos.len();
                                img_infos.push(vk::DescriptorImageInfo::default().sampler(shadow_sampler));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(vert_ds)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLER)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                        }
                    }
                }
                if !writes.is_empty() {
                    unsafe { device.update_descriptor_sets(&writes, &[]); }
                }
            }

            // Build push constant data from the first permutation
            let mut push_constant_data: Vec<u8> = Vec::new();
            let mut push_constant_stage_flags = vk::ShaderStageFlags::empty();
            if let Some(first_prep) = perm_preps.first() {
                let all_pcs: Vec<&crate::reflected_pipeline::PushConstantInfo> = first_prep.vert_refl
                    .push_constants.iter()
                    .chain(first_prep.frag_refl.push_constants.iter())
                    .collect();
                if !all_pcs.is_empty() {
                    let total_size = all_pcs.iter().map(|pc| pc.size as usize).max().unwrap_or(0);
                    push_constant_data.resize(total_size, 0u8);
                    for pc in &all_pcs {
                        push_constant_stage_flags |=
                            crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
                        for field in &pc.fields {
                            let offset = field.offset as usize;
                            match field.name.as_str() {
                                "light_dir" => {
                                    let v = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
                                    let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                    if offset + bytes.len() <= push_constant_data.len() {
                                        push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                                    }
                                }
                                "view_pos" => {
                                    let v = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
                                    let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                    if offset + bytes.len() <= push_constant_data.len() {
                                        push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // For each permutation group: create pipeline + per-material descriptors
            let mut perm_idx: usize = 0;
            for prep in perm_preps {
                // Load shaders
                let perm_vert_path = format!("{}.vert.spv", prep.perm_base);
                let perm_frag_path = format!("{}.frag.spv", prep.perm_base);
                let p_vert_code = crate::spv_loader::load_spirv(Path::new(&perm_vert_path))?;
                let p_frag_code = crate::spv_loader::load_spirv(Path::new(&perm_frag_path))?;
                let p_vert_module = crate::spv_loader::create_shader_module(&device, &p_vert_code)?;
                let p_frag_module = crate::spv_loader::create_shader_module(&device, &p_frag_code)?;

                // Create set 1 layout for this permutation
                let perm_layouts = unsafe {
                    crate::reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &prep.perm_merged)?
                };
                let material_set_layout = perm_layouts.get(&prep.frag_set_idx).copied()
                    .unwrap_or(vk::DescriptorSetLayout::null());
                // Clean up other layouts
                for (&idx, &l) in &perm_layouts {
                    if idx != prep.frag_set_idx {
                        unsafe { device.destroy_descriptor_set_layout(l, None); }
                    }
                }

                // Map materials to this permutation
                for &mi in &prep.material_indices {
                    if mi < material_to_permutation.len() {
                        material_to_permutation[mi] = perm_idx;
                    }
                }

                // Allocate per-material descriptor sets and UBOs
                let mut perm_desc_sets: Vec<vk::DescriptorSet> = Vec::new();
                let mut perm_ubos: Vec<GpuBuffer> = Vec::new();

                for (local_i, &mi) in prep.material_indices.iter().enumerate() {
                    // Create material UBO
                    let mat_data = if mi < gltf_s.materials.len() {
                        MaterialUboData::from_gltf_material(&gltf_s.materials[mi])
                    } else {
                        MaterialUboData::default_values()
                    };

                    let mat_buf = create_buffer_with_data(
                        &device, ctx.allocator_mut(),
                        bytemuck::bytes_of(&mat_data),
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        &format!("perm_{}_material_{}", perm_idx, mi),
                    )?;

                    // Allocate descriptor set
                    let ds = if material_set_layout != vk::DescriptorSetLayout::null() {
                        let info = vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(std::slice::from_ref(&material_set_layout));
                        unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?[0] }
                    } else {
                        vk::DescriptorSet::null()
                    };

                    // Write descriptors for this material
                    {
                        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                        let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                        if let Some(bindings) = prep.perm_merged.get(&prep.frag_set_idx) {
                            for binding in bindings {
                                let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                                match vk_type {
                                    vk::DescriptorType::UNIFORM_BUFFER => {
                                        let (buffer, range) = if binding.name == "Material" {
                                            (mat_buf.buffer, std::mem::size_of::<MaterialUboData>() as u64)
                                        } else if binding.name == "Light" {
                                            (light_buffer.buffer, binding.size as u64)
                                        } else if binding.name == "SceneLight" {
                                            if let Some(ref sl_buf) = scene_light_ubo {
                                                (sl_buf.buffer, binding.size as u64)
                                            } else {
                                                continue;
                                            }
                                        } else {
                                            continue;
                                        };
                                        let idx = buf_infos.len();
                                        buf_infos.push(
                                            vk::DescriptorBufferInfo::default()
                                                .buffer(buffer).offset(0).range(range),
                                        );
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(ds).dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                    vk::DescriptorType::STORAGE_BUFFER => {
                                        if binding.name == "lights" {
                                            if let Some(ref lights_buf) = lights_ssbo {
                                                let idx = buf_infos.len();
                                                let ssbo_size = (sm.lights.len() as u64) * 64;
                                                let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                                buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                                writes.push(
                                                    vk::WriteDescriptorSet::default()
                                                        .dst_set(ds).dst_binding(binding.binding)
                                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                                );
                                            }
                                        } else if binding.name == "shadow_matrices" {
                                            if let Some(ref sm_buf) = shadow_matrices_ssbo {
                                                let idx = buf_infos.len();
                                                let ssbo_size = (shadow_count_val as u64) * 80;
                                                let range = if ssbo_size > 0 { ssbo_size } else { 80 };
                                                buf_infos.push(vk::DescriptorBufferInfo::default().buffer(sm_buf.buffer).offset(0).range(range));
                                                writes.push(
                                                    vk::WriteDescriptorSet::default()
                                                        .dst_set(ds).dst_binding(binding.binding)
                                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                                );
                                            }
                                        }
                                    }
                                    vk::DescriptorType::SAMPLER => {
                                        let actual_sampler = if binding.name == "shadow_maps" && shadow_enabled {
                                            shadow_sampler
                                        } else {
                                            ibl_assets.sampler_for_binding(&binding.name).unwrap_or(sampler)
                                        };
                                        let idx = img_infos.len();
                                        img_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(ds).dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                    vk::DescriptorType::SAMPLED_IMAGE => {
                                        let is_cube = crate::reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                                        let view = if binding.name == "shadow_maps" && shadow_enabled {
                                            shadow_array_view
                                        } else if is_cube || binding.name == "brdf_lut" {
                                            ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                                        } else {
                                            let per_mat_view = if mi < per_material_textures.len() {
                                                per_material_textures[mi].get(&binding.name).map(|img| img.view)
                                            } else {
                                                None
                                            };
                                            per_mat_view.unwrap_or_else(|| {
                                                let fallback = match binding.name.as_str() {
                                                    "emissive_tex" => default_black_texture.view,
                                                    "normal_tex" => default_normal_texture.view,
                                                    _ => default_texture.view,
                                                };
                                                texture_images.get(&binding.name).map(|img| img.view).unwrap_or(fallback)
                                            })
                                        };
                                        let idx = img_infos.len();
                                        img_infos.push(
                                            vk::DescriptorImageInfo::default()
                                                .image_view(view)
                                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                                        );
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(ds).dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                                .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if !writes.is_empty() {
                            unsafe { device.update_descriptor_sets(&writes, &[]); }
                        }
                    }

                    perm_desc_sets.push(ds);
                    perm_ubos.push(mat_buf);
                    let _ = local_i; // suppress unused warning
                }

                // Create pipeline layout: set 0 (shared) + set 1 (per-permutation)
                let mut layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
                if shared_set0_layout != vk::DescriptorSetLayout::null() {
                    layouts.push(shared_set0_layout);
                }
                if material_set_layout != vk::DescriptorSetLayout::null() {
                    layouts.push(material_set_layout);
                }

                // Push constant ranges for this permutation
                let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
                for pc in &prep.vert_refl.push_constants {
                    push_ranges.push(
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(pc.size),
                    );
                }
                for pc in &prep.frag_refl.push_constants {
                    push_ranges.push(
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                            .offset(0)
                            .size(pc.size),
                    );
                }

                let perm_pipeline_layout = unsafe {
                    let layout_info = vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push_ranges);
                    device.create_pipeline_layout(&layout_info, None)
                        .map_err(|e| format!("Failed to create permutation pipeline layout: {:?}", e))?
                };

                // Create graphics pipeline with override_stride=48 for glTF
                let entry_name = c"main";
                let shader_stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(p_vert_module).name(entry_name),
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(p_frag_module).name(entry_name),
                ];

                let (binding_desc, attr_descs) =
                    crate::reflected_pipeline::create_reflected_vertex_input_with_stride(&prep.vert_refl, 48);

                let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                    .vertex_binding_descriptions(&binding_desc)
                    .vertex_attribute_descriptions(&attr_descs);
                let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

                let viewport = vk::Viewport::default()
                    .width(width as f32).height(height as f32).max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
                let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                    .viewports(std::slice::from_ref(&viewport))
                    .scissors(std::slice::from_ref(&scissor));

                let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                    .polygon_mode(vk::PolygonMode::FILL)
                    .cull_mode(vk::CullModeFlags::NONE)
                    .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                    .line_width(1.0);
                let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1);
                let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_test_enable(true).depth_write_enable(true)
                    .depth_compare_op(vk::CompareOp::LESS);
                let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(std::slice::from_ref(&color_blend_attachment));

                let perm_pipeline = unsafe {
                    device
                        .create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            &[vk::GraphicsPipelineCreateInfo::default()
                                .stages(&shader_stages)
                                .vertex_input_state(&vertex_input)
                                .input_assembly_state(&input_assembly)
                                .viewport_state(&viewport_state)
                                .rasterization_state(&rasterizer)
                                .multisample_state(&multisampling)
                                .depth_stencil_state(&depth_stencil)
                                .color_blend_state(&color_blending)
                                .layout(perm_pipeline_layout)
                                .render_pass(render_pass)
                                .subpass(0)],
                            None,
                        )
                        .map_err(|e| format!("Failed to create permutation pipeline '{}': {:?}", prep.suffix, e))?[0]
                };

                info!("Loaded permutation '{}' from {} ({} material(s))",
                      prep.suffix, prep.perm_base, prep.material_indices.len());

                permutations.push(PermutationPipeline {
                    suffix: prep.suffix,
                    base_path: prep.perm_base,
                    vert_module: p_vert_module,
                    frag_module: p_frag_module,
                    vert_refl: prep.vert_refl,
                    frag_refl: prep.frag_refl,
                    material_set_layout,
                    pipeline_layout: perm_pipeline_layout,
                    pipeline: perm_pipeline,
                    per_material_desc_sets: perm_desc_sets,
                    per_material_ubos: perm_ubos,
                    material_indices: prep.material_indices,
                });

                perm_idx += 1;
            }

            info!("Multi-pipeline setup complete: {} permutation(s) for {} material(s)",
                  permutations.len(), total_materials);

            // Use a null pipeline/layout for the base (we use permutation pipelines instead)
            let ds_layouts = HashMap::new(); // no shared ds_layouts in multi mode
            let descriptor_sets = Vec::new();

            Ok(PersistentRenderer {
                pipeline: vk::Pipeline::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_pool,
                ds_layouts,
                descriptor_sets,
                vert_module,
                frag_module,
                render_pass,
                framebuffer,
                color_image,
                depth_image,
                width,
                height,
                vbo,
                ibo,
                num_indices,
                draw_ranges,
                per_material_desc_sets: Vec::new(),
                per_material_buffers: Vec::new(),
                per_material_textures,
                vert_set_idx: 0,
                frag_set_idx: 1,
                vert_ds,
                mvp_buffer,
                light_buffer,
                material_buffer,
                has_multi_light,
                lights_ssbo,
                scene_light_ubo,
                shadow_enabled,
                shadow_image,
                shadow_image_alloc,
                shadow_array_view,
                shadow_per_layer_views,
                shadow_framebuffers,
                shadow_render_pass,
                shadow_sampler,
                shadow_pipeline,
                shadow_pipeline_layout,
                shadow_vert_module,
                shadow_matrices_ssbo,
                shadow_count: shadow_count_val,
                scene_manager: sm,
                push_constant_data,
                push_constant_stage_flags,
                sampler,
                default_texture,
                default_black_texture,
                default_normal_texture,
                texture_images,
                ibl_assets,
                multi_pipeline: true,
                permutations,
                material_to_permutation,
                shared_set0_layout,
                auto_eye,
                auto_target,
                auto_up,
                auto_far,
                has_scene_bounds,
            })
        } else {
            // ---------------------------------------------------------------
            // SINGLE-PIPELINE MODE (original behavior)
            // ---------------------------------------------------------------
            let ds_layouts = unsafe {
                crate::reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &merged)?
            };

            let (mut pool_sizes, _) = crate::reflected_pipeline::compute_pool_sizes(&merged);
            for ps in &mut pool_sizes {
                let frag_count: u32 = merged.get(&frag_set_idx).map(|bindings| {
                    bindings.iter().filter(|b| {
                        crate::reflected_pipeline::binding_type_to_vk_public(&b.binding_type) == ps.ty
                    }).count() as u32
                }).unwrap_or(0);
                if frag_count > 0 {
                    ps.descriptor_count += frag_count * (num_materials as u32 - 1);
                }
            }
            let max_sets = 1 + num_materials as u32;

            let descriptor_pool = unsafe {
                device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(max_sets)
                            .pool_sizes(&pool_sizes),
                        None,
                    )
                    .map_err(|e| format!("Failed to create descriptor pool: {:?}", e))?
            };

            let vert_ds = if let Some(&layout) = ds_layouts.get(&vert_set_idx) {
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&layout));
                unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?[0] }
            } else {
                vk::DescriptorSet::null()
            };

            let mut per_material_desc_sets: Vec<vk::DescriptorSet> = Vec::new();
            let mut per_material_buffers: Vec<GpuBuffer> = Vec::new();

            if let Some(&frag_layout) = ds_layouts.get(&frag_set_idx) {
                let layouts_vec: Vec<vk::DescriptorSetLayout> = vec![frag_layout; num_materials];
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts_vec);
                per_material_desc_sets = unsafe {
                    device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed: {:?}", e))?
                };
            }

            let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
            for pc in &vert_refl.push_constants {
                push_ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                        .offset(0)
                        .size(pc.size),
                );
            }
            for pc in &frag_refl.push_constants {
                push_ranges.push(
                    vk::PushConstantRange::default()
                        .stage_flags(crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags))
                        .offset(0)
                        .size(pc.size),
                );
            }

            let mut push_constant_data: Vec<u8> = Vec::new();
            let mut push_constant_stage_flags = vk::ShaderStageFlags::empty();
            {
                let all_pcs: Vec<&crate::reflected_pipeline::PushConstantInfo> = vert_refl
                    .push_constants.iter()
                    .chain(frag_refl.push_constants.iter())
                    .collect();
                if !all_pcs.is_empty() {
                    let total_size = all_pcs.iter().map(|pc| pc.size as usize).max().unwrap_or(0);
                    push_constant_data.resize(total_size, 0u8);
                    for pc in &all_pcs {
                        push_constant_stage_flags |=
                            crate::reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
                        for field in &pc.fields {
                            let offset = field.offset as usize;
                            match field.name.as_str() {
                                "light_dir" => {
                                    let v = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
                                    let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                    push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                                }
                                "view_pos" => {
                                    let v = glam::Vec3::new(0.0, 0.0, 3.0);
                                    let bytes = bytemuck::cast_slice::<f32, u8>(v.as_ref());
                                    push_constant_data[offset..offset + bytes.len()].copy_from_slice(bytes);
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            let pipeline_layout = unsafe {
                crate::reflected_pipeline::create_pipeline_layout_from_merged(&device, &ds_layouts, &push_ranges)?
            };

            // Write vertex set: MVP + Light (+ SceneLight UBO + lights SSBO + shadow bindings)
            {
                let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                if let Some(bindings) = merged.get(&vert_set_idx) {
                    for binding in bindings {
                        let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                        if vk_type == vk::DescriptorType::UNIFORM_BUFFER {
                            let (buffer, range) = match binding.name.as_str() {
                                "MVP" => (mvp_buffer.buffer, binding.size as u64),
                                "Light" => (light_buffer.buffer, binding.size as u64),
                                "SceneLight" => {
                                    if let Some(ref sl_buf) = scene_light_ubo {
                                        (sl_buf.buffer, binding.size as u64)
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            };
                            let idx = buf_infos.len();
                            buf_infos.push(vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(range));
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(vert_ds)
                                    .dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                            );
                        } else if vk_type == vk::DescriptorType::STORAGE_BUFFER {
                            if binding.name == "lights" {
                                if let Some(ref lights_buf) = lights_ssbo {
                                    let idx = buf_infos.len();
                                    let ssbo_size = (sm.lights.len() as u64) * 64;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(vert_ds)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                    );
                                }
                            } else if binding.name == "shadow_matrices" {
                                if let Some(ref sm_buf) = shadow_matrices_ssbo {
                                    let idx = buf_infos.len();
                                    let ssbo_size = (shadow_count_val as u64) * 80;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 80 };
                                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(sm_buf.buffer).offset(0).range(range));
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(vert_ds)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                    );
                                }
                            }
                        } else if vk_type == vk::DescriptorType::SAMPLED_IMAGE && binding.name == "shadow_maps" {
                            if shadow_enabled {
                                let idx = img_infos.len();
                                img_infos.push(
                                    vk::DescriptorImageInfo::default()
                                        .image_view(shadow_array_view)
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                                );
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(vert_ds)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                        } else if vk_type == vk::DescriptorType::SAMPLER && binding.name == "shadow_maps" {
                            // Combined sampler for shadow maps
                            if shadow_enabled {
                                let idx = img_infos.len();
                                img_infos.push(vk::DescriptorImageInfo::default().sampler(shadow_sampler));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(vert_ds)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLER)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                        }
                    }
                }
                if !writes.is_empty() {
                    unsafe { device.update_descriptor_sets(&writes, &[]); }
                }
            }

            for mi in 0..num_materials {
                let mat_data = if let Some(ref gs) = gltf_scene {
                    if mi < gs.materials.len() {
                        MaterialUboData::from_gltf_material(&gs.materials[mi])
                    } else {
                        MaterialUboData::default_values()
                    }
                } else {
                    MaterialUboData::default_values()
                };

                let mat_buf = create_buffer_with_data(
                    &device, ctx.allocator_mut(),
                    bytemuck::bytes_of(&mat_data),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    &format!("pbr_material_{}", mi),
                )?;
                per_material_buffers.push(mat_buf);
            }

            for mi in 0..num_materials {
                let ds = per_material_desc_sets[mi];
                let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                if let Some(bindings) = merged.get(&frag_set_idx) {
                    for binding in bindings {
                        let vk_type = crate::reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                        match vk_type {
                            vk::DescriptorType::UNIFORM_BUFFER => {
                                let (buffer, range) = if binding.name == "Material" {
                                    (per_material_buffers[mi].buffer, std::mem::size_of::<MaterialUboData>() as u64)
                                } else if binding.name == "Light" {
                                    (light_buffer.buffer, binding.size as u64)
                                } else if binding.name == "SceneLight" {
                                    if let Some(ref sl_buf) = scene_light_ubo {
                                        (sl_buf.buffer, binding.size as u64)
                                    } else {
                                        continue;
                                    }
                                } else {
                                    continue;
                                };
                                let idx = buf_infos.len();
                                buf_infos.push(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buffer).offset(0).range(range),
                                );
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                );
                            }
                            vk::DescriptorType::STORAGE_BUFFER => {
                                if binding.name == "lights" {
                                    if let Some(ref lights_buf) = lights_ssbo {
                                        let idx = buf_infos.len();
                                        let ssbo_size = (sm.lights.len() as u64) * 64;
                                        let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                        buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(ds).dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                } else if binding.name == "shadow_matrices" {
                                    if let Some(ref sm_buf) = shadow_matrices_ssbo {
                                        let idx = buf_infos.len();
                                        let ssbo_size = (shadow_count_val as u64) * 80;
                                        let range = if ssbo_size > 0 { ssbo_size } else { 80 };
                                        buf_infos.push(vk::DescriptorBufferInfo::default().buffer(sm_buf.buffer).offset(0).range(range));
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(ds).dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                }
                            }
                            vk::DescriptorType::SAMPLER => {
                                let actual_sampler = if binding.name == "shadow_maps" && shadow_enabled {
                                    shadow_sampler
                                } else {
                                    ibl_assets.sampler_for_binding(&binding.name).unwrap_or(sampler)
                                };
                                let idx = img_infos.len();
                                img_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLER)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                            vk::DescriptorType::SAMPLED_IMAGE => {
                                let is_cube = crate::reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                                let view = if binding.name == "shadow_maps" && shadow_enabled {
                                    shadow_array_view
                                } else if is_cube || binding.name == "brdf_lut" {
                                    ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                                } else {
                                    let per_mat_view = if mi < per_material_textures.len() {
                                        per_material_textures[mi].get(&binding.name).map(|img| img.view)
                                    } else {
                                        None
                                    };
                                    per_mat_view.unwrap_or_else(|| {
                                        let fallback = match binding.name.as_str() {
                                            "emissive_tex" => default_black_texture.view,
                                            "normal_tex" => default_normal_texture.view,
                                            _ => default_texture.view,
                                        };
                                        texture_images.get(&binding.name).map(|img| img.view).unwrap_or(fallback)
                                    })
                                };
                                let idx = img_infos.len();
                                img_infos.push(
                                    vk::DescriptorImageInfo::default()
                                        .image_view(view)
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                                );
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                        .image_info(unsafe { std::slice::from_raw_parts(&img_infos[idx] as *const _, 1) }),
                                );
                            }
                            _ => {}
                        }
                    }
                }
                if !writes.is_empty() {
                    unsafe { device.update_descriptor_sets(&writes, &[]); }
                }
            }

            let descriptor_sets: Vec<vk::DescriptorSet> = {
                let max_idx = std::cmp::max(vert_set_idx, frag_set_idx);
                let mut sets = vec![vk::DescriptorSet::null(); (max_idx + 1) as usize];
                sets[vert_set_idx as usize] = vert_ds;
                sets
            };

            let entry_name = c"main";
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module).name(entry_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module).name(entry_name),
            ];

            // Use stride override for glTF (buffer is always packed at 48-byte stride)
            let (binding_desc, attr_descs) = if is_gltf {
                crate::reflected_pipeline::create_reflected_vertex_input_with_stride(&vert_refl, gltf_vertex_stride)
            } else {
                crate::reflected_pipeline::create_reflected_vertex_input(&vert_refl)
            };

            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_desc)
                .vertex_attribute_descriptions(&attr_descs);
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewport = vk::Viewport::default()
                .width(width as f32).height(height as f32).max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(std::slice::from_ref(&viewport))
                .scissors(std::slice::from_ref(&scissor));

            let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(true).depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS);
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&color_blend_attachment));

            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::GraphicsPipelineCreateInfo::default()
                            .stages(&shader_stages)
                            .vertex_input_state(&vertex_input)
                            .input_assembly_state(&input_assembly)
                            .viewport_state(&viewport_state)
                            .rasterization_state(&rasterizer)
                            .multisample_state(&multisampling)
                            .depth_stencil_state(&depth_stencil)
                            .color_blend_state(&color_blending)
                            .layout(pipeline_layout)
                            .render_pass(render_pass)
                            .subpass(0)],
                        None,
                    )
                    .map_err(|e| format!("Failed to create pipeline: {:?}", e))?[0]
            };

            info!("PersistentRenderer initialized (single-pipeline): {}x{}", width, height);

            Ok(PersistentRenderer {
                pipeline,
                pipeline_layout,
                descriptor_pool,
                ds_layouts,
                descriptor_sets,
                vert_module,
                frag_module,
                render_pass,
                framebuffer,
                color_image,
                depth_image,
                width,
                height,
                vbo,
                ibo,
                num_indices,
                draw_ranges,
                per_material_desc_sets,
                per_material_buffers,
                per_material_textures,
                vert_set_idx,
                frag_set_idx,
                vert_ds,
                mvp_buffer,
                light_buffer,
                material_buffer,
                has_multi_light,
                lights_ssbo,
                scene_light_ubo,
                shadow_enabled,
                shadow_image,
                shadow_image_alloc,
                shadow_array_view,
                shadow_per_layer_views,
                shadow_framebuffers,
                shadow_render_pass,
                shadow_sampler,
                shadow_pipeline,
                shadow_pipeline_layout,
                shadow_vert_module,
                shadow_matrices_ssbo,
                shadow_count: shadow_count_val,
                scene_manager: sm,
                push_constant_data,
                push_constant_stage_flags,
                sampler,
                default_texture,
                default_black_texture,
                default_normal_texture,
                texture_images,
                ibl_assets,
                multi_pipeline: false,
                permutations: Vec::new(),
                material_to_permutation: Vec::new(),
                shared_set0_layout: vk::DescriptorSetLayout::null(),
                auto_eye,
                auto_target,
                auto_up,
                auto_far,
                has_scene_bounds,
            })
        }
    }

    /// Update camera uniforms (MVP + light view_pos).
    pub fn update_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) {
        let model = glam::Mat4::IDENTITY;
        let view = crate::camera::look_at(eye, target, up);
        let proj = crate::camera::perspective(fov_y, aspect, near, far);

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        if let Some(ref mut alloc) = self.mvp_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                mapped[..192].copy_from_slice(&mvp_data);
            }
        }

        // Update view_pos in light buffer (offset 16, size 12)
        if let Some(ref mut alloc) = self.light_buffer.allocation {
            if let Some(mapped) = alloc.mapped_slice_mut() {
                let bytes = bytemuck::cast_slice::<f32, u8>(eye.as_ref());
                mapped[16..28].copy_from_slice(bytes);
            }
        }

        // Update view_pos in SceneLight UBO (offset 0, size 12) for multi-light mode
        if self.has_multi_light {
            if let Some(ref mut sl_buf) = self.scene_light_ubo {
                if let Some(ref mut alloc) = sl_buf.allocation {
                    if let Some(mapped) = alloc.mapped_slice_mut() {
                        let bytes = bytemuck::cast_slice::<f32, u8>(eye.as_ref());
                        mapped[0..12].copy_from_slice(bytes);
                    }
                }
            }
        }
    }

    /// Record and submit rendering commands. The offscreen color_image
    /// ends up in TRANSFER_SRC_OPTIMAL layout ready for blit.
    pub fn render_frame(&self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        let device = &ctx.device;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin command buffer: {:?}", e))?;
        }

        // --- Shadow pass (before main color pass) ---
        if self.shadow_enabled && self.shadow_count > 0 {
            ctx.cmd_begin_label(cmd, "Shadow Pass", [0.5, 0.5, 0.0, 1.0]);
            let shadow_clear = [vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            }];

            for layer in 0..self.shadow_count as usize {
                if layer >= self.shadow_framebuffers.len() || layer >= self.scene_manager.shadow_entries.len() {
                    break;
                }

                let shadow_rp_begin = vk::RenderPassBeginInfo::default()
                    .render_pass(self.shadow_render_pass)
                    .framebuffer(self.shadow_framebuffers[layer])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: SHADOW_MAP_RESOLUTION,
                            height: SHADOW_MAP_RESOLUTION,
                        },
                    })
                    .clear_values(&shadow_clear);

                unsafe {
                    device.cmd_begin_render_pass(cmd, &shadow_rp_begin, vk::SubpassContents::INLINE);
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.shadow_pipeline);
                    device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbo.buffer], &[0]);
                    device.cmd_bind_index_buffer(cmd, self.ibo.buffer, 0, vk::IndexType::UINT32);

                    // Push the light's view-projection matrix as MVP (model is identity)
                    let light_vp = &self.scene_manager.shadow_entries[layer].view_projection;
                    let mvp_bytes: &[u8] = bytemuck::cast_slice(light_vp);
                    device.cmd_push_constants(
                        cmd,
                        self.shadow_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        mvp_bytes,
                    );

                    // Draw all meshes
                    if !self.draw_ranges.is_empty() {
                        for range in &self.draw_ranges {
                            device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
                        }
                    } else {
                        device.cmd_draw_indexed(cmd, self.num_indices, 1, 0, 0, 0);
                    }

                    device.cmd_end_render_pass(cmd);
                }
            }
            ctx.cmd_end_label(cmd);
        }

        // --- Main color pass ---
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.05, 0.05, 0.08, 1.0] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D { width: self.width, height: self.height },
            })
            .clear_values(&clear_values);

        ctx.cmd_begin_label(cmd, "Raster Pass", [0.2, 0.8, 0.2, 1.0]);
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vbo.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.ibo.buffer, 0, vk::IndexType::UINT32);

            if self.multi_pipeline && !self.permutations.is_empty() {
                // ---- Multi-pipeline draw ----
                // Bind shared set 0 (MVP + Light) using the first permutation's layout
                if self.vert_ds != vk::DescriptorSet::null() {
                    device.cmd_bind_descriptor_sets(
                        cmd, vk::PipelineBindPoint::GRAPHICS,
                        self.permutations[0].pipeline_layout, 0,
                        std::slice::from_ref(&self.vert_ds), &[],
                    );
                }

                let mut current_perm_idx: i32 = -1;

                for range in &self.draw_ranges {
                    let mat_idx = range.material_index;
                    let perm_idx = if mat_idx < self.material_to_permutation.len() {
                        self.material_to_permutation[mat_idx]
                    } else {
                        0
                    };
                    let perm = &self.permutations[perm_idx];

                    // Switch pipeline if permutation changed
                    if perm_idx as i32 != current_perm_idx {
                        current_perm_idx = perm_idx as i32;
                        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, perm.pipeline);
                        if !self.push_constant_data.is_empty() {
                            device.cmd_push_constants(
                                cmd, perm.pipeline_layout, self.push_constant_stage_flags,
                                0, &self.push_constant_data,
                            );
                        }
                        // Re-bind set 0 after pipeline change
                        if self.vert_ds != vk::DescriptorSet::null() {
                            device.cmd_bind_descriptor_sets(
                                cmd, vk::PipelineBindPoint::GRAPHICS,
                                perm.pipeline_layout, 0,
                                std::slice::from_ref(&self.vert_ds), &[],
                            );
                        }
                    }

                    // Find per-material descriptor set index within this permutation
                    let local_mat_idx = perm.material_indices.iter().position(|&mi| mi == mat_idx);
                    if let Some(local_idx) = local_mat_idx {
                        if local_idx < perm.per_material_desc_sets.len() {
                            device.cmd_bind_descriptor_sets(
                                cmd, vk::PipelineBindPoint::GRAPHICS,
                                perm.pipeline_layout, 1,
                                std::slice::from_ref(&perm.per_material_desc_sets[local_idx]), &[],
                            );
                        }
                    }

                    device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
                }
            } else {
                // ---- Single-pipeline draw (original) ----
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

                if !self.push_constant_data.is_empty() {
                    device.cmd_push_constants(
                        cmd, self.pipeline_layout, self.push_constant_stage_flags,
                        0, &self.push_constant_data,
                    );
                }

                // Bind vertex set (shared)
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, self.vert_set_idx, std::slice::from_ref(&self.vert_ds), &[]);

                if !self.draw_ranges.is_empty() && !self.per_material_desc_sets.is_empty() {
                    let mut current_mat: i32 = -1;
                    for range in &self.draw_ranges {
                        if range.material_index as i32 != current_mat {
                            current_mat = range.material_index as i32;
                            let mat_idx = (current_mat as usize).min(self.per_material_desc_sets.len() - 1);
                            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, self.frag_set_idx, std::slice::from_ref(&self.per_material_desc_sets[mat_idx]), &[]);
                        }
                        device.cmd_draw_indexed(cmd, range.index_count, 1, range.index_offset, 0, 0);
                    }
                } else {
                    // Fallback: single draw
                    if !self.per_material_desc_sets.is_empty() {
                        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, self.frag_set_idx, std::slice::from_ref(&self.per_material_desc_sets[0]), &[]);
                    }
                    device.cmd_draw_indexed(cmd, self.num_indices, 1, 0, 0, 0);
                }
            }

            device.cmd_end_render_pass(cmd);
        }
        ctx.cmd_end_label(cmd);

        Ok(cmd)
    }

    /// Blit the offscreen color image to a swapchain image.
    /// Assumes cmd is already recording. Transitions swapchain image layout.
    pub fn cmd_blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swapchain_image: vk::Image,
        swapchain_extent: vk::Extent2D,
    ) {
        unsafe {
            // Transition swapchain image: UNDEFINED -> TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1).layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );

            // Blit offscreen -> swapchain
            let blit_region = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: self.width as i32,
                        y: self.height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: swapchain_extent.width as i32,
                        y: swapchain_extent.height as i32,
                        z: 1,
                    },
                ]);

            device.cmd_blit_image(
                cmd,
                self.color_image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit_region],
                vk::Filter::LINEAR,
            );

            // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .image(swapchain_image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1).layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }
    }

    /// Clean up all GPU resources.
    pub fn cleanup(&mut self, ctx: &mut VulkanContext) {
        let device = ctx.device.clone();

        unsafe {
            // Clean up multi-pipeline resources
            for perm in self.permutations.drain(..) {
                device.destroy_pipeline(perm.pipeline, None);
                device.destroy_pipeline_layout(perm.pipeline_layout, None);
                if perm.material_set_layout != vk::DescriptorSetLayout::null() {
                    device.destroy_descriptor_set_layout(perm.material_set_layout, None);
                }
                device.destroy_shader_module(perm.vert_module, None);
                device.destroy_shader_module(perm.frag_module, None);
                for mut ubo in perm.per_material_ubos {
                    ubo.destroy(&device, ctx.allocator_mut());
                }
            }
            if self.shared_set0_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(self.shared_set0_layout, None);
            }

            // Clean up single-pipeline resources
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(self.pipeline_layout, None);
            }
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for (_, layout) in &self.ds_layouts {
                device.destroy_descriptor_set_layout(*layout, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }

        self.color_image.destroy(&device, ctx.allocator_mut());
        self.depth_image.destroy(&device, ctx.allocator_mut());
        self.default_texture.destroy(&device, ctx.allocator_mut());
        self.default_black_texture.destroy(&device, ctx.allocator_mut());
        self.default_normal_texture.destroy(&device, ctx.allocator_mut());
        for (_, mut tex) in self.texture_images.drain() {
            tex.destroy(&device, ctx.allocator_mut());
        }
        self.ibl_assets.destroy(&device, ctx.allocator_mut());
        self.mvp_buffer.destroy(&device, ctx.allocator_mut());
        self.light_buffer.destroy(&device, ctx.allocator_mut());
        if let Some(mut buf) = self.lights_ssbo.take() {
            buf.destroy(&device, ctx.allocator_mut());
        }
        if let Some(mut buf) = self.scene_light_ubo.take() {
            buf.destroy(&device, ctx.allocator_mut());
        }

        // Clean up shadow resources
        if self.shadow_enabled {
            unsafe {
                if self.shadow_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.shadow_pipeline, None);
                }
                if self.shadow_pipeline_layout != vk::PipelineLayout::null() {
                    device.destroy_pipeline_layout(self.shadow_pipeline_layout, None);
                }
                if self.shadow_vert_module != vk::ShaderModule::null() {
                    device.destroy_shader_module(self.shadow_vert_module, None);
                }
                for fb in self.shadow_framebuffers.drain(..) {
                    device.destroy_framebuffer(fb, None);
                }
                for view in self.shadow_per_layer_views.drain(..) {
                    device.destroy_image_view(view, None);
                }
                if self.shadow_array_view != vk::ImageView::null() {
                    device.destroy_image_view(self.shadow_array_view, None);
                }
                if self.shadow_render_pass != vk::RenderPass::null() {
                    device.destroy_render_pass(self.shadow_render_pass, None);
                }
                if self.shadow_sampler != vk::Sampler::null() {
                    device.destroy_sampler(self.shadow_sampler, None);
                }
            }
            if let Some(alloc) = self.shadow_image_alloc.take() {
                let _ = ctx.allocator_mut().free(alloc);
            }
            if self.shadow_image != vk::Image::null() {
                unsafe { device.destroy_image(self.shadow_image, None); }
            }
            if let Some(mut buf) = self.shadow_matrices_ssbo.take() {
                buf.destroy(&device, ctx.allocator_mut());
            }
        }

        self.material_buffer.destroy(&device, ctx.allocator_mut());
        for mut buf in self.per_material_buffers.drain(..) {
            buf.destroy(&device, ctx.allocator_mut());
        }
        for tex_map in self.per_material_textures.drain(..) {
            for (_, mut tex) in tex_map {
                tex.destroy(&device, ctx.allocator_mut());
            }
        }
        self.vbo.destroy(&device, ctx.allocator_mut());
        self.ibo.destroy(&device, ctx.allocator_mut());
    }
}

// ===========================================================================
// Renderer trait implementation for PersistentRenderer
// ===========================================================================

impl scene_manager::Renderer for PersistentRenderer {
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String> {
        self.render_frame(ctx)
    }

    fn update_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) {
        PersistentRenderer::update_camera(self, eye, target, up, fov_y, aspect, near, far);
    }

    fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swap_image: vk::Image,
        extent: vk::Extent2D,
    ) {
        self.cmd_blit_to_swapchain(device, cmd, swap_image, extent);
    }

    fn output_image(&self) -> vk::Image {
        self.color_image.image
    }

    fn output_format(&self) -> vk::Format {
        vk::Format::R8G8B8A8_UNORM
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn has_scene_bounds(&self) -> bool {
        self.has_scene_bounds
    }

    fn auto_eye(&self) -> glam::Vec3 {
        self.auto_eye
    }

    fn auto_target(&self) -> glam::Vec3 {
        self.auto_target
    }

    fn auto_up(&self) -> glam::Vec3 {
        self.auto_up
    }

    fn auto_far(&self) -> f32 {
        self.auto_far
    }

    fn wait_stage(&self) -> vk::PipelineStageFlags {
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
    }

    fn destroy(&mut self, ctx: &mut VulkanContext) {
        self.cleanup(ctx);
    }
}
