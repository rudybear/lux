//! Mesh shader renderer: builds meshlets, creates mesh+fragment pipeline, dispatches mesh tasks.
//!
//! Follows the raster_renderer pattern for render pass/framebuffer/depth buffer
//! and the rt_renderer pattern for SoA storage buffers.
//!
//! Supports both single-pipeline mode (original) and multi-pipeline mode
//! (when a .manifest.json exists AND the scene has multiple materials).

use ash::vk;
use bytemuck;
use gpu_allocator::vulkan::Allocator;
use log::info;
use std::collections::HashMap;
use std::path::Path;

use crate::camera::DefaultCamera;
use crate::meshlet;
use crate::reflected_pipeline;
use crate::scene;
use crate::scene_manager::{self, GpuBuffer, GpuImage, IblAssets};
use crate::spv_loader;
use crate::vulkan_context::VulkanContext;

/// Material UBO data matching `properties Material { ... }` in gltf_pbr_layered.lux (std140 layout, 144 bytes).
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
// Manifest loading (same pattern as raster_renderer)
// ===========================================================================

/// A single manifest permutation entry.
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
fn try_load_manifest(pipeline_base: &str) -> Option<ShaderManifest> {
    let path1 = format!("{}.manifest.json", pipeline_base);
    if Path::new(&path1).exists() {
        info!("Loading mesh shader manifest: {}", path1);
        if let Some(manifest) = parse_manifest_json(Path::new(&path1)) {
            if !manifest.permutations.is_empty() {
                return Some(manifest);
            }
        }
    }

    // Try legacy subdirectory format
    let p = std::path::Path::new(pipeline_base);
    if let (Some(dir), Some(filename)) = (p.parent(), p.file_name()) {
        let path2 = dir.join("gltf_pbr").join(format!("{}.manifest.json", filename.to_string_lossy()));
        if path2.exists() {
            info!("Loading mesh shader manifest: {}", path2.display());
            if let Some(manifest) = parse_manifest_json(&path2) {
                if !manifest.permutations.is_empty() {
                    return Some(manifest);
                }
            }
        }
    }

    None
}

/// Find the best matching permutation suffix for a set of material features.
/// Returns "" (base) if no exact match found.
fn find_permutation_suffix(manifest: &ShaderManifest, features: &std::collections::BTreeSet<String>) -> String {
    let mut wanted: HashMap<String, bool> = HashMap::new();
    for fname in &manifest.features {
        wanted.insert(fname.clone(), features.contains(fname));
    }
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
    String::new()
}

// ===========================================================================
// Multi-pipeline data structures
// ===========================================================================

/// A group of meshlets that share a common material.
/// Built by partitioning meshlets per draw range so each group maps to a
/// specific material (and therefore a specific shader permutation).
struct MeshletGroup {
    first_meshlet: u32,     // offset into global meshlet descriptor array
    meshlet_count: u32,     // how many meshlets in this group
    material_index: usize,  // scene material index
    permutation_index: usize, // index into mesh_permutations
}

/// Per-permutation pipeline data for mesh shader multi-material rendering.
/// The mesh shader (.mesh.spv) is SHARED across all permutations (it only reads
/// geometry data), while the fragment shader (.frag.spv) differs per permutation
/// (different BRDF features, texture bindings).
struct MeshPermutationPipeline {
    #[allow(dead_code)]
    suffix: String,
    frag_module: vk::ShaderModule,
    #[allow(dead_code)]
    frag_refl: reflected_pipeline::ReflectionData,
    material_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    per_material_desc_sets: Vec<vk::DescriptorSet>,
    per_material_ubos: Vec<GpuBuffer>,
    material_indices: Vec<usize>,
}

/// Mesh shader renderer state (persistent, suitable for interactive multi-frame rendering).
pub struct MeshShaderRenderer {
    // Pipeline (single-pipeline mode)
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    ds_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    // Shader modules
    mesh_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,

    // Render target (offscreen)
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    color_image: GpuImage,
    depth_image: GpuImage,
    width: u32,
    height: u32,

    // Meshlet data
    meshlet_count: u32,

    // SoA storage buffers (meshlet descriptors, vertices, triangles, positions, normals, tex_coords, tangents)
    meshlet_desc_buffer: GpuBuffer,
    meshlet_verts_buffer: GpuBuffer,
    meshlet_tris_buffer: GpuBuffer,
    positions_buffer: GpuBuffer,
    normals_buffer: GpuBuffer,
    tex_coords_buffer: GpuBuffer,
    tangents_buffer: GpuBuffer,

    // Buffer sizes for descriptor writes (needed by multi-pipeline setup)
    meshlet_desc_size: u64,
    meshlet_verts_size: u64,
    meshlet_tris_size: u64,
    positions_size: u64,
    normals_size: u64,
    tex_coords_size: u64,
    tangents_size: u64,

    // Uniforms
    mvp_buffer: GpuBuffer,
    light_buffer: GpuBuffer,
    material_buffer: GpuBuffer,

    // Textures
    sampler: vk::Sampler,
    default_texture: GpuImage,
    default_black_texture: GpuImage,
    default_normal_texture: GpuImage,
    texture_images: HashMap<String, GpuImage>,
    per_material_textures: Vec<HashMap<String, GpuImage>>,
    ibl_assets: IblAssets,

    // Multi-pipeline mode
    multi_pipeline: bool,
    mesh_permutations: Vec<MeshPermutationPipeline>,
    material_to_permutation: Vec<usize>,
    shared_set0_layout: vk::DescriptorSetLayout,
    shared_set0: vk::DescriptorSet,
    meshlet_groups: Vec<MeshletGroup>,

    // Multi-light support (P17.2)
    has_multi_light: bool,
    lights_ssbo: Option<GpuBuffer>,
    scene_light_ubo: Option<GpuBuffer>,

    // Bindless mode (single pipeline, materials SSBO + texture array)
    bindless_mode: bool,
    bindless_materials_ssbo: Option<scene_manager::BindlessMaterialsSSBO>,

    // Cached reflection for multi-pipeline setup
    mesh_refl: Option<reflected_pipeline::ReflectionData>,

    // Auto-camera
    pub auto_eye: glam::Vec3,
    pub auto_target: glam::Vec3,
    pub auto_up: glam::Vec3,
    pub auto_far: f32,
    pub has_scene_bounds: bool,
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
    scene_manager::create_offscreen_image(
        device, allocator, width, height, format, usage, aspect, name,
    )
}

/// Delegate to scene_manager::create_texture_image with FRAGMENT_SHADER dst_stage.
fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
) -> Result<GpuImage, String> {
    scene_manager::create_texture_image(
        ctx,
        width,
        height,
        pixels,
        name,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    )
}

/// Create a 1x1 white RGBA default texture image.
fn create_default_white_texture(ctx: &mut VulkanContext) -> Result<GpuImage, String> {
    let white_pixel: [u8; 4] = [255, 255, 255, 255];
    create_texture_image(ctx, 1, 1, &white_pixel, "default_white_1x1")
}

/// Delegate to scene_manager::load_ibl_assets with FRAGMENT_SHADER dst_stage.
fn load_ibl_assets(ctx: &mut VulkanContext, ibl_name: &str) -> IblAssets {
    scene_manager::load_ibl_assets(ctx, vk::PipelineStageFlags::FRAGMENT_SHADER, ibl_name)
}

impl MeshShaderRenderer {
    /// Create a new mesh shader renderer.
    ///
    /// Loads reflection, builds meshlets from scene geometry, uploads storage buffers,
    /// creates render pass/framebuffer/pipeline, and writes descriptors.
    ///
    /// Supports both single-pipeline mode (original) and multi-pipeline mode
    /// (when a .manifest.json exists AND the scene has multiple materials AND
    /// the mesh shader declares a meshletOffset push constant).
    pub fn new(
        ctx: &mut VulkanContext,
        scene_source: &str,
        pipeline_base: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
        demo_lights: bool,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();

        // --- Phase 0: Load scene geometry (needed for permutation resolution) ---
        let is_gltf = scene_source.ends_with(".glb") || scene_source.ends_with(".gltf");

        let mut gltf_scene = if is_gltf {
            Some(crate::gltf_loader::load_gltf(Path::new(scene_source))?)
        } else {
            None
        };

        // Resolve single-material permutation from manifest + scene features
        let resolved_frag_base = if let Some(ref gs) = gltf_scene {
            if let Some(manifest) = try_load_manifest(pipeline_base) {
                let demo_light_vec = if demo_lights { crate::setup_demo_lights() } else { vec![] };
                let scene_features = scene_manager::detect_scene_features(gs, &demo_light_vec);
                let suffix = find_permutation_suffix(&manifest, &scene_features);
                if !suffix.is_empty() {
                    let candidate = format!("{}{}", pipeline_base, suffix);
                    if Path::new(&format!("{}.frag.spv", candidate)).exists() {
                        info!("Resolved mesh single-material permutation: {}", suffix);
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
        };

        // --- Phase 1: Load reflection JSON ---
        // Both mesh and fragment shaders use resolved permutation (mesh shader
        // varyings must match fragment shader inputs — permutation mesh outputs
        // tangent/bitangent when normal_map is enabled)
        let mesh_json_path = format!("{}.mesh.json", resolved_frag_base);
        let frag_json_path = format!("{}.frag.json", resolved_frag_base);

        let mesh_refl = reflected_pipeline::load_reflection(Path::new(&mesh_json_path))?;
        let frag_refl = reflected_pipeline::load_reflection(Path::new(&frag_json_path))?;

        info!(
            "Mesh shader reflection loaded: mesh={} frag={} descriptor_sets: mesh={:?}, frag={:?}",
            mesh_json_path,
            frag_json_path,
            mesh_refl.descriptor_sets.keys().collect::<Vec<_>>(),
            frag_refl.descriptor_sets.keys().collect::<Vec<_>>(),
        );

        // Get mesh output info from reflection (for meshlet building limits)
        let mesh_output = mesh_refl.mesh_output.as_ref();
        let max_verts = mesh_output.map(|m| m.max_vertices).unwrap_or(64);
        let max_prims = mesh_output.map(|m| m.max_primitives).unwrap_or(124);
        info!(
            "Mesh output limits from reflection: max_vertices={}, max_primitives={}",
            max_verts, max_prims
        );

        // --- Bindless detection: check if the fragment reflection has bindless enabled ---
        let use_bindless = frag_refl.bindless.as_ref().map_or(false, |b| b.enabled) && ctx.bindless_supported;
        if use_bindless {
            info!("Mesh shader bindless mode detected: redirecting to new_bindless");
            return Self::new_bindless(ctx, scene_source, pipeline_base, width, height, ibl_name,
                                      mesh_refl, frag_refl, gltf_scene, demo_lights);
        }

        // Check for manifest to decide multi-pipeline mode
        let manifest = try_load_manifest(pipeline_base);
        let has_multiple_materials = gltf_scene.as_ref().map(|s| s.materials.len() > 1).unwrap_or(false);

        // Check if the mesh shader declares a meshletOffset push constant
        let has_meshlet_offset_pc = mesh_refl.push_constants.iter().any(|pc| {
            pc.fields.iter().any(|f| f.name == "meshletOffset")
        });

        let use_multi_pipeline = if manifest.is_some() && has_multiple_materials {
            if has_meshlet_offset_pc {
                info!(
                    "Multi-pipeline mode enabled: manifest found, {} materials, meshletOffset push constant present",
                    gltf_scene.as_ref().map(|s| s.materials.len()).unwrap_or(0)
                );
                true
            } else {
                info!(
                    "WARNING: Manifest found with {} material(s), but mesh shader multi-pipeline \
                     requires meshletOffset push constant support. Falling back to single pipeline.",
                    gltf_scene.as_ref().map(|s| s.materials.len()).unwrap_or(0)
                );
                false
            }
        } else {
            false
        };

        let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
        let mut auto_target = glam::Vec3::ZERO;
        let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
        let mut auto_far = 100.0f32;
        let mut has_scene_bounds = false;

        let mut tangent_data_raw: Vec<[f32; 4]> = Vec::new();
        let mut draw_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();

        let (vertices, indices): (Vec<scene::PbrVertex>, Vec<u32>) =
            if let Some(ref mut gltf_s) = gltf_scene {
                if gltf_s.meshes.is_empty() {
                    return Err(format!("No meshes found in glTF file: {}", scene_source));
                }
                let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
                if draw_items.is_empty() {
                    return Err(format!("No draw items in glTF scene: {}", scene_source));
                }

                let mut all_verts: Vec<scene::PbrVertex> = Vec::new();
                let mut all_indices: Vec<u32> = Vec::new();
                let mut vertex_offset: u32 = 0;
                let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
                let mut index_offset: u32 = 0;

                for item in &draw_items {
                    let mesh = &gltf_s.meshes[item.mesh_index];
                    let world = item.world_transform;
                    let normal_mat =
                        glam::Mat3::from_mat4(world).inverse().transpose();

                    for v in &mesh.vertices {
                        let pos =
                            world.transform_point3(glam::Vec3::from(v.position));
                        let norm =
                            (normal_mat * glam::Vec3::from(v.normal)).normalize();
                        vertex_positions.push([pos.x, pos.y, pos.z]);
                        all_verts.push(scene::PbrVertex {
                            position: pos.into(),
                            normal: norm.into(),
                            uv: v.uv,
                        });
                        if mesh.has_tangents {
                            let tan3 = (normal_mat * glam::Vec3::from_slice(&v.tangent[..3])).normalize();
                            tangent_data_raw.push([tan3.x, tan3.y, tan3.z, v.tangent[3]]);
                        } else {
                            tangent_data_raw.push([1.0, 0.0, 0.0, 1.0]);
                        }
                    }
                    for &idx in &mesh.indices {
                        all_indices.push(idx + vertex_offset);
                    }

                    // Track draw ranges per draw item (for per-material meshlet building)
                    draw_ranges.push(crate::gltf_loader::DrawRange {
                        index_offset,
                        index_count: mesh.indices.len() as u32,
                        material_index: item.material_index,
                    });
                    index_offset += mesh.indices.len() as u32;

                    vertex_offset += mesh.vertices.len() as u32;
                }

                // Compute auto-camera
                has_scene_bounds = true;
                let (eye, target, up, far) =
                    scene_manager::compute_auto_camera_from_draw_items(
                        &vertex_positions,
                        &draw_items,
                    );
                auto_eye = eye;
                auto_target = target;
                auto_up = up;
                auto_far = far;

                (all_verts, all_indices)
            } else {
                scene::generate_sphere(32, 32)
            };
        if tangent_data_raw.is_empty() {
            tangent_data_raw = vec![[1.0, 0.0, 0.0, 1.0]; vertices.len()];
        }

        // --- Phase 2: Build meshlets ---
        // Decide whether to build per-draw-range (multi-material) or full-buffer (single)
        let has_multiple_draw_ranges = draw_ranges.len() > 1;
        let mut meshlet_groups: Vec<MeshletGroup> = Vec::new();

        let (meshlet_descs, meshlet_vertices_vec, meshlet_triangles_vec) =
            if has_multiple_draw_ranges && use_multi_pipeline {
                // Multi-material path: build meshlets per draw range so each group
                // is associated with a specific material for permutation selection.
                let mut all_meshlets: Vec<meshlet::MeshletDescriptor> = Vec::new();
                let mut all_meshlet_vertices: Vec<u32> = Vec::new();
                let mut all_meshlet_triangles: Vec<u32> = Vec::new();

                for range in &draw_ranges {
                    let range_index_count = range.index_count;
                    if range_index_count == 0 {
                        continue;
                    }

                    // Build meshlets from this draw range's index slice
                    let range_start = range.index_offset as usize;
                    let range_end = range_start + range_index_count as usize;
                    let range_indices = &indices[range_start..range_end];

                    let build_result = meshlet::build_meshlets(
                        range_indices,
                        vertices.len() as u32,
                        max_verts,
                        max_prims,
                    );

                    if build_result.meshlets.is_empty() {
                        continue;
                    }

                    // Record the meshlet group before concatenation
                    let group = MeshletGroup {
                        first_meshlet: all_meshlets.len() as u32,
                        meshlet_count: build_result.meshlets.len() as u32,
                        material_index: range.material_index,
                        permutation_index: 0, // set later during multi-pipeline setup
                    };

                    // Offset meshlet descriptors to account for global vertex/triangle arrays
                    let vertex_base = all_meshlet_vertices.len() as u32;
                    let triangle_base = all_meshlet_triangles.len() as u32;
                    let mut offset_meshlets = build_result.meshlets;
                    for m in &mut offset_meshlets {
                        m.vertex_offset += vertex_base;
                        m.triangle_offset += triangle_base;
                    }

                    // Concatenate into global arrays
                    all_meshlets.extend_from_slice(&offset_meshlets);
                    all_meshlet_vertices.extend_from_slice(&build_result.meshlet_vertices);
                    all_meshlet_triangles.extend_from_slice(&build_result.meshlet_triangles);

                    meshlet_groups.push(group);
                }

                info!(
                    "Built {} meshlets across {} material group(s) (max_verts={}, max_prims={})",
                    all_meshlets.len(),
                    meshlet_groups.len(),
                    max_verts,
                    max_prims
                );

                (all_meshlets, all_meshlet_vertices, all_meshlet_triangles)
            } else {
                // Single draw range (original path): build meshlets from the full index buffer
                let meshlet_result = meshlet::build_meshlets(
                    &indices,
                    vertices.len() as u32,
                    max_verts,
                    max_prims,
                );

                info!(
                    "Built {} meshlets from {} triangles ({} vertices)",
                    meshlet_result.meshlets.len(),
                    indices.len() / 3,
                    vertices.len()
                );

                (
                    meshlet_result.meshlets,
                    meshlet_result.meshlet_vertices,
                    meshlet_result.meshlet_triangles,
                )
            };

        let meshlet_count = meshlet_descs.len() as u32;

        // --- Phase 3: Upload storage buffers ---
        let meshlet_desc_size = meshlet_descs.len() as u64 * 16;
        let meshlet_desc_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_descs),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_descriptors",
        )?;

        let meshlet_verts_size = if meshlet_vertices_vec.is_empty() { 4u64 } else { meshlet_vertices_vec.len() as u64 * 4 };
        let meshlet_verts_data: Vec<u32> = if meshlet_vertices_vec.is_empty() { vec![0u32] } else { meshlet_vertices_vec };
        let meshlet_verts_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_verts_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_vertices",
        )?;

        let meshlet_tris_size = if meshlet_triangles_vec.is_empty() { 4u64 } else { meshlet_triangles_vec.len() as u64 * 4 };
        let meshlet_tris_data: Vec<u32> = if meshlet_triangles_vec.is_empty() { vec![0u32] } else { meshlet_triangles_vec };
        let meshlet_tris_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&meshlet_tris_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_meshlet_triangles",
        )?;

        // SoA vertex attribute buffers (same pattern as rt_renderer)
        let positions_data: Vec<[f32; 4]> = vertices
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2], 1.0])
            .collect();
        let normals_data: Vec<[f32; 4]> = vertices
            .iter()
            .map(|v| [v.normal[0], v.normal[1], v.normal[2], 0.0])
            .collect();
        let tex_coords_data: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv).collect();

        let positions_size = positions_data.len() as u64 * 16;
        let normals_size = normals_data.len() as u64 * 16;
        let tex_coords_size = tex_coords_data.len() as u64 * 8;

        let positions_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&positions_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_positions",
        )?;
        let normals_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&normals_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_normals",
        )?;
        let tex_coords_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&tex_coords_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_tex_coords",
        )?;

        let tangents_data: Vec<[f32; 4]> = tangent_data_raw;
        let tangents_size = tangents_data.len() as u64 * 16;
        let tangents_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            bytemuck::cast_slice(&tangents_data),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "mesh_tangents",
        )?;

        info!(
            "Created mesh SoA storage buffers: positions={}B, normals={}B, texCoords={}B, tangents={}B, meshlets={}B",
            positions_size, normals_size, tex_coords_size, tangents_size, meshlet_desc_size
        );

        // --- Phase 4: Uniform buffers ---
        let aspect = width as f32 / height as f32;
        let (model, view, proj) = if has_scene_bounds {
            (
                glam::Mat4::IDENTITY,
                crate::camera::look_at(auto_eye, auto_target, auto_up),
                crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far),
            )
        } else {
            (
                DefaultCamera::model(),
                DefaultCamera::view(),
                DefaultCamera::projection(aspect),
            )
        };

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        let mvp_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            &mvp_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "mesh_mvp",
        )?;

        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let camera_pos = if has_scene_bounds {
            auto_eye
        } else {
            DefaultCamera::EYE
        };
        let mut light_data = [0u8; 32];
        light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
        light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
        light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

        let light_buffer = create_buffer_with_data(
            &device,
            ctx.allocator_mut(),
            &light_data,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "mesh_light",
        )?;

        // --- Multi-light detection and buffer creation (P17.2) ---
        // Scan the FRAGMENT reflection for a "lights" storage_buffer binding
        let frag_merged_for_detect = reflected_pipeline::merge_descriptor_sets(&[&frag_refl]);
        let has_multi_light = frag_merged_for_detect.values().any(|bindings| {
            bindings.iter().any(|b| b.name == "lights" && b.binding_type == "storage_buffer")
        });

        let mut sm = scene_manager::SceneManager::new();
        if demo_lights {
            sm.override_lights(crate::setup_demo_lights());
        } else if let Some(ref gs) = gltf_scene {
            sm.populate_lights_from_gltf(&gs.lights);
        } else {
            sm.populate_lights_from_gltf(&[]);
        }

        let mut lights_ssbo: Option<GpuBuffer> = None;
        if has_multi_light {
            let light_floats = sm.pack_lights_buffer();
            let data: &[u8] = if light_floats.is_empty() {
                &[0u8; 64]
            } else {
                bytemuck::cast_slice(&light_floats)
            };
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), data,
                vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_lights_ssbo",
            )?;
            info!("Mesh multi-light SSBO: {} lights, {} bytes", sm.lights.len(), data.len());
            lights_ssbo = Some(buf);
        }

        // SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
        let mut scene_light_ubo: Option<GpuBuffer> = None;
        if has_multi_light {
            let mut scene_light_data = [0u8; 16];
            scene_light_data[0..12].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
            let light_count = sm.lights.len() as i32;
            scene_light_data[12..16].copy_from_slice(&light_count.to_le_bytes());
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), &scene_light_data,
                vk::BufferUsageFlags::UNIFORM_BUFFER, "mesh_scene_light_ubo",
            )?;
            scene_light_ubo = Some(buf);
        }

        // Material uniform buffer
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
            &device,
            ctx.allocator_mut(),
            bytemuck::bytes_of(&material_data),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "mesh_material",
        )?;

        // --- Phase 5: Textures ---
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
        let default_black_texture =
            create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "default_black")?;
        let default_normal_texture =
            create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "default_normal")?;

        // Build per-material texture maps (for multi-pipeline mode)
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
                        let label = format!("mesh_{}[{}]", name, mat_idx);
                        let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                        tex_map.insert(name.to_string(), gpu_img);
                    }
                }

                if mat.base_color_image.is_some() && !tex_map.contains_key("albedo_tex") {
                    if let Some(tex_data) = mat.base_color_image.as_ref() {
                        let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &format!("mesh_albedo_tex[{}]", mat_idx))?;
                        tex_map.insert("albedo_tex".to_string(), gpu_img);
                    }
                }

                per_material_textures.push(tex_map);
            }
        }

        // Also build the single-material texture_images map (for single-pipeline fallback)
        let gltf_material = gltf_scene.as_ref().and_then(|s| {
            if !s.materials.is_empty() {
                Some(&s.materials[0])
            } else {
                None
            }
        });

        let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
        let texture_names = [
            "base_color_tex",
            "normal_tex",
            "metallic_roughness_tex",
            "occlusion_tex",
            "emissive_tex",
            "albedo_tex",
        ];
        for tex_name in &texture_names {
            let tex_image_opt = match *tex_name {
                "base_color_tex" | "albedo_tex" => {
                    gltf_material.and_then(|m| m.base_color_image.as_ref())
                }
                "normal_tex" => gltf_material.and_then(|m| m.normal_image.as_ref()),
                "metallic_roughness_tex" => {
                    gltf_material.and_then(|m| m.metallic_roughness_image.as_ref())
                }
                "occlusion_tex" => gltf_material.and_then(|m| m.occlusion_image.as_ref()),
                "emissive_tex" => gltf_material.and_then(|m| m.emissive_image.as_ref()),
                _ => None,
            };
            if let Some(tex_data) = tex_image_opt {
                let gpu_img = create_texture_image(
                    ctx,
                    tex_data.width,
                    tex_data.height,
                    &tex_data.pixels,
                    tex_name,
                )?;
                texture_images.insert(tex_name.to_string(), gpu_img);
            }
        }

        if gltf_material.is_none() {
            let tex_pixels = scene::generate_procedural_texture(512);
            let proc_texture =
                create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_albedo")?;
            texture_images.insert("albedo_tex".to_string(), proc_texture);
            let proc_texture2 =
                create_texture_image(ctx, 512, 512, &tex_pixels, "procedural_base_color")?;
            texture_images.insert("base_color_tex".to_string(), proc_texture2);
        }

        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // --- Phase 6: Render pass, framebuffer ---
        let color_image = create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
            "mesh_color",
        )?;

        let depth_image = create_offscreen_image(
            &device,
            ctx.allocator_mut(),
            width,
            height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            "mesh_depth",
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

        let mesh_dependencies = [
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
                        .dependencies(&mesh_dependencies),
                    None,
                )
                .map_err(|e| format!("Failed to create mesh shader render pass: {:?}", e))?
        };

        let fb_attachments = [color_image.view, depth_image.view];
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&fb_attachments)
                        .width(width)
                        .height(height)
                        .layers(1),
                    None,
                )
                .map_err(|e| format!("Failed to create mesh shader framebuffer: {:?}", e))?
        };

        // Load mesh shader module (use resolved permutation for single-pipeline,
        // as it must match the fragment shader's expected varyings)
        let mesh_path = format!("{}.mesh.spv", resolved_frag_base);
        let mesh_code = spv_loader::load_spirv(Path::new(&mesh_path))?;
        let mesh_module = spv_loader::create_shader_module(&device, &mesh_code)?;

        // ===================================================================
        // Branch: multi-pipeline vs single-pipeline
        // ===================================================================
        if use_multi_pipeline {
            // ---------------------------------------------------------------
            // MULTI-PIPELINE MODE
            // ---------------------------------------------------------------
            let gltf_s = gltf_scene.as_ref().unwrap();
            let groups = scene_manager::group_materials_by_features(gltf_s, &sm.lights);
            let total_materials = gltf_s.materials.len();

            let mut material_to_permutation: Vec<usize> = vec![0; total_materials];
            let mut mesh_permutations: Vec<MeshPermutationPipeline> = Vec::new();

            // Create shared set 0 layout from mesh shader reflection
            // (meshlet buffers + SoA + MVP + Light)
            let shared_set0_layout = {
                let set0_merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl]);
                let set0_layouts = unsafe {
                    reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &set0_merged)?
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

            // Set 0 bindings (shared) from mesh shader reflection
            let mesh_merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl]);
            if let Some(bindings) = mesh_merged.get(&0) {
                for b in bindings {
                    let vk_type = reflected_pipeline::binding_type_to_vk_public(&b.binding_type);
                    *type_counts.entry(vk_type).or_insert(0) += 1;
                }
            }

            // Pre-pass: load per-permutation reflections and count descriptors
            struct PermPrep {
                suffix: String,
                perm_base: String,
                material_indices: Vec<usize>,
                frag_refl: reflected_pipeline::ReflectionData,
                perm_merged: HashMap<u32, Vec<reflected_pipeline::BindingInfo>>,
                frag_set_idx: u32,
            }
            let mut perm_preps: Vec<PermPrep> = Vec::new();

            for (suffix, material_indices) in &groups {
                let mut resolved_suffix = suffix.clone();
                let mut perm_base = format!("{}{}", pipeline_base, resolved_suffix);
                let perm_frag_path = format!("{}.frag.spv", perm_base);

                if !Path::new(&perm_frag_path).exists() {
                    info!("Missing fragment shader for mesh permutation '{}', falling back to base", resolved_suffix);
                    perm_base = pipeline_base.to_string();
                    resolved_suffix = String::new();
                }

                let perm_frag_json = format!("{}.frag.json", perm_base);
                let p_frag_refl = reflected_pipeline::load_reflection(Path::new(&perm_frag_json))?;
                let p_merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl, &p_frag_refl]);

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
                        let vk_type = reflected_pipeline::binding_type_to_vk_public(&b.binding_type);
                        *type_counts.entry(vk_type).or_insert(0) += material_indices.len() as u32;
                    }
                }
                max_sets += material_indices.len() as u32;

                perm_preps.push(PermPrep {
                    suffix: resolved_suffix,
                    perm_base,
                    material_indices: material_indices.clone(),
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
                    .map_err(|e| format!("Failed to create mesh multi-pipeline descriptor pool: {:?}", e))?
            };

            // Allocate and write shared set 0 (meshlet data + SoA buffers + MVP + Light)
            let shared_set0 = if shared_set0_layout != vk::DescriptorSetLayout::null() {
                let info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(std::slice::from_ref(&shared_set0_layout));
                unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed to alloc shared set 0: {:?}", e))?[0] }
            } else {
                vk::DescriptorSet::null()
            };

            // Write set 0: meshlet SSBOs + SoA buffers + MVP + Light
            {
                let set0_bindings = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl]);
                let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                let ssbo_map: HashMap<&str, (vk::Buffer, u64)> = [
                    ("meshlet_descriptors", (meshlet_desc_buffer.buffer, meshlet_desc_size)),
                    ("meshlet_vertices", (meshlet_verts_buffer.buffer, meshlet_verts_size)),
                    ("meshlet_triangles", (meshlet_tris_buffer.buffer, meshlet_tris_size)),
                    ("positions", (positions_buffer.buffer, positions_size)),
                    ("normals", (normals_buffer.buffer, normals_size)),
                    ("tex_coords", (tex_coords_buffer.buffer, tex_coords_size)),
                    ("tangents", (tangents_buffer.buffer, tangents_size)),
                ].into_iter().collect();

                if let Some(bindings) = set0_bindings.get(&0) {
                    for binding in bindings {
                        let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                        match vk_type {
                            vk::DescriptorType::UNIFORM_BUFFER => {
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
                                        .dst_set(shared_set0)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                );
                            }
                            vk::DescriptorType::STORAGE_BUFFER => {
                                // Check known SSBOs first, then multi-light
                                if let Some(&(buffer, size)) = ssbo_map.get(binding.name.as_str()) {
                                    let idx = buf_infos.len();
                                    buf_infos.push(vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(size));
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(shared_set0)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                    );
                                } else if binding.name == "lights" {
                                    if let Some(ref lights_buf) = lights_ssbo {
                                        let idx = buf_infos.len();
                                        let ssbo_size = (sm.lights.len() as u64) * 64;
                                        let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                        buf_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                        writes.push(
                                            vk::WriteDescriptorSet::default()
                                                .dst_set(shared_set0)
                                                .dst_binding(binding.binding)
                                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                                .buffer_info(unsafe { std::slice::from_raw_parts(&buf_infos[idx] as *const _, 1) }),
                                        );
                                    }
                                } else {
                                    info!("Unknown SSBO '{}' in set 0, skipping", binding.name);
                                    continue;
                                }
                            }
                            _ => continue,
                        }
                    }
                }
                if !writes.is_empty() {
                    unsafe { device.update_descriptor_sets(&writes, &[]); }
                }
            }

            // For each permutation: create per-material descriptor sets, UBOs, pipeline
            let mut perm_idx: usize = 0;
            for prep in perm_preps {
                // Load per-permutation fragment shader
                let perm_frag_path = format!("{}.frag.spv", prep.perm_base);
                let p_frag_code = spv_loader::load_spirv(Path::new(&perm_frag_path))?;
                let p_frag_module = spv_loader::create_shader_module(&device, &p_frag_code)?;

                // Create set 1 layout for this permutation
                let perm_layouts = unsafe {
                    reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &prep.perm_merged)?
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

                for &mi in &prep.material_indices {
                    // Create Material UBO for this material
                    let mat_data = if mi < gltf_s.materials.len() {
                        MaterialUboData::from_gltf_material(&gltf_s.materials[mi])
                    } else {
                        MaterialUboData::default_values()
                    };

                    let mat_buf = create_buffer_with_data(
                        &device, ctx.allocator_mut(),
                        bytemuck::bytes_of(&mat_data),
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        &format!("mesh_perm_{}_material_{}", perm_idx, mi),
                    )?;

                    // Allocate descriptor set
                    let ds = if material_set_layout != vk::DescriptorSetLayout::null() {
                        let info = vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(std::slice::from_ref(&material_set_layout));
                        unsafe { device.allocate_descriptor_sets(&info).map_err(|e| format!("Failed to alloc material DS: {:?}", e))?[0] }
                    } else {
                        vk::DescriptorSet::null()
                    };

                    // Write descriptors for set 1
                    {
                        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
                        let mut img_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
                        let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();

                        if let Some(bindings) = prep.perm_merged.get(&prep.frag_set_idx) {
                            for binding in bindings {
                                let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
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

                    perm_desc_sets.push(ds);
                    perm_ubos.push(mat_buf);
                }

                // Create pipeline layout: set 0 (shared) + set 1 (per-permutation material)
                let mut layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
                if shared_set0_layout != vk::DescriptorSetLayout::null() {
                    layouts.push(shared_set0_layout);
                }
                if material_set_layout != vk::DescriptorSetLayout::null() {
                    layouts.push(material_set_layout);
                }

                // Push constant ranges from mesh + fragment stages
                let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
                for pc in &mesh_refl.push_constants {
                    push_ranges.push(
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::MESH_EXT)
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
                        .map_err(|e| format!("Failed to create mesh permutation pipeline layout: {:?}", e))?
                };

                // Create graphics pipeline (shared mesh shader + permutation fragment shader)
                let entry_name = c"main";
                let shader_stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::MESH_EXT)
                        .module(mesh_module) // shared mesh shader
                        .name(entry_name),
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(p_frag_module)
                        .name(entry_name),
                ];

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
                    .cull_mode(vk::CullModeFlags::BACK)
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

                let perm_pipeline = unsafe {
                    device
                        .create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            &[vk::GraphicsPipelineCreateInfo::default()
                                .stages(&shader_stages)
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
                        .map_err(|e| format!("Failed to create mesh permutation pipeline '{}': {:?}", prep.suffix, e))?[0]
                };

                info!("Loaded mesh permutation '{}' from {} ({} material(s))",
                      prep.suffix, prep.perm_base, prep.material_indices.len());

                mesh_permutations.push(MeshPermutationPipeline {
                    suffix: prep.suffix,
                    frag_module: p_frag_module,
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

            // Assign permutation indices to meshlet groups
            for group in &mut meshlet_groups {
                if group.material_index < material_to_permutation.len() {
                    group.permutation_index = material_to_permutation[group.material_index];
                }
            }

            info!("Mesh multi-pipeline setup complete: {} permutation(s) for {} material(s)",
                  mesh_permutations.len(), total_materials);

            // Load base frag module (kept for cleanup symmetry)
            let frag_path = format!("{}.frag.spv", pipeline_base);
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let frag_module = spv_loader::create_shader_module(&device, &frag_code)?;

            Ok(MeshShaderRenderer {
                pipeline: vk::Pipeline::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_pool,
                ds_layouts: HashMap::new(),
                descriptor_sets: Vec::new(),
                mesh_module,
                frag_module,
                render_pass,
                framebuffer,
                color_image,
                depth_image,
                width,
                height,
                meshlet_count,
                meshlet_desc_buffer,
                meshlet_verts_buffer,
                meshlet_tris_buffer,
                positions_buffer,
                normals_buffer,
                tex_coords_buffer,
                tangents_buffer,
                meshlet_desc_size,
                meshlet_verts_size,
                meshlet_tris_size,
                positions_size,
                normals_size,
                tex_coords_size,
                tangents_size,
                mvp_buffer,
                light_buffer,
                material_buffer,
                sampler,
                default_texture,
                default_black_texture,
                default_normal_texture,
                texture_images,
                per_material_textures,
                ibl_assets,
                has_multi_light,
                lights_ssbo,
                scene_light_ubo,
                multi_pipeline: true,
                mesh_permutations,
                material_to_permutation,
                shared_set0_layout,
                shared_set0,
                meshlet_groups,
                bindless_mode: false,
                bindless_materials_ssbo: None,
                mesh_refl: None,
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

            // Merge descriptor sets from both stages
            let merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl, &frag_refl]);

            // Create descriptor set layouts
            let ds_layouts = unsafe {
                reflected_pipeline::create_descriptor_set_layouts_from_merged(&device, &merged)?
            };

            let (pool_sizes, max_sets) = reflected_pipeline::compute_pool_sizes(&merged);

            info!(
                "Mesh shader reflection-driven descriptors: {} sets, {} pool size entries",
                max_sets,
                pool_sizes.len()
            );

            // Create descriptor pool
            let descriptor_pool = unsafe {
                let pool_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&pool_sizes);
                device
                    .create_descriptor_pool(&pool_info, None)
                    .map_err(|e| format!("Failed to create mesh shader descriptor pool: {:?}", e))?
            };

            // Allocate descriptor sets
            let max_set_idx = ds_layouts.keys().copied().max().unwrap_or(0);
            let mut ordered_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
            for i in 0..=max_set_idx {
                if let Some(&layout) = ds_layouts.get(&i) {
                    ordered_layouts.push(layout);
                }
            }

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&ordered_layouts);

            let descriptor_sets = unsafe {
                device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| format!("Failed to allocate mesh shader descriptor sets: {:?}", e))?
            };

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

            // Pipeline layout (no push constants for mesh shader by default)
            let pipeline_layout = unsafe {
                reflected_pipeline::create_pipeline_layout_from_merged(&device, &ds_layouts, &[])?
            };

            // Build SSBO map by name -> (buffer, size)
            let ssbo_map: HashMap<String, (vk::Buffer, u64)> = [
                ("meshlet_descriptors".to_string(), (meshlet_desc_buffer.buffer, meshlet_desc_size)),
                ("meshlet_vertices".to_string(), (meshlet_verts_buffer.buffer, meshlet_verts_size)),
                ("meshlet_triangles".to_string(), (meshlet_tris_buffer.buffer, meshlet_tris_size)),
                ("positions".to_string(), (positions_buffer.buffer, positions_size)),
                ("normals".to_string(), (normals_buffer.buffer, normals_size)),
                ("tex_coords".to_string(), (tex_coords_buffer.buffer, tex_coords_size)),
                ("tangents".to_string(), (tangents_buffer.buffer, tangents_size)),
            ].into_iter().collect();

            // Write descriptors (same pattern as before)
            let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
            let total_bindings: usize = merged.values().map(|v| v.len()).sum();
            buffer_infos.reserve(total_bindings);
            image_infos.reserve(total_bindings);

            for (&set_idx, bindings) in &merged {
                let ds = match ds_map.get(&set_idx) {
                    Some(&ds) => ds,
                    None => continue,
                };
                for binding in bindings {
                    let vk_type =
                        reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                    match vk_type {
                        vk::DescriptorType::UNIFORM_BUFFER => {
                            let (buffer, range) = match binding.name.as_str() {
                                "MVP" => (mvp_buffer.buffer, binding.size as u64),
                                "Light" => (light_buffer.buffer, binding.size as u64),
                                "Material" => (material_buffer.buffer, std::mem::size_of::<MaterialUboData>() as u64),
                                "SceneLight" => {
                                    if let Some(ref sl_buf) = scene_light_ubo {
                                        (sl_buf.buffer, binding.size as u64)
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            };
                            let idx = buffer_infos.len();
                            buffer_infos.push(
                                vk::DescriptorBufferInfo::default()
                                    .buffer(buffer)
                                    .offset(0)
                                    .range(range),
                            );
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(unsafe {
                                        std::slice::from_raw_parts(
                                            &buffer_infos[idx] as *const _,
                                            1,
                                        )
                                    }),
                            );
                        }
                        vk::DescriptorType::STORAGE_BUFFER => {
                            // Check known geometry SSBOs first, then multi-light
                            if let Some(&(buf, sz)) = ssbo_map.get(&binding.name) {
                                let idx = buffer_infos.len();
                                buffer_infos.push(
                                    vk::DescriptorBufferInfo::default()
                                        .buffer(buf)
                                        .offset(0)
                                        .range(sz),
                                );
                                writes.push(
                                    vk::WriteDescriptorSet::default()
                                        .dst_set(ds)
                                        .dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(unsafe {
                                            std::slice::from_raw_parts(
                                                &buffer_infos[idx] as *const _,
                                                1,
                                            )
                                        }),
                                );
                            } else if binding.name == "lights" {
                                if let Some(ref lights_buf) = lights_ssbo {
                                    let idx = buffer_infos.len();
                                    let ssbo_size = (sm.lights.len() as u64) * 64;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                    buffer_infos.push(
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(lights_buf.buffer)
                                            .offset(0)
                                            .range(range),
                                    );
                                    writes.push(
                                        vk::WriteDescriptorSet::default()
                                            .dst_set(ds)
                                            .dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe {
                                                std::slice::from_raw_parts(
                                                    &buffer_infos[idx] as *const _,
                                                    1,
                                                )
                                            }),
                                    );
                                }
                            } else {
                                info!(
                                    "Unknown SSBO name '{}' at set={} binding={}, skipping",
                                    binding.name, set_idx, binding.binding
                                );
                                continue;
                            }
                        }
                        vk::DescriptorType::SAMPLER => {
                            let actual_sampler = ibl_assets
                                .sampler_for_binding(&binding.name)
                                .unwrap_or(sampler);
                            let idx = image_infos.len();
                            image_infos
                                .push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .image_info(unsafe {
                                        std::slice::from_raw_parts(
                                            &image_infos[idx] as *const _,
                                            1,
                                        )
                                    }),
                            );
                        }
                        vk::DescriptorType::SAMPLED_IMAGE => {
                            let is_cube =
                                reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                            let view = if is_cube {
                                ibl_assets
                                    .view_for_binding(&binding.name)
                                    .unwrap_or(default_texture.view)
                            } else if binding.name == "brdf_lut" {
                                ibl_assets
                                    .view_for_binding(&binding.name)
                                    .unwrap_or(default_texture.view)
                            } else {
                                let fallback = match binding.name.as_str() {
                                    "emissive_tex" => default_black_texture.view,
                                    "normal_tex" => default_normal_texture.view,
                                    _ => default_texture.view,
                                };
                                texture_images
                                    .get(&binding.name)
                                    .map(|img| img.view)
                                    .unwrap_or(fallback)
                            };
                            let idx = image_infos.len();
                            image_infos.push(
                                vk::DescriptorImageInfo::default()
                                    .image_view(view)
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                            );
                            writes.push(
                                vk::WriteDescriptorSet::default()
                                    .dst_set(ds)
                                    .dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(unsafe {
                                        std::slice::from_raw_parts(
                                            &image_infos[idx] as *const _,
                                            1,
                                        )
                                    }),
                            );
                        }
                        _ => {
                            info!(
                                "Unhandled descriptor type {:?} at set={} binding={} name='{}'",
                                vk_type, set_idx, binding.binding, binding.name
                            );
                        }
                    }
                }
            }

            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }

            // Load fragment shader module (using resolved permutation)
            let frag_path = format!("{}.frag.spv", resolved_frag_base);
            let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
            let frag_module = spv_loader::create_shader_module(&device, &frag_code)?;

            // Create graphics pipeline with mesh + fragment stages (no vertex input)
            let entry_name = c"main";
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::MESH_EXT)
                    .module(mesh_module)
                    .name(entry_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(entry_name),
            ];

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

            // Mesh shader pipelines do NOT have vertex_input_state or input_assembly_state.
            let pipeline = unsafe {
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[vk::GraphicsPipelineCreateInfo::default()
                            .stages(&shader_stages)
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
                    .map_err(|e| format!("Failed to create mesh shader pipeline: {:?}", e))?[0]
            };

            info!(
                "MeshShaderRenderer initialized (single-pipeline): {}x{}, {} meshlets",
                width, height, meshlet_count
            );

            Ok(MeshShaderRenderer {
                pipeline,
                pipeline_layout,
                descriptor_pool,
                ds_layouts,
                descriptor_sets,
                mesh_module,
                frag_module,
                render_pass,
                framebuffer,
                color_image,
                depth_image,
                width,
                height,
                meshlet_count,
                meshlet_desc_buffer,
                meshlet_verts_buffer,
                meshlet_tris_buffer,
                positions_buffer,
                normals_buffer,
                tex_coords_buffer,
                tangents_buffer,
                meshlet_desc_size,
                meshlet_verts_size,
                meshlet_tris_size,
                positions_size,
                normals_size,
                tex_coords_size,
                tangents_size,
                mvp_buffer,
                light_buffer,
                material_buffer,
                sampler,
                default_texture,
                default_black_texture,
                default_normal_texture,
                texture_images,
                per_material_textures,
                ibl_assets,
                has_multi_light,
                lights_ssbo,
                scene_light_ubo,
                multi_pipeline: false,
                mesh_permutations: Vec::new(),
                material_to_permutation: Vec::new(),
                shared_set0_layout: vk::DescriptorSetLayout::null(),
                shared_set0: vk::DescriptorSet::null(),
                meshlet_groups: Vec::new(),
                bindless_mode: false,
                bindless_materials_ssbo: None,
                mesh_refl: None,
                auto_eye,
                auto_target,
                auto_up,
                auto_far,
                has_scene_bounds,
            })
        }
    }

    /// Bindless constructor: single mesh+frag pipeline with materials SSBO + texture array.
    ///
    /// Uses `meshletOffset` and `material_index` push constants per dispatch to select
    /// the correct meshlet region and material in the bindless uber-shader.
    fn new_bindless(
        ctx: &mut VulkanContext,
        scene_source: &str,
        pipeline_base: &str,
        width: u32,
        height: u32,
        ibl_name: &str,
        mesh_refl: reflected_pipeline::ReflectionData,
        frag_refl: reflected_pipeline::ReflectionData,
        gltf_scene: Option<crate::gltf_loader::GltfScene>,
        demo_lights: bool,
    ) -> Result<Self, String> {
        let device = ctx.device.clone();

        let mesh_output = mesh_refl.mesh_output.as_ref();
        let max_verts = mesh_output.map(|m| m.max_vertices).unwrap_or(64);
        let max_prims = mesh_output.map(|m| m.max_primitives).unwrap_or(124);

        // --- Phase 1: Load scene geometry ---
        let mut auto_eye = glam::Vec3::new(0.0, 0.0, 3.0);
        let mut auto_target = glam::Vec3::ZERO;
        let mut auto_up = glam::Vec3::new(0.0, 1.0, 0.0);
        let mut auto_far = 100.0f32;
        let mut has_scene_bounds = false;

        let mut tangent_data_raw: Vec<[f32; 4]> = Vec::new();
        let mut draw_ranges: Vec<crate::gltf_loader::DrawRange> = Vec::new();

        let mut gltf_scene = gltf_scene;

        let (vertices, indices): (Vec<scene::PbrVertex>, Vec<u32>) =
            if let Some(ref mut gltf_s) = gltf_scene {
                if gltf_s.meshes.is_empty() {
                    return Err(format!("No meshes found in glTF file: {}", scene_source));
                }
                let draw_items = crate::gltf_loader::flatten_scene(gltf_s);
                if draw_items.is_empty() {
                    return Err(format!("No draw items in glTF scene: {}", scene_source));
                }

                let mut all_verts: Vec<scene::PbrVertex> = Vec::new();
                let mut all_indices: Vec<u32> = Vec::new();
                let mut vertex_offset: u32 = 0;
                let mut vertex_positions: Vec<[f32; 3]> = Vec::new();
                let mut index_offset: u32 = 0;

                for item in &draw_items {
                    let mesh = &gltf_s.meshes[item.mesh_index];
                    let world = item.world_transform;
                    let normal_mat = glam::Mat3::from_mat4(world).inverse().transpose();

                    for v in &mesh.vertices {
                        let pos = world.transform_point3(glam::Vec3::from(v.position));
                        let norm = (normal_mat * glam::Vec3::from(v.normal)).normalize();
                        vertex_positions.push([pos.x, pos.y, pos.z]);
                        all_verts.push(scene::PbrVertex {
                            position: pos.into(),
                            normal: norm.into(),
                            uv: v.uv,
                        });
                        if mesh.has_tangents {
                            let tan3 = (normal_mat * glam::Vec3::from_slice(&v.tangent[..3])).normalize();
                            tangent_data_raw.push([tan3.x, tan3.y, tan3.z, v.tangent[3]]);
                        } else {
                            tangent_data_raw.push([1.0, 0.0, 0.0, 1.0]);
                        }
                    }
                    for &idx in &mesh.indices {
                        all_indices.push(idx + vertex_offset);
                    }

                    draw_ranges.push(crate::gltf_loader::DrawRange {
                        index_offset,
                        index_count: mesh.indices.len() as u32,
                        material_index: item.material_index,
                    });
                    index_offset += mesh.indices.len() as u32;
                    vertex_offset += mesh.vertices.len() as u32;
                }

                has_scene_bounds = true;
                let (eye, target, up, far) = scene_manager::compute_auto_camera_from_draw_items(&vertex_positions, &draw_items);
                auto_eye = eye;
                auto_target = target;
                auto_up = up;
                auto_far = far;

                (all_verts, all_indices)
            } else {
                scene::generate_sphere(32, 32)
            };

        if tangent_data_raw.is_empty() {
            tangent_data_raw = vec![[1.0, 0.0, 0.0, 1.0]; vertices.len()];
        }

        // --- Phase 2: Build meshlets per draw range ---
        let has_multiple_draw_ranges = draw_ranges.len() > 1;
        let mut meshlet_groups: Vec<MeshletGroup> = Vec::new();

        let (meshlet_descs, meshlet_vertices_vec, meshlet_triangles_vec) =
            if has_multiple_draw_ranges {
                let mut all_meshlets: Vec<crate::meshlet::MeshletDescriptor> = Vec::new();
                let mut all_meshlet_vertices: Vec<u32> = Vec::new();
                let mut all_meshlet_triangles: Vec<u32> = Vec::new();

                for range in &draw_ranges {
                    if range.index_count == 0 { continue; }
                    let range_start = range.index_offset as usize;
                    let range_end = range_start + range.index_count as usize;
                    let range_indices = &indices[range_start..range_end];

                    let build_result = meshlet::build_meshlets(range_indices, vertices.len() as u32, max_verts, max_prims);
                    if build_result.meshlets.is_empty() { continue; }

                    let group = MeshletGroup {
                        first_meshlet: all_meshlets.len() as u32,
                        meshlet_count: build_result.meshlets.len() as u32,
                        material_index: range.material_index,
                        permutation_index: 0,
                    };

                    let vertex_base = all_meshlet_vertices.len() as u32;
                    let triangle_base = all_meshlet_triangles.len() as u32;
                    let mut offset_meshlets = build_result.meshlets;
                    for m in &mut offset_meshlets {
                        m.vertex_offset += vertex_base;
                        m.triangle_offset += triangle_base;
                    }

                    all_meshlets.extend_from_slice(&offset_meshlets);
                    all_meshlet_vertices.extend_from_slice(&build_result.meshlet_vertices);
                    all_meshlet_triangles.extend_from_slice(&build_result.meshlet_triangles);
                    meshlet_groups.push(group);
                }

                info!("Bindless mesh: built {} meshlets across {} material group(s)", all_meshlets.len(), meshlet_groups.len());
                (all_meshlets, all_meshlet_vertices, all_meshlet_triangles)
            } else {
                let meshlet_result = meshlet::build_meshlets(&indices, vertices.len() as u32, max_verts, max_prims);
                // Single group covering all meshlets
                meshlet_groups.push(MeshletGroup {
                    first_meshlet: 0,
                    meshlet_count: meshlet_result.meshlets.len() as u32,
                    material_index: 0,
                    permutation_index: 0,
                });
                info!("Bindless mesh: built {} meshlets from {} triangles", meshlet_result.meshlets.len(), indices.len() / 3);
                (meshlet_result.meshlets, meshlet_result.meshlet_vertices, meshlet_result.meshlet_triangles)
            };

        let meshlet_count = meshlet_descs.len() as u32;

        // --- Phase 3: Upload storage buffers ---
        let meshlet_desc_size = meshlet_descs.len() as u64 * 16;
        let meshlet_desc_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&meshlet_descs), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_meshlet_descriptors")?;

        let meshlet_verts_size = if meshlet_vertices_vec.is_empty() { 4u64 } else { meshlet_vertices_vec.len() as u64 * 4 };
        let meshlet_verts_data: Vec<u32> = if meshlet_vertices_vec.is_empty() { vec![0u32] } else { meshlet_vertices_vec };
        let meshlet_verts_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&meshlet_verts_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_meshlet_vertices")?;

        let meshlet_tris_size = if meshlet_triangles_vec.is_empty() { 4u64 } else { meshlet_triangles_vec.len() as u64 * 4 };
        let meshlet_tris_data: Vec<u32> = if meshlet_triangles_vec.is_empty() { vec![0u32] } else { meshlet_triangles_vec };
        let meshlet_tris_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&meshlet_tris_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_meshlet_triangles")?;

        let positions_data: Vec<[f32; 4]> = vertices.iter().map(|v| [v.position[0], v.position[1], v.position[2], 1.0]).collect();
        let normals_data: Vec<[f32; 4]> = vertices.iter().map(|v| [v.normal[0], v.normal[1], v.normal[2], 0.0]).collect();
        let tex_coords_data: Vec<[f32; 2]> = vertices.iter().map(|v| v.uv).collect();

        let positions_size = positions_data.len() as u64 * 16;
        let normals_size = normals_data.len() as u64 * 16;
        let tex_coords_size = tex_coords_data.len() as u64 * 8;

        let positions_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&positions_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_positions")?;
        let normals_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&normals_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_normals")?;
        let tex_coords_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&tex_coords_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_tex_coords")?;

        let tangents_data: Vec<[f32; 4]> = tangent_data_raw;
        let tangents_size = tangents_data.len() as u64 * 16;
        let tangents_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::cast_slice(&tangents_data), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_tangents")?;

        // --- Phase 4: Uniform buffers ---
        let aspect = width as f32 / height as f32;
        let (model, view, proj) = if has_scene_bounds {
            (glam::Mat4::IDENTITY, crate::camera::look_at(auto_eye, auto_target, auto_up), crate::camera::perspective(45.0f32.to_radians(), aspect, 0.1, auto_far))
        } else {
            (DefaultCamera::model(), DefaultCamera::view(), DefaultCamera::projection(aspect))
        };

        let mut mvp_data = [0u8; 192];
        mvp_data[0..64].copy_from_slice(bytemuck::cast_slice(model.as_ref()));
        mvp_data[64..128].copy_from_slice(bytemuck::cast_slice(view.as_ref()));
        mvp_data[128..192].copy_from_slice(bytemuck::cast_slice(proj.as_ref()));

        let mvp_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), &mvp_data, vk::BufferUsageFlags::UNIFORM_BUFFER, "mesh_bl_mvp")?;

        let light_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let camera_pos = if has_scene_bounds { auto_eye } else { DefaultCamera::EYE };
        let mut light_data = [0u8; 32];
        light_data[0..12].copy_from_slice(bytemuck::cast_slice(light_dir.as_ref()));
        light_data[12..16].copy_from_slice(&0.0f32.to_le_bytes());
        light_data[16..28].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
        light_data[28..32].copy_from_slice(&0.0f32.to_le_bytes());

        let light_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), &light_data, vk::BufferUsageFlags::UNIFORM_BUFFER, "mesh_bl_light")?;

        // --- Multi-light detection and buffer creation (P17.2) ---
        let frag_merged_for_detect = reflected_pipeline::merge_descriptor_sets(&[&frag_refl]);
        let has_multi_light = frag_merged_for_detect.values().any(|bindings| {
            bindings.iter().any(|b| b.name == "lights" && b.binding_type == "storage_buffer")
        });

        let mut sm = scene_manager::SceneManager::new();
        if demo_lights {
            sm.override_lights(crate::setup_demo_lights());
        } else if let Some(ref gs) = gltf_scene {
            sm.populate_lights_from_gltf(&gs.lights);
        } else {
            sm.populate_lights_from_gltf(&[]);
        }

        let mut lights_ssbo: Option<GpuBuffer> = None;
        if has_multi_light {
            let light_floats = sm.pack_lights_buffer();
            let data: &[u8] = if light_floats.is_empty() {
                &[0u8; 64]
            } else {
                bytemuck::cast_slice(&light_floats)
            };
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), data,
                vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_lights_ssbo",
            )?;
            info!("Mesh bindless multi-light SSBO: {} lights, {} bytes", sm.lights.len(), data.len());
            lights_ssbo = Some(buf);
        }

        // SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
        let mut scene_light_ubo: Option<GpuBuffer> = None;
        if has_multi_light {
            let mut scene_light_data = [0u8; 16];
            scene_light_data[0..12].copy_from_slice(bytemuck::cast_slice(camera_pos.as_ref()));
            let light_count = sm.lights.len() as i32;
            scene_light_data[12..16].copy_from_slice(&light_count.to_le_bytes());
            let buf = create_buffer_with_data(
                &device, ctx.allocator_mut(), &scene_light_data,
                vk::BufferUsageFlags::UNIFORM_BUFFER, "mesh_bl_scene_light_ubo",
            )?;
            scene_light_ubo = Some(buf);
        }

        // Material UBO (for legacy single-material fallback fields)
        let material_data = if let Some(ref gs) = gltf_scene {
            if !gs.materials.is_empty() { MaterialUboData::from_gltf_material(&gs.materials[0]) }
            else { MaterialUboData::default_values() }
        } else { MaterialUboData::default_values() };

        let material_buffer = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::bytes_of(&material_data), vk::BufferUsageFlags::UNIFORM_BUFFER, "mesh_bl_material")?;

        // --- Phase 5: Textures ---
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT),
                None,
            ).map_err(|e| format!("Failed to create sampler: {:?}", e))?
        };

        let default_texture = create_default_white_texture(ctx)?;
        let default_black_texture = create_texture_image(ctx, 1, 1, &[0u8, 0, 0, 255], "bl_default_black")?;
        let default_normal_texture = create_texture_image(ctx, 1, 1, &[128u8, 128, 255, 255], "bl_default_normal")?;

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
                        let label = format!("mesh_bl_{}[{}]", name, mat_idx);
                        let gpu_img = create_texture_image(ctx, tex_data.width, tex_data.height, &tex_data.pixels, &label)?;
                        tex_map.insert(name.to_string(), gpu_img);
                    }
                }
                per_material_textures.push(tex_map);
            }
        }

        // Single-material fallback texture_images map
        let mut texture_images: HashMap<String, GpuImage> = HashMap::new();
        if gltf_scene.is_none() {
            let tex_pixels = scene::generate_procedural_texture(512);
            let proc_texture = create_texture_image(ctx, 512, 512, &tex_pixels, "bl_procedural_albedo")?;
            texture_images.insert("base_color_tex".to_string(), proc_texture);
        }

        let ibl_assets = load_ibl_assets(ctx, ibl_name);

        // --- Phase 6: Build bindless texture array + materials SSBO ---
        let (bindless_tex_array, dedup_map) = scene_manager::build_bindless_texture_array_from_gpu_images(
            &per_material_textures, sampler,
            default_texture.view, default_black_texture.view, default_normal_texture.view,
        );

        info!("Mesh bindless: {} unique textures in bindless array", bindless_tex_array.texture_count);

        let materials_ssbo = if let Some(ref gs) = gltf_scene {
            scene_manager::build_materials_ssbo(
                &device, ctx.allocator_mut(), gs, &dedup_map,
                &per_material_textures,
                default_texture.view, default_black_texture.view, default_normal_texture.view,
            )?
        } else {
            let default_mat = scene_manager::BindlessMaterialData {
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
                emissive_factor: [0.0, 0.0, 0.0],
                metallic_factor: 0.0, roughness_factor: 1.0, emissive_strength: 1.0,
                ior: 1.5, clearcoat_factor: 0.0, clearcoat_roughness: 0.0,
                sheen_roughness: 0.0, transmission_factor: 0.0,
                _pad_before_sheen: 0.0, sheen_color_factor: [0.0, 0.0, 0.0], _pad_after_sheen: 0.0,
                base_color_tex_index: -1, normal_tex_index: -1, metallic_roughness_tex_index: -1,
                occlusion_tex_index: -1, emissive_tex_index: -1, clearcoat_tex_index: -1,
                clearcoat_roughness_tex_index: -1, sheen_color_tex_index: -1, transmission_tex_index: -1,
                material_flags: 0, index_offset: 0.0, _padding: 0,
            };
            let buf = create_buffer_with_data(&device, ctx.allocator_mut(), bytemuck::bytes_of(&default_mat), vk::BufferUsageFlags::STORAGE_BUFFER, "mesh_bl_materials_ssbo")?;
            scene_manager::BindlessMaterialsSSBO { buffer: buf, material_count: 1 }
        };

        info!("Mesh bindless: {} materials in SSBO", materials_ssbo.material_count);

        // --- Phase 7: Render pass, framebuffer ---
        let color_image = create_offscreen_image(&device, ctx.allocator_mut(), width, height, vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC, vk::ImageAspectFlags::COLOR, "mesh_bl_color")?;
        let depth_image = create_offscreen_image(&device, ctx.allocator_mut(), width, height, vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH, "mesh_bl_depth")?;

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
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL).dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0).dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];

        let render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default().attachments(&attachments).subpasses(std::slice::from_ref(&subpass)).dependencies(&dependencies),
                None,
            ).map_err(|e| format!("Failed to create mesh bindless render pass: {:?}", e))?
        };

        let fb_attachments = [color_image.view, depth_image.view];
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default().render_pass(render_pass).attachments(&fb_attachments).width(width).height(height).layers(1),
                None,
            ).map_err(|e| format!("Failed to create mesh bindless framebuffer: {:?}", e))?
        };

        // --- Phase 8: Descriptor set layouts (bindless-aware) ---
        let merged = reflected_pipeline::merge_descriptor_sets(&[&mesh_refl, &frag_refl]);
        let ds_layouts = unsafe {
            reflected_pipeline::create_descriptor_set_layouts_bindless(&device, &merged)?
        };

        // Push constant ranges from mesh + frag
        let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
        for pc in &mesh_refl.push_constants {
            push_ranges.push(vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::MESH_EXT).offset(0).size(pc.size));
        }
        for pc in &frag_refl.push_constants {
            // Fragment push constants share offset 0 with mesh stage in bindless mode
            // (mesh: {meshletOffset: u32, material_index: u32}, frag: {model: mat4, material_index: u32})
            // Use the maximum of both for a combined push constant range
            let frag_flags = reflected_pipeline::stage_flags_from_strings_public(&pc.stage_flags);
            push_ranges.push(vk::PushConstantRange::default().stage_flags(frag_flags).offset(0).size(pc.size));
        }

        let pipeline_layout = unsafe {
            reflected_pipeline::create_pipeline_layout_from_merged(&device, &ds_layouts, &push_ranges)?
        };

        // --- Phase 9: Descriptor pool + allocation ---
        let (pool_sizes, max_sets) = reflected_pipeline::compute_pool_sizes_bindless(&merged);

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(max_sets).pool_sizes(&pool_sizes).flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            ).map_err(|e| format!("Failed to create mesh bindless descriptor pool: {:?}", e))?
        };

        let max_set_idx = ds_layouts.keys().copied().max().unwrap_or(0);
        let mut ordered_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
        let mut set_indices: Vec<u32> = Vec::new();
        for i in 0..=max_set_idx {
            if let Some(&layout) = ds_layouts.get(&i) {
                ordered_layouts.push(layout);
                set_indices.push(i);
            }
        }

        let actual_tex_count = bindless_tex_array.texture_count.max(1);
        let mut variable_counts: Vec<u32> = vec![0; ordered_layouts.len()];
        for (layout_idx, &set_idx) in set_indices.iter().enumerate() {
            if let Some(bindings) = merged.get(&set_idx) {
                for b in bindings {
                    if b.binding_type == "bindless_combined_image_sampler_array" {
                        variable_counts[layout_idx] = actual_tex_count;
                    }
                }
            }
        }

        let mut variable_count_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&variable_counts);

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&ordered_layouts)
            .push_next(&mut variable_count_info);

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate mesh bindless descriptor sets: {:?}", e))?
        };

        let mut ds_map: HashMap<u32, vk::DescriptorSet> = HashMap::new();
        for (idx, &set_idx) in set_indices.iter().enumerate() {
            ds_map.insert(set_idx, descriptor_sets[idx]);
        }

        // --- Phase 10: Write descriptors (non-bindless sets) ---
        {
            let ssbo_map: HashMap<&str, (vk::Buffer, u64)> = [
                ("meshlet_descriptors", (meshlet_desc_buffer.buffer, meshlet_desc_size)),
                ("meshlet_vertices", (meshlet_verts_buffer.buffer, meshlet_verts_size)),
                ("meshlet_triangles", (meshlet_tris_buffer.buffer, meshlet_tris_size)),
                ("positions", (positions_buffer.buffer, positions_size)),
                ("normals", (normals_buffer.buffer, normals_size)),
                ("tex_coords", (tex_coords_buffer.buffer, tex_coords_size)),
                ("tangents", (tangents_buffer.buffer, tangents_size)),
                ("materials", (materials_ssbo.buffer.buffer, materials_ssbo.material_count as u64 * std::mem::size_of::<scene_manager::BindlessMaterialData>() as u64)),
            ].into_iter().collect();

            let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
            let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
            let total_bindings: usize = merged.values().map(|v| v.len()).sum();
            buffer_infos.reserve(total_bindings);
            image_infos.reserve(total_bindings);

            for (&set_idx, bindings) in &merged {
                let ds = match ds_map.get(&set_idx) {
                    Some(&ds) => ds,
                    None => continue,
                };

                for binding in bindings {
                    // Skip the bindless texture array (written separately)
                    if binding.binding_type == "bindless_combined_image_sampler_array" {
                        continue;
                    }

                    let vk_type = reflected_pipeline::binding_type_to_vk_public(&binding.binding_type);
                    match vk_type {
                        vk::DescriptorType::UNIFORM_BUFFER => {
                            let (buffer, range) = match binding.name.as_str() {
                                "MVP" => (mvp_buffer.buffer, binding.size as u64),
                                "Light" => (light_buffer.buffer, binding.size as u64),
                                "Material" => (material_buffer.buffer, std::mem::size_of::<MaterialUboData>() as u64),
                                "SceneLight" => {
                                    if let Some(ref sl_buf) = scene_light_ubo {
                                        (sl_buf.buffer, binding.size as u64)
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            };
                            let idx = buffer_infos.len();
                            buffer_infos.push(vk::DescriptorBufferInfo::default().buffer(buffer).offset(0).range(range));
                            writes.push(
                                vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(unsafe { std::slice::from_raw_parts(&buffer_infos[idx] as *const _, 1) }),
                            );
                        }
                        vk::DescriptorType::STORAGE_BUFFER => {
                            if let Some(&(buf, sz)) = ssbo_map.get(binding.name.as_str()) {
                                let idx = buffer_infos.len();
                                buffer_infos.push(vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(sz));
                                writes.push(
                                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(binding.binding)
                                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                        .buffer_info(unsafe { std::slice::from_raw_parts(&buffer_infos[idx] as *const _, 1) }),
                                );
                            } else if binding.name == "lights" {
                                if let Some(ref lights_buf) = lights_ssbo {
                                    let idx = buffer_infos.len();
                                    let ssbo_size = (sm.lights.len() as u64) * 64;
                                    let range = if ssbo_size > 0 { ssbo_size } else { 64 };
                                    buffer_infos.push(vk::DescriptorBufferInfo::default().buffer(lights_buf.buffer).offset(0).range(range));
                                    writes.push(
                                        vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(binding.binding)
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(unsafe { std::slice::from_raw_parts(&buffer_infos[idx] as *const _, 1) }),
                                    );
                                }
                            }
                        }
                        vk::DescriptorType::SAMPLER => {
                            let actual_sampler = ibl_assets.sampler_for_binding(&binding.name).unwrap_or(sampler);
                            let idx = image_infos.len();
                            image_infos.push(vk::DescriptorImageInfo::default().sampler(actual_sampler));
                            writes.push(
                                vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .image_info(unsafe { std::slice::from_raw_parts(&image_infos[idx] as *const _, 1) }),
                            );
                        }
                        vk::DescriptorType::SAMPLED_IMAGE => {
                            let is_cube = reflected_pipeline::is_cube_image_binding(&binding.binding_type);
                            let view = if is_cube || binding.name == "brdf_lut" {
                                ibl_assets.view_for_binding(&binding.name).unwrap_or(default_texture.view)
                            } else {
                                let fallback = match binding.name.as_str() {
                                    "emissive_tex" => default_black_texture.view,
                                    "normal_tex" => default_normal_texture.view,
                                    _ => default_texture.view,
                                };
                                texture_images.get(&binding.name).map(|img| img.view).unwrap_or(fallback)
                            };
                            let idx = image_infos.len();
                            image_infos.push(vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
                            writes.push(
                                vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(binding.binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(unsafe { std::slice::from_raw_parts(&image_infos[idx] as *const _, 1) }),
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

        // --- Write bindless texture array ---
        if bindless_tex_array.texture_count > 0 {
            let bindless_set_idx = frag_refl.bindless.as_ref().and_then(|b| b.texture_array.as_ref()).map(|ta| ta.set).unwrap_or(2);
            let bindless_binding = frag_refl.bindless.as_ref().and_then(|b| b.texture_array.as_ref()).map(|ta| ta.binding).unwrap_or(0);

            if let Some(&ds) = ds_map.get(&bindless_set_idx) {
                let img_infos: Vec<vk::DescriptorImageInfo> = (0..bindless_tex_array.texture_count as usize)
                    .map(|i| {
                        vk::DescriptorImageInfo::default()
                            .sampler(bindless_tex_array.samplers[i])
                            .image_view(bindless_tex_array.image_views[i])
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    })
                    .collect();

                let write = vk::WriteDescriptorSet::default()
                    .dst_set(ds).dst_binding(bindless_binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_infos);

                unsafe { device.update_descriptor_sets(&[write], &[]); }
                info!("Mesh bindless: wrote {} textures to set {} binding {}", bindless_tex_array.texture_count, bindless_set_idx, bindless_binding);
            }
        }

        // --- Phase 11: Load shaders and create pipeline ---
        let mesh_path = format!("{}.mesh.spv", pipeline_base);
        let frag_path = format!("{}.frag.spv", pipeline_base);

        let mesh_code = spv_loader::load_spirv(Path::new(&mesh_path))?;
        let frag_code = spv_loader::load_spirv(Path::new(&frag_path))?;
        let mesh_module = spv_loader::create_shader_module(&device, &mesh_code)?;
        let frag_module = spv_loader::create_shader_module(&device, &frag_code)?;

        let entry_name = c"main";
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::MESH_EXT).module(mesh_module).name(entry_name),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(entry_name),
        ];

        let viewport = vk::Viewport::default().width(width as f32).height(height as f32).max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL).cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE).line_width(1.0);
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true).depth_write_enable(true).depth_compare_op(vk::CompareOp::LESS);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages).viewport_state(&viewport_state)
                    .rasterization_state(&rasterizer).multisample_state(&multisampling)
                    .depth_stencil_state(&depth_stencil).color_blend_state(&color_blending)
                    .layout(pipeline_layout).render_pass(render_pass).subpass(0)],
                None,
            ).map_err(|e| format!("Failed to create mesh bindless pipeline: {:?}", e))?[0]
        };

        info!("MeshShaderRenderer initialized (bindless): {}x{}, {} meshlets, {} material groups",
              width, height, meshlet_count, meshlet_groups.len());

        Ok(MeshShaderRenderer {
            pipeline,
            pipeline_layout,
            descriptor_pool,
            ds_layouts,
            descriptor_sets,
            mesh_module,
            frag_module,
            render_pass,
            framebuffer,
            color_image,
            depth_image,
            width,
            height,
            meshlet_count,
            meshlet_desc_buffer,
            meshlet_verts_buffer,
            meshlet_tris_buffer,
            positions_buffer,
            normals_buffer,
            tex_coords_buffer,
            tangents_buffer,
            meshlet_desc_size,
            meshlet_verts_size,
            meshlet_tris_size,
            positions_size,
            normals_size,
            tex_coords_size,
            tangents_size,
            mvp_buffer,
            light_buffer,
            material_buffer,
            sampler,
            default_texture,
            default_black_texture,
            default_normal_texture,
            texture_images,
            per_material_textures,
            ibl_assets,
            has_multi_light,
            lights_ssbo,
            scene_light_ubo,
            multi_pipeline: false, // not multi-pipeline mode
            mesh_permutations: Vec::new(),
            material_to_permutation: Vec::new(),
            shared_set0_layout: vk::DescriptorSetLayout::null(),
            shared_set0: vk::DescriptorSet::null(),
            meshlet_groups,
            bindless_mode: true,
            bindless_materials_ssbo: Some(materials_ssbo),
            mesh_refl: None,
            auto_eye,
            auto_target,
            auto_up,
            auto_far,
            has_scene_bounds,
        })
    }

    /// Update camera uniforms (MVP + light view_pos).
    pub fn update_camera_internal(
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

    /// Record mesh shader rendering commands.
    pub fn render_frame(
        &self,
        ctx: &VulkanContext,
    ) -> Result<vk::CommandBuffer, String> {
        let device = &ctx.device;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| format!("Failed to allocate mesh command buffer: {:?}", e))?[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| format!("Failed to begin mesh command buffer: {:?}", e))?;
        }

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
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            })
            .clear_values(&clear_values);

        let ms_loader = ctx
            .mesh_shader_loader
            .as_ref()
            .ok_or("Mesh shader loader not available")?;

        ctx.cmd_begin_label(cmd, "Mesh Dispatch", [0.2, 0.2, 0.8, 1.0]);
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);

            if self.bindless_mode {
                // ---- Bindless draw ----
                // Single pipeline, all descriptor sets bound once, per-group push constants
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &self.descriptor_sets,
                    &[],
                );

                if self.meshlet_groups.is_empty() {
                    // Fallback: dispatch all meshlets with default push constants
                    let pc_data: [u32; 2] = [0, 0]; // meshletOffset=0, material_index=0
                    device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::MESH_EXT,
                        0,
                        bytemuck::cast_slice(&pc_data),
                    );
                    ms_loader.cmd_draw_mesh_tasks(cmd, self.meshlet_count, 1, 1);
                } else {
                    for group in &self.meshlet_groups {
                        // Push {meshletOffset: u32, material_index: u32} (8 bytes at offset 0)
                        let pc_data: [u32; 2] = [group.first_meshlet, group.material_index as u32];
                        device.cmd_push_constants(
                            cmd,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::MESH_EXT,
                            0,
                            bytemuck::cast_slice(&pc_data),
                        );
                        ms_loader.cmd_draw_mesh_tasks(cmd, group.meshlet_count, 1, 1);
                    }
                }
            } else if self.multi_pipeline && !self.mesh_permutations.is_empty() {
                // ---- Multi-pipeline draw ----
                // Bind shared set 0 once (meshlet data + SoA buffers + MVP + Light)
                if self.shared_set0 != vk::DescriptorSet::null() {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.mesh_permutations[0].pipeline_layout,
                        0,
                        std::slice::from_ref(&self.shared_set0),
                        &[],
                    );
                }

                let mut current_perm_idx: i32 = -1;

                for group in &self.meshlet_groups {
                    let perm = &self.mesh_permutations[group.permutation_index];

                    // Switch pipeline if permutation changed
                    if group.permutation_index as i32 != current_perm_idx {
                        current_perm_idx = group.permutation_index as i32;
                        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, perm.pipeline);
                        // Re-bind set 0 after pipeline change
                        if self.shared_set0 != vk::DescriptorSet::null() {
                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                perm.pipeline_layout,
                                0,
                                std::slice::from_ref(&self.shared_set0),
                                &[],
                            );
                        }
                    }

                    // Find per-material descriptor set index within this permutation
                    let local_mat_idx = perm.material_indices.iter().position(|&mi| mi == group.material_index);
                    if let Some(local_idx) = local_mat_idx {
                        if local_idx < perm.per_material_desc_sets.len() {
                            device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                perm.pipeline_layout,
                                1,
                                std::slice::from_ref(&perm.per_material_desc_sets[local_idx]),
                                &[],
                            );
                        }
                    }

                    // Push meshletOffset so the mesh shader reads the correct region
                    let meshlet_offset: u32 = group.first_meshlet;
                    device.cmd_push_constants(
                        cmd,
                        perm.pipeline_layout,
                        vk::ShaderStageFlags::MESH_EXT,
                        0,
                        &meshlet_offset.to_le_bytes(),
                    );

                    // Dispatch meshlets for this group
                    ms_loader.cmd_draw_mesh_tasks(cmd, group.meshlet_count, 1, 1);
                }
            } else {
                // ---- Single-pipeline draw (original) ----
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &self.descriptor_sets,
                    &[],
                );

                // Dispatch mesh shader workgroups: one workgroup per meshlet
                ms_loader.cmd_draw_mesh_tasks(cmd, self.meshlet_count, 1, 1);
            }

            device.cmd_end_render_pass(cmd);
        }
        ctx.cmd_end_label(cmd);

        Ok(cmd)
    }

    /// Blit the offscreen color image to a swapchain image.
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
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
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
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    /// Clean up all GPU resources.
    pub fn cleanup(&mut self, ctx: &mut VulkanContext) {
        let device = ctx.device.clone();

        unsafe {
            // Clean up multi-pipeline resources
            for perm in self.mesh_permutations.drain(..) {
                if perm.pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(perm.pipeline, None);
                }
                if perm.pipeline_layout != vk::PipelineLayout::null() {
                    device.destroy_pipeline_layout(perm.pipeline_layout, None);
                }
                if perm.material_set_layout != vk::DescriptorSetLayout::null() {
                    device.destroy_descriptor_set_layout(perm.material_set_layout, None);
                }
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
            device.destroy_shader_module(self.mesh_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }

        self.color_image.destroy(&device, ctx.allocator_mut());
        self.depth_image.destroy(&device, ctx.allocator_mut());
        self.default_texture.destroy(&device, ctx.allocator_mut());
        self.default_black_texture
            .destroy(&device, ctx.allocator_mut());
        self.default_normal_texture
            .destroy(&device, ctx.allocator_mut());
        for (_, mut tex) in self.texture_images.drain() {
            tex.destroy(&device, ctx.allocator_mut());
        }
        for tex_map in self.per_material_textures.drain(..) {
            for (_, mut tex) in tex_map {
                tex.destroy(&device, ctx.allocator_mut());
            }
        }
        // Clean up bindless materials SSBO
        if let Some(ref mut ssbo) = self.bindless_materials_ssbo {
            ssbo.buffer.destroy(&device, ctx.allocator_mut());
        }
        self.bindless_materials_ssbo = None;

        self.ibl_assets.destroy(&device, ctx.allocator_mut());
        self.mvp_buffer.destroy(&device, ctx.allocator_mut());
        self.light_buffer.destroy(&device, ctx.allocator_mut());
        if let Some(mut buf) = self.lights_ssbo.take() {
            buf.destroy(&device, ctx.allocator_mut());
        }
        if let Some(mut buf) = self.scene_light_ubo.take() {
            buf.destroy(&device, ctx.allocator_mut());
        }
        self.material_buffer.destroy(&device, ctx.allocator_mut());
        self.meshlet_desc_buffer
            .destroy(&device, ctx.allocator_mut());
        self.meshlet_verts_buffer
            .destroy(&device, ctx.allocator_mut());
        self.meshlet_tris_buffer
            .destroy(&device, ctx.allocator_mut());
        self.positions_buffer.destroy(&device, ctx.allocator_mut());
        self.normals_buffer.destroy(&device, ctx.allocator_mut());
        self.tex_coords_buffer
            .destroy(&device, ctx.allocator_mut());
        self.tangents_buffer
            .destroy(&device, ctx.allocator_mut());
    }
}

// ===========================================================================
// Renderer trait implementation for MeshShaderRenderer
// ===========================================================================

impl scene_manager::Renderer for MeshShaderRenderer {
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
        self.update_camera_internal(eye, target, up, fov_y, aspect, near, far);
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
