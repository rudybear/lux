//! glTF 2.0 scene loader for the Rust/ash playground.
//!
//! Uses the `gltf` crate to parse .glb/.gltf files.
//! Produces GPU-ready interleaved vertex data matching Lux PBR vertex layout:
//!   position(vec3) + normal(vec3) + uv(vec2) + tangent(vec4) = 48 bytes per vertex.
//!
//! Also extracts texture images from GLB files, decoding PNG/JPEG to RGBA u8 pixels.

use glam::{Mat4, Quat, Vec3};
use log::info;
use std::path::Path;

// ===========================================================================
// Data structures
// ===========================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GltfVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

#[derive(Debug)]
pub struct GltfMesh {
    pub name: String,
    pub vertices: Vec<GltfVertex>,
    pub indices: Vec<u32>,
    pub material_index: usize,
    pub has_tangents: bool,
    pub vertex_stride: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
        }
    }
}

/// Decoded RGBA image data extracted from a GLB texture.
#[derive(Debug, Clone)]
pub struct TextureImage {
    /// RGBA u8 pixel data.
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct GltfMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub alpha_mode: String,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// Base color / albedo texture (RGBA pixels).
    pub base_color_image: Option<TextureImage>,
    /// Normal map texture (RGBA pixels).
    pub normal_image: Option<TextureImage>,
    /// Metallic-roughness texture (RGBA pixels).
    pub metallic_roughness_image: Option<TextureImage>,
    /// Occlusion texture (RGBA pixels).
    pub occlusion_image: Option<TextureImage>,
    /// Emissive texture (RGBA pixels).
    pub emissive_image: Option<TextureImage>,

    // --- KHR_materials_* extension fields ---
    pub has_clearcoat: bool,
    pub clearcoat_factor: f32,
    pub clearcoat_roughness_factor: f32,
    pub clearcoat_image: Option<TextureImage>,
    pub clearcoat_roughness_image: Option<TextureImage>,

    pub has_sheen: bool,
    pub sheen_color_factor: [f32; 3],
    pub sheen_roughness_factor: f32,
    pub sheen_color_image: Option<TextureImage>,

    pub has_transmission: bool,
    pub transmission_factor: f32,
    pub transmission_image: Option<TextureImage>,

    pub ior: f32,
    pub emissive_strength: f32,
    pub is_unlit: bool,

    // KHR_texture_transform
    pub base_color_uv_xform: UvTransform,
    pub normal_uv_xform: UvTransform,
    pub metallic_roughness_uv_xform: UvTransform,

    // Custom properties (from glTF extras, for extended bindless structs)
    pub custom_float_properties: std::collections::BTreeMap<String, f32>,
    pub custom_vec4_properties: std::collections::BTreeMap<String, [f32; 4]>,
}

impl Default for GltfMaterial {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            emissive: [0.0, 0.0, 0.0],
            alpha_mode: "OPAQUE".to_string(),
            alpha_cutoff: 0.5,
            double_sided: false,
            base_color_image: None,
            normal_image: None,
            metallic_roughness_image: None,
            occlusion_image: None,
            emissive_image: None,
            has_clearcoat: false,
            clearcoat_factor: 0.0,
            clearcoat_roughness_factor: 0.0,
            clearcoat_image: None,
            clearcoat_roughness_image: None,
            has_sheen: false,
            sheen_color_factor: [0.0, 0.0, 0.0],
            sheen_roughness_factor: 0.0,
            sheen_color_image: None,
            has_transmission: false,
            transmission_factor: 0.0,
            transmission_image: None,
            ior: 1.5,
            emissive_strength: 1.0,
            is_unlit: false,
            base_color_uv_xform: UvTransform::default(),
            normal_uv_xform: UvTransform::default(),
            metallic_roughness_uv_xform: UvTransform::default(),
            custom_float_properties: std::collections::BTreeMap::new(),
            custom_vec4_properties: std::collections::BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GltfNode {
    pub name: String,
    pub local_transform: Mat4,
    pub world_transform: Mat4,
    pub mesh_index: i32,
    pub camera_index: i32,
    pub light_index: i32,
    pub children: Vec<usize>,
    pub parent: i32,
}

#[derive(Debug, Clone)]
pub struct GltfCamera {
    pub name: String,
    pub camera_type: String,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub position: Vec3,
    pub direction: Vec3,
}

#[derive(Debug, Clone)]
pub struct GltfLight {
    pub name: String,
    pub light_type: String,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub position: Vec3,
    pub direction: Vec3,
}

/// Gaussian splat data extracted from KHR_gaussian_splatting glTF extension.
pub struct GaussianSplatData {
    pub positions: Vec<[f32; 4]>,
    pub rotations: Vec<[f32; 4]>,
    pub scales: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh_coefficients: Vec<Vec<[f32; 4]>>,
    pub sh_degree: u32,
    pub num_splats: u32,
    pub has_splats: bool,
    /// True when loaded from KHR_gaussian_splatting extension (opacities are linear [0,1],
    /// need logit conversion for shader compatibility).
    pub khr_format: bool,
}

impl Default for GaussianSplatData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: 0,
            num_splats: 0,
            has_splats: false,
            khr_format: false,
        }
    }
}

pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    pub materials: Vec<GltfMaterial>,
    pub nodes: Vec<GltfNode>,
    pub cameras: Vec<GltfCamera>,
    pub lights: Vec<GltfLight>,
    pub root_nodes: Vec<usize>,
    /// Maps glTF mesh index → (start, count) in `meshes` vec (one entry per primitive).
    pub mesh_primitive_ranges: Vec<(usize, usize)>,
    /// Gaussian splat data from KHR_gaussian_splatting extension (if present).
    pub splat_data: GaussianSplatData,
}

pub struct DrawItem {
    pub world_transform: Mat4,
    pub mesh_index: usize,
    pub material_index: usize,
}

pub struct DrawRange {
    pub index_offset: u32,
    pub index_count: u32,
    pub material_index: usize,
}

// ===========================================================================
// Scene traversal
// ===========================================================================

/// Traverse the scene graph and produce a flat draw list sorted by material.
pub fn flatten_scene(scene: &mut GltfScene) -> Vec<DrawItem> {
    let mut items = Vec::new();

    fn traverse(
        scene: &mut GltfScene,
        node_idx: usize,
        parent_world: Mat4,
        items: &mut Vec<DrawItem>,
    ) {
        let local = scene.nodes[node_idx].local_transform;
        let world = parent_world * local;
        scene.nodes[node_idx].world_transform = world;

        let mesh_idx = scene.nodes[node_idx].mesh_index;
        if mesh_idx >= 0 {
            let mi = mesh_idx as usize;
            if mi < scene.mesh_primitive_ranges.len() {
                let (start, count) = scene.mesh_primitive_ranges[mi];
                for pi in start..start + count {
                    items.push(DrawItem {
                        world_transform: world,
                        mesh_index: pi,
                        material_index: scene.meshes[pi].material_index,
                    });
                }
            } else if mi < scene.meshes.len() {
                items.push(DrawItem {
                    world_transform: world,
                    mesh_index: mi,
                    material_index: scene.meshes[mi].material_index,
                });
            }
        }

        let children: Vec<usize> = scene.nodes[node_idx].children.clone();
        for child in children {
            if child < scene.nodes.len() {
                traverse(scene, child, world, items);
            }
        }
    }

    let roots: Vec<usize> = scene.root_nodes.clone();
    for root in roots {
        if root < scene.nodes.len() {
            traverse(scene, root, Mat4::IDENTITY, &mut items);
        }
    }

    // Extract light positions/directions from node world transforms.
    // We iterate by index to avoid simultaneous borrows of nodes and lights.
    for ni in 0..scene.nodes.len() {
        let light_index = scene.nodes[ni].light_index;
        if light_index >= 0 && (light_index as usize) < scene.lights.len() {
            let world = scene.nodes[ni].world_transform;
            let li = light_index as usize;
            // Position from column 3 (w_axis) of world transform
            scene.lights[li].position = Vec3::new(
                world.w_axis.x,
                world.w_axis.y,
                world.w_axis.z,
            );
            // Direction: transform -Z through the rotation part of the world matrix
            let dir = world.transform_vector3(Vec3::new(0.0, 0.0, -1.0));
            scene.lights[li].direction = dir.normalize_or_zero();
        }
    }

    items.sort_by_key(|d| d.material_index);
    items
}

// ===========================================================================
// Loading
// ===========================================================================

/// Decode a glTF image to RGBA u8 pixels using the `image` crate.
fn decode_gltf_image(gltf_image: &gltf::image::Data) -> Option<TextureImage> {
    let (width, height) = (gltf_image.width, gltf_image.height);
    let pixels = &gltf_image.pixels;

    // The gltf crate already decodes images; we just need to ensure RGBA8 format
    let rgba_pixels = match gltf_image.format {
        gltf::image::Format::R8G8B8A8 => {
            // Already RGBA8
            pixels.clone()
        }
        gltf::image::Format::R8G8B8 => {
            // Convert RGB to RGBA
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in pixels.chunks(3) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
            rgba
        }
        gltf::image::Format::R8 => {
            // Grayscale to RGBA
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for &v in pixels {
                rgba.push(v);
                rgba.push(v);
                rgba.push(v);
                rgba.push(255);
            }
            rgba
        }
        gltf::image::Format::R8G8 => {
            // RG to RGBA (common for normal maps or metallic-roughness)
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in pixels.chunks(2) {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(0);
                rgba.push(255);
            }
            rgba
        }
        gltf::image::Format::R16 | gltf::image::Format::R16G16 |
        gltf::image::Format::R16G16B16 | gltf::image::Format::R16G16B16A16 => {
            // 16-bit formats: convert to 8-bit RGBA
            let component_count = match gltf_image.format {
                gltf::image::Format::R16 => 1,
                gltf::image::Format::R16G16 => 2,
                gltf::image::Format::R16G16B16 => 3,
                gltf::image::Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for pixel_idx in 0..(width * height) as usize {
                let base = pixel_idx * component_count * 2;
                let mut channels = [0u8; 4];
                for c in 0..component_count {
                    let offset = base + c * 2;
                    if offset + 1 < pixels.len() {
                        let val16 = u16::from_le_bytes([pixels[offset], pixels[offset + 1]]);
                        channels[c] = (val16 >> 8) as u8;
                    }
                }
                // Fill missing channels
                if component_count == 1 {
                    channels[1] = channels[0];
                    channels[2] = channels[0];
                }
                if component_count < 4 {
                    channels[3] = 255;
                }
                rgba.extend_from_slice(&channels);
            }
            rgba
        }
        _ => {
            // Unknown format, skip
            return None;
        }
    };

    if rgba_pixels.len() == (width * height * 4) as usize {
        Some(TextureImage {
            pixels: rgba_pixels,
            width,
            height,
        })
    } else {
        None
    }
}

/// Extract a texture image by glTF texture index.
fn extract_texture(
    images: &[gltf::image::Data],
    texture: Option<gltf::Texture>,
) -> Option<TextureImage> {
    let tex = texture?;
    let img_index = tex.source().index();
    if img_index < images.len() {
        decode_gltf_image(&images[img_index])
    } else {
        None
    }
}

/// Import glTF without validation (KHR_gaussian_splatting uses custom attribute
/// names like "KHR_gaussian_splatting:ROTATION" that the gltf crate's strict
/// validation rejects as invalid semantic names).
fn import_without_validation(path: &Path) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>), String> {
    let file_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let gltf_data = gltf::Gltf::from_slice_without_validation(&file_data)
        .map_err(|e| format!("Failed to parse glTF {}: {}", path.display(), e))?;
    let base = path.parent();
    let blob = gltf_data.blob;
    let document = gltf_data.document;
    let buffers = gltf::import_buffers(&document, base, blob)
        .map_err(|e| format!("Failed to load glTF buffers {}: {}", path.display(), e))?;
    let images = gltf::import_images(&document, base, &buffers)
        .map_err(|e| format!("Failed to load glTF images {}: {}", path.display(), e))?;
    Ok((document, buffers, images))
}

/// Load a .glb or .gltf file.
pub fn load_gltf(path: &Path) -> Result<GltfScene, String> {
    let (document, buffers, images) =
        import_without_validation(path)?;

    info!("Loaded glTF: {} images, {} materials", images.len(), document.materials().len());

    let mut scene = GltfScene {
        meshes: Vec::new(),
        materials: Vec::new(),
        nodes: Vec::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        root_nodes: Vec::new(),
        mesh_primitive_ranges: Vec::new(),
        splat_data: GaussianSplatData::default(),
    };

    // --- Materials (with texture extraction) ---
    let all_textures: Vec<gltf::Texture> = document.textures().collect();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let bc = pbr.base_color_factor();

        let base_color_image = extract_texture(&images, pbr.base_color_texture().map(|t| t.texture()));
        let base_color_uv_xform = pbr.base_color_texture()
            .and_then(|info| info.texture_transform())
            .map(|t| UvTransform {
                offset: t.offset(),
                scale: t.scale(),
                rotation: t.rotation(),
            })
            .unwrap_or_default();

        let normal_image = extract_texture(&images, mat.normal_texture().map(|t| t.texture()));
        // NormalTexture does not expose texture_transform() directly; use default
        let normal_uv_xform = UvTransform::default();

        let metallic_roughness_image = extract_texture(&images, pbr.metallic_roughness_texture().map(|t| t.texture()));
        let metallic_roughness_uv_xform = pbr.metallic_roughness_texture()
            .and_then(|info| info.texture_transform())
            .map(|t| UvTransform {
                offset: t.offset(),
                scale: t.scale(),
                rotation: t.rotation(),
            })
            .unwrap_or_default();

        let occlusion_image = extract_texture(&images, mat.occlusion_texture().map(|t| t.texture()));
        let emissive_image = extract_texture(&images, mat.emissive_texture().map(|t| t.texture()));

        // --- KHR_materials_* extensions ---
        // Clearcoat: parsed from raw extension JSON (gltf crate v1.4.1 lacks built-in support)
        let (has_clearcoat, clearcoat_factor, clearcoat_roughness_factor,
             clearcoat_image, clearcoat_roughness_image) =
            if let Some(cc_val) = mat.extension_value("KHR_materials_clearcoat") {
                let cc_factor = cc_val.get("clearcoatFactor")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let cc_rough = cc_val.get("clearcoatRoughnessFactor")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let cc_tex = cc_val.get("clearcoatTexture")
                    .and_then(|t| t.get("index"))
                    .and_then(|v| v.as_u64())
                    .and_then(|idx| all_textures.get(idx as usize))
                    .and_then(|tex| extract_texture(&images, Some(tex.clone())));
                let cc_rough_tex = cc_val.get("clearcoatRoughnessTexture")
                    .and_then(|t| t.get("index"))
                    .and_then(|v| v.as_u64())
                    .and_then(|idx| all_textures.get(idx as usize))
                    .and_then(|tex| extract_texture(&images, Some(tex.clone())));
                (true, cc_factor, cc_rough, cc_tex, cc_rough_tex)
            } else {
                (false, 0.0, 0.0, None, None)
            };

        // Sheen: parsed from raw extension JSON (gltf crate v1.4.1 lacks built-in support)
        let (has_sheen, sheen_color_factor, sheen_roughness_factor, sheen_color_image) =
            if let Some(sh_val) = mat.extension_value("KHR_materials_sheen") {
                let sh_color = if let Some(arr) = sh_val.get("sheenColorFactor").and_then(|v| v.as_array()) {
                    let r = arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let g = arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    let b = arr.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                    [r, g, b]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let sh_rough = sh_val.get("sheenRoughnessFactor")
                    .and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let sh_tex = sh_val.get("sheenColorTexture")
                    .and_then(|t| t.get("index"))
                    .and_then(|v| v.as_u64())
                    .and_then(|idx| all_textures.get(idx as usize))
                    .and_then(|tex| extract_texture(&images, Some(tex.clone())));
                (true, sh_color, sh_rough, sh_tex)
            } else {
                (false, [0.0, 0.0, 0.0], 0.0, None)
            };

        // Transmission: uses gltf crate's built-in API
        let (has_transmission, transmission_factor, transmission_image) =
            if let Some(tr) = mat.transmission() {
                let tr_tex = tr.transmission_texture().and_then(|t| {
                    extract_texture(&images, Some(t.texture()))
                });
                (true, tr.transmission_factor(), tr_tex)
            } else {
                (false, 0.0, None)
            };

        let ior = mat.ior().unwrap_or(1.5);
        let emissive_strength = mat.emissive_strength().unwrap_or(1.0);
        let is_unlit = mat.unlit();

        let mat_name = mat.name().unwrap_or("unnamed").to_string();
        info!(
            "Material '{}': base_color={}, normal={}, metallic_roughness={}, occlusion={}, emissive={}",
            mat_name,
            base_color_image.as_ref().map_or("none".to_string(), |i| format!("{}x{}", i.width, i.height)),
            normal_image.as_ref().map_or("none".to_string(), |i| format!("{}x{}", i.width, i.height)),
            metallic_roughness_image.as_ref().map_or("none".to_string(), |i| format!("{}x{}", i.width, i.height)),
            occlusion_image.as_ref().map_or("none".to_string(), |i| format!("{}x{}", i.width, i.height)),
            emissive_image.as_ref().map_or("none".to_string(), |i| format!("{}x{}", i.width, i.height)),
        );

        scene.materials.push(GltfMaterial {
            name: mat_name,
            base_color: bc,
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: mat.emissive_factor(),
            alpha_mode: match mat.alpha_mode() {
                gltf::material::AlphaMode::Opaque => "OPAQUE",
                gltf::material::AlphaMode::Mask => "MASK",
                gltf::material::AlphaMode::Blend => "BLEND",
            }
            .to_string(),
            alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
            double_sided: mat.double_sided(),
            base_color_image,
            normal_image,
            metallic_roughness_image,
            occlusion_image,
            emissive_image,
            has_clearcoat,
            clearcoat_factor,
            clearcoat_roughness_factor,
            clearcoat_image,
            clearcoat_roughness_image,
            has_sheen,
            sheen_color_factor,
            sheen_roughness_factor,
            sheen_color_image,
            has_transmission,
            transmission_factor,
            transmission_image,
            ior,
            emissive_strength,
            is_unlit,
            base_color_uv_xform,
            normal_uv_xform,
            metallic_roughness_uv_xform,
            custom_float_properties: std::collections::BTreeMap::new(),
            custom_vec4_properties: std::collections::BTreeMap::new(),
        });

        // Load custom properties from glTF extras
        if let Some(raw_extras) = mat.extras().as_ref() {
            if let Ok(extras_json) = serde_json::from_str::<serde_json::Value>(raw_extras.get()) {
                if let Some(lux_props) = extras_json.get("lux_properties").and_then(|v| v.as_object()) {
                    let current_mat = scene.materials.last_mut().unwrap();
                    for (key, val) in lux_props {
                        if let Some(f) = val.as_f64() {
                            current_mat.custom_float_properties.insert(key.clone(), f as f32);
                        }
                    }
                }
            }
        }
    }
    if scene.materials.is_empty() {
        scene.materials.push(GltfMaterial::default());
    }

    // --- Meshes ---
    for mesh in document.meshes() {
        let prim_start = scene.meshes.len();
        for prim in mesh.primitives() {
            // Skip POINTS primitives (splats) — they are handled separately below
            if prim.mode() == gltf::mesh::Mode::Points {
                continue;
            }
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            if positions.is_empty() {
                continue;
            }

            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            let texcoords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|iter| iter.into_f32().collect())
                .unwrap_or_default();

            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            let has_tangents = !tangents.is_empty();

            let mut vertices = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                vertices.push(GltfVertex {
                    position: positions[i],
                    normal: if i < normals.len() {
                        normals[i]
                    } else {
                        [0.0, 1.0, 0.0]
                    },
                    uv: if i < texcoords.len() {
                        texcoords[i]
                    } else {
                        [0.0, 0.0]
                    },
                    tangent: if i < tangents.len() {
                        tangents[i]
                    } else {
                        [1.0, 0.0, 0.0, 1.0]
                    },
                });
            }

            let indices: Vec<u32> = reader
                .read_indices()
                .map(|iter| iter.into_u32().collect())
                .unwrap_or_else(|| (0..vertices.len() as u32).collect());

            let mat_idx = prim.material().index().unwrap_or(0);

            scene.meshes.push(GltfMesh {
                name: mesh.name().unwrap_or("unnamed").to_string(),
                vertex_stride: std::mem::size_of::<GltfVertex>(),
                vertices,
                indices,
                material_index: mat_idx,
                has_tangents,
            });
        }
        scene.mesh_primitive_ranges.push((prim_start, scene.meshes.len() - prim_start));
    }

    // --- KHR_gaussian_splatting ---
    // Scan primitives for POINTS mode with the KHR_gaussian_splatting extension.
    // The extension stores splat attributes (position, rotation, scale, opacity, SH)
    // as regular glTF accessors referenced from the primitive's extension JSON.
    // Supports multi-primitive merging: accumulates ALL splat primitives across all meshes,
    // with zero-padding for SH degree mismatches between primitives.

    // Per-primitive metadata for deferred node transform application
    struct SplatPrimInfo {
        mesh_index: usize,    // glTF mesh index this primitive belongs to
        start_splat: usize,   // index of first splat in accumulated arrays
        splat_count: usize,   // number of splats in this primitive
    }
    let mut splat_prim_infos: Vec<SplatPrimInfo> = Vec::new();
    let mut global_max_sh_degree: u32 = 0;

    // Helper: read a float accessor by index from the buffer data.
    let read_accessor_f32 = |acc_idx: usize| -> Vec<f32> {
        if let Some(accessor) = document.accessors().nth(acc_idx) {
            let view = match accessor.view() {
                Some(v) => v,
                None => return Vec::new(),
            };
            let buf_idx = view.buffer().index();
            if buf_idx >= buffers.len() {
                return Vec::new();
            }
            let data = &buffers[buf_idx];
            let offset = view.offset() + accessor.offset();
            let count = accessor.count();
            let comp_count = match accessor.dimensions() {
                gltf::accessor::Dimensions::Scalar => 1,
                gltf::accessor::Dimensions::Vec2 => 2,
                gltf::accessor::Dimensions::Vec3 => 3,
                gltf::accessor::Dimensions::Vec4 => 4,
                _ => 1,
            };
            let stride = view.stride().unwrap_or(comp_count * 4);
            let mut out = Vec::with_capacity(count * comp_count);
            for i in 0..count {
                let base = offset + i * stride;
                for c in 0..comp_count {
                    let off = base + c * 4;
                    if off + 4 <= data.len() {
                        let val = f32::from_le_bytes([
                            data[off], data[off + 1], data[off + 2], data[off + 3],
                        ]);
                        out.push(val);
                    }
                }
            }
            out
        } else {
            Vec::new()
        }
    };

    for (mi, mesh) in document.meshes().enumerate() {
        for prim in mesh.primitives() {
            // glTF mode 0 = POINTS
            if prim.mode() != gltf::mesh::Mode::Points {
                continue;
            }
            let ext = match prim.extension_value("KHR_gaussian_splatting") {
                Some(v) => v.clone(),
                None => continue,
            };

            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

            // Read positions from the standard POSITION attribute (vec3 -> pad to vec4)
            let prim_positions: Vec<[f32; 4]> = reader
                .read_positions()
                .map(|iter| iter.map(|p| [p[0], p[1], p[2], 1.0]).collect())
                .unwrap_or_default();

            let prim_num_splats = prim_positions.len() as u32;
            if prim_num_splats == 0 {
                continue;
            }

            if scene.splat_data.has_splats {
                info!(
                    "Merging additional splat primitive from mesh: {}",
                    mesh.name().unwrap_or("unnamed")
                );
            } else {
                info!(
                    "Detected KHR_gaussian_splatting primitive in mesh: {}",
                    mesh.name().unwrap_or("unnamed")
                );
            }

            // Mark as KHR format (opacities are linear [0,1], need logit conversion)
            scene.splat_data.khr_format = true;

            // Parse the extension's "attributes" sub-object for accessor indices
            let ext_attrs = ext.get("attributes");

            // Parse rotation accessor (vec4 quaternions)
            let prim_rotations: Vec<[f32; 4]> = ext_attrs
                .and_then(|a| a.get("ROTATION"))
                .and_then(|v| v.as_u64())
                .map(|idx| {
                    let raw = read_accessor_f32(idx as usize);
                    raw.chunks(4)
                        .map(|c| [
                            c.first().copied().unwrap_or(0.0),
                            c.get(1).copied().unwrap_or(0.0),
                            c.get(2).copied().unwrap_or(0.0),
                            c.get(3).copied().unwrap_or(1.0),
                        ])
                        .collect()
                })
                .unwrap_or_else(|| vec![[0.0, 0.0, 0.0, 1.0]; prim_num_splats as usize]);

            // Parse scale accessor (vec3 -> pad to vec4)
            let prim_scales: Vec<[f32; 4]> = ext_attrs
                .and_then(|a| a.get("SCALE"))
                .and_then(|v| v.as_u64())
                .map(|idx| {
                    let raw = read_accessor_f32(idx as usize);
                    raw.chunks(3)
                        .map(|c| [
                            c.first().copied().unwrap_or(1.0),
                            c.get(1).copied().unwrap_or(1.0),
                            c.get(2).copied().unwrap_or(1.0),
                            0.0,
                        ])
                        .collect()
                })
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 0.0]; prim_num_splats as usize]);

            // Parse opacity accessor (scalar)
            let prim_opacities: Vec<f32> = ext_attrs
                .and_then(|a| a.get("OPACITY"))
                .and_then(|v| v.as_u64())
                .map(|idx| read_accessor_f32(idx as usize))
                .unwrap_or_else(|| vec![1.0; prim_num_splats as usize]);

            // Parse spherical harmonics coefficients
            // Extension format: "sh": [{"coefficients": <accessor_idx>, "degree": <N>}, ...]
            // KHR format stores all coefficients for a degree in one flat accessor.
            // The shader expects individual coefficient buffers (one per SH basis function).
            // Degree 0: 1 coeff  -> buffer index 0
            // Degree 1: 3 coeffs -> buffer indices 1,2,3
            // Degree 2: 5 coeffs -> buffer indices 4,5,6,7,8
            // Degree 3: 7 coeffs -> buffer indices 9,10,11,12,13,14,15
            let mut prim_sh_degree = 0u32;
            let mut prim_sh_coefficients: Vec<Vec<[f32; 4]>> = Vec::new();
            let coeff_base: [usize; 4] = [0, 1, 4, 9];
            if let Some(sh_arr) = ext.get("sh").and_then(|v| v.as_array()) {
                for sh_entry in sh_arr {
                    let degree = sh_entry.get("degree")
                        .and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    if degree > prim_sh_degree {
                        prim_sh_degree = degree;
                    }
                    if let Some(acc_idx) = sh_entry.get("coefficients").and_then(|v| v.as_u64()) {
                        let raw = read_accessor_f32(acc_idx as usize);
                        let n = prim_num_splats as usize;
                        let num_coeffs = (2 * degree + 1) as usize; // coefficients per degree
                        let floats_per_splat = num_coeffs * 3; // 3 channels (RGB) per coefficient

                        if degree == 0 {
                            // DC term: vec3 per splat -> 1 buffer
                            let base_idx = coeff_base[0];
                            if base_idx >= prim_sh_coefficients.len() {
                                prim_sh_coefficients.resize(base_idx + 1, Vec::new());
                            }
                            prim_sh_coefficients[base_idx] = raw.chunks(3)
                                .map(|c| [
                                    c.first().copied().unwrap_or(0.0),
                                    c.get(1).copied().unwrap_or(0.0),
                                    c.get(2).copied().unwrap_or(0.0),
                                    0.0,
                                ])
                                .collect();
                        } else {
                            // Higher degrees: flat scalar array, split into individual coefficients
                            // Layout per splat: [c0_r, c0_g, c0_b, c1_r, c1_g, c1_b, ...]
                            let base_idx = coeff_base[degree as usize];
                            let total_coeffs = base_idx + num_coeffs;
                            if total_coeffs > prim_sh_coefficients.len() {
                                prim_sh_coefficients.resize(total_coeffs, Vec::new());
                            }
                            // Initialize individual coefficient buffers
                            for ci in 0..num_coeffs {
                                prim_sh_coefficients[base_idx + ci] = Vec::with_capacity(n);
                            }
                            // Unpack: for each splat, extract individual vec3 coefficients
                            for si in 0..n {
                                let splat_offset = si * floats_per_splat;
                                for ci in 0..num_coeffs {
                                    let off = splat_offset + ci * 3;
                                    prim_sh_coefficients[base_idx + ci].push([
                                        raw.get(off).copied().unwrap_or(0.0),
                                        raw.get(off + 1).copied().unwrap_or(0.0),
                                        raw.get(off + 2).copied().unwrap_or(0.0),
                                        0.0,
                                    ]);
                                }
                            }
                        }
                    }
                }
            }

            // Update global max SH degree
            if prim_sh_degree > global_max_sh_degree {
                global_max_sh_degree = prim_sh_degree;
            }

            // Record per-primitive metadata for node transform application
            splat_prim_infos.push(SplatPrimInfo {
                mesh_index: mi,
                start_splat: scene.splat_data.num_splats as usize,
                splat_count: prim_num_splats as usize,
            });

            // Append positions, rotations, scales, opacities to scene-level arrays
            scene.splat_data.positions.extend_from_slice(&prim_positions);
            scene.splat_data.rotations.extend_from_slice(&prim_rotations);
            scene.splat_data.scales.extend_from_slice(&prim_scales);
            scene.splat_data.opacities.extend_from_slice(&prim_opacities);

            // Append SH coefficients with zero-padding for missing higher degrees
            // Ensure scene-level sh_coefficients has enough degree slots
            if prim_sh_coefficients.len() > scene.splat_data.sh_coefficients.len() {
                let old_size = scene.splat_data.sh_coefficients.len();
                scene.splat_data.sh_coefficients.resize_with(
                    prim_sh_coefficients.len(), Vec::new,
                );
                // Zero-fill new degree arrays for previously accumulated splats
                for d in old_size..prim_sh_coefficients.len() {
                    if !prim_sh_coefficients[d].is_empty() && prim_num_splats > 0 {
                        let floats_per_splat = prim_sh_coefficients[d].len() / prim_num_splats as usize;
                        scene.splat_data.sh_coefficients[d] =
                            vec![[0.0f32; 4]; scene.splat_data.num_splats as usize * floats_per_splat];
                    }
                }
            }
            // Append this primitive's SH data for each degree
            for d in 0..prim_sh_coefficients.len() {
                scene.splat_data.sh_coefficients[d]
                    .extend_from_slice(&prim_sh_coefficients[d]);
            }
            // For degrees that exist at the scene level but not in this primitive, zero-pad
            for d in prim_sh_coefficients.len()..scene.splat_data.sh_coefficients.len() {
                if !scene.splat_data.sh_coefficients[d].is_empty()
                    && scene.splat_data.num_splats > 0
                {
                    let floats_per_splat = scene.splat_data.sh_coefficients[d].len()
                        / scene.splat_data.num_splats as usize;
                    let padding = vec![[0.0f32; 4]; prim_num_splats as usize * floats_per_splat];
                    scene.splat_data.sh_coefficients[d].extend_from_slice(&padding);
                }
            }

            scene.splat_data.num_splats += prim_num_splats;
            scene.splat_data.has_splats = true;

            info!(
                "Gaussian splats (cumulative): {} splats, primitive SH degree {}",
                scene.splat_data.num_splats, prim_sh_degree
            );
        }
    }

    // Finalize global SH degree and apply KHR opacity logit conversion
    if scene.splat_data.has_splats {
        scene.splat_data.sh_degree = global_max_sh_degree;

        // Convert KHR linear values to log/logit space for shader compatibility.
        // The compute shader applies exp() to scales and sigmoid() to opacity,
        // so we must store raw log-space scales and logit-space opacity.
        // Only for KHR format; internal format (_SCALE, _OPACITY) already stores raw values.
        if scene.splat_data.khr_format {
            // Convert linear scales to log-space: ln(scale)
            for scale in &mut scene.splat_data.scales {
                scale[0] = scale[0].max(1e-7).ln();
                scale[1] = scale[1].max(1e-7).ln();
                scale[2] = scale[2].max(1e-7).ln();
            }
            info!(
                "Converted KHR linear scales to log-space for {} splats",
                scene.splat_data.scales.len()
            );

            // Convert linear opacity [0,1] to logit: ln(p / (1 - p))
            for opacity in &mut scene.splat_data.opacities {
                let p = opacity.clamp(1e-6, 1.0 - 1e-6);
                *opacity = (p / (1.0 - p)).ln();
            }
            info!(
                "Converted KHR linear opacity to logit for {} splats",
                scene.splat_data.opacities.len()
            );
        }

        info!(
            "Total gaussian splats: {}, max SH degree {}",
            scene.splat_data.num_splats, scene.splat_data.sh_degree
        );
    }

    // --- Nodes ---
    for node in document.nodes() {
        let (t, r, s) = node.transform().decomposed();
        let translation = Mat4::from_translation(Vec3::from(t));
        let rotation = Mat4::from_quat(Quat::from_array(r));
        let scale = Mat4::from_scale(Vec3::from(s));
        let local_transform = translation * rotation * scale;

        let children: Vec<usize> = node.children().map(|c| c.index()).collect();
        let mesh_index = node.mesh().map(|m| m.index() as i32).unwrap_or(-1);
        let camera_index = node.camera().map(|c| c.index() as i32).unwrap_or(-1);

        // Set light index from node extension (KHR_lights_punctual)
        let light_index = node.light().map(|l| l.index() as i32).unwrap_or(-1);

        scene.nodes.push(GltfNode {
            name: node.name().unwrap_or("unnamed").to_string(),
            local_transform,
            world_transform: Mat4::IDENTITY,
            mesh_index,
            camera_index,
            light_index,
            children,
            parent: -1,
        });
    }

    // Set parents
    for i in 0..scene.nodes.len() {
        let children = scene.nodes[i].children.clone();
        for &child in &children {
            if child < scene.nodes.len() {
                scene.nodes[child].parent = i as i32;
            }
        }
    }

    // Root nodes
    if let Some(gltf_scene) = document.default_scene().or_else(|| document.scenes().next()) {
        scene.root_nodes = gltf_scene.nodes().map(|n| n.index()).collect();
    }

    // --- Apply node transforms to splat data ---
    // Now that nodes are loaded and parent relationships are set, compute world
    // transforms and apply them to splat positions, rotations, and scales.
    if scene.splat_data.has_splats && !splat_prim_infos.is_empty() {
        // First compute world transforms for all nodes (top-down BFS from roots)
        fn compute_world(nodes: &mut [GltfNode], node_idx: usize, parent_world: Mat4) {
            let world = parent_world * nodes[node_idx].local_transform;
            nodes[node_idx].world_transform = world;
            let children = nodes[node_idx].children.clone();
            for child in children {
                if child < nodes.len() {
                    compute_world(nodes, child, world);
                }
            }
        }
        for &root in &scene.root_nodes {
            if root < scene.nodes.len() {
                compute_world(&mut scene.nodes, root, Mat4::IDENTITY);
            }
        }

        // Build mapping: glTF mesh index -> node world transform
        // (use the first node referencing each mesh)
        let mut mesh_to_world: std::collections::HashMap<usize, Mat4> =
            std::collections::HashMap::new();
        for node in &scene.nodes {
            if node.mesh_index >= 0 {
                let mi = node.mesh_index as usize;
                mesh_to_world.entry(mi).or_insert(node.world_transform);
            }
        }

        // Apply transforms to each splat primitive's data
        for prim_info in &splat_prim_infos {
            let world = match mesh_to_world.get(&prim_info.mesh_index) {
                Some(m) => *m,
                None => continue,
            };

            // Check if transform is identity (skip if so)
            let cols = world.to_cols_array();
            let identity_cols = Mat4::IDENTITY.to_cols_array();
            let is_identity = cols.iter().zip(identity_cols.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);
            if is_identity {
                continue;
            }

            info!(
                "Applying node transform to splat range [{}, {})",
                prim_info.start_splat,
                prim_info.start_splat + prim_info.splat_count
            );

            // Extract rotation matrix (upper-left 3x3) and scale from world matrix
            let col0 = Vec3::new(world.x_axis.x, world.x_axis.y, world.x_axis.z);
            let col1 = Vec3::new(world.y_axis.x, world.y_axis.y, world.y_axis.z);
            let col2 = Vec3::new(world.z_axis.x, world.z_axis.y, world.z_axis.z);
            let sx = col0.length();
            let sy = col1.length();
            let sz = col2.length();
            let uniform_scale = (sx * sy * sz).cbrt(); // geometric mean

            // Normalize rotation matrix (remove scale)
            let r0 = if sx > 1e-6 { col0 / sx } else { col0 };
            let r1 = if sy > 1e-6 { col1 / sy } else { col1 };
            let r2 = if sz > 1e-6 { col2 / sz } else { col2 };

            // Convert rotation matrix to quaternion (Shepperd's method)
            let trace = r0.x + r1.y + r2.z;
            let node_quat: [f32; 4]; // [x, y, z, w]
            if trace > 0.0 {
                let s = 0.5 / (trace + 1.0f32).sqrt();
                let w = 0.25 / s;
                let x = (r1.z - r2.y) * s;
                let y = (r2.x - r0.z) * s;
                let z = (r0.y - r1.x) * s;
                node_quat = [x, y, z, w];
            } else if r0.x > r1.y && r0.x > r2.z {
                let s = 2.0 * (1.0 + r0.x - r1.y - r2.z).sqrt();
                let w = (r1.z - r2.y) / s;
                let x = 0.25 * s;
                let y = (r1.x + r0.y) / s;
                let z = (r2.x + r0.z) / s;
                node_quat = [x, y, z, w];
            } else if r1.y > r2.z {
                let s = 2.0 * (1.0 + r1.y - r0.x - r2.z).sqrt();
                let w = (r2.x - r0.z) / s;
                let x = (r1.x + r0.y) / s;
                let y = 0.25 * s;
                let z = (r2.y + r1.z) / s;
                node_quat = [x, y, z, w];
            } else {
                let s = 2.0 * (1.0 + r2.z - r0.x - r1.y).sqrt();
                let w = (r0.y - r1.x) / s;
                let x = (r2.x + r0.z) / s;
                let y = (r2.y + r1.z) / s;
                let z = 0.25 * s;
                node_quat = [x, y, z, w];
            }
            // Normalize quaternion
            let qlen = (node_quat[0] * node_quat[0]
                + node_quat[1] * node_quat[1]
                + node_quat[2] * node_quat[2]
                + node_quat[3] * node_quat[3])
                .sqrt();
            let nq = if qlen > 1e-6 {
                [
                    node_quat[0] / qlen,
                    node_quat[1] / qlen,
                    node_quat[2] / qlen,
                    node_quat[3] / qlen,
                ]
            } else {
                node_quat
            };

            let end_splat = prim_info.start_splat + prim_info.splat_count;

            // Apply to positions: worldPos = world * vec4(pos, 1.0)
            for si in prim_info.start_splat..end_splat {
                if si >= scene.splat_data.positions.len() { break; }
                let pos = scene.splat_data.positions[si];
                let p = glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                let wp = world * p;
                scene.splat_data.positions[si] = [wp.x, wp.y, wp.z, 1.0];
            }

            // Apply to rotations: q_combined = q_node * q_splat (Hamilton product)
            for si in prim_info.start_splat..end_splat {
                if si >= scene.splat_data.rotations.len() { break; }
                let q = scene.splat_data.rotations[si];
                let qx = q[0]; let qy = q[1]; let qz = q[2]; let qw = q[3];
                scene.splat_data.rotations[si] = [
                    nq[3] * qx + nq[0] * qw + nq[1] * qz - nq[2] * qy,
                    nq[3] * qy - nq[0] * qz + nq[1] * qw + nq[2] * qx,
                    nq[3] * qz + nq[0] * qy - nq[1] * qx + nq[2] * qw,
                    nq[3] * qw - nq[0] * qx - nq[1] * qy - nq[2] * qz,
                ];
            }

            // Apply to scales: log_scale + log(uniform_scale)
            let log_scale = uniform_scale.ln();
            for si in prim_info.start_splat..end_splat {
                if si >= scene.splat_data.scales.len() { break; }
                scene.splat_data.scales[si][0] += log_scale;
                scene.splat_data.scales[si][1] += log_scale;
                scene.splat_data.scales[si][2] += log_scale;
            }
        }
    }

    // --- Cameras ---
    for cam in document.cameras() {
        let gcam = match cam.projection() {
            gltf::camera::Projection::Perspective(p) => GltfCamera {
                name: cam.name().unwrap_or("camera").to_string(),
                camera_type: "perspective".to_string(),
                fov_y: p.yfov(),
                aspect: p.aspect_ratio().unwrap_or(1.0),
                near: p.znear(),
                far: p.zfar().unwrap_or(1000.0),
                position: Vec3::ZERO,
                direction: Vec3::NEG_Z,
            },
            gltf::camera::Projection::Orthographic(o) => GltfCamera {
                name: cam.name().unwrap_or("camera").to_string(),
                camera_type: "orthographic".to_string(),
                fov_y: 0.0,
                aspect: o.xmag() / o.ymag(),
                near: o.znear(),
                far: o.zfar(),
                position: Vec3::ZERO,
                direction: Vec3::NEG_Z,
            },
        };
        scene.cameras.push(gcam);
    }

    // --- Lights (KHR_lights_punctual) ---
    if let Some(lights) = document.lights() {
        for light in lights {
            let glight = GltfLight {
                name: light.name().unwrap_or("light").to_string(),
                light_type: match light.kind() {
                    gltf::khr_lights_punctual::Kind::Directional => "directional",
                    gltf::khr_lights_punctual::Kind::Point => "point",
                    gltf::khr_lights_punctual::Kind::Spot { .. } => "spot",
                }
                .to_string(),
                color: light.color(),
                intensity: light.intensity(),
                range: light.range().unwrap_or(0.0),
                inner_cone_angle: match light.kind() {
                    gltf::khr_lights_punctual::Kind::Spot {
                        inner_cone_angle, ..
                    } => inner_cone_angle,
                    _ => 0.0,
                },
                outer_cone_angle: match light.kind() {
                    gltf::khr_lights_punctual::Kind::Spot {
                        outer_cone_angle, ..
                    } => outer_cone_angle,
                    _ => std::f32::consts::FRAC_PI_4,
                },
                position: Vec3::ZERO,
                direction: Vec3::NEG_Z,
            };
            scene.lights.push(glight);
        }
    }

    Ok(scene)
}
