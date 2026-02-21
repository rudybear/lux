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

pub struct GltfScene {
    pub meshes: Vec<GltfMesh>,
    pub materials: Vec<GltfMaterial>,
    pub nodes: Vec<GltfNode>,
    pub cameras: Vec<GltfCamera>,
    pub lights: Vec<GltfLight>,
    pub root_nodes: Vec<usize>,
}

pub struct DrawItem {
    pub world_transform: Mat4,
    pub mesh_index: usize,
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
            if mi < scene.meshes.len() {
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

/// Load a .glb or .gltf file.
pub fn load_gltf(path: &Path) -> Result<GltfScene, String> {
    let (document, buffers, images) =
        gltf::import(path).map_err(|e| format!("Failed to load glTF {}: {}", path.display(), e))?;

    info!("Loaded glTF: {} images, {} materials", images.len(), document.materials().len());

    let mut scene = GltfScene {
        meshes: Vec::new(),
        materials: Vec::new(),
        nodes: Vec::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        root_nodes: Vec::new(),
    };

    // --- Materials (with texture extraction) ---
    let all_textures: Vec<gltf::Texture> = document.textures().collect();
    for mat in document.materials() {
        let pbr = mat.pbr_metallic_roughness();
        let bc = pbr.base_color_factor();

        let base_color_image = extract_texture(&images, pbr.base_color_texture().map(|t| t.texture()));
        let normal_image = extract_texture(&images, mat.normal_texture().map(|t| t.texture()));
        let metallic_roughness_image = extract_texture(&images, pbr.metallic_roughness_texture().map(|t| t.texture()));
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
        });
    }
    if scene.materials.is_empty() {
        scene.materials.push(GltfMaterial::default());
    }

    // --- Meshes ---
    for mesh in document.meshes() {
        for prim in mesh.primitives() {
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

        scene.nodes.push(GltfNode {
            name: node.name().unwrap_or("unnamed").to_string(),
            local_transform,
            world_transform: Mat4::IDENTITY,
            mesh_index,
            camera_index,
            light_index: -1,
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
