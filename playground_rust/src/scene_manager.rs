//! Shared scene management: types, IBL loading, texture upload, auto-camera.
//!
//! Extracts duplicated code from `rt_renderer.rs` and `raster_renderer.rs`
//! into a single module. Both renderers use `SceneManager` for scene data
//! and implement the `Renderer` trait for unified rendering.

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use log::info;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use glam::Vec3;

use crate::camera::DefaultCamera;
use crate::screenshot;
use crate::vulkan_context::VulkanContext;

// ===========================================================================
// Scene light management
// ===========================================================================

#[derive(Debug, Clone)]
pub struct SceneLight {
    pub light_type: u32, // 0=directional, 1=point, 2=spot
    pub position: Vec3,
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cone_angle: f32,
    pub outer_cone_angle: f32,
    pub casts_shadow: bool,
    pub shadow_index: i32,
}

impl Default for SceneLight {
    fn default() -> Self {
        Self {
            light_type: 0,
            position: Vec3::ZERO,
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::ONE,
            intensity: 1.0,
            range: 0.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: std::f32::consts::FRAC_PI_4,
            casts_shadow: false,
            shadow_index: -1,
        }
    }
}

/// Shadow map entry: stores the view-projection matrix and bias parameters
/// for a single shadow-casting light.
#[derive(Debug, Clone)]
pub struct ShadowEntry {
    pub view_projection: [f32; 16], // mat4 in column-major
    pub bias: f32,
    pub normal_bias: f32,
    pub resolution: f32,
    pub light_size: f32,
}

impl Default for ShadowEntry {
    fn default() -> Self {
        Self {
            view_projection: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            bias: 0.005,
            normal_bias: 0.02,
            resolution: 1024.0,
            light_size: 0.02,
        }
    }
}

/// Maximum number of shadow-casting lights supported.
pub const MAX_SHADOW_MAPS: usize = 8;

/// Shadow map resolution (width and height).
pub const SHADOW_MAP_RESOLUTION: u32 = 2048;

/// Placeholder for a future gaussian splat renderer.
///
/// When a glTF scene contains KHR_gaussian_splatting data, the scene manager
/// can hold an optional `SplatRenderer` to handle the splat render path.
pub struct SplatRenderer {
    pub num_splats: u32,
    pub sh_degree: u32,
    // Future: GPU buffers for positions, rotations, scales, opacities, SH coeffs
}

impl SplatRenderer {
    /// Create a splat renderer from loaded gaussian splat data.
    pub fn new(splat_data: &crate::gltf_loader::GaussianSplatData) -> Self {
        info!(
            "SplatRenderer: initialized with {} splats, SH degree {}",
            splat_data.num_splats, splat_data.sh_degree
        );
        Self {
            num_splats: splat_data.num_splats,
            sh_degree: splat_data.sh_degree,
        }
    }
}

/// Central scene manager: holds lights and provides buffer packing for GPU upload.
pub struct SceneManager {
    pub lights: Vec<SceneLight>,
    pub shadow_entries: Vec<ShadowEntry>,
    /// Optional gaussian splat renderer, initialized when the scene contains
    /// KHR_gaussian_splatting data.
    pub splat_renderer: Option<SplatRenderer>,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            lights: Vec::new(),
            shadow_entries: Vec::new(),
            splat_renderer: None,
        }
    }

    /// Initialize the splat renderer if the loaded glTF scene has gaussian splat data.
    pub fn init_splat_renderer_if_needed(&mut self, scene: &crate::gltf_loader::GltfScene) {
        if scene.splat_data.has_splats {
            self.splat_renderer = Some(SplatRenderer::new(&scene.splat_data));
        }
    }

    /// Returns true if this scene manager has an active splat renderer.
    pub fn has_splat_renderer(&self) -> bool {
        self.splat_renderer.is_some()
    }

    /// Pack all lights into a flat f32 buffer for GPU SSBO upload.
    ///
    /// Each light occupies 4 vec4s (16 floats = 64 bytes):
    ///   vec4: (light_type, intensity, range, inner_cone)
    ///   vec4: (position.xyz, outer_cone)
    ///   vec4: (direction.xyz, shadow_index)
    ///   vec4: (color.xyz, pad)
    pub fn pack_lights_buffer(&self) -> Vec<f32> {
        let mut buf = Vec::new();
        for l in &self.lights {
            // vec4: (light_type, intensity, range, inner_cone)
            buf.push(l.light_type as f32);
            buf.push(l.intensity);
            buf.push(l.range);
            buf.push(l.inner_cone_angle);
            // vec4: (position.xyz, outer_cone)
            buf.push(l.position.x);
            buf.push(l.position.y);
            buf.push(l.position.z);
            buf.push(l.outer_cone_angle);
            // vec4: (direction.xyz, shadow_index)
            buf.push(l.direction.x);
            buf.push(l.direction.y);
            buf.push(l.direction.z);
            buf.push(l.shadow_index as f32);
            // vec4: (color.xyz, pad)
            buf.push(l.color.x);
            buf.push(l.color.y);
            buf.push(l.color.z);
            buf.push(0.0); // pad
        }
        buf
    }

    /// Populate lights from a glTF scene's extracted light data.
    ///
    /// If the scene has no lights, a default directional light is added.
    pub fn populate_lights_from_gltf(&mut self, gltf_lights: &[crate::gltf_loader::GltfLight]) {
        self.lights.clear();
        for gl in gltf_lights {
            let light_type = match gl.light_type.as_str() {
                "directional" => 0,
                "point" => 1,
                "spot" => 2,
                _ => 0,
            };
            self.lights.push(SceneLight {
                light_type,
                position: gl.position,
                direction: gl.direction,
                color: Vec3::new(gl.color[0], gl.color[1], gl.color[2]),
                intensity: gl.intensity,
                range: gl.range,
                inner_cone_angle: gl.inner_cone_angle,
                outer_cone_angle: gl.outer_cone_angle,
                ..Default::default()
            });
        }
        // Add default directional light if scene has none
        if self.lights.is_empty() {
            self.lights.push(SceneLight {
                light_type: 0,
                direction: Vec3::new(1.0, 0.8, 0.6).normalize(),
                color: Vec3::new(1.0, 0.98, 0.95),
                intensity: 1.0,
                ..Default::default()
            });
        }
        info!(
            "SceneManager: {} light(s) populated",
            self.lights.len()
        );
    }

    /// Replace current lights with the given demo lights.
    pub fn override_lights(&mut self, demo_lights: Vec<SceneLight>) {
        self.lights = demo_lights;
        info!(
            "SceneManager: overridden with {} demo light(s)",
            self.lights.len()
        );
    }

    /// Compute shadow view-projection matrices for all shadow-capable lights.
    ///
    /// For directional lights: builds an orthographic projection sized to the
    /// camera frustum. For spot lights: builds a perspective projection from the
    /// light's position using its cone angle and range.
    ///
    /// Assigns `shadow_index` on each light (-1 if no shadow).
    /// Stores results in `self.shadow_entries`.
    pub fn compute_shadow_data(
        &mut self,
        camera_view: &[f32; 16],
        camera_proj: &[f32; 16],
        _near: f32,
        _far: f32,
    ) {
        use glam::{Mat4, Vec3, Vec4};

        self.shadow_entries.clear();
        let mut shadow_idx: i32 = 0;

        // Parse camera matrices (column-major)
        let cam_view = Mat4::from_cols_array(camera_view);
        let cam_proj = Mat4::from_cols_array(camera_proj);
        let cam_inv_vp = (cam_proj * cam_view).inverse();

        // Compute camera frustum corners in world space (NDC cube corners)
        let ndc_corners: [[f32; 3]; 8] = [
            [-1.0, -1.0, 0.0], [ 1.0, -1.0, 0.0], [-1.0,  1.0, 0.0], [ 1.0,  1.0, 0.0],
            [-1.0, -1.0, 1.0], [ 1.0, -1.0, 1.0], [-1.0,  1.0, 1.0], [ 1.0,  1.0, 1.0],
        ];
        let mut frustum_ws: Vec<Vec3> = Vec::with_capacity(8);
        for ndc in &ndc_corners {
            let clip = cam_inv_vp * Vec4::new(ndc[0], ndc[1], ndc[2], 1.0);
            if clip.w.abs() > 1e-8 {
                frustum_ws.push(Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w));
            }
        }

        // Compute frustum center and radius for directional light sizing
        let frustum_center = if !frustum_ws.is_empty() {
            let sum: Vec3 = frustum_ws.iter().copied().sum();
            sum / frustum_ws.len() as f32
        } else {
            Vec3::ZERO
        };

        let frustum_radius = frustum_ws.iter()
            .map(|p| (*p - frustum_center).length())
            .fold(0.0f32, f32::max);

        for light in self.lights.iter_mut() {
            if shadow_idx as usize >= MAX_SHADOW_MAPS {
                light.shadow_index = -1;
                continue;
            }

            // Only directional and spot lights cast shadows
            match light.light_type {
                0 => {
                    // Directional light: orthographic projection
                    let light_dir = light.direction.normalize();
                    // Build a stable up vector
                    let up = if light_dir.y.abs() > 0.99 {
                        Vec3::new(0.0, 0.0, 1.0)
                    } else {
                        Vec3::new(0.0, 1.0, 0.0)
                    };
                    let light_right = light_dir.cross(up).normalize();
                    let light_up = light_right.cross(light_dir).normalize();

                    // Light view: looking along light_dir from behind the frustum
                    let light_pos = frustum_center - light_dir * frustum_radius * 2.0;

                    let light_view = Mat4::from_cols(
                        Vec4::new(light_right.x, light_up.x, -light_dir.x, 0.0),
                        Vec4::new(light_right.y, light_up.y, -light_dir.y, 0.0),
                        Vec4::new(light_right.z, light_up.z, -light_dir.z, 0.0),
                        Vec4::new(
                            -light_right.dot(light_pos),
                            -light_up.dot(light_pos),
                            light_dir.dot(light_pos),
                            1.0,
                        ),
                    );

                    // Ortho projection sized to frustum
                    let extent = frustum_radius.max(1.0);
                    let ortho_near = 0.0f32;
                    let ortho_far = frustum_radius * 4.0;

                    // Column-major orthographic projection (Vulkan depth [0,1])
                    let light_proj = Mat4::from_cols(
                        Vec4::new(1.0 / extent, 0.0, 0.0, 0.0),
                        Vec4::new(0.0, -1.0 / extent, 0.0, 0.0), // Y-flip for Vulkan
                        Vec4::new(0.0, 0.0, 1.0 / (ortho_far - ortho_near), 0.0),
                        Vec4::new(0.0, 0.0, -ortho_near / (ortho_far - ortho_near), 1.0),
                    );

                    let vp = light_proj * light_view;
                    light.shadow_index = shadow_idx;
                    light.casts_shadow = true;

                    self.shadow_entries.push(ShadowEntry {
                        view_projection: vp.to_cols_array(),
                        bias: 0.005,
                        normal_bias: 0.02,
                        resolution: SHADOW_MAP_RESOLUTION as f32,
                        light_size: 0.02,
                    });
                    shadow_idx += 1;
                }
                2 => {
                    // Spot light: perspective projection from light position
                    let light_dir = light.direction.normalize();
                    let up = if light_dir.y.abs() > 0.99 {
                        Vec3::new(0.0, 0.0, 1.0)
                    } else {
                        Vec3::new(0.0, 1.0, 0.0)
                    };
                    let light_right = light_dir.cross(up).normalize();
                    let light_up = light_right.cross(light_dir).normalize();

                    let light_view = Mat4::from_cols(
                        Vec4::new(light_right.x, light_up.x, -light_dir.x, 0.0),
                        Vec4::new(light_right.y, light_up.y, -light_dir.y, 0.0),
                        Vec4::new(light_right.z, light_up.z, -light_dir.z, 0.0),
                        Vec4::new(
                            -light_right.dot(light.position),
                            -light_up.dot(light.position),
                            light_dir.dot(light.position),
                            1.0,
                        ),
                    );

                    // Use the outer cone angle for the FOV
                    let fov = light.outer_cone_angle * 2.0;
                    let fov = fov.max(0.1); // Clamp to avoid degenerate
                    let spot_near = 0.1f32;
                    let spot_far = if light.range > 0.0 { light.range } else { 100.0 };
                    let f_val = 1.0 / (fov / 2.0).tan();

                    // Vulkan perspective (Y-flip, depth [0,1])
                    let light_proj = Mat4::from_cols(
                        Vec4::new(f_val, 0.0, 0.0, 0.0),
                        Vec4::new(0.0, -f_val, 0.0, 0.0), // Y-flip
                        Vec4::new(0.0, 0.0, spot_far / (spot_near - spot_far), -1.0),
                        Vec4::new(0.0, 0.0, (spot_near * spot_far) / (spot_near - spot_far), 0.0),
                    );

                    let vp = light_proj * light_view;
                    light.shadow_index = shadow_idx;
                    light.casts_shadow = true;

                    self.shadow_entries.push(ShadowEntry {
                        view_projection: vp.to_cols_array(),
                        bias: 0.005,
                        normal_bias: 0.02,
                        resolution: SHADOW_MAP_RESOLUTION as f32,
                        light_size: 0.02,
                    });
                    shadow_idx += 1;
                }
                _ => {
                    // Point lights don't cast shadows in this implementation
                    light.shadow_index = -1;
                }
            }
        }

        info!(
            "SceneManager: computed {} shadow map(s) for {} light(s)",
            self.shadow_entries.len(),
            self.lights.len()
        );
    }

    /// Pack all shadow entries into a flat f32 buffer for GPU SSBO upload.
    ///
    /// Each entry occupies 20 floats (80 bytes):
    ///   16 floats: mat4 viewProjection (column-major)
    ///   4 floats: bias, normalBias, resolution, light_size
    pub fn pack_shadow_buffer(&self) -> Vec<f32> {
        let mut buf = Vec::with_capacity(self.shadow_entries.len() * 20);
        for entry in &self.shadow_entries {
            // 16 floats for mat4
            buf.extend_from_slice(&entry.view_projection);
            // 4 floats: bias, normal_bias, resolution, light_size
            buf.push(entry.bias);
            buf.push(entry.normal_bias);
            buf.push(entry.resolution);
            buf.push(entry.light_size);
        }
        buf
    }

    /// Number of active shadow maps.
    pub fn shadow_count(&self) -> usize {
        self.shadow_entries.len()
    }
}

// ===========================================================================
// Shared GPU resource types
// ===========================================================================

/// GPU buffer with its allocation. Defined once and used by both renderers.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
}

impl GpuBuffer {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

/// GPU image with its allocation.
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
}

impl GpuImage {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
    }
}

/// IBL cubemap texture with its associated resources.
pub struct IblTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub allocation: Option<Allocation>,
}

impl IblTexture {
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
        unsafe {
            device.destroy_image(self.image, None);
        }
    }
}

/// Loaded IBL assets: specular cubemap, irradiance cubemap, BRDF LUT.
pub struct IblAssets {
    pub env_specular: Option<IblTexture>,
    pub env_irradiance: Option<IblTexture>,
    pub brdf_lut: Option<IblTexture>,
}

impl IblAssets {
    pub fn empty() -> Self {
        Self {
            env_specular: None,
            env_irradiance: None,
            brdf_lut: None,
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(ref mut tex) = self.env_specular {
            tex.destroy(device, allocator);
        }
        if let Some(ref mut tex) = self.env_irradiance {
            tex.destroy(device, allocator);
        }
        if let Some(ref mut tex) = self.brdf_lut {
            tex.destroy(device, allocator);
        }
    }

    /// Get the image view for a named IBL binding (if available).
    pub fn view_for_binding(&self, name: &str) -> Option<vk::ImageView> {
        match name {
            "env_specular" => self.env_specular.as_ref().map(|t| t.view),
            "env_irradiance" => self.env_irradiance.as_ref().map(|t| t.view),
            "brdf_lut" => self.brdf_lut.as_ref().map(|t| t.view),
            _ => None,
        }
    }

    /// Get the sampler for a named IBL binding (if available).
    pub fn sampler_for_binding(&self, name: &str) -> Option<vk::Sampler> {
        match name {
            "env_specular" => self.env_specular.as_ref().map(|t| t.sampler),
            "env_irradiance" => self.env_irradiance.as_ref().map(|t| t.sampler),
            "brdf_lut" => self.brdf_lut.as_ref().map(|t| t.sampler),
            _ => None,
        }
    }
}

/// IBL manifest (parsed from manifest.json).
#[derive(serde::Deserialize, Default)]
pub struct IblManifest {
    #[serde(default)]
    pub specular_face_size: u32,
    #[serde(default)]
    pub specular_mip_count: u32,
    #[serde(default)]
    pub irradiance_face_size: u32,
    #[serde(default)]
    pub brdf_lut_size: u32,
    #[serde(default)]
    pub specular: Option<IblSpecularManifest>,
    #[serde(default)]
    pub irradiance: Option<IblIrradianceManifest>,
    #[serde(default)]
    pub brdf_lut: Option<IblBrdfLutManifest>,
}

#[derive(serde::Deserialize, Default)]
pub struct IblSpecularManifest {
    #[serde(default)]
    pub face_size: u32,
    #[serde(default)]
    pub mip_count: u32,
}

#[derive(serde::Deserialize, Default)]
pub struct IblIrradianceManifest {
    #[serde(default)]
    pub face_size: u32,
}

#[derive(serde::Deserialize, Default)]
pub struct IblBrdfLutManifest {
    #[serde(default)]
    pub size: u32,
}

impl IblManifest {
    pub fn spec_face_size(&self) -> u32 {
        if self.specular_face_size > 0 {
            self.specular_face_size
        } else if let Some(ref s) = self.specular {
            if s.face_size > 0 { s.face_size } else { 256 }
        } else {
            256
        }
    }

    pub fn spec_mip_count(&self) -> u32 {
        if self.specular_mip_count > 0 {
            self.specular_mip_count
        } else if let Some(ref s) = self.specular {
            if s.mip_count > 0 { s.mip_count } else { 5 }
        } else {
            5
        }
    }

    pub fn irr_face_size(&self) -> u32 {
        if self.irradiance_face_size > 0 {
            self.irradiance_face_size
        } else if let Some(ref i) = self.irradiance {
            if i.face_size > 0 { i.face_size } else { 32 }
        } else {
            32
        }
    }

    pub fn lut_size(&self) -> u32 {
        if self.brdf_lut_size > 0 {
            self.brdf_lut_size
        } else if let Some(ref b) = self.brdf_lut {
            if b.size > 0 { b.size } else { 512 }
        } else {
            512
        }
    }
}

// ===========================================================================
// Renderer trait
// ===========================================================================

/// Unified renderer trait implemented by both RTRenderer and PersistentRenderer.
///
/// Allows main.rs to handle both render paths through dynamic dispatch.
pub trait Renderer {
    /// Render a frame. For headless, this renders once. For interactive, call per frame.
    fn render(&mut self, ctx: &VulkanContext) -> Result<vk::CommandBuffer, String>;

    /// Update lights for animation. Default: no-op.
    fn update_lights(&mut self, _lights: &[SceneLight]) {}

    /// Update the camera from orbit parameters.
    fn update_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
    );

    /// Blit the rendered output to a swapchain image. The command buffer should be recording.
    fn blit_to_swapchain(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        swap_image: vk::Image,
        extent: vk::Extent2D,
    );

    /// The output image (for screenshot readback).
    fn output_image(&self) -> vk::Image;

    /// The output image format.
    fn output_format(&self) -> vk::Format;

    /// Render width.
    fn width(&self) -> u32;

    /// Render height.
    fn height(&self) -> u32;

    /// Whether this renderer has auto-camera data from scene bounds.
    fn has_scene_bounds(&self) -> bool;

    /// Auto-camera eye position.
    fn auto_eye(&self) -> glam::Vec3;

    /// Auto-camera target position.
    fn auto_target(&self) -> glam::Vec3;

    /// Auto-camera up vector.
    fn auto_up(&self) -> glam::Vec3;

    /// Auto-camera far plane.
    fn auto_far(&self) -> f32;

    /// The wait stage flags for swapchain submission.
    fn wait_stage(&self) -> vk::PipelineStageFlags;

    /// Destroy all GPU resources.
    fn destroy(&mut self, ctx: &mut VulkanContext);
}

// ===========================================================================
// Shared utility functions
// ===========================================================================

/// Check whether a scene source string refers to a glTF file.
pub fn is_gltf_file(scene_source: &str) -> bool {
    scene_source.ends_with(".glb") || scene_source.ends_with(".gltf")
}

/// Detect which material features are used across the scene.
pub fn detect_scene_features(scene: &crate::gltf_loader::GltfScene, lights: &[SceneLight]) -> std::collections::BTreeSet<String> {
    let mut features = std::collections::BTreeSet::new();
    for mat in &scene.materials {
        if mat.normal_image.is_some() {
            features.insert("has_normal_map".to_string());
        }
        if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
            features.insert("has_emission".to_string());
        }
        if mat.has_clearcoat {
            features.insert("has_clearcoat".to_string());
        }
        if mat.has_sheen {
            features.insert("has_sheen".to_string());
        }
        if mat.has_transmission {
            features.insert("has_transmission".to_string());
        }
    }
    // Detect shadow support
    if lights.iter().any(|l| l.casts_shadow) {
        features.insert("has_shadows".to_string());
    }
    // Detect gaussian splat data
    if scene.splat_data.has_splats {
        features.insert("has_gaussian_splats".to_string());
    }
    if !features.is_empty() {
        info!("Detected material features: {:?}", features);
    }
    features
}

/// Build a pipeline path from detected features.
pub fn build_pipeline_path(base_path: &str, features: &std::collections::BTreeSet<String>) -> String {
    if features.is_empty() {
        return base_path.to_string();
    }
    let suffix: Vec<String> = features.iter().map(|f| f.strip_prefix("has_").unwrap_or(f).to_string()).collect();
    let candidate = format!("shadercache/gltf_pbr_layered+{}", suffix.join("+"));
    let frag = format!("{}.frag.spv", candidate);
    let rgen = format!("{}.rgen.spv", candidate);
    if std::path::Path::new(&frag).exists() || std::path::Path::new(&rgen).exists() {
        info!("Auto-selected pipeline variant: {}", candidate);
        return candidate;
    }
    base_path.to_string()
}

/// Detect features for a single material (not the whole scene).
///
/// Returns a set of feature flags like "has_normal_map", "has_sheen", etc.
pub fn detect_material_features(scene: &crate::gltf_loader::GltfScene, material_index: usize, lights: &[SceneLight]) -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    if material_index >= scene.materials.len() {
        return features;
    }
    let mat = &scene.materials[material_index];
    if mat.normal_image.is_some() {
        features.insert("has_normal_map".to_string());
    }
    if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
        features.insert("has_emission".to_string());
    }
    if mat.has_clearcoat {
        features.insert("has_clearcoat".to_string());
    }
    if mat.has_sheen {
        features.insert("has_sheen".to_string());
    }
    if mat.has_transmission {
        features.insert("has_transmission".to_string());
    }
    // Detect shadow support: if any light casts shadows, enable has_shadows
    if lights.iter().any(|l| l.casts_shadow) {
        features.insert("has_shadows".to_string());
    }
    features
}

/// Convert a feature set to a permutation suffix string like "+normal_map+sheen".
///
/// Features are sorted (BTreeSet is already sorted), "has_" prefix is stripped.
/// Empty set returns "".
pub fn features_to_suffix(features: &BTreeSet<String>) -> String {
    if features.is_empty() {
        return String::new();
    }
    let mut suffix = String::new();
    for f in features {
        suffix.push('+');
        suffix.push_str(f.strip_prefix("has_").unwrap_or(f));
    }
    suffix
}

/// Group materials by feature set, returning a map of suffix -> [material_indices].
///
/// Each unique combination of features gets its own permutation suffix.
pub fn group_materials_by_features(scene: &crate::gltf_loader::GltfScene, lights: &[SceneLight]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..scene.materials.len() {
        let feats = detect_material_features(scene, i, lights);
        let suffix = features_to_suffix(&feats);
        groups.entry(suffix).or_default().push(i);
    }

    info!("Material permutation groups:");
    for (suffix, indices) in &groups {
        let label = if suffix.is_empty() { "(base)" } else { suffix.as_str() };
        info!("  \"{}\": materials {:?}", label, indices);
    }

    groups
}

/// Compute auto-camera from scene bounds using node transforms (for glTF scenes).
///
/// This code is shared between RT and raster renderers. Both use the same
/// node-transform-based camera placement algorithm.
pub fn compute_auto_camera_from_draw_items(
    vertices_positions: &[[f32; 3]],
    draw_items: &[crate::gltf_loader::DrawItem],
) -> (glam::Vec3, glam::Vec3, glam::Vec3, f32) {
    let mut bbox_min = glam::Vec3::splat(f32::MAX);
    let mut bbox_max = glam::Vec3::splat(f32::MIN);
    for pos in vertices_positions {
        bbox_min = bbox_min.min(glam::Vec3::from(*pos));
        bbox_max = bbox_max.max(glam::Vec3::from(*pos));
    }
    let center = (bbox_min + bbox_max) * 0.5;
    let extent = bbox_max - bbox_min;

    // Extract camera direction from first draw item's world transform
    let mut cam_dir = glam::Vec3::new(0.0, 0.0, 1.0);
    let mut cam_up_vec = glam::Vec3::new(0.0, 1.0, 0.0);
    let mut used_blender_dir = false;

    if !draw_items.is_empty() {
        let upper = glam::Mat3::from_mat4(draw_items[0].world_transform);
        let row_norms = glam::Vec3::new(
            upper.row(0).length(),
            upper.row(1).length(),
            upper.row(2).length(),
        );
        let r_approx = glam::Mat3::from_cols(
            upper.col(0) / row_norms.x.max(1e-8),
            upper.col(1) / row_norms.y.max(1e-8),
            upper.col(2) / row_norms.z.max(1e-8),
        );
        // Blender front (-Y local) and up (+Z local) in world space
        let front = r_approx.transpose() * glam::Vec3::new(0.0, -1.0, 0.0);
        let up_w = r_approx.transpose() * glam::Vec3::new(0.0, 0.0, 1.0);

        let fl = front.length();
        let ul = up_w.length();
        if fl > 0.5 && ul > 0.5 {
            let candidate_dir = front / fl;
            let candidate_up = up_w / ul;
            if candidate_dir.y.abs() < 0.9 {
                cam_dir = candidate_dir;
                cam_up_vec = candidate_up;
                if cam_dir.dot(cam_up_vec).abs() > 0.9 {
                    cam_up_vec = glam::Vec3::new(0.0, 1.0, 0.0);
                }
                used_blender_dir = true;
            }
        }
    }

    // Fallback: look along the longest horizontal extent of the bounding box
    if !used_blender_dir {
        let ex = extent.x.abs();
        let ez = extent.z.abs();
        cam_dir = if ex >= ez {
            glam::Vec3::new(1.0, 0.0, 0.0)
        } else {
            glam::Vec3::new(0.0, 0.0, 1.0)
        };
        cam_up_vec = glam::Vec3::new(0.0, 1.0, 0.0);
    }

    // Perpendicular extent via projection
    let view_right = (-cam_dir).cross(cam_up_vec);
    let vr_len = view_right.length();
    let (view_right, cam_up_vec) = if vr_len < 1e-6 {
        let cu = glam::Vec3::new(0.0, 0.0, 1.0);
        let vr = (-cam_dir).cross(cu);
        (vr / vr.length().max(1e-8), cu)
    } else {
        (view_right / vr_len, cam_up_vec)
    };
    let view_up = view_right.cross(-cam_dir).normalize();

    let proj_right = view_right.abs().dot(extent.abs());
    let proj_up = view_up.abs().dot(extent.abs());
    let max_perp_extent = proj_right.max(proj_up);

    let fov_y = DefaultCamera::FOV_Y_DEG.to_radians();
    let distance = (max_perp_extent / 2.0) / (fov_y / 2.0).tan() * 1.1;

    let elev_rad = 5.0f32.to_radians();
    let eye = center + distance * cam_dir + distance * elev_rad.sin() * view_up;

    let target = center;
    let up = cam_up_vec;
    let far = (distance * 3.0f32).max(100.0);

    info!(
        "Auto-camera: eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2}) distance={:.2}",
        eye.x, eye.y, eye.z, target.x, target.y, target.z, distance
    );

    (eye, target, up, far)
}

// ===========================================================================
// Buffer creation helpers
// ===========================================================================

/// Create a GPU buffer, upload data, and return the buffer + allocation.
pub fn create_buffer_with_data(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[u8],
    usage: vk::BufferUsageFlags,
    name: &str,
) -> Result<GpuBuffer, String> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&buffer_info, None)
            .map_err(|e| format!("Failed to create buffer '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mut allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate memory for '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind buffer memory '{}': {:?}", name, e))?;
    }

    // Copy data
    if let Some(mapped) = allocation.mapped_slice_mut() {
        mapped[..data.len()].copy_from_slice(data);
    } else {
        return Err(format!("Buffer '{}' is not host-visible", name));
    }

    Ok(GpuBuffer {
        buffer,
        allocation: Some(allocation),
    })
}

/// Create an offscreen render target image.
pub fn create_offscreen_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    name: &str,
) -> Result<GpuImage, String> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create image '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate image memory '{}': {:?}", name, e))?;

    unsafe {
        device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind image memory '{}': {:?}", name, e))?;
    }

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create image view '{}': {:?}", name, e))?
    };

    Ok(GpuImage {
        image,
        view,
        allocation: Some(allocation),
    })
}

// ===========================================================================
// Texture upload helpers (parameterized by dst_stage)
// ===========================================================================

/// Create a texture image from pixel data and upload it via a staging buffer.
///
/// `dst_stage` controls the final pipeline barrier:
/// - For raster: `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`
/// - For RT: `VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR`
pub fn create_texture_image(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixels: &[u8],
    name: &str,
    dst_stage: vk::PipelineStageFlags,
) -> Result<GpuImage, String> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let data_size = (width * height * 4) as u64;

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(data_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        ctx.device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { ctx.device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "texture_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate staging memory: {:?}", e))?;

    unsafe {
        ctx.device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixels.len()].copy_from_slice(pixels);
    }

    // Create GPU image
    let device_clone = ctx.device.clone();
    let image = create_offscreen_image(
        &device_clone,
        ctx.allocator_mut(),
        width,
        height,
        format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        name,
    )?;

    // Copy staging -> image
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        dst_stage,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok(image)
}

/// Create a 2D image from raw pixel data (arbitrary format) and upload via staging.
///
/// Used for BRDF LUT (float16 RGBA).
pub fn create_texture_image_raw(
    ctx: &mut VulkanContext,
    width: u32,
    height: u32,
    pixel_data: &[u8],
    format: vk::Format,
    name: &str,
    dst_stage: vk::PipelineStageFlags,
) -> Result<GpuImage, String> {
    let data_size = pixel_data.len() as u64;

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(data_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        ctx.device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { ctx.device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "texture_raw_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate staging memory: {:?}", e))?;

    unsafe {
        ctx.device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixel_data.len()].copy_from_slice(pixel_data);
    }

    // Create GPU image
    let device_clone = ctx.device.clone();
    let image = create_offscreen_image(
        &device_clone,
        ctx.allocator_mut(),
        width,
        height,
        format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        name,
    )?;

    // Copy staging -> image
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(),
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
    );

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D::default())
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    screenshot::cmd_transition_image(
        &ctx.device,
        cmd,
        image.image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER,
        dst_stage,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    Ok(image)
}

// ===========================================================================
// Cubemap creation (parameterized by dst_stage)
// ===========================================================================

/// Create a cubemap VkImage with CUBE_COMPATIBLE flag, upload data via staging, create CUBE view.
///
/// `dst_stage` controls the final pipeline barrier.
pub fn create_cubemap_image(
    ctx: &mut VulkanContext,
    face_size: u32,
    mip_count: u32,
    format: vk::Format,
    pixel_data: &[u8],
    bytes_per_pixel: u32,
    name: &str,
    dst_stage: vk::PipelineStageFlags,
) -> Result<(vk::Image, vk::ImageView, Allocation), String> {
    let device_clone = ctx.device.clone();
    let device = &device_clone;

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: face_size,
            height: face_size,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);

    let image = unsafe {
        device
            .create_image(&image_info, None)
            .map_err(|e| format!("Failed to create cubemap image '{}': {:?}", name, e))?
    };

    let requirements = unsafe { device.get_image_memory_requirements(image) };

    let allocation = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate cubemap image memory '{}': {:?}", name, e))?;

    unsafe {
        ctx.device
            .bind_image_memory(image, allocation.memory(), allocation.offset())
            .map_err(|e| format!("Failed to bind cubemap image memory '{}': {:?}", name, e))?;
    }

    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::default()
        .size(pixel_data.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe {
        ctx.device
            .create_buffer(&staging_info, None)
            .map_err(|e| format!("Failed to create cubemap staging buffer: {:?}", e))?
    };

    let staging_reqs = unsafe { ctx.device.get_buffer_memory_requirements(staging_buffer) };

    let mut staging_alloc = ctx
        .allocator_mut()
        .allocate(&AllocationCreateDesc {
            name: "cubemap_staging",
            requirements: staging_reqs,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| format!("Failed to allocate cubemap staging memory: {:?}", e))?;

    unsafe {
        ctx.device
            .bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())
            .map_err(|e| format!("Failed to bind cubemap staging memory: {:?}", e))?;
    }

    if let Some(mapped) = staging_alloc.mapped_slice_mut() {
        mapped[..pixel_data.len()].copy_from_slice(pixel_data);
    }

    // Transition image and copy all mip levels / faces
    let cmd = ctx.begin_single_commands()?;

    screenshot::cmd_transition_image_layers(
        &ctx.device, cmd, image,
        vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER,
        mip_count, 6,
    );

    // Copy each mip level and face from the staging buffer
    let mut buffer_offset: u64 = 0;
    let mut regions: Vec<vk::BufferImageCopy> = Vec::new();

    for mip in 0..mip_count {
        let mip_size = std::cmp::max(face_size >> mip, 1);
        let face_bytes = (mip_size * mip_size * bytes_per_pixel) as u64;

        for face in 0..6u32 {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .base_array_layer(face)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: mip_size,
                        height: mip_size,
                        depth: 1,
                    }),
            );
            buffer_offset += face_bytes;
        }
    }

    unsafe {
        ctx.device.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }

    // Transition to SHADER_READ_ONLY_OPTIMAL
    screenshot::cmd_transition_image_layers(
        &ctx.device, cmd, image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::TRANSFER, dst_stage,
        mip_count, 6,
    );

    ctx.end_single_commands(cmd)?;

    // Cleanup staging
    let _ = ctx.allocator_mut().free(staging_alloc);
    unsafe {
        ctx.device.destroy_buffer(staging_buffer, None);
    }

    // Create CUBE image view
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(6),
        );

    let view = unsafe {
        ctx.device
            .create_image_view(&view_info, None)
            .map_err(|e| format!("Failed to create cubemap image view '{}': {:?}", name, e))?
    };

    Ok((image, view, allocation))
}

// ===========================================================================
// IBL loading (parameterized by dst_stage)
// ===========================================================================

/// Try to find and load IBL assets from assets/ibl/<name>/ directories.
///
/// `dst_stage` controls the final pipeline barrier for all uploaded textures.
pub fn load_ibl_assets(ctx: &mut VulkanContext, dst_stage: vk::PipelineStageFlags, ibl_name: &str) -> IblAssets {
    // Search for IBL asset directories
    let ibl_base = Path::new("assets/ibl");
    if !ibl_base.exists() {
        info!("IBL assets directory not found at {:?}, using defaults", ibl_base);
        return IblAssets::empty();
    }

    // If a specific IBL name was requested, try it first
    let requested = if !ibl_name.is_empty() {
        let path = ibl_base.join(ibl_name);
        if path.is_dir() && path.join("manifest.json").exists() {
            info!("Using requested IBL: {}", ibl_name);
            Some(path)
        } else {
            info!("Requested IBL '{}' not found, falling back to auto-detect", ibl_name);
            None
        }
    } else {
        None
    };

    // Find IBL directory: prefer requested, then "pisa" then "neutral", matching C++ engine
    let ibl_dir = {
        let mut found = requested;
        if found.is_none() {
            let preferred = ["pisa", "neutral"];
            for name in &preferred {
                let path = ibl_base.join(name);
                if path.is_dir() && path.join("manifest.json").exists() {
                    found = Some(path);
                    break;
                }
            }
        }
        // Fall back to alphabetical
        if found.is_none() {
            if let Ok(entries) = std::fs::read_dir(ibl_base) {
                let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
                sorted_entries.sort_by_key(|e| e.file_name());
                for entry in sorted_entries {
                    let path = entry.path();
                    if path.is_dir() && path.join("manifest.json").exists() {
                        found = Some(path);
                        break;
                    }
                }
            }
        }
        match found {
            Some(dir) => dir,
            None => {
                info!("No IBL manifest.json found in {:?}", ibl_base);
                return IblAssets::empty();
            }
        }
    };

    info!("Loading IBL assets from {:?}", ibl_dir);

    // Parse manifest
    let manifest_path = ibl_dir.join("manifest.json");
    let manifest: IblManifest = match std::fs::read_to_string(&manifest_path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => {
            info!("Failed to read IBL manifest");
            return IblAssets::empty();
        }
    };

    let mut assets = IblAssets::empty();

    // Load specular cubemap
    let spec_path = ibl_dir.join("specular.bin");
    if spec_path.exists() {
        match load_specular_cubemap(ctx, &spec_path, &manifest, dst_stage) {
            Ok(tex) => {
                info!("Loaded specular cubemap: {}x{}, {} mips",
                    manifest.spec_face_size(), manifest.spec_face_size(), manifest.spec_mip_count());
                assets.env_specular = Some(tex);
            }
            Err(e) => info!("Failed to load specular cubemap: {}", e),
        }
    }

    // Load irradiance cubemap
    let irr_path = ibl_dir.join("irradiance.bin");
    if irr_path.exists() {
        match load_irradiance_cubemap(ctx, &irr_path, &manifest, dst_stage) {
            Ok(tex) => {
                info!("Loaded irradiance cubemap: {}x{}",
                    manifest.irr_face_size(), manifest.irr_face_size());
                assets.env_irradiance = Some(tex);
            }
            Err(e) => info!("Failed to load irradiance cubemap: {}", e),
        }
    }

    // Load BRDF LUT
    let brdf_path = ibl_dir.join("brdf_lut.bin");
    if brdf_path.exists() {
        match load_brdf_lut(ctx, &brdf_path, &manifest, dst_stage) {
            Ok(tex) => {
                info!("Loaded BRDF LUT: {}x{}", manifest.lut_size(), manifest.lut_size());
                assets.brdf_lut = Some(tex);
            }
            Err(e) => info!("Failed to load BRDF LUT: {}", e),
        }
    }

    info!("IBL assets loaded: specular={}, irradiance={}, brdf_lut={}",
        assets.env_specular.is_some(),
        assets.env_irradiance.is_some(),
        assets.brdf_lut.is_some());

    assets
}

/// Load the specular cubemap from specular.bin (float16 RGBA, mip-major, face-major).
fn load_specular_cubemap(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
    dst_stage: vk::PipelineStageFlags,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read specular.bin: {}", e))?;

    let face_size = manifest.spec_face_size();
    let mip_count = manifest.spec_mip_count();
    let bytes_per_pixel = 8u32; // float16 RGBA = 4 * 2 bytes

    let (image, view, allocation) = create_cubemap_image(
        ctx,
        face_size,
        mip_count,
        vk::Format::R16G16B16A16_SFLOAT,
        &raw_data,
        bytes_per_pixel,
        "ibl_specular",
        dst_stage,
    )?;

    // Create sampler with linear + mipmap filtering
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(mip_count as f32);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create specular sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image,
        view,
        sampler,
        allocation: Some(allocation),
    })
}

/// Load the irradiance cubemap from irradiance.bin (float16 RGBA, 6 faces, 1 mip).
fn load_irradiance_cubemap(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
    dst_stage: vk::PipelineStageFlags,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read irradiance.bin: {}", e))?;

    let face_size = manifest.irr_face_size();
    let bytes_per_pixel = 8u32; // float16 RGBA

    let (image, view, allocation) = create_cubemap_image(
        ctx,
        face_size,
        1,
        vk::Format::R16G16B16A16_SFLOAT,
        &raw_data,
        bytes_per_pixel,
        "ibl_irradiance",
        dst_stage,
    )?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create irradiance sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image,
        view,
        sampler,
        allocation: Some(allocation),
    })
}

/// Load the BRDF LUT from brdf_lut.bin (float16 RG, padded to RGBA).
fn load_brdf_lut(
    ctx: &mut VulkanContext,
    path: &Path,
    manifest: &IblManifest,
    dst_stage: vk::PipelineStageFlags,
) -> Result<IblTexture, String> {
    let raw_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read brdf_lut.bin: {}", e))?;

    let lut_size = manifest.lut_size();
    let expected_rg_bytes = (lut_size * lut_size * 2 * 2) as usize; // float16 RG = 2 * 2 bytes

    if raw_data.len() < expected_rg_bytes {
        return Err(format!(
            "BRDF LUT data too small: {} < {} expected",
            raw_data.len(),
            expected_rg_bytes
        ));
    }

    // Pad RG float16 to RGBA float16 (each pixel: 4 bytes -> 8 bytes)
    let pixel_count = (lut_size * lut_size) as usize;
    let mut rgba_data: Vec<u8> = Vec::with_capacity(pixel_count * 8);
    let zero_u16: [u8; 2] = [0, 0]; // float16 zero

    for i in 0..pixel_count {
        let src_offset = i * 4; // 2 channels * 2 bytes each
        // R channel
        rgba_data.extend_from_slice(&raw_data[src_offset..src_offset + 2]);
        // G channel
        rgba_data.extend_from_slice(&raw_data[src_offset + 2..src_offset + 4]);
        // B channel (zero)
        rgba_data.extend_from_slice(&zero_u16);
        // A channel (zero)
        rgba_data.extend_from_slice(&zero_u16);
    }

    let gpu_image = create_texture_image_raw(
        ctx,
        lut_size,
        lut_size,
        &rgba_data,
        vk::Format::R16G16B16A16_SFLOAT,
        "ibl_brdf_lut",
        dst_stage,
    )?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe {
        ctx.device
            .create_sampler(&sampler_info, None)
            .map_err(|e| format!("Failed to create BRDF LUT sampler: {:?}", e))?
    };

    Ok(IblTexture {
        image: gpu_image.image,
        view: gpu_image.view,
        sampler,
        allocation: gpu_image.allocation,
    })
}

// ===========================================================================
// Bindless rendering support
// ===========================================================================

/// Material flags for bindless material data.
pub const BINDLESS_MAT_FLAG_NORMAL_MAP: u32 = 0x1;
pub const BINDLESS_MAT_FLAG_CLEARCOAT: u32 = 0x2;
pub const BINDLESS_MAT_FLAG_SHEEN: u32 = 0x4;
pub const BINDLESS_MAT_FLAG_EMISSION: u32 = 0x8;
pub const BINDLESS_MAT_FLAG_TRANSMISSION: u32 = 0x10;

/// Bindless material data matching the shader's `BindlessMaterialData` struct (std430, 128 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct BindlessMaterialData {
    pub base_color_factor: [f32; 4],          // offset 0
    pub emissive_factor: [f32; 3],            // offset 16
    pub metallic_factor: f32,                 // offset 28
    pub roughness_factor: f32,                // offset 32
    pub emissive_strength: f32,               // offset 36
    pub ior: f32,                             // offset 40
    pub clearcoat_factor: f32,                // offset 44
    pub clearcoat_roughness: f32,             // offset 48
    pub transmission_factor: f32,             // offset 52 (matches shader index 8)
    pub sheen_roughness: f32,                 // offset 56 (matches shader index 9)
    pub _pad_before_sheen: f32,              // offset 60
    pub sheen_color_factor: [f32; 3],         // offset 64
    pub _pad_after_sheen: f32,               // offset 76
    pub base_color_tex_index: i32,            // offset 80
    pub normal_tex_index: i32,                // offset 84
    pub metallic_roughness_tex_index: i32,    // offset 88
    pub occlusion_tex_index: i32,             // offset 92
    pub emissive_tex_index: i32,              // offset 96
    pub clearcoat_tex_index: i32,             // offset 100
    pub clearcoat_roughness_tex_index: i32,   // offset 104
    pub sheen_color_tex_index: i32,           // offset 108
    pub transmission_tex_index: i32,          // offset 112
    pub material_flags: u32,                  // offset 116
    pub index_offset: f32,                    // offset 120 — per-geometry index offset for RT (float for shader compat)
    pub _padding: u32,                        // offset 124
}
// Total: 128 bytes (std430)

unsafe impl bytemuck::Pod for BindlessMaterialData {}
unsafe impl bytemuck::Zeroable for BindlessMaterialData {}

/// Dynamic bindless material field descriptor (loaded from reflection JSON)
#[derive(Debug, Clone)]
pub struct BindlessFieldInfo {
    pub name: String,
    pub field_type: String,
    pub offset: u32,
    pub size: u32,
}

/// Dynamic struct layout from reflection JSON
#[derive(Debug, Clone)]
pub struct BindlessStructLayout {
    pub fields: Vec<BindlessFieldInfo>,
    pub struct_size: u32,
    pub has_custom_properties: bool,
}

impl BindlessStructLayout {
    pub fn from_reflection(bindless_json: &serde_json::Value) -> Self {
        let struct_size = bindless_json.get("struct_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(128) as u32;

        let mut fields = Vec::new();
        let has_custom = bindless_json.get("custom_properties").is_some();

        if let Some(field_array) = bindless_json.get("struct_fields").and_then(|v| v.as_array()) {
            for field in field_array {
                fields.push(BindlessFieldInfo {
                    name: field.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    field_type: field.get("type").and_then(|v| v.as_str()).unwrap_or("scalar").to_string(),
                    offset: field.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                    size: field.get("size").and_then(|v| v.as_u64()).unwrap_or(4) as u32,
                });
            }
        }

        Self { fields, struct_size, has_custom_properties: has_custom }
    }

    pub fn find_field(&self, name: &str) -> Option<&BindlessFieldInfo> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Build materials SSBO using reflection-driven dynamic layout.
///
/// When custom properties are present, writes to a raw byte buffer
/// using field offsets from reflection JSON instead of the fixed struct.
pub fn build_materials_ssbo_dynamic(
    materials: &[crate::gltf_loader::GltfMaterial],
    layout: &BindlessStructLayout,
    tex_index_fn: impl Fn(usize, &str) -> i32,
) -> Vec<u8> {
    let mat_count = materials.len().max(1);
    let mut buffer = vec![0u8; mat_count * layout.struct_size as usize];

    for (i, mat) in materials.iter().enumerate() {
        let base = i * layout.struct_size as usize;

        let write_f32 = |buf: &mut Vec<u8>, field_name: &str, val: f32| {
            if let Some(f) = layout.find_field(field_name) {
                let off = base + f.offset as usize;
                if off + 4 <= buf.len() {
                    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        };

        let write_i32 = |buf: &mut Vec<u8>, field_name: &str, val: i32| {
            if let Some(f) = layout.find_field(field_name) {
                let off = base + f.offset as usize;
                if off + 4 <= buf.len() {
                    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        };

        let write_u32 = |buf: &mut Vec<u8>, field_name: &str, val: u32| {
            if let Some(f) = layout.find_field(field_name) {
                let off = base + f.offset as usize;
                if off + 4 <= buf.len() {
                    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        };

        // baseColorFactor (vec4)
        if let Some(f) = layout.find_field("baseColorFactor") {
            let off = base + f.offset as usize;
            if off + 16 <= buffer.len() {
                for (j, &val) in mat.base_color.iter().enumerate() {
                    buffer[off + j * 4..off + (j + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        // emissiveFactor (vec3)
        if let Some(f) = layout.find_field("emissiveFactor") {
            let off = base + f.offset as usize;
            if off + 12 <= buffer.len() {
                for (j, &val) in mat.emissive.iter().enumerate() {
                    buffer[off + j * 4..off + (j + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        // Scalar fields
        write_f32(&mut buffer, "metallicFactor", mat.metallic);
        write_f32(&mut buffer, "roughnessFactor", mat.roughness);
        write_f32(&mut buffer, "emissionStrength", mat.emissive_strength);
        write_f32(&mut buffer, "ior", mat.ior);
        write_f32(&mut buffer, "clearcoatFactor", mat.clearcoat_factor);
        write_f32(&mut buffer, "clearcoatRoughness", mat.clearcoat_roughness_factor);
        write_f32(&mut buffer, "transmissionFactor", mat.transmission_factor);
        write_f32(&mut buffer, "sheenRoughness", mat.sheen_roughness_factor);

        // sheenColorFactor (vec3)
        if let Some(f) = layout.find_field("sheenColorFactor") {
            let off = base + f.offset as usize;
            if off + 12 <= buffer.len() {
                for (j, &val) in mat.sheen_color_factor.iter().enumerate() {
                    buffer[off + j * 4..off + (j + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
            }
        }

        // Texture indices
        write_i32(&mut buffer, "base_color_tex_index", tex_index_fn(i, "base_color"));
        write_i32(&mut buffer, "normal_tex_index", tex_index_fn(i, "normal"));
        write_i32(&mut buffer, "metallic_roughness_tex_index", tex_index_fn(i, "metallic_roughness"));
        write_i32(&mut buffer, "occlusion_tex_index", tex_index_fn(i, "occlusion"));
        write_i32(&mut buffer, "emissive_tex_index", tex_index_fn(i, "emissive"));
        write_i32(&mut buffer, "clearcoat_tex_index", tex_index_fn(i, "clearcoat"));
        write_i32(&mut buffer, "clearcoat_roughness_tex_index", tex_index_fn(i, "clearcoat_roughness"));
        write_i32(&mut buffer, "sheen_color_tex_index", tex_index_fn(i, "sheen_color"));
        write_i32(&mut buffer, "transmission_tex_index", tex_index_fn(i, "transmission"));

        // Feature flags (computed from material properties, same logic as fixed builder)
        let mut flags: u32 = 0;
        if mat.normal_image.is_some() {
            flags |= BINDLESS_MAT_FLAG_NORMAL_MAP;
        }
        if mat.has_clearcoat {
            flags |= BINDLESS_MAT_FLAG_CLEARCOAT;
        }
        if mat.has_sheen {
            flags |= BINDLESS_MAT_FLAG_SHEEN;
        }
        if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
            flags |= BINDLESS_MAT_FLAG_EMISSION;
        }
        if mat.has_transmission {
            flags |= BINDLESS_MAT_FLAG_TRANSMISSION;
        }
        write_u32(&mut buffer, "material_flags", flags);

        // Write custom float properties from glTF extras
        for (key, &val) in &mat.custom_float_properties {
            write_f32(&mut buffer, key, val);
        }

        // Write custom vec4 properties from glTF extras
        for (key, val) in &mat.custom_vec4_properties {
            if let Some(f) = layout.find_field(key) {
                let off = base + f.offset as usize;
                if off + 16 <= buffer.len() {
                    for (j, &v) in val.iter().enumerate() {
                        buffer[off + j * 4..off + (j + 1) * 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
    }

    buffer
}

/// Collected bindless texture array: all unique (imageView, sampler) pairs.
pub struct BindlessTextureArray {
    pub image_views: Vec<vk::ImageView>,
    pub samplers: Vec<vk::Sampler>,
    pub texture_count: u32,
}

/// Materials SSBO for bindless rendering: one `BindlessMaterialData` per scene material.
pub struct BindlessMaterialsSSBO {
    pub buffer: GpuBuffer,
    pub material_count: u32,
}

/// Build a bindless texture array by collecting all unique (imageView, sampler) pairs
/// from per-material texture maps.
///
/// The per-material texture maps are `Vec<HashMap<String, (vk::ImageView, vk::Sampler)>>`
/// where each entry maps a texture slot name (e.g., "base_color_tex") to a GPU resource pair.
///
/// Returns the array and a dedup map: imageView -> index in the array.
pub fn build_bindless_texture_array_from_views(
    per_mat_textures: &[std::collections::HashMap<String, (vk::ImageView, vk::Sampler)>],
) -> (BindlessTextureArray, std::collections::HashMap<vk::ImageView, u32>) {
    use std::collections::HashMap;

    let mut views: Vec<vk::ImageView> = Vec::new();
    let mut samplers: Vec<vk::Sampler> = Vec::new();
    let mut dedup: HashMap<vk::ImageView, u32> = HashMap::new();

    // Only include material texture slots (not IBL cubemaps or other non-material textures)
    let valid_slots = [
        "base_color_tex", "normal_tex", "metallic_roughness_tex",
        "occlusion_tex", "emissive_tex", "clearcoat_tex",
        "clearcoat_roughness_tex", "sheen_color_tex", "transmission_tex",
    ];
    for (mat_i, mat_tex) in per_mat_textures.iter().enumerate() {
        for (slot, &(view, sampler)) in mat_tex {
            if !valid_slots.contains(&slot.as_str()) { continue; }
            if !dedup.contains_key(&view) {
                let idx = views.len() as u32;
                dedup.insert(view, idx);
                views.push(view);
                samplers.push(sampler);
                info!("  bindless tex[{}] = mat {} slot '{}'  (view={:?})", idx, mat_i, slot, view);
            }
        }
    }

    let count = views.len() as u32;
    info!("Bindless texture array: {} unique textures", count);

    (
        BindlessTextureArray {
            image_views: views,
            samplers,
            texture_count: count,
        },
        dedup,
    )
}

/// Build a bindless texture array from per-material GpuImage maps.
///
/// This variant works with `Vec<HashMap<String, GpuImage>>` as used in the raster
/// and mesh renderers. A shared sampler handle is used for all textures.
pub fn build_bindless_texture_array_from_gpu_images(
    per_mat_textures: &[std::collections::HashMap<String, GpuImage>],
    sampler: vk::Sampler,
    default_white_view: vk::ImageView,
    default_black_view: vk::ImageView,
    default_normal_view: vk::ImageView,
) -> (BindlessTextureArray, std::collections::HashMap<vk::ImageView, u32>) {
    use std::collections::HashMap;

    let mut views: Vec<vk::ImageView> = Vec::new();
    let mut samplers: Vec<vk::Sampler> = Vec::new();
    let mut dedup: HashMap<vk::ImageView, u32> = HashMap::new();

    // Always add default textures at known indices
    let add_view = |views: &mut Vec<vk::ImageView>,
                    samplers: &mut Vec<vk::Sampler>,
                    dedup: &mut HashMap<vk::ImageView, u32>,
                    view: vk::ImageView,
                    samp: vk::Sampler| -> u32 {
        if let Some(&idx) = dedup.get(&view) {
            idx
        } else {
            let idx = views.len() as u32;
            dedup.insert(view, idx);
            views.push(view);
            samplers.push(samp);
            idx
        }
    };

    add_view(&mut views, &mut samplers, &mut dedup, default_white_view, sampler);
    add_view(&mut views, &mut samplers, &mut dedup, default_black_view, sampler);
    add_view(&mut views, &mut samplers, &mut dedup, default_normal_view, sampler);

    for mat_tex in per_mat_textures {
        for (_slot, img) in mat_tex {
            add_view(&mut views, &mut samplers, &mut dedup, img.view, sampler);
        }
    }

    let count = views.len() as u32;
    info!("Bindless texture array (GpuImage): {} unique textures", count);

    (
        BindlessTextureArray {
            image_views: views,
            samplers,
            texture_count: count,
        },
        dedup,
    )
}

/// Build the materials SSBO for bindless rendering.
///
/// Creates one `BindlessMaterialData` per scene material, with texture indices
/// pointing into the bindless texture array (via the dedup map) and material flags.
pub fn build_materials_ssbo(
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    scene: &crate::gltf_loader::GltfScene,
    dedup: &std::collections::HashMap<vk::ImageView, u32>,
    per_mat_textures: &[std::collections::HashMap<String, GpuImage>],
    default_white_view: vk::ImageView,
    default_black_view: vk::ImageView,
    default_normal_view: vk::ImageView,
) -> Result<BindlessMaterialsSSBO, String> {
    let mut materials: Vec<BindlessMaterialData> = Vec::new();

    let lookup = |mat_idx: usize, slot: &str, default_view: vk::ImageView| -> i32 {
        if mat_idx < per_mat_textures.len() {
            if let Some(img) = per_mat_textures[mat_idx].get(slot) {
                if let Some(&idx) = dedup.get(&img.view) {
                    return idx as i32;
                }
            }
        }
        // Return default texture index
        dedup.get(&default_view).copied().unwrap_or(0) as i32
    };

    for (mi, mat) in scene.materials.iter().enumerate() {
        let mut flags: u32 = 0;
        if mat.normal_image.is_some() {
            flags |= BINDLESS_MAT_FLAG_NORMAL_MAP;
        }
        if mat.has_clearcoat {
            flags |= BINDLESS_MAT_FLAG_CLEARCOAT;
        }
        if mat.has_sheen {
            flags |= BINDLESS_MAT_FLAG_SHEEN;
        }
        if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
            flags |= BINDLESS_MAT_FLAG_EMISSION;
        }
        if mat.has_transmission {
            flags |= BINDLESS_MAT_FLAG_TRANSMISSION;
        }

        materials.push(BindlessMaterialData {
            base_color_factor: mat.base_color,
            emissive_factor: mat.emissive,
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            emissive_strength: mat.emissive_strength,
            ior: mat.ior,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness: mat.clearcoat_roughness_factor,
            sheen_roughness: mat.sheen_roughness_factor,
            transmission_factor: mat.transmission_factor,
            _pad_before_sheen: 0.0,
            sheen_color_factor: mat.sheen_color_factor,
            _pad_after_sheen: 0.0,
            base_color_tex_index: lookup(mi, "base_color_tex", default_white_view),
            normal_tex_index: lookup(mi, "normal_tex", default_normal_view),
            metallic_roughness_tex_index: lookup(mi, "metallic_roughness_tex", default_white_view),
            occlusion_tex_index: lookup(mi, "occlusion_tex", default_white_view),
            emissive_tex_index: lookup(mi, "emissive_tex", default_black_view),
            clearcoat_tex_index: lookup(mi, "clearcoat_tex", default_white_view),
            clearcoat_roughness_tex_index: lookup(mi, "clearcoat_roughness_tex", default_white_view),
            sheen_color_tex_index: lookup(mi, "sheen_color_tex", default_black_view),
            transmission_tex_index: lookup(mi, "transmission_tex", default_black_view),
            material_flags: flags,
            index_offset: 0.0,
            _padding: 0,
        });
    }

    // Ensure at least one material entry
    if materials.is_empty() {
        materials.push(BindlessMaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            metallic_factor: 0.0,
            roughness_factor: 1.0,
            emissive_strength: 1.0,
            ior: 1.5,
            clearcoat_factor: 0.0,
            clearcoat_roughness: 0.0,
            sheen_roughness: 0.0,
            transmission_factor: 0.0,
            _pad_before_sheen: 0.0,
            sheen_color_factor: [0.0, 0.0, 0.0],
            _pad_after_sheen: 0.0,
            base_color_tex_index: 0,
            normal_tex_index: -1,
            metallic_roughness_tex_index: -1,
            occlusion_tex_index: -1,
            emissive_tex_index: -1,
            clearcoat_tex_index: -1,
            clearcoat_roughness_tex_index: -1,
            sheen_color_tex_index: -1,
            transmission_tex_index: -1,
            material_flags: 0,
            index_offset: 0.0,
            _padding: 0,
        });
    }

    let material_count = materials.len() as u32;
    let data: &[u8] = bytemuck::cast_slice(&materials);

    info!(
        "Bindless materials SSBO: {} materials, {} bytes (128 bytes each)",
        material_count,
        data.len()
    );

    let buffer = create_buffer_with_data(
        device,
        allocator,
        data,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "bindless_materials_ssbo",
    )?;

    Ok(BindlessMaterialsSSBO {
        buffer,
        material_count,
    })
}

/// Build the materials SSBO for bindless RT rendering.
///
/// This variant works with per-material texture maps as used in the RT renderer
/// (Vec<HashMap<String, (vk::ImageView, vk::Sampler)>>).
pub fn build_materials_ssbo_from_views(
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    scene: &crate::gltf_loader::GltfScene,
    dedup: &std::collections::HashMap<vk::ImageView, u32>,
    per_mat_textures: &[std::collections::HashMap<String, (vk::ImageView, vk::Sampler)>],
    default_white_view: vk::ImageView,
    default_black_view: vk::ImageView,
    default_normal_view: vk::ImageView,
) -> Result<BindlessMaterialsSSBO, String> {
    let mut materials: Vec<BindlessMaterialData> = Vec::new();

    let lookup = |mat_idx: usize, slot: &str, _default_view: vk::ImageView| -> i32 {
        if mat_idx < per_mat_textures.len() {
            if let Some(&(view, _sampler)) = per_mat_textures[mat_idx].get(slot) {
                if let Some(&idx) = dedup.get(&view) {
                    return idx as i32;
                }
            }
        }
        -1  // No texture for this slot — shader uses material factor directly
    };

    for (mi, mat) in scene.materials.iter().enumerate() {
        let mut flags: u32 = 0;
        if mat.normal_image.is_some() {
            flags |= BINDLESS_MAT_FLAG_NORMAL_MAP;
        }
        if mat.has_clearcoat {
            flags |= BINDLESS_MAT_FLAG_CLEARCOAT;
        }
        if mat.has_sheen {
            flags |= BINDLESS_MAT_FLAG_SHEEN;
        }
        if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
            flags |= BINDLESS_MAT_FLAG_EMISSION;
        }
        if mat.has_transmission {
            flags |= BINDLESS_MAT_FLAG_TRANSMISSION;
        }

        materials.push(BindlessMaterialData {
            base_color_factor: mat.base_color,
            emissive_factor: mat.emissive,
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            emissive_strength: mat.emissive_strength,
            ior: mat.ior,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness: mat.clearcoat_roughness_factor,
            sheen_roughness: mat.sheen_roughness_factor,
            transmission_factor: mat.transmission_factor,
            _pad_before_sheen: 0.0,
            sheen_color_factor: mat.sheen_color_factor,
            _pad_after_sheen: 0.0,
            base_color_tex_index: lookup(mi, "base_color_tex", default_white_view),
            normal_tex_index: lookup(mi, "normal_tex", default_normal_view),
            metallic_roughness_tex_index: lookup(mi, "metallic_roughness_tex", default_white_view),
            occlusion_tex_index: lookup(mi, "occlusion_tex", default_white_view),
            emissive_tex_index: lookup(mi, "emissive_tex", default_black_view),
            clearcoat_tex_index: lookup(mi, "clearcoat_tex", default_white_view),
            clearcoat_roughness_tex_index: lookup(mi, "clearcoat_roughness_tex", default_white_view),
            sheen_color_tex_index: lookup(mi, "sheen_color_tex", default_black_view),
            transmission_tex_index: lookup(mi, "transmission_tex", default_black_view),
            material_flags: flags,
            index_offset: 0.0,
            _padding: 0,
        });
    }

    if materials.is_empty() {
        materials.push(bytemuck::Zeroable::zeroed());
    }

    let material_count = materials.len() as u32;
    let data: &[u8] = bytemuck::cast_slice(&materials);

    info!(
        "Bindless materials SSBO (from views): {} materials, {} bytes",
        material_count,
        data.len()
    );

    let buffer = create_buffer_with_data(
        device,
        allocator,
        data,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "bindless_materials_ssbo",
    )?;

    Ok(BindlessMaterialsSSBO {
        buffer,
        material_count,
    })
}

/// Build a bindless materials SSBO indexed by **geometry** (draw range), not by material.
///
/// `gl_GeometryIndexEXT` in the RT closest-hit shader returns the geometry index within the
/// BLAS, which corresponds to the draw range index. Each SSBO entry at index `i` must contain
/// the material data for `draw_ranges[i].material_index`.
pub fn build_materials_ssbo_by_geometry(
    device: &ash::Device,
    allocator: &mut gpu_allocator::vulkan::Allocator,
    scene: &crate::gltf_loader::GltfScene,
    dedup: &std::collections::HashMap<vk::ImageView, u32>,
    per_mat_textures: &[std::collections::HashMap<String, (vk::ImageView, vk::Sampler)>],
    default_white_view: vk::ImageView,
    default_black_view: vk::ImageView,
    default_normal_view: vk::ImageView,
    draw_ranges: &[crate::gltf_loader::DrawRange],
) -> Result<BindlessMaterialsSSBO, String> {
    let lookup = |mat_idx: usize, slot: &str, _default_view: vk::ImageView| -> i32 {
        if mat_idx < per_mat_textures.len() {
            if let Some(&(view, _sampler)) = per_mat_textures[mat_idx].get(slot) {
                if let Some(&idx) = dedup.get(&view) {
                    return idx as i32;
                }
            }
        }
        -1  // No texture for this slot — shader uses material factor directly
    };

    let mut materials: Vec<BindlessMaterialData> = Vec::new();

    // One entry per draw range (geometry), NOT per material
    for (geo_idx, range) in draw_ranges.iter().enumerate() {
        let mi = range.material_index;
        let mat = if mi < scene.materials.len() {
            &scene.materials[mi]
        } else {
            info!(
                "Geometry {} references material {} which is out of range ({}), using defaults",
                geo_idx, mi, scene.materials.len()
            );
            materials.push(bytemuck::Zeroable::zeroed());
            continue;
        };

        let mut flags: u32 = 0;
        if mat.normal_image.is_some() {
            flags |= BINDLESS_MAT_FLAG_NORMAL_MAP;
        }
        if mat.has_clearcoat {
            flags |= BINDLESS_MAT_FLAG_CLEARCOAT;
        }
        if mat.has_sheen {
            flags |= BINDLESS_MAT_FLAG_SHEEN;
        }
        if mat.emissive_image.is_some() || mat.emissive.iter().any(|&e| e > 0.0) {
            flags |= BINDLESS_MAT_FLAG_EMISSION;
        }
        if mat.has_transmission {
            flags |= BINDLESS_MAT_FLAG_TRANSMISSION;
        }

        materials.push(BindlessMaterialData {
            base_color_factor: mat.base_color,
            emissive_factor: mat.emissive,
            metallic_factor: mat.metallic,
            roughness_factor: mat.roughness,
            emissive_strength: mat.emissive_strength,
            ior: mat.ior,
            clearcoat_factor: mat.clearcoat_factor,
            clearcoat_roughness: mat.clearcoat_roughness_factor,
            sheen_roughness: mat.sheen_roughness_factor,
            transmission_factor: mat.transmission_factor,
            _pad_before_sheen: 0.0,
            sheen_color_factor: mat.sheen_color_factor,
            _pad_after_sheen: 0.0,
            base_color_tex_index: lookup(mi, "base_color_tex", default_white_view),
            normal_tex_index: lookup(mi, "normal_tex", default_normal_view),
            metallic_roughness_tex_index: lookup(mi, "metallic_roughness_tex", default_white_view),
            occlusion_tex_index: lookup(mi, "occlusion_tex", default_white_view),
            emissive_tex_index: lookup(mi, "emissive_tex", default_black_view),
            clearcoat_tex_index: lookup(mi, "clearcoat_tex", default_white_view),
            clearcoat_roughness_tex_index: lookup(mi, "clearcoat_roughness_tex", default_white_view),
            sheen_color_tex_index: lookup(mi, "sheen_color_tex", default_black_view),
            transmission_tex_index: lookup(mi, "transmission_tex", default_black_view),
            material_flags: flags,
            index_offset: range.index_offset as f32,
            _padding: 0,
        });

        let m = materials.last().unwrap();
        info!(
            "Geometry {} -> material {} '{}': flags=0x{:x}, metallic={:.2}, roughness={:.2}, \
             sheen_rough={:.2}, transmission={:.2}, index_offset={}, \
             tex[bc={}, norm={}, mr={}, occ={}, em={}, cc={}, ccr={}, sheen={}, trans={}]",
            geo_idx, mi, mat.name,
            flags, m.metallic_factor, m.roughness_factor,
            m.sheen_roughness, m.transmission_factor, m.index_offset,
            m.base_color_tex_index, m.normal_tex_index, m.metallic_roughness_tex_index,
            m.occlusion_tex_index, m.emissive_tex_index, m.clearcoat_tex_index,
            m.clearcoat_roughness_tex_index, m.sheen_color_tex_index, m.transmission_tex_index,
        );
    }

    if materials.is_empty() {
        materials.push(bytemuck::Zeroable::zeroed());
    }

    let material_count = materials.len() as u32;
    let data: &[u8] = bytemuck::cast_slice(&materials);

    info!(
        "Bindless materials SSBO (by geometry): {} entries (geometries), {} bytes",
        material_count,
        data.len()
    );

    let buffer = create_buffer_with_data(
        device,
        allocator,
        data,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "bindless_materials_ssbo_by_geo",
    )?;

    Ok(BindlessMaterialsSSBO {
        buffer,
        material_count,
    })
}
