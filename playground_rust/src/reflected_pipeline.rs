//! Reflection-driven Vulkan pipeline setup from .lux.json metadata.
//!
//! Reads .lux.json sidecar files and creates `VkDescriptorSetLayout`,
//! `VkPipelineLayout`, `VkPipeline`, and vertex input state automatically.

use ash::vk;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ===========================================================================
// Reflection data structures (deserialized from .lux.json)
// ===========================================================================

#[derive(Debug, Deserialize)]
pub struct ReflectionData {
    pub version: u32,
    #[serde(default)]
    pub source: String,
    pub stage: String,
    pub execution_model: String,
    #[serde(default)]
    pub inputs: Vec<VarInfo>,
    #[serde(default)]
    pub outputs: Vec<VarInfo>,
    #[serde(default)]
    pub descriptor_sets: HashMap<String, Vec<BindingInfo>>,
    #[serde(default)]
    pub push_constants: Vec<PushConstantInfo>,
    #[serde(default)]
    pub vertex_attributes: Vec<VertexAttribute>,
    #[serde(default)]
    pub vertex_stride: u32,
    #[serde(default)]
    pub ray_payloads: Vec<RayPayloadInfo>,
    #[serde(default)]
    pub hit_attributes: Vec<HitAttributeInfo>,
    #[serde(default)]
    pub callable_data: Vec<CallableDataInfo>,
}

#[derive(Debug, Deserialize)]
pub struct VarInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub var_type: String,
    #[serde(default)]
    pub location: i32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BindingInfo {
    pub binding: u32,
    #[serde(rename = "type")]
    pub binding_type: String,
    pub name: String,
    #[serde(default)]
    pub fields: Vec<FieldInfo>,
    #[serde(default)]
    pub size: u32,
    #[serde(default)]
    pub stage_flags: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FieldInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub field_type: String,
    #[serde(default)]
    pub offset: u32,
    #[serde(default)]
    pub size: u32,
}

#[derive(Debug, Deserialize)]
pub struct PushConstantInfo {
    pub name: String,
    #[serde(default)]
    pub fields: Vec<FieldInfo>,
    #[serde(default)]
    pub size: u32,
    #[serde(default)]
    pub stage_flags: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct VertexAttribute {
    pub location: u32,
    #[serde(rename = "type")]
    pub attr_type: String,
    pub name: String,
    pub format: String,
    pub offset: u32,
}

#[derive(Debug, Deserialize)]
pub struct RayPayloadInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub payload_type: String,
    #[serde(default)]
    pub location: i32,
}

#[derive(Debug, Deserialize)]
pub struct HitAttributeInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub attr_type: String,
}

#[derive(Debug, Deserialize)]
pub struct CallableDataInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub data_type: String,
    #[serde(default)]
    pub location: i32,
}

// ===========================================================================
// Loading
// ===========================================================================

/// Load and parse a .lux.json reflection file.
pub fn load_reflection(path: &Path) -> Result<ReflectionData, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read reflection JSON {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse reflection JSON {}: {}", path.display(), e))
}

// ===========================================================================
// Format mapping
// ===========================================================================

/// Map reflection format string to Vulkan format.
pub fn reflection_format_to_vk(fmt: &str) -> vk::Format {
    match fmt {
        "R32_SFLOAT" => vk::Format::R32_SFLOAT,
        "R32G32_SFLOAT" => vk::Format::R32G32_SFLOAT,
        "R32G32B32_SFLOAT" => vk::Format::R32G32B32_SFLOAT,
        "R32G32B32A32_SFLOAT" => vk::Format::R32G32B32A32_SFLOAT,
        "R32_SINT" => vk::Format::R32_SINT,
        "R32G32_SINT" => vk::Format::R32G32_SINT,
        "R32G32B32_SINT" => vk::Format::R32G32B32_SINT,
        "R32G32B32A32_SINT" => vk::Format::R32G32B32A32_SINT,
        "R32_UINT" => vk::Format::R32_UINT,
        "R32G32_UINT" => vk::Format::R32G32_UINT,
        "R32G32B32_UINT" => vk::Format::R32G32B32_UINT,
        "R32G32B32A32_UINT" => vk::Format::R32G32B32A32_UINT,
        _ => vk::Format::R32G32B32A32_SFLOAT,
    }
}

// ===========================================================================
// Descriptor set layout creation
// ===========================================================================

fn binding_type_to_vk(btype: &str) -> vk::DescriptorType {
    match btype {
        "uniform_buffer" => vk::DescriptorType::UNIFORM_BUFFER,
        "sampler" => vk::DescriptorType::SAMPLER,
        "sampled_image" => vk::DescriptorType::SAMPLED_IMAGE,
        "sampled_cube_image" => vk::DescriptorType::SAMPLED_IMAGE,
        "storage_image" => vk::DescriptorType::STORAGE_IMAGE,
        "acceleration_structure" => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        _ => vk::DescriptorType::UNIFORM_BUFFER,
    }
}

/// Check if a binding type represents a cubemap image.
pub fn is_cube_image_binding(btype: &str) -> bool {
    btype == "sampled_cube_image"
}

fn stage_flags_from_strings(flags: &[String]) -> vk::ShaderStageFlags {
    let mut result = vk::ShaderStageFlags::empty();
    for f in flags {
        match f.as_str() {
            "vertex" => result |= vk::ShaderStageFlags::VERTEX,
            "fragment" => result |= vk::ShaderStageFlags::FRAGMENT,
            "raygen" => result |= vk::ShaderStageFlags::RAYGEN_KHR,
            "closest_hit" => result |= vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            "miss" => result |= vk::ShaderStageFlags::MISS_KHR,
            "any_hit" => result |= vk::ShaderStageFlags::ANY_HIT_KHR,
            "intersection" => result |= vk::ShaderStageFlags::INTERSECTION_KHR,
            "callable" => result |= vk::ShaderStageFlags::CALLABLE_KHR,
            _ => {}
        }
    }
    if result.is_empty() {
        // Safe fallback: make binding visible to all shader stages
        result = vk::ShaderStageFlags::ALL;
    }
    result
}

/// Create descriptor set layouts from merged vertex + fragment reflection data.
pub unsafe fn create_descriptor_set_layouts(
    device: &ash::Device,
    vert_reflection: &ReflectionData,
    frag_reflection: &ReflectionData,
) -> Result<HashMap<u32, vk::DescriptorSetLayout>, String> {
    // Merge descriptor sets from both stages
    let mut merged: HashMap<u32, HashMap<u32, vk::DescriptorSetLayoutBinding>> = HashMap::new();

    let merge_stage = |merged: &mut HashMap<u32, HashMap<u32, vk::DescriptorSetLayoutBinding>>,
                       refl: &ReflectionData| {
        for (set_key, bindings) in &refl.descriptor_sets {
            let set_idx: u32 = set_key.parse().unwrap_or(0);
            let set_map = merged.entry(set_idx).or_default();
            for b in bindings {
                let entry = set_map.entry(b.binding).or_insert_with(|| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(binding_type_to_vk(&b.binding_type))
                        .descriptor_count(1)
                        .stage_flags(stage_flags_from_strings(&b.stage_flags))
                });
                *entry = entry.stage_flags(entry.stage_flags | stage_flags_from_strings(&b.stage_flags));
            }
        }
    };

    merge_stage(&mut merged, vert_reflection);
    merge_stage(&mut merged, frag_reflection);

    let mut layouts = HashMap::new();
    for (set_idx, binding_map) in &merged {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
            binding_map.values().copied().collect();
        bindings.sort_by_key(|b| b.binding);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout: {:?}", e))?;
        layouts.insert(*set_idx, layout);
    }

    Ok(layouts)
}

/// Create pipeline layout from descriptor set layouts and push constant ranges.
pub unsafe fn create_reflected_pipeline_layout(
    device: &ash::Device,
    set_layouts: &HashMap<u32, vk::DescriptorSetLayout>,
    vert_reflection: &ReflectionData,
    frag_reflection: &ReflectionData,
) -> Result<vk::PipelineLayout, String> {
    let max_set = set_layouts.keys().copied().max().unwrap_or(0);
    let mut ordered: Vec<vk::DescriptorSetLayout> = Vec::new();
    for i in 0..=max_set {
        if let Some(&layout) = set_layouts.get(&i) {
            ordered.push(layout);
        }
    }

    let mut push_ranges: Vec<vk::PushConstantRange> = Vec::new();
    for pc in &vert_reflection.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(stage_flags_from_strings(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }
    for pc in &frag_reflection.push_constants {
        push_ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(stage_flags_from_strings(&pc.stage_flags))
                .offset(0)
                .size(pc.size),
        );
    }

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&ordered)
        .push_constant_ranges(&push_ranges);

    device
        .create_pipeline_layout(&layout_info, None)
        .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))
}

/// Create vertex input binding and attribute descriptions from reflection.
pub fn create_reflected_vertex_input(
    vert_reflection: &ReflectionData,
) -> (
    Vec<vk::VertexInputBindingDescription>,
    Vec<vk::VertexInputAttributeDescription>,
) {
    let binding = vec![vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(vert_reflection.vertex_stride)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attributes: Vec<vk::VertexInputAttributeDescription> = vert_reflection
        .vertex_attributes
        .iter()
        .map(|attr| {
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(attr.location)
                .format(reflection_format_to_vk(&attr.format))
                .offset(attr.offset)
        })
        .collect();

    (binding, attributes)
}

// ===========================================================================
// Merged descriptor set layout from multiple reflection files
// ===========================================================================

/// Merge descriptor sets from an arbitrary set of reflection files.
/// Returns a map of (set_index -> Vec<BindingInfo>), with stage_flags merged.
pub fn merge_descriptor_sets(
    reflections: &[&ReflectionData],
) -> HashMap<u32, Vec<BindingInfo>> {
    let mut merged: HashMap<u32, HashMap<u32, BindingInfo>> = HashMap::new();

    for refl in reflections {
        for (set_key, bindings) in &refl.descriptor_sets {
            let set_idx: u32 = set_key.parse().unwrap_or(0);
            let set_map = merged.entry(set_idx).or_default();
            for b in bindings {
                let entry = set_map.entry(b.binding).or_insert_with(|| b.clone());
                // Merge stage flags from additional stages
                let mut existing_flags: Vec<String> = entry.stage_flags.clone();
                for flag in &b.stage_flags {
                    if !existing_flags.contains(flag) {
                        existing_flags.push(flag.clone());
                    }
                }
                entry.stage_flags = existing_flags;
            }
        }
    }

    let mut result = HashMap::new();
    for (set_idx, binding_map) in merged {
        let mut bindings: Vec<BindingInfo> = binding_map.into_values().collect();
        bindings.sort_by_key(|b| b.binding);
        result.insert(set_idx, bindings);
    }

    result
}

/// Create VkDescriptorSetLayouts from merged binding info.
pub unsafe fn create_descriptor_set_layouts_from_merged(
    device: &ash::Device,
    merged: &HashMap<u32, Vec<BindingInfo>>,
) -> Result<HashMap<u32, vk::DescriptorSetLayout>, String> {
    let mut layouts = HashMap::new();

    for (&set_idx, bindings) in merged {
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(binding_type_to_vk(&b.binding_type))
                    .descriptor_count(1)
                    .stage_flags(stage_flags_from_strings(&b.stage_flags))
            })
            .collect();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&vk_bindings);

        let layout = device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(|e| format!("Failed to create descriptor set layout for set {}: {:?}", set_idx, e))?;

        layouts.insert(set_idx, layout);
    }

    Ok(layouts)
}

/// Compute descriptor pool sizes from merged reflection data.
pub fn compute_pool_sizes(
    merged: &HashMap<u32, Vec<BindingInfo>>,
) -> (Vec<vk::DescriptorPoolSize>, u32) {
    let mut type_counts: HashMap<vk::DescriptorType, u32> = HashMap::new();
    let mut total_sets = 0u32;

    for bindings in merged.values() {
        total_sets += 1;
        for b in bindings {
            let vk_type = binding_type_to_vk(&b.binding_type);
            *type_counts.entry(vk_type).or_insert(0) += 1;
        }
    }

    let pool_sizes: Vec<vk::DescriptorPoolSize> = type_counts
        .iter()
        .map(|(&ty, &count)| {
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(count)
        })
        .collect();

    (pool_sizes, total_sets)
}

/// Create a pipeline layout from merged descriptor set layouts (ordered by set index).
pub unsafe fn create_pipeline_layout_from_merged(
    device: &ash::Device,
    set_layouts: &HashMap<u32, vk::DescriptorSetLayout>,
    push_constant_ranges: &[vk::PushConstantRange],
) -> Result<vk::PipelineLayout, String> {
    let max_set = set_layouts.keys().copied().max().unwrap_or(0);
    let mut ordered: Vec<vk::DescriptorSetLayout> = Vec::new();
    for i in 0..=max_set {
        if let Some(&layout) = set_layouts.get(&i) {
            ordered.push(layout);
        }
    }

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&ordered)
        .push_constant_ranges(push_constant_ranges);

    device
        .create_pipeline_layout(&layout_info, None)
        .map_err(|e| format!("Failed to create pipeline layout: {:?}", e))
}

/// Public accessor: convert binding_type string to VkDescriptorType.
pub fn binding_type_to_vk_public(btype: &str) -> vk::DescriptorType {
    binding_type_to_vk(btype)
}

/// Public accessor: convert stage flag strings to VkShaderStageFlags.
pub fn stage_flags_from_strings_public(flags: &[String]) -> vk::ShaderStageFlags {
    stage_flags_from_strings(flags)
}
