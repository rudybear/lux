#pragma once
/**
 * ReflectedPipeline — data-driven Vulkan pipeline setup from .lux.json reflection metadata.
 *
 * Reads .lux.json sidecar files and creates VkDescriptorSetLayout, VkPipelineLayout,
 * VkPipeline, and vertex input state automatically — no hardcoded descriptor sets needed.
 */

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declaration for JSON parsing (uses minimal built-in parser)
struct ReflectionData {
    int version = 1;
    std::string source;
    std::string stage;
    std::string execution_model;

    struct VarInfo {
        std::string name;
        std::string type;
        int location = -1;
    };

    struct FieldInfo {
        std::string name;
        std::string type;
        int offset = 0;
        int size = 0;
    };

    struct BindingInfo {
        int binding = 0;
        std::string type; // "uniform_buffer", "sampler", "sampled_image", "acceleration_structure", "storage_image"
        std::string name;
        std::vector<FieldInfo> fields;
        int size = 0;
        std::vector<std::string> stage_flags;
    };

    struct PushConstantInfo {
        std::string name;
        std::vector<FieldInfo> fields;
        int size = 0;
        std::vector<std::string> stage_flags;
    };

    struct VertexAttribute {
        int location = 0;
        std::string type;
        std::string name;
        std::string format;
        int offset = 0;
    };

    std::vector<VarInfo> inputs;
    std::vector<VarInfo> outputs;
    std::unordered_map<int, std::vector<BindingInfo>> descriptor_sets; // set index -> bindings
    std::vector<PushConstantInfo> push_constants;
    std::vector<VertexAttribute> vertex_attributes;
    int vertex_stride = 0;
};

/**
 * Parse a .lux.json file into a ReflectionData struct.
 */
ReflectionData parseReflectionJson(const std::string& json_path);

/**
 * Map a Vulkan format string from reflection metadata to VkFormat.
 */
VkFormat reflectionFormatToVkFormat(const std::string& fmt);

/**
 * Create descriptor set layouts from merged vertex + fragment reflection data.
 * Returns map of set index -> VkDescriptorSetLayout.
 */
std::unordered_map<int, VkDescriptorSetLayout> createDescriptorSetLayouts(
    VkDevice device,
    const ReflectionData& vertReflection,
    const ReflectionData& fragReflection
);

/**
 * Create descriptor set layouts from multiple stages (e.g. raygen + closest_hit + miss).
 * Returns map of set index -> VkDescriptorSetLayout.
 */
std::unordered_map<int, VkDescriptorSetLayout> createDescriptorSetLayoutsMultiStage(
    VkDevice device,
    const std::vector<ReflectionData>& stages
);

/**
 * Create pipeline layout from descriptor set layouts and push constant ranges.
 */
VkPipelineLayout createReflectedPipelineLayout(
    VkDevice device,
    const std::unordered_map<int, VkDescriptorSetLayout>& setLayouts,
    const ReflectionData& vertReflection,
    const ReflectionData& fragReflection
);

/**
 * Create pipeline layout from descriptor set layouts and push constant ranges (multi-stage).
 */
VkPipelineLayout createReflectedPipelineLayoutMultiStage(
    VkDevice device,
    const std::unordered_map<int, VkDescriptorSetLayout>& setLayouts,
    const std::vector<ReflectionData>& stages
);

/**
 * Get merged binding info from all stages, keyed by (set, binding).
 * Useful for iterating over all bindings when creating descriptor writes.
 */
struct MergedBindingInfo {
    int set;
    int binding;
    std::string type;
    std::string name;
    int size;
    VkShaderStageFlags stageFlags;
};

std::vector<MergedBindingInfo> getMergedBindings(
    const std::vector<ReflectionData>& stages
);

/**
 * Create vertex input state from vertex reflection data.
 */
struct ReflectedVertexInput {
    VkVertexInputBindingDescription binding;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

ReflectedVertexInput createReflectedVertexInput(const ReflectionData& vertReflection);
