#pragma once
/**
 * ReflectedPipeline — data-driven Vulkan pipeline setup from .lux.json reflection metadata.
 *
 * Reads .lux.json sidecar files and creates VkDescriptorSetLayout, VkPipelineLayout,
 * VkPipeline, and vertex input state automatically — no hardcoded descriptor sets needed.
 *
 * On Metal (LUX_METAL_BACKEND): provides format mappings and shared manifest/permutation logic.
 */

#ifndef LUX_METAL_BACKEND
#include <vulkan/vulkan.h>
#endif
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <optional>
#include <cstdint>

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
        std::string type; // "uniform_buffer", "sampler", "sampled_image", "acceleration_structure", "storage_image", "storage_buffer", "bindless_combined_image_sampler_array"
        std::string name;
        std::vector<FieldInfo> fields;
        int size = 0;
        int max_count = 0;  // for bindless arrays (e.g. 1024)
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

    struct MeshOutputInfo {
        uint32_t maxVertices = 64;
        uint32_t maxPrimitives = 124;
        uint32_t workgroupSize = 32;
        std::string outputTopology = "triangles";
    };
    std::optional<MeshOutputInfo> meshOutput;

    // Bindless descriptor support
    bool bindlessEnabled = false;

    // Dynamic bindless struct layout (from reflection "bindless" section)
    uint32_t bindlessStructSize = 128;  // default to core 128-byte struct
    bool bindlessHasCustomProperties = false;
    struct BindlessFieldMeta {
        std::string name;
        std::string type;
        uint32_t offset = 0;
        uint32_t size = 0;
    };
    std::vector<BindlessFieldMeta> bindlessStructFields;
};

/**
 * Parse a .lux.json file into a ReflectionData struct.
 */
ReflectionData parseReflectionJson(const std::string& json_path);

#ifndef LUX_METAL_BACKEND

/**
 * Map a Vulkan format string from reflection metadata to VkFormat.
 */
VkFormat reflectionFormatToVkFormat(const std::string& fmt);

/**
 * Map a binding type string to VkDescriptorType.
 */
VkDescriptorType bindingTypeToVkDescriptorType(const std::string& type);

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
    int max_count = 0;  // for bindless arrays
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

ReflectedVertexInput createReflectedVertexInput(const ReflectionData& vertReflection, int overrideStride = 0);

#endif // !LUX_METAL_BACKEND

// Metal format mappings (available on Apple platforms with Metal backend)
#if defined(__APPLE__) && defined(LUX_METAL_BACKEND)
#include <Metal/Metal.hpp>
MTL::VertexFormat reflectionFormatToMTLVertexFormat(const std::string& fmt);
MTL::PixelFormat reflectionFormatToMTLPixelFormat(const std::string& fmt);
#endif

/**
 * Shader manifest: lists all available permutations for a pipeline.
 */
struct ManifestPermutation {
    std::string suffix;                                // e.g. "+normal_map+sheen"
    std::unordered_map<std::string, bool> features;    // feature name -> enabled
};

struct ShaderManifest {
    std::string pipeline;                              // e.g. "GltfForward"
    std::vector<std::string> featureNames;             // all declared features
    std::vector<ManifestPermutation> permutations;     // all permutations
};

/**
 * Parse a .manifest.json file into a ShaderManifest struct.
 * Returns empty manifest if file doesn't exist.
 */
ShaderManifest parseManifestJson(const std::string& manifest_path);

/**
 * Try to load manifest from a pipeline base path.
 * Checks for basePath + ".manifest.json" and the legacy gltf_pbr/ subdirectory.
 */
ShaderManifest tryLoadManifest(const std::string& pipelineBase);

/**
 * Find the best matching permutation suffix for a set of material features.
 * Returns "" (base) if no exact match found.
 */
std::string findPermutationSuffix(const ShaderManifest& manifest,
                                   const std::set<std::string>& materialFeatures);
