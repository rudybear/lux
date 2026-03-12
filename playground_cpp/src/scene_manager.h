#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "gltf_loader.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include "material_ubo.h"
#include "splat_renderer.h"
#include <glm/glm.hpp>
#include <memory>

// GPU-compatible scene light representation (64 bytes per light, std430 layout)
struct SceneLight {
    enum Type { Directional = 0, Point = 1, Spot = 2 };
    Type type = Directional;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 0.0f;          // 0 = infinite
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7854f; // pi/4
    bool castsShadow = false;
    int shadowIndex = -1;
};

// Shadow cascade data for directional light CSM
struct ShadowCascade {
    glm::mat4 viewProjection;
    float splitDepth;
};

// Shadow map entry: packed for GPU (std430 layout, 80 bytes)
struct ShadowEntry {
    glm::mat4 viewProjection{1.0f};
    float bias = 0.005f;
    float normalBias = 0.02f;
    float resolution = 1024.0f;
    float light_size = 0.02f;
};

// Bindless texture array: all unique textures from the scene for descriptor binding
struct BindlessTextureArray {
    std::vector<VkImageView> imageViews;
    std::vector<VkSampler> samplers;
    uint32_t textureCount = 0;
};

// Bindless materials SSBO: all materials packed into a single storage buffer
struct BindlessMaterialsSSBO {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t materialCount = 0;
};

// Shared scene management: loads scenes, uploads meshes/textures/IBL, computes auto-camera.
// Used by both RTRenderer and RasterRenderer to avoid code duplication.
class SceneManager {
public:
    void loadScene(VulkanContext& ctx, const std::string& sceneSource);
    void loadProceduralTestScene(VulkanContext& ctx);
    void uploadToGPU(VulkanContext& ctx, int vertexStride = 32);
    void uploadTextures(VulkanContext& ctx);
    void loadIBLAssets(VulkanContext& ctx, VkPipelineStageFlags dstStage,
                       const std::string& requestedName = "");

    // Auto-camera
    bool hasSceneBounds() const { return m_hasSceneBounds; }
    glm::vec3 getAutoEye() const { return m_autoEye; }
    glm::vec3 getAutoTarget() const { return m_autoTarget; }
    glm::vec3 getAutoUp() const { return m_autoUp; }
    float getAutoFar() const { return m_autoFar; }
    void overrideAutoCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up, float farPlane) {
        m_autoEye = eye; m_autoTarget = target; m_autoUp = up; m_autoFar = farPlane;
        m_hasSceneBounds = true;
    }

    // Resource access
    GPUTexture& getTextureForBinding(const std::string& name);
    const std::vector<Vertex>& getVertices() const { return m_vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_indices; }
    const GPUMesh& getMesh() const { return m_mesh; }
    const GltfScene& getGltfScene() const { return m_gltfScene; }
    bool hasGltfScene() const { return m_hasGltfScene; }
    const std::string& getSceneSource() const { return m_sceneSource; }

    // Light management
    void addLight(const SceneLight& light) { m_lights.push_back(light); }
    void clearLights() { m_lights.clear(); }
    const std::vector<SceneLight>& getLights() const { return m_lights; }
    std::vector<SceneLight>& getLightsMutable() { return m_lights; }
    int getLightCount() const { return static_cast<int>(m_lights.size()); }

    // Convert glTF lights to SceneLights (called after scene load)
    void populateLightsFromGltf(const GltfScene& gltfScene) {
        m_lights.clear();
        for (const auto& gl : gltfScene.lights) {
            SceneLight sl;
            if (gl.type == "directional") sl.type = SceneLight::Directional;
            else if (gl.type == "point") sl.type = SceneLight::Point;
            else if (gl.type == "spot") sl.type = SceneLight::Spot;
            sl.position = gl.position;
            sl.direction = gl.direction;
            sl.color = gl.color;
            sl.intensity = gl.intensity;
            sl.range = gl.range;
            sl.innerConeAngle = gl.innerConeAngle;
            sl.outerConeAngle = gl.outerConeAngle;
            m_lights.push_back(sl);
        }
        // Add a default directional light if scene has none
        if (m_lights.empty()) {
            SceneLight defaultLight;
            defaultLight.type = SceneLight::Directional;
            defaultLight.direction = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
            defaultLight.color = glm::vec3(1.0f, 0.98f, 0.95f);
            defaultLight.intensity = 1.0f;
            m_lights.push_back(defaultLight);
        }
    }

    // Pack lights into GPU buffer (std430 layout: 64 bytes per light)
    std::vector<float> packLightsBuffer() const {
        std::vector<float> buf;
        for (const auto& l : m_lights) {
            // vec4: (light_type, intensity, range, inner_cone)
            buf.push_back(static_cast<float>(l.type));
            buf.push_back(l.intensity);
            buf.push_back(l.range);
            buf.push_back(l.innerConeAngle);
            // vec4: (position.xyz, outer_cone)
            buf.push_back(l.position.x);
            buf.push_back(l.position.y);
            buf.push_back(l.position.z);
            buf.push_back(l.outerConeAngle);
            // vec4: (direction.xyz, shadow_index)
            buf.push_back(l.direction.x);
            buf.push_back(l.direction.y);
            buf.push_back(l.direction.z);
            buf.push_back(static_cast<float>(l.shadowIndex));
            // vec4: (color.xyz, pad)
            buf.push_back(l.color.x);
            buf.push_back(l.color.y);
            buf.push_back(l.color.z);
            buf.push_back(0.0f); // pad
        }
        return buf;
    }

    // Shadow map support
    void computeShadowData(const glm::mat4& cameraView, const glm::mat4& cameraProj,
                           float nearClip, float farClip);
    std::vector<float> packShadowBuffer() const;
    const std::vector<ShadowEntry>& getShadowEntries() const { return m_shadowEntries; }
    int getShadowMapCount() const { return static_cast<int>(m_shadowEntries.size()); }

    // Per-material draw ranges and textures for multi-material rendering
    const std::vector<DrawRange>& getDrawRanges() const { return m_drawRanges; }
    const std::vector<std::unordered_map<std::string, GPUTexture>>& getPerMaterialTextures() const { return m_perMaterialTextures; }

    // IBL texture access for descriptor binding
    std::unordered_map<std::string, GPUTexture>& getIBLTextures() { return m_iblTextures; }
    std::unordered_map<std::string, GPUTexture>& getNamedTextures() { return m_namedTextures; }

    void cleanup(VulkanContext& ctx);

    // Bindless support: build texture array and materials SSBO from scene data
    BindlessTextureArray buildBindlessTextureArray(VulkanContext& ctx);
    BindlessMaterialsSSBO buildMaterialsSSBO(VulkanContext& ctx, const BindlessTextureArray& texArray);
    BindlessMaterialsSSBO buildMaterialsSSBOByGeometry(VulkanContext& ctx, const BindlessTextureArray& texArray);

    // Scene feature detection for dynamic pipeline selection
    std::set<std::string> detectSceneFeatures() const;
    static std::string buildPipelinePath(const std::string& basePath,
                                          const std::set<std::string>& features);

    // Per-material feature detection for permutation selection
    std::set<std::string> detectMaterialFeatures(int materialIndex) const;

    // Group materials by feature set -> {suffix: [material_indices]}
    std::map<std::string, std::vector<int>> groupMaterialsByFeatures() const;

    // Build permutation suffix from features -> "+normal_map+sheen"
    static std::string featuresToSuffix(const std::set<std::string>& features);

    static bool isGltfFile(const std::string& source);

    // Gaussian splatting support
    bool hasSplatData() const { return m_hasGltfScene && m_gltfScene.splat_data.has_splats; }
    SplatRenderer* getSplatRenderer() { return splat_renderer_.get(); }
    void initSplatRenderer(VulkanContext& ctx, const std::string& shaderBase, uint32_t width, uint32_t height);

private:
    std::string m_sceneSource;

    // Scene geometry
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::vector<Vertex> m_splatBoundsVerts;  // splat positions as vertices for auto-camera
    GPUMesh m_mesh = {};

    // glTF scene data
    GltfScene m_gltfScene;
    bool m_hasGltfScene = false;

    // Scene lights (populated from glTF or added manually)
    std::vector<SceneLight> m_lights;

    // Shadow map entries (populated by computeShadowData)
    std::vector<ShadowEntry> m_shadowEntries;

    // Textures
    std::unordered_map<std::string, GPUTexture> m_namedTextures;
    std::unordered_map<std::string, GPUTexture> m_iblTextures;
    GPUTexture m_defaultWhiteTexture = {};
    GPUTexture m_defaultBlackTexture = {};
    GPUTexture m_defaultNormalTexture = {};

    // Per-material draw ranges for multi-draw
    std::vector<DrawRange> m_drawRanges;

    // Per-material textures (one map per material index)
    std::vector<std::unordered_map<std::string, GPUTexture>> m_perMaterialTextures;

    // Auto-camera
    bool m_hasSceneBounds = false;
    glm::vec3 m_autoEye = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 m_autoTarget = glm::vec3(0.0f);
    glm::vec3 m_autoUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float m_autoFar = 100.0f;

    // Bindless texture dedup map: VkImage -> index in bindless texture array
    std::unordered_map<uint64_t, int32_t> m_bindlessTexDedup;

    // Gaussian splatting renderer (created when scene has splat data)
    std::unique_ptr<SplatRenderer> splat_renderer_;

    void createDefaultTextures(VulkanContext& ctx);
    void uploadGltfTextures(VulkanContext& ctx);
    GPUTexture uploadCubemapF16(VulkanContext& ctx, uint32_t faceSize, uint32_t mipCount,
                                const std::vector<uint16_t>& data, VkPipelineStageFlags dstStage);
    void computeAutoCamera(const std::vector<Vertex>& vertices);
};
