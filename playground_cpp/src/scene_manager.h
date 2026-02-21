#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "gltf_loader.h"
#include <string>
#include <vector>
#include <unordered_map>

// Shared scene management: loads scenes, uploads meshes/textures/IBL, computes auto-camera.
// Used by both RTRenderer and RasterRenderer to avoid code duplication.
class SceneManager {
public:
    void loadScene(VulkanContext& ctx, const std::string& sceneSource);
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

    // Resource access
    GPUTexture& getTextureForBinding(const std::string& name);
    const std::vector<Vertex>& getVertices() const { return m_vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_indices; }
    const GPUMesh& getMesh() const { return m_mesh; }
    const GltfScene& getGltfScene() const { return m_gltfScene; }
    bool hasGltfScene() const { return m_hasGltfScene; }
    const std::string& getSceneSource() const { return m_sceneSource; }

    // IBL texture access for descriptor binding
    std::unordered_map<std::string, GPUTexture>& getIBLTextures() { return m_iblTextures; }
    std::unordered_map<std::string, GPUTexture>& getNamedTextures() { return m_namedTextures; }

    void cleanup(VulkanContext& ctx);

    static bool isGltfFile(const std::string& source);

private:
    std::string m_sceneSource;

    // Scene geometry
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    GPUMesh m_mesh = {};

    // glTF scene data
    GltfScene m_gltfScene;
    bool m_hasGltfScene = false;

    // Textures
    std::unordered_map<std::string, GPUTexture> m_namedTextures;
    std::unordered_map<std::string, GPUTexture> m_iblTextures;
    GPUTexture m_defaultWhiteTexture = {};
    GPUTexture m_defaultBlackTexture = {};
    GPUTexture m_defaultNormalTexture = {};

    // Auto-camera
    bool m_hasSceneBounds = false;
    glm::vec3 m_autoEye = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 m_autoTarget = glm::vec3(0.0f);
    glm::vec3 m_autoUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float m_autoFar = 100.0f;

    void createDefaultTextures(VulkanContext& ctx);
    void uploadGltfTextures(VulkanContext& ctx);
    GPUTexture uploadCubemapF16(VulkanContext& ctx, uint32_t faceSize, uint32_t mipCount,
                                const std::vector<uint16_t>& data, VkPipelineStageFlags dstStage);
    void computeAutoCamera(const std::vector<Vertex>& vertices);
};
