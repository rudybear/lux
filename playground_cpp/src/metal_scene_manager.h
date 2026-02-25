#pragma once

#include "metal_scene.h"
#include "metal_context.h"
#include "gltf_loader.h"
#include "scene_geometry.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

class MetalSceneManager {
public:
    void loadScene(const std::string& sceneSource);
    void uploadToGPU(MetalContext& ctx, int vertexStride = 32);
    void uploadTextures(MetalContext& ctx);
    void loadIBLAssets(MetalContext& ctx, const std::string& requestedName = "");

    // Auto-camera
    bool hasSceneBounds() const { return m_hasSceneBounds; }
    glm::vec3 getAutoEye() const { return m_autoEye; }
    glm::vec3 getAutoTarget() const { return m_autoTarget; }
    glm::vec3 getAutoUp() const { return m_autoUp; }
    float getAutoFar() const { return m_autoFar; }

    // Resource access
    const MetalGPUMesh& getMesh() const { return m_mesh; }
    const GltfScene& getGltfScene() const { return m_gltfScene; }
    bool hasGltfScene() const { return m_hasGltfScene; }
    const std::string& getSceneSource() const { return m_sceneSource; }

    // Per-material draw ranges and textures
    const std::vector<MetalDrawRange>& getDrawRanges() const { return m_drawRanges; }
    const std::vector<std::unordered_map<std::string, MetalGPUTexture>>& getPerMaterialTextures() const { return m_perMaterialTextures; }

    // IBL textures
    MetalGPUTexture& getIBLTexture(const std::string& name) { return m_iblTextures[name]; }
    bool hasIBLTexture(const std::string& name) const { return m_iblTextures.count(name) > 0; }

    // Named textures (procedural, etc.)
    MetalGPUTexture& getNamedTexture(const std::string& name) { return m_namedTextures[name]; }

    // Default textures
    MetalGPUTexture& getDefaultWhite() { return m_defaultWhiteTexture; }
    MetalGPUTexture& getDefaultBlack() { return m_defaultBlackTexture; }
    MetalGPUTexture& getDefaultNormal() { return m_defaultNormalTexture; }

    // Texture lookup for binding by name (checks per-material, named, IBL, defaults)
    MetalGPUTexture& getTextureForBinding(const std::string& name, int materialIndex = -1);

    // Scene feature detection
    std::set<std::string> detectSceneFeatures() const;
    std::set<std::string> detectMaterialFeatures(int materialIndex) const;
    std::map<std::string, std::vector<int>> groupMaterialsByFeatures() const;

    // Vertex/index data access (for SoA uploads in mesh renderer)
    const std::vector<Vertex>& getVertices() const { return m_vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_indices; }

    void cleanup();

    static bool isGltfFile(const std::string& source);

private:
    std::string m_sceneSource;

    // Scene geometry
    std::vector<Vertex> m_vertices;
    std::vector<uint32_t> m_indices;
    MetalGPUMesh m_mesh = {};

    // glTF scene data
    GltfScene m_gltfScene;
    bool m_hasGltfScene = false;

    // Textures
    std::unordered_map<std::string, MetalGPUTexture> m_namedTextures;
    std::unordered_map<std::string, MetalGPUTexture> m_iblTextures;
    MetalGPUTexture m_defaultWhiteTexture = {};
    MetalGPUTexture m_defaultBlackTexture = {};
    MetalGPUTexture m_defaultNormalTexture = {};

    // Per-material draw ranges and textures
    std::vector<MetalDrawRange> m_drawRanges;
    std::vector<std::unordered_map<std::string, MetalGPUTexture>> m_perMaterialTextures;

    // Auto-camera
    bool m_hasSceneBounds = false;
    glm::vec3 m_autoEye = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 m_autoTarget = glm::vec3(0.0f);
    glm::vec3 m_autoUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float m_autoFar = 100.0f;

    void createDefaultTextures(MetalContext& ctx);
    void uploadGltfTextures(MetalContext& ctx);
    void computeAutoCamera(const std::vector<Vertex>& vertices);
};
