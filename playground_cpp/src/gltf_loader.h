#pragma once
/**
 * glTF 2.0 scene loader for the C++/Vulkan playground.
 *
 * Uses cgltf (header-only, fetched at build time or vendored in deps/).
 * Produces GPU-ready interleaved vertex data matching Lux PBR vertex layout:
 *   position(vec3) + normal(vec3) + uv(vec2) + tangent(vec4) = 48 bytes per vertex.
 */

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>

// ===========================================================================
// Data structures
// ===========================================================================

struct GltfVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec4 tangent{0.0f, 0.0f, 0.0f, 1.0f};  // tangent with handedness
};
static_assert(sizeof(GltfVertex) == 48, "GltfVertex must be 48 bytes");

struct GltfMesh {
    std::string name;
    std::vector<GltfVertex> vertices;
    std::vector<uint32_t> indices;
    int materialIndex = 0;
    bool hasTangents = false;
    uint32_t vertexStride = sizeof(GltfVertex);
};

struct GltfTextureData {
    std::vector<uint8_t> pixels;   // Decoded RGBA uint8 pixel data
    int width = 0;
    int height = 0;
    bool valid() const { return !pixels.empty() && width > 0 && height > 0; }
};

struct GltfMaterial {
    std::string name;
    glm::vec4 baseColor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic = 0.0f;
    float roughness = 1.0f;
    glm::vec3 emissive{0.0f};
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;

    // --- KHR_materials_* extension fields ---
    bool hasClearcoat = false;
    float clearcoatFactor = 0.0f;
    float clearcoatRoughnessFactor = 0.0f;
    GltfTextureData clearcoat_tex;
    GltfTextureData clearcoat_roughness_tex;

    bool hasSheen = false;
    glm::vec3 sheenColorFactor{0.0f};
    float sheenRoughnessFactor = 0.0f;
    GltfTextureData sheen_color_tex;

    bool hasTransmission = false;
    float transmissionFactor = 0.0f;
    GltfTextureData transmission_tex;

    float ior = 1.5f;
    float emissiveStrength = 1.0f;
    bool isUnlit = false;

    // Decoded texture image data (RGBA8) extracted from GLB
    GltfTextureData base_color_tex;
    GltfTextureData normal_tex;
    GltfTextureData metallic_roughness_tex;
    GltfTextureData occlusion_tex;
    GltfTextureData emissive_tex;
};

struct GltfNode {
    std::string name;
    glm::mat4 localTransform{1.0f};
    glm::mat4 worldTransform{1.0f};
    int meshIndex = -1;
    int cameraIndex = -1;
    int lightIndex = -1;
    std::vector<int> children;
    int parent = -1;
};

struct GltfCamera {
    std::string name;
    std::string type = "perspective";
    float fovY = glm::radians(60.0f);
    float aspect = 1.0f;
    float zNear = 0.01f;
    float zFar = 1000.0f;
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
};

struct GltfLight {
    std::string name;
    std::string type = "directional";  // directional, point, spot
    glm::vec3 color{1.0f};
    float intensity = 1.0f;
    float range = 0.0f;  // 0 = infinite
    float innerConeAngle = 0.0f;
    float outerConeAngle = 0.7854f;  // pi/4
    glm::vec3 position{0.0f};
    glm::vec3 direction{0.0f, 0.0f, -1.0f};
};

struct GltfScene {
    std::vector<GltfMesh> meshes;
    std::vector<GltfMaterial> materials;
    std::vector<GltfNode> nodes;
    std::vector<GltfCamera> cameras;
    std::vector<GltfLight> lights;
    std::vector<int> rootNodes;
};

struct DrawItem {
    glm::mat4 worldTransform;
    int meshIndex;
    int materialIndex;
};

// ===========================================================================
// API
// ===========================================================================

/**
 * Load a .glb or .gltf file.
 * Requires cgltf.h to be available in the include path.
 */
GltfScene loadGltf(const std::string& path);

/**
 * Traverse the scene graph and produce a flat draw list sorted by material.
 */
std::vector<DrawItem> flattenScene(GltfScene& scene);
