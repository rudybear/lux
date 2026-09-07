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
#include <array>
#include <unordered_map>
#include <cstdint>
#include <cmath>

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

struct UVTransform {
    glm::vec2 offset{0.0f};
    glm::vec2 scale{1.0f};
    float rotation = 0.0f;
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

    // KHR_texture_transform
    UVTransform base_color_uv_xform;
    UVTransform normal_uv_xform;
    UVTransform metallic_roughness_uv_xform;

    // Custom properties from glTF extras (for extended bindless structs)
    std::unordered_map<std::string, float> custom_float_properties;
    std::unordered_map<std::string, std::array<float, 4>> custom_vec_properties;
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

// ---------------------------------------------------------------------------
// Dynamic (4D) Gaussian splats: morph-target keyframe animation.
// See docs/lux-4d-spec.md and mobiledlss/gltf/dyn_splat_gltf.py (reference
// writer/reader for the MOBILEDLSS_dynamic_splats convention).
// ---------------------------------------------------------------------------

// One sparse morph target: deltas from the base attributes at `indices`.
// Parallel arrays; indices.size() == dpos.size()/3 == drot.size()/4 == dsh0.size()/3.
struct SplatMorphTarget {
    std::vector<uint32_t> indices;  // global indices into the base splat arrays
    std::vector<float> dpos;        // flat vec3 per sparse entry
    std::vector<float> drot;        // flat vec4 per sparse entry
    std::vector<float> dsh0;        // flat vec3 per sparse entry
};

// One animation keyframe: a time (seconds) and the per-target weight vector
// at that time (as authored -- typically one-hot, weights[0] is all-zero).
struct SplatKeyframe {
    float time = 0.0f;
    std::vector<float> weights;  // length == targets.size()
};

struct SplatDynamics {
    bool has_motion = false;
    std::vector<SplatMorphTarget> targets;    // length K (one per non-base keyframe)
    std::vector<SplatKeyframe> keyframes;     // length K+1, keyframes[0] = base (all-zero weights)
    // mesh.extras.MOBILEDLSS_dynamic_splats, when present (informative --
    // the animation above is authoritative for interpolation semantics).
    std::vector<int> extrasFrames;   // video-frame index per keyframe, parallel to `keyframes`
    bool hasExtrasFps = false;
    float extrasFps = 30.0f;
    std::string rotationBlend = "lerp_renormalize";
};

// Result of evaluating the animation at a point in time: which segment
// (interval between two adjacent keyframes) is active, and the blend
// weights for its low/high target. `hasLow` is false only for the first
// segment, where the "low" side is the (all-zero-delta) base frame.
struct SplatMorphState {
    int highTargetIndex = 0;   // valid index into dynamics.targets
    int lowTargetIndex = -1;   // valid index into dynamics.targets, or -1
    float weightLow = 0.0f;
    float weightHigh = 1.0f;
};

struct GaussianSplatData {
    std::vector<float> positions;    // xyz packed as vec4 (w=1)
    std::vector<float> rotations;    // xyzw quaternion per splat
    std::vector<float> scales;       // log-space xyz per splat
    std::vector<float> opacities;    // logit-space per splat
    // Per-gaussian foreground/actor coverage flag, 0.0/1.0, length == num_splats.
    // Sourced from the custom `_FOREGROUND` vertex attribute (SCALAR, UNSIGNED_BYTE
    // normalized, 255 = foreground) when the primitive has one; else falls back to
    // "gaussian has any morph-target delta" (dynamics.targets[*].indices union);
    // else all-zero. See docs/lux-4d-spec.md and splat_expander.py's
    // `foreground_coverage` config flag (packs this into out_depth's .g channel).
    std::vector<float> foreground;
    std::vector<std::vector<float>> sh_coefficients; // per-degree SH coefficients
    uint32_t sh_degree = 0;
    uint32_t num_splats = 0;
    bool has_splats = false;
    bool khr_format = false;  // true when loaded from KHR_gaussian_splatting attributes
    // KHR_gaussian_splatting extension-object metadata (ratified spec), when present.
    // "srgb_rec709_display" (default assumption) or "lin_rec709_display"; the extension
    // is optional, and the field defaults to the spec's implicit sRGB assumption when absent.
    std::string color_space = "srgb_rec709_display";
    std::string kernel = "ellipse";
    // Dynamic (morph-target-animated) splats, when the source glTF's splat
    // primitive has `targets`. Static assets leave this default-constructed
    // (has_motion == false).
    SplatDynamics dynamics;
};

/**
 * Evaluate the animation at `timeSeconds`, clamped to the keyframe range.
 * Returns which target(s) are active and their blend weights.
 */
SplatMorphState evaluateSplatMorphState(const SplatDynamics& dyn, float timeSeconds);

/**
 * Map an integer video-frame index to a time in seconds: `frame / fps` when
 * `mesh.extras.MOBILEDLSS_dynamic_splats.fps` is present, otherwise index
 * directly into the animation's own keyframe times (clamped).
 */
float splatFrameToTime(const SplatDynamics& dyn, int frame);

// A precomputed per-segment (union of the segment's low+high target sparse
// indices) delta list, ready to upload as the morph-apply compute shader's
// `morph_index` / `morph_delta_*_lo` / `morph_delta_*_hi` GPU buffers. See
// SPECIFICATION.md 12.8 and docs/language-reference.md.
struct SplatMorphSegment {
    std::vector<uint32_t> index;      // merged, de-duplicated indices
    std::vector<float> dposLo, dposHi;   // flat vec3 per merged entry (0 where absent)
    std::vector<float> drotLo, drotHi;   // flat vec4 per merged entry
    std::vector<float> dsh0Lo, dsh0Hi;   // flat vec3 per merged entry
};

/**
 * Build the K precomputed per-segment merged delta lists from `dyn.targets`
 * (segment i pairs low=targets[i-1] (or none if i==0) with high=targets[i]).
 * Call once at load time; segments don't depend on animation time.
 */
std::vector<SplatMorphSegment> buildSplatMorphSegments(const SplatDynamics& dyn);

struct GltfScene {
    std::vector<GltfMesh> meshes;
    std::vector<GltfMaterial> materials;
    std::vector<GltfNode> nodes;
    std::vector<GltfCamera> cameras;
    std::vector<GltfLight> lights;
    std::vector<int> rootNodes;
    // Maps glTF mesh index -> (start, count) in meshes vec (one entry per primitive)
    std::vector<std::pair<size_t, size_t>> meshPrimitiveRanges;
    // KHR_gaussian_splatting data (populated when extension is present)
    GaussianSplatData splat_data;
};

struct DrawItem {
    glm::mat4 worldTransform;
    int meshIndex;
    int materialIndex;
};

struct DrawRange {
    uint32_t indexOffset;
    uint32_t indexCount;
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
