#pragma once
/**
 * MaterialUBOData: CPU-side struct matching the std140 layout of the
 * `properties Material { ... }` block in gltf_pbr_layered.lux.
 *
 * Field order and alignment must match the shader declaration exactly.
 */

#include <glm/glm.hpp>
#include <cstdint>

struct MaterialUBOData {
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};  // offset  0, size 16
    glm::vec3 emissiveFactor{0.0f};                        // offset 16, size 12
    float metallicFactor = 1.0f;                            // offset 28, size  4
    float roughnessFactor = 1.0f;                           // offset 32, size  4
    float emissiveStrength = 1.0f;                          // offset 36, size  4
    float ior = 1.5f;                                       // offset 40, size  4
    float clearcoatFactor = 0.0f;                           // offset 44, size  4
    float clearcoatRoughnessFactor = 0.0f;                  // offset 48, size  4
    float sheenRoughnessFactor = 0.0f;                      // offset 52, size  4
    float transmissionFactor = 0.0f;                        // offset 56, size  4
    float _pad_before_sheen = 0.0f;                         // offset 60, size  4 (pad to align vec3 to 16)
    glm::vec3 sheenColorFactor{0.0f};                       // offset 64, size 12
    float _pad0 = 0.0f;                                     // offset 76, size  4
    // KHR_texture_transform (offset 80 -> 144)
    glm::vec4 baseColorUvSt{0.0f, 0.0f, 1.0f, 1.0f};      // offset 80: offset.xy, scale.xy
    glm::vec4 normalUvSt{0.0f, 0.0f, 1.0f, 1.0f};          // offset 96
    glm::vec4 mrUvSt{0.0f, 0.0f, 1.0f, 1.0f};              // offset 112
    float baseColorUvRot = 0.0f;                              // offset 128
    float normalUvRot = 0.0f;                                 // offset 132
    float mrUvRot = 0.0f;                                     // offset 136
    float _pad1 = 0.0f;                                       // offset 140
};
static_assert(sizeof(MaterialUBOData) == 144, "MaterialUBOData must be 144 bytes (std140)");

// Material flags bits for bindless rendering
constexpr uint32_t BINDLESS_MAT_FLAG_NORMAL_MAP    = 0x1;
constexpr uint32_t BINDLESS_MAT_FLAG_CLEARCOAT     = 0x2;
constexpr uint32_t BINDLESS_MAT_FLAG_SHEEN         = 0x4;
constexpr uint32_t BINDLESS_MAT_FLAG_EMISSION      = 0x8;
constexpr uint32_t BINDLESS_MAT_FLAG_TRANSMISSION  = 0x10;

/**
 * BindlessMaterialData: CPU-side struct matching the std430 layout of the
 * BindlessMaterialData struct in the bindless uber-shader SSBO.
 *
 * Field order and alignment must match the SPIR-V struct exactly (128-byte stride).
 */
struct BindlessMaterialData {
    glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};  // offset 0
    glm::vec3 emissiveFactor{0.0f};                        // offset 16
    float metallicFactor = 1.0f;                            // offset 28
    float roughnessFactor = 1.0f;                           // offset 32
    float emissiveStrength = 1.0f;                          // offset 36
    float ior = 1.5f;                                       // offset 40
    float clearcoatFactor = 0.0f;                           // offset 44
    float clearcoatRoughness = 0.0f;                        // offset 48
    float transmissionFactor = 0.0f;                        // offset 52 (matches shader index 8)
    float sheenRoughness = 0.0f;                            // offset 56 (matches shader index 9)
    float _pad_before_sheen = 0.0f;                         // offset 60
    glm::vec3 sheenColorFactor{0.0f};                       // offset 64
    float _pad_after_sheen = 0.0f;                          // offset 76
    int32_t baseColorTexIndex = -1;                         // offset 80
    int32_t normalTexIndex = -1;                            // offset 84
    int32_t metallicRoughnessTexIndex = -1;                 // offset 88
    int32_t occlusionTexIndex = -1;                         // offset 92
    int32_t emissiveTexIndex = -1;                          // offset 96
    int32_t clearcoatTexIndex = -1;                         // offset 100
    int32_t clearcoatRoughnessTexIndex = -1;                // offset 104
    int32_t sheenColorTexIndex = -1;                        // offset 108
    int32_t transmissionTexIndex = -1;                      // offset 112
    uint32_t materialFlags = 0;                             // offset 116
    float indexOffset = 0.0f;                               // offset 120 — per-geometry index offset for RT (float for shader compat)
    uint32_t _padding = 0;                                  // offset 124
};  // total = 128 bytes
static_assert(sizeof(BindlessMaterialData) == 128, "BindlessMaterialData must be 128 bytes (std430)");
