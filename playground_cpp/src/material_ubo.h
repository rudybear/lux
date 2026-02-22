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
};
static_assert(sizeof(MaterialUBOData) == 80, "MaterialUBOData must be 80 bytes (std140)");
