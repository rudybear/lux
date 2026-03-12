#pragma once

#include <glm/glm.hpp>
#include <vector>

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

// Pack lights into GPU buffer (std430 layout: 64 bytes per light)
inline std::vector<float> packLightsBuffer(const std::vector<SceneLight>& lights) {
    std::vector<float> buf;
    for (const auto& l : lights) {
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

// Pack shadow entries into GPU buffer (80 bytes per entry: mat4 + 4 floats)
inline std::vector<float> packShadowBuffer(const std::vector<ShadowEntry>& entries) {
    std::vector<float> buf;
    for (const auto& entry : entries) {
        // mat4 viewProjection (column-major, 16 floats)
        const float* mat = &entry.viewProjection[0][0];
        for (int i = 0; i < 16; i++) {
            buf.push_back(mat[i]);
        }
        // bias, normalBias, resolution, light_size
        buf.push_back(entry.bias);
        buf.push_back(entry.normalBias);
        buf.push_back(entry.resolution);
        buf.push_back(entry.light_size);
    }
    return buf;
}
