#include "scene_geometry.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace Scene {

void generateSphere(uint32_t stacks, uint32_t slices,
                    std::vector<Vertex>& outVertices,
                    std::vector<uint32_t>& outIndices) {
    outVertices.clear();
    outIndices.clear();

    for (uint32_t i = 0; i <= stacks; i++) {
        float phi = static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(stacks);
        float vCoord = static_cast<float>(i) / static_cast<float>(stacks);

        for (uint32_t j = 0; j <= slices; j++) {
            float theta = 2.0f * static_cast<float>(M_PI) * static_cast<float>(j) / static_cast<float>(slices);
            float uCoord = static_cast<float>(j) / static_cast<float>(slices);

            float x = std::sin(phi) * std::cos(theta);
            float y = std::cos(phi);
            float z = std::sin(phi) * std::sin(theta);

            Vertex v;
            v.position = glm::vec3(x, y, z);
            v.normal = glm::vec3(x, y, z);
            v.uv = glm::vec2(uCoord, vCoord);
            outVertices.push_back(v);
        }
    }

    for (uint32_t i = 0; i < stacks; i++) {
        for (uint32_t j = 0; j < slices; j++) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;

            outIndices.push_back(a);
            outIndices.push_back(b);
            outIndices.push_back(a + 1);

            outIndices.push_back(b);
            outIndices.push_back(b + 1);
            outIndices.push_back(a + 1);
        }
    }
}

void generateTriangle(std::vector<TriangleVertex>& outVertices) {
    outVertices.clear();
    outVertices.push_back({{0.0f, 0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}});
    outVertices.push_back({{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}});
    outVertices.push_back({{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}});
}

std::vector<uint8_t> generateProceduralTexture(uint32_t size) {
    std::vector<uint8_t> pixels(size * size * 4);

    for (uint32_t y = 0; y < size; y++) {
        for (uint32_t x = 0; x < size; x++) {
            float u = static_cast<float>(x) / static_cast<float>(size);
            float v = static_cast<float>(y) / static_cast<float>(size);

            int cx = static_cast<int>(u * 8.0f) % 2;
            int cy = static_cast<int>(v * 8.0f) % 2;
            int checker = cx ^ cy;

            int r, g, b;
            if (checker) {
                r = static_cast<int>(200.0f + 40.0f * std::sin(u * 12.0f));
                g = static_cast<int>(120.0f + 30.0f * std::sin(v * 10.0f));
                b = static_cast<int>(80.0f + 20.0f * std::cos(u * 8.0f + v * 6.0f));
            } else {
                r = static_cast<int>(60.0f + 30.0f * std::sin(u * 10.0f + v * 4.0f));
                g = static_cast<int>(150.0f + 40.0f * std::cos(v * 8.0f));
                b = static_cast<int>(170.0f + 50.0f * std::sin(u * 6.0f));
            }

            uint32_t idx = (y * size + x) * 4;
            pixels[idx + 0] = static_cast<uint8_t>(std::clamp(r, 0, 255));
            pixels[idx + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
            pixels[idx + 2] = static_cast<uint8_t>(std::clamp(b, 0, 255));
            pixels[idx + 3] = 255;
        }
    }

    return pixels;
}

} // namespace Scene
