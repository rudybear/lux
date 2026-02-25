#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

// PBR sphere vertex: position + normal + UV = 32 bytes
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
};
static_assert(sizeof(Vertex) == 32, "Vertex must be 32 bytes");

// Simple colored triangle vertex: position + color = 24 bytes
struct TriangleVertex {
    glm::vec3 position;
    glm::vec3 color;
};
static_assert(sizeof(TriangleVertex) == 24, "TriangleVertex must be 24 bytes");

namespace Scene {

// Generate a UV sphere mesh
void generateSphere(uint32_t stacks, uint32_t slices,
                    std::vector<Vertex>& outVertices,
                    std::vector<uint32_t>& outIndices);

// Generate the hardcoded colored triangle (3 vertices in clip space)
void generateTriangle(std::vector<TriangleVertex>& outVertices);

// Generate procedural checker texture (RGBA8, size x size)
std::vector<uint8_t> generateProceduralTexture(uint32_t size);

} // namespace Scene
