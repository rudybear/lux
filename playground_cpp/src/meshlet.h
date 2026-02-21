#pragma once

#include <cstdint>
#include <vector>

// GPU-side meshlet descriptor (matches shader uvec4 layout)
struct MeshletDescriptor {
    uint32_t vertexOffset;
    uint32_t vertexCount;
    uint32_t triangleOffset;
    uint32_t triangleCount;
};

struct MeshletBuildResult {
    std::vector<MeshletDescriptor> meshlets;
    std::vector<uint32_t> meshletVertices;    // global vertex indices per meshlet
    std::vector<uint32_t> meshletTriangles;   // local triangle indices (3 per tri)
};

// Build meshlets from indexed triangle mesh using greedy partitioning
MeshletBuildResult buildMeshlets(
    const uint32_t* indices,
    uint32_t indexCount,
    uint32_t vertexCount,
    uint32_t maxVertices = 64,
    uint32_t maxPrimitives = 124
);
