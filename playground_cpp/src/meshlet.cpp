#include "meshlet.h"
#include <unordered_set>
#include <algorithm>

MeshletBuildResult buildMeshlets(
    const uint32_t* indices,
    uint32_t indexCount,
    uint32_t vertexCount,
    uint32_t maxVertices,
    uint32_t maxPrimitives)
{
    (void)vertexCount; // unused but kept for API symmetry

    MeshletBuildResult result;

    uint32_t triangleCount = indexCount / 3;
    if (triangleCount == 0) return result;

    // Current meshlet state
    std::unordered_set<uint32_t> currentVertices;
    std::vector<uint32_t> currentVertexList;   // ordered vertex list
    std::vector<uint32_t> currentTriangles;    // local indices (3 per tri)

    auto finalizeMeshlet = [&]() {
        if (currentTriangles.empty()) return;

        MeshletDescriptor desc;
        desc.vertexOffset = static_cast<uint32_t>(result.meshletVertices.size());
        desc.vertexCount = static_cast<uint32_t>(currentVertexList.size());
        desc.triangleOffset = static_cast<uint32_t>(result.meshletTriangles.size());
        desc.triangleCount = static_cast<uint32_t>(currentTriangles.size() / 3);

        result.meshlets.push_back(desc);
        result.meshletVertices.insert(result.meshletVertices.end(),
            currentVertexList.begin(), currentVertexList.end());
        result.meshletTriangles.insert(result.meshletTriangles.end(),
            currentTriangles.begin(), currentTriangles.end());

        currentVertices.clear();
        currentVertexList.clear();
        currentTriangles.clear();
    };

    for (uint32_t t = 0; t < triangleCount; ++t) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];

        // Count how many new vertices this triangle would add
        uint32_t newVerts = 0;
        if (currentVertices.find(i0) == currentVertices.end()) newVerts++;
        if (currentVertices.find(i1) == currentVertices.end()) newVerts++;
        if (currentVertices.find(i2) == currentVertices.end()) newVerts++;

        // Check if adding this triangle would exceed limits
        if (currentVertexList.size() + newVerts > maxVertices ||
            currentTriangles.size() / 3 + 1 > maxPrimitives) {
            finalizeMeshlet();
        }

        // Add vertices (get or assign local index)
        auto getOrAddVertex = [&](uint32_t globalIdx) -> uint32_t {
            if (currentVertices.find(globalIdx) == currentVertices.end()) {
                uint32_t localIdx = static_cast<uint32_t>(currentVertexList.size());
                currentVertices.insert(globalIdx);
                currentVertexList.push_back(globalIdx);
                return localIdx;
            }
            // Find local index
            for (uint32_t i = 0; i < currentVertexList.size(); ++i) {
                if (currentVertexList[i] == globalIdx) return i;
            }
            return 0; // Should not happen
        };

        uint32_t l0 = getOrAddVertex(i0);
        uint32_t l1 = getOrAddVertex(i1);
        uint32_t l2 = getOrAddVertex(i2);

        currentTriangles.push_back(l0);
        currentTriangles.push_back(l1);
        currentTriangles.push_back(l2);
    }

    // Finalize last meshlet
    finalizeMeshlet();

    return result;
}
