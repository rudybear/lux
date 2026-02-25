#include "rt_renderer.h"
#include "spv_loader.h"
#include "camera.h"
#include "reflected_pipeline.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <algorithm>

// --------------------------------------------------------------------------
// Alignment helper
// --------------------------------------------------------------------------

static VkDeviceSize alignUp(VkDeviceSize size, VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// --------------------------------------------------------------------------
// Helper: fill MaterialUBOData from a GltfMaterial
// --------------------------------------------------------------------------

static MaterialUBOData materialDataFromGltf(const GltfMaterial& mat) {
    MaterialUBOData d{};
    d.baseColorFactor = mat.baseColor;
    d.metallicFactor = mat.metallic;
    d.roughnessFactor = mat.roughness;
    d.emissiveFactor = mat.emissive;
    d.emissiveStrength = mat.emissiveStrength;
    d.ior = mat.ior;
    d.clearcoatFactor = mat.clearcoatFactor;
    d.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
    d.sheenColorFactor = mat.sheenColorFactor;
    d.sheenRoughnessFactor = mat.sheenRoughnessFactor;
    d.transmissionFactor = mat.transmissionFactor;
    d.baseColorUvSt = glm::vec4(mat.base_color_uv_xform.offset, mat.base_color_uv_xform.scale);
    d.normalUvSt = glm::vec4(mat.normal_uv_xform.offset, mat.normal_uv_xform.scale);
    d.mrUvSt = glm::vec4(mat.metallic_roughness_uv_xform.offset, mat.metallic_roughness_uv_xform.scale);
    d.baseColorUvRot = mat.base_color_uv_xform.rotation;
    d.normalUvRot = mat.normal_uv_xform.rotation;
    d.mrUvRot = mat.metallic_roughness_uv_xform.rotation;
    return d;
}

// --------------------------------------------------------------------------
// Storage image creation (RT output)
// --------------------------------------------------------------------------

void RTRenderer::createStorageImage(VulkanContext& ctx) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {renderWidth, renderHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                                     &storageImage, &storageAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create RT storage image");
    }

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = storageImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(ctx.device, &viewInfo, nullptr, &storageImageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create RT storage image view");
    }

    // Transition to GENERAL layout for use as storage image
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = storageImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// BLAS creation — multi-geometry when draw ranges are available
// --------------------------------------------------------------------------

void RTRenderer::createBLAS(VulkanContext& ctx) {
    const auto& vertices = m_scene->getVertices();
    const auto& indices = m_scene->getIndices();

    sphereMesh = Scene::uploadMesh(ctx.allocator, ctx.device, ctx.commandPool,
                                   ctx.graphicsQueue, vertices, indices);

    // Create SoA storage buffers for closest_hit vertex interpolation
    {
        std::vector<glm::vec4> posVec4(vertices.size());
        std::vector<glm::vec4> norVec4(vertices.size());
        std::vector<glm::vec2> uvVec2(vertices.size());
        for (size_t i = 0; i < vertices.size(); i++) {
            posVec4[i] = glm::vec4(vertices[i].position, 1.0f);
            norVec4[i] = glm::vec4(vertices[i].normal, 0.0f);
            uvVec2[i] = vertices[i].uv;
        }

        VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        positionsSize = posVec4.size() * sizeof(glm::vec4);
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            posVec4.data(), positionsSize, ssboUsage,
                            positionsBuffer, positionsAllocation);

        normalsSize = norVec4.size() * sizeof(glm::vec4);
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            norVec4.data(), normalsSize, ssboUsage,
                            normalsBuffer, normalsAllocation);

        texCoordsSize = uvVec2.size() * sizeof(glm::vec2);
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            uvVec2.data(), texCoordsSize, ssboUsage,
                            texCoordsBuffer, texCoordsAllocation);

        indexStorageSize = indices.size() * sizeof(uint32_t);
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            indices.data(), indexStorageSize, ssboUsage,
                            indexStorageBuffer, indexStorageAllocation);

        std::cout << "[info] Created SoA storage buffers: positions=" << positionsSize
                  << "B, normals=" << normalsSize << "B, texCoords=" << texCoordsSize
                  << "B, indices=" << indexStorageSize << "B" << std::endl;
    }

    VkDeviceAddress vertexAddress = ctx.getBufferDeviceAddress(sphereMesh.vertexBuffer);
    VkDeviceAddress indexAddress = ctx.getBufferDeviceAddress(sphereMesh.indexBuffer);

    const auto& drawRanges = m_scene->getDrawRanges();

    // Decide whether to use multi-geometry BLAS
    // Multi-geometry: one geometry per draw range, enabling per-geometry hit group selection.
    // In bindless mode, gl_GeometryIndexEXT maps to material index.
    bool useMultiGeometry = (m_multiMaterial || m_bindlessMode) && drawRanges.size() > 1;

    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges;
    std::vector<uint32_t> primitiveCounts;

    if (useMultiGeometry) {
        // Multi-geometry BLAS: one geometry per draw range
        m_geometryCount = static_cast<uint32_t>(drawRanges.size());

        for (auto& range : drawRanges) {
            VkAccelerationStructureGeometryTrianglesDataKHR trianglesData = {};
            trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            trianglesData.vertexData.deviceAddress = vertexAddress;
            trianglesData.vertexStride = sizeof(Vertex);
            trianglesData.maxVertex = sphereMesh.vertexCount - 1;
            trianglesData.indexType = VK_INDEX_TYPE_UINT32;
            // Offset the index data to this draw range's start
            trianglesData.indexData.deviceAddress = indexAddress +
                static_cast<VkDeviceSize>(range.indexOffset) * sizeof(uint32_t);

            VkAccelerationStructureGeometryKHR geo = {};
            geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geo.geometry.triangles = trianglesData;
            geometries.push_back(geo);

            uint32_t primCount = range.indexCount / 3;
            primitiveCounts.push_back(primCount);

            VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
            rangeInfo.primitiveCount = primCount;
            rangeInfo.primitiveOffset = 0;  // offset already baked into indexData address
            rangeInfo.firstVertex = 0;
            rangeInfo.transformOffset = 0;
            buildRanges.push_back(rangeInfo);
        }

        std::cout << "[info] Multi-geometry BLAS: " << m_geometryCount
                  << " geometries from " << drawRanges.size() << " draw ranges" << std::endl;
    } else {
        // Single-geometry BLAS (original path)
        m_geometryCount = 1;

        VkAccelerationStructureGeometryTrianglesDataKHR trianglesData = {};
        trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        trianglesData.vertexData.deviceAddress = vertexAddress;
        trianglesData.vertexStride = sizeof(Vertex);
        trianglesData.maxVertex = sphereMesh.vertexCount - 1;
        trianglesData.indexType = VK_INDEX_TYPE_UINT32;
        trianglesData.indexData.deviceAddress = indexAddress;

        VkAccelerationStructureGeometryKHR geo = {};
        geo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geo.geometry.triangles = trianglesData;
        geometries.push_back(geo);

        uint32_t primCount = sphereMesh.indexCount / 3;
        primitiveCounts.push_back(primCount);

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
        rangeInfo.primitiveCount = primCount;
        rangeInfo.primitiveOffset = 0;
        rangeInfo.firstVertex = 0;
        rangeInfo.transformOffset = 0;
        buildRanges.push_back(rangeInfo);
    }

    // Query build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = static_cast<uint32_t>(geometries.size());
    buildInfo.pGeometries = geometries.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    ctx.pfnGetAccelerationStructureBuildSizesKHR(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        primitiveCounts.data(),
        &sizeInfo);

    // Create AS buffer
    VkBufferCreateInfo asBufInfo = {};
    asBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    asBufInfo.size = sizeInfo.accelerationStructureSize;
    asBufInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo asAllocInfo = {};
    asAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    asAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkResult result = vmaCreateBuffer(ctx.allocator, &asBufInfo, &asAllocInfo,
                                      &blasBuffer, &blasAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create BLAS buffer");
    }

    // Create scratch buffer
    VkBufferCreateInfo scratchBufInfo = {};
    scratchBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchBufInfo.size = sizeInfo.buildScratchSize;
    scratchBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo scratchAllocInfo = {};
    scratchAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    scratchAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    result = vmaCreateBuffer(ctx.allocator, &scratchBufInfo, &scratchAllocInfo,
                             &blasScratchBuffer, &blasScratchAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create BLAS scratch buffer");
    }

    // Create acceleration structure handle
    VkAccelerationStructureCreateInfoKHR asCreateInfo = {};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = blasBuffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    result = ctx.pfnCreateAccelerationStructureKHR(ctx.device, &asCreateInfo, nullptr, &blas);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create BLAS");
    }

    // Build BLAS
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = ctx.getBufferDeviceAddress(blasScratchBuffer);

    // Build range info pointer array (one pointer per geometry, but Vulkan expects
    // a pointer to the whole array, and each element corresponds to one geometry)
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = buildRanges.data();

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    ctx.pfnCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    ctx.endSingleTimeCommands(cmd);

    uint32_t totalPrims = 0;
    for (auto pc : primitiveCounts) totalPrims += pc;
    std::cout << "[info] BLAS built (" << totalPrims << " triangles, "
              << geometries.size() << " geometry/geometries)" << std::endl;
}

// --------------------------------------------------------------------------
// TLAS creation
// --------------------------------------------------------------------------

void RTRenderer::createTLAS(VulkanContext& ctx) {
    // Get BLAS device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blas;
    VkDeviceAddress blasAddress = ctx.pfnGetAccelerationStructureDeviceAddressKHR(ctx.device, &addressInfo);

    // Create instance data (identity transform, single instance)
    VkAccelerationStructureInstanceKHR instance = {};
    // Identity 3x4 transform matrix (row-major)
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    // SBT offset is 0 — geometry index selects hit record within the hit region
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blasAddress;

    // Upload instance data to GPU buffer
    VkBufferCreateInfo instanceBufInfo = {};
    instanceBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    instanceBufInfo.size = sizeof(VkAccelerationStructureInstanceKHR);
    instanceBufInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    VmaAllocationCreateInfo instanceAllocInfo = {};
    instanceAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VkResult result = vmaCreateBuffer(ctx.allocator, &instanceBufInfo, &instanceAllocInfo,
                                      &instanceBuffer, &instanceAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create instance buffer");
    }

    void* mapped;
    vmaMapMemory(ctx.allocator, instanceAllocation, &mapped);
    memcpy(mapped, &instance, sizeof(VkAccelerationStructureInstanceKHR));
    vmaUnmapMemory(ctx.allocator, instanceAllocation);

    VkDeviceAddress instanceAddress = ctx.getBufferDeviceAddress(instanceBuffer);

    // Set up geometry for TLAS (instances)
    VkAccelerationStructureGeometryInstancesDataKHR instancesData = {};
    instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = instanceAddress;

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances = instancesData;

    uint32_t instanceCount = 1;

    // Query TLAS build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    ctx.pfnGetAccelerationStructureBuildSizesKHR(
        ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &instanceCount,
        &sizeInfo);

    // Create TLAS buffer
    VkBufferCreateInfo tlasBufInfo = {};
    tlasBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    tlasBufInfo.size = sizeInfo.accelerationStructureSize;
    tlasBufInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo tlasAllocInfo = {};
    tlasAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    tlasAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    result = vmaCreateBuffer(ctx.allocator, &tlasBufInfo, &tlasAllocInfo,
                             &tlasBuffer, &tlasAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS buffer");
    }

    // Create scratch buffer for TLAS
    VkBufferCreateInfo scratchBufInfo = {};
    scratchBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    scratchBufInfo.size = sizeInfo.buildScratchSize;
    scratchBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo scratchAllocInfo = {};
    scratchAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    scratchAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    result = vmaCreateBuffer(ctx.allocator, &scratchBufInfo, &scratchAllocInfo,
                             &tlasScratchBuffer, &tlasScratchAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS scratch buffer");
    }

    // Create TLAS handle
    VkAccelerationStructureCreateInfoKHR asCreateInfo = {};
    asCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    asCreateInfo.buffer = tlasBuffer;
    asCreateInfo.size = sizeInfo.accelerationStructureSize;
    asCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    result = ctx.pfnCreateAccelerationStructureKHR(ctx.device, &asCreateInfo, nullptr, &tlas);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS");
    }

    // Build TLAS
    buildInfo.dstAccelerationStructure = tlas;
    buildInfo.scratchData.deviceAddress = ctx.getBufferDeviceAddress(tlasScratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = instanceCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Memory barrier between BLAS and TLAS builds
    VkMemoryBarrier memBarrier = {};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    ctx.pfnCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    ctx.endSingleTimeCommands(cmd);

    std::cout << "[info] TLAS built (1 instance)" << std::endl;
}

// --------------------------------------------------------------------------
// RT pipeline creation — multi-hit-group when multiple permutations exist
// --------------------------------------------------------------------------

void RTRenderer::createRTPipeline(VulkanContext& ctx) {
    // Build shader stages and groups.
    // Single-material: stages=[rgen, rmiss, rchit], groups=[rgen, miss, hit]
    // Multi-material:  stages=[rgen, rmiss, rchit_0, rchit_1, ...],
    //                  groups=[rgen, miss, hit_0, hit_1, ...]

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    // Stage 0: raygen
    {
        VkPipelineShaderStageCreateInfo s = {};
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        s.module = rgenModule;
        s.pName = "main";
        stages.push_back(s);
    }
    // Stage 1: miss
    {
        VkPipelineShaderStageCreateInfo s = {};
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        s.module = rmissModule;
        s.pName = "main";
        stages.push_back(s);
    }

    // Group 0: raygen (GENERAL)
    {
        VkRayTracingShaderGroupCreateInfoKHR g = {};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader = 0;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }
    // Group 1: miss (GENERAL)
    {
        VkRayTracingShaderGroupCreateInfoKHR g = {};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        g.generalShader = 1;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }

    if (m_multiMaterial && !m_hitPermutations.empty()) {
        // Multi-material: one rchit stage + hit group per permutation
        for (size_t pi = 0; pi < m_hitPermutations.size(); pi++) {
            uint32_t stageIndex = static_cast<uint32_t>(stages.size());

            VkPipelineShaderStageCreateInfo s = {};
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            s.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            s.module = m_hitPermutations[pi].rchitModule;
            s.pName = "main";
            stages.push_back(s);

            VkRayTracingShaderGroupCreateInfoKHR g = {};
            g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            g.generalShader = VK_SHADER_UNUSED_KHR;
            g.closestHitShader = stageIndex;
            g.anyHitShader = VK_SHADER_UNUSED_KHR;
            g.intersectionShader = VK_SHADER_UNUSED_KHR;
            groups.push_back(g);

            // Record the group index (groups 0,1 are rgen,rmiss; hit groups start at 2)
            m_hitPermutations[pi].hitGroupIndex = static_cast<uint32_t>(groups.size()) - 1;
        }
    } else {
        // Single-material: one rchit stage + one hit group
        uint32_t stageIndex = static_cast<uint32_t>(stages.size());

        VkPipelineShaderStageCreateInfo s = {};
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        s.module = rchitModule;
        s.pName = "main";
        stages.push_back(s);

        VkRayTracingShaderGroupCreateInfoKHR g = {};
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = stageIndex;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(g);
    }

    // Create descriptor set layouts from superset reflection data
    reflectedSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stageReflections);

    // Pipeline layout from reflection
    pipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, reflectedSetLayouts, stageReflections);

    // Create RT pipeline
    VkRayTracingPipelineCreateInfoKHR rtPipeInfo = {};
    rtPipeInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtPipeInfo.stageCount = static_cast<uint32_t>(stages.size());
    rtPipeInfo.pStages = stages.data();
    rtPipeInfo.groupCount = static_cast<uint32_t>(groups.size());
    rtPipeInfo.pGroups = groups.data();
    rtPipeInfo.maxPipelineRayRecursionDepth = 1;
    rtPipeInfo.layout = pipelineLayout;

    VkResult result = ctx.pfnCreateRayTracingPipelinesKHR(
        ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
        1, &rtPipeInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline");
    }

    std::cout << "[info] RT pipeline created (" << stages.size() << " stages, "
              << groups.size() << " groups"
              << (m_multiMaterial ? ", multi-material" : "") << ")" << std::endl;
}

// --------------------------------------------------------------------------
// SBT creation — multi-hit-record when multiple geometries exist
// --------------------------------------------------------------------------

void RTRenderer::createSBT(VulkanContext& ctx) {
    uint32_t handleSize = ctx.rtPipelineProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = ctx.rtPipelineProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = ctx.rtPipelineProperties.shaderGroupBaseAlignment;

    uint32_t handleSizeAligned = static_cast<uint32_t>(alignUp(handleSize, handleAlignment));

    // Total shader groups in the pipeline
    // Bindless: one hit group (uber-shader), same as single-material
    uint32_t hitGroupCount = m_multiMaterial ? static_cast<uint32_t>(m_hitPermutations.size()) : 1;
    uint32_t totalGroupCount = 2 + hitGroupCount;  // rgen + miss + N hit groups

    // Get all shader group handles from the pipeline
    std::vector<uint8_t> handles(totalGroupCount * handleSize);
    VkResult result = ctx.pfnGetRayTracingShaderGroupHandlesKHR(
        ctx.device, pipeline, 0, totalGroupCount,
        handles.size(), handles.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get ray tracing shader group handles");
    }

    // Compute region sizes
    VkDeviceSize raygenRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize missRegionSize = alignUp(handleSizeAligned, baseAlignment);

    // Hit region: one record per BLAS geometry (not per hit group!)
    // Each geometry maps to a specific hit group handle.
    // Vulkan selects: instanceSBTOffset + geometryIndex * sbt.stride
    // In bindless mode: N geometries all map to the same single hit group.
    uint32_t hitRecordCount = (m_multiMaterial || m_bindlessMode) ? m_geometryCount : 1;
    VkDeviceSize hitRegionSize = alignUp(
        static_cast<VkDeviceSize>(hitRecordCount) * handleSizeAligned, baseAlignment);
    VkDeviceSize totalSbtSize = raygenRegionSize + missRegionSize + hitRegionSize;

    // Create SBT buffer
    VkBufferCreateInfo sbtBufInfo = {};
    sbtBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbtBufInfo.size = totalSbtSize;
    sbtBufInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo sbtAllocInfo = {};
    sbtAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    sbtAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    result = vmaCreateBuffer(ctx.allocator, &sbtBufInfo, &sbtAllocInfo,
                             &sbtBuffer, &sbtAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create SBT buffer");
    }

    // Map and write SBT data
    void* mapped;
    vmaMapMemory(ctx.allocator, sbtAllocation, &mapped);
    uint8_t* sbtData = static_cast<uint8_t*>(mapped);
    memset(sbtData, 0, totalSbtSize);

    // Copy raygen handle (group 0)
    memcpy(sbtData, handles.data() + 0 * handleSize, handleSize);

    // Copy miss handle (group 1)
    memcpy(sbtData + raygenRegionSize, handles.data() + 1 * handleSize, handleSize);

    // Copy hit records
    uint8_t* hitBase = sbtData + raygenRegionSize + missRegionSize;

    if (m_bindlessMode && m_geometryCount > 1) {
        // Bindless: all geometries use the same hit group (group 2).
        // Write N identical hit records so geometryIndex * stride lands on valid data.
        for (uint32_t gi = 0; gi < m_geometryCount; gi++) {
            memcpy(hitBase + gi * handleSizeAligned,
                   handles.data() + 2 * handleSize,
                   handleSize);
        }
    } else if (m_multiMaterial && !m_geometryToHitGroup.empty()) {
        // Multi-material: write one hit record per geometry, each pointing to
        // the correct permutation's hit group handle
        for (uint32_t gi = 0; gi < m_geometryCount; gi++) {
            // m_geometryToHitGroup[gi] is the group index (2-based)
            uint32_t groupIdx = m_geometryToHitGroup[gi];
            memcpy(hitBase + gi * handleSizeAligned,
                   handles.data() + groupIdx * handleSize,
                   handleSize);
        }
    } else {
        // Single-material: one hit record for group 2
        memcpy(hitBase, handles.data() + 2 * handleSize, handleSize);
    }

    vmaUnmapMemory(ctx.allocator, sbtAllocation);

    // Build strided device address regions
    VkDeviceAddress sbtBaseAddress = ctx.getBufferDeviceAddress(sbtBuffer);

    raygenSBT.deviceAddress = sbtBaseAddress;
    raygenSBT.stride = raygenRegionSize;
    raygenSBT.size = raygenRegionSize;

    missSBT.deviceAddress = sbtBaseAddress + raygenRegionSize;
    missSBT.stride = handleSizeAligned;
    missSBT.size = missRegionSize;

    hitSBT.deviceAddress = sbtBaseAddress + raygenRegionSize + missRegionSize;
    hitSBT.stride = handleSizeAligned;
    hitSBT.size = hitRegionSize;

    callableSBT = {};

    std::cout << "[info] SBT created (handleSize=" << handleSize
              << ", aligned=" << handleSizeAligned
              << ", baseAlign=" << baseAlignment
              << ", hitRecords=" << hitRecordCount << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Reflection-driven descriptor set creation and update
// --------------------------------------------------------------------------

void RTRenderer::createDescriptorSet(VulkanContext& ctx) {
    // Create camera UBO (128 bytes: inverse_view + inverse_projection)
    VkBufferCreateInfo camBufInfo = {};
    camBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    camBufInfo.size = 128; // 2 x mat4
    camBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo camAllocInfo = {};
    camAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VkResult result = vmaCreateBuffer(ctx.allocator, &camBufInfo, &camAllocInfo,
                                      &cameraBuffer, &cameraAllocation, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create camera UBO");
    }

    // Update camera data
    updateCameraUBO(ctx);

    // Create Material uniform buffer(s)
    // In bindless mode, material data is in the materials SSBO, not individual UBOs.
    // We still create a dummy single-material UBO for bindings that reference "Material"
    // (e.g. if the shader has both bindless and a legacy Material UBO -- though typically
    // a bindless shader won't have a Material UBO).
    if (!m_bindlessMode && m_multiMaterial && m_scene && m_scene->hasGltfScene()) {
        // Multi-material: create one UBO per material
        const auto& materials = m_scene->getGltfScene().materials;
        size_t matCount = materials.size();
        m_perMaterialBuffers.resize(matCount);
        m_perMaterialAllocations.resize(matCount);

        for (size_t mi = 0; mi < matCount; mi++) {
            MaterialUBOData materialData = materialDataFromGltf(materials[mi]);

            VkBufferCreateInfo matBufInfo = {};
            matBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            matBufInfo.size = sizeof(MaterialUBOData);
            matBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            VmaAllocationCreateInfo matAllocInfo = {};
            matAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

            vmaCreateBuffer(ctx.allocator, &matBufInfo, &matAllocInfo,
                            &m_perMaterialBuffers[mi], &m_perMaterialAllocations[mi], nullptr);

            void* matMapped;
            vmaMapMemory(ctx.allocator, m_perMaterialAllocations[mi], &matMapped);
            memcpy(matMapped, &materialData, sizeof(MaterialUBOData));
            vmaUnmapMemory(ctx.allocator, m_perMaterialAllocations[mi]);
        }

        // Also create the single m_materialBuffer pointing to material 0 for
        // backward compatibility with descriptor writes that reference "Material"
        m_materialBuffer = m_perMaterialBuffers[0];
        m_materialAllocation = VK_NULL_HANDLE; // owned by per-material array

        std::cout << "[info] Created " << matCount << " per-material UBOs" << std::endl;
    } else if (!m_bindlessMode) {
        // Single-material: one Material UBO (original path)
        MaterialUBOData materialData{};
        if (m_scene && m_scene->hasGltfScene()) {
            materialData = materialDataFromGltf(m_scene->getGltfScene().materials[0]);
        }

        VkBufferCreateInfo matBufInfo = {};
        matBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matBufInfo.size = sizeof(MaterialUBOData);
        matBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo matAllocInfo = {};
        matAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        result = vmaCreateBuffer(ctx.allocator, &matBufInfo, &matAllocInfo,
                                 &m_materialBuffer, &m_materialAllocation, nullptr);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create material UBO");
        }

        void* matMapped;
        vmaMapMemory(ctx.allocator, m_materialAllocation, &matMapped);
        memcpy(matMapped, &materialData, sizeof(MaterialUBOData));
        vmaUnmapMemory(ctx.allocator, m_materialAllocation);
    }

    // Get merged bindings from superset reflection data for pool sizing
    auto mergedBindings = getMergedBindings(stageReflections);

    // Identify the "material set" — the set that contains the Material UBO and textures
    int materialSetIdx = -1;
    for (auto& b : mergedBindings) {
        if (b.name == "Material") { materialSetIdx = b.set; break; }
    }

    // Count pool sizes from reflection data
    // In multi-material mode, we need N copies of the material set
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    uint32_t maxSets = 0;

    const auto& drawRanges = m_scene->getDrawRanges();
    size_t matCount = (m_scene && m_scene->hasGltfScene()) ?
        m_scene->getGltfScene().materials.size() : 1;
    bool multiDesc = !m_bindlessMode && m_multiMaterial && materialSetIdx >= 0 && matCount > 1;

    // Determine actual texture count for bindless
    uint32_t bindlessTexCount = m_bindlessMode ? m_bindlessTextures.textureCount : 0;
    // Must have at least 1 for valid descriptor writes
    if (m_bindlessMode && bindlessTexCount == 0) bindlessTexCount = 1;

    for (auto& b : mergedBindings) {
        VkDescriptorType vkType = bindingTypeToVkDescriptorType(b.type);

        if (b.type == "bindless_combined_image_sampler_array") {
            // Pool needs enough combined image samplers for the actual texture count
            typeCounts[VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] += bindlessTexCount;
        } else if (multiDesc && b.set == materialSetIdx) {
            // Need one descriptor per material for bindings in the material set
            typeCounts[vkType] += static_cast<uint32_t>(matCount);
        } else {
            typeCounts[vkType]++;
        }
    }

    // Count descriptor sets needed
    for (auto& [setIdx, layout] : reflectedSetLayouts) {
        if (multiDesc && setIdx == materialSetIdx) {
            maxSets += static_cast<uint32_t>(matCount);
        } else {
            maxSets++;
        }
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    // Bindless requires UPDATE_AFTER_BIND on the pool
    if (m_bindlessMode) {
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    }

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create RT descriptor pool");
    }

    // Allocate descriptor sets
    for (auto& [setIdx, layout] : reflectedSetLayouts) {
        if (multiDesc && setIdx == materialSetIdx) continue; // allocated per-material below

        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layout;

        // For the bindless texture set, use variable descriptor count
        VkDescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo = {};
        uint32_t variableDescCount = bindlessTexCount;

        if (m_bindlessMode && setIdx == m_bindlessTextureSet) {
            variableCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
            variableCountInfo.descriptorSetCount = 1;
            variableCountInfo.pDescriptorCounts = &variableDescCount;
            allocSetInfo.pNext = &variableCountInfo;
        }

        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate RT descriptor set for set " + std::to_string(setIdx));
        }
        reflectedDescSets[setIdx] = set;
    }

    // Write descriptor sets (non-material bindings, plus bindless-specific bindings)
    {
        struct DescWriteInfo {
            VkDescriptorBufferInfo bufferInfo;
            VkDescriptorImageInfo imageInfo;
            VkWriteDescriptorSetAccelerationStructureKHR asInfo;
        };
        std::vector<DescWriteInfo> writeInfos(mergedBindings.size());
        std::vector<VkWriteDescriptorSet> writes;

        // For bindless texture array, we need a separate vector of image infos
        std::vector<VkDescriptorImageInfo> bindlessImageInfos;

        for (size_t i = 0; i < mergedBindings.size(); i++) {
            auto& b = mergedBindings[i];
            if (multiDesc && b.set == materialSetIdx) continue; // handled per-material

            // Skip bindless texture array — handled separately below
            if (b.type == "bindless_combined_image_sampler_array") continue;

            auto setIt = reflectedDescSets.find(b.set);
            if (setIt == reflectedDescSets.end()) continue;

            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = setIt->second;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;

            if (b.type == "uniform_buffer") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                if (b.name == "Material" && !m_bindlessMode) {
                    // Single-material path — write material 0
                    writeInfos[i].bufferInfo = {m_materialBuffer, 0,
                        static_cast<VkDeviceSize>(b.size > 0 ? b.size : sizeof(MaterialUBOData))};
                } else if (b.name == "Material" && m_bindlessMode) {
                    // Bindless mode: skip Material UBO (material data is in SSBO)
                    continue;
                } else {
                    writeInfos[i].bufferInfo = {cameraBuffer, 0,
                        static_cast<VkDeviceSize>(b.size > 0 ? b.size : 128)};
                }
                w.pBufferInfo = &writeInfos[i].bufferInfo;
            } else if (b.type == "acceleration_structure") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                writeInfos[i].asInfo = {};
                writeInfos[i].asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                writeInfos[i].asInfo.accelerationStructureCount = 1;
                writeInfos[i].asInfo.pAccelerationStructures = &tlas;
                w.pNext = &writeInfos[i].asInfo;
            } else if (b.type == "storage_image") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = storageImageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                w.pImageInfo = &writeInfos[i].imageInfo;
            } else if (b.type == "storage_buffer") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                VkBuffer buf = VK_NULL_HANDLE;
                VkDeviceSize bufSize = 0;
                if (b.name == "positions") { buf = positionsBuffer; bufSize = positionsSize; }
                else if (b.name == "normals") { buf = normalsBuffer; bufSize = normalsSize; }
                else if (b.name == "tex_coords") { buf = texCoordsBuffer; bufSize = texCoordsSize; }
                else if (b.name == "indices") { buf = indexStorageBuffer; bufSize = indexStorageSize; }
                else if (b.name == "materials" && m_bindlessMode && m_bindlessMaterials.buffer != VK_NULL_HANDLE) {
                    buf = m_bindlessMaterials.buffer;
                    bufSize = static_cast<VkDeviceSize>(m_bindlessMaterials.materialCount) * sizeof(BindlessMaterialData);
                }
                if (buf == VK_NULL_HANDLE) {
                    std::cout << "[warn] Unknown storage buffer name: " << b.name << std::endl;
                    continue;
                }
                writeInfos[i].bufferInfo = {buf, 0, bufSize};
                w.pBufferInfo = &writeInfos[i].bufferInfo;
            } else if (b.type == "sampler") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                auto& tex = m_scene->getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.sampler = tex.sampler;
                w.pImageInfo = &writeInfos[i].imageInfo;
            } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                auto& tex = m_scene->getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = tex.imageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                w.pImageInfo = &writeInfos[i].imageInfo;
            } else {
                continue;
            }

            writes.push_back(w);
        }

        // Bindless texture array write
        if (m_bindlessMode && m_bindlessTextureSet >= 0 && bindlessTexCount > 0) {
            auto setIt = reflectedDescSets.find(m_bindlessTextureSet);
            if (setIt != reflectedDescSets.end()) {
                bindlessImageInfos.resize(bindlessTexCount);
                for (uint32_t ti = 0; ti < bindlessTexCount; ti++) {
                    if (ti < m_bindlessTextures.textureCount) {
                        bindlessImageInfos[ti].sampler = m_bindlessTextures.samplers[ti];
                        bindlessImageInfos[ti].imageView = m_bindlessTextures.imageViews[ti];
                        bindlessImageInfos[ti].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    } else {
                        // Pad with a valid fallback (first texture or default)
                        if (m_bindlessTextures.textureCount > 0) {
                            bindlessImageInfos[ti].sampler = m_bindlessTextures.samplers[0];
                            bindlessImageInfos[ti].imageView = m_bindlessTextures.imageViews[0];
                        } else {
                            auto& fallback = m_scene->getTextureForBinding("base_color_tex");
                            bindlessImageInfos[ti].sampler = fallback.sampler;
                            bindlessImageInfos[ti].imageView = fallback.imageView;
                        }
                        bindlessImageInfos[ti].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    }
                }

                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = setIt->second;
                w.dstBinding = static_cast<uint32_t>(m_bindlessTextureBinding);
                w.descriptorCount = bindlessTexCount;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = bindlessImageInfos.data();
                writes.push_back(w);
            }
        }

        if (!writes.empty()) {
            vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                                   writes.data(), 0, nullptr);
        }
    }

    // Multi-material: allocate and write per-material descriptor sets for the material set
    if (multiDesc) {
        auto& matSetLayout = reflectedSetLayouts[materialSetIdx];
        auto& perMatTextures = m_scene->getPerMaterialTextures();
        m_perMaterialDescSets.resize(matCount);

        for (size_t mi = 0; mi < matCount; mi++) {
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = descriptorPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &matSetLayout;

            if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_perMaterialDescSets[mi]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate per-material RT descriptor set");
            }

            // Write this material's descriptors
            struct DescWriteInfo {
                VkDescriptorBufferInfo bufferInfo;
                VkDescriptorImageInfo imageInfo;
                VkWriteDescriptorSetAccelerationStructureKHR asInfo;
            };
            std::vector<DescWriteInfo> writeInfos;
            writeInfos.reserve(mergedBindings.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(mergedBindings.size());

            for (auto& b : mergedBindings) {
                if (b.set != materialSetIdx) continue;

                DescWriteInfo info = {};
                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = m_perMaterialDescSets[mi];
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;

                if (b.type == "uniform_buffer") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    if (b.name == "Material") {
                        info.bufferInfo = {m_perMaterialBuffers[mi], 0, sizeof(MaterialUBOData)};
                    } else {
                        // Camera or other UBO in the material set
                        info.bufferInfo = {cameraBuffer, 0,
                            static_cast<VkDeviceSize>(b.size > 0 ? b.size : 128)};
                    }
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.type == "sampler") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {}; info.imageInfo.sampler = iblIt->second.sampler;
                    } else {
                        auto& tex = m_scene->getTextureForBinding(b.name);
                        info.imageInfo = {}; info.imageInfo.sampler = tex.sampler;
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {};
                        info.imageInfo.imageView = iblIt->second.imageView;
                        info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    } else {
                        GPUTexture* texPtr = nullptr;
                        if (mi < perMatTextures.size()) {
                            auto it = perMatTextures[mi].find(b.name);
                            if (it != perMatTextures[mi].end())
                                texPtr = const_cast<GPUTexture*>(&it->second);
                        }
                        if (texPtr && texPtr->imageView != VK_NULL_HANDLE) {
                            info.imageInfo = {};
                            info.imageInfo.imageView = texPtr->imageView;
                            info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        } else {
                            auto& tex = m_scene->getTextureForBinding(b.name);
                            info.imageInfo = {};
                            info.imageInfo.imageView = tex.imageView;
                            info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        }
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else if (b.type == "storage_buffer") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    VkBuffer buf = VK_NULL_HANDLE;
                    VkDeviceSize bufSize = 0;
                    if (b.name == "positions") { buf = positionsBuffer; bufSize = positionsSize; }
                    else if (b.name == "normals") { buf = normalsBuffer; bufSize = normalsSize; }
                    else if (b.name == "tex_coords") { buf = texCoordsBuffer; bufSize = texCoordsSize; }
                    else if (b.name == "indices") { buf = indexStorageBuffer; bufSize = indexStorageSize; }
                    if (buf == VK_NULL_HANDLE) {
                        std::cout << "[warn] Unknown storage buffer name in material set: " << b.name << std::endl;
                        continue;
                    }
                    info.bufferInfo = {buf, 0, bufSize};
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.type == "acceleration_structure") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                    info.asInfo = {};
                    info.asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
                    info.asInfo.accelerationStructureCount = 1;
                    info.asInfo.pAccelerationStructures = &tlas;
                    writeInfos.push_back(info);
                    w.pNext = &writeInfos.back().asInfo;
                } else if (b.type == "storage_image") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    info.imageInfo = {};
                    info.imageInfo.imageView = storageImageView;
                    info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else {
                    continue;
                }
                writes.push_back(w);
            }

            if (!writes.empty()) {
                vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                                       writes.data(), 0, nullptr);
            }
        }

        // Set the "active" material desc set to material 0 for the reflectedDescSets map
        // (used in render() for initial binding)
        reflectedDescSets[materialSetIdx] = m_perMaterialDescSets[0];

        std::cout << "[info] RT multi-material descriptor sets created: "
                  << matCount << " material set(s)" << std::endl;
    }

    std::cout << "[info] RT descriptor set created (reflection-driven: "
              << reflectedSetLayouts.size() << " set(s), "
              << mergedBindings.size() << " binding(s)"
              << (m_bindlessMode ? ", bindless" : (multiDesc ? ", multi-material" : ""))
              << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Camera UBO update
// --------------------------------------------------------------------------

void RTRenderer::updateCameraUBO(VulkanContext& ctx) {
    float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);
    Camera::RTCameraData camData;
    if (m_scene && m_scene->hasSceneBounds()) {
        camData = Camera::getRTCameraData(aspect, m_scene->getAutoEye(),
                                           m_scene->getAutoTarget(),
                                           m_scene->getAutoUp(),
                                           m_scene->getAutoFar());
    } else {
        camData = Camera::getRTCameraData(aspect);
    }

    void* mapped;
    vmaMapMemory(ctx.allocator, cameraAllocation, &mapped);
    uint8_t* dst = static_cast<uint8_t*>(mapped);
    memcpy(dst, &camData.inverseView, sizeof(glm::mat4));
    memcpy(dst + sizeof(glm::mat4), &camData.inverseProjection, sizeof(glm::mat4));
    vmaUnmapMemory(ctx.allocator, cameraAllocation);
}

void RTRenderer::updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                               glm::vec3 up, float fovY, float aspect,
                               float nearPlane, float farPlane) {
    // Build inverse view/projection from orbit camera parameters
    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 proj = glm::perspective(fovY, aspect, nearPlane, farPlane);
    proj[1][1] *= -1.0f; // Vulkan Y flip

    Camera::RTCameraData camData;
    camData.inverseView = glm::inverse(view);
    camData.inverseProjection = glm::inverse(proj);

    void* mapped;
    vmaMapMemory(ctx.allocator, cameraAllocation, &mapped);
    uint8_t* dst = static_cast<uint8_t*>(mapped);
    memcpy(dst, &camData.inverseView, sizeof(glm::mat4));
    memcpy(dst + sizeof(glm::mat4), &camData.inverseProjection, sizeof(glm::mat4));
    vmaUnmapMemory(ctx.allocator, cameraAllocation);
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void RTRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                  VkImage swapImage, VkExtent2D extent) {
    // Transition swapchain image to TRANSFER_DST
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit RT output to swapchain
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {
        static_cast<int32_t>(renderWidth),
        static_cast<int32_t>(renderHeight), 1
    };
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {
        static_cast<int32_t>(extent.width),
        static_cast<int32_t>(extent.height), 1
    };

    vkCmdBlitImage(cmd,
        storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_LINEAR);

    // Transition swapchain image to PRESENT
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

void RTRenderer::init(VulkanContext& ctx,
                      const std::string& rgenSpvPath,
                      const std::string& rmissSpvPath,
                      const std::string& rchitSpvPath,
                      uint32_t width, uint32_t height,
                      SceneManager& scene) {
    if (!ctx.supportsRT()) {
        throw std::runtime_error("Ray tracing is not supported on this device");
    }

    renderWidth = width;
    renderHeight = height;
    m_scene = &scene;

    std::cout << "[info] Initializing RT renderer (" << width << "x" << height << ")" << std::endl;

    // Derive pipeline base from rgen path (strip .rgen.spv suffix)
    m_pipelineBase = rgenSpvPath;
    auto pos = m_pipelineBase.find(".rgen.spv");
    if (pos != std::string::npos) {
        m_pipelineBase = m_pipelineBase.substr(0, pos);
    }

    namespace fs = std::filesystem;

    // Check for manifest to determine multi-material mode
    ShaderManifest manifest = tryLoadManifest(m_pipelineBase);
    bool hasMultipleMaterials = m_scene->hasGltfScene() &&
        m_scene->getGltfScene().materials.size() > 1;
    bool hasManifest = !manifest.permutations.empty();

    // --- Detect bindless mode ---
    // Check if the rchit reflection JSON has bindlessEnabled AND the device supports it.
    // Bindless uses a single uber-shader rchit with gl_GeometryIndexEXT for material lookup.
    {
        std::string rchitJsonPath = m_pipelineBase + ".rchit.json";
        if (fs::exists(rchitJsonPath)) {
            ReflectionData rchitRefl = parseReflectionJson(rchitJsonPath);
            if (rchitRefl.bindlessEnabled && ctx.supportsBindless()) {
                m_bindlessMode = true;
                // Extract bindless set/binding info from the reflection data
                for (auto& [setIdx, bindings] : rchitRefl.descriptor_sets) {
                    for (auto& b : bindings) {
                        if (b.type == "bindless_combined_image_sampler_array") {
                            m_bindlessTextureSet = setIdx;
                            m_bindlessTextureBinding = b.binding;
                            m_bindlessMaxCount = b.max_count > 0 ?
                                static_cast<uint32_t>(b.max_count) : 1024;
                        }
                        if (b.type == "storage_buffer" && b.name == "materials") {
                            m_bindlessMaterialsSet = setIdx;
                            m_bindlessMaterialsBinding = b.binding;
                        }
                    }
                }
                std::cout << "[info] Bindless RT mode detected (texture set=" << m_bindlessTextureSet
                          << " binding=" << m_bindlessTextureBinding
                          << " max=" << m_bindlessMaxCount
                          << ", materials set=" << m_bindlessMaterialsSet
                          << " binding=" << m_bindlessMaterialsBinding << ")" << std::endl;
            }
        }
    }

    if (m_bindlessMode) {
        // --- Bindless RT mode ---
        // Single uber-shader rchit, multi-geometry BLAS for per-material gl_GeometryIndexEXT.
        // No permutation suffix needed for bindless shaders.
        m_multiMaterial = false; // bindless replaces multi-material permutation mode

        // Load reflection JSON for all RT stages (using base path, no permutation)
        std::string rgenJsonPath = m_pipelineBase + ".rgen.json";
        std::string rmissJsonPath = m_pipelineBase + ".rmiss.json";
        std::string rchitJsonPath = m_pipelineBase + ".rchit.json";

        if (fs::exists(rgenJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rgenJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rgenJsonPath));
        }
        if (fs::exists(rchitJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rchitJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rchitJsonPath));
        }
        if (fs::exists(rmissJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rmissJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rmissJsonPath));
        }

        if (stageReflections.empty()) {
            throw std::runtime_error("No reflection JSON files found for RT pipeline: " + m_pipelineBase);
        }

        // Build bindless resources from scene
        m_bindlessTextures = m_scene->buildBindlessTextureArray(ctx);
        m_bindlessMaterials = m_scene->buildMaterialsSSBOByGeometry(ctx, m_bindlessTextures);

        // Enable multi-geometry BLAS: one geometry per draw range so that
        // gl_GeometryIndexEXT maps directly to the material index
        const auto& drawRanges = m_scene->getDrawRanges();
        if (drawRanges.size() > 1) {
            m_geometryCount = static_cast<uint32_t>(drawRanges.size());
        }

        std::cout << "[info] Bindless RT: " << m_bindlessTextures.textureCount << " textures, "
                  << m_bindlessMaterials.materialCount << " materials, "
                  << m_geometryCount << " geometries" << std::endl;

    } else if (false && hasManifest && hasMultipleMaterials) {
        // NOTE: Permutation-based multi-material RT is disabled.
        // Vulkan RT binds descriptor sets ONCE before traceRays, so per-geometry
        // material switching requires bindless textures (see bindless mode above).
        m_multiMaterial = true;

        auto groups = m_scene->groupMaterialsByFeatures();
        size_t totalMaterials = m_scene->getGltfScene().materials.size();
        m_geometryToPermutation.resize(0); // built from draw ranges below

        int permIdx = 0;
        for (auto& [suffix, materialIndices] : groups) {
            std::string resolvedSuffix = suffix;
            std::string permBase = m_pipelineBase + resolvedSuffix;
            std::string rchitPath = permBase + ".rchit.spv";

            if (!fs::exists(rchitPath)) {
                std::cerr << "[warn] Missing rchit shader for permutation '"
                          << resolvedSuffix << "', falling back to base" << std::endl;
                permBase = m_pipelineBase;
                rchitPath = permBase + ".rchit.spv";
                resolvedSuffix = "";
            }

            RTHitPermutation perm;
            perm.suffix = resolvedSuffix;
            perm.basePath = permBase;
            perm.materialIndices = materialIndices;

            // Load rchit shader module
            auto rchitCode = SpvLoader::loadSPIRV(rchitPath);
            perm.rchitModule = SpvLoader::createShaderModule(ctx.device, rchitCode);

            // Load rchit reflection JSON
            std::string rchitJsonPath = permBase + ".rchit.json";
            if (fs::exists(rchitJsonPath)) {
                perm.rchitRefl = parseReflectionJson(rchitJsonPath);
            }

            m_hitPermutations.push_back(std::move(perm));
            permIdx++;

            std::cout << "[info] Loaded RT permutation '" << resolvedSuffix << "' from " << permBase
                      << " (" << materialIndices.size() << " material(s))" << std::endl;
        }

        // Build materialToPermutation mapping
        std::vector<int> materialToPermutation(totalMaterials, 0);
        for (size_t pi = 0; pi < m_hitPermutations.size(); pi++) {
            for (int mi : m_hitPermutations[pi].materialIndices) {
                if (mi < static_cast<int>(totalMaterials)) {
                    materialToPermutation[mi] = static_cast<int>(pi);
                }
            }
        }

        // Build geometry-to-hit-group mapping from draw ranges
        const auto& drawRanges = m_scene->getDrawRanges();
        m_geometryToPermutation.resize(drawRanges.size());
        m_geometryToHitGroup.resize(drawRanges.size());

        for (size_t gi = 0; gi < drawRanges.size(); gi++) {
            int matIdx = drawRanges[gi].materialIndex;
            int pi = (matIdx < static_cast<int>(totalMaterials))
                ? materialToPermutation[matIdx] : 0;
            m_geometryToPermutation[gi] = pi;
            // Hit group index: groups 0,1 are rgen,rmiss; hit groups start at 2
            m_geometryToHitGroup[gi] = 2 + static_cast<uint32_t>(pi);
        }

        // Build superset reflection from all permutation rchit shaders
        // This ensures the descriptor layout covers bindings from ALL permutations
        stageReflections.clear();

        // Add raygen reflection
        std::string rgenJsonPath = m_pipelineBase + ".rgen.json";
        if (fs::exists(rgenJsonPath)) {
            stageReflections.push_back(parseReflectionJson(rgenJsonPath));
        }

        // Add all permutation rchit reflections (superset merging)
        for (auto& perm : m_hitPermutations) {
            std::string rchitJsonPath = perm.basePath + ".rchit.json";
            if (fs::exists(rchitJsonPath)) {
                stageReflections.push_back(parseReflectionJson(rchitJsonPath));
            }
        }

        // Add miss reflection
        std::string rmissJsonPath = m_pipelineBase + ".rmiss.json";
        if (fs::exists(rmissJsonPath)) {
            stageReflections.push_back(parseReflectionJson(rmissJsonPath));
        }

        if (stageReflections.empty()) {
            throw std::runtime_error("No reflection JSON files found for RT pipeline: " + m_pipelineBase);
        }

        std::cout << "[info] Multi-material RT mode: " << m_hitPermutations.size()
                  << " permutation(s), " << totalMaterials << " material(s), "
                  << drawRanges.size() << " draw range(s)" << std::endl;
    } else {
        // Single-material mode (no bindless, no multi-material)
        m_multiMaterial = false;

        // If manifest exists but only one material, resolve the best permutation
        std::string resolvedBase = m_pipelineBase;
        if (hasManifest && m_scene->hasGltfScene()) {
            auto sceneFeatures = m_scene->detectSceneFeatures();
            std::string suffix = findPermutationSuffix(manifest, sceneFeatures);
            if (!suffix.empty()) {
                std::string candidateBase = m_pipelineBase + suffix;
                if (fs::exists(candidateBase + ".rchit.spv")) {
                    resolvedBase = candidateBase;
                    std::cout << "[info] Resolved single-material RT permutation: " << suffix << std::endl;
                }
            }
        }

        // Load reflection JSON for all RT stages
        std::string rgenJsonPath = resolvedBase + ".rgen.json";
        // raygen always uses base (no permutation suffix for rgen)
        if (!fs::exists(rgenJsonPath)) rgenJsonPath = m_pipelineBase + ".rgen.json";
        std::string rmissJsonPath = resolvedBase + ".rmiss.json";
        if (!fs::exists(rmissJsonPath)) rmissJsonPath = m_pipelineBase + ".rmiss.json";
        std::string rchitJsonPath = resolvedBase + ".rchit.json";

        if (fs::exists(rgenJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rgenJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rgenJsonPath));
        }
        if (fs::exists(rchitJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rchitJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rchitJsonPath));
        }
        if (fs::exists(rmissJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << rmissJsonPath << std::endl;
            stageReflections.push_back(parseReflectionJson(rmissJsonPath));
        }

        if (stageReflections.empty()) {
            throw std::runtime_error("No reflection JSON files found for RT pipeline: " + m_pipelineBase);
        }
    }

    // Load shader modules (raygen and miss are always from the base pipeline)
    auto rgenCode = SpvLoader::loadSPIRV(rgenSpvPath);
    rgenModule = SpvLoader::createShaderModule(ctx.device, rgenCode);

    auto rmissCode = SpvLoader::loadSPIRV(rmissSpvPath);
    rmissModule = SpvLoader::createShaderModule(ctx.device, rmissCode);

    if (!m_multiMaterial) {
        // Single-material or bindless: load the rchit module.
        // In bindless mode, use base rchit (no permutation suffix).
        // In single-material mode, try resolving a permutation.
        std::string resolvedRchitPath = rchitSpvPath;
        if (!m_bindlessMode && hasManifest && m_scene->hasGltfScene()) {
            auto sceneFeatures = m_scene->detectSceneFeatures();
            std::string suffix = findPermutationSuffix(manifest, sceneFeatures);
            if (!suffix.empty()) {
                std::string candidatePath = m_pipelineBase + suffix + ".rchit.spv";
                if (fs::exists(candidatePath)) {
                    resolvedRchitPath = candidatePath;
                }
            }
        }
        auto rchitCode = SpvLoader::loadSPIRV(resolvedRchitPath);
        rchitModule = SpvLoader::createShaderModule(ctx.device, rchitCode);
    }
    // Multi-material: rchit modules are already loaded in m_hitPermutations

    // Create resources
    createStorageImage(ctx);
    createBLAS(ctx);
    createTLAS(ctx);
    createRTPipeline(ctx);
    createSBT(ctx);
    createDescriptorSet(ctx);

    // Set initial camera from scene auto-camera (used by headless path;
    // interactive path overrides via updateCamera() before render())
    updateCameraUBO(ctx);

    std::cout << "[info] RT renderer initialized successfully"
              << (m_bindlessMode ? " (bindless)" : (m_multiMaterial ? " (multi-material)" : ""))
              << std::endl;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RTRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Transition storage image to GENERAL for writing
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = storageImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Bind RT pipeline and reflection-driven descriptors
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);

    // Bind descriptor sets in order
    int maxSet = -1;
    for (auto& [idx, _] : reflectedDescSets) {
        maxSet = std::max(maxSet, idx);
    }
    for (int i = 0; i <= maxSet; i++) {
        auto it = reflectedDescSets.find(i);
        if (it != reflectedDescSets.end()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    pipelineLayout, static_cast<uint32_t>(i),
                                    1, &it->second, 0, nullptr);
        }
    }

    // Trace rays
    ctx.cmdBeginLabel(cmd, "RT Trace", 0.8f, 0.2f, 0.2f, 1.0f);
    ctx.pfnCmdTraceRaysKHR(cmd,
        &raygenSBT, &missSBT, &hitSBT, &callableSBT,
        renderWidth, renderHeight, 1);
    ctx.cmdEndLabel(cmd);

    // Transition storage image to TRANSFER_SRC_OPTIMAL for readback
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void RTRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    // Pipeline resources
    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);

    for (auto& [_, layout] : reflectedSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    reflectedSetLayouts.clear();
    reflectedDescSets.clear();
    m_perMaterialDescSets.clear();

    // Shader modules
    if (rgenModule) vkDestroyShaderModule(ctx.device, rgenModule, nullptr);
    if (rmissModule) vkDestroyShaderModule(ctx.device, rmissModule, nullptr);
    if (rchitModule) vkDestroyShaderModule(ctx.device, rchitModule, nullptr);

    // Multi-material permutation shader modules
    for (auto& perm : m_hitPermutations) {
        if (perm.rchitModule) vkDestroyShaderModule(ctx.device, perm.rchitModule, nullptr);
    }
    m_hitPermutations.clear();

    // Acceleration structures
    if (blas && ctx.pfnDestroyAccelerationStructureKHR) {
        ctx.pfnDestroyAccelerationStructureKHR(ctx.device, blas, nullptr);
    }
    if (tlas && ctx.pfnDestroyAccelerationStructureKHR) {
        ctx.pfnDestroyAccelerationStructureKHR(ctx.device, tlas, nullptr);
    }

    // Buffers
    if (blasBuffer) vmaDestroyBuffer(ctx.allocator, blasBuffer, blasAllocation);
    if (tlasBuffer) vmaDestroyBuffer(ctx.allocator, tlasBuffer, tlasAllocation);
    if (blasScratchBuffer) vmaDestroyBuffer(ctx.allocator, blasScratchBuffer, blasScratchAllocation);
    if (tlasScratchBuffer) vmaDestroyBuffer(ctx.allocator, tlasScratchBuffer, tlasScratchAllocation);
    if (instanceBuffer) vmaDestroyBuffer(ctx.allocator, instanceBuffer, instanceAllocation);
    if (sbtBuffer) vmaDestroyBuffer(ctx.allocator, sbtBuffer, sbtAllocation);
    if (cameraBuffer) vmaDestroyBuffer(ctx.allocator, cameraBuffer, cameraAllocation);

    // Bindless materials SSBO (owned by RT renderer, not scene manager)
    if (m_bindlessMaterials.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, m_bindlessMaterials.buffer, m_bindlessMaterials.allocation);
        m_bindlessMaterials.buffer = VK_NULL_HANDLE;
        m_bindlessMaterials.allocation = VK_NULL_HANDLE;
    }
    // Bindless texture array: image views and samplers are owned by SceneManager, not freed here
    m_bindlessTextures = {};

    // Per-material UBOs
    if (!m_perMaterialBuffers.empty()) {
        for (size_t i = 0; i < m_perMaterialBuffers.size(); i++) {
            if (m_perMaterialBuffers[i]) {
                vmaDestroyBuffer(ctx.allocator, m_perMaterialBuffers[i], m_perMaterialAllocations[i]);
            }
        }
        m_perMaterialBuffers.clear();
        m_perMaterialAllocations.clear();
        // m_materialBuffer was aliased, do not double-free
        m_materialBuffer = VK_NULL_HANDLE;
        m_materialAllocation = VK_NULL_HANDLE;
    } else {
        // Single-material path: m_materialBuffer owns its allocation
        if (m_materialBuffer) vmaDestroyBuffer(ctx.allocator, m_materialBuffer, m_materialAllocation);
    }

    // Storage image
    if (storageImageView) vkDestroyImageView(ctx.device, storageImageView, nullptr);
    if (storageImage) vmaDestroyImage(ctx.allocator, storageImage, storageAllocation);

    // RT-local mesh (BLAS geometry copy)
    Scene::destroyMesh(ctx.allocator, sphereMesh);

    // SoA storage buffers
    if (positionsBuffer) vmaDestroyBuffer(ctx.allocator, positionsBuffer, positionsAllocation);
    if (normalsBuffer) vmaDestroyBuffer(ctx.allocator, normalsBuffer, normalsAllocation);
    if (texCoordsBuffer) vmaDestroyBuffer(ctx.allocator, texCoordsBuffer, texCoordsAllocation);
    if (indexStorageBuffer) vmaDestroyBuffer(ctx.allocator, indexStorageBuffer, indexStorageAllocation);
}
