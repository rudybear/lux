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

// --------------------------------------------------------------------------
// Alignment helper
// --------------------------------------------------------------------------

static VkDeviceSize alignUp(VkDeviceSize size, VkDeviceSize alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
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
// BLAS creation from sphere mesh
// --------------------------------------------------------------------------

static bool isGltfFileRT(const std::string& source) {
    if (source.size() > 4 && source.substr(source.size()-4) == ".glb") return true;
    if (source.size() > 5 && source.substr(source.size()-5) == ".gltf") return true;
    return false;
}

void RTRenderer::createBLAS(VulkanContext& ctx) {
    // Load mesh from glTF or generate procedural sphere
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    if (isGltfFileRT(m_sceneSource)) {
        std::cout << "[info] Loading glTF for RT BLAS: " << m_sceneSource << std::endl;
        GltfScene gltfScene = loadGltf(m_sceneSource);

        if (!gltfScene.meshes.empty()) {
            auto& gmesh = gltfScene.meshes[0];
            vertices.resize(gmesh.vertices.size());
            for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                vertices[i].position = gmesh.vertices[i].position;
                vertices[i].normal = gmesh.vertices[i].normal;
                vertices[i].uv = gmesh.vertices[i].uv;
            }
            indices = gmesh.indices;
            std::cout << "[info] Loaded glTF mesh for BLAS: " << vertices.size()
                      << " vertices, " << indices.size() << " indices" << std::endl;
        } else {
            std::cout << "[warn] glTF has no meshes, falling back to sphere" << std::endl;
            Scene::generateSphere(32, 32, vertices, indices);
        }
    } else {
        Scene::generateSphere(32, 32, vertices, indices);
    }

    sphereMesh = Scene::uploadMesh(ctx.allocator, ctx.device, ctx.commandPool,
                                   ctx.graphicsQueue, vertices, indices);

    // Set up geometry description for triangles
    VkDeviceAddress vertexAddress = ctx.getBufferDeviceAddress(sphereMesh.vertexBuffer);
    VkDeviceAddress indexAddress = ctx.getBufferDeviceAddress(sphereMesh.indexBuffer);

    VkAccelerationStructureGeometryTrianglesDataKHR trianglesData = {};
    trianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    trianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    trianglesData.vertexData.deviceAddress = vertexAddress;
    trianglesData.vertexStride = sizeof(Vertex);
    trianglesData.maxVertex = sphereMesh.vertexCount - 1;
    trianglesData.indexType = VK_INDEX_TYPE_UINT32;
    trianglesData.indexData.deviceAddress = indexAddress;

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = trianglesData;

    uint32_t primitiveCount = sphereMesh.indexCount / 3;

    // Query build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
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
        &primitiveCount,
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

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    ctx.pfnCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    ctx.endSingleTimeCommands(cmd);

    std::cout << "[info] BLAS built (" << primitiveCount << " triangles)" << std::endl;
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
// RT pipeline creation (reflection-driven descriptor sets)
// --------------------------------------------------------------------------

void RTRenderer::createRTPipeline(VulkanContext& ctx) {
    // Shader stages:
    //   0 = raygen
    //   1 = miss
    //   2 = closest hit
    VkPipelineShaderStageCreateInfo stages[3] = {};

    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgenModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmissModule;
    stages[1].pName = "main";

    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = rchitModule;
    stages[2].pName = "main";

    // Shader groups:
    //   Group 0 (raygen): GENERAL, generalShader=0
    //   Group 1 (miss):   GENERAL, generalShader=1
    //   Group 2 (hit):    TRIANGLES_HIT_GROUP, closestHitShader=2
    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {};

    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Create descriptor set layouts from reflection data (NO hardcoded bindings)
    reflectedSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stageReflections);

    // Pipeline layout from reflection
    pipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, reflectedSetLayouts, stageReflections);

    // Create RT pipeline
    VkRayTracingPipelineCreateInfoKHR rtPipeInfo = {};
    rtPipeInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtPipeInfo.stageCount = 3;
    rtPipeInfo.pStages = stages;
    rtPipeInfo.groupCount = 3;
    rtPipeInfo.pGroups = groups;
    rtPipeInfo.maxPipelineRayRecursionDepth = 1;
    rtPipeInfo.layout = pipelineLayout;

    VkResult result = ctx.pfnCreateRayTracingPipelinesKHR(
        ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
        1, &rtPipeInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing pipeline");
    }

    std::cout << "[info] RT pipeline created (reflection-driven)" << std::endl;
}

// --------------------------------------------------------------------------
// SBT creation
// --------------------------------------------------------------------------

void RTRenderer::createSBT(VulkanContext& ctx) {
    uint32_t handleSize = ctx.rtPipelineProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = ctx.rtPipelineProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = ctx.rtPipelineProperties.shaderGroupBaseAlignment;

    uint32_t handleSizeAligned = static_cast<uint32_t>(alignUp(handleSize, handleAlignment));
    uint32_t groupCount = 3;

    // Get shader group handles
    uint32_t sbtSize = groupCount * handleSizeAligned;
    std::vector<uint8_t> handles(groupCount * handleSize);

    VkResult result = ctx.pfnGetRayTracingShaderGroupHandlesKHR(
        ctx.device, pipeline, 0, groupCount,
        handles.size(), handles.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to get ray tracing shader group handles");
    }

    // Compute region sizes (each region must be base-aligned)
    VkDeviceSize raygenRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize missRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize hitRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize totalSbtSize = raygenRegionSize + missRegionSize + hitRegionSize;

    // Create single SBT buffer
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

    // Map and copy handles to aligned offsets
    void* mapped;
    vmaMapMemory(ctx.allocator, sbtAllocation, &mapped);
    uint8_t* sbtData = static_cast<uint8_t*>(mapped);

    // Zero the entire buffer first
    memset(sbtData, 0, totalSbtSize);

    // Copy raygen handle (group 0)
    memcpy(sbtData, handles.data() + 0 * handleSize, handleSize);

    // Copy miss handle (group 1)
    memcpy(sbtData + raygenRegionSize, handles.data() + 1 * handleSize, handleSize);

    // Copy hit handle (group 2)
    memcpy(sbtData + raygenRegionSize + missRegionSize,
           handles.data() + 2 * handleSize, handleSize);

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
              << ", baseAlign=" << baseAlignment << ")" << std::endl;
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

    // Get merged bindings from reflection data for pool sizing
    auto mergedBindings = getMergedBindings(stageReflections);

    // Count pool sizes from reflection data
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "storage_image") vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else if (b.type == "acceleration_structure") vkType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        typeCounts[vkType]++;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    uint32_t maxSets = static_cast<uint32_t>(reflectedSetLayouts.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create RT descriptor pool");
    }

    // Allocate descriptor sets
    for (auto& [setIdx, layout] : reflectedSetLayouts) {
        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layout;

        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate RT descriptor set");
        }
        reflectedDescSets[setIdx] = set;
    }

    // Write descriptor sets based on reflection binding names
    // We need to keep descriptor info structs alive until vkUpdateDescriptorSets
    struct DescWriteInfo {
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorImageInfo imageInfo;
        VkWriteDescriptorSetAccelerationStructureKHR asInfo;
    };
    std::vector<DescWriteInfo> writeInfos(mergedBindings.size());
    std::vector<VkWriteDescriptorSet> writes;

    for (size_t i = 0; i < mergedBindings.size(); i++) {
        auto& b = mergedBindings[i];
        auto setIt = reflectedDescSets.find(b.set);
        if (setIt == reflectedDescSets.end()) continue;

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = setIt->second;
        w.dstBinding = static_cast<uint32_t>(b.binding);
        w.descriptorCount = 1;

        if (b.type == "uniform_buffer") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            // Match by name: "Camera" or "Light" -> cameraBuffer
            writeInfos[i].bufferInfo = {cameraBuffer, 0,
                static_cast<VkDeviceSize>(b.size > 0 ? b.size : 128)};
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
        } else {
            continue; // Skip unsupported types
        }

        writes.push_back(w);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    std::cout << "[info] RT descriptor set created (reflection-driven: "
              << reflectedSetLayouts.size() << " set(s), "
              << mergedBindings.size() << " binding(s))" << std::endl;
}

// --------------------------------------------------------------------------
// Camera UBO update
// --------------------------------------------------------------------------

void RTRenderer::updateCameraUBO(VulkanContext& ctx) {
    float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);
    Camera::RTCameraData camData = Camera::getRTCameraData(aspect);

    void* mapped;
    vmaMapMemory(ctx.allocator, cameraAllocation, &mapped);
    uint8_t* dst = static_cast<uint8_t*>(mapped);
    memcpy(dst, &camData.inverseView, sizeof(glm::mat4));
    memcpy(dst + sizeof(glm::mat4), &camData.inverseProjection, sizeof(glm::mat4));
    vmaUnmapMemory(ctx.allocator, cameraAllocation);
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

void RTRenderer::init(VulkanContext& ctx,
                      const std::string& rgenSpvPath,
                      const std::string& rmissSpvPath,
                      const std::string& rchitSpvPath,
                      uint32_t width, uint32_t height,
                      const std::string& sceneSource) {
    if (!ctx.supportsRT()) {
        throw std::runtime_error("Ray tracing is not supported on this device");
    }

    renderWidth = width;
    renderHeight = height;
    m_sceneSource = sceneSource;

    std::cout << "[info] Initializing RT renderer (" << width << "x" << height << ")" << std::endl;

    // Derive pipeline base from rgen path (strip .rgen.spv suffix)
    m_pipelineBase = rgenSpvPath;
    auto pos = m_pipelineBase.find(".rgen.spv");
    if (pos != std::string::npos) {
        m_pipelineBase = m_pipelineBase.substr(0, pos);
    }

    // Load reflection JSON for all RT stages
    namespace fs = std::filesystem;
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

    // Load shader modules
    auto rgenCode = SpvLoader::loadSPIRV(rgenSpvPath);
    rgenModule = SpvLoader::createShaderModule(ctx.device, rgenCode);

    auto rmissCode = SpvLoader::loadSPIRV(rmissSpvPath);
    rmissModule = SpvLoader::createShaderModule(ctx.device, rmissCode);

    auto rchitCode = SpvLoader::loadSPIRV(rchitSpvPath);
    rchitModule = SpvLoader::createShaderModule(ctx.device, rchitCode);

    // Create resources
    createStorageImage(ctx);
    createBLAS(ctx);
    createTLAS(ctx);
    createRTPipeline(ctx);
    createSBT(ctx);
    createDescriptorSet(ctx);

    std::cout << "[info] RT renderer initialized successfully" << std::endl;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RTRenderer::render(VulkanContext& ctx) {
    // Update camera UBO each frame
    updateCameraUBO(ctx);

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
    ctx.pfnCmdTraceRaysKHR(cmd,
        &raygenSBT, &missSBT, &hitSBT, &callableSBT,
        renderWidth, renderHeight, 1);

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

    // Shader modules
    if (rgenModule) vkDestroyShaderModule(ctx.device, rgenModule, nullptr);
    if (rmissModule) vkDestroyShaderModule(ctx.device, rmissModule, nullptr);
    if (rchitModule) vkDestroyShaderModule(ctx.device, rchitModule, nullptr);

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

    // Storage image
    if (storageImageView) vkDestroyImageView(ctx.device, storageImageView, nullptr);
    if (storageImage) vmaDestroyImage(ctx.allocator, storageImage, storageAllocation);

    // Sphere mesh
    Scene::destroyMesh(ctx.allocator, sphereMesh);
}
