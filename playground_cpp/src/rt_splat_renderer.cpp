/// RT Gaussian Splat Renderer — 3DGRT algorithm implementation
///
/// Renders Gaussian splat scenes via Vulkan ray tracing:
///   - AABB BLAS: one bounding box per Gaussian (3-sigma extent from covariance)
///   - Intersection shader: analytical ray-Gaussian peak response
///   - Closest-hit shader: packs hit data into vec4 payload
///   - Miss shader: signals no-hit via zero payload
///   - Raygen shader: multi-round trace loop with SH color + alpha compositing

#include "rt_splat_renderer.h"
#include "vulkan_context.h"
#include "gltf_loader.h"
#include "spv_loader.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>

// ---------------------------------------------------------------------------
// VMA buffer helpers (same pattern as splat_renderer.cpp)
// ---------------------------------------------------------------------------

static void createVmaBuffer(VmaAllocator allocator, VkDeviceSize size,
                            VkBufferUsageFlags usage, VmaMemoryUsage memUsage,
                            VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = std::max(size, VkDeviceSize(16));
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memUsage;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("RTSplatRenderer: failed to create VMA buffer");
    }
}

static void uploadVmaBuffer(VmaAllocator allocator, VmaAllocation allocation,
                            const void* data, VkDeviceSize size) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, allocation, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, allocation);
}

static void destroyVmaBuffer(VmaAllocator allocator, VkBuffer& buffer, VmaAllocation& alloc) {
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer, alloc);
        buffer = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
    }
}

// Align up to alignment boundary
static VkDeviceSize alignUp(VkDeviceSize val, VkDeviceSize alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

RTSplatRenderer::~RTSplatRenderer() {}

void RTSplatRenderer::init(VulkanContext& ctx, const GaussianSplatData& data,
                           const std::string& shaderBase,
                           uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;
    numSplats_ = data.num_splats;
    shDegree_ = data.sh_degree;

    createStorageImage(ctx);
    createBuffers(ctx, data);

    // Compute AABBs while we still have the splat data, then build BLAS
    std::vector<VkAabbPositionsKHR> aabbs;
    computeAABBs(data, aabbs);
    createBLAS(ctx, aabbs);

    createTLAS(ctx);
    createRTPipeline(ctx, shaderBase);
    createSBT(ctx);
    createDescriptorSet(ctx);
}

void RTSplatRenderer::updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                                   float fovY, float aspect,
                                   float nearPlane, float farPlane) {
    glm::mat4 view = glm::lookAt(eye, target, up);
    glm::mat4 proj = glm::perspective(fovY, aspect, nearPlane, farPlane);
    proj[1][1] *= -1.0f; // Vulkan Y-flip

    invView_ = glm::inverse(view);
    invProj_ = glm::inverse(proj);
}

// ---------------------------------------------------------------------------
// Storage image (RT output)
// ---------------------------------------------------------------------------

void RTSplatRenderer::createStorageImage(VulkanContext& ctx) {
    VkImageCreateInfo imgInfo = {};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgInfo.extent = {width_, height_, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imgInfo, &allocInfo,
                   &storageImage_, &storageAlloc_, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = storageImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(ctx.device, &viewInfo, nullptr, &storageView_);

    // Transition to GENERAL layout
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = storageImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    ctx.endSingleTimeCommands(cmd);
}

// ---------------------------------------------------------------------------
// Upload splat data to GPU
// ---------------------------------------------------------------------------

void RTSplatRenderer::createBuffers(VulkanContext& ctx, const GaussianSplatData& data) {
    const VkBufferUsageFlags ssboUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // Positions (vec4 per splat)
    VkDeviceSize posSize = data.positions.size() * sizeof(float);
    createVmaBuffer(ctx.allocator, posSize, ssboUsage,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, posBuffer_, posAlloc_);
    uploadVmaBuffer(ctx.allocator, posAlloc_, data.positions.data(), posSize);

    // Rotations (vec4 per splat)
    VkDeviceSize rotSize = data.rotations.size() * sizeof(float);
    createVmaBuffer(ctx.allocator, rotSize, ssboUsage,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, rotBuffer_, rotAlloc_);
    uploadVmaBuffer(ctx.allocator, rotAlloc_, data.rotations.data(), rotSize);

    // Scales: loader stores as vec3, shader expects vec4 (std430 alignment)
    {
        std::vector<float> scale4(numSplats_ * 4, 0.0f);
        for (uint32_t i = 0; i < numSplats_; ++i) {
            scale4[i * 4 + 0] = data.scales[i * 3 + 0];
            scale4[i * 4 + 1] = data.scales[i * 3 + 1];
            scale4[i * 4 + 2] = data.scales[i * 3 + 2];
        }
        VkDeviceSize scaleSize = scale4.size() * sizeof(float);
        createVmaBuffer(ctx.allocator, scaleSize, ssboUsage,
                        VMA_MEMORY_USAGE_CPU_TO_GPU, scaleBuffer_, scaleAlloc_);
        uploadVmaBuffer(ctx.allocator, scaleAlloc_, scale4.data(), scaleSize);
    }

    // Opacities (scalar per splat)
    VkDeviceSize opSize = data.opacities.size() * sizeof(float);
    createVmaBuffer(ctx.allocator, opSize, ssboUsage,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, opacityBuffer_, opacityAlloc_);
    uploadVmaBuffer(ctx.allocator, opacityAlloc_, data.opacities.data(), opSize);

    // SH coefficient buffers — pad vec3 data to vec4 for std430 alignment
    for (size_t i = 0; i < data.sh_coefficients.size(); ++i) {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        const auto& coeffs = data.sh_coefficients[i];
        size_t floatsPerSplat = coeffs.size() / numSplats_;
        if (floatsPerSplat == 3) {
            // Pad vec3 → vec4
            std::vector<float> padded(numSplats_ * 4, 0.0f);
            for (uint32_t s = 0; s < numSplats_; ++s) {
                padded[s * 4 + 0] = coeffs[s * 3 + 0];
                padded[s * 4 + 1] = coeffs[s * 3 + 1];
                padded[s * 4 + 2] = coeffs[s * 3 + 2];
            }
            VkDeviceSize shSize = padded.size() * sizeof(float);
            createVmaBuffer(ctx.allocator, shSize, ssboUsage,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            uploadVmaBuffer(ctx.allocator, alloc, padded.data(), shSize);
        } else {
            VkDeviceSize shSize = coeffs.size() * sizeof(float);
            createVmaBuffer(ctx.allocator, shSize, ssboUsage,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            uploadVmaBuffer(ctx.allocator, alloc, coeffs.data(), shSize);
        }
        shBuffers_.push_back(buf);
        shAllocs_.push_back(alloc);
    }

    // Camera UBO (2 x mat4 = 128 bytes)
    createVmaBuffer(ctx.allocator, 128,
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, cameraBuffer_, cameraAlloc_);
}

// ---------------------------------------------------------------------------
// Compute AABBs: 3-sigma bounding box from Gaussian covariance
// ---------------------------------------------------------------------------

void RTSplatRenderer::computeAABBs(const GaussianSplatData& data,
                                   std::vector<VkAabbPositionsKHR>& aabbs) {
    aabbs.resize(numSplats_);

    for (uint32_t i = 0; i < numSplats_; ++i) {
        // Position (xyz from vec4)
        float cx = data.positions[i * 4 + 0];
        float cy = data.positions[i * 4 + 1];
        float cz = data.positions[i * 4 + 2];

        // Quaternion (xyzw)
        float qi = data.rotations[i * 4 + 0];
        float qj = data.rotations[i * 4 + 1];
        float qk = data.rotations[i * 4 + 2];
        float qr = data.rotations[i * 4 + 3];

        // Scale (exp of log-scale, xyz — stored as vec3)
        float sx = std::exp(data.scales[i * 3 + 0]);
        float sy = std::exp(data.scales[i * 3 + 1]);
        float sz = std::exp(data.scales[i * 3 + 2]);

        // Build rotation matrix from quaternion
        float qi2 = qi * qi, qj2 = qj * qj, qk2 = qk * qk;
        float qri = qr * qi, qrj = qr * qj, qrk = qr * qk;
        float qij = qi * qj, qik = qi * qk, qjk = qj * qk;

        // R[row][col]
        float R[3][3];
        R[0][0] = 1.0f - 2.0f * (qj2 + qk2);
        R[0][1] = 2.0f * (qij - qrk);
        R[0][2] = 2.0f * (qik + qrj);
        R[1][0] = 2.0f * (qij + qrk);
        R[1][1] = 1.0f - 2.0f * (qi2 + qk2);
        R[1][2] = 2.0f * (qjk - qri);
        R[2][0] = 2.0f * (qik - qrj);
        R[2][1] = 2.0f * (qjk + qri);
        R[2][2] = 1.0f - 2.0f * (qi2 + qj2);

        // Covariance diagonal: Sigma_jj = sum_k R_jk^2 * S_k^2
        // Half-extent at 3 sigma: 3 * sqrt(Sigma_jj)
        float s2[3] = {sx * sx, sy * sy, sz * sz};
        float halfExt[3];
        for (int j = 0; j < 3; ++j) {
            float cov_jj = 0.0f;
            for (int k = 0; k < 3; ++k) {
                cov_jj += R[j][k] * R[j][k] * s2[k];
            }
            halfExt[j] = 3.0f * std::sqrt(cov_jj);
        }

        aabbs[i].minX = cx - halfExt[0];
        aabbs[i].minY = cy - halfExt[1];
        aabbs[i].minZ = cz - halfExt[2];
        aabbs[i].maxX = cx + halfExt[0];
        aabbs[i].maxY = cy + halfExt[1];
        aabbs[i].maxZ = cz + halfExt[2];
    }
}

// ---------------------------------------------------------------------------
// BLAS: single AABB geometry containing all Gaussians
// ---------------------------------------------------------------------------

void RTSplatRenderer::createBLAS(VulkanContext& ctx,
                                  const std::vector<VkAabbPositionsKHR>& aabbs) {
    // Upload AABB data to GPU buffer
    VkDeviceSize aabbSize = aabbs.size() * sizeof(VkAabbPositionsKHR);
    createVmaBuffer(ctx.allocator, aabbSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU, aabbBuffer_, aabbAlloc_);
    uploadVmaBuffer(ctx.allocator, aabbAlloc_, aabbs.data(), aabbSize);

    VkDeviceAddress aabbAddress = ctx.getBufferDeviceAddress(aabbBuffer_);

    // BLAS geometry: single AABB geometry containing all Gaussians
    VkAccelerationStructureGeometryKHR blasGeo = {};
    blasGeo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    blasGeo.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    blasGeo.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    blasGeo.geometry.aabbs.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    blasGeo.geometry.aabbs.data.deviceAddress = aabbAddress;
    blasGeo.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

    // Query build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &blasGeo;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx.pfnGetAccelerationStructureBuildSizesKHR(ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &numSplats_, &sizeInfo);

    // Create BLAS buffer
    createVmaBuffer(ctx.allocator, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, blasBuffer_, blasAlloc_);

    // Create BLAS handle
    VkAccelerationStructureCreateInfoKHR blasCI = {};
    blasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blasCI.buffer = blasBuffer_;
    blasCI.size = sizeInfo.accelerationStructureSize;
    blasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    ctx.pfnCreateAccelerationStructureKHR(ctx.device, &blasCI, nullptr, &blas_);

    // Scratch buffer
    createVmaBuffer(ctx.allocator, sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, blasScratchBuffer_, blasScratchAlloc_);

    // Build
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = blas_;
    buildInfo.scratchData.deviceAddress = ctx.getBufferDeviceAddress(blasScratchBuffer_);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = numSplats_;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    ctx.pfnCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    ctx.endSingleTimeCommands(cmd);
}

// ---------------------------------------------------------------------------
// TLAS: single instance pointing to the AABB BLAS
// ---------------------------------------------------------------------------

void RTSplatRenderer::createTLAS(VulkanContext& ctx) {
    // Get BLAS device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blas_;
    VkDeviceAddress blasAddress =
        ctx.pfnGetAccelerationStructureDeviceAddressKHR(ctx.device, &addressInfo);

    // Single instance, identity transform
    VkAccelerationStructureInstanceKHR instance = {};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blasAddress;

    // Upload instance data
    VkDeviceSize instanceSize = sizeof(VkAccelerationStructureInstanceKHR);
    createVmaBuffer(ctx.allocator, instanceSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU, instanceBuffer_, instanceAlloc_);
    uploadVmaBuffer(ctx.allocator, instanceAlloc_, &instance, instanceSize);

    VkDeviceAddress instanceAddress = ctx.getBufferDeviceAddress(instanceBuffer_);

    // TLAS geometry
    VkAccelerationStructureGeometryKHR tlasGeo = {};
    tlasGeo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeo.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeo.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeo.geometry.instances.data.deviceAddress = instanceAddress;

    // Query build sizes
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeo;

    uint32_t instanceCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    ctx.pfnGetAccelerationStructureBuildSizesKHR(ctx.device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo);

    // Create TLAS buffer
    createVmaBuffer(ctx.allocator, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, tlasBuffer_, tlasAlloc_);

    // Create TLAS handle
    VkAccelerationStructureCreateInfoKHR tlasCI = {};
    tlasCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCI.buffer = tlasBuffer_;
    tlasCI.size = sizeInfo.accelerationStructureSize;
    tlasCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    ctx.pfnCreateAccelerationStructureKHR(ctx.device, &tlasCI, nullptr, &tlas_);

    // Scratch buffer
    createVmaBuffer(ctx.allocator, sizeInfo.buildScratchSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY, tlasScratchBuffer_, tlasScratchAlloc_);

    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.dstAccelerationStructure = tlas_;
    buildInfo.scratchData.deviceAddress = ctx.getBufferDeviceAddress(tlasScratchBuffer_);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    ctx.pfnCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    ctx.endSingleTimeCommands(cmd);
}

// ---------------------------------------------------------------------------
// RT Pipeline: raygen + miss + procedural hit group (intersection + closest_hit)
// ---------------------------------------------------------------------------

void RTSplatRenderer::createRTPipeline(VulkanContext& ctx,
                                       const std::string& shaderBase) {
    // Load shader modules
    rgenModule_  = SpvLoader::createShaderModule(ctx.device,
                       SpvLoader::loadSPIRV(shaderBase + ".rgen.spv"));
    rmissModule_ = SpvLoader::createShaderModule(ctx.device,
                       SpvLoader::loadSPIRV(shaderBase + ".rmiss.spv"));
    rchitModule_ = SpvLoader::createShaderModule(ctx.device,
                       SpvLoader::loadSPIRV(shaderBase + ".rchit.spv"));
    rintModule_  = SpvLoader::createShaderModule(ctx.device,
                       SpvLoader::loadSPIRV(shaderBase + ".rint.spv"));

    // Shader stages
    std::vector<VkPipelineShaderStageCreateInfo> stages(4);
    for (auto& s : stages) {
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        s.pName = "main";
    }
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgenModule_;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmissModule_;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = rchitModule_;
    stages[3].stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    stages[3].module = rintModule_;

    // Shader groups: raygen(0), miss(1), hit(2: intersection + closest_hit)
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(3);
    for (auto& g : groups) {
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.generalShader = VK_SHADER_UNUSED_KHR;
        g.closestHitShader = VK_SHADER_UNUSED_KHR;
        g.anyHitShader = VK_SHADER_UNUSED_KHR;
        g.intersectionShader = VK_SHADER_UNUSED_KHR;
    }

    // Group 0: raygen (general)
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0; // stage index

    // Group 1: miss (general)
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;

    // Group 2: procedural hit group (intersection + closest_hit)
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    groups[2].closestHitShader = 2;
    groups[2].intersectionShader = 3;

    // Descriptor set layouts matching compiled shader bindings:
    //   Set 0 (raygen): binding 0=Camera UBO, 1=TLAS, 2=storage image,
    //                    3..3+numSH-1=SH buffers, 3+numSH=splat_pos
    //   Set 1 (intersection): binding 0=splat_pos, 1=splat_rot, 2=splat_scale, 3=splat_opacity
    uint32_t numSH = static_cast<uint32_t>(shBuffers_.size());

    // --- Set 0: raygen ---
    {
        uint32_t numBindings0 = 3 + numSH + 1; // camera + tlas + image + SH buffers + pos
        std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings0);
        uint32_t b = 0;

        // Binding 0: Camera UBO
        bindings[b] = {};
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b++;

        // Binding 1: acceleration structure
        bindings[b] = {};
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b++;

        // Binding 2: storage image
        bindings[b] = {};
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b++;

        // Bindings 3..3+numSH-1: SH coefficient buffers
        for (uint32_t i = 0; i < numSH; ++i) {
            bindings[b] = {};
            bindings[b].binding = b;
            bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[b].descriptorCount = 1;
            bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            b++;
        }

        // Binding 3+numSH: splat_pos for SH direction computation
        bindings[b] = {};
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        b++;

        VkDescriptorSetLayoutCreateInfo layoutCI = {};
        layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = numBindings0;
        layoutCI.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(ctx.device, &layoutCI, nullptr, &setLayout0_);
    }

    // --- Set 1: intersection ---
    {
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (int i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        }

        VkDescriptorSetLayoutCreateInfo layoutCI = {};
        layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCI.bindingCount = 4;
        layoutCI.pBindings = bindings;
        vkCreateDescriptorSetLayout(ctx.device, &layoutCI, nullptr, &setLayout1_);
    }

    // Pipeline layout: [set 0, set 1]
    VkDescriptorSetLayout setLayouts[2] = {setLayout0_, setLayout1_};
    VkPipelineLayoutCreateInfo plCI = {};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 2;
    plCI.pSetLayouts = setLayouts;
    vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &pipelineLayout_);

    // Create RT pipeline
    VkRayTracingPipelineCreateInfoKHR rtCI = {};
    rtCI.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtCI.stageCount = static_cast<uint32_t>(stages.size());
    rtCI.pStages = stages.data();
    rtCI.groupCount = static_cast<uint32_t>(groups.size());
    rtCI.pGroups = groups.data();
    rtCI.maxPipelineRayRecursionDepth = 1;
    rtCI.layout = pipelineLayout_;

    ctx.pfnCreateRayTracingPipelinesKHR(ctx.device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                        1, &rtCI, nullptr, &pipeline_);
}

// ---------------------------------------------------------------------------
// Shader Binding Table
// ---------------------------------------------------------------------------

void RTSplatRenderer::createSBT(VulkanContext& ctx) {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps = {};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props2);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    uint32_t handleSizeAligned = static_cast<uint32_t>(alignUp(handleSize, handleAlignment));

    uint32_t groupCount = 3; // raygen, miss, hit
    uint32_t sbtSize = groupCount * handleSizeAligned;

    std::vector<uint8_t> handles(groupCount * handleSize);
    ctx.pfnGetRayTracingShaderGroupHandlesKHR(ctx.device, pipeline_,
        0, groupCount, handles.size(), handles.data());

    // SBT regions
    VkDeviceSize raygenRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize missRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize hitRegionSize = alignUp(handleSizeAligned, baseAlignment);
    VkDeviceSize totalSize = raygenRegionSize + missRegionSize + hitRegionSize;

    createVmaBuffer(ctx.allocator, totalSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU, sbtBuffer_, sbtAlloc_);

    // Write handles
    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, sbtAlloc_, &mapped);
    auto* dst = static_cast<uint8_t*>(mapped);

    // Raygen at offset 0
    std::memcpy(dst, handles.data() + 0 * handleSize, handleSize);
    // Miss at offset raygenRegionSize
    std::memcpy(dst + raygenRegionSize, handles.data() + 1 * handleSize, handleSize);
    // Hit at offset raygenRegionSize + missRegionSize
    std::memcpy(dst + raygenRegionSize + missRegionSize,
                handles.data() + 2 * handleSize, handleSize);

    vmaUnmapMemory(ctx.allocator, sbtAlloc_);

    VkDeviceAddress sbtBase = ctx.getBufferDeviceAddress(sbtBuffer_);

    raygenSBT_.deviceAddress = sbtBase;
    raygenSBT_.stride = raygenRegionSize;
    raygenSBT_.size = raygenRegionSize;

    missSBT_.deviceAddress = sbtBase + raygenRegionSize;
    missSBT_.stride = handleSizeAligned;
    missSBT_.size = missRegionSize;

    hitSBT_.deviceAddress = sbtBase + raygenRegionSize + missRegionSize;
    hitSBT_.stride = handleSizeAligned;
    hitSBT_.size = hitRegionSize;

    callableSBT_ = {}; // unused
}

// ---------------------------------------------------------------------------
// Descriptor set
// ---------------------------------------------------------------------------

void RTSplatRenderer::createDescriptorSet(VulkanContext& ctx) {
    uint32_t numSH = static_cast<uint32_t>(shBuffers_.size());

    // Pool sizes (need 2 sets)
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 + 1 + numSH}, // intersection(4) + raygen(pos+SH)
    };

    VkDescriptorPoolCreateInfo poolCI = {};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets = 2;
    poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCI.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &descriptorPool_);

    // Allocate both sets
    VkDescriptorSetLayout layouts[2] = {setLayout0_, setLayout1_};
    VkDescriptorSet sets[2];
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, sets);
    descSet0_ = sets[0];
    descSet1_ = sets[1];

    std::vector<VkWriteDescriptorSet> writes;
    VkWriteDescriptorSet w = {};

    // ========== Set 0 (raygen) ==========

    // Binding 0: Camera UBO
    VkDescriptorBufferInfo cameraInfo = {cameraBuffer_, 0, 128};
    w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet0_;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w.pBufferInfo = &cameraInfo;
    writes.push_back(w);

    // Binding 1: TLAS
    VkWriteDescriptorSetAccelerationStructureKHR asWrite = {};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas_;
    w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet0_;
    w.dstBinding = 1;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w.pNext = &asWrite;
    writes.push_back(w);

    // Binding 2: Storage image
    VkDescriptorImageInfo imgInfo = {};
    imgInfo.imageView = storageView_;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet0_;
    w.dstBinding = 2;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo = &imgInfo;
    writes.push_back(w);

    // Bindings 3..3+numSH-1: SH coefficient buffers
    std::vector<VkDescriptorBufferInfo> shInfos(numSH);
    for (uint32_t i = 0; i < numSH; ++i) {
        shInfos[i] = {shBuffers_[i], 0, VK_WHOLE_SIZE};
        w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descSet0_;
        w.dstBinding = 3 + i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &shInfos[i];
        writes.push_back(w);
    }

    // Binding 3+numSH: splat_pos for raygen
    VkDescriptorBufferInfo posRaygenInfo = {posBuffer_, 0, VK_WHOLE_SIZE};
    w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet0_;
    w.dstBinding = 3 + numSH;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &posRaygenInfo;
    writes.push_back(w);

    // ========== Set 1 (intersection) ==========

    // Bindings 0-3: splat_pos, splat_rot, splat_scale, splat_opacity
    VkBuffer isectBuffers[4] = {posBuffer_, rotBuffer_, scaleBuffer_, opacityBuffer_};
    VkDescriptorBufferInfo isectInfos[4];
    for (int i = 0; i < 4; ++i) {
        isectInfos[i] = {isectBuffers[i], 0, VK_WHOLE_SIZE};
        w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = descSet1_;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &isectInfos[i];
        writes.push_back(w);
    }

    vkUpdateDescriptorSets(ctx.device,
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// ---------------------------------------------------------------------------
// Update camera UBO
// ---------------------------------------------------------------------------

void RTSplatRenderer::updateCameraUBO(VulkanContext& ctx) {
    struct CameraData {
        glm::mat4 inv_view;
        glm::mat4 inv_proj;
    } cam;
    cam.inv_view = invView_;
    cam.inv_proj = invProj_;

    uploadVmaBuffer(ctx.allocator, cameraAlloc_, &cam, sizeof(cam));
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void RTSplatRenderer::render(VulkanContext& ctx) {
    updateCameraUBO(ctx);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Clear storage image to opaque black before tracing
    VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkImageSubresourceRange clearRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, storageImage_, VK_IMAGE_LAYOUT_GENERAL,
                         &clearColor, 1, &clearRange);

    // Barrier: clear -> RT shader write
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = storageImage_;
    barrier.subresourceRange = clearRange;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_);
    VkDescriptorSet sets[2] = {descSet0_, descSet1_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            pipelineLayout_, 0, 2, sets, 0, nullptr);

    ctx.pfnCmdTraceRaysKHR(cmd,
        &raygenSBT_, &missSBT_, &hitSBT_, &callableSBT_,
        width_, height_, 1);

    // Barrier: ensure RT shader writes are visible before any reads
    VkImageMemoryBarrier postBarrier = {};
    postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.image = storageImage_;
    postBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    postBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &postBarrier);

    ctx.endSingleTimeCommands(cmd);
}

// ---------------------------------------------------------------------------
// Blit to swapchain
// ---------------------------------------------------------------------------

void RTSplatRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                      VkImage swapImage, VkExtent2D extent) {
    // Transition storage image: GENERAL -> TRANSFER_SRC
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.image = storageImage_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Transition swap image: UNDEFINED -> TRANSFER_DST
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = swapImage;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit
    VkImageBlit region = {};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {static_cast<int32_t>(width_),
                            static_cast<int32_t>(height_), 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                            static_cast<int32_t>(extent.height), 1};
    vkCmdBlitImage(cmd,
        storageImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region, VK_FILTER_LINEAR);

    // Transition swap image: TRANSFER_DST -> PRESENT_SRC
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.image = swapImage;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Transition storage image back: TRANSFER_SRC -> GENERAL
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.image = storageImage_;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void RTSplatRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    // Shader modules
    if (rgenModule_)  vkDestroyShaderModule(ctx.device, rgenModule_, nullptr);
    if (rmissModule_) vkDestroyShaderModule(ctx.device, rmissModule_, nullptr);
    if (rchitModule_) vkDestroyShaderModule(ctx.device, rchitModule_, nullptr);
    if (rintModule_)  vkDestroyShaderModule(ctx.device, rintModule_, nullptr);

    // Pipeline
    if (pipeline_)       vkDestroyPipeline(ctx.device, pipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(ctx.device, pipelineLayout_, nullptr);

    // Descriptors
    if (descriptorPool_) vkDestroyDescriptorPool(ctx.device, descriptorPool_, nullptr);
    if (setLayout0_)     vkDestroyDescriptorSetLayout(ctx.device, setLayout0_, nullptr);
    if (setLayout1_)     vkDestroyDescriptorSetLayout(ctx.device, setLayout1_, nullptr);

    // SBT
    destroyVmaBuffer(ctx.allocator, sbtBuffer_, sbtAlloc_);

    // Acceleration structures
    if (tlas_) ctx.pfnDestroyAccelerationStructureKHR(ctx.device, tlas_, nullptr);
    if (blas_) ctx.pfnDestroyAccelerationStructureKHR(ctx.device, blas_, nullptr);
    destroyVmaBuffer(ctx.allocator, tlasBuffer_, tlasAlloc_);
    destroyVmaBuffer(ctx.allocator, blasBuffer_, blasAlloc_);
    destroyVmaBuffer(ctx.allocator, instanceBuffer_, instanceAlloc_);
    destroyVmaBuffer(ctx.allocator, aabbBuffer_, aabbAlloc_);
    destroyVmaBuffer(ctx.allocator, tlasScratchBuffer_, tlasScratchAlloc_);
    destroyVmaBuffer(ctx.allocator, blasScratchBuffer_, blasScratchAlloc_);

    // Camera UBO
    destroyVmaBuffer(ctx.allocator, cameraBuffer_, cameraAlloc_);

    // Splat buffers
    destroyVmaBuffer(ctx.allocator, posBuffer_, posAlloc_);
    destroyVmaBuffer(ctx.allocator, rotBuffer_, rotAlloc_);
    destroyVmaBuffer(ctx.allocator, scaleBuffer_, scaleAlloc_);
    destroyVmaBuffer(ctx.allocator, opacityBuffer_, opacityAlloc_);
    for (size_t i = 0; i < shBuffers_.size(); ++i) {
        destroyVmaBuffer(ctx.allocator, shBuffers_[i], shAllocs_[i]);
    }
    shBuffers_.clear();
    shAllocs_.clear();

    // Storage image
    if (storageView_)  vkDestroyImageView(ctx.device, storageView_, nullptr);
    if (storageImage_) vmaDestroyImage(ctx.allocator, storageImage_, storageAlloc_);
}
