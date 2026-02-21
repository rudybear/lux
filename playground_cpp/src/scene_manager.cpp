#include "scene_manager.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <cfloat>

// --------------------------------------------------------------------------
// Static helper: detect glTF files
// --------------------------------------------------------------------------

bool SceneManager::isGltfFile(const std::string& source) {
    if (source.size() > 4 && source.substr(source.size()-4) == ".glb") return true;
    if (source.size() > 5 && source.substr(source.size()-5) == ".gltf") return true;
    return false;
}

// --------------------------------------------------------------------------
// Load scene geometry (CPU-side)
// --------------------------------------------------------------------------

void SceneManager::loadScene(VulkanContext& ctx, const std::string& sceneSource) {
    m_sceneSource = sceneSource;

    if (isGltfFile(sceneSource)) {
        std::cout << "[info] Loading glTF scene: " << sceneSource << std::endl;
        m_gltfScene = loadGltf(sceneSource);
        m_hasGltfScene = true;

        if (!m_gltfScene.meshes.empty()) {
            auto drawItems = flattenScene(m_gltfScene);

            // Collect all vertices/indices, transformed by world transform
            uint32_t vertexOffset = 0;
            for (auto& item : drawItems) {
                auto& gmesh = m_gltfScene.meshes[item.meshIndex];
                glm::mat4 world = item.worldTransform;
                glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(world)));

                for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                    Vertex v;
                    glm::vec4 pos4 = world * glm::vec4(gmesh.vertices[i].position, 1.0f);
                    v.position = glm::vec3(pos4);
                    v.normal = glm::normalize(normalMat * gmesh.vertices[i].normal);
                    v.uv = gmesh.vertices[i].uv;
                    m_vertices.push_back(v);
                }
                for (auto idx : gmesh.indices) {
                    m_indices.push_back(idx + vertexOffset);
                }
                vertexOffset += static_cast<uint32_t>(gmesh.vertices.size());
            }
            std::cout << "[info] Loaded glTF mesh: " << m_vertices.size()
                      << " vertices, " << m_indices.size() << " indices" << std::endl;
        } else {
            std::cout << "[warn] glTF has no meshes, falling back to sphere" << std::endl;
            Scene::generateSphere(32, 32, m_vertices, m_indices);
        }
    } else if (sceneSource == "sphere" || sceneSource == "triangle" ||
               sceneSource == "fullscreen") {
        // For sphere, generate geometry; for triangle/fullscreen, vertices may be empty
        if (sceneSource == "sphere") {
            Scene::generateSphere(32, 32, m_vertices, m_indices);
        }
    } else {
        // Default: treat as sphere
        Scene::generateSphere(32, 32, m_vertices, m_indices);
    }

    // Compute auto-camera from vertex bounds
    if (!m_vertices.empty()) {
        computeAutoCamera(m_vertices);
    }
}

// --------------------------------------------------------------------------
// Upload mesh to GPU
// --------------------------------------------------------------------------

void SceneManager::uploadToGPU(VulkanContext& ctx, int vertexStride) {
    if (m_vertices.empty() && m_indices.empty()) return;

    if (vertexStride == 48 && m_hasGltfScene) {
        // 48-byte stride upload with tangents for raster path
        auto drawItems = flattenScene(m_gltfScene);

        std::vector<float> vdata;
        std::vector<uint32_t> allIndices;
        uint32_t vertexOffset = 0;

        for (auto& item : drawItems) {
            auto& gmesh = m_gltfScene.meshes[item.meshIndex];
            glm::mat4 world = item.worldTransform;
            glm::mat3 upper3x3 = glm::mat3(world);
            glm::mat3 normalMat = glm::transpose(glm::inverse(upper3x3));

            for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                auto& gv = gmesh.vertices[i];
                glm::vec4 pos4 = world * glm::vec4(gv.position, 1.0f);
                glm::vec3 pos(pos4);
                glm::vec3 norm = glm::normalize(normalMat * gv.normal);

                vdata.push_back(pos.x);
                vdata.push_back(pos.y);
                vdata.push_back(pos.z);
                vdata.push_back(norm.x);
                vdata.push_back(norm.y);
                vdata.push_back(norm.z);
                vdata.push_back(gv.uv.x);
                vdata.push_back(gv.uv.y);
                if (gmesh.hasTangents) {
                    glm::vec3 tang = glm::normalize(upper3x3 * glm::vec3(gv.tangent));
                    vdata.push_back(tang.x);
                    vdata.push_back(tang.y);
                    vdata.push_back(tang.z);
                    vdata.push_back(gv.tangent.w);
                } else {
                    vdata.push_back(1.0f);
                    vdata.push_back(0.0f);
                    vdata.push_back(0.0f);
                    vdata.push_back(1.0f);
                }
            }

            for (auto idx : gmesh.indices) {
                allIndices.push_back(idx + vertexOffset);
            }
            vertexOffset += static_cast<uint32_t>(gmesh.vertices.size());
        }

        // Upload raw vertex data directly (bypasses Scene::uploadMesh which uses 32-byte Vertex)
        VkDeviceSize vbufSize = vdata.size() * sizeof(float);
        VkBufferCreateInfo vbufInfo = {};
        vbufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vbufInfo.size = vbufSize;
        vbufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

        VmaAllocationCreateInfo vbufAllocInfo = {};
        vbufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateBuffer(ctx.allocator, &vbufInfo, &vbufAllocInfo,
                        &m_mesh.vertexBuffer, &m_mesh.vertexAllocation, nullptr);

        VkBuffer vStagingBuffer;
        VmaAllocation vStagingAlloc;
        VkBufferCreateInfo vStagingInfo = {};
        vStagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vStagingInfo.size = vbufSize;
        vStagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo vStagingAllocInfo = {};
        vStagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &vStagingInfo, &vStagingAllocInfo,
                        &vStagingBuffer, &vStagingAlloc, nullptr);

        void* vMapped;
        vmaMapMemory(ctx.allocator, vStagingAlloc, &vMapped);
        memcpy(vMapped, vdata.data(), vbufSize);
        vmaUnmapMemory(ctx.allocator, vStagingAlloc);

        VkDeviceSize ibufSize = allIndices.size() * sizeof(uint32_t);
        VkBufferCreateInfo ibufInfo = {};
        ibufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ibufInfo.size = ibufSize;
        ibufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

        VmaAllocationCreateInfo ibufAllocInfo = {};
        ibufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        vmaCreateBuffer(ctx.allocator, &ibufInfo, &ibufAllocInfo,
                        &m_mesh.indexBuffer, &m_mesh.indexAllocation, nullptr);

        VkBuffer iStagingBuffer;
        VmaAllocation iStagingAlloc;
        VkBufferCreateInfo iStagingInfo = {};
        iStagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        iStagingInfo.size = ibufSize;
        iStagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo iStagingAllocInfo = {};
        iStagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &iStagingInfo, &iStagingAllocInfo,
                        &iStagingBuffer, &iStagingAlloc, nullptr);

        void* iMapped;
        vmaMapMemory(ctx.allocator, iStagingAlloc, &iMapped);
        memcpy(iMapped, allIndices.data(), ibufSize);
        vmaUnmapMemory(ctx.allocator, iStagingAlloc);

        VkCommandBuffer copyCmd = ctx.beginSingleTimeCommands();
        VkBufferCopy vCopy = {};
        vCopy.size = vbufSize;
        vkCmdCopyBuffer(copyCmd, vStagingBuffer, m_mesh.vertexBuffer, 1, &vCopy);
        VkBufferCopy iCopy = {};
        iCopy.size = ibufSize;
        vkCmdCopyBuffer(copyCmd, iStagingBuffer, m_mesh.indexBuffer, 1, &iCopy);
        ctx.endSingleTimeCommands(copyCmd);

        vmaDestroyBuffer(ctx.allocator, vStagingBuffer, vStagingAlloc);
        vmaDestroyBuffer(ctx.allocator, iStagingBuffer, iStagingAlloc);

        m_mesh.vertexCount = vertexOffset;
        m_mesh.indexCount = static_cast<uint32_t>(allIndices.size());

        std::cout << "[info] Uploaded glTF mesh with 48-byte stride ("
                  << m_mesh.vertexCount << " verts, " << m_mesh.indexCount << " indices)" << std::endl;
    } else if (vertexStride == 48 && !m_hasGltfScene) {
        // 48-byte stride for non-glTF (sphere) - pad with tangent
        std::vector<float> vdata;
        vdata.reserve(m_vertices.size() * 12);
        for (auto& v : m_vertices) {
            vdata.push_back(v.position.x);
            vdata.push_back(v.position.y);
            vdata.push_back(v.position.z);
            vdata.push_back(v.normal.x);
            vdata.push_back(v.normal.y);
            vdata.push_back(v.normal.z);
            vdata.push_back(v.uv.x);
            vdata.push_back(v.uv.y);
            // Default tangent
            vdata.push_back(1.0f);
            vdata.push_back(0.0f);
            vdata.push_back(0.0f);
            vdata.push_back(1.0f);
        }

        VkDeviceSize vbufSize = vdata.size() * sizeof(float);
        VkBufferCreateInfo vbufInfo = {};
        vbufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vbufInfo.size = vbufSize;
        vbufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        VmaAllocationCreateInfo vbufAllocInfo = {};
        vbufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &vbufInfo, &vbufAllocInfo,
                        &m_mesh.vertexBuffer, &m_mesh.vertexAllocation, nullptr);

        VkBuffer vStagingBuf;
        VmaAllocation vStagingAlloc;
        VkBufferCreateInfo vStgInfo = {};
        vStgInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vStgInfo.size = vbufSize;
        vStgInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo vStgAllocInfo = {};
        vStgAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &vStgInfo, &vStgAllocInfo,
                        &vStagingBuf, &vStagingAlloc, nullptr);

        void* vMapped;
        vmaMapMemory(ctx.allocator, vStagingAlloc, &vMapped);
        memcpy(vMapped, vdata.data(), vbufSize);
        vmaUnmapMemory(ctx.allocator, vStagingAlloc);

        VkDeviceSize ibufSize = m_indices.size() * sizeof(uint32_t);
        VkBufferCreateInfo ibufInfo = {};
        ibufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ibufInfo.size = ibufSize;
        ibufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        VmaAllocationCreateInfo ibufAllocInfo = {};
        ibufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &ibufInfo, &ibufAllocInfo,
                        &m_mesh.indexBuffer, &m_mesh.indexAllocation, nullptr);

        VkBuffer iStagingBuf;
        VmaAllocation iStagingAlloc;
        VkBufferCreateInfo iStgInfo = {};
        iStgInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        iStgInfo.size = ibufSize;
        iStgInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo iStgAllocInfo = {};
        iStgAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
        vmaCreateBuffer(ctx.allocator, &iStgInfo, &iStgAllocInfo,
                        &iStagingBuf, &iStagingAlloc, nullptr);

        void* iMapped;
        vmaMapMemory(ctx.allocator, iStagingAlloc, &iMapped);
        memcpy(iMapped, m_indices.data(), ibufSize);
        vmaUnmapMemory(ctx.allocator, iStagingAlloc);

        VkCommandBuffer copyCmd = ctx.beginSingleTimeCommands();
        VkBufferCopy vCopy = {}; vCopy.size = vbufSize;
        vkCmdCopyBuffer(copyCmd, vStagingBuf, m_mesh.vertexBuffer, 1, &vCopy);
        VkBufferCopy iCopy = {}; iCopy.size = ibufSize;
        vkCmdCopyBuffer(copyCmd, iStagingBuf, m_mesh.indexBuffer, 1, &iCopy);
        ctx.endSingleTimeCommands(copyCmd);

        vmaDestroyBuffer(ctx.allocator, vStagingBuf, vStagingAlloc);
        vmaDestroyBuffer(ctx.allocator, iStagingBuf, iStagingAlloc);

        m_mesh.vertexCount = static_cast<uint32_t>(m_vertices.size());
        m_mesh.indexCount = static_cast<uint32_t>(m_indices.size());

        std::cout << "[info] Uploaded sphere mesh with 48-byte stride" << std::endl;
    } else {
        // Standard 32-byte Vertex upload
        if (!m_vertices.empty() && !m_indices.empty()) {
            m_mesh = Scene::uploadMesh(ctx.allocator, ctx.device, ctx.commandPool,
                                       ctx.graphicsQueue, m_vertices, m_indices);
        }
    }
}

// --------------------------------------------------------------------------
// Upload textures
// --------------------------------------------------------------------------

void SceneManager::uploadTextures(VulkanContext& ctx) {
    createDefaultTextures(ctx);
    uploadGltfTextures(ctx);

    // If no named textures at all (non-glTF scene), create a procedural one
    if (m_namedTextures.empty()) {
        std::cout << "Generating procedural texture..." << std::endl;
        auto texPixels = Scene::generateProceduralTexture(512);
        auto gpuTex = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                            ctx.graphicsQueue, texPixels, 512, 512);
        m_namedTextures["albedo_tex"] = gpuTex;
        m_namedTextures["base_color_tex"] = gpuTex;
    }
}

// --------------------------------------------------------------------------
// Default textures
// --------------------------------------------------------------------------

void SceneManager::createDefaultTextures(VulkanContext& ctx) {
    std::vector<uint8_t> white = {255, 255, 255, 255};
    m_defaultWhiteTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                  ctx.graphicsQueue, white, 1, 1);
    std::vector<uint8_t> black = {0, 0, 0, 255};
    m_defaultBlackTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                  ctx.graphicsQueue, black, 1, 1);
    // Flat normal texture (128,128,255) = tangent-space (0,0,1)
    std::vector<uint8_t> flatNormal = {128, 128, 255, 255};
    m_defaultNormalTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                   ctx.graphicsQueue, flatNormal, 1, 1);
    std::cout << "[info] Created default fallback textures (white, black, flat-normal)" << std::endl;
}

// --------------------------------------------------------------------------
// Upload glTF textures
// --------------------------------------------------------------------------

void SceneManager::uploadGltfTextures(VulkanContext& ctx) {
    if (!m_hasGltfScene || m_gltfScene.materials.empty()) return;

    auto& mat = m_gltfScene.materials[0];

    auto uploadIfValid = [&](const GltfTextureData& texData, const std::string& name) {
        if (texData.valid()) {
            auto gpuTex = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                ctx.graphicsQueue, texData.pixels,
                                                static_cast<uint32_t>(texData.width),
                                                static_cast<uint32_t>(texData.height));
            m_namedTextures[name] = gpuTex;
            std::cout << "[info] Uploaded GPU texture: " << name
                      << " (" << texData.width << "x" << texData.height << ")" << std::endl;
        }
    };

    uploadIfValid(mat.base_color_tex, "base_color_tex");
    uploadIfValid(mat.normal_tex, "normal_tex");
    uploadIfValid(mat.metallic_roughness_tex, "metallic_roughness_tex");
    uploadIfValid(mat.occlusion_tex, "occlusion_tex");
    uploadIfValid(mat.emissive_tex, "emissive_tex");
}

// --------------------------------------------------------------------------
// Cubemap upload (F16 RGBA data, 6 faces, N mip levels)
// --------------------------------------------------------------------------

GPUTexture SceneManager::uploadCubemapF16(VulkanContext& ctx, uint32_t faceSize,
                                           uint32_t mipCount,
                                           const std::vector<uint16_t>& data,
                                           VkPipelineStageFlags dstStage) {
    GPUTexture tex;
    tex.width = faceSize;
    tex.height = faceSize;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageInfo.extent = {faceSize, faceSize, 1};
    imageInfo.mipLevels = mipCount;
    imageInfo.arrayLayers = 6;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(ctx.allocator, &imageInfo, &imageAllocInfo,
                                     &tex.image, &tex.allocation, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "[warn] Failed to create cubemap image" << std::endl;
        return tex;
    }

    VkDeviceSize dataSize = data.size() * sizeof(uint16_t);
    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = dataSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo stagingAllocInfo = {};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    vmaCreateBuffer(ctx.allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, stagingAlloc, &mapped);
    memcpy(mapped, data.data(), dataSize);
    vmaUnmapMemory(ctx.allocator, stagingAlloc);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 6;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkDeviceSize offset = 0;
    for (uint32_t mip = 0; mip < mipCount; mip++) {
        uint32_t mipSize = std::max(1u, faceSize >> mip);
        VkDeviceSize faceBytes = static_cast<VkDeviceSize>(mipSize) * mipSize * 4 * sizeof(uint16_t);
        for (uint32_t face = 0; face < 6; face++) {
            VkBufferImageCopy region = {};
            region.bufferOffset = offset;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = mip;
            region.imageSubresource.baseArrayLayer = face;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {mipSize, mipSize, 1};
            vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            offset += faceBytes;
        }
    }

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;
    vkCreateImageView(ctx.device, &viewInfo, nullptr, &tex.imageView);

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipCount);
    vkCreateSampler(ctx.device, &samplerInfo, nullptr, &tex.sampler);

    return tex;
}

// --------------------------------------------------------------------------
// IBL asset loading
// --------------------------------------------------------------------------

void SceneManager::loadIBLAssets(VulkanContext& ctx, VkPipelineStageFlags dstStage) {
    namespace fs = std::filesystem;

    // Auto-detect IBL assets: prefer "pisa" then "neutral"
    std::string iblNames[] = {"pisa", "neutral"};
    std::string iblDir;
    for (auto& name : iblNames) {
        std::string testPath = "assets/ibl/" + name + "/manifest.json";
        if (fs::exists(testPath)) {
            iblDir = "assets/ibl/" + name;
            break;
        }
    }
    if (iblDir.empty()) {
        std::cout << "[info] No IBL assets found, skipping IBL loading" << std::endl;
        return;
    }

    std::string manifestPath = iblDir + "/manifest.json";
    std::ifstream manifestFile(manifestPath);
    if (!manifestFile.is_open()) return;
    std::stringstream ss;
    ss << manifestFile.rdbuf();
    std::string manifestContent = ss.str();

    auto extractNestedInt = [&](const std::string& json, const std::string& section,
                                const std::string& key) -> int {
        auto secPos = json.find("\"" + section + "\"");
        if (secPos == std::string::npos) return 0;
        auto bracePos = json.find("{", secPos);
        if (bracePos == std::string::npos) return 0;
        int depth = 1;
        size_t endPos = bracePos + 1;
        while (endPos < json.size() && depth > 0) {
            if (json[endPos] == '{') depth++;
            else if (json[endPos] == '}') depth--;
            endPos++;
        }
        std::string sub = json.substr(bracePos, endPos - bracePos);
        auto kpos = sub.find("\"" + key + "\"");
        if (kpos == std::string::npos) return 0;
        kpos = sub.find(":", kpos);
        if (kpos == std::string::npos) return 0;
        kpos++;
        while (kpos < sub.size() && (sub[kpos] == ' ' || sub[kpos] == '\t' ||
               sub[kpos] == '\n' || sub[kpos] == '\r')) kpos++;
        return std::stoi(sub.substr(kpos));
    };

    int specSize = extractNestedInt(manifestContent, "specular", "face_size");
    int specMips = extractNestedInt(manifestContent, "specular", "mip_count");
    int irrSize = extractNestedInt(manifestContent, "irradiance", "face_size");
    int brdfSize = extractNestedInt(manifestContent, "brdf_lut", "size");

    if (specSize <= 0 || specMips <= 0 || irrSize <= 0 || brdfSize <= 0) {
        std::cerr << "[warn] Invalid IBL manifest values" << std::endl;
        return;
    }

    std::cout << "[info] Loading IBL assets (spec=" << specSize << "x" << specSize
              << " mips=" << specMips << ", irr=" << irrSize << "x" << irrSize
              << ", brdf=" << brdfSize << "x" << brdfSize << ")" << std::endl;

    // Specular cubemap
    {
        std::string specPath = iblDir + "/specular.bin";
        std::ifstream specFile(specPath, std::ios::binary | std::ios::ate);
        if (specFile.is_open()) {
            size_t fileSize = specFile.tellg();
            specFile.seekg(0);
            std::vector<uint16_t> specData(fileSize / sizeof(uint16_t));
            specFile.read(reinterpret_cast<char*>(specData.data()), fileSize);
            auto specTex = uploadCubemapF16(ctx, static_cast<uint32_t>(specSize),
                                             static_cast<uint32_t>(specMips), specData, dstStage);
            if (specTex.image != VK_NULL_HANDLE) {
                m_iblTextures["env_specular"] = specTex;
                std::cout << "[info] Loaded IBL specular cubemap" << std::endl;
            }
        }
    }

    // Irradiance cubemap
    {
        std::string irrPath = iblDir + "/irradiance.bin";
        std::ifstream irrFile(irrPath, std::ios::binary | std::ios::ate);
        if (irrFile.is_open()) {
            size_t fileSize = irrFile.tellg();
            irrFile.seekg(0);
            std::vector<uint16_t> irrData(fileSize / sizeof(uint16_t));
            irrFile.read(reinterpret_cast<char*>(irrData.data()), fileSize);
            auto irrTex = uploadCubemapF16(ctx, static_cast<uint32_t>(irrSize), 1, irrData, dstStage);
            if (irrTex.image != VK_NULL_HANDLE) {
                m_iblTextures["env_irradiance"] = irrTex;
                std::cout << "[info] Loaded IBL irradiance cubemap" << std::endl;
            }
        }
    }

    // BRDF LUT (2D texture, RG16F padded to RGBA16F)
    {
        std::string brdfPath = iblDir + "/brdf_lut.bin";
        std::ifstream brdfFile(brdfPath, std::ios::binary | std::ios::ate);
        if (brdfFile.is_open()) {
            size_t fileSize = brdfFile.tellg();
            brdfFile.seekg(0);
            std::vector<uint16_t> rawData(fileSize / sizeof(uint16_t));
            brdfFile.read(reinterpret_cast<char*>(rawData.data()), fileSize);

            size_t totalPixels = static_cast<size_t>(brdfSize) * brdfSize;
            std::vector<uint16_t> rgbaData;
            if (rawData.size() == totalPixels * 2) {
                rgbaData.resize(totalPixels * 4);
                for (size_t p = 0; p < totalPixels; p++) {
                    rgbaData[p * 4 + 0] = rawData[p * 2 + 0];
                    rgbaData[p * 4 + 1] = rawData[p * 2 + 1];
                    rgbaData[p * 4 + 2] = 0;
                    rgbaData[p * 4 + 3] = 0x3C00; // 1.0 in half-float
                }
            } else {
                rgbaData = rawData;
            }

            GPUTexture brdfTex;
            brdfTex.width = static_cast<uint32_t>(brdfSize);
            brdfTex.height = static_cast<uint32_t>(brdfSize);

            VkDeviceSize imgSize = rgbaData.size() * sizeof(uint16_t);
            VkBuffer stagBuf;
            VmaAllocation stagAlloc;
            VkBufferCreateInfo stagInfo = {};
            stagInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagInfo.size = imgSize;
            stagInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagAllocInfo = {};
            stagAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &stagInfo, &stagAllocInfo, &stagBuf, &stagAlloc, nullptr);

            void* mapped;
            vmaMapMemory(ctx.allocator, stagAlloc, &mapped);
            memcpy(mapped, rgbaData.data(), imgSize);
            vmaUnmapMemory(ctx.allocator, stagAlloc);

            VkImageCreateInfo imgInfo = {};
            imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imgInfo.imageType = VK_IMAGE_TYPE_2D;
            imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            imgInfo.extent = {static_cast<uint32_t>(brdfSize), static_cast<uint32_t>(brdfSize), 1};
            imgInfo.mipLevels = 1;
            imgInfo.arrayLayers = 1;
            imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo imgAllocInfo = {};
            imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            vmaCreateImage(ctx.allocator, &imgInfo, &imgAllocInfo, &brdfTex.image, &brdfTex.allocation, nullptr);

            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = brdfTex.image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy region = {};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {static_cast<uint32_t>(brdfSize), static_cast<uint32_t>(brdfSize), 1};
            vkCmdCopyBufferToImage(cmd, stagBuf, brdfTex.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            ctx.endSingleTimeCommands(cmd);
            vmaDestroyBuffer(ctx.allocator, stagBuf, stagAlloc);

            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = brdfTex.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            vkCreateImageView(ctx.device, &viewInfo, nullptr, &brdfTex.imageView);

            VkSamplerCreateInfo samplerInfo = {};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            vkCreateSampler(ctx.device, &samplerInfo, nullptr, &brdfTex.sampler);

            m_iblTextures["brdf_lut"] = brdfTex;
            std::cout << "[info] Loaded IBL BRDF LUT" << std::endl;
        }
    }

    std::cout << "[info] IBL loading complete: " << m_iblTextures.size() << " texture(s)" << std::endl;
}

// --------------------------------------------------------------------------
// Get texture for a binding name, with fallbacks
// --------------------------------------------------------------------------

GPUTexture& SceneManager::getTextureForBinding(const std::string& name) {
    // Check IBL textures first
    auto iblIt = m_iblTextures.find(name);
    if (iblIt != m_iblTextures.end()) return iblIt->second;

    // Check named textures
    auto it = m_namedTextures.find(name);
    if (it != m_namedTextures.end()) return it->second;

    // For "albedo_tex" used by pbr_surface, try "base_color_tex" as well
    if (name == "albedo_tex") {
        it = m_namedTextures.find("base_color_tex");
        if (it != m_namedTextures.end()) return it->second;
    }

    // Use semantically correct defaults for missing textures
    if (name == "emissive_tex") return m_defaultBlackTexture;
    if (name == "normal_tex") return m_defaultNormalTexture;

    return m_defaultWhiteTexture;
}

// --------------------------------------------------------------------------
// Compute auto-camera from vertex bounds
// --------------------------------------------------------------------------

void SceneManager::computeAutoCamera(const std::vector<Vertex>& vertices) {
    if (vertices.empty()) return;

    glm::vec3 bboxMin(std::numeric_limits<float>::max());
    glm::vec3 bboxMax(std::numeric_limits<float>::lowest());
    for (auto& v : vertices) {
        bboxMin = glm::min(bboxMin, v.position);
        bboxMax = glm::max(bboxMax, v.position);
    }
    glm::vec3 center = (bboxMin + bboxMax) * 0.5f;
    glm::vec3 extent = bboxMax - bboxMin;

    // Determine camera direction from first node's transform (matches raster renderer)
    glm::vec3 camDir(0.0f, 0.0f, 1.0f);
    glm::vec3 camUp(0.0f, 1.0f, 0.0f);

    if (m_hasGltfScene) {
        auto drawItems = flattenScene(m_gltfScene);
        if (!drawItems.empty()) {
            glm::mat4 world = drawItems[0].worldTransform;
            glm::mat3 upper3x3(world);
            glm::vec3 row0 = glm::vec3(upper3x3[0]);
            glm::vec3 row1 = glm::vec3(upper3x3[1]);
            glm::vec3 row2 = glm::vec3(upper3x3[2]);
            float n0 = glm::length(row0), n1 = glm::length(row1), n2 = glm::length(row2);
            if (n0 > 1e-8f) upper3x3[0] /= n0;
            if (n1 > 1e-8f) upper3x3[1] /= n1;
            if (n2 > 1e-8f) upper3x3[2] /= n2;

            // Blender front (-Y local) and up (+Z local) in world space
            glm::mat3 Rt = glm::transpose(upper3x3);
            glm::vec3 front = Rt * glm::vec3(0.0f, -1.0f, 0.0f);
            glm::vec3 upW = Rt * glm::vec3(0.0f, 0.0f, 1.0f);

            float fl = glm::length(front);
            float ul = glm::length(upW);
            if (fl > 0.5f && ul > 0.5f) {
                glm::vec3 candidateDir = front / fl;
                glm::vec3 candidateUp = upW / ul;
                if (std::abs(candidateDir.y) < 0.9f) {
                    camDir = candidateDir;
                    camUp = candidateUp;
                    if (std::abs(glm::dot(camDir, camUp)) > 0.9f) {
                        camUp = glm::vec3(0.0f, 1.0f, 0.0f);
                    }
                }
            }
        }
    }

    // Compute perpendicular extent for distance calculation
    glm::vec3 viewRight = glm::cross(-camDir, camUp);
    float vrLen = glm::length(viewRight);
    if (vrLen < 1e-6f) {
        camUp = glm::vec3(0.0f, 0.0f, 1.0f);
        viewRight = glm::cross(-camDir, camUp);
        vrLen = glm::length(viewRight);
    }
    viewRight /= vrLen;
    glm::vec3 viewUp = glm::normalize(glm::cross(viewRight, -camDir));

    float projRight = std::abs(viewRight.x) * extent.x + std::abs(viewRight.y) * extent.y + std::abs(viewRight.z) * extent.z;
    float projUp = std::abs(viewUp.x) * extent.x + std::abs(viewUp.y) * extent.y + std::abs(viewUp.z) * extent.z;
    float maxPerpExtent = glm::max(projRight, projUp);

    float fovY = glm::radians(45.0f);
    float distance = (maxPerpExtent / 2.0f) / std::tan(fovY / 2.0f) * 1.1f;

    float elevRad = glm::radians(5.0f);
    glm::vec3 eye = center + distance * camDir + distance * std::sin(elevRad) * viewUp;

    m_autoEye = eye;
    m_autoTarget = center;
    m_autoUp = camUp;
    m_autoFar = glm::max(distance * 3.0f, 100.0f);
    m_hasSceneBounds = true;

    std::cout << "[info] Auto-camera: eye=(" << m_autoEye.x << "," << m_autoEye.y << "," << m_autoEye.z
              << ") target=(" << m_autoTarget.x << "," << m_autoTarget.y << "," << m_autoTarget.z
              << ") distance=" << distance << std::endl;
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void SceneManager::cleanup(VulkanContext& ctx) {
    // Destroy named textures (track already-destroyed to avoid double-free)
    std::vector<VkImage> destroyed;
    for (auto& [name, tex] : m_namedTextures) {
        if (tex.image != VK_NULL_HANDLE) {
            bool alreadyDestroyed = false;
            for (auto img : destroyed) {
                if (img == tex.image) { alreadyDestroyed = true; break; }
            }
            if (!alreadyDestroyed) {
                destroyed.push_back(tex.image);
                Scene::destroyTexture(ctx.allocator, ctx.device, tex);
            }
        }
    }
    m_namedTextures.clear();

    for (auto& [name, tex] : m_iblTextures) {
        if (tex.image != VK_NULL_HANDLE) {
            Scene::destroyTexture(ctx.allocator, ctx.device, tex);
        }
    }
    m_iblTextures.clear();

    Scene::destroyTexture(ctx.allocator, ctx.device, m_defaultWhiteTexture);
    Scene::destroyTexture(ctx.allocator, ctx.device, m_defaultBlackTexture);
    Scene::destroyTexture(ctx.allocator, ctx.device, m_defaultNormalTexture);

    Scene::destroyMesh(ctx.allocator, m_mesh);
}
