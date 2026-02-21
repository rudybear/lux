#include "mesh_renderer.h"
#include "spv_loader.h"
#include "camera.h"
#include "reflected_pipeline.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <filesystem>

// --------------------------------------------------------------------------
// Offscreen render target creation
// --------------------------------------------------------------------------

void MeshRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {m_width, m_height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &m_colorImage, &m_colorAlloc, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_colorImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_colorView);

    // Depth image
    VkImageCreateInfo depthInfo = imageInfo;
    depthInfo.format = VK_FORMAT_D32_SFLOAT;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    vmaCreateImage(ctx.allocator, &depthInfo, &allocInfo,
                   &m_depthImage, &m_depthAlloc, nullptr);

    VkImageViewCreateInfo depthViewInfo = viewInfo;
    depthViewInfo.image = m_depthImage;
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    vkCreateImageView(ctx.device, &depthViewInfo, nullptr, &m_depthView);
}

// --------------------------------------------------------------------------
// Render pass creation
// --------------------------------------------------------------------------

void MeshRenderer::createRenderPass(VulkanContext& ctx) {
    VkAttachmentDescription colorAtt = {};
    colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentDescription depthAtt = {};
    depthAtt.format = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription attachments[] = {colorAtt, depthAtt};

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &m_renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create mesh shader render pass");
    }
}

// --------------------------------------------------------------------------
// Framebuffer creation
// --------------------------------------------------------------------------

void MeshRenderer::createFramebuffer(VulkanContext& ctx) {
    VkImageView views[] = {m_colorView, m_depthView};

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = m_renderPass;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = views;
    fbInfo.width = m_width;
    fbInfo.height = m_height;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &m_framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create mesh shader framebuffer");
    }
}

// --------------------------------------------------------------------------
// Vertex data upload (SoA storage buffers, same pattern as RT renderer)
// --------------------------------------------------------------------------

void MeshRenderer::uploadVertexData(VulkanContext& ctx) {
    const auto& vertices = m_scene->getVertices();
    VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Positions: vec4 per vertex (x,y,z,1)
    std::vector<glm::vec4> posVec4(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        posVec4[i] = glm::vec4(vertices[i].position, 1.0f);
    }
    m_positionsSize = posVec4.size() * sizeof(glm::vec4);
    Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                        posVec4.data(), m_positionsSize, ssboUsage,
                        m_positionsBuffer, m_positionsAlloc);

    // Normals: vec4 per vertex (nx,ny,nz,0)
    std::vector<glm::vec4> norVec4(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        norVec4[i] = glm::vec4(vertices[i].normal, 0.0f);
    }
    m_normalsSize = norVec4.size() * sizeof(glm::vec4);
    Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                        norVec4.data(), m_normalsSize, ssboUsage,
                        m_normalsBuffer, m_normalsAlloc);

    // Tex coords: vec2 per vertex
    std::vector<glm::vec2> uvVec2(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        uvVec2[i] = vertices[i].uv;
    }
    m_texCoordsSize = uvVec2.size() * sizeof(glm::vec2);
    Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                        uvVec2.data(), m_texCoordsSize, ssboUsage,
                        m_texCoordsBuffer, m_texCoordsAlloc);

    // Tangents: vec4 per vertex (tx,ty,tz,handedness)
    // Extract from GltfVertex data if available, otherwise default to (1,0,0,1)
    std::vector<glm::vec4> tanVec4(vertices.size(), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    if (m_scene->hasGltfScene()) {
        const auto& gltfScene = m_scene->getGltfScene();
        auto drawItems = flattenScene(const_cast<GltfScene&>(gltfScene));
        size_t vertexOffset = 0;
        for (auto& item : drawItems) {
            auto& gmesh = gltfScene.meshes[item.meshIndex];
            glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(item.worldTransform)));
            for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                if (vertexOffset + i < tanVec4.size()) {
                    glm::vec3 tan3 = glm::normalize(normalMat * glm::vec3(gmesh.vertices[i].tangent));
                    tanVec4[vertexOffset + i] = glm::vec4(tan3, gmesh.vertices[i].tangent.w);
                }
            }
            vertexOffset += gmesh.vertices.size();
        }
    }
    m_tangentsSize = tanVec4.size() * sizeof(glm::vec4);
    Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                        tanVec4.data(), m_tangentsSize, ssboUsage,
                        m_tangentsBuffer, m_tangentsAlloc);
    std::cout << "[mesh] Created SoA storage buffers: positions=" << m_positionsSize
              << "B, normals=" << m_normalsSize << "B, texCoords=" << m_texCoordsSize
              << "B, tangents=" << m_tangentsSize
              << "B" << std::endl;
}

// --------------------------------------------------------------------------
// Meshlet data build and upload
// --------------------------------------------------------------------------

void MeshRenderer::uploadMeshletData(VulkanContext& ctx) {
    const auto& indices = m_scene->getIndices();
    const auto& vertices = m_scene->getVertices();

    // Get optimal limits from hardware
    uint32_t maxVerts = 64;
    uint32_t maxPrims = 124;
    if (ctx.meshShaderSupported) {
        // Use conservative defaults that work everywhere
        uint32_t hwMaxVerts = ctx.meshShaderProperties.maxMeshOutputVertices;
        uint32_t hwMaxPrims = ctx.meshShaderProperties.maxMeshOutputPrimitives;
        if (hwMaxVerts > 0) maxVerts = std::min(hwMaxVerts, 64u);
        if (hwMaxPrims > 0) maxPrims = std::min(hwMaxPrims, 124u);
    }

    // Build meshlets
    auto buildResult = buildMeshlets(
        indices.data(), static_cast<uint32_t>(indices.size()),
        static_cast<uint32_t>(vertices.size()),
        maxVerts, maxPrims);

    m_totalMeshlets = static_cast<uint32_t>(buildResult.meshlets.size());
    std::cout << "[mesh] Built " << m_totalMeshlets << " meshlets"
              << " (max_verts=" << maxVerts << ", max_prims=" << maxPrims << ")" << std::endl;

    VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Upload meshlet descriptors
    m_meshletDescSize = buildResult.meshlets.size() * sizeof(MeshletDescriptor);
    Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                        buildResult.meshlets.data(), m_meshletDescSize, ssboUsage,
                        m_meshletDescBuffer, m_meshletDescAlloc);

    // Upload meshlet vertex indices
    m_meshletVertSize = buildResult.meshletVertices.size() * sizeof(uint32_t);
    if (m_meshletVertSize > 0) {
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            buildResult.meshletVertices.data(), m_meshletVertSize, ssboUsage,
                            m_meshletVertBuffer, m_meshletVertAlloc);
    } else {
        // Create a minimal buffer for descriptor binding
        uint32_t dummy = 0;
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            &dummy, sizeof(uint32_t), ssboUsage,
                            m_meshletVertBuffer, m_meshletVertAlloc);
        m_meshletVertSize = sizeof(uint32_t);
    }

    // Upload meshlet triangle indices
    m_meshletTriSize = buildResult.meshletTriangles.size() * sizeof(uint32_t);
    if (m_meshletTriSize > 0) {
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            buildResult.meshletTriangles.data(), m_meshletTriSize, ssboUsage,
                            m_meshletTriBuffer, m_meshletTriAlloc);
    } else {
        // Create a minimal buffer for descriptor binding
        uint32_t dummy = 0;
        Scene::uploadBuffer(ctx.allocator, ctx.device, ctx.commandPool, ctx.graphicsQueue,
                            &dummy, sizeof(uint32_t), ssboUsage,
                            m_meshletTriBuffer, m_meshletTriAlloc);
        m_meshletTriSize = sizeof(uint32_t);
    }
}

// --------------------------------------------------------------------------
// Pipeline creation (mesh shader + fragment, no vertex input)
// --------------------------------------------------------------------------

void MeshRenderer::createPipeline(VulkanContext& ctx) {
    // Create descriptor set layouts from reflection data
    std::vector<ReflectionData> stages = {m_meshReflection, m_fragReflection};
    m_setLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);

    // Pipeline layout from reflection
    m_pipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, m_setLayouts, stages);

    // Shader stages: mesh + fragment (no vertex!)
    VkPipelineShaderStageCreateInfo shaderStages[2] = {};

    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    shaderStages[0].module = m_meshModule;
    shaderStages[0].pName = "main";

    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = m_fragModule;
    shaderStages[1].pName = "main";

    // Empty vertex input state (mesh shaders don't use vertex input)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport/scissor
    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_width);
    viewport.height = static_cast<float>(m_height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {m_width, m_height};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // Graphics pipeline create info
    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = shaderStages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = m_pipelineLayout;
    pipeInfo.renderPass = m_renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &m_pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create mesh shader pipeline");
    }

    std::cout << "[mesh] Pipeline created (reflection-driven)" << std::endl;
}

// --------------------------------------------------------------------------
// Reflection-driven descriptor set creation and update
// --------------------------------------------------------------------------

void MeshRenderer::setupDescriptors(VulkanContext& ctx) {
    // Get merged bindings from reflection data for pool sizing
    std::vector<ReflectionData> stages = {m_meshReflection, m_fragReflection};
    auto mergedBindings = getMergedBindings(stages);

    // Count pool sizes from reflection data
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "storage_image") vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else if (b.type == "storage_buffer") vkType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        typeCounts[vkType]++;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    uint32_t maxSets = static_cast<uint32_t>(m_setLayouts.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create mesh shader descriptor pool");
    }

    // Allocate descriptor sets
    for (auto& [setIdx, layout] : m_setLayouts) {
        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = m_descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layout;

        VkDescriptorSet set;
        if (vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate mesh shader descriptor set");
        }
        m_descriptorSets[setIdx] = set;
    }

    // Write descriptor sets based on reflection binding names
    struct DescWriteInfo {
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorImageInfo imageInfo;
    };
    std::vector<DescWriteInfo> writeInfos(mergedBindings.size());
    std::vector<VkWriteDescriptorSet> writes;

    for (size_t i = 0; i < mergedBindings.size(); i++) {
        auto& b = mergedBindings[i];
        auto setIt = m_descriptorSets.find(b.set);
        if (setIt == m_descriptorSets.end()) continue;

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = setIt->second;
        w.dstBinding = static_cast<uint32_t>(b.binding);
        w.descriptorCount = 1;

        if (b.type == "uniform_buffer") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            if (b.name == "MVP") {
                writeInfos[i].bufferInfo = {m_mvpBuffer, 0,
                    static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            } else if (b.name == "Light") {
                writeInfos[i].bufferInfo = {m_lightBuffer, 0,
                    static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
            } else {
                // Unknown UBO name - use MVP as fallback
                writeInfos[i].bufferInfo = {m_mvpBuffer, 0,
                    static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            }
            w.pBufferInfo = &writeInfos[i].bufferInfo;
        } else if (b.type == "storage_buffer") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceSize bufSize = 0;
            if (b.name == "meshlet_descriptors") { buf = m_meshletDescBuffer; bufSize = m_meshletDescSize; }
            else if (b.name == "meshlet_vertices") { buf = m_meshletVertBuffer; bufSize = m_meshletVertSize; }
            else if (b.name == "meshlet_triangles") { buf = m_meshletTriBuffer; bufSize = m_meshletTriSize; }
            else if (b.name == "positions") { buf = m_positionsBuffer; bufSize = m_positionsSize; }
            else if (b.name == "normals") { buf = m_normalsBuffer; bufSize = m_normalsSize; }
            else if (b.name == "tex_coords") { buf = m_texCoordsBuffer; bufSize = m_texCoordsSize; }
            else if (b.name == "tangents") { buf = m_tangentsBuffer; bufSize = m_tangentsSize; }
            if (buf == VK_NULL_HANDLE) {
                std::cout << "[warn] Unknown storage buffer name: " << b.name << std::endl;
                continue;
            }
            writeInfos[i].bufferInfo = {buf, 0, bufSize};
            w.pBufferInfo = &writeInfos[i].bufferInfo;
        } else if (b.type == "sampler") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            // Check IBL textures first, then regular textures
            auto& iblTextures = m_scene->getIBLTextures();
            auto iblIt = iblTextures.find(b.name);
            if (iblIt != iblTextures.end()) {
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.sampler = iblIt->second.sampler;
            } else {
                auto& tex = m_scene->getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.sampler = tex.sampler;
            }
            w.pImageInfo = &writeInfos[i].imageInfo;
        } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            // Check IBL textures first, then regular textures
            auto& iblTextures = m_scene->getIBLTextures();
            auto iblIt = iblTextures.find(b.name);
            if (iblIt != iblTextures.end()) {
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = iblIt->second.imageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } else {
                auto& tex = m_scene->getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = tex.imageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
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

    std::cout << "[mesh] Reflection-driven descriptors created: "
              << m_setLayouts.size() << " set(s), "
              << mergedBindings.size() << " binding(s)" << std::endl;
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

void MeshRenderer::init(VulkanContext& ctx, SceneManager& scene,
                        const std::string& pipelineBase, uint32_t width, uint32_t height) {
    if (!ctx.supportsMeshShader()) {
        throw std::runtime_error("Mesh shader pipeline requires VK_EXT_mesh_shader support");
    }

    m_scene = &scene;
    m_pipelineBase = pipelineBase;
    m_width = width;
    m_height = height;

    std::cout << "[mesh] Initializing mesh shader renderer (" << width << "x" << height << ")" << std::endl;

    // Load reflection JSON
    namespace fs = std::filesystem;
    std::string meshJsonPath = pipelineBase + ".mesh.json";
    std::string fragJsonPath = pipelineBase + ".frag.json";

    if (fs::exists(meshJsonPath)) {
        std::cout << "[info] Loading reflection JSON: " << meshJsonPath << std::endl;
        m_meshReflection = parseReflectionJson(meshJsonPath);
    }
    if (fs::exists(fragJsonPath)) {
        std::cout << "[info] Loading reflection JSON: " << fragJsonPath << std::endl;
        m_fragReflection = parseReflectionJson(fragJsonPath);
    }

    // Load shader modules
    auto meshCode = SpvLoader::loadSPIRV(pipelineBase + ".mesh.spv");
    m_meshModule = SpvLoader::createShaderModule(ctx.device, meshCode);

    auto fragCode = SpvLoader::loadSPIRV(pipelineBase + ".frag.spv");
    m_fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

    // Upload vertex data as SoA storage buffers
    uploadVertexData(ctx);

    // Build meshlets and upload
    uploadMeshletData(ctx);

    // Create uniform buffers
    float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);

    // MVP buffer (3 x mat4 = 192 bytes)
    struct MVPData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };
    MVPData mvpData;
    if (m_scene->hasSceneBounds()) {
        mvpData.model = glm::mat4(1.0f);
        mvpData.view = glm::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp());
        mvpData.projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, m_scene->getAutoFar());
        mvpData.projection[1][1] *= -1.0f; // Vulkan Y-flip
    } else {
        auto cam = Camera::getDefaultMatrices(aspect);
        mvpData = {cam.model, cam.view, cam.projection};
    }

    VkBufferCreateInfo mvpBufInfo = {};
    mvpBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    mvpBufInfo.size = sizeof(MVPData);
    mvpBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo mvpAllocInfo = {};
    mvpAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &mvpBufInfo, &mvpAllocInfo,
                    &m_mvpBuffer, &m_mvpAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, m_mvpAlloc, &mapped);
    memcpy(mapped, &mvpData, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, m_mvpAlloc);

    // Light buffer (32 bytes: vec3 light_dir padded + vec3 view_pos padded)
    struct LightData {
        glm::vec3 lightDir;
        float _pad0;
        glm::vec3 viewPos;
        float _pad1;
    };

    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    glm::vec3 viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
    LightData lightData = {lightDir, 0.0f, viewPos, 0.0f};

    VkBufferCreateInfo lightBufInfo = {};
    lightBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    lightBufInfo.size = sizeof(LightData);
    lightBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo lightAllocInfo = {};
    lightAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &lightBufInfo, &lightAllocInfo,
                    &m_lightBuffer, &m_lightAlloc, nullptr);

    vmaMapMemory(ctx.allocator, m_lightAlloc, &mapped);
    memcpy(mapped, &lightData, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, m_lightAlloc);

    // Create offscreen render target
    createOffscreenTarget(ctx);
    createRenderPass(ctx);
    createFramebuffer(ctx);

    // Create pipeline and descriptors
    createPipeline(ctx);
    setupDescriptors(ctx);

    std::cout << "[mesh] Mesh shader renderer initialized: " << pipelineBase << std::endl;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void MeshRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkClearValue clearValues[2];
    clearValues[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = m_renderPass;
    rpBegin.framebuffer = m_framebuffer;
    rpBegin.renderArea = {{0, 0}, {m_width, m_height}};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // Bind descriptor sets in order
    int maxSet = -1;
    for (auto& [idx, _] : m_descriptorSets) {
        maxSet = std::max(maxSet, idx);
    }
    for (int i = 0; i <= maxSet; i++) {
        auto it = m_descriptorSets.find(i);
        if (it != m_descriptorSets.end()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_pipelineLayout, static_cast<uint32_t>(i),
                                    1, &it->second, 0, nullptr);
        }
    }

    // Draw mesh tasks - one workgroup per meshlet
    ctx.pfnCmdDrawMeshTasksEXT(cmd, m_totalMeshlets, 1, 1);

    vkCmdEndRenderPass(cmd);
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Orbit camera: update MVP + Light uniform buffers per frame
// --------------------------------------------------------------------------

void MeshRenderer::updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                                glm::vec3 up, float fovY, float aspect,
                                float nearPlane, float farPlane) {
    if (!m_mvpBuffer || !m_lightBuffer) return;

    // Update MVP
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    mvp.model = glm::mat4(1.0f);
    mvp.view = Camera::lookAt(eye, target, up);
    mvp.projection = Camera::perspective(fovY, aspect, nearPlane, farPlane);

    void* mapped;
    vmaMapMemory(ctx.allocator, m_mvpAlloc, &mapped);
    memcpy(mapped, &mvp, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, m_mvpAlloc);

    // Update Light uniform (preserve light_dir, update view_pos)
    struct LightData {
        glm::vec3 lightDir; float _pad0;
        glm::vec3 viewPos; float _pad1;
    };
    LightData light;
    vmaMapMemory(ctx.allocator, m_lightAlloc, &mapped);
    memcpy(&light, mapped, sizeof(LightData));
    light.viewPos = eye;
    memcpy(mapped, &light, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, m_lightAlloc);
}

// --------------------------------------------------------------------------
// Blit to swapchain (same pattern as raster renderer)
// --------------------------------------------------------------------------

void MeshRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                    VkImage swapImage, VkExtent2D extent) {
    // Transition swapchain image: UNDEFINED -> TRANSFER_DST
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
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(m_width),
                                static_cast<int32_t>(m_height), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        m_colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_LINEAR);

    // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
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
// Interactive mode: render to swapchain
// --------------------------------------------------------------------------

void MeshRenderer::renderToSwapchain(VulkanContext& ctx,
                                     VkImage swapImage, VkImageView swapView,
                                     VkFormat swapFormat, VkExtent2D extent,
                                     VkSemaphore waitSem, VkSemaphore signalSem,
                                     VkFence fence) {
    (void)swapView;
    (void)swapFormat;

    // Record all commands (offscreen render + blit to swapchain) into a single
    // command buffer and submit with proper semaphore/fence synchronization.
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // --- Offscreen render pass ---
    VkClearValue clearValues[2];
    clearValues[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = m_renderPass;
    rpBegin.framebuffer = m_framebuffer;
    rpBegin.renderArea = {{0, 0}, {m_width, m_height}};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    // Bind descriptor sets in order
    int maxSet = -1;
    for (auto& [idx, _] : m_descriptorSets) {
        maxSet = std::max(maxSet, idx);
    }
    for (int i = 0; i <= maxSet; i++) {
        auto it = m_descriptorSets.find(i);
        if (it != m_descriptorSets.end()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_pipelineLayout, static_cast<uint32_t>(i),
                                    1, &it->second, 0, nullptr);
        }
    }

    // Draw mesh tasks - one workgroup per meshlet
    ctx.pfnCmdDrawMeshTasksEXT(cmd, m_totalMeshlets, 1, 1);

    vkCmdEndRenderPass(cmd);

    // --- Blit offscreen -> swapchain ---
    blitToSwapchain(ctx, cmd, swapImage, extent);

    vkEndCommandBuffer(cmd);

    // Submit with proper semaphore/fence synchronization
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSem;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSem;

    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fence);
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MeshRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    if (m_pipeline) vkDestroyPipeline(ctx.device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(ctx.device, m_pipelineLayout, nullptr);

    if (m_descriptorPool) vkDestroyDescriptorPool(ctx.device, m_descriptorPool, nullptr);
    for (auto& [_, layout] : m_setLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    m_setLayouts.clear();
    m_descriptorSets.clear();

    if (m_meshModule) vkDestroyShaderModule(ctx.device, m_meshModule, nullptr);
    if (m_fragModule) vkDestroyShaderModule(ctx.device, m_fragModule, nullptr);

    if (m_framebuffer) vkDestroyFramebuffer(ctx.device, m_framebuffer, nullptr);
    if (m_renderPass) vkDestroyRenderPass(ctx.device, m_renderPass, nullptr);

    if (m_colorView) vkDestroyImageView(ctx.device, m_colorView, nullptr);
    if (m_colorImage) vmaDestroyImage(ctx.allocator, m_colorImage, m_colorAlloc);
    if (m_depthView) vkDestroyImageView(ctx.device, m_depthView, nullptr);
    if (m_depthImage) vmaDestroyImage(ctx.allocator, m_depthImage, m_depthAlloc);

    if (m_meshletDescBuffer) vmaDestroyBuffer(ctx.allocator, m_meshletDescBuffer, m_meshletDescAlloc);
    if (m_meshletVertBuffer) vmaDestroyBuffer(ctx.allocator, m_meshletVertBuffer, m_meshletVertAlloc);
    if (m_meshletTriBuffer) vmaDestroyBuffer(ctx.allocator, m_meshletTriBuffer, m_meshletTriAlloc);
    if (m_positionsBuffer) vmaDestroyBuffer(ctx.allocator, m_positionsBuffer, m_positionsAlloc);
    if (m_normalsBuffer) vmaDestroyBuffer(ctx.allocator, m_normalsBuffer, m_normalsAlloc);
    if (m_texCoordsBuffer) vmaDestroyBuffer(ctx.allocator, m_texCoordsBuffer, m_texCoordsAlloc);
    if (m_tangentsBuffer) vmaDestroyBuffer(ctx.allocator, m_tangentsBuffer, m_tangentsAlloc);
    if (m_mvpBuffer) vmaDestroyBuffer(ctx.allocator, m_mvpBuffer, m_mvpAlloc);
    if (m_lightBuffer) vmaDestroyBuffer(ctx.allocator, m_lightBuffer, m_lightAlloc);
}
