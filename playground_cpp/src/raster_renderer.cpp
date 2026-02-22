#include "raster_renderer.h"
#include "spv_loader.h"
#include "camera.h"
#include "gltf_loader.h"
#include "reflected_pipeline.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <array>
#include <fstream>
#include <sstream>
#include <filesystem>

// --------------------------------------------------------------------------
// Offscreen render target creation
// --------------------------------------------------------------------------

void RasterRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color attachment
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {renderWidth, renderHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &offscreenImage, &offscreenAllocation, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = offscreenImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &offscreenImageView);

    // Depth attachment (raster scenes with 3D geometry)
    if (m_needsDepth) {
        VkImageCreateInfo depthInfo = imageInfo;
        depthInfo.format = VK_FORMAT_D32_SFLOAT;
        depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        vmaCreateImage(ctx.allocator, &depthInfo, &allocInfo,
                       &depthImage, &depthAllocation, nullptr);

        VkImageViewCreateInfo depthViewInfo = viewInfo;
        depthViewInfo.image = depthImage;
        depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
        depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        vkCreateImageView(ctx.device, &depthViewInfo, nullptr, &depthImageView);
    }
}

// --------------------------------------------------------------------------
// Render pass creation
// --------------------------------------------------------------------------

void RasterRenderer::createRenderPass(VulkanContext& ctx) {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> colorRefs;
    VkAttachmentReference depthRef = {};

    // Color attachment
    VkAttachmentDescription colorAtt = {};
    colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    attachments.push_back(colorAtt);

    VkAttachmentReference colorRef = {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs.push_back(colorRef);

    // Depth attachment (raster scenes with 3D geometry)
    if (m_needsDepth) {
        VkAttachmentDescription depthAtt = {};
        depthAtt.format = VK_FORMAT_D32_SFLOAT;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAtt);

        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
    if (m_needsDepth) {
        subpass.pDepthStencilAttachment = &depthRef;
    }

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
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

// --------------------------------------------------------------------------
// Framebuffer creation
// --------------------------------------------------------------------------

void RasterRenderer::createFramebuffer(VulkanContext& ctx) {
    std::vector<VkImageView> attachments = {offscreenImageView};
    if (m_needsDepth) {
        attachments.push_back(depthImageView);
    }

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbInfo.pAttachments = attachments.data();
    fbInfo.width = renderWidth;
    fbInfo.height = renderHeight;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer");
    }
}

// --------------------------------------------------------------------------
// Triangle pipeline
// --------------------------------------------------------------------------

void RasterRenderer::createPipelineTriangle(VulkanContext& ctx) {
    // Pipeline layout: no descriptor sets
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Vertex input: position (vec3, offset 0) + color (vec3, offset 12)
    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(TriangleVertex); // 24 bytes
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2] = {};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;

    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = 12;

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create triangle graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Fullscreen pipeline
// --------------------------------------------------------------------------

void RasterRenderer::createPipelineFullscreen(VulkanContext& ctx) {
    // Pipeline layout: no descriptor sets
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // No vertex input (uses gl_VertexIndex)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

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

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fullscreen graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// PBR resources: uniforms only (descriptors are now reflection-driven)
// --------------------------------------------------------------------------

void RasterRenderer::setupPBRResources(VulkanContext& ctx) {
    float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);

    // Create MVP uniform buffer (3 x mat4 = 192 bytes)
    struct MVPData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };
    MVPData mvpData;
    if (m_scene->hasSceneBounds()) {
        // Use auto-camera computed from scene bounds
        mvpData.model = glm::mat4(1.0f); // identity - transforms already baked into vertices
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
                    &mvpBuffer, &mvpAllocation, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, mvpAllocation, &mapped);
    memcpy(mapped, &mvpData, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, mvpAllocation);

    // Create Light uniform buffer (32 bytes: vec3 light_dir padded + vec3 view_pos padded)
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
                    &lightBuffer, &lightAllocation, nullptr);

    vmaMapMemory(ctx.allocator, lightAllocation, &mapped);
    memcpy(mapped, &lightData, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, lightAllocation);

    // Create Material uniform buffer (80 bytes, std140 layout)
    MaterialUBOData materialData{};
    if (m_scene && m_scene->hasGltfScene()) {
        const auto& mat = m_scene->getGltfScene().materials[0];
        materialData.baseColorFactor = mat.baseColor;
        materialData.metallicFactor = mat.metallic;
        materialData.roughnessFactor = mat.roughness;
        materialData.emissiveFactor = mat.emissive;
        materialData.emissiveStrength = mat.emissiveStrength;
        materialData.ior = mat.ior;
        materialData.clearcoatFactor = mat.clearcoatFactor;
        materialData.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
        materialData.sheenColorFactor = mat.sheenColorFactor;
        materialData.sheenRoughnessFactor = mat.sheenRoughnessFactor;
        materialData.transmissionFactor = mat.transmissionFactor;
    }

    VkBufferCreateInfo matBufInfo = {};
    matBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    matBufInfo.size = sizeof(MaterialUBOData);
    matBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo matAllocInfo = {};
    matAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &matBufInfo, &matAllocInfo,
                    &m_materialBuffer, &m_materialAllocation, nullptr);

    vmaMapMemory(ctx.allocator, m_materialAllocation, &mapped);
    memcpy(mapped, &materialData, sizeof(MaterialUBOData));
    vmaUnmapMemory(ctx.allocator, m_materialAllocation);
}

// --------------------------------------------------------------------------
// Reflection-driven descriptor set creation
// --------------------------------------------------------------------------

void RasterRenderer::setupReflectedDescriptors(VulkanContext& ctx) {
    // Use reflected_pipeline utilities to create descriptor set layouts from reflection JSON
    std::vector<ReflectionData> stages = {vertReflection, fragReflection};
    reflectedSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);

    // Get merged binding info for pool sizing and descriptor writes
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

    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    // Allocate descriptor sets
    for (auto& [setIdx, layout] : reflectedSetLayouts) {
        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layout;

        VkDescriptorSet set;
        vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &set);
        reflectedDescSets[setIdx] = set;
    }

    // Write descriptor sets based on reflection binding names
    // We need to keep descriptor info structs alive until vkUpdateDescriptorSets returns
    struct DescWriteInfo {
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorImageInfo imageInfo;
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
            // Match by name: "MVP" -> mvpBuffer, "Light" -> lightBuffer
            if (b.name == "MVP") {
                writeInfos[i].bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            } else if (b.name == "Light") {
                writeInfos[i].bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
            } else if (b.name == "Material") {
                writeInfos[i].bufferInfo = {m_materialBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : sizeof(MaterialUBOData))};
            } else {
                // Unknown UBO name - use MVP as fallback
                writeInfos[i].bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            }
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

    std::cout << "[info] Reflection-driven descriptors created: "
              << reflectedSetLayouts.size() << " set(s), "
              << mergedBindings.size() << " binding(s)" << std::endl;
}

// --------------------------------------------------------------------------
// PBR pipeline (reflection-driven layout)
// --------------------------------------------------------------------------

void RasterRenderer::createPipelinePBR(VulkanContext& ctx) {
    // Pipeline layout from reflection data
    pipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, reflectedSetLayouts,
        {vertReflection, fragReflection});

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Vertex input from reflection data (supports both 32-byte and 48-byte stride)
    auto reflectedInput = createReflectedVertexInput(vertReflection);

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &reflectedInput.binding;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(reflectedInput.attributes.size());
    vertexInput.pVertexAttributeDescriptions = reflectedInput.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

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

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create PBR graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

void RasterRenderer::init(VulkanContext& ctx, SceneManager& scene,
                          const std::string& pipelineBase, const std::string& renderPath,
                          uint32_t width, uint32_t height) {
    m_scene = &scene;
    m_renderPath = renderPath;
    m_pipelineBase = pipelineBase;
    renderWidth = width;
    renderHeight = height;

    const std::string& sceneSource = m_scene->getSceneSource();

    // Determine if depth buffer is needed based on scene
    m_needsDepth = (sceneSource == "sphere" || SceneManager::isGltfFile(sceneSource));

    // Upload scene resources
    if (sceneSource == "triangle") {
        // Generate triangle mesh
        std::vector<TriangleVertex> triVerts;
        Scene::generateTriangle(triVerts);

        VkDeviceSize bufSize = triVerts.size() * sizeof(TriangleVertex);
        VkBufferCreateInfo bufInfo = {};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = bufSize;
        bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        vmaCreateBuffer(ctx.allocator, &bufInfo, &allocInfo,
                        &triangleVB, &triangleVBAllocation, nullptr);

        void* mapped;
        vmaMapMemory(ctx.allocator, triangleVBAllocation, &mapped);
        memcpy(mapped, triVerts.data(), bufSize);
        vmaUnmapMemory(ctx.allocator, triangleVBAllocation);
    } else if (sceneSource != "fullscreen") {
        // For sphere and glTF scenes, setup PBR resources (uniforms)
        setupPBRResources(ctx);
    }

    // Create offscreen target (depends on depth requirement)
    createOffscreenTarget(ctx);
    createRenderPass(ctx);
    createFramebuffer(ctx);

    // Create pipeline from shader files
    namespace fs = std::filesystem;
    std::string fragPath = pipelineBase + ".frag.spv";

    if (renderPath == "fullscreen") {
        const auto& fullscreenSpv = SpvLoader::getFullscreenVertSPIRV();
        vertModule = SpvLoader::createShaderModule(ctx.device, fullscreenSpv);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);
        createPipelineFullscreen(ctx);
    } else if (renderPath == "raster") {
        std::string vertPath = pipelineBase + ".vert.spv";
        auto vertCode = SpvLoader::loadSPIRV(vertPath);
        vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

        std::string vertJsonPath = pipelineBase + ".vert.json";
        std::string fragJsonPath = pipelineBase + ".frag.json";

        if (fs::exists(vertJsonPath) && fs::exists(fragJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << vertJsonPath << std::endl;
            vertReflection = parseReflectionJson(vertJsonPath);
            std::cout << "[info] Loading reflection JSON: " << fragJsonPath << std::endl;
            fragReflection = parseReflectionJson(fragJsonPath);
        }

        if (sceneSource == "triangle") {
            createPipelineTriangle(ctx);
        } else {
            setupReflectedDescriptors(ctx);
            createPipelinePBR(ctx);
        }
    } else {
        throw std::runtime_error("Unsupported render path for raster renderer: " + renderPath);
    }

    // Bind scene resources to pipeline (push constants)
    {
        glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
        auto buildPushData = [&](const ReflectionData& refl) {
            for (auto& pc : refl.push_constants) {
                if (pc.size <= 0) continue;
                pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(pc.size));
                pushConstantData.resize(pushConstantSize, 0);
                pushConstantStageFlags |= (refl.stage == "vertex")
                    ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
                for (auto& f : pc.fields) {
                    if (f.name == "light_dir" && static_cast<uint32_t>(f.offset) + 12 <= pushConstantSize) {
                        memcpy(pushConstantData.data() + f.offset, &lightDir, sizeof(glm::vec3));
                    } else if (f.name == "view_pos" && static_cast<uint32_t>(f.offset) + 12 <= pushConstantSize) {
                        glm::vec3 eye = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
                        memcpy(pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                    }
                }
            }
        };
        buildPushData(vertReflection);
        buildPushData(fragReflection);
    }

    std::cout << "[info] Raster renderer initialized: " << pipelineBase
              << " (path=" << renderPath << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RasterRenderer::render(VulkanContext& ctx) {
    const std::string& sceneSource = m_scene->getSceneSource();

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Set clear values
    std::vector<VkClearValue> clearValues;
    VkClearValue colorClear = {};
    if (m_needsDepth) {
        colorClear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    } else {
        colorClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }
    clearValues.push_back(colorClear);

    if (m_needsDepth) {
        VkClearValue depthClear = {};
        depthClear.depthStencil = {1.0f, 0};
        clearValues.push_back(depthClear);
    }

    VkRenderPassBeginInfo rpBeginInfo = {};
    rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBeginInfo.renderPass = renderPass;
    rpBeginInfo.framebuffer = framebuffer;
    rpBeginInfo.renderArea.offset = {0, 0};
    rpBeginInfo.renderArea.extent = {renderWidth, renderHeight};
    rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBeginInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Push constants (if any)
    if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
        vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                           0, pushConstantSize, pushConstantData.data());
    }

    if (sceneSource == "triangle") {
        // Triangle scene: bind triangle vertex buffer and draw
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (sceneSource == "fullscreen" || m_renderPath == "fullscreen") {
        // Fullscreen scene: draw fullscreen triangle (no vertex buffer)
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else {
        // Sphere, glTF, or default PBR scene: bind reflection-driven descriptors
        const auto& mesh = m_scene->getMesh();

        // Bind descriptor sets in order
        int maxSet = -1;
        for (auto& [idx, _] : reflectedDescSets) {
            maxSet = std::max(maxSet, idx);
        }
        for (int i = 0; i <= maxSet; i++) {
            auto it = reflectedDescSets.find(i);
            if (it != reflectedDescSets.end()) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, static_cast<uint32_t>(i),
                                        1, &it->second, 0, nullptr);
            }
        }

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void RasterRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
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
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(renderWidth),
                                static_cast<int32_t>(renderHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        offscreenImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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

void RasterRenderer::renderToSwapchain(VulkanContext& ctx,
                                       VkImage swapImage, VkImageView swapView,
                                       VkFormat swapFormat, VkExtent2D extent,
                                       VkSemaphore waitSem, VkSemaphore signalSem,
                                       VkFence fence) {
    const std::string& sceneSource = m_scene->getSceneSource();

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
    std::vector<VkClearValue> clearValues;
    VkClearValue colorClear = {};
    if (m_needsDepth) {
        colorClear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    } else {
        colorClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }
    clearValues.push_back(colorClear);
    if (m_needsDepth) {
        VkClearValue depthClear = {};
        depthClear.depthStencil = {1.0f, 0};
        clearValues.push_back(depthClear);
    }

    VkRenderPassBeginInfo rpBeginInfo = {};
    rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBeginInfo.renderPass = renderPass;
    rpBeginInfo.framebuffer = framebuffer;
    rpBeginInfo.renderArea.offset = {0, 0};
    rpBeginInfo.renderArea.extent = {renderWidth, renderHeight};
    rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBeginInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
        vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                           0, pushConstantSize, pushConstantData.data());
    }

    if (sceneSource == "triangle") {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (sceneSource == "fullscreen" || m_renderPath == "fullscreen") {
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else {
        const auto& mesh = m_scene->getMesh();

        int maxSet = -1;
        for (auto& [idx, _] : reflectedDescSets) {
            maxSet = std::max(maxSet, idx);
        }
        for (int i = 0; i <= maxSet; i++) {
            auto it = reflectedDescSets.find(i);
            if (it != reflectedDescSets.end()) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, static_cast<uint32_t>(i),
                                        1, &it->second, 0, nullptr);
            }
        }
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

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
// Orbit camera: update MVP + Light uniform buffers per frame
// --------------------------------------------------------------------------

void RasterRenderer::updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                                  glm::vec3 up, float fovY, float aspect,
                                  float nearPlane, float farPlane) {
    if (!mvpBuffer || !lightBuffer) return;

    // Update MVP
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    mvp.model = glm::mat4(1.0f);
    mvp.view = Camera::lookAt(eye, target, up);
    mvp.projection = Camera::perspective(fovY, aspect, nearPlane, farPlane);

    void* mapped;
    vmaMapMemory(ctx.allocator, mvpAllocation, &mapped);
    memcpy(mapped, &mvp, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, mvpAllocation);

    // Update Light uniform (preserve light_dir, update view_pos)
    struct LightData {
        glm::vec3 lightDir; float _pad0;
        glm::vec3 viewPos; float _pad1;
    };
    LightData light;
    vmaMapMemory(ctx.allocator, lightAllocation, &mapped);
    memcpy(&light, mapped, sizeof(LightData));
    light.viewPos = eye;
    memcpy(mapped, &light, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, lightAllocation);
}


// Cleanup
// --------------------------------------------------------------------------

void RasterRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    if (framebuffer) vkDestroyFramebuffer(ctx.device, framebuffer, nullptr);
    if (renderPass) vkDestroyRenderPass(ctx.device, renderPass, nullptr);

    if (vertModule) vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    if (fragModule) vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    if (offscreenImageView) vkDestroyImageView(ctx.device, offscreenImageView, nullptr);
    if (offscreenImage) vmaDestroyImage(ctx.allocator, offscreenImage, offscreenAllocation);

    if (depthImageView) vkDestroyImageView(ctx.device, depthImageView, nullptr);
    if (depthImage) vmaDestroyImage(ctx.allocator, depthImage, depthAllocation);

    if (triangleVB) vmaDestroyBuffer(ctx.allocator, triangleVB, triangleVBAllocation);

    if (mvpBuffer) vmaDestroyBuffer(ctx.allocator, mvpBuffer, mvpAllocation);
    if (lightBuffer) vmaDestroyBuffer(ctx.allocator, lightBuffer, lightAllocation);
    if (m_materialBuffer) vmaDestroyBuffer(ctx.allocator, m_materialBuffer, m_materialAllocation);

    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    for (auto& [_, layout] : reflectedSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    reflectedSetLayouts.clear();
    reflectedDescSets.clear();

    // Note: mesh and textures are owned by SceneManager, not cleaned up here
}
