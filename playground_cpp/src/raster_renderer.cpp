#include "raster_renderer.h"
#include "spv_loader.h"
#include "camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <array>

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

    // Depth attachment (PBR mode)
    if (mode == RasterMode::PBR) {
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

    // Depth attachment (PBR only)
    if (mode == RasterMode::PBR) {
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
    if (mode == RasterMode::PBR) {
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
    if (mode == RasterMode::PBR) {
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
// PBR pipeline
// --------------------------------------------------------------------------

void RasterRenderer::setupPBRResources(VulkanContext& ctx) {
    float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);
    auto cam = Camera::getDefaultMatrices(aspect);

    // Create MVP uniform buffer (3 x mat4 = 192 bytes)
    struct MVPData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };
    MVPData mvpData = {cam.model, cam.view, cam.projection};

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
    LightData lightData = {lightDir, 0.0f, Camera::DEFAULT_EYE, 0.0f};

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

    // Generate and upload sphere mesh
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Scene::generateSphere(32, 32, vertices, indices);
    mesh = Scene::uploadMesh(ctx.allocator, ctx.device, ctx.commandPool,
                             ctx.graphicsQueue, vertices, indices);

    // Generate and upload procedural texture
    std::cout << "Generating procedural texture..." << std::endl;
    auto texPixels = Scene::generateProceduralTexture(512);
    texture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                   ctx.graphicsQueue, texPixels, 512, 512);

    // Create descriptor set layout 0: MVP uniform
    VkDescriptorSetLayoutBinding mvpBinding = {};
    mvpBinding.binding = 0;
    mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    mvpBinding.descriptorCount = 1;
    mvpBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layout0Info = {};
    layout0Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout0Info.bindingCount = 1;
    layout0Info.pBindings = &mvpBinding;

    vkCreateDescriptorSetLayout(ctx.device, &layout0Info, nullptr, &descSetLayout0);

    // Create descriptor set layout 1: light + sampler + texture
    VkDescriptorSetLayoutBinding fragBindings[3] = {};

    fragBindings[0].binding = 0;
    fragBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    fragBindings[0].descriptorCount = 1;
    fragBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    fragBindings[1].binding = 1;
    fragBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    fragBindings[1].descriptorCount = 1;
    fragBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    fragBindings[2].binding = 2;
    fragBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    fragBindings[2].descriptorCount = 1;
    fragBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout1Info = {};
    layout1Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout1Info.bindingCount = 3;
    layout1Info.pBindings = fragBindings;

    vkCreateDescriptorSetLayout(ctx.device, &layout1Info, nullptr, &descSetLayout1);

    // Create descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    // Allocate descriptor sets
    VkDescriptorSetLayout layouts[] = {descSetLayout0, descSetLayout1};
    VkDescriptorSetAllocateInfo allocSetInfo = {};
    allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocSetInfo.descriptorPool = descriptorPool;
    allocSetInfo.descriptorSetCount = 2;
    allocSetInfo.pSetLayouts = layouts;

    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(ctx.device, &allocSetInfo, sets);
    descSet0 = sets[0];
    descSet1 = sets[1];

    // Update descriptor set 0: MVP
    VkDescriptorBufferInfo mvpBufDesc = {};
    mvpBufDesc.buffer = mvpBuffer;
    mvpBufDesc.offset = 0;
    mvpBufDesc.range = 192;

    VkWriteDescriptorSet writes[4] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descSet0;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &mvpBufDesc;

    // Update descriptor set 1: Light + sampler + texture
    VkDescriptorBufferInfo lightBufDesc = {};
    lightBufDesc.buffer = lightBuffer;
    lightBufDesc.offset = 0;
    lightBufDesc.range = 32;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet1;
    writes[1].dstBinding = 0;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &lightBufDesc;

    VkDescriptorImageInfo samplerDesc = {};
    samplerDesc.sampler = texture.sampler;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descSet1;
    writes[2].dstBinding = 1;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[2].pImageInfo = &samplerDesc;

    VkDescriptorImageInfo texDesc = {};
    texDesc.imageView = texture.imageView;
    texDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = descSet1;
    writes[3].dstBinding = 2;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[3].pImageInfo = &texDesc;

    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);
}

void RasterRenderer::createPipelinePBR(VulkanContext& ctx) {
    // Pipeline layout with 2 descriptor sets
    VkDescriptorSetLayout layouts[] = {descSetLayout0, descSetLayout1};

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = layouts;

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

    // Vertex input: position (vec3) + normal (vec3) + uv (vec2) = 32 bytes
    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(Vertex); // 32 bytes
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[3] = {};
    attrs[0].location = 0; // position
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;

    attrs[1].location = 1; // normal
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = 12;

    attrs[2].location = 2; // uv
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[2].offset = 24;

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3;
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
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
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

void RasterRenderer::init(VulkanContext& ctx, RasterMode renderMode,
                          const std::string& vertSpvPath, const std::string& fragSpvPath,
                          uint32_t width, uint32_t height) {
    mode = renderMode;
    renderWidth = width;
    renderHeight = height;

    // Load shader modules
    if (mode == RasterMode::Fullscreen) {
        // Use built-in fullscreen vertex shader
        const auto& fullscreenSpv = SpvLoader::getFullscreenVertSPIRV();
        vertModule = SpvLoader::createShaderModule(ctx.device, fullscreenSpv);
    } else {
        auto vertCode = SpvLoader::loadSPIRV(vertSpvPath);
        vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
    }

    auto fragCode = SpvLoader::loadSPIRV(fragSpvPath);
    fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

    // Create offscreen target
    createOffscreenTarget(ctx);
    createRenderPass(ctx);
    createFramebuffer(ctx);

    // Setup mode-specific resources
    switch (mode) {
        case RasterMode::Triangle: {
            // Upload triangle vertex data
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

            createPipelineTriangle(ctx);
            break;
        }
        case RasterMode::Fullscreen:
            createPipelineFullscreen(ctx);
            break;
        case RasterMode::PBR:
            setupPBRResources(ctx);
            createPipelinePBR(ctx);
            break;
    }
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RasterRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Set clear values
    std::vector<VkClearValue> clearValues;
    VkClearValue colorClear = {};
    if (mode == RasterMode::PBR) {
        colorClear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    } else {
        colorClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }
    clearValues.push_back(colorClear);

    if (mode == RasterMode::PBR) {
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

    switch (mode) {
        case RasterMode::Triangle: {
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
            vkCmdDraw(cmd, 3, 1, 0, 0);
            break;
        }
        case RasterMode::Fullscreen:
            vkCmdDraw(cmd, 3, 1, 0, 0);
            break;
        case RasterMode::PBR: {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSet0, 0, nullptr);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 1, 1, &descSet1, 0, nullptr);

            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
            vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
            break;
        }
    }

    vkCmdEndRenderPass(cmd);
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Interactive mode: render to swapchain
// --------------------------------------------------------------------------

void RasterRenderer::renderToSwapchain(VulkanContext& ctx,
                                       VkImage swapImage, VkImageView swapView,
                                       VkFormat swapFormat, VkExtent2D extent,
                                       VkSemaphore waitSem, VkSemaphore signalSem,
                                       VkFence fence) {
    // For interactive mode, we render to offscreen then blit to swapchain
    // First do the offscreen render
    render(ctx);

    // Then blit offscreen -> swapchain
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

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

    // Blit
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

    // Transition swapchain image to PRESENT
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
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

    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    if (descSetLayout0) vkDestroyDescriptorSetLayout(ctx.device, descSetLayout0, nullptr);
    if (descSetLayout1) vkDestroyDescriptorSetLayout(ctx.device, descSetLayout1, nullptr);

    Scene::destroyMesh(ctx.allocator, mesh);
    Scene::destroyTexture(ctx.allocator, ctx.device, texture);
}
