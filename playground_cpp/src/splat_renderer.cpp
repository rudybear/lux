#include "splat_renderer.h"
#include "vulkan_context.h"
#include "gltf_loader.h"
#include "spv_loader.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <iostream>

// --------------------------------------------------------------------------
// VMA buffer helper
// --------------------------------------------------------------------------

static void createVmaBuffer(VmaAllocator allocator, VkDeviceSize size,
                             VkBufferUsageFlags usage, VmaMemoryUsage memUsage,
                             VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = std::max(size, VkDeviceSize(16)); // avoid zero-size buffers
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memUsage;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA buffer");
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

// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

SplatRenderer::~SplatRenderer() {
    // Resources should be cleaned up via cleanup(ctx) before destruction.
    // This is a safety net — VMA allocator is gone by now so we can't free here.
}

// --------------------------------------------------------------------------
// Offscreen render target (VMA)
// --------------------------------------------------------------------------

void SplatRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &colorImage_, &colorAlloc_, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = colorImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &colorView_);

    // Depth image
    VkImageCreateInfo depthInfo = imageInfo;
    depthInfo.format = VK_FORMAT_D32_SFLOAT;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    vmaCreateImage(ctx.allocator, &depthInfo, &allocInfo,
                   &depthImage_, &depthAlloc_, nullptr);

    VkImageViewCreateInfo depthViewInfo = viewInfo;
    depthViewInfo.image = depthImage_;
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    vkCreateImageView(ctx.device, &depthViewInfo, nullptr, &depthView_);
}

// --------------------------------------------------------------------------
// Render pass (color + depth, finalLayout = TRANSFER_SRC for blit/screenshot)
// --------------------------------------------------------------------------

void SplatRenderer::createRenderPass(VkDevice device) {
    VkAttachmentDescription attachments[2] = {};

    // Color
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // Depth
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = deps;

    vkCreateRenderPass(device, &rpInfo, nullptr, &renderPass_);
}

// --------------------------------------------------------------------------
// Framebuffer
// --------------------------------------------------------------------------

void SplatRenderer::createFramebuffer(VkDevice device) {
    VkImageView fbViews[2] = {colorView_, depthView_};

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPass_;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = fbViews;
    fbInfo.width = width_;
    fbInfo.height = height_;
    fbInfo.layers = 1;

    vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffer_);
}

// --------------------------------------------------------------------------
// Pipeline creation
// --------------------------------------------------------------------------

void SplatRenderer::createPipelines(VkDevice device, const std::string& shaderBase) {
    // --- Descriptor set layouts ---

    // Compute: bindings 0..9 = 10 SSBOs
    std::vector<VkDescriptorSetLayoutBinding> computeBindings(10);
    for (uint32_t i = 0; i < 10; ++i) {
        computeBindings[i] = {};
        computeBindings[i].binding = i;
        computeBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        computeBindings[i].descriptorCount = 1;
        computeBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo computeLayoutInfo = {};
    computeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    computeLayoutInfo.bindingCount = static_cast<uint32_t>(computeBindings.size());
    computeLayoutInfo.pBindings = computeBindings.data();
    vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &computeSetLayout_);

    // Render: 4 SSBOs (projected_centers, conics, colors, sorted_indices)
    std::vector<VkDescriptorSetLayoutBinding> renderBindings(4);
    for (uint32_t i = 0; i < 4; ++i) {
        renderBindings[i] = {};
        renderBindings[i].binding = i;
        renderBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        renderBindings[i].descriptorCount = 1;
        renderBindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo renderLayoutInfo = {};
    renderLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    renderLayoutInfo.bindingCount = static_cast<uint32_t>(renderBindings.size());
    renderLayoutInfo.pBindings = renderBindings.data();
    vkCreateDescriptorSetLayout(device, &renderLayoutInfo, nullptr, &renderSetLayout_);

    // --- Pipeline layouts ---

    // Compute push constants: view(64) + proj(64) + camPos(12) + pad(4) + focal(8) + screen(8) + numSplats(4) + pad(12) = 176 bytes
    VkPushConstantRange computePush = {};
    computePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    computePush.offset = 0;
    computePush.size = 176;

    VkPipelineLayoutCreateInfo computePipeLayoutInfo = {};
    computePipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computePipeLayoutInfo.setLayoutCount = 1;
    computePipeLayoutInfo.pSetLayouts = &computeSetLayout_;
    computePipeLayoutInfo.pushConstantRangeCount = 1;
    computePipeLayoutInfo.pPushConstantRanges = &computePush;
    vkCreatePipelineLayout(device, &computePipeLayoutInfo, nullptr, &computeLayout_);

    // Render push constants: screen_size(8) + visible_count(4) + alpha_cutoff(4) = 16 bytes
    VkPushConstantRange renderPush = {};
    renderPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    renderPush.offset = 0;
    renderPush.size = 16;

    VkPipelineLayoutCreateInfo renderPipeLayoutInfo = {};
    renderPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    renderPipeLayoutInfo.setLayoutCount = 1;
    renderPipeLayoutInfo.pSetLayouts = &renderSetLayout_;
    renderPipeLayoutInfo.pushConstantRangeCount = 1;
    renderPipeLayoutInfo.pPushConstantRanges = &renderPush;
    vkCreatePipelineLayout(device, &renderPipeLayoutInfo, nullptr, &renderLayout_);

    // --- Load compute shader ---
    auto compCode = SpvLoader::loadSPIRV(shaderBase + ".comp.spv");
    VkShaderModule compModule = SpvLoader::createShaderModule(device, compCode);

    VkComputePipelineCreateInfo compPipeInfo = {};
    compPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compPipeInfo.stage.module = compModule;
    compPipeInfo.stage.pName = "main";
    compPipeInfo.layout = computeLayout_;

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &computePipeline_);
    vkDestroyShaderModule(device, compModule, nullptr);

    // --- Load graphics shaders ---
    auto vertCode = SpvLoader::loadSPIRV(shaderBase + ".vert.spv");
    auto fragCode = SpvLoader::loadSPIRV(shaderBase + ".frag.spv");
    VkShaderModule vertModule = SpvLoader::createShaderModule(device, vertCode);
    VkShaderModule fragModule = SpvLoader::createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.width = static_cast<float>(width_);
    viewport.height = static_cast<float>(height_);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.extent = {width_, height_};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // premultiplied alpha
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = renderLayout_;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &renderPipeline_);

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    // --- Descriptor pool ---
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 20;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool_);

    VkDescriptorSetLayout layouts[2] = {computeSetLayout_, renderSetLayout_};
    VkDescriptorSetAllocateInfo setAllocInfo = {};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = descriptorPool_;
    setAllocInfo.descriptorSetCount = 2;
    setAllocInfo.pSetLayouts = layouts;

    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(device, &setAllocInfo, sets);
    computeDescSet_ = sets[0];
    renderDescSet_ = sets[1];
}

// --------------------------------------------------------------------------
// Buffer creation and data upload
// --------------------------------------------------------------------------

void SplatRenderer::createBuffers(VulkanContext& ctx, const GaussianSplatData& data) {
    numSplats_ = data.num_splats;
    shDegree_ = data.sh_degree;
    if (numSplats_ == 0) return;

    VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Input buffers (CPU-visible for upload)
    // Positions: loader stores as vec4 (x,y,z,1) already
    const auto& pos4 = data.positions;  // already numSplats * 4 floats
    hostPositions_ = pos4;  // cache for CPU sort

    createVmaBuffer(ctx.allocator, pos4.size() * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, posBuffer_, posAlloc_);
    uploadVmaBuffer(ctx.allocator, posAlloc_, pos4.data(), pos4.size() * sizeof(float));

    // Rotations (vec4 xyzw)
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, rotBuffer_, rotAlloc_);
    uploadVmaBuffer(ctx.allocator, rotAlloc_, data.rotations.data(),
                    numSplats_ * 4 * sizeof(float));

    // Scales as vec4 (x,y,z,0)
    std::vector<float> scale4(numSplats_ * 4);
    for (uint32_t i = 0; i < numSplats_; ++i) {
        scale4[i * 4 + 0] = data.scales[i * 3 + 0];
        scale4[i * 4 + 1] = data.scales[i * 3 + 1];
        scale4[i * 4 + 2] = data.scales[i * 3 + 2];
        scale4[i * 4 + 3] = 0.0f;
    }
    createVmaBuffer(ctx.allocator, scale4.size() * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, scaleBuffer_, scaleAlloc_);
    uploadVmaBuffer(ctx.allocator, scaleAlloc_, scale4.data(), scale4.size() * sizeof(float));

    // Opacities
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, opacityBuffer_, opacityAlloc_);
    uploadVmaBuffer(ctx.allocator, opacityAlloc_, data.opacities.data(),
                    numSplats_ * sizeof(float));

    // SH coefficient buffers — pad vec3 data to vec4 for std430 alignment
    for (const auto& coeffs : data.sh_coefficients) {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        // Shader expects vec4 per splat; source data may be vec3
        uint32_t srcFloatsPerSplat = coeffs.empty() ? 0 : static_cast<uint32_t>(coeffs.size() / numSplats_);
        if (srcFloatsPerSplat == 3) {
            // Pad vec3 → vec4
            std::vector<float> padded(numSplats_ * 4, 0.0f);
            for (uint32_t i = 0; i < numSplats_; ++i) {
                padded[i * 4 + 0] = coeffs[i * 3 + 0];
                padded[i * 4 + 1] = coeffs[i * 3 + 1];
                padded[i * 4 + 2] = coeffs[i * 3 + 2];
            }
            createVmaBuffer(ctx.allocator, padded.size() * sizeof(float), ssbo,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            uploadVmaBuffer(ctx.allocator, alloc, padded.data(), padded.size() * sizeof(float));
        } else {
            VkDeviceSize sz = coeffs.empty() ? sizeof(float) : coeffs.size() * sizeof(float);
            createVmaBuffer(ctx.allocator, sz, ssbo, VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            if (!coeffs.empty()) {
                uploadVmaBuffer(ctx.allocator, alloc, coeffs.data(), coeffs.size() * sizeof(float));
            }
        }
        shBuffers_.push_back(buf);
        shAllocs_.push_back(alloc);
    }

    // Projected output buffers (GPU only, written by compute)
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projCenterBuffer_, projCenterAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projConicBuffer_, projConicAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projColorBuffer_, projColorAlloc_);

    // Sort keys (GPU only, written by compute, read back for CPU sort)
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, sortKeysBuffer_, sortKeysAlloc_);

    // Sorted indices (CPU-visible for upload after sort)
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, sortedIndicesBuffer_, sortedIndicesAlloc_);

    // Visible count (GPU only, atomic counter)
    createVmaBuffer(ctx.allocator, sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, visibleCountBuffer_, visibleCountAlloc_);

    // --- Update descriptor sets ---
    auto writeSSBO = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo bufInfo = {};
        bufInfo.buffer = buffer;
        bufInfo.offset = 0;
        bufInfo.range = size;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    };

    // Compute set: positions(0), rotations(1), scales(2), opacities(3), sh(4),
    //   projected_centers(5), projected_conics(6), projected_colors(7),
    //   sort_keys(8), visible_count(9)
    writeSSBO(computeDescSet_, 0, posBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 1, rotBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 2, scaleBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 3, opacityBuffer_, numSplats_ * sizeof(float));

    VkBuffer shBuf = shBuffers_.empty() ? posBuffer_ : shBuffers_[0];
    VkDeviceSize shSize = shBuffers_.empty() ? sizeof(float) : numSplats_ * 4 * sizeof(float);
    writeSSBO(computeDescSet_, 4, shBuf, std::max(shSize, VkDeviceSize(sizeof(float))));

    writeSSBO(computeDescSet_, 5, projCenterBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 6, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 7, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 8, sortKeysBuffer_, numSplats_ * sizeof(float));
    writeSSBO(computeDescSet_, 9, visibleCountBuffer_, sizeof(uint32_t));

    // Render set: projected_centers(0), conics(1), colors(2), sorted_indices(3)
    writeSSBO(renderDescSet_, 0, projCenterBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 1, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 2, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 3, sortedIndicesBuffer_, numSplats_ * sizeof(uint32_t));
}

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------

void SplatRenderer::init(VulkanContext& ctx, const GaussianSplatData& data,
                          const std::string& shaderBase, uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;

    createOffscreenTarget(ctx);
    createRenderPass(ctx.device);
    createFramebuffer(ctx.device);
    createPipelines(ctx.device, shaderBase);
    createBuffers(ctx, data);

    // Default camera from bounding box
    if (data.num_splats > 0) {
        glm::vec3 minB(1e9f), maxB(-1e9f);
        for (uint32_t i = 0; i < data.num_splats; ++i) {
            float x = data.positions[i * 4 + 0];
            float y = data.positions[i * 4 + 1];
            float z = data.positions[i * 4 + 2];
            minB = glm::min(minB, glm::vec3(x, y, z));
            maxB = glm::max(maxB, glm::vec3(x, y, z));
        }
        glm::vec3 center = (minB + maxB) * 0.5f;
        float radius = glm::length(maxB - minB) * 0.5f;
        if (radius < 0.001f) radius = 1.0f;

        camPos_ = center + glm::vec3(0.0f, radius * 0.5f, radius * 2.5f);
        float aspect = static_cast<float>(width) / static_cast<float>(height);
        float fov = glm::radians(45.0f);
        viewMatrix_ = glm::lookAt(camPos_, center, glm::vec3(0.0f, 1.0f, 0.0f));
        projMatrix_ = glm::perspective(fov, aspect, 0.01f, radius * 10.0f);
        projMatrix_[1][1] *= -1.0f; // Vulkan Y-flip

        // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = w/(2*tan(fov_x/2))
        // For square pixels: fx = fy = h/(2*tan(fov_y/2))
        focalY_ = 0.5f * static_cast<float>(height) / tanf(fov * 0.5f);
        focalX_ = focalY_;  // square pixels
    }

    std::cout << "[info] SplatRenderer initialized: " << numSplats_ << " splats, "
              << width << "x" << height << std::endl;
}

// --------------------------------------------------------------------------
// Update camera
// --------------------------------------------------------------------------

void SplatRenderer::updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                                  float fovY, float aspect, float nearPlane, float farPlane) {
    camPos_ = eye;
    viewMatrix_ = glm::lookAt(eye, target, up);
    projMatrix_ = glm::perspective(fovY, aspect, nearPlane, farPlane);
    projMatrix_[1][1] *= -1.0f; // Vulkan Y-flip

    // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = fy for square pixels
    focalY_ = 0.5f * static_cast<float>(height_) / tanf(fovY * 0.5f);
    focalX_ = focalY_;
}

// --------------------------------------------------------------------------
// Render
// --------------------------------------------------------------------------

void SplatRenderer::render(VulkanContext& ctx) {
    if (numSplats_ == 0) return;

    // CPU sort (uses cached host positions + current view matrix)
    {
        const float* vm = &viewMatrix_[0][0];
        std::vector<float> depths(numSplats_);
        for (uint32_t i = 0; i < numSplats_; ++i) {
            float x = hostPositions_[i * 4 + 0];
            float y = hostPositions_[i * 4 + 1];
            float z = hostPositions_[i * 4 + 2];
            depths[i] = vm[2] * x + vm[6] * y + vm[10] * z + vm[14];
        }
        std::vector<uint32_t> indices(numSplats_);
        std::iota(indices.begin(), indices.end(), 0u);
        std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
            return depths[a] < depths[b];  // back-to-front (most negative Z = farthest first)
        });
        uploadVmaBuffer(ctx.allocator, sortedIndicesAlloc_,
                        indices.data(), numSplats_ * sizeof(uint32_t));
    }

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // --- Compute dispatch ---
    struct ComputePush {
        float view[16];       // offset 0
        float proj[16];       // offset 64
        float camPos[3];      // offset 128
        float _pad0;          // offset 140
        float screenW, screenH; // offset 144 (screen_size)
        uint32_t numSplats;   // offset 152 (total_splats)
        float focalX;         // offset 156
        float focalY;         // offset 160
        int32_t shDegree;     // offset 164
        float _pad1[2];       // offset 168 (pad to 176)
    } push = {};

    std::memcpy(push.view, &viewMatrix_[0][0], 64);
    std::memcpy(push.proj, &projMatrix_[0][0], 64);
    push.camPos[0] = camPos_.x;
    push.camPos[1] = camPos_.y;
    push.camPos[2] = camPos_.z;
    push.screenW = static_cast<float>(width_);
    push.screenH = static_cast<float>(height_);
    push.numSplats = numSplats_;
    push.focalX = focalX_;
    push.focalY = focalY_;
    push.shDegree = static_cast<int32_t>(shDegree_);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_,
                            0, 1, &computeDescSet_, 0, nullptr);
    vkCmdPushConstants(cmd, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    uint32_t groupCount = (numSplats_ + 255) / 256;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    // Barrier: compute -> vertex/fragment read
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    // --- Render pass ---
    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = renderPass_;
    rpBegin.framebuffer = framebuffer_;
    rpBegin.renderArea.extent = {width_, height_};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, renderLayout_,
                            0, 1, &renderDescSet_, 0, nullptr);

    // Push constants: screen_size(vec2) + visible_count(uint) + alpha_cutoff(float) = 16 bytes
    struct RenderPush {
        float screenW, screenH;
        uint32_t visibleCount;
        float alphaCutoff;
    } renderPush = {};
    renderPush.screenW = static_cast<float>(width_);
    renderPush.screenH = static_cast<float>(height_);
    renderPush.visibleCount = numSplats_;  // all splats visible (CPU sort)
    renderPush.alphaCutoff = 0.004f;
    vkCmdPushConstants(cmd, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(renderPush), &renderPush);

    // Instanced draw: 6 vertices (quad) x numSplats instances
    vkCmdDraw(cmd, 6, numSplats_, 0, 0);

    vkCmdEndRenderPass(cmd);

    // Submit and wait
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void SplatRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
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
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit offscreen -> swapchain
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(width_),
                                static_cast<int32_t>(height_), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        colorImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
// Cleanup
// --------------------------------------------------------------------------

void SplatRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    // Framebuffer and render pass
    if (framebuffer_ != VK_NULL_HANDLE) vkDestroyFramebuffer(ctx.device, framebuffer_, nullptr);
    if (renderPass_ != VK_NULL_HANDLE)  vkDestroyRenderPass(ctx.device, renderPass_, nullptr);

    // Image views
    if (colorView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, colorView_, nullptr);
    if (depthView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, depthView_, nullptr);

    // Images (VMA)
    if (colorImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, colorImage_, colorAlloc_);
    if (depthImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, depthImage_, depthAlloc_);

    // Pipelines
    if (computePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, computePipeline_, nullptr);
    if (renderPipeline_ != VK_NULL_HANDLE)  vkDestroyPipeline(ctx.device, renderPipeline_, nullptr);
    if (computeLayout_ != VK_NULL_HANDLE)   vkDestroyPipelineLayout(ctx.device, computeLayout_, nullptr);
    if (renderLayout_ != VK_NULL_HANDLE)    vkDestroyPipelineLayout(ctx.device, renderLayout_, nullptr);

    // Descriptor layouts and pool
    if (computeSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, computeSetLayout_, nullptr);
    if (renderSetLayout_ != VK_NULL_HANDLE)  vkDestroyDescriptorSetLayout(ctx.device, renderSetLayout_, nullptr);
    if (descriptorPool_ != VK_NULL_HANDLE)   vkDestroyDescriptorPool(ctx.device, descriptorPool_, nullptr);

    // Buffers (VMA)
    destroyVmaBuffer(ctx.allocator, posBuffer_, posAlloc_);
    destroyVmaBuffer(ctx.allocator, rotBuffer_, rotAlloc_);
    destroyVmaBuffer(ctx.allocator, scaleBuffer_, scaleAlloc_);
    destroyVmaBuffer(ctx.allocator, opacityBuffer_, opacityAlloc_);
    destroyVmaBuffer(ctx.allocator, projCenterBuffer_, projCenterAlloc_);
    destroyVmaBuffer(ctx.allocator, projConicBuffer_, projConicAlloc_);
    destroyVmaBuffer(ctx.allocator, projColorBuffer_, projColorAlloc_);
    destroyVmaBuffer(ctx.allocator, sortKeysBuffer_, sortKeysAlloc_);
    destroyVmaBuffer(ctx.allocator, sortedIndicesBuffer_, sortedIndicesAlloc_);
    destroyVmaBuffer(ctx.allocator, visibleCountBuffer_, visibleCountAlloc_);

    for (size_t i = 0; i < shBuffers_.size(); ++i) {
        destroyVmaBuffer(ctx.allocator, shBuffers_[i], shAllocs_[i]);
    }
    shBuffers_.clear();
    shAllocs_.clear();

    // Zero out all handles
    framebuffer_ = VK_NULL_HANDLE;
    renderPass_ = VK_NULL_HANDLE;
    colorView_ = VK_NULL_HANDLE;
    depthView_ = VK_NULL_HANDLE;
    colorImage_ = VK_NULL_HANDLE;
    depthImage_ = VK_NULL_HANDLE;
    computePipeline_ = VK_NULL_HANDLE;
    renderPipeline_ = VK_NULL_HANDLE;
    computeLayout_ = VK_NULL_HANDLE;
    renderLayout_ = VK_NULL_HANDLE;
    computeSetLayout_ = VK_NULL_HANDLE;
    renderSetLayout_ = VK_NULL_HANDLE;
    descriptorPool_ = VK_NULL_HANDLE;
}
