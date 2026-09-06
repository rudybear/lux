#include "unet_runner.h"
#include "vulkan_context.h"
#include "spv_loader.h"
#include "dlss_io.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

VkBuffer createHostVisibleBuffer(VmaAllocator allocator, VkDeviceSize sizeBytes, VmaAllocation& allocOut) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = std::max<VkDeviceSize>(sizeBytes, 16);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    VkBuffer buf = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator, &info, &allocInfo, &buf, &allocOut, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("unet_runner: failed to create buffer");
    }
    return buf;
}

VkDescriptorSetLayout makeSetLayout(VkDevice device, uint32_t n) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(n);
    for (uint32_t i = 0; i < n; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = n;
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout makePipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout, uint32_t pushSize) {
    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = pushSize;
    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 1;
    info.pSetLayouts = &setLayout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipeline makeComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string& spvPath) {
    auto code = SpvLoader::loadSPIRV(spvPath);
    VkShaderModule module = SpvLoader::createShaderModule(device, code);
    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("unet_runner: failed to create pipeline for " + spvPath);
    }
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

void writeDescriptorSet(VkDevice device, VkDescriptorSet set, const std::vector<VkBuffer>& buffers) {
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

double dispatchOne(VulkanContext& ctx, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set,
                    const void* pushData, uint32_t pushSize, uint32_t totalThreads) {
    auto t0 = std::chrono::high_resolution_clock::now();
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    uint32_t groups = (totalThreads + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);
    ctx.endSingleTimeCommands(cmd);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 2-D tiled dispatch (the conv3x3/conv3x3_s2/upsample_concat_conv kernels):
// one workgroup per (tileW x tileH) output tile -- tileW/tileH must match
// the --define workgroup_size_x/_y each kernel's SPIR-V module was
// compiled with (examples/unet_*.lux header comments); vkCmdDispatch's
// group counts are in *workgroups*, not threads, so no local-size args
// are needed here (baked into the shader module's OpExecutionMode).
double dispatchTiled(VulkanContext& ctx, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set,
                      const void* pushData, uint32_t pushSize,
                      uint32_t outW, uint32_t outH, uint32_t tileW, uint32_t tileH) {
    auto t0 = std::chrono::high_resolution_clock::now();
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    uint32_t groupsX = (outW + tileW - 1) / tileW;
    uint32_t groupsY = (outH + tileH - 1) / tileH;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
    ctx.endSingleTimeCommands(cmd);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Tile sizes must match the --define workgroup_size_x/_y each kernel was
// compiled with (examples/unet_*.lux header comments).
// See metal_unet_runner.cpp's matching constants for why the "Lo" variants
// exist (a single worst-case-sized tile caps every caller's occupancy at
// the worst caller's requirement -- costly for high-res/low-channel
// layers). Vulkan's local size is baked into each kernel's SPIR-V module
// (OpExecutionMode LocalSize), so dispatchTiled here only needs the
// spatial tile dims to compute group counts -- no Z argument required.
constexpr uint32_t kConv3x3TileW = 8, kConv3x3TileH = 8;
constexpr uint32_t kConv3x3LoTileW = 10, kConv3x3LoTileH = 10, kConv3x3LoCinMax = 48;
constexpr uint32_t kConv3x3S2TileW = 8, kConv3x3S2TileH = 3;
constexpr uint32_t kUpConcatTileW = 8, kUpConcatTileH = 4;
constexpr uint32_t kUpConcatLoTileW = 8, kUpConcatLoTileH = 8;
constexpr uint32_t kUpConcatLoCinAMax = 48, kUpConcatLoCinBMax = 32;

struct Layer {
    std::string name, type;
    uint32_t cin, cout, kh, kw, weightOffset, biasOffset;
};

struct Manifest {
    uint32_t inChannels, outChannels, K, hidden, upscale, paramStride;
    std::vector<Layer> layers;
};

Manifest loadSidecar(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("unet_runner: cannot open " + path);
    Manifest m;
    uint32_t numLayers;
    f >> m.inChannels >> m.outChannels >> m.K >> m.hidden >> m.upscale >> m.paramStride >> numLayers;
    for (uint32_t i = 0; i < numLayers; ++i) {
        Layer l;
        f >> l.name >> l.type >> l.cin >> l.cout >> l.kh >> l.kw >> l.weightOffset >> l.biasOffset;
        m.layers.push_back(l);
    }
    return m;
}

std::vector<float> loadWeightsFp16AsFp32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("unet_runner: cannot open " + path);
    size_t sizeBytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    size_t count = sizeBytes / 2;
    std::vector<uint16_t> raw(count);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(sizeBytes));
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = DlssIO::halfToFloat(raw[i]);
    return out;
}

} // namespace

int runUnetDump(const std::string& inputNpyPath, const std::string& weightsBlobPath,
                const std::string& manifestJsonPath, const std::string& outputNpyPath,
                const std::string& kernelPipelineDir) {
    fs::path sidecarPath = fs::path(manifestJsonPath).replace_extension(".layers.txt");
    Manifest manifest;
    std::vector<float> weightsHost;
    DlssIO::NpyArray input;
    try {
        manifest = loadSidecar(sidecarPath.string());
        weightsHost = loadWeightsFp16AsFp32(weightsBlobPath);
        input = DlssIO::readNpyFloat32(inputNpyPath);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }
    if (input.shape.size() != 3 || static_cast<uint32_t>(input.shape[2]) != manifest.inChannels) {
        std::cerr << "[error] unet_runner: input shape " << (input.shape.size() >= 3 ? input.shape[2] : -1)
                  << " channels, expected " << manifest.inChannels << std::endl;
        return 1;
    }
    uint32_t netH = static_cast<uint32_t>(input.shape[0]);
    uint32_t netW = static_cast<uint32_t>(input.shape[1]);

    VulkanContext ctx;
    try {
        ctx.init(false, true, nullptr, false);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }

    // One shared weights buffer for every layer/kernel.
    VmaAllocation aWeights;
    VkBuffer bWeights = createHostVisibleBuffer(ctx.allocator, weightsHost.size() * sizeof(float), aWeights);
    {
        void* mapped = nullptr;
        vmaMapMemory(ctx.allocator, aWeights, &mapped);
        std::memcpy(mapped, weightsHost.data(), weightsHost.size() * sizeof(float));
        vmaUnmapMemory(ctx.allocator, aWeights);
    }

    // 3 kernel kinds needing 3 buffers (conv3x3[_s2]_lrelu, conv1x1) and 1
    // needing 4 (upsample_concat_conv_lrelu).
    VkDescriptorSetLayout layout3 = makeSetLayout(ctx.device, 3);
    VkDescriptorSetLayout layout4 = makeSetLayout(ctx.device, 4);
    // conv3x3[_s2]/conv1x1 push: 6 uint = 24 bytes; upsample_concat push: 7 uint = 28 bytes.
    VkPipelineLayout pl3 = makePipelineLayout(ctx.device, layout3, 24);
    VkPipelineLayout pl4 = makePipelineLayout(ctx.device, layout4, 28);

    VkPipeline pConv3x3, pConv3x3Lo, pConv3x3S2, pUpConcat, pUpConcatLo, pConv1x1;
    try {
        pConv3x3 = makeComputePipeline(ctx.device, pl3, kernelPipelineDir + "/unet_conv3x3_lrelu.comp.spv");
        pConv3x3Lo = makeComputePipeline(ctx.device, pl3, kernelPipelineDir + "/unet_conv3x3_lrelu_lo.comp.spv");
        pConv3x3S2 = makeComputePipeline(ctx.device, pl3, kernelPipelineDir + "/unet_conv3x3_s2_lrelu.comp.spv");
        pUpConcat = makeComputePipeline(ctx.device, pl4, kernelPipelineDir + "/unet_upsample_concat_conv_lrelu.comp.spv");
        pUpConcatLo = makeComputePipeline(ctx.device, pl4, kernelPipelineDir + "/unet_upsample_concat_conv_lrelu_lo.comp.spv");
        pConv1x1 = makeComputePipeline(ctx.device, pl3, kernelPipelineDir + "/unet_conv1x1.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * 3 + 4 * 3};
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool);

    auto allocSet = [&](VkDescriptorSetLayout l) {
        VkDescriptorSetAllocateInfo a = {};
        a.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        a.descriptorPool = pool;
        a.descriptorSetCount = 1;
        a.pSetLayouts = &l;
        VkDescriptorSet s = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(ctx.device, &a, &s);
        return s;
    };

    // Working activation buffers -- allocate generously for the largest
    // possible layer output (input resolution * max channel count) and
    // reuse two ping-pong buffers plus a small stack of skip buffers.
    size_t maxElems = static_cast<size_t>(netH) * netW * 256;  // generous upper bound
    VmaAllocation aBufA, aBufB;
    VkBuffer bufA = createHostVisibleBuffer(ctx.allocator, maxElems * sizeof(float), aBufA);
    VkBuffer bufB = createHostVisibleBuffer(ctx.allocator, maxElems * sizeof(float), aBufB);
    {
        void* mapped = nullptr;
        vmaMapMemory(ctx.allocator, aBufA, &mapped);
        std::memcpy(mapped, input.data.data(), input.data.size() * sizeof(float));
        vmaUnmapMemory(ctx.allocator, aBufA);
    }

    struct SkipBuf { VkBuffer buf; VmaAllocation alloc; uint32_t h, w, c; };
    std::vector<SkipBuf> skipStack;

    VkBuffer curBuf = bufA;
    VmaAllocation curAlloc = aBufA;
    VkBuffer otherBuf = bufB;
    VmaAllocation otherAlloc = aBufB;
    uint32_t curH = netH, curW = netW;

    struct Push3 { uint32_t a, b, cin, cout, weightOffset, biasOffset; };
    struct Push4 { uint32_t outH, outW, cinA, cinB, cout, weightOffset, biasOffset; };

    double totalMs = 0;
    for (size_t i = 0; i < manifest.layers.size(); ++i) {
        const Layer& layer = manifest.layers[i];
        VkDescriptorSet set;
        double ms;
        uint32_t newH = curH, newW = curW;

        if (layer.type == "conv3x3_lrelu") {
            set = allocSet(layout3);
            writeDescriptorSet(ctx.device, set, {curBuf, bWeights, otherBuf});
            Push3 push = {curH, curW, layer.cin, layer.cout, layer.weightOffset, layer.biasOffset};
            if (layer.cin <= kConv3x3LoCinMax) {
                ms = dispatchTiled(ctx, pConv3x3Lo, pl3, set, &push, sizeof(push), curW, curH, kConv3x3LoTileW, kConv3x3LoTileH);
            } else {
                ms = dispatchTiled(ctx, pConv3x3, pl3, set, &push, sizeof(push), curW, curH, kConv3x3TileW, kConv3x3TileH);
            }
        } else if (layer.type == "conv1x1") {
            set = allocSet(layout3);
            writeDescriptorSet(ctx.device, set, {curBuf, bWeights, otherBuf});
            Push3 push = {curH, curW, layer.cin, layer.cout, layer.weightOffset, layer.biasOffset};
            ms = dispatchOne(ctx, pConv1x1, pl3, set, &push, sizeof(push), curH * curW);
        } else if (layer.type == "conv3x3_s2_lrelu") {
            set = allocSet(layout3);
            writeDescriptorSet(ctx.device, set, {curBuf, bWeights, otherBuf});
            Push3 push = {curH, curW, layer.cin, layer.cout, layer.weightOffset, layer.biasOffset};
            newH = curH / 2; newW = curW / 2;
            ms = dispatchTiled(ctx, pConv3x3S2, pl3, set, &push, sizeof(push), newW, newH, kConv3x3S2TileW, kConv3x3S2TileH);
        } else if (layer.type == "upsample_concat_conv_lrelu") {
            SkipBuf skip = skipStack.back();
            skipStack.pop_back();
            uint32_t cinB = skip.c;
            uint32_t cinA = layer.cin - cinB;
            set = allocSet(layout4);
            writeDescriptorSet(ctx.device, set, {curBuf, skip.buf, bWeights, otherBuf});
            Push4 push = {skip.h, skip.w, cinA, cinB, layer.cout, layer.weightOffset, layer.biasOffset};
            newH = skip.h; newW = skip.w;
            if (cinA <= kUpConcatLoCinAMax && cinB <= kUpConcatLoCinBMax) {
                ms = dispatchTiled(ctx, pUpConcatLo, pl4, set, &push, sizeof(push), newW, newH, kUpConcatLoTileW, kUpConcatLoTileH);
            } else {
                ms = dispatchTiled(ctx, pUpConcat, pl4, set, &push, sizeof(push), newW, newH, kUpConcatTileW, kUpConcatTileH);
            }
            vmaDestroyBuffer(ctx.allocator, skip.buf, skip.alloc);
        } else {
            std::cerr << "[error] unet_runner: unknown layer type " << layer.type << std::endl;
            return 1;
        }

        totalMs += ms;
        std::cout << "[unet] layer " << i << " (" << layer.name << ", " << layer.type << "): "
                  << ms << " ms" << std::endl;

        curH = newH; curW = newW;
        std::swap(curBuf, otherBuf);
        std::swap(curAlloc, otherAlloc);

        // Push a skip connection right after stem (0), down1.refine (2),
        // down2.refine (4) -- matches ParamPredUNet._features's x0/x1/x2.
        if (i == 0 || i == 2 || i == 4) {
            size_t n = static_cast<size_t>(curH) * curW * layer.cout;
            VmaAllocation skipAlloc;
            VkBuffer skipBuf = createHostVisibleBuffer(ctx.allocator, n * sizeof(float), skipAlloc);
            void* dst = nullptr;
            void* src = nullptr;
            vmaMapMemory(ctx.allocator, skipAlloc, &dst);
            vmaMapMemory(ctx.allocator, curAlloc, &src);
            std::memcpy(dst, src, n * sizeof(float));
            vmaUnmapMemory(ctx.allocator, curAlloc);
            vmaUnmapMemory(ctx.allocator, skipAlloc);
            skipStack.push_back({skipBuf, skipAlloc, curH, curW, layer.cout});
        }
    }

    std::cout << "[unet] total GPU dispatch time: " << totalMs << " ms" << std::endl;

    size_t outN = static_cast<size_t>(curH) * curW * manifest.outChannels;
    std::vector<float> outData(outN);
    {
        void* mapped = nullptr;
        vmaMapMemory(ctx.allocator, curAlloc, &mapped);
        std::memcpy(outData.data(), mapped, outN * sizeof(float));
        vmaUnmapMemory(ctx.allocator, curAlloc);
    }
    DlssIO::writeNpyFloat32(outputNpyPath, outData, {curH, curW, manifest.outChannels});
    std::cout << "[unet] wrote " << outputNpyPath << " (" << curH << "x" << curW << "x"
              << manifest.outChannels << ")" << std::endl;

    return 0;
}
