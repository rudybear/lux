#include "reconstruct_runner.h"
#include "vulkan_context.h"
#include "spv_loader.h"
#include "dlss_io.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

namespace fs = std::filesystem;

namespace {

// --- Small local helpers (same conventions as splat_renderer.cpp) ---------

VkBuffer createHostVisibleBuffer(VmaAllocator allocator, VkDeviceSize sizeBytes,
                                  VmaAllocation& allocOut) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = std::max<VkDeviceSize>(sizeBytes, 16);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // host-visible + device-accessible (UMA)

    VkBuffer buf = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator, &info, &allocInfo, &buf, &allocOut, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("reconstruct_runner: failed to create buffer");
    }
    return buf;
}

void uploadFloats(VmaAllocator allocator, VmaAllocation alloc, const std::vector<float>& data) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    if (!data.empty()) std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

void zeroBuffer(VmaAllocator allocator, VmaAllocation alloc, size_t numFloats) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    std::memset(mapped, 0, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

std::vector<float> downloadFloats(VmaAllocator allocator, VmaAllocation alloc, size_t numFloats) {
    std::vector<float> out(numFloats);
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    std::memcpy(out.data(), mapped, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
    return out;
}

void copyBufferHost(VmaAllocator allocator, VmaAllocation dst, VmaAllocation src, size_t numFloats) {
    void* dstPtr = nullptr;
    void* srcPtr = nullptr;
    vmaMapMemory(allocator, dst, &dstPtr);
    vmaMapMemory(allocator, src, &srcPtr);
    std::memcpy(dstPtr, srcPtr, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, src);
    vmaUnmapMemory(allocator, dst);
}

VkDescriptorSetLayout makeSetLayout(VkDevice device, uint32_t numStorageBuffers) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(numStorageBuffers);
    for (uint32_t i = 0; i < numStorageBuffers; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = numStorageBuffers;
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout makePipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout,
                                     uint32_t pushConstantSize) {
    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = pushConstantSize;

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
        throw std::runtime_error("reconstruct_runner: failed to create compute pipeline for " + spvPath);
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

void dispatchOne(VulkanContext& ctx, VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet set, const void* pushData, uint32_t pushSize,
                  uint32_t totalThreads) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    uint32_t groups = (totalThreads + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);
    // Each dispatch is its own fully-synchronous submission (full queue
    // drain via endSingleTimeCommands' vkQueueWaitIdle) -- deliberately
    // conservative: docs/lux-4d-spec.md section 3's MV-race debugging
    // found that a MoltenVK/Apple-Silicon compute-to-compute dependency
    // chained through several later barriers was NOT always reliably
    // observed by a later stage in the same command buffer, while a full
    // drain between stages always was. Exactness matters far more than
    // per-frame latency for this offline validation tool.
    ctx.endSingleTimeCommands(cmd);
}

struct Meta {
    int s = 2, k = 4, param_stride = 1, hidden = 8;
    int proxy_w = 0, proxy_h = 0, target_w = 0, target_h = 0, net_w = 0, net_h = 0;
    int num_frames = 0;
};

Meta loadMeta(const std::string& path) {
    Meta m;
    auto get = [&](const char* key) -> long { return DlssIO::readJsonIntField(path, key); };
    m.s = static_cast<int>(get("s"));
    m.k = static_cast<int>(get("k"));
    m.param_stride = static_cast<int>(get("param_stride"));
    m.hidden = static_cast<int>(get("hidden"));
    m.proxy_w = static_cast<int>(get("proxy_w"));
    m.proxy_h = static_cast<int>(get("proxy_h"));
    m.target_w = static_cast<int>(get("target_w"));
    m.target_h = static_cast<int>(get("target_h"));
    m.net_w = static_cast<int>(get("net_w"));
    m.net_h = static_cast<int>(get("net_h"));
    m.num_frames = static_cast<int>(get("num_frames"));
    if (m.proxy_w <= 0 || m.proxy_h <= 0 || m.target_w <= 0 || m.target_h <= 0 ||
        m.net_w <= 0 || m.net_h <= 0 || m.num_frames <= 0) {
        throw std::runtime_error("reconstruct_runner: invalid/missing field in " + path);
    }
    return m;
}

} // namespace

int runReconstructDump(const std::string& dumpDir, const std::string& outDirIn,
                        const std::string& pipelineBase) {
    std::string outDir = outDirIn.empty() ? dumpDir : outDirIn;
    fs::create_directories(outDir);

    Meta meta;
    try {
        meta = loadMeta(dumpDir + "/meta.json");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }
    const int S = meta.s, K = meta.k, PS = meta.param_stride, HIDDEN = meta.hidden;
    const int SP = S * PS;
    const int totalCh = SP * SP * K * K + SP * SP + HIDDEN;
    std::cout << "[reconstruct] s=" << S << " k=" << K << " param_stride=" << PS
              << " hidden=" << HIDDEN << " proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " target=" << meta.target_w << "x" << meta.target_h
              << " net=" << meta.net_w << "x" << meta.net_h
              << " frames=" << meta.num_frames << std::endl;

    VulkanContext ctx;
    try {
        ctx.init(false, true, nullptr, false);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }

    // --- Pipelines (3 independent compute stages, each its own set layout
    // matching the StorageBufferDecl order reconstruct_expander.py emits) ---
    VkDescriptorSetLayout warpLayout = makeSetLayout(ctx.device, 3);
    VkDescriptorSetLayout applyLayout = makeSetLayout(ctx.device, 5);
    VkDescriptorSetLayout blendLayout = makeSetLayout(ctx.device, 5);

    VkPipelineLayout warpPL = makePipelineLayout(ctx.device, warpLayout, 16);   // 4 uint
    VkPipelineLayout applyPL = makePipelineLayout(ctx.device, applyLayout, 32); // 6 uint + 2 scalar
    VkPipelineLayout blendPL = makePipelineLayout(ctx.device, blendLayout, 16); // 2 uint (padded)

    VkPipeline warpPipe, applyPipe, blendPipe;
    try {
        warpPipe = makeComputePipeline(ctx.device, warpPL, pipelineBase + ".warp.comp.spv");
        applyPipe = makeComputePipeline(ctx.device, applyPL, pipelineBase + ".apply.comp.spv");
        blendPipe = makeComputePipeline(ctx.device, blendPL, pipelineBase + ".blend.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 + 5 + 5};
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 3;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool);

    VkDescriptorSetLayout layouts[3] = {warpLayout, applyLayout, blendLayout};
    VkDescriptorSetAllocateInfo dsAlloc = {};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = pool;
    dsAlloc.descriptorSetCount = 3;
    dsAlloc.pSetLayouts = layouts;
    VkDescriptorSet sets[3];
    vkAllocateDescriptorSets(ctx.device, &dsAlloc, sets);
    VkDescriptorSet warpSet = sets[0], applySet = sets[1], blendSet = sets[2];

    // --- Buffers ---
    const size_t proxyColorN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 3;
    const size_t mvProxyN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 2;
    const size_t targetColorN = static_cast<size_t>(meta.target_w) * meta.target_h * 3;
    const size_t targetScalarN = static_cast<size_t>(meta.target_w) * meta.target_h;
    const size_t hiddenN = targetScalarN * HIDDEN;
    const size_t packedN = static_cast<size_t>(meta.net_w) * meta.net_h * totalCh;

    VmaAllocation aProxyColor, aMvProxy, aWarped, aPrevColor;
    VmaAllocation aPacked, aSpatial, aAlpha, aHidden;
    VmaAllocation aDisocc, aOutColor;

    VkBuffer bProxyColor = createHostVisibleBuffer(ctx.allocator, proxyColorN * 4, aProxyColor);
    VkBuffer bMvProxy = createHostVisibleBuffer(ctx.allocator, mvProxyN * 4, aMvProxy);
    VkBuffer bWarped = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aWarped);
    VkBuffer bPrevColor = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aPrevColor);
    VkBuffer bPacked = createHostVisibleBuffer(ctx.allocator, packedN * 4, aPacked);
    VkBuffer bSpatial = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aSpatial);
    VkBuffer bAlpha = createHostVisibleBuffer(ctx.allocator, targetScalarN * 4, aAlpha);
    VkBuffer bHidden = createHostVisibleBuffer(ctx.allocator, hiddenN * 4, aHidden);
    VkBuffer bDisocc = createHostVisibleBuffer(ctx.allocator, targetScalarN * 4, aDisocc);
    VkBuffer bOutColor = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aOutColor);

    writeDescriptorSet(ctx.device, warpSet, {bPrevColor, bMvProxy, bWarped});
    writeDescriptorSet(ctx.device, applySet, {bPacked, bProxyColor, bSpatial, bAlpha, bHidden});
    writeDescriptorSet(ctx.device, blendSet, {bSpatial, bWarped, bAlpha, bDisocc, bOutColor});

    // prev_color starts at zero, matching mobiledlss.train.train.rollout's
    // `prev_color = torch.zeros(B, 3, H, W, ...)` initial state.
    zeroBuffer(ctx.allocator, aPrevColor, targetColorN);

    struct WarpPush { uint32_t target_w, target_h, proxy_w, proxy_h; };
    struct ApplyPush {
        uint32_t target_w, target_h, proxy_w, proxy_h, net_w, net_h;
        float jitter_x, jitter_y;
    };
    struct BlendPush { uint32_t target_w, target_h, _pad0, _pad1; };

    WarpPush warpPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                          static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h)};
    BlendPush blendPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h), 0, 0};
    uint32_t totalTargetThreads = static_cast<uint32_t>(targetScalarN);

    for (int t = 0; t < meta.num_frames; ++t) {
        std::string suf = "_f" + std::to_string(t) + ".npy";
        auto proxyColor = DlssIO::readNpyFloat32(dumpDir + "/proxy_color" + suf);
        auto mvProxy = DlssIO::readNpyFloat32(dumpDir + "/mv_proxy" + suf);
        auto jitter = DlssIO::readNpyFloat32(dumpDir + "/jitter" + suf);
        auto packed = DlssIO::readNpyFloat32(dumpDir + "/packed_params" + suf);
        auto disocc = DlssIO::readNpyFloat32(dumpDir + "/disocc" + suf);

        if (proxyColor.data.size() != proxyColorN || mvProxy.data.size() != mvProxyN ||
            jitter.data.size() != 2 || packed.data.size() != packedN ||
            disocc.data.size() != targetScalarN) {
            std::cerr << "[error] frame " << t << ": dump array size mismatch against meta.json"
                       << " (proxy_color=" << proxyColor.data.size() << "/" << proxyColorN
                       << " mv_proxy=" << mvProxy.data.size() << "/" << mvProxyN
                       << " packed_params=" << packed.data.size() << "/" << packedN
                       << " disocc=" << disocc.data.size() << "/" << targetScalarN << ")" << std::endl;
            return 1;
        }

        uploadFloats(ctx.allocator, aProxyColor, proxyColor.data);
        uploadFloats(ctx.allocator, aMvProxy, mvProxy.data);
        uploadFloats(ctx.allocator, aPacked, packed.data);
        uploadFloats(ctx.allocator, aDisocc, disocc.data);

        dispatchOne(ctx, warpPipe, warpPL, warpSet, &warpPush, sizeof(warpPush), totalTargetThreads);

        ApplyPush applyPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                                static_cast<uint32_t>(meta.net_w), static_cast<uint32_t>(meta.net_h),
                                jitter.data[0], jitter.data[1]};
        dispatchOne(ctx, applyPipe, applyPL, applySet, &applyPush, sizeof(applyPush), totalTargetThreads);

        dispatchOne(ctx, blendPipe, blendPL, blendSet, &blendPush, sizeof(blendPush), totalTargetThreads);

        auto outColor = downloadFloats(ctx.allocator, aOutColor, targetColorN);
        auto hidden = downloadFloats(ctx.allocator, aHidden, hiddenN);
        DlssIO::writeNpyFloat32(outDir + "/out_f" + std::to_string(t) + ".npy", outColor,
                                 {meta.target_h, meta.target_w, 3});
        DlssIO::writeNpyFloat32(outDir + "/hidden_f" + std::to_string(t) + ".npy", hidden,
                                 {meta.target_h, meta.target_w, HIDDEN});

        // Carry `out` forward as next frame's `prev_color` (host-side copy
        // -- all buffers are host-visible/coherent VMA_MEMORY_USAGE_CPU_TO_GPU,
        // so no GPU copy command or extra barrier is needed here; the prior
        // dispatch's own endSingleTimeCommands() already fully drained the
        // queue, guaranteeing the CPU sees the GPU's writes).
        copyBufferHost(ctx.allocator, aPrevColor, aOutColor, targetColorN);

        std::cout << "[reconstruct] frame " << t << " done" << std::endl;
    }

    vkDestroyPipeline(ctx.device, warpPipe, nullptr);
    vkDestroyPipeline(ctx.device, applyPipe, nullptr);
    vkDestroyPipeline(ctx.device, blendPipe, nullptr);
    vkDestroyPipelineLayout(ctx.device, warpPL, nullptr);
    vkDestroyPipelineLayout(ctx.device, applyPL, nullptr);
    vkDestroyPipelineLayout(ctx.device, blendPL, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, warpLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, applyLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, blendLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    vmaDestroyBuffer(ctx.allocator, bProxyColor, aProxyColor);
    vmaDestroyBuffer(ctx.allocator, bMvProxy, aMvProxy);
    vmaDestroyBuffer(ctx.allocator, bWarped, aWarped);
    vmaDestroyBuffer(ctx.allocator, bPrevColor, aPrevColor);
    vmaDestroyBuffer(ctx.allocator, bPacked, aPacked);
    vmaDestroyBuffer(ctx.allocator, bSpatial, aSpatial);
    vmaDestroyBuffer(ctx.allocator, bAlpha, aAlpha);
    vmaDestroyBuffer(ctx.allocator, bHidden, aHidden);
    vmaDestroyBuffer(ctx.allocator, bDisocc, aDisocc);
    vmaDestroyBuffer(ctx.allocator, bOutColor, aOutColor);

    std::cout << "[reconstruct] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}
