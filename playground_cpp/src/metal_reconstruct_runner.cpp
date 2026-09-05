#include "metal_reconstruct_runner.h"
#include "metal_context.h"
#include "metal_shader_transpiler.h"
#include "dlss_io.h"

#include <Metal/Metal.hpp>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <vector>

namespace fs = std::filesystem;

namespace {

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
        throw std::runtime_error("metal reconstruct_runner: invalid/missing field in " + path);
    }
    return m;
}

struct CompiledStage {
    TranspiledShader shader;
    MTL::ComputePipelineState* pipeline = nullptr;
};

CompiledStage compileStage(MetalContext& ctx, ShaderTranspiler& transpiler, const std::string& spvPath) {
    CompiledStage cs;
    transpiler.transpileInto(cs.shader, spvPath, SpvExecModel::GLCompute);
    NS::Error* error = nullptr;
    cs.pipeline = ctx.device->newComputePipelineState(cs.shader.function, &error);
    if (!cs.pipeline) {
        std::string msg = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error("metal reconstruct_runner: failed to create pipeline for " +
                                  spvPath + ": " + msg);
    }
    return cs;
}

void dispatchOne(MetalContext& ctx, CompiledStage& stage, const std::vector<MTL::Buffer*>& buffers,
                  const void* pushData, uint32_t pushSize, uint32_t totalThreads,
                  double* gpuMsOut) {
    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(stage.pipeline);
    for (size_t i = 0; i < buffers.size(); ++i) {
        uint32_t idx = stage.shader.findBufferIndex(0, static_cast<uint32_t>(i));
        if (idx != UINT32_MAX) enc->setBuffer(buffers[i], 0, idx);
    }
    if (stage.shader.pushConstantBufferIndex != UINT32_MAX) {
        enc->setBytes(pushData, pushSize, stage.shader.pushConstantBufferIndex);
    }
    uint32_t tgSize = 256;
    uint32_t groups = (totalThreads + tgSize - 1) / tgSize;
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(tgSize, 1, 1));
    enc->endEncoding();
    ctx.submitAndWait(cmdBuf);
    if (gpuMsOut) {
        *gpuMsOut = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
    }
}

} // namespace

int runReconstructDumpMetal(const std::string& dumpDir, const std::string& outDirIn,
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
    std::cout << "[reconstruct-metal] s=" << S << " k=" << K << " param_stride=" << PS
              << " hidden=" << HIDDEN << " proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " target=" << meta.target_w << "x" << meta.target_h
              << " net=" << meta.net_w << "x" << meta.net_h
              << " frames=" << meta.num_frames << std::endl;

    MetalContext ctx;
    ctx.initHeadless();

    ShaderTranspiler transpiler;
    transpiler.init(ctx.device);

    CompiledStage warpStage, applyStage, blendStage;
    try {
        warpStage = compileStage(ctx, transpiler, pipelineBase + ".warp.comp.spv");
        applyStage = compileStage(ctx, transpiler, pipelineBase + ".apply.comp.spv");
        blendStage = compileStage(ctx, transpiler, pipelineBase + ".blend.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    const size_t proxyColorN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 3;
    const size_t mvProxyN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 2;
    const size_t targetColorN = static_cast<size_t>(meta.target_w) * meta.target_h * 3;
    const size_t targetScalarN = static_cast<size_t>(meta.target_w) * meta.target_h;
    const size_t hiddenN = targetScalarN * HIDDEN;
    const size_t packedN = static_cast<size_t>(meta.net_w) * meta.net_h * totalCh;

    auto mkBuf = [&](size_t numFloats) {
        return ctx.newBuffer(std::max<size_t>(numFloats, 4) * sizeof(float), MTL::ResourceStorageModeShared);
    };
    MTL::Buffer* bProxyColor = mkBuf(proxyColorN);
    MTL::Buffer* bMvProxy = mkBuf(mvProxyN);
    MTL::Buffer* bWarped = mkBuf(targetColorN);
    MTL::Buffer* bPrevColor = mkBuf(targetColorN);
    MTL::Buffer* bPacked = mkBuf(packedN);
    MTL::Buffer* bSpatial = mkBuf(targetColorN);
    MTL::Buffer* bAlpha = mkBuf(targetScalarN);
    MTL::Buffer* bHidden = mkBuf(hiddenN);
    MTL::Buffer* bDisocc = mkBuf(targetScalarN);
    MTL::Buffer* bOutColor = mkBuf(targetColorN);

    std::memset(bPrevColor->contents(), 0, targetColorN * sizeof(float));

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

    double sumWarpMs = 0, sumApplyMs = 0, sumBlendMs = 0;

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
            std::cerr << "[error] frame " << t << ": dump array size mismatch against meta.json" << std::endl;
            return 1;
        }

        std::memcpy(bProxyColor->contents(), proxyColor.data.data(), proxyColorN * sizeof(float));
        std::memcpy(bMvProxy->contents(), mvProxy.data.data(), mvProxyN * sizeof(float));
        std::memcpy(bPacked->contents(), packed.data.data(), packedN * sizeof(float));
        std::memcpy(bDisocc->contents(), disocc.data.data(), targetScalarN * sizeof(float));

        double warpMs = 0, applyMs = 0, blendMs = 0;
        dispatchOne(ctx, warpStage, {bPrevColor, bMvProxy, bWarped}, &warpPush, sizeof(warpPush),
                    totalTargetThreads, &warpMs);

        ApplyPush applyPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                                static_cast<uint32_t>(meta.net_w), static_cast<uint32_t>(meta.net_h),
                                jitter.data[0], jitter.data[1]};
        dispatchOne(ctx, applyStage, {bPacked, bProxyColor, bSpatial, bAlpha, bHidden}, &applyPush,
                    sizeof(applyPush), totalTargetThreads, &applyMs);

        dispatchOne(ctx, blendStage, {bSpatial, bWarped, bAlpha, bDisocc, bOutColor}, &blendPush,
                    sizeof(blendPush), totalTargetThreads, &blendMs);

        sumWarpMs += warpMs; sumApplyMs += applyMs; sumBlendMs += blendMs;

        std::vector<float> outColor(targetColorN);
        std::memcpy(outColor.data(), bOutColor->contents(), targetColorN * sizeof(float));
        std::vector<float> hidden(hiddenN);
        std::memcpy(hidden.data(), bHidden->contents(), hiddenN * sizeof(float));
        DlssIO::writeNpyFloat32(outDir + "/out_f" + std::to_string(t) + ".npy", outColor,
                                 {meta.target_h, meta.target_w, 3});
        DlssIO::writeNpyFloat32(outDir + "/hidden_f" + std::to_string(t) + ".npy", hidden,
                                 {meta.target_h, meta.target_w, HIDDEN});

        std::memcpy(bPrevColor->contents(), bOutColor->contents(), targetColorN * sizeof(float));

        std::cout << "[reconstruct-metal] frame " << t << " done"
                  << " (warp=" << warpMs << "ms apply=" << applyMs << "ms blend=" << blendMs << "ms)"
                  << std::endl;
    }

    int T = meta.num_frames;
    std::cout << "[reconstruct-metal] mean GPU time over " << T << " frames at "
              << meta.target_w << "x" << meta.target_h << ": warp=" << (sumWarpMs / T)
              << "ms apply=" << (sumApplyMs / T) << "ms blend=" << (sumBlendMs / T)
              << "ms total=" << ((sumWarpMs + sumApplyMs + sumBlendMs) / T) << "ms" << std::endl;

    std::cout << "[reconstruct-metal] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}
