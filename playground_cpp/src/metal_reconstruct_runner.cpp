#include "metal_reconstruct_runner.h"
#include "metal_context.h"
#include "metal_shader_transpiler.h"
#include "dlss_io.h"

#include <Metal/Metal.hpp>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <vector>
#include <array>

namespace fs = std::filesystem;

namespace {

struct Meta {
    int s = 2, k = 4, param_stride = 1, hidden = 8;
    int proxy_w = 0, proxy_h = 0, target_w = 0, target_h = 0, net_w = 0, net_h = 0;
    int num_frames = 0;
    // Scene-memory path (SPECIFICATION.md 12.9's `memory: { channels,
    // hidden }` sub-block) -- memory_channels == 0 means the compiled
    // pipeline (and this dump) has no memory path, exactly the stage-1
    // 2-way blend.
    int memory_channels = 0, memory_hidden = 0, tex_w = 0, tex_h = 0;
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
    long mc = get("memory_channels");
    long mh = get("memory_hidden");
    long tw = get("tex_w");
    long th = get("tex_h");
    if (mc > 0 && mh > 0 && tw > 0 && th > 0) {
        m.memory_channels = static_cast<int>(mc);
        m.memory_hidden = static_cast<int>(mh);
        m.tex_w = static_cast<int>(tw);
        m.tex_h = static_cast<int>(th);
    }
    return m;
}

// numpy's `.npy`/`Camera.camtoworld` convention is row-major "math"
// indexing (flat[i*4+j] == M[row i, col j]); GLM/GLSL/SPIR-V mat4 push
// constants are column-major memory layout (flat[col*4+row] == M[row,col],
// see splat_renderer.cpp's own `std::memcpy(push.view, &viewMatrix_[0][0],
// 64)` -- GLM's raw storage already matches this directly). Loading a
// row-major .npy straight into a push constant without transposing would
// silently feed the shader M^T instead of M.
std::array<float, 16> transposeRowMajorToColMajor4x4(const std::vector<float>& rowMajor16) {
    std::array<float, 16> out{};
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            out[col * 4 + row] = rowMajor16[row * 4 + col];
    return out;
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

struct BguvPush {
    uint32_t width, height;
    float _pad0[2];
    float k_params[4];       // fx, fy, cx, cy
    float cam_to_world[16];  // column-major (GLM convention)
    float bg_sphere[4];      // centre.xyz, radius
};
static_assert(sizeof(BguvPush) == 112, "BguvPush must match reconstruct_expander.py's bguv push layout");

struct MemoryPush {
    uint32_t width, height, tex_w, tex_h;
};
static_assert(sizeof(MemoryPush) == 16, "MemoryPush must match reconstruct_expander.py's memory push layout");

struct WarpPush { uint32_t target_w, target_h, proxy_w, proxy_h; };
struct ApplyPush {
    uint32_t target_w, target_h, proxy_w, proxy_h, net_w, net_h;
    float jitter_x, jitter_y;
};
struct BlendPush { uint32_t target_w, target_h, _pad0, _pad1; };

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
    const bool hasMemory = meta.memory_channels > 0;
    const int NBLEND = hasMemory ? 3 : 1;
    const int totalCh = SP * SP * K * K + SP * SP * NBLEND + HIDDEN;
    const int MC = meta.memory_channels, MH = meta.memory_hidden, TEXW = meta.tex_w, TEXH = meta.tex_h;
    std::cout << "[reconstruct-metal] s=" << S << " k=" << K << " param_stride=" << PS
              << " hidden=" << HIDDEN << " proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " target=" << meta.target_w << "x" << meta.target_h
              << " net=" << meta.net_w << "x" << meta.net_h
              << " frames=" << meta.num_frames
              << (hasMemory ? (" memory(channels=" + std::to_string(MC) + " hidden=" + std::to_string(MH) +
                                " tex=" + std::to_string(TEXW) + "x" + std::to_string(TEXH) + ")") : "")
              << std::endl;

    MetalContext ctx;
    ctx.initHeadless();

    ShaderTranspiler transpiler;
    transpiler.init(ctx.device);

    CompiledStage warpStage, applyStage, blendStage, bguvStage, memoryStage;
    try {
        warpStage = compileStage(ctx, transpiler, pipelineBase + ".warp.comp.spv");
        applyStage = compileStage(ctx, transpiler, pipelineBase + ".apply.comp.spv");
        blendStage = compileStage(ctx, transpiler, pipelineBase + ".blend.comp.spv");
        if (hasMemory) {
            bguvStage = compileStage(ctx, transpiler, pipelineBase + ".bguv.comp.spv");
            memoryStage = compileStage(ctx, transpiler, pipelineBase + ".memory.comp.spv");
        }
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
    const size_t blendN = targetScalarN * NBLEND;  // alpha_out (N=1) or blend_out (N=3)

    auto mkBuf = [&](size_t numFloats) {
        return ctx.newBuffer(std::max<size_t>(numFloats, 4) * sizeof(float), MTL::ResourceStorageModeShared);
    };
    MTL::Buffer* bProxyColor = mkBuf(proxyColorN);
    MTL::Buffer* bMvProxy = mkBuf(mvProxyN);
    MTL::Buffer* bWarped = mkBuf(targetColorN);
    MTL::Buffer* bPrevColor = mkBuf(targetColorN);
    MTL::Buffer* bPacked = mkBuf(packedN);
    MTL::Buffer* bSpatial = mkBuf(targetColorN);
    MTL::Buffer* bBlend = mkBuf(blendN);
    MTL::Buffer* bHidden = mkBuf(hiddenN);
    MTL::Buffer* bDisocc = mkBuf(targetScalarN);
    MTL::Buffer* bOutColor = mkBuf(targetColorN);

    std::memset(bPrevColor->contents(), 0, targetColorN * sizeof(float));

    // --- Scene-memory buffers (loaded once -- texture/decoder weights are
    // constant for the whole dump; bguv/memory_color are recomputed every
    // frame from that frame's camera). ---
    MTL::Buffer* bBguvTarget = nullptr;
    MTL::Buffer* bTexture = nullptr;
    MTL::Buffer* bFc1W = nullptr; MTL::Buffer* bFc1B = nullptr;
    MTL::Buffer* bFc2W = nullptr; MTL::Buffer* bFc2B = nullptr;
    MTL::Buffer* bBgFeaturesDummy = nullptr;  // memory stage's proxy-feature output, unused at target res
    MTL::Buffer* bMemoryColor = nullptr;
    std::array<float, 4> bgSphere{};

    if (hasMemory) {
        auto tex = DlssIO::readNpyFloat32(dumpDir + "/texture.npy");
        auto sph = DlssIO::readNpyFloat32(dumpDir + "/bg_sphere.npy");
        if (tex.data.size() != static_cast<size_t>(MC) * TEXH * TEXW || sph.data.size() != 4) {
            std::cerr << "[error] texture.npy/bg_sphere.npy size mismatch against meta.json" << std::endl;
            return 1;
        }
        std::copy(sph.data.begin(), sph.data.end(), bgSphere.begin());

        auto fc1w = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc1_w");
        auto fc1b = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc1_b");
        auto fc2w = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc2_w");
        auto fc2b = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc2_b");
        if (fc1w.data.size() != static_cast<size_t>(MH) * MC || fc1b.data.size() != static_cast<size_t>(MH) ||
            fc2w.data.size() != static_cast<size_t>(3) * MH || fc2b.data.size() != 3) {
            std::cerr << "[error] memory_head.npz array size mismatch against meta.json" << std::endl;
            return 1;
        }

        bTexture = mkBuf(tex.data.size());
        std::memcpy(bTexture->contents(), tex.data.data(), tex.data.size() * sizeof(float));
        bFc1W = mkBuf(fc1w.data.size());
        std::memcpy(bFc1W->contents(), fc1w.data.data(), fc1w.data.size() * sizeof(float));
        bFc1B = mkBuf(fc1b.data.size());
        std::memcpy(bFc1B->contents(), fc1b.data.data(), fc1b.data.size() * sizeof(float));
        bFc2W = mkBuf(fc2w.data.size());
        std::memcpy(bFc2W->contents(), fc2w.data.data(), fc2w.data.size() * sizeof(float));
        bFc2B = mkBuf(fc2b.data.size());
        std::memcpy(bFc2B->contents(), fc2b.data.data(), fc2b.data.size() * sizeof(float));

        bBguvTarget = mkBuf(targetScalarN * 2);
        bBgFeaturesDummy = mkBuf(targetScalarN * static_cast<size_t>(MC));
        bMemoryColor = mkBuf(targetColorN);
    }

    WarpPush warpPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                          static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h)};
    BlendPush blendPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h), 0, 0};
    uint32_t totalTargetThreads = static_cast<uint32_t>(targetScalarN);

    double sumWarpMs = 0, sumApplyMs = 0, sumBlendMs = 0, sumBguvMs = 0, sumMemoryMs = 0;

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

        double warpMs = 0, applyMs = 0, blendMs = 0, bguvMs = 0, memoryMs = 0;

        if (hasMemory) {
            auto kParams = DlssIO::readNpyFloat32(dumpDir + "/k_params" + suf);
            auto camToWorld = DlssIO::readNpyFloat32(dumpDir + "/cam_to_world" + suf);
            if (kParams.data.size() != 4 || camToWorld.data.size() != 16) {
                std::cerr << "[error] frame " << t << ": k_params/cam_to_world size mismatch" << std::endl;
                return 1;
            }
            BguvPush bguvPush{};
            bguvPush.width = static_cast<uint32_t>(meta.target_w);
            bguvPush.height = static_cast<uint32_t>(meta.target_h);
            std::copy(kParams.data.begin(), kParams.data.end(), bguvPush.k_params);
            auto colMajor = transposeRowMajorToColMajor4x4(camToWorld.data);
            std::copy(colMajor.begin(), colMajor.end(), bguvPush.cam_to_world);
            std::copy(bgSphere.begin(), bgSphere.end(), bguvPush.bg_sphere);

            dispatchOne(ctx, bguvStage, {bBguvTarget}, &bguvPush, sizeof(bguvPush),
                        totalTargetThreads, &bguvMs);

            MemoryPush memPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                   static_cast<uint32_t>(TEXW), static_cast<uint32_t>(TEXH)};
            dispatchOne(ctx, memoryStage,
                        {bBguvTarget, bTexture, bFc1W, bFc1B, bFc2W, bFc2B, bBgFeaturesDummy, bMemoryColor},
                        &memPush, sizeof(memPush), totalTargetThreads, &memoryMs);
        }

        dispatchOne(ctx, warpStage, {bPrevColor, bMvProxy, bWarped}, &warpPush, sizeof(warpPush),
                    totalTargetThreads, &warpMs);

        ApplyPush applyPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                                static_cast<uint32_t>(meta.net_w), static_cast<uint32_t>(meta.net_h),
                                jitter.data[0], jitter.data[1]};
        dispatchOne(ctx, applyStage, {bPacked, bProxyColor, bSpatial, bBlend, bHidden}, &applyPush,
                    sizeof(applyPush), totalTargetThreads, &applyMs);

        if (hasMemory) {
            dispatchOne(ctx, blendStage, {bSpatial, bWarped, bBlend, bDisocc, bMemoryColor, bOutColor},
                        &blendPush, sizeof(blendPush), totalTargetThreads, &blendMs);
        } else {
            dispatchOne(ctx, blendStage, {bSpatial, bWarped, bBlend, bDisocc, bOutColor}, &blendPush,
                        sizeof(blendPush), totalTargetThreads, &blendMs);
        }

        sumWarpMs += warpMs; sumApplyMs += applyMs; sumBlendMs += blendMs;
        sumBguvMs += bguvMs; sumMemoryMs += memoryMs;

        std::vector<float> outColor(targetColorN);
        std::memcpy(outColor.data(), bOutColor->contents(), targetColorN * sizeof(float));
        std::vector<float> hidden(hiddenN);
        std::memcpy(hidden.data(), bHidden->contents(), hiddenN * sizeof(float));
        DlssIO::writeNpyFloat32(outDir + "/out_f" + std::to_string(t) + ".npy", outColor,
                                 {meta.target_h, meta.target_w, 3});
        DlssIO::writeNpyFloat32(outDir + "/hidden_f" + std::to_string(t) + ".npy", hidden,
                                 {meta.target_h, meta.target_w, HIDDEN});
        if (hasMemory) {
            std::vector<float> bguvOut(targetScalarN * 2);
            std::memcpy(bguvOut.data(), bBguvTarget->contents(), targetScalarN * 2 * sizeof(float));
            DlssIO::writeNpyFloat32(outDir + "/bguv_target_f" + std::to_string(t) + ".npy", bguvOut,
                                     {meta.target_h, meta.target_w, 2});
        }

        std::memcpy(bPrevColor->contents(), bOutColor->contents(), targetColorN * sizeof(float));

        std::cout << "[reconstruct-metal] frame " << t << " done"
                  << " (warp=" << warpMs << "ms apply=" << applyMs << "ms blend=" << blendMs << "ms"
                  << (hasMemory ? (" bguv=" + std::to_string(bguvMs) + "ms memory=" + std::to_string(memoryMs) + "ms") : "")
                  << ")" << std::endl;
    }

    int T = meta.num_frames;
    double totalPerFrame = (sumWarpMs + sumApplyMs + sumBlendMs + sumBguvMs + sumMemoryMs) / T;
    std::cout << "[reconstruct-metal] mean GPU time over " << T << " frames at "
              << meta.target_w << "x" << meta.target_h << ": warp=" << (sumWarpMs / T)
              << "ms apply=" << (sumApplyMs / T) << "ms blend=" << (sumBlendMs / T)
              << (hasMemory ? ("ms bguv=" + std::to_string(sumBguvMs / T) +
                                "ms memory=" + std::to_string(sumMemoryMs / T)) : "")
              << "ms total=" << totalPerFrame << "ms" << std::endl;

    std::cout << "[reconstruct-metal] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}

int runDumpBgFeaturesMetal(const std::string& dumpDir, const std::string& outDirIn,
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
    if (meta.memory_channels <= 0) {
        std::cerr << "[error] --dump-bg-features: dump directory's meta.json has no memory_channels "
                     "(this dump/pipeline has no scene-memory path)" << std::endl;
        return 1;
    }
    const int MC = meta.memory_channels, TEXW = meta.tex_w, TEXH = meta.tex_h;
    std::cout << "[dump-bg-features-metal] proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " channels=" << MC << " tex=" << TEXW << "x" << TEXH
              << " frames=" << meta.num_frames << std::endl;

    MetalContext ctx;
    ctx.initHeadless();
    ShaderTranspiler transpiler;
    transpiler.init(ctx.device);

    CompiledStage bguvStage, memoryStage;
    try {
        bguvStage = compileStage(ctx, transpiler, pipelineBase + ".bguv.comp.spv");
        memoryStage = compileStage(ctx, transpiler, pipelineBase + ".memory.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    auto tex = DlssIO::readNpyFloat32(dumpDir + "/texture.npy");
    auto sph = DlssIO::readNpyFloat32(dumpDir + "/bg_sphere.npy");
    if (tex.data.size() != static_cast<size_t>(MC) * TEXH * TEXW || sph.data.size() != 4) {
        std::cerr << "[error] texture.npy/bg_sphere.npy size mismatch against meta.json" << std::endl;
        return 1;
    }
    std::array<float, 4> bgSphere{};
    std::copy(sph.data.begin(), sph.data.end(), bgSphere.begin());

    auto mkBuf = [&](size_t numFloats) {
        return ctx.newBuffer(std::max<size_t>(numFloats, 4) * sizeof(float), MTL::ResourceStorageModeShared);
    };
    const size_t proxyScalarN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h;
    MTL::Buffer* bTexture = mkBuf(tex.data.size());
    std::memcpy(bTexture->contents(), tex.data.data(), tex.data.size() * sizeof(float));
    // Decoder weights aren't needed for this proxy-feature-only dump, but
    // the memory stage's binding layout is fixed -- bind zero-sized dummies.
    MTL::Buffer* bFc1W = mkBuf(1); MTL::Buffer* bFc1B = mkBuf(1);
    MTL::Buffer* bFc2W = mkBuf(1); MTL::Buffer* bFc2B = mkBuf(1);
    MTL::Buffer* bBguvProxy = mkBuf(proxyScalarN * 2);
    MTL::Buffer* bBgFeatures = mkBuf(proxyScalarN * static_cast<size_t>(MC));
    MTL::Buffer* bMemoryColorDummy = mkBuf(proxyScalarN * 3);

    uint32_t totalProxyThreads = static_cast<uint32_t>(proxyScalarN);

    for (int t = 0; t < meta.num_frames; ++t) {
        std::string suf = "_f" + std::to_string(t) + ".npy";
        auto kParams = DlssIO::readNpyFloat32(dumpDir + "/k_params_proxy" + suf);
        auto camToWorld = DlssIO::readNpyFloat32(dumpDir + "/cam_to_world" + suf);
        if (kParams.data.size() != 4 || camToWorld.data.size() != 16) {
            std::cerr << "[error] frame " << t << ": k_params_proxy/cam_to_world size mismatch" << std::endl;
            return 1;
        }
        BguvPush bguvPush{};
        bguvPush.width = static_cast<uint32_t>(meta.proxy_w);
        bguvPush.height = static_cast<uint32_t>(meta.proxy_h);
        std::copy(kParams.data.begin(), kParams.data.end(), bguvPush.k_params);
        auto colMajor = transposeRowMajorToColMajor4x4(camToWorld.data);
        std::copy(colMajor.begin(), colMajor.end(), bguvPush.cam_to_world);
        std::copy(bgSphere.begin(), bgSphere.end(), bguvPush.bg_sphere);

        dispatchOne(ctx, bguvStage, {bBguvProxy}, &bguvPush, sizeof(bguvPush), totalProxyThreads, nullptr);

        MemoryPush memPush = {static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                               static_cast<uint32_t>(TEXW), static_cast<uint32_t>(TEXH)};
        dispatchOne(ctx, memoryStage,
                    {bBguvProxy, bTexture, bFc1W, bFc1B, bFc2W, bFc2B, bBgFeatures, bMemoryColorDummy},
                    &memPush, sizeof(memPush), totalProxyThreads, nullptr);

        std::vector<float> feat(proxyScalarN * static_cast<size_t>(MC));
        std::memcpy(feat.data(), bBgFeatures->contents(), feat.size() * sizeof(float));
        DlssIO::writeNpyFloat32(outDir + "/bg_features_f" + std::to_string(t) + ".npy", feat,
                                 {meta.proxy_h, meta.proxy_w, MC});
        std::cout << "[dump-bg-features-metal] frame " << t << " done" << std::endl;
    }

    std::cout << "[dump-bg-features-metal] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}
