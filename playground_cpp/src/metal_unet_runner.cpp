#include "metal_unet_runner.h"
#include "metal_context.h"
#include "metal_shader_transpiler.h"
#include "dlss_io.h"

#include <Metal/Metal.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>

namespace fs = std::filesystem;

namespace {

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
    if (!f.is_open()) throw std::runtime_error("metal unet_runner: cannot open " + path);
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
    if (!f.is_open()) throw std::runtime_error("metal unet_runner: cannot open " + path);
    size_t sizeBytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    size_t count = sizeBytes / 2;
    std::vector<uint16_t> raw(count);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(sizeBytes));
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) out[i] = DlssIO::halfToFloat(raw[i]);
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
        throw std::runtime_error("metal unet_runner: failed to create pipeline for " + spvPath + ": " + msg);
    }
    return cs;
}

double dispatchOne(MetalContext& ctx, CompiledStage& stage, const std::vector<MTL::Buffer*>& buffers,
                    const void* pushData, uint32_t pushSize, uint32_t totalThreads) {
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
    return (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;
}

} // namespace

int runUnetDumpMetal(const std::string& inputNpyPath, const std::string& weightsBlobPath,
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
        std::cerr << "[error] metal unet_runner: input channel mismatch" << std::endl;
        return 1;
    }
    uint32_t netH = static_cast<uint32_t>(input.shape[0]);
    uint32_t netW = static_cast<uint32_t>(input.shape[1]);

    MetalContext ctx;
    ctx.initHeadless();
    ShaderTranspiler transpiler;
    transpiler.init(ctx.device);

    CompiledStage sConv3x3, sConv3x3S2, sUpConcat, sConv1x1;
    try {
        sConv3x3 = compileStage(ctx, transpiler, kernelPipelineDir + "/unet_conv3x3_lrelu.comp.spv");
        sConv3x3S2 = compileStage(ctx, transpiler, kernelPipelineDir + "/unet_conv3x3_s2_lrelu.comp.spv");
        sUpConcat = compileStage(ctx, transpiler, kernelPipelineDir + "/unet_upsample_concat_conv_lrelu.comp.spv");
        sConv1x1 = compileStage(ctx, transpiler, kernelPipelineDir + "/unet_conv1x1.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    MTL::Buffer* bWeights = ctx.newBuffer(weightsHost.size() * sizeof(float), MTL::ResourceStorageModeShared);
    std::memcpy(bWeights->contents(), weightsHost.data(), weightsHost.size() * sizeof(float));

    size_t maxElems = static_cast<size_t>(netH) * netW * 256;
    MTL::Buffer* bufA = ctx.newBuffer(maxElems * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* bufB = ctx.newBuffer(maxElems * sizeof(float), MTL::ResourceStorageModeShared);
    std::memcpy(bufA->contents(), input.data.data(), input.data.size() * sizeof(float));

    struct SkipBuf { MTL::Buffer* buf; uint32_t h, w, c; };
    std::vector<SkipBuf> skipStack;

    MTL::Buffer* curBuf = bufA;
    MTL::Buffer* otherBuf = bufB;
    uint32_t curH = netH, curW = netW;

    struct Push3 { uint32_t a, b, cin, cout, weightOffset, biasOffset; };
    struct Push4 { uint32_t outH, outW, cinA, cinB, cout, weightOffset, biasOffset; };

    double totalMs = 0;
    for (size_t i = 0; i < manifest.layers.size(); ++i) {
        const Layer& layer = manifest.layers[i];
        double ms;
        uint32_t newH = curH, newW = curW;

        if (layer.type == "conv3x3_lrelu" || layer.type == "conv1x1") {
            Push3 push = {curH, curW, layer.cin, layer.cout, layer.weightOffset, layer.biasOffset};
            CompiledStage& stage = (layer.type == "conv1x1") ? sConv1x1 : sConv3x3;
            ms = dispatchOne(ctx, stage, {curBuf, bWeights, otherBuf}, &push, sizeof(push), curH * curW * layer.cout);
        } else if (layer.type == "conv3x3_s2_lrelu") {
            Push3 push = {curH, curW, layer.cin, layer.cout, layer.weightOffset, layer.biasOffset};
            newH = curH / 2; newW = curW / 2;
            ms = dispatchOne(ctx, sConv3x3S2, {curBuf, bWeights, otherBuf}, &push, sizeof(push), newH * newW * layer.cout);
        } else if (layer.type == "upsample_concat_conv_lrelu") {
            SkipBuf skip = skipStack.back();
            skipStack.pop_back();
            uint32_t cinB = skip.c;
            uint32_t cinA = layer.cin - cinB;
            Push4 push = {skip.h, skip.w, cinA, cinB, layer.cout, layer.weightOffset, layer.biasOffset};
            newH = skip.h; newW = skip.w;
            ms = dispatchOne(ctx, sUpConcat, {curBuf, skip.buf, bWeights, otherBuf}, &push, sizeof(push),
                              newH * newW * layer.cout);
        } else {
            std::cerr << "[error] metal unet_runner: unknown layer type " << layer.type << std::endl;
            return 1;
        }

        totalMs += ms;
        std::cout << "[unet-metal] layer " << i << " (" << layer.name << ", " << layer.type << "): "
                  << ms << " ms" << std::endl;

        curH = newH; curW = newW;
        std::swap(curBuf, otherBuf);

        if (i == 0 || i == 2 || i == 4) {
            size_t n = static_cast<size_t>(curH) * curW * layer.cout;
            MTL::Buffer* skipBuf = ctx.newBuffer(n * sizeof(float), MTL::ResourceStorageModeShared);
            std::memcpy(skipBuf->contents(), curBuf->contents(), n * sizeof(float));
            skipStack.push_back({skipBuf, curH, curW, layer.cout});
        }
    }

    std::cout << "[unet-metal] total GPU dispatch time: " << totalMs << " ms" << std::endl;

    size_t outN = static_cast<size_t>(curH) * curW * manifest.outChannels;
    std::vector<float> outData(outN);
    std::memcpy(outData.data(), curBuf->contents(), outN * sizeof(float));
    DlssIO::writeNpyFloat32(outputNpyPath, outData, {curH, curW, manifest.outChannels});
    std::cout << "[unet-metal] wrote " << outputNpyPath << " (" << curH << "x" << curW << "x"
              << manifest.outChannels << ")" << std::endl;

    return 0;
}
