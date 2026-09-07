#pragma once

// Stage B3: MPSGraph port of mobiledlss.train.model.ParamPredUNet's
// forward_packed, ported from bench/mpsgraph/bench.mm's graph construction
// (NHWC activations / HWIO weights, LeakyReLU 0.1, floor-mode nearest
// upsample matching PyTorch exactly -- see MPSGraphUNet.mm's kernel-source
// comments for the derivation). Weights from
// demo/ios_assets/exported/unet_weights.fp16.bin + .layers.txt (mobiledlss's
// tools/export_lux_weights.py format, the same sidecar
// playground_cpp/src/metal_unet_runner.cpp already parses).
//
// Runs at the *padded* net resolution NetInputAssembly reports
// (getNetW()/getNetH() -- e.g. 240x136 for the juggle scene's 480x270
// proxy at param_stride=2), encoded into the caller's own MTLCommandBuffer
// (wrapped as an MPSCommandBuffer) so it shares the frame's command buffer
// with the splat render + input assembly passes.

#include <Metal/Metal.hpp>
#include <string>
#include <cstdint>

class MetalContext;

class MPSGraphUNet {
public:
    ~MPSGraphUNet();

    // `weightsBinPath`/`layersTxtPath`: exported/unet_weights.fp16.bin and
    // .layers.txt. `netW`/`netH`: the *padded* net resolution (must be
    // divisible by 8 -- NetInputAssembly::getNetW()/getNetH()).
    void init(MetalContext& ctx, const std::string& weightsBinPath,
              const std::string& layersTxtPath, uint32_t netW, uint32_t netH);

    // Encodes the whole graph into `cmdBuf` (a metal-cpp MTL::CommandBuffer*,
    // bridged to id<MTLCommandBuffer> internally -- see .mm) reading
    // `inputBuffer` (fp16 NHWC, netW*netH*inChannels -- NetInputAssembly's
    // output buffer, used directly with no copy) and writing `outputBuffer`
    // (fp16 NHWC, netW*netH*outChannels -- caller-allocated).
    void encode(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Buffer* inputBuffer,
                MTL::Buffer* outputBuffer);

    uint32_t getInChannels() const { return inChannels_; }
    uint32_t getOutChannels() const { return outChannels_; }
    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    uint32_t getK() const { return k_; }
    uint32_t getHiddenChannels() const { return hidden_; }
    uint32_t getUpscale() const { return upscale_; }
    uint32_t getParamStride() const { return paramStride_; }

private:
    void* graph_ = nullptr;          // MPSGraph* (opaque -- keeps this header plain C++)
    void* inputTensor_ = nullptr;    // MPSGraphTensor*
    void* outputTensor_ = nullptr;   // MPSGraphTensor*
    void* executable_ = nullptr;     // MPSGraphExecutable* (compiled once in init())

    uint32_t netW_ = 0, netH_ = 0;
    uint32_t inChannels_ = 0, outChannels_ = 0;
    uint32_t k_ = 0, hidden_ = 0, upscale_ = 0, paramStride_ = 0;
};
