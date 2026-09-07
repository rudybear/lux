#pragma once

// Live-inference stage: MPSGraph port of mobiledlss.train.model.ParamPredUNet's
// forward_packed (NHWC activations / HWIO weights, LeakyReLU 0.1, floor-mode
// nearest upsample matching PyTorch exactly). Weights from
// demo/ios_assets/exported/unet_weights.fp16.bin + .layers.txt (mobiledlss's
// tools/export_lux_weights.py format, the same sidecar
// playground_cpp/src/metal_unet_runner.cpp already parses).
//
// Moved from playground_ios/Source/MPSGraphUNet.{h,mm} into playground_cpp/src
// so the macOS `--live-bench` CLI mode and the iOS live demo share the exact
// same validated graph -- MPSGraph/MetalPerformanceShadersGraph is available
// on macOS too (Metal.framework + MetalPerformanceShadersGraph.framework),
// only the surrounding driver (metal_live_reconstruct.*) differs per host.
//
// Runs at the *padded* net resolution NetInputAssembly reports (getNetW()/
// getNetH()).
//
// encode()'s contract (changed from the iOS-only version, which always
// committed+waited internally before returning): it encodes into a
// caller-supplied MTL::CommandBuffer* WITHOUT committing or waiting, and
// RETURNS the command buffer the caller must keep encoding into afterwards.
// This is required because MPSGraphExecutable::encodeToCommandBuffer: may
// internally call [MPSCommandBuffer commitAndContinue], which commits the
// work encoded so far and swaps in a NEW underlying MTLCommandBuffer under
// the hood (see Apple's MPSCommandBuffer.h doc comment) -- the original
// object the caller passed in may no longer be the "live" one. A caller that
// wants to encode further work depending on this pass's output (the
// reconstruct pass, here) MUST use the returned pointer, not its own
// original one, and must commit/wait on whichever buffer the LAST such call
// in the frame returns.
//
// Ownership: the returned pointer carries an EXTRA retain the caller now
// owns (see the .mm's comment on the swapped-root case -- a plain
// non-owning pointer to it is a real use-after-free once encode()'s own
// autoreleasepool drains). The caller must `->release()` the returned
// command buffer once fully done with it (after commit()+waitUntilCompleted()).

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

    // Encodes the whole graph, reading `inputBuffer` (fp16 NHWC,
    // netW*netH*inChannels -- NetInputAssembly's output buffer, used
    // directly with no copy) and writing `outputBuffer` (fp16 NHWC,
    // netW*netH*outChannels -- caller-allocated). Returns the command buffer
    // to keep encoding into (see class comment) -- ordinarily the same
    // `cmdBuf` passed in, but not guaranteed to be.
    MTL::CommandBuffer* encode(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Buffer* inputBuffer,
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
