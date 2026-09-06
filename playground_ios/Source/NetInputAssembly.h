#pragma once

// Stage B2: assembles ParamPredUNet's 26-channel NHWC input tensor
// (mobiledlss/train/model.py::build_input) on-GPU, as a Metal compute pass,
// from the proxy-res (480x270) DLSS attachments MetalSplatRenderer produces
// (B1) plus the exported per-scene memory texture/bg-sphere. See
// docs/rendering-engines.md's playground_ios section for the validation
// numbers against mobiledlss.train.model.build_input.
//
// Channel order (build_input's exact `parts` concat order):
//   color(3), depth_n(1), mv_n(2), disocc(1), fg(1), jitter(2),
//   texture_feat(8), hidden(hiddenChannels) = 18 + hiddenChannels total.

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>
#include <cstdint>
#include <string>

class MetalContext;

class NetInputAssembly {
public:
    ~NetInputAssembly();

    // `textureNpyPath`/`bgSphereNpyPath`: exported/texture.npy ([C,H,W] float32)
    // and bg_sphere.npy ([cx,cy,cz,r] float32), read via DlssIO::readNpyFloat32.
    // `depthAlphaOffset`: index of the alpha component in the proxy renderer's
    // expected-depth texture (MetalSplatRenderer's RG32Float -> 1;
    // MetalSplatLuxcRenderer's RGBA32Float -> 3; i.e. kExpectedDepthChannels-1
    // for either backend) -- premultiplied depth is always at index 0.
    void init(MetalContext& ctx, const std::string& textureNpyPath,
              const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
              uint32_t paramStride, uint32_t hiddenChannels, uint32_t depthAlphaOffset);

    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    uint32_t getChannels() const { return kNonHiddenChannels + hiddenChannels_; }
    static constexpr uint32_t kNonHiddenChannels = 18;

    // Runs the unpremultiply-depth pass (ping-ponged history) + the main
    // assembly kernel for one frame, writing the packed fp16 NHWC result
    // into getOutputBuffer(). `hiddenIn` (NHWC, netW*netH*hiddenChannels
    // fp16 values) may be null, in which case hidden channels are written
    // as zero (B2's validation config -- see SplatView.mm).
    void run(MetalContext& ctx, MTL::Texture* currColorTex, MTL::Texture* currDepthTex,
             MTL::Texture* currMotionTex, MTL::Buffer* hiddenIn, glm::vec3 eye,
             glm::vec3 rAxis, glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy,
             float cx, float cy, float jitterProxyX, float jitterProxyY);

    MTL::Buffer* getOutputBuffer() const { return outputBuffer_; }

private:
    MetalContext* ctx_ = nullptr;
    MTL::ComputePipelineState* unpremulPipeline_ = nullptr;
    MTL::ComputePipelineState* assemblePipeline_ = nullptr;

    MTL::Buffer* bgTextureBuffer_ = nullptr;  // [C, texH, texW] float32
    uint32_t texChannels_ = 0, texW_ = 0, texH_ = 0;
    glm::vec4 bgSphere_{0.0f};  // cx, cy, cz, r

    MTL::Texture* depthPing_[2] = {nullptr, nullptr};  // R32Float, proxy res, un-premultiplied
    int depthPingIndex_ = 0;
    bool firstFrame_ = true;

    MTL::Buffer* outputBuffer_ = nullptr;  // half, netW*netH*getChannels()
    MTL::Buffer* zeroHiddenBuffer_ = nullptr;

    uint32_t proxyW_ = 0, proxyH_ = 0, paramStride_ = 1, hiddenChannels_ = 0;
    uint32_t depthAlphaOffset_ = 1;
    uint32_t netW_ = 0, netH_ = 0;
};
