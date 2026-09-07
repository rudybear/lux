#pragma once

// Live-inference stage: assembles ParamPredUNet's 26-channel NHWC input
// tensor (mobiledlss/train/model.py::build_input) on-GPU, as a Metal compute
// pass, from the proxy-res DLSS attachments MetalSplatLuxcRenderer produces
// plus the exported per-scene memory texture/bg-sphere.
//
// Moved verbatim (logic-for-logic) from playground_ios/Source/
// NetInputAssembly.{h,mm} into playground_cpp/src so both the iOS live demo
// and a macOS `--live-bench` CLI mode (metal_main.cpp) share the exact same
// validated kernel -- see metal_live_reconstruct.h for the driver that owns
// an instance of this. The one behavioral change from the iOS-only version:
// run() now takes the caller's own MTL::CommandBuffer* and encodes into it
// with NO commit()/waitUntilCompleted() of its own (the iOS version began
// and blocked on its own command buffer every frame) -- the caller (
// MetalLiveReconstruct::encodeFrame) is responsible for committing once, at
// the end of the whole per-frame chain.
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
    void init(MetalContext& ctx, const std::string& textureNpyPath,
              const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
              uint32_t paramStride, uint32_t hiddenChannels);

    // Net resolution is the *padded* one (mobiledlss.train.export._PaddedExportModel's
    // convention: proxy h/w replicate-padded up to a multiple of 8*paramStride before
    // box-pooling, so the network itself always runs at a multiple-of-8 resolution --
    // e.g. proxy 480x270, paramStride=2 -> net 240x136, not the "true" 240x135).
    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    uint32_t getChannels() const { return kNonHiddenChannels + hiddenChannels_; }
    static constexpr uint32_t kNonHiddenChannels = 18;

    // Encodes the unpremultiply-depth pass (ping-ponged history) + the main
    // assembly kernel for one frame into `cmdBuf` (caller commits/waits),
    // writing the packed fp16 NHWC result into getOutputBuffer(). `hiddenIn`
    // (NHWC, netW*netH*hiddenChannels fp16 values) may be null, in which case
    // hidden channels are written as zero.
    //
    // `currAuxTex`: MetalSplatLuxcRenderer::getAuxTexture() -- ONE RGBA32Float
    // texture packing (mv.x*alpha, mv.y*alpha, depth*alpha, alpha). `.w` is
    // this attachment's OWN genuine alpha -- un-premultiply mv/depth by it,
    // same convention as color's own `.a`.
    // `currFgTex`: MetalSplatLuxcRenderer::getFgTexture() (RGBA16Float:
    // fg*alpha, 0, 0, alpha) -- a SEPARATE attachment, valid whenever
    // hasForegroundCoverage(); may be null when !hasForegroundCoverage() (fg
    // then stays whatever setFgSource() selects, ignoring this texture).
    void run(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Texture* currColorTex,
             MTL::Texture* currAuxTex, MTL::Texture* currFgTex, MTL::Buffer* hiddenIn, glm::vec3 eye,
             glm::vec3 rAxis, glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy,
             float cx, float cy, float jitterProxyX, float jitterProxyY);

    // fg channel source (mobiledlss/train/model.py::build_input's proxy_fg):
    // NOT proxy alpha (measured to collapse the model from 36.4 to 24.3dB).
    // kFgSourceFgTexture reads lux's real per-pixel foreground_coverage output
    // from currFgTex; kFgSourceConstantZero (background) is the fallback for a
    // pipeline compiled without foreground_coverage.
    enum FgSource : uint32_t { kFgSourceConstantZero = 0, kFgSourceFgTexture = 1 };
    void setFgSource(FgSource src) { fgSource_ = src; }

    MTL::Buffer* getOutputBuffer() const { return outputBuffer_; }

    // Exposed for the reconstruct pass, which needs its own proxy-resolution
    // disocclusion. CALL THESE AFTER run() returns (run() flips the internal
    // ping-pong index at the very end of each call, so "this frame's" and
    // "previous frame's" swap sides relative to which index is "current" --
    // these two methods already account for that).
    MTL::Texture* getDepthWrittenThisFrame() const { return depthPing_[1 - depthPingIndex_]; }
    MTL::Texture* getDepthFromPreviousFrame() const { return depthPing_[depthPingIndex_]; }
    // True iff the frame that was *just* run() was the first one (i.e. its
    // disocclusion was forced to 1 with no real previous-frame history).
    bool wasFirstFrame() const { return wasFirstFrame_; }
    // True iff the *next* run() call will be the first one.
    bool isNextFrameFirst() const { return firstFrame_; }

    // Forces the next run() to be treated as the very first one again (full
    // disocclusion, no real depth history) -- for a display-mode switch back
    // into Reconstruction after skipping frames.
    void reset() {
        depthPingIndex_ = 0;
        firstFrame_ = true;
        wasFirstFrame_ = true;
    }

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
    bool wasFirstFrame_ = true;

    MTL::Buffer* outputBuffer_ = nullptr;  // half, netW*netH*getChannels()
    MTL::Buffer* zeroHiddenBuffer_ = nullptr;

    uint32_t proxyW_ = 0, proxyH_ = 0, paramStride_ = 1, hiddenChannels_ = 0;
    uint32_t fgSource_ = kFgSourceConstantZero;
    uint32_t netW_ = 0, netH_ = 0;
};
