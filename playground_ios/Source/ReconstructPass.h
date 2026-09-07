#pragma once

// Stage B4: live reconstruct-with-memory (mobiledlss.train.reconstruct.reconstruct +
// blend3 + mobiledlss.train.scene_texture.MemoryColorHead), as a single Metal
// compute kernel over target-resolution (960x540) pixels:
//   - pixel_shuffle_params: un-multiplex the UNet's net-res packed output
//     (kernel logits, blend logits, hidden_raw) to per-target-pixel values.
//   - apply_kernel: gather K*K jitter-aware nearest taps from the proxy
//     colour, softmax-weighted -> `spatial`.
//   - warp: backward-warp the previous frame's output by this frame's
//     (nearest-upsampled, scaled) proxy MV -> `warped`.
//   - bg-sphere UV (target res) + texture sample + MemoryColorHead decode
//     -> `memory` colour.
//   - blend3 (3-way, disocclusion-gated renormalised weights) -> output.
//   - hidden_new (broadcast from hidden_raw) is kept for next frame's
//     warp+downsample (ping-ponged here internally).
//
// Weights from demo/ios_assets/exported/{texture.npy,bg_sphere.npy,memory_head.npz}
// (mobiledlss's export_ios_weights.py); reads DlssIO::readNpzMemberFloat32
// for memory_head.npz (already linked, dlss_io.cpp).

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>
#include <cstdint>
#include <string>

class MetalContext;

class ReconstructPass {
public:
    ~ReconstructPass();

    void init(MetalContext& ctx, const std::string& textureNpyPath, const std::string& bgSphereNpyPath,
              const std::string& memoryHeadNpzPath, uint32_t targetW, uint32_t targetH, uint32_t proxyW,
              uint32_t proxyH, uint32_t netW, uint32_t netH, uint32_t k, uint32_t hiddenChannels,
              uint32_t nBlend, uint32_t s, uint32_t paramStride);

    // Step 0 (call BEFORE NetInputAssembly::run()/MPSGraphUNet::encode() this
    // frame): warp last frame's raw hidden state (target res) by this
    // frame's proxy MV, box-downsample to net res -- mobiledlss.train.
    // reconstruct.warp()+downsample_hidden()'s exact recurrence
    // (mobiledlss.train.train.rollout's per-frame hidden handling). Writes
    // into `hiddenInOut` (fp16 NHWC, netW*netH*hiddenChannels -- pass this
    // buffer as NetInputAssembly::run()'s hiddenIn). On the very first frame
    // (`firstFrame`), just zero-fills (no history yet).
    void prepareHiddenInput(MetalContext& ctx, MTL::Texture* proxyMotionTex, bool firstFrame,
                             MTL::Buffer* hiddenInOut);

    MTL::Buffer* getHiddenInputBuffer() const { return hiddenInputBuffer_; }

    // Runs the whole reconstruct pass for one frame, writing into `outColorTex`
    // (any writable texture -- the caller's drawable, or an offscreen target
    // for PSNR/dump purposes). `curDepthTex`/`prevDepthTex`:
    // NetInputAssembly::getCurDepthTexture()/getPrevDepthTexture(), captured
    // *before* that frame's NetInputAssembly::run() call (which flips them).
    // `firstFrame`: NetInputAssembly::isFirstFrame() (same one-shot
    // "treat as fully disoccluded" seed as B2). Eye/rAxis/uAxis/fAxis/fx/fy/
    // cx/cy are the *target*-resolution camera for this frame (bg-sphere UV).
    void run(MetalContext& ctx, MTL::Texture* proxyColorTex, MTL::Texture* proxyMotionTex,
             MTL::Texture* curDepthTex, MTL::Texture* prevDepthTex, bool firstFrame,
             MTL::Buffer* unetOutputBuffer, MTL::Texture* outColorTex, glm::vec3 eye, glm::vec3 rAxis,
             glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy, float cx, float cy, float jitterTargetX,
             float jitterTargetY);

private:
    MTL::ComputePipelineState* pipeline_ = nullptr;
    MTL::ComputePipelineState* hiddenPipeline_ = nullptr;
    MTL::Buffer* hiddenInputBuffer_ = nullptr;  // fp16 NHWC, netW*netH*hiddenChannels

    MTL::Buffer* bgTextureBuffer_ = nullptr;
    uint32_t texChannels_ = 0, texW_ = 0, texH_ = 0;
    glm::vec4 bgSphere_{0.0f};

    MTL::Buffer* fc1w_ = nullptr;  // [hidden, channels]
    MTL::Buffer* fc1b_ = nullptr;  // [hidden]
    MTL::Buffer* fc2w_ = nullptr;  // [3, hidden]
    MTL::Buffer* fc2b_ = nullptr;  // [3]
    uint32_t memHidden_ = 16;

    // Previous-frame output colour + raw hidden state, both at target
    // resolution, ping-ponged (index 0/1 swapped each run()).
    MTL::Texture* prevColor_[2] = {nullptr, nullptr};
    MTL::Buffer* prevHidden_[2] = {nullptr, nullptr};  // fp16 NHWC, targetW*targetH*hiddenChannels
    int pingIndex_ = 0;
    bool firstRun_ = true;

    uint32_t targetW_ = 0, targetH_ = 0, proxyW_ = 0, proxyH_ = 0, netW_ = 0, netH_ = 0;
    uint32_t k_ = 4, hiddenChannels_ = 8, nBlend_ = 3, s_ = 2, paramStride_ = 2;
};
