#pragma once

// Live-inference stage: reconstruct-with-memory (mobiledlss.train.reconstruct.reconstruct +
// blend3 + mobiledlss.train.scene_texture.MemoryColorHead), as a single Metal
// compute kernel over target-resolution pixels:
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
// Moved from playground_ios/Source/ReconstructPass.{h,mm} into
// playground_cpp/src -- see net_input_assembly.h's header comment for why.
// Weights from demo/ios_assets/exported/{texture.npy,bg_sphere.npy,memory_head.npz}.
//
// Both entry points now take the caller's own MTL::CommandBuffer* and encode
// into it with NO commit()/waitUntilCompleted() of their own (the iOS-only
// version began and blocked on its own command buffer every call) -- the
// caller (MetalLiveReconstruct::encodeFrame) commits once, at the end of the
// whole per-frame chain.

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>
#include <cstdint>
#include <string>

class MetalContext;

class LiveReconstructPass {
public:
    ~LiveReconstructPass();

    void init(MetalContext& ctx, const std::string& textureNpyPath, const std::string& bgSphereNpyPath,
              const std::string& memoryHeadNpzPath, uint32_t targetW, uint32_t targetH, uint32_t proxyW,
              uint32_t proxyH, uint32_t netW, uint32_t netH, uint32_t k, uint32_t hiddenChannels,
              uint32_t nBlend, uint32_t s, uint32_t paramStride);

    // Step 0 (call BEFORE NetInputAssembly::run()/MPSGraphUNet::encode() this
    // frame, into the SAME cmdBuf): warp last frame's raw hidden state
    // (target res) by this frame's proxy MV, box-downsample to net res --
    // mobiledlss.train.reconstruct.warp()+downsample_hidden()'s exact
    // recurrence. Writes into `hiddenInOut` (fp16 NHWC,
    // netW*netH*hiddenChannels -- pass this buffer as NetInputAssembly::
    // run()'s hiddenIn). On the very first frame (`firstFrame`), just
    // zero-fills (no history yet).
    void prepareHiddenInput(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Texture* proxyMotionTex,
                             bool firstFrame, MTL::Buffer* hiddenInOut);

    MTL::Buffer* getHiddenInputBuffer() const { return hiddenInputBuffer_; }

    // Warm start (Task A cold-start investigation): bilinear-upsamples
    // `proxyColorTex` (premultiplied-over-black, same convention run()'s own
    // apply_kernel gather now reads it in -- see that comment) into BOTH
    // prevColor_ ping-pong slots, composited exactly like the app's own
    // Bilinear display mode (SplatView.mm's upscale_proxy, bilinear==1
    // branch, post its own premultiply fix -- no un-premultiply division).
    // Call once, right after this frame's proxy render, whenever this frame
    // is about to run as `firstFrame` (i.e. right after a resetHistory()) --
    // MetalLiveReconstruct::encodeFrame() does this.
    //
    // NOTE on why this does NOT change the very first frame's own output:
    // `run()`'s `firstFrame` branch forces `disocc=1` for that frame (which
    // it must -- see mobiledlss.train.train.rollout's own `t==0:
    // disocc_t=torch.ones(...)`, the exact convention this mirrors), and
    // `renormalized_blend_weights`/blend3's disocclusion gate zeroes the
    // *history* blend weight whenever disocc>0.5 -- so `wH * warped` is 0
    // regardless of what `warped` (sourced from prevColor_) equals, on
    // BOTH the very first frame (matching PyTorch's own `prev_color =
    // torch.zeros(...)` at t==0, equally inert there for the same reason)
    // and every frame after until prevColor_ is naturally overwritten by
    // this frame's own real output anyway. This call exists to replace
    // what was previously an implicitly-relied-on-but-never-actually-
    // written GPU texture (prevColor_ ping-pong slots are
    // StorageModePrivate and were never cleared/initialized before this)
    // with well-defined content, and to keep parity with `warm start`
    // wording -- not because it measurably changes --live-psnr's frame-0
    // PSNR under the current (training-matching) disocclusion gating.
    void seedHistoryFromProxy(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Texture* proxyColorTex);

    // Debug output (rollout-PSNR capture): per target-pixel
    // post-disocclusion-renormalization blend weights, half2 (wS spatial, wM
    // memory -- wH omitted, it's `1 - wS - wM`), written by every run() call
    // unconditionally.
    MTL::Buffer* getBlendDebugBuffer() const { return blendDebugBuffer_; }

    // Debug outputs (Task B chroma-fringe investigation): the three raw
    // colour sources blend3 mixes together, half4 (rgb, 1.0) per target
    // pixel, written by every run() call unconditionally -- same
    // always-write precedent as getBlendDebugBuffer() above. Lets a caller
    // isolate which of {spatial, warped/history, memory} carries a given
    // pixel's error without re-deriving them from scratch.
    MTL::Buffer* getSpatialDebugBuffer() const { return spatialDebugBuffer_; }
    MTL::Buffer* getWarpedDebugBuffer() const { return warpedDebugBuffer_; }
    MTL::Buffer* getMemoryDebugBuffer() const { return memoryDebugBuffer_; }

    // For a frame-N "everything reconstruct consumes" validation dump
    // (reproducing the step in PyTorch): the *history* this frame's run() is
    // about to read -- i.e. last frame's own composited output/hidden,
    // BEFORE run() flips pingIndex_. Call before run(), not after.
    MTL::Texture* getPrevColorTexture() const { return prevColor_[pingIndex_]; }
    MTL::Buffer* getPrevHiddenBuffer() const { return prevHidden_[pingIndex_]; }

    // Forces the next prepareHiddenInput()/run() pair to be treated as the
    // very first frame again (zero hidden, no warp/blend history) -- pair
    // with NetInputAssembly::reset() when re-entering Reconstruction display
    // mode after frames were skipped.
    void reset() {
        pingIndex_ = 0;
        firstRun_ = true;
    }

    // Runs the whole reconstruct pass for one frame, encoding into `cmdBuf`
    // (caller commits/waits), writing into `outColorTex` (any writable
    // texture -- the caller's own offscreen target, or the drawable).
    // `curDepthTex`/`prevDepthTex`: NetInputAssembly::getDepthWrittenThisFrame()/
    // getDepthFromPreviousFrame(), captured *before* that frame's
    // NetInputAssembly::run() call (which flips them). `firstFrame`:
    // NetInputAssembly::isNextFrameFirst() (same one-shot "treat as fully
    // disoccluded" seed). Eye/rAxis/uAxis/fAxis/fx/fy/cx/cy are the
    // *target*-resolution camera for this frame (bg-sphere UV).
    void run(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Texture* proxyColorTex,
             MTL::Texture* proxyMotionTex, MTL::Texture* curDepthTex, MTL::Texture* prevDepthTex, bool firstFrame,
             MTL::Buffer* unetOutputBuffer, MTL::Texture* outColorTex, glm::vec3 eye, glm::vec3 rAxis,
             glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy, float cx, float cy, float jitterTargetX,
             float jitterTargetY);

private:
    MTL::ComputePipelineState* pipeline_ = nullptr;
    MTL::ComputePipelineState* hiddenPipeline_ = nullptr;
    MTL::ComputePipelineState* warmStartPipeline_ = nullptr;
    MTL::Buffer* hiddenInputBuffer_ = nullptr;  // fp16 NHWC, netW*netH*hiddenChannels

    MTL::Buffer* blendDebugBuffer_ = nullptr;  // half2 (wS, wM), targetW*targetH
    MTL::Buffer* spatialDebugBuffer_ = nullptr;  // half4 (rgb, 1.0), targetW*targetH
    MTL::Buffer* warpedDebugBuffer_ = nullptr;   // half4 (rgb, 1.0), targetW*targetH
    MTL::Buffer* memoryDebugBuffer_ = nullptr;   // half4 (rgb, 1.0), targetW*targetH

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
