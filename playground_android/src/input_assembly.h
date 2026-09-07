#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

struct VulkanContext;

// Stage 3 of the mobile-DLSS Android live demo (docs/rendering-engines.md):
// assembles ParamPredUNet's 26-channel NHWC input tensor
// (mobiledlss/train/model.py::build_input) on-GPU from the proxy-res
// (480x270) DLSS attachments the proxy SplatRenderer produces, as two
// hand-written GLSL compute shaders (playground_android/shaders_glsl/
// input_assembly_{unpremul_depth,assemble}.comp, compiled with the NDK's
// glslc -- NOT luxc; this is new, not-yet-declarative code per the task
// brief, mirroring how docs/rendering-engines.md's fused-GPU-compute UNet
// kernels started as hand-written `compute{}` blocks before any `network`
// syntax existed).
//
// Deliberately does NOT bind SplatRenderer's colour/motion/expected-depth
// VkImages directly as sampled/storage images in the compute descriptor
// sets -- those images are only created with COLOR_ATTACHMENT_BIT |
// TRANSFER_SRC_BIT | TRANSFER_DST_BIT usage (splat_renderer.cpp, a reused-
// by-reference file this task must not modify), not SAMPLED_BIT/STORAGE_BIT.
// Instead, each frame's raw texel bytes are pulled off the GPU with a plain
// vkCmdCopyImageToBuffer (the images are already left in
// TRANSFER_SRC_OPTIMAL after SplatRenderer::render() -- see
// createRenderPass's finalLayout -- so no extra barrier is needed) into
// persistent storage buffers, and both compute passes below operate on
// buffers only -- the exact same "everything is a storage buffer" style
// playground_cpp/src/reconstruct_runner.cpp already uses for the
// reconstruct passes. Sampling that the Metal reference (NetInputAssembly.mm)
// did with real hardware texture units (bg-scene-texture bilinear,
// zero-padded previous-depth bilinear) is done as plain manual buffer
// arithmetic in GLSL instead (see the .comp source for the derivation of
// each).
class InputAssembly {
public:
    ~InputAssembly();

    // netW/netH: mobiledlss.train.export._PaddedExportModel's convention --
    // proxy h/w replicate-padded up to a multiple of 8*paramStride before
    // box-pooling, so the network always runs at a multiple-of-8 resolution
    // (proxy 480x270, paramStride=2 -> net 240x136, not "true" 240x135).
    // auxFormat/fgFormat: SplatRenderer::getAuxFormat()/getFgFormat() --
    // defaulted to their current values (RGBA32F/RGBA16F) so existing call
    // sites don't need updating; only used to SIZE the raw-texel storage
    // buffers (see input_assembly.cpp's bytesPerTexel()) -- the GLSL unpack
    // paths in shaders_glsl/input_assembly_*.comp still hardcode those same
    // two formats' byte layouts, so passing anything else here would size
    // the buffer correctly but still be interpreted wrong on the shader
    // side (bytesPerTexel() throws on any other format instead of silently
    // doing that).
    void init(VulkanContext& ctx, const std::string& textureNpyPath,
              const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
              uint32_t paramStride, uint32_t hiddenChannels,
              const std::string& shaderDir,
              VkFormat auxFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
              VkFormat fgFormat = VK_FORMAT_R16G16B16A16_SFLOAT);

    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    static constexpr uint32_t kNonHiddenChannels = 18;
    uint32_t getChannels() const { return kNonHiddenChannels + hiddenChannels_; }

    // fg channel source (see NetInputAssembly.h's identical enum/comment):
    // NEVER proxy alpha -- measured on iOS to collapse the model
    // 36.4->24.3dB (the net reads "alpha coverage" as "moving actor here,
    // distrust history/memory" everywhere alpha>0, not just where the
    // actor truly is). run() always requests kFgSourceExpectedDepthG now
    // that gaussian_splat_dlss's foreground_coverage output is wired
    // (unpremul_depth.comp's CurFg buffer -- originally read via out_depth's
    // .g channel, now via the separate out_fg attachment lux commit 6ed0334
    // introduced; the enum name predates that repacking and is kept
    // unchanged since its MEANING -- "real per-pixel fg, not constant 0" --
    // hasn't -- see input_assembly.cpp/the .comp sources); kFgSourceConstantZero
    // is kept only as the pre-foreground-coverage fallback value.
    enum FgSource : uint32_t { kFgSourceConstantZero = 0, kFgSourceExpectedDepthG = 1 };

    // Task 2 (Reconstruction-mode time budget) timing breakdown for one
    // run() call -- CPU wall-clock only (these are all synchronous
    // beginSingleTimeCommands()/endSingleTimeCommands() round trips, i.e.
    // full queue drains, so CPU wall time here already includes GPU
    // execution + submit/wait overhead; readbackMs's 2 vkCmdCopyImageToBuffer
    // calls have no compute work at all, so they isolate pure copy+drain
    // cost).
    struct Timings {
        double readbackMs = 0.0;  // 3x copyImageToBuffer (color/aux/fg -> storage buffers)
        double computeMs = 0.0;   // unpremul + assemble compute dispatches
    };

    // Runs both passes for one frame: (1) un-premultiply this frame's proxy
    // depth+fg into the ping-pong/current buffers, (2) assemble the packed
    // fp32 NHWC tensor into getOutputHostPtr(). `colorImage`/`auxImage`/
    // `fgImage` are the proxy SplatRenderer's raw attachments (already in
    // TRANSFER_SRC_OPTIMAL after render()) -- lux commit 6ed0334 packed the
    // old separate motion/depth attachments into one `auxImage`
    // (SplatRenderer::getAuxImage(), RGBA32F: x=mv.x*a, y=mv.y*a, z=depth*a,
    // w=a) plus a separate `fgImage` (getFgImage(), RGBA16F: x=fg*a, w=a);
    // `hiddenIn` may be null (zero hidden state -- Stage 3/4 validation
    // config; Stage 5 feeds the real recurrent state). eye/rAxis/uAxis/fAxis
    // are world-space camera position/right/up(down)/forward axes (see
    // android_main.cpp's buildCvViewRowMajor -- its r/u/f ARE these axes
    // directly, no camera-to-world matrix construction needed). `outTimings`
    // (optional) receives this call's breakdown for task 2's profiling.
    void run(VulkanContext& ctx, VkImage colorImage, VkImage auxImage, VkImage fgImage,
             uint32_t proxyW, uint32_t proxyH, const float* hiddenIn,
             float eyeX, float eyeY, float eyeZ,
             float rX, float rY, float rZ, float uX, float uY, float uZ,
             float fX, float fY, float fZ,
             float fx, float fy, float cx, float cy,
             float jitterProxyX, float jitterProxyY, Timings* outTimings = nullptr);

    // Host-visible pointer to this frame's packed NHWC fp32 output
    // (netW*netH*getChannels() floats) -- valid immediately after run()
    // returns (all buffers are VMA_MEMORY_USAGE_CPU_TO_GPU / coherent, and
    // run() fully drains the queue via ctx.endSingleTimeCommands()).
    const float* getOutputHostPtr() const;
    void dumpToNpy(const std::string& path) const;

    // Proxy-res unpremultiplied depth this frame just wrote / the frame
    // before it -- for Stage 5's target-resolution disocclusion (its own
    // separate computation from these, per NetInputAssembly.h's
    // getDepthWrittenThisFrame/getDepthFromPreviousFrame comment: "CALL
    // THESE AFTER run() returns" -- run() flips the ping-pong index at the
    // very end of each call, so these two already account for that).
    const float* getDepthWrittenThisFrame() const;
    const float* getDepthFromPreviousFrame() const;
    bool wasFirstFrame() const { return wasFirstFrame_; }

    uint32_t getTexChannels() const { return texChannels_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;

    uint32_t proxyW_ = 0, proxyH_ = 0, paramStride_ = 1, hiddenChannels_ = 0;
    uint32_t netW_ = 0, netH_ = 0;
    uint32_t texChannels_ = 0, texW_ = 0, texH_ = 0;
    bool firstFrame_ = true;
    bool wasFirstFrame_ = true;
    std::vector<float> bgSphere_;  // cx, cy, cz, r
};
