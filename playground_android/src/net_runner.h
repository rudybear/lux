#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Stage 4 of the mobile-DLSS Android live demo (docs/rendering-engines.md):
// runs mobiledlss/train/export.py::export_tflite's exported ParamPredUNet
// (demo/ios_assets/exported/unet_ps2_mem.tflite -- fp16 weights, K=4, s=2,
// param_stride=2, hidden=8, texture_channels=8, scene memory / 3-way blend)
// via the TensorFlow Lite C API + GPU delegate.
//
// Input-contract mismatch this class bridges: export_tflite's exported
// graph takes 8 *proxy-resolution* tensors (proxy_color/depth/mv/disocc/fg,
// jitter_proxy, hidden_in at net res, texture_feat -- build_input's own box
// -pooling runs INSIDE the exported graph as TF ops), but Stage 3's
// InputAssembly already produces the POST-pooled, POST-normalize 26-channel
// NET-resolution tensor (matching NetInputAssembly.mm's "pre-pooled" design,
// meant for the OTHER, fused-GPU-compute UNet path documented in
// docs/rendering-engines.md's "Fused-GPU-compute ParamPredUNet" section).
// Rather than rewriting Stage 3's shader to emit two different resolutions,
// this class reconstructs an EXACT (not approximate) proxy-resolution input
// set by nearest-upsampling each already-pooled net-res channel back up by
// param_stride and inverting the two post-pool scalar transforms
// (mv_scale multiply, depth normalize) build_input applies AFTER pooling --
// avg_pool2d of a nearest-upsampled, piecewise-constant-per-block field is
// mathematically the identity, so re-running the graph's own internal
// box-pool on this reconstructed input reproduces Stage 3's exact net-res
// values bit-for-bit (mod fp32 rounding), not merely approximately.
class NetRunner {
public:
    ~NetRunner();

    // netW/netH: Stage 3's InputAssembly::getNetW()/getNetH() (240x136).
    // paramStride/hiddenChannels/texChannels must match Stage 3's config
    // (2, 8, 8). modelPath: the .tflite file's on-device path.
    void init(const std::string& modelPath, uint32_t netW, uint32_t netH,
              uint32_t paramStride, uint32_t hiddenChannels, uint32_t texChannels);

    bool isGpuDelegateActive() const { return gpuDelegateActive_; }
    uint32_t getOutputChannels() const { return outputChannels_; }  // sp*sp*K*K + sp*sp*3 + hidden

    struct RunTimingsMs {
        double adapterMs = 0.0;   // net-res -> proxy-res reconstruction (CPU)
        double uploadMs = 0.0;    // TfLiteTensorCopyFromBuffer x8 (CPU buffer I/O)
        double inferMs = 0.0;     // TfLiteInterpreterInvoke
        double downloadMs = 0.0;  // TfLiteTensorCopyToBuffer x1
    };

    // netTensor26ch: Stage 3's packed [netH, netW, 18+hiddenChannels] fp32
    // output (InputAssembly::getOutputHostPtr()). hiddenInNetRes: recurrent
    // hidden state at net res ([netH,netW,hiddenChannels]) or nullptr for
    // zero (Stage 4 validation config; Stage 5 feeds the real state).
    // Returns a pointer (owned by this object, valid until the next run())
    // to the packed NHWC output, [netH, netW, getOutputChannels()].
    const float* run(const float* netTensor26ch, const float* hiddenInNetRes, RunTimingsMs& timings);

private:
    struct Impl;
    Impl* impl_ = nullptr;

    uint32_t netW_ = 0, netH_ = 0, paramStride_ = 1, hiddenChannels_ = 0, texChannels_ = 0;
    uint32_t proxyW_ = 0, proxyH_ = 0;  // netW_*paramStride_, netH_*paramStride_
    uint32_t outputChannels_ = 0;
    bool gpuDelegateActive_ = false;

    std::vector<float> bufColor_, bufDepth_, bufMv_, bufDisocc_, bufFg_, bufJitter_, bufHidden_, bufTex_;
    std::vector<float> bufOutput_;
};
