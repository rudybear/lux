#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Forward-declared opaque type for the LiteRT Next GPU backend (net_runner_
// litert.h/.cpp) -- NetRunner only needs a pointer to it, so this header
// doesn't have to include any LiteRT headers or force every TU that
// includes net_runner.h to see them.
struct LiteRtNetRunnerImpl;

// Stage 4 of the mobile-DLSS Android live demo (docs/rendering-engines.md):
// runs mobiledlss/train/export.py::export_tflite_pooled_input's exported
// ParamPredUNet (demo/ios_assets/exported/unet_ps2_mem_pooled.tflite --
// fp16 weights, K=4, s=2, param_stride=2, hidden=8, texture_channels=8,
// scene memory / 3-way blend) via the TensorFlow Lite C API + GPU delegate.
//
// Task 4 (docs/rendering-engines.md, remove the readback stalls/copies
// where cheap) follow-up: this used to bridge a real input-contract
// mismatch -- export_tflite's exported graph took 8 *proxy-resolution*
// tensors and ran build_input's box-pooling as TF ops INSIDE the graph,
// while Stage 3's InputAssembly already produces the POST-pooled 26-channel
// NET-resolution tensor, so NetRunner had to reconstruct an exact
// proxy-resolution input set (nearest-upsample + inverse post-pool
// transforms) just so the model could re-pool it right back down --
// measured (task 2's RECON_TIMING) at ~53ms of the ~103ms total net cost,
// pure wasted work. export_tflite_pooled_input's graph starts *after*
// build_input (ParamPredUNet._features_from_pooled: stem->down->
// bottleneck->up->head), taking the already-pooled 26-channel tensor
// directly -- exactly what InputAssembly::getOutputHostPtr() already is,
// same channel order (`build_input`'s `parts` concat order: color(3),
// depthN(1), mvN(2), disocc(1), fg(1), jitter(2), texture_feat(8),
// hidden(hiddenChannels)). So run() now feeds that pointer straight to the
// model's single input tensor -- no adapter, no per-channel reconstruction
// -- except for one thing InputAssembly's own tensor does NOT carry: the
// real recurrent hidden state (InputAssembly::run() is always called with
// hiddenIn=nullptr in this app, per its own call sites' comments, so its
// own hidden channels are always zero; Stage 5/6's recurrence is fed in
// here instead, exactly as it always was). run() overwrites just the last
// hiddenChannels floats of each pixel (a cheap ~netW*netH*hiddenChannels
// copy, not a full proxy-res reconstruction) when hiddenInNetRes is
// non-null.
//
// Validated (desktop, mobiledlss/.venv): the pooled-input fp16 TFLite
// model's output on the device's own dumped frame-30 net-input tensor
// matches the real on-device XNNPACK run of the OLD proxy-input model to
// max|diff|=1.3e-5 (essentially exact -- both are the same fp16-weight
// model, just re-entered at a different point in the same graph).
class NetRunner {
public:
    ~NetRunner();

    // netW/netH: Stage 3's InputAssembly::getNetW()/getNetH() (240x136).
    // paramStride/hiddenChannels/texChannels must match Stage 3's config
    // (2, 8, 8) -- paramStride/texChannels are kept only for logging/
    // bookkeeping now (no longer used to size an adapter reconstruction).
    // modelPath: the pooled-input .tflite file's on-device path.
    void init(const std::string& modelPath, uint32_t netW, uint32_t netH,
              uint32_t paramStride, uint32_t hiddenChannels, uint32_t texChannels);

    bool isGpuDelegateActive() const { return gpuDelegateActive_; }
    uint32_t getOutputChannels() const { return outputChannels_; }  // sp*sp*K*K + sp*sp*3 + hidden

    struct RunTimingsMs {
        double adapterMs = 0.0;   // hidden-channel overwrite into the staging buffer (0 if hiddenInNetRes is null -- straight passthrough then, no copy at all)
        double uploadMs = 0.0;    // TfLiteTensorCopyFromBuffer x1 (CPU buffer I/O)
        double inferMs = 0.0;     // TfLiteInterpreterInvoke
        double downloadMs = 0.0;  // TfLiteTensorCopyToBuffer x1
    };

    // netTensor26ch: Stage 3's packed [netH, netW, 18+hiddenChannels] fp32
    // output (InputAssembly::getOutputHostPtr()) -- fed to the model
    // (almost) as-is. hiddenInNetRes: recurrent hidden state at net res
    // ([netH,netW,hiddenChannels]) or nullptr for zero (Stage 4 validation
    // config; Stage 5 feeds the real state) -- overwrites netTensor26ch's
    // own (always-zero) trailing hidden channels in the staging buffer, per
    // this class's header comment. Returns a pointer (owned by this object,
    // valid until the next run()) to the packed NHWC output, [netH, netW,
    // getOutputChannels()].
    const float* run(const float* netTensor26ch, const float* hiddenInNetRes, RunTimingsMs& timings);

private:
    struct Impl;
    Impl* impl_ = nullptr;

    // LiteRT Next GPU backend (net_runner_litert.h/.cpp): the default
    // backend as of the LiteRT-Next-GPU-path task (docs/rendering-
    // engines.md) -- ~15-18ms median end-to-end on this device vs XNNPACK's
    // ~37-44ms, same fp16-weight-quantization-level accuracy. init() tries
    // this first; on ANY failure (accelerator unavailable, compile error,
    // ...) it falls back to the XNNPACK Impl above, exactly like the old
    // force_gpu_net TFLite-GPU-delegate fallback used to. A `xnnpack_net`
    // marker file (cwd-relative, same convention as force_gpu_net/
    // gpu_net_config.txt) forces the XNNPACK path even when LiteRT GPU is
    // available, for A/B comparison.
    LiteRtNetRunnerImpl* litertImpl_ = nullptr;
    bool useLiteRt_ = false;

    uint32_t netW_ = 0, netH_ = 0, paramStride_ = 1, hiddenChannels_ = 0, texChannels_ = 0;
    uint32_t inputChannels_ = 0;  // netTensor26ch's channel count (10 + hiddenChannels_ + texChannels_)
    uint32_t outputChannels_ = 0;
    bool gpuDelegateActive_ = false;

    std::vector<float> bufPooled_;  // staging buffer, only populated when hiddenInNetRes != nullptr
    std::vector<float> bufOutput_;
};
