#include "net_runner.h"

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define LOG_TAG "lux_android_net"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
inline double msSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Stage 3's net-res channel layout (InputAssembly / NetInputAssembly.h):
// color(3) depthN(1) mvN(2) disocc(1) fg(1) jitter(2) texfeat(8) hidden(*).
constexpr int kColorOff = 0, kDepthOff = 3, kMvOff = 4, kDisoccOff = 6, kFgOff = 7,
              kJitterOff = 8, kTexOff = 10;
constexpr float kMvScale = 1.0f / 16.0f;
}  // namespace

struct NetRunner::Impl {
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteDelegate* gpuDelegate = nullptr;
    TfLiteInterpreter* interpreter = nullptr;

    // Input tensor indices, resolved once by name at init time (onnx2tf +
    // flatc preserved export.py's _input_names order/names exactly -- see
    // the export_and_validate_tflite.py validation this was checked
    // against -- but resolving by name is one extra safety net against a
    // future re-export reordering them).
    int idxColor = -1, idxDepth = -1, idxMv = -1, idxDisocc = -1, idxFg = -1,
        idxJitter = -1, idxHidden = -1, idxTex = -1;
    TfLiteTensor* outputTensor = nullptr;
};

NetRunner::~NetRunner() {
    if (!impl_) return;
    if (impl_->interpreter) TfLiteInterpreterDelete(impl_->interpreter);
    if (impl_->gpuDelegate) TfLiteGpuDelegateV2Delete(impl_->gpuDelegate);
    if (impl_->options) TfLiteInterpreterOptionsDelete(impl_->options);
    if (impl_->model) TfLiteModelDelete(impl_->model);
    delete impl_;
}

void NetRunner::init(const std::string& modelPath, uint32_t netW, uint32_t netH,
                      uint32_t paramStride, uint32_t hiddenChannels, uint32_t texChannels) {
    impl_ = new Impl();
    netW_ = netW;
    netH_ = netH;
    paramStride_ = paramStride;
    hiddenChannels_ = hiddenChannels;
    texChannels_ = texChannels;
    proxyW_ = netW * paramStride;
    proxyH_ = netH * paramStride;

    impl_->model = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!impl_->model) throw std::runtime_error("NetRunner: TfLiteModelCreateFromFile failed: " + modelPath);

    impl_->options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(impl_->options, 4);

    // GPU delegate is OFF by default: on this exact device/model
    // (Pixel 9 Pro XL, Mali-G715, TFLite GPU delegate 2.16.1,
    // unet_ps2_mem.tflite's onnx2tf-converted graph) it silently produces
    // WRONG results -- max|diff| vs. torch jumped from 1.17e-2 (CPU/
    // XNNPACK, matching the desktop tf.lite.Interpreter validation exactly)
    // to 26.7 (GPU delegate) on the identical frame-30 input, isolated by
    // A/B toggling this flag with everything else held fixed (same
    // interpreter, same input buffers, same model file). Not root-caused
    // to a specific op -- a known category of issue for TFLite's GPU
    // delegate on less-common op patterns (onnx2tf's conversion of the
    // packed multi-output slice, transposes, etc. -- see export.py's
    // `_export_tflite_onnx2tf` docstring for the conversion bugs already
    // known in this exact graph). CPU/XNNPACK is the only currently-
    // correct backend for this model on this device; re-enable via the
    // `force_gpu_net` marker file only for further debugging.
    FILE* marker = fopen("force_gpu_net", "r");
    bool forceGpu = (marker != nullptr);
    if (marker) fclose(marker);
    TfLiteGpuDelegateOptionsV2 gpuOpts = TfLiteGpuDelegateOptionsV2Default();
    gpuOpts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    gpuOpts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    impl_->gpuDelegate = forceGpu ? TfLiteGpuDelegateV2Create(&gpuOpts) : nullptr;
    if (forceGpu) LOGI("NetRunner: force_gpu_net marker present, enabling (KNOWN-INCORRECT) GPU delegate");
    if (impl_->gpuDelegate) {
        TfLiteInterpreterOptionsAddDelegate(impl_->options, impl_->gpuDelegate);
    } else if (forceGpu) {
        LOGE("TfLiteGpuDelegateV2Create failed even with force_gpu_net set; using CPU (XNNPACK)");
    } else {
        LOGI("NetRunner: GPU delegate disabled by default (see init()'s comment); using CPU (XNNPACK)");
    }

    impl_->interpreter = TfLiteInterpreterCreate(impl_->model, impl_->options);
    if (!impl_->interpreter) throw std::runtime_error("NetRunner: TfLiteInterpreterCreate failed");

    if (TfLiteInterpreterAllocateTensors(impl_->interpreter) != kTfLiteOk) {
        if (impl_->gpuDelegate) {
            // GPU delegate rejected the graph (or a subset of ops) outright
            // at allocate-time on some builds; retry CPU-only rather than
            // fail the whole demo stage.
            LOGE("AllocateTensors failed with GPU delegate; retrying CPU-only");
            TfLiteInterpreterDelete(impl_->interpreter);
            TfLiteGpuDelegateV2Delete(impl_->gpuDelegate);
            impl_->gpuDelegate = nullptr;
            TfLiteInterpreterOptionsDelete(impl_->options);
            impl_->options = TfLiteInterpreterOptionsCreate();
            TfLiteInterpreterOptionsSetNumThreads(impl_->options, 4);
            impl_->interpreter = TfLiteInterpreterCreate(impl_->model, impl_->options);
            if (!impl_->interpreter || TfLiteInterpreterAllocateTensors(impl_->interpreter) != kTfLiteOk) {
                throw std::runtime_error("NetRunner: AllocateTensors failed (CPU fallback too)");
            }
        } else {
            throw std::runtime_error("NetRunner: AllocateTensors failed");
        }
    }
    gpuDelegateActive_ = (impl_->gpuDelegate != nullptr);

    int32_t nIn = TfLiteInterpreterGetInputTensorCount(impl_->interpreter);
    LOGI("NetRunner: model=%s inputs=%d gpu_delegate=%d", modelPath.c_str(), nIn, gpuDelegateActive_);
    for (int32_t i = 0; i < nIn; i++) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(impl_->interpreter, i);
        std::string name = TfLiteTensorName(t) ? TfLiteTensorName(t) : "";
        LOGI("  IN[%d] name=%s dims=%d", i, name.c_str(), TfLiteTensorNumDims(t));
        if (name == "proxy_color") impl_->idxColor = i;
        else if (name == "proxy_depth") impl_->idxDepth = i;
        else if (name == "proxy_mv") impl_->idxMv = i;
        else if (name == "proxy_disocc") impl_->idxDisocc = i;
        else if (name == "proxy_fg") impl_->idxFg = i;
        else if (name == "jitter_proxy") impl_->idxJitter = i;
        else if (name == "hidden_in") impl_->idxHidden = i;
        else if (name == "texture_feat") impl_->idxTex = i;
    }
    if (impl_->idxColor < 0 || impl_->idxDepth < 0 || impl_->idxMv < 0 || impl_->idxDisocc < 0 ||
        impl_->idxFg < 0 || impl_->idxJitter < 0 || impl_->idxHidden < 0 || impl_->idxTex < 0) {
        // Name-based resolution failed (e.g. a re-export without flatc on
        // PATH, per export.py's _export_tflite_onnx2tf docstring) -- fall
        // back to export.py's fixed _input_names positional order.
        LOGE("NetRunner: input tensor name lookup incomplete; falling back to positional order");
        impl_->idxColor = 0; impl_->idxDepth = 1; impl_->idxMv = 2; impl_->idxDisocc = 3;
        impl_->idxFg = 4; impl_->idxJitter = 5; impl_->idxHidden = 6; impl_->idxTex = 7;
    }

    impl_->outputTensor = const_cast<TfLiteTensor*>(TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0));
    int32_t outDims = TfLiteTensorNumDims(impl_->outputTensor);
    outputChannels_ = static_cast<uint32_t>(TfLiteTensorDim(impl_->outputTensor, outDims - 1));
    LOGI("NetRunner: output channels=%u (expect sp*sp*K*K + sp*sp*3 + hidden)", outputChannels_);

    const size_t proxyN = static_cast<size_t>(proxyW_) * proxyH_;
    bufColor_.resize(proxyN * 3);
    bufDepth_.resize(proxyN);
    bufMv_.resize(proxyN * 2);
    bufDisocc_.resize(proxyN);
    bufFg_.resize(proxyN);
    bufJitter_.resize(2);
    bufHidden_.resize(static_cast<size_t>(netW_) * netH_ * hiddenChannels_);
    bufTex_.resize(proxyN * texChannels_);
    bufOutput_.resize(static_cast<size_t>(netW_) * netH_ * outputChannels_);
}

const float* NetRunner::run(const float* netTensor26ch, const float* hiddenInNetRes, RunTimingsMs& timings) {
    auto tAdapter = std::chrono::high_resolution_clock::now();
    const uint32_t ch = 18u + hiddenChannels_;
    const uint32_t ps = paramStride_;

    // --- Exact net-res -> proxy-res reconstruction (see net_runner.h's
    // header comment for why nearest-upsample + inverse post-pool transform
    // is mathematically exact here, not an approximation). ---
    for (uint32_t gy = 0; gy < netH_; gy++) {
        for (uint32_t gx = 0; gx < netW_; gx++) {
            const float* px = netTensor26ch + (static_cast<size_t>(gy) * netW_ + gx) * ch;
            float depthN = px[kDepthOff];
            float depthRaw = (depthN < 0.999999f) ? depthN / (1.0f - depthN) : 1.0e6f;
            for (uint32_t dy = 0; dy < ps; dy++) {
                for (uint32_t dx = 0; dx < ps; dx++) {
                    uint32_t py = gy * ps + dy, pxi = gx * ps + dx;
                    size_t pidx = static_cast<size_t>(py) * proxyW_ + pxi;
                    bufColor_[pidx * 3 + 0] = px[kColorOff + 0];
                    bufColor_[pidx * 3 + 1] = px[kColorOff + 1];
                    bufColor_[pidx * 3 + 2] = px[kColorOff + 2];
                    bufDepth_[pidx] = depthRaw;
                    bufMv_[pidx * 2 + 0] = px[kMvOff + 0] / kMvScale;
                    bufMv_[pidx * 2 + 1] = px[kMvOff + 1] / kMvScale;
                    bufDisocc_[pidx] = px[kDisoccOff];
                    bufFg_[pidx] = px[kFgOff];
                    for (uint32_t c = 0; c < texChannels_; c++) {
                        bufTex_[pidx * texChannels_ + c] = px[kTexOff + c];
                    }
                }
            }
        }
    }
    bufJitter_[0] = netTensor26ch[kJitterOff + 0];
    bufJitter_[1] = netTensor26ch[kJitterOff + 1];
    if (hiddenInNetRes != nullptr) {
        std::memcpy(bufHidden_.data(), hiddenInNetRes, bufHidden_.size() * sizeof(float));
    } else {
        std::fill(bufHidden_.begin(), bufHidden_.end(), 0.0f);
    }
    timings.adapterMs = msSince(tAdapter);

    auto tUpload = std::chrono::high_resolution_clock::now();
    auto upload = [&](int idx, const std::vector<float>& buf) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(impl_->interpreter, idx);
        TfLiteTensorCopyFromBuffer(t, buf.data(), buf.size() * sizeof(float));
    };
    upload(impl_->idxColor, bufColor_);
    upload(impl_->idxDepth, bufDepth_);
    upload(impl_->idxMv, bufMv_);
    upload(impl_->idxDisocc, bufDisocc_);
    upload(impl_->idxFg, bufFg_);
    upload(impl_->idxJitter, bufJitter_);
    upload(impl_->idxHidden, bufHidden_);
    upload(impl_->idxTex, bufTex_);
    timings.uploadMs = msSince(tUpload);

    auto tInfer = std::chrono::high_resolution_clock::now();
    TfLiteStatus st = TfLiteInterpreterInvoke(impl_->interpreter);
    timings.inferMs = msSince(tInfer);
    if (st != kTfLiteOk) throw std::runtime_error("NetRunner: TfLiteInterpreterInvoke failed");

    auto tDownload = std::chrono::high_resolution_clock::now();
    TfLiteTensorCopyToBuffer(impl_->outputTensor, bufOutput_.data(), bufOutput_.size() * sizeof(float));
    timings.downloadMs = msSince(tDownload);

    return bufOutput_.data();
}
