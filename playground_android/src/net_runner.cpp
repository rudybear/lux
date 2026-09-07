#include "net_runner.h"

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>

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
}  // namespace

struct NetRunner::Impl {
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteDelegate* gpuDelegate = nullptr;
    TfLiteDelegate* xnnpackDelegate = nullptr;  // only set when explicitly force-created (see init())
    TfLiteInterpreter* interpreter = nullptr;

    // Single input tensor index, resolved by name ("net_input" --
    // export_tflite_pooled_input's input_names=["net_input"]) with
    // positional fallback to 0 (a single-input graph has nowhere else it
    // could be).
    int idxInput = 0;
    TfLiteTensor* outputTensor = nullptr;
};

NetRunner::~NetRunner() {
    if (!impl_) return;
    if (impl_->interpreter) TfLiteInterpreterDelete(impl_->interpreter);
    if (impl_->gpuDelegate) TfLiteGpuDelegateV2Delete(impl_->gpuDelegate);
    if (impl_->xnnpackDelegate) TfLiteXNNPackDelegateDelete(impl_->xnnpackDelegate);
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
    // build_input's parts concat order: color(3) depthN(1) mvN(2) disocc(1)
    // fg(1) jitter(2) [=10, _NON_HIDDEN_IN_CHANNELS] + texture_feat(tex) +
    // hidden(hidden) -- matches export_tflite_pooled_input's in_ch exactly.
    inputChannels_ = 10u + hiddenChannels_ + texChannels_;

    impl_->model = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!impl_->model) throw std::runtime_error("NetRunner: TfLiteModelCreateFromFile failed: " + modelPath);

    impl_->options = TfLiteInterpreterOptionsCreate();
    // Task 3 CPU-config sweep: cpu_threads.txt (cwd-relative), a bare
    // integer, overrides the thread count (default 4) without a rebuild.
    int numThreads = 4;
    FILE* threadsFile = fopen("cpu_threads.txt", "r");
    if (threadsFile) {
        int v = 0;
        if (fscanf(threadsFile, "%d", &v) == 1 && v > 0) numThreads = v;
        fclose(threadsFile);
    }
    TfLiteInterpreterOptionsSetNumThreads(impl_->options, numThreads);

    // Task 3 follow-up: XNNPACK is applied automatically by
    // TfLiteInterpreterCreate whenever no other delegate is registered (its
    // the CPU path's default backend since ~TF 2.3) -- that implicit
    // instance runs fp32 compute even though the model's WEIGHTS are
    // fp16-quantized on disk (dequantized to fp32 at load time). This
    // block, gated by gpu_net_config.txt's xnnpack_fp16=1 (default 0, i.e.
    // unchanged behaviour), instead creates an EXPLICIT XNNPack delegate
    // with TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 set, which runs the
    // arithmetic itself in fp16 on CPUs with native fp16 vector support
    // (the Tensor G4's cores are ARMv9/ARMv8.2+, which have it) -- measure
    // via the NET timing infer= line with/without this flag. Needs
    // third_party/tflite/include/.../delegates/xnnpack/xnnpack_delegate.h
    // (fetched from the exact tensorflow v2.16.1 tag to match this repo's
    // prebuilt libtensorflowlite_jni.so, which already exports these
    // symbols -- confirmed via `llvm-nm -D`ing it -- even though no header
    // for them shipped in the original third_party/tflite/include drop).
    bool xnnpackForceFp16 = false;
    FILE* xnnCfg = fopen("gpu_net_config.txt", "r");
    if (xnnCfg) {
        char line[128];
        while (fgets(line, sizeof(line), xnnCfg)) {
            std::string s(line);
            auto eq = s.find('=');
            if (eq == std::string::npos) continue;
            std::string key = s.substr(0, eq);
            std::string val = s.substr(eq + 1);
            while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == ' ')) val.pop_back();
            if (key == "xnnpack_fp16") xnnpackForceFp16 = (val == "1");
        }
        fclose(xnnCfg);
    }
    if (xnnpackForceFp16) {
        TfLiteXNNPackDelegateOptions xnnOpts = TfLiteXNNPackDelegateOptionsDefault();
        xnnOpts.num_threads = numThreads;
        xnnOpts.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
        impl_->xnnpackDelegate = TfLiteXNNPackDelegateCreate(&xnnOpts);
        if (impl_->xnnpackDelegate) {
            TfLiteInterpreterOptionsAddDelegate(impl_->options, impl_->xnnpackDelegate);
            LOGI("NetRunner: explicit XNNPack delegate created with FORCE_FP16 (gpu_net_config.txt xnnpack_fp16=1)");
        } else {
            LOGE("NetRunner: TfLiteXNNPackDelegateCreate(FORCE_FP16) failed; falling back to the implicit fp32 XNNPack path");
        }
    }

    // GPU delegate is OFF by default: on this exact device/model
    // (Pixel 9 Pro XL, Mali-G715, TFLite GPU delegate 2.16.1) it produces
    // WRONG results regardless of precision config or CL/GL backend --
    // task 3's GPU-delegate correctness bisect (docs/rendering-engines.md)
    // found MAX_PRECISION+fp32 still outputs all-zero on the full 72-node
    // graph, OpenCL is entirely unloadable on this device ("undefined
    // symbol: clGetCommandBufferInfoKHR"), and isolated single-op probes
    // for the 3 likely-suspect ops (nearest-upsample, LeakyReLU, 4D jitter
    // broadcast) all show CPU/GPU matching exactly -- not root-caused to a
    // specific op, but conclusively not fixable by a delegate-options
    // change alone. CPU/XNNPACK is the only currently-correct backend for
    // this model on this device; re-enable via the `force_gpu_net` marker
    // file only for further debugging (gpu_probe.{h,cpp}'s isolated probes
    // are the better tool for that now, not this whole-model toggle).
    FILE* marker = fopen("force_gpu_net", "r");
    bool forceGpu = (marker != nullptr);
    if (marker) fclose(marker);

    // Task 3 (docs/rendering-engines.md, GPU-delegate correctness bisect):
    // runtime-configurable via gpu_net_config.txt (cwd-relative, same as
    // force_gpu_net) instead of a rebuild per experiment -- one
    // "key=value" per line, keys: precision=max|min (max ->
    // is_precision_loss_allowed=0 + priority1=MAX_PRECISION, matching
    // TfLiteGpuDelegateOptionsV2Default() exactly; min -> this file's
    // prior hardcoded MIN_LATENCY override, kept only for A/B comparison)
    // and backend=auto|cl|gl (-> experimental_flags CL_ONLY/GL_ONLY).
    std::string precisionCfg = "max", backendCfg = "auto";
    FILE* cfgFile = fopen("gpu_net_config.txt", "r");
    if (cfgFile) {
        char line[128];
        while (fgets(line, sizeof(line), cfgFile)) {
            std::string s(line);
            auto eq = s.find('=');
            if (eq == std::string::npos) continue;
            std::string key = s.substr(0, eq);
            std::string val = s.substr(eq + 1);
            while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == ' ')) val.pop_back();
            if (key == "precision") precisionCfg = val;
            else if (key == "backend") backendCfg = val;
        }
        fclose(cfgFile);
    }

    TfLiteGpuDelegateOptionsV2 gpuOpts = TfLiteGpuDelegateOptionsV2Default();
    gpuOpts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    if (precisionCfg == "min") {
        gpuOpts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    } else {
        gpuOpts.is_precision_loss_allowed = 0;
        gpuOpts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }
    if (backendCfg == "cl") {
        gpuOpts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
    } else if (backendCfg == "gl") {
        gpuOpts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
    }
    impl_->gpuDelegate = forceGpu ? TfLiteGpuDelegateV2Create(&gpuOpts) : nullptr;
    if (forceGpu) {
        LOGI("NetRunner: force_gpu_net marker present, enabling GPU delegate (precision=%s backend=%s, "
             "is_precision_loss_allowed=%d priority1=%d experimental_flags=%lld)",
             precisionCfg.c_str(), backendCfg.c_str(), gpuOpts.is_precision_loss_allowed,
             gpuOpts.inference_priority1, static_cast<long long>(gpuOpts.experimental_flags));
    }
    if (impl_->gpuDelegate) {
        TfLiteInterpreterOptionsAddDelegate(impl_->options, impl_->gpuDelegate);
    } else if (forceGpu) {
        LOGE("TfLiteGpuDelegateV2Create failed even with force_gpu_net set; using CPU (XNNPACK)");
    } else {
        LOGI("NetRunner: GPU delegate disabled by default (see init()'s comment); using CPU (XNNPACK%s)",
             xnnpackForceFp16 ? ", explicit FORCE_FP16" : "");
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
            TfLiteInterpreterOptionsSetNumThreads(impl_->options, numThreads);
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
    LOGI("NetRunner: model=%s inputs=%d gpu_delegate=%d cpu_threads=%d xnnpack_force_fp16=%d",
         modelPath.c_str(), nIn, gpuDelegateActive_, numThreads, xnnpackForceFp16 && impl_->xnnpackDelegate != nullptr);
    impl_->idxInput = 0;
    for (int32_t i = 0; i < nIn; i++) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(impl_->interpreter, i);
        std::string name = TfLiteTensorName(t) ? TfLiteTensorName(t) : "";
        LOGI("  IN[%d] name=%s dims=%d", i, name.c_str(), TfLiteTensorNumDims(t));
        if (name == "net_input") impl_->idxInput = i;
    }
    if (nIn != 1) {
        LOGE("NetRunner: expected a single-input pooled model, got %d inputs -- using index 0 regardless", nIn);
    }

    impl_->outputTensor = const_cast<TfLiteTensor*>(TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0));
    int32_t outDims = TfLiteTensorNumDims(impl_->outputTensor);
    outputChannels_ = static_cast<uint32_t>(TfLiteTensorDim(impl_->outputTensor, outDims - 1));
    LOGI("NetRunner: output channels=%u (expect sp*sp*K*K + sp*sp*3 + hidden)", outputChannels_);

    bufOutput_.resize(static_cast<size_t>(netW_) * netH_ * outputChannels_);
    // bufPooled_ is only actually allocated/used in run() when hiddenInNetRes
    // != nullptr (see its call site) -- reserve now so that first real (post
    // -warmup) frame doesn't pay a vector-growth cost mid-measurement.
    bufPooled_.resize(static_cast<size_t>(netW_) * netH_ * inputChannels_);
}

const float* NetRunner::run(const float* netTensor26ch, const float* hiddenInNetRes, RunTimingsMs& timings) {
    auto tAdapter = std::chrono::high_resolution_clock::now();
    const float* toUpload = netTensor26ch;
    if (hiddenInNetRes != nullptr) {
        // InputAssembly's own tensor always has zero in its trailing
        // hiddenChannels_ (its run() is always called with hiddenIn=nullptr
        // in this app -- see net_runner.h's class comment): copy the whole
        // tensor once, then overwrite just that trailing slice per pixel
        // with the real recurrent state. Far cheaper than the old adapter
        // (touches inputChannels_ floats/pixel at NET res, not a proxy-res
        // reconstruction of every channel).
        std::memcpy(bufPooled_.data(), netTensor26ch, bufPooled_.size() * sizeof(float));
        const size_t n = static_cast<size_t>(netW_) * netH_;
        for (size_t p = 0; p < n; p++) {
            std::memcpy(&bufPooled_[p * inputChannels_ + (inputChannels_ - hiddenChannels_)],
                        &hiddenInNetRes[p * hiddenChannels_], hiddenChannels_ * sizeof(float));
        }
        toUpload = bufPooled_.data();
    }
    timings.adapterMs = msSince(tAdapter);

    auto tUpload = std::chrono::high_resolution_clock::now();
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(impl_->interpreter, impl_->idxInput);
    TfLiteTensorCopyFromBuffer(inputTensor, toUpload,
                                static_cast<size_t>(netW_) * netH_ * inputChannels_ * sizeof(float));
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
