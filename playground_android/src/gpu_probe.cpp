#include "gpu_probe.h"
#include "dlss_io.h"

#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define LOG_TAG "lux_android_net"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace fs = std::filesystem;

namespace {

struct RunResult {
    bool ok = false;
    std::vector<float> output;
};

// Runs `modelPath` on `inputs` (positional order -- matches onnx2tf's
// copy_onnx_input_output_names_to_tflite export, but position works
// regardless) with CPU or GL-forced GPU delegate.
RunResult runOnce(const std::string& modelPath, const std::vector<std::vector<float>>& inputs, bool useGpu) {
    RunResult result;
    TfLiteModel* model = TfLiteModelCreateFromFile(modelPath.c_str());
    if (!model) {
        LOGE("GPU_PROBE: failed to load model %s", modelPath.c_str());
        return result;
    }
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 4);
    TfLiteDelegate* gpuDelegate = nullptr;
    if (useGpu) {
        TfLiteGpuDelegateOptionsV2 gpuOpts = TfLiteGpuDelegateOptionsV2Default();
        gpuOpts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
        gpuOpts.is_precision_loss_allowed = 0;
        gpuOpts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        // Force GL: OpenCL isn't loadable at all on this device (NetRunner's
        // own init() log: "Can not open OpenCL library... undefined symbol:
        // clGetCommandBufferInfoKHR"), so testing "auto" here would just
        // silently retest the same GL fallback path -- force it explicitly
        // so the probe's config matches what actually runs.
        gpuOpts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
        gpuDelegate = TfLiteGpuDelegateV2Create(&gpuOpts);
        if (gpuDelegate) TfLiteInterpreterOptionsAddDelegate(options, gpuDelegate);
    }
    TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, options);
    if (!interp || TfLiteInterpreterAllocateTensors(interp) != kTfLiteOk) {
        LOGE("GPU_PROBE: %s AllocateTensors failed (useGpu=%d)", modelPath.c_str(), useGpu);
        if (interp) TfLiteInterpreterDelete(interp);
        if (gpuDelegate) TfLiteGpuDelegateV2Delete(gpuDelegate);
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
        return result;
    }
    int32_t nIn = TfLiteInterpreterGetInputTensorCount(interp);
    if (static_cast<size_t>(nIn) != inputs.size()) {
        LOGE("GPU_PROBE: %s expects %d inputs, probe supplied %zu", modelPath.c_str(), nIn, inputs.size());
    }
    for (int32_t i = 0; i < nIn && static_cast<size_t>(i) < inputs.size(); i++) {
        TfLiteTensor* t = TfLiteInterpreterGetInputTensor(interp, i);
        TfLiteTensorCopyFromBuffer(t, inputs[i].data(), inputs[i].size() * sizeof(float));
    }
    TfLiteStatus invokeStatus = TfLiteInterpreterInvoke(interp);
    if (invokeStatus != kTfLiteOk) {
        LOGE("GPU_PROBE: %s Invoke failed (useGpu=%d, status=%d)", modelPath.c_str(), useGpu, invokeStatus);
    } else {
        const TfLiteTensor* outT = TfLiteInterpreterGetOutputTensor(interp, 0);
        size_t n = TfLiteTensorByteSize(outT) / sizeof(float);
        result.output.resize(n);
        TfLiteTensorCopyToBuffer(outT, result.output.data(), n * sizeof(float));
        result.ok = true;
    }
    TfLiteInterpreterDelete(interp);
    if (gpuDelegate) TfLiteGpuDelegateV2Delete(gpuDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
    return result;
}

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -1.0;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

}  // namespace

void runGpuProbes(const std::string& probesRootDir) {
    std::error_code ec;
    if (!fs::exists(probesRootDir, ec) || !fs::is_directory(probesRootDir, ec)) {
        LOGI("GPU_PROBE: %s not found, skipping probes", probesRootDir.c_str());
        return;
    }
    for (const auto& entry : fs::directory_iterator(probesRootDir, ec)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        std::string dir = entry.path().string();
        std::string modelPath = dir + "/model_float32.tflite";
        if (!fs::exists(modelPath)) {
            LOGI("GPU_PROBE: %s: no model_float32.tflite, skipping", name.c_str());
            continue;
        }

        std::vector<std::vector<float>> inputs;
        for (int i = 0;; i++) {
            std::string p = dir + "/input" + std::to_string(i) + ".npy";
            if (!fs::exists(p)) break;
            inputs.push_back(DlssIO::readNpyFloat32(p).data);
        }
        if (inputs.empty()) {
            LOGI("GPU_PROBE: %s: no input*.npy, skipping", name.c_str());
            continue;
        }
        std::vector<float> ref;
        std::string refPath = dir + "/ref_output.npy";
        if (fs::exists(refPath)) ref = DlssIO::readNpyFloat32(refPath).data;

        RunResult cpu = runOnce(modelPath, inputs, /*useGpu=*/false);
        RunResult gpu = runOnce(modelPath, inputs, /*useGpu=*/true);

        if (!cpu.ok || !gpu.ok) {
            LOGI("GPU_PROBE %-20s cpu_ok=%d gpu_ok=%d (one or both failed to run at all -- see errors above)",
                 name.c_str(), cpu.ok, gpu.ok);
            continue;
        }
        double cpuVsRef = ref.empty() ? -1.0 : maxAbsDiff(cpu.output, ref);
        double gpuVsRef = ref.empty() ? -1.0 : maxAbsDiff(gpu.output, ref);
        double cpuVsGpu = maxAbsDiff(cpu.output, gpu.output);
        LOGI("GPU_PROBE %-20s cpu_vs_pytorch_ref=%.6f gpu_vs_pytorch_ref=%.6f cpu_vs_gpu=%.6f "
             "cpu_out[0..2]=(%.4f,%.4f,%.4f) gpu_out[0..2]=(%.4f,%.4f,%.4f) n=%zu",
             name.c_str(), cpuVsRef, gpuVsRef, cpuVsGpu,
             cpu.output.size() > 2 ? cpu.output[0] : 0.f, cpu.output.size() > 2 ? cpu.output[1] : 0.f,
             cpu.output.size() > 2 ? cpu.output[2] : 0.f,
             gpu.output.size() > 2 ? gpu.output[0] : 0.f, gpu.output.size() > 2 ? gpu.output[1] : 0.f,
             gpu.output.size() > 2 ? gpu.output[2] : 0.f, cpu.output.size());
    }
}
