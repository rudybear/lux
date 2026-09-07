#include "net_runner_litert.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

#include <android/log.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>

#define LOG_TAG "lux_android_net"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
inline double msSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}
const char* bufferTypeName(LiteRtTensorBufferType t) {
    switch (t) {
        case kLiteRtTensorBufferTypeHostMemory: return "HostMemory";
        case kLiteRtTensorBufferTypeAhwb: return "Ahwb";
        case kLiteRtTensorBufferTypeGlBuffer: return "GlBuffer";
        case kLiteRtTensorBufferTypeGlTexture: return "GlTexture";
        case kLiteRtTensorBufferTypeOpenClBuffer: return "OpenClBuffer";
        case kLiteRtTensorBufferTypeOpenClBufferFp16: return "OpenClBufferFp16";
        case kLiteRtTensorBufferTypeOpenClTexture: return "OpenClTexture";
        case kLiteRtTensorBufferTypeOpenClTextureFp16: return "OpenClTextureFp16";
        case kLiteRtTensorBufferTypeOpenClBufferPacked: return "OpenClBufferPacked";
        case kLiteRtTensorBufferTypeOpenClImageBuffer: return "OpenClImageBuffer";
        case kLiteRtTensorBufferTypeOpenClImageBufferFp16: return "OpenClImageBufferFp16";
        default: return "Other";
    }
}
}  // namespace

struct LiteRtNetRunnerImpl {
    LiteRtEnvironment environment = nullptr;
    LiteRtModel model = nullptr;
    LiteRtCompiledModel compiledModel = nullptr;
    LiteRtTensorBuffer inputBuf = nullptr;
    LiteRtTensorBuffer outputBuf = nullptr;
    size_t inputFloats = 0;
    size_t outputFloats = 0;

    // Headless EGL/GLES context (see LiteRtNetRunner_Create's comment on
    // why this is needed at all): this app is Vulkan-only, so without this
    // there is no current GL context on the calling thread, and this
    // device's LiteRT GPU accelerator falls back from OpenCL (blocked by
    // Android's app-sandbox SELinux policy -- untrusted_app can't dlopen
    // the vendor CL driver, confirmed by "OpenCL not supported on this
    // platform" appearing ONLY inside the app, never in the equivalent
    // `adb shell`-run standalone harness) to OpenGL, whose managed tensor
    // buffers (kLiteRtTensorBufferTypeGlBuffer) need a current context to
    // create. Deliberately never torn down (see LiteRtNetRunner_Destroy) --
    // same process-lifetime-singleton reasoning as the LiteRtEnvironment
    // leak below, and avoids a second teardown-ordering landmine.
    EGLDisplay eglDisplay = EGL_NO_DISPLAY;
    EGLSurface eglSurface = EGL_NO_SURFACE;
    EGLContext eglContext = EGL_NO_CONTEXT;
    std::vector<float> outStaging;  // returned to caller (LiteRtLockTensorBuffer's mapped
                                     // pointer is only valid between Lock/Unlock, so run()
                                     // copies it out here before unlocking -- same "valid
                                     // until next run()" contract as the XNNPACK Impl's
                                     // bufOutput_).
};

LiteRtNetRunnerImpl* LiteRtNetRunner_Create(const std::string& modelPath, uint32_t netW, uint32_t netH,
                                             uint32_t inputChannels, uint32_t* outOutputChannels) {
    auto* impl = new LiteRtNetRunnerImpl();
    impl->inputFloats = static_cast<size_t>(netW) * netH * inputChannels;

    auto fail = [&](const char* what, LiteRtStatus st) {
        LOGE("LiteRtNetRunner_Create: %s failed (status=%d)", what, static_cast<int>(st));
        // Best-effort cleanup of whatever got created before the failure --
        // never destroy the environment here either, see net_runner_litert.h.
        if (impl->compiledModel) LiteRtDestroyCompiledModel(impl->compiledModel);
        if (impl->model) LiteRtDestroyModel(impl->model);
        delete impl;
        return static_cast<LiteRtNetRunnerImpl*>(nullptr);
    };

    // --- headless EGL/GLES3 context, see the struct field comment above ---
    impl->eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (impl->eglDisplay == EGL_NO_DISPLAY) {
        LOGE("LiteRtNetRunner_Create: eglGetDisplay failed");
        delete impl;
        return nullptr;
    }
    EGLint eglMajor = 0, eglMinor = 0;
    if (!eglInitialize(impl->eglDisplay, &eglMajor, &eglMinor)) {
        LOGE("LiteRtNetRunner_Create: eglInitialize failed (0x%x)", eglGetError());
        delete impl;
        return nullptr;
    }
    const EGLint configAttribs[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
                                     EGL_OPENGL_ES3_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
                                     EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
    EGLConfig eglConfig;
    EGLint numConfigs = 0;
    if (!eglChooseConfig(impl->eglDisplay, configAttribs, &eglConfig, 1, &numConfigs) || numConfigs < 1) {
        LOGE("LiteRtNetRunner_Create: eglChooseConfig failed (0x%x)", eglGetError());
        delete impl;
        return nullptr;
    }
    const EGLint pbufferAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    impl->eglSurface = eglCreatePbufferSurface(impl->eglDisplay, eglConfig, pbufferAttribs);
    if (impl->eglSurface == EGL_NO_SURFACE) {
        LOGE("LiteRtNetRunner_Create: eglCreatePbufferSurface failed (0x%x)", eglGetError());
        delete impl;
        return nullptr;
    }
    const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    impl->eglContext = eglCreateContext(impl->eglDisplay, eglConfig, EGL_NO_CONTEXT, contextAttribs);
    if (impl->eglContext == EGL_NO_CONTEXT) {
        LOGE("LiteRtNetRunner_Create: eglCreateContext failed (0x%x)", eglGetError());
        delete impl;
        return nullptr;
    }
    if (!eglMakeCurrent(impl->eglDisplay, impl->eglSurface, impl->eglSurface, impl->eglContext)) {
        LOGE("LiteRtNetRunner_Create: eglMakeCurrent failed (0x%x)", eglGetError());
        delete impl;
        return nullptr;
    }
    LOGI("LiteRtNetRunner_Create: headless EGL context ready (EGL %d.%d, GL_VERSION=%s)", eglMajor, eglMinor,
         reinterpret_cast<const char*>(glGetString(GL_VERSION)));

    LiteRtStatus st = LiteRtCreateEnvironment(/*num_options=*/0, nullptr, &impl->environment);
    if (st != kLiteRtStatusOk) return fail("LiteRtCreateEnvironment", st);

    st = LiteRtCreateModelFromFile(impl->environment, modelPath.c_str(), &impl->model);
    if (st != kLiteRtStatusOk) return fail("LiteRtCreateModelFromFile", st);

    LiteRtOptions options;
    st = LiteRtCreateOptions(&options);
    if (st != kLiteRtStatusOk) return fail("LiteRtCreateOptions", st);
    st = LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorGpu);
    if (st != kLiteRtStatusOk) {
        LiteRtDestroyOptions(options);
        return fail("LiteRtSetOptionsHardwareAccelerators", st);
    }

    st = LiteRtCreateCompiledModel(impl->environment, impl->model, options, &impl->compiledModel);
    LiteRtDestroyOptions(options);
    if (st != kLiteRtStatusOk) return fail("LiteRtCreateCompiledModel(GPU)", st);

    LiteRtSubgraph subgraph;
    st = LiteRtGetModelSubgraph(impl->model, 0, &subgraph);
    if (st != kLiteRtStatusOk) return fail("LiteRtGetModelSubgraph", st);

    LiteRtParamIndex numInputs = 0, numOutputs = 0;
    LiteRtGetNumSubgraphInputs(subgraph, &numInputs);
    LiteRtGetNumSubgraphOutputs(subgraph, &numOutputs);
    if (numInputs != 1 || numOutputs != 1) {
        LOGE("LiteRtNetRunner_Create: expected a single-input/single-output pooled model, got %u/%u",
             static_cast<unsigned>(numInputs), static_cast<unsigned>(numOutputs));
        return fail("input/output count check", kLiteRtStatusErrorInvalidArgument);
    }

    // Output channel count comes from the model itself (its output tensor's
    // last dim), same as the XNNPACK Impl's TfLiteTensorDim query -- needed
    // to size outStaging_/outputFloats before the buffer-requirements block
    // below can create the output managed buffer.
    {
        LiteRtTensor outTensor;
        st = LiteRtGetSubgraphOutput(subgraph, 0, &outTensor);
        if (st != kLiteRtStatusOk) return fail("LiteRtGetSubgraphOutput", st);
        LiteRtRankedTensorType rankedType;
        st = LiteRtGetRankedTensorType(outTensor, &rankedType);
        if (st != kLiteRtStatusOk) return fail("LiteRtGetRankedTensorType(output)", st);
        int32_t rank = rankedType.layout.rank;
        if (rank < 1) return fail("output tensor rank check", kLiteRtStatusErrorInvalidArgument);
        uint32_t outputChannels = static_cast<uint32_t>(rankedType.layout.dimensions[rank - 1]);
        impl->outputFloats = static_cast<size_t>(netW) * netH * outputChannels;
        impl->outStaging.resize(impl->outputFloats);
        *outOutputChannels = outputChannels;
        LOGI("LiteRtNetRunner_Create: output_channels=%u output_floats=%zu", outputChannels, impl->outputFloats);
    }

    // Task 3 (docs/rendering-engines.md, avoid CPU copies where cheap):
    // the buffer type the GPU accelerator actually wants is whatever
    // LiteRtGetTensorBufferRequirementsSupportedTensorBufferType reports
    // here -- logged for both input and output so on-device logcat shows
    // exactly which stage(s) still round-trip through a CPU memcpy (see
    // this function's LOGI lines below). On this device/library the CL
    // backend reports OpenCL buffer types, NOT host memory, but
    // LiteRtCreateManagedTensorBuffer + Lock/Unlock (used below) works
    // uniformly across all of them -- Lock returns a CPU-mapped view
    // regardless of the underlying storage, at the cost of an implicit
    // copy for non-host types. Wiring InputAssembly/ReconstructLive's own
    // Vulkan buffers directly into one of these types (AHWB is the
    // natural interop point -- both Vulkan and OpenCL can import the same
    // AHardwareBuffer) would remove that copy, but requires InputAssembly
    // to produce an AHWB-backed staging buffer instead of a host `float*`,
    // which it does not today -- see this file's design comment / the
    // task report for the concrete follow-up.
    {
        LiteRtTensorBufferRequirements reqs;
        if (LiteRtGetCompiledModelInputBufferRequirements(impl->compiledModel, 0, 0, &reqs) == kLiteRtStatusOk) {
            LiteRtTensorBufferType t;
            if (LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(reqs, 0, &t) == kLiteRtStatusOk) {
                LOGI("LiteRtNetRunner: input buffer type = %s (%d)", bufferTypeName(t), static_cast<int>(t));
            }
        }
        if (LiteRtGetCompiledModelOutputBufferRequirements(impl->compiledModel, 0, 0, &reqs) == kLiteRtStatusOk) {
            LiteRtTensorBufferType t;
            if (LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(reqs, 0, &t) == kLiteRtStatusOk) {
                LOGI("LiteRtNetRunner: output buffer type = %s (%d)", bufferTypeName(t), static_cast<int>(t));
            }
        }
    }

    auto makeManagedBuffer = [&](bool isInput, LiteRtTensorBuffer* outBuf) -> LiteRtStatus {
        LiteRtTensorBufferRequirements reqs;
        LiteRtStatus rst = isInput
                                ? LiteRtGetCompiledModelInputBufferRequirements(impl->compiledModel, 0, 0, &reqs)
                                : LiteRtGetCompiledModelOutputBufferRequirements(impl->compiledModel, 0, 0, &reqs);
        if (rst != kLiteRtStatusOk) return rst;
        LiteRtTensorBufferType bufType;
        rst = LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(reqs, 0, &bufType);
        if (rst != kLiteRtStatusOk) return rst;
        size_t bufSize;
        rst = LiteRtGetTensorBufferRequirementsBufferSize(reqs, &bufSize);
        if (rst != kLiteRtStatusOk) return rst;

        LiteRtTensor tensor;
        rst = isInput ? LiteRtGetSubgraphInput(subgraph, 0, &tensor) : LiteRtGetSubgraphOutput(subgraph, 0, &tensor);
        if (rst != kLiteRtStatusOk) return rst;
        LiteRtRankedTensorType rankedType;
        rst = LiteRtGetRankedTensorType(tensor, &rankedType);
        if (rst != kLiteRtStatusOk) return rst;

        return LiteRtCreateManagedTensorBuffer(impl->environment, bufType, &rankedType, bufSize, outBuf);
    };

    st = makeManagedBuffer(/*isInput=*/true, &impl->inputBuf);
    if (st != kLiteRtStatusOk) return fail("create input tensor buffer", st);
    st = makeManagedBuffer(/*isInput=*/false, &impl->outputBuf);
    if (st != kLiteRtStatusOk) return fail("create output tensor buffer", st);

    LOGI("LiteRtNetRunner_Create: OK (model=%s, GPU accelerator active)", modelPath.c_str());
    return impl;
}

const float* LiteRtNetRunner_Run(LiteRtNetRunnerImpl* impl, const float* input,
                                  LiteRtNetRunnerTimingsMs& timings) {
    // Cheap no-op if already current on this thread (same thread as
    // Create(), in this app) -- defends against the GL context somehow not
    // being current (e.g. a future caller running Create()/Run() on
    // different threads), see the struct field comment above.
    eglMakeCurrent(impl->eglDisplay, impl->eglSurface, impl->eglSurface, impl->eglContext);

    auto tUpload = std::chrono::high_resolution_clock::now();
    void* hostMem = nullptr;
    LiteRtStatus st = LiteRtLockTensorBuffer(impl->inputBuf, &hostMem, kLiteRtTensorBufferLockModeWrite);
    if (st != kLiteRtStatusOk) throw std::runtime_error("LiteRtNetRunner_Run: LiteRtLockTensorBuffer(input) failed");
    std::memcpy(hostMem, input, impl->inputFloats * sizeof(float));
    LiteRtUnlockTensorBuffer(impl->inputBuf);
    timings.uploadMs = msSince(tUpload);

    auto tInfer = std::chrono::high_resolution_clock::now();
    st = LiteRtRunCompiledModel(impl->compiledModel, /*signature_index=*/0, 1, &impl->inputBuf, 1, &impl->outputBuf);
    timings.inferMs = msSince(tInfer);
    if (st != kLiteRtStatusOk) throw std::runtime_error("LiteRtNetRunner_Run: LiteRtRunCompiledModel failed");

    auto tDownload = std::chrono::high_resolution_clock::now();
    st = LiteRtLockTensorBuffer(impl->outputBuf, &hostMem, kLiteRtTensorBufferLockModeRead);
    if (st != kLiteRtStatusOk) throw std::runtime_error("LiteRtNetRunner_Run: LiteRtLockTensorBuffer(output) failed");
    std::memcpy(impl->outStaging.data(), hostMem, impl->outputFloats * sizeof(float));
    LiteRtUnlockTensorBuffer(impl->outputBuf);
    timings.downloadMs = msSince(tDownload);

    return impl->outStaging.data();
}

void LiteRtNetRunner_Destroy(LiteRtNetRunnerImpl* impl) {
    if (!impl) return;
    if (impl->inputBuf) LiteRtDestroyTensorBuffer(impl->inputBuf);
    if (impl->outputBuf) LiteRtDestroyTensorBuffer(impl->outputBuf);
    if (impl->compiledModel) LiteRtDestroyCompiledModel(impl->compiledModel);
    if (impl->model) LiteRtDestroyModel(impl->model);
    // impl->environment is intentionally leaked -- see net_runner_litert.h.
    delete impl;
}
