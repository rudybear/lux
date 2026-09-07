#pragma once

// LiteRT Next GPU backend for NetRunner (mobile-DLSS Android live demo).
//
// Verified on-device (Pixel 9 Pro XL, bench/android_net/litert_harness +
// bench/android_net/litert_app in mobiledlss): LiteRT Next's C API
// (LiteRtCreateCompiledModel with kLiteRtHwAcceleratorGpu, ML Drift/OpenCL
// backend, com.google.ai.edge.litert 2.1.6 -- 2.2.0 crashes natively on
// this device, see third_party/litert/README) runs the pooled-input model
// at a median ~15-18ms end-to-end (incl. forced readback sync) vs XNNPACK's
// ~37-44ms, output matching the fp32 reference to max|diff|~0.21 mean~0.008
// -- same ballpark as the existing XNNPACK path's own fp16-weight
// quantization error, not a new source of error.
//
// This header exposes a tiny opaque-pointer C++ API (not the raw LiteRT C
// API directly) so net_runner.h doesn't need to include any LiteRT headers
// (litert/c/litert_common.h etc.) or link against libLiteRt.so unless this
// backend is actually selected at runtime -- net_runner.cpp is the only
// caller, from inside its own init()/run()/~NetRunner().
#include <cstdint>
#include <string>

struct LiteRtNetRunnerImpl;

struct LiteRtNetRunnerTimingsMs {
    double uploadMs = 0.0;    // LiteRtLockTensorBuffer(write) + memcpy + Unlock
    double inferMs = 0.0;     // LiteRtRunCompiledModel
    double downloadMs = 0.0;  // LiteRtLockTensorBuffer(read) + memcpy + Unlock (forces the GPU sync)
};

// Creates the LiteRT environment + GPU-accelerated CompiledModel for
// modelPath (a single-input, single-output NHWC graph -- exactly
// export_tflite_pooled_input's pooled-input graph, see net_runner.h).
// netW/netH/inputChannels give the input tensor's flattened element count
// (netW*netH*inputChannels); the output tensor's channel count is read back
// from the model itself (its last dim, same as the XNNPACK Impl's
// TfLiteTensorDim(outputTensor, outDims-1) query) into *outOutputChannels,
// so NetRunner::init() can size its own bufOutput_/outputChannels_ exactly
// as it already does for the XNNPACK path. Returns nullptr on ANY failure
// (GPU accelerator unavailable, compile failure, ...) -- caller falls back
// to the XNNPACK Impl in that case, exactly like the old force_gpu_net
// TFLite-GPU-delegate fallback path.
LiteRtNetRunnerImpl* LiteRtNetRunner_Create(const std::string& modelPath, uint32_t netW, uint32_t netH,
                                             uint32_t inputChannels, uint32_t* outOutputChannels);

// Uploads `input` (inputFloats floats, from Create), runs the compiled
// model on the GPU accelerator, and returns a pointer (owned by impl, valid
// until the next call) to the output tensor's outputFloats floats.
const float* LiteRtNetRunner_Run(LiteRtNetRunnerImpl* impl, const float* input,
                                  LiteRtNetRunnerTimingsMs& timings);

// Destroys the tensor buffers + compiled model + model. Deliberately does
// NOT destroy the LiteRtEnvironment -- doing so after any GPU-accelerator
// use segfaults during teardown on this device/library version (root cause
// not pinned down; reproduced with a minimal standalone harness, isolated
// to LiteRtDestroyEnvironment specifically -- destroying everything else in
// any order is fine). Harmless to leak: NetRunner is a process-lifetime
// singleton in this app (one AppState, no re-init), so the environment
// would only ever be destroyed once, at process exit, where the OS reclaims
// it anyway.
void LiteRtNetRunner_Destroy(LiteRtNetRunnerImpl* impl);
