#pragma once

#include <string>

// Task 3 (docs/rendering-engines.md, GPU-delegate correctness bisect):
// runs every probe model under `probesRootDir` (one subdirectory per probe,
// each containing model_float32.tflite, input0.npy[, input1.npy, ...],
// ref_output.npy -- built by the sibling mobiledlss checkout's throwaway
// build_probes.py script, NOT checked into this repo) with CPU (XNNPACK)
// and GPU (GL-forced, matching what NetRunner's own delegate actually falls
// back to on this device -- OpenCL isn't loadable at all here, see
// net_runner.cpp's init() log) delegates, logging max|diff| of each against
// the PyTorch-eager reference AND against each other, via LOGI so
// `adb logcat -d | grep GPU_PROBE` gives the verdict. Isolated, minimal
// single-input(s)/single-output TFLite harness -- does NOT reuse NetRunner's
// 8-named-input contract, since these probes are arbitrary small ops, not
// the real ParamPredUNet graph.
void runGpuProbes(const std::string& probesRootDir);
