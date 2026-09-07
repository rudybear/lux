#include "MPSGraphUNet.h"
#include "metal_context.h"
#include "dlss_io.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace {

struct Layer {
    std::string name, type;
    uint32_t cin, cout, kh, kw, weightOffset, biasOffset;
    uint32_t stride() const { return type.find("_s2_") != std::string::npos ? 2u : 1u; }
    uint32_t pad() const { return kh == 3 ? 1u : 0u; }
    bool activate() const { return type != "conv1x1"; }
};

struct Manifest {
    uint32_t inChannels, outChannels, k, hidden, upscale, paramStride;
    std::vector<Layer> layers;
};

// Same plain-text sidecar tools/export_lux_weights.py writes and
// playground_cpp/src/metal_unet_runner.cpp already parses this way.
Manifest loadSidecar(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("MPSGraphUNet: cannot open " + path);
    Manifest m{};
    uint32_t numLayers;
    f >> m.inChannels >> m.outChannels >> m.k >> m.hidden >> m.upscale >> m.paramStride >> numLayers;
    for (uint32_t i = 0; i < numLayers; ++i) {
        Layer l;
        f >> l.name >> l.type >> l.cin >> l.cout >> l.kh >> l.kw >> l.weightOffset >> l.biasOffset;
        m.layers.push_back(l);
    }
    return m;
}

std::vector<uint16_t> loadBlobFp16(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("MPSGraphUNet: cannot open " + path);
    size_t sizeBytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint16_t> raw(sizeBytes / 2);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(sizeBytes));
    return raw;
}

// tools/export_lux_weights.py stores each conv weight channel-last as
// [cout, kh, kw, cin] ("OHWI" -- convenient for a per-output-pixel NHWC GPU
// kernel). MPSGraph's convolution2DWithSourceTensor:...weightsLayout:HWIO
// wants [kh, kw, cin, cout] instead -- a pure reorder (both are fp16 bit
// patterns already; no numeric conversion needed).
std::vector<uint16_t> ohwiToHwio(const uint16_t* src, uint32_t cout, uint32_t kh, uint32_t kw, uint32_t cin) {
    std::vector<uint16_t> dst(static_cast<size_t>(cout) * kh * kw * cin);
    for (uint32_t o = 0; o < cout; o++)
        for (uint32_t y = 0; y < kh; y++)
            for (uint32_t x = 0; x < kw; x++)
                for (uint32_t i = 0; i < cin; i++) {
                    size_t srcIdx = ((static_cast<size_t>(o) * kh + y) * kw + x) * cin + i;
                    size_t dstIdx = ((static_cast<size_t>(y) * kw + x) * cin + i) * cout + o;
                    dst[dstIdx] = src[srcIdx];
                }
    return dst;
}

}  // namespace

MPSGraphUNet::~MPSGraphUNet() {
    // ARC manages the Objective-C objects behind these opaque void* handles
    // (see the __bridge_transfer release below) -- nothing to do here.
    if (executable_) { MPSGraphExecutable* e = (__bridge_transfer MPSGraphExecutable*)executable_; (void)e; }
    if (graph_) { MPSGraph* g = (__bridge_transfer MPSGraph*)graph_; (void)g; }
}

void MPSGraphUNet::init(MetalContext& ctx, const std::string& weightsBinPath,
                         const std::string& layersTxtPath, uint32_t netW, uint32_t netH) {
    @autoreleasepool {
        Manifest manifest = loadSidecar(layersTxtPath);
        std::vector<uint16_t> blob = loadBlobFp16(weightsBinPath);

        netW_ = netW;
        netH_ = netH;
        inChannels_ = manifest.inChannels;
        outChannels_ = manifest.outChannels;
        k_ = manifest.k;
        hidden_ = manifest.hidden;
        upscale_ = manifest.upscale;
        paramStride_ = manifest.paramStride;

        if (netW % 8 != 0 || netH % 8 != 0) {
            throw std::runtime_error("MPSGraphUNet: net resolution must be a multiple of 8 (3 stride-2 stages)");
        }

        id<MTLDevice> device = (__bridge id<MTLDevice>)ctx.device;
        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];

        NSArray<NSNumber*>* inShape = @[@1, @(netH), @(netW), @(inChannels_)];
        MPSGraphTensor* input = [graph placeholderWithShape:inShape dataType:MPSDataTypeFloat16 name:@"input"];

        // Per-layer HWIO weight + [1,1,1,cout] bias constants, built once from the
        // exported blob (transposed OHWI->HWIO -- see ohwiToHwio above).
        auto convBlock = [&](MPSGraphTensor* x, const Layer& l) -> MPSGraphTensor* {
            std::vector<uint16_t> wHwio = ohwiToHwio(blob.data() + l.weightOffset, l.cout, l.kh, l.kw, l.cin);
            NSData* wData = [NSData dataWithBytes:wHwio.data() length:wHwio.size() * sizeof(uint16_t)];
            NSData* bData = [NSData dataWithBytes:blob.data() + l.biasOffset length:static_cast<size_t>(l.cout) * sizeof(uint16_t)];
            NSArray<NSNumber*>* wshape = @[@(l.kh), @(l.kw), @(l.cin), @(l.cout)];
            NSArray<NSNumber*>* bshape = @[@1, @1, @1, @(l.cout)];
            MPSGraphTensor* wt = [graph constantWithData:wData shape:wshape dataType:MPSDataTypeFloat16];
            MPSGraphTensor* bt = [graph constantWithData:bData shape:bshape dataType:MPSDataTypeFloat16];
            MPSGraphConvolution2DOpDescriptor* desc = [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:l.stride()
                              strideInY:l.stride()
                        dilationRateInX:1
                        dilationRateInY:1
                                 groups:1
                            paddingLeft:l.pad()
                           paddingRight:l.pad()
                             paddingTop:l.pad()
                          paddingBottom:l.pad()
                           paddingStyle:MPSGraphPaddingStyleExplicit
                             dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                          weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
            MPSGraphTensor* conv = [graph convolution2DWithSourceTensor:x weightsTensor:wt descriptor:desc name:nil];
            MPSGraphTensor* biased = [graph additionWithPrimaryTensor:conv secondaryTensor:bt name:nil];
            if (l.activate()) return [graph leakyReLUWithTensor:biased alpha:0.1 name:nil];
            return biased;
        };

        // PyTorch F.interpolate(mode="nearest", scale_factor=2)'s floor-division
        // index mapping -- MPSGraphResizeNearestRoundingModeFloor, NOT the
        // resizeTensor: family's default RoundPreferCeil (verified mismatched
        // against a hand-computed 2x2 case in bench/mpsgraph/bench.mm's own
        // bring-up -- see that file's comment on this exact pitfall).
        auto upsample2x = [&](MPSGraphTensor* x, int newH, int newW) -> MPSGraphTensor* {
            int32_t sz[2] = {newH, newW};
            MPSGraphTensor* sizeT = [graph constantWithData:[NSData dataWithBytes:sz length:sizeof(sz)]
                                                        shape:@[@2]
                                                     dataType:MPSDataTypeInt32];
            return [graph resizeNearestWithTensor:x
                                        sizeTensor:sizeT
                               nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                      centerResult:NO
                                      alignCorners:NO
                                            layout:MPSGraphTensorNamedDataLayoutNHWC
                                              name:nil];
        };

        int H = static_cast<int>(netH), W = static_cast<int>(netW);
        int H1 = H / 2, W1 = W / 2, H2 = H1 / 2, W2 = W1 / 2;

        const auto& L = manifest.layers;
        MPSGraphTensor* x0 = convBlock(input, L[0]);
        MPSGraphTensor* x1d = convBlock(x0, L[1]);
        MPSGraphTensor* x1 = convBlock(x1d, L[2]);
        MPSGraphTensor* x2d = convBlock(x1, L[3]);
        MPSGraphTensor* x2 = convBlock(x2d, L[4]);
        MPSGraphTensor* x3d = convBlock(x2, L[5]);
        MPSGraphTensor* x3 = convBlock(x3d, L[6]);
        MPSGraphTensor* xb = convBlock(x3, L[7]);

        MPSGraphTensor* u3in = [graph concatTensor:upsample2x(xb, H2, W2) withTensor:x2 dimension:3 name:nil];
        MPSGraphTensor* u3 = convBlock(u3in, L[8]);

        MPSGraphTensor* u2in = [graph concatTensor:upsample2x(u3, H1, W1) withTensor:x1 dimension:3 name:nil];
        MPSGraphTensor* u2 = convBlock(u2in, L[9]);

        MPSGraphTensor* u1in = [graph concatTensor:upsample2x(u2, H, W) withTensor:x0 dimension:3 name:nil];
        MPSGraphTensor* u1 = convBlock(u1in, L[10]);

        MPSGraphTensor* output = convBlock(u1, L[11]);

        MPSGraphShapedType* feedType = [[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16];
        MPSGraphExecutable* exe = [graph compileWithDevice:gdev
                                                      feeds:@{input : feedType}
                                              targetTensors:@[output]
                                           targetOperations:nil
                                      compilationDescriptor:nil];

        graph_ = (__bridge_retained void*)graph;
        inputTensor_ = (__bridge void*)input;    // owned by `graph`, kept alive by graph_
        outputTensor_ = (__bridge void*)output;  // ditto
        executable_ = (__bridge_retained void*)exe;
    }
}

void MPSGraphUNet::encode(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, MTL::Buffer* inputBuffer,
                           MTL::Buffer* outputBuffer) {
    @autoreleasepool {
        MPSGraphExecutable* exe = (__bridge MPSGraphExecutable*)executable_;
        id<MTLCommandBuffer> mtlCB = (__bridge id<MTLCommandBuffer>)cmdBuf;
        MPSCommandBuffer* mpsCB = [MPSCommandBuffer commandBufferWithCommandBuffer:mtlCB];

        id<MTLBuffer> inBuf = (__bridge id<MTLBuffer>)inputBuffer;
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)outputBuffer;

        NSArray<NSNumber*>* inShape = @[@1, @(netH_), @(netW_), @(inChannels_)];
        NSArray<NSNumber*>* outShape = @[@1, @(netH_), @(netW_), @(outChannels_)];
        MPSGraphTensorData* inputTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf
                                                                                shape:inShape
                                                                             dataType:MPSDataTypeFloat16];
        MPSGraphTensorData* outputTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf
                                                                                 shape:outShape
                                                                              dataType:MPSDataTypeFloat16];
        // resultsArray: writes directly into our own outputBuffer (no internal
        // allocation + extra copy) -- readable by the next pass (B4's reconstruct)
        // with zero host round-trip.
        [exe encodeToCommandBuffer:mpsCB
                        inputsArray:@[inputTD]
                       resultsArray:@[outputTD]
                executionDescriptor:nil];
        // MPSGraph's encode may internally commitAndContinue, replacing the
        // underlying MTLCommandBuffer -- the caller must keep encoding into
        // mpsCB.rootCommandBuffer afterwards, not the original cmdBuf (see
        // MPSGraph.h's encode doc comment, and bench.mm's identical note).
        // Since metal-cpp's MTL::CommandBuffer* is just a thin wrapper around
        // the same id<MTLCommandBuffer>, and this project's convention is
        // "commit immediately after each renderer's own pass" (see
        // MetalSplatRenderer::render()), the caller commits+waits right after
        // this call, so root-swap risk is contained to within this one call.
        // Always commit+wait the *current* root (whether or not MPSGraph swapped
        // it out from under us) -- this call owns cmdBuf's lifetime entirely, by
        // this project's "one pass, one command buffer, commit inside the call"
        // convention (matches MetalSplatRenderer::render()).
        id<MTLCommandBuffer> root = mpsCB.rootCommandBuffer;
        [root commit];
        [root waitUntilCompleted];
    }
}
