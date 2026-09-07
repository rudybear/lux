#include "NetInputAssembly.h"
#include "metal_context.h"
#include "dlss_io.h"

#include <stdexcept>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// Uniforms (matches the MSL struct below byte-for-byte: every vector is a
// full float4 -- no packed_float3 -- so C++ and MSL agree on layout with no
// padding surprises, same convention metal_splat_renderer.cpp's raw-byte
// ComputeUniforms uses, just via a plain struct instead of manual offsets).
// ---------------------------------------------------------------------------
struct InputAssemblyUniforms {
    float eye[4];
    float rAxis[4];
    float uAxis[4];
    float fAxis[4];
    float bgCenter[4];
    float fx, fy, cx, cy;
    float bgRadius, mvScale, jitterX, jitterY;
    uint32_t proxyW, proxyH, netW, netH;
    uint32_t paramStride, hiddenChannels, texW, texH;
    uint32_t firstFrame, texChannels, _pad1, _pad2;
};

static const char* kUnpremulDepthMSL = R"(
#include <metal_stdlib>
using namespace metal;

kernel void unpremul_depth(texture2d<float, access::read> depthTex [[texture(0)]],
                            texture2d<float, access::write> outTex [[texture(1)]],
                            constant uint& alphaOffset [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= depthTex.get_width() || gid.y >= depthTex.get_height()) return;
    float4 raw = depthTex.read(gid);  // premul at .x; alpha at .y (RG32F) or .w (RGBA32F)
    float a = (alphaOffset == 1) ? raw.y : raw.w;
    float depth = (a > 1e-6) ? (raw.x / a) : 0.0;
    outTex.write(float4(depth, 0.0, 0.0, 0.0), gid);
}
)";

// Build-input assembly kernel: mobiledlss/train/model.py::build_input, with
// mobiledlss/datagen/camera.py::sphere_uv (bg-sphere UV + texture sample)
// and mobiledlss/datagen/motion.py::disocclusion_mask folded in (their
// inputs -- texture_feat and proxy_disocc -- are exactly the things
// build_input expects the *caller* to have already produced). See
// NetInputAssembly.h's channel-order comment.
static const char* kAssembleMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct InputAssemblyUniforms {
    float4 eye;
    float4 rAxis;
    float4 uAxis;
    float4 fAxis;
    float4 bgCenter;
    float fx, fy, cx, cy;
    float bgRadius, mvScale, jitterX, jitterY;
    uint proxyW, proxyH, netW, netH;
    uint paramStride, hiddenChannels, texW, texH;
    uint firstFrame, texChannels, _pad1, _pad2;
};

// mobiledlss.train.scene_texture.SceneTexture.forward's exact sampling
// convention (derived from its grid_sample(padding_mode='border',
// align_corners=False) call on a 1-texel wrap-padded texture -- see
// NetInputAssembly.mm's derivation comment): u wraps (repeat), v clamps
// (border), texel centres at integer pixel coordinates.
static void sampleBgTexture(device const float* tex, uint texW, uint texH, uint texC,
                             float u, float v, thread float* outFeat) {
    float px = u * float(texW) - 0.5;
    float py = v * float(texH) - 0.5;
    int x0 = int(floor(px));
    int y0 = int(floor(py));
    float fx = px - float(x0);
    float fy = py - float(y0);
    int x0w = ((x0 % int(texW)) + int(texW)) % int(texW);
    int x1w = ((x0 + 1) % int(texW) + int(texW)) % int(texW);
    int y0c = clamp(y0, 0, int(texH) - 1);
    int y1c = clamp(y0 + 1, 0, int(texH) - 1);
    for (uint c = 0; c < texC; c++) {
        device const float* plane = tex + c * texH * texW;
        float t00 = plane[y0c * texW + x0w];
        float t10 = plane[y0c * texW + x1w];
        float t01 = plane[y1c * texW + x0w];
        float t11 = plane[y1c * texW + x1w];
        outFeat[c] = mix(mix(t00, t10, fx), mix(t01, t11, fx), fy);
    }
}

// mobiledlss.datagen.motion.reproject_depth's exact sampling convention
// (grid_sample(padding_mode='zeros', align_corners=False), continuous pixel
// coordinate = proxy_index - mv, see NetInputAssembly.mm's derivation).
static float sampleDepthZeroPad(texture2d<float, access::sample> tex, uint w, uint h,
                                 float cx, float cy) {
    constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_zero);
    // Metal's clamp_to_zero address mode already implements exactly this
    // zero-contribution-outside-bounds bilinear tap behaviour for
    // coord::pixel sampling with texel centres at integer+0.5 -- so a plain
    // hardware sample at (cx+0.5, cy+0.5) matches grid_sample's zeros
    // padding_mode directly (no manual four-tap code needed here, unlike
    // sampleBgTexture's buffer-backed manual implementation above).
    return tex.sample(s, float2(cx + 0.5, cy + 0.5)).x;
}

kernel void assemble_net_input(
    texture2d<float, access::read> colorTex [[texture(0)]],   // RGBA16Float, premultiplied
    texture2d<float, access::read> depthTex [[texture(1)]],   // RG32Float: depth*a, a
    texture2d<float, access::read> motionTex [[texture(2)]],  // RGBA32Float: mv*a, 0, 0, a
    texture2d<float, access::read> curUnpremulDepth [[texture(3)]],   // R32Float
    texture2d<float, access::sample> prevUnpremulDepth [[texture(4)]], // R32Float
    device const float* bgTexture [[buffer(0)]],
    constant InputAssemblyUniforms& u [[buffer(1)]],
    device const half* hiddenIn [[buffer(2)]],   // NHWC, netW*netH*hiddenChannels (may be all-zero)
    device half* out [[buffer(3)]],              // NHWC, netW*netH*(18+hiddenChannels)
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.netW || gid.y >= u.netH) return;

    float3 colorSum = float3(0.0);
    float depthSum = 0.0;
    float2 mvSum = float2(0.0);
    float fgSum = 0.0;
    float disoccSum = 0.0;
    float texFeatSum[8] = {0,0,0,0,0,0,0,0};

    float invN = 1.0 / float(u.paramStride * u.paramStride);
    const float relTol = 0.05;

    for (uint dy = 0; dy < u.paramStride; dy++) {
        for (uint dx = 0; dx < u.paramStride; dx++) {
            // Replicate-pad (mobiledlss.train.export._PaddedExportModel): proxy rows/cols
            // at or past the real proxy size reuse the last real row/col, exactly like
            // F.pad(..., mode='replicate') applied before build_input's box-pool.
            uint px = min(gid.x * u.paramStride + dx, u.proxyW - 1);
            uint py = min(gid.y * u.paramStride + dy, u.proxyH - 1);
            uint2 pgid(px, py);

            float4 colorRaw = colorTex.read(pgid);
            float ca = colorRaw.a;
            float3 rgb = (ca > 1e-6) ? (colorRaw.rgb / ca) : float3(0.0);
            colorSum += rgb;
            fgSum += ca;

            float depthVal = curUnpremulDepth.read(pgid).x;
            depthSum += depthVal;

            float4 motionRaw = motionTex.read(pgid);
            float ma = motionRaw.a;
            float2 mv = (ma > 1e-6) ? (motionRaw.xy / ma) : float2(0.0);
            mvSum += mv;

            // --- disocclusion_mask (motion.py) ---
            float d;
            if (u.firstFrame != 0) {
                d = 1.0;
            } else {
                float pxCode = float(px) + 0.5 - mv.x;
                float pyCode = float(py) + 0.5 - mv.y;
                bool inside = pxCode >= 0.0 && pxCode <= float(u.proxyW) &&
                              pyCode >= 0.0 && pyCode <= float(u.proxyH);
                if (inside) {
                    float cxx = float(px) - mv.x;  // continuous coord = pxCode - 0.5
                    float cyy = float(py) - mv.y;
                    float prevDepth = sampleDepthZeroPad(prevUnpremulDepth, u.proxyW, u.proxyH, cxx, cyy);
                    bool closerBefore = prevDepth < depthVal * (1.0 - relTol);
                    d = closerBefore ? 1.0 : 0.0;
                } else {
                    d = 1.0;
                }
            }
            disoccSum += d;

            // --- bg-sphere UV (camera.py::sphere_uv) + texture sample ---
            float3 dirCam = float3((float(px) + 0.5 - u.cx) / u.fx,
                                    (float(py) + 0.5 - u.cy) / u.fy, 1.0);
            float3 dirWorld = normalize(dirCam.x * u.rAxis.xyz + dirCam.y * u.uAxis.xyz + dirCam.z * u.fAxis.xyz);
            float3 o = u.eye.xyz;
            float3 oc = o - u.bgCenter.xyz;
            float b = dot(dirWorld, oc);
            float disc = b * b - (dot(oc, oc) - u.bgRadius * u.bgRadius);
            float t = -b + sqrt(max(disc, 0.0));
            float3 p = o + dirWorld * t - u.bgCenter.xyz;
            float3 d3 = normalize(p);
            float uu = atan2(d3.z, d3.x) / (2.0 * M_PI_F) + 0.5;
            float vv = acos(clamp(d3.y, -1.0, 1.0)) / M_PI_F;

            float feat[8];
            sampleBgTexture(bgTexture, u.texW, u.texH, u.texChannels, uu, vv, feat);
            for (uint c = 0; c < u.texChannels; c++) texFeatSum[c] += feat[c];
        }
    }

    float3 color = colorSum * invN;
    float depthAvg = depthSum * invN;
    float2 mvAvg = mvSum * invN;
    float fg = fgSum * invN;
    float disocc = disoccSum * invN;

    float depthN = depthAvg / (depthAvg + 1.0);  // normalize_depth, AFTER pooling
    float2 mvN = mvAvg * u.mvScale;

    uint base = (gid.y * u.netW + gid.x) * (18 + u.hiddenChannels);
    out[base + 0] = half(color.r);
    out[base + 1] = half(color.g);
    out[base + 2] = half(color.b);
    out[base + 3] = half(depthN);
    out[base + 4] = half(mvN.x);
    out[base + 5] = half(mvN.y);
    out[base + 6] = half(disocc);
    out[base + 7] = half(fg);
    out[base + 8] = half(u.jitterX);
    out[base + 9] = half(u.jitterY);
    for (uint c = 0; c < u.texChannels; c++) {
        out[base + 10 + c] = half(texFeatSum[c] * invN);
    }
    uint hiddenBase = (gid.y * u.netW + gid.x) * u.hiddenChannels;
    for (uint c = 0; c < u.hiddenChannels; c++) {
        out[base + 10 + u.texChannels + c] = hiddenIn[hiddenBase + c];
    }
}
)";

static MTL::CompileOptions* safeMathCompileOptions() {
    auto* options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(false);
    if (@available(iOS 18.0, *)) {
        options->setMathMode(MTL::MathModeSafe);
    }
    return options;
}

static MTL::ComputePipelineState* buildPipeline(MetalContext& ctx, const char* src, const char* fnName) {
    NS::Error* error = nullptr;
    auto* nsSrc = NS::String::string(src, NS::UTF8StringEncoding);
    auto* opts = safeMathCompileOptions();
    auto* lib = ctx.device->newLibrary(nsSrc, opts, &error);
    opts->release();
    if (!lib) {
        std::string msg = std::string("Failed to compile ") + fnName + " shader";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }
    auto* fn = lib->newFunction(NS::String::string(fnName, NS::UTF8StringEncoding));
    if (!fn) {
        lib->release();
        throw std::runtime_error(std::string(fnName) + " function not found");
    }
    auto* pipeline = ctx.device->newComputePipelineState(fn, &error);
    fn->release();
    lib->release();
    if (!pipeline) {
        std::string msg = std::string("Failed to create ") + fnName + " pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }
    return pipeline;
}

NetInputAssembly::~NetInputAssembly() {
    if (unpremulPipeline_) unpremulPipeline_->release();
    if (assemblePipeline_) assemblePipeline_->release();
    if (bgTextureBuffer_) bgTextureBuffer_->release();
    if (depthPing_[0]) depthPing_[0]->release();
    if (depthPing_[1]) depthPing_[1]->release();
    if (outputBuffer_) outputBuffer_->release();
    if (zeroHiddenBuffer_) zeroHiddenBuffer_->release();
}

void NetInputAssembly::init(MetalContext& ctx, const std::string& textureNpyPath,
                             const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
                             uint32_t paramStride, uint32_t hiddenChannels, uint32_t depthAlphaOffset) {
    ctx_ = &ctx;
    proxyW_ = proxyW;
    proxyH_ = proxyH;
    paramStride_ = paramStride;
    hiddenChannels_ = hiddenChannels;
    depthAlphaOffset_ = depthAlphaOffset;
    // mobiledlss.train.export._PaddedExportModel: replicate-pad proxy h/w up to a
    // multiple of 8*paramStride before box-pooling, so the network always runs at a
    // multiple-of-8 resolution (e.g. 480x270, ps=2 -> pad proxy to 480x272 -> net 240x136).
    auto roundUp = [](uint32_t x, uint32_t m) { return ((x + m - 1) / m) * m; };
    uint32_t multiple = 8 * paramStride;
    uint32_t paddedProxyW = roundUp(proxyW, multiple);
    uint32_t paddedProxyH = roundUp(proxyH, multiple);
    netW_ = paddedProxyW / paramStride;
    netH_ = paddedProxyH / paramStride;

    unpremulPipeline_ = buildPipeline(ctx, kUnpremulDepthMSL, "unpremul_depth");
    assemblePipeline_ = buildPipeline(ctx, kAssembleMSL, "assemble_net_input");

    // --- exported/texture.npy: [C, H, W] float32 (tools/reconstruct_reference_dump.py's
    // convention, read via the already-linked DlssIO::readNpyFloat32). ---
    DlssIO::NpyArray texArr = DlssIO::readNpyFloat32(textureNpyPath);
    if (texArr.shape.size() != 3) throw std::runtime_error("texture.npy: expected 3D [C,H,W]");
    texChannels_ = static_cast<uint32_t>(texArr.shape[0]);
    texH_ = static_cast<uint32_t>(texArr.shape[1]);
    texW_ = static_cast<uint32_t>(texArr.shape[2]);
    bgTextureBuffer_ = ctx.newBuffer(texArr.data.data(), texArr.data.size() * sizeof(float),
                                      MTL::ResourceStorageModeShared);

    DlssIO::NpyArray sphereArr = DlssIO::readNpyFloat32(bgSphereNpyPath);
    if (sphereArr.data.size() != 4) throw std::runtime_error("bg_sphere.npy: expected 4 floats");
    bgSphere_ = glm::vec4(sphereArr.data[0], sphereArr.data[1], sphereArr.data[2], sphereArr.data[3]);

    for (int i = 0; i < 2; i++) {
        auto* desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Float, proxyW, proxyH, false);
        desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        desc->setStorageMode(MTL::StorageModePrivate);
        depthPing_[i] = ctx.newTexture(desc);
    }
    depthPingIndex_ = 0;
    firstFrame_ = true;

    uint32_t channels = getChannels();
    outputBuffer_ = ctx.newBuffer(static_cast<size_t>(netW_) * netH_ * channels * sizeof(uint16_t),
                                   MTL::ResourceStorageModeShared);
    zeroHiddenBuffer_ = ctx.newBuffer(static_cast<size_t>(netW_) * netH_ * hiddenChannels_ * sizeof(uint16_t),
                                       MTL::ResourceStorageModeShared);
    std::memset(zeroHiddenBuffer_->contents(), 0, static_cast<size_t>(netW_) * netH_ * hiddenChannels_ * sizeof(uint16_t));
}

void NetInputAssembly::run(MetalContext& ctx, MTL::Texture* currColorTex, MTL::Texture* currDepthTex,
                            MTL::Texture* currMotionTex, MTL::Buffer* hiddenIn, glm::vec3 eye,
                            glm::vec3 rAxis, glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy,
                            float cx, float cy, float jitterProxyX, float jitterProxyY) {
    MTL::Texture* curDepth = depthPing_[depthPingIndex_];
    MTL::Texture* prevDepth = depthPing_[1 - depthPingIndex_];

    auto* cmdBuf = ctx.beginCommandBuffer();

    // Pass 1: un-premultiply this frame's proxy depth into curDepth.
    {
        auto* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(unpremulPipeline_);
        enc->setTexture(currDepthTex, 0);
        enc->setTexture(curDepth, 1);
        enc->setBytes(&depthAlphaOffset_, sizeof(depthAlphaOffset_), 0);
        MTL::Size grid(proxyW_, proxyH_, 1);
        NS::UInteger tew = unpremulPipeline_->threadExecutionWidth();
        NS::UInteger maxT = unpremulPipeline_->maxTotalThreadsPerThreadgroup();
        NS::UInteger th = maxT / tew;
        if (th == 0) th = 1;
        MTL::Size tg(tew, th, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }

    // Pass 2: assemble the packed NHWC input.
    {
        InputAssemblyUniforms u{};
        u.eye[0] = eye.x; u.eye[1] = eye.y; u.eye[2] = eye.z; u.eye[3] = 0.0f;
        u.rAxis[0] = rAxis.x; u.rAxis[1] = rAxis.y; u.rAxis[2] = rAxis.z; u.rAxis[3] = 0.0f;
        u.uAxis[0] = uAxis.x; u.uAxis[1] = uAxis.y; u.uAxis[2] = uAxis.z; u.uAxis[3] = 0.0f;
        u.fAxis[0] = fAxis.x; u.fAxis[1] = fAxis.y; u.fAxis[2] = fAxis.z; u.fAxis[3] = 0.0f;
        u.bgCenter[0] = bgSphere_.x; u.bgCenter[1] = bgSphere_.y; u.bgCenter[2] = bgSphere_.z; u.bgCenter[3] = 0.0f;
        u.fx = fx; u.fy = fy; u.cx = cx; u.cy = cy;
        u.bgRadius = bgSphere_.w;
        u.mvScale = 1.0f / 16.0f;
        u.jitterX = jitterProxyX;
        u.jitterY = jitterProxyY;
        u.proxyW = proxyW_; u.proxyH = proxyH_; u.netW = netW_; u.netH = netH_;
        u.paramStride = paramStride_; u.hiddenChannels = hiddenChannels_;
        u.texW = texW_; u.texH = texH_;
        u.firstFrame = firstFrame_ ? 1 : 0;
        u.texChannels = texChannels_;

        MTL::Buffer* hidden = hiddenIn ? hiddenIn : zeroHiddenBuffer_;

        auto* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(assemblePipeline_);
        enc->setTexture(currColorTex, 0);
        enc->setTexture(currDepthTex, 1);
        enc->setTexture(currMotionTex, 2);
        enc->setTexture(curDepth, 3);
        enc->setTexture(prevDepth, 4);
        enc->setBuffer(bgTextureBuffer_, 0, 0);
        enc->setBytes(&u, sizeof(u), 1);
        enc->setBuffer(hidden, 0, 2);
        enc->setBuffer(outputBuffer_, 0, 3);
        MTL::Size grid(netW_, netH_, 1);
        NS::UInteger tew = assemblePipeline_->threadExecutionWidth();
        NS::UInteger maxT = assemblePipeline_->maxTotalThreadsPerThreadgroup();
        NS::UInteger th = maxT / tew;
        if (th == 0) th = 1;
        MTL::Size tg(tew, th, 1);
        enc->dispatchThreads(grid, tg);
        enc->endEncoding();
    }

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    wasFirstFrame_ = firstFrame_;
    depthPingIndex_ = 1 - depthPingIndex_;
    firstFrame_ = false;
}
