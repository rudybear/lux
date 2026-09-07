#include "ReconstructPass.h"
#include "metal_context.h"
#include "dlss_io.h"

#include <stdexcept>
#include <cstring>
#include <vector>

// --- Shared derivations (see NetInputAssembly.mm for the full write-up):
// bg-sphere UV bilinear sample: SceneTexture.forward's wrap-U/clamp-V
// grid_sample(padding_mode='border', align_corners=False) convention, texel
// centres at integer pixel coords, continuous coord = uv*size - 0.5.
// Zero-padded bilinear (warp(), reproject_depth()): continuous coord =
// index - raw_mv (the "+0.5/-0.5" of the PyTorch formula cancel exactly),
// Metal coord::pixel sampling with address::clamp_to_zero at (c+0.5).
static const char* kHiddenWarpMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct HiddenUniforms {
    uint netW, netH, proxyW, proxyH, targetW, targetH, paramStrideTimesS, hiddenChannels;
};

kernel void warp_downsample_hidden(
    texture2d<float, access::read> proxyMotion [[texture(0)]],   // RGBA32Float: mv.x*a, mv.y*a, depth*a, a (lux 6ed0334's packed out_aux; only .xy/.w read here)
    texture2d<float, access::sample> prevHiddenR [[texture(1)]],   // unused placeholder (keep binding count stable)
    device const half* prevHidden [[buffer(0)]],   // NHWC, targetW*targetH*hiddenChannels
    device half* hiddenOut [[buffer(1)]],          // NHWC, netW*netH*hiddenChannels
    constant HiddenUniforms& u [[buffer(2)]],
    constant uint& firstFrame [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.netW || gid.y >= u.netH) return;
    uint sp = u.paramStrideTimesS;
    float acc[16];
    for (uint c = 0; c < u.hiddenChannels; c++) acc[c] = 0.0;

    if (firstFrame == 0) {
        for (uint dy = 0; dy < sp; dy++) {
            for (uint dx = 0; dx < sp; dx++) {
                uint X = gid.x * sp + dx;
                uint Y = gid.y * sp + dy;
                if (X >= u.targetW || Y >= u.targetH) continue;
                // mv is at proxy resolution; target/proxy ratio is `s` (upscale), not `sp`
                // (sp = s*paramStride). X*proxyW/targetW addresses the right proxy texel.
                uint pmx = min(X * u.proxyW / u.targetW, u.proxyW - 1);
                uint pmy = min(Y * u.proxyH / u.targetH, u.proxyH - 1);
                float4 mraw = proxyMotion.read(uint2(pmx, pmy));
                float ma = mraw.a;
                float2 mv = (ma > 1e-6) ? (mraw.xy / ma) : float2(0.0);
                float s = float(u.targetW) / float(u.proxyW);
                mv *= s;

                float cx = float(X) - mv.x;
                float cy = float(Y) - mv.y;
                bool inside = cx >= -0.5 && cx < float(u.targetW) - 0.5 && cy >= -0.5 && cy < float(u.targetH) - 0.5;
                if (inside) {
                    // Manual zero-padded bilinear over the flat NHWC buffer (no hardware
                    // texture available for an arbitrary-channel-count buffer).
                    float px = cx, py = cy;
                    int x0 = int(floor(px)), y0 = int(floor(py));
                    float fx = px - float(x0), fy = py - float(y0);
                    for (uint c = 0; c < u.hiddenChannels; c++) {
                        float v00 = 0, v10 = 0, v01 = 0, v11 = 0;
                        if (x0 >= 0 && x0 < int(u.targetW) && y0 >= 0 && y0 < int(u.targetH))
                            v00 = float(prevHidden[(uint(y0) * u.targetW + uint(x0)) * u.hiddenChannels + c]);
                        if (x0 + 1 >= 0 && x0 + 1 < int(u.targetW) && y0 >= 0 && y0 < int(u.targetH))
                            v10 = float(prevHidden[(uint(y0) * u.targetW + uint(x0 + 1)) * u.hiddenChannels + c]);
                        if (x0 >= 0 && x0 < int(u.targetW) && y0 + 1 >= 0 && y0 + 1 < int(u.targetH))
                            v01 = float(prevHidden[(uint(y0 + 1) * u.targetW + uint(x0)) * u.hiddenChannels + c]);
                        if (x0 + 1 >= 0 && x0 + 1 < int(u.targetW) && y0 + 1 >= 0 && y0 + 1 < int(u.targetH))
                            v11 = float(prevHidden[(uint(y0 + 1) * u.targetW + uint(x0 + 1)) * u.hiddenChannels + c]);
                        float v = mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
                        acc[c] += v;
                    }
                }
            }
        }
    }
    float invN = 1.0 / float(sp * sp);
    uint base = (gid.y * u.netW + gid.x) * u.hiddenChannels;
    for (uint c = 0; c < u.hiddenChannels; c++) hiddenOut[base + c] = half(acc[c] * invN);
}
)";

static const char* kReconstructMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct ReconUniforms {
    float4 eye, rAxis, uAxis, fAxis, bgCenter;
    float fx, fy, cx, cy, bgRadius;
    float jitterX, jitterY;
    uint targetW, targetH, proxyW, proxyH, netW, netH;
    uint k, hiddenChannels, nBlend, s, sp, texW, texH, texChannels, memHidden, firstFrame;
};

static void sampleBgTex(device const float* tex, uint texW, uint texH, uint texC,
                         float u, float v, thread float* outFeat) {
    float px = u * float(texW) - 0.5;
    float py = v * float(texH) - 0.5;
    int x0 = int(floor(px)); int y0 = int(floor(py));
    float fx = px - float(x0); float fy = py - float(y0);
    int x0w = ((x0 % int(texW)) + int(texW)) % int(texW);
    int x1w = ((x0 + 1) % int(texW) + int(texW)) % int(texW);
    int y0c = clamp(y0, 0, int(texH) - 1);
    int y1c = clamp(y0 + 1, 0, int(texH) - 1);
    for (uint c = 0; c < texC; c++) {
        device const float* plane = tex + c * texH * texW;
        float t00 = plane[y0c * texW + x0w], t10 = plane[y0c * texW + x1w];
        float t01 = plane[y1c * texW + x0w], t11 = plane[y1c * texW + x1w];
        outFeat[c] = mix(mix(t00, t10, fx), mix(t01, t11, fx), fy);
    }
}

kernel void reconstruct_frame(
    texture2d<float, access::read> proxyColor [[texture(0)]],    // RGBA16Float premul
    texture2d<float, access::read> proxyMotion [[texture(1)]],   // RGBA32Float: mv.x*a, mv.y*a, depth*a, a (out_aux; only .xy/.w read here)
    texture2d<float, access::read> curDepth [[texture(2)]],      // R32Float, unpremul (proxy)
    texture2d<float, access::sample> prevDepth [[texture(3)]],   // R32Float, unpremul (proxy)
    texture2d<float, access::sample> prevColor [[texture(4)]],   // RGBA16Float, target res
    texture2d<float, access::write> outColor [[texture(5)]],     // target res
    device const float* bgTexture [[buffer(0)]],
    device const half* unetOut [[buffer(1)]],       // NHWC, netW*netH*(sp*sp*K*K + nBlend*sp*sp + hidden)
    device const float* fc1w [[buffer(2)]], device const float* fc1b [[buffer(3)]],
    device const float* fc2w [[buffer(4)]], device const float* fc2b [[buffer(5)]],
    device half* hiddenNewOut [[buffer(6)]],        // NHWC, targetW*targetH*hiddenChannels
    constant ReconUniforms& u [[buffer(7)]],
    device half* blendDebugOut [[buffer(8)]],       // half2 (wS, wM) per pixel -- rollout-PSNR diagnostics
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= u.targetW || gid.y >= u.targetH) return;
    uint X = gid.x, Y = gid.y;
    uint netX = min(X / u.sp, u.netW - 1);
    uint netY = min(Y / u.sp, u.netH - 1);
    uint subX = X % u.sp, subY = Y % u.sp;
    uint outCh = u.sp * u.sp * u.k * u.k + u.nBlend * u.sp * u.sp + u.hiddenChannels;
    device const half* netPixel = unetOut + (uint(netY) * u.netW + netX) * outCh;

    // --- pixel_shuffle_params: kernel softmax ---
    uint KK = u.k * u.k;
    float kernelLogits[16];
    float kmax = -1e30;
    for (uint kk = 0; kk < KK; kk++) {
        float v = float(netPixel[kk * u.sp * u.sp + subY * u.sp + subX]);
        kernelLogits[kk] = v;
        kmax = max(kmax, v);
    }
    float ksum = 0.0;
    float kernelW[16];
    for (uint kk = 0; kk < KK; kk++) { kernelW[kk] = exp(kernelLogits[kk] - kmax); ksum += kernelW[kk]; }
    for (uint kk = 0; kk < KK; kk++) kernelW[kk] /= ksum;

    // --- blend logits (n_blend=3 -> softmax; n_blend=1 -> sigmoid) ---
    float blendW[3] = {1.0, 0.0, 0.0};
    uint blendBase = KK * u.sp * u.sp;
    if (u.nBlend == 1) {
        float v = float(netPixel[blendBase + subY * u.sp + subX]);
        blendW[0] = 1.0 / (1.0 + exp(-v));
    } else {
        float bmax = -1e30, bl[3];
        for (uint bb = 0; bb < u.nBlend; bb++) {
            bl[bb] = float(netPixel[blendBase + bb * u.sp * u.sp + subY * u.sp + subX]);
            bmax = max(bmax, bl[bb]);
        }
        float bsum = 0.0;
        for (uint bb = 0; bb < u.nBlend; bb++) { blendW[bb] = exp(bl[bb] - bmax); bsum += blendW[bb]; }
        for (uint bb = 0; bb < u.nBlend; bb++) blendW[bb] /= bsum;
    }

    // --- hidden_raw: broadcast (same value for the whole sp x sp block) ---
    uint hiddenBase = blendBase + u.nBlend * u.sp * u.sp;
    uint hbaseOut = (Y * u.targetW + X) * u.hiddenChannels;
    for (uint c = 0; c < u.hiddenChannels; c++) hiddenNewOut[hbaseOut + c] = netPixel[hiddenBase + c];

    // --- apply_kernel: jitter-aware K-tap gather from proxy colour ---
    float cxp = (float(X) + 0.5 + u.jitterX) / float(u.s) - 0.5;
    float cyp = (float(Y) + 0.5 + u.jitterY) / float(u.s) - 0.5;
    int anchorX = int(round(cxp)), anchorY = int(round(cyp));
    int off0 = -(int(u.k) / 2 - 1);  // K=4 -> -1
    float3 spatial = float3(0.0);
    for (uint dyi = 0; dyi < u.k; dyi++) {
        int tapY = clamp(anchorY + off0 + int(dyi), 0, int(u.proxyH) - 1);
        for (uint dxi = 0; dxi < u.k; dxi++) {
            int tapX = clamp(anchorX + off0 + int(dxi), 0, int(u.proxyW) - 1);
            float4 raw = proxyColor.read(uint2(uint(tapX), uint(tapY)));
            float a = raw.a;
            float3 rgb = (a > 1e-6) ? raw.rgb / a : float3(0.0);
            uint tapIdx = dyi * u.k + dxi;
            spatial += rgb * kernelW[tapIdx];
        }
    }

    // --- warp: backward-warp previous frame's output by this frame's (scaled) proxy MV ---
    uint pmx = min(X * u.proxyW / u.targetW, u.proxyW - 1);
    uint pmy = min(Y * u.proxyH / u.targetH, u.proxyH - 1);
    float4 mraw = proxyMotion.read(uint2(pmx, pmy));
    float ma = mraw.a;
    float2 mv = (ma > 1e-6) ? (mraw.xy / ma) : float2(0.0);
    mv *= float(u.s);
    float wcx = float(X) - mv.x, wcy = float(Y) - mv.y;
    constexpr sampler zeroPad(coord::pixel, filter::linear, address::clamp_to_zero);
    float3 warped = u.firstFrame != 0 ? float3(0.0) : prevColor.sample(zeroPad, float2(wcx + 0.5, wcy + 0.5)).rgb;

    // --- disocc_target: proxy-res disocclusion (motion.py::disocclusion_mask), nearest-upsampled ---
    float disocc = 1.0;
    if (u.firstFrame == 0) {
        float depthVal = curDepth.read(uint2(pmx, pmy)).x;
        float pxCode = float(pmx) + 0.5 - mv.x / float(u.s);
        float pyCode = float(pmy) + 0.5 - mv.y / float(u.s);
        bool inside = pxCode >= 0.0 && pxCode <= float(u.proxyW) && pyCode >= 0.0 && pyCode <= float(u.proxyH);
        if (inside) {
            float ccx = float(pmx) - mv.x / float(u.s), ccy = float(pmy) - mv.y / float(u.s);
            float prevD = prevDepth.sample(zeroPad, float2(ccx + 0.5, ccy + 0.5)).x;
            disocc = (prevD < depthVal * (1.0 - 0.05)) ? 1.0 : 0.0;
        } else {
            disocc = 1.0;
        }
    }

    // --- bg-sphere UV (target res) + texture sample + MemoryColorHead decode ---
    float3 dirCam = float3((float(X) + 0.5 - u.cx) / u.fx, (float(Y) + 0.5 - u.cy) / u.fy, 1.0);
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
    sampleBgTex(bgTexture, u.texW, u.texH, u.texChannels, uu, vv, feat);

    float hid[16];
    for (uint h = 0; h < u.memHidden; h++) {
        float acc = fc1b[h];
        for (uint c = 0; c < u.texChannels; c++) acc += fc1w[h * u.texChannels + c] * feat[c];
        hid[h] = acc > 0.0 ? acc : acc * 0.1;  // LeakyReLU(0.1)
    }
    float3 memory;
    for (uint o3 = 0; o3 < 3; o3++) {
        float acc = fc2b[o3];
        for (uint h = 0; h < u.memHidden; h++) acc += fc2w[o3 * u.memHidden + h] * hid[h];
        memory[o3] = 1.0 / (1.0 + exp(-acc));  // sigmoid
    }

    // --- blend3 with disocclusion-gated renormalised weights ---
    float wS = blendW[0], wH = blendW[1], wM = u.nBlend == 1 ? 0.0 : blendW[2];
    if (disocc > 0.5) wH = 0.0;
    float total = max(wS + wH + wM, 1e-8);
    wS /= total; wH /= total; wM /= total;
    float3 outRgb = wS * spatial + wH * warped + wM * memory;

    outColor.write(float4(outRgb, 1.0), gid);
    uint dbgBase = (Y * u.targetW + X) * 2;
    blendDebugOut[dbgBase + 0] = half(wS);
    blendDebugOut[dbgBase + 1] = half(wM);
}
)";

static MTL::ComputePipelineState* buildPipeline(MetalContext& ctx, const char* src, const char* fnName) {
    NS::Error* error = nullptr;
    auto* nsSrc = NS::String::string(src, NS::UTF8StringEncoding);
    auto* opts = MTL::CompileOptions::alloc()->init();
    opts->setFastMathEnabled(false);
    auto* lib = ctx.device->newLibrary(nsSrc, opts, &error);
    opts->release();
    if (!lib) {
        std::string msg = std::string("ReconstructPass: failed to compile ") + fnName;
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }
    auto* fn = lib->newFunction(NS::String::string(fnName, NS::UTF8StringEncoding));
    if (!fn) { lib->release(); throw std::runtime_error(std::string(fnName) + " not found"); }
    auto* pipeline = ctx.device->newComputePipelineState(fn, &error);
    fn->release();
    lib->release();
    if (!pipeline) throw std::runtime_error(std::string("failed to create pipeline for ") + fnName);
    return pipeline;
}

ReconstructPass::~ReconstructPass() {
    if (pipeline_) pipeline_->release();
    if (hiddenPipeline_) hiddenPipeline_->release();
    if (bgTextureBuffer_) bgTextureBuffer_->release();
    if (blendDebugBuffer_) blendDebugBuffer_->release();
    if (fc1w_) fc1w_->release();
    if (fc1b_) fc1b_->release();
    if (fc2w_) fc2w_->release();
    if (fc2b_) fc2b_->release();
    for (int i = 0; i < 2; i++) {
        if (prevColor_[i]) prevColor_[i]->release();
        if (prevHidden_[i]) prevHidden_[i]->release();
    }
    if (hiddenInputBuffer_) hiddenInputBuffer_->release();
}

void ReconstructPass::init(MetalContext& ctx, const std::string& textureNpyPath,
                            const std::string& bgSphereNpyPath, const std::string& memoryHeadNpzPath,
                            uint32_t targetW, uint32_t targetH, uint32_t proxyW, uint32_t proxyH, uint32_t netW,
                            uint32_t netH, uint32_t k, uint32_t hiddenChannels, uint32_t nBlend, uint32_t s,
                            uint32_t paramStride) {
    targetW_ = targetW; targetH_ = targetH; proxyW_ = proxyW; proxyH_ = proxyH; netW_ = netW; netH_ = netH;
    k_ = k; hiddenChannels_ = hiddenChannels; nBlend_ = nBlend; s_ = s; paramStride_ = paramStride;

    pipeline_ = buildPipeline(ctx, kReconstructMSL, "reconstruct_frame");
    hiddenPipeline_ = buildPipeline(ctx, kHiddenWarpMSL, "warp_downsample_hidden");

    DlssIO::NpyArray texArr = DlssIO::readNpyFloat32(textureNpyPath);
    texChannels_ = static_cast<uint32_t>(texArr.shape[0]);
    texH_ = static_cast<uint32_t>(texArr.shape[1]);
    texW_ = static_cast<uint32_t>(texArr.shape[2]);
    bgTextureBuffer_ = ctx.newBuffer(texArr.data.data(), texArr.data.size() * sizeof(float), MTL::ResourceStorageModeShared);

    DlssIO::NpyArray sphereArr = DlssIO::readNpyFloat32(bgSphereNpyPath);
    bgSphere_ = glm::vec4(sphereArr.data[0], sphereArr.data[1], sphereArr.data[2], sphereArr.data[3]);

    DlssIO::NpyArray w1 = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc1_w");
    DlssIO::NpyArray b1 = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc1_b");
    DlssIO::NpyArray w2 = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc2_w");
    DlssIO::NpyArray b2 = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc2_b");
    memHidden_ = static_cast<uint32_t>(b1.data.size());
    fc1w_ = ctx.newBuffer(w1.data.data(), w1.data.size() * sizeof(float), MTL::ResourceStorageModeShared);
    fc1b_ = ctx.newBuffer(b1.data.data(), b1.data.size() * sizeof(float), MTL::ResourceStorageModeShared);
    fc2w_ = ctx.newBuffer(w2.data.data(), w2.data.size() * sizeof(float), MTL::ResourceStorageModeShared);
    fc2b_ = ctx.newBuffer(b2.data.data(), b2.data.size() * sizeof(float), MTL::ResourceStorageModeShared);

    for (int i = 0; i < 2; i++) {
        auto* desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, targetW, targetH, false);
        desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        desc->setStorageMode(MTL::StorageModePrivate);
        prevColor_[i] = ctx.newTexture(desc);
        prevHidden_[i] = ctx.newBuffer(static_cast<size_t>(targetW) * targetH * hiddenChannels * sizeof(uint16_t),
                                        MTL::ResourceStorageModeShared);
        std::memset(prevHidden_[i]->contents(), 0, static_cast<size_t>(targetW) * targetH * hiddenChannels * sizeof(uint16_t));
    }
    hiddenInputBuffer_ = ctx.newBuffer(static_cast<size_t>(netW) * netH * hiddenChannels * sizeof(uint16_t),
                                        MTL::ResourceStorageModeShared);
    blendDebugBuffer_ = ctx.newBuffer(static_cast<size_t>(targetW) * targetH * 2 * sizeof(uint16_t),
                                       MTL::ResourceStorageModeShared);
    pingIndex_ = 0;
    firstRun_ = true;
}

void ReconstructPass::prepareHiddenInput(MetalContext& ctx, MTL::Texture* proxyMotionTex, bool firstFrame,
                                          MTL::Buffer* hiddenInOut) {
    struct HiddenUniforms {
        uint32_t netW, netH, proxyW, proxyH, targetW, targetH, spTimesOne, hiddenChannels;
    } u{netW_, netH_, proxyW_, proxyH_, targetW_, targetH_, s_ * paramStride_, hiddenChannels_};
    uint32_t first = firstFrame ? 1 : 0;

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(hiddenPipeline_);
    enc->setTexture(proxyMotionTex, 0);
    enc->setTexture(proxyMotionTex, 1);  // unused placeholder binding
    enc->setBuffer(prevHidden_[pingIndex_], 0, 0);
    enc->setBuffer(hiddenInOut, 0, 1);
    enc->setBytes(&u, sizeof(u), 2);
    enc->setBytes(&first, sizeof(first), 3);
    MTL::Size grid(netW_, netH_, 1);
    NS::UInteger tew = hiddenPipeline_->threadExecutionWidth();
    NS::UInteger th = hiddenPipeline_->maxTotalThreadsPerThreadgroup() / tew;
    if (th == 0) th = 1;
    enc->dispatchThreads(grid, MTL::Size(tew, th, 1));
    enc->endEncoding();
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

void ReconstructPass::run(MetalContext& ctx, MTL::Texture* proxyColorTex, MTL::Texture* proxyMotionTex,
                           MTL::Texture* curDepthTex, MTL::Texture* prevDepthTex, bool firstFrame,
                           MTL::Buffer* unetOutputBuffer, MTL::Texture* outColorTex, glm::vec3 eye,
                           glm::vec3 rAxis, glm::vec3 uAxis, glm::vec3 fAxis, float fx, float fy, float cx,
                           float cy, float jitterTargetX, float jitterTargetY) {
    struct ReconUniforms {
        float eye[4], rAxis[4], uAxis[4], fAxis[4], bgCenter[4];
        float fx, fy, cx, cy, bgRadius;
        float jitterX, jitterY;
        uint32_t targetW, targetH, proxyW, proxyH, netW, netH;
        uint32_t k, hiddenChannels, nBlend, s, sp, texW, texH, texChannels, memHidden, firstFrame;
    } u{};
    u.eye[0] = eye.x; u.eye[1] = eye.y; u.eye[2] = eye.z;
    u.rAxis[0] = rAxis.x; u.rAxis[1] = rAxis.y; u.rAxis[2] = rAxis.z;
    u.uAxis[0] = uAxis.x; u.uAxis[1] = uAxis.y; u.uAxis[2] = uAxis.z;
    u.fAxis[0] = fAxis.x; u.fAxis[1] = fAxis.y; u.fAxis[2] = fAxis.z;
    u.bgCenter[0] = bgSphere_.x; u.bgCenter[1] = bgSphere_.y; u.bgCenter[2] = bgSphere_.z;
    u.fx = fx; u.fy = fy; u.cx = cx; u.cy = cy; u.bgRadius = bgSphere_.w;
    u.jitterX = jitterTargetX; u.jitterY = jitterTargetY;
    u.targetW = targetW_; u.targetH = targetH_; u.proxyW = proxyW_; u.proxyH = proxyH_;
    u.netW = netW_; u.netH = netH_;
    u.k = k_; u.hiddenChannels = hiddenChannels_; u.nBlend = nBlend_; u.s = s_; u.sp = s_ * paramStride_;
    u.texW = texW_; u.texH = texH_; u.texChannels = texChannels_; u.memHidden = memHidden_;
    u.firstFrame = firstFrame ? 1 : 0;

    int nextIndex = 1 - pingIndex_;

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(pipeline_);
    enc->setTexture(proxyColorTex, 0);
    enc->setTexture(proxyMotionTex, 1);
    enc->setTexture(curDepthTex, 2);
    enc->setTexture(prevDepthTex, 3);
    enc->setTexture(prevColor_[pingIndex_], 4);
    enc->setTexture(outColorTex, 5);
    enc->setBuffer(bgTextureBuffer_, 0, 0);
    enc->setBuffer(unetOutputBuffer, 0, 1);
    enc->setBuffer(fc1w_, 0, 2);
    enc->setBuffer(fc1b_, 0, 3);
    enc->setBuffer(fc2w_, 0, 4);
    enc->setBuffer(fc2b_, 0, 5);
    enc->setBuffer(prevHidden_[nextIndex], 0, 6);
    enc->setBytes(&u, sizeof(u), 7);
    enc->setBuffer(blendDebugBuffer_, 0, 8);
    MTL::Size grid(targetW_, targetH_, 1);
    NS::UInteger tew = pipeline_->threadExecutionWidth();
    NS::UInteger th = pipeline_->maxTotalThreadsPerThreadgroup() / tew;
    if (th == 0) th = 1;
    enc->dispatchThreads(grid, MTL::Size(tew, th, 1));
    enc->endEncoding();

    // Also copy outColorTex -> prevColor_[nextIndex] (this frame's own output,
    // for next frame's warp) via a blit.
    auto* blit = cmdBuf->blitCommandEncoder();
    blit->copyFromTexture(outColorTex, 0, 0, MTL::Origin(0, 0, 0), MTL::Size(targetW_, targetH_, 1),
                           prevColor_[nextIndex], 0, 0, MTL::Origin(0, 0, 0));
    blit->endEncoding();

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    pingIndex_ = nextIndex;
    firstRun_ = false;
}
