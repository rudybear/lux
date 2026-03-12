#include "metal_splat_renderer.h"
#include "gltf_loader.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>

// --------------------------------------------------------------------------
// Embedded MSL: Gaussian splat projection compute kernel
// --------------------------------------------------------------------------

static const char* kSplatComputeMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct ComputeUniforms {
    float4x4 view;
    float4x4 proj;
    packed_float3 camPos;
    float _pad0;
    float screenW;
    float screenH;
    uint numSplats;
    float focalX;
    float focalY;
    int shDegree;
    float _pad1[2];
};

// Build rotation matrix from quaternion (xyzw)
float3x3 quatToMat(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;

    return float3x3(
        float3(1.0 - (yy + zz), xy + wz, xz - wy),
        float3(xy - wz, 1.0 - (xx + zz), yz + wx),
        float3(xz + wy, yz - wx, 1.0 - (xx + yy))
    );
}

// SH degree 0: base color from DC coefficient
float3 shColor(float3 sh0, float3 dir) {
    // SH DC: C0 = 0.28209479 (Y_0^0)
    float C0 = 0.28209479;
    return max(float3(0.0), float3(0.5) + C0 * sh0);
}

// Compute 2D covariance from 3D covariance
float3 computeCov2D(float3 mean, float3x3 cov3D,
                    float4x4 viewMat, float focalX, float focalY) {
    // Transform mean to camera space
    float4 t4 = viewMat * float4(mean, 1.0);
    float3 t = t4.xyz;

    // Use positive depth (GLM right-handed: objects in front have negative z)
    float tz = -t.z;
    if (tz < 0.001) tz = 0.001;
    float tz2 = tz * tz;

    // Clamp tx/ty to avoid extreme off-screen projections
    float limx = 1.3 * focalX / tz;
    float limy = 1.3 * focalY / tz;
    float txtz = min(limx, max(-limx, t.x / tz));
    float tytz = min(limy, max(-limy, t.y / tz));

    // Jacobian of projection with positive depth (matches Lux stdlib convention)
    float3x3 J = float3x3(
        float3(focalX / tz, 0.0, -(focalX * txtz) / tz),
        float3(0.0, focalY / tz, -(focalY * tytz) / tz),
        float3(0.0, 0.0, 0.0)
    );

    // W = view matrix upper-left 3x3 (rows stored as columns, same convention as J)
    float3x3 W = float3x3(
        float3(viewMat[0][0], viewMat[1][0], viewMat[2][0]),
        float3(viewMat[0][1], viewMat[1][1], viewMat[2][1]),
        float3(viewMat[0][2], viewMat[1][2], viewMat[2][2])
    );

    float3x3 T = J * W;
    float3x3 cov = T * cov3D * transpose(T);

    float cov00 = cov[0][0];
    float cov01 = cov[0][1];
    float cov11 = cov[1][1];

    // Return upper-triangle of 2D covariance + low-pass filter
    return float3(cov00 + 0.3, cov01, cov11 + 0.3);
}

kernel void splat_project(
    device const float4* positions [[buffer(0)]],
    device const float4* rotations [[buffer(1)]],
    device const float4* scales    [[buffer(2)]],
    device const float*  opacities [[buffer(3)]],
    device const float4* sh0       [[buffer(4)]],
    device float4* projCenters     [[buffer(5)]],
    device float4* projConics      [[buffer(6)]],
    device float4* projColors      [[buffer(7)]],
    constant ComputeUniforms& u    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= u.numSplats) return;

    float3 pos = positions[gid].xyz;
    float4 rot = rotations[gid];
    float3 scl = scales[gid].xyz;
    float opacity = opacities[gid];

    // Activate opacity from logit space: sigmoid
    float alpha = 1.0 / (1.0 + exp(-opacity));

    // Activate scale from log space
    float3 s = exp(scl);

    // Build 3D covariance: R * S * S^T * R^T
    float3x3 R = quatToMat(rot);
    float3x3 S = float3x3(
        float3(s.x, 0.0, 0.0),
        float3(0.0, s.y, 0.0),
        float3(0.0, 0.0, s.z)
    );
    float3x3 M = R * S;
    float3x3 cov3D = M * transpose(M);

    // Project to screen
    float4 p_hom = u.proj * u.view * float4(pos, 1.0);
    float p_w = max(p_hom.w, 0.0001);
    float3 p_ndc = p_hom.xyz / p_w;

    // NDC to screen pixels
    // Metal NDC: X [-1,1] left-to-right, Y [-1,1] bottom-to-top
    // Screen: X [0,W] left-to-right, Y [0,H] top-to-bottom
    float2 center = float2(
        (p_ndc.x * 0.5 + 0.5) * u.screenW,
        (1.0 - (p_ndc.y * 0.5 + 0.5)) * u.screenH
    );

    // Frustum culling (depth range [0,1] for Metal with GLM_FORCE_DEPTH_ZERO_TO_ONE)
    if (p_ndc.z < 0.0 || p_ndc.z > 1.0 ||
        center.x < -200.0 || center.x > u.screenW + 200.0 ||
        center.y < -200.0 || center.y > u.screenH + 200.0) {
        projCenters[gid] = float4(0.0, 0.0, -1e6, 0.0);
        projConics[gid] = float4(0.0);
        projColors[gid] = float4(0.0);
        return;
    }

    // Compute 2D covariance
    float3 cov2D = computeCov2D(pos, cov3D, u.view, u.focalX, u.focalY);

    // Invert 2D covariance to get conic
    float det = cov2D.x * cov2D.z - cov2D.y * cov2D.y;
    if (det <= 0.0) {
        projCenters[gid] = float4(0.0, 0.0, -1e6, 0.0);
        projConics[gid] = float4(0.0);
        projColors[gid] = float4(0.0);
        return;
    }
    float inv_det = 1.0 / det;
    float3 conic = float3(cov2D.z * inv_det, -cov2D.y * inv_det, cov2D.x * inv_det);

    // Compute radius for the quad (3 sigma)
    float mid = 0.5 * (cov2D.x + cov2D.z);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float radius = ceil(3.0 * sqrt(lambda1));

    // Evaluate SH color (degree 0)
    float3 viewDir = normalize(pos - float3(u.camPos));
    float3 color = (u.shDegree >= 0)
        ? shColor(sh0[gid].xyz, viewDir)
        : float3(0.5);

    // Store results
    projCenters[gid] = float4(center, p_ndc.z, radius);
    projConics[gid] = float4(conic, alpha);
    projColors[gid] = float4(color, alpha);
}
)";

// --------------------------------------------------------------------------
// Embedded MSL: Gaussian splat vertex + fragment shaders
// --------------------------------------------------------------------------

static const char* kSplatRenderMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct RenderUniforms {
    float screenW;
    float screenH;
    uint visibleCount;
    float alphaCutoff;
};

struct VertexOut {
    float4 position [[position]];
    float3 conic;
    float4 color;
    float2 center;
    float2 offset;
};

vertex VertexOut splat_vertex(
    uint vid [[vertex_id]],
    uint iid [[instance_id]],
    device const float4* projCenters [[buffer(0)]],
    device const float4* projConics  [[buffer(1)]],
    device const float4* projColors  [[buffer(2)]],
    device const uint*   sortedIdx   [[buffer(3)]],
    constant RenderUniforms& u       [[buffer(4)]])
{
    VertexOut out;

    uint idx = sortedIdx[iid];
    float4 centerData = projCenters[idx];
    float4 conicData = projConics[idx];
    float4 colorData = projColors[idx];

    float2 center = centerData.xy;
    float radius = centerData.w;

    // Quad offsets: 2 triangles forming a [-1,1] quad
    float2 quadPos[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)
    };
    float2 offset = quadPos[vid] * radius;

    // Screen-space to NDC
    // Screen: X [0,W] left-to-right, Y [0,H] top-to-bottom
    // Metal NDC: X [-1,1] left-to-right, Y [-1,1] bottom-to-top
    float2 screenPos = center + offset;
    float2 ndc = float2(
        screenPos.x / u.screenW * 2.0 - 1.0,
        1.0 - screenPos.y / u.screenH * 2.0
    );

    out.position = float4(ndc, centerData.z, 1.0);
    out.conic = conicData.xyz;
    out.color = colorData;
    out.center = center;
    out.offset = offset;

    return out;
}

fragment float4 splat_fragment(
    VertexOut in [[stage_in]],
    constant RenderUniforms& u [[buffer(4)]])
{
    // Evaluate gaussian: exp(-0.5 * (offset^T * conic * offset))
    float2 d = in.offset;
    float power = -0.5 * (in.conic.x * d.x * d.x +
                           2.0 * in.conic.y * d.x * d.y +
                           in.conic.z * d.y * d.y);

    if (power > 0.0) discard_fragment();

    float alpha = min(0.99, in.color.a * exp(power));
    if (alpha < u.alphaCutoff) discard_fragment();

    // Premultiplied alpha output
    return float4(in.color.rgb * alpha, alpha);
}
)";

// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

MetalSplatRenderer::~MetalSplatRenderer() {
    // Resources should be cleaned up via cleanup() before destruction
}

// --------------------------------------------------------------------------
// Render target creation
// --------------------------------------------------------------------------

void MetalSplatRenderer::createRenderTargets(MetalContext& ctx) {
    auto* colorDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, width_, height_, false);
    colorDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    colorDesc->setStorageMode(MTL::StorageModePrivate);
    colorTarget_ = ctx.newTexture(colorDesc);
    colorTarget_->setLabel(NS::String::string("SplatColor", NS::UTF8StringEncoding));

    auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, width_, height_, false);
    depthDesc->setUsage(MTL::TextureUsageRenderTarget);
    depthDesc->setStorageMode(MTL::StorageModePrivate);
    depthTarget_ = ctx.newTexture(depthDesc);
    depthTarget_->setLabel(NS::String::string("SplatDepth", NS::UTF8StringEncoding));
}

// --------------------------------------------------------------------------
// Pipeline creation (MSL from embedded strings)
// --------------------------------------------------------------------------

void MetalSplatRenderer::createPipelines(MetalContext& ctx) {
    NS::Error* error = nullptr;

    // --- Compute pipeline ---
    auto* computeSrc = NS::String::string(kSplatComputeMSL, NS::UTF8StringEncoding);
    auto* computeLib = ctx.device->newLibrary(computeSrc, nullptr, &error);
    if (!computeLib) {
        std::string msg = "Failed to compile splat compute shader";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    auto* computeFunc = computeLib->newFunction(
        NS::String::string("splat_project", NS::UTF8StringEncoding));
    if (!computeFunc) {
        computeLib->release();
        throw std::runtime_error("splat_project function not found in compute library");
    }

    computePipeline_ = ctx.device->newComputePipelineState(computeFunc, &error);
    computeFunc->release();
    computeLib->release();

    if (!computePipeline_) {
        std::string msg = "Failed to create splat compute pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    // --- Render pipeline ---
    auto* renderSrc = NS::String::string(kSplatRenderMSL, NS::UTF8StringEncoding);
    auto* renderLib = ctx.device->newLibrary(renderSrc, nullptr, &error);
    if (!renderLib) {
        std::string msg = "Failed to compile splat render shader";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    auto* vertFunc = renderLib->newFunction(
        NS::String::string("splat_vertex", NS::UTF8StringEncoding));
    auto* fragFunc = renderLib->newFunction(
        NS::String::string("splat_fragment", NS::UTF8StringEncoding));
    if (!vertFunc || !fragFunc) {
        if (vertFunc) vertFunc->release();
        if (fragFunc) fragFunc->release();
        renderLib->release();
        throw std::runtime_error("splat_vertex or splat_fragment function not found");
    }

    auto* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);

    // Color attachment with premultiplied alpha blending
    auto* colorAtt = pipeDesc->colorAttachments()->object(0);
    colorAtt->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    colorAtt->setBlendingEnabled(true);
    colorAtt->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    colorAtt->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAtt->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAtt->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    colorAtt->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAtt->setAlphaBlendOperation(MTL::BlendOperationAdd);

    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    renderPipeline_ = ctx.device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();
    vertFunc->release();
    fragFunc->release();
    renderLib->release();

    if (!renderPipeline_) {
        std::string msg = "Failed to create splat render pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    // --- Depth stencil state (depth test on, write off for transparency) ---
    auto* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    dsDesc->setDepthWriteEnabled(false);
    depthStencilState_ = ctx.device->newDepthStencilState(dsDesc);
    dsDesc->release();
}

// --------------------------------------------------------------------------
// Buffer creation and data upload
// --------------------------------------------------------------------------

void MetalSplatRenderer::createBuffers(MetalContext& ctx, const GaussianSplatData& data) {
    numSplats_ = data.num_splats;
    shDegree_ = data.sh_degree;
    if (numSplats_ == 0) return;

    // Input buffers (shared memory for CPU upload)

    // Positions: already vec4 (x,y,z,1)
    hostPositions_ = data.positions;
    posBuffer_ = ctx.newBuffer(hostPositions_.data(),
                               hostPositions_.size() * sizeof(float),
                               MTL::ResourceStorageModeShared);
    posBuffer_->setLabel(NS::String::string("SplatPositions", NS::UTF8StringEncoding));

    // Rotations (vec4 xyzw)
    rotBuffer_ = ctx.newBuffer(data.rotations.data(),
                               numSplats_ * 4 * sizeof(float),
                               MTL::ResourceStorageModeShared);
    rotBuffer_->setLabel(NS::String::string("SplatRotations", NS::UTF8StringEncoding));

    // Scales as vec4 (x,y,z,0)
    std::vector<float> scale4(numSplats_ * 4);
    for (uint32_t i = 0; i < numSplats_; ++i) {
        scale4[i * 4 + 0] = data.scales[i * 3 + 0];
        scale4[i * 4 + 1] = data.scales[i * 3 + 1];
        scale4[i * 4 + 2] = data.scales[i * 3 + 2];
        scale4[i * 4 + 3] = 0.0f;
    }
    scaleBuffer_ = ctx.newBuffer(scale4.data(),
                                 scale4.size() * sizeof(float),
                                 MTL::ResourceStorageModeShared);
    scaleBuffer_->setLabel(NS::String::string("SplatScales", NS::UTF8StringEncoding));

    // Opacities
    opacityBuffer_ = ctx.newBuffer(data.opacities.data(),
                                   numSplats_ * sizeof(float),
                                   MTL::ResourceStorageModeShared);
    opacityBuffer_->setLabel(NS::String::string("SplatOpacities", NS::UTF8StringEncoding));

    // SH coefficients (degree 0) - pad vec3 to vec4 for alignment
    if (!data.sh_coefficients.empty() && !data.sh_coefficients[0].empty()) {
        const auto& coeffs = data.sh_coefficients[0];
        uint32_t srcFloatsPerSplat = static_cast<uint32_t>(coeffs.size() / numSplats_);

        if (srcFloatsPerSplat == 3) {
            std::vector<float> padded(numSplats_ * 4, 0.0f);
            for (uint32_t i = 0; i < numSplats_; ++i) {
                padded[i * 4 + 0] = coeffs[i * 3 + 0];
                padded[i * 4 + 1] = coeffs[i * 3 + 1];
                padded[i * 4 + 2] = coeffs[i * 3 + 2];
            }
            shBuffer_ = ctx.newBuffer(padded.data(),
                                      padded.size() * sizeof(float),
                                      MTL::ResourceStorageModeShared);
        } else {
            shBuffer_ = ctx.newBuffer(coeffs.data(),
                                      coeffs.size() * sizeof(float),
                                      MTL::ResourceStorageModeShared);
        }
    } else {
        // No SH data — create a dummy buffer
        std::vector<float> dummy(numSplats_ * 4, 0.0f);
        shBuffer_ = ctx.newBuffer(dummy.data(),
                                  dummy.size() * sizeof(float),
                                  MTL::ResourceStorageModeShared);
    }
    shBuffer_->setLabel(NS::String::string("SplatSH0", NS::UTF8StringEncoding));

    // Projected output buffers (shared for compute write / render read)
    size_t vec4Size = numSplats_ * 4 * sizeof(float);

    projCenterBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projCenterBuffer_->setLabel(NS::String::string("ProjCenters", NS::UTF8StringEncoding));

    projConicBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projConicBuffer_->setLabel(NS::String::string("ProjConics", NS::UTF8StringEncoding));

    projColorBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projColorBuffer_->setLabel(NS::String::string("ProjColors", NS::UTF8StringEncoding));

    // Sorted indices buffer (shared for CPU upload)
    sortedIndicesBuffer_ = ctx.newBuffer(numSplats_ * sizeof(uint32_t),
                                         MTL::ResourceStorageModeShared);
    sortedIndicesBuffer_->setLabel(NS::String::string("SortedIndices", NS::UTF8StringEncoding));

    // Initialize sorted indices to identity
    auto* idxPtr = static_cast<uint32_t*>(sortedIndicesBuffer_->contents());
    for (uint32_t i = 0; i < numSplats_; ++i) idxPtr[i] = i;

    // Note: uniforms are passed via setBytes() in renderToTarget, no persistent buffers needed
}

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------

void MetalSplatRenderer::init(MetalContext& ctx, const GaussianSplatData& data,
                               uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;

    createRenderTargets(ctx);
    createPipelines(ctx);
    createBuffers(ctx, data);

    // Default camera from bounding box
    if (data.num_splats > 0) {
        glm::vec3 minB(1e9f), maxB(-1e9f);
        for (uint32_t i = 0; i < data.num_splats; ++i) {
            float x = data.positions[i * 4 + 0];
            float y = data.positions[i * 4 + 1];
            float z = data.positions[i * 4 + 2];
            minB = glm::min(minB, glm::vec3(x, y, z));
            maxB = glm::max(maxB, glm::vec3(x, y, z));
        }
        glm::vec3 center = (minB + maxB) * 0.5f;
        float radius = glm::length(maxB - minB) * 0.5f;
        if (radius < 0.001f) radius = 1.0f;

        // Auto-detect up axis: if Z-extent > Y-extent, model is Z-up (typical for 3DGS captures)
        bool zUp = (maxB.z - minB.z) > (maxB.y - minB.y) * 1.5f;
        camPos_ = zUp
            ? center + glm::vec3(0.0f, radius * 2.5f, radius * 0.3f)   // look from +Y with Z-up
            : center + glm::vec3(0.0f, radius * 0.3f, radius * 2.5f);  // look from +Z with Y-up
        float aspect = static_cast<float>(width) / static_cast<float>(height);
        float fov = glm::radians(45.0f);
        // Auto-detect up axis: if Z-extent > Y-extent, use Z-up camera
        glm::vec3 upVec = zUp
            ? glm::vec3(0.0f, 0.0f, -1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        viewMatrix_ = glm::lookAt(camPos_, center, upVec);
        projMatrix_ = glm::perspective(fov, aspect, 0.01f, radius * 10.0f);
        // Metal NDC: no Y-flip needed (Metal uses [0,1] depth, top-left origin handled by viewport)

        focalY_ = 0.5f * static_cast<float>(height) / tanf(fov * 0.5f);
        focalX_ = focalY_;
    }

    std::cout << "[metal] MetalSplatRenderer initialized: " << numSplats_ << " splats, "
              << width << "x" << height << std::endl;
}

// --------------------------------------------------------------------------
// Update camera
// --------------------------------------------------------------------------

void MetalSplatRenderer::updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                                       float fovY, float aspect,
                                       float nearPlane, float farPlane) {
    camPos_ = eye;
    viewMatrix_ = glm::lookAt(eye, target, up);
    projMatrix_ = glm::perspective(fovY, aspect, nearPlane, farPlane);
    // No Y-flip for Metal

    focalY_ = 0.5f * static_cast<float>(height_) / tanf(fovY * 0.5f);
    focalX_ = focalY_;
}

// --------------------------------------------------------------------------
// CPU depth sort (back-to-front)
// --------------------------------------------------------------------------

void MetalSplatRenderer::cpuSort() {
    const float* vm = &viewMatrix_[0][0];
    std::vector<float> depths(numSplats_);
    for (uint32_t i = 0; i < numSplats_; ++i) {
        float x = hostPositions_[i * 4 + 0];
        float y = hostPositions_[i * 4 + 1];
        float z = hostPositions_[i * 4 + 2];
        depths[i] = vm[2] * x + vm[6] * y + vm[10] * z + vm[14];
    }

    std::vector<uint32_t> indices(numSplats_);
    std::iota(indices.begin(), indices.end(), 0u);
    std::sort(indices.begin(), indices.end(), [&](uint32_t a, uint32_t b) {
        return depths[a] < depths[b]; // back-to-front
    });

    auto* dst = static_cast<uint32_t*>(sortedIndicesBuffer_->contents());
    std::memcpy(dst, indices.data(), numSplats_ * sizeof(uint32_t));
}

// --------------------------------------------------------------------------
// Render (offscreen)
// --------------------------------------------------------------------------

void MetalSplatRenderer::render(MetalContext& ctx) {
    if (numSplats_ == 0) return;

    cpuSort();
    renderToTarget(ctx, colorTarget_, depthTarget_, width_, height_);
}

// --------------------------------------------------------------------------
// Render to drawable (interactive)
// --------------------------------------------------------------------------

void MetalSplatRenderer::renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) {
    if (numSplats_ == 0) return;

    cpuSort();

    auto* drawableTex = drawable->texture();
    uint32_t drawW = static_cast<uint32_t>(drawableTex->width());
    uint32_t drawH = static_cast<uint32_t>(drawableTex->height());

    // Rebuild depth buffer if drawable size changed
    if (depthTarget_ &&
        (depthTarget_->width() != drawW || depthTarget_->height() != drawH)) {
        depthTarget_->release();
        auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatDepth32Float, drawW, drawH, false);
        depthDesc->setUsage(MTL::TextureUsageRenderTarget);
        depthDesc->setStorageMode(MTL::StorageModePrivate);
        depthTarget_ = ctx.newTexture(depthDesc);
    }

    renderToTarget(ctx, drawableTex, depthTarget_, drawW, drawH);

    // Present
    auto* cmdBuf = ctx.beginCommandBuffer();
    cmdBuf->presentDrawable(drawable);
    cmdBuf->commit();
}

// --------------------------------------------------------------------------
// Internal render to a specific target
// --------------------------------------------------------------------------

void MetalSplatRenderer::renderToTarget(MetalContext& ctx, MTL::Texture* colorTex,
                                         MTL::Texture* depthTex,
                                         uint32_t drawW, uint32_t drawH) {
    // --- Build compute uniforms as raw bytes (matching MSL packed_float3 layout) ---
    uint8_t uniformData[176] = {};
    std::memcpy(uniformData + 0, &viewMatrix_[0][0], 64);    // view: offset 0
    std::memcpy(uniformData + 64, &projMatrix_[0][0], 64);   // proj: offset 64
    float cp[3] = {camPos_.x, camPos_.y, camPos_.z};
    std::memcpy(uniformData + 128, cp, 12);                   // camPos: offset 128
    // _pad0 at offset 140 (already zero)
    float sw = static_cast<float>(drawW);
    float sh = static_cast<float>(drawH);
    std::memcpy(uniformData + 144, &sw, 4);                   // screenW: offset 144
    std::memcpy(uniformData + 148, &sh, 4);                   // screenH: offset 148
    std::memcpy(uniformData + 152, &numSplats_, 4);           // numSplats: offset 152
    std::memcpy(uniformData + 156, &focalX_, 4);              // focalX: offset 156
    std::memcpy(uniformData + 160, &focalY_, 4);              // focalY: offset 160
    int32_t shd = static_cast<int32_t>(shDegree_);
    std::memcpy(uniformData + 164, &shd, 4);                  // shDegree: offset 164

    // --- Build render uniforms ---
    struct RenderUniforms {
        float screenW, screenH;
        uint32_t visibleCount;
        float alphaCutoff;
    };
    RenderUniforms renderUniforms = {};
    renderUniforms.screenW = static_cast<float>(drawW);
    renderUniforms.screenH = static_cast<float>(drawH);
    renderUniforms.visibleCount = numSplats_;
    renderUniforms.alphaCutoff = 0.004f;

    // --- Command buffer with compute + render ---
    auto* cmdBuf = ctx.beginCommandBuffer();

    // --- Compute dispatch ---
    auto* compEnc = cmdBuf->computeCommandEncoder();
    compEnc->setComputePipelineState(computePipeline_);

    compEnc->setBuffer(posBuffer_, 0, 0);
    compEnc->setBuffer(rotBuffer_, 0, 1);
    compEnc->setBuffer(scaleBuffer_, 0, 2);
    compEnc->setBuffer(opacityBuffer_, 0, 3);
    compEnc->setBuffer(shBuffer_, 0, 4);
    compEnc->setBuffer(projCenterBuffer_, 0, 5);
    compEnc->setBuffer(projConicBuffer_, 0, 6);
    compEnc->setBuffer(projColorBuffer_, 0, 7);
    compEnc->setBytes(uniformData, 176, 8);

    uint32_t threadGroupSize = static_cast<uint32_t>(computePipeline_->maxTotalThreadsPerThreadgroup());
    if (threadGroupSize > 256) threadGroupSize = 256;
    uint32_t groupCount = (numSplats_ + threadGroupSize - 1) / threadGroupSize;

    compEnc->dispatchThreadgroups(MTL::Size(groupCount, 1, 1),
                                  MTL::Size(threadGroupSize, 1, 1));
    compEnc->endEncoding();

    // --- Render pass ---
    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();

    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(colorTex);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    colorAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 1.0));

    auto* depthAtt = rpDesc->depthAttachment();
    depthAtt->setTexture(depthTex);
    depthAtt->setLoadAction(MTL::LoadActionClear);
    depthAtt->setStoreAction(MTL::StoreActionDontCare);
    depthAtt->setClearDepth(1.0);

    auto* renderEnc = cmdBuf->renderCommandEncoder(rpDesc);
    rpDesc->release();

    renderEnc->setRenderPipelineState(renderPipeline_);
    renderEnc->setDepthStencilState(depthStencilState_);
    renderEnc->setCullMode(MTL::CullModeNone);

    MTL::Viewport viewport = {};
    viewport.originX = 0.0;
    viewport.originY = 0.0;
    viewport.width = static_cast<double>(drawW);
    viewport.height = static_cast<double>(drawH);
    viewport.znear = 0.0;
    viewport.zfar = 1.0;
    renderEnc->setViewport(viewport);

    // Bind buffers for vertex + fragment
    renderEnc->setVertexBuffer(projCenterBuffer_, 0, 0);
    renderEnc->setVertexBuffer(projConicBuffer_, 0, 1);
    renderEnc->setVertexBuffer(projColorBuffer_, 0, 2);
    renderEnc->setVertexBuffer(sortedIndicesBuffer_, 0, 3);
    renderEnc->setVertexBytes(&renderUniforms, sizeof(renderUniforms), 4);
    renderEnc->setFragmentBytes(&renderUniforms, sizeof(renderUniforms), 4);

    // Instanced draw: 6 vertices (quad) x numSplats instances
    renderEnc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0u, 6u, numSplats_);

    renderEnc->endEncoding();

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MetalSplatRenderer::cleanup() {
    if (colorTarget_) { colorTarget_->release(); colorTarget_ = nullptr; }
    if (depthTarget_) { depthTarget_->release(); depthTarget_ = nullptr; }

    if (computePipeline_) { computePipeline_->release(); computePipeline_ = nullptr; }
    if (renderPipeline_) { renderPipeline_->release(); renderPipeline_ = nullptr; }
    if (depthStencilState_) { depthStencilState_->release(); depthStencilState_ = nullptr; }

    if (posBuffer_) { posBuffer_->release(); posBuffer_ = nullptr; }
    if (rotBuffer_) { rotBuffer_->release(); rotBuffer_ = nullptr; }
    if (scaleBuffer_) { scaleBuffer_->release(); scaleBuffer_ = nullptr; }
    if (opacityBuffer_) { opacityBuffer_->release(); opacityBuffer_ = nullptr; }
    if (shBuffer_) { shBuffer_->release(); shBuffer_ = nullptr; }

    if (projCenterBuffer_) { projCenterBuffer_->release(); projCenterBuffer_ = nullptr; }
    if (projConicBuffer_) { projConicBuffer_->release(); projConicBuffer_ = nullptr; }
    if (projColorBuffer_) { projColorBuffer_->release(); projColorBuffer_ = nullptr; }

    if (sortedIndicesBuffer_) { sortedIndicesBuffer_->release(); sortedIndicesBuffer_ = nullptr; }
}
