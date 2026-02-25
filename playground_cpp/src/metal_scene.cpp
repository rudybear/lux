#include "metal_scene.h"
#include "metal_context.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace MetalScene {

MetalGPUMesh uploadMesh(MetalContext& ctx, const void* vertexData, size_t vertexSize,
                        const uint32_t* indexData, size_t indexCount, int stride) {
    MetalGPUMesh mesh;
    mesh.vertexStride = stride;
    mesh.indexCount = static_cast<uint32_t>(indexCount);
    mesh.vertexCount = static_cast<uint32_t>(vertexSize / stride);

    // Create vertex buffer (shared memory for simplicity; private + staging for perf)
    mesh.vertexBuffer = ctx.newBuffer(vertexData, vertexSize, MTL::ResourceStorageModeShared);
    if (!mesh.vertexBuffer) {
        throw std::runtime_error("Failed to create Metal vertex buffer");
    }

    // Create index buffer
    size_t indexSize = indexCount * sizeof(uint32_t);
    mesh.indexBuffer = ctx.newBuffer(indexData, indexSize, MTL::ResourceStorageModeShared);
    if (!mesh.indexBuffer) {
        throw std::runtime_error("Failed to create Metal index buffer");
    }

    return mesh;
}

MetalGPUTexture uploadTexture(MetalContext& ctx, const uint8_t* pixels,
                               uint32_t width, uint32_t height, int channels,
                               bool generateMips) {
    MetalGPUTexture tex;
    tex.width = width;
    tex.height = height;

    // Calculate mip count
    uint32_t mipCount = 1;
    if (generateMips) {
        mipCount = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
    }

    // Create texture descriptor
    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, width, height, generateMips);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setMipmapLevelCount(mipCount);

    tex.texture = ctx.newTexture(desc);
    if (!tex.texture) {
        throw std::runtime_error("Failed to create Metal texture");
    }

    // Convert to RGBA if needed
    std::vector<uint8_t> rgbaData;
    const uint8_t* uploadData = pixels;
    if (channels == 3) {
        rgbaData.resize(width * height * 4);
        for (uint32_t i = 0; i < width * height; i++) {
            rgbaData[i * 4 + 0] = pixels[i * 3 + 0];
            rgbaData[i * 4 + 1] = pixels[i * 3 + 1];
            rgbaData[i * 4 + 2] = pixels[i * 3 + 2];
            rgbaData[i * 4 + 3] = 255;
        }
        uploadData = rgbaData.data();
    }

    // Upload via staging buffer + blit command encoder
    size_t bytesPerRow = width * 4;
    size_t dataSize = bytesPerRow * height;
    auto* staging = ctx.newBuffer(uploadData, dataSize, MTL::ResourceStorageModeShared);

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* blit = cmdBuf->blitCommandEncoder();
    blit->copyFromBuffer(staging, 0, bytesPerRow, 0,
                         MTL::Size(width, height, 1),
                         tex.texture, 0, 0,
                         MTL::Origin(0, 0, 0));
    if (generateMips) {
        blit->generateMipmaps(tex.texture);
    }
    blit->endEncoding();
    ctx.submitAndWait(cmdBuf);
    staging->release();

    // Create default sampler
    tex.sampler = createSampler(ctx);

    return tex;
}

MetalGPUTexture uploadCubemap(MetalContext& ctx, const uint8_t* const faces[6],
                               uint32_t faceSize) {
    MetalGPUTexture tex;
    tex.width = faceSize;
    tex.height = faceSize;

    auto* desc = MTL::TextureDescriptor::textureCubeDescriptor(
        MTL::PixelFormatRGBA8Unorm, faceSize, false);
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);

    tex.texture = ctx.newTexture(desc);

    size_t bytesPerRow = faceSize * 4;
    size_t bytesPerFace = bytesPerRow * faceSize;

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* blit = cmdBuf->blitCommandEncoder();

    MTL::Buffer* stagingBuffers[6] = {};
    for (int face = 0; face < 6; face++) {
        stagingBuffers[face] = ctx.newBuffer(faces[face], bytesPerFace, MTL::ResourceStorageModeShared);
        blit->copyFromBuffer(stagingBuffers[face], 0, bytesPerRow, 0,
                             MTL::Size(faceSize, faceSize, 1),
                             tex.texture, face, 0,
                             MTL::Origin(0, 0, 0));
    }

    blit->endEncoding();
    ctx.submitAndWait(cmdBuf);

    for (int face = 0; face < 6; face++) {
        stagingBuffers[face]->release();
    }

    tex.sampler = createSampler(ctx, MTL::SamplerMinMagFilterLinear,
                                MTL::SamplerAddressModeClampToEdge, false);
    return tex;
}

MetalGPUTexture uploadCubemapF16(MetalContext& ctx, uint32_t faceSize, uint32_t mipCount,
                                  const uint16_t* data, size_t dataSize) {
    MetalGPUTexture tex;
    tex.width = faceSize;
    tex.height = faceSize;

    auto* desc = MTL::TextureDescriptor::textureCubeDescriptor(
        MTL::PixelFormatRGBA16Float, faceSize, (mipCount > 1));
    desc->setUsage(MTL::TextureUsageShaderRead);
    desc->setStorageMode(MTL::StorageModePrivate);
    desc->setMipmapLevelCount(mipCount);

    tex.texture = ctx.newTexture(desc);

    // Upload each mip level for each face
    auto* staging = ctx.newBuffer(data, dataSize * sizeof(uint16_t), MTL::ResourceStorageModeShared);

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* blit = cmdBuf->blitCommandEncoder();

    size_t offset = 0;
    for (uint32_t mip = 0; mip < mipCount; mip++) {
        uint32_t mipSize = std::max(1u, faceSize >> mip);
        size_t bytesPerRow = mipSize * 4 * sizeof(uint16_t); // RGBA16F = 8 bytes per pixel
        size_t bytesPerFace = bytesPerRow * mipSize;

        for (int face = 0; face < 6; face++) {
            blit->copyFromBuffer(staging, offset, bytesPerRow, 0,
                                 MTL::Size(mipSize, mipSize, 1),
                                 tex.texture, face, mip,
                                 MTL::Origin(0, 0, 0));
            offset += bytesPerFace;
        }
    }

    blit->endEncoding();
    ctx.submitAndWait(cmdBuf);
    staging->release();

    tex.sampler = createSampler(ctx, MTL::SamplerMinMagFilterLinear,
                                MTL::SamplerAddressModeClampToEdge, (mipCount > 1));
    return tex;
}

MTL::SamplerState* createSampler(MetalContext& ctx,
                                  MTL::SamplerMinMagFilter minMag,
                                  MTL::SamplerAddressMode address,
                                  bool mipFilter) {
    auto* desc = MTL::SamplerDescriptor::alloc()->init();
    desc->setMinFilter(minMag);
    desc->setMagFilter(minMag);
    desc->setSAddressMode(address);
    desc->setTAddressMode(address);
    desc->setRAddressMode(address);
    if (mipFilter) {
        desc->setMipFilter(MTL::SamplerMipFilterLinear);
    }
    desc->setMaxAnisotropy(16);

    auto* sampler = ctx.device->newSamplerState(desc);
    desc->release();
    return sampler;
}

MetalGPUTexture createDefaultTexture(MetalContext& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint8_t pixels[4] = {r, g, b, a};
    return uploadTexture(ctx, pixels, 1, 1, 4, false);
}

void destroyMesh(MetalGPUMesh& mesh) {
    if (mesh.vertexBuffer) { mesh.vertexBuffer->release(); mesh.vertexBuffer = nullptr; }
    if (mesh.indexBuffer) { mesh.indexBuffer->release(); mesh.indexBuffer = nullptr; }
}

void destroyTexture(MetalGPUTexture& tex) {
    if (tex.texture) { tex.texture->release(); tex.texture = nullptr; }
    if (tex.sampler) { tex.sampler->release(); tex.sampler = nullptr; }
}

} // namespace MetalScene
