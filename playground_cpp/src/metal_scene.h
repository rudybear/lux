#pragma once

#include <Metal/Metal.hpp>
#include <vector>
#include <cstdint>
#include <string>

class MetalContext;

// GPU mesh data for Metal
struct MetalGPUMesh {
    MTL::Buffer* vertexBuffer = nullptr;
    MTL::Buffer* indexBuffer = nullptr;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    int vertexStride = 0;
};

// GPU texture data for Metal
struct MetalGPUTexture {
    MTL::Texture* texture = nullptr;
    MTL::SamplerState* sampler = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Per-material draw range (same as Vulkan's DrawRange concept)
struct MetalDrawRange {
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    int materialIndex = 0;
};

namespace MetalScene {

// Upload vertex + index data to Metal buffers
MetalGPUMesh uploadMesh(MetalContext& ctx, const void* vertexData, size_t vertexSize,
                        const uint32_t* indexData, size_t indexCount, int stride);

// Upload RGBA8 texture with mipmap generation
MetalGPUTexture uploadTexture(MetalContext& ctx, const uint8_t* pixels,
                               uint32_t width, uint32_t height, int channels = 4,
                               bool generateMips = true);

// Upload cubemap from 6 faces (each face is width x width RGBA8)
MetalGPUTexture uploadCubemap(MetalContext& ctx, const uint8_t* const faces[6],
                               uint32_t faceSize);

// Upload cubemap from F16 data (for IBL)
MetalGPUTexture uploadCubemapF16(MetalContext& ctx, uint32_t faceSize, uint32_t mipCount,
                                  const uint16_t* data, size_t dataSize);

// Create sampler state
MTL::SamplerState* createSampler(MetalContext& ctx,
                                  MTL::SamplerMinMagFilter minMag = MTL::SamplerMinMagFilterLinear,
                                  MTL::SamplerAddressMode address = MTL::SamplerAddressModeRepeat,
                                  bool mipFilter = true);

// Create a default 1x1 texture of a given color
MetalGPUTexture createDefaultTexture(MetalContext& ctx, uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

// Cleanup
void destroyMesh(MetalGPUMesh& mesh);
void destroyTexture(MetalGPUTexture& tex);

} // namespace MetalScene
