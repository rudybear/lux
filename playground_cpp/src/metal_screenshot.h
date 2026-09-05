#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <cstdint>
#include <vector>

class MetalContext;

namespace MetalScreenshot {

// Capture a Metal texture to a PNG file.
// Supports RGBA8Unorm and BGRA8Unorm pixel formats.
void saveTextureToPNG(MetalContext& ctx, MTL::Texture* texture,
                      uint32_t width, uint32_t height,
                      const std::string& outputPath);

// Reads back the raw bytes of a Metal texture (no format conversion, no PNG
// encoding) via a blit into a shared staging buffer -- used for the DLSS
// auxiliary attachments (RGBA32F motion, RG32F expected depth). `bytesPerPixel`
// must match the texture's actual pixel format.
std::vector<uint8_t> readTextureRaw(MetalContext& ctx, MTL::Texture* texture,
                                     uint32_t width, uint32_t height,
                                     uint32_t bytesPerPixel);

} // namespace MetalScreenshot
