#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <cstdint>

class MetalContext;

namespace MetalScreenshot {

// Capture a Metal texture to a PNG file.
// Supports RGBA8Unorm and BGRA8Unorm pixel formats.
void saveTextureToPNG(MetalContext& ctx, MTL::Texture* texture,
                      uint32_t width, uint32_t height,
                      const std::string& outputPath);

} // namespace MetalScreenshot
