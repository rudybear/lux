#include "metal_screenshot.h"
#include "metal_context.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace MetalScreenshot {

void saveTextureToPNG(MetalContext& ctx, MTL::Texture* texture,
                      uint32_t width, uint32_t height,
                      const std::string& outputPath) {
    if (!texture) {
        throw std::runtime_error("Cannot save screenshot: null texture");
    }

    // Create a shared (CPU-readable) staging buffer
    size_t bytesPerRow = width * 4;
    size_t bufferSize = bytesPerRow * height;
    auto* stagingBuffer = ctx.newBuffer(bufferSize, MTL::ResourceStorageModeShared);

    // Blit texture to staging buffer
    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* blit = cmdBuf->blitCommandEncoder();
    blit->copyFromTexture(texture, 0, 0,
                          MTL::Origin(0, 0, 0),
                          MTL::Size(width, height, 1),
                          stagingBuffer, 0, bytesPerRow, 0);
    blit->endEncoding();
    ctx.submitAndWait(cmdBuf);

    // Read back pixel data
    std::vector<uint8_t> pixels(bufferSize);
    memcpy(pixels.data(), stagingBuffer->contents(), bufferSize);
    stagingBuffer->release();

    // Handle BGRA → RGBA swizzle if needed
    MTL::PixelFormat format = texture->pixelFormat();
    if (format == MTL::PixelFormatBGRA8Unorm || format == MTL::PixelFormatBGRA8Unorm_sRGB) {
        for (size_t i = 0; i < bufferSize; i += 4) {
            std::swap(pixels[i], pixels[i + 2]); // swap R and B
        }
    }

    // Write PNG
    int result = stbi_write_png(outputPath.c_str(), static_cast<int>(width), static_cast<int>(height),
                                4, pixels.data(), static_cast<int>(bytesPerRow));
    if (!result) {
        throw std::runtime_error("Failed to write PNG: " + outputPath);
    }

    std::cout << "[metal] Screenshot saved: " << outputPath
              << " (" << width << "x" << height << ")" << std::endl;
}

} // namespace MetalScreenshot
