#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstdint>

namespace SpvLoader {

// Load SPIR-V binary from file, verifying magic number
std::vector<uint32_t> loadSPIRV(const std::string& path);

// Create a VkShaderModule from SPIR-V code
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code);

// Detect shader stage from filename extension
// Supported: .vert -> VERTEX, .frag -> FRAGMENT, .rgen -> RAYGEN_KHR,
//            .rchit -> CLOSEST_HIT_KHR, .rmiss -> MISS_KHR
VkShaderStageFlagBits detectStage(const std::string& filename);

// Get the hardcoded fullscreen vertex shader SPIR-V
const std::vector<uint32_t>& getFullscreenVertSPIRV();

} // namespace SpvLoader
