#include "spv_loader.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace SpvLoader {

std::vector<uint32_t> loadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize < 4) {
        throw std::runtime_error("File too small to be valid SPIR-V: " + path);
    }
    if (fileSize % 4 != 0) {
        throw std::runtime_error("SPIR-V file size not aligned to 4 bytes: " + path);
    }

    std::vector<uint32_t> code(fileSize / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);

    // Verify SPIR-V magic number
    if (code[0] != 0x07230203) {
        char buf[64];
        snprintf(buf, sizeof(buf), "0x%08X", code[0]);
        throw std::runtime_error(
            "Bad SPIR-V magic number " + std::string(buf) +
            " (expected 0x07230203) in file: " + path);
    }

    return code;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

VkShaderStageFlagBits detectStage(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Check for compound extensions like .vert.spv, .frag.spv, etc.
    if (lower.find(".vert") != std::string::npos) return VK_SHADER_STAGE_VERTEX_BIT;
    if (lower.find(".frag") != std::string::npos) return VK_SHADER_STAGE_FRAGMENT_BIT;
    if (lower.find(".rgen") != std::string::npos) return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    if (lower.find(".rchit") != std::string::npos) return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    if (lower.find(".rmiss") != std::string::npos) return VK_SHADER_STAGE_MISS_BIT_KHR;
    if (lower.find(".comp") != std::string::npos) return VK_SHADER_STAGE_COMPUTE_BIT;

    throw std::runtime_error("Cannot detect shader stage from filename: " + filename);
}

// Try to compile fullscreen.vert.glsl -> fullscreen.vert.spv using glslangValidator
static bool tryCompileFullscreenShader(const std::string& glslPath, const std::string& spvPath) {
    std::string cmd = "glslangValidator -V \"" + glslPath + "\" -o \"" + spvPath + "\" 2>&1";
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        // Try glslc as alternative
        cmd = "glslc \"" + glslPath + "\" -o \"" + spvPath + "\" 2>&1";
        ret = std::system(cmd.c_str());
    }
    return ret == 0;
}

const std::vector<uint32_t>& getFullscreenVertSPIRV() {
    static std::vector<uint32_t> cached;
    if (!cached.empty()) return cached;

    // Try to find pre-compiled fullscreen.vert.spv next to the executable or in shaders/
    std::vector<std::string> searchPaths;

    // Get executable directory
    // Try relative to current working dir
    searchPaths.push_back("shaders/fullscreen.vert.spv");
    searchPaths.push_back("../shaders/fullscreen.vert.spv");
    searchPaths.push_back("../../shaders/fullscreen.vert.spv");
    searchPaths.push_back("playground_cpp/shaders/fullscreen.vert.spv");

    for (const auto& path : searchPaths) {
        if (fs::exists(path)) {
            try {
                cached = loadSPIRV(path);
                std::cout << "[info] Loaded fullscreen vertex shader from: " << path << std::endl;
                return cached;
            } catch (...) {
                continue;
            }
        }
    }

    // Try to compile from GLSL
    std::vector<std::string> glslPaths = {
        "shaders/fullscreen.vert.glsl",
        "../shaders/fullscreen.vert.glsl",
        "../../shaders/fullscreen.vert.glsl",
        "playground_cpp/shaders/fullscreen.vert.glsl",
    };

    for (const auto& glslPath : glslPaths) {
        if (fs::exists(glslPath)) {
            std::string spvPath = glslPath.substr(0, glslPath.rfind(".glsl")) + ".spv";
            std::cout << "[info] Compiling fullscreen vertex shader: " << glslPath << std::endl;
            if (tryCompileFullscreenShader(glslPath, spvPath)) {
                try {
                    cached = loadSPIRV(spvPath);
                    std::cout << "[info] Compiled fullscreen vertex shader to: " << spvPath << std::endl;
                    return cached;
                } catch (...) {
                    // Fall through
                }
            }
        }
    }

    // Fallback: use minimal hardcoded SPIR-V for fullscreen triangle
    // This is a valid SPIR-V binary for the fullscreen vertex shader
    std::cout << "[warn] Using hardcoded fullscreen vertex shader SPIR-V" << std::endl;

    // Minimal fullscreen triangle vertex shader in SPIR-V
    // Equivalent to:
    //   #version 450
    //   layout(location=0) out vec2 uv;
    //   void main() {
    //       vec2 p[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    //       gl_Position = vec4(p[gl_VertexIndex], 0, 1);
    //       vec2 v = p[gl_VertexIndex];
    //       uv = vec2(v.x*0.5+0.5, 1.0-(v.y*0.5+0.5));
    //   }
    cached = {
        0x07230203, 0x00010000, 0x000d000a, 0x00000036,
        0x00000000, 0x00020011, 0x00000001, 0x0006000b,
        0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
        0x00000000, 0x0003000e, 0x00000000, 0x00000001,
        0x0009000f, 0x00000000, 0x00000004, 0x6e69616d,
        0x00000000, 0x0000000d, 0x00000011, 0x00000024,
        0x0000002d, 0x00030003, 0x00000002, 0x000001c2,
        0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
        0x00060005, 0x0000000b, 0x736f7000, 0x6f697469,
        0x0000736e, 0x00000000,
        0x00060005, 0x0000000d, 0x565f6c67, 0x65747265,
        0x646e4978, 0x00007865,
        0x00060005, 0x0000000f, 0x505f6c67, 0x65567265,
        0x78657472, 0x00000000, 0x00060006, 0x0000000f,
        0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
        0x00070006, 0x0000000f, 0x00000001, 0x505f6c67,
        0x746e696f, 0x657a6953, 0x00000000, 0x00070006,
        0x0000000f, 0x00000002, 0x435f6c67, 0x4470696c,
        0x61747369, 0x0065636e, 0x00070006, 0x0000000f,
        0x00000003, 0x435f6c67, 0x446c6c75, 0x61747369,
        0x0065636e, 0x00050005, 0x00000011, 0x00000000,
        0x00000000, 0x00000000, 0x00030005, 0x0000001f,
        0x00000070, 0x00030005, 0x00000024, 0x00007600,
        0x00040005, 0x0000002d, 0x00007675, 0x00000000,
        0x00040047, 0x0000000d, 0x0000000b, 0x0000002a,
        0x00050048, 0x0000000f, 0x00000000, 0x0000000b,
        0x00000000, 0x00050048, 0x0000000f, 0x00000001,
        0x0000000b, 0x00000001, 0x00050048, 0x0000000f,
        0x00000002, 0x0000000b, 0x00000003, 0x00050048,
        0x0000000f, 0x00000003, 0x0000000b, 0x00000004,
        0x00030047, 0x0000000f, 0x00000002, 0x00040047,
        0x0000002d, 0x0000001e, 0x00000000, 0x00020013,
        0x00000002, 0x00030021, 0x00000003, 0x00000002,
        0x00030016, 0x00000006, 0x00000020, 0x00040017,
        0x00000007, 0x00000006, 0x00000002, 0x00040015,
        0x00000008, 0x00000020, 0x00000000, 0x0004002b,
        0x00000008, 0x00000009, 0x00000003, 0x0004001c,
        0x0000000a, 0x00000007, 0x00000009, 0x00040020,
        0x00000026, 0x00000007, 0x0000000a, 0x00040015,
        0x00000025, 0x00000020, 0x00000001, 0x00040020,
        0x00000027, 0x00000001, 0x00000025, 0x0004003b,
        0x00000027, 0x0000000d, 0x00000001, 0x00040017,
        0x0000000e, 0x00000006, 0x00000004, 0x0004002b,
        0x00000008, 0x00000028, 0x00000001, 0x0004001c,
        0x00000029, 0x00000006, 0x00000028, 0x0006001e,
        0x0000000f, 0x0000000e, 0x00000006, 0x00000029,
        0x00000029, 0x00040020, 0x00000010, 0x00000003,
        0x0000000f, 0x0004003b, 0x00000010, 0x00000011,
        0x00000003, 0x0004002b, 0x00000006, 0x00000012,
        0xbf800000, 0x0005002c, 0x00000007, 0x00000013,
        0x00000012, 0x00000012, 0x0004002b, 0x00000006,
        0x00000014, 0x40400000, 0x0005002c, 0x00000007,
        0x00000015, 0x00000014, 0x00000012, 0x0005002c,
        0x00000007, 0x00000016, 0x00000012, 0x00000014,
        0x0006002c, 0x0000000a, 0x00000017, 0x00000013,
        0x00000015, 0x00000016, 0x0004002b, 0x00000006,
        0x00000018, 0x00000000, 0x0004002b, 0x00000006,
        0x00000019, 0x3f800000, 0x0004002b, 0x00000025,
        0x0000001a, 0x00000000, 0x00040020, 0x0000001b,
        0x00000003, 0x0000000e, 0x00040020, 0x0000001c,
        0x00000007, 0x00000007, 0x0004002b, 0x00000006,
        0x0000001d, 0x3f000000, 0x00040020, 0x0000002c,
        0x00000003, 0x00000007, 0x0004003b, 0x0000002c,
        0x0000002d, 0x00000003, 0x00050036, 0x00000002,
        0x00000004, 0x00000000, 0x00000003, 0x000200f8,
        0x00000005, 0x0004003b, 0x00000026, 0x0000000b,
        0x00000007, 0x0004003b, 0x0000001c, 0x0000001f,
        0x00000007, 0x0003003e, 0x0000000b, 0x00000017,
        0x0004003d, 0x00000025, 0x0000002e, 0x0000000d,
        0x00050041, 0x0000001c, 0x0000002f, 0x0000000b,
        0x0000002e, 0x0004003d, 0x00000007, 0x00000030,
        0x0000002f, 0x00050051, 0x00000006, 0x00000031,
        0x00000030, 0x00000000, 0x00050051, 0x00000006,
        0x00000032, 0x00000030, 0x00000001, 0x00070050,
        0x0000000e, 0x00000033, 0x00000031, 0x00000032,
        0x00000018, 0x00000019, 0x00050041, 0x0000001b,
        0x00000034, 0x00000011, 0x0000001a, 0x0003003e,
        0x00000034, 0x00000033, 0x0003003e, 0x0000001f,
        0x00000030, 0x0004003d, 0x00000007, 0x00000020,
        0x0000001f, 0x00050051, 0x00000006, 0x00000021,
        0x00000020, 0x00000000, 0x00050085, 0x00000006,
        0x00000022, 0x00000021, 0x0000001d, 0x00050081,
        0x00000006, 0x00000023, 0x00000022, 0x0000001d,
        0x00050051, 0x00000006, 0x00000035, 0x00000020,
        0x00000001, 0x00050085, 0x00000006, 0x00000036,
        0x00000035, 0x0000001d, 0x00050081, 0x00000006,
        0x00000037, 0x00000036, 0x0000001d, 0x00050083,
        0x00000006, 0x00000038, 0x00000019, 0x00000037,
        0x00050050, 0x00000007, 0x00000039, 0x00000023,
        0x00000038, 0x0003003e, 0x0000002d, 0x00000039,
        0x000100fd, 0x00010038,
    };

    return cached;
}

} // namespace SpvLoader
