#pragma once
// Android port of playground_cpp/src/vulkan_context.{h,cpp}'s window/surface/
// swapchain setup. We can't touch vulkan_context.cpp (it #includes GLFW,
// which doesn't exist on Android) so this file re-implements just the
// instance/surface/device/swapchain bring-up using the same vk-bootstrap
// library, targeting VK_KHR_android_surface + an ANativeWindow instead of a
// GLFWwindow. It fills in the same `VulkanContext` struct (vulkan_context.h,
// included unmodified) so splat_renderer.cpp / scene_manager.cpp / etc. can
// be reused byte-for-byte.

#include "vulkan_context.h"

struct ANativeWindow;

namespace AndroidVulkan {

// Creates instance + VK_KHR_android_surface surface + device + queue +
// command pool + VMA allocator, mirroring VulkanContext::init(..., headless=false, ...)
// but without GLFW. Throws std::runtime_error on failure.
void init(VulkanContext& ctx, ANativeWindow* window, bool forceValidation = false);

// Mirrors VulkanContext::createSwapchain (same vk-bootstrap SwapchainBuilder
// call) -- duplicated here only because it lives in the GLFW-including TU.
void createSwapchain(VulkanContext& ctx, uint32_t width, uint32_t height);

// Mirrors VulkanContext::cleanup().
void cleanup(VulkanContext& ctx);

}  // namespace AndroidVulkan
