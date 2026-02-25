#pragma once

// Forward declarations for Metal-cpp types
namespace CA { class MetalLayer; }
namespace MTL { class Device; }
struct GLFWwindow;

// Create a CAMetalLayer and attach it to a GLFW window (ObjC++ implementation)
CA::MetalLayer* createMetalLayer(GLFWwindow* window, MTL::Device* device);

// Update the drawable size of a CAMetalLayer to match the GLFW window's framebuffer
void updateDrawableSize(CA::MetalLayer* layer, GLFWwindow* window);
