#pragma once

#include <glm/glm.hpp>

namespace Camera {

// Perspective projection matrix with Vulkan clip space (Y-flip, depth [0,1])
glm::mat4 perspective(float fovY, float aspect, float nearPlane, float farPlane);

// Look-at view matrix
glm::mat4 lookAt(glm::vec3 eye, glm::vec3 target, glm::vec3 up);

// Default camera parameters
constexpr glm::vec3 DEFAULT_EYE    = glm::vec3(0.0f, 0.0f, 3.0f);
constexpr glm::vec3 DEFAULT_TARGET = glm::vec3(0.0f, 0.0f, 0.0f);
constexpr glm::vec3 DEFAULT_UP     = glm::vec3(0.0f, 1.0f, 0.0f);
constexpr float DEFAULT_FOV        = 45.0f; // degrees
constexpr float DEFAULT_NEAR       = 0.1f;
constexpr float DEFAULT_FAR        = 100.0f;

// Get default MVP matrices for PBR rendering
struct CameraMatrices {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

CameraMatrices getDefaultMatrices(float aspect);

// Get inverse matrices for ray tracing
struct RTCameraData {
    glm::mat4 inverseView;
    glm::mat4 inverseProjection;
};

RTCameraData getRTCameraData(float aspect);

} // namespace Camera
