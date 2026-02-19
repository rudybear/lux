#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace Camera {

glm::mat4 perspective(float fovY, float aspect, float nearPlane, float farPlane) {
    // Vulkan clip space: Y-down (flip Y), depth [0, 1]
    // This matches the Python playground's perspective() function
    float f = 1.0f / std::tan(fovY / 2.0f);

    glm::mat4 m(0.0f);
    m[0][0] = f / aspect;
    m[1][1] = -f;  // Vulkan Y-flip
    m[2][2] = farPlane / (nearPlane - farPlane);
    m[2][3] = -1.0f;
    m[3][2] = (nearPlane * farPlane) / (nearPlane - farPlane);

    return m;
}

glm::mat4 lookAt(glm::vec3 eye, glm::vec3 target, glm::vec3 up) {
    // Standard look-at matrix matching the Python playground
    glm::vec3 f = glm::normalize(target - eye);
    glm::vec3 s = glm::normalize(glm::cross(f, up));
    glm::vec3 u = glm::cross(s, f);

    glm::mat4 m(1.0f);
    m[0][0] = s.x;  m[1][0] = s.y;  m[2][0] = s.z;
    m[0][1] = u.x;  m[1][1] = u.y;  m[2][1] = u.z;
    m[0][2] = -f.x; m[1][2] = -f.y; m[2][2] = -f.z;
    m[3][0] = -glm::dot(s, eye);
    m[3][1] = -glm::dot(u, eye);
    m[3][2] = glm::dot(f, eye);

    return m;
}

CameraMatrices getDefaultMatrices(float aspect) {
    CameraMatrices cam;
    cam.model = glm::mat4(1.0f); // identity
    cam.view = lookAt(DEFAULT_EYE, DEFAULT_TARGET, DEFAULT_UP);
    cam.projection = perspective(
        glm::radians(DEFAULT_FOV), aspect, DEFAULT_NEAR, DEFAULT_FAR);
    return cam;
}

RTCameraData getRTCameraData(float aspect) {
    RTCameraData data;
    glm::mat4 view = lookAt(DEFAULT_EYE, DEFAULT_TARGET, DEFAULT_UP);
    glm::mat4 proj = perspective(
        glm::radians(DEFAULT_FOV), aspect, DEFAULT_NEAR, DEFAULT_FAR);
    data.inverseView = glm::inverse(view);
    data.inverseProjection = glm::inverse(proj);
    return data;
}

} // namespace Camera
