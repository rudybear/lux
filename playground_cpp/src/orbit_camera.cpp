#include "orbit_camera.h"
#include "dlss_io.h"

#include <array>
#include <cmath>

namespace LiveOrbitCamera {

Result compute(int frame, uint32_t width, uint32_t height, bool metalYConvention) {
    Result out;
    const float elRad = glm::radians(kElevationDeg);
    const float az = glm::radians(kDegPerFrame * static_cast<float>(frame));
    const glm::vec3 eye = kFgCenter + kOrbitRadius * glm::vec3(cosf(elRad) * cosf(az),
                                                                -sinf(elRad),
                                                                cosf(elRad) * sinf(az));
    // PanopticSports is a y-down world -- world "up" for the look_at basis is (0,-1,0).
    const glm::vec3 up(0.0f, -1.0f, 0.0f);
    const glm::vec3 f = glm::normalize(kFgCenter - eye);
    const glm::vec3 r = glm::normalize(glm::cross(-up, f));
    const glm::vec3 u = glm::cross(f, r);

    std::array<float, 16> viewCv = {
        r.x, r.y, r.z, -glm::dot(r, eye),
        u.x, u.y, u.z, -glm::dot(u, eye),
        f.x, f.y, f.z, -glm::dot(f, eye),
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    const glm::mat4 viewGl = DlssIO::cvViewToGl(viewCv);

    const float fy = 0.5f * static_cast<float>(height) / tanf(glm::radians(kFovYDeg) * 0.5f);
    const float fx = fy;
    const float cx = 0.5f * static_cast<float>(width);
    const float cy = 0.5f * static_cast<float>(height);
    const glm::mat4 proj = DlssIO::buildIntrinsicsProjection(
        fx, fy, cx, cy, static_cast<float>(width), static_cast<float>(height),
        0.01f, 1000.0f, metalYConvention);

    out.eye = eye;
    out.r = r; out.u = u; out.f = f;
    out.viewGl = viewGl;
    out.proj = proj;
    out.fx = fx;
    out.fy = fy;
    return out;
}

float halton(int index, int base) {
    float f = 1.0f, res = 0.0f;
    int i = index;
    while (i > 0) {
        f /= static_cast<float>(base);
        res += f * static_cast<float>(i % base);
        i /= base;
    }
    return res;
}

void taaJitterTargetPx(int frame, int period, float& jxOut, float& jyOut) {
    const int i = (frame % period) + 1;
    jxOut = halton(i, 2) - 0.5f;
    jyOut = halton(i, 3) - 0.5f;
}

}  // namespace LiveOrbitCamera
