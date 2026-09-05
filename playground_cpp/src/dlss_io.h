#pragma once

// Shared (Vulkan- and Metal-independent) host-side helpers for the DLSS
// input-contract outputs (docs/lux-4d-spec.md sections 3-4):
//   - --camera-json <file>: OpenCV camera bridge (world->camera viewmat +
//     pinhole intrinsics K) -> a lux (glTF/OpenGL-convention) view + custom
//     off-axis projection matrix that reproduces the OpenCV pinhole exactly.
//   - --output-aux <prefix>: <prefix>_color.png / _depth.npy / _mv.npy (+
//     normalized preview PNGs), a minimal dependency-free .npy writer.
//
// GPU image readback (VkImage/MTLTexture -> CPU float array) is backend-
// specific and lives in main.cpp / metal_main.cpp; this file only handles
// the backend-agnostic math and file I/O once the data is on the CPU.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace DlssIO {

struct CameraJsonData {
    std::array<float, 16> viewmatCv;  // row-major, world -> camera, OpenCV convention
    std::array<float, 9> K;           // row-major 3x3 pinhole intrinsics
    uint32_t width = 0;
    uint32_t height = 0;
};

// Parses `{"viewmat_cv": [...16 floats...], "K": [...9 floats...],
// "width": W, "height": H}`. Returns false (and leaves `out` untouched) if
// the file is missing or doesn't contain both arrays with the right length.
// Deliberately NOT a general JSON parser (this codebase avoids pulling in
// nlohmann::json etc. -- see readShaderShDegree/readShaderBoolFlag in
// splat_renderer.cpp for the same convention): just extracts the numeric
// arrays following the two known keys.
bool loadCameraJson(const std::string& path, CameraJsonData& out);

// viewmat_gl = diag(1,-1,-1,1) * viewmat_cv (see docs/lux-4d-spec.md
// section 4): converts an OpenCV-convention (+x right, +y down, +z forward)
// world->camera matrix into lux's glTF/OpenGL convention (+y up, -z
// forward). Row-major input -> GLM column-major mat4.
glm::mat4 cvViewToGl(const std::array<float, 16>& viewmatCvRowMajor);

// Builds an off-axis (arbitrary principal point) Vulkan-clip-space
// projection matrix directly from OpenCV pinhole intrinsics (fx, fy, cx,
// cy in pixels) and the image size, such that lux's existing NDC->pixel
// mapping (`pixel = (ndc*0.5+0.5)*screen_size`, pixel-center convention
// (i+0.5, j+0.5)) reproduces `u = fx*x/z + cx`, `v = fy*y/z + cy` exactly
// for camera-space points converted from lux's GL convention. See
// docs/lux-4d-spec.md section 4 for the derivation. `nearPlane`/`farPlane`
// only affect the depth-test Z-buffer, not pixel placement.
glm::mat4 buildIntrinsicsProjection(float fx, float fy, float cx, float cy,
                                     float width, float height,
                                     float nearPlane, float farPlane);

// Writes a float32 .npy file, row-major, little-endian, with the given
// shape (e.g. {height, width, channels} or {height, width}). ~20 lines,
// no new dependencies (docs/lux-4d-spec.md section 3).
void writeNpyFloat32(const std::string& path, const std::vector<float>& data,
                      const std::vector<int64_t>& shape);

// Writes a normalized grayscale PNG preview of a float buffer with
// `channels` interleaved components per pixel (1 = raw value, e.g. depth;
// 2 = magnitude, e.g. motion vectors). Min/max normalized over the whole
// image (0 always maps to black); NaN/Inf-free by construction (the
// upstream premultiplied-alpha readback below already handles alpha==0).
void writeNormalizedPreviewPNG(const std::string& path, const std::vector<float>& data,
                                uint32_t width, uint32_t height, int channels);

// Un-premultiplies an RGBA-blended auxiliary attachment (motion or depth,
// both written as value*alpha by the fragment shader, see
// docs/lux-4d-spec.md section 3) by the alpha channel read back from the
// color attachment, producing the final per-pixel value (0 where alpha==0,
// i.e. nothing was drawn). `channels` is 1 (depth) or 2 (motion).
std::vector<float> unpremultiplyByAlpha(const std::vector<float>& rawAux,
                                         const std::vector<float>& alphaChannel,
                                         uint32_t width, uint32_t height, int channels);

} // namespace DlssIO
