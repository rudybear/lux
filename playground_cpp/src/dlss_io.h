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
// `metalYConvention`: false (default, Vulkan) assumes the engine's NDC->pixel
// mapping is the direct `pixel = (ndc*0.5+0.5)*screen_size`. Metal's splat
// renderer instead applies an EXTRA Y flip outside the projection matrix
// (`screen.y = (1-(ndc.y*0.5+0.5))*H`, see metal_splat_renderer.cpp), which
// negates the sign of the y-row's coefficients relative to the Vulkan
// derivation above -- pass true for MetalSplatRenderer::updateCameraExplicit.
glm::mat4 buildIntrinsicsProjection(float fx, float fy, float cx, float cy,
                                     float width, float height,
                                     float nearPlane, float farPlane,
                                     bool metalYConvention = false);

// Writes a float32 .npy file, row-major, little-endian, with the given
// shape (e.g. {height, width, channels} or {height, width}). ~20 lines,
// no new dependencies (docs/lux-4d-spec.md section 3).
void writeNpyFloat32(const std::string& path, const std::vector<float>& data,
                      const std::vector<int64_t>& shape);

// Reads a float32 .npy file written by numpy (`<f4`, C order, version 1.0
// or 2.0 header). Throws std::runtime_error on any other dtype/order, or
// if the file doesn't exist. docs/lux-reconstruct-spec.md's dump loader.
struct NpyArray {
    std::vector<float> data;
    std::vector<int64_t> shape;
};
NpyArray readNpyFloat32(const std::string& path);

// Minimal flat-JSON integer field reader (same "find the key, strtol what
// follows the colon" approach as loadCameraJson's own scalarAfterKey) --
// used to read a --reconstruct-dump directory's meta.json (s, k,
// param_stride, hidden, proxy_w/h, target_w/h, net_w/h, num_frames).
// Returns -1 if the key isn't found.
long readJsonIntField(const std::string& jsonPath, const std::string& key);

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

// IEEE-754 binary16 -> float32. Portable (no _Float16/F16C dependency);
// used to convert the splat color attachment's raw bytes (now
// RGBA16_SFLOAT/RGBA16Float, see docs/lux-4d-spec.md section 3's PSNR
// follow-up) back to float32 on the host, both for PNG saving (round, not
// truncate, to 8-bit) and for the `--output-aux`-only `_color.npy` dump.
float halfToFloat(uint16_t h);

// Rounds a linear [0,1] float to an 8-bit UNORM byte: clamp then
// round-half-away-from-zero (NOT truncate) -- e.g. 0.999 -> 255, not 254.
uint8_t floatToUnorm8Rounded(float x);

// Converts a raw RGBA16_SFLOAT/RGBA16Float color attachment readback
// (8 bytes/pixel: 4x uint16 half-floats, premultiplied alpha per
// docs/lux-4d-spec.md section 3's fragment output convention) into:
//   - `outRgba8`: an 8-bit RGBA buffer (4 bytes/pixel) for PNG saving,
//     alpha un-premultiplied out of rgb first (so the saved PNG shows
//     true un-premultiplied color), each channel rounded (not truncated).
//   - the returned float32 [H,W,4] un-premultiplied RGBA buffer (for
//     `_color.npy`; alpha itself is NOT premultiplied, so this channel is
//     the plain accumulated alpha).
std::vector<float> convertRgba16fColorAttachment(const std::vector<uint8_t>& rawHalfBytes,
                                                  uint32_t width, uint32_t height,
                                                  std::vector<uint8_t>& outRgba8);

} // namespace DlssIO
