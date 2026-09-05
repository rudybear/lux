#include "dlss_io.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace DlssIO {

namespace {

std::vector<double> extractNumbers(const std::string& s) {
    std::vector<double> out;
    const char* p = s.c_str();
    while (*p) {
        // Skip to the start of the next number.
        while (*p && !(std::isdigit(static_cast<unsigned char>(*p)) || *p == '-' || *p == '+' || *p == '.')) {
            p++;
        }
        if (!*p) break;
        char* end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) { p++; continue; }
        out.push_back(v);
        p = end;
    }
    return out;
}

std::vector<double> extractArrayAfterKey(const std::string& content, const std::string& key) {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return {};
    auto bracketStart = content.find('[', pos);
    if (bracketStart == std::string::npos) return {};
    auto bracketEnd = content.find(']', bracketStart);
    if (bracketEnd == std::string::npos) return {};
    return extractNumbers(content.substr(bracketStart, bracketEnd - bracketStart));
}

long scalarAfterKey(const std::string& content, const std::string& key) {
    auto pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    auto colon = content.find(':', pos);
    if (colon == std::string::npos) return -1;
    return std::strtol(content.c_str() + colon + 1, nullptr, 10);
}

} // namespace

bool loadCameraJson(const std::string& path, CameraJsonData& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto vm = extractArrayAfterKey(content, "viewmat_cv");
    auto k = extractArrayAfterKey(content, "K");
    if (vm.size() != 16 || k.size() != 9) return false;

    for (int i = 0; i < 16; ++i) out.viewmatCv[i] = static_cast<float>(vm[i]);
    for (int i = 0; i < 9; ++i) out.K[i] = static_cast<float>(k[i]);

    long w = scalarAfterKey(content, "width");
    long h = scalarAfterKey(content, "height");
    if (w <= 0 || h <= 0) return false;
    out.width = static_cast<uint32_t>(w);
    out.height = static_cast<uint32_t>(h);
    return true;
}

glm::mat4 cvViewToGl(const std::array<float, 16>& m) {
    // Row-major input: m[r*4+c]. GLM mat4 storage is column-major: M[c][r].
    glm::mat4 cv(1.0f);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            cv[c][r] = m[r * 4 + c];

    // viewmat_gl = diag(1,-1,-1,1) * viewmat_cv negates rows 1 and 2.
    glm::mat4 gl = cv;
    for (int c = 0; c < 4; ++c) {
        gl[c][1] = -cv[c][1];
        gl[c][2] = -cv[c][2];
    }
    return gl;
}

glm::mat4 buildIntrinsicsProjection(float fx, float fy, float cx, float cy,
                                     float width, float height,
                                     float nearPlane, float farPlane,
                                     bool metalYConvention) {
    // Derivation (docs/lux-4d-spec.md section 4): the Vulkan splat
    // pipeline's vertex stage maps NDC -> pixel via
    // `pixel = (ndc*0.5 + 0.5) * screen_size`. Requiring this to reproduce
    // the OpenCV pinhole `u = fx*x_cv/z_cv + cx`, `v = fy*y_cv/z_cv + cy`
    // for camera-space points converted from lux's GL convention
    // (x_gl=x_cv, y_gl=-y_cv, z_gl=-z_cv, so x_cv/z_cv = x_gl/z_gl and
    // y_cv/z_cv = y_gl/(-z_gl)... using z_cv = -z_gl) and clip.w = -z_gl
    // (the usual convention) gives, after solving clip_x = ndc_x * clip.w
    // and clip_y = ndc_y * clip.w for the matrix coefficients:
    //   clip.x = (2*fx/W)*x_gl + (1 - 2*cx/W)*z_gl
    //   clip.y =      -(2*fy/H)*y_gl + (1 - 2*cy/H)*z_gl
    //   clip.w = -z_gl
    // Metal's splat renderer instead maps
    // `screen.y = (1-(ndc.y*0.5+0.5))*H` (an EXTRA flip outside the
    // projection matrix, see metal_splat_renderer.cpp's `center`/`ndc`),
    // which negates the y-row's sign in the same derivation:
    //   clip.y = (2*fy/H)*y_gl - (1 - 2*cy/H)*z_gl
    // The z row (depth-test only, doesn't affect pixel placement) is
    // borrowed from a standard glm::perspective matrix with the same
    // near/far planes, since it already matches this codebase's Vulkan
    // depth convention everywhere else.
    glm::mat4 ref = glm::perspective(glm::radians(60.0f), width / height, nearPlane, farPlane);

    glm::mat4 P(0.0f);
    P[0][0] = 2.0f * fx / width;
    P[2][0] = 1.0f - 2.0f * cx / width;
    if (metalYConvention) {
        P[1][1] = 2.0f * fy / height;
        P[2][1] = -(1.0f - 2.0f * cy / height);
    } else {
        P[1][1] = -2.0f * fy / height;
        P[2][1] = 1.0f - 2.0f * cy / height;
    }
    P[2][2] = ref[2][2];
    P[2][3] = -1.0f;
    P[3][2] = ref[3][2];
    return P;
}

void writeNpyFloat32(const std::string& path, const std::vector<float>& data,
                      const std::vector<int64_t>& shape) {
    // Minimal NPY v1.0 writer: magic + version + header length (u16 LE) +
    // header string (padded so the total prefix is a multiple of 64 bytes,
    // ending in '\n') + raw little-endian float32 data. See
    // https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    std::ostringstream shapeStr;
    shapeStr << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shapeStr << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) shapeStr << ", ";
    }
    shapeStr << ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeStr.str() + ", }";
    // Total prefix length (10-byte fixed preamble + header + \n) must be a
    // multiple of 64; pad the header with spaces before the final newline.
    size_t preambleLen = 10;
    size_t unpadded = preambleLen + header.size() + 1;
    size_t padded = ((unpadded + 63) / 64) * 64;
    size_t padLen = padded - unpadded;
    header.append(padLen, ' ');
    header.push_back('\n');

    std::ofstream f(path, std::ios::binary);
    f.write("\x93NUMPY", 6);
    uint8_t version[2] = {1, 0};
    f.write(reinterpret_cast<const char*>(version), 2);
    uint16_t headerLen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&headerLen), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
}

void writeNormalizedPreviewPNG(const std::string& path, const std::vector<float>& data,
                                uint32_t width, uint32_t height, int channels) {
    size_t numPixels = static_cast<size_t>(width) * height;
    std::vector<float> magnitude(numPixels, 0.0f);
    for (size_t i = 0; i < numPixels; ++i) {
        float m = 0.0f;
        for (int c = 0; c < channels; ++c) {
            float v = data[i * channels + c];
            m += v * v;
        }
        magnitude[i] = std::sqrt(m);
    }
    float lo = *std::min_element(magnitude.begin(), magnitude.end());
    float hi = *std::max_element(magnitude.begin(), magnitude.end());
    float range = (hi - lo) > 1e-8f ? (hi - lo) : 1.0f;

    std::vector<uint8_t> gray(numPixels);
    for (size_t i = 0; i < numPixels; ++i) {
        float t = (magnitude[i] - lo) / range;
        t = std::clamp(t, 0.0f, 1.0f);
        gray[i] = static_cast<uint8_t>(t * 255.0f + 0.5f);
    }
    stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 1,
                    gray.data(), static_cast<int>(width));
}

std::vector<float> unpremultiplyByAlpha(const std::vector<float>& rawAux,
                                         const std::vector<float>& alphaChannel,
                                         uint32_t width, uint32_t height, int channels) {
    size_t numPixels = static_cast<size_t>(width) * height;
    std::vector<float> out(rawAux.size());
    for (size_t i = 0; i < numPixels; ++i) {
        float a = alphaChannel[i];
        for (int c = 0; c < channels; ++c) {
            out[i * channels + c] = (a > 1e-6f) ? (rawAux[i * channels + c] / a) : 0.0f;
        }
    }
    return out;
}

} // namespace DlssIO
