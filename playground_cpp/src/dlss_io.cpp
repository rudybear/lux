#include "dlss_io.h"
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

long readJsonIntField(const std::string& jsonPath, const std::string& key) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) return -1;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return scalarAfterKey(content, key);
}

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

namespace {

// Shared by readNpyFloat32 (file) and readNpzMemberFloat32 (in-memory,
// extracted from one stored .npz entry) -- both are the identical .npy
// binary format, just from a different backing stream.
NpyArray parseNpyStream(std::istream& f, const std::string& debugName) {
    char magic[6];
    f.read(magic, 6);
    if (f.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("parseNpyStream: bad magic in " + debugName);
    }
    uint8_t version[2];
    f.read(reinterpret_cast<char*>(version), 2);
    uint32_t headerLen;
    if (version[0] == 1) {
        uint16_t h16;
        f.read(reinterpret_cast<char*>(&h16), 2);
        headerLen = h16;
    } else {
        f.read(reinterpret_cast<char*>(&headerLen), 4);
    }
    std::string header(headerLen, '\0');
    f.read(header.data(), static_cast<std::streamsize>(headerLen));

    if (header.find("'<f4'") == std::string::npos) {
        throw std::runtime_error("parseNpyStream: expected dtype '<f4' in " + debugName + " header: " + header);
    }
    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("parseNpyStream: fortran-order arrays not supported: " + debugName);
    }

    // Parse the 'shape': (a, b, c) tuple -- ints and commas only, so a
    // simple scan suffices (no general Python-literal parser needed).
    NpyArray out;
    size_t shapeKey = header.find("'shape':");
    if (shapeKey == std::string::npos) {
        throw std::runtime_error("parseNpyStream: no 'shape' key in " + debugName);
    }
    size_t open = header.find('(', shapeKey);
    size_t close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("parseNpyStream: malformed shape tuple in " + debugName);
    }
    std::string shapeBody = header.substr(open + 1, close - open - 1);
    std::string cur;
    for (char c : shapeBody) {
        if (c == ',') {
            if (!cur.empty()) out.shape.push_back(std::stoll(cur));
            cur.clear();
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.shape.push_back(std::stoll(cur));

    int64_t total = 1;
    for (int64_t d : out.shape) total *= d;
    out.data.resize(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(total * static_cast<int64_t>(sizeof(float))));
    if (f.gcount() != static_cast<std::streamsize>(total * static_cast<int64_t>(sizeof(float)))) {
        throw std::runtime_error("parseNpyStream: short read on " + debugName);
    }
    return out;
}

uint16_t readLE16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t readLE32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t readLE64(const uint8_t* p) {
    uint64_t lo = readLE32(p);
    uint64_t hi = readLE32(p + 4);
    return lo | (hi << 32);
}

// `np.savez` (numpy >= ~1.22 or so) writes each member by streaming
// directly into the zip (it doesn't know the compressed/uncompressed size
// until the write completes), so Python's zipfile always escapes both
// 32-bit size fields in the LOCAL file header to 0xFFFFFFFF and puts the
// real 64-bit sizes in a Zip64 extended-information extra field (tag
// 0x0001) instead -- even for small, well under 4GB, arrays. `uncomp32`/
// `comp32` are the local header's raw (possibly-escaped) 32-bit fields;
// returns the resolved (uncompressed size, compressed size) pair.
std::pair<uint64_t, uint64_t> resolveZip64Sizes(uint32_t uncomp32, uint32_t comp32,
                                                 const std::vector<uint8_t>& extra) {
    if (uncomp32 != 0xFFFFFFFFu && comp32 != 0xFFFFFFFFu) {
        return {uncomp32, comp32};
    }
    size_t pos = 0;
    while (pos + 4 <= extra.size()) {
        uint16_t tag = readLE16(&extra[pos]);
        uint16_t size = readLE16(&extra[pos + 2]);
        if (tag == 0x0001 && pos + 4 + size <= extra.size()) {
            // Zip64 extra field data order (only escaped fields are
            // present, in this fixed order): original (uncompressed)
            // size, compressed size, relative header offset, disk start
            // number. Both size fields are escaped together here.
            size_t off = pos + 4;
            uint64_t uncomp = uncomp32, comp = comp32;
            if (uncomp32 == 0xFFFFFFFFu && off + 8 <= extra.size()) { uncomp = readLE64(&extra[off]); off += 8; }
            if (comp32 == 0xFFFFFFFFu && off + 8 <= extra.size()) { comp = readLE64(&extra[off]); off += 8; }
            return {uncomp, comp};
        }
        pos += 4 + size;
    }
    throw std::runtime_error("resolveZip64Sizes: escaped size but no Zip64 extra field found");
}

} // namespace

NpyArray readNpyFloat32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("readNpyFloat32: failed to open " + path);
    }
    return parseNpyStream(f, path);
}

NpyArray readNpzMemberFloat32(const std::string& path, const std::string& member) {
    // np.savez(path, name=array, ...) with no explicit compression uses
    // zipfile.ZIP_STORED (uncompressed) by default, so this only needs to
    // walk ZIP local file headers looking for `member + ".npy"` (numpy's
    // own per-array naming convention inside the archive) -- no deflate,
    // no central-directory parsing needed.
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("readNpzMemberFloat32: failed to open " + path);
    }
    const std::string wantName = member + ".npy";
    while (true) {
        uint8_t sigBytes[4];
        f.read(reinterpret_cast<char*>(sigBytes), 4);
        if (f.gcount() != 4) break;
        uint32_t sig = readLE32(sigBytes);
        if (sig != 0x04034b50u) break;  // not a local file header -> central directory / EOCD reached

        uint8_t hdr[26];
        f.read(reinterpret_cast<char*>(hdr), 26);
        if (f.gcount() != 26) {
            throw std::runtime_error("readNpzMemberFloat32: truncated local file header in " + path);
        }
        uint16_t compMethod = readLE16(hdr + 4);
        uint32_t compSize32 = readLE32(hdr + 14);
        uint32_t uncompSize32 = readLE32(hdr + 18);
        uint16_t nameLen = readLE16(hdr + 22);
        uint16_t extraLen = readLE16(hdr + 24);

        std::string name(nameLen, '\0');
        f.read(name.data(), nameLen);
        std::vector<uint8_t> extra(extraLen);
        if (extraLen > 0) f.read(reinterpret_cast<char*>(extra.data()), extraLen);

        auto [uncompSize64, compSize64] = resolveZip64Sizes(uncompSize32, compSize32, extra);
        (void)uncompSize64;

        if (name == wantName) {
            if (compMethod != 0) {
                throw std::runtime_error("readNpzMemberFloat32: member '" + name +
                                          "' is compressed (only ZIP_STORED is supported) in " + path);
            }
            std::vector<char> buf(static_cast<size_t>(compSize64));
            f.read(buf.data(), static_cast<std::streamsize>(compSize64));
            if (static_cast<uint64_t>(f.gcount()) != compSize64) {
                throw std::runtime_error("readNpzMemberFloat32: short read on member '" + name + "' in " + path);
            }
            std::string membuf(buf.begin(), buf.end());
            std::istringstream iss(membuf, std::ios::binary);
            return parseNpyStream(iss, path + ":" + name);
        }
        f.seekg(static_cast<std::streamoff>(compSize64), std::ios::cur);
    }
    throw std::runtime_error("readNpzMemberFloat32: member '" + wantName + "' not found in " + path);
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

float halfToFloat(uint16_t h) {
    // Standard IEEE-754 binary16 -> binary32 bit-twiddling expansion.
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp16 = (h >> 10) & 0x1Fu;
    uint32_t mant16 = h & 0x3FFu;
    uint32_t bits;
    if (exp16 == 0) {
        if (mant16 == 0) {
            bits = sign;  // +-0
        } else {
            // Subnormal half -> normalize into a float32.
            int e = -1;
            uint32_t m = mant16;
            do { m <<= 1; e++; } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            uint32_t exp32 = static_cast<uint32_t>(127 - 15 - e);
            bits = sign | (exp32 << 23) | (m << 13);
        }
    } else if (exp16 == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant16 << 13);  // Inf/NaN
    } else {
        uint32_t exp32 = exp16 - 15 + 127;
        bits = sign | (exp32 << 23) | (mant16 << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint8_t floatToUnorm8Rounded(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lround(x * 255.0f));
}

std::vector<float> convertRgba16fColorAttachment(const std::vector<uint8_t>& rawHalfBytes,
                                                  uint32_t width, uint32_t height,
                                                  std::vector<uint8_t>& outRgba8) {
    size_t numPixels = static_cast<size_t>(width) * height;
    std::vector<float> out(numPixels * 4);
    outRgba8.assign(numPixels * 4, 0);
    const uint16_t* half = reinterpret_cast<const uint16_t*>(rawHalfBytes.data());
    for (size_t i = 0; i < numPixels; ++i) {
        float rPre = halfToFloat(half[i * 4 + 0]);
        float gPre = halfToFloat(half[i * 4 + 1]);
        float bPre = halfToFloat(half[i * 4 + 2]);
        float a = halfToFloat(half[i * 4 + 3]);

        float r, g, b;
        if (a > 1e-6f) {
            r = rPre / a; g = gPre / a; b = bPre / a;
        } else {
            r = g = b = 0.0f;
        }
        out[i * 4 + 0] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = a;

        outRgba8[i * 4 + 0] = floatToUnorm8Rounded(r);
        outRgba8[i * 4 + 1] = floatToUnorm8Rounded(g);
        outRgba8[i * 4 + 2] = floatToUnorm8Rounded(b);
        outRgba8[i * 4 + 3] = floatToUnorm8Rounded(a);
    }
    return out;
}

} // namespace DlssIO
