#include "gltf_loader.h"

#include <algorithm>
#include <functional>
#include <fstream>
#include <map>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cctype>
#include <iostream>

// cgltf — single-header glTF parser
// Define the implementation once in this translation unit.
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

// stb_image — single-header image decoder for PNG/JPEG from GLB buffers
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"

// ===========================================================================
// Helper: extract texture image data from GLB buffer views via stb_image
// ===========================================================================

static GltfTextureData extractTextureData(const cgltf_texture_view& texView) {
    GltfTextureData result;
    if (!texView.texture || !texView.texture->image) return result;

    const cgltf_image* image = texView.texture->image;

    // Case 1: Image data is embedded in a buffer view (typical for .glb)
    if (image->buffer_view && image->buffer_view->buffer && image->buffer_view->buffer->data) {
        const uint8_t* rawData = static_cast<const uint8_t*>(image->buffer_view->buffer->data)
                                 + image->buffer_view->offset;
        size_t rawSize = image->buffer_view->size;

        int w = 0, h = 0, channels = 0;
        stbi_uc* decoded = stbi_load_from_memory(rawData, static_cast<int>(rawSize),
                                                  &w, &h, &channels, 4); // force RGBA
        if (decoded) {
            result.width = w;
            result.height = h;
            result.pixels.assign(decoded, decoded + w * h * 4);
            stbi_image_free(decoded);
            std::cout << "  [texture] Decoded " << w << "x" << h
                      << " (" << channels << " ch -> RGBA)" << std::endl;
        } else {
            std::cerr << "  [texture] Failed to decode embedded image: "
                      << stbi_failure_reason() << std::endl;
        }
    }
    // Case 2: URI-based external file (for .gltf with separate image files)
    else if (image->uri && strlen(image->uri) > 0) {
        // Not handling data URIs or external files in this implementation;
        // GLB embeds images in buffer views which is the primary use case
        std::cerr << "  [texture] External URI textures not supported: " << image->uri << std::endl;
    }

    return result;
}

// ===========================================================================
// Helper: read accessor data
// ===========================================================================

static std::vector<float> readFloatAccessor(const cgltf_accessor* accessor) {
    std::vector<float> result;
    if (!accessor) return result;

    size_t components = cgltf_num_components(accessor->type);
    result.resize(accessor->count * components);
    cgltf_accessor_unpack_floats(accessor, result.data(), result.size());
    return result;
}

static std::vector<uint32_t> readIndexAccessor(const cgltf_accessor* accessor) {
    std::vector<uint32_t> result;
    if (!accessor) return result;

    result.resize(accessor->count);
    for (size_t i = 0; i < accessor->count; i++) {
        result[i] = static_cast<uint32_t>(cgltf_accessor_read_index(accessor, i));
    }
    return result;
}

// Read a (possibly sparse) accessor as compact (index, value) pairs, without
// materializing a full dense array. Morph target deltas are typically sparse
// with **no bufferView** (the dense base is implicitly all-zero) -- cgltf
// exposes this directly via accessor->is_sparse / accessor->sparse, so no
// custom JSON scanning is needed here (unlike the KHR extension object,
// which cgltf doesn't understand natively).
static void readSparseAccessor(const cgltf_accessor* accessor,
                                std::vector<uint32_t>& outIndices,
                                std::vector<float>& outValues) {
    outIndices.clear();
    outValues.clear();
    if (!accessor) return;
    size_t components = cgltf_num_components(accessor->type);

    if (!accessor->is_sparse) {
        // Dense target (valid glTF, just less common): every index is "touched".
        outValues = readFloatAccessor(accessor);
        outIndices.resize(accessor->count);
        for (size_t i = 0; i < accessor->count; i++) outIndices[i] = static_cast<uint32_t>(i);
        return;
    }

    const cgltf_accessor_sparse& sparse = accessor->sparse;
    const uint8_t* indexData = cgltf_buffer_view_data(sparse.indices_buffer_view);
    if (!indexData) return;
    indexData += sparse.indices_byte_offset;
    size_t indexStride = cgltf_component_size(sparse.indices_component_type);

    outIndices.resize(sparse.count);
    for (size_t i = 0; i < sparse.count; i++) {
        outIndices[i] = static_cast<uint32_t>(
            cgltf_component_read_index(indexData + i * indexStride, sparse.indices_component_type));
    }

    if (accessor->component_type == cgltf_component_type_r_32f) {
        // Fast path: sparse.values is tightly-packed float data per spec
        // (byteStride is disallowed on a sparse values bufferView).
        const uint8_t* valueData = cgltf_buffer_view_data(sparse.values_buffer_view);
        outValues.resize(sparse.count * components);
        if (valueData) {
            std::memcpy(outValues.data(), valueData + sparse.values_byte_offset,
                        outValues.size() * sizeof(float));
        }
    } else {
        // Fallback for exotic component types: unpack densely then gather.
        auto dense = readFloatAccessor(accessor);
        outValues.resize(sparse.count * components);
        for (size_t i = 0; i < sparse.count; i++) {
            std::memcpy(&outValues[i * components], &dense[outIndices[i] * components],
                        components * sizeof(float));
        }
    }
}

// ===========================================================================
// Minimal best-effort JSON scanning for unprocessed extension blobs
// ===========================================================================
// cgltf surfaces extensions it doesn't natively understand (like
// KHR_gaussian_splatting) as a raw JSON text blob (cgltf_extension::data).
// These helpers pull simple flat string/int fields out of such a blob
// without needing a full JSON parser — sufficient for the KHR_gaussian_splatting
// extension object, whose top-level fields are plain strings/ints/arrays.

// Extract a top-level string value for `"key":"value"` from a raw JSON blob.
static bool jsonExtractString(const std::string& json, const std::string& key, std::string& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    pos = json.find('"', pos);
    if (pos == std::string::npos) return false;
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return false;
    out = json.substr(pos + 1, end - pos - 1);
    return true;
}

// Extract a top-level integer value for `"key":N` from a raw JSON blob.
static bool jsonExtractInt(const std::string& json, const std::string& key, int& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
    size_t start = pos;
    while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) || json[pos] == '-')) pos++;
    if (pos == start) return false;
    out = std::stoi(json.substr(start, pos - start));
    return true;
}

// Extract a top-level float value for `"key":N` (or N.N) from a raw JSON blob.
static bool jsonExtractFloat(const std::string& json, const std::string& key, float& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) pos++;
    size_t start = pos;
    while (pos < json.size() && (std::isdigit(static_cast<unsigned char>(json[pos])) ||
           json[pos] == '-' || json[pos] == '+' || json[pos] == '.' ||
           json[pos] == 'e' || json[pos] == 'E')) pos++;
    if (pos == start) return false;
    out = std::stof(json.substr(start, pos - start));
    return true;
}

// Extract a top-level integer array `"key":[1,2,3]` from a raw JSON blob.
static bool jsonExtractIntArray(const std::string& json, const std::string& key, std::vector<int>& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    size_t arrStart = json.find('[', pos);
    size_t arrEnd = json.find(']', arrStart == std::string::npos ? pos : arrStart);
    if (arrStart == std::string::npos || arrEnd == std::string::npos) return false;
    std::string arrText = json.substr(arrStart + 1, arrEnd - arrStart - 1);
    out.clear();
    size_t tokStart = 0;
    while (tokStart <= arrText.size()) {
        size_t comma = arrText.find(',', tokStart);
        std::string tok = arrText.substr(tokStart, comma == std::string::npos ? std::string::npos : comma - tokStart);
        size_t a = tok.find_first_not_of(" \t\n\r");
        if (a != std::string::npos) {
            size_t b = tok.find_last_not_of(" \t\n\r");
            out.push_back(std::stoi(tok.substr(a, b - a + 1)));
        }
        if (comma == std::string::npos) break;
        tokStart = comma + 1;
    }
    return true;
}

// Find the raw JSON text of a named object value `"key":{...}` from a raw JSON blob.
static bool jsonExtractObject(const std::string& json, const std::string& key, std::string& out) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return false;
    size_t braceStart = json.find('{', pos);
    if (braceStart == std::string::npos) return false;
    int depth = 0;
    size_t i = braceStart;
    for (; i < json.size(); i++) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') {
            depth--;
            if (depth == 0) { i++; break; }
        }
    }
    out = json.substr(braceStart, i - braceStart);
    return true;
}

// Find the raw JSON text of `data->meshes[meshIndex]`'s `"extras"` object by
// manually scanning cgltf's retained raw JSON buffer (data->json/json_size).
// Needed because this vendored cgltf.h calls cgltf_parse_json_extras() for
// PRIMITIVE-level extras but never for the owning cgltf_mesh object itself,
// so `mesh.extras.data` is always null even when a writer (e.g. the ratified
// MOBILEDLSS_dynamic_splats convention) puts extras on the mesh.
static bool jsonMeshExtras(const cgltf_data* data, size_t meshIndex, std::string& out) {
    if (!data->json || data->json_size == 0) return false;
    std::string json(data->json, data->json_size);

    size_t pos = json.find("\"meshes\"");
    if (pos == std::string::npos) return false;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return false;

    // Walk balanced brace/bracket depth from just past the array's '[':
    // depth==0 means "directly inside the meshes array" -- a '{' seen there
    // starts the meshIndex-th element, and the matching '}' (depth back to 0)
    // ends it.
    int depth = 0;
    size_t objStart = std::string::npos;
    size_t curIndex = 0;
    for (size_t i = pos + 1; i < json.size(); i++) {
        char c = json[i];
        if (c == '{' || c == '[') {
            if (depth == 0 && c == '{') objStart = i;
            depth++;
        } else if (c == '}' || c == ']') {
            depth--;
            if (depth == 0) {
                if (c == '}' && objStart != std::string::npos) {
                    if (curIndex == meshIndex) {
                        std::string meshObj = json.substr(objStart, i - objStart + 1);
                        return jsonExtractObject(meshObj, "extras", out);
                    }
                    curIndex++;
                    objStart = std::string::npos;
                } else if (c == ']') {
                    return false;  // end of the meshes array, index not found
                }
            }
        }
    }
    return false;
}

// Find the raw JSON text of a named unprocessed extension on a primitive, if any.
static const char* findExtensionData(const cgltf_primitive& prim, const char* name) {
    for (size_t i = 0; i < prim.extensions_count; i++) {
        if (prim.extensions[i].name && std::strcmp(prim.extensions[i].name, name) == 0) {
            return prim.extensions[i].data;
        }
    }
    return nullptr;
}

// ===========================================================================
// SH coefficient rotation for node transforms
// ===========================================================================

// Evaluate SH basis functions for a single degree L at direction d.
// Constants match splat_expander.py exactly (Condon-Shortley phase convention).
// out: array of (2*L+1) values
static void evalSHBasisDegree(const glm::vec3& d, int L, float* out) {
    float x = d.x, y = d.y, z = d.z;
    switch (L) {
    case 1: {
        const float C1 = 0.48860251f;
        out[0] = -C1 * y;   // sh1: Y_1^{-1}
        out[1] =  C1 * z;   // sh2: Y_1^0
        out[2] = -C1 * x;   // sh3: Y_1^1
        break;
    }
    case 2: {
        float x2 = x*x, y2 = y*y, z2 = z*z;
        out[0] =  1.09254843f * x * y;                    // sh4
        out[1] = -1.09254843f * y * z;                    // sh5
        out[2] =  0.31539157f * (2.0f*z2 - x2 - y2);     // sh6
        out[3] = -1.09254843f * x * z;                    // sh7
        out[4] =  0.54627422f * (x2 - y2);                // sh8
        break;
    }
    case 3: {
        float x2 = x*x, y2 = y*y, z2 = z*z;
        out[0] = -0.59004359f * y * (3.0f*x2 - y2);              // sh9
        out[1] =  2.89061144f * x * y * z;                       // sh10
        out[2] = -0.45704580f * y * (4.0f*z2 - x2 - y2);        // sh11
        out[3] =  0.37317633f * z * (2.0f*z2 - 3.0f*x2 - 3.0f*y2); // sh12
        out[4] = -0.45704580f * x * (4.0f*z2 - x2 - y2);        // sh13
        out[5] =  1.44530572f * z * (x2 - y2);                   // sh14
        out[6] = -0.59004359f * x * (x2 - 3.0f*y2);             // sh15
        break;
    }
    }
}

// Solve NxN linear system via Gaussian elimination with partial pivoting.
// aug: N x (N+1) augmented matrix [A | b], overwritten in-place.
// x: output solution vector (N elements).
static bool solveLinearSystem(float* aug, int N, float* x) {
    for (int col = 0; col < N; col++) {
        int maxRow = col;
        float maxVal = std::abs(aug[col * (N+1) + col]);
        for (int row = col + 1; row < N; row++) {
            float val = std::abs(aug[row * (N+1) + col]);
            if (val > maxVal) { maxVal = val; maxRow = row; }
        }
        if (maxVal < 1e-12f) return false;
        if (maxRow != col) {
            for (int j = col; j <= N; j++)
                std::swap(aug[col * (N+1) + j], aug[maxRow * (N+1) + j]);
        }
        float pivot = aug[col * (N+1) + col];
        for (int row = col + 1; row < N; row++) {
            float factor = aug[row * (N+1) + col] / pivot;
            for (int j = col; j <= N; j++)
                aug[row * (N+1) + j] -= factor * aug[col * (N+1) + j];
        }
    }
    for (int row = N - 1; row >= 0; row--) {
        x[row] = aug[row * (N+1) + N];
        for (int j = row + 1; j < N; j++)
            x[row] -= aug[row * (N+1) + j] * x[j];
        x[row] /= aug[row * (N+1) + row];
    }
    return true;
}

// Compute the (2L+1)x(2L+1) SH rotation matrix for degree L given rotation R.
// The matrix M satisfies: c' = M * c, where evaluating c' with world-space
// directions gives the same result as evaluating c with local-space directions.
static void computeSHRotMatrix(int L, const glm::mat3& R, float* M) {
    int N = 2 * L + 1;

    // Well-distributed test directions on the unit sphere
    const glm::vec3 rawDirs[] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };
    glm::vec3 testDirs[7];
    for (int i = 0; i < 7; i++) testDirs[i] = glm::normalize(rawDirs[i]);

    // Build A[i][j] = B_j(d_i) and A_rot[i][j] = B_j(R^T * d_i)
    std::vector<float> A(N * N), A_rot(N * N);
    float basis[7], basis_rot[7];
    glm::mat3 Rt = glm::transpose(R);

    for (int i = 0; i < N; i++) {
        evalSHBasisDegree(testDirs[i], L, basis);
        evalSHBasisDegree(Rt * testDirs[i], L, basis_rot);
        for (int j = 0; j < N; j++) {
            A[i * N + j] = basis[j];
            A_rot[i * N + j] = basis_rot[j];
        }
    }

    // Solve A * M = A_rot column by column
    std::vector<float> aug(N * (N + 1));
    float x[7];
    for (int col = 0; col < N; col++) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++)
                aug[i * (N+1) + k] = A[i * N + k];
            aug[i * (N+1) + N] = A_rot[i * N + col];
        }
        solveLinearSystem(aug.data(), N, x);
        for (int i = 0; i < N; i++)
            M[i * N + col] = x[i];
    }
}

// Rotate SH coefficients for a range of splats, for one degree.
// startCoeff: first coefficient index (1 for L=1, 4 for L=2, 9 for L=3)
static void rotateSHDegree(
    const glm::mat3& R,
    std::vector<std::vector<float>>& sh_coefficients,
    int startCoeff, int degree,
    size_t startSplat, size_t splatCount)
{
    int N = 2 * degree + 1;
    float M[49]; // max 7x7 for degree 3
    computeSHRotMatrix(degree, R, M);

    for (size_t si = startSplat; si < startSplat + splatCount; si++) {
        for (int ch = 0; ch < 3; ch++) {
            float old_c[7];
            for (int m = 0; m < N; m++) {
                int idx = startCoeff + m;
                if (idx < static_cast<int>(sh_coefficients.size()) &&
                    (si * 3 + ch) < sh_coefficients[idx].size())
                    old_c[m] = sh_coefficients[idx][si * 3 + ch];
                else
                    old_c[m] = 0.0f;
            }
            float new_c[7];
            for (int i = 0; i < N; i++) {
                new_c[i] = 0.0f;
                for (int j = 0; j < N; j++)
                    new_c[i] += M[i * N + j] * old_c[j];
            }
            for (int m = 0; m < N; m++) {
                int idx = startCoeff + m;
                if (idx < static_cast<int>(sh_coefficients.size()) &&
                    (si * 3 + ch) < sh_coefficients[idx].size())
                    sh_coefficients[idx][si * 3 + ch] = new_c[m];
            }
        }
    }
}

// Rotate all SH coefficients for a range of splats (all applicable degrees).
static void rotateSHCoeffs(
    const glm::mat3& R,
    std::vector<std::vector<float>>& sh_coefficients,
    size_t startSplat, size_t splatCount,
    uint32_t maxSHDegree)
{
    // Degree 0 is rotation-invariant (skip)
    if (maxSHDegree >= 1 && sh_coefficients.size() > 3)
        rotateSHDegree(R, sh_coefficients, 1, 1, startSplat, splatCount);
    if (maxSHDegree >= 2 && sh_coefficients.size() > 8)
        rotateSHDegree(R, sh_coefficients, 4, 2, startSplat, splatCount);
    if (maxSHDegree >= 3 && sh_coefficients.size() > 15)
        rotateSHDegree(R, sh_coefficients, 9, 3, startSplat, splatCount);
}

// ===========================================================================
// Node transform
// ===========================================================================

static glm::mat4 nodeTransform(const cgltf_node* node) {
    glm::mat4 m(1.0f);
    if (node->has_matrix) {
        memcpy(&m, node->matrix, sizeof(float) * 16);
        return m;
    }

    glm::mat4 t(1.0f);
    if (node->has_translation) {
        t[3][0] = node->translation[0];
        t[3][1] = node->translation[1];
        t[3][2] = node->translation[2];
    }

    glm::mat4 r(1.0f);
    if (node->has_rotation) {
        float x = node->rotation[0], y = node->rotation[1];
        float z = node->rotation[2], w = node->rotation[3];
        r[0][0] = 1 - 2*(y*y + z*z); r[0][1] = 2*(x*y + z*w);   r[0][2] = 2*(x*z - y*w);
        r[1][0] = 2*(x*y - z*w);     r[1][1] = 1 - 2*(x*x + z*z); r[1][2] = 2*(y*z + x*w);
        r[2][0] = 2*(x*z + y*w);     r[2][1] = 2*(y*z - x*w);     r[2][2] = 1 - 2*(x*x + y*y);
    }

    glm::mat4 s(1.0f);
    if (node->has_scale) {
        s[0][0] = node->scale[0];
        s[1][1] = node->scale[1];
        s[2][2] = node->scale[2];
    }

    return t * r * s;
}

// ===========================================================================
// Loading
// ===========================================================================

GltfScene loadGltf(const std::string& path) {
    cgltf_options options = {};
    cgltf_data* data = nullptr;

    cgltf_result result = cgltf_parse_file(&options, path.c_str(), &data);
    if (result != cgltf_result_success) {
        throw std::runtime_error("Failed to parse glTF: " + path);
    }

    result = cgltf_load_buffers(&options, data, path.c_str());
    if (result != cgltf_result_success) {
        cgltf_free(data);
        throw std::runtime_error("Failed to load glTF buffers: " + path);
    }

    GltfScene scene;

    // --- Materials ---
    for (size_t i = 0; i < data->materials_count; i++) {
        auto& mat = data->materials[i];
        GltfMaterial gmat;
        gmat.name = mat.name ? mat.name : "unnamed";

        if (mat.has_pbr_metallic_roughness) {
            auto& pbr = mat.pbr_metallic_roughness;
            gmat.baseColor = glm::vec4(pbr.base_color_factor[0], pbr.base_color_factor[1],
                                        pbr.base_color_factor[2], pbr.base_color_factor[3]);
            gmat.metallic = pbr.metallic_factor;
            gmat.roughness = pbr.roughness_factor;

            // Extract PBR textures from GLB buffer views
            if (pbr.base_color_texture.texture) {
                std::cout << "[info] Extracting base_color texture for material: " << gmat.name << std::endl;
                gmat.base_color_tex = extractTextureData(pbr.base_color_texture);
                if (pbr.base_color_texture.has_transform) {
                    gmat.base_color_uv_xform.offset = glm::vec2(
                        pbr.base_color_texture.transform.offset[0],
                        pbr.base_color_texture.transform.offset[1]);
                    gmat.base_color_uv_xform.scale = glm::vec2(
                        pbr.base_color_texture.transform.scale[0],
                        pbr.base_color_texture.transform.scale[1]);
                    gmat.base_color_uv_xform.rotation = pbr.base_color_texture.transform.rotation;
                }
            }
            if (pbr.metallic_roughness_texture.texture) {
                std::cout << "[info] Extracting metallic_roughness texture for material: " << gmat.name << std::endl;
                gmat.metallic_roughness_tex = extractTextureData(pbr.metallic_roughness_texture);
                if (pbr.metallic_roughness_texture.has_transform) {
                    gmat.metallic_roughness_uv_xform.offset = glm::vec2(
                        pbr.metallic_roughness_texture.transform.offset[0],
                        pbr.metallic_roughness_texture.transform.offset[1]);
                    gmat.metallic_roughness_uv_xform.scale = glm::vec2(
                        pbr.metallic_roughness_texture.transform.scale[0],
                        pbr.metallic_roughness_texture.transform.scale[1]);
                    gmat.metallic_roughness_uv_xform.rotation = pbr.metallic_roughness_texture.transform.rotation;
                }
            }
        }

        // Normal map
        if (mat.normal_texture.texture) {
            std::cout << "[info] Extracting normal texture for material: " << gmat.name << std::endl;
            gmat.normal_tex = extractTextureData(mat.normal_texture);
            if (mat.normal_texture.has_transform) {
                gmat.normal_uv_xform.offset = glm::vec2(
                    mat.normal_texture.transform.offset[0],
                    mat.normal_texture.transform.offset[1]);
                gmat.normal_uv_xform.scale = glm::vec2(
                    mat.normal_texture.transform.scale[0],
                    mat.normal_texture.transform.scale[1]);
                gmat.normal_uv_xform.rotation = mat.normal_texture.transform.rotation;
            }
        }

        // Occlusion map
        if (mat.occlusion_texture.texture) {
            std::cout << "[info] Extracting occlusion texture for material: " << gmat.name << std::endl;
            gmat.occlusion_tex = extractTextureData(mat.occlusion_texture);
        }

        // Emissive map
        if (mat.emissive_texture.texture) {
            std::cout << "[info] Extracting emissive texture for material: " << gmat.name << std::endl;
            gmat.emissive_tex = extractTextureData(mat.emissive_texture);
        }

        gmat.emissive = glm::vec3(mat.emissive_factor[0], mat.emissive_factor[1], mat.emissive_factor[2]);
        if (mat.alpha_mode == cgltf_alpha_mode_mask) gmat.alphaMode = "MASK";
        else if (mat.alpha_mode == cgltf_alpha_mode_blend) gmat.alphaMode = "BLEND";
        gmat.alphaCutoff = mat.alpha_cutoff;
        gmat.doubleSided = mat.double_sided;

        // --- KHR_materials_* extensions ---
        if (mat.has_clearcoat) {
            gmat.hasClearcoat = true;
            gmat.clearcoatFactor = mat.clearcoat.clearcoat_factor;
            gmat.clearcoatRoughnessFactor = mat.clearcoat.clearcoat_roughness_factor;
            if (mat.clearcoat.clearcoat_texture.texture) {
                std::cout << "[info] Extracting clearcoat texture for material: " << gmat.name << std::endl;
                gmat.clearcoat_tex = extractTextureData(mat.clearcoat.clearcoat_texture);
            }
            if (mat.clearcoat.clearcoat_roughness_texture.texture) {
                std::cout << "[info] Extracting clearcoat roughness texture for material: " << gmat.name << std::endl;
                gmat.clearcoat_roughness_tex = extractTextureData(mat.clearcoat.clearcoat_roughness_texture);
            }
        }
        if (mat.has_sheen) {
            gmat.hasSheen = true;
            gmat.sheenColorFactor = glm::vec3(mat.sheen.sheen_color_factor[0],
                                               mat.sheen.sheen_color_factor[1],
                                               mat.sheen.sheen_color_factor[2]);
            gmat.sheenRoughnessFactor = mat.sheen.sheen_roughness_factor;
            if (mat.sheen.sheen_color_texture.texture) {
                std::cout << "[info] Extracting sheen color texture for material: " << gmat.name << std::endl;
                gmat.sheen_color_tex = extractTextureData(mat.sheen.sheen_color_texture);
            }
        }
        if (mat.has_transmission) {
            gmat.hasTransmission = true;
            gmat.transmissionFactor = mat.transmission.transmission_factor;
            if (mat.transmission.transmission_texture.texture) {
                std::cout << "[info] Extracting transmission texture for material: " << gmat.name << std::endl;
                gmat.transmission_tex = extractTextureData(mat.transmission.transmission_texture);
            }
        }
        if (mat.has_ior) {
            gmat.ior = mat.ior.ior;
        }
        if (mat.has_emissive_strength) {
            gmat.emissiveStrength = mat.emissive_strength.emissive_strength;
        }
        if (mat.unlit) {
            gmat.isUnlit = true;
        }

        // Load custom properties from glTF extras
        if (mat.extras.data) {
            // cgltf stores extras as raw JSON; parse for lux_properties
            // Format: { "lux_properties": { "prop_name": 1.0, "prop_vec": [1,2,3,4] } }
            std::string extrasJson(static_cast<const char*>(mat.extras.data),
                                   mat.extras.data ? strlen(static_cast<const char*>(mat.extras.data)) : 0);
            // Simple check: if "lux_properties" appears in the extras JSON,
            // a full JSON parser (nlohmann::json) should be used at the call site
            // to populate custom_float_properties / custom_vec_properties.
            // For now, store the raw extras string for later parsing.
            (void)extrasJson;
        }

        scene.materials.push_back(gmat);
    }

    if (scene.materials.empty()) {
        scene.materials.push_back(GltfMaterial{"default"});
    }

    // --- Meshes ---
    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        auto& mesh = data->meshes[mi];
        size_t primStart = scene.meshes.size();
        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            auto& prim = mesh.primitives[pi];
            if (prim.type == cgltf_primitive_type_points) continue;
            GltfMesh gmesh;
            gmesh.name = mesh.name ? mesh.name : "unnamed";

            // Find accessors
            const cgltf_accessor* posAccessor = nullptr;
            const cgltf_accessor* normAccessor = nullptr;
            const cgltf_accessor* uvAccessor = nullptr;
            const cgltf_accessor* tangentAccessor = nullptr;

            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                auto& attr = prim.attributes[ai];
                if (attr.type == cgltf_attribute_type_position) posAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) uvAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_tangent) tangentAccessor = attr.data;
            }

            if (!posAccessor) continue;

            auto positions = readFloatAccessor(posAccessor);
            auto normals = readFloatAccessor(normAccessor);
            auto uvs = readFloatAccessor(uvAccessor);
            auto tangents = readFloatAccessor(tangentAccessor);

            size_t numVerts = posAccessor->count;
            gmesh.vertices.resize(numVerts);
            gmesh.hasTangents = !tangents.empty();

            for (size_t v = 0; v < numVerts; v++) {
                gmesh.vertices[v].position = glm::vec3(
                    positions[v*3], positions[v*3+1], positions[v*3+2]);
                if (!normals.empty()) {
                    gmesh.vertices[v].normal = glm::vec3(
                        normals[v*3], normals[v*3+1], normals[v*3+2]);
                } else {
                    gmesh.vertices[v].normal = glm::vec3(0, 1, 0);
                }
                if (!uvs.empty()) {
                    gmesh.vertices[v].uv = glm::vec2(uvs[v*2], uvs[v*2+1]);
                } else {
                    gmesh.vertices[v].uv = glm::vec2(0, 0);
                }
                if (!tangents.empty()) {
                    gmesh.vertices[v].tangent = glm::vec4(
                        tangents[v*4], tangents[v*4+1], tangents[v*4+2], tangents[v*4+3]);
                } else {
                    gmesh.vertices[v].tangent = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                }
            }

            if (prim.indices) {
                gmesh.indices = readIndexAccessor(prim.indices);
            } else {
                gmesh.indices.resize(numVerts);
                for (size_t i = 0; i < numVerts; i++) gmesh.indices[i] = static_cast<uint32_t>(i);
            }

            gmesh.materialIndex = prim.material
                ? static_cast<int>(cgltf_material_index(data, prim.material))
                : 0;

            scene.meshes.push_back(std::move(gmesh));
        }
        scene.meshPrimitiveRanges.push_back({primStart, scene.meshes.size() - primStart});
    }

    // --- KHR_gaussian_splatting ---
    // Scan primitives for POINTS topology with gaussian splatting attributes.
    // The KHR_gaussian_splatting extension encodes splat data as POINTS primitives
    // with custom attributes: POSITION (vec3), _ROTATION (vec4 quaternion),
    // _SCALE (vec3 log-space), _OPACITY (scalar logit-space), and
    // _SH_0.._SH_N for spherical harmonics coefficients.
    // Helper: check if attribute name matches a splat attribute (supports both
    // internal format: _ROTATION, _SCALE, _OPACITY, _SH_N
    // and Khronos conformance format: KHR_gaussian_splatting:ROTATION, etc.)
    auto isSplatAttr = [](const std::string& name, const std::string& suffix) -> bool {
        return name == "_" + suffix
            || name == "KHR_gaussian_splatting:" + suffix;
    };

    // Per-primitive metadata for deferred node transform application
    struct SplatPrimInfo {
        size_t meshIndex;     // cgltf mesh index this primitive belongs to
        size_t startSplat;    // index of first splat in accumulated arrays
        size_t splatCount;    // number of splats in this primitive
        int nodeIndex = -1;   // scene node index for per-instance transforms
    };
    std::vector<SplatPrimInfo> splatPrimInfos;

    // Track max SH degree across all primitives for zero-padding
    uint32_t globalMaxSHDegree = 0;

    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        auto& mesh = data->meshes[mi];
        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            auto& prim = mesh.primitives[pi];
            if (prim.type != cgltf_primitive_type_points) continue;

            // Check for gaussian splatting attributes (both naming conventions)
            bool hasRotation = false, hasScale = false;
            bool primIsKHR = false;
            bool hasInternalAttrs = false;  // lux's raw internal fixture convention
                                             // (_ROTATION/_SCALE/_OPACITY): these are
                                             // already shader-native (logit opacity,
                                             // log-space scale), unlike KHR semantics.
            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                if (prim.attributes[ai].name) {
                    std::string attrName(prim.attributes[ai].name);
                    if (isSplatAttr(attrName, "ROTATION")) hasRotation = true;
                    if (isSplatAttr(attrName, "SCALE")) hasScale = true;
                    if (attrName.rfind("KHR_gaussian_splatting:", 0) == 0) primIsKHR = true;
                    if (attrName == "_ROTATION" || attrName == "_SCALE" || attrName == "_OPACITY")
                        hasInternalAttrs = true;
                }
            }

            // Legacy pre-ratification draft layout: ROTATION/SCALE/OPACITY (and SH)
            // accessor indices live nested under extensions.KHR_gaussian_splatting,
            // rather than as primitive.attributes entries. Fall back to parsing that
            // raw extension JSON blob when the attribute scan above found nothing.
            const char* splatExtJson = findExtensionData(prim, "KHR_gaussian_splatting");
            std::string extText = splatExtJson ? std::string(splatExtJson) : std::string();
            int legacyRotationIdx = -1, legacyScaleIdx = -1, legacyOpacityIdx = -1;
            bool legacyNestedAttrs = false;
            if (splatExtJson) {
                // Only treat the primitive as KHR-format (linear opacity/scale
                // semantics) when it isn't already using lux's internal
                // shader-native attribute convention — some hand-authored test
                // fixtures carry a (redundant, historically unused) extension
                // object alongside _ROTATION/_SCALE/_OPACITY attributes, and
                // those must keep their original shader-native interpretation.
                if (!hasInternalAttrs) primIsKHR = true;
                if (!hasRotation && !hasScale) {
                    bool foundRot = jsonExtractInt(extText, "ROTATION", legacyRotationIdx);
                    bool foundScale = jsonExtractInt(extText, "SCALE", legacyScaleIdx);
                    jsonExtractInt(extText, "OPACITY", legacyOpacityIdx);
                    hasRotation = hasRotation || foundRot;
                    hasScale = hasScale || foundScale;
                    legacyNestedAttrs = foundRot || foundScale;
                }
                std::string cs, kernel;
                if (jsonExtractString(extText, "colorSpace", cs)) scene.splat_data.color_space = cs;
                if (jsonExtractString(extText, "kernel", kernel)) scene.splat_data.kernel = kernel;
            }

            if (!hasRotation && !hasScale) continue;

            if (scene.splat_data.has_splats) {
                std::cout << "[info] Merging additional splat primitive from mesh: "
                          << (mesh.name ? mesh.name : "unnamed") << std::endl;
            } else {
                std::cout << "[info] Detected KHR_gaussian_splatting primitive in mesh: "
                          << (mesh.name ? mesh.name : "unnamed")
                          << (legacyNestedAttrs ? " (legacy pre-ratification draft layout)" : "")
                          << std::endl;
            }

            if (primIsKHR) scene.splat_data.khr_format = true;

            // Temporary per-primitive buffers
            std::vector<float> primPositions;
            std::vector<float> primRotations;
            std::vector<float> primScales;
            std::vector<float> primOpacities;
            std::vector<float> primForeground;  // _FOREGROUND attribute, if present
            std::vector<std::vector<float>> primSHCoeffs; // per-degree
            uint32_t primSHDegree = 0;
            uint32_t primNumSplats = 0;

            // Parse all gaussian splatting attributes
            // Conformance SH format: KHR_gaussian_splatting:SH_DEGREE_N_COEF_M (vec3 each)
            // Internal SH format: _SH_N (packed float array per degree)
            // We collect conformance SH coefficients per degree, then pack them.
            std::map<int, std::vector<std::vector<float>>> khrSHByDegree; // degree -> [coef_idx] -> vec3 per splat

            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                auto& attr = prim.attributes[ai];
                if (!attr.name || !attr.data) continue;
                std::string attrName(attr.name);

                if (attr.type == cgltf_attribute_type_position) {
                    // Pack positions as vec4 (xyz, w=1)
                    auto pos3 = readFloatAccessor(attr.data);
                    primNumSplats = static_cast<uint32_t>(attr.data->count);
                    primPositions.resize(attr.data->count * 4);
                    for (size_t si = 0; si < attr.data->count; si++) {
                        primPositions[si * 4 + 0] = pos3[si * 3 + 0];
                        primPositions[si * 4 + 1] = pos3[si * 3 + 1];
                        primPositions[si * 4 + 2] = pos3[si * 3 + 2];
                        primPositions[si * 4 + 3] = 1.0f;
                    }
                } else if (isSplatAttr(attrName, "ROTATION")) {
                    primRotations = readFloatAccessor(attr.data);
                } else if (isSplatAttr(attrName, "SCALE")) {
                    primScales = readFloatAccessor(attr.data);
                } else if (isSplatAttr(attrName, "OPACITY")) {
                    primOpacities = readFloatAccessor(attr.data);
                } else if (attrName == "_FOREGROUND" && !std::getenv("LUX_DEBUG_IGNORE_FOREGROUND_ATTR")) {
                    // Custom mobiledlss attribute: SCALAR UNSIGNED_BYTE normalized
                    // (255 = foreground). cgltf_accessor_unpack_floats already
                    // applies the normalized-int -> [0,1] float conversion.
                    // LUX_DEBUG_IGNORE_FOREGROUND_ATTR=1 forces the morph-delta
                    // fallback below even when the attribute is present, for
                    // validating the two sources agree (see mobiledlss's
                    // juggle_p0.8_stride{2,4}.glb: both give 65,641/119,826).
                    primForeground = readFloatAccessor(attr.data);
                } else if (attrName.rfind("_SH_", 0) == 0) {
                    // Internal format: _SH_0, _SH_1, ... _SH_N (packed float array)
                    int degree = std::stoi(attrName.substr(4));
                    if (degree >= static_cast<int>(primSHCoeffs.size())) {
                        primSHCoeffs.resize(degree + 1);
                    }
                    primSHCoeffs[degree] = readFloatAccessor(attr.data);
                    if (static_cast<uint32_t>(degree) > primSHDegree) {
                        primSHDegree = static_cast<uint32_t>(degree);
                    }
                } else if (attrName.rfind("KHR_gaussian_splatting:SH_DEGREE_", 0) == 0) {
                    // Conformance format: KHR_gaussian_splatting:SH_DEGREE_N_COEF_M
                    // Each is a VEC3 accessor (one vec3 per splat for this coefficient)
                    // Parse degree and coef index from name
                    // Format: KHR_gaussian_splatting:SH_DEGREE_<D>_COEF_<C>
                    std::string rest = attrName.substr(33); // after "KHR_gaussian_splatting:SH_DEGREE_"
                    size_t underscorePos = rest.find("_COEF_");
                    if (underscorePos != std::string::npos) {
                        int degree = std::stoi(rest.substr(0, underscorePos));
                        int coefIdx = std::stoi(rest.substr(underscorePos + 6));
                        auto coeffData = readFloatAccessor(attr.data);
                        if (degree >= static_cast<int>(khrSHByDegree.size())) {
                            khrSHByDegree[degree]; // ensure entry exists
                        }
                        if (coefIdx >= static_cast<int>(khrSHByDegree[degree].size())) {
                            khrSHByDegree[degree].resize(coefIdx + 1);
                        }
                        khrSHByDegree[degree][coefIdx] = coeffData;
                    }
                }
            }

            // Legacy pre-ratification draft layout: resolve ROTATION/SCALE/OPACITY/SH
            // accessor indices parsed from extensions.KHR_gaussian_splatting directly,
            // since they aren't reachable via prim.attributes in this layout.
            if (legacyNestedAttrs) {
                if (legacyRotationIdx >= 0 && static_cast<size_t>(legacyRotationIdx) < data->accessors_count)
                    primRotations = readFloatAccessor(&data->accessors[legacyRotationIdx]);
                if (legacyScaleIdx >= 0 && static_cast<size_t>(legacyScaleIdx) < data->accessors_count)
                    primScales = readFloatAccessor(&data->accessors[legacyScaleIdx]);
                if (legacyOpacityIdx >= 0 && static_cast<size_t>(legacyOpacityIdx) < data->accessors_count)
                    primOpacities = readFloatAccessor(&data->accessors[legacyOpacityIdx]);

                // Legacy "sh" array: [{"degree":0,"coefficients":[N]}, {"degree":1,"coefficients":[N,N,N]}, ...]
                // Best-effort scan: walk each "degree"/"coefficients" pair in document order.
                size_t searchPos = 0;
                while (true) {
                    size_t degPos = extText.find("\"degree\"", searchPos);
                    if (degPos == std::string::npos) break;
                    int degree = 0;
                    jsonExtractInt(extText.substr(degPos), "degree", degree);
                    size_t coefPos = extText.find("\"coefficients\"", degPos);
                    if (coefPos == std::string::npos) break;
                    size_t arrStart = extText.find('[', coefPos);
                    size_t arrEnd = extText.find(']', arrStart == std::string::npos ? coefPos : arrStart);
                    if (arrStart == std::string::npos || arrEnd == std::string::npos) break;
                    std::string arrText = extText.substr(arrStart + 1, arrEnd - arrStart - 1);
                    std::vector<int> coeffIndices;
                    size_t tokStart = 0;
                    while (tokStart <= arrText.size()) {
                        size_t comma = arrText.find(',', tokStart);
                        std::string tok = arrText.substr(tokStart, comma == std::string::npos ? std::string::npos : comma - tokStart);
                        size_t a = tok.find_first_not_of(" \t\n\r");
                        if (a != std::string::npos) {
                            size_t b = tok.find_last_not_of(" \t\n\r");
                            coeffIndices.push_back(std::stoi(tok.substr(a, b - a + 1)));
                        }
                        if (comma == std::string::npos) break;
                        tokStart = comma + 1;
                    }
                    for (size_t ci = 0; ci < coeffIndices.size(); ci++) {
                        int accIdx = coeffIndices[ci];
                        if (accIdx < 0 || static_cast<size_t>(accIdx) >= data->accessors_count) continue;
                        auto coeffData = readFloatAccessor(&data->accessors[accIdx]);
                        if (degree >= static_cast<int>(khrSHByDegree.size())) khrSHByDegree[degree];
                        if (ci >= khrSHByDegree[degree].size()) khrSHByDegree[degree].resize(ci + 1);
                        khrSHByDegree[degree][ci] = coeffData;
                    }
                    searchPos = arrEnd + 1;
                }
            }

            // Pack KHR conformance SH data into our internal per-primitive format
            // Internal: sh_coefficients[degree_idx] = flat float array (all splats, all coefficients for that degree)
            // Degree 0 has 1 coefficient (3 floats per splat = DC color)
            // Degree 1 has 3 coefficients, degree 2 has 5, degree 3 has 7
            if (!khrSHByDegree.empty()) {
                int maxDeg = 0;
                for (auto& [deg, coeffs] : khrSHByDegree) {
                    if (deg > maxDeg) maxDeg = deg;
                }
                primSHDegree = static_cast<uint32_t>(maxDeg);

                // Unpack KHR per-degree data into per-coefficient arrays
                // Each coefficient gets its own array (vec3 per splat), matching
                // the shader's splat_sh0..splat_sh15 buffer layout.
                // Coefficient index mapping:
                //   Degree 0: 1 coeff  -> indices 0
                //   Degree 1: 3 coeffs -> indices 1,2,3
                //   Degree 2: 5 coeffs -> indices 4,5,6,7,8
                //   Degree 3: 7 coeffs -> indices 9,10,11,12,13,14,15
                static const int coeffBase[] = {0, 1, 4, 9};
                for (auto& [deg, coeffsByIdx] : khrSHByDegree) {
                    int base = coeffBase[deg];
                    for (size_t ci = 0; ci < coeffsByIdx.size(); ci++) {
                        int coeffIdx = base + static_cast<int>(ci);
                        if (coeffIdx >= static_cast<int>(primSHCoeffs.size())) {
                            primSHCoeffs.resize(coeffIdx + 1);
                        }
                        if (!coeffsByIdx[ci].empty()) {
                            primSHCoeffs[coeffIdx] = coeffsByIdx[ci]; // vec3 per splat
                        }
                    }
                }
            }

            // Update global max SH degree
            if (primSHDegree > globalMaxSHDegree) {
                globalMaxSHDegree = primSHDegree;
            }

            // Record per-primitive metadata for node transform application
            SplatPrimInfo info;
            info.meshIndex = mi;
            info.startSplat = scene.splat_data.num_splats;
            info.splatCount = primNumSplats;
            splatPrimInfos.push_back(info);

            // --- Dynamic splats: morph targets (see docs/lux-4d-spec.md) ---
            // Only the first primitive carrying `targets` populates
            // scene.splat_data.dynamics; multi-primitive dynamic scenes are
            // not yet supported end-to-end (documented limitation).
            if (prim.targets_count > 0 && !scene.splat_data.dynamics.has_motion) {
                scene.splat_data.dynamics.has_motion = true;
                for (size_t ti = 0; ti < prim.targets_count; ti++) {
                    auto& target = prim.targets[ti];
                    SplatMorphTarget morphTarget;
                    std::vector<uint32_t> posIdx, rotIdx, shIdx;
                    std::vector<float> posVal, rotVal, shVal;
                    for (size_t tai = 0; tai < target.attributes_count; tai++) {
                        auto& tattr = target.attributes[tai];
                        if (!tattr.name || !tattr.data) continue;
                        std::string tname(tattr.name);
                        if (tattr.type == cgltf_attribute_type_position) {
                            readSparseAccessor(tattr.data, posIdx, posVal);
                        } else if (isSplatAttr(tname, "ROTATION")) {
                            readSparseAccessor(tattr.data, rotIdx, rotVal);
                        } else if (tname == "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0") {
                            readSparseAccessor(tattr.data, shIdx, shVal);
                        }
                        // Other target semantics (scale/opacity/higher SH degrees)
                        // are static in v1; ignore rather than choke, per spec.
                    }
                    // Union the (usually identical) index sets across the three
                    // semantics, since the runtime representation assumes one
                    // shared index list per target.
                    std::vector<uint32_t> unionIdx = posIdx;
                    unionIdx.insert(unionIdx.end(), rotIdx.begin(), rotIdx.end());
                    unionIdx.insert(unionIdx.end(), shIdx.begin(), shIdx.end());
                    std::sort(unionIdx.begin(), unionIdx.end());
                    unionIdx.erase(std::unique(unionIdx.begin(), unionIdx.end()), unionIdx.end());

                    std::unordered_map<uint32_t, size_t> posMap, rotMap, shMap;
                    for (size_t i = 0; i < posIdx.size(); i++) posMap[posIdx[i]] = i;
                    for (size_t i = 0; i < rotIdx.size(); i++) rotMap[rotIdx[i]] = i;
                    for (size_t i = 0; i < shIdx.size(); i++) shMap[shIdx[i]] = i;

                    morphTarget.indices.reserve(unionIdx.size());
                    morphTarget.dpos.assign(unionIdx.size() * 3, 0.0f);
                    morphTarget.drot.assign(unionIdx.size() * 4, 0.0f);
                    morphTarget.dsh0.assign(unionIdx.size() * 3, 0.0f);
                    for (size_t u = 0; u < unionIdx.size(); u++) {
                        // Offset by info.startSplat so indices are valid into the
                        // scene-level (concatenated across primitives) arrays.
                        morphTarget.indices.push_back(unionIdx[u] + static_cast<uint32_t>(info.startSplat));
                        auto pit = posMap.find(unionIdx[u]);
                        if (pit != posMap.end())
                            std::memcpy(&morphTarget.dpos[u * 3], &posVal[pit->second * 3], 3 * sizeof(float));
                        auto rit = rotMap.find(unionIdx[u]);
                        if (rit != rotMap.end())
                            std::memcpy(&morphTarget.drot[u * 4], &rotVal[rit->second * 4], 4 * sizeof(float));
                        auto sit = shMap.find(unionIdx[u]);
                        if (sit != shMap.end())
                            std::memcpy(&morphTarget.dsh0[u * 3], &shVal[sit->second * 3], 3 * sizeof(float));
                    }
                    scene.splat_data.dynamics.targets.push_back(std::move(morphTarget));
                }

                // --- node.weights (initial pose) + animation (LINEAR sampler
                // on the node's `weights` path) -> keyframe times/weights. ---
                cgltf_node* ownerNode = nullptr;
                for (size_t ni = 0; ni < data->nodes_count; ni++) {
                    if (data->nodes[ni].mesh == &mesh) { ownerNode = &data->nodes[ni]; break; }
                }
                size_t numTargets = scene.splat_data.dynamics.targets.size();
                bool gotAnimation = false;
                if (ownerNode) {
                    for (size_t ai = 0; ai < data->animations_count; ai++) {
                        auto& anim = data->animations[ai];
                        for (size_t ci = 0; ci < anim.channels_count; ci++) {
                            auto& channel = anim.channels[ci];
                            if (channel.target_node != ownerNode ||
                                channel.target_path != cgltf_animation_path_type_weights) continue;
                            auto* sampler = channel.sampler;
                            if (!sampler) continue;
                            auto times = readFloatAccessor(sampler->input);
                            auto weightsFlat = readFloatAccessor(sampler->output);
                            if (numTargets == 0 || times.empty()) continue;
                            size_t numKeyframes = times.size();
                            if (weightsFlat.size() != numKeyframes * numTargets) continue;
                            scene.splat_data.dynamics.keyframes.resize(numKeyframes);
                            for (size_t ki = 0; ki < numKeyframes; ki++) {
                                scene.splat_data.dynamics.keyframes[ki].time = times[ki];
                                scene.splat_data.dynamics.keyframes[ki].weights.assign(
                                    weightsFlat.begin() + ki * numTargets,
                                    weightsFlat.begin() + (ki + 1) * numTargets);
                            }
                            gotAnimation = true;
                            std::cout << "[info] Dynamic splats: " << numKeyframes
                                      << " keyframes, " << numTargets << " morph targets, "
                                      << (sampler->interpolation == cgltf_interpolation_type_linear
                                              ? "LINEAR" : "non-LINEAR (unsupported, treated as LINEAR)")
                                      << " interpolation" << std::endl;
                            break;
                        }
                        if (gotAnimation) break;
                    }
                }
                if (!gotAnimation) {
                    std::cerr << "[warn] Dynamic splat primitive has morph targets but no "
                                 "weights animation was found; motion will be disabled." << std::endl;
                    scene.splat_data.dynamics.has_motion = false;
                }

                // --- mesh.extras.MOBILEDLSS_dynamic_splats (informative) ---
                // NOTE: this vendored cgltf.h only calls cgltf_parse_json_extras()
                // for PRIMITIVE-level extras (out_prim->extras), never for the
                // owning cgltf_mesh itself -- mesh.extras.data is therefore
                // always null, even though the ratified writer puts
                // MOBILEDLSS_dynamic_splats on the *mesh* object. Fall back to a
                // manual scan of the raw JSON buffer cgltf retains
                // (data->json/json_size) instead of relying on cgltf for this.
                std::string meshExtrasJson;
                if (jsonMeshExtras(data, mi, meshExtrasJson)) {
                    std::string convJson;
                    if (jsonExtractObject(meshExtrasJson, "MOBILEDLSS_dynamic_splats", convJson)) {
                        float fps = 30.0f;
                        if (jsonExtractFloat(convJson, "fps", fps)) {
                            scene.splat_data.dynamics.extrasFps = fps;
                            scene.splat_data.dynamics.hasExtrasFps = true;
                        }
                        jsonExtractIntArray(convJson, "keyframes", scene.splat_data.dynamics.extrasFrames);
                        std::string blend;
                        if (jsonExtractString(convJson, "rotationBlend", blend)) {
                            scene.splat_data.dynamics.rotationBlend = blend;
                        }
                    }
                }
            }

            // Append positions, rotations, scales, opacities to scene-level arrays
            scene.splat_data.positions.insert(scene.splat_data.positions.end(),
                primPositions.begin(), primPositions.end());
            scene.splat_data.rotations.insert(scene.splat_data.rotations.end(),
                primRotations.begin(), primRotations.end());
            scene.splat_data.scales.insert(scene.splat_data.scales.end(),
                primScales.begin(), primScales.end());
            scene.splat_data.opacities.insert(scene.splat_data.opacities.end(),
                primOpacities.begin(), primOpacities.end());
            // Only append when this primitive actually had a _FOREGROUND attribute
            // and every primitive seen so far also had one; otherwise leave
            // scene-level `foreground` short so the post-loop fallback (below)
            // detects the mismatch and rebuilds it from morph-target deltas.
            if (!primForeground.empty() &&
                scene.splat_data.foreground.size() == scene.splat_data.num_splats) {
                scene.splat_data.foreground.insert(scene.splat_data.foreground.end(),
                    primForeground.begin(), primForeground.end());
            }

            // Append SH coefficients with zero-padding for missing higher degrees
            // Ensure scene-level sh_coefficients has enough degree slots
            if (primSHCoeffs.size() > scene.splat_data.sh_coefficients.size()) {
                // Expand scene-level array; new degree slots need zero-padding for
                // previously accumulated splats
                size_t oldSize = scene.splat_data.sh_coefficients.size();
                scene.splat_data.sh_coefficients.resize(primSHCoeffs.size());
                // Zero-fill new degree arrays for previously accumulated splats
                for (size_t d = oldSize; d < primSHCoeffs.size(); d++) {
                    if (d < primSHCoeffs.size() && !primSHCoeffs[d].empty() && primNumSplats > 0) {
                        // Determine floats-per-splat from this primitive's data
                        size_t floatsPerSplat = primSHCoeffs[d].size() / primNumSplats;
                        scene.splat_data.sh_coefficients[d].resize(
                            scene.splat_data.num_splats * floatsPerSplat, 0.0f);
                    }
                }
            }
            // Now append this primitive's SH data for each degree
            for (size_t d = 0; d < primSHCoeffs.size(); d++) {
                scene.splat_data.sh_coefficients[d].insert(
                    scene.splat_data.sh_coefficients[d].end(),
                    primSHCoeffs[d].begin(), primSHCoeffs[d].end());
            }
            // For degrees that exist at the scene level but not in this primitive, zero-pad
            for (size_t d = primSHCoeffs.size(); d < scene.splat_data.sh_coefficients.size(); d++) {
                if (!scene.splat_data.sh_coefficients[d].empty() && scene.splat_data.num_splats > 0) {
                    size_t floatsPerSplat = scene.splat_data.sh_coefficients[d].size() / scene.splat_data.num_splats;
                    scene.splat_data.sh_coefficients[d].resize(
                        scene.splat_data.sh_coefficients[d].size() + primNumSplats * floatsPerSplat, 0.0f);
                }
            }

            scene.splat_data.num_splats += primNumSplats;
            scene.splat_data.has_splats = true;

            std::cout << "[info] Gaussian splats (cumulative): " << scene.splat_data.num_splats
                      << " splats, primitive SH degree " << primSHDegree << std::endl;
        }
    }

    // Finalize global SH degree
    if (scene.splat_data.has_splats) {
        scene.splat_data.sh_degree = globalMaxSHDegree;

        // Convert KHR linear opacity to logit space for shader compatibility.
        // The compute shader applies sigmoid() to opacity, so we store logit-space values.
        // Only for KHR format; internal format (_SCALE, _OPACITY) already stores raw values.
        //
        // SCALE is trickier: the *ratified* KHR_gaussian_splatting spec requires SCALE to
        // be linear ("Scale values are linear and MUST NOT be negative"), but the
        // currently-published Khronos conformance test assets predate an April-2026
        // editorial pass and still store log-space values (matching the pre-ratification
        // draft and raw 3DGS training convention) — including negative values, which are
        // invalid under the ratified rule. Since both encodings are seen in the wild for
        // the exact same attribute name/location, auto-detect: any negative value means
        // the data is already log-space (leave as-is, as the compute shader applies exp());
        // all-non-negative, boundedly-small values are treated as ratified linear scale and
        // converted to log-space here to match the shader's exp()-based reconstruction.
        if (scene.splat_data.khr_format) {
            // Convert linear opacity [0,1] to logit: log(p / (1 - p))
            for (size_t i = 0; i < scene.splat_data.opacities.size(); ++i) {
                float p = std::clamp(scene.splat_data.opacities[i], 1e-6f, 1.0f - 1e-6f);
                scene.splat_data.opacities[i] = std::log(p / (1.0f - p));
            }
            std::cout << "[info] Converted KHR linear opacity to logit for "
                      << scene.splat_data.opacities.size() << " splats" << std::endl;

            bool scaleLooksLinear = !scene.splat_data.scales.empty();
            for (float s : scene.splat_data.scales) {
                if (s < 0.0f || s >= 20.0f) { scaleLooksLinear = false; break; }
            }
            if (scaleLooksLinear) {
                for (auto& s : scene.splat_data.scales) {
                    s = std::log(std::max(s, 1e-12f));
                }
                std::cout << "[info] Converted ratified linear KHR SCALE to log-space for "
                          << scene.splat_data.num_splats << " splats" << std::endl;
            } else {
                std::cout << "[info] KHR SCALE already log-space (legacy/conformance layout)"
                          << std::endl;
            }
        }

        std::cout << "[info] Total gaussian splats: " << scene.splat_data.num_splats
                  << ", max SH degree " << scene.splat_data.sh_degree << std::endl;
        if (scene.splat_data.khr_format) {
            std::cout << "[info] KHR_gaussian_splatting kernel=" << scene.splat_data.kernel
                      << " colorSpace=" << scene.splat_data.color_space << std::endl;
        }

        // --- Foreground/actor coverage flag ---
        // Preferred source: the custom `_FOREGROUND` vertex attribute, appended
        // above per-primitive. If any primitive lacked it (size mismatch against
        // the final splat count), rebuild it from scratch using the fallback:
        // a gaussian is foreground iff it has any morph-target delta (i.e. is
        // in the union of dynamics.targets[*].indices). Static (non-dynamic)
        // scenes with no attribute get an all-zero flag.
        if (scene.splat_data.foreground.size() != scene.splat_data.num_splats) {
            if (!scene.splat_data.foreground.empty()) {
                std::cout << "[warn] _FOREGROUND attribute present on only some splat "
                             "primitives; falling back to morph-delta-derived foreground "
                             "flag for all " << scene.splat_data.num_splats << " splats"
                          << std::endl;
            }
            scene.splat_data.foreground.assign(scene.splat_data.num_splats, 0.0f);
            if (scene.splat_data.dynamics.has_motion) {
                size_t marked = 0;
                for (auto& target : scene.splat_data.dynamics.targets) {
                    for (uint32_t idx : target.indices) {
                        if (idx < scene.splat_data.foreground.size() &&
                            scene.splat_data.foreground[idx] == 0.0f) {
                            scene.splat_data.foreground[idx] = 1.0f;
                            marked++;
                        }
                    }
                }
                std::cout << "[info] Foreground coverage: derived from morph-target "
                             "deltas (" << marked << "/" << scene.splat_data.num_splats
                          << " splats)" << std::endl;
            }
        } else if (!scene.splat_data.foreground.empty()) {
            size_t fgCount = 0;
            for (float f : scene.splat_data.foreground) if (f > 0.5f) fgCount++;
            std::cout << "[info] Foreground coverage: from _FOREGROUND attribute ("
                      << fgCount << "/" << scene.splat_data.num_splats << " splats)"
                      << std::endl;
        }
    }

    // --- Nodes ---
    for (size_t ni = 0; ni < data->nodes_count; ni++) {
        auto& node = data->nodes[ni];
        GltfNode gnode;
        gnode.name = node.name ? node.name : ("node_" + std::to_string(ni));
        gnode.localTransform = nodeTransform(&node);
        gnode.worldTransform = glm::mat4(1.0f);
        gnode.meshIndex = node.mesh ? static_cast<int>(cgltf_mesh_index(data, node.mesh)) : -1;
        gnode.cameraIndex = node.camera ? static_cast<int>(cgltf_camera_index(data, node.camera)) : -1;

        if (node.light) {
            // Find light index by matching pointer into data->lights array
            for (size_t li = 0; li < data->lights_count; li++) {
                if (&data->lights[li] == node.light) {
                    gnode.lightIndex = static_cast<int>(li);
                    break;
                }
            }
        }

        for (size_t ci = 0; ci < node.children_count; ci++) {
            gnode.children.push_back(static_cast<int>(cgltf_node_index(data, node.children[ci])));
        }

        scene.nodes.push_back(std::move(gnode));
    }

    // Set parents
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        for (int child : scene.nodes[i].children) {
            if (child >= 0 && child < static_cast<int>(scene.nodes.size())) {
                scene.nodes[child].parent = static_cast<int>(i);
            }
        }
    }

    // Root nodes from scene
    if (data->scenes_count > 0) {
        auto& s = data->scenes[data->scene ? cgltf_scene_index(data, data->scene) : 0];
        for (size_t i = 0; i < s.nodes_count; i++) {
            scene.rootNodes.push_back(static_cast<int>(cgltf_node_index(data, s.nodes[i])));
        }
    }

    // --- Apply node transforms to splat data ---
    // Now that nodes are loaded and parent relationships are set, compute world
    // transforms and apply them to splat positions, rotations, and scales.
    if (scene.splat_data.has_splats && !splatPrimInfos.empty()) {
        // First compute world transforms for all nodes (top-down BFS from roots)
        // We need to do this before flattenScene which also computes them
        std::function<void(int, const glm::mat4&)> computeWorld =
            [&](int nodeIdx, const glm::mat4& parentWorld) {
                auto& node = scene.nodes[nodeIdx];
                node.worldTransform = parentWorld * node.localTransform;
                for (int child : node.children) {
                    if (child >= 0 && child < static_cast<int>(scene.nodes.size())) {
                        computeWorld(child, node.worldTransform);
                    }
                }
            };
        glm::mat4 identity(1.0f);
        for (int root : scene.rootNodes) {
            if (root >= 0 && root < static_cast<int>(scene.nodes.size())) {
                computeWorld(root, identity);
            }
        }

        // Build mapping: cgltf mesh index -> list of scene node indices
        std::unordered_map<size_t, std::vector<size_t>> meshToNodeIndices;
        for (size_t ni = 0; ni < scene.nodes.size(); ni++) {
            int mi = scene.nodes[ni].meshIndex;
            if (mi >= 0) {
                meshToNodeIndices[static_cast<size_t>(mi)].push_back(ni);
            }
        }

        // For meshes referenced by multiple nodes, duplicate splat data per instance
        {
            size_t origNumSplats = scene.splat_data.num_splats;
            std::vector<size_t> shFPS(scene.splat_data.sh_coefficients.size(), 0);
            for (size_t d = 0; d < shFPS.size(); d++) {
                if (!scene.splat_data.sh_coefficients[d].empty() && origNumSplats > 0) {
                    shFPS[d] = scene.splat_data.sh_coefficients[d].size() / origNumSplats;
                }
            }

            std::vector<SplatPrimInfo> expandedInfos;
            for (auto& info : splatPrimInfos) {
                auto it = meshToNodeIndices.find(info.meshIndex);
                if (it == meshToNodeIndices.end() || it->second.empty()) {
                    expandedInfos.push_back(info);
                    continue;
                }
                auto& nodeList = it->second;

                // First node uses existing data
                SplatPrimInfo first = info;
                first.nodeIndex = static_cast<int>(nodeList[0]);
                expandedInfos.push_back(first);

                if (nodeList.size() > 1) {
                    // Copy original data into temporaries for safe self-insert
                    std::vector<float> origPos(
                        scene.splat_data.positions.begin() + info.startSplat * 4,
                        scene.splat_data.positions.begin() + (info.startSplat + info.splatCount) * 4);
                    std::vector<float> origRot(
                        scene.splat_data.rotations.begin() + info.startSplat * 4,
                        scene.splat_data.rotations.begin() + (info.startSplat + info.splatCount) * 4);
                    std::vector<float> origScl(
                        scene.splat_data.scales.begin() + info.startSplat * 3,
                        scene.splat_data.scales.begin() + (info.startSplat + info.splatCount) * 3);
                    std::vector<float> origOpa(
                        scene.splat_data.opacities.begin() + info.startSplat,
                        scene.splat_data.opacities.begin() + info.startSplat + info.splatCount);
                    std::vector<std::vector<float>> origSH(scene.splat_data.sh_coefficients.size());
                    for (size_t d = 0; d < origSH.size(); d++) {
                        if (shFPS[d] > 0) {
                            size_t s = info.startSplat * shFPS[d];
                            size_t e = (info.startSplat + info.splatCount) * shFPS[d];
                            origSH[d].assign(
                                scene.splat_data.sh_coefficients[d].begin() + s,
                                scene.splat_data.sh_coefficients[d].begin() + e);
                        }
                    }

                    // Additional nodes: duplicate splat data
                    for (size_t k = 1; k < nodeList.size(); k++) {
                        SplatPrimInfo copy;
                        copy.meshIndex = info.meshIndex;
                        copy.nodeIndex = static_cast<int>(nodeList[k]);
                        copy.startSplat = scene.splat_data.num_splats;
                        copy.splatCount = info.splatCount;

                        scene.splat_data.positions.insert(scene.splat_data.positions.end(),
                            origPos.begin(), origPos.end());
                        scene.splat_data.rotations.insert(scene.splat_data.rotations.end(),
                            origRot.begin(), origRot.end());
                        scene.splat_data.scales.insert(scene.splat_data.scales.end(),
                            origScl.begin(), origScl.end());
                        scene.splat_data.opacities.insert(scene.splat_data.opacities.end(),
                            origOpa.begin(), origOpa.end());
                        for (size_t d = 0; d < origSH.size(); d++) {
                            if (!origSH[d].empty()) {
                                scene.splat_data.sh_coefficients[d].insert(
                                    scene.splat_data.sh_coefficients[d].end(),
                                    origSH[d].begin(), origSH[d].end());
                            }
                        }

                        scene.splat_data.num_splats += info.splatCount;
                        expandedInfos.push_back(copy);
                    }
                }
            }
            splatPrimInfos = expandedInfos;

            if (scene.splat_data.num_splats != origNumSplats) {
                std::cout << "[info] Expanded multi-instance splat meshes: "
                          << origNumSplats << " -> " << scene.splat_data.num_splats << " splats" << std::endl;
            }
        }

        // Apply transforms to each splat entry using its specific node's world transform
        for (auto& info : splatPrimInfos) {
            if (info.nodeIndex < 0 || info.nodeIndex >= static_cast<int>(scene.nodes.size())) continue;
            glm::mat4 world = scene.nodes[info.nodeIndex].worldTransform;

            // Check if transform is identity (skip if so)
            bool isIdentity = true;
            for (int c = 0; c < 4 && isIdentity; c++) {
                for (int r = 0; r < 4 && isIdentity; r++) {
                    float expected = (c == r) ? 1.0f : 0.0f;
                    if (std::abs(world[c][r] - expected) > 1e-6f) isIdentity = false;
                }
            }
            if (isIdentity) continue;

            std::cout << "[info] Applying node transform to splat range ["
                      << info.startSplat << ", " << info.startSplat + info.splatCount
                      << ")" << std::endl;

            // Extract rotation quaternion from the 3x3 upper-left of the world matrix
            // (assumes no shear; uses normalized rotation matrix)
            glm::mat3 rotMat(world);
            // Extract scale from column lengths
            float sx = glm::length(glm::vec3(rotMat[0]));
            float sy = glm::length(glm::vec3(rotMat[1]));
            float sz = glm::length(glm::vec3(rotMat[2]));
            float uniformScale = std::cbrt(sx * sy * sz); // geometric mean

            // Normalize rotation matrix (remove scale)
            if (sx > 1e-6f) rotMat[0] /= sx;
            if (sy > 1e-6f) rotMat[1] /= sy;
            if (sz > 1e-6f) rotMat[2] /= sz;

            // Convert rotation matrix to quaternion (Shepperd's method)
            glm::vec4 nodeQuat;
            float trace = rotMat[0][0] + rotMat[1][1] + rotMat[2][2];
            if (trace > 0.0f) {
                float s = 0.5f / std::sqrt(trace + 1.0f);
                nodeQuat.w = 0.25f / s;
                nodeQuat.x = (rotMat[1][2] - rotMat[2][1]) * s;
                nodeQuat.y = (rotMat[2][0] - rotMat[0][2]) * s;
                nodeQuat.z = (rotMat[0][1] - rotMat[1][0]) * s;
            } else if (rotMat[0][0] > rotMat[1][1] && rotMat[0][0] > rotMat[2][2]) {
                float s = 2.0f * std::sqrt(1.0f + rotMat[0][0] - rotMat[1][1] - rotMat[2][2]);
                nodeQuat.w = (rotMat[1][2] - rotMat[2][1]) / s;
                nodeQuat.x = 0.25f * s;
                nodeQuat.y = (rotMat[1][0] + rotMat[0][1]) / s;
                nodeQuat.z = (rotMat[2][0] + rotMat[0][2]) / s;
            } else if (rotMat[1][1] > rotMat[2][2]) {
                float s = 2.0f * std::sqrt(1.0f + rotMat[1][1] - rotMat[0][0] - rotMat[2][2]);
                nodeQuat.w = (rotMat[2][0] - rotMat[0][2]) / s;
                nodeQuat.x = (rotMat[1][0] + rotMat[0][1]) / s;
                nodeQuat.y = 0.25f * s;
                nodeQuat.z = (rotMat[2][1] + rotMat[1][2]) / s;
            } else {
                float s = 2.0f * std::sqrt(1.0f + rotMat[2][2] - rotMat[0][0] - rotMat[1][1]);
                nodeQuat.w = (rotMat[0][1] - rotMat[1][0]) / s;
                nodeQuat.x = (rotMat[2][0] + rotMat[0][2]) / s;
                nodeQuat.y = (rotMat[2][1] + rotMat[1][2]) / s;
                nodeQuat.z = 0.25f * s;
            }
            // Normalize quaternion
            float qlen = std::sqrt(nodeQuat.x*nodeQuat.x + nodeQuat.y*nodeQuat.y +
                                   nodeQuat.z*nodeQuat.z + nodeQuat.w*nodeQuat.w);
            if (qlen > 1e-6f) { nodeQuat.x /= qlen; nodeQuat.y /= qlen; nodeQuat.z /= qlen; nodeQuat.w /= qlen; }

            // Apply to positions: worldPos = world * vec4(pos, 1.0)
            for (size_t si = info.startSplat; si < info.startSplat + info.splatCount; si++) {
                size_t base = si * 4;
                if (base + 3 >= scene.splat_data.positions.size()) break;
                glm::vec4 pos(scene.splat_data.positions[base],
                              scene.splat_data.positions[base + 1],
                              scene.splat_data.positions[base + 2], 1.0f);
                glm::vec4 worldPos = world * pos;
                scene.splat_data.positions[base + 0] = worldPos.x;
                scene.splat_data.positions[base + 1] = worldPos.y;
                scene.splat_data.positions[base + 2] = worldPos.z;
                scene.splat_data.positions[base + 3] = 1.0f;
            }

            // Apply to rotations: multiply splat quaternion by node quaternion
            // q_combined = q_node * q_splat (Hamilton product)
            for (size_t si = info.startSplat; si < info.startSplat + info.splatCount; si++) {
                size_t base = si * 4;
                if (base + 3 >= scene.splat_data.rotations.size()) break;
                float qx = scene.splat_data.rotations[base + 0];
                float qy = scene.splat_data.rotations[base + 1];
                float qz = scene.splat_data.rotations[base + 2];
                float qw = scene.splat_data.rotations[base + 3];
                // Hamilton product: nodeQuat * splatQuat
                scene.splat_data.rotations[base + 0] = nodeQuat.w*qx + nodeQuat.x*qw + nodeQuat.y*qz - nodeQuat.z*qy;
                scene.splat_data.rotations[base + 1] = nodeQuat.w*qy - nodeQuat.x*qz + nodeQuat.y*qw + nodeQuat.z*qx;
                scene.splat_data.rotations[base + 2] = nodeQuat.w*qz + nodeQuat.x*qy - nodeQuat.y*qx + nodeQuat.z*qw;
                scene.splat_data.rotations[base + 3] = nodeQuat.w*qw - nodeQuat.x*qx - nodeQuat.y*qy - nodeQuat.z*qz;
            }

            // Apply to scales: multiply by uniform scale factor
            // Scales are in log-space, so add log(uniformScale) to each component
            float logScale = std::log(uniformScale);
            for (size_t si = info.startSplat; si < info.startSplat + info.splatCount; si++) {
                size_t base = si * 3;
                if (base + 2 >= scene.splat_data.scales.size()) break;
                scene.splat_data.scales[base + 0] += logScale;
                scene.splat_data.scales[base + 1] += logScale;
                scene.splat_data.scales[base + 2] += logScale;
            }

            // Rotate SH coefficients for degrees 1+ when node has rotation
            {
                glm::mat3 identity(1.0f);
                bool rotIsIdentity = true;
                for (int c = 0; c < 3 && rotIsIdentity; c++)
                    for (int r = 0; r < 3 && rotIsIdentity; r++)
                        if (std::abs(rotMat[c][r] - identity[c][r]) > 1e-6f)
                            rotIsIdentity = false;

                if (!rotIsIdentity && scene.splat_data.sh_degree >= 1) {
                    std::cout << "[info] Rotating SH coefficients (degree 1-"
                              << scene.splat_data.sh_degree << ") for node "
                              << info.nodeIndex << std::endl;
                    rotateSHCoeffs(rotMat, scene.splat_data.sh_coefficients,
                                   info.startSplat, info.splatCount,
                                   scene.splat_data.sh_degree);
                }
            }
        }
    }

    // --- Cameras ---
    for (size_t ci = 0; ci < data->cameras_count; ci++) {
        auto& cam = data->cameras[ci];
        GltfCamera gcam;
        gcam.name = cam.name ? cam.name : "camera";
        if (cam.type == cgltf_camera_type_perspective) {
            gcam.type = "perspective";
            gcam.fovY = cam.data.perspective.yfov;
            gcam.aspect = cam.data.perspective.aspect_ratio > 0 ? cam.data.perspective.aspect_ratio : 1.0f;
            gcam.zNear = cam.data.perspective.znear;
            gcam.zFar = cam.data.perspective.zfar > 0 ? cam.data.perspective.zfar : 1000.0f;
        }
        scene.cameras.push_back(gcam);
    }

    // --- Lights (KHR_lights_punctual) ---
    if (data->lights_count > 0) {
        for (size_t li = 0; li < data->lights_count; li++) {
            auto& light = data->lights[li];
            GltfLight glight;
            glight.name = light.name ? light.name : "light";
            switch (light.type) {
                case cgltf_light_type_directional: glight.type = "directional"; break;
                case cgltf_light_type_point: glight.type = "point"; break;
                case cgltf_light_type_spot: glight.type = "spot"; break;
                default: glight.type = "directional"; break;
            }
            glight.color = glm::vec3(light.color[0], light.color[1], light.color[2]);
            glight.intensity = light.intensity;
            glight.range = light.range;
            if (light.type == cgltf_light_type_spot) {
                glight.innerConeAngle = light.spot_inner_cone_angle;
                glight.outerConeAngle = light.spot_outer_cone_angle;
            }
            scene.lights.push_back(glight);
        }
    }

    cgltf_free(data);
    return scene;
}

std::vector<DrawItem> flattenScene(GltfScene& scene) {
    std::vector<DrawItem> items;

    std::function<void(int, const glm::mat4&)> traverse =
        [&](int nodeIdx, const glm::mat4& parentWorld) {
            auto& node = scene.nodes[nodeIdx];
            node.worldTransform = parentWorld * node.localTransform;

            if (node.meshIndex >= 0) {
                size_t mi = static_cast<size_t>(node.meshIndex);
                if (mi < scene.meshPrimitiveRanges.size()) {
                    auto [start, count] = scene.meshPrimitiveRanges[mi];
                    for (size_t pi = start; pi < start + count; pi++) {
                        DrawItem item;
                        item.worldTransform = node.worldTransform;
                        item.meshIndex = static_cast<int>(pi);
                        item.materialIndex = scene.meshes[pi].materialIndex;
                        items.push_back(item);
                    }
                } else if (mi < scene.meshes.size()) {
                    DrawItem item;
                    item.worldTransform = node.worldTransform;
                    item.meshIndex = node.meshIndex;
                    item.materialIndex = scene.meshes[mi].materialIndex;
                    items.push_back(item);
                }
            }

            // Extract world-space position and direction for lights attached to this node
            if (node.lightIndex >= 0 && node.lightIndex < static_cast<int>(scene.lights.size())) {
                auto& light = scene.lights[node.lightIndex];
                // Extract position from world transform column 3
                light.position = glm::vec3(node.worldTransform[3]);
                // Extract forward direction (-Z in local space, transformed)
                light.direction = glm::normalize(glm::vec3(
                    node.worldTransform * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
            }

            for (int child : node.children) {
                if (child >= 0 && child < static_cast<int>(scene.nodes.size())) {
                    traverse(child, node.worldTransform);
                }
            }
        };

    glm::mat4 identity(1.0f);
    for (int root : scene.rootNodes) {
        if (root >= 0 && root < static_cast<int>(scene.nodes.size())) {
            traverse(root, identity);
        }
    }

    std::sort(items.begin(), items.end(),
              [](const DrawItem& a, const DrawItem& b) {
                  return a.materialIndex < b.materialIndex;
              });

    return items;
}

// ===========================================================================
// Dynamic (4D) Gaussian splats: animation evaluation + segment building
// ===========================================================================

SplatMorphState evaluateSplatMorphState(const SplatDynamics& dyn, float timeSeconds) {
    SplatMorphState state;
    if (!dyn.has_motion || dyn.keyframes.size() < 2 || dyn.targets.empty()) {
        state.highTargetIndex = 0;
        state.lowTargetIndex = -1;
        state.weightLow = 0.0f;
        state.weightHigh = 0.0f;
        return state;
    }

    const auto& kf = dyn.keyframes;
    float t = timeSeconds;
    if (t <= kf.front().time) {
        // Exactly at (or before) the base frame: no motion.
        state.highTargetIndex = 0;
        state.lowTargetIndex = -1;
        state.weightLow = 0.0f;
        state.weightHigh = 0.0f;
        return state;
    }
    if (t >= kf.back().time) t = kf.back().time;

    // Find i such that kf[i].time <= t <= kf[i+1].time.
    size_t i = 0;
    for (; i + 1 < kf.size(); i++) {
        if (t <= kf[i + 1].time) break;
    }
    if (i + 1 >= kf.size()) i = kf.size() - 2;

    float t0 = kf[i].time, t1 = kf[i + 1].time;
    float alpha = (t1 > t0) ? (t - t0) / (t1 - t0) : 1.0f;

    // keyframe index m (1..K) corresponds to targets[m-1]; keyframe 0 is the base.
    state.highTargetIndex = static_cast<int>(i);       // targets[i] == keyframe (i+1)
    state.lowTargetIndex = (i >= 1) ? static_cast<int>(i - 1) : -1;  // targets[i-1] == keyframe i (or base)
    state.weightLow = 1.0f - alpha;
    state.weightHigh = alpha;
    return state;
}

float splatFrameToTime(const SplatDynamics& dyn, int frame) {
    if (dyn.hasExtrasFps) {
        return static_cast<float>(frame) / dyn.extrasFps;
    }
    if (dyn.keyframes.empty()) return 0.0f;
    int idx = frame;
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(dyn.keyframes.size())) idx = static_cast<int>(dyn.keyframes.size()) - 1;
    return dyn.keyframes[idx].time;
}

std::vector<SplatMorphSegment> buildSplatMorphSegments(const SplatDynamics& dyn) {
    std::vector<SplatMorphSegment> segments;
    size_t numTargets = dyn.targets.size();
    if (numTargets == 0) return segments;
    segments.resize(numTargets);

    for (size_t seg = 0; seg < numTargets; seg++) {
        const SplatMorphTarget* lo = (seg >= 1) ? &dyn.targets[seg - 1] : nullptr;
        const SplatMorphTarget& hi = dyn.targets[seg];

        std::vector<uint32_t> unionIdx = hi.indices;
        if (lo) unionIdx.insert(unionIdx.end(), lo->indices.begin(), lo->indices.end());
        std::sort(unionIdx.begin(), unionIdx.end());
        unionIdx.erase(std::unique(unionIdx.begin(), unionIdx.end()), unionIdx.end());

        std::unordered_map<uint32_t, size_t> loMap, hiMap;
        if (lo) for (size_t i = 0; i < lo->indices.size(); i++) loMap[lo->indices[i]] = i;
        for (size_t i = 0; i < hi.indices.size(); i++) hiMap[hi.indices[i]] = i;

        SplatMorphSegment& s = segments[seg];
        s.index = unionIdx;
        s.dposLo.assign(unionIdx.size() * 3, 0.0f);
        s.dposHi.assign(unionIdx.size() * 3, 0.0f);
        s.drotLo.assign(unionIdx.size() * 4, 0.0f);
        s.drotHi.assign(unionIdx.size() * 4, 0.0f);
        s.dsh0Lo.assign(unionIdx.size() * 3, 0.0f);
        s.dsh0Hi.assign(unionIdx.size() * 3, 0.0f);

        for (size_t u = 0; u < unionIdx.size(); u++) {
            uint32_t idx = unionIdx[u];
            if (lo) {
                auto it = loMap.find(idx);
                if (it != loMap.end()) {
                    std::memcpy(&s.dposLo[u * 3], &lo->dpos[it->second * 3], 3 * sizeof(float));
                    std::memcpy(&s.drotLo[u * 4], &lo->drot[it->second * 4], 4 * sizeof(float));
                    std::memcpy(&s.dsh0Lo[u * 3], &lo->dsh0[it->second * 3], 3 * sizeof(float));
                }
            }
            auto it = hiMap.find(idx);
            if (it != hiMap.end()) {
                std::memcpy(&s.dposHi[u * 3], &hi.dpos[it->second * 3], 3 * sizeof(float));
                std::memcpy(&s.drotHi[u * 4], &hi.drot[it->second * 4], 4 * sizeof(float));
                std::memcpy(&s.dsh0Hi[u * 3], &hi.dsh0[it->second * 3], 3 * sizeof(float));
            }
        }
    }
    return segments;
}
