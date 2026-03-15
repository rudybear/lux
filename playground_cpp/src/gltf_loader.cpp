#include "gltf_loader.h"

#include <algorithm>
#include <functional>
#include <fstream>
#include <map>
#include <stdexcept>
#include <cstring>
#include <cmath>
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
            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                if (prim.attributes[ai].name) {
                    std::string attrName(prim.attributes[ai].name);
                    if (isSplatAttr(attrName, "ROTATION")) hasRotation = true;
                    if (isSplatAttr(attrName, "SCALE")) hasScale = true;
                    if (attrName.rfind("KHR_gaussian_splatting:", 0) == 0) primIsKHR = true;
                }
            }

            if (!hasRotation && !hasScale) continue;

            if (scene.splat_data.has_splats) {
                std::cout << "[info] Merging additional splat primitive from mesh: "
                          << (mesh.name ? mesh.name : "unnamed") << std::endl;
            } else {
                std::cout << "[info] Detected KHR_gaussian_splatting primitive in mesh: "
                          << (mesh.name ? mesh.name : "unnamed") << std::endl;
            }

            if (primIsKHR) scene.splat_data.khr_format = true;

            // Temporary per-primitive buffers
            std::vector<float> primPositions;
            std::vector<float> primRotations;
            std::vector<float> primScales;
            std::vector<float> primOpacities;
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

            // Append positions, rotations, scales, opacities to scene-level arrays
            scene.splat_data.positions.insert(scene.splat_data.positions.end(),
                primPositions.begin(), primPositions.end());
            scene.splat_data.rotations.insert(scene.splat_data.rotations.end(),
                primRotations.begin(), primRotations.end());
            scene.splat_data.scales.insert(scene.splat_data.scales.end(),
                primScales.begin(), primScales.end());
            scene.splat_data.opacities.insert(scene.splat_data.opacities.end(),
                primOpacities.begin(), primOpacities.end());

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
        // KHR scales are already in log-space per spec (no conversion needed).
        // Only for KHR format; internal format (_SCALE, _OPACITY) already stores raw values.
        if (scene.splat_data.khr_format) {
            // Convert linear opacity [0,1] to logit: log(p / (1 - p))
            for (size_t i = 0; i < scene.splat_data.opacities.size(); ++i) {
                float p = std::clamp(scene.splat_data.opacities[i], 1e-6f, 1.0f - 1e-6f);
                scene.splat_data.opacities[i] = std::log(p / (1.0f - p));
            }
            std::cout << "[info] Converted KHR linear opacity to logit for "
                      << scene.splat_data.opacities.size() << " splats" << std::endl;
        }

        std::cout << "[info] Total gaussian splats: " << scene.splat_data.num_splats
                  << ", max SH degree " << scene.splat_data.sh_degree << std::endl;
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

        // Build mapping: cgltf mesh index -> node world transform
        // (use the first node referencing each mesh)
        std::unordered_map<size_t, glm::mat4> meshToWorldTransform;
        for (size_t ni = 0; ni < scene.nodes.size(); ni++) {
            int mi = scene.nodes[ni].meshIndex;
            if (mi >= 0 && meshToWorldTransform.find(static_cast<size_t>(mi)) == meshToWorldTransform.end()) {
                meshToWorldTransform[static_cast<size_t>(mi)] = scene.nodes[ni].worldTransform;
            }
        }

        // Apply transforms to each splat primitive's data
        for (auto& info : splatPrimInfos) {
            auto it = meshToWorldTransform.find(info.meshIndex);
            if (it == meshToWorldTransform.end()) continue;

            glm::mat4 world = it->second;

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
