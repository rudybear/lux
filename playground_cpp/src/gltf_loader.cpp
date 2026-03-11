#include "gltf_loader.h"

#include <algorithm>
#include <functional>
#include <fstream>
#include <map>
#include <stdexcept>
#include <cstring>
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

    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        auto& mesh = data->meshes[mi];
        for (size_t pi = 0; pi < mesh.primitives_count; pi++) {
            auto& prim = mesh.primitives[pi];
            if (prim.type != cgltf_primitive_type_points) continue;

            // Check for gaussian splatting attributes (both naming conventions)
            bool hasRotation = false, hasScale = false;
            for (size_t ai = 0; ai < prim.attributes_count; ai++) {
                if (prim.attributes[ai].name) {
                    std::string attrName(prim.attributes[ai].name);
                    if (isSplatAttr(attrName, "ROTATION")) hasRotation = true;
                    if (isSplatAttr(attrName, "SCALE")) hasScale = true;
                }
            }

            if (!hasRotation && !hasScale) continue;

            std::cout << "[info] Detected KHR_gaussian_splatting primitive in mesh: "
                      << (mesh.name ? mesh.name : "unnamed") << std::endl;

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
                    scene.splat_data.num_splats = static_cast<uint32_t>(attr.data->count);
                    scene.splat_data.positions.resize(attr.data->count * 4);
                    for (size_t si = 0; si < attr.data->count; si++) {
                        scene.splat_data.positions[si * 4 + 0] = pos3[si * 3 + 0];
                        scene.splat_data.positions[si * 4 + 1] = pos3[si * 3 + 1];
                        scene.splat_data.positions[si * 4 + 2] = pos3[si * 3 + 2];
                        scene.splat_data.positions[si * 4 + 3] = 1.0f;
                    }
                } else if (isSplatAttr(attrName, "ROTATION")) {
                    scene.splat_data.rotations = readFloatAccessor(attr.data);
                } else if (isSplatAttr(attrName, "SCALE")) {
                    scene.splat_data.scales = readFloatAccessor(attr.data);
                } else if (isSplatAttr(attrName, "OPACITY")) {
                    scene.splat_data.opacities = readFloatAccessor(attr.data);
                } else if (attrName.rfind("_SH_", 0) == 0) {
                    // Internal format: _SH_0, _SH_1, ... _SH_N (packed float array)
                    int degree = std::stoi(attrName.substr(4));
                    if (degree >= static_cast<int>(scene.splat_data.sh_coefficients.size())) {
                        scene.splat_data.sh_coefficients.resize(degree + 1);
                    }
                    scene.splat_data.sh_coefficients[degree] = readFloatAccessor(attr.data);
                    if (static_cast<uint32_t>(degree) > scene.splat_data.sh_degree) {
                        scene.splat_data.sh_degree = static_cast<uint32_t>(degree);
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

            // Pack KHR conformance SH data into our internal format
            // Internal: sh_coefficients[degree_idx] = flat float array (all splats, all coefficients for that degree)
            // Degree 0 has 1 coefficient (3 floats per splat = DC color)
            // Degree 1 has 3 coefficients, degree 2 has 5, degree 3 has 7
            if (!khrSHByDegree.empty()) {
                int maxDegree = 0;
                for (auto& [deg, coeffs] : khrSHByDegree) {
                    if (deg > maxDegree) maxDegree = deg;
                }
                scene.splat_data.sh_degree = static_cast<uint32_t>(maxDegree);

                // For degree 0: pack as DC color (SH0 coefficient)
                // Each KHR SH coefficient is a VEC3 (r,g,b) per splat
                // Our internal format expects sh_coefficients[0] = [r0,g0,b0, r1,g1,b1, ...]
                for (auto& [deg, coeffsByIdx] : khrSHByDegree) {
                    // Ensure degree index exists in our array
                    // For KHR format, we store ALL coefficients for a degree in one flat array
                    // Number of coefficients per degree: 2*l+1
                    size_t numCoeffs = coeffsByIdx.size();
                    size_t numSplats = scene.splat_data.num_splats;

                    // Flatten: for each splat, write all coefficients (each is vec3)
                    std::vector<float> packed(numSplats * numCoeffs * 3, 0.0f);
                    for (size_t ci = 0; ci < numCoeffs; ci++) {
                        if (ci < coeffsByIdx.size() && !coeffsByIdx[ci].empty()) {
                            for (size_t si = 0; si < numSplats; si++) {
                                size_t srcBase = si * 3;
                                size_t dstBase = si * numCoeffs * 3 + ci * 3;
                                if (srcBase + 2 < coeffsByIdx[ci].size() && dstBase + 2 < packed.size()) {
                                    packed[dstBase + 0] = coeffsByIdx[ci][srcBase + 0];
                                    packed[dstBase + 1] = coeffsByIdx[ci][srcBase + 1];
                                    packed[dstBase + 2] = coeffsByIdx[ci][srcBase + 2];
                                }
                            }
                        }
                    }

                    if (deg >= static_cast<int>(scene.splat_data.sh_coefficients.size())) {
                        scene.splat_data.sh_coefficients.resize(deg + 1);
                    }
                    scene.splat_data.sh_coefficients[deg] = std::move(packed);
                }
            }

            scene.splat_data.has_splats = true;
            std::cout << "[info] Gaussian splats: " << scene.splat_data.num_splats
                      << " splats, SH degree " << scene.splat_data.sh_degree << std::endl;

            // Only process the first gaussian splatting primitive
            break;
        }
        if (scene.splat_data.has_splats) break;
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
