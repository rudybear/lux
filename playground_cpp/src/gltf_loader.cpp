#include "gltf_loader.h"

#include <algorithm>
#include <functional>
#include <fstream>
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
            }
            if (pbr.metallic_roughness_texture.texture) {
                std::cout << "[info] Extracting metallic_roughness texture for material: " << gmat.name << std::endl;
                gmat.metallic_roughness_tex = extractTextureData(pbr.metallic_roughness_texture);
            }
        }

        // Normal map
        if (mat.normal_texture.texture) {
            std::cout << "[info] Extracting normal texture for material: " << gmat.name << std::endl;
            gmat.normal_tex = extractTextureData(mat.normal_texture);
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

        scene.materials.push_back(gmat);
    }

    if (scene.materials.empty()) {
        scene.materials.push_back(GltfMaterial{"default"});
    }

    // --- Meshes ---
    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        auto& mesh = data->meshes[mi];
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

            if (node.meshIndex >= 0 && node.meshIndex < static_cast<int>(scene.meshes.size())) {
                DrawItem item;
                item.worldTransform = node.worldTransform;
                item.meshIndex = node.meshIndex;
                item.materialIndex = scene.meshes[node.meshIndex].materialIndex;
                items.push_back(item);
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
