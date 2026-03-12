"""Generate a test .glb with both mesh triangles AND Gaussian splats (hybrid scene)."""

import struct
import json
import math
import sys


def _logit(p):
    p = max(1e-7, min(1.0 - 1e-7, p))
    return math.log(p / (1.0 - p))


def generate_hybrid(output_path):
    """Create a GLB with a colored cube mesh + surrounding splat cloud."""

    # --- Cube mesh data (8 vertices, 12 triangles = 36 indices) ---
    cube_half = 0.3  # cube from -0.3 to +0.3
    cube_verts = [
        # pos(3) + normal(3) = 6 floats per vertex
        # Front face (z+)
        (-cube_half, -cube_half,  cube_half,  0, 0, 1),
        ( cube_half, -cube_half,  cube_half,  0, 0, 1),
        ( cube_half,  cube_half,  cube_half,  0, 0, 1),
        (-cube_half,  cube_half,  cube_half,  0, 0, 1),
        # Back face (z-)
        ( cube_half, -cube_half, -cube_half,  0, 0, -1),
        (-cube_half, -cube_half, -cube_half,  0, 0, -1),
        (-cube_half,  cube_half, -cube_half,  0, 0, -1),
        ( cube_half,  cube_half, -cube_half,  0, 0, -1),
        # Top face (y+)
        (-cube_half,  cube_half,  cube_half,  0, 1, 0),
        ( cube_half,  cube_half,  cube_half,  0, 1, 0),
        ( cube_half,  cube_half, -cube_half,  0, 1, 0),
        (-cube_half,  cube_half, -cube_half,  0, 1, 0),
        # Bottom face (y-)
        (-cube_half, -cube_half, -cube_half,  0, -1, 0),
        ( cube_half, -cube_half, -cube_half,  0, -1, 0),
        ( cube_half, -cube_half,  cube_half,  0, -1, 0),
        (-cube_half, -cube_half,  cube_half,  0, -1, 0),
        # Right face (x+)
        ( cube_half, -cube_half,  cube_half,  1, 0, 0),
        ( cube_half, -cube_half, -cube_half,  1, 0, 0),
        ( cube_half,  cube_half, -cube_half,  1, 0, 0),
        ( cube_half,  cube_half,  cube_half,  1, 0, 0),
        # Left face (x-)
        (-cube_half, -cube_half, -cube_half, -1, 0, 0),
        (-cube_half, -cube_half,  cube_half, -1, 0, 0),
        (-cube_half,  cube_half,  cube_half, -1, 0, 0),
        (-cube_half,  cube_half, -cube_half, -1, 0, 0),
    ]
    cube_indices = [
        0,1,2, 0,2,3,     # front
        4,5,6, 4,6,7,     # back
        8,9,10, 8,10,11,  # top
        12,13,14, 12,14,15, # bottom
        16,17,18, 16,18,19, # right
        20,21,22, 20,22,23, # left
    ]

    # Pack mesh vertex data
    mesh_pos_data = bytearray()
    mesh_norm_data = bytearray()
    pos_min = [float('inf')] * 3
    pos_max = [float('-inf')] * 3
    for v in cube_verts:
        mesh_pos_data += struct.pack('<3f', v[0], v[1], v[2])
        mesh_norm_data += struct.pack('<3f', v[3], v[4], v[5])
        for j in range(3):
            pos_min[j] = min(pos_min[j], v[j])
            pos_max[j] = max(pos_max[j], v[j])

    mesh_idx_data = bytearray()
    for idx in cube_indices:
        mesh_idx_data += struct.pack('<H', idx)  # uint16

    num_mesh_verts = len(cube_verts)
    num_mesh_indices = len(cube_indices)

    # --- Splat data: ring of splats around the cube ---
    num_splats = 50
    splat_pos_data = bytearray()
    splat_rot_data = bytearray()
    splat_scale_data = bytearray()
    splat_opa_data = bytearray()
    splat_sh_data = bytearray()

    log_scale = math.log(0.05)
    opacity_linear = 0.9  # KHR linear opacity

    splat_positions = []
    for i in range(num_splats):
        angle = 2.0 * math.pi * i / num_splats
        radius = 0.8
        x = radius * math.cos(angle)
        y = 0.0  # ring at y=0
        z = radius * math.sin(angle)
        splat_positions.append((x, y, z))

        splat_pos_data += struct.pack('<3f', x, y, z)
        splat_rot_data += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)
        splat_scale_data += struct.pack('<3f', log_scale, log_scale, log_scale)
        splat_opa_data += struct.pack('<f', opacity_linear)  # KHR linear format

        # Color: cycle through green/yellow
        sh_c0 = 0.28209479177387814
        t = i / num_splats
        r = t / sh_c0
        g = 0.8 / sh_c0
        b = (0.2 * (1 - t)) / sh_c0
        splat_sh_data += struct.pack('<3f', r, g, b)

    # Compute splat position bounds
    splat_pos_min = [float('inf')] * 3
    splat_pos_max = [float('-inf')] * 3
    for x, y, z in splat_positions:
        splat_pos_min = [min(splat_pos_min[j], v) for j, v in enumerate([x, y, z])]
        splat_pos_max = [max(splat_pos_max[j], v) for j, v in enumerate([x, y, z])]

    # --- Build buffer layout ---
    # All data concatenated into one binary buffer
    # Order: mesh_pos | mesh_norm | mesh_idx | splat_pos | splat_rot | splat_scale | splat_opa | splat_sh
    sections = [
        ("mesh_pos",    mesh_pos_data),
        ("mesh_norm",   mesh_norm_data),
        ("mesh_idx",    mesh_idx_data),
        ("splat_pos",   splat_pos_data),
        ("splat_rot",   splat_rot_data),
        ("splat_scale", splat_scale_data),
        ("splat_opa",   splat_opa_data),
        ("splat_sh",    splat_sh_data),
    ]

    buffer_bin = bytearray()
    offsets = {}
    for name, data in sections:
        # Align to 4 bytes
        while len(buffer_bin) % 4 != 0:
            buffer_bin += b'\x00'
        offsets[name] = len(buffer_bin)
        buffer_bin += data

    total_buffer = len(buffer_bin)

    # Buffer views (one per section)
    buffer_views = []
    for name, data in sections:
        bv = {"buffer": 0, "byteOffset": offsets[name], "byteLength": len(data)}
        if name == "mesh_idx":
            bv["target"] = 34963  # ELEMENT_ARRAY_BUFFER
        elif name in ("mesh_pos", "mesh_norm"):
            bv["target"] = 34962  # ARRAY_BUFFER
            bv["byteStride"] = 12
        buffer_views.append(bv)

    # Accessors
    # 0: mesh positions (VEC3)
    # 1: mesh normals (VEC3)
    # 2: mesh indices (SCALAR, uint16)
    # 3: splat positions (VEC3)
    # 4: splat rotations (VEC4)
    # 5: splat scales (VEC3)
    # 6: splat opacities (SCALAR)
    # 7: splat SH (VEC3)
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": num_mesh_verts, "type": "VEC3",
         "min": pos_min, "max": pos_max},
        {"bufferView": 1, "componentType": 5126, "count": num_mesh_verts, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5123, "count": num_mesh_indices, "type": "SCALAR"},
        {"bufferView": 3, "componentType": 5126, "count": num_splats, "type": "VEC3",
         "min": splat_pos_min, "max": splat_pos_max},
        {"bufferView": 4, "componentType": 5126, "count": num_splats, "type": "VEC4"},
        {"bufferView": 5, "componentType": 5126, "count": num_splats, "type": "VEC3"},
        {"bufferView": 6, "componentType": 5126, "count": num_splats, "type": "SCALAR"},
        {"bufferView": 7, "componentType": 5126, "count": num_splats, "type": "VEC3"},
    ]

    # Meshes: one mesh with TWO primitives (triangles + points/splats)
    meshes = [{
        "primitives": [
            # Primitive 0: triangle mesh (cube)
            {
                "mode": 4,  # TRIANGLES
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": 2,
            },
            # Primitive 1: Gaussian splats (POINTS)
            {
                "mode": 0,  # POINTS
                "attributes": {
                    "POSITION": 3,
                    "_ROTATION": 4,
                    "_SCALE": 5,
                    "_OPACITY": 6,
                    "_SH_0": 7,
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "attributes": {
                            "ROTATION": 4,
                            "SCALE": 5,
                            "OPACITY": 6,
                        },
                        "sh": [{"coefficients": 7, "degree": 0}],
                    }
                }
            },
        ]
    }]

    # Material for the cube (simple gray PBR)
    materials = [{
        "name": "cube_material",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.2, 0.2, 1.0],  # red cube
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
        }
    }]
    # Assign material to the mesh primitive
    meshes[0]["primitives"][0]["material"] = 0

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-generate-hybrid"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_buffer}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": materials,
        "meshes": meshes,
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    # Encode GLB
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('ascii')
    while len(buffer_bin) % 4 != 0:
        buffer_bin += b'\x00'

    glb_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_bin)
    header = struct.pack('<III', 0x46546C67, 2, glb_length)
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(buffer_bin), 0x004E4942) + bytes(buffer_bin)

    with open(output_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    print(f"Generated hybrid GLB: {output_path}")
    print(f"  Cube mesh: {num_mesh_verts} vertices, {num_mesh_indices // 3} triangles")
    print(f"  Splat ring: {num_splats} splats (SH degree 0)")
    print(f"  GLB size: {glb_length} bytes")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "tests/assets/hybrid_mesh_splats.glb"
    generate_hybrid(output)
