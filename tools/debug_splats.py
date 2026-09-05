"""Generate a diagnostic splat scene: 5 large colored splats in known positions."""

import struct
import json
import sys


def generate_debug_splats(output_path):
    SH_C0 = 0.28209479177387814

    # 5 splats: red, green, blue, yellow, white
    # Positions in Y-up space, centered at origin
    splats = [
        # (x, y, z, r, g, b, scale)
        (0.0, 0.0, 0.0,    1.0, 0.0, 0.0, 0.15),   # center: red
        (-0.4, 0.0, 0.0,   0.0, 1.0, 0.0, 0.12),   # left: green
        (0.4, 0.0, 0.0,    0.0, 0.0, 1.0, 0.12),   # right: blue
        (0.0, 0.3, 0.0,    1.0, 1.0, 0.0, 0.10),   # top: yellow
        (0.0, -0.3, 0.0,   1.0, 1.0, 1.0, 0.10),   # bottom: white
    ]

    n = len(splats)
    pos_data = bytearray()
    rot_data = bytearray()
    scale_data = bytearray()
    opacity_data = bytearray()
    sh_data = bytearray()

    pos_min = [float('inf')] * 3
    pos_max = [float('-inf')] * 3

    for x, y, z, r, g, b, s in splats:
        pos_data += struct.pack('<3f', x, y, z)
        pos_min = [min(pos_min[j], v) for j, v in enumerate([x, y, z])]
        pos_max = [max(pos_max[j], v) for j, v in enumerate([x, y, z])]

        # Identity quaternion (XYZW, scalar-last)
        rot_data += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)

        # Linear scale (ratified KHR_gaussian_splatting spec, isotropic)
        scale_data += struct.pack('<3f', s, s, s)

        # Linear opacity (ratified KHR_gaussian_splatting spec, high opacity)
        opacity_data += struct.pack('<f', 0.99)

        # SH DC: color = SH_C0 * sh_dc + 0.5, so sh_dc = (color - 0.5) / SH_C0
        sh_r = (r - 0.5) / SH_C0
        sh_g = (g - 0.5) / SH_C0
        sh_b = (b - 0.5) / SH_C0
        sh_data += struct.pack('<3f', sh_r, sh_g, sh_b)

    # Build GLB
    buffers = [pos_data, rot_data, scale_data, opacity_data, sh_data]
    offsets = []
    running = 0
    for part in buffers:
        offsets.append(running)
        running += len(part)
    total_size = running

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-debug-splats"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "extensionsRequired": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_size}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[i], "byteLength": len(buffers[i])}
            for i in range(5)
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": n, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5126, "count": n, "type": "VEC3"},
        ],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                # Ratified KHR_gaussian_splatting layout.
                "attributes": {
                    "POSITION": 0,
                    "KHR_gaussian_splatting:ROTATION": 1,
                    "KHR_gaussian_splatting:SCALE": 2,
                    "KHR_gaussian_splatting:OPACITY": 3,
                    "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4,
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "kernel": "ellipse",
                        "colorSpace": "srgb_rec709_display",
                        "sortingMethod": "cameraDistance",
                        "projection": "perspective",
                    }
                }
            }]
        }],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_str = json.dumps(gltf_json, separators=(',', ':'))
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('ascii')

    bin_data = bytearray()
    for part in buffers:
        bin_data += part
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'

    glb_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack('<III', 0x46546C67, 2, glb_length)
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data

    with open(output_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    print(f"Generated {output_path}: {n} debug splats")
    for x, y, z, r, g, b, s in splats:
        print(f"  pos=({x},{y},{z}) color=({r},{g},{b}) scale={s}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "debug_splats.glb"
    generate_debug_splats(output)
