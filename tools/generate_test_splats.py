"""Generate procedural test Gaussian splat scenes as .glb files."""

import struct
import json
import math
import sys


def _fibonacci_sphere(n):
    """Generate n points on a unit sphere using Fibonacci spiral sampling."""
    points = []
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = 2.0 * math.pi * i / golden_ratio
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        points.append((x, y, z))
    return points


def _logit(p):
    """Inverse sigmoid: logit(p) = log(p / (1 - p))."""
    p = max(1e-7, min(1.0 - 1e-7, p))
    return math.log(p / (1.0 - p))


def generate_test_splats(output_path, num_splats=1000, pattern="sphere"):
    """Generate a test .glb with KHR_gaussian_splatting data."""
    points = _fibonacci_sphere(num_splats)

    # Pack binary data
    pos_data = bytearray()
    rot_data = bytearray()
    scale_data = bytearray()
    opacity_data = bytearray()
    sh_data = bytearray()

    log_scale = math.log(0.02)
    opacity_logit = _logit(0.95)

    for i, (x, y, z) in enumerate(points):
        # Position: vec3 float
        pos_data += struct.pack('<3f', x, y, z)
        # Rotation: xyzw quaternion (identity = [0, 0, 0, 1])
        rot_data += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)
        # Scale: log-space vec3
        scale_data += struct.pack('<3f', log_scale, log_scale, log_scale)
        # Opacity: logit-space scalar
        opacity_data += struct.pack('<f', opacity_logit)
        # SH degree 0: DC coefficient = color / SH_C0 where SH_C0 = 0.28209479
        # Red gradient based on latitude (z coordinate): red at top, blue at bottom
        latitude = (z + 1.0) / 2.0  # 0..1
        sh_c0 = 0.28209479177387814
        r = latitude / sh_c0
        g = 0.2 / sh_c0
        b = (1.0 - latitude) / sh_c0
        sh_data += struct.pack('<3f', r, g, b)

    # Build accessor metadata
    # Buffer layout: positions | rotations | scales | opacities | sh_dc
    pos_offset = 0
    pos_size = len(pos_data)
    rot_offset = pos_offset + pos_size
    rot_size = len(rot_data)
    scale_offset = rot_offset + rot_size
    scale_size = len(scale_data)
    opa_offset = scale_offset + scale_size
    opa_size = len(opacity_data)
    sh_offset = opa_offset + opa_size
    sh_size = len(sh_data)
    total_buffer = pos_size + rot_size + scale_size + opa_size + sh_size

    buffer_bin = pos_data + rot_data + scale_data + opacity_data + sh_data

    # Compute position bounding box for POSITION accessor (required by spec)
    pos_min = [float('inf')] * 3
    pos_max = [float('-inf')] * 3
    for x, y, z in points:
        pos_min = [min(pos_min[j], v) for j, v in enumerate([x, y, z])]
        pos_max = [max(pos_max[j], v) for j, v in enumerate([x, y, z])]

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-generate-test-splats"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_buffer}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset,   "byteLength": pos_size},
            {"buffer": 0, "byteOffset": rot_offset,    "byteLength": rot_size},
            {"buffer": 0, "byteOffset": scale_offset,  "byteLength": scale_size},
            {"buffer": 0, "byteOffset": opa_offset,    "byteLength": opa_size},
            {"buffer": 0, "byteOffset": sh_offset,     "byteLength": sh_size},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": num_splats, "type": "VEC3",
             "min": pos_min, "max": pos_max},   # positions
            {"bufferView": 1, "componentType": 5126, "count": num_splats, "type": "VEC4"},   # rotations
            {"bufferView": 2, "componentType": 5126, "count": num_splats, "type": "VEC3"},   # scales
            {"bufferView": 3, "componentType": 5126, "count": num_splats, "type": "SCALAR"}, # opacities
            {"bufferView": 4, "componentType": 5126, "count": num_splats, "type": "VEC3"},   # sh_dc
        ],
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": {
                    "POSITION": 0,
                    "_ROTATION": 1,
                    "_SCALE": 2,
                    "_OPACITY": 3,
                    "_SH_0": 4,
                },
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "attributes": {
                            "ROTATION": 1,
                            "SCALE": 2,
                            "OPACITY": 3,
                        },
                        "sh": [{"coefficients": 4, "degree": 0}],
                    }
                }
            }]
        }],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    # Encode GLB
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    # Pad JSON to 4-byte alignment
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('ascii')
    # Pad binary to 4-byte alignment
    while len(buffer_bin) % 4 != 0:
        buffer_bin += b'\x00'

    # GLB header: magic + version + length
    glb_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_bin)
    header = struct.pack('<III', 0x46546C67, 2, glb_length)  # glTF magic, version 2
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes  # JSON chunk
    bin_chunk = struct.pack('<II', len(buffer_bin), 0x004E4942) + buffer_bin    # BIN chunk

    with open(output_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    print(f"  {num_splats} splats, SH degree 0, pattern={pattern}")
    print(f"  Buffer: {total_buffer} bytes, GLB: {glb_length} bytes")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "test_splats.glb"
    generate_test_splats(output)
    print(f"Generated {output}")
