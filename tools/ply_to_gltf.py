"""Convert 3DGS PLY files to glTF with KHR_gaussian_splatting extension."""

import struct
import sys
import json


def read_ply(path):
    """Read a 3DGS PLY file. Returns dict of arrays."""
    with open(path, 'rb') as f:
        # Parse header
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError(f"Not a PLY file: {line!r}")

        properties = []
        vertex_count = 0
        is_binary_le = False

        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            parts = line.split()
            if parts[0] == 'format':
                if parts[1] == 'binary_little_endian':
                    is_binary_le = True
                elif parts[1] == 'ascii':
                    is_binary_le = False
                else:
                    raise ValueError(f"Unsupported PLY format: {parts[1]}")
            elif parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2])
            elif parts[0] == 'property':
                dtype = parts[1]
                name = parts[2]
                properties.append((name, dtype))

        if vertex_count == 0:
            raise ValueError("No vertices in PLY file")

        # Build struct format
        dtype_map = {
            'float': 'f', 'float32': 'f',
            'double': 'd', 'float64': 'd',
            'uchar': 'B', 'uint8': 'B',
            'int': 'i', 'int32': 'i',
            'short': 'h', 'int16': 'h',
        }
        fmt_chars = []
        for name, dtype in properties:
            if dtype not in dtype_map:
                raise ValueError(f"Unsupported PLY property type: {dtype}")
            fmt_chars.append(dtype_map[dtype])

        struct_fmt = '<' + ''.join(fmt_chars)
        struct_size = struct.calcsize(struct_fmt)

        # Read all vertex data
        result = {name: [] for name, _ in properties}

        if is_binary_le:
            for _ in range(vertex_count):
                raw = f.read(struct_size)
                if len(raw) != struct_size:
                    raise ValueError("Unexpected end of binary PLY data")
                values = struct.unpack(struct_fmt, raw)
                for val, (name, _) in zip(values, properties):
                    result[name].append(float(val))
        else:
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                values = line.split()
                for val, (name, _) in zip(values, properties):
                    result[name].append(float(val))

    result['_vertex_count'] = vertex_count
    result['_properties'] = [name for name, _ in properties]
    return result


def _detect_sh_degree(property_names):
    """Detect SH degree from f_rest_N property names."""
    rest_count = sum(1 for n in property_names if n.startswith('f_rest_'))
    if rest_count == 0:
        return 0
    # Each degree l has (2l+1) coefficients per channel (3 channels: RGB)
    # degree 0: 3 DC coeffs (f_dc_0..2) -- always present
    # degree 1: 9 rest coeffs (f_rest_0..8)
    # degree 2: 15 more (f_rest_9..23)
    # degree 3: 21 more (f_rest_24..44)
    # Total rest: 3 * sum(2l+1 for l=1..degree)
    for degree in range(1, 5):
        expected = 3 * sum(2 * l + 1 for l in range(1, degree + 1))
        if rest_count == expected:
            return degree
    # Fallback: use as many as fit
    return min(3, rest_count // 9)


def ply_to_gltf(ply_path, gltf_path, sh_degree=None):
    """Convert PLY to glTF (.glb) with KHR_gaussian_splatting."""
    data = read_ply(ply_path)
    n = data['_vertex_count']
    props = data['_properties']
    print(f"Read {n} splats, properties: {len(props)}")

    if sh_degree is None:
        sh_degree = _detect_sh_degree(props)
    print(f"SH degree: {sh_degree}")

    # Extract positions (Y-up conversion: swap Y and Z)
    pos_buf = bytearray()
    for i in range(n):
        x = data['x'][i]
        y = data['z'][i]   # PLY Z -> glTF Y (up)
        z = -data['y'][i]  # PLY Y -> glTF -Z (forward)
        pos_buf += struct.pack('<3f', x, y, z)

    # Extract rotations: PLY stores WXYZ, glTF KHR_gaussian_splatting uses XYZW
    rot_buf = bytearray()
    for i in range(n):
        w = data['rot_0'][i]
        rx = data['rot_1'][i]
        ry = data['rot_2'][i]
        rz = data['rot_3'][i]
        # Normalize
        length = (w*w + rx*rx + ry*ry + rz*rz) ** 0.5
        if length > 1e-8:
            w /= length; rx /= length; ry /= length; rz /= length
        # Coordinate swap: rotate the quaternion to match Y-up
        # Apply the same axis swap as positions: (x, z, -y)
        rot_buf += struct.pack('<4f', rx, rz, -ry, w)

    # Extract scales (already in log-space)
    scale_buf = bytearray()
    for i in range(n):
        sx = data['scale_0'][i]
        sy = data['scale_1'][i]
        sz = data['scale_2'][i]
        # Apply same axis reorder as positions
        scale_buf += struct.pack('<3f', sx, sz, sy)

    # Extract opacity (already in logit-space)
    opa_buf = bytearray()
    for i in range(n):
        opa_buf += struct.pack('<f', data['opacity'][i])

    # Extract SH coefficients
    # DC band (degree 0): f_dc_0, f_dc_1, f_dc_2 -> vec3
    sh_dc_buf = bytearray()
    for i in range(n):
        r = data['f_dc_0'][i]
        g = data['f_dc_1'][i]
        b = data['f_dc_2'][i]
        sh_dc_buf += struct.pack('<3f', r, g, b)

    # Higher degree SH bands
    sh_rest_bufs = []
    if sh_degree >= 1:
        rest_names = sorted(
            [p for p in props if p.startswith('f_rest_')],
            key=lambda p: int(p.split('_')[-1])
        )
        coeffs_per_degree = []
        offset = 0
        for l in range(1, sh_degree + 1):
            count = 3 * (2 * l + 1)  # 3 channels * (2l+1) coefficients
            degree_names = rest_names[offset:offset + count]
            offset += count

            buf = bytearray()
            for i in range(n):
                for name in degree_names:
                    buf += struct.pack('<f', data[name][i])
            sh_rest_bufs.append((l, count, buf))

    # Build buffer views and accessors
    buffer_parts = [pos_buf, rot_buf, scale_buf, opa_buf, sh_dc_buf]
    for _, _, buf in sh_rest_bufs:
        buffer_parts.append(buf)

    offsets = []
    running = 0
    for part in buffer_parts:
        offsets.append(running)
        running += len(part)
    total_size = running

    buffer_views = []
    for idx, part in enumerate(buffer_parts):
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offsets[idx],
            "byteLength": len(part),
        })

    # Compute position min/max for glTF compliance
    pos_min = [float('inf')] * 3
    pos_max = [float('-inf')] * 3
    for i in range(n):
        vals = struct.unpack_from('<3f', pos_buf, i * 12)
        for j in range(3):
            if vals[j] < pos_min[j]:
                pos_min[j] = vals[j]
            if vals[j] > pos_max[j]:
                pos_max[j] = vals[j]

    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
         "min": pos_min, "max": pos_max},   # positions
        {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4"},   # rotations
        {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},   # scales
        {"bufferView": 3, "componentType": 5126, "count": n, "type": "SCALAR"}, # opacities
        {"bufferView": 4, "componentType": 5126, "count": n, "type": "VEC3"},   # sh_dc
    ]

    sh_entries = [{"coefficients": 4, "degree": 0}]
    for idx, (l, count, _) in enumerate(sh_rest_bufs):
        acc_idx = 5 + idx
        # Each degree has count floats per splat; represent as SCALAR array
        accessors.append({
            "bufferView": 5 + idx,
            "componentType": 5126,
            "count": n * count,
            "type": "SCALAR",
        })
        sh_entries.append({"coefficients": acc_idx, "degree": l})

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-ply-to-gltf"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_size}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": dict([
                    ("POSITION", 0),
                    ("_ROTATION", 1),
                    ("_SCALE", 2),
                    ("_OPACITY", 3),
                ] + [("_SH_%d" % e["degree"], e["coefficients"]) for e in sh_entries]),
                "extensions": {
                    "KHR_gaussian_splatting": {
                        "attributes": {
                            "ROTATION": 1,
                            "SCALE": 2,
                            "OPACITY": 3,
                        },
                        "sh": sh_entries,
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
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('ascii')

    bin_data = bytearray()
    for part in buffer_parts:
        bin_data += part
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'

    glb_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack('<III', 0x46546C67, 2, glb_length)
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data

    with open(gltf_path, 'wb') as f:
        f.write(header + json_chunk + bin_chunk)

    print(f"Written {gltf_path}: {n} splats, SH degree {sh_degree}, {glb_length} bytes")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m tools.ply_to_gltf input.ply output.glb [--sh-degree N]")
        sys.exit(1)

    sh_deg = None
    if len(sys.argv) >= 5 and sys.argv[3] == '--sh-degree':
        sh_deg = int(sys.argv[4])

    ply_to_gltf(sys.argv[1], sys.argv[2], sh_degree=sh_deg)
