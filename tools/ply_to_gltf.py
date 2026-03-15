"""Convert 3DGS PLY files to glTF with KHR_gaussian_splatting extension.

Enhanced converter with:
- Coordinate system conversion (Z-up -> Y-up)
- Opacity transform (logit -> linear sigmoid)
- Scales kept in log-space per KHR spec (no transform)
- SH channel reordering (3DGS channel-first -> KHR coefficient-first VEC3)
- KHR-conformant attribute naming (KHR_gaussian_splatting:SH_DEGREE_N_COEF_M)
- SH degree auto-detection
- Round-trip verification
- Batch processing
- Info, centering, and decimation options
"""

import struct
import sys
import os
import json
import math
import argparse
import random

import numpy as np


# ---------------------------------------------------------------------------
# PLY reading
# ---------------------------------------------------------------------------

def read_ply(path):
    """Read a 3DGS PLY file. Returns dict of numpy arrays plus metadata.

    Returns a dict with:
      - Each property name -> numpy array of float64
      - '_vertex_count' -> int
      - '_properties' -> list of property name strings
      - '_property_types' -> list of (name, dtype) tuples
    """
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

        # Build struct format for binary reading
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

    # Convert lists to numpy arrays
    for name, _ in properties:
        result[name] = np.array(result[name], dtype=np.float64)

    result['_vertex_count'] = vertex_count
    result['_properties'] = [name for name, _ in properties]
    result['_property_types'] = properties
    return result


def read_ply_header(path):
    """Read only the PLY header. Returns (vertex_count, properties) without loading data."""
    with open(path, 'rb') as f:
        line = f.readline().decode('ascii').strip()
        if line != 'ply':
            raise ValueError(f"Not a PLY file: {line!r}")

        properties = []
        vertex_count = 0

        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            parts = line.split()
            if parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2])
            elif parts[0] == 'property':
                dtype = parts[1]
                name = parts[2]
                properties.append((name, dtype))

    return vertex_count, properties


# ---------------------------------------------------------------------------
# PLY writing (for round-trip verification)
# ---------------------------------------------------------------------------

def write_ply(path, data, vertex_count, properties):
    """Write a 3DGS PLY file from dict of numpy arrays.

    Args:
        path: Output file path.
        data: Dict mapping property names to numpy arrays.
        vertex_count: Number of vertices.
        properties: List of property name strings to write.
    """
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
    ]
    for name in properties:
        header_lines.append(f"property float {name}")
    header_lines.append("end_header")
    header_str = "\n".join(header_lines) + "\n"

    with open(path, 'wb') as f:
        f.write(header_str.encode('ascii'))
        for i in range(vertex_count):
            for name in properties:
                f.write(struct.pack('<f', float(data[name][i])))


# ---------------------------------------------------------------------------
# SH degree detection
# ---------------------------------------------------------------------------

def detect_sh_degree(property_names):
    """Detect SH degree from f_rest_N property names.

    Returns:
        int: SH degree (0-3). Returns 0 if no f_rest_* properties found.
    """
    rest_count = sum(1 for n in property_names if n.startswith('f_rest_'))
    if rest_count == 0:
        return 0
    # Each degree l has (2l+1) coefficients per channel (3 channels: RGB)
    # degree 0: 3 DC coeffs (f_dc_0..2) -- always present
    # degree 1: 9 rest coeffs (f_rest_0..8)
    # degree 2: 24 total rest (9+15) -> (f_rest_0..23)
    # degree 3: 45 total rest (9+15+21) -> (f_rest_0..44)
    # Total rest: 3 * sum(2l+1 for l=1..degree)
    for degree in range(1, 5):
        expected = 3 * sum(2 * l + 1 for l in range(1, degree + 1))
        if rest_count == expected:
            return degree
    # Fallback: use as many as fit
    return min(3, rest_count // 9)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Sigmoid function: 1 / (1 + exp(-x)). Handles large negative inputs."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def logit(p):
    """Inverse sigmoid: log(p / (1 - p))."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def auto_detect_opacity_space(values):
    """Auto-detect if opacity values are already in linear [0,1] space.

    Returns True if values appear to already be linear (all in [0,1]).
    Returns False if values appear to be in logit space.
    """
    return bool(np.all(values >= 0.0) and np.all(values <= 1.0))


def auto_detect_scale_space(values):
    """Auto-detect if scale values are already in linear space.

    Returns True if all values are positive and < 10 (plausible linear scales).
    Returns False if values appear to be in log space (may have negatives).
    """
    return bool(np.all(values > 0.0) and np.all(values < 10.0))


def convert_positions_z_up_to_y_up(positions):
    """Convert positions from Z-up (3DGS) to Y-up (glTF).

    Applies -90 degree rotation around X axis:
        x' = x
        y' = z
        z' = -y
    """
    result = np.empty_like(positions)
    result[:, 0] = positions[:, 0]      # x' = x
    result[:, 1] = positions[:, 2]      # y' = z
    result[:, 2] = -positions[:, 1]     # z' = -y
    return result


def convert_quaternions_z_up_to_y_up(quaternions_wxyz):
    """Convert quaternions from Z-up (3DGS WXYZ) to Y-up (glTF XYZW).

    Input: (N, 4) array in WXYZ order (3DGS convention).
    Output: (N, 4) array in XYZW order (glTF convention) with Y-up axes.

    The coordinate system rotation is: (x, y, z) -> (x, z, -y).
    For quaternions, we apply the same axis permutation to the imaginary parts.
    """
    w = quaternions_wxyz[:, 0]
    qx = quaternions_wxyz[:, 1]
    qy = quaternions_wxyz[:, 2]
    qz = quaternions_wxyz[:, 3]

    # Normalize
    length = np.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    mask = length > 1e-8
    w = np.where(mask, w / length, w)
    qx = np.where(mask, qx / length, qx)
    qy = np.where(mask, qy / length, qy)
    qz = np.where(mask, qz / length, qz)

    # Apply axis swap: (qx, qy, qz) -> (qx, qz, -qy) and output as XYZW
    result = np.column_stack([qx, qz, -qy, w])
    return result


def convert_scales_z_up_to_y_up(scales):
    """Convert scales from Z-up to Y-up by reordering axes.

    Input:  (sx, sy, sz) in Z-up space
    Output: (sx, sz, sy) in Y-up space (y and z swapped)
    """
    result = np.empty_like(scales)
    result[:, 0] = scales[:, 0]     # x stays
    result[:, 1] = scales[:, 2]     # new y = old z
    result[:, 2] = scales[:, 1]     # new z = old y
    return result


# ---------------------------------------------------------------------------
# PLY info
# ---------------------------------------------------------------------------

def ply_info(path):
    """Print information about a PLY file without converting.

    Returns a dict with info for programmatic use.
    """
    data = read_ply(path)
    n = data['_vertex_count']
    props = data['_properties']
    sh_degree = detect_sh_degree(props)

    # Positions
    positions = np.column_stack([data['x'], data['y'], data['z']])
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    center = (pos_min + pos_max) / 2.0
    extent = pos_max - pos_min

    # Opacity
    opacities = data.get('opacity', np.array([]))
    opacity_auto_linear = auto_detect_opacity_space(opacities) if len(opacities) > 0 else None

    # Scales
    has_scale = 'scale_0' in data
    if has_scale:
        scales = np.column_stack([data['scale_0'], data['scale_1'], data['scale_2']])
        scale_auto_linear = auto_detect_scale_space(scales)
    else:
        scales = None
        scale_auto_linear = None

    info = {
        'splat_count': n,
        'property_count': len(props),
        'properties': props,
        'sh_degree': sh_degree,
        'position_min': pos_min.tolist(),
        'position_max': pos_max.tolist(),
        'center': center.tolist(),
        'extent': extent.tolist(),
        'opacity_auto_linear': opacity_auto_linear,
        'scale_auto_linear': scale_auto_linear,
    }

    print(f"PLY Info: {path}")
    print(f"  Splat count:    {n}")
    print(f"  Properties:     {len(props)}")
    print(f"  SH degree:      {sh_degree}")
    print(f"  Position min:   ({pos_min[0]:.4f}, {pos_min[1]:.4f}, {pos_min[2]:.4f})")
    print(f"  Position max:   ({pos_max[0]:.4f}, {pos_max[1]:.4f}, {pos_max[2]:.4f})")
    print(f"  Center:         ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"  Extent:         ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})")

    if len(opacities) > 0:
        print(f"  Opacity range:  [{opacities.min():.4f}, {opacities.max():.4f}]")
        if opacity_auto_linear:
            print(f"  Opacity space:  appears linear (all in [0,1])")
        else:
            print(f"  Opacity space:  appears logit (raw training output)")

    if has_scale:
        print(f"  Scale range:    [{scales.min():.4f}, {scales.max():.4f}]")
        if scale_auto_linear:
            print(f"  Scale space:    appears linear (positive, < 10)")
        else:
            print(f"  Scale space:    appears log (raw training output)")

    return info


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------

def decimate_data(data, target_count, method='opacity'):
    """Reduce splat count by keeping the N most opaque (or random) splats.

    Args:
        data: PLY data dict.
        target_count: Number of splats to keep.
        method: 'opacity' (keep highest opacity) or 'random'.

    Returns:
        New data dict with reduced splat count.
    """
    n = data['_vertex_count']
    if target_count >= n:
        return data

    if method == 'opacity' and 'opacity' in data:
        opacities = data['opacity']
        indices = np.argsort(-opacities)[:target_count]
        indices = np.sort(indices)  # Maintain original order
    else:
        indices = np.sort(np.random.choice(n, target_count, replace=False))

    result = {}
    for key, val in data.items():
        if key.startswith('_'):
            result[key] = val
        else:
            result[key] = val[indices]

    result['_vertex_count'] = target_count
    return result


# ---------------------------------------------------------------------------
# Center point cloud
# ---------------------------------------------------------------------------

def center_positions(data):
    """Center the point cloud at the origin (modify in place).

    Returns the offset that was subtracted.
    """
    positions = np.column_stack([data['x'], data['y'], data['z']])
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    data['x'] = data['x'] - center[0]
    data['y'] = data['y'] - center[1]
    data['z'] = data['z'] - center[2]
    return center


# ---------------------------------------------------------------------------
# GLB encoding
# ---------------------------------------------------------------------------

def encode_glb(gltf_json, buffer_parts):
    """Encode a glTF JSON and binary buffer parts into a GLB byte string."""
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

    return header + json_chunk + bin_chunk


# ---------------------------------------------------------------------------
# GLB decoding (for round-trip verification)
# ---------------------------------------------------------------------------

def load_glb(path):
    """Parse a .glb and return (gltf_json, bin_data)."""
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, f"Not a GLB file: {path}"

        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        assert chunk_type == 0x4E4F534A  # JSON
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes)

        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        assert chunk_type == 0x004E4942  # BIN
        bin_data = f.read(chunk_len)

    return gltf, bin_data


def read_accessor(gltf, bin_data, accessor_idx):
    """Read accessor data as a numpy array."""
    acc = gltf['accessors'][accessor_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)

    type_map = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    count = acc['count']
    components = type_map[acc['type']]

    assert acc['componentType'] == 5126, f"Unsupported component type: {acc['componentType']}"
    data = np.frombuffer(bin_data, dtype=np.float32,
                         count=count * components,
                         offset=offset)
    return data.reshape(count, components) if components > 1 else data.copy()


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def ply_to_gltf(ply_path, gltf_path, sh_degree=None,
                convert_coords=True, raw_opacity=False, raw_scale=False,
                do_center=False, decimate_count=None, quiet=False):
    """Convert PLY to glTF (.glb) with KHR_gaussian_splatting.

    Args:
        ply_path: Input PLY file path.
        gltf_path: Output GLB file path.
        sh_degree: SH degree override (None = auto-detect).
        convert_coords: If True, convert Z-up to Y-up (default True).
        raw_opacity: If True, skip opacity sigmoid transform.
        raw_scale: If True, skip scale exp transform.
        do_center: If True, center point cloud at origin.
        decimate_count: If set, reduce to this many splats.
        quiet: If True, suppress output.

    Returns:
        dict with conversion metadata.
    """
    data = read_ply(ply_path)
    n = data['_vertex_count']
    props = data['_properties']
    if not quiet:
        print(f"Read {n} splats, properties: {len(props)}")

    if sh_degree is None:
        sh_degree = detect_sh_degree(props)
    if not quiet:
        print(f"SH degree: {sh_degree}")

    # Optional centering
    offset = None
    if do_center:
        offset = center_positions(data)
        if not quiet:
            print(f"Centered: offset ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")

    # Optional decimation
    if decimate_count is not None and decimate_count < n:
        data = decimate_data(data, decimate_count)
        n = data['_vertex_count']
        if not quiet:
            print(f"Decimated to {n} splats")

    # Build position array
    positions = np.column_stack([data['x'], data['y'], data['z']])

    if convert_coords:
        positions = convert_positions_z_up_to_y_up(positions)
    pos_buf = positions.astype(np.float32).tobytes()

    # Build rotation array
    rot_wxyz = np.column_stack([
        data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']
    ])
    if convert_coords:
        rot_xyzw = convert_quaternions_z_up_to_y_up(rot_wxyz)
    else:
        # No coord convert, but still need WXYZ -> XYZW reorder
        w = rot_wxyz[:, 0]
        qx = rot_wxyz[:, 1]
        qy = rot_wxyz[:, 2]
        qz = rot_wxyz[:, 3]
        length = np.sqrt(w * w + qx * qx + qy * qy + qz * qz)
        mask = length > 1e-8
        w = np.where(mask, w / length, w)
        qx = np.where(mask, qx / length, qx)
        qy = np.where(mask, qy / length, qy)
        qz = np.where(mask, qz / length, qz)
        rot_xyzw = np.column_stack([qx, qy, qz, w])
    rot_buf = rot_xyzw.astype(np.float32).tobytes()

    # Build scale array
    scales = np.column_stack([data['scale_0'], data['scale_1'], data['scale_2']])
    if convert_coords:
        scales = convert_scales_z_up_to_y_up(scales)

    # KHR_gaussian_splatting spec stores scales in log-space (same as 3DGS PLY).
    # No transform needed — pass through as-is.  The compute shader applies exp().
    if not quiet:
        print("Scales kept in log-space (KHR spec)")
    scale_buf = scales.astype(np.float32).tobytes()

    # Build opacity array
    opacities = data['opacity'].copy()

    # Auto-detect or apply opacity transform
    if not raw_opacity:
        if auto_detect_opacity_space(opacities):
            if not quiet:
                print("Opacities appear already linear, keeping as-is")
        else:
            opacities = sigmoid(opacities)
            if not quiet:
                print("Applied sigmoid() opacity transform")
    opa_buf = opacities.astype(np.float32).tobytes()

    # SH DC band (degree 0): f_dc_0, f_dc_1, f_dc_2 → 1 VEC3 per splat
    sh_dc = np.column_stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']])
    sh_dc_buf = sh_dc.astype(np.float32).tobytes()

    # Higher degree SH bands — reorder from 3DGS channel-first to KHR coefficient-first.
    #
    # 3DGS PLY stores f_rest_* in GLOBAL channel-first order (due to .transpose(1,2)
    # in the training code's save):
    #   [R_coeff0, R_coeff1, ..., R_coeffK, G_coeff0, ..., G_coeffK, B_coeff0, ..., B_coeffK]
    # where K = total higher-order coefficients across ALL degrees.
    #
    # KHR_gaussian_splatting spec stores per-coefficient VEC3:
    #   each coefficient has (R, G, B) together as a VEC3 attribute.
    sh_coeff_bufs = []  # list of (degree, coef_within_degree, bytes_buffer)
    if sh_degree >= 1:
        rest_names = sorted(
            [p for p in props if p.startswith('f_rest_')],
            key=lambda p: int(p.split('_')[-1])
        )
        total_rest = len(rest_names)
        coeffs_per_channel = total_rest // 3

        # Read all f_rest as (N, total_rest) array
        rest_data = np.column_stack([data[name] for name in rest_names])

        # Reshape: (N, 3_channels, coeffs_per_channel) — channel-first layout
        rest_data = rest_data.reshape(n, 3, coeffs_per_channel)

        # Transpose to coefficient-first: (N, coeffs_per_channel, 3_rgb)
        rest_data = rest_data.transpose(0, 2, 1)

        # Split by degree and create individual VEC3 buffers per coefficient
        coeff_idx = 0
        for l in range(1, sh_degree + 1):
            num_coeffs = 2 * l + 1
            for c in range(num_coeffs):
                coeff_vec3 = rest_data[:, coeff_idx, :]  # (N, 3)
                buf = coeff_vec3.astype(np.float32).tobytes()
                sh_coeff_bufs.append((l, c, buf))
                coeff_idx += 1

        if not quiet:
            print(f"Reordered {total_rest} SH coefficients: channel-first -> "
                  f"{len(sh_coeff_bufs)} coefficient-first VEC3 buffers")

    # Build buffer views and accessors
    buffer_parts = [pos_buf, rot_buf, scale_buf, opa_buf, sh_dc_buf]
    for _, _, buf in sh_coeff_bufs:
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
    pos_min = positions.min(axis=0).astype(np.float32).tolist()
    pos_max = positions.max(axis=0).astype(np.float32).tolist()

    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
         "min": pos_min, "max": pos_max},   # positions
        {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC4"},   # rotations
        {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},   # scales
        {"bufferView": 3, "componentType": 5126, "count": n, "type": "SCALAR"}, # opacities
        {"bufferView": 4, "componentType": 5126, "count": n, "type": "VEC3"},   # sh_dc
    ]

    for idx in range(len(sh_coeff_bufs)):
        accessors.append({
            "bufferView": 5 + idx,
            "componentType": 5126,
            "count": n,
            "type": "VEC3",
        })

    # Build KHR-conformant primitive attributes
    attributes = {
        "POSITION": 0,
        "KHR_gaussian_splatting:ROTATION": 1,
        "KHR_gaussian_splatting:SCALE": 2,
        "KHR_gaussian_splatting:OPACITY": 3,
        "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4,
    }
    for idx, (l, c, _) in enumerate(sh_coeff_bufs):
        attributes[f"KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{c}"] = 5 + idx

    # Build extension "sh" array with per-coefficient accessor indices per degree.
    # coefficients is an array of accessor indices (one VEC3 accessor per coefficient).
    sh_entries = [{"degree": 0, "coefficients": [4]}]
    for l in range(1, sh_degree + 1):
        degree_acc_indices = [
            5 + idx for idx, (dl, _, _) in enumerate(sh_coeff_bufs) if dl == l
        ]
        sh_entries.append({"degree": l, "coefficients": degree_acc_indices})

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-ply-to-gltf"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_size}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{
            "primitives": [{
                "mode": 0,
                "attributes": attributes,
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

    glb_bytes = encode_glb(gltf_json, buffer_parts)
    with open(gltf_path, 'wb') as f:
        f.write(glb_bytes)

    glb_length = len(glb_bytes)
    if not quiet:
        print(f"Written {gltf_path}: {n} splats, SH degree {sh_degree}, {glb_length} bytes")

    return {
        'splat_count': n,
        'sh_degree': sh_degree,
        'glb_size': glb_length,
        'position_min': pos_min,
        'position_max': pos_max,
        'center_offset': offset.tolist() if offset is not None else None,
    }


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def verify_round_trip(ply_path, glb_path, convert_coords=True,
                      raw_opacity=False, raw_scale=False, threshold=1e-5,
                      quiet=False):
    """Verify PLY -> glTF conversion by reading back and comparing.

    Reads the original PLY and the generated GLB, applies inverse transforms
    to the GLB data, and compares all attributes numerically.

    Returns:
        dict mapping attribute names to max absolute errors.
        Raises ValueError if any error exceeds threshold.
    """
    # Read original PLY
    orig = read_ply(ply_path)
    n_orig = orig['_vertex_count']

    # Read GLB
    gltf, bin_data = load_glb(glb_path)
    mesh = gltf['meshes'][0]
    prim = mesh['primitives'][0]
    ext = prim.get('extensions', {}).get('KHR_gaussian_splatting', {})
    attrs = prim['attributes']

    glb_positions = read_accessor(gltf, bin_data, attrs['POSITION'])
    glb_rotations = read_accessor(gltf, bin_data,
                                  attrs.get('_ROTATION', ext['attributes']['ROTATION']))
    glb_scales = read_accessor(gltf, bin_data,
                               attrs.get('_SCALE', ext['attributes']['SCALE']))
    glb_opacities = read_accessor(gltf, bin_data,
                                  attrs.get('_OPACITY', ext['attributes']['OPACITY']))

    n_glb = gltf['accessors'][attrs['POSITION']]['count']
    assert n_orig == n_glb, f"Vertex count mismatch: PLY={n_orig}, GLB={n_glb}"

    errors = {}

    # Compare positions
    orig_pos = np.column_stack([orig['x'], orig['y'], orig['z']]).astype(np.float32)
    if convert_coords:
        orig_pos_conv = convert_positions_z_up_to_y_up(orig_pos)
    else:
        orig_pos_conv = orig_pos
    errors['position'] = float(np.max(np.abs(glb_positions - orig_pos_conv)))

    # Compare rotations (glTF stores XYZW)
    orig_rot_wxyz = np.column_stack([
        orig['rot_0'], orig['rot_1'], orig['rot_2'], orig['rot_3']
    ]).astype(np.float32)
    if convert_coords:
        orig_rot_conv = convert_quaternions_z_up_to_y_up(orig_rot_wxyz.astype(np.float64)).astype(np.float32)
    else:
        w = orig_rot_wxyz[:, 0]
        qx = orig_rot_wxyz[:, 1]
        qy = orig_rot_wxyz[:, 2]
        qz = orig_rot_wxyz[:, 3]
        length = np.sqrt(w * w + qx * qx + qy * qy + qz * qz)
        mask = length > 1e-8
        w = np.where(mask, w / length, w)
        qx = np.where(mask, qx / length, qx)
        qy = np.where(mask, qy / length, qy)
        qz = np.where(mask, qz / length, qz)
        orig_rot_conv = np.column_stack([qx, qy, qz, w])
    errors['rotation'] = float(np.max(np.abs(glb_rotations - orig_rot_conv)))

    # Compare scales (kept in log-space per KHR spec, no transform)
    orig_scales = np.column_stack([
        orig['scale_0'], orig['scale_1'], orig['scale_2']
    ]).astype(np.float32)
    if convert_coords:
        orig_scales = convert_scales_z_up_to_y_up(orig_scales)
    errors['scale'] = float(np.max(np.abs(glb_scales - orig_scales)))

    # Compare opacities
    orig_opa = orig['opacity'].astype(np.float32)
    if not raw_opacity:
        if not auto_detect_opacity_space(orig_opa):
            orig_opa = sigmoid(orig_opa).astype(np.float32)
    errors['opacity'] = float(np.max(np.abs(glb_opacities - orig_opa)))

    # Compare SH DC
    sh_bands = ext.get('sh', [])
    if sh_bands:
        coeffs = sh_bands[0]['coefficients']
        # Handle both old (single index) and new (array of indices) formats
        sh0_idx = coeffs[0] if isinstance(coeffs, list) else coeffs
        glb_sh0 = read_accessor(gltf, bin_data, sh0_idx)
        orig_sh0 = np.column_stack([
            orig['f_dc_0'], orig['f_dc_1'], orig['f_dc_2']
        ]).astype(np.float32)
        errors['sh_dc'] = float(np.max(np.abs(glb_sh0 - orig_sh0)))

    # Report
    max_error = max(errors.values())
    if not quiet:
        print(f"\nRound-trip verification ({ply_path} -> {glb_path}):")
        for attr, err in sorted(errors.items()):
            status = "OK" if err <= threshold else "FAIL"
            print(f"  {attr:12s}: max error = {err:.2e}  [{status}]")
        print(f"  Overall: {'PASS' if max_error <= threshold else 'FAIL'} (threshold={threshold})")

    if max_error > threshold:
        raise ValueError(
            f"Round-trip verification failed: max error {max_error:.2e} > {threshold:.2e}. "
            f"Errors: {errors}"
        )

    return errors


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_convert(input_dir, output_dir=None, **kwargs):
    """Convert all PLY files in a directory to GLB.

    Args:
        input_dir: Directory containing .ply files.
        output_dir: Output directory (default: same as input_dir).
        **kwargs: Extra args passed to ply_to_gltf.

    Returns:
        List of (input_path, output_path, result_or_error) tuples.
    """
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.ply')
    ])

    if not ply_files:
        print(f"No .ply files found in {input_dir}")
        return []

    results = []
    for idx, filename in enumerate(ply_files, 1):
        ply_path = os.path.join(input_dir, filename)
        glb_name = os.path.splitext(filename)[0] + '.glb'
        glb_path = os.path.join(output_dir, glb_name)

        print(f"[{idx}/{len(ply_files)}] {filename} -> {glb_name}")
        try:
            result = ply_to_gltf(ply_path, glb_path, quiet=True, **kwargs)
            results.append((ply_path, glb_path, result))
            print(f"  OK: {result['splat_count']} splats, SH{result['sh_degree']}, "
                  f"{result['glb_size']} bytes")
        except Exception as e:
            results.append((ply_path, glb_path, e))
            print(f"  ERROR: {e}")

    # Summary
    ok = sum(1 for _, _, r in results if not isinstance(r, Exception))
    fail = len(results) - ok
    print(f"\nBatch complete: {ok} succeeded, {fail} failed, {len(results)} total")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert 3DGS PLY files to glTF (.glb) with KHR_gaussian_splatting extension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s input.ply output.glb
  %(prog)s input.ply output.glb --no-convert --raw-opacity --raw-scale
  %(prog)s --info input.ply
  %(prog)s --batch ./splats/ --output-dir ./glb/
  %(prog)s input.ply output.glb --verify
  %(prog)s input.ply output.glb --center --decimate 50000
""",
    )
    parser.add_argument('input', nargs='?', help='Input PLY file path')
    parser.add_argument('output', nargs='?', help='Output GLB file path')

    # Conversion flags
    coord_group = parser.add_mutually_exclusive_group()
    coord_group.add_argument('--convert', action='store_true', default=True,
                             help='Convert Z-up to Y-up coordinates (default)')
    coord_group.add_argument('--no-convert', action='store_false', dest='convert',
                             help='Skip coordinate conversion')

    parser.add_argument('--raw-opacity', action='store_true', default=False,
                        help='Skip sigmoid opacity transform (PLY already in linear [0,1])')
    parser.add_argument('--raw-scale', action='store_true', default=False,
                        help='Skip exp scale transform (PLY already in linear space)')
    parser.add_argument('--sh-degree', type=int, default=None, metavar='N',
                        help='Override SH degree (default: auto-detect)')

    # Processing
    parser.add_argument('--center', action='store_true', default=False,
                        help='Center the point cloud at origin')
    parser.add_argument('--decimate', type=int, default=None, metavar='N',
                        help='Keep only N splats (by opacity)')

    # Verification
    parser.add_argument('--verify', action='store_true', default=False,
                        help='Verify round-trip accuracy after conversion')
    parser.add_argument('--verify-threshold', type=float, default=1e-5, metavar='T',
                        help='Max acceptable error for verification (default: 1e-5)')

    # Info / batch modes
    parser.add_argument('--info', action='store_true', default=False,
                        help='Print PLY info without converting')
    parser.add_argument('--batch', type=str, default=None, metavar='DIR',
                        help='Batch-convert all PLY files in directory')
    parser.add_argument('--output-dir', type=str, default=None, metavar='DIR',
                        help='Output directory for batch mode')

    return parser


def main(argv=None):
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Info mode
    if args.info:
        if not args.input:
            parser.error("--info requires an input PLY file")
        ply_info(args.input)
        return

    # Batch mode
    if args.batch:
        batch_convert(
            args.batch,
            output_dir=args.output_dir,
            sh_degree=args.sh_degree,
            convert_coords=args.convert,
            raw_opacity=args.raw_opacity,
            raw_scale=args.raw_scale,
            do_center=args.center,
            decimate_count=args.decimate,
        )
        return

    # Single file mode
    if not args.input:
        parser.error("Input PLY file is required")
    if not args.output:
        # Default output name
        args.output = os.path.splitext(args.input)[0] + '.glb'

    ply_to_gltf(
        args.input, args.output,
        sh_degree=args.sh_degree,
        convert_coords=args.convert,
        raw_opacity=args.raw_opacity,
        raw_scale=args.raw_scale,
        do_center=args.center,
        decimate_count=args.decimate,
    )

    if args.verify:
        verify_round_trip(
            args.input, args.output,
            convert_coords=args.convert,
            raw_opacity=args.raw_opacity,
            raw_scale=args.raw_scale,
            threshold=args.verify_threshold,
        )


if __name__ == "__main__":
    main()
