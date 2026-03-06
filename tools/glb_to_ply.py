"""Convert a KHR_gaussian_splatting .glb to .ply for external viewers.

Usage:
    python tools/glb_to_ply.py tests/assets/debug_splats.glb debug_splats.ply
    python tools/glb_to_ply.py tests/assets/test_splats.glb test_splats.ply

Then open the .ply in SuperSplat (https://playcanvas.com/supersplat),
gsplat.tech, or antimatter15/splat.
"""

import struct
import json
import sys
import math
import numpy as np


def load_glb(path):
    """Parse a .glb and return (json_data, bin_data)."""
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        assert magic == 0x46546C67, f"Not a GLB file: {path}"

        # JSON chunk
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        assert chunk_type == 0x4E4F534A  # JSON
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes)

        # BIN chunk
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

    # Only support float (5126)
    assert acc['componentType'] == 5126, f"Unsupported component type: {acc['componentType']}"
    data = np.frombuffer(bin_data, dtype=np.float32,
                         count=count * components,
                         offset=offset)
    return data.reshape(count, components) if components > 1 else data


def convert_glb_to_ply(glb_path, ply_path, verbose=True):
    gltf, bin_data = load_glb(glb_path)

    # Find the gaussian splatting primitive
    mesh = gltf['meshes'][0]
    prim = mesh['primitives'][0]
    ext = prim.get('extensions', {}).get('KHR_gaussian_splatting', {})

    attrs = prim['attributes']
    n = gltf['accessors'][attrs['POSITION']]['count']

    # Read data
    positions = read_accessor(gltf, bin_data, attrs['POSITION'])  # (N, 3)
    rotations = read_accessor(gltf, bin_data, attrs.get('_ROTATION', ext['attributes']['ROTATION']))  # (N, 4) XYZW
    scales = read_accessor(gltf, bin_data, attrs.get('_SCALE', ext['attributes']['SCALE']))  # (N, 3) log-scale
    opacities = read_accessor(gltf, bin_data, attrs.get('_OPACITY', ext['attributes']['OPACITY']))  # (N,) logit

    # SH coefficients
    sh_bands = ext.get('sh', [])
    sh_degree = sh_bands[0]['degree'] if sh_bands else 0
    sh0_idx = sh_bands[0]['coefficients'] if sh_bands else attrs.get('_SH_0')
    sh0 = read_accessor(gltf, bin_data, sh0_idx)  # (N, 3)

    if verbose:
        print(f"Loaded {n} splats from {glb_path}")
        print(f"  SH degree: {sh_degree}")
        print(f"  Position range: [{positions.min(axis=0)}] to [{positions.max(axis=0)}]")
        print(f"  Scale (log) range: [{scales.min():.4f}, {scales.max():.4f}]")
        print(f"  Scale (exp) range: [{np.exp(scales.min()):.6f}, {np.exp(scales.max()):.6f}]")
        print(f"  Opacity (logit) range: [{opacities.min():.4f}, {opacities.max():.4f}]")
        sigmoid = 1.0 / (1.0 + np.exp(-opacities))
        print(f"  Opacity (sigmoid) range: [{sigmoid.min():.4f}, {sigmoid.max():.4f}]")

        # Decode SH DC to color
        SH_C0 = 0.28209479177387814
        colors = SH_C0 * sh0 + 0.5
        colors = np.clip(colors, 0, 1)
        print(f"  Color range: [{colors.min(axis=0)}] to [{colors.max(axis=0)}]")

        # Print first few splats
        for i in range(min(n, 5)):
            c = colors[i]
            s = np.exp(scales[i])
            o = 1.0 / (1.0 + math.exp(-float(opacities[i])))
            print(f"  Splat {i}: pos={positions[i]} color=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) "
                  f"scale=({s[0]:.4f},{s[1]:.4f},{s[2]:.4f}) opacity={o:.4f}")

    # Write PLY in the format expected by most splat viewers
    # Standard 3DGS PLY format:
    #   x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity,
    #   scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
    # Note: PLY viewers expect quaternion as WXYZ (scalar-first)
    # and opacity as logit (inverse sigmoid)

    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(ply_path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            x, y, z = positions[i]
            # Normals (unused, set to 0)
            nx = ny = nz = 0.0
            # SH DC coefficients (raw, not decoded to color)
            dc0, dc1, dc2 = sh0[i]
            # Opacity (logit space, as stored)
            opa = float(opacities[i])
            # Scale (log space, as stored)
            s0, s1, s2 = scales[i]
            # Rotation: GLB stores XYZW, PLY expects WXYZ
            qx, qy, qz, qw = rotations[i]

            f.write(struct.pack('<17f',
                x, y, z, nx, ny, nz,
                dc0, dc1, dc2, opa,
                s0, s1, s2,
                qw, qx, qy, qz))  # WXYZ order for PLY

    if verbose:
        print(f"\nWrote {ply_path} ({n} splats)")
        print(f"Open in: https://playcanvas.com/supersplat")
        print(f"     or: https://antimatter15.com/splat/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python tools/glb_to_ply.py <input.glb> [output.ply]")
        sys.exit(1)
    glb_path = sys.argv[1]
    ply_path = sys.argv[2] if len(sys.argv) > 2 else glb_path.rsplit('.', 1)[0] + '.ply'
    convert_glb_to_ply(glb_path, ply_path)
