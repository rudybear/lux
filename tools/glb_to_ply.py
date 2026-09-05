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


def _find_splat_primitive(gltf):
    """Find the first POINTS-mode primitive carrying Gaussian splat attributes.

    Supports the ratified layout (KHR_gaussian_splatting:ROTATION etc. living
    directly in primitive.attributes), the legacy pre-ratification draft
    layout (ROTATION/SCALE/OPACITY nested under
    extensions.KHR_gaussian_splatting.attributes), and lux's internal
    hand-authored fixture convention (_ROTATION/_SCALE/... in
    primitive.attributes).
    """
    for mesh in gltf.get('meshes', []):
        for prim in mesh.get('primitives', []):
            if prim.get('mode', 4) != 0:
                continue
            attrs = prim.get('attributes', {})
            ext = prim.get('extensions', {}).get('KHR_gaussian_splatting', {})
            has_rotation = (
                'KHR_gaussian_splatting:ROTATION' in attrs
                or '_ROTATION' in attrs
                or 'ROTATION' in ext.get('attributes', {})
            )
            if has_rotation:
                return prim
    raise ValueError("No KHR_gaussian_splatting primitive found in glTF")


def _resolve_attr(attrs, ext, suffix):
    """Resolve a splat attribute accessor index across ratified/legacy/internal layouts."""
    if f"KHR_gaussian_splatting:{suffix}" in attrs:
        return attrs[f"KHR_gaussian_splatting:{suffix}"]
    if f"_{suffix}" in attrs:
        return attrs[f"_{suffix}"]
    return ext['attributes'][suffix]


def _is_linear_scale(values):
    """Heuristic: ratified KHR_gaussian_splatting:SCALE is linear (non-negative);
    the legacy/pre-editorial-review draft (matching the currently-published
    Khronos conformance test assets) and lux's internal _SCALE convention both
    store log-space values, which can be negative. Treat all-non-negative,
    boundedly-small values as linear; anything with negatives (or very large
    magnitude) as log-space.
    """
    return bool(np.all(values >= 0.0) and np.all(values < 20.0))


def _is_linear_opacity(values):
    """Heuristic: linear opacity is always in [0, 1]; logit-space values
    routinely fall outside that range."""
    return bool(np.all(values >= 0.0) and np.all(values <= 1.0))


def convert_glb_to_ply(glb_path, ply_path, verbose=True):
    gltf, bin_data = load_glb(glb_path)

    # Find the gaussian splatting primitive (not necessarily mesh 0 / primitive 0 —
    # hybrid mesh+splat assets interleave regular mesh primitives with the splat one).
    prim = _find_splat_primitive(gltf)
    ext = prim.get('extensions', {}).get('KHR_gaussian_splatting', {})

    attrs = prim['attributes']
    n = gltf['accessors'][attrs['POSITION']]['count']

    # Read data
    positions = read_accessor(gltf, bin_data, attrs['POSITION'])  # (N, 3)
    rotations = read_accessor(gltf, bin_data, _resolve_attr(attrs, ext, 'ROTATION'))  # (N, 4) XYZW
    scales = read_accessor(gltf, bin_data, _resolve_attr(attrs, ext, 'SCALE'))
    opacities = read_accessor(gltf, bin_data, _resolve_attr(attrs, ext, 'OPACITY'))

    # PLY output always uses log-space scale / logit-space opacity (matching
    # the raw 3DGS training convention). Ratified-layout GLBs store linear
    # scale and linear opacity, so convert back; legacy/internal layouts are
    # already in log/logit space.
    if _is_linear_scale(scales):
        scales = np.log(np.maximum(scales, 1e-12))
    if _is_linear_opacity(opacities):
        opacities = np.log(np.clip(opacities, 1e-7, 1 - 1e-7) / (1 - np.clip(opacities, 1e-7, 1 - 1e-7)))

    # SH coefficients: ratified layout keeps SH_DEGREE_0_COEF_0 directly in
    # primitive.attributes; legacy layout nests a "sh" array in the extension.
    sh0_idx = attrs.get('KHR_gaussian_splatting:SH_DEGREE_0_COEF_0', attrs.get('_SH_0'))
    sh_degree = 0
    if sh0_idx is None:
        sh_bands = ext.get('sh', [])
        sh_degree = sh_bands[0]['degree'] if sh_bands else 0
        coeffs = sh_bands[0]['coefficients'] if sh_bands else None
        sh0_idx = coeffs[0] if isinstance(coeffs, list) else coeffs
    else:
        prefix = 'KHR_gaussian_splatting:SH_DEGREE_'
        sh_degree = max(
            (int(name[len(prefix):].split('_COEF_')[0]) for name in attrs
             if name.startswith(prefix)),
            default=0,
        )
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
