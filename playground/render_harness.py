"""Offscreen render harness for Lux shader playground.

Loads compiled SPIR-V vertex and fragment shaders, renders a colored
triangle to a 512x512 offscreen texture, and saves the result as PNG.

Usage:
    python render_harness.py <name>.vert.spv <name>.frag.spv -o output.png

Dependencies:
    pip install wgpu Pillow numpy
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image


# ---------------------------------------------------------------------------
# Shader loading helpers
# ---------------------------------------------------------------------------

def _try_spirv_to_wgsl(spv_path: Path) -> str | None:
    """Attempt SPIR-V -> WGSL conversion using naga-cli (if available).

    spirv-cross does NOT support WGSL output, so we look for naga-cli
    which ships with some wgpu/Rust toolchains.  Returns None on failure.
    """
    for tool in ("naga", "naga-cli"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wgsl", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                [tool, str(spv_path), tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                wgsl = Path(tmp_path).read_text(encoding="utf-8")
                Path(tmp_path).unlink(missing_ok=True)
                return wgsl
            Path(tmp_path).unlink(missing_ok=True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def load_shader_module(device: wgpu.GPUDevice, spv_path: Path) -> wgpu.GPUShaderModule:
    """Load a SPIR-V binary as a wgpu shader module.

    Primary path: pass raw SPIR-V bytes to wgpu (supported by wgpu-native
    0.29.x via the spirv feature).  Fallback: convert to WGSL via naga-cli.
    """
    spv_bytes = spv_path.read_bytes()

    # Verify SPIR-V magic number
    if len(spv_bytes) < 4:
        raise ValueError(f"{spv_path}: file too small to be valid SPIR-V")
    magic = struct.unpack("<I", spv_bytes[:4])[0]
    if magic != 0x07230203:
        raise ValueError(
            f"{spv_path}: bad SPIR-V magic 0x{magic:08X} "
            f"(expected 0x07230203)"
        )

    # Try loading SPIR-V directly
    try:
        module = device.create_shader_module(code=spv_bytes)
        return module
    except Exception as exc:
        msg = str(exc).encode("ascii", errors="replace").decode("ascii")
        print(f"[warn] direct SPIR-V load failed ({msg}); trying WGSL fallback...")

    # Fallback: SPIR-V -> WGSL via naga-cli
    wgsl = _try_spirv_to_wgsl(spv_path)
    if wgsl is not None:
        print(f"[info] converted {spv_path.name} to WGSL via naga")
        return device.create_shader_module(code=wgsl)

    raise RuntimeError(
        f"Cannot load {spv_path}: wgpu rejected the SPIR-V and no WGSL "
        f"converter is available.  Install naga-cli or use a wgpu version "
        f"with SPIR-V support."
    )


# ---------------------------------------------------------------------------
# GLB / Gaussian splat loading
# ---------------------------------------------------------------------------

def _is_splat_attr(name: str, suffix: str) -> bool:
    """Check if a glTF attribute name matches a splat attribute.

    Supports both internal format (_ROTATION, _SCALE, _OPACITY, _SH_N)
    and KHR conformance format (KHR_gaussian_splatting:ROTATION, etc.).
    """
    return name == f"_{suffix}" or name == f"KHR_gaussian_splatting:{suffix}"


def _node_local_transform(node: dict) -> np.ndarray:
    """Compute a 4x4 local transform matrix from a glTF node's TRS or matrix."""
    if 'matrix' in node:
        # glTF stores matrices in column-major order
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4)

    m = np.eye(4, dtype=np.float64)

    if 'scale' in node:
        s = node['scale']
        sm = np.eye(4, dtype=np.float64)
        sm[0, 0], sm[1, 1], sm[2, 2] = s[0], s[1], s[2]
        m = sm

    if 'rotation' in node:
        x, y, z, w = node['rotation']
        rm = np.eye(4, dtype=np.float64)
        rm[0, 0] = 1 - 2 * (y * y + z * z)
        rm[0, 1] = 2 * (x * y + z * w)
        rm[0, 2] = 2 * (x * z - y * w)
        rm[1, 0] = 2 * (x * y - z * w)
        rm[1, 1] = 1 - 2 * (x * x + z * z)
        rm[1, 2] = 2 * (y * z + x * w)
        rm[2, 0] = 2 * (x * z + y * w)
        rm[2, 1] = 2 * (y * z - x * w)
        rm[2, 2] = 1 - 2 * (x * x + y * y)
        m = rm @ m

    if 'translation' in node:
        t = node['translation']
        tm = np.eye(4, dtype=np.float64)
        tm[0, 3], tm[1, 3], tm[2, 3] = t[0], t[1], t[2]
        m = tm @ m

    return m


def _mat3_to_quaternion(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an XYZW quaternion (Shepperd's method)."""
    m = rot_mat
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[1, 2] - m[2, 1]) * s
        y = (m[2, 0] - m[0, 2]) * s
        z = (m[0, 1] - m[1, 0]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[1, 2] - m[2, 1]) / s
        x = 0.25 * s
        y = (m[1, 0] + m[0, 1]) / s
        z = (m[2, 0] + m[0, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[2, 0] - m[0, 2]) / s
        x = (m[1, 0] + m[0, 1]) / s
        y = 0.25 * s
        z = (m[2, 1] + m[1, 2]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[0, 1] - m[1, 0]) / s
        x = (m[2, 0] + m[0, 2]) / s
        y = (m[2, 1] + m[1, 2]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    qlen = np.linalg.norm(q)
    if qlen > 1e-6:
        q /= qlen
    return q


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1 * q2. Both in XYZW format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def load_splat_glb(path: Path) -> dict:
    """Parse a GLB file and extract Gaussian splat data.

    Supports multi-primitive merging across all meshes, KHR_gaussian_splatting
    opacity logit conversion, node transform application, and hybrid
    mesh+splat scenes (POINTS primitives are splats, others are mesh geometry).

    Returns a dict with numpy arrays:
        positions   (N, 3)  - world-space xyz
        rotations   (N, 4)  - XYZW quaternion (world-space)
        scales      (N, 3)  - log-space (world-transformed)
        opacities   (N,)    - logit-space (ready for sigmoid in rasterizer)
        sh_coeffs   list of (N, C) arrays per SH degree
        sh_degree   int     - max SH degree across all primitives
        num_splats  int
        has_mesh    bool    - True if non-POINTS mesh geometry also exists
    """
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file: bad magic 0x{magic:08X}")

    # JSON chunk
    json_len, json_type = struct.unpack_from('<II', data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("GLB: first chunk is not JSON")
    gltf = json.loads(data[20:20 + json_len])

    # BIN chunk
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from('<II', data, bin_off)
    if bin_type != 0x004E4942:
        raise ValueError("GLB: second chunk is not BIN")
    bin_data = data[bin_off + 8:bin_off + 8 + bin_len]

    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']

    def read_accessor(idx):
        acc = accessors[idx]
        bv = buffer_views[acc['bufferView']]
        offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
        count = acc['count']
        components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc['type']]
        arr = np.frombuffer(bin_data, dtype=np.float32,
                            count=count * components, offset=offset)
        return arr.reshape(count, components) if components > 1 else arr.copy()

    # -----------------------------------------------------------------------
    # Scan all meshes for POINTS primitives (splats) and non-POINTS (mesh)
    # -----------------------------------------------------------------------
    # Per-primitive metadata for deferred node transform application
    class _PrimInfo:
        __slots__ = ('mesh_index', 'start_splat', 'splat_count')
        def __init__(self, mi, start, count):
            self.mesh_index = mi
            self.start_splat = start
            self.splat_count = count

    all_positions = []      # list of (Ni, 3) arrays
    all_rotations = []      # list of (Ni, 4) arrays
    all_scales = []         # list of (Ni, 3) arrays
    all_opacities = []      # list of (Ni,) arrays
    # Per-degree: list of (Ni, C) arrays per primitive
    all_sh_per_degree = {}  # degree -> list of (Ni, C) arrays
    prim_infos = []
    global_max_sh_degree = 0
    total_splats = 0
    khr_format = False
    has_mesh = False

    for mi, mesh in enumerate(gltf.get('meshes', [])):
        for prim in mesh.get('primitives', []):
            # glTF mode: 0 = POINTS, 4 = TRIANGLES (default if absent)
            mode = prim.get('mode', 4)
            if mode != 0:
                # Non-POINTS primitive -- this is regular mesh geometry
                has_mesh = True
                continue

            attrs = prim.get('attributes', {})

            # Detect splat attributes: check for KHR extension or _ROTATION/_SCALE
            # prim_is_khr is True ONLY when attribute names use the
            # "KHR_gaussian_splatting:" prefix (conformance format with linear
            # opacity).  Having the extension object alone does NOT imply
            # linear opacity -- internal format stores logit values.
            has_rotation = False
            has_scale = False
            prim_is_khr = False
            gs_ext = prim.get('extensions', {}).get('KHR_gaussian_splatting')

            # Always scan top-level attributes for both naming conventions
            for aname in attrs:
                if _is_splat_attr(aname, 'ROTATION'):
                    has_rotation = True
                if _is_splat_attr(aname, 'SCALE'):
                    has_scale = True
                if aname.startswith('KHR_gaussian_splatting:'):
                    prim_is_khr = True

            # Also check the extension object's attributes sub-dict
            if gs_ext is not None:
                gs_attrs_dict = gs_ext.get('attributes', {})
                if 'ROTATION' in gs_attrs_dict:
                    has_rotation = True
                if 'SCALE' in gs_attrs_dict:
                    has_scale = True

            if not has_rotation and not has_scale:
                continue

            if prim_is_khr:
                khr_format = True

            # Read position data
            if 'POSITION' not in attrs:
                continue
            positions = read_accessor(attrs['POSITION'])
            prim_n = positions.shape[0]
            if prim_n == 0:
                continue

            # Read splat-specific attributes.
            # Three formats:
            #   1. Conformance: attrs have "KHR_gaussian_splatting:ROTATION" etc.
            #   2. Extension object: gs_ext.attributes = {ROTATION: idx, ...}
            #   3. Underscore: attrs have "_ROTATION", "_SCALE" etc.
            prim_sh = {}

            if prim_is_khr:
                # Conformance format: read from top-level attrs with prefix
                rot_key = next((k for k in attrs if k.endswith(':ROTATION')), None)
                scl_key = next((k for k in attrs if k.endswith(':SCALE')), None)
                opa_key = next((k for k in attrs if k.endswith(':OPACITY')), None)

                rotations = read_accessor(attrs[rot_key]) if rot_key else np.zeros((prim_n, 4), dtype=np.float32)
                scales = read_accessor(attrs[scl_key]) if scl_key else np.zeros((prim_n, 3), dtype=np.float32)
                opacities = read_accessor(attrs[opa_key]) if opa_key else np.zeros(prim_n, dtype=np.float32)

                # SH from conformance attributes or extension sh list
                for aname in attrs:
                    if 'SH_DEGREE_' in aname:
                        # KHR_gaussian_splatting:SH_DEGREE_N_COEF_M
                        # For simplicity, treat each as a separate degree entry
                        # The read_accessor returns the vec3 per splat
                        rest = aname.split('SH_DEGREE_')[1]
                        parts = rest.split('_COEF_')
                        if len(parts) == 2:
                            degree = int(parts[0])
                            coeffs = read_accessor(attrs[aname])
                            # For degree 0, there's only 1 coef (the DC)
                            if degree not in prim_sh:
                                prim_sh[degree] = coeffs
                            else:
                                # Multiple coefs for same degree: concat columns
                                prim_sh[degree] = np.hstack([prim_sh[degree], coeffs])
                            if degree > global_max_sh_degree:
                                global_max_sh_degree = degree

                # Also check extension sh list (some conformance files may use it)
                if gs_ext is not None:
                    for sh_entry in gs_ext.get('sh', []):
                        degree = sh_entry['degree']
                        coeffs = read_accessor(sh_entry['coefficients'])
                        if degree not in prim_sh:
                            prim_sh[degree] = coeffs
                        if degree > global_max_sh_degree:
                            global_max_sh_degree = degree

            elif gs_ext is not None and gs_ext.get('attributes'):
                # Extension object format: accessor indices in gs_ext.attributes
                gs_attrs = gs_ext['attributes']
                rotations = read_accessor(gs_attrs['ROTATION']) if 'ROTATION' in gs_attrs else np.zeros((prim_n, 4), dtype=np.float32)
                scales = read_accessor(gs_attrs['SCALE']) if 'SCALE' in gs_attrs else np.zeros((prim_n, 3), dtype=np.float32)
                opacities = read_accessor(gs_attrs['OPACITY']) if 'OPACITY' in gs_attrs else np.zeros(prim_n, dtype=np.float32)

                # SH from extension sh list
                for sh_entry in gs_ext.get('sh', []):
                    degree = sh_entry['degree']
                    coeffs = read_accessor(sh_entry['coefficients'])
                    prim_sh[degree] = coeffs
                    if degree > global_max_sh_degree:
                        global_max_sh_degree = degree

            else:
                # Underscore attribute format (_ROTATION, _SCALE, _OPACITY, _SH_N)
                rot_key = next((k for k in attrs if _is_splat_attr(k, 'ROTATION')), None)
                scl_key = next((k for k in attrs if _is_splat_attr(k, 'SCALE')), None)
                opa_key = next((k for k in attrs if _is_splat_attr(k, 'OPACITY')), None)

                rotations = read_accessor(attrs[rot_key]) if rot_key else np.zeros((prim_n, 4), dtype=np.float32)
                scales = read_accessor(attrs[scl_key]) if scl_key else np.zeros((prim_n, 3), dtype=np.float32)
                opacities = read_accessor(attrs[opa_key]) if opa_key else np.zeros(prim_n, dtype=np.float32)

                # SH from _SH_N attributes
                for aname in attrs:
                    if aname.startswith('_SH_'):
                        degree = int(aname[4:])
                        prim_sh[degree] = read_accessor(attrs[aname])
                        if degree > global_max_sh_degree:
                            global_max_sh_degree = degree

            # Ensure correct shapes
            if positions.ndim == 1:
                positions = positions.reshape(-1, 3)
            if rotations.ndim == 1:
                rotations = rotations.reshape(-1, 4)
            if scales.ndim == 1:
                scales = scales.reshape(-1, 3)

            # Record per-primitive info for node transform
            prim_infos.append(_PrimInfo(mi, total_splats, prim_n))

            all_positions.append(positions)
            all_rotations.append(rotations)
            all_scales.append(scales)
            all_opacities.append(opacities)

            # Accumulate SH by degree
            for degree, coeffs in prim_sh.items():
                if degree not in all_sh_per_degree:
                    all_sh_per_degree[degree] = []
                all_sh_per_degree[degree].append((total_splats, prim_n, coeffs))

            total_splats += prim_n

    # -----------------------------------------------------------------------
    # Handle legacy single-primitive path (no POINTS mode set but has KHR ext)
    # -----------------------------------------------------------------------
    if total_splats == 0:
        # Fall back: try the original path for files that don't set mode=0
        for mi, mesh in enumerate(gltf.get('meshes', [])):
            for prim in mesh.get('primitives', []):
                gs_ext = prim.get('extensions', {}).get('KHR_gaussian_splatting')
                if gs_ext is None:
                    continue

                # Check if this primitive uses conformance attribute naming
                attrs = prim.get('attributes', {})
                for aname in attrs:
                    if aname.startswith('KHR_gaussian_splatting:'):
                        khr_format = True
                        break
                gs_attrs = gs_ext.get('attributes', {})

                if 'POSITION' not in attrs:
                    continue
                positions = read_accessor(attrs['POSITION'])
                prim_n = positions.shape[0]
                if prim_n == 0:
                    continue

                rotations = read_accessor(gs_attrs['ROTATION']) if 'ROTATION' in gs_attrs else np.zeros((prim_n, 4), dtype=np.float32)
                scales = read_accessor(gs_attrs['SCALE']) if 'SCALE' in gs_attrs else np.zeros((prim_n, 3), dtype=np.float32)
                opacities = read_accessor(gs_attrs['OPACITY']) if 'OPACITY' in gs_attrs else np.zeros(prim_n, dtype=np.float32)

                if positions.ndim == 1:
                    positions = positions.reshape(-1, 3)
                if rotations.ndim == 1:
                    rotations = rotations.reshape(-1, 4)
                if scales.ndim == 1:
                    scales = scales.reshape(-1, 3)

                prim_sh = {}
                for sh_entry in gs_ext.get('sh', []):
                    degree = sh_entry['degree']
                    coeffs = read_accessor(sh_entry['coefficients'])
                    prim_sh[degree] = coeffs
                    if degree > global_max_sh_degree:
                        global_max_sh_degree = degree

                prim_infos.append(_PrimInfo(mi, total_splats, prim_n))
                all_positions.append(positions)
                all_rotations.append(rotations)
                all_scales.append(scales)
                all_opacities.append(opacities)

                for degree, coeffs in prim_sh.items():
                    if degree not in all_sh_per_degree:
                        all_sh_per_degree[degree] = []
                    all_sh_per_degree[degree].append((total_splats, prim_n, coeffs))

                total_splats += prim_n

    if total_splats == 0:
        raise ValueError(f"No Gaussian splat primitives found in {path}")

    # -----------------------------------------------------------------------
    # Merge accumulated arrays
    # -----------------------------------------------------------------------
    merged_positions = np.concatenate(all_positions, axis=0)   # (N, 3)
    merged_rotations = np.concatenate(all_rotations, axis=0)   # (N, 4)
    merged_scales = np.concatenate(all_scales, axis=0)         # (N, 3)
    merged_opacities = np.concatenate(all_opacities, axis=0)   # (N,)

    # Merge SH coefficients with zero-padding for primitives missing higher degrees
    sh_coeffs_list = []
    for degree in sorted(all_sh_per_degree.keys()):
        entries = all_sh_per_degree[degree]
        # Determine the number of components per splat for this degree
        # from the first entry that has data
        components = None
        for _, _, coeffs in entries:
            if coeffs.ndim == 2:
                components = coeffs.shape[1]
            else:
                components = 1
            break
        if components is None:
            continue

        merged = np.zeros((total_splats, components), dtype=np.float32)
        for start, count, coeffs in entries:
            if coeffs.ndim == 1:
                # Scalar per splat
                merged[start:start + count, 0] = coeffs[:count]
            else:
                # Pad or truncate to match target components
                c = min(coeffs.shape[1], components)
                merged[start:start + count, :c] = coeffs[:count, :c]
        sh_coeffs_list.append(merged)

    # -----------------------------------------------------------------------
    # KHR linear → log/logit conversion
    # -----------------------------------------------------------------------
    # KHR_gaussian_splatting stores scales and opacity in linear space.
    # The rasterizer applies exp() to scales and sigmoid() to opacity,
    # so we must convert back to log/logit space.
    # Non-KHR formats already store log/logit values.
    if khr_format:
        # Convert linear scales to log-space: log(scale)
        merged_scales = np.log(np.maximum(merged_scales, 1e-7)).astype(np.float32)

        # Convert linear opacity [0,1] to logit: log(p / (1-p))
        p = np.clip(merged_opacities, 1e-6, 1.0 - 1e-6)
        merged_opacities = np.log(p / (1.0 - p)).astype(np.float32)

    # -----------------------------------------------------------------------
    # Node transform application
    # -----------------------------------------------------------------------
    nodes = gltf.get('nodes', [])
    if nodes and prim_infos:
        # Compute local transforms for each node
        local_transforms = []
        for node in nodes:
            local_transforms.append(_node_local_transform(node))

        # Build parent mapping from children lists
        parents = [-1] * len(nodes)
        for ni, node in enumerate(nodes):
            for ci in node.get('children', []):
                if 0 <= ci < len(nodes):
                    parents[ci] = ni

        # Determine root nodes from scene
        root_nodes = []
        scenes = gltf.get('scenes', [])
        if scenes:
            scene_idx = gltf.get('scene', 0)
            if 0 <= scene_idx < len(scenes):
                root_nodes = scenes[scene_idx].get('nodes', [])
        if not root_nodes:
            # Fallback: nodes with no parent
            root_nodes = [i for i in range(len(nodes)) if parents[i] == -1]

        # Compute world transforms (top-down BFS from roots)
        world_transforms = [np.eye(4, dtype=np.float64)] * len(nodes)

        def _compute_world(ni, parent_world):
            world_transforms[ni] = parent_world @ local_transforms[ni]
            for ci in nodes[ni].get('children', []):
                if 0 <= ci < len(nodes):
                    _compute_world(ci, world_transforms[ni])

        for root in root_nodes:
            if 0 <= root < len(nodes):
                _compute_world(root, np.eye(4, dtype=np.float64))

        # Build mapping: glTF mesh index -> first node's world transform
        mesh_to_world = {}
        for ni, node in enumerate(nodes):
            mi = node.get('mesh', -1)
            if mi >= 0 and mi not in mesh_to_world:
                mesh_to_world[mi] = world_transforms[ni]

        # Apply transforms to each primitive's splat range
        for info in prim_infos:
            world = mesh_to_world.get(info.mesh_index)
            if world is None:
                continue

            # Check if transform is identity (skip if so)
            if np.allclose(world, np.eye(4), atol=1e-6):
                continue

            sl = slice(info.start_splat, info.start_splat + info.splat_count)

            # --- Transform positions: world @ (pos, 1) -> xyz ---
            pos = merged_positions[sl]  # (K, 3)
            ones = np.ones((pos.shape[0], 1), dtype=np.float64)
            pos4 = np.hstack([pos.astype(np.float64), ones])  # (K, 4)
            world_pos = (world @ pos4.T).T  # (K, 4)
            merged_positions[sl] = world_pos[:, :3].astype(np.float32)

            # --- Extract rotation and scale from world matrix ---
            rot_mat = world[:3, :3].copy()
            sx = np.linalg.norm(rot_mat[:, 0])
            sy = np.linalg.norm(rot_mat[:, 1])
            sz = np.linalg.norm(rot_mat[:, 2])
            uniform_scale = (sx * sy * sz) ** (1.0 / 3.0)  # geometric mean

            # Normalize rotation matrix (remove scale)
            if sx > 1e-6:
                rot_mat[:, 0] /= sx
            if sy > 1e-6:
                rot_mat[:, 1] /= sy
            if sz > 1e-6:
                rot_mat[:, 2] /= sz

            node_quat = _mat3_to_quaternion(rot_mat)  # XYZW

            # --- Transform rotations: q_node * q_splat ---
            rots = merged_rotations[sl].astype(np.float64)  # (K, 4)
            for i in range(rots.shape[0]):
                rots[i] = _quat_multiply(node_quat, rots[i])
            merged_rotations[sl] = rots.astype(np.float32)

            # --- Transform scales: log_scale + log(uniform_scale) ---
            if uniform_scale > 1e-6:
                log_scale = math.log(uniform_scale)
                merged_scales[sl] = (merged_scales[sl].astype(np.float64) + log_scale).astype(np.float32)

    return {
        'positions': merged_positions,
        'rotations': merged_rotations,
        'scales': merged_scales,
        'opacities': merged_opacities,
        'sh_coeffs': sh_coeffs_list,
        'sh_degree': global_max_sh_degree,
        'num_splats': total_splats,
        'has_mesh': has_mesh,
    }


# ---------------------------------------------------------------------------
# Camera math helpers (numpy)
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 view matrix (column-major, OpenGL convention)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, :3] = 0.0
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Compute a 4x4 perspective matrix (Vulkan Y-flip applied)."""
    t = math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * t)
    m[1, 1] = -1.0 / t  # Vulkan Y-flip
    m[2, 2] = far / (near - far)
    m[2, 3] = (near * far) / (near - far)
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# Vertex data
# ---------------------------------------------------------------------------

# Hardcoded colored triangle in clip space (no MVP transform needed).
# Each vertex: position (vec3) + color (vec3) = 6 x float32
TRIANGLE_VERTICES = [
    # position (x, y, z),     color (r, g, b)
    ( 0.0,  0.5, 0.0,        1.0, 0.0, 0.0),   # top,          red
    (-0.5, -0.5, 0.0,        0.0, 1.0, 0.0),   # bottom-left,  green
    ( 0.5, -0.5, 0.0,        0.0, 0.0, 1.0),   # bottom-right, blue
]

VERTEX_FORMAT = f"{len(TRIANGLE_VERTICES) * 6}f"
VERTEX_DATA = struct.pack(
    VERTEX_FORMAT,
    *[component for vertex in TRIANGLE_VERTICES for component in vertex],
)
FLOATS_PER_VERTEX = 6
VERTEX_STRIDE = FLOATS_PER_VERTEX * 4  # 24 bytes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_triangle(
    vert_spv_path: Path,
    frag_spv_path: Path,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render one frame and return an (H, W, 4) uint8 RGBA numpy array."""

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Shaders ---
    vert_module = load_shader_module(device, vert_spv_path)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Vertex buffer ---
    vbo = device.create_buffer_with_data(
        data=VERTEX_DATA, usage=wgpu.BufferUsage.VERTEX
    )

    # --- Render target texture ---
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # --- Render pipeline ---
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex=wgpu.VertexState(
            module=vert_module,
            entry_point="main",
            buffers=[
                wgpu.VertexBufferLayout(
                    array_stride=VERTEX_STRIDE,
                    step_mode=wgpu.VertexStepMode.vertex,
                    attributes=[
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x3,
                            offset=0,
                            shader_location=0,  # position
                        ),
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x3,
                            offset=12,
                            shader_location=1,  # color
                        ),
                    ],
                )
            ],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
        ),
        multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
        fragment=wgpu.FragmentState(
            module=frag_module,
            entry_point="main",
            targets=[
                wgpu.ColorTargetState(format=wgpu.TextureFormat.rgba8unorm),
            ],
        ),
    )

    # --- Encode render pass ---
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.0, 0.0, 0.0, 1.0),
            )
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_vertex_buffer(0, vbo)
    render_pass.draw(3)
    render_pass.end()

    # --- Readback: copy texture -> buffer ---
    bytes_per_row = width * 4
    # wgpu requires bytes_per_row aligned to 256
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255

    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=texture),
        wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer,
            offset=0,
            bytes_per_row=bytes_per_row_aligned,
            rows_per_image=height,
        ),
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    # --- Map buffer and extract pixels ---
    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()

    # Handle potential row padding
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, : width * 4].reshape(height, width, 4).copy()

    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# Fullscreen quad rendering (fragment-only shaders)
# ---------------------------------------------------------------------------

# Built-in WGSL vertex shader for fullscreen coverage.
# Draws a single triangle that covers the entire screen and passes
# UV coordinates (0,0)-(1,1) to the fragment shader at location 0.
_FULLSCREEN_VERT_WGSL = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices that cover [-1,1] clip space
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    // Map clip coords to UV: [-1,1] -> [0,1], with Y flipped so
    // UV (0,0) = top-left to match Vulkan/wgpu convention
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}
"""


def render_fullscreen(
    frag_spv_path: Path,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render a fullscreen quad with the given fragment shader.

    The fragment shader must accept `in uv: vec2` at location 0.
    Returns an (H, W, 4) uint8 RGBA numpy array.
    """

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Shaders ---
    vert_module = device.create_shader_module(code=_FULLSCREEN_VERT_WGSL)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Render target texture ---
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # --- Render pipeline (no vertex buffers needed) ---
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex=wgpu.VertexState(
            module=vert_module,
            entry_point="main",
            buffers=[],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
        ),
        multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
        fragment=wgpu.FragmentState(
            module=frag_module,
            entry_point="main",
            targets=[
                wgpu.ColorTargetState(format=wgpu.TextureFormat.rgba8unorm),
            ],
        ),
    )

    # --- Encode render pass ---
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.0, 0.0, 0.0, 1.0),
            )
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.draw(3)  # 3 vertices for fullscreen triangle
    render_pass.end()

    # --- Readback: copy texture -> buffer ---
    bytes_per_row = width * 4
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255

    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=texture),
        wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer,
            offset=0,
            bytes_per_row=bytes_per_row_aligned,
            rows_per_image=height,
        ),
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    # --- Map buffer and extract pixels ---
    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, : width * 4].reshape(height, width, 4).copy()

    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# Gaussian splat rendering
# ---------------------------------------------------------------------------

def render_splats(
    comp_spv_path: Path,
    vert_spv_path: Path,
    frag_spv_path: Path,
    splat_glb_path: Path,
    width: int = 512,
    height: int = 512,
    eye: np.ndarray = None,
    target: np.ndarray = None,
) -> np.ndarray:
    """Render gaussian splats and return an (H, W, 4) uint8 RGBA array.

    Pure CPU/numpy implementation matching the GPU pipeline exactly:
    compute projection → CPU depth sort → rasterize 2D Gaussians.
    """
    splat = load_splat_glb(splat_glb_path)
    return render_splats_from_data(splat, width, height, eye=eye, target=target)


def render_splats_from_data(
    splat: dict,
    width: int = 512,
    height: int = 512,
    eye: np.ndarray = None,
    target: np.ndarray = None,
) -> np.ndarray:
    """Render preloaded splat data and return an (H, W, 4) uint8 RGBA array."""
    n = splat['num_splats']

    positions = splat['positions']      # (N, 3)
    rotations = splat['rotations']      # (N, 4) XYZW
    scales_log = splat['scales']        # (N, 3) log-space
    opacities_logit = splat['opacities']  # (N,) logit-space
    sh_coeffs = splat['sh_coeffs']      # list of (N, 3) arrays

    # --- Camera setup (matches C++/Rust auto-camera) ---
    bb_min, bb_max = positions.min(axis=0), positions.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5
    if radius < 0.001:
        radius = 1.0

    if target is None:
        target = center
    if eye is None:
        eye = center + np.array([0.0, radius * 0.5, radius * 2.5], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_y = math.radians(45.0)
    aspect = width / height
    near_plane = 0.01
    far_plane = radius * 10.0

    view = _look_at(eye, target, up)
    proj = _perspective(fov_y, aspect, near_plane, far_plane)

    focal_y = 0.5 * height / math.tan(fov_y / 2.0)
    focal_x = focal_y  # square pixels

    # --- Preprocess each splat (CPU equivalent of compute shader) ---
    # Arrays for projected splat data
    proj_centers = np.zeros((n, 4), dtype=np.float32)  # ndc_x, ndc_y, ndc_z, radius
    proj_conics = np.zeros((n, 4), dtype=np.float32)   # conic_x, conic_y, conic_z, opacity
    proj_colors = np.zeros((n, 4), dtype=np.float32)   # r, g, b, opacity
    view_depths = np.zeros(n, dtype=np.float32)

    SH_C0 = 0.28209479177387814

    for i in range(n):
        pos = positions[i]
        # Transform to view space
        vp = view @ np.append(pos, 1.0)
        vx, vy, vz = vp[:3]

        # Near-plane cull
        if vz > -0.1:
            view_depths[i] = 1e6
            continue

        # Positive depth
        t = -vz
        t2 = t * t

        # Quaternion to rotation matrix (XYZW = scalar-last)
        qi, qj, qk, qr = rotations[i]
        qi2, qj2, qk2 = qi*qi, qj*qj, qk*qk
        qri, qrj, qrk = qr*qi, qr*qj, qr*qk
        qij, qik, qjk = qi*qj, qi*qk, qj*qk

        R = np.array([
            [1 - 2*(qj2+qk2), 2*(qij-qrk),   2*(qik+qrj)],
            [2*(qij+qrk),     1 - 2*(qi2+qk2), 2*(qjk-qri)],
            [2*(qik-qrj),     2*(qjk+qri),     1 - 2*(qi2+qj2)],
        ], dtype=np.float64)

        # Scale (exp of log-scale)
        s = np.exp(scales_log[i]).astype(np.float64)

        # 3D covariance: Sigma = R @ diag(s^2) @ R^T
        S2 = s * s
        cov3d = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(a, 3):
                val = sum(R[a, k] * R[b, k] * S2[k] for k in range(3))
                cov3d[a, b] = val
                cov3d[b, a] = val

        # Jacobian of projection
        j00 = focal_x / t
        j02 = -focal_x * vx / t2
        j11 = focal_y / t
        j12 = -focal_y * vy / t2

        # J = [[j00, 0, j02], [0, j11, j12]]
        # 2D covariance = J @ cov3d @ J^T
        c00 = j00*j00*cov3d[0,0] + 2*j00*j02*cov3d[0,2] + j02*j02*cov3d[2,2]
        c01 = j00*j11*cov3d[0,1] + j00*j12*cov3d[0,2] + j02*j11*cov3d[1,2] + j02*j12*cov3d[2,2]
        c11 = j11*j11*cov3d[1,1] + 2*j11*j12*cov3d[1,2] + j12*j12*cov3d[2,2]

        # Low-pass filter
        c00 += 0.3
        c11 += 0.3

        det = c00 * c11 - c01 * c01
        if det <= 0:
            view_depths[i] = 1e6
            continue

        inv_det = 1.0 / det
        conic_x = float(c11 * inv_det)
        conic_y = float(-c01 * inv_det)
        conic_z = float(c00 * inv_det)

        # Radius from eigenvalues
        mid = 0.5 * (c00 + c11)
        half_diff = math.sqrt(max(mid*mid - det, 0.0))
        lambda_max = mid + half_diff
        splat_radius = math.ceil(3.0 * math.sqrt(lambda_max))

        # Project to NDC
        clip = proj @ np.array([vx, vy, vz, 1.0], dtype=np.float64)
        ndc_x = float(clip[0] / clip[3])
        ndc_y = float(clip[1] / clip[3])
        ndc_z = float(clip[2] / clip[3])

        # Opacity (sigmoid)
        opacity = 1.0 / (1.0 + math.exp(-float(opacities_logit[i])))

        # SH color (degree 0)
        if sh_coeffs:
            sh0 = sh_coeffs[0][i]
            color = np.clip(SH_C0 * sh0 + 0.5, 0, 1)
        else:
            color = np.array([1.0, 1.0, 1.0])

        proj_centers[i] = [ndc_x, ndc_y, ndc_z, splat_radius]
        proj_conics[i] = [conic_x, conic_y, conic_z, opacity]
        proj_colors[i] = [color[0], color[1], color[2], opacity]
        view_depths[i] = vz  # negative = in front

    # --- Sort back-to-front ---
    sorted_indices = np.argsort(view_depths)  # most negative first = farthest

    # --- Rasterize (premultiplied alpha, back-to-front, matches GPU exactly) ---
    # GPU clears to opaque black (0,0,0,1)
    fb = np.zeros((height, width, 3), dtype=np.float64)
    alpha_cutoff = 0.004
    srgb = True  # match the default gaussian_splat.lux color_space: srgb

    for idx in sorted_indices:
        ndc_x, ndc_y, ndc_z, splat_radius = proj_centers[idx]
        conic_x, conic_y, conic_z, opacity = proj_conics[idx]
        cr, cg, cb, _ = proj_colors[idx]

        if splat_radius <= 0 or opacity <= 0:
            continue

        # NDC to pixel center
        px = (ndc_x * 0.5 + 0.5) * width
        py = (ndc_y * 0.5 + 0.5) * height

        r = int(splat_radius)
        x0 = max(0, int(px - r))
        x1 = min(width, int(px + r) + 1)
        y0 = max(0, int(py - r))
        y1 = min(height, int(py + r) + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Vectorized Gaussian evaluation over the bounding box
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx = xx - px
        dy = yy - py

        power = -0.5 * (conic_x * dx*dx + 2*conic_y * dx*dy + conic_z * dy*dy)

        # Discard fragments outside Gaussian tail
        gauss = np.where(power > -4.0, np.exp(power), 0.0)
        alpha = gauss * opacity
        alpha = np.where(alpha >= alpha_cutoff, alpha, 0.0)

        if not np.any(alpha > 0):
            continue

        # sRGB conversion (linear → sRGB)
        if srgb:
            cr = cr ** 0.45454545 if cr > 0 else 0.0
            cg = cg ** 0.45454545 if cg > 0 else 0.0
            cb = cb ** 0.45454545 if cb > 0 else 0.0

        # GPU premultiplied alpha blending: dst = src_premul + (1 - src_a) * dst
        # src_premul.rgb = color.rgb * alpha
        one_minus_a = 1.0 - alpha
        region = fb[y0:y1, x0:x1]
        region[:, :, 0] = cr * alpha + one_minus_a * region[:, :, 0]
        region[:, :, 1] = cg * alpha + one_minus_a * region[:, :, 1]
        region[:, :, 2] = cb * alpha + one_minus_a * region[:, :, 2]

    # Convert to uint8 RGBA (opaque output, matches GPU clear alpha=1)
    fb = np.clip(fb, 0, 1)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[:, :, :3] = (fb * 255).astype(np.uint8)
    result[:, :, 3] = 255
    return result


def interactive_splat_viewer(
    splat_glb_path: Path,
    width: int = 512,
    height: int = 512,
) -> None:
    """Open an interactive matplotlib window for orbiting around a splat scene.

    Controls: drag to orbit, scroll to zoom, ESC/Q to exit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton

    splat = load_splat_glb(splat_glb_path)
    n = splat['num_splats']
    print(f"Loaded {n} splats (SH degree {splat['sh_degree']})")

    positions = splat['positions']
    bb_min, bb_max = positions.min(axis=0), positions.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5
    if radius < 0.001:
        radius = 1.0

    # Orbit camera state: azimuth, elevation, distance
    state = {
        'azimuth': 0.0,       # radians
        'elevation': 0.2,     # radians
        'distance': radius * 3.0,
        'dragging': False,
        'last_x': 0,
        'last_y': 0,
    }

    def _eye_from_orbit():
        az, el, dist = state['azimuth'], state['elevation'], state['distance']
        x = dist * math.cos(el) * math.sin(az)
        y = dist * math.sin(el)
        z = dist * math.cos(el) * math.cos(az)
        return center + np.array([x, y, z], dtype=np.float32)

    def _render():
        eye = _eye_from_orbit()
        return render_splats_from_data(splat, width, height, eye=eye, target=center)

    # Initial render
    print("Rendering initial view...")
    pixels = _render()

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    fig.canvas.manager.set_window_title('Lux Gaussian Splat Viewer')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    img_plot = ax.imshow(pixels)

    def _refresh():
        print("Re-rendering...", end=' ', flush=True)
        pixels = _render()
        img_plot.set_data(pixels)
        fig.canvas.draw_idle()
        print("done.")

    def on_press(event):
        if event.button == MouseButton.LEFT:
            state['dragging'] = True
            state['last_x'] = event.x
            state['last_y'] = event.y

    def on_release(event):
        if event.button == MouseButton.LEFT and state['dragging']:
            state['dragging'] = False
            _refresh()

    def on_motion(event):
        if not state['dragging'] or event.x is None or event.y is None:
            return
        dx = event.x - state['last_x']
        dy = event.y - state['last_y']
        state['last_x'] = event.x
        state['last_y'] = event.y
        state['azimuth'] -= dx * 0.01
        state['elevation'] = max(-math.pi / 2 + 0.01,
                                  min(math.pi / 2 - 0.01,
                                      state['elevation'] + dy * 0.01))

    def on_scroll(event):
        factor = 0.9 if event.step > 0 else 1.1
        state['distance'] = max(radius * 0.5, state['distance'] * factor)
        _refresh()

    def on_key(event):
        if event.key in ('escape', 'q'):
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Interactive viewer: drag to orbit, scroll to zoom, ESC/Q to exit")
    plt.show()


def save_png(pixels: np.ndarray, output_path: Path) -> None:
    """Save an (H, W, 4) uint8 RGBA array as a PNG file."""
    img = Image.fromarray(pixels, "RGBA")
    img.save(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render compiled Lux SPIR-V shaders offscreen",
    )
    parser.add_argument("vert_spv", nargs='?', type=Path,
                        help="Vertex shader .vert.spv (triangle mode)")
    parser.add_argument("frag_spv", nargs='?', type=Path,
                        help="Fragment shader .frag.spv (triangle mode)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.png"),
                        help="Output PNG path (default: output.png)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    # Splat mode
    parser.add_argument("--splat-comp", type=Path, help="Compute shader .comp.spv")
    parser.add_argument("--splat-vert", type=Path, help="Vertex shader .vert.spv")
    parser.add_argument("--splat-frag", type=Path, help="Fragment shader .frag.spv")
    parser.add_argument("--splat-scene", type=Path, help="Splat scene .glb file")
    parser.add_argument("--interactive", action="store_true",
                        help="Open interactive orbit viewer (splat mode only)")
    # Fullscreen mode
    parser.add_argument("--fullscreen", type=Path,
                        help="Fragment shader for fullscreen quad mode")
    args = parser.parse_args()

    is_splat = args.splat_scene is not None

    if is_splat:
        if not args.splat_scene.exists():
            print(f"Error: --splat-scene not found: {args.splat_scene}", file=sys.stderr)
            sys.exit(1)
        if args.interactive:
            interactive_splat_viewer(
                args.splat_scene, width=args.width, height=args.height)
            sys.exit(0)
        print(f"Splat mode (CPU): scene={args.splat_scene.name}")
        pixels = render_splats(
            args.splat_comp, args.splat_vert, args.splat_frag, args.splat_scene,
            width=args.width, height=args.height,
        )
    elif args.fullscreen:
        if not args.fullscreen.exists():
            print(f"Error: shader not found: {args.fullscreen}", file=sys.stderr)
            sys.exit(1)
        print(f"Fullscreen mode: {args.fullscreen}")
        pixels = render_fullscreen(
            args.fullscreen, width=args.width, height=args.height)
    else:
        if args.vert_spv is None or args.frag_spv is None:
            parser.error("Provide vert_spv and frag_spv, or use --splat-* / --fullscreen")
        if not args.vert_spv.exists():
            print(f"Error: vertex shader not found: {args.vert_spv}", file=sys.stderr)
            sys.exit(1)
        if not args.frag_spv.exists():
            print(f"Error: fragment shader not found: {args.frag_spv}", file=sys.stderr)
            sys.exit(1)
        print(f"Triangle mode: {args.vert_spv}, {args.frag_spv}")
        pixels = render_triangle(
            args.vert_spv, args.frag_spv,
            width=args.width, height=args.height,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_png(pixels, args.output)
    print(f"Saved {args.width}x{args.height} render to {args.output}")

    non_black = (pixels[:, :, :3].sum(axis=2) > 0).sum()
    total = pixels.shape[0] * pixels.shape[1]
    coverage = non_black / total * 100
    print(f"Coverage: {coverage:.1f}% ({non_black} non-black pixels)")


if __name__ == "__main__":
    main()
