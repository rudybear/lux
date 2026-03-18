"""Diagnostic tool: read a KHR_gaussian_splatting GLB and inspect splat data.

Extracts ~100 splats from the center of the scene (median position +/- small
radius) and prints per-splat attributes alongside aggregate statistics.
Useful for verifying data integrity when renderers produce garbled output.

Usage:
    python tools/diag_splats.py C:/Users/rudyb/Downloads/point_cloud_noconv.glb
"""

import json
import math
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# GLB parsing (same approach as render_harness.py / glb_to_ply.py)
# ---------------------------------------------------------------------------

def load_glb(path):
    """Parse a .glb and return (gltf_json, bin_data)."""
    with open(path, 'rb') as f:
        magic, version, length = struct.unpack('<III', f.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"Not a GLB file (magic=0x{magic:08X}): {path}")
        print(f"GLB header: version={version}, total_length={length} bytes")

        # JSON chunk
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError("First chunk is not JSON")
        json_bytes = f.read(chunk_len)
        gltf = json.loads(json_bytes)
        print(f"JSON chunk: {chunk_len} bytes")

        # BIN chunk
        chunk_len, chunk_type = struct.unpack('<II', f.read(8))
        if chunk_type != 0x004E4942:
            raise ValueError("Second chunk is not BIN")
        bin_data = f.read(chunk_len)
        print(f"BIN  chunk: {chunk_len} bytes")

    return gltf, bin_data


def read_accessor(gltf, bin_data, accessor_idx):
    """Read an accessor as a numpy array. Supports float (5126) and
    unsigned byte (5121) component types."""
    acc = gltf['accessors'][accessor_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)

    type_map = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    count = acc['count']
    components = type_map.get(acc['type'], 1)
    comp_type = acc['componentType']

    if comp_type == 5126:  # FLOAT
        dtype = np.float32
    elif comp_type == 5121:  # UNSIGNED_BYTE
        dtype = np.uint8
    elif comp_type == 5123:  # UNSIGNED_SHORT
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported componentType: {comp_type}")

    byte_stride = bv.get('byteStride', 0)
    elem_size = np.dtype(dtype).itemsize * components

    if byte_stride and byte_stride != elem_size:
        # Interleaved buffer: read element-by-element respecting stride
        result = np.empty(count * components, dtype=dtype)
        for i in range(count):
            src_off = offset + i * byte_stride
            elem = np.frombuffer(bin_data, dtype=dtype,
                                 count=components, offset=src_off)
            result[i * components:(i + 1) * components] = elem
        data = result
    else:
        data = np.frombuffer(bin_data, dtype=dtype,
                             count=count * components, offset=offset)

    if dtype != np.float32:
        data = data.astype(np.float32)
        # Apply normalization if accessor says so
        if acc.get('normalized', False):
            if comp_type == 5121:
                data /= 255.0
            elif comp_type == 5123:
                data /= 65535.0

    return data.reshape(count, components) if components > 1 else data.copy()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814  # 0.5 * sqrt(1/pi)


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def logit(p):
    """Inverse sigmoid: log(p/(1-p))."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Attribute discovery
# ---------------------------------------------------------------------------

def find_splat_accessor(attrs, gs_ext, name):
    """Find the accessor index for a splat attribute by checking multiple
    naming conventions (KHR prefix, extension dict, underscore prefix)."""
    # KHR conformance prefix
    khr_key = f'KHR_gaussian_splatting:{name}'
    if khr_key in attrs:
        return attrs[khr_key], 'khr_attr'
    # Underscore prefix
    us_key = f'_{name}'
    if us_key in attrs:
        return attrs[us_key], 'underscore_attr'
    # Extension attributes dict
    if gs_ext is not None:
        gs_attrs = gs_ext.get('attributes', {})
        if name in gs_attrs:
            return gs_attrs[name], 'ext_attrs'
    return None, None


def find_sh_accessors(attrs, gs_ext):
    """Locate SH coefficient accessors. Returns dict: degree -> list of
    accessor indices.  Avoids double-counting when both attribute names
    and extension sh list reference the same accessor."""
    sh_map = {}
    seen_indices = set()

    # Check KHR conformance attributes first (preferred source)
    for aname in attrs:
        if 'SH_DEGREE_' in aname:
            rest = aname.split('SH_DEGREE_')[1]
            parts = rest.split('_COEF_')
            if len(parts) == 2:
                degree = int(parts[0])
                idx = attrs[aname]
                if idx not in seen_indices:
                    sh_map.setdefault(degree, []).append(idx)
                    seen_indices.add(idx)

    # Check extension sh list (skip indices already found)
    if gs_ext is not None:
        for sh_entry in gs_ext.get('sh', []):
            degree = sh_entry['degree']
            coeff_val = sh_entry['coefficients']
            if isinstance(coeff_val, list):
                for idx in coeff_val:
                    if idx not in seen_indices:
                        sh_map.setdefault(degree, []).append(idx)
                        seen_indices.add(idx)
            else:
                if coeff_val not in seen_indices:
                    sh_map.setdefault(degree, []).append(coeff_val)
                    seen_indices.add(coeff_val)

    # Check underscore _SH_N attributes
    for aname in attrs:
        if aname.startswith('_SH_'):
            try:
                degree = int(aname[4:])
                idx = attrs[aname]
                if idx not in seen_indices:
                    sh_map.setdefault(degree, []).append(idx)
                    seen_indices.add(idx)
            except ValueError:
                pass

    return sh_map


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def diagnose(glb_path, sample_target=100, radius_percentile=10):
    print(f"=== Gaussian Splat Diagnostic: {glb_path} ===\n")

    gltf, bin_data = load_glb(glb_path)

    # Print top-level glTF info
    asset = gltf.get('asset', {})
    print(f"\nAsset: generator={asset.get('generator','?')}, "
          f"version={asset.get('version','?')}")
    exts_used = gltf.get('extensionsUsed', [])
    exts_req = gltf.get('extensionsRequired', [])
    print(f"extensionsUsed: {exts_used}")
    print(f"extensionsRequired: {exts_req}")
    print(f"Meshes: {len(gltf.get('meshes', []))}")
    print(f"Accessors: {len(gltf.get('accessors', []))}")
    print(f"BufferViews: {len(gltf.get('bufferViews', []))}")
    print(f"Buffers: {len(gltf.get('buffers', []))}")

    # Find the splat primitive
    mesh = gltf['meshes'][0]
    prim = mesh['primitives'][0]
    attrs = prim.get('attributes', {})
    gs_ext = prim.get('extensions', {}).get('KHR_gaussian_splatting')
    mode = prim.get('mode', 4)

    print(f"\nPrimitive mode: {mode} ({'POINTS' if mode == 0 else 'TRIANGLES' if mode == 4 else mode})")
    print(f"Attributes: {list(attrs.keys())}")
    if gs_ext:
        print(f"KHR_gaussian_splatting extension: {json.dumps(gs_ext, indent=2)[:500]}")

    # Detect KHR conformance format (linear opacity) vs internal (logit opacity)
    is_khr_conformance = any(k.startswith('KHR_gaussian_splatting:') for k in attrs)
    print(f"KHR conformance attribute naming: {is_khr_conformance}")

    # Read position data
    pos_acc_idx = attrs['POSITION']
    positions = read_accessor(gltf, bin_data, pos_acc_idx)
    n = positions.shape[0]
    print(f"\n--- Position Data ({n} splats) ---")
    print(f"  min: ({positions[:, 0].min():.4f}, {positions[:, 1].min():.4f}, {positions[:, 2].min():.4f})")
    print(f"  max: ({positions[:, 0].max():.4f}, {positions[:, 1].max():.4f}, {positions[:, 2].max():.4f})")
    median = np.median(positions, axis=0)
    mean = np.mean(positions, axis=0)
    print(f"  mean:   ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"  median: ({median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f})")

    # Read rotations
    rot_idx, rot_source = find_splat_accessor(attrs, gs_ext, 'ROTATION')
    if rot_idx is not None:
        rotations = read_accessor(gltf, bin_data, rot_idx)
        print(f"\n--- Rotation Data (source: {rot_source}, accessor {rot_idx}) ---")
        print(f"  shape: {rotations.shape}, dtype: {rotations.dtype}")
        # Check quaternion norms
        if rotations.ndim == 2 and rotations.shape[1] == 4:
            norms = np.linalg.norm(rotations, axis=1)
            print(f"  quaternion norms: min={norms.min():.6f}, max={norms.max():.6f}, "
                  f"mean={norms.mean():.6f}")
            bad_norms = np.sum((norms < 0.99) | (norms > 1.01))
            if bad_norms > 0:
                print(f"  WARNING: {bad_norms} quaternions have norm outside [0.99, 1.01]!")
        else:
            print(f"  WARNING: unexpected rotation shape {rotations.shape}")
    else:
        rotations = None
        print("\n  WARNING: no rotation data found!")

    # Read scales
    scl_idx, scl_source = find_splat_accessor(attrs, gs_ext, 'SCALE')
    if scl_idx is not None:
        scales = read_accessor(gltf, bin_data, scl_idx)
        print(f"\n--- Scale Data (source: {scl_source}, accessor {scl_idx}) ---")
        print(f"  shape: {scales.shape}, dtype: {scales.dtype}")
        if scales.ndim == 2:
            print(f"  log-scale range: [{scales.min():.4f}, {scales.max():.4f}]")
            exp_scales = np.exp(scales)
            print(f"  exp(scale) range: [{exp_scales.min():.6f}, {exp_scales.max():.6f}]")
            # Check for NaN/Inf
            nan_count = np.sum(np.isnan(scales))
            inf_count = np.sum(np.isinf(scales))
            if nan_count or inf_count:
                print(f"  WARNING: {nan_count} NaN, {inf_count} Inf values in scales!")
            # Detect if scales might already be linear (not log-space)
            all_positive = np.all(scales > 0)
            has_negatives = np.any(scales < 0)
            print(f"  All positive (might be linear): {all_positive}")
            print(f"  Has negatives (likely log-space): {has_negatives}")
        else:
            print(f"  WARNING: unexpected scale shape {scales.shape}")
    else:
        scales = None
        print("\n  WARNING: no scale data found!")

    # Read opacities
    opa_idx, opa_source = find_splat_accessor(attrs, gs_ext, 'OPACITY')
    if opa_idx is not None:
        opacities = read_accessor(gltf, bin_data, opa_idx)
        print(f"\n--- Opacity Data (source: {opa_source}, accessor {opa_idx}) ---")
        print(f"  shape: {opacities.shape}, dtype: {opacities.dtype}")
        print(f"  raw range: [{opacities.min():.4f}, {opacities.max():.4f}]")

        # Determine if these are linear or logit
        in_01 = np.all(opacities >= 0) and np.all(opacities <= 1.0)
        print(f"  All in [0,1] (linear): {in_01}")
        if in_01:
            print(f"  Interpretation: LINEAR opacity (KHR conformance)")
            lin_opacities = opacities
            logit_opacities = logit(opacities)
        else:
            print(f"  Interpretation: LOGIT opacity (internal format)")
            logit_opacities = opacities
            lin_opacities = sigmoid(opacities)

        print(f"  linear opacity range: [{lin_opacities.min():.6f}, {lin_opacities.max():.6f}]")
        print(f"  logit opacity range:  [{logit_opacities.min():.4f}, {logit_opacities.max():.4f}]")

        # NaN check
        nan_count = np.sum(np.isnan(opacities))
        if nan_count:
            print(f"  WARNING: {nan_count} NaN values in opacities!")
    else:
        opacities = None
        lin_opacities = None
        logit_opacities = None
        print("\n  WARNING: no opacity data found!")

    # Read SH coefficients
    sh_map = find_sh_accessors(attrs, gs_ext)
    print(f"\n--- SH Data ---")
    print(f"  Degrees found: {sorted(sh_map.keys())}")

    sh_dc = None
    for degree in sorted(sh_map.keys()):
        acc_indices = sh_map[degree]
        # Read and merge all coefficient accessors for this degree
        parts = [read_accessor(gltf, bin_data, idx) for idx in acc_indices]
        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = np.hstack(parts) if all(p.ndim == 2 for p in parts) else parts[0]
        print(f"  Degree {degree}: {len(acc_indices)} accessor(s), "
              f"merged shape={merged.shape}, range=[{merged.min():.4f}, {merged.max():.4f}]")
        if degree == 0:
            sh_dc = merged  # (N, 3) DC coefficients

    if sh_dc is not None:
        if sh_dc.ndim == 1:
            sh_dc = sh_dc.reshape(-1, 3)
        dc_colors = SH_C0 * sh_dc + 0.5
        dc_colors_clipped = np.clip(dc_colors, 0, 1)
        print(f"\n  DC color (SH_C0 * dc + 0.5):")
        print(f"    raw range: [{dc_colors.min():.4f}, {dc_colors.max():.4f}]")
        print(f"    clipped to [0,1]:")
        print(f"      R: [{dc_colors_clipped[:, 0].min():.4f}, {dc_colors_clipped[:, 0].max():.4f}], "
              f"mean={dc_colors_clipped[:, 0].mean():.4f}")
        print(f"      G: [{dc_colors_clipped[:, 1].min():.4f}, {dc_colors_clipped[:, 1].max():.4f}], "
              f"mean={dc_colors_clipped[:, 1].mean():.4f}")
        print(f"      B: [{dc_colors_clipped[:, 2].min():.4f}, {dc_colors_clipped[:, 2].max():.4f}], "
              f"mean={dc_colors_clipped[:, 2].mean():.4f}")
        nan_count = np.sum(np.isnan(sh_dc))
        if nan_count:
            print(f"    WARNING: {nan_count} NaN values in SH DC!")
    else:
        dc_colors = None
        print("  WARNING: no SH DC (degree 0) found!")

    # -----------------------------------------------------------------------
    # Sample ~100 splats near the center (median position)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"=== Sampling ~{sample_target} splats near scene center ===")
    print(f"{'='*70}")

    # Compute distance from each splat to the median position
    dists = np.linalg.norm(positions - median, axis=1)
    dist_sorted = np.sort(dists)

    # Start with a small radius and expand until we get enough splats
    # Use percentile-based radius
    if n <= sample_target:
        indices = np.arange(n)
    else:
        # Find radius that captures ~sample_target splats
        target_radius = dist_sorted[min(sample_target, n - 1)]
        indices = np.where(dists <= target_radius)[0]

    actual_count = len(indices)
    if actual_count > sample_target * 2:
        # Subsample if too many
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, sample_target, replace=False)
        indices = np.sort(indices)
        actual_count = len(indices)

    max_dist = dists[indices].max() if len(indices) > 0 else 0
    print(f"  Center (median): ({median[0]:.4f}, {median[1]:.4f}, {median[2]:.4f})")
    print(f"  Sampling radius: {max_dist:.4f}")
    print(f"  Sampled {actual_count} splats (of {n} total)\n")

    # Print per-splat data for the sample
    print(f"{'Idx':>8s}  {'pos_x':>9s} {'pos_y':>9s} {'pos_z':>9s}  "
          f"{'qx':>8s} {'qy':>8s} {'qz':>8s} {'qw':>8s}  "
          f"{'logS_x':>8s} {'logS_y':>8s} {'logS_z':>8s}  "
          f"{'expS_x':>8s} {'expS_y':>8s} {'expS_z':>8s}  "
          f"{'opa_lin':>8s} {'opa_logit':>9s}  "
          f"{'dc_r':>7s} {'dc_g':>7s} {'dc_b':>7s}  "
          f"{'col_r':>6s} {'col_g':>6s} {'col_b':>6s}")
    print("-" * 220)

    for i in indices[:actual_count]:
        px, py, pz = positions[i]

        if rotations is not None and rotations.ndim == 2:
            qx, qy, qz, qw = rotations[i]
        else:
            qx = qy = qz = 0.0
            qw = 1.0

        if scales is not None and scales.ndim == 2:
            sx, sy, sz = scales[i]
            esx, esy, esz = np.exp(scales[i])
        else:
            sx = sy = sz = 0.0
            esx = esy = esz = 1.0

        opa_l = float(lin_opacities[i]) if lin_opacities is not None else 0.0
        opa_g = float(logit_opacities[i]) if logit_opacities is not None else 0.0

        if sh_dc is not None:
            dcr, dcg, dcb = sh_dc[i] if sh_dc.ndim == 2 else (sh_dc[i], 0, 0)
            cr = SH_C0 * dcr + 0.5
            cg = SH_C0 * dcg + 0.5
            cb = SH_C0 * dcb + 0.5
        else:
            dcr = dcg = dcb = 0.0
            cr = cg = cb = 0.5

        print(f"{i:8d}  {px:9.4f} {py:9.4f} {pz:9.4f}  "
              f"{qx:8.4f} {qy:8.4f} {qz:8.4f} {qw:8.4f}  "
              f"{sx:8.4f} {sy:8.4f} {sz:8.4f}  "
              f"{esx:8.5f} {esy:8.5f} {esz:8.5f}  "
              f"{opa_l:8.5f} {opa_g:9.4f}  "
              f"{dcr:7.3f} {dcg:7.3f} {dcb:7.3f}  "
              f"{cr:6.3f} {cg:6.3f} {cb:6.3f}")

    # -----------------------------------------------------------------------
    # Aggregate statistics for the sample
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"=== Aggregate statistics for {actual_count} sampled splats ===")
    print(f"{'='*70}")

    # Positions
    s_pos = positions[indices]
    print(f"\nPositions:")
    print(f"  mean:  ({s_pos[:, 0].mean():.4f}, {s_pos[:, 1].mean():.4f}, {s_pos[:, 2].mean():.4f})")
    print(f"  std:   ({s_pos[:, 0].std():.4f}, {s_pos[:, 1].std():.4f}, {s_pos[:, 2].std():.4f})")

    # Scales
    if scales is not None and scales.ndim == 2:
        s_scales = scales[indices]
        s_exp = np.exp(s_scales)
        print(f"\nLog-scales:")
        print(f"  mean:  ({s_scales[:, 0].mean():.4f}, {s_scales[:, 1].mean():.4f}, {s_scales[:, 2].mean():.4f})")
        print(f"  std:   ({s_scales[:, 0].std():.4f}, {s_scales[:, 1].std():.4f}, {s_scales[:, 2].std():.4f})")
        print(f"Exp(scales):")
        print(f"  mean:  ({s_exp[:, 0].mean():.6f}, {s_exp[:, 1].mean():.6f}, {s_exp[:, 2].mean():.6f})")
        print(f"  std:   ({s_exp[:, 0].std():.6f}, {s_exp[:, 1].std():.6f}, {s_exp[:, 2].std():.6f})")
        print(f"  min:   ({s_exp[:, 0].min():.6f}, {s_exp[:, 1].min():.6f}, {s_exp[:, 2].min():.6f})")
        print(f"  max:   ({s_exp[:, 0].max():.6f}, {s_exp[:, 1].max():.6f}, {s_exp[:, 2].max():.6f})")

    # Opacities
    if lin_opacities is not None:
        s_opa = lin_opacities[indices]
        print(f"\nLinear opacities:")
        print(f"  mean: {s_opa.mean():.6f}")
        print(f"  std:  {s_opa.std():.6f}")
        print(f"  min:  {s_opa.min():.6f}")
        print(f"  max:  {s_opa.max():.6f}")
        # Histogram
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(s_opa, bins=bins)
        print(f"  histogram: {list(zip([f'{b:.1f}' for b in bins[:-1]], hist.tolist()))}")

    # DC Colors
    if sh_dc is not None:
        s_dc = sh_dc[indices]
        if s_dc.ndim == 1:
            s_dc = s_dc.reshape(-1, 3)
        s_colors = SH_C0 * s_dc + 0.5
        s_colors_clipped = np.clip(s_colors, 0, 1)
        print(f"\nDC Colors (SH_C0 * dc + 0.5):")
        print(f"  mean:  ({s_colors[:, 0].mean():.4f}, {s_colors[:, 1].mean():.4f}, {s_colors[:, 2].mean():.4f})")
        print(f"  std:   ({s_colors[:, 0].std():.4f}, {s_colors[:, 1].std():.4f}, {s_colors[:, 2].std():.4f})")
        print(f"  min:   ({s_colors[:, 0].min():.4f}, {s_colors[:, 1].min():.4f}, {s_colors[:, 2].min():.4f})")
        print(f"  max:   ({s_colors[:, 0].max():.4f}, {s_colors[:, 1].max():.4f}, {s_colors[:, 2].max():.4f})")
        out_of_range = np.sum((s_colors < -0.5) | (s_colors > 1.5))
        print(f"  values outside [-0.5, 1.5]: {out_of_range} (of {s_colors.size})")

    # Rotations
    if rotations is not None and rotations.ndim == 2:
        s_rot = rotations[indices]
        s_norms = np.linalg.norm(s_rot, axis=1)
        print(f"\nQuaternion norms:")
        print(f"  mean: {s_norms.mean():.6f}")
        print(f"  std:  {s_norms.std():.6f}")
        print(f"  min:  {s_norms.min():.6f}")
        print(f"  max:  {s_norms.max():.6f}")

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"=== Sanity Checks ===")
    print(f"{'='*70}")

    issues = 0

    # Check quaternion ordering (XYZW vs WXYZ)
    if rotations is not None and rotations.ndim == 2:
        # In XYZW format, w is column 3. For identity quats, w should be ~1
        w_col3 = rotations[:, 3]
        w_col0 = rotations[:, 0]
        # Count how many have |w| near 1
        near1_col3 = np.sum(np.abs(w_col3) > 0.9)
        near1_col0 = np.sum(np.abs(w_col0) > 0.9)
        print(f"\nQuaternion ordering test:")
        print(f"  |col[3]| > 0.9: {near1_col3}/{n} ({100*near1_col3/n:.1f}%) -- expected high if XYZW")
        print(f"  |col[0]| > 0.9: {near1_col0}/{n} ({100*near1_col0/n:.1f}%) -- expected high if WXYZ")
        if near1_col0 > near1_col3 * 2:
            print(f"  LIKELY WXYZ FORMAT (scalar-first), not XYZW!")
            issues += 1
        elif near1_col3 > near1_col0:
            print(f"  Appears to be XYZW format (scalar-last) -- correct for glTF")

    # Check opacity space
    if opacities is not None:
        if is_khr_conformance and not in_01:
            print(f"\nOpacity space mismatch:")
            print(f"  KHR conformance naming detected but opacity NOT in [0,1]!")
            print(f"  This suggests opacity is in logit space despite KHR naming.")
            issues += 1
        elif not is_khr_conformance and in_01:
            print(f"\nOpacity note:")
            print(f"  Non-KHR naming but opacity IS in [0,1].")
            print(f"  This might be linear opacity without KHR naming convention.")

    # Check scale space
    if scales is not None and scales.ndim == 2:
        if np.all(scales > 0) and np.all(scales < 5):
            print(f"\nScale space warning:")
            print(f"  All scale values are positive and small -- might already be LINEAR (not log)!")
            print(f"  If the renderer applies exp(), these would become [1.0, {np.exp(scales.max()):.1f}]")
            issues += 1

    # Check for all-zero data
    for name, arr in [('positions', positions), ('rotations', rotations),
                      ('scales', scales), ('opacities', opacities),
                      ('sh_dc', sh_dc)]:
        if arr is not None:
            zero_frac = np.sum(arr == 0) / arr.size
            if zero_frac > 0.5:
                print(f"\n  WARNING: {name} has {100*zero_frac:.1f}% zero values!")
                issues += 1

    # Check for NaN/Inf in all arrays
    for name, arr in [('positions', positions), ('rotations', rotations),
                      ('scales', scales), ('opacities', opacities),
                      ('sh_dc', sh_dc)]:
        if arr is not None:
            nans = np.sum(np.isnan(arr))
            infs = np.sum(np.isinf(arr))
            if nans or infs:
                print(f"\n  WARNING: {name} has {nans} NaN and {infs} Inf values!")
                issues += 1

    if issues == 0:
        print(f"\n  All sanity checks passed.")
    else:
        print(f"\n  {issues} potential issue(s) detected.")

    print(f"\n=== Diagnostic complete ===")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        path = "C:/Users/rudyb/Downloads/point_cloud_noconv.glb"
    else:
        path = sys.argv[1]

    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    diagnose(path, sample_target=sample)
