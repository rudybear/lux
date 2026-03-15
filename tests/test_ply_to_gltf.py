"""Tests for the enhanced PLY-to-glTF Gaussian splat converter.

Tests cover:
- PLY header parsing and data reading
- SH degree auto-detection (degree 0, 1, 2, 3)
- Coordinate conversion (Z-up to Y-up)
- Opacity sigmoid transform and auto-detection
- Scale exp transform and auto-detection
- Round-trip verification (PLY -> glTF -> compare)
- POSITION accessor min/max validation
- Centering and decimation
- Batch processing
- Edge cases (single splat, degree-0 only, empty properties)
"""

import json
import math
import os
import struct
import tempfile

import numpy as np
import pytest

from tools.ply_to_gltf import (
    auto_detect_opacity_space,
    auto_detect_scale_space,
    batch_convert,
    center_positions,
    convert_positions_z_up_to_y_up,
    convert_quaternions_z_up_to_y_up,
    convert_scales_z_up_to_y_up,
    decimate_data,
    detect_sh_degree,
    encode_glb,
    load_glb,
    logit,
    ply_info,
    ply_to_gltf,
    read_accessor,
    read_ply,
    read_ply_header,
    sigmoid,
    verify_round_trip,
    write_ply,
)


# ---------------------------------------------------------------------------
# Helpers: generate test PLY files
# ---------------------------------------------------------------------------

def _make_ply_binary(tmp_path, splats, sh_degree=0, filename="test.ply",
                     extra_rest_coeffs=None):
    """Create a binary PLY file with the given splat data.

    Args:
        tmp_path: Directory for the file.
        splats: List of dicts with keys: x, y, z, rot_0..3, scale_0..2,
                opacity, f_dc_0..2. Optional: f_rest_* for higher SH.
        sh_degree: SH degree (determines f_rest_* property count).
        filename: Output filename.
        extra_rest_coeffs: If provided, list of f_rest values per splat
                          (flat array of all rest coeffs for each splat).

    Returns:
        Path to the created PLY file.
    """
    n = len(splats)
    # Base properties
    base_props = [
        'x', 'y', 'z',
        'nx', 'ny', 'nz',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
        'opacity',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
    ]

    # f_rest properties for higher SH degrees
    rest_count = 0
    if sh_degree >= 1:
        rest_count = 3 * sum(2 * l + 1 for l in range(1, sh_degree + 1))
    rest_props = [f'f_rest_{i}' for i in range(rest_count)]
    all_props = base_props + rest_props

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for p in all_props:
        header += f"property float {p}\n"
    header += "end_header\n"

    path = os.path.join(str(tmp_path), filename)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        for i, s in enumerate(splats):
            # Base data
            vals = [
                s.get('x', 0.0), s.get('y', 0.0), s.get('z', 0.0),
                0.0, 0.0, 0.0,  # normals
                s.get('f_dc_0', 0.0), s.get('f_dc_1', 0.0), s.get('f_dc_2', 0.0),
                s.get('opacity', 0.0),
                s.get('scale_0', 0.0), s.get('scale_1', 0.0), s.get('scale_2', 0.0),
                s.get('rot_0', 1.0), s.get('rot_1', 0.0), s.get('rot_2', 0.0),
                s.get('rot_3', 0.0),
            ]
            # SH rest coefficients
            if rest_count > 0:
                if extra_rest_coeffs is not None:
                    vals.extend(extra_rest_coeffs[i])
                else:
                    vals.extend([0.0] * rest_count)
            for v in vals:
                f.write(struct.pack('<f', v))

    return path


def _make_ply_ascii(tmp_path, splats, filename="test_ascii.ply"):
    """Create an ASCII PLY file with basic properties."""
    n = len(splats)
    props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity', 'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3']

    header = "ply\nformat ascii 1.0\n"
    header += f"element vertex {n}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    lines = [header]
    for s in splats:
        vals = [
            s.get('x', 0.0), s.get('y', 0.0), s.get('z', 0.0),
            s.get('f_dc_0', 0.0), s.get('f_dc_1', 0.0), s.get('f_dc_2', 0.0),
            s.get('opacity', 0.0),
            s.get('scale_0', 0.0), s.get('scale_1', 0.0), s.get('scale_2', 0.0),
            s.get('rot_0', 1.0), s.get('rot_1', 0.0), s.get('rot_2', 0.0),
            s.get('rot_3', 0.0),
        ]
        lines.append(' '.join(str(v) for v in vals) + '\n')

    path = os.path.join(str(tmp_path), filename)
    with open(path, 'w') as f:
        f.writelines(lines)
    return path


def _default_splat(**overrides):
    """Return a default splat dict with optional overrides."""
    d = {
        'x': 1.0, 'y': 2.0, 'z': 3.0,
        'f_dc_0': 0.5, 'f_dc_1': 0.3, 'f_dc_2': 0.1,
        'opacity': 2.0,  # logit space (sigmoid(2.0) ~ 0.88)
        'scale_0': -3.0, 'scale_1': -3.5, 'scale_2': -4.0,  # log space
        'rot_0': 1.0, 'rot_1': 0.0, 'rot_2': 0.0, 'rot_3': 0.0,
    }
    d.update(overrides)
    return d


# =========================================================================
# 1. PLY Header Parsing Tests
# =========================================================================

class TestPlyParsing:
    """Test PLY header parsing and data reading."""

    def test_read_binary_ply(self, tmp_path):
        """Read a binary PLY file and verify vertex count and property names."""
        splats = [_default_splat(), _default_splat(x=2.0)]
        path = _make_ply_binary(tmp_path, splats)
        data = read_ply(path)
        assert data['_vertex_count'] == 2
        assert 'x' in data['_properties']
        assert 'rot_0' in data['_properties']
        assert len(data['x']) == 2

    def test_read_ascii_ply(self, tmp_path):
        """Read an ASCII PLY file and verify values match."""
        splats = [_default_splat(x=5.0, y=6.0, z=7.0)]
        path = _make_ply_ascii(tmp_path, splats)
        data = read_ply(path)
        assert data['_vertex_count'] == 1
        assert float(data['x'][0]) == pytest.approx(5.0)
        assert float(data['y'][0]) == pytest.approx(6.0)
        assert float(data['z'][0]) == pytest.approx(7.0)

    def test_read_ply_returns_numpy_arrays(self, tmp_path):
        """PLY reader should return numpy arrays for property data."""
        splats = [_default_splat()]
        path = _make_ply_binary(tmp_path, splats)
        data = read_ply(path)
        assert isinstance(data['x'], np.ndarray)

    def test_read_ply_header_only(self, tmp_path):
        """read_ply_header should return count and properties without loading data."""
        splats = [_default_splat(), _default_splat()]
        path = _make_ply_binary(tmp_path, splats)
        count, props = read_ply_header(path)
        assert count == 2
        names = [name for name, _ in props]
        assert 'x' in names
        assert 'opacity' in names

    def test_read_ply_not_a_ply(self, tmp_path):
        """Reading a non-PLY file should raise ValueError."""
        path = os.path.join(str(tmp_path), "bad.ply")
        with open(path, 'w') as f:
            f.write("not a ply file\n")
        with pytest.raises(ValueError, match="Not a PLY file"):
            read_ply(path)

    def test_read_ply_property_types_stored(self, tmp_path):
        """_property_types should store (name, dtype) tuples."""
        splats = [_default_splat()]
        path = _make_ply_binary(tmp_path, splats)
        data = read_ply(path)
        assert '_property_types' in data
        assert all(isinstance(t, tuple) and len(t) == 2 for t in data['_property_types'])


# =========================================================================
# 2. SH Degree Detection Tests
# =========================================================================

class TestSHDegreeDetection:
    """Test SH degree auto-detection from property names."""

    def test_degree_0_no_rest(self):
        """No f_rest_* properties -> degree 0."""
        props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        assert detect_sh_degree(props) == 0

    def test_degree_1_nine_rest(self):
        """9 f_rest_* properties -> degree 1."""
        props = ['f_dc_0', 'f_dc_1', 'f_dc_2']
        # degree 1: 3 * (2*1+1) = 9 rest coefficients
        props += [f'f_rest_{i}' for i in range(9)]
        assert detect_sh_degree(props) == 1

    def test_degree_2_twenty_four_rest(self):
        """24 f_rest_* properties -> degree 2."""
        props = ['f_dc_0', 'f_dc_1', 'f_dc_2']
        # degree 2: 3*(3+5) = 24 rest coefficients
        props += [f'f_rest_{i}' for i in range(24)]
        assert detect_sh_degree(props) == 2

    def test_degree_3_forty_five_rest(self):
        """45 f_rest_* properties -> degree 3."""
        props = ['f_dc_0', 'f_dc_1', 'f_dc_2']
        # degree 3: 3*(3+5+7) = 45 rest coefficients
        props += [f'f_rest_{i}' for i in range(45)]
        assert detect_sh_degree(props) == 3

    def test_degree_fallback(self):
        """Unusual rest count should use fallback calculation."""
        props = ['f_dc_0', 'f_dc_1', 'f_dc_2']
        # 12 rest coeffs doesn't match any exact degree
        props += [f'f_rest_{i}' for i in range(12)]
        degree = detect_sh_degree(props)
        assert degree == 1  # 12 // 9 = 1

    def test_degree_from_ply_file(self, tmp_path):
        """Detection should work on actual PLY file property names."""
        splats = [_default_splat()]
        # SH degree 1: 9 rest coeffs
        path = _make_ply_binary(tmp_path, splats, sh_degree=1,
                                extra_rest_coeffs=[[0.0] * 9])
        data = read_ply(path)
        assert detect_sh_degree(data['_properties']) == 1


# =========================================================================
# 3. Coordinate Conversion Tests
# =========================================================================

class TestCoordinateConversion:
    """Test Z-up to Y-up coordinate system conversion."""

    def test_position_z_up_to_y_up(self):
        """Z-up (x, y, z) -> Y-up (x, z, -y)."""
        pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = convert_positions_z_up_to_y_up(pos)
        np.testing.assert_allclose(result[0], [1.0, 3.0, -2.0])

    def test_position_conversion_batch(self):
        """Batch position conversion should handle multiple points."""
        pos = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        result = convert_positions_z_up_to_y_up(pos)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1], [0.0, 0.0, -1.0])
        np.testing.assert_allclose(result[2], [0.0, 1.0, 0.0])

    def test_position_origin_unchanged(self):
        """Origin point should remain at origin after conversion."""
        pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        result = convert_positions_z_up_to_y_up(pos)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0])

    def test_quaternion_identity_stays_identity(self):
        """Identity quaternion (w=1, xyz=0) should remain identity after conversion.

        In WXYZ format: [1, 0, 0, 0] -> XYZW: [0, 0, 0, 1]
        """
        quat_wxyz = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result = convert_quaternions_z_up_to_y_up(quat_wxyz)
        # Identity in XYZW is [0, 0, 0, 1]
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0, 1.0], atol=1e-7)

    def test_quaternion_normalization(self):
        """Non-unit quaternions should be normalized."""
        quat_wxyz = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        result = convert_quaternions_z_up_to_y_up(quat_wxyz)
        length = np.sqrt(np.sum(result[0] ** 2))
        assert length == pytest.approx(1.0, abs=1e-7)

    def test_scale_axis_reorder(self):
        """Scale Y and Z axes should swap during conversion."""
        scales = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        result = convert_scales_z_up_to_y_up(scales)
        np.testing.assert_allclose(result[0], [1.0, 3.0, 2.0])

    def test_no_convert_preserves_positions(self, tmp_path):
        """With --no-convert, positions should pass through unchanged (aside from float32)."""
        splats = [_default_splat(x=1.0, y=2.0, z=3.0)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, convert_coords=False,
                     raw_opacity=True, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        pos = read_accessor(gltf, bin_data, 0)
        assert pos[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert pos[0, 1] == pytest.approx(2.0, abs=1e-5)
        assert pos[0, 2] == pytest.approx(3.0, abs=1e-5)


# =========================================================================
# 4. Opacity Sigmoid Transform Tests
# =========================================================================

class TestOpacityTransform:
    """Test opacity sigmoid transform and auto-detection."""

    def test_sigmoid_basic(self):
        """sigmoid(0) = 0.5."""
        assert float(sigmoid(np.array([0.0]))[0]) == pytest.approx(0.5)

    def test_sigmoid_positive(self):
        """sigmoid(large positive) ~ 1.0."""
        result = sigmoid(np.array([10.0]))
        assert float(result[0]) == pytest.approx(1.0, abs=1e-4)

    def test_sigmoid_negative(self):
        """sigmoid(large negative) ~ 0.0."""
        result = sigmoid(np.array([-10.0]))
        assert float(result[0]) == pytest.approx(0.0, abs=1e-4)

    def test_sigmoid_logit_roundtrip(self):
        """sigmoid(logit(x)) should round-trip for values in (0, 1)."""
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.95])
        result = sigmoid(logit(values))
        np.testing.assert_allclose(result, values, atol=1e-6)

    def test_auto_detect_linear_opacity(self):
        """Values all in [0, 1] should be detected as linear."""
        values = np.array([0.0, 0.5, 0.8, 1.0])
        assert auto_detect_opacity_space(values) is True

    def test_auto_detect_logit_opacity(self):
        """Values outside [0, 1] should be detected as logit."""
        values = np.array([-2.0, 0.5, 3.0])
        assert auto_detect_opacity_space(values) is False

    def test_auto_detect_negative_opacity(self):
        """Negative values should be detected as logit."""
        values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        assert auto_detect_opacity_space(values) is False

    def test_opacity_transform_in_conversion(self, tmp_path):
        """Logit opacity should be transformed to linear sigmoid in GLB output."""
        logit_val = 2.0  # sigmoid(2.0) ~ 0.8808
        expected = 1.0 / (1.0 + math.exp(-logit_val))
        splats = [_default_splat(opacity=logit_val)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        attrs = gltf['meshes'][0]['primitives'][0]['attributes']
        opa = read_accessor(gltf, bin_data, attrs['KHR_gaussian_splatting:OPACITY'])
        assert float(opa[0]) == pytest.approx(expected, abs=1e-5)

    def test_raw_opacity_skips_transform(self, tmp_path):
        """--raw-opacity should pass through opacity values unchanged."""
        splats = [_default_splat(opacity=0.75)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_opacity=True, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        attrs = gltf['meshes'][0]['primitives'][0]['attributes']
        opa = read_accessor(gltf, bin_data, attrs['KHR_gaussian_splatting:OPACITY'])
        assert float(opa[0]) == pytest.approx(0.75, abs=1e-5)


# =========================================================================
# 5. Scale Exp Transform Tests
# =========================================================================

class TestScaleTransform:
    """Test scale exp transform and auto-detection."""

    def test_auto_detect_linear_scale(self):
        """Positive values < 10 should be detected as linear."""
        values = np.array([[0.5, 0.3, 0.1], [1.0, 2.0, 0.05]])
        assert auto_detect_scale_space(values) is True

    def test_auto_detect_log_scale_negative(self):
        """Negative values should be detected as log-space."""
        values = np.array([[-3.0, -4.0, -5.0]])
        assert auto_detect_scale_space(values) is False

    def test_auto_detect_log_scale_large(self):
        """Values >= 10 should be detected as log-space."""
        values = np.array([[0.5, 0.3, 15.0]])
        assert auto_detect_scale_space(values) is False

    def test_scale_kept_in_log_space(self, tmp_path):
        """KHR spec: scales stay in log-space (no exp transform)."""
        log_scale = -3.0
        splats = [_default_splat(scale_0=log_scale, scale_1=log_scale, scale_2=log_scale)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_opacity=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        attrs = gltf['meshes'][0]['primitives'][0]['attributes']
        scales = read_accessor(gltf, bin_data, attrs['KHR_gaussian_splatting:SCALE'])
        # Scales kept in log-space per KHR spec — all same value after coord swap
        for j in range(3):
            assert float(scales[0, j]) == pytest.approx(log_scale, abs=1e-5)

    def test_raw_scale_skips_transform(self, tmp_path):
        """--raw-scale should pass through scale values unchanged."""
        splats = [_default_splat(scale_0=0.5, scale_1=0.3, scale_2=0.1)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_opacity=True, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        attrs = gltf['meshes'][0]['primitives'][0]['attributes']
        scales = read_accessor(gltf, bin_data, attrs['KHR_gaussian_splatting:SCALE'])
        # With convert_coords=True, axes reorder: (0.5, 0.3, 0.1) -> (0.5, 0.1, 0.3)
        assert float(scales[0, 0]) == pytest.approx(0.5, abs=1e-5)
        assert float(scales[0, 1]) == pytest.approx(0.1, abs=1e-5)
        assert float(scales[0, 2]) == pytest.approx(0.3, abs=1e-5)


# =========================================================================
# 6. Round-Trip Verification Tests
# =========================================================================

class TestRoundTrip:
    """Test PLY -> glTF -> verify round-trip accuracy."""

    def test_round_trip_basic(self, tmp_path):
        """Basic PLY -> GLB round-trip should pass verification."""
        splats = [
            _default_splat(x=1.0, y=2.0, z=3.0),
            _default_splat(x=-1.0, y=0.5, z=-0.5),
        ]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)
        errors = verify_round_trip(ply_path, glb_path, quiet=True)
        assert all(v < 1e-5 for v in errors.values())

    def test_round_trip_no_convert(self, tmp_path):
        """Round-trip without coordinate conversion should also pass."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, convert_coords=False, quiet=True)
        errors = verify_round_trip(ply_path, glb_path,
                                    convert_coords=False, quiet=True)
        assert all(v < 1e-5 for v in errors.values())

    def test_round_trip_raw_modes(self, tmp_path):
        """Round-trip with raw opacity and raw scale should pass."""
        splats = [_default_splat(opacity=0.8, scale_0=0.1, scale_1=0.2, scale_2=0.3)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_opacity=True, raw_scale=True, quiet=True)
        errors = verify_round_trip(ply_path, glb_path,
                                    raw_opacity=True, raw_scale=True, quiet=True)
        assert all(v < 1e-5 for v in errors.values())

    def test_round_trip_sh_dc(self, tmp_path):
        """SH DC coefficients should survive the round-trip."""
        splats = [_default_splat(f_dc_0=1.5, f_dc_1=-0.3, f_dc_2=0.7)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)
        errors = verify_round_trip(ply_path, glb_path, quiet=True)
        assert errors['sh_dc'] < 1e-5

    def test_round_trip_fails_on_bad_data(self, tmp_path):
        """Verification should raise ValueError when data doesn't match."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        # Convert with coords, verify without -> mismatch
        ply_to_gltf(ply_path, glb_path, convert_coords=True, quiet=True)
        with pytest.raises(ValueError, match="Round-trip verification failed"):
            verify_round_trip(ply_path, glb_path, convert_coords=False,
                              raw_opacity=True, raw_scale=True,
                              threshold=1e-5, quiet=True)

    def test_round_trip_multiple_splats(self, tmp_path):
        """Round-trip with multiple splats should match all of them."""
        splats = [
            _default_splat(x=float(i), y=float(i * 0.1), z=float(-i * 0.5),
                          opacity=float(i) * 0.5 + 0.1,  # logit space
                          rot_0=1.0, rot_1=0.0, rot_2=0.0, rot_3=0.0)
            for i in range(10)
        ]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)
        errors = verify_round_trip(ply_path, glb_path, quiet=True)
        assert all(v < 1e-4 for v in errors.values())


# =========================================================================
# 7. POSITION Accessor Min/Max Tests
# =========================================================================

class TestPositionMinMax:
    """Test that POSITION accessor has min/max fields (glTF spec requirement)."""

    def test_position_has_min_max(self, tmp_path):
        """POSITION accessor must have min and max fields."""
        splats = [_default_splat(x=1.0, y=2.0, z=3.0)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)

        gltf, _ = load_glb(glb_path)
        pos_accessor = gltf['accessors'][0]
        assert 'min' in pos_accessor
        assert 'max' in pos_accessor
        assert len(pos_accessor['min']) == 3
        assert len(pos_accessor['max']) == 3

    def test_position_min_max_correct(self, tmp_path):
        """Position min/max should match actual vertex positions."""
        splats = [
            _default_splat(x=-5.0, y=0.0, z=10.0),
            _default_splat(x=5.0, y=-3.0, z=-10.0),
            _default_splat(x=0.0, y=7.0, z=0.0),
        ]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        # Use no-convert so positions are predictable
        ply_to_gltf(ply_path, glb_path, convert_coords=False,
                     raw_opacity=True, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        pos_accessor = gltf['accessors'][0]
        positions = read_accessor(gltf, bin_data, 0)

        actual_min = positions.min(axis=0).tolist()
        actual_max = positions.max(axis=0).tolist()

        for i in range(3):
            assert pos_accessor['min'][i] == pytest.approx(actual_min[i], abs=1e-5)
            assert pos_accessor['max'][i] == pytest.approx(actual_max[i], abs=1e-5)

    def test_position_min_max_with_conversion(self, tmp_path):
        """Min/max should be computed AFTER coordinate conversion."""
        splats = [_default_splat(x=1.0, y=2.0, z=3.0)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, convert_coords=True,
                     raw_opacity=True, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        pos_accessor = gltf['accessors'][0]
        positions = read_accessor(gltf, bin_data, 0)
        # After Z-up to Y-up: (1, 2, 3) -> (1, 3, -2)
        assert pos_accessor['min'][0] == pytest.approx(1.0, abs=1e-5)
        assert pos_accessor['min'][1] == pytest.approx(3.0, abs=1e-5)
        assert pos_accessor['min'][2] == pytest.approx(-2.0, abs=1e-5)


# =========================================================================
# 8. Auto-Detection Tests
# =========================================================================

class TestAutoDetection:
    """Test auto-detection of linear vs log/logit space."""

    def test_auto_detect_opacity_all_zero_one(self):
        """All values in [0,1] -> linear."""
        vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert auto_detect_opacity_space(vals) is True

    def test_auto_detect_opacity_single_zero(self):
        """Single 0.0 value -> linear."""
        vals = np.array([0.0])
        assert auto_detect_opacity_space(vals) is True

    def test_auto_detect_opacity_single_negative(self):
        """Single negative value -> logit."""
        vals = np.array([-1.0])
        assert auto_detect_opacity_space(vals) is False

    def test_auto_detect_opacity_above_one(self):
        """Value > 1 -> logit."""
        vals = np.array([0.5, 1.5])
        assert auto_detect_opacity_space(vals) is False

    def test_auto_detect_scale_positive_small(self):
        """Small positive values -> linear."""
        vals = np.array([[0.01, 0.02, 0.03], [0.5, 1.0, 2.0]])
        assert auto_detect_scale_space(vals) is True

    def test_auto_detect_scale_has_zero(self):
        """Zero value -> not linear (must be positive)."""
        vals = np.array([[0.0, 0.5, 1.0]])
        assert auto_detect_scale_space(vals) is False

    def test_auto_detect_scale_has_negative(self):
        """Negative value -> log-space."""
        vals = np.array([[-3.0, -4.0, -5.0]])
        assert auto_detect_scale_space(vals) is False

    def test_auto_detect_in_conversion_linear_opacity(self, tmp_path):
        """If opacity values are all in [0,1], auto-detect should keep them as-is."""
        splats = [_default_splat(opacity=0.8)]  # Already linear
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        attrs = gltf['meshes'][0]['primitives'][0]['attributes']
        opa = read_accessor(gltf, bin_data, attrs['KHR_gaussian_splatting:OPACITY'])
        # 0.8 is in [0,1] -> auto-detect says linear -> kept as-is
        assert float(opa[0]) == pytest.approx(0.8, abs=1e-5)


# =========================================================================
# 9. Edge Cases
# =========================================================================

class TestEdgeCases:
    """Test edge cases: single splat, degree-0 only, centering, decimation."""

    def test_single_splat(self, tmp_path):
        """Converting a PLY with a single splat should work."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        result = ply_to_gltf(ply_path, glb_path, quiet=True)
        assert result['splat_count'] == 1
        assert result['sh_degree'] == 0

    def test_degree_0_only_sh(self, tmp_path):
        """SH degree 0 should only have DC band, no rest buffers."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats, sh_degree=0)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)

        gltf, _ = load_glb(glb_path)
        # 5 accessors: pos, rot, scale, opacity, sh_dc
        assert len(gltf['accessors']) == 5

    def test_sh_degree_1_extra_accessors(self, tmp_path):
        """SH degree 1 should have 8 accessors (5 base + 3 per-coefficient VEC3)."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats, sh_degree=1,
                                    extra_rest_coeffs=[[0.01 * i for i in range(9)]])
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)

        gltf, _ = load_glb(glb_path)
        # 5 base (pos, rot, scale, opa, sh_dc) + 3 VEC3 per-coefficient for degree 1
        assert len(gltf['accessors']) == 8

    def test_center_positions(self, tmp_path):
        """--center should center the point cloud at origin."""
        splats = [
            _default_splat(x=10.0, y=20.0, z=30.0),
            _default_splat(x=20.0, y=40.0, z=50.0),
        ]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        result = ply_to_gltf(ply_path, glb_path, do_center=True,
                              convert_coords=False, raw_opacity=True,
                              raw_scale=True, quiet=True)

        gltf, bin_data = load_glb(glb_path)
        pos = read_accessor(gltf, bin_data, 0)
        center = (pos.min(axis=0) + pos.max(axis=0)) / 2.0
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-5)

    def test_decimate_reduces_count(self, tmp_path):
        """--decimate should reduce the splat count."""
        splats = [_default_splat(x=float(i), opacity=float(i) * 0.1)
                  for i in range(20)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        result = ply_to_gltf(ply_path, glb_path, decimate_count=5,
                              quiet=True)
        assert result['splat_count'] == 5

    def test_decimate_larger_than_count(self, tmp_path):
        """Decimating to more than count should keep all splats."""
        splats = [_default_splat() for _ in range(3)]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        result = ply_to_gltf(ply_path, glb_path, decimate_count=100,
                              quiet=True)
        assert result['splat_count'] == 3

    def test_decimate_by_opacity_keeps_highest(self):
        """Opacity-based decimation should keep the highest opacity splats."""
        data = {
            'x': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            'opacity': np.array([0.1, 0.9, 0.3, 0.8, 0.2]),
            '_vertex_count': 5,
            '_properties': ['x', 'opacity'],
        }
        result = decimate_data(data, 2, method='opacity')
        assert result['_vertex_count'] == 2
        # Should keep indices 1 (0.9) and 3 (0.8)
        np.testing.assert_array_equal(result['opacity'], [0.9, 0.8])

    def test_extensions_used_not_required(self, tmp_path):
        """glTF output must use extensionsUsed (NOT extensionsRequired)."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)

        gltf, _ = load_glb(glb_path)
        assert 'extensionsUsed' in gltf
        assert 'KHR_gaussian_splatting' in gltf['extensionsUsed']
        assert 'extensionsRequired' not in gltf

    def test_glb_valid_structure(self, tmp_path):
        """GLB output should have valid glTF 2.0 structure."""
        splats = [_default_splat()]
        ply_path = _make_ply_binary(tmp_path, splats)
        glb_path = os.path.join(str(tmp_path), "test.glb")
        ply_to_gltf(ply_path, glb_path, quiet=True)

        gltf, _ = load_glb(glb_path)
        assert gltf['asset']['version'] == '2.0'
        assert gltf['asset']['generator'] == 'lux-ply-to-gltf'
        assert len(gltf['meshes']) == 1
        assert len(gltf['nodes']) == 1
        assert gltf['scene'] == 0


# =========================================================================
# 10. PLY Info Tests
# =========================================================================

class TestPlyInfo:
    """Test PLY info reporting."""

    def test_info_basic(self, tmp_path):
        """ply_info should return splat count and SH degree."""
        splats = [_default_splat(x=1.0, y=2.0, z=3.0)]
        ply_path = _make_ply_binary(tmp_path, splats)
        info = ply_info(ply_path)
        assert info['splat_count'] == 1
        assert info['sh_degree'] == 0
        assert len(info['position_min']) == 3
        assert len(info['position_max']) == 3

    def test_info_multiple_splats(self, tmp_path):
        """ply_info should report correct bounds for multiple splats."""
        splats = [
            _default_splat(x=-5.0, y=0.0, z=10.0),
            _default_splat(x=5.0, y=3.0, z=-10.0),
        ]
        ply_path = _make_ply_binary(tmp_path, splats)
        info = ply_info(ply_path)
        assert info['splat_count'] == 2
        assert info['position_min'][0] == pytest.approx(-5.0)
        assert info['position_max'][0] == pytest.approx(5.0)


# =========================================================================
# 11. Write PLY Tests
# =========================================================================

class TestWritePly:
    """Test PLY writing for round-trip support."""

    def test_write_and_read_back(self, tmp_path):
        """Written PLY should read back with matching values."""
        n = 3
        data = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([4.0, 5.0, 6.0]),
            'z': np.array([7.0, 8.0, 9.0]),
        }
        props = ['x', 'y', 'z']
        path = os.path.join(str(tmp_path), "written.ply")
        write_ply(path, data, n, props)

        readback = read_ply(path)
        assert readback['_vertex_count'] == 3
        np.testing.assert_allclose(readback['x'], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(readback['y'], [4.0, 5.0, 6.0], atol=1e-5)
        np.testing.assert_allclose(readback['z'], [7.0, 8.0, 9.0], atol=1e-5)


# =========================================================================
# 12. Batch Processing Tests
# =========================================================================

class TestBatchProcessing:
    """Test batch directory conversion."""

    def test_batch_converts_all_plys(self, tmp_path):
        """Batch mode should convert all PLY files in a directory."""
        input_dir = os.path.join(str(tmp_path), "input")
        output_dir = os.path.join(str(tmp_path), "output")
        os.makedirs(input_dir)

        for name in ['a', 'b', 'c']:
            splats = [_default_splat(x=float(ord(name)))]
            _make_ply_binary(tmp_path, splats,
                            filename=os.path.join("input", f"{name}.ply"))

        results = batch_convert(input_dir, output_dir,
                                raw_opacity=True, raw_scale=True)
        assert len(results) == 3
        for ply_path, glb_path, result in results:
            assert not isinstance(result, Exception)
            assert os.path.exists(glb_path)

    def test_batch_empty_dir(self, tmp_path):
        """Batch on an empty directory should return empty results."""
        input_dir = os.path.join(str(tmp_path), "empty")
        os.makedirs(input_dir)
        results = batch_convert(input_dir)
        assert results == []

    def test_batch_default_output_dir(self, tmp_path):
        """Default output directory should be same as input directory."""
        input_dir = os.path.join(str(tmp_path), "both")
        os.makedirs(input_dir)
        splats = [_default_splat()]
        _make_ply_binary(tmp_path, splats,
                        filename=os.path.join("both", "test.ply"))

        results = batch_convert(input_dir, raw_opacity=True, raw_scale=True)
        assert len(results) == 1
        _, glb_path, _ = results[0]
        assert os.path.dirname(glb_path) == input_dir


# =========================================================================
# 13. GLB Encoding Tests
# =========================================================================

class TestGlbEncoding:
    """Test low-level GLB encoding."""

    def test_encode_glb_magic(self):
        """GLB output should start with the glTF magic bytes."""
        gltf_json = {"asset": {"version": "2.0"}}
        data = encode_glb(gltf_json, [b'\x00' * 4])
        magic = struct.unpack('<I', data[:4])[0]
        assert magic == 0x46546C67  # 'glTF'

    def test_encode_glb_version(self):
        """GLB header should contain version 2."""
        gltf_json = {"asset": {"version": "2.0"}}
        data = encode_glb(gltf_json, [b'\x00' * 4])
        version = struct.unpack('<I', data[4:8])[0]
        assert version == 2

    def test_encode_glb_alignment(self):
        """GLB binary chunk should be 4-byte aligned."""
        gltf_json = {"asset": {"version": "2.0"}}
        # Non-aligned buffer (3 bytes)
        data = encode_glb(gltf_json, [b'\x01\x02\x03'])
        length = struct.unpack('<I', data[8:12])[0]
        assert length == len(data)
        # JSON chunk and BIN chunk should both be 4-byte aligned
        json_len = struct.unpack('<I', data[12:16])[0]
        assert json_len % 4 == 0

    def test_encode_glb_roundtrip(self, tmp_path):
        """Encoded GLB should be parseable by load_glb."""
        gltf_json = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 4}],
        }
        data = encode_glb(gltf_json, [b'\x01\x02\x03\x04'])
        path = os.path.join(str(tmp_path), "test.glb")
        with open(path, 'wb') as f:
            f.write(data)
        loaded_gltf, loaded_bin = load_glb(path)
        assert loaded_gltf['asset']['version'] == '2.0'
        assert loaded_bin[:4] == b'\x01\x02\x03\x04'


# =========================================================================
# 14. Center Positions Tests
# =========================================================================

class TestCenterPositions:
    """Test point cloud centering."""

    def test_center_basic(self):
        """Centering should move the midpoint to origin."""
        data = {
            'x': np.array([10.0, 20.0]),
            'y': np.array([30.0, 40.0]),
            'z': np.array([50.0, 60.0]),
        }
        offset = center_positions(data)
        np.testing.assert_allclose(offset, [15.0, 35.0, 55.0])
        np.testing.assert_allclose(data['x'], [-5.0, 5.0])
        np.testing.assert_allclose(data['y'], [-5.0, 5.0])
        np.testing.assert_allclose(data['z'], [-5.0, 5.0])

    def test_center_already_centered(self):
        """Centering an already-centered cloud should be a no-op."""
        data = {
            'x': np.array([-1.0, 1.0]),
            'y': np.array([-2.0, 2.0]),
            'z': np.array([-3.0, 3.0]),
        }
        offset = center_positions(data)
        np.testing.assert_allclose(offset, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(data['x'], [-1.0, 1.0])


# =========================================================================
# 15. Integration with existing test assets
# =========================================================================

class TestExistingAssets:
    """Test against existing test asset files if available."""

    @pytest.fixture
    def assets_dir(self):
        """Path to the test assets directory."""
        d = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.isdir(d):
            pytest.skip("tests/assets directory not found")
        return d

    def test_load_existing_glb(self, assets_dir):
        """Existing GLB test assets should be loadable."""
        glb_files = [f for f in os.listdir(assets_dir) if f.endswith('.glb')]
        assert len(glb_files) > 0, "No GLB files in assets dir"
        for name in glb_files:
            path = os.path.join(assets_dir, name)
            gltf, bin_data = load_glb(path)
            assert gltf['asset']['version'] == '2.0'
            assert 'KHR_gaussian_splatting' in gltf.get('extensionsUsed', [])

    def test_existing_glb_has_position_min_max(self, assets_dir):
        """Existing GLB test assets should have POSITION min/max."""
        glb_files = [f for f in os.listdir(assets_dir) if f.endswith('.glb')]
        for name in glb_files:
            path = os.path.join(assets_dir, name)
            gltf, _ = load_glb(path)
            pos_accessor = gltf['accessors'][0]
            assert 'min' in pos_accessor, f"{name} missing POSITION min"
            assert 'max' in pos_accessor, f"{name} missing POSITION max"
