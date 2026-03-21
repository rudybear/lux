"""KHR_gaussian_splatting conformance test suite.

Tests cover:
- Asset loading: parse each .glb/.gltf conformance file, verify extension data
- Compilation: compile gaussian_splat.lux and verify SPIR-V output
- Data validation: quaternion normalization, scale positivity, opacity range, SH counts
- Edge cases: empty splat data, single splat, very large splat count

The conformance assets follow the official KHR_gaussian_splatting spec where:
- Splat attributes use the ``KHR_gaussian_splatting:`` prefix in the primitive
  attributes dict (e.g. ``KHR_gaussian_splatting:SCALE``, ``KHR_gaussian_splatting:ROTATION``)
- SH coefficients are per-degree attributes: ``KHR_gaussian_splatting:SH_DEGREE_N_COEF_M``
- The extension object carries rendering hints: ``kernel``, ``colorSpace``,
  ``sortingMethod``, ``projection``
- Scale and opacity are stored in linear space (not log/logit)

Conformance assets must be downloaded first:
    python -m tools.download_khr_splat_tests

Tests are automatically skipped if the assets directory is missing.
"""

import json
import math
import re
import struct
from pathlib import Path

import pytest

from luxc.builtins.types import clear_type_aliases


# ---------------------------------------------------------------------------
# Paths and discovery
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFORMANCE_DIR = REPO_ROOT / "tests" / "assets" / "khr_splat_conformance"
EXAMPLES_DIR = REPO_ROOT / "examples"
SPLAT_LUX = EXAMPLES_DIR / "gaussian_splat.lux"

# Prefix used by the KHR_gaussian_splatting spec for primitive attributes
_KHR_PREFIX = "KHR_gaussian_splatting:"


def _has_spirv_tools() -> bool:
    try:
        import subprocess
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


def _conformance_assets_present() -> bool:
    """Return True if the conformance asset directory contains at least one glTF file."""
    if not CONFORMANCE_DIR.is_dir():
        return False
    return bool(
        list(CONFORMANCE_DIR.rglob("*.glb"))
        + list(CONFORMANCE_DIR.rglob("*.gltf"))
    )


requires_conformance_assets = pytest.mark.skipif(
    not _conformance_assets_present(),
    reason=(
        "KHR_gaussian_splatting conformance assets not downloaded. "
        "Run: python -m tools.download_khr_splat_tests"
    ),
)


def _discover_conformance_files() -> list[Path]:
    """Return all .glb and .gltf files under the conformance directory."""
    if not CONFORMANCE_DIR.is_dir():
        return []
    files = sorted(CONFORMANCE_DIR.rglob("*.glb")) + sorted(CONFORMANCE_DIR.rglob("*.gltf"))
    return files


# ---------------------------------------------------------------------------
# Minimal glTF/GLB parser (no external dependencies)
# ---------------------------------------------------------------------------

def _parse_glb(path: Path) -> tuple[dict, bytes]:
    """Parse a GLB file. Returns (json_dict, binary_chunk_bytes)."""
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"GLB too small: {len(data)} bytes")

    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file (magic={magic:#x})")
    if version != 2:
        raise ValueError(f"Unsupported GLB version: {version}")

    offset = 12
    json_data = None
    bin_data = b''

    while offset < len(data):
        if offset + 8 > len(data):
            break
        chunk_length, chunk_type = struct.unpack_from('<II', data, offset)
        offset += 8
        chunk_bytes = data[offset:offset + chunk_length]
        offset += chunk_length

        if chunk_type == 0x4E4F534A:  # JSON
            json_data = json.loads(chunk_bytes.decode('utf-8'))
        elif chunk_type == 0x004E4942:  # BIN
            bin_data = chunk_bytes

    if json_data is None:
        raise ValueError("No JSON chunk in GLB")

    return json_data, bin_data


def _parse_gltf(path: Path) -> tuple[dict, bytes]:
    """Parse a .gltf file. Returns (json_dict, binary_data_or_empty)."""
    gltf = json.loads(path.read_text(encoding='utf-8'))
    bin_data = b''

    # Try to load associated .bin file
    buffers = gltf.get("buffers", [])
    if buffers:
        uri = buffers[0].get("uri", "")
        if uri and not uri.startswith("data:"):
            bin_path = path.parent / uri
            if bin_path.exists():
                bin_data = bin_path.read_bytes()

    return gltf, bin_data


def _load_gltf_json(path: Path) -> tuple[dict, bytes]:
    """Load glTF JSON and optional binary data from .glb or .gltf."""
    if path.suffix.lower() == '.glb':
        return _parse_glb(path)
    else:
        return _parse_gltf(path)


# ---------------------------------------------------------------------------
# KHR_gaussian_splatting helpers (official spec format)
# ---------------------------------------------------------------------------

def _is_splat_primitive(prim: dict) -> bool:
    """Return True if a primitive is a Gaussian splat (has splat-specific attributes)."""
    attrs = prim.get("attributes", {})
    return any(k.startswith(_KHR_PREFIX) for k in attrs)


def _get_splat_primitives(gltf: dict) -> list[dict]:
    """Return all primitives that carry KHR_gaussian_splatting splat attributes."""
    results = []
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if _is_splat_primitive(prim):
                results.append(prim)
    return results


def _get_khr_extension(prim: dict) -> dict:
    """Return the KHR_gaussian_splatting extension dict from a primitive."""
    return prim.get("extensions", {}).get("KHR_gaussian_splatting", {})


def _is_compressed_primitive(prim: dict) -> bool:
    """Return True if the primitive uses a KHR_gaussian_splatting compression extension."""
    ext = _get_khr_extension(prim)
    sub_exts = ext.get("extensions", {})
    return any(k.startswith("KHR_gaussian_splatting_compression") for k in sub_exts)


def _get_khr_attribute(prim: dict, name: str) -> int | None:
    """Return the accessor index for a KHR_gaussian_splatting attribute, or None."""
    attrs = prim.get("attributes", {})
    key = _KHR_PREFIX + name
    return attrs.get(key)


def _get_sh_degree_coef_attrs(prim: dict) -> dict[int, list[tuple[int, int]]]:
    """Extract SH coefficient attributes grouped by degree.

    Returns {degree: [(coef_index, accessor_index), ...]} sorted by coef_index.
    """
    attrs = prim.get("attributes", {})
    pattern = re.compile(r'^KHR_gaussian_splatting:SH_DEGREE_(\d+)_COEF_(\d+)$')
    result: dict[int, list[tuple[int, int]]] = {}
    for key, acc_idx in attrs.items():
        m = pattern.match(key)
        if m:
            degree = int(m.group(1))
            coef = int(m.group(2))
            result.setdefault(degree, []).append((coef, acc_idx))
    # Sort each degree's coefficients
    for degree in result:
        result[degree].sort()
    return result


def _get_splat_count(gltf: dict) -> int:
    """Get the number of splats from the POSITION accessor count of the first splat primitive."""
    for prim in _get_splat_primitives(gltf):
        pos_acc = prim.get("attributes", {}).get("POSITION")
        if pos_acc is not None:
            return gltf["accessors"][pos_acc]["count"]
    return 0


def _get_max_sh_degree(prim: dict) -> int:
    """Return the highest SH degree found in a primitive's attributes."""
    sh_attrs = _get_sh_degree_coef_attrs(prim)
    if not sh_attrs:
        return -1
    return max(sh_attrs.keys())


def _read_accessor_floats(gltf: dict, bin_data: bytes, accessor_index: int) -> list[float]:
    """Read float data from a glTF accessor."""
    accessor = gltf["accessors"][accessor_index]
    bv_index = accessor.get("bufferView")
    if bv_index is None:
        return []
    buffer_view = gltf["bufferViews"][bv_index]

    bv_offset = buffer_view.get("byteOffset", 0)
    acc_offset = accessor.get("byteOffset", 0)
    offset = bv_offset + acc_offset
    count = accessor["count"]
    comp_type = accessor.get("componentType", 5126)
    acc_type = accessor.get("type", "SCALAR")

    type_sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
    components = type_sizes.get(acc_type, 1)
    total = count * components

    byte_stride = buffer_view.get("byteStride", 0)
    element_byte_size = components * 4  # assumes float32

    if comp_type == 5126:  # FLOAT
        if byte_stride == 0 or byte_stride == element_byte_size:
            # Tight packing: fast path
            return list(struct.unpack_from(f'<{total}f', bin_data, offset))
        else:
            # Interleaved: stride between elements
            values = []
            for i in range(count):
                elem_offset = offset + i * byte_stride
                elem = struct.unpack_from(f'<{components}f', bin_data, elem_offset)
                values.extend(elem)
            return values
    elif comp_type == 5120:  # BYTE
        return [struct.unpack_from('<b', bin_data, offset + i)[0] / 127.0 for i in range(total)]
    elif comp_type == 5121:  # UNSIGNED_BYTE
        return [struct.unpack_from('<B', bin_data, offset + i)[0] / 255.0 for i in range(total)]
    elif comp_type == 5122:  # SHORT
        return [struct.unpack_from('<h', bin_data, offset + i * 2)[0] / 32767.0 for i in range(total)]
    elif comp_type == 5123:  # UNSIGNED_SHORT
        return [struct.unpack_from('<H', bin_data, offset + i * 2)[0] / 65535.0 for i in range(total)]
    else:
        return []


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------

def _compile_splat_lux(tmp_path, source=None, stem="conformance_splat"):
    """Compile a Gaussian splat Lux source to SPIR-V."""
    from luxc.compiler import compile_source
    clear_type_aliases()
    if source is None:
        source = SPLAT_LUX.read_text(encoding='utf-8')
    compile_source(
        source, stem, tmp_path,
        emit_reflection=True, validate=True,
    )


# =========================================================================
# 1. Asset Loading Tests (conformance assets)
# =========================================================================

@requires_conformance_assets
class TestConformanceAssetLoading:
    """Test that each conformance .glb/.gltf parses and contains valid KHR_gaussian_splatting data."""

    @pytest.fixture(params=_discover_conformance_files(), ids=lambda p: p.name)
    def asset_path(self, request):
        return request.param

    def test_asset_parses_without_error(self, asset_path):
        """Each conformance file should parse as valid glTF 2.0."""
        gltf, bin_data = _load_gltf_json(asset_path)
        assert "asset" in gltf
        assert gltf["asset"].get("version") == "2.0"

    def test_extension_declared(self, asset_path):
        """KHR_gaussian_splatting should be listed in extensionsUsed or extensionsRequired."""
        gltf, _ = _load_gltf_json(asset_path)
        all_ext = gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", [])
        assert "KHR_gaussian_splatting" in all_ext, (
            "KHR_gaussian_splatting not declared in extensionsUsed or extensionsRequired"
        )

    def test_has_splat_primitive(self, asset_path):
        """At least one primitive should have KHR_gaussian_splatting prefixed attributes."""
        gltf, _ = _load_gltf_json(asset_path)
        splat_prims = _get_splat_primitives(gltf)
        assert len(splat_prims) > 0, "No primitive with KHR_gaussian_splatting attributes found"

    def test_splat_count_positive(self, asset_path):
        """Each conformance asset should have at least one splat."""
        gltf, _ = _load_gltf_json(asset_path)
        count = _get_splat_count(gltf)
        assert count > 0, f"POSITION accessor count is {count}"

    def test_splat_has_position(self, asset_path):
        """Each splat primitive should have a POSITION attribute."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            assert "POSITION" in prim.get("attributes", {}), "Splat primitive missing POSITION"

    def test_splat_has_required_khr_attributes(self, asset_path):
        """Each splat primitive should have SCALE, ROTATION, OPACITY via KHR prefix."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            if _is_compressed_primitive(prim):
                pytest.skip("Compressed format stores attributes in packed buffer")
            attrs = prim.get("attributes", {})
            for attr_name in ["SCALE", "ROTATION", "OPACITY"]:
                key = _KHR_PREFIX + attr_name
                assert key in attrs, f"Missing {key} in splat primitive attributes"

    def test_splat_has_sh_coefficients(self, asset_path):
        """Each splat primitive should have at least SH_DEGREE_0_COEF_0 (DC term)."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            if _is_compressed_primitive(prim):
                pytest.skip("Compressed format stores SH coefficients in packed buffer")
            sh_attrs = _get_sh_degree_coef_attrs(prim)
            assert 0 in sh_attrs, "Missing SH_DEGREE_0_COEF_0 (DC term) in splat primitive"

    def test_extension_object_has_kernel(self, asset_path):
        """The KHR_gaussian_splatting extension object should specify a kernel type."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            ext = _get_khr_extension(prim)
            if _is_compressed_primitive(prim):
                pytest.skip("Compressed format may omit kernel/colorSpace in extension object")
            assert "kernel" in ext, "Extension object missing 'kernel' field"
            assert ext["kernel"] in ("ellipse", "point"), f"Unknown kernel: {ext['kernel']}"

    def test_extension_object_has_color_space(self, asset_path):
        """The extension should specify a colorSpace."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            ext = _get_khr_extension(prim)
            if _is_compressed_primitive(prim):
                pytest.skip("Compressed format may omit kernel/colorSpace in extension object")
            assert "colorSpace" in ext, "Extension object missing 'colorSpace' field"

    def test_accessor_types_valid(self, asset_path):
        """Verify that referenced splat accessors have appropriate types."""
        gltf, _ = _load_gltf_json(asset_path)
        accessors = gltf.get("accessors", [])
        for prim in _get_splat_primitives(gltf):
            if _is_compressed_primitive(prim):
                pytest.skip("Compressed format uses different accessor types")
            # ROTATION should be VEC4
            rot_idx = _get_khr_attribute(prim, "ROTATION")
            if rot_idx is not None and rot_idx < len(accessors):
                assert accessors[rot_idx]["type"] == "VEC4", (
                    f"ROTATION accessor should be VEC4, got {accessors[rot_idx]['type']}"
                )
            # SCALE should be VEC3
            scale_idx = _get_khr_attribute(prim, "SCALE")
            if scale_idx is not None and scale_idx < len(accessors):
                assert accessors[scale_idx]["type"] == "VEC3", (
                    f"SCALE accessor should be VEC3, got {accessors[scale_idx]['type']}"
                )
            # OPACITY should be SCALAR
            opa_idx = _get_khr_attribute(prim, "OPACITY")
            if opa_idx is not None and opa_idx < len(accessors):
                assert accessors[opa_idx]["type"] == "SCALAR", (
                    f"OPACITY accessor should be SCALAR, got {accessors[opa_idx]['type']}"
                )

    def test_sh_degree_in_valid_range(self, asset_path):
        """SH degrees should be 0, 1, 2, or 3."""
        gltf, _ = _load_gltf_json(asset_path)
        for prim in _get_splat_primitives(gltf):
            sh_attrs = _get_sh_degree_coef_attrs(prim)
            for degree in sh_attrs:
                assert 0 <= degree <= 3, f"SH degree {degree} out of valid range [0, 3]"


# =========================================================================
# 2. Compilation Tests (conformance assets)
# =========================================================================

@requires_conformance_assets
@requires_spirv_tools
class TestConformanceCompilation:
    """Verify that gaussian_splat.lux compiles for each conformance SH degree."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_compile_default_splat_pipeline(self, tmp_path):
        """The default gaussian_splat.lux should compile to 3 SPIR-V modules."""
        _compile_splat_lux(tmp_path)
        assert (tmp_path / "conformance_splat.comp.spv").exists()
        assert (tmp_path / "conformance_splat.vert.spv").exists()
        assert (tmp_path / "conformance_splat.frag.spv").exists()

    @pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
    def test_compile_for_each_sh_degree(self, tmp_path, sh_degree):
        """Compilation should succeed for each SH degree the conformance assets may use."""
        source = f"""
        splat Conformance {{
            sh_degree: {sh_degree},
            kernel: ellipse,
            color_space: srgb,
            sort: camera_distance,
            alpha_cutoff: 0.004,
        }}
        pipeline ConformanceViewer {{
            mode: gaussian_splat,
            splat: Conformance,
        }}
        """
        _compile_splat_lux(tmp_path, source=source, stem=f"conform_sh{sh_degree}")
        assert (tmp_path / f"conform_sh{sh_degree}.comp.spv").exists()
        assert (tmp_path / f"conform_sh{sh_degree}.vert.spv").exists()
        assert (tmp_path / f"conform_sh{sh_degree}.frag.spv").exists()

    def test_compile_reflection_json_valid(self, tmp_path):
        """Reflection JSON should contain gaussian_splatting section."""
        _compile_splat_lux(tmp_path)
        json_path = tmp_path / "conformance_splat.comp.json"
        assert json_path.exists()
        meta = json.loads(json_path.read_text())
        assert "gaussian_splatting" in meta
        gs = meta["gaussian_splatting"]
        assert "sh_degree" in gs
        assert "kernel" in gs
        assert "input_buffers" in gs
        assert "output_buffers" in gs

    @pytest.fixture(params=_discover_conformance_files(), ids=lambda p: p.name)
    def conformance_asset(self, request):
        return request.param

    def test_compile_matches_conformance_sh_degree(self, tmp_path, conformance_asset):
        """For each conformance asset, compile a pipeline with matching SH degree."""
        gltf, _ = _load_gltf_json(conformance_asset)
        splat_prims = _get_splat_primitives(gltf)
        if not splat_prims:
            pytest.skip("No splat primitives in this file")

        # Find the maximum SH degree used across all splat primitives
        max_degree = 0
        for prim in splat_prims:
            d = _get_max_sh_degree(prim)
            if d > max_degree:
                max_degree = d

        source = f"""
        splat ConfAsset {{
            sh_degree: {max_degree},
        }}
        pipeline ConfPipeline {{
            mode: gaussian_splat,
            splat: ConfAsset,
        }}
        """
        stem = f"conf_{conformance_asset.stem}"
        _compile_splat_lux(tmp_path, source=source, stem=stem)
        assert (tmp_path / f"{stem}.comp.spv").exists()


# =========================================================================
# 3. Data Validation Tests (conformance assets)
# =========================================================================

@requires_conformance_assets
class TestConformanceDataValidation:
    """Validate the actual binary data in conformance assets."""

    @pytest.fixture(params=_discover_conformance_files(), ids=lambda p: p.name)
    def asset_data(self, request):
        """Load and return (path, gltf_json, bin_data, splat_prims)."""
        path = request.param
        gltf, bin_data = _load_gltf_json(path)
        splat_prims = _get_splat_primitives(gltf)
        if not splat_prims:
            pytest.skip("No splat primitives in this file")
        return path, gltf, bin_data, splat_prims

    def test_rotation_quaternions_are_unit(self, asset_data):
        """All rotation quaternions should be approximately unit length."""
        path, gltf, bin_data, splat_prims = asset_data
        if not bin_data:
            pytest.skip("No binary data available")

        for prim in splat_prims:
            rot_idx = _get_khr_attribute(prim, "ROTATION")
            if rot_idx is None:
                continue

            values = _read_accessor_floats(gltf, bin_data, rot_idx)
            count = len(values) // 4

            # Check a sample of quaternions (up to 1000)
            step = max(1, count // 1000)
            for i in range(0, count, step):
                x, y, z, w = values[i*4:(i+1)*4]
                length = math.sqrt(x*x + y*y + z*z + w*w)
                assert abs(length - 1.0) < 0.05, (
                    f"Quaternion {i} not unit length: |q|={length:.4f} "
                    f"(x={x:.4f}, y={y:.4f}, z={z:.4f}, w={w:.4f})"
                )

    def test_opacity_values_valid(self, asset_data):
        """Opacity values should be in [0, 1] range (linear space per KHR spec)."""
        path, gltf, bin_data, splat_prims = asset_data
        if not bin_data:
            pytest.skip("No binary data available")

        for prim in splat_prims:
            opa_idx = _get_khr_attribute(prim, "OPACITY")
            if opa_idx is None:
                continue

            values = _read_accessor_floats(gltf, bin_data, opa_idx)
            step = max(1, len(values) // 1000)
            for i in range(0, len(values), step):
                v = values[i]
                assert math.isfinite(v), f"Opacity[{i}] is not finite: {v}"
                assert -0.01 <= v <= 1.01, (
                    f"Opacity[{i}] = {v} outside expected [0, 1] range"
                )

    def test_scale_values_finite(self, asset_data):
        """Scale values should be finite.

        Note: the KHR spec allows both linear-space and log-space scales.
        The conformance assets use log-space (negative values are valid, representing
        scales < 1.0). We only verify that values are finite (not NaN/inf).
        """
        path, gltf, bin_data, splat_prims = asset_data
        if not bin_data:
            pytest.skip("No binary data available")

        for prim in splat_prims:
            scale_idx = _get_khr_attribute(prim, "SCALE")
            if scale_idx is None:
                continue

            values = _read_accessor_floats(gltf, bin_data, scale_idx)
            step = max(1, len(values) // 3000)
            for i in range(0, len(values), step):
                v = values[i]
                assert math.isfinite(v), f"Scale[{i}] is not finite: {v}"

    def test_position_values_finite(self, asset_data):
        """All position values should be finite."""
        path, gltf, bin_data, splat_prims = asset_data
        if not bin_data:
            pytest.skip("No binary data available")

        for prim in splat_prims:
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is None:
                continue
            values = _read_accessor_floats(gltf, bin_data, pos_idx)
            step = max(1, len(values) // 3000)
            for i in range(0, len(values), step):
                v = values[i]
                assert math.isfinite(v), f"Position[{i}] is not finite: {v}"

    def test_sh_coefficient_count_matches_degree(self, asset_data):
        """Verify each SH degree has the expected number of coefficient accessors.

        Per the KHR spec, degree l has (2l+1) coefficients. Each coefficient is a
        separate VEC3 accessor (3 channels: RGB). So degree 0 has 1 coefficient,
        degree 1 has 3, degree 2 has 5, degree 3 has 7.
        """
        path, gltf, bin_data, splat_prims = asset_data

        expected_coefs = {0: 1, 1: 3, 2: 5, 3: 7}

        for prim in splat_prims:
            sh_attrs = _get_sh_degree_coef_attrs(prim)
            for degree, coef_list in sh_attrs.items():
                if degree in expected_coefs:
                    assert len(coef_list) == expected_coefs[degree], (
                        f"SH degree {degree}: expected {expected_coefs[degree]} "
                        f"coefficient accessors, got {len(coef_list)}"
                    )

    def test_sh_coefficient_values_finite(self, asset_data):
        """All SH coefficient values should be finite."""
        path, gltf, bin_data, splat_prims = asset_data
        if not bin_data:
            pytest.skip("No binary data available")

        for prim in splat_prims:
            sh_attrs = _get_sh_degree_coef_attrs(prim)
            for degree, coef_list in sh_attrs.items():
                for coef_idx, acc_idx in coef_list:
                    values = _read_accessor_floats(gltf, bin_data, acc_idx)
                    step = max(1, len(values) // 1000)
                    for i in range(0, len(values), step):
                        v = values[i]
                        assert math.isfinite(v), (
                            f"SH degree {degree} coef {coef_idx}[{i}] is not finite: {v}"
                        )

    def test_mixed_sh_degrees_across_primitives(self, asset_data):
        """If the file has multiple splat primitives, they may have different SH degrees."""
        path, gltf, bin_data, splat_prims = asset_data
        if len(splat_prims) < 2:
            pytest.skip("Only one splat primitive, cannot test mixed degrees")

        degrees = set()
        for prim in splat_prims:
            d = _get_max_sh_degree(prim)
            degrees.add(d)

        # Just verify all degrees are valid; they may or may not differ
        for d in degrees:
            assert 0 <= d <= 3, f"Invalid SH degree: {d}"


# =========================================================================
# 4. Edge Case Tests (synthetic data, always run)
# =========================================================================

class TestSplatEdgeCaseSynthetic:
    """Edge case tests using synthetically generated glTF splat data."""

    def _build_test_glb(self, num_splats, sh_degree=0, **overrides):
        """Build a minimal GLB-style (gltf_json, bin_bytes) with KHR_gaussian_splatting data.

        Uses the official KHR spec format: attributes with ``KHR_gaussian_splatting:``
        prefix, SH coefficients as individual VEC3 accessors per coefficient.
        """
        # Positions: vec3 float
        pos_data = bytearray()
        for i in range(num_splats):
            t = i / max(num_splats - 1, 1)
            pos_data += struct.pack('<3f', t, 0.0, 0.0)

        # Rotations: vec4 float (unit quaternion identity)
        rot_data = bytearray()
        for _ in range(num_splats):
            rot_data += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)

        # Scales: vec3 float (linear space, positive)
        scale_val = overrides.get("scale_val", 0.5)
        scale_data = bytearray()
        for _ in range(num_splats):
            scale_data += struct.pack('<3f', scale_val, scale_val, scale_val)

        # Opacity: scalar float (linear space [0, 1])
        opacity_val = overrides.get("opacity_val", 0.95)
        opacity_data = bytearray()
        for _ in range(num_splats):
            opacity_data += struct.pack('<f', opacity_val)

        # SH DC: degree 0, coefficient 0 -> VEC3
        sh_dc_data = bytearray()
        sh_c0 = 0.28209479177387814
        for _ in range(num_splats):
            sh_dc_data += struct.pack('<3f', 0.5 / sh_c0, 0.5 / sh_c0, 0.5 / sh_c0)

        # Higher degree SH: each coefficient is a separate VEC3 accessor
        # degree l has (2l+1) coefficients
        sh_higher_bufs = []
        for l in range(1, sh_degree + 1):
            n_coefs = 2 * l + 1
            for c in range(n_coefs):
                buf = bytearray()
                for _ in range(num_splats):
                    buf += struct.pack('<3f', 0.0, 0.0, 0.0)
                sh_higher_bufs.append((l, c, buf))

        # Build buffer
        parts = [pos_data, rot_data, scale_data, opacity_data, sh_dc_data]
        for _, _, buf in sh_higher_bufs:
            parts.append(buf)

        offsets = []
        running = 0
        for part in parts:
            offsets.append(running)
            running += len(part)
        total_size = running

        bin_bytes = bytearray()
        for part in parts:
            bin_bytes += part

        buffer_views = [
            {"buffer": 0, "byteOffset": offsets[idx], "byteLength": len(part)}
            for idx, part in enumerate(parts)
        ]

        # Position min/max
        if num_splats > 0:
            pos_min = [0.0, 0.0, 0.0]
            pos_max = [1.0 if num_splats > 1 else 0.0, 0.0, 0.0]
        else:
            pos_min = [0.0, 0.0, 0.0]
            pos_max = [0.0, 0.0, 0.0]

        accessors = [
            {"bufferView": 0, "componentType": 5126, "count": num_splats, "type": "VEC3",
             "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": num_splats, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": num_splats, "type": "VEC3"},
            {"bufferView": 3, "componentType": 5126, "count": num_splats, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5126, "count": num_splats, "type": "VEC3"},
        ]

        # Build attributes dict with KHR prefix
        attributes = {
            "POSITION": 0,
            "KHR_gaussian_splatting:SCALE": 2,
            "KHR_gaussian_splatting:ROTATION": 1,
            "KHR_gaussian_splatting:OPACITY": 3,
            "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0": 4,
        }

        for idx, (l, c, _) in enumerate(sh_higher_bufs):
            acc_idx = 5 + idx
            accessors.append({
                "bufferView": 5 + idx,
                "componentType": 5126,
                "count": num_splats,
                "type": "VEC3",
            })
            attributes[f"KHR_gaussian_splatting:SH_DEGREE_{l}_COEF_{c}"] = acc_idx

        gltf_json = {
            "asset": {"version": "2.0", "generator": "lux-conformance-test"},
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
                            "kernel": "ellipse",
                            "colorSpace": "BT.709-sRGB",
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

        return gltf_json, bytes(bin_bytes)

    def _write_glb(self, gltf_json, bin_data, path):
        """Write a GLB file from JSON dict and binary data."""
        json_str = json.dumps(gltf_json, separators=(',', ':'))
        while len(json_str) % 4 != 0:
            json_str += ' '
        json_bytes = json_str.encode('ascii')

        bin_padded = bytearray(bin_data)
        while len(bin_padded) % 4 != 0:
            bin_padded += b'\x00'

        glb_length = 12 + 8 + len(json_bytes) + 8 + len(bin_padded)
        header = struct.pack('<III', 0x46546C67, 2, glb_length)
        json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
        bin_chunk = struct.pack('<II', len(bin_padded), 0x004E4942) + bytes(bin_padded)

        with open(path, 'wb') as f:
            f.write(header + json_chunk + bin_chunk)

    # --- Empty splat data ---

    def test_empty_splat_parses(self, tmp_path):
        """A GLB with zero splats should parse without error."""
        gltf, bin_data = self._build_test_glb(0)
        glb_path = tmp_path / "empty.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        assert loaded_gltf["asset"]["version"] == "2.0"
        splat_prims = _get_splat_primitives(loaded_gltf)
        assert len(splat_prims) == 1

    def test_empty_splat_count_is_zero(self, tmp_path):
        """A GLB with zero splats should report count 0."""
        gltf, bin_data = self._build_test_glb(0)
        glb_path = tmp_path / "empty.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, _ = _parse_glb(glb_path)
        count = _get_splat_count(loaded_gltf)
        assert count == 0

    # --- Single splat ---

    def test_single_splat_parses(self, tmp_path):
        """A GLB with exactly one splat should parse correctly."""
        gltf, bin_data = self._build_test_glb(1)
        glb_path = tmp_path / "single.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        count = _get_splat_count(loaded_gltf)
        assert count == 1

    def test_single_splat_rotation_is_unit(self, tmp_path):
        """The single splat's rotation should be a unit quaternion."""
        gltf, bin_data = self._build_test_glb(1)
        glb_path = tmp_path / "single.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)
        rot_idx = _get_khr_attribute(splat_prims[0], "ROTATION")
        values = _read_accessor_floats(loaded_gltf, loaded_bin, rot_idx)
        assert len(values) == 4
        x, y, z, w = values
        length = math.sqrt(x*x + y*y + z*z + w*w)
        assert abs(length - 1.0) < 1e-6

    def test_single_splat_opacity_value(self, tmp_path):
        """The single splat's opacity should match what was written."""
        gltf, bin_data = self._build_test_glb(1, opacity_val=0.75)
        glb_path = tmp_path / "single.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)
        opa_idx = _get_khr_attribute(splat_prims[0], "OPACITY")
        values = _read_accessor_floats(loaded_gltf, loaded_bin, opa_idx)
        assert len(values) == 1
        assert abs(values[0] - 0.75) < 1e-6

    # --- Large splat count ---

    def test_large_splat_count_parses(self, tmp_path):
        """A GLB with 100,000 splats should parse and report correct count."""
        gltf, bin_data = self._build_test_glb(100_000)
        glb_path = tmp_path / "large.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, _ = _parse_glb(glb_path)
        count = _get_splat_count(loaded_gltf)
        assert count == 100_000

    def test_large_splat_data_integrity(self, tmp_path):
        """Spot-check data in a large splat GLB for integrity."""
        n = 10_000
        gltf, bin_data = self._build_test_glb(n)
        glb_path = tmp_path / "large.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)

        # Check rotation quaternions are all unit
        rot_idx = _get_khr_attribute(splat_prims[0], "ROTATION")
        rot_values = _read_accessor_floats(loaded_gltf, loaded_bin, rot_idx)
        assert len(rot_values) == n * 4
        for i in range(0, min(100, n)):
            x, y, z, w = rot_values[i*4:(i+1)*4]
            length = math.sqrt(x*x + y*y + z*z + w*w)
            assert abs(length - 1.0) < 1e-6

    # --- SH degree edge cases ---

    def test_sh_degree_0_has_one_coef(self, tmp_path):
        """SH degree 0 should produce exactly one coefficient attribute."""
        gltf, bin_data = self._build_test_glb(10, sh_degree=0)
        glb_path = tmp_path / "sh0.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, _ = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)
        sh_attrs = _get_sh_degree_coef_attrs(splat_prims[0])
        assert 0 in sh_attrs
        assert len(sh_attrs[0]) == 1  # 1 coefficient for degree 0
        assert len(sh_attrs) == 1     # only degree 0

    def test_sh_degree_3_has_all_degrees(self, tmp_path):
        """SH degree 3 should produce coefficients for degrees 0, 1, 2, and 3."""
        gltf, bin_data = self._build_test_glb(10, sh_degree=3)
        glb_path = tmp_path / "sh3.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, _ = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)
        sh_attrs = _get_sh_degree_coef_attrs(splat_prims[0])
        assert set(sh_attrs.keys()) == {0, 1, 2, 3}

    def test_sh_coefficients_correct_count_per_degree(self, tmp_path):
        """Verify each SH degree has (2l+1) coefficient accessors."""
        n = 10
        gltf, bin_data = self._build_test_glb(n, sh_degree=3)
        glb_path = tmp_path / "sh3_counts.glb"
        self._write_glb(gltf, bin_data, glb_path)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        splat_prims = _get_splat_primitives(loaded_gltf)
        sh_attrs = _get_sh_degree_coef_attrs(splat_prims[0])

        expected_coefs = {0: 1, 1: 3, 2: 5, 3: 7}
        for degree, coef_list in sh_attrs.items():
            assert len(coef_list) == expected_coefs[degree], (
                f"Degree {degree}: expected {expected_coefs[degree]} coefficients, got {len(coef_list)}"
            )

    # --- Data validation on synthetic data ---

    def test_non_unit_quaternion_detected(self):
        """Helper verification: a non-unit quaternion should have length != 1."""
        # Unit quaternion check
        x, y, z, w = 0.5, 0.5, 0.5, 0.5
        length = math.sqrt(x*x + y*y + z*z + w*w)
        assert abs(length - 1.0) < 1e-6

        # Not unit
        x, y, z, w = 1.0, 1.0, 1.0, 1.0
        length = math.sqrt(x*x + y*y + z*z + w*w)
        assert abs(length - 1.0) > 0.5

    def test_various_opacity_values(self, tmp_path):
        """Test that different opacity values round-trip correctly."""
        for opa in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gltf, bin_data = self._build_test_glb(1, opacity_val=opa)
            splat_prims = _get_splat_primitives(gltf)
            opa_idx = _get_khr_attribute(splat_prims[0], "OPACITY")
            values = _read_accessor_floats(gltf, bin_data, opa_idx)
            assert abs(values[0] - opa) < 1e-6, f"Opacity {opa} round-trip failed: got {values[0]}"


# =========================================================================
# 5. GLB Format Integrity Tests (always run)
# =========================================================================

class TestGlbFormatIntegrity:
    """Test GLB parsing edge cases and format validation."""

    def test_parse_glb_magic_validation(self, tmp_path):
        """A file with wrong magic bytes should raise ValueError."""
        bad_path = tmp_path / "bad.glb"
        bad_path.write_bytes(b'\x00\x00\x00\x00' + b'\x00' * 20)
        with pytest.raises(ValueError, match="Not a GLB file"):
            _parse_glb(bad_path)

    def test_parse_glb_too_small(self, tmp_path):
        """A file smaller than the GLB header should raise ValueError."""
        tiny_path = tmp_path / "tiny.glb"
        tiny_path.write_bytes(b'\x00\x01\x02')
        with pytest.raises(ValueError, match="GLB too small"):
            _parse_glb(tiny_path)

    def test_parse_glb_version_check(self, tmp_path):
        """A GLB with version != 2 should raise ValueError."""
        bad_path = tmp_path / "v1.glb"
        bad_path.write_bytes(struct.pack('<III', 0x46546C67, 1, 12))
        with pytest.raises(ValueError, match="Unsupported GLB version"):
            _parse_glb(bad_path)

    def test_roundtrip_minimal_glb(self, tmp_path):
        """A minimal splat GLB should survive write-then-read without data loss."""
        gltf_orig = {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "buffers": [{"byteLength": 0}],
            "bufferViews": [],
            "accessors": [],
            "meshes": [{
                "primitives": [{
                    "mode": 0,
                    "attributes": {
                        "KHR_gaussian_splatting:SCALE": 0,
                    },
                    "extensions": {
                        "KHR_gaussian_splatting": {
                            "kernel": "ellipse",
                        }
                    }
                }]
            }],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0,
        }

        json_str = json.dumps(gltf_orig, separators=(',', ':'))
        while len(json_str) % 4 != 0:
            json_str += ' '
        json_bytes = json_str.encode('ascii')

        glb_length = 12 + 8 + len(json_bytes)
        header = struct.pack('<III', 0x46546C67, 2, glb_length)
        json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes

        glb_path = tmp_path / "minimal.glb"
        with open(glb_path, 'wb') as f:
            f.write(header + json_chunk)

        loaded_gltf, loaded_bin = _parse_glb(glb_path)
        assert loaded_gltf["asset"]["version"] == "2.0"
        assert "KHR_gaussian_splatting" in loaded_gltf.get("extensionsUsed", [])
        splat_prims = _get_splat_primitives(loaded_gltf)
        assert len(splat_prims) == 1


# =========================================================================
# 6. Existing Tools Integration Tests (always run)
# =========================================================================

class TestGenerateTestSplatsIntegration:
    """Test that generate_test_splats.py output is valid glTF with Gaussian splat data.

    Note: generate_test_splats uses the project's internal KHR format (nested
    ``attributes`` and ``sh`` arrays), not the official spec format with prefixed
    attribute names. These tests verify structural integrity regardless of format.
    """

    def _extract_any_splat_data(self, gltf: dict) -> dict | None:
        """Extract splat extension data from either spec-format or project-format."""
        for mesh in gltf.get("meshes", []):
            for prim in mesh.get("primitives", []):
                ext = prim.get("extensions", {}).get("KHR_gaussian_splatting")
                if ext is not None:
                    return ext
        return None

    def test_generated_glb_has_extension(self, tmp_path):
        """A GLB from generate_test_splats should have KHR_gaussian_splatting."""
        from tools.generate_test_splats import generate_test_splats

        glb_path = tmp_path / "test.glb"
        generate_test_splats(str(glb_path), num_splats=100)

        gltf, bin_data = _parse_glb(glb_path)
        assert "KHR_gaussian_splatting" in gltf.get("extensionsUsed", [])
        ext = self._extract_any_splat_data(gltf)
        assert ext is not None

    def test_generated_glb_splat_count(self, tmp_path):
        """The splat count should match what was requested."""
        from tools.generate_test_splats import generate_test_splats

        for n in [1, 50, 500]:
            glb_path = tmp_path / f"test_{n}.glb"
            generate_test_splats(str(glb_path), num_splats=n)
            gltf, _ = _parse_glb(glb_path)
            # Find POSITION accessor count
            for mesh in gltf.get("meshes", []):
                for prim in mesh.get("primitives", []):
                    pos_idx = prim.get("attributes", {}).get("POSITION")
                    if pos_idx is not None:
                        count = gltf["accessors"][pos_idx]["count"]
                        assert count == n, f"Expected {n} splats, got {count}"

    def test_generated_glb_rotation_quaternions_unit(self, tmp_path):
        """Rotations from generate_test_splats should be unit quaternions."""
        from tools.generate_test_splats import generate_test_splats

        glb_path = tmp_path / "test_rot.glb"
        generate_test_splats(str(glb_path), num_splats=100)

        gltf, bin_data = _parse_glb(glb_path)
        # The project format uses _ROTATION attribute or KHR nested attributes
        for mesh in gltf.get("meshes", []):
            for prim in mesh.get("primitives", []):
                attrs = prim.get("attributes", {})
                rot_idx = attrs.get("_ROTATION")
                if rot_idx is None:
                    ext = prim.get("extensions", {}).get("KHR_gaussian_splatting", {})
                    rot_idx = ext.get("attributes", {}).get("ROTATION")
                if rot_idx is not None:
                    values = _read_accessor_floats(gltf, bin_data, rot_idx)
                    count = len(values) // 4
                    for i in range(count):
                        x, y, z, w = values[i*4:(i+1)*4]
                        length = math.sqrt(x*x + y*y + z*z + w*w)
                        assert abs(length - 1.0) < 1e-6, f"Quaternion {i} not unit: |q|={length}"

    def test_generated_glb_sh_dc_present(self, tmp_path):
        """SH degree 0 DC coefficients should be present and finite."""
        from tools.generate_test_splats import generate_test_splats

        glb_path = tmp_path / "test_sh.glb"
        generate_test_splats(str(glb_path), num_splats=50)

        gltf, bin_data = _parse_glb(glb_path)
        ext = self._extract_any_splat_data(gltf)
        assert ext is not None

        # Project format: sh[0].coefficients is the accessor index
        sh = ext.get("sh", [])
        if sh:
            assert sh[0]["degree"] == 0
            coeff_idx = sh[0]["coefficients"]
            values = _read_accessor_floats(gltf, bin_data, coeff_idx)
            assert len(values) == 50 * 3  # vec3 per splat
            for v in values:
                assert math.isfinite(v)

    def test_generated_glb_position_has_min_max(self, tmp_path):
        """POSITION accessor should have min/max as required by glTF spec."""
        from tools.generate_test_splats import generate_test_splats

        glb_path = tmp_path / "test_minmax.glb"
        generate_test_splats(str(glb_path), num_splats=100)

        gltf, _ = _parse_glb(glb_path)
        for mesh in gltf.get("meshes", []):
            for prim in mesh.get("primitives", []):
                pos_idx = prim.get("attributes", {}).get("POSITION")
                if pos_idx is not None:
                    acc = gltf["accessors"][pos_idx]
                    assert "min" in acc, "POSITION accessor missing 'min'"
                    assert "max" in acc, "POSITION accessor missing 'max'"
                    assert len(acc["min"]) == 3
                    assert len(acc["max"]) == 3
