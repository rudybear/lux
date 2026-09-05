"""End-to-end tests for the DLSS input-contract outputs (docs/lux-4d-spec.md
section 3) and the OpenCV camera bridge (section 4).

Unlike tests/test_splat_dlss_outputs.py (compiler-only, no GPU), these tests
build/compile examples/gaussian_splat_dlss.lux and actually run the
playground_cpp binaries headlessly, comparing rendered aux dumps
(<prefix>_color.png, _depth.npy, _mv.npy) against independently-computed
expected values.

Skips cleanly (does not fail) when a playground binary hasn't been built --
see docs/rendering-engines.md for build instructions:
    cd playground_cpp && cmake -B build && cmake --build build --target lux-playground
    cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal
"""

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playground.dynamic_splat_gltf import load_dynamic_splat_glb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VULKAN_BIN = REPO_ROOT / "playground_cpp" / "build" / "lux-playground"
METAL_BIN = REPO_ROOT / "playground_cpp" / "build-metal" / "lux-playground-metal"
PIPELINE_BASE = "examples/gaussian_splat_dlss"


def _compile_dlss_pipeline():
    """Ensure examples/gaussian_splat_dlss.{comp,vert,frag,morph.comp}.spv exist."""
    from luxc.builtins.types import clear_type_aliases
    clear_type_aliases()
    subprocess.run(
        [sys.executable, "-m", "luxc", "examples/gaussian_splat_dlss.lux"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )


def _backend_params():
    params = []
    if VULKAN_BIN.exists():
        params.append(pytest.param(str(VULKAN_BIN), id="vulkan"))
    if METAL_BIN.exists():
        params.append(pytest.param(str(METAL_BIN), id="metal"))
    if not params:
        params.append(pytest.param(None, id="no-binary", marks=pytest.mark.skip(
            reason="Neither lux-playground nor lux-playground-metal is built; "
                    "see docs/rendering-engines.md")))
    return params


@pytest.fixture(scope="module", autouse=True)
def _ensure_compiled():
    _compile_dlss_pipeline()


@pytest.fixture(params=_backend_params())
def binary(request):
    if request.param is None:
        pytest.skip("no playground binary built")
    return request.param


def _supports_dlss_flags(binary_path):
    """Metal support is added incrementally; skip cleanly if this binary's
    --help doesn't advertise the flags this test needs."""
    out = subprocess.run([binary_path, "--help"], capture_output=True, text=True).stdout
    return "--camera-json" in out and "--output-aux" in out


# ---------------------------------------------------------------------------
# Synthetic splat asset generation (KHR_gaussian_splatting ratified layout,
# same schema as tools/generate_test_splats.py, generalized to explicit
# per-splat positions/scale/opacity/color).
# ---------------------------------------------------------------------------

def _write_splats_glb(path, positions, scale=0.08, opacity=0.98, color=(1.0, 1.0, 1.0)):
    n = len(positions)
    pos_data = bytearray()
    rot_data = bytearray()
    scale_data = bytearray()
    opacity_data = bytearray()
    sh_data = bytearray()

    sh_c0 = 0.28209479177387814
    r, g, b = (c / sh_c0 for c in color)

    for (x, y, z) in positions:
        pos_data += struct.pack('<3f', x, y, z)
        rot_data += struct.pack('<4f', 0.0, 0.0, 0.0, 1.0)
        scale_data += struct.pack('<3f', scale, scale, scale)
        opacity_data += struct.pack('<f', opacity)
        sh_data += struct.pack('<3f', r, g, b)

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

    pos_min = [min(p[i] for p in positions) for i in range(3)]
    pos_max = [max(p[i] for p in positions) for i in range(3)]

    gltf_json = {
        "asset": {"version": "2.0", "generator": "lux-test-dlss-outputs"},
        "extensionsUsed": ["KHR_gaussian_splatting"],
        "extensionsRequired": ["KHR_gaussian_splatting"],
        "buffers": [{"byteLength": total_buffer}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_size},
            {"buffer": 0, "byteOffset": rot_offset, "byteLength": rot_size},
            {"buffer": 0, "byteOffset": scale_offset, "byteLength": scale_size},
            {"buffer": 0, "byteOffset": opa_offset, "byteLength": opa_size},
            {"buffer": 0, "byteOffset": sh_offset, "byteLength": sh_size},
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
    while len(buffer_bin) % 4 != 0:
        buffer_bin += b'\x00'

    glb_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_bin)
    header = struct.pack('<III', 0x46546C67, 2, glb_length)
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack('<II', len(buffer_bin), 0x004E4942) + buffer_bin

    Path(path).write_bytes(header + json_chunk + bin_chunk)


# ---------------------------------------------------------------------------
# Camera helpers -- simple axis-aligned OpenCV cameras: R = I, so
# p_cam = p_world - eye exactly (no rotation ambiguity), which keeps the
# expected-value math trivial and exact.
# ---------------------------------------------------------------------------

def _identity_camera_json(path, eye, width, height, fov_y_deg=50.0):
    """viewmat_cv with R=I, translation -eye (world point `eye + d*z_axis`
    ends up at camera-space depth d along OpenCV's +z-forward axis)."""
    fy = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2.0)
    K = [fy, 0.0, width / 2.0, 0.0, fy, height / 2.0, 0.0, 0.0, 1.0]
    viewmat_cv = [
        1.0, 0.0, 0.0, -eye[0],
        0.0, 1.0, 0.0, -eye[1],
        0.0, 0.0, 1.0, -eye[2],
        0.0, 0.0, 0.0, 1.0,
    ]
    json.dump({"viewmat_cv": viewmat_cv, "K": K, "width": width, "height": height},
              open(path, "w"))
    return K


def _run(binary_path, args, cwd=REPO_ROOT, timeout=60):
    result = subprocess.run([binary_path, *args], cwd=cwd, capture_output=True,
                             text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"playground exited {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def _alpha_weighted_centroid(png_path):
    """Sub-pixel (x, y) centroid of a rendered splat blob, weighted by the
    color image's alpha channel, using pixel-center (i+0.5, j+0.5)
    convention."""
    arr = np.array(Image.open(png_path).convert("RGBA"), dtype=np.float64)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    ys, xs = np.mgrid[0:h, 0:w]
    total = alpha.sum()
    assert total > 0, f"no visible splat in {png_path}"
    cx = float(((xs + 0.5) * alpha).sum() / total)
    cy = float(((ys + 0.5) * alpha).sum() / total)
    return cx, cy


# ===========================================================================
# Section 4: camera bridge
# ===========================================================================

class TestCameraBridge:
    def test_known_point_projects_to_expected_pixel(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --camera-json yet")

        width, height = 128, 128
        depth = 4.0
        eye = (0.0, 0.0, -depth)  # world point (dx,dy,0) -> camera-space (dx,dy,depth)
        dx, dy = 0.35, -0.2       # world-space offset of the splat from the optical axis

        glb_path = tmp_path / "one_splat.glb"
        _write_splats_glb(glb_path, [(dx, dy, 0.0)], scale=0.10, opacity=0.99)

        cam_path = tmp_path / "cam.json"
        K = _identity_camera_json(cam_path, eye, width, height)
        fx, cx = K[0], K[2]
        fy, cy = K[4], K[5]
        expected_u = fx * dx / depth + cx
        expected_v = fy * dy / depth + cy

        color_path = tmp_path / "out.png"
        _run(binary, ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam_path), "--output", str(color_path)])

        cx_actual, cy_actual = _alpha_weighted_centroid(color_path)
        assert abs(cx_actual - expected_u) < 0.5, (cx_actual, expected_u)
        assert abs(cy_actual - expected_v) < 0.5, (cy_actual, expected_v)


# ===========================================================================
# Section 3: expected depth
# ===========================================================================

class TestExpectedDepth:
    def test_single_opaque_splat_depth_equals_camera_z(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --output-aux yet")

        width, height = 96, 96
        depth = 5.5
        eye = (0.0, 0.0, -depth)

        # Un-premultiplying against the aux attachment's OWN full-precision
        # alpha channel (see splat_expander._build_fragment_body) recovers
        # the exact per-splat depth regardless of the fragment's alpha
        # value (for a single, non-overlapping splat), so any alpha
        # threshold above 0 is mathematically fine; a larger/more opaque
        # splat is used here just to get a comfortable number of covered
        # pixels to test at.
        glb_path = tmp_path / "one_splat.glb"
        _write_splats_glb(glb_path, [(0.0, 0.0, 0.0)], scale=0.25, opacity=0.999)

        cam_path = tmp_path / "cam.json"
        _identity_camera_json(cam_path, eye, width, height)

        aux_prefix = tmp_path / "aux"
        _run(binary, ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam_path),
                      "--output", str(tmp_path / "out.png"),
                      "--output-aux", str(aux_prefix)])

        depth_map = np.load(str(aux_prefix) + "_depth.npy")
        alpha = np.array(Image.open(str(aux_prefix) + "_color.png").convert("RGBA"))[:, :, 3] / 255.0
        mask = alpha > 0.1
        assert mask.sum() > 0, "splat not visible at high alpha"
        measured = depth_map[mask]
        assert np.max(np.abs(measured - depth)) < 1e-3, (measured.min(), measured.max(), depth)

    def test_overlapping_splats_depth_matches_over_compositing(self, binary, tmp_path):
        """Regression test for a real bug: out_depth was declared vec2 (x =
        depth*alpha, y = alpha), but Vulkan's fixed-function alpha blend
        factors (ONE_MINUS_SRC_ALPHA) read "source alpha" from the 4th
        component of the fragment shader's output for that attachment. A
        vec2 output has no 4th component; on MoltenVK/Apple GPUs this was
        empirically observed to blend as if src alpha were 0 (dst_new = src
        + dst_old, an undecayed running sum) instead of the correct
        back-to-front "over" composite (dst_new = src + dst_old*(1-alpha)).
        For two co-axial overlapping splats at different depths this gives
        a visibly different, order-INdependent answer (a simple opacity-
        weighted average) instead of the correct occlusion-weighted one.
        `out_motion` (already vec4, alpha genuinely at .w) never had this
        bug -- see out_depth's vec4 fix in
        luxc/expansion/splat_expander.py.
        """
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --output-aux yet")

        width, height = 64, 64
        # Two splats directly along the optical axis (same x, y) so they
        # fully overlap at the center pixel; default sort is camera_distance
        # (back-to-front), so the far one (z=6) is drawn before the near one
        # (z=4).
        alpha_each = 0.5
        z_far, z_near = 6.0, 4.0
        glb_path = tmp_path / "two_splats.glb"
        _write_splats_glb(glb_path, [(0.0, 0.0, z_far), (0.0, 0.0, z_near)],
                           scale=0.6, opacity=alpha_each)

        cam_path = tmp_path / "cam.json"
        _identity_camera_json(cam_path, (0.0, 0.0, 0.0), width, height)

        aux_prefix = tmp_path / "aux"
        _run(binary, ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam_path),
                      "--output", str(tmp_path / "out.png"),
                      "--output-aux", str(aux_prefix)])

        depth_map = np.load(str(aux_prefix) + "_depth.npy")
        measured = float(depth_map[height // 2, width // 2])

        # Correct back-to-front "over" composite (far drawn first, near on
        # top): the near splat's contribution isn't attenuated by the far
        # one, but the far splat's IS attenuated by (1 - alpha_near).
        premul = alpha_each * z_far
        cum_alpha = alpha_each
        premul = alpha_each * z_near + premul * (1 - alpha_each)
        cum_alpha = alpha_each + cum_alpha * (1 - alpha_each)
        correct = premul / cum_alpha

        # The buggy (undecayed-sum) behavior degenerates to a simple,
        # order-independent opacity-weighted average -- provably different
        # from `correct` whenever the two alphas are equal and nonzero.
        buggy = (alpha_each * z_far + alpha_each * z_near) / (alpha_each + alpha_each)
        assert abs(correct - buggy) > 0.1, "test's own two values must differ to be meaningful"

        assert abs(measured - correct) < 0.05, (
            f"measured depth {measured} should match the correct back-to-front "
            f"'over' composite {correct}, not the buggy undecayed-sum value {buggy}"
        )


# ===========================================================================
# Section 3: motion vectors
# ===========================================================================

class TestMotionVectors:
    def test_first_frame_mv_is_zero(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --output-aux yet")

        width, height = 64, 64
        glb_path = tmp_path / "scene.glb"
        _write_splats_glb(glb_path, [(0.0, 0.0, 0.0), (0.3, 0.1, -0.2)], scale=0.15, opacity=0.9)
        cam_path = tmp_path / "cam.json"
        _identity_camera_json(cam_path, (0.0, 0.0, -4.0), width, height)

        aux_prefix = tmp_path / "aux"
        _run(binary, ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam_path),
                      "--output", str(tmp_path / "out.png"),
                      "--output-aux", str(aux_prefix)])

        mv = np.load(str(aux_prefix) + "_mv.npy")
        assert np.max(np.abs(mv)) < 1e-2  # float rounding only, not real motion

    def test_mv_matches_cpu_reprojection_of_depth(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --camera-json-prev yet")

        width, height = 128, 128
        # A grid of well-separated, near-opaque splats: with heavy overlap,
        # "expected depth" is a genuine alpha-weighted average across
        # multiple different depths (matches gsplat's "ED" mode by design),
        # which this per-pixel single-depth reprojection can't model -- so
        # keep splats far enough apart in screen space that at most one
        # dominates each covered pixel.
        rng = np.random.default_rng(0)
        grid = np.linspace(-0.5, 0.5, 4)
        positions = [(float(gx), float(gy), float(rng.uniform(-0.05, 0.05)))
                     for gx in grid for gy in grid]
        glb_path = tmp_path / "scene.glb"
        _write_splats_glb(glb_path, positions, scale=0.10, opacity=0.999)

        eye0 = (0.0, 0.0, -4.0)
        eye1 = (0.15, -0.05, -4.0)  # small lateral truck between "prev" and "curr"
        cam0 = tmp_path / "cam0.json"
        cam1 = tmp_path / "cam1.json"
        K = _identity_camera_json(cam0, eye0, width, height)
        _identity_camera_json(cam1, eye1, width, height)
        fx, cx = K[0], K[2]
        fy, cy = K[4], K[5]

        aux_prefix = tmp_path / "aux"
        # "current" = cam1, "previous" = cam0 (camera moved from eye0 to eye1).
        _run(binary, ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam1), "--camera-json-prev", str(cam0),
                      "--output", str(tmp_path / "out.png"),
                      "--output-aux", str(aux_prefix)])

        depth = np.load(str(aux_prefix) + "_depth.npy")
        mv = np.load(str(aux_prefix) + "_mv.npy")
        alpha = np.array(Image.open(str(aux_prefix) + "_color.png").convert("RGBA"))[:, :, 3] / 255.0

        ys, xs = np.mgrid[0:height, 0:width]
        u = xs + 0.5
        v = ys + 0.5

        # Unproject each pixel with the CURRENT camera (R=I, eye=eye1):
        # world = eye1 + depth * K^-1 [u,v,1].
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth
        world_x = x_cam + eye1[0]
        world_y = y_cam + eye1[1]
        world_z = z_cam + eye1[2]

        # Reproject into the PREVIOUS camera (R=I, eye=eye0). Background
        # pixels have depth==0, giving pz_cam==0 (divide-by-zero -> inf) --
        # harmless since only the alpha>0.9-masked entries are asserted on.
        px_cam = world_x - eye0[0]
        py_cam = world_y - eye0[1]
        pz_cam = world_z - eye0[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            prev_u = fx * px_cam / pz_cam + cx
            prev_v = fy * py_cam / pz_cam + cy

        mask = alpha > 0.9
        assert mask.sum() > 50, f"too few high-alpha pixels ({mask.sum()})"

        expected_prev_u = u[mask] - mv[:, :, 0][mask]
        expected_prev_v = v[mask] - mv[:, :, 1][mask]
        err_u = np.abs(expected_prev_u - prev_u[mask])
        err_v = np.abs(expected_prev_v - prev_v[mask])
        assert np.median(err_u) < 0.05, np.median(err_u)
        assert np.median(err_v) < 0.05, np.median(err_v)


# ===========================================================================
# Section 3: sub-pixel jitter
# ===========================================================================

class TestJitter:
    def _render_centroid(self, binary, tmp_path, jitter, name):
        width, height = 96, 96
        glb_path = tmp_path / "one_splat.glb"
        _write_splats_glb(glb_path, [(0.0, 0.0, 0.0)], scale=0.10, opacity=0.99)
        cam_path = tmp_path / "cam.json"
        _identity_camera_json(cam_path, (0.03, 0.02, -4.0), width, height)
        out_path = tmp_path / f"{name}.png"
        args = ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                "--headless", "--width", str(width), "--height", str(height),
                "--camera-json", str(cam_path), "--output", str(out_path)]
        if jitter is not None:
            args += ["--jitter", str(jitter[0]), str(jitter[1])]
        _run(binary, args)
        return _alpha_weighted_centroid(out_path)

    def test_jitter_x_shifts_right(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --camera-json yet")
        cx0, cy0 = self._render_centroid(binary, tmp_path, None, "unjittered")
        cx1, cy1 = self._render_centroid(binary, tmp_path, (0.5, 0.0), "jit_x")
        assert abs((cx1 - cx0) - 0.5) < 0.1, (cx0, cx1)
        assert abs(cy1 - cy0) < 0.1, (cy0, cy1)

    def test_jitter_y_shifts_down(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --camera-json yet")
        cx0, cy0 = self._render_centroid(binary, tmp_path, None, "unjittered")
        cx1, cy1 = self._render_centroid(binary, tmp_path, (0.0, 0.5), "jit_y")
        assert abs((cy1 - cy0) - 0.5) < 0.1, (cy0, cy1)
        assert abs(cx1 - cx0) < 0.1, (cx0, cx1)

    def test_jitter_does_not_affect_motion_vectors(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --output-aux yet")

        width, height = 96, 96
        glb_path = tmp_path / "scene.glb"
        # A single, isolated, near-opaque splat: its mv is one constant
        # value across the whole splat (un-premultiplied exactly regardless
        # of per-fragment alpha, see TestExpectedDepth above), so this
        # isolates "does jitter leak into mv" from the separate, expected
        # effect of overlapping splats' alpha-weighted blend shifting
        # slightly when jitter moves sub-pixel sample positions.
        _write_splats_glb(glb_path, [(0.0, 0.0, 0.0)], scale=0.20, opacity=0.999)
        eye0 = (0.0, 0.0, -4.0)
        eye1 = (0.1, 0.0, -4.0)
        cam0 = tmp_path / "cam0.json"
        cam1 = tmp_path / "cam1.json"
        _identity_camera_json(cam0, eye0, width, height)
        _identity_camera_json(cam1, eye1, width, height)

        def render(name, jitter):
            aux_prefix = tmp_path / name
            args = ["--scene", str(glb_path), "--pipeline", PIPELINE_BASE,
                    "--headless", "--width", str(width), "--height", str(height),
                    "--camera-json", str(cam1), "--camera-json-prev", str(cam0),
                    "--output", str(tmp_path / f"{name}.png"),
                    "--output-aux", str(aux_prefix)]
            if jitter is not None:
                args += ["--jitter", str(jitter[0]), str(jitter[1])]
            _run(binary, args)
            mv = np.load(str(aux_prefix) + "_mv.npy")
            alpha = np.array(Image.open(str(aux_prefix) + "_color.png").convert("RGBA"))[:, :, 3] / 255.0
            return mv, alpha

        mv_off, alpha_off = render("no_jitter", None)
        mv_on, alpha_on = render("with_jitter", (1.3, -0.7))
        # Only compare pixels solidly covered in BOTH renders -- jitter
        # shifts which pixels straddle a splat's edge by up to ~1px, and
        # those edge pixels legitimately gain/lose coverage (and therefore
        # go from mv==0 to mv!=0 or vice versa) independent of whether
        # jitter leaks into the MV value itself, which is what this test
        # checks.
        mask = (alpha_off > 0.5) & (alpha_on > 0.5)
        assert mask.sum() > 20, f"too few stably-covered pixels ({mask.sum()})"
        assert np.max(np.abs(mv_off[mask] - mv_on[mask])) < 1e-3


# ===========================================================================
# Section 3: --time-prev/--frame-prev (real actor motion in a single-shot
# headless MV render, not just camera-only)
# ===========================================================================

class TestPreviousMorphTime:
    """Without --time-prev/--frame-prev, splat_prev_pos defaults to the
    current frame's own (post-morph) position, so a single headless render's
    mv only ever reflects camera motion -- fine for validating the
    projection math (see TestMotionVectors above), but it means a *moving*
    actor's own motion never shows up unless you have two real frames to
    diff. --time-prev/--frame-prev fixes that by evaluating the morph at a
    second, different time into splat_prev_pos."""

    ASSET = REPO_ROOT / "tests" / "assets" / "dynamic_splats_synthetic.glb"

    def test_camera_only_mv_is_zero_without_time_prev(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --output-aux yet")
        width, height = 128, 128
        cam = tmp_path / "cam.json"
        _identity_camera_json(cam, (0.0, 0.0, -4.0), width, height)
        aux_prefix = tmp_path / "aux"
        _run(binary, ["--scene", str(self.ASSET), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam), "--time", "0.13333334028720856",
                      "--output", str(tmp_path / "out.png"), "--output-aux", str(aux_prefix)])
        mv = np.load(str(aux_prefix) + "_mv.npy")
        # Static camera, no --time-prev: prev positions == current positions
        # by construction, so mv must be ~0 everywhere (not just where
        # nothing moves -- literally the whole frame, camera-only).
        assert np.max(np.abs(mv)) < 1e-3

    def test_time_prev_reveals_actor_motion_only_on_moving_gaussians(self, binary, tmp_path):
        if not _supports_dlss_flags(binary):
            pytest.skip(f"{binary} doesn't support --time-prev yet")
        asset = load_dynamic_splat_glb(self.ASSET)
        moving = set()
        for t in asset.targets:
            moving.update(int(i) for i in t.indices)
        static_idx = [i for i in range(asset.num_splats) if i not in moving]
        moving_idx = sorted(moving)
        assert len(static_idx) > 10 and len(moving_idx) > 10

        width, height = 128, 128
        eye = (0.0, 0.0, -4.0)
        cam = tmp_path / "cam.json"
        K = _identity_camera_json(cam, eye, width, height)
        fx, cx, fy, cy = K[0], K[2], K[4], K[5]

        def project(p):
            x, y, z = p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]
            return fx * x / z + cx, fy * y / z + cy

        aux_prefix = tmp_path / "aux"
        # Static camera (same cam for current+prev, no --camera-json-prev):
        # isolates pure actor motion between two keyframes.
        _run(binary, ["--scene", str(self.ASSET), "--pipeline", PIPELINE_BASE,
                      "--headless", "--width", str(width), "--height", str(height),
                      "--camera-json", str(cam),
                      "--time", "0.13333334028720856", "--time-prev", "0.06666667014360428",
                      "--output", str(tmp_path / "out.png"), "--output-aux", str(aux_prefix)])
        mv = np.load(str(aux_prefix) + "_mv.npy")

        moving_px = [project(asset.base_pos[i]) for i in moving_idx]

        def min_dist_to_moving(u, v):
            return min(((u - mu) ** 2 + (v - mv_) ** 2) ** 0.5 for mu, mv_ in moving_px)

        def mv_window_max(u, v, radius=2):
            px, py = int(round(u - 0.5)), int(round(v - 0.5))
            x0, x1 = max(0, px - radius), min(width, px + radius + 1)
            y0, y1 = max(0, py - radius), min(height, py + radius + 1)
            if x1 <= x0 or y1 <= y0:
                return None
            window = mv[y0:y1, x0:x1, :]
            return float(np.max(np.linalg.norm(window, axis=-1))) if window.size else None

        # Only sample static gaussians whose projection is well clear of
        # every moving gaussian's projection (>= 10px, generous given the
        # splats' own screen-space radius) -- with 400 splats packed into a
        # 128x128 frame, many static/moving pairs project close enough to
        # bleed alpha-blended mv contributions into each other's immediate
        # neighborhood, which isn't the "does mv leak onto UNRELATED static
        # gaussians" question this test asks.
        isolated_static = [i for i in static_idx
                            if min_dist_to_moving(*project(asset.base_pos[i])) >= 10.0]
        assert len(isolated_static) >= 5, "too few isolated static gaussians to test against"

        static_max = max(v for i in isolated_static
                          if (v := mv_window_max(*project(asset.base_pos[i]))) is not None)
        moving_max = max(v for i in moving_idx
                          if (v := mv_window_max(*project(asset.base_pos[i]))) is not None)
        assert static_max < 0.5, f"isolated static gaussians should show ~0 mv, got {static_max}"
        assert moving_max > 1.0, f"moving gaussians should show real mv, got {moving_max}"
        assert moving_max > 10 * max(static_max, 1e-6)
