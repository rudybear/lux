"""Tests for dynamic (morph-target-animated) Gaussian splat glTF loading.

Covers: sparse morph-target parsing (including sparse-without-bufferView
accessors, whose dense base is implicitly all-zero), keyframe evaluation
reproducing the dense frame, mid-keyframe evaluation matching a lerp, and
that static (non-animated) splat assets are correctly *not* detected as
dynamic.

See docs/lux-4d-spec.md sections 1 and 5, and the reference writer/reader at
mobiledlss/gltf/dyn_splat_gltf.py (this test suite does not import mobiledlss
or torch -- playground/dynamic_splat_gltf.py is a self-contained numpy-only
reimplementation of the reading + evaluation semantics).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.dynamic_splat_gltf import (  # noqa: E402
    load_dynamic_splat_glb,
    is_dynamic_splat_glb,
    evaluate_at,
    evaluate_weights_at,
)

ASSETS = Path(__file__).resolve().parent / "assets"
DYNAMIC_GLB = ASSETS / "dynamic_splats_synthetic.glb"


@pytest.fixture(scope="module")
def asset():
    if not DYNAMIC_GLB.exists():
        pytest.skip(f"missing test fixture {DYNAMIC_GLB}; regenerate via docs/lux-4d-spec.md's writer call")
    return load_dynamic_splat_glb(DYNAMIC_GLB)


# ---------------------------------------------------------------------------
# 1. Sparse-target parsing
# ---------------------------------------------------------------------------

class TestSparseTargetParsing:
    def test_detects_dynamic(self):
        assert is_dynamic_splat_glb(DYNAMIC_GLB) is True

    def test_static_assets_not_dynamic(self):
        for name in ["test_splats.glb", "luigi.glb", "debug_splats.glb"]:
            p = ASSETS / name
            if not p.exists():
                continue
            assert is_dynamic_splat_glb(p) is False, f"{name} should not be detected as dynamic"

    def test_base_shapes(self, asset):
        n = asset.num_splats
        assert n == 400
        assert asset.base_pos.shape == (n, 3)
        assert asset.base_rot.shape == (n, 4)
        assert asset.base_sh0.shape == (n, 3)
        assert asset.base_scale.shape == (n, 3)
        assert asset.base_opacity.shape == (n,)

    def test_num_targets_and_keyframes(self, asset):
        # stride=2 over 9 frames -> keyframes [0,2,4,6,8] -> 4 targets
        assert len(asset.targets) == 4
        assert asset.keyframe_times.shape[0] == 5
        assert asset.keyframe_weights.shape == (5, 4)

    def test_targets_are_sparse_and_no_bufferview_base(self):
        """Directly inspect the raw glTF JSON: target accessors must be
        sparse, and (since the base is implicitly zero) must have no
        bufferView of their own."""
        import struct
        data = DYNAMIC_GLB.read_bytes()
        jlen, _ = struct.unpack_from("<II", data, 12)
        gltf = json.loads(data[20:20 + jlen])
        prim = gltf["meshes"][0]["primitives"][0]
        assert "targets" in prim and len(prim["targets"]) == 4
        for tgt in prim["targets"]:
            for semantic, acc_idx in tgt.items():
                acc = gltf["accessors"][acc_idx]
                assert "sparse" in acc, f"{semantic} target accessor must be sparse"
                assert "bufferView" not in acc, f"{semantic} target accessor must have no dense bufferView"
                assert acc["count"] == 400

    def test_moving_gaussians_count(self, asset):
        # _synthetic_scene(..., moving=60): only ~60 gaussians move per target
        # (delta_eps thresholding may drop a few that don't clear the threshold
        # between adjacent sampled frames -- allow a small margin).
        for tgt in asset.targets:
            assert 0 < len(tgt.indices) <= 60

    def test_weights_are_one_hot(self, asset):
        w = asset.keyframe_weights
        assert np.all(w[0] == 0.0)
        for i in range(1, w.shape[0]):
            assert np.count_nonzero(w[i]) == 1
            assert w[i, i - 1] == 1.0

    def test_extras_present(self, asset):
        assert asset.extras.get("version") == 1
        assert asset.extras.get("keyframes") == [0, 2, 4, 6, 8]
        assert asset.extras.get("fps") == 30.0
        assert asset.fps == 30.0
        assert asset.extras_keyframes == [0, 2, 4, 6, 8]


# ---------------------------------------------------------------------------
# 2. Keyframe / interpolation semantics
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_at_base_time_is_base(self, asset):
        out = evaluate_at(asset, asset.keyframe_times[0])
        assert np.array_equal(out["positions"], asset.base_pos)
        expected_rot = asset.base_rot / np.linalg.norm(asset.base_rot, axis=1, keepdims=True)
        assert np.allclose(out["rotations"], expected_rot)

    def test_evaluate_at_keyframe_reproduces_dense_frame(self, asset):
        """At an exact keyframe time, evaluate_at must exactly reproduce the
        dense (fully-applied) frame: base + 1.0*delta_k, 0*others."""
        for k, tgt in enumerate(asset.targets):
            t = asset.keyframe_times[k + 1]
            out = evaluate_at(asset, t)
            expected_pos = asset.base_pos.copy()
            expected_pos[tgt.indices] += tgt.dpos
            expected_rot = asset.base_rot.copy()
            expected_rot[tgt.indices] += tgt.drot
            expected_rot = expected_rot / np.linalg.norm(expected_rot, axis=1, keepdims=True)
            expected_sh0 = asset.base_sh0.copy()
            expected_sh0[tgt.indices] += tgt.dsh0

            assert np.array_equal(out["positions"], expected_pos), f"keyframe {k+1} position mismatch"
            assert np.allclose(out["rotations"], expected_rot, atol=0, rtol=0) or np.array_equal(out["rotations"], expected_rot)
            assert np.array_equal(out["sh0"], expected_sh0), f"keyframe {k+1} sh0 mismatch"

    def test_mid_keyframe_equals_lerp(self, asset):
        """Evaluating halfway between two keyframes must equal lerping the
        two (already-renormalized) endpoint frames -- matches the reference
        semantics in mobiledlss/gltf/dyn_splat_gltf.py's evaluate_at()."""
        t0, t1 = asset.keyframe_times[1], asset.keyframe_times[2]
        tm = 0.5 * (t0 + t1)
        f0 = evaluate_at(asset, t0)
        f1 = evaluate_at(asset, t1)
        fm = evaluate_at(asset, tm)

        expected_pos = 0.5 * f0["positions"] + 0.5 * f1["positions"]
        expected_rot_raw = 0.5 * f0["rotations"] + 0.5 * f1["rotations"]
        expected_rot = expected_rot_raw / np.linalg.norm(expected_rot_raw, axis=1, keepdims=True)
        expected_sh0 = 0.5 * f0["sh0"] + 0.5 * f1["sh0"]

        assert np.allclose(fm["positions"], expected_pos, atol=1e-6)
        assert np.allclose(fm["rotations"], expected_rot, atol=1e-6)
        assert np.allclose(fm["sh0"], expected_sh0, atol=1e-6)

    def test_weights_lerp_between_adjacent_keyframes_only(self, asset):
        t0, t1 = asset.keyframe_times[1], asset.keyframe_times[2]
        tm = 0.5 * (t0 + t1)
        w = evaluate_weights_at(asset, tm)
        # Only target[0] (low) and target[1] (high) may be nonzero.
        assert np.allclose(w[0], 0.5)
        assert np.allclose(w[1], 0.5)
        assert np.allclose(w[2:], 0.0)

    def test_clamped_outside_range(self, asset):
        before = evaluate_at(asset, -1.0)
        at_zero = evaluate_at(asset, float(asset.keyframe_times[0]))
        assert np.array_equal(before["positions"], at_zero["positions"])

        after = evaluate_at(asset, 1e6)
        at_last = evaluate_at(asset, float(asset.keyframe_times[-1]))
        assert np.array_equal(after["positions"], at_last["positions"])

    def test_frame_to_time_uses_fps(self, asset):
        assert asset.frame_to_time(0) == pytest.approx(0.0)
        assert asset.frame_to_time(4) == pytest.approx(4.0 / 30.0)


# ---------------------------------------------------------------------------
# 3. Cross-check against the mobiledlss reference implementation, when
#    available (skipped otherwise so this test file is portable).
# ---------------------------------------------------------------------------

MOBILEDLSS_PY = Path.home() / "sources" / "mobiledlss" / ".venv" / "bin" / "python"
MOBILEDLSS_ROOT = Path.home() / "sources" / "mobiledlss"


@pytest.mark.skipif(not MOBILEDLSS_PY.exists(), reason="mobiledlss venv not available on this machine")
def test_cross_check_against_mobiledlss_reference(asset):
    """Independent cross-check: ask mobiledlss's own evaluate_at() (torch-based
    reference) for the dense frame at keyframe 2 and compare positions."""
    script = f"""
import sys, json
sys.path.insert(0, {str(MOBILEDLSS_ROOT / "tests")!r})
sys.path.insert(0, {str(MOBILEDLSS_ROOT)!r})
from mobiledlss.gltf.dyn_splat_gltf import read_dynamic_splat_glb, evaluate_at
scene = read_dynamic_splat_glb({str(DYNAMIC_GLB)!r})
out = evaluate_at(scene, scene.keyframes, scene.keyframes[2])
print(json.dumps(out["means"].numpy().tolist()))
"""
    result = subprocess.run([str(MOBILEDLSS_PY), "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    ref_positions = np.array(json.loads(result.stdout))

    ours = evaluate_at(asset, float(asset.keyframe_times[2]))
    assert np.allclose(ours["positions"], ref_positions, atol=1e-5)
