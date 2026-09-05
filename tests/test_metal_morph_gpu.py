"""Verifies MetalSplatRenderer's GPU morph-apply compute kernel
(metal_splat_renderer.cpp's kSplatMorphMSL) against the independent
numpy-only Python reference evaluator (playground/dynamic_splat_gltf.py's
evaluate_at), which is itself validated against mobiledlss's own
torch-based evaluate_at() in tests/test_dynamic_splats.py.

This replaces the old CPU-side morph mirror in MetalSplatRenderer with a
real Metal compute kernel (SPECIFICATION.md 12.8, docs/lux-4d-spec.md) --
the --dump-splat-buffers debug flag (metal_main.cpp) writes the
GPU-kernel-updated splat_pos/rot/sh0 buffers to .npy right after
setMorphTime(), so this test never needs to touch the Metal API directly.

Skips cleanly if lux-playground-metal isn't built.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from playground.dynamic_splat_gltf import load_dynamic_splat_glb, evaluate_at  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
METAL_BIN = REPO_ROOT / "playground_cpp" / "build-metal" / "lux-playground-metal"
ASSET = REPO_ROOT / "tests" / "assets" / "dynamic_splats_synthetic.glb"


@pytest.fixture(scope="module")
def asset():
    if not ASSET.exists():
        pytest.skip(f"missing test fixture {ASSET}")
    return load_dynamic_splat_glb(ASSET)


@pytest.fixture
def metal_bin():
    if not METAL_BIN.exists():
        pytest.skip(f"{METAL_BIN} not built; see docs/rendering-engines.md")
    return str(METAL_BIN)


def _run_and_dump(metal_bin, time_seconds, tmp_path, name):
    prefix = tmp_path / name
    result = subprocess.run(
        [metal_bin, "--scene", str(ASSET), "--headless", "--width", "16", "--height", "16",
         "--time", repr(time_seconds), "--output", str(tmp_path / f"{name}.png"),
         "--dump-splat-buffers", str(prefix)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    pos = np.load(str(prefix) + "_pos.npy")[:, :3]
    rot = np.load(str(prefix) + "_rot.npy")
    sh0 = np.load(str(prefix) + "_sh0.npy")[:, :3]
    return pos, rot, sh0


class TestMetalMorphGpuKernel:
    def test_exact_keyframe(self, asset, metal_bin, tmp_path):
        t = float(asset.keyframe_times[2])
        pos, rot, sh0 = _run_and_dump(metal_bin, t, tmp_path, "kf2")
        ref = evaluate_at(asset, t)

        # Empirically: positions/sh0 come back EXACTLY bit-identical
        # (max diff == 0.0); rotations differ by at most 1 ULP at float32
        # (1.1920929e-07 == 2**-23), from the GPU/CPU renormalization
        # division -- about as "bit-exact" as two independent compute
        # backends (Metal shader vs numpy) can get.
        assert np.max(np.abs(pos - ref["positions"])) == 0.0
        assert np.max(np.abs(sh0 - ref["sh0"])) == 0.0
        assert np.max(np.abs(rot - ref["rotations"])) <= 1.1920929e-07 + 1e-9

    def test_mid_keyframe(self, asset, metal_bin, tmp_path):
        t0, t1 = float(asset.keyframe_times[1]), float(asset.keyframe_times[2])
        t = 0.5 * (t0 + t1)
        pos, rot, sh0 = _run_and_dump(metal_bin, t, tmp_path, "mid")
        ref = evaluate_at(asset, t)

        assert np.max(np.abs(pos - ref["positions"])) < 1e-5
        assert np.max(np.abs(sh0 - ref["sh0"])) < 1e-5
        assert np.max(np.abs(rot - ref["rotations"])) < 1e-5

    def test_base_frame_unchanged(self, asset, metal_bin, tmp_path):
        """At/before the base frame (t=0), attributes equal the base pose
        exactly (no morph contribution)."""
        pos, rot, sh0 = _run_and_dump(metal_bin, 0.0, tmp_path, "base")
        ref = evaluate_at(asset, 0.0)
        assert np.array_equal(pos, ref["positions"].astype(np.float32))
        assert np.max(np.abs(rot - ref["rotations"])) < 1e-6
        assert np.array_equal(sh0, ref["sh0"].astype(np.float32))
