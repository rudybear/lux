"""Regression test for the SplatDynamics::hasExtrasFps bug: `--frame N`
must map to time N/fps whenever mesh.extras.MOBILEDLSS_dynamic_splats.fps
is present, matching `--time <N/fps>` bit-for-bit -- not clamp to the last
keyframe for N >= keyframe_count.

Root cause (playground_cpp/src/gltf_loader.cpp): the vendored cgltf.h only
calls cgltf_parse_json_extras() for PRIMITIVE-level extras, never for the
owning cgltf_mesh object itself, so `mesh.extras.data` was always null even
though the ratified MOBILEDLSS_dynamic_splats writer puts extras on the
*mesh*. Fixed by jsonMeshExtras(), a manual scan of cgltf's retained raw
JSON buffer (data->json/json_size).

tests/assets/dynamic_splats_synthetic.glb has 5 keyframes (fps=30,
extras.keyframes stride 2) -- frame 4 is within range either way, so this
also exercises frame 20 (well past the keyframe count) to prove the fix
isn't accidentally passing via the old clamp-to-last-keyframe fallback path.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
VULKAN_BIN = REPO_ROOT / "playground_cpp" / "build" / "lux-playground"
METAL_BIN = REPO_ROOT / "playground_cpp" / "build-metal" / "lux-playground-metal"
ASSET = REPO_ROOT / "tests" / "assets" / "dynamic_splats_synthetic.glb"
FPS = 30.0


def _backend_params():
    params = []
    if VULKAN_BIN.exists():
        params.append(pytest.param((str(VULKAN_BIN), "examples/gaussian_splat_dynamic"), id="vulkan"))
    if METAL_BIN.exists():
        params.append(pytest.param((str(METAL_BIN), None), id="metal"))
    if not params:
        params.append(pytest.param((None, None), id="no-binary",
                                    marks=pytest.mark.skip(reason="no playground binary built")))
    return params


@pytest.fixture(params=_backend_params())
def binary_and_pipeline(request):
    if request.param[0] is None:
        pytest.skip("no playground binary built")
    return request.param


def _render(binary, pipeline, extra_args, out_path):
    args = [binary, "--scene", str(ASSET), "--headless",
            "--width", "32", "--height", "32", "--output", str(out_path)]
    if pipeline:
        args += ["--pipeline", pipeline]
    args += extra_args
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result.stdout


@pytest.mark.parametrize("frame", [4, 20])
def test_frame_maps_to_time_via_fps(binary_and_pipeline, frame, tmp_path):
    binary, pipeline = binary_and_pipeline
    expected_time = frame / FPS

    out_frame = tmp_path / f"frame{frame}.png"
    stdout_frame = _render(binary, pipeline, ["--frame", str(frame)], out_frame)
    assert f"t={expected_time:g}" in stdout_frame or f"t={expected_time}" in stdout_frame, stdout_frame

    out_time = tmp_path / f"time{frame}.png"
    _render(binary, pipeline, ["--time", repr(expected_time)], out_time)

    a = np.array(Image.open(out_frame).convert("RGBA"))
    b = np.array(Image.open(out_time).convert("RGBA"))
    assert np.array_equal(a, b), f"--frame {frame} did not match --time {expected_time!r}"
