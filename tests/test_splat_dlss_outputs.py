"""Tests for the splat `motion_vectors` / `expected_depth` compiler outputs.

See docs/lux-4d-spec.md section 3 ("DLSS input contract outputs") and
SPECIFICATION.md 12.8. These are luxc-side (compiler) tests: they compile
small `splat { ... }` sources with the new boolean members on/off and
inspect the emitted SPIR-V reflection JSON, without needing a GPU.
"""

import json
from pathlib import Path

import pytest

from luxc.builtins.types import clear_type_aliases
from luxc.compiler import compile_source


def _compile(tmp_path: Path, stem: str, *, motion_vectors: bool, expected_depth: bool,
             motion_keyframes: bool = False) -> dict:
    """Compile a minimal gaussian_splat pipeline and return {stage: reflection_json}."""
    members = [
        "sh_degree: 0",
        "kernel: ellipse",
        "color_space: linear",
        "sort: camera_distance",
        "alpha_cutoff: 0.004",
    ]
    if motion_keyframes:
        members.append("motion: keyframes")
    if motion_vectors:
        members.append("motion_vectors: true")
    if expected_depth:
        members.append("expected_depth: true")
    source = (
        "splat TestCloud {\n    " + ",\n    ".join(members) + ",\n}\n\n"
        "pipeline TestViewer {\n    mode: gaussian_splat,\n    splat: TestCloud,\n}\n"
    )
    clear_type_aliases()
    compile_source(source, stem, tmp_path, emit_reflection=True, validate=True)
    stages = {}
    for suffix in ("comp", "vert", "frag", "morph.comp"):
        p = tmp_path / f"{stem}.{suffix}.json"
        if p.exists():
            stages[suffix] = json.loads(p.read_text())
    return stages


class TestFlagsOff:
    """With both flags false, output must be identical to the pre-existing shape."""

    def test_no_extra_buffers_or_outputs(self, tmp_path):
        stages = self._compile_off(tmp_path)
        comp = stages["comp"]
        buf_names = {b["name"] for b in comp["descriptor_sets"]["0"]}
        assert "splat_prev_pos" not in buf_names
        assert "projected_mv" not in buf_names
        assert "projected_depth" not in buf_names
        assert [f["name"] for f in comp["push_constants"][0]["fields"]] == [
            "view_matrix", "proj_matrix", "cam_pos", "screen_size",
            "total_splats", "focal_x", "focal_y", "sh_degree",
        ]
        assert comp["push_constants"][0]["size"] == 176

        frag = stages["frag"]
        assert [o["name"] for o in frag["outputs"]] == ["out_color"]
        assert [i["name"] for i in frag["inputs"]] == [
            "frag_conic", "frag_color", "frag_center", "frag_offset"]

        vert = stages["vert"]
        assert [o["name"] for o in vert["outputs"]] == [
            "frag_conic", "frag_color", "frag_center", "frag_offset"]

    def _compile_off(self, tmp_path):
        return _compile(tmp_path, "off_test", motion_vectors=False, expected_depth=False)


class TestExpectedDepthOnly:
    def test_depth_buffer_and_output(self, tmp_path):
        stages = _compile(tmp_path, "depth_test", motion_vectors=False, expected_depth=True)
        comp = stages["comp"]
        buf_names = {b["name"] for b in comp["descriptor_sets"]["0"]}
        assert "projected_depth" in buf_names
        assert "projected_mv" not in buf_names
        assert "splat_prev_pos" not in buf_names
        # No extra push fields needed for depth alone.
        assert comp["push_constants"][0]["size"] == 176

        frag = stages["frag"]
        assert [o["name"] for o in frag["outputs"]] == ["out_color", "out_depth"]
        # vec4, not vec2: Vulkan's fixed-function alpha blend reads "source
        # alpha" from the 4th component of the fragment output for this
        # attachment, which a vec2 output doesn't have -- see
        # splat_expander.py's out_depth comment.
        assert frag["outputs"][1]["type"] == "vec4"

        gs = comp["gaussian_splatting"]
        assert gs["expected_depth"] is True
        assert gs["motion_vectors"] is False


class TestMotionVectorsOnly:
    def test_mv_buffers_push_and_output(self, tmp_path):
        stages = _compile(tmp_path, "mv_test", motion_vectors=True, expected_depth=False)
        comp = stages["comp"]
        buf_names = {b["name"] for b in comp["descriptor_sets"]["0"]}
        assert "splat_prev_pos" in buf_names
        assert "projected_mv" in buf_names
        assert "projected_depth" not in buf_names

        push_fields = [f["name"] for f in comp["push_constants"][0]["fields"]]
        assert "proj_matrix_unjittered" in push_fields
        assert "prev_view_proj_unjittered" in push_fields
        assert comp["push_constants"][0]["size"] == 304

        frag = stages["frag"]
        assert [o["name"] for o in frag["outputs"]] == ["out_color", "out_motion"]
        assert frag["outputs"][1]["type"] == "vec4"

        vert = stages["vert"]
        assert "frag_mv" in [o["name"] for o in vert["outputs"]]

        gs = comp["gaussian_splatting"]
        assert gs["motion_vectors"] is True
        assert gs["expected_depth"] is False
        assert "splat_prev_pos" in gs["input_buffers"]
        assert "projected_mv" in gs["output_buffers"]


class TestBothOutputs:
    def test_both_and_with_motion_keyframes(self, tmp_path):
        # Also exercise this alongside `motion: keyframes` (the morph-apply
        # stage), since the real playground asset (dynamic splats) uses both
        # together -- the morph stage itself is untouched by these flags.
        stages = _compile(tmp_path, "both_test", motion_vectors=True,
                           expected_depth=True, motion_keyframes=True)
        assert "morph.comp" in stages  # morph-apply stage still emitted

        frag = stages["frag"]
        assert [o["name"] for o in frag["outputs"]] == ["out_color", "out_motion", "out_depth"]
        assert [o["location"] for o in frag["outputs"]] == [0, 1, 2]

        vert = stages["vert"]
        out_names = [o["name"] for o in vert["outputs"]]
        assert out_names == ["frag_conic", "frag_color", "frag_center",
                              "frag_offset", "frag_mv", "frag_depth"]

        comp = stages["comp"]
        assert comp["push_constants"][0]["size"] == 304
        # morph stage's own reflection is unaffected by the new flags.
        morph = stages["morph.comp"]["gaussian_splatting_morph"]
        assert morph["role"] == "morph_apply"


class TestBoolLiteralParsing:
    """`motion_vectors`/`expected_depth` default false when omitted."""

    def test_defaults_false(self, tmp_path):
        stages = _compile(tmp_path, "default_test", motion_vectors=False, expected_depth=False)
        gs = stages["comp"]["gaussian_splatting"]
        assert gs["motion_vectors"] is False
        assert gs["expected_depth"] is False
