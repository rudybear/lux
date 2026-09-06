"""Compiler-only tests for the `reconstruct` block (luxc/expansion/
reconstruct_expander.py, SPECIFICATION.md section 12.9) -- no GPU, just
compiles a small `.lux` source and checks the generated stage/reflection
structure. See tests/test_dlss_outputs.py's TestExpectedDepth /
tests/test_gaussian_splatting.py for the GPU-level analogues this mirrors.
"""

import json
from pathlib import Path

from luxc.builtins.types import clear_type_aliases
from luxc.compiler import compile_source

REPO_ROOT = Path(__file__).resolve().parent.parent


def _compile(tmp_path: Path, stem: str, s: int = 2, k: int = 4,
             param_stride: int = 1, hidden: int = 8) -> dict:
    src = f"""
reconstruct R {{
    s: {s},
    k: {k},
    param_stride: {param_stride},
    hidden: {hidden},
}}

pipeline P {{
    mode: reconstruct,
    reconstruct: R,
}}
"""
    clear_type_aliases()
    compile_source(src, stem, tmp_path, emit_reflection=True, validate=True)
    stages = {}
    for suffix in ("warp", "apply", "blend"):
        p = tmp_path / f"{stem}.{suffix}.comp.json"
        assert p.exists(), f"missing {p}"
        stages[suffix] = json.loads(p.read_text())
    return stages


class TestReconstructStages:
    def test_three_compute_stages_emitted(self, tmp_path):
        stages = _compile(tmp_path, "recon_test")
        for suffix in ("warp", "apply", "blend"):
            assert stages[suffix]["stage"] == "compute"
            assert stages[suffix]["compute"]["workgroup_size"] == [256, 1, 1]

    def test_warp_buffers_and_push(self, tmp_path):
        stages = _compile(tmp_path, "recon_test")
        buf_names = [b["name"] for b in stages["warp"]["descriptor_sets"]["0"]]
        assert buf_names == ["prev_color", "mv_proxy", "warped_color"]
        push_fields = [f["name"] for f in stages["warp"]["push_constants"][0]["fields"]]
        assert push_fields == ["target_w", "target_h", "proxy_w", "proxy_h"]

    def test_apply_buffers_and_push_default_config(self, tmp_path):
        stages = _compile(tmp_path, "recon_test", s=2, k=4, param_stride=1, hidden=8)
        buf_names = [b["name"] for b in stages["apply"]["descriptor_sets"]["0"]]
        assert buf_names == ["packed_params", "proxy_color", "spatial_out", "alpha_out", "hidden_out"]
        push_fields = [f["name"] for f in stages["apply"]["push_constants"][0]["fields"]]
        assert push_fields == ["target_w", "target_h", "proxy_w", "proxy_h", "net_w", "net_h",
                                "jitter_x", "jitter_y"]

    def test_blend_buffers_and_push(self, tmp_path):
        stages = _compile(tmp_path, "recon_test")
        buf_names = [b["name"] for b in stages["blend"]["descriptor_sets"]["0"]]
        assert buf_names == ["spatial_out", "warped_color", "alpha_out", "disocc", "out_color"]
        push_fields = [f["name"] for f in stages["blend"]["push_constants"][0]["fields"]]
        assert push_fields == ["target_w", "target_h"]

    def test_param_stride_2_compiles(self, tmp_path):
        # sp = s*param_stride = 4 -> more kernel-logit channels, larger un-multiplex
        # factor; this exercises the general (not just sp==s) index math.
        stages = _compile(tmp_path, "recon_ps2", s=2, k=4, param_stride=2, hidden=8)
        buf_names = [b["name"] for b in stages["apply"]["descriptor_sets"]["0"]]
        assert buf_names == ["packed_params", "proxy_color", "spatial_out", "alpha_out", "hidden_out"]

    def test_different_k_compiles(self, tmp_path):
        # K=6 (a plausible alternative tap count) exercises the general
        # _tap_offsets(K)/softmax unroll, not just the K=4 case used elsewhere.
        _compile(tmp_path, "recon_k6", s=2, k=6, param_stride=1, hidden=4)


def _compile_mem(tmp_path: Path, stem: str, s: int = 2, k: int = 4,
                  param_stride: int = 1, hidden: int = 8,
                  mem_channels: int = 8, mem_hidden: int = 16) -> dict:
    src = f"""
reconstruct R {{
    s: {s},
    k: {k},
    param_stride: {param_stride},
    hidden: {hidden},
    memory: {{
        channels: {mem_channels},
        hidden: {mem_hidden},
    }},
}}

pipeline P {{
    mode: reconstruct,
    reconstruct: R,
}}
"""
    clear_type_aliases()
    compile_source(src, stem, tmp_path, emit_reflection=True, validate=True)
    stages = {}
    for suffix in ("warp", "apply", "bguv", "memory", "blend"):
        p = tmp_path / f"{stem}.{suffix}.comp.json"
        assert p.exists(), f"missing {p}"
        stages[suffix] = json.loads(p.read_text())
    return stages


class TestReconstructMemory:
    """`memory: { channels, hidden }` sub-block -- explicit per-scene memory
    (mobiledlss's docs/scene-memory-spec.md), 3-way blend."""

    def test_no_memory_block_keeps_2way_shape(self, tmp_path):
        # Backward compatibility: a reconstruct block with no `memory` sub-
        # block emits exactly the original 3 stages / 2-way blend buffers,
        # byte-identical to the stage-1 port (see the tests above).
        stages = _compile(tmp_path, "recon_no_mem")
        assert set(stages.keys()) == {"warp", "apply", "blend"}
        buf_names = [b["name"] for b in stages["apply"]["descriptor_sets"]["0"]]
        assert "alpha_out" in buf_names
        assert "blend_out" not in buf_names

    def test_five_compute_stages_emitted(self, tmp_path):
        stages = _compile_mem(tmp_path, "recon_mem_test")
        for suffix in ("warp", "apply", "bguv", "memory", "blend"):
            assert stages[suffix]["stage"] == "compute"
            assert stages[suffix]["compute"]["workgroup_size"] == [256, 1, 1]

    def test_apply_uses_3way_blend_out(self, tmp_path):
        stages = _compile_mem(tmp_path, "recon_mem_test")
        buf_names = [b["name"] for b in stages["apply"]["descriptor_sets"]["0"]]
        assert buf_names == ["packed_params", "proxy_color", "spatial_out", "blend_out", "hidden_out"]

    def test_bguv_buffers_and_push(self, tmp_path):
        stages = _compile_mem(tmp_path, "recon_mem_test")
        buf_names = [b["name"] for b in stages["bguv"]["descriptor_sets"]["0"]]
        assert buf_names == ["bguv_out"]
        push_fields = [f["name"] for f in stages["bguv"]["push_constants"][0]["fields"]]
        assert push_fields == ["width", "height", "k_params", "cam_to_world", "bg_sphere"]

    def test_memory_buffers_and_push(self, tmp_path):
        stages = _compile_mem(tmp_path, "recon_mem_test")
        buf_names = [b["name"] for b in stages["memory"]["descriptor_sets"]["0"]]
        assert buf_names == ["bguv_in", "bg_texture", "mem_fc1_w", "mem_fc1_b",
                              "mem_fc2_w", "mem_fc2_b", "bg_features_out", "memory_color_out"]
        push_fields = [f["name"] for f in stages["memory"]["push_constants"][0]["fields"]]
        assert push_fields == ["width", "height", "tex_w", "tex_h"]

    def test_blend_uses_3way_buffers(self, tmp_path):
        stages = _compile_mem(tmp_path, "recon_mem_test")
        buf_names = [b["name"] for b in stages["blend"]["descriptor_sets"]["0"]]
        assert buf_names == ["spatial_out", "warped_color", "blend_out", "disocc",
                              "memory_color", "out_color"]

    def test_different_channels_and_hidden_compile(self, tmp_path):
        # A non-default memory architecture exercises the general (not just
        # 8/16) fully-unrolled decoder-MLP codegen.
        _compile_mem(tmp_path, "recon_mem_c4h8", mem_channels=4, mem_hidden=8)

    def test_param_stride_2_with_memory_compiles(self, tmp_path):
        _compile_mem(tmp_path, "recon_mem_ps2", param_stride=2)
