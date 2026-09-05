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
