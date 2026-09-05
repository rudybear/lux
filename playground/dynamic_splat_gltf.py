"""Loader + evaluator for dynamic (morph-target-animated) Gaussian splat glTF assets.

Implements the `MOBILEDLSS_dynamic_splats` convention documented alongside the
mobiledlss `dyn_splat_gltf.py` reference writer: a `KHR_gaussian_splatting`
POINTS primitive whose `targets` list carries one **sparse** morph target per
keyframe (deltas from the base/keyframe-0 attributes for POSITION,
`KHR_gaussian_splatting:ROTATION`, and `KHR_gaussian_splatting:SH_DEGREE_0_COEF_0`),
driven by a standard glTF `animation` with a `LINEAR` sampler on the node's
`weights` path.

This module is intentionally dependency-free beyond numpy (no torch, no
mobiledlss import) so it can run inside lux's own test suite. It mirrors --
but does not import -- `mobiledlss/gltf/dyn_splat_gltf.py`'s `read_dynamic_splat_glb`
and `evaluate_at`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXT = "KHR_gaussian_splatting"
A_ROT = f"{EXT}:ROTATION"
A_SCALE = f"{EXT}:SCALE"
A_OPACITY = f"{EXT}:OPACITY"
A_SH0 = f"{EXT}:SH_DEGREE_0_COEF_0"
CONVENTION = "MOBILEDLSS_dynamic_splats"

_FLOAT = 5126
_UINT = 5125
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
SH0_C0 = 0.28209479177387814  # Y_0^0; rgb = 0.5 + C0 * sh0


def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    return sh0 * SH0_C0 + 0.5


@dataclass
class MorphTarget:
    """A single sparse morph target: deltas from the base attributes."""
    indices: np.ndarray             # (M,) uint32, into the base N splats
    dpos: np.ndarray                # (M, 3) float32
    drot: np.ndarray                # (M, 4) float32
    dsh0: np.ndarray                # (M, 3) float32


@dataclass
class DynamicSplatAsset:
    num_splats: int
    base_pos: np.ndarray            # (N, 3)
    base_rot: np.ndarray            # (N, 4) xyzw
    base_sh0: np.ndarray            # (N, 3)
    base_scale: np.ndarray          # (N, 3) -- static, not animated in v1
    base_opacity: np.ndarray        # (N,)   -- static, not animated in v1
    targets: list[MorphTarget]      # length K
    keyframe_times: np.ndarray      # (K+1,) seconds, keyframe_times[0] == time of the base frame
    keyframe_weights: np.ndarray    # (K+1, K) one-hot per non-base keyframe (row 0 all-zero)
    extras: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return float(self.extras.get("fps", 30.0))

    @property
    def extras_keyframes(self) -> list[int] | None:
        return self.extras.get("keyframes")

    def frame_to_time(self, frame: int) -> float:
        """Map an integer video-frame index to a time in seconds.

        Uses `extras.fps` when present (frame / fps); otherwise falls back to
        indexing directly into the animation's own keyframe times.
        """
        if "fps" in self.extras:
            return frame / float(self.extras["fps"])
        idx = max(0, min(frame, len(self.keyframe_times) - 1))
        return float(self.keyframe_times[idx])


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, _ver, _len = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB file")
    off, gltf, bin_data = 12, None, b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:
            bin_data = chunk
        off += 8 + clen
    if gltf is None:
        raise ValueError(f"{path}: no JSON chunk found")
    return gltf, bin_data


def _read_accessor(gltf: dict, bin_data: bytes, index: int) -> np.ndarray:
    """Read an accessor, honoring sparse overrides and bufferView-less (all-zero) bases.

    This is the crux of dynamic-splat support: morph target accessors are
    typically **sparse with no bufferView** (the dense base is implicitly
    all-zero, since morph targets store *deltas*).
    """
    a = gltf["accessors"][index]
    n = _NCOMP[a["type"]]
    dtype = {_FLOAT: np.float32, _UINT: np.uint32}[a["componentType"]]
    if "bufferView" in a:
        bv = gltf["bufferViews"][a["bufferView"]]
        offset = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        out = np.frombuffer(bin_data, dtype, count=a["count"] * n, offset=offset).reshape(a["count"], n).copy()
    else:
        # No bufferView: dense base is implicitly zero (valid glTF for a
        # sparse accessor -- required for morph-target deltas on gaussians
        # that never move).
        out = np.zeros((a["count"], n), dtype=dtype)
    if "sparse" in a:
        s = a["sparse"]
        iv = gltf["bufferViews"][s["indices"]["bufferView"]]
        i_off = iv.get("byteOffset", 0) + s["indices"].get("byteOffset", 0)
        vv = gltf["bufferViews"][s["values"]["bufferView"]]
        v_off = vv.get("byteOffset", 0) + s["values"].get("byteOffset", 0)
        idx_dtype = {_UINT: np.uint32, 5123: np.uint16, 5121: np.uint8}[s["indices"]["componentType"]]
        idx = np.frombuffer(bin_data, idx_dtype, count=s["count"], offset=i_off).astype(np.uint32)
        vals = np.frombuffer(bin_data, np.float32, count=s["count"] * n, offset=v_off).reshape(-1, n)
        out[idx] = vals
    return out


def _read_sparse_indices_only(gltf: dict, bin_data: bytes, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, values) for a sparse accessor without materializing the full dense array."""
    a = gltf["accessors"][index]
    n = _NCOMP[a["type"]]
    if "sparse" not in a:
        # Dense accessor with no sparse override: every index is "touched".
        dense = _read_accessor(gltf, bin_data, index)
        return np.arange(dense.shape[0], dtype=np.uint32), dense
    s = a["sparse"]
    iv = gltf["bufferViews"][s["indices"]["bufferView"]]
    i_off = iv.get("byteOffset", 0) + s["indices"].get("byteOffset", 0)
    vv = gltf["bufferViews"][s["values"]["bufferView"]]
    v_off = vv.get("byteOffset", 0) + s["values"].get("byteOffset", 0)
    idx_dtype = {_UINT: np.uint32, 5123: np.uint16, 5121: np.uint8}[s["indices"]["componentType"]]
    idx = np.frombuffer(bin_data, idx_dtype, count=s["count"], offset=i_off).astype(np.uint32)
    vals = np.frombuffer(bin_data, np.float32, count=s["count"] * n, offset=v_off).reshape(-1, n)
    return idx, vals


def is_dynamic_splat_glb(path: str | Path) -> bool:
    """True if the splat POINTS primitive in `path` carries morph targets.

    Static splat assets (no `targets`, no `animation`) must be loaded via the
    ordinary static path unchanged -- this is the gate callers should check
    before invoking `load_dynamic_splat_glb`.
    """
    gltf, _bin = _read_glb(Path(path))
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("targets"):
                return True
    return False


def load_dynamic_splat_glb(path: str | Path) -> DynamicSplatAsset:
    """Load a dynamic-splat GLB (base attributes + sparse morph targets + animation)."""
    gltf, bin_data = _read_glb(Path(path))
    mesh = gltf["meshes"][0]
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]

    base_pos = _read_accessor(gltf, bin_data, attrs["POSITION"])
    base_rot = _read_accessor(gltf, bin_data, attrs[A_ROT])
    base_sh0 = _read_accessor(gltf, bin_data, attrs[A_SH0])
    base_scale = _read_accessor(gltf, bin_data, attrs[A_SCALE]) if A_SCALE in attrs else np.zeros_like(base_pos)
    base_opacity = _read_accessor(gltf, bin_data, attrs[A_OPACITY]).reshape(-1) if A_OPACITY in attrs else np.zeros(base_pos.shape[0], dtype=np.float32)
    num_splats = base_pos.shape[0]

    targets: list[MorphTarget] = []
    for tgt in prim.get("targets", []):
        pidx, pvals = _read_sparse_indices_only(gltf, bin_data, tgt["POSITION"])
        ridx, rvals = _read_sparse_indices_only(gltf, bin_data, tgt[A_ROT])
        sidx, svals = _read_sparse_indices_only(gltf, bin_data, tgt[A_SH0])
        # The reference writer always uses the same index set for all three
        # semantics of a given target (they're all gated on the same "moved"
        # mask), but don't assume it: union them defensively.
        if not (np.array_equal(pidx, ridx) and np.array_equal(pidx, sidx)):
            union = np.unique(np.concatenate([pidx, ridx, sidx]))
            dpos = np.zeros((len(union), 3), dtype=np.float32)
            drot = np.zeros((len(union), 4), dtype=np.float32)
            dsh0 = np.zeros((len(union), 3), dtype=np.float32)
            pos_map = {int(v): i for i, v in enumerate(union)}
            for i, v in enumerate(pidx):
                dpos[pos_map[int(v)]] = pvals[i]
            for i, v in enumerate(ridx):
                drot[pos_map[int(v)]] = rvals[i]
            for i, v in enumerate(sidx):
                dsh0[pos_map[int(v)]] = svals[i]
            targets.append(MorphTarget(union.astype(np.uint32), dpos, drot, dsh0))
        else:
            targets.append(MorphTarget(pidx, pvals, rvals, svals))

    node = gltf["nodes"][0]
    anim = gltf["animations"][0]
    sampler = anim["samplers"][0]
    if sampler.get("interpolation", "LINEAR") != "LINEAR":
        raise ValueError("dynamic splat animation sampler must be LINEAR")
    times = _read_accessor(gltf, bin_data, sampler["input"]).reshape(-1)
    weights_flat = _read_accessor(gltf, bin_data, sampler["output"]).reshape(-1)
    num_keyframes = times.shape[0]
    num_targets = len(targets)
    weights = weights_flat.reshape(num_keyframes, num_targets)

    extras = mesh.get("extras", {}).get(CONVENTION, {})

    return DynamicSplatAsset(
        num_splats=num_splats,
        base_pos=base_pos, base_rot=base_rot, base_sh0=base_sh0,
        base_scale=base_scale, base_opacity=base_opacity,
        targets=targets,
        keyframe_times=times,
        keyframe_weights=weights,
        extras=extras,
    )


def evaluate_weights_at(asset: DynamicSplatAsset, time: float) -> np.ndarray:
    """glTF LINEAR-sampler semantics: piecewise-linear interpolation of the
    per-target weights vector at `time`, clamped to the animation's time range."""
    times = asset.keyframe_times
    weights = asset.keyframe_weights
    if time <= times[0]:
        return weights[0].copy()
    if time >= times[-1]:
        return weights[-1].copy()
    # bisect: find i such that times[i] <= time <= times[i+1]
    i = int(np.searchsorted(times, time, side="right")) - 1
    i = max(0, min(i, len(times) - 2))
    t0, t1 = times[i], times[i + 1]
    a = 0.0 if t1 <= t0 else (time - t0) / (t1 - t0)
    return (1.0 - a) * weights[i] + a * weights[i + 1]


def evaluate_at(asset: DynamicSplatAsset, time: float) -> dict:
    """Evaluate animated attributes at `time` (seconds), per the spec's semantics:

        attr(t) = base + sum_j w_j(t) * delta_j

    followed by quaternion renormalization. Generic: sums over however many
    target weights are nonzero (not hardcoded to exactly two), so this is
    both the "keyframe" and "generic >2 non-zero weights" reference path.
    """
    w = evaluate_weights_at(asset, time)
    pos = asset.base_pos.copy()
    rot = asset.base_rot.copy()
    sh0 = asset.base_sh0.copy()
    for j, wj in enumerate(w):
        if wj == 0.0:
            continue
        tgt = asset.targets[j]
        pos[tgt.indices] += wj * tgt.dpos
        rot[tgt.indices] += wj * tgt.drot
        sh0[tgt.indices] += wj * tgt.dsh0
    rot_norm = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    return {
        "positions": pos,
        "rotations": rot_norm,
        "sh0": sh0,
        "colors": sh0_to_rgb(sh0).clip(0.0, 1.0),
        "scales": asset.base_scale,
        "opacities": asset.base_opacity,
        "weights": w,
    }
