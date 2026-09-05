"""Generates tests/assets/cameras/juggle_az0.json: the single fixed camera
used to render tests/assets (out-of-repo) juggle_stride4.glb frames 0/40/80
for the docs/lux-4d-spec.md section D real-scene check.

Self-contained (numpy only, no mobiledlss import -- lux stays free of
mobiledlss dependencies) reproduction of mobiledlss/mobiledlss/datagen/
camera.py's look_at() + intrinsics() + orbit_path() for a single fixed
azimuth (az=0), matching that module's exact formulas line-for-line so the
camera this JSON encodes is provably identical to what orbit_path(...)[0]
would produce.

Scene: PanopticSports "juggle" (39 keyframes, stride 4, fps=30), a y-down
world (world-up = (0,-1,0)). Actor/action center taken as
world (-0.22, -1.04, 0.09) (given). Orbit parameters: radius=2.0,
elevation_deg=15, az=0 (i.e. the very first camera on a would-be orbit
path -- see orbit_path(..., start_deg=0.0)):

    eye = center + radius * (cos(el)*cos(az), -sin(el) + bob, cos(el)*sin(az))

with bob=0 (bob_amplitude=0 for a static single camera). K uses fovY=50deg
on a 960x540 image, fx=fy (square pixels): fy = 0.5*height/tan(fovY/2).

viewmat_cv is OpenCV world->camera (+x right, +y down, +z forward);
--camera-json converts it to lux's GL convention via
viewmat_gl = diag(1,-1,-1,1) * viewmat_cv (see docs/lux-4d-spec.md
section 4 and playground_cpp/src/dlss_io.cpp).
"""
import json
import math
from pathlib import Path

import numpy as np


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-convention world->camera 4x4 (camera looks down +z, x right,
    y down). `up` is the world's up direction; PanopticSports is y-down so
    up=(0,-1,0) (matches mobiledlss.datagen.camera.look_at's default)."""
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(-up, f)
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)
    R = np.stack([r, u, f], axis=0)
    view = np.eye(4)
    view[:3, :3] = R
    view[:3, 3] = -R @ eye
    return view


def intrinsics(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    fy = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([[fy, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    center = np.array([-0.22, -1.04, 0.09])
    world_up = np.array([0.0, -1.0, 0.0])
    radius = 2.0
    elevation_deg = 15.0
    az_deg = 0.0
    width, height = 960, 540
    fov_y_deg = 50.0

    el = math.radians(elevation_deg)
    az = math.radians(az_deg)
    eye = center + radius * np.array([
        math.cos(el) * math.cos(az),
        -math.sin(el),
        math.cos(el) * math.sin(az),
    ])
    viewmat_cv = look_at(eye, center, world_up)
    K = intrinsics(width, height, fov_y_deg)

    out = {
        "viewmat_cv": viewmat_cv.flatten().tolist(),
        "K": K.flatten().tolist(),
        "width": width,
        "height": height,
    }
    out_path = Path(__file__).parent / "juggle_az0.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"eye = {eye.tolist()}")
    print(f"wrote {out_path}")
