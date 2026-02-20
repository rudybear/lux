"""Offline IBL preprocessing tool for PBR rendering.

Converts .exr or .hdr equirectangular panorama images into pre-filtered IBL
assets suitable for real-time PBR rendering with the split-sum approximation.

Output structure (playground/assets/ibl/<name>/):
    specular.bin    -- float16 RGBA, 6 faces x 9 mip levels (256..1)
    irradiance.bin  -- float16 RGBA, 6 faces x 1 mip level (32x32)
    brdf_lut.bin    -- float16 RG, 512x512
    manifest.json   -- metadata

Usage:
    python -m playground.preprocess_ibl <input.hdr|input.exr>
    python -m playground.preprocess_ibl --download <env_name>
    python -m playground.preprocess_ibl --download-models
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from math import acos, atan2, cos, floor, log2, pi, sin, sqrt
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYGROUND_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PLAYGROUND_DIR / "assets"

SPECULAR_FACE_SIZE = 256
SPECULAR_MIP_COUNT = 9  # 256, 128, 64, 32, 16, 8, 4, 2, 1
IRRADIANCE_FACE_SIZE = 32
BRDF_LUT_SIZE = 512

SPECULAR_SAMPLES = 256
IRRADIANCE_SAMPLES = 2048
BRDF_LUT_SAMPLES = 1024

# Standard OpenGL cubemap face directions
# Each face: (forward, right, up)
CUBEMAP_FACES = {
    0: {"name": "+X", "forward": np.array([1, 0, 0], dtype=np.float32),
        "right": np.array([0, 0, -1], dtype=np.float32),
        "up": np.array([0, -1, 0], dtype=np.float32)},
    1: {"name": "-X", "forward": np.array([-1, 0, 0], dtype=np.float32),
        "right": np.array([0, 0, 1], dtype=np.float32),
        "up": np.array([0, -1, 0], dtype=np.float32)},
    2: {"name": "+Y", "forward": np.array([0, 1, 0], dtype=np.float32),
        "right": np.array([1, 0, 0], dtype=np.float32),
        "up": np.array([0, 0, 1], dtype=np.float32)},
    3: {"name": "-Y", "forward": np.array([0, -1, 0], dtype=np.float32),
        "right": np.array([1, 0, 0], dtype=np.float32),
        "up": np.array([0, 0, -1], dtype=np.float32)},
    4: {"name": "+Z", "forward": np.array([0, 0, 1], dtype=np.float32),
        "right": np.array([1, 0, 0], dtype=np.float32),
        "up": np.array([0, -1, 0], dtype=np.float32)},
    5: {"name": "-Z", "forward": np.array([0, 0, -1], dtype=np.float32),
        "right": np.array([-1, 0, 0], dtype=np.float32),
        "up": np.array([0, -1, 0], dtype=np.float32)},
}

# HDR environment download URLs
# Note: These files are stored in Git LFS, so we use media.githubusercontent.com
# which resolves LFS pointers to actual file content.
HDR_BASE_URL = "https://media.githubusercontent.com/media/KhronosGroup/glTF-Sample-Environments/main"
AVAILABLE_ENVS = {
    "neutral": f"{HDR_BASE_URL}/neutral.hdr",
    "pisa": f"{HDR_BASE_URL}/pisa.hdr",
    "papermill": f"{HDR_BASE_URL}/papermill.hdr",
    "doge2": f"{HDR_BASE_URL}/doge2.hdr",
    "ennis": f"{HDR_BASE_URL}/ennis.hdr",
    "field": f"{HDR_BASE_URL}/field.hdr",
}

GLTF_MODELS_BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
GLTF_MODELS = {
    "MetalRoughSpheres": (
        f"{GLTF_MODELS_BASE}/MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb"
    ),
    "MetalRoughSpheresNoTextures": (
        f"{GLTF_MODELS_BASE}/MetalRoughSpheresNoTextures/glTF-Binary/"
        "MetalRoughSpheresNoTextures.glb"
    ),
}


# ---------------------------------------------------------------------------
# HDR/EXR loading
# ---------------------------------------------------------------------------

def _load_radiance_hdr(path: Path) -> np.ndarray:
    """Load a Radiance .hdr (RGBE) file as float32 RGB.

    Implements the RGBE format reader without external dependencies.
    Returns array of shape (H, W, 3) in linear HDR.
    """
    with open(path, "rb") as f:
        # Read header lines until empty line
        header_lines = []
        while True:
            line = f.readline()
            if not line or line.strip() == b"":
                break
            header_lines.append(line.decode("ascii", errors="replace").strip())

        # Parse resolution line: "-Y height +X width"
        res_line = f.readline().decode("ascii").strip()
        parts = res_line.split()
        if len(parts) != 4:
            raise ValueError(f"Cannot parse resolution: {res_line}")
        height = int(parts[1])
        width = int(parts[3])

        # Read scanlines (adaptive RLE or uncompressed)
        img = np.zeros((height, width, 3), dtype=np.float32)

        for y in range(height):
            # Read scanline header
            b0 = f.read(1)
            b1 = f.read(1)
            if not b0 or not b1:
                raise ValueError(f"Unexpected EOF at scanline {y}")

            if b0[0] == 2 and b1[0] == 2:
                # New-style adaptive RLE
                b2 = f.read(1)
                b3 = f.read(1)
                scanline_width = (b2[0] << 8) | b3[0]
                if scanline_width != width:
                    raise ValueError(f"Scanline width mismatch: {scanline_width} vs {width}")

                # Read 4 channels separately (R, G, B, E)
                channels = []
                for ch in range(4):
                    channel_data = bytearray()
                    while len(channel_data) < width:
                        count_byte = f.read(1)[0]
                        if count_byte > 128:
                            # RLE run
                            run_len = count_byte - 128
                            val = f.read(1)[0]
                            channel_data.extend([val] * run_len)
                        else:
                            # Literal run
                            channel_data.extend(f.read(count_byte))
                    channels.append(channel_data[:width])

                # Convert RGBE to float
                for x in range(width):
                    r, g, b, e = channels[0][x], channels[1][x], channels[2][x], channels[3][x]
                    if e > 0:
                        scale = 2.0 ** (e - 128 - 8)
                        img[y, x, 0] = r * scale
                        img[y, x, 1] = g * scale
                        img[y, x, 2] = b * scale
            else:
                # Old-style: 4 bytes per pixel (RGBE)
                b2 = f.read(1)
                b3 = f.read(1)
                r, g, b_val, e = b0[0], b1[0], b2[0], b3[0]
                if e > 0:
                    scale = 2.0 ** (e - 128 - 8)
                    img[y, 0, 0] = r * scale
                    img[y, 0, 1] = g * scale
                    img[y, 0, 2] = b_val * scale
                # Read remaining pixels
                for x in range(1, width):
                    pixel = f.read(4)
                    r, g, b_val, e = pixel[0], pixel[1], pixel[2], pixel[3]
                    if e > 0:
                        scale = 2.0 ** (e - 128 - 8)
                        img[y, x, 0] = r * scale
                        img[y, x, 1] = g * scale
                        img[y, x, 2] = b_val * scale

    return img


def load_panorama(path: Path) -> np.ndarray:
    """Load an equirectangular panorama image as float32 RGB(A).

    Returns an array of shape (H, W, 3) in linear HDR.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Native HDR loader for Radiance .hdr files
    if suffix == ".hdr":
        try:
            img = _load_radiance_hdr(path)
            print(f"  Loaded via native HDR reader: {img.shape[1]}x{img.shape[0]}, "
                  f"range [{img.min():.3f}, {img.max():.3f}]")
            return img
        except Exception as e_hdr:
            print(f"  Native HDR reader failed: {e_hdr}")

    # Try imageio
    try:
        import imageio.v3 as iio
        img = iio.imread(str(path))
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        print(f"  Loaded via imageio: {img.shape[1]}x{img.shape[0]}, "
              f"range [{img.min():.3f}, {img.max():.3f}]")
        return img
    except Exception as e_imageio:
        print(f"  imageio failed: {e_imageio}")

    # Fallback for .exr: try OpenEXR + Imath
    if suffix == ".exr":
        try:
            import Imath
            import OpenEXR

            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            dw = header["dataWindow"]
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            float_type = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = {}
            for ch_name in ["R", "G", "B"]:
                raw = exr_file.channel(ch_name, float_type)
                channels[ch_name] = np.frombuffer(raw, dtype=np.float32).reshape(
                    (height, width)
                )
            img = np.stack([channels["R"], channels["G"], channels["B"]], axis=-1)
            print(f"  Loaded via OpenEXR: {width}x{height}, "
                  f"range [{img.min():.3f}, {img.max():.3f}]")
            return img
        except Exception as e_exr:
            print(f"  OpenEXR fallback failed: {e_exr}")

    raise RuntimeError(
        f"Cannot load '{path}'. Supported formats: .hdr (native), "
        f".exr (needs OpenEXR+Imath), or any format supported by imageio."
    )


# ---------------------------------------------------------------------------
# Panorama sampling
# ---------------------------------------------------------------------------

def direction_to_equirect_uv(direction: np.ndarray) -> Tuple[float, float]:
    """Convert a 3D direction vector to equirectangular UV coordinates.

    Returns (u, v) in [0,1] range.
    """
    x, y, z = direction[0], direction[1], direction[2]
    u = 0.5 + atan2(x, z) / (2.0 * pi)
    v = 0.5 - np.arcsin(np.clip(y, -1.0, 1.0)) / pi
    return u, v


def sample_panorama(panorama: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Sample the equirectangular panorama at a 3D direction using bilinear interpolation."""
    h, w, _ = panorama.shape
    u, v = direction_to_equirect_uv(direction)

    # Convert to pixel coordinates
    px = u * w - 0.5
    py = v * h - 0.5

    x0 = int(floor(px))
    y0 = int(floor(py))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = px - x0
    fy = py - y0

    # Wrap x, clamp y
    x0 = x0 % w
    x1 = x1 % w
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    c00 = panorama[y0, x0]
    c10 = panorama[y0, x1]
    c01 = panorama[y1, x0]
    c11 = panorama[y1, x1]

    return (c00 * (1 - fx) * (1 - fy) +
            c10 * fx * (1 - fy) +
            c01 * (1 - fx) * fy +
            c11 * fx * fy)


def sample_panorama_vectorized(panorama: np.ndarray,
                               directions: np.ndarray) -> np.ndarray:
    """Sample the panorama for an array of directions (N, 3) -> (N, 3).

    Uses bilinear interpolation, vectorized with numpy.
    """
    h, w, _ = panorama.shape
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]

    u = 0.5 + np.arctan2(x, z) / (2.0 * pi)
    v = 0.5 - np.arcsin(np.clip(y, -1.0, 1.0)) / pi

    px = u * w - 0.5
    py = v * h - 0.5

    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)

    x0 = x0 % w
    x1 = x1 % w
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    c00 = panorama[y0, x0]
    c10 = panorama[y0, x1]
    c01 = panorama[y1, x0]
    c11 = panorama[y1, x1]

    fx = fx[:, np.newaxis]
    fy = fy[:, np.newaxis]

    return (c00 * (1 - fx) * (1 - fy) +
            c10 * fx * (1 - fy) +
            c01 * (1 - fx) * fy +
            c11 * fx * fy)


# ---------------------------------------------------------------------------
# Cubemap face direction generation
# ---------------------------------------------------------------------------

def cubemap_face_direction(face_idx: int, u: float, v: float) -> np.ndarray:
    """Compute the 3D direction for texel (u, v) on a cubemap face.

    u, v are in [-1, 1] range.
    """
    face = CUBEMAP_FACES[face_idx]
    d = face["forward"] + u * face["right"] + v * face["up"]
    norm = np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return d / norm


def cubemap_face_directions(face_idx: int, size: int) -> np.ndarray:
    """Generate all 3D directions for a cubemap face.

    Returns array of shape (size, size, 3).
    """
    face = CUBEMAP_FACES[face_idx]
    # texel centers
    coords = np.linspace(-1.0 + 1.0 / size, 1.0 - 1.0 / size, size,
                         dtype=np.float32)
    # u varies along columns, v varies along rows (v is flipped for top-to-bottom)
    uu, vv = np.meshgrid(coords, coords)  # Vulkan/WebGPU: row 0 = top = v=0

    directions = (face["forward"][np.newaxis, np.newaxis, :]
                  + uu[:, :, np.newaxis] * face["right"][np.newaxis, np.newaxis, :]
                  + vv[:, :, np.newaxis] * face["up"][np.newaxis, np.newaxis, :])

    # Normalize
    norms = np.linalg.norm(directions, axis=2, keepdims=True)
    return directions / norms


# ---------------------------------------------------------------------------
# Hammersley quasi-random sequence
# ---------------------------------------------------------------------------

def radical_inverse_vdc(bits: int) -> float:
    """Van der Corput radical inverse (bit reversal)."""
    bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
    bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
    bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return bits * 2.3283064365386963e-10  # / 0x100000000


def hammersley_sequence(n: int) -> np.ndarray:
    """Generate n Hammersley 2D points. Returns (n, 2) array."""
    points = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        points[i, 0] = i / n
        points[i, 1] = radical_inverse_vdc(i)
    return points


# ---------------------------------------------------------------------------
# GGX importance sampling
# ---------------------------------------------------------------------------

def importance_sample_ggx_batch(xi: np.ndarray, roughness: float,
                                N: np.ndarray) -> np.ndarray:
    """Importance sample GGX distribution for a batch of random samples.

    Args:
        xi: (num_samples, 2) Hammersley points
        roughness: surface roughness
        N: (3,) normal direction

    Returns:
        H: (num_samples, 3) half vectors in world space
    """
    a = roughness * roughness
    a2 = a * a

    phi = 2.0 * pi * xi[:, 0]
    denom = 1.0 + (a2 - 1.0) * xi[:, 1]
    cos_theta = np.sqrt((1.0 - xi[:, 1]) / np.maximum(denom, 1e-8))
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))

    # Tangent-space half vector
    Hx = np.cos(phi) * sin_theta
    Hy = np.sin(phi) * sin_theta
    Hz = cos_theta

    # Build tangent frame from N
    N = N / np.linalg.norm(N)
    if abs(N[1]) < 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent = np.cross(up, N)
    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(N, tangent)

    # Transform to world space
    H = (Hx[:, np.newaxis] * tangent[np.newaxis, :]
         + Hy[:, np.newaxis] * bitangent[np.newaxis, :]
         + Hz[:, np.newaxis] * N[np.newaxis, :])

    # Normalize
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Geometry / visibility functions for BRDF LUT
# ---------------------------------------------------------------------------

def v_smith_ggx_correlated(n_dot_v: float, n_dot_l: float,
                           roughness: float) -> float:
    """Exact height-correlated Smith GGX visibility (V form).

    This is the V term (already includes 1/(4*NdotL*NdotV) denominator),
    matching the Khronos glTF Sample Renderer and our runtime v_ggx_correlated.

    Uses a2 = roughness^4 (perceptualRoughness -> alpha -> alpha^2).
    """
    a = roughness * roughness
    a2 = a * a  # roughness^4
    ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2)
    ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2)
    denom = ggxv + ggxl
    if denom < 1e-8:
        return 0.0
    return 0.5 / denom


# ---------------------------------------------------------------------------
# Specular pre-filtering
# ---------------------------------------------------------------------------

def prefilter_specular_face(panorama: np.ndarray, face_idx: int, face_size: int,
                            roughness: float, num_samples: int) -> np.ndarray:
    """Pre-filter one cubemap face for a given roughness level.

    Returns (face_size, face_size, 4) float32 RGBA.
    """
    directions = cubemap_face_directions(face_idx, face_size)
    result = np.zeros((face_size, face_size, 4), dtype=np.float32)
    xi = hammersley_sequence(num_samples)

    for row in range(face_size):
        for col in range(face_size):
            N = directions[row, col].astype(np.float64)
            N = N / np.linalg.norm(N)

            if roughness < 1e-4:
                # Mip 0: mirror reflection, just sample the environment
                color = sample_panorama(panorama, N.astype(np.float32))
                result[row, col] = np.array([color[0], color[1], color[2], 1.0],
                                            dtype=np.float32)
                continue

            # Importance sample GGX
            H_batch = importance_sample_ggx_batch(xi, roughness, N)
            V = N.copy()

            # Reflect V about H to get L: L = 2 * dot(V, H) * H - V
            VdotH = np.sum(V[np.newaxis, :] * H_batch, axis=1)
            L_batch = 2.0 * VdotH[:, np.newaxis] * H_batch - V[np.newaxis, :]

            # N dot L
            NdotL = np.sum(N[np.newaxis, :] * L_batch, axis=1)

            # Filter valid samples (NdotL > 0)
            valid = NdotL > 0.0
            if not np.any(valid):
                result[row, col] = np.array([0, 0, 0, 1], dtype=np.float32)
                continue

            L_valid = L_batch[valid].astype(np.float32)
            NdotL_valid = NdotL[valid].astype(np.float32)

            # Normalize L directions for sampling
            L_norms = np.linalg.norm(L_valid, axis=1, keepdims=True)
            L_valid = L_valid / np.maximum(L_norms, 1e-8)

            # Sample panorama at each L direction
            colors = sample_panorama_vectorized(panorama, L_valid)

            # Weight by NdotL
            weights = NdotL_valid
            total_weight = weights.sum()

            if total_weight > 0:
                weighted_color = (colors * weights[:, np.newaxis]).sum(axis=0)
                weighted_color /= total_weight
            else:
                weighted_color = np.zeros(3, dtype=np.float32)

            result[row, col] = np.array(
                [weighted_color[0], weighted_color[1], weighted_color[2], 1.0],
                dtype=np.float32,
            )

    return result


def generate_specular(panorama: np.ndarray) -> bytes:
    """Generate the full pre-filtered specular cubemap (all faces, all mips).

    Layout in memory: for each mip level (0..4), for each face (0..5):
        face_size x face_size x 4 float16 values

    Returns raw bytes.
    """
    print("\n[Specular] Generating pre-filtered specular cubemap...")
    all_data = bytearray()

    for mip in range(SPECULAR_MIP_COUNT):
        face_size = SPECULAR_FACE_SIZE >> mip
        roughness = mip / max(SPECULAR_MIP_COUNT - 1, 1)
        num_samples = 1 if roughness < 1e-4 else SPECULAR_SAMPLES
        print(f"  Mip {mip}: {face_size}x{face_size}, "
              f"roughness={roughness:.2f}, samples={num_samples}")

        for face_idx in range(6):
            t0 = time.time()
            face_data = prefilter_specular_face(
                panorama, face_idx, face_size, roughness, num_samples
            )
            elapsed = time.time() - t0
            face_name = CUBEMAP_FACES[face_idx]["name"]
            print(f"    Face {face_name}: {elapsed:.1f}s")

            # Convert to float16 and append
            face_fp16 = face_data.astype(np.float16)
            all_data.extend(face_fp16.tobytes())

    print(f"  Specular total size: {len(all_data)} bytes")
    return bytes(all_data)


# ---------------------------------------------------------------------------
# Irradiance convolution
# ---------------------------------------------------------------------------

def convolve_irradiance_face(panorama: np.ndarray, face_idx: int,
                             face_size: int, num_samples: int) -> np.ndarray:
    """Compute diffuse irradiance for one cubemap face.

    Uses cosine-weighted hemisphere sampling.
    Returns (face_size, face_size, 4) float32 RGBA.
    """
    directions = cubemap_face_directions(face_idx, face_size)
    result = np.zeros((face_size, face_size, 4), dtype=np.float32)
    xi = hammersley_sequence(num_samples)

    for row in range(face_size):
        for col in range(face_size):
            N = directions[row, col].astype(np.float64)
            N = N / np.linalg.norm(N)

            # Build tangent frame
            if abs(N[1]) < 0.999:
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            tangent = np.cross(up, N)
            tangent = tangent / np.linalg.norm(tangent)
            bitangent = np.cross(N, tangent)

            # Cosine-weighted hemisphere sampling
            # pdf = cos(theta) / pi, sampling: theta = acos(sqrt(1-xi1)),
            # phi = 2*pi*xi0
            phi = 2.0 * pi * xi[:, 0]
            cos_theta = np.sqrt(1.0 - xi[:, 1])
            sin_theta = np.sqrt(xi[:, 1])

            # Tangent-space directions
            Lx = np.cos(phi) * sin_theta
            Ly = np.sin(phi) * sin_theta
            Lz = cos_theta

            # Transform to world space
            L = (Lx[:, np.newaxis] * tangent[np.newaxis, :]
                 + Ly[:, np.newaxis] * bitangent[np.newaxis, :]
                 + Lz[:, np.newaxis] * N[np.newaxis, :])

            # Normalize
            L_norms = np.linalg.norm(L, axis=1, keepdims=True)
            L = (L / np.maximum(L_norms, 1e-8)).astype(np.float32)

            # Sample panorama
            colors = sample_panorama_vectorized(panorama, L)

            # For cosine-weighted sampling, each sample already accounts
            # for the cosine weighting, so we just average
            irradiance = colors.mean(axis=0)

            result[row, col] = np.array(
                [irradiance[0], irradiance[1], irradiance[2], 1.0],
                dtype=np.float32,
            )

    return result


def generate_irradiance(panorama: np.ndarray) -> bytes:
    """Generate the diffuse irradiance cubemap.

    Returns raw bytes (float16 RGBA).
    """
    print("\n[Irradiance] Generating diffuse irradiance cubemap...")
    all_data = bytearray()

    for face_idx in range(6):
        t0 = time.time()
        face_data = convolve_irradiance_face(
            panorama, face_idx, IRRADIANCE_FACE_SIZE, IRRADIANCE_SAMPLES
        )
        elapsed = time.time() - t0
        face_name = CUBEMAP_FACES[face_idx]["name"]
        print(f"  Face {face_name}: {IRRADIANCE_FACE_SIZE}x{IRRADIANCE_FACE_SIZE}, "
              f"{elapsed:.1f}s")

        face_fp16 = face_data.astype(np.float16)
        all_data.extend(face_fp16.tobytes())

    print(f"  Irradiance total size: {len(all_data)} bytes")
    return bytes(all_data)


# ---------------------------------------------------------------------------
# BRDF integration LUT
# ---------------------------------------------------------------------------

def integrate_brdf(n_dot_v: float, roughness: float,
                   num_samples: int, xi_seq: np.ndarray) -> Tuple[float, float]:
    """Integrate the BRDF for a single (NdotV, roughness) pair.

    Uses exact height-correlated Smith GGX visibility (V form), matching
    the Khronos glTF Sample Renderer's BRDF LUT generation.

    Returns (scale, bias) for the split-sum approximation.
    """
    V = np.array([sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v], dtype=np.float64)
    N = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    A = 0.0
    B = 0.0

    # Importance sample GGX
    H_batch = importance_sample_ggx_batch(xi_seq, roughness, N)

    for i in range(num_samples):
        H = H_batch[i]
        VdotH = max(np.dot(V, H), 0.0)
        L = 2.0 * VdotH * H - V
        NdotL = max(L[2], 0.0)

        if NdotL > 0.0:
            NdotH = max(H[2], 0.0)
            # Exact Smith visibility (V form, includes 1/(4*NdotL*NdotV))
            V_smith = v_smith_ggx_correlated(n_dot_v, NdotL, roughness)
            # Khronos formula: V * VdotH * NdotL / NdotH
            V_pdf = V_smith * VdotH * NdotL / max(NdotH, 1e-8)
            Fc = (1.0 - VdotH) ** 5.0
            A += (1.0 - Fc) * V_pdf
            B += Fc * V_pdf

    # Factor of 4 compensates for the 1/4 in the V form
    scale = 4.0 * A / num_samples
    bias = 4.0 * B / num_samples
    return scale, bias


def generate_brdf_lut() -> bytes:
    """Generate the BRDF integration LUT.

    512x512 float16 RG texture, indexed by (NdotV, roughness).
    Row = roughness (0 at top, 1 at bottom), Col = NdotV (0 at left, 1 at right).

    Returns raw bytes.
    """
    print("\n[BRDF LUT] Generating BRDF integration LUT...")
    size = BRDF_LUT_SIZE
    lut = np.zeros((size, size, 2), dtype=np.float32)
    xi_seq = hammersley_sequence(BRDF_LUT_SAMPLES)

    t0 = time.time()
    total_pixels = size * size
    report_interval = max(total_pixels // 20, 1)
    pixel_count = 0

    for row in range(size):
        # roughness: row 0 = near 0, row (size-1) = 1.0
        roughness = (row + 0.5) / size
        roughness = max(roughness, 0.01)  # avoid divide-by-zero at roughness=0

        for col in range(size):
            n_dot_v = (col + 0.5) / size
            n_dot_v = max(n_dot_v, 0.001)  # avoid edge singularity

            s, b = integrate_brdf(n_dot_v, roughness, BRDF_LUT_SAMPLES, xi_seq)
            lut[row, col, 0] = s
            lut[row, col, 1] = b

            pixel_count += 1
            if pixel_count % report_interval == 0:
                pct = pixel_count / total_pixels * 100
                elapsed = time.time() - t0
                print(f"  Progress: {pct:.0f}% ({pixel_count}/{total_pixels}), "
                      f"elapsed: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  BRDF LUT complete: {size}x{size}, {elapsed:.1f}s")

    lut_fp16 = lut.astype(np.float16)
    data = lut_fp16.tobytes()
    print(f"  BRDF LUT size: {len(data)} bytes")
    return data


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(output_dir: Path) -> None:
    """Write manifest.json with metadata about the generated assets."""
    manifest = {
        "specular": {
            "file": "specular.bin",
            "format": "float16_rgba",
            "face_size": SPECULAR_FACE_SIZE,
            "mip_count": SPECULAR_MIP_COUNT,
            "mip_sizes": [SPECULAR_FACE_SIZE >> m for m in range(SPECULAR_MIP_COUNT)],
            "num_faces": 6,
            "samples_per_texel": SPECULAR_SAMPLES,
            "layout": "mip-major, then face-major (mip0-face0..5, mip1-face0..5, ...)",
        },
        "irradiance": {
            "file": "irradiance.bin",
            "format": "float16_rgba",
            "face_size": IRRADIANCE_FACE_SIZE,
            "mip_count": 1,
            "num_faces": 6,
            "samples_per_texel": IRRADIANCE_SAMPLES,
            "layout": "face0, face1, ..., face5",
        },
        "brdf_lut": {
            "file": "brdf_lut.bin",
            "format": "float16_rg",
            "size": BRDF_LUT_SIZE,
            "samples_per_texel": BRDF_LUT_SAMPLES,
            "layout": "row-major, row=roughness (0..1), col=NdotV (0..1)",
        },
        "cubemap_faces": [
            "+X (right)", "-X (left)", "+Y (up)", "-Y (down)",
            "+Z (front)", "-Z (back)",
        ],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Wrote {manifest_path}")


# ---------------------------------------------------------------------------
# Asset downloading
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL to a destination path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LuxShaderPlayground/1.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            block_size = 64 * 1024
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Downloaded: {downloaded}/{total} bytes ({pct:.0f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded} bytes", end="", flush=True)
        print()
        print(f"  OK: {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:
        print(f"\n  FAILED: {e}")
        return False


def download_environment(name: str) -> bool:
    """Download a named HDR environment map."""
    if name not in AVAILABLE_ENVS:
        print(f"  Unknown environment '{name}'. Available: {', '.join(AVAILABLE_ENVS)}")
        return False

    url = AVAILABLE_ENVS[name]
    dest = ASSETS_DIR / f"{name}.hdr"

    if dest.exists():
        print(f"  Already exists: {dest} ({dest.stat().st_size} bytes)")
        answer = input("  Re-download? [y/N] ").strip().lower()
        if answer != "y":
            print("  Skipped.")
            return True

    return download_file(url, dest)


def download_models() -> bool:
    """Download glTF test models."""
    success = True
    for model_name, url in GLTF_MODELS.items():
        dest = ASSETS_DIR / f"{model_name}.glb"
        if dest.exists():
            print(f"  Already exists: {dest} ({dest.stat().st_size} bytes)")
            continue
        if not download_file(url, dest):
            success = False
    return success


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_panorama(input_path: Path) -> None:
    """Full IBL preprocessing pipeline for a panorama file."""
    input_path = Path(input_path).resolve()
    name = input_path.stem
    output_dir = PLAYGROUND_DIR / "assets" / "ibl" / name

    print("=" * 60)
    print(f"  IBL Preprocessing: {input_path.name}")
    print(f"  Output directory:  {output_dir}")
    print("=" * 60)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load panorama
    print("\n[Load] Loading panorama...")
    panorama = load_panorama(input_path)
    print(f"  Panorama shape: {panorama.shape}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: BRDF LUT (does not depend on panorama)
    brdf_data = generate_brdf_lut()
    brdf_path = output_dir / "brdf_lut.bin"
    with open(brdf_path, "wb") as f:
        f.write(brdf_data)
    print(f"  Wrote {brdf_path}")

    # Phase 2: Specular pre-filtering
    specular_data = generate_specular(panorama)
    specular_path = output_dir / "specular.bin"
    with open(specular_path, "wb") as f:
        f.write(specular_data)
    print(f"  Wrote {specular_path}")

    # Phase 3: Irradiance convolution
    irradiance_data = generate_irradiance(panorama)
    irradiance_path = output_dir / "irradiance.bin"
    with open(irradiance_path, "wb") as f:
        f.write(irradiance_data)
    print(f"  Wrote {irradiance_path}")

    # Phase 4: Manifest
    write_manifest(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  IBL Preprocessing Complete!")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Files:")
    for p in sorted(output_dir.iterdir()):
        print(f"    {p.name:20s}  {p.stat().st_size:>10,} bytes")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IBL preprocessing tool for PBR rendering. "
                    "Converts equirectangular HDR/EXR panoramas into "
                    "pre-filtered cubemap assets.",
        prog="python -m playground.preprocess_ibl",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "input", nargs="?", default=None,
        help="Path to .hdr or .exr equirectangular panorama file",
    )
    group.add_argument(
        "--download", metavar="ENV_NAME",
        help=f"Download a named HDR environment. "
             f"Available: {', '.join(sorted(AVAILABLE_ENVS))}",
    )
    group.add_argument(
        "--download-models", action="store_true",
        help="Download glTF test models (MetalRoughSpheres)",
    )

    args = parser.parse_args()

    if args.download:
        print("=" * 60)
        print(f"  Downloading HDR environment: {args.download}")
        print("=" * 60)
        success = download_environment(args.download)
        if not success:
            sys.exit(1)
        # Auto-process after download
        hdr_path = ASSETS_DIR / f"{args.download}.hdr"
        if hdr_path.exists():
            print(f"\nAuto-processing downloaded environment...")
            process_panorama(hdr_path)

    elif args.download_models:
        print("=" * 60)
        print("  Downloading glTF test models")
        print("=" * 60)
        success = download_models()
        if not success:
            sys.exit(1)
        print("\nDone.")

    else:
        process_panorama(Path(args.input))


if __name__ == "__main__":
    main()
