"""Scene utility functions: geometry generators and camera math for the Lux engine."""

import math
import struct
import numpy as np


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Column-major perspective projection matrix (Vulkan clip space: y-down, z [0,1])."""
    f = 1.0 / math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f
    m[2, 2] = far / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (near * far) / (near - far)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Column-major look-at view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0];  m[1, 0] = s[1];  m[2, 0] = s[2]
    m[0, 1] = u[0];  m[1, 1] = u[1];  m[2, 1] = u[2]
    m[0, 2] = -f[0]; m[1, 2] = -f[1]; m[2, 2] = -f[2]
    m[3, 0] = -np.dot(s, eye)
    m[3, 1] = -np.dot(u, eye)
    m[3, 2] = np.dot(f, eye)
    return m


def generate_sphere(stacks: int = 32, slices: int = 32, radius: float = 1.0):
    """Generate a UV sphere. Returns (vertices, indices).
    Vertices: list of (px,py,pz, nx,ny,nz, u,v) — 8 floats per vertex.
    Indices: list of uint32 triangle indices.
    """
    vertices = []
    indices = []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        v_coord = i / stacks
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            u_coord = j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            vertices.extend([x * radius, y * radius, z * radius, x, y, z, u_coord, v_coord])
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices.extend([a, b, a + 1, b, b + 1, a + 1])
    return vertices, indices


def generate_triangle():
    """Generate a colored triangle in clip space.
    Returns vertex_data (bytes), num_vertices=3, stride=24.
    """
    verts = [
        0.0,  0.5, 0.0,  1.0, 0.0, 0.0,
       -0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
        0.5, -0.5, 0.0,  0.0, 0.0, 1.0,
    ]
    return struct.pack(f"{len(verts)}f", *verts), 3, 24


def generate_procedural_texture(size: int = 512) -> np.ndarray:
    """Generate a colorful checker texture. Returns (size,size,4) uint8 RGBA."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            cx = int(u * 8) % 2
            cy = int(v * 8) % 2
            checker = cx ^ cy
            if checker:
                r = int(200 + 40 * math.sin(u * 12.0))
                g = int(120 + 30 * math.sin(v * 10.0))
                b = int(80 + 20 * math.cos(u * 8.0 + v * 6.0))
            else:
                r = int(60 + 30 * math.sin(u * 10.0 + v * 4.0))
                g = int(150 + 40 * math.cos(v * 8.0))
                b = int(170 + 50 * math.sin(u * 6.0))
            img[y, x] = [min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)), 255]
    return img
