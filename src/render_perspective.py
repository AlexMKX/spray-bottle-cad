"""CPU pinhole renderer for diagnostic color, visible part IDs and metric depth."""

import numpy as np
import trimesh

PARTS = {"body": 1, "pump": 2, "label": 3, "lettering": 4}
COLORS = {1: (169, 201, 233), 2: (45, 42, 43), 3: (52, 49, 52), 4: (228, 231, 235)}


def load_parts(directory):
    return {name: trimesh.load_mesh(directory / f"{name}.stl")
            for name in PARTS if name != "lettering" or (directory / "lettering.stl").exists()}


def render(parts, camera, rotation, translation, size):
    width, height = size
    depth = np.full((height, width), np.inf, dtype=np.float32)
    ids = np.zeros((height, width), dtype=np.uint8)
    color = np.full((height, width, 3), 240, dtype=np.uint8)
    light = np.array([-0.35, -0.45, -0.82])
    for name, mesh in parts.items():
        camera_xyz = mesh.vertices @ rotation.T + translation
        if np.any(camera_xyz[:, 2] <= 0):
            raise ValueError("Model intersects or lies behind the camera plane")
        homogeneous = camera_xyz @ camera.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
        reciprocal_depth = 1 / camera_xyz[:, 2]
        intensity = 0.52 + 0.48 * np.clip(mesh.face_normals @ rotation.T @ light, 0, 1)
        part = PARTS[name]
        for face, shade in zip(mesh.faces, intensity):
            tri = pixels[face]
            left, top = np.maximum(0, np.ceil(tri.min(axis=0))).astype(int)
            right, bottom = np.minimum([width - 1, height - 1],
                                       np.floor(tri.max(axis=0))).astype(int)
            if left > right or top > bottom:
                continue
            a, b = tri[1:] - tri[0]
            area = a[0] * b[1] - a[1] * b[0]
            if abs(area) < 1e-10:
                continue
            y, x = np.ogrid[top:bottom + 1, left:right + 1]
            x, y = x - tri[0, 0], y - tri[0, 1]
            u = (x * b[1] - y * b[0]) / area
            v = (a[0] * y - a[1] * x) / area
            w = 1 - u - v
            inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
            if not inside.any():
                continue
            region = np.s_[top:bottom + 1, left:right + 1]
            d0, d1, d2 = reciprocal_depth[face]
            candidate = 1 / (w[inside] * d0 + u[inside] * d1 + v[inside] * d2)
            iy, ix = np.nonzero(inside)
            depth_region = depth[region]
            closer = candidate < depth_region[iy, ix]
            iy, ix = iy[closer], ix[closer]
            depth_region[iy, ix] = candidate[closer]
            ids[region][iy, ix] = part
            color[region][iy, ix] = np.rint(np.asarray(COLORS[part]) * shade)
    return color, ids, depth
