from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibrate_camera import calibrate, detect_grid, mat_points, project
from render_perspective import render
from visual_validate import compare


def test_lattice_to_camera_recovers_projection_above_mat():
    width, height = 640, 480
    camera = np.array([[535., 0, 319.5], [0, 535, 239.5], [0, 0, 1]])
    cols, rows, pitch = 14, 12, 10
    coordinates = np.array([(x, y) for x in range(cols + 1) for y in range(rows + 1)])
    xyz = np.column_stack([coordinates[:, 0] * pitch, -coordinates[:, 1] * pitch,
                           np.zeros(len(coordinates))])
    grids = []
    for offset in (-0.22, 0.05, 0.27):
        rvec = np.array([2.46, offset, -0.12])
        translation = np.array([-66., -63., 248.])
        pixels = project(xyz, rvec, translation, camera)
        mesh = pixels.reshape(cols + 1, rows + 1, 2)
        photo = np.full((height, width, 3), (70, 74, 66), dtype=np.uint8)
        for line in (*mesh, *mesh.transpose(1, 0, 2)):
            cv2.line(photo, tuple(np.rint(line[0]).astype(int)),
                     tuple(np.rint(line[-1]).astype(int)), (150, 164, 155), 2, cv2.LINE_AA)
        approximate = mesh[[0, -1, -1, 0], [0, 0, -1, -1]] + [[1, -2], [-2, 1], [2, 2], [0, 1]]
        grids.append(detect_grid(photo, {"name": "test", "grid_cells": [cols, rows],
                                         "grid_seed_px": approximate,
                                         "grid_exclude_px": [320, 200, 385, 280]}))
    estimated, poses = calibrate(grids, (width, height), pitch)
    assert abs(estimated[0, 0] - camera[0, 0]) < 12
    rotation, position, _, holdout, error = poses[-1]
    assert np.sqrt(np.mean(error[holdout] ** 2)) < 0.8
    xyz = [[40, -35, 0], [90, -50, 42]]
    expected = project(xyz, rvec, translation, camera)
    actual = project(xyz, cv2.Rodrigues(rotation)[0], position, estimated)
    assert np.max(np.abs(actual - expected)) < 2.5
    np.testing.assert_allclose(mat_points(actual[:1], estimated, rotation, position),
                               [[40, -35]], atol=1e-8)


def test_perspective_raster_visible_ids_and_optical_depth():
    triangle = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    near = triangle.copy().apply_translation([0, 0, -4])
    camera = np.array([[40., 0, 32], [0, 40, 32], [0, 0, 1]])
    _, ids, depths = render({"pump": near, "body": triangle}, camera,
                            np.eye(3), np.array([0, 0, 10]), (64, 64))
    assert ids[32, 32] == 2
    assert depths[32, 32] == 6
    assert ids[32, 37] == 0
    assert np.isinf(depths[0, 0])
    mask = ids != 0
    assert compare(mask, mask) == {"iou": 1, "chamfer_px": 0}
    assert compare(np.roll(mask, 2, axis=1), mask)["iou"] < 1
