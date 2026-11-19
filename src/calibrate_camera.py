"""Fit a shared pinhole camera from indexed intersections of a square mat grid."""

import cv2
import numpy as np


def warp_points(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), matrix)[:, 0]


def detect_grid(image, view):
    """Refine four indexed seed corners with both grid line families in the JPG."""
    cols, rows = view["grid_cells"]
    corners = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    homography = cv2.getPerspectiveTransform(corners, np.float32(view["grid_seed_px"]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    logical = np.float32([(x, y) for y in range(1, rows) for x in range(1, cols)])
    left, top, right, bottom = view["grid_exclude_px"]
    # Sample the bright lines on both sides of each intersection, away from its crossing.
    offsets = np.arange(-0.44, 0.45, 0.02)
    arm = (abs(offsets) > 0.16) & (abs(offsets) < 0.42)
    center = abs(offsets) < 0.14
    indices = np.flatnonzero(center)
    patch = np.stack(np.meshgrid(offsets, offsets), axis=-1)
    for _ in range(3):
        predicted = warp_points(logical, homography)
        use = ~((predicted[:, 0] > left) & (predicted[:, 0] < right)
                & (predicted[:, 1] > top) & (predicted[:, 1] < bottom))
        lattice = logical[use]
        points = lattice[:, None, None, :] + patch
        mapped = warp_points(points, homography).reshape(points.shape).astype(np.float32)
        sample_shape = (-1, len(offsets))
        samples = cv2.remap(gray, mapped[..., 0].reshape(sample_shape),
                            mapped[..., 1].reshape(sample_shape), cv2.INTER_LINEAR)
        samples = samples.reshape(points.shape[:-1])
        profiles = np.stack([np.median(samples[:, arm, :], axis=1),
                             np.median(samples[:, :, arm], axis=2)], axis=1)
        peak = indices[np.argmax(profiles[:, :, indices], axis=2)]
        axes = np.arange(2)[None, :]
        index = np.arange(len(lattice))[:, None]
        height = profiles[index, axes, peak]
        supported = (height - np.median(profiles[:, :, arm], axis=2) > 12).all(axis=1)
        supported &= ((peak > indices[0]) & (peak < indices[-1])).all(axis=1)
        prev = profiles[index, axes, peak - 1]
        next_ = profiles[index, axes, peak + 1]
        curvature = prev - 2 * height + next_
        correction = np.divide(0.5 * (prev - next_), curvature,
                               out=np.zeros_like(height), where=curvature < -0.01)
        pixels = warp_points(lattice + offsets[peak] + correction * 0.02, homography)
        lattice, pixels = lattice[supported], pixels[supported]
        if len(lattice) < 24:
            raise ValueError(f"{view['name']}: insufficient supported grid intersections")
        homography, inliers = cv2.findHomography(lattice, pixels, cv2.RANSAC, 2.5)
        if homography is None or inliers.sum() < 24:
            raise ValueError(f"{view['name']}: no consistent grid homography")
    valid = inliers.ravel().astype(bool)
    return lattice[valid], pixels[valid]


def project(points, rotation, translation, camera):
    pixels, _ = cv2.projectPoints(np.asarray(points, dtype=float).reshape(-1, 3),
                                 rotation, translation, camera, None)
    return pixels[:, 0]


def calibrate(grids, size, pitch):
    """Square pixels, principal point fixed at image center, zero distortion."""
    world = [np.column_stack([cells[:, 0] * pitch, -cells[:, 1] * pitch,
                              np.zeros(len(cells))]).astype(np.float32) for cells, _ in grids]
    observed = [pixels.astype(np.float32) for _, pixels in grids]
    holdout = [(cells[:, 0] + 2 * cells[:, 1]) % 5 == 0 for cells, _ in grids]
    width, height = size
    matrix = np.array([[width * 1.3, 0, (width - 1) / 2],
                       [0, width * 1.3, (height - 1) / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    _, matrix, _, rotations, translations = cv2.calibrateCamera(
        [x[~hold] for x, hold in zip(world, holdout)],
        [x[~hold] for x, hold in zip(observed, holdout)],
        (width, height), matrix, None, flags=flags)
    if not width * 0.35 < matrix[0, 0] < width * 4:
        raise ValueError("Focal length is poorly constrained by these views")
    poses = []
    for xyz, pixels, hold, rvec, tvec in zip(world, observed, holdout, rotations, translations):
        fitted = project(xyz, rvec, tvec, matrix)
        errors = np.linalg.norm(pixels - fitted, axis=1)
        if hold.sum() < 5 or np.sqrt(np.mean(errors[hold] ** 2)) > 4:
            raise ValueError("Insufficient grid coverage or held-out grid RMS exceeds 4 px")
        rotation, _ = cv2.Rodrigues(rvec)
        if (-rotation.T @ tvec)[2, 0] <= 0:
            raise ValueError("Camera is below the mat. Check grid corner order")
        poses.append((rotation, tvec.ravel(), fitted, hold, errors))
    return matrix, poses


def mat_points(pixels, camera, rotation, translation):
    """Back-project image pixels onto the plane z=0 in metric mat coordinates."""
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(camera).T @ rotation
    center = -rotation.T @ translation
    distance = -center[2] / rays[:, 2]
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError("Base-edge points do not meet the mat in front of the camera")
    return (center + distance[:, None] * rays)[:, :2]
