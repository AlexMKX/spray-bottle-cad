"""Register upright bottle views, render in their cameras, and report unscaled errors."""

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_camera import calibrate, detect_grid, mat_points, project
from model import DIM
from render_perspective import load_parts, render

ROOT = Path(__file__).resolve().parents[1]


def bottle_pose(view, camera, mat_rotation, mat_translation):
    """Use the projected bottom edge for XY and yaw, with the CAD base at z=0."""
    pixels = np.asarray(view["base_edge_px"], dtype=float)
    edge = mat_points(pixels, camera, mat_rotation, mat_translation)
    direction = edge[1] - edge[0]
    span = np.linalg.norm(direction)
    if span < 1:
        raise ValueError("Degenerate base edge")
    # Local directions of the annotated edge, ordered image left to right.
    angle, face, along = {
        "-y": (0, [0, -1], "width"),
        "+y": (np.pi, [0, 1], "width"),
        "-x": (-np.pi / 2, [-1, 0], "depth"),
        "+x": (np.pi / 2, [1, 0], "depth"),
    }[view["base_edge"]]
    yaw = np.arctan2(direction[1], direction[0]) - angle
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    radius = DIM["body_edge_radius"]
    # A fully filleted box contacts the plane on its inset flat footprint.
    offset = np.array(face) * (np.array([DIM["body_width"], DIM["body_depth"]]) / 2 - radius)
    origin = np.r_[edge.mean(axis=0) - rotation[:2, :2] @ offset, 0]
    return (
        mat_rotation @ rotation,
        mat_rotation @ origin + mat_translation,
        {"base_xy_mm": origin[:2].tolist(), "yaw_deg": float((np.degrees(yaw) + 180) % 360 - 180),
         "observed_edge_span_mm": float(span),
         "cad_flat_edge_span_mm": DIM[f"body_{along}"] - 2 * radius},
    )


def photo_mask(image, roi, opening_roi):
    left, top, right, bottom = roi
    margin = 26
    x, y = max(0, left - margin), max(0, top - margin)
    crop = image[y:bottom + margin, x:right + margin]
    rect = (left - x, top - y, right - left, bottom - top)
    levels = np.zeros(crop.shape[:2], np.uint8)
    cv2.setRNGSeed(0)
    cv2.grabCut(crop, levels, rect, np.zeros((1, 65)), np.zeros((1, 65)),
                6, cv2.GC_INIT_WITH_RECT)
    fg = (levels == cv2.GC_FGD) | (levels == cv2.GC_PR_FGD)
    # Green mat seen through the opening may be joined to the foreground by GrabCut.
    b, g, r = cv2.split(crop.astype(np.int16))
    fg &= ~((g > r + 6) & (g >= b))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n < 2:
        raise ValueError("Photo foreground segmentation is empty")
    mask = np.zeros(image.shape[:2], dtype=bool)
    component = (labels == 1 + np.argmax(stats[1:, 4])).astype(np.uint8)
    outer, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                offset=(x, y))
    filled = cv2.drawContours(mask.astype(np.uint8), outer, -1, 1, cv2.FILLED).astype(bool)
    mask[y:y + crop.shape[0], x:x + crop.shape[1]] = component != 0
    if opening_roi is not None:
        left, top, right, bottom = opening_roi
        opening = np.s_[top:bottom, left:right]
        hole = (filled[opening] & ~mask[opening]).astype(np.uint8)
        # Join mat regions separated by a thin grid line inside the known opening ROI.
        hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, regions, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
        if n > 1:
            filled[opening] &= regions != 1 + np.argmax(stats[1:, 4])
    return filled


def compare(observed, predicted):
    if not observed.any() or not predicted.any():
        raise ValueError("Empty object mask")
    if any(predicted[edge].any() for edge in (0, -1)) or predicted[:, [0, -1]].any():
        raise ValueError("Perspective render touches the frame edge")
    a, b = (mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            for mask in (observed, predicted))
    ad, bd = (cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2,
                                   cv2.DIST_MASK_PRECISE) for boundary in (a, b))
    return {"iou": float(np.count_nonzero(observed & predicted) / np.count_nonzero(observed | predicted)),
            "chamfer_px": float((ad[b].mean() + bd[a].mean()) / 2)}


def diagnostics(directory, photo, color, observed, ids, pixels, fitted, holdout, pump):
    predicted = ids != 0
    overlay = photo.copy()
    overlay[predicted] = cv2.addWeighted(photo, 0.48, color, 0.52, 0)[predicted]
    for mask, line in [(observed, (30, 70, 255)), (predicted, (255, 220, 30))]:
        edges, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, edges, -1, line, 2, cv2.LINE_AA)
    for point, rgb in zip(pump, [(30, 70, 255), (255, 220, 30)]):
        cv2.drawMarker(overlay, tuple(np.rint(point).astype(int)), rgb, cv2.MARKER_CROSS, 18, 2)
    diff = np.full_like(photo, 242)
    diff[observed & predicted] = (155, 190, 159)
    diff[observed & ~predicted] = (30, 70, 255)
    diff[predicted & ~observed] = (255, 220, 30)
    grid = photo.copy()
    for actual, expected, held in zip(pixels, fitted, holdout):
        observed_pt, fitted_pt = [tuple(np.rint(point).astype(int)) for point in (actual, expected)]
        cv2.circle(grid, observed_pt, 4 if held else 2, (0, 220, 255) if held else (0, 255, 100), 1)
        cv2.drawMarker(grid, fitted_pt, (240, 130, 30), cv2.MARKER_CROSS, 7)
    y, x = np.where(observed | predicted)
    viewport = np.s_[max(0, y.min() - 22):y.max() + 23, max(0, x.min() - 22):x.max() + 23]
    for name, frame in (("photo", photo), ("render", color), ("overlay", overlay), ("difference", diff)):
        if not cv2.imwrite(str(directory / f"{name}.png"), frame[viewport]):
            raise OSError(f"Could not write {name}.png")
    cv2.imwrite(str(directory / "grid.png"), grid)
    cv2.imwrite(str(directory / "photo-mask.png"), observed.astype(np.uint8) * 255)
    cv2.imwrite(str(directory / "model-part-ids.png"), ids)


def run(args):
    config = json.loads(args.manifest.read_text())
    views = config["views"]
    pitch = args.grid_pitch_mm if args.grid_pitch_mm is not None else config["grid_pitch_mm"]
    if not np.isfinite(pitch) or pitch <= 0 or len(views) < 3:
        raise ValueError("A positive grid pitch and at least three views are required")
    size = config["image_size"]
    names = [v["name"] for v in views]
    if len(names) != len(set(names)) or any(not name.isascii() or not name.replace("-", "").isalnum() for name in names):
        raise ValueError("View names must be distinct ASCII letters, digits or hyphens")
    photos = []
    for view in views:
        path = args.photos / view["photo"]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Original photo required: {path}")
        if list(image.shape[:2][::-1]) != size:
            raise ValueError(f"{path}: expected {size[0]}×{size[1]} pixels")
        photos.append(image)
    grids = [detect_grid(photo, view) for photo, view in zip(photos, views)]
    camera, poses = calibrate(grids, size, pitch)
    parts = load_parts(args.cad)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"intrinsics": camera.tolist(), "grid_pitch_mm": pitch,
              "grid_pitch_measured": args.grid_pitch_mm is not None or config["grid_pitch_measured"],
              "camera_model": "shared focal length, square pixels, centered principal point, no distortion",
              "object_pose": "upright, base on mat, yaw and XY from annotated base edge",
              "views": []}
    for view, photo, (_, pixels), (mat_r, mat_t, fitted, hold, err) in zip(views, photos, grids, poses):
        rotation, translation, pose = bottle_pose(view, camera, mat_r, mat_t)
        color, ids, depth = render(parts, camera, rotation, translation, size)
        observed = photo_mask(photo, view["object_roi_px"], view.get("opening_roi_px"))
        metrics = compare(observed, ids != 0)
        pump_observed = np.asarray(view["pump_top_px"])
        pump_model_z = parts["pump"].bounds[1, 2]
        pump_top = project([[DIM["nozzle_center_x"], 0, pump_model_z]],
                           cv2.Rodrigues(rotation)[0], translation, camera)[0]
        directory = args.out / view["name"]
        directory.mkdir(exist_ok=True)
        diagnostics(directory, photo, color, observed, ids, pixels, fitted, hold,
                    (pump_observed, pump_top))
        # Optical-axis depth in mm. Infinite values denote background.
        np.save(directory / "depth-mm.npy", depth)
        result = {"view": view["name"], "grid_intersections": len(pixels),
                  "grid_holdout_intersections": int(hold.sum()),
                  "grid_holdout_rms_px": float(np.sqrt(np.mean(err[hold] ** 2))),
                  **metrics, "pump_top_error_px": float(np.linalg.norm(pump_observed - pump_top)),
                  "mat_to_camera": {"rotation": mat_r.tolist(), "translation_mm": mat_t.tolist()},
                  "object_to_camera": {"rotation": rotation.tolist(), "translation_mm": translation.tolist()},
                  "pose": pose}
        report["views"].append(result)
        print(f"{view['name']}: grid RMS {result['grid_holdout_rms_px']:.2f} px, "
              f"IoU {metrics['iou']:.3f}, Chamfer {metrics['chamfer_px']:.1f} px, "
              f"pump top {result['pump_top_error_px']:.1f} px")
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_html(args.out, report)
    if args.min_iou is not None:
        if not report["grid_pitch_measured"]:
            raise ValueError("A measured grid pitch is required when enforcing --min-iou")
        failed = [v["view"] for v in report["views"] if v["iou"] < args.min_iou]
        if failed:
            raise ValueError(f"IoU below threshold for {', '.join(failed)}")


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def write_html(directory, report):
    pitch_status = "measured" if report["grid_pitch_measured"] else "assumed"
    content = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Perspective CAD comparison</title>",
        "<style>body{font:16px system-ui;margin:2rem;color:#222} section{margin-bottom:2rem}"
        ".panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;height:auto}figcaption{margin-top:.4rem}</style>",
        "<h1>Perspective CAD comparison</h1>",
        f"<p>Grid pitch: {report['grid_pitch_mm']:g} mm ({pitch_status}). "
        f"Focal length: {report['intrinsics'][0][0]:.1f} px. Centered pinhole model.</p>",
        "<p>Orange marks the photo contour; cyan marks CAD. Green marks overlap. "
        "Crosses locate the observed and CAD pump top centers. Pump top is independent of pose fitting. "
        "Grid RMS is measured on held-out intersections. Base contact and the mat pitch affect scale.</p>",
    ]
    for view in report["views"]:
        name = html.escape(view["view"])
        content.append(f"<section><h2>{name}</h2><p>IoU {view['iou']:.3f} · "
                       f"Chamfer {view['chamfer_px']:.1f} px · "
                       f"pump top {view['pump_top_error_px']:.1f} px · "
                       f"grid RMS {view['grid_holdout_rms_px']:.2f} px</p><div class='panels'>")
        for file, caption in (("photo", "Photo"), ("render", "CAD render"),
                              ("overlay", "Overlay"), ("difference", "Difference")):
            uri = image_uri(directory / view["view"] / f"{file}.png")
            content.append(f"<figure><img src='{uri}' alt='{caption}'>"
                           f"<figcaption>{caption}</figcaption></figure>")
        grid_uri = image_uri(directory / view["view"] / "grid.png")
        content.append(f"</div><details><summary>Reprojected grid</summary>"
                       f"<img src='{grid_uri}' alt='Detected and reprojected grid intersections'>"
                       "</details></section>")
    summary = html.escape(json.dumps(report, indent=2))
    content.append(f"<details><summary>Camera, pose and metrics JSON</summary>"
                   f"<pre>{summary}</pre></details></html>")
    (directory / "comparison.html").write_text("\n".join(content), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=Path, default=ROOT / "refs/photos")
    parser.add_argument("--manifest", type=Path, default=ROOT / "refs/calibration/upright.json")
    parser.add_argument("--cad", type=Path, default=ROOT / "out")
    parser.add_argument("--out", type=Path, default=ROOT / "out/visual")
    parser.add_argument("--grid-pitch-mm", type=float, help="Measured pitch, overriding manifest value")
    parser.add_argument("--min-iou", type=float, help="Optional minimum IoU for every view")
    args = parser.parse_args()
    if args.min_iou is not None and not 0 < args.min_iou <= 1:
        parser.error("--min-iou must be in (0, 1]")
    try:
        run(args)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f"Visual validation failed: {exc}\n")


if __name__ == "__main__":
    main()
