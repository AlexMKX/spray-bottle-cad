# ISE MOUNT Pump Spray 80 ml — parametric CAD reconstruction

Source-based CAD reconstruction of the 80 ml spray bottle from seven user-provided photos. **Visual prototype, not calibrated mechanical geometry**.

![Illustrative project sketch](docs/preview.svg)

The sketch is illustrative; **real CAD renders are created by GitHub Actions** and available as build artifacts.

## Build

Requires Python 3.11+, OpenSCAD and Xvfb (Linux):

```bash
sudo apt-get install -y openscad xvfb fonts-dejavu-core
python -m pip install -r requirements.txt
bash scripts/build.sh
```

Outputs: `out/body.stl`, `out/pump.stl`, `out/silhouette.stl`, `out/spray-bottle.step`, six PNG camera renders and photo-silhouette metrics `out/comparison.json`. GitHub Actions repeats this build on push.

## Source structure

- `src/model.py`: parametric CadQuery body, through-handle, spray head, sticker and letters (all dimensions in mm, estimates).
- `src/render.py`: six headless OpenSCAD camera previews.
- `src/compare.py`: GrabCut on original photos when available; otherwise compare against saved reference silhouette contours.
- `refs/silhouettes.json`: contours **extracted from the user's actual photographs**, stored as text to make CI reproducible without multi-megabyte photos.
- `src/grid_scale.py`: experimental grid pitch estimation, not a metric camera calibration.
- `tests/`: 4 geometry/raster comparison smoke tests.

Original high-resolution images are in the separately delivered project archive, and can be placed under `refs/photos/`. They are deliberately not committed to this **public** GitHub repository. The vector silhouettes preserve privacy and provide lightweight shape reference for CI.

## Photo ↔ render QA

The pipeline compares normalized silhouettes with IoU (higher is better) and symmetric contour Chamfer distance (lower is better). Reference contours were obtained using OpenCV GrabCut from three user-provided photos, then simplified at 2 pixels tolerance. With actual photos present, the pipeline recomputes segmentation from them instead.

Preliminary full-photo baseline: front **0.713**, rear **0.644**, side **0.843** IoU. These are uncalibrated perspectives; shape similarity is not dimensional accuracy. **No strict IoU quality gate** until the camera pose is calibrated.

For an accurate comparison: measure some bottle dimensions, place ChArUco or AprilTag markers by it, calibrate camera intrinsics/pose, render with matching projection, optimize camera/object pose, evaluate keypoint reprojection errors and silhouette differences in a known scale.

## Printability and safety

Main body and pump are separate visual solids, not a usable threaded cap or sealed reservoir. Do not use as a fluid container for alcohol/fuel or print a fitted part without physical measurements.
