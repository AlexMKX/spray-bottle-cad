#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python src/model.py --out out
python src/render.py --out out
python src/compare.py --photos refs/photos --out out
python -m pytest -q
