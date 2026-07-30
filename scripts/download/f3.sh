#!/usr/bin/env bash
# F3 Netherlands (Dutch North Sea) — seismic volume 401x701x255 + facies labels (Alaudah 2019 benchmark).
# License: CC BY-SA. Source: Zenodo record 3755060, data.zip, ~1.05 GB.
#   https://zenodo.org/record/3755060
# NOTE: facies + implied horizons here; ONLAP = derive stratal terminations from these. The full 384 km^2
#       SEG-Y amplitude cube is on the dGB OSR (terranubis.com/osr, OpendTect format) if you need more.
set -euo pipefail
DEST="${1:-datasets/f3}"; mkdir -p "$DEST"
echo "[f3] -> $DEST/data.zip (~1.05 GB)"
curl -L -C - --retry 3 -o "$DEST/data.zip" "https://zenodo.org/record/3755060/files/data.zip?download=1"
if command -v unzip >/dev/null 2>&1; then ( cd "$DEST" && unzip -n -q data.zip ); else echo "[f3] install 'unzip' to extract"; fi
echo "[f3] done: $(du -sh "$DEST" | cut -f1)"
