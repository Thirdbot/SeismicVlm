#!/usr/bin/env bash
# Penobscot 3D (offshore Nova Scotia) — seismic + 7 interpreted horizons + labels. License: CC BY-SA.
# Source: Baroni et al. 2019 "Penobscot Dataset", Zenodo record 3924682. Single HDF5, ~2.27 GB.
#   https://zenodo.org/records/3924682
# NOTE: this .h5 = seismic + horizons + facies-style labels. FAULT sticks are best taken from the dGB
#       OpendTect/OSR project (terranubis.com/osr) and exported — supplement after this pulls the cube.
set -euo pipefail
DEST="${1:-datasets/penobscot}"; mkdir -p "$DEST"
echo "[penobscot] -> $DEST/dataset.h5 (~2.27 GB)"
curl -L -C - --retry 3 -o "$DEST/dataset.h5" "https://zenodo.org/records/3924682/files/dataset.h5?download=1"
echo "[penobscot] done: $(du -h "$DEST/dataset.h5" | cut -f1)"
