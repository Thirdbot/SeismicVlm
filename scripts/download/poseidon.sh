#!/usr/bin/env bash
# Poseidon 3D (Browse Basin, offshore NW Australia) — imaged 3D seismic in MDIO format. License: CC BY 4.0.
# Source: AWS Open Data, PUBLIC bucket (no credentials). https://registry.opendata.aws/tgs-opendata-poseidon/
#   s3://tgs-opendata-poseidon/{near,mid,far,full_stack_agc}.mdio/   (each = a directory of zarr chunks)
# LARGE (~2900 km^2). This pulls ONLY one stack (default full_stack_agc). CHECK the size BEFORE syncing:
#   aws s3 ls --no-sign-request --summarize --human-readable --recursive \
#       s3://tgs-opendata-poseidon/full_stack_agc.mdio/ | tail -3
set -euo pipefail
DEST="${1:-datasets/poseidon}"; STACK="${2:-full_stack_agc.mdio}"; mkdir -p "$DEST/$STACK"
echo "[poseidon] sync $STACK  (Ctrl-C if the size check above was too big for your disk)"
aws s3 sync --no-sign-request "s3://tgs-opendata-poseidon/$STACK/" "$DEST/$STACK/"
echo "[poseidon] done: $(du -sh "$DEST/$STACK" | cut -f1)"
