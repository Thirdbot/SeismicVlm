#!/usr/bin/env bash
# Volve (central Norwegian North Sea) — FULL E&P dataset incl. seismic + geomodel (faults/horizons/closure).
# License: EQUINOR OPEN DATA LICENCE — you MUST accept it on the portal; this cannot be automated here.
# Size: ~40k files / ~TB total — does NOT fit ~84 GB free. Pull ONLY the seismic subfolder, selectively.
#
# Steps (manual):
#   1) Portal: https://www.equinor.com/energy/volve-data-sharing  ->  "Go to the Volve Dataset, data.equinor.com"
#   2) Accept the licence; obtain the container SAS URL from data.equinor.com (Azure Blob Storage).
#   3) Install azcopy:  https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-v10
#   4) azcopy copy "<CONTAINER_SAS_URL>/Volve/Seismic/ST0202R08/..." "datasets/volve/" --recursive
#
# The reservoir geomodel (faults/horizons/closure) is in Petrel/RMS formats — needs OpendTect/Petrel to
# export to a parseable form. Confirm that export is possible BEFORE committing to Volve as the closure source.
echo "Volve is NOT auto-runnable: Equinor licence acceptance + Azure/azcopy + ~TB size (free disk ~84 GB)."
echo "Follow the manual steps in the header of this script."
exit 1
