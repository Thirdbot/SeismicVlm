"""Smeaheia real-field dataset (Equinor/Gassnova, CO2DataShare).

`build_csv.py` turns the 2D SEG-Y lines into the UNIFIED CSV (fault panels where 3D sticks project +
negative panels elsewhere); `segy.py` holds the SEG-Y/stick/horizon machinery. This exposes `scenes()`
= the same loader (data.loader) over that CSV, so stages/eval consume real exactly like synthetic.

Future real-field datasets (F3, GN1101, Thebe, …) follow this pattern: a folder with its own build_csv
that emits the same schema.
"""
import os

from hybrid.data.smeaheia.build_csv import real_csv_scenes, build_real_csv, CSV_OUT


def scenes(test_frac=0.25, neg_per_pos=3, seed=42):
    """Real Smeaheia scenes (build the CSV if missing, then load+split). Returns (all, train, test),
    positives + negatives, CPU-offloaded smaps (big panels)."""
    if not os.path.exists(CSV_OUT):
        build_real_csv()
    return real_csv_scenes(test_frac=test_frac, neg_per_pos=neg_per_pos, seed=seed)
