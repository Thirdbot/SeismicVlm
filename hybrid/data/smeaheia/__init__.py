"""Smeaheia real-field dataset (Equinor/Gassnova, CO2DataShare).

`build_csv.py` turns the 2D SEG-Y lines into the UNIFIED CSV (fault panels where 3D sticks project +
negative panels elsewhere); `segy.py` holds the SEG-Y/stick/horizon machinery. This exposes `scenes()`
= the same loader (data.loader) over that CSV, so stages/eval consume real exactly like synthetic.

Future real-field datasets (F3, GN1101, Thebe, …) follow this pattern: a folder with its own build_csv
that emits the same schema.
"""
import os

from hybrid.data.smeaheia.build_csv import real_csv_scenes, build_real_csv, CSV_OUT

CUBE_CSV = "data/real_data/real_field_cube.csv"   # GN1101-3D-cube-derived GT (correct masks, sliced ⊥ strike)


def scenes(test_frac=0.25, neg_per_pos=3, seed=42):
    """Real Smeaheia scenes (build the CSV if missing, then load+split). Returns (all, train, test).
    DEFAULT source = the GN1101 3-D-cube GT (build_from_cube: fault surfaces sliced from the volume with
    depth→time — masks that actually sit on the faults). Set SMEAHEIA_LINES=1 for the legacy 2-D-line
    stick-projection CSV (depth-as-time misplacement — kept for comparison only)."""
    if os.environ.get("SMEAHEIA_LINES") == "1":
        if not os.path.exists(CSV_OUT):
            build_real_csv()
        return real_csv_scenes(test_frac=test_frac, neg_per_pos=neg_per_pos, seed=seed, csv=CSV_OUT)
    if not os.path.exists(CUBE_CSV):
        from hybrid.data.smeaheia.build_from_cube import build
        build()
    return real_csv_scenes(test_frac=test_frac, neg_per_pos=neg_per_pos, seed=seed, csv=CUBE_CSV)
