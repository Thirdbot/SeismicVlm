# Real-field data (Smeaheia + Thebe)

Real 2-D/3-D seismic used to **benchmark and fine-tune the VISION part** of the model
(SFM encoder → DETR reader → per-instance masks + attributes). Narration is **frozen** here —
this is a syn→real transfer for detection/masking/attributes only.

This folder holds the two datasets the loader expects on disk here:
- **Smeaheia** — you download + place it (below).
- **Thebe** — **auto-downloads** into `data/real_data/thebe/` (nothing to place).

CRACKS and the synthetic set auto-download from HuggingFace and live elsewhere — see the repo-root `SETUP.md §3`.

---

## Smeaheia — © Equinor & Gassnova (attribution REQUIRED)

Smeaheia Dataset, via CO2DataShare.
- Dataset: https://co2datashare.org/dataset/smeaheia-dataset
- License: https://co2datashare.org/view/license/26af9426-203f-4993-9d41-2e1bf191ceaf
  (modified CC BY 4.0 — research/ML/derivatives/redistribution allowed **with attribution to Equinor & Gassnova**;
  **may not sell** the material.)

Any published result or redistributed derivative MUST credit **Equinor and Gassnova** and link the license above.

### Layout — place the downloaded files here

The **default** build slices the GN1101 **3-D cube** perpendicular to each fault's local strike
(`hybrid/data/smeaheia/build_from_cube.py`) — masks that actually sit on the faults (supersedes the legacy
2-D-line projection). Place:

```
data/real_data/
  # --- YOU PLACE (the cube build reads exactly these three) ---
  smeaheia_3d/Seismic_3D_Surveys/data/GN1101_Scaled(Realized)   # the GN1101 3-D cube SEG-Y  (the CUBE)
  raw/fault_sticks.zip                                          # fault sticks (filtered to fault_Sticks_GN1101_2012)
  horizons/   <name>.shp                                        # POINTZ horizon shapefiles  →  THROW
  # --- BUILT automatically ---
  smeaheia_3d/…/gn1101_geom.npz                                 # cached cube geometry
  cube_panels/   cube_masks/                                    # rendered fault panels + masks
  real_field_cube.csv                                           # the CSV the loader reads
```
> `segy/` and `render/` are used **only** by the legacy `SMEAHEIA_LINES=1` 2-D-line path — the default **cube**
> build does not touch them.

### What each input is GT for

| input | → GT for |
|---|---|
| GN1101 cube + fault sticks | the images · fault **masks** · bbox · **dip** |
| **horizons (`*.shp`)** | **throw** (horizon offset across a fault) |

> ⚠ **Throw needs the horizons.** Cube + sticks alone give masks + dip (e.g. `215 fault / 215 background`) but
> **`throw 0/N`** — the build now prints `WARNING: NO HORIZONS loaded`. Add the `horizons/*.shp` files to get throw;
> masks and dip are unaffected either way. Real is **fault-only** (no closure/area/count on these surveys).

### Build

Auto-builds on the first `scenes()` call, or run it explicitly:
```bash
python -m hybrid.data.smeaheia.build_from_cube          # default → data/real_data/real_field_cube.csv (cube GT)
SMEAHEIA_LINES=1 python -m hybrid.data.smeaheia.build_csv   # legacy 2-D-line projection (comparison only)
```
Success prints `CUBE_BUILD_DONE … N panels (X fault / Y background) · throw M/N`. `throw 0/N` ⇒ horizons missing.

---

## Thebe — auto-downloads (Harvard Dataverse, [DOI 10.7910/DVN/YBYGBK](https://doi.org/10.7910/DVN/YBYGBK))

Two ways in — auto-download, or (if the Dataverse API is flaky for you) manual placement:

**A. Auto** — `hybrid.data.thebe.scenes()` streams chunks into `data/real_data/thebe/` on first use.
- **`N_CHUNKS=2`** by default (~200 of 1803 crosslines, ~3 GB). Set **`N_CHUNKS=18`** for the full ~30 GB volume.
- `REAL_CAP` caps how many panels are **built/encoded** (default 100 000) — separate from the download.
- The access API returns **HTTP 202 while it stages a file from cold storage**; the loader polls with backoff
  until it serves bytes (tune `THEBE_STAGE_TRIES` / `THEBE_STAGE_WAIT`). It caches each file once staged.

**B. Manual (reliable)** — if the API keeps 202-ing or erroring, download the files from the dataset **page**
([DOI 10.7910/DVN/YBYGBK](https://doi.org/10.7910/DVN/YBYGBK) → pick the version, e.g. **V1**) and drop the
fault `.npy` + seismic `.npz` files into `data/real_data/thebe/raw/`. The build **discovers and pairs them by
filename** (and prints `local pair: <fault> ↔ <seismic>` for each) — no API call. Then:
```bash
python -m hybrid.data.thebe.build_csv        # uses raw/ if files are present; else auto-downloads
```
- **If you see `np.load: No data left in file`**, a chunk was truncated/empty — delete `data/real_data/thebe/raw/` and re-run.

---

## Use it (current commands)

```bash
CKPT=hybrid/checkpoints/reader.pt DATASETS=smeaheia python -m hybrid.eval.benchmark   # benchmark on real
ACTIVE_CLASSES=fault SURVEY=smeaheia STEPS=1000 scripts/alone.sh                       # fine-tune one survey
ACTIVE_CLASSES=fault WEIGHTS=thebe:4,cracks:3,smeaheia:3 scripts/joint.sh              # weighted round-robin joint
```

Splits are deterministic (seed 42) and **leak-safe**: Smeaheia by source-line/fault, Thebe/CRACKS contiguous by
crossline/section — a fault's overlapping windows never straddle train/test.
