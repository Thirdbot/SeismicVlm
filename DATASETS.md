# Datasets — where every file goes

Only **two things are manual** (the SFM encoder and, if you want it, Smeaheia). Everything else
**downloads and materializes itself** the first time you run a stage. Don't pre-create the auto folders —
the loaders make them; a hand-made folder with the wrong name just hides the data from the loader.

## The full tree (after setup)

```
hybrid/checkpoints/
  SFM-Base-512.pth            ← YOU PLACE · required encoder (hard error without it) · ~330 MB
  stage1_*/  reader.pt  stage2_grounding.pt  stage3_answer.pt  stage3_narrator.pt   ← AUTO (training writes them)

data/synthetic/                                   ← AUTO · HF thirdExec/synthetic-seismic-vlm
  multimodal_multi_image_dataset.csv                (= the SYNTH_CSV default)
  images/synth_XXXXX_0.png   masks/synth_XXXXX_N.png

data/real_data/
  thebe/                                          ← AUTO · Harvard Dataverse (streamed by chunk)
    raw/  images/  masks/  thebe.csv
  cracks/                                         ← AUTO · HF gOLIVES/CRACKS
    images/  masks/  cracks.csv
  # --- Smeaheia: YOU PLACE these, then it builds real_field_cube.csv itself ---
  smeaheia_3d/Seismic_3D_Surveys/data/GN1101_Scaled(Realized)   ← the GN1101 3-D cube SEG-Y
  raw/fault_sticks.zip                                          ← fault sticks (fault_Sticks_GN1101_2012)
  horizons/<name>.shp                                           ← POINTZ horizon shapefiles → THROW
  cube_panels/  cube_masks/  real_field_cube.csv               ← AUTO (built from the three above)
```

## Per-dataset

| dataset | get it | placement | download cap | panels built |
|---|---|---|---|---|
| **Synthetic** | AUTO — HF [`thirdExec/synthetic-seismic-vlm`](https://huggingface.co/datasets/thirdExec/synthetic-seismic-vlm). Own set: `SYNTH_CSV=/path`. | `data/synthetic/` | full | all |
| **CRACKS** | AUTO — HF [`gOLIVES/CRACKS`](https://huggingface.co/datasets/gOLIVES/CRACKS) | `data/real_data/cracks/` | full | `REAL_CAP` (10 000) |
| **Thebe** | AUTO — Dataverse [DOI 10.7910/DVN/YBYGBK](https://doi.org/10.7910/DVN/YBYGBK) | `data/real_data/thebe/` | ⚠ `N_CHUNKS=2` (~3 GB, ~200 xlines). **`N_CHUNKS=18` = full ~30 GB** | `REAL_CAP` (100 000) |
| **Smeaheia** | MANUAL — [co2datashare.org/dataset/smeaheia-dataset](https://co2datashare.org/dataset/smeaheia-dataset) (© Equinor & Gassnova — attribution required) | see tree above · details in [`data/real_data/README.md`](data/real_data/README.md) | — | all |

## Build commands

```bash
python -m hybrid.run_train                       # triggers synthetic auto-download + trains
python -m hybrid.data.smeaheia.build_from_cube   # (optional) build Smeaheia now; else scenes() builds on first use
# Thebe / CRACKS build lazily the first time a run touches them
```

## Notes / gotchas

- **SFM encoder has no auto-download** — you must place `hybrid/checkpoints/SFM-Base-512.pth` yourself
  (Seismic Foundation Model, ViT-B/16 @512). *(Maintainer: add the exact source URL here.)*
- **HuggingFace downloads failing on a fresh machine?** The stack is new (`hf-hub 1.22`); the Xet backend is
  the usual culprit. Try **`export HF_HUB_DISABLE_XET=1`** and retry. CLI is **`hf`** (`hf auth login`), *not*
  the removed `huggingface-cli`. See `SETUP.md §3c`.
- **Thebe `np.load: No data left in file`** = a truncated chunk cache. The loader now validates + re-fetches;
  if it persists, `rm -rf data/real_data/thebe/raw/` and re-run.
- **Smeaheia `throw 0/N`** = the `horizons/*.shp` files are missing. Masks + dip still work; add the horizons for throw.
- **Nothing here is in git** (all gitignored) except each dataset's small README — a fork downloads/places fresh.
```
