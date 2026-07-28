"""Schema gate — verify the dataset CSV's object_type / values.derive keys match the registry BEFORE
a (GPU-expensive) reader rebuild. Pure pandas + registry (no torch, no NCS encode) → runs in seconds.

Checks, against registry.CLASS_ID / OBJECT_DERIVED / SECTION_DERIVED:
  · every registry CLASS appears in the data (else that class is gated → never trained)
  · every OBJECT-derived key appears, AND on its klass (closure) — scope correctness
  · every SECTION-derived key appears (missing = gated-safe, just untrained)
  · data keys with no registry row are listed (informational — ignored by design)

A registry key absent from the data is SAFE (present-gating skips it), so this never blocks; it just
tells you what will actually train. Exit code 1 only if a registry CLASS is missing (the one case that
usually means a spelling/regen mismatch worth fixing before you spend the GPU).

Run:  python -m hybrid.eval.schema_check            (path auto-read from data/synthetic)
      CSV=/path/to/other.csv python -m hybrid.eval.schema_check
"""
import ast
import collections
import json
import os
import pathlib
import re
import sys

import pandas as pd

from hybrid.model.registry import CLASS_ID, OBJECT_DERIVED, SECTION_DERIVED


def _trainer_csv():
    """The CSV path the trainer actually uses — read from scenes.py WITHOUT importing it (that pulls the
    NCS encoder / transformers). Keeps this gate torch-free so it can run while the GPU is busy."""
    txt = pathlib.Path("hybrid/data/synthetic/__init__.py").read_text()
    m = re.search(r'^CSV\s*=\s*["\'](.+?)["\']', txt, re.M)
    return m.group(1) if m else None


def _parse(cell):
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str):
        return []
    try:
        return json.loads(cell)
    except Exception:
        try:
            return ast.literal_eval(cell)
        except Exception:
            return []


def main():
    csv = os.environ.get("CSV") or _trainer_csv()
    print(f"[schema] CSV {csv}", flush=True)
    if not csv or not os.path.exists(csv):
        print("[schema] CSV not found — set CSV=… or fix hybrid/data/synthetic/__init__.py:CSV", flush=True)
        sys.exit(2)

    df = pd.read_csv(csv)
    otype = collections.Counter()
    derive_by_type = collections.defaultdict(collections.Counter)   # object_type -> derive-key counts
    all_derive = collections.Counter()
    for cell in df["regions"].dropna():
        for reg in _parse(cell):
            if not isinstance(reg, dict):
                continue
            ot = reg.get("object_type")
            otype[ot] += 1
            for k in ((reg.get("values") or {}).get("derive") or {}):
                derive_by_type[ot][k] += 1
                all_derive[k] += 1

    print(f"[schema] object_types: {dict(otype)}", flush=True)

    # ---- classes ----
    missing_cls = [c for c in CLASS_ID if c not in otype]
    cls_line = "  ".join(f"{c}{'✓' if c in otype else '✗GATED'}" for c in CLASS_ID)
    print(f"[classes]         {cls_line}", flush=True)

    # ---- object-derived (must appear, and on their klass) ----
    obj_lines, obj_scope_warn = [], []
    for _i, key, _m, _kd, _l, klass in OBJECT_DERIVED:
        present = key in all_derive
        on_klass = derive_by_type.get(klass, {}).get(key, 0)
        tag = "✓" if present else "✗gated"
        if present and on_klass == 0:
            tag = "⚠SCOPE"                              # exists but not on the expected klass
            obj_scope_warn.append((key, klass))
        obj_lines.append(f"{key}{tag}")
    print(f"[object-derived]  {'  '.join(obj_lines)}   (expected on: closure)", flush=True)

    # ---- section-derived (missing = gated-safe) ----
    sec_lines = []
    for _i, key, _m, _kd, _l in SECTION_DERIVED:
        sec_lines.append(f"{key}{'✓' if key in all_derive else '✗gated'}")
    print(f"[section-derived] {'  '.join(sec_lines)}", flush=True)

    # ---- data keys with no registry row (informational) ----
    reg_keys = {k for _i, k, *_ in OBJECT_DERIVED} | {k for _i, k, *_ in SECTION_DERIVED}
    data_only = sorted(k for k in all_derive if k not in reg_keys)
    if data_only:
        print(f"[data-only keys]  {', '.join(data_only)}  (no registry row — ignored by design)", flush=True)

    # ---- verdict ----
    ok = not missing_cls and not obj_scope_warn
    if missing_cls:
        print(f"[VERDICT] ATTENTION — registry classes absent from data: {missing_cls} "
              f"(likely a spelling/regen mismatch; those classes won't train)", flush=True)
    if obj_scope_warn:
        print(f"[VERDICT] ATTENTION — object-derived keys not on their klass: {obj_scope_warn} "
              f"(scope mismatch — the object head won't see them)", flush=True)
    if ok:
        gated = ([c for c in CLASS_ID if c not in otype]
                 + [k for _i, k, *_ in OBJECT_DERIVED if k not in all_derive]
                 + [k for _i, k, *_ in SECTION_DERIVED if k not in all_derive])
        print(f"[VERDICT] READY — all registry classes + object-derived present. "
              f"Gated (safe, untrained): {gated or 'none'}", flush=True)
    print("SCHEMA_CHECK_DONE", flush=True)
    sys.exit(0 if not missing_cls else 1)


if __name__ == "__main__":
    main()
