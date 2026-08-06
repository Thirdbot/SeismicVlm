"""Assemble benchmark logs into the paper tables. Reads the machine-readable `[METRICS]` JSON lines that
`hybrid.eval.benchmark` emits (one per dataset+checkpoint, never pooled), groups by dataset, and prints:
  (1) the full per-checkpoint metric rows,
  (2) the {alone}-vs-{joint} complementarity table,
  (3) the config tradeoff (4:3:3 mask-best vs 8:1:1 attribute-best).

  python -m hybrid.eval.report                      # reads runs/bench_*.log
  python -m hybrid.eval.report runs/a.log runs/b.log

Honest by construction: Dice is the DEPLOY number (detect(), not teacher-forced); dip is shown only where
independently claimable (smeaheia); attributes carry their constant-predictor baseline.
"""
import glob
import json
import os
import sys

DATASETS = ["thebe", "cracks", "smeaheia", "synthetic"]


def role(ckpt):
    """Infer a checkpoint's role from its filename (for the comparison tables)."""
    c = ckpt.lower()
    if "alone" in c:
        return "alone"
    if "811" in c or "w811" in c:
        return "joint 8:1:1"
    if "w433" in c or "full" in c:
        return "joint 4:3:3"
    if "_rr" in c or "joint_rr" in c:
        return "joint 8:1:1 (capped)"
    return ckpt.replace("reader_", "").replace(".pt", "")


def load(paths):
    rows = []
    for p in paths:
        try:
            for line in open(p):
                i = line.find("[METRICS] ")
                if i >= 0:
                    rows.append(json.loads(line[i + 10:]))
        except OSError:
            print(f"[report] WARNING: cannot read {p}", flush=True)
    return rows


def f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "—"    # nan → —


def beats(mae, const):                                                          # ✓ if attribute beats its constant
    return " ✓" if (isinstance(mae, (int, float)) and isinstance(const, (int, float)) and mae == mae and mae < const) else ""


def main():
    paths = sys.argv[1:] or sorted(glob.glob("runs/bench_*.log"))
    if not paths:
        print("[report] no logs (pass paths or run scripts/benchmark.sh → runs/bench_*.log)", flush=True); return
    rows = load(paths)
    if not rows:
        print("[report] no [METRICS] lines found in the logs", flush=True); return
    idx = {(r["dataset"], role(r["ckpt"])): r for r in rows}                     # (dataset, role) → metrics
    roles = [rl for rl in ["alone", "joint 4:3:3", "joint 8:1:1", "joint 8:1:1 (capped)"]
             if any(k[1] == rl for k in idx)]
    dsets = [d for d in DATASETS if any(k[0] == d for k in idx)]

    print(f"# Benchmark report ({len(rows)} dataset-checkpoint rows from {len(paths)} logs)\n")

    print("## Full metrics (per dataset · per checkpoint · never pooled)\n")
    print("| dataset | checkpoint | n | Dice oracle/deploy | pixP/R/F1 | tol-F1 | detF1 | class | dip(const) | throw(const) |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for d in dsets:
        for r in [x for x in rows if x["dataset"] == d]:
            cls = f"{r['cls_hit']}/{r['cls_tot']}" if r["cls_tot"] else "—"
            dipc = "" if r.get("dip_claimable") else " (circ)"
            print(f"| {d} | {role(r['ckpt'])} | {r['n']} | {f(r['dice_oracle'])}/{f(r['dice_deploy'])} | "
                  f"{f(r['pixP'],2)}/{f(r['pixR'],2)}/{f(r['pixF1'],2)} | {f(r['tolf1'])} | {f(r['detF1'])} | {cls} | "
                  f"{f(r['dip'],2)}{dipc}({f(r['dip_const'],2)}){beats(r['dip'],r['dip_const'])} | "
                  f"{f(r['throw'],1)}({f(r['throw_const'],1)}){beats(r['throw'],r['throw_const'])} |")

    if "alone" in roles and any(rl.startswith("joint") for rl in roles):
        jr = "joint 4:3:3" if "joint 4:3:3" in roles else next(rl for rl in roles if rl.startswith("joint"))
        print(f"\n## Complementarity — {{alone}} vs {{{jr}}} (each survey on its OWN held-out)\n")
        print("| survey | Dice(deploy) alone→joint | detF1 alone→joint | dip alone→joint (claimable) |")
        print("|---|--:|--:|--:|")
        for d in dsets:
            a, j = idx.get((d, "alone")), idx.get((d, jr))
            if not (a and j):
                continue
            dip = f"{f(a['dip'],2)}→{f(j['dip'],2)}" + (beats(j["dip"], j["dip_const"]) if j.get("dip_claimable") else " (circ)")
            print(f"| {d} | {f(a['dice_deploy'])}→{f(j['dice_deploy'])} | {f(a['detF1'])}→{f(j['detF1'])} | {dip} |")

    if "joint 4:3:3" in roles and "joint 8:1:1" in roles:
        print("\n## Config tradeoff — 4:3:3 (mask-best) vs 8:1:1 (attribute-best)\n")
        print("| survey | Dice(deploy) 4:3:3 / 8:1:1 | dip 4:3:3 / 8:1:1 | throw 4:3:3 / 8:1:1 |")
        print("|---|--:|--:|--:|")
        for d in dsets:
            a, b = idx.get((d, "joint 4:3:3")), idx.get((d, "joint 8:1:1"))
            if not (a and b):
                continue
            print(f"| {d} | {f(a['dice_deploy'])} / {f(b['dice_deploy'])} | {f(a['dip'],2)} / {f(b['dip'],2)} | "
                  f"{f(a['throw'],1)} / {f(b['throw'],1)} |")
    print("\n_Dice=deploy (detect, not teacher-forced) · dip claimable only for smeaheia (circ=mask-derived GT) · "
          "✓ = beats its constant predictor._")


if __name__ == "__main__":
    main()
