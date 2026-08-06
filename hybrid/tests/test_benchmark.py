"""FAIR-AND-SQUARE TEST of benchmark.py's metric functions against hand-computed known cases. If these
pass, the numbers the benchmark reports are correct arithmetic; if not, the benchmark is buggy. No model.

  python -m hybrid.tests.test_benchmark
"""
import os
os.environ.setdefault("SFM_CKPT", "hybrid/checkpoints/SFM-Base-512.pth")
import torch

from hybrid.eval.benchmark import px_prf, tol_f1, _iou, _tdice, _prf, _const
from hybrid.model.geometry import field_dice


def block(r0, r1, c0, c1, H=20, W=20):
    m = torch.zeros(H, W); m[r0:r1, c0:c1] = 1.0
    return m


def main():
    fails = []

    def check(name, cond, got, want):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: got {got} · want {want}", flush=True)
        if not cond:
            fails.append(name)

    print("=== perfect overlap → all 1.0 ===", flush=True)
    g = block(0, 10, 0, 20); p = g.clone()
    check("IoU=1", abs(_iou(p, g) - 1) < 1e-6, f"{_iou(p,g):.3f}", "1.0")
    check("tDice=1", abs(_tdice(p, g) - 1) < 1e-6, f"{_tdice(p,g):.3f}", "1.0")
    P, R, F = px_prf(p, g)
    check("pixel P/R/F=1", abs(P - 1) < 1e-6 and abs(R - 1) < 1e-6, f"{P:.2f}/{R:.2f}/{F:.2f}", "1/1/1")
    check("tol-F1=1", abs(tol_f1(p, g)[0] - 1) < 1e-6, f"{tol_f1(p,g)[0]:.3f}", "1.0")

    print("=== half overlap (p rows0-10, g rows5-15) → IoU 0.333, Dice 0.5, P/R 0.5 ===", flush=True)
    p = block(0, 10, 0, 20); g = block(5, 15, 0, 20)                # overlap 5×20=100; |p|=|g|=200
    check("IoU=0.333", abs(_iou(p, g) - 1 / 3) < 1e-3, f"{_iou(p,g):.3f}", "0.333")
    check("tDice=0.5", abs(_tdice(p, g) - 0.5) < 1e-3, f"{_tdice(p,g):.3f}", "0.5")
    P, R, F = px_prf(p, g)
    check("pixel P=R=0.5", abs(P - 0.5) < 1e-3 and abs(R - 0.5) < 1e-3, f"{P:.2f}/{R:.2f}", "0.5/0.5")

    print("=== disjoint → 0 ===", flush=True)
    p = block(0, 5, 0, 20); g = block(10, 15, 0, 20)
    check("IoU=0", _iou(p, g) == 0, f"{_iou(p,g):.3f}", "0.0")
    check("tDice=0", _tdice(p, g) == 0, f"{_tdice(p,g):.3f}", "0.0")

    print("=== tolerance-F1: a 1px-offset line is forgiven at τ=2 but strict-F1 is 0 ===", flush=True)
    g = block(0, 20, 10, 11); p = block(0, 20, 11, 12)             # vertical lines, 1px apart, no overlap
    _, _, sF = px_prf(p, g)
    check("strict pixel-F1 ≈ 0 (no overlap)", sF < 0.05, f"{sF:.2f}", "≈0")
    check("tol-F1@2px ≥ 0.9 (offset forgiven)", tol_f1(p, g, tau=2)[0] >= 0.9, f"{tol_f1(p,g,2)[0]:.2f}", "≥0.9")

    print("=== field_dice (logit input) perfect → 1 ===", flush=True)
    g = block(5, 15, 5, 15); logit = g * 20 - 10                    # sigmoid≈1 on g, ≈0 off
    check("field_dice=1", field_dice(logit, g) > 0.98, f"{field_dice(logit, g):.3f}", "≈1.0")

    print("=== _const (constant-predictor MAE = MAD from median) ===", flush=True)
    check("const([70,70,70])=0", _const([70, 70, 70]) == 0, f"{_const([70,70,70]):.2f}", "0")
    check("const([60,70,80])=6.67", abs(_const([60, 70, 80]) - 20 / 3) < 1e-2, f"{_const([60,70,80]):.2f}", "6.67")

    print("=== _prf (detection) ===", flush=True)
    P, R, F = _prf(8, 2, 2)
    check("_prf(8,2,2)=0.8/0.8/0.8", abs(P - .8) < 1e-9 and abs(F - .8) < 1e-9, f"{P:.2f}/{R:.2f}/{F:.2f}", "0.8/0.8/0.8")

    print("=== centroid L2 ×100 (localization) ===", flush=True)
    def cdist(a, b): return 100.0 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    check("centroid same=0", cdist([.5, .5], [.5, .5]) == 0, f"{cdist([.5,.5],[.5,.5]):.1f}", "0")
    check("centroid Δrow0.1=10", abs(cdist([.6, .5], [.5, .5]) - 10) < 1e-6, f"{cdist([.6,.5],[.5,.5]):.1f}", "10")

    print(f"\n[TEST_BENCHMARK] {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}", flush=True)
    print("TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
