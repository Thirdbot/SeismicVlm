# Benchmark report (9 dataset-checkpoint rows from 5 logs)

## Full metrics (per dataset · per checkpoint · never pooled)

| dataset | checkpoint | n | Dice oracle/deploy | pixP/R/F1 | tol-F1 | detF1 | class | dip(const) | throw(const) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| thebe | alone | 31114 | 0.345/0.318 | 0.33/0.48/0.35 | 0.398 | 0.411 | 13909/13909 | 3.98(3.73) | —(—) |
| thebe | joint 8:1:1 | 31114 | 0.357/0.330 | 0.31/0.56/0.36 | 0.408 | 0.391 | 13388/13388 | 3.94(3.47) | —(—) |
| thebe | joint 4:3:3 | 31114 | 0.349/0.317 | 0.35/0.45/0.35 | 0.403 | 0.404 | 13604/13604 | 5.94(3.52) | —(—) |
| cracks | alone | 1679 | 0.068/0.052 | 0.05/0.17/0.07 | 0.088 | 0.420 | 707/707 | —(—) | —(—) |
| cracks | joint 8:1:1 | 1679 | 0.112/0.107 | 0.08/0.25/0.11 | 0.143 | 0.546 | 1083/1083 | —(—) | —(—) |
| cracks | joint 4:3:3 | 1679 | 0.123/0.121 | 0.11/0.21/0.12 | 0.161 | 0.650 | 1338/1338 | —(—) | —(—) |
| smeaheia | alone | 34 | 0.010/0.000 | 0.01/0.01/0.01 | 0.017 | 0.000 | — | — (sticks)(—) | —(—) |
| smeaheia | joint 8:1:1 | 34 | 0.057/0.015 | 0.07/0.19/0.06 | 0.076 | 0.045 | 1/1 | 65.15 (sticks)(0.00) | 19.0(0.0) |
| smeaheia | joint 4:3:3 | 34 | 0.010/0.007 | 0.08/0.01/0.01 | 0.020 | 0.157 | 4/4 | 22.34 (sticks)(23.16) ✓ | 118.4(117.5) |

## Complementarity — {alone} vs {joint 4:3:3} (each survey on its OWN held-out)

| survey | Dice(deploy) alone→joint | detF1 alone→joint | dip alone→joint (claimable) |
|---|--:|--:|--:|
| thebe | 0.318→0.317 | 0.411→0.404 | 3.98→5.94 |
| cracks | 0.052→0.121 | 0.420→0.650 | —→— |
| smeaheia | 0.000→0.007 | 0.000→0.157 | —→22.34 ✓ (sticks) |

## Config tradeoff — 4:3:3 (mask-best) vs 8:1:1 (attribute-best)

| survey | Dice(deploy) 4:3:3 / 8:1:1 | dip 4:3:3 / 8:1:1 | throw 4:3:3 / 8:1:1 |
|---|--:|--:|--:|
| thebe | 0.317 / 0.330 | 5.94 / 3.94 | — / — |
| cracks | 0.121 / 0.107 | — / — | — / — |
| smeaheia | 0.007 / 0.015 | 22.34 / 65.15 | 118.4 / 19.0 |

_Dice=deploy (detect, not teacher-forced) · dip read from fault-trace geometry; smeaheia dip ALSO from independent projected sticks (an extra cross-check, marked 'sticks') · ✓ = beats its constant predictor._
