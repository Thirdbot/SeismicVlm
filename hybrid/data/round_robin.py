"""Round-robin scene feeders for COMPLEMENTARY-JOINT real-field finetuning.

The mask decoder + class-driven heads are ONE shared copy; a sequential survey-by-survey finetune
overwrites earlier surveys (forgetting). Round-robin refreshes every survey every few steps so nothing is
forgotten, and each survey shapes the shared heads — the mechanism behind cross-survey complementarity
(a survey too small to train alone, e.g. Smeaheia's 144 faults, stands in the collective). Pure ordering
logic — no torch, no I/O — fair-and-square tested in hybrid/tests/test_round_robin.py.
"""
import random


def round_robin(scenes_by_ds, n_cycles, seed=0):
    """[(name, scene), ...] of length n_cycles*len(datasets): one item per dataset per cycle, in fixed
    dataset order, recycling (re-shuffling) any dataset whose pool is exhausted. Equal turns → the small
    attribute-rich survey (Smeaheia) shapes the shared heads as much as Thebe; recycle → refreshed every
    cycle so nothing is forgotten between rare appearances."""
    rng = random.Random(seed)
    pools = {name: rng.sample(list(sc), len(sc)) for name, sc in scenes_by_ds.items()}
    idx = {name: 0 for name in scenes_by_ds}
    order = list(scenes_by_ds.keys())
    seq = []
    for _ in range(n_cycles):
        for name in order:
            p = pools[name]
            if idx[name] >= len(p):                 # exhausted → reshuffle, recycle
                rng.shuffle(p); idx[name] = 0
            seq.append((name, p[idx[name]])); idx[name] += 1
    return seq


def weighted_round_robin(scenes_by_ds, weights, total_steps, seed=0):
    """Deficit weighted round-robin: each step every dataset accrues weight/Σweight credit, the highest-credit
    dataset is drawn (credit −1). Over total_steps a dataset gets ≈ total_steps·weight/Σweight draws, SMOOTHLY
    interleaved (not blocked) — so the large survey (Thebe) trains adequately while the small sets stay
    refreshed every few steps (no-forget) at a sane recycle rate. Exhausted pools reshuffle+recycle."""
    empty = [n for n, sc in scenes_by_ds.items() if not sc]
    if empty:                                                   # a survey with 0 TRAIN scenes (e.g. a bad/capped build)
        print(f"[round-robin] WARNING: empty training pool(s) {empty} — excluded from the schedule", flush=True)
    scenes_by_ds = {n: sc for n, sc in scenes_by_ds.items() if sc}
    if not scenes_by_ds:
        raise ValueError("[round-robin] all training pools are empty — nothing to train on (check the dataset builds)")
    rng = random.Random(seed)
    pools = {n: rng.sample(list(sc), len(sc)) for n, sc in scenes_by_ds.items()}
    idx = {n: 0 for n in scenes_by_ds}
    credit = {n: 0.0 for n in scenes_by_ds}
    wsum = float(sum(weights.get(n, 1) for n in scenes_by_ds))
    seq = []
    for _ in range(total_steps):
        for n in scenes_by_ds:
            credit[n] += weights[n] / wsum
        pick = max(scenes_by_ds, key=lambda n: credit[n])
        credit[pick] -= 1.0
        p = pools[pick]
        if idx[pick] >= len(p):
            rng.shuffle(p); idx[pick] = 0
        seq.append((pick, p[idx[pick]])); idx[pick] += 1
    return seq
