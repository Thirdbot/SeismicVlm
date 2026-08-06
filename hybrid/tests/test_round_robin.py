"""FAIR-AND-SQUARE TEST of the round-robin feeder for the complementary-joint run (C). The feeder gives
each dataset EQUAL TURNS per cycle (balance-by-steps, not by size), recycling exhausted small sets so they
are refreshed every cycle (the no-forgetting mechanism). This tests the ordering logic before it drives
any training. The function under test is defined here and will be lifted into run_joint_rr.py.

  python -m hybrid.tests.test_round_robin
"""
from collections import Counter

from hybrid.data.round_robin import round_robin, weighted_round_robin      # the promoted main functions


def main():
    fails = []

    def check(name, cond, got, want):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: got {got} · want {want}", flush=True)
        if not cond:
            fails.append(name)

    # datasets of very different sizes (like Thebe 37856 / CRACKS 297 / Smeaheia 262)
    big = list(range(1000))          # "thebe"
    small = [f"c{i}" for i in range(5)]   # "cracks" (tiny → must recycle)
    tiny = [f"s{i}" for i in range(3)]    # "smeaheia"
    N = 20
    seq = round_robin({"thebe": big, "cracks": small, "smeaheia": tiny}, n_cycles=N)

    print("=== equal turns (each dataset appears exactly n_cycles times) ===", flush=True)
    from collections import Counter
    cnt = Counter(name for name, _ in seq)
    check("thebe turns = N", cnt["thebe"] == N, cnt["thebe"], N)
    check("cracks turns = N", cnt["cracks"] == N, cnt["cracks"], N)
    check("smeaheia turns = N", cnt["smeaheia"] == N, cnt["smeaheia"], N)
    check("total length = N*3", len(seq) == N * 3, len(seq), N * 3)

    print("=== interleaving (one of each per cycle, fixed order, no clumping) ===", flush=True)
    cyc0 = [name for name, _ in seq[:3]]
    check("cycle order = [thebe,cracks,smeaheia]", cyc0 == ["thebe", "cracks", "smeaheia"], cyc0, "[thebe,cracks,smeaheia]")
    max_run = 1; run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i][0] == seq[i-1][0] else 1
        max_run = max(max_run, run)
    check("no dataset runs >1 in a row", max_run == 1, f"max_run {max_run}", "1")

    print("=== recycling (tiny sets reused ~evenly, big set never repeats within N) ===", flush=True)
    small_items = [x for n, x in seq if n == "cracks"]           # 20 draws from 5 items → each ~4×
    sc = Counter(small_items)
    check("cracks 5 items all seen", len(sc) == 5, len(sc), 5)
    check("cracks reuse ~even (3-5 each)", all(3 <= v <= 5 for v in sc.values()), dict(sc), "3-5 each")
    big_items = [x for n, x in seq if n == "thebe"]              # 20 draws from 1000 → all distinct
    check("thebe no repeat within N", len(set(big_items)) == len(big_items), f"{len(set(big_items))}/{len(big_items)}", "all distinct")

    print("=== determinism (same seed → same sequence) ===", flush=True)
    seq2 = round_robin({"thebe": big, "cracks": small, "smeaheia": tiny}, n_cycles=N)
    check("seed reproducible", seq == seq2, "equal", "equal")

    print("=== WEIGHTED round-robin (thebe:8, cracks:1, smeaheia:1) ===", flush=True)
    T = 10000
    wseq = weighted_round_robin({"thebe": big, "cracks": small, "smeaheia": tiny},
                                {"thebe": 8, "cracks": 1, "smeaheia": 1}, total_steps=T)
    wc = Counter(n for n, _ in wseq)
    check("total = T", len(wseq) == T, len(wseq), T)
    check("thebe ≈ 80%", abs(wc["thebe"] / T - 0.8) < 0.02, f"{wc['thebe']/T:.2%}", "~80%")
    check("cracks ≈ 10%", abs(wc["cracks"] / T - 0.1) < 0.02, f"{wc['cracks']/T:.2%}", "~10%")
    check("smeaheia ≈ 10%", abs(wc["smeaheia"] / T - 0.1) < 0.02, f"{wc['smeaheia']/T:.2%}", "~10%")
    # small sets stay refreshed: max gap between smeaheia appearances is bounded (not starved)
    pos = [i for i, (n, _) in enumerate(wseq) if n == "smeaheia"]
    max_gap = max(pos[i+1] - pos[i] for i in range(len(pos)-1))
    check("smeaheia refreshed (max gap < 15)", max_gap < 15, f"max gap {max_gap}", "<15")
    # no long thebe bursts (smooth interleave, not 8-in-a-row blocks)
    mr = 1; run = 1
    for i in range(1, len(wseq)):
        run = run + 1 if wseq[i][0] == wseq[i-1][0] else 1; mr = max(mr, run)
    check("thebe bursts bounded (max run < 12)", mr < 12, f"max run {mr}", "<12")

    print(f"\n[TEST_ROUND_ROBIN] {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}", flush=True)
    print("TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
