"""Academic evaluation entry point — run the full benchmark suite.

  hybrid/run_eval.py                 # synthetic component benchmark (copy / reader / mask)
  DATASET=smeaheia hybrid/run_eval.py  # same benchmark on real held-out
  REAL_TRANSFER=1 hybrid/run_eval.py   # real-field before/after (adapter isolation)

Config is env-driven (no argparse): DATASET (synthetic|smeaheia), CKPT (narrator .pt),
READER (reader .pt), SCENES (cap). See hybrid/eval/{components,real_transfer,schema_check}.py.
"""
import os

from hybrid.eval import schema_check


def main():
    print("=" * 70, flush=True)
    print("SCHEMA CHECK", flush=True)
    print("=" * 70, flush=True)
    try:
        schema_check.main()
    except SystemExit:
        pass

    if os.environ.get("REAL_TRANSFER"):
        print("\n" + "=" * 70 + "\nREAL-FIELD BEFORE / AFTER (adapter isolation)\n" + "=" * 70, flush=True)
        from hybrid.eval import real_transfer
        real_transfer.main()
    else:
        ds = os.environ.get("DATASET", "synthetic")
        print("\n" + "=" * 70 + f"\nCOMPONENT BENCHMARK — {ds}\n" + "=" * 70, flush=True)
        from hybrid.eval import components
        components.main()


if __name__ == "__main__":
    main()
