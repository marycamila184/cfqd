"""Batch runner (T13).

    python scripts/run_batch.py --config configs/experiments/smoke.yaml --dry-run
    python scripts/run_batch.py --config configs/experiments/v1.yaml --jobs 4
    python scripts/run_batch.py --config configs/experiments/v1.yaml --only dataset=german_credit method=wachter

Resumable: configurations whose output files exist are skipped (use --force to redo).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfqd.experiment import dry_run, load_experiment, run_grid  # noqa: E402


def parse_only(items):
    if not items:
        return None
    conds = [tuple(i.split("=", 1)) for i in items]

    def ok(key):
        return all(str(getattr(key, f)) == v for f, v in conds)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML")
    ap.add_argument("--dry-run", action="store_true", help="print the grid and its cost, run nothing")
    ap.add_argument("--jobs", type=int, default=1, help="parallel configurations (processes)")
    ap.add_argument("--force", action="store_true", help="rerun configurations already done")
    ap.add_argument("--out", help="override the output folder of the YAML")
    ap.add_argument("--only", nargs="*", help="filters field=value (dataset, model, method, seed)")
    a = ap.parse_args(argv)
    exp = load_experiment(a.config)
    if a.out:
        exp["out"] = a.out
    print(dry_run(exp))
    if a.dry_run:
        return
    run_grid(exp, jobs=a.jobs, force=a.force, only=parse_only(a.only))


if __name__ == "__main__":
    main()
