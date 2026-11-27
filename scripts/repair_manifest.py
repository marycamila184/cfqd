"""Repair the manifest of a finished batch run.

    python scripts/repair_manifest.py --exp results/full --config configs/experiments/full.yaml

Runs before 2026-11-20 appended every configuration to one shared manifest.jsonl from
parallel workers, which broke some lines and lost others (the result CSVs are written
atomically per configuration and were not affected). This script:

1. writes a manifest/<key>.json for every configuration that has its result files but no
   readable manifest entry, marked "reconstructed": true and "seconds": null; the code
   hash is the one shared by the other entries of the run, the dataset md5 comes from
   data/manifest.jsonl, budgets and parameters from the experiment config;
2. rewrites manifest.jsonl from the per-configuration files plus the readable old lines.

Run it only when no batch is writing to that folder.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfqd.experiment import expand_grid, is_done, load_experiment, read_manifest, write_manifest_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/full")
    ap.add_argument("--config", default="configs/experiments/full.yaml")
    a = ap.parse_args(argv)
    exp = load_experiment(REPO / a.config)
    exp["out"] = a.exp
    exp_dir = REPO / a.exp
    recs = read_manifest(exp_dir)
    have = {r["key"] for r in recs}
    code = Counter(r.get("code") for r in recs).most_common(1)[0][0] if recs else None
    data_md5 = {}
    dm = REPO / "data" / "manifest.jsonl"
    if dm.exists():
        for line in dm.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                data_md5[r["dataset"]] = (r.get("md5"), r.get("generator_hash"))
    (exp_dir / "manifest").mkdir(parents=True, exist_ok=True)
    fixed = []
    for k in expand_grid(exp)[0]:
        if k.stem in have or not is_done(k, exp):
            continue
        md5, gh = data_md5.get(k.dataset, (None, None))
        entry = dict(key=k.stem, dataset=k.dataset, model=k.model, method=k.method, seed=k.seed,
                     budgets=sorted(exp["budgets"]), n_instances=exp["n_instances"],
                     method_params=exp["methods"][k.method], effort_edges=exp.get("effort_edges"),
                     train_predictions=exp.get("train_predictions"), code=code, dataset_md5=md5,
                     generator_hash=gh, seconds=None, finished_at=None, reconstructed=True)
        (exp_dir / "manifest" / f"{k.stem}.json").write_text(json.dumps(entry), encoding="utf-8")
        fixed.append(k.stem)
    n = write_manifest_jsonl(exp_dir)
    print(f"reconstructed {len(fixed)} entries: {fixed}")
    print(f"manifest.jsonl rewritten with {n} entries")


if __name__ == "__main__":
    main()
