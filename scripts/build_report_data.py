"""Aggregate the batch summaries into the data files of the group results page.

    python scripts/build_report_data.py [--exp results/full] [--out results/report]

The page itself is report/index.html (versioned); this script only writes its data.
Writes JSON files {rows, meta} (meta carries this script's own code, so every number
on the page can be traced back):

    overview.json    dataset x model x method at the largest budget: mean and sd across seeds
    curves.json      dataset x model x method x budget: means over seeds and instances
    progress.json    configurations finished per dataset (vs. the grid of the experiment)
    accuracy.json    test accuracy of the explained models, mean/sd over seeds, vs Guidotti
    carla.json       CARLA success-rate check (results/carla_check), vs the published rates

Aggregation: each (dataset, model, method, seed) is first averaged over its instances;
mean and sd are then taken across seeds (sd = sample standard deviation, ddof=1).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfqd import datasets as D  # noqa: E402
from cfqd import models as M  # noqa: E402
from cfqd.descriptor import Descriptor  # noqa: E402
from cfqd.experiment import expand_grid, load_experiment  # noqa: E402

METRICS = ["success", "coverage", "n_strategies", "n_valid", "dist_min", "l0_mean", "diversity",
           "redundancy_mean", "validity", "qd_score"]
CARLA = {("wachter", "adult", "nn"): 0.50, ("wachter", "compas_guidotti", "nn"): 0.66}
CODE = Path(__file__).read_text(encoding="utf-8")


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write(out: Path, name: str, df: pd.DataFrame, inputs: str) -> None:
    rows = [{k: clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    meta = [{"label": "build_report_data.py", "value": CODE, "type": "code", "lang": "python"},
            {"label": "Input", "value": inputs},
            {"label": "Built", "value": datetime.now(timezone.utc).isoformat(timespec="seconds")}]
    (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}), encoding="utf-8")
    print(f"{name}.json: {len(rows)} rows")


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    files = sorted((exp_dir / "summary").glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def seed_level(s: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "method", "seed", "budget"]
    agg = s.groupby(keys)[METRICS].mean()
    agg["violations"] = s.groupby(keys)["violations"].sum()
    agg["n_queries"] = s.groupby(keys)["n_queries"].median()
    agg["instances"] = s.groupby(keys)["instance_id"].nunique()
    return agg.reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/full")
    ap.add_argument("--config", default="configs/experiments/full.yaml")
    ap.add_argument("--out", default="results/report")
    a = ap.parse_args(argv)
    exp_dir, out = REPO / a.exp, REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    s = load_summaries(exp_dir)
    s["success"] = s["success"].astype(float)
    sl = seed_level(s)
    src = f"{a.exp}/summary/*.csv ({len(s):,} instance x budget rows)"

    # overview at the largest budget
    bmax = int(sl["budget"].max())
    top = sl[sl["budget"] == bmax]
    g = top.groupby(["dataset", "model", "method"])
    ov = g[METRICS].mean()
    for m in ["success", "coverage", "n_strategies", "dist_min"]:
        ov[f"{m}_sd"] = g[m].std(ddof=1)
    ov["violations"] = g["violations"].sum()
    ov["queries_median"] = g["n_queries"].median()
    ov["seeds"] = g["seed"].nunique()
    ov["instances"] = g["instances"].sum()
    ov["budget"] = bmax
    ov = ov.reset_index()
    ov["key"] = ov["dataset"] + "|" + ov["model"] + "|" + ov["method"]
    exp_cfg = load_experiment(REPO / a.config)
    grid = {}
    for ds in sorted(set(ov["dataset"])):
        spec = D.spec_from_config(D.load_config(ds))
        desc = Descriptor(spec, exp_cfg.get("effort_edges"))
        grid[ds] = (desc.n_feasible_cells, ", ".join(g.replace("_", " ") for g in spec.groups))
    ov["feasible_cells"] = ov["dataset"].map(lambda d: grid[d][0])
    ov["groups"] = ov["dataset"].map(lambda d: grid[d][1])
    write(out, "overview", ov, src + f"; budget = {bmax}; feasible cells from cfqd.descriptor.Descriptor")

    # budget curves
    cv = sl.groupby(["dataset", "model", "method", "budget"])[
        ["success", "coverage", "n_strategies", "n_valid", "dist_min"]].mean()
    cv["queries_median"] = sl.groupby(["dataset", "model", "method", "budget"])["n_queries"].median()
    cv["seeds"] = sl.groupby(["dataset", "model", "method", "budget"])["seed"].nunique()
    cv = cv.reset_index()
    write(out, "curves", cv, src)

    # progress against the experiment grid
    exp = load_experiment(REPO / a.config)
    keys = expand_grid(exp)[0]
    done = {p.stem for p in (exp_dir / "summary").glob("*.csv")}
    pr = pd.DataFrame([dict(dataset=k.dataset, done=k.stem in done) for k in keys])
    pr = pr.groupby("dataset")["done"].agg(done="sum", total="count").reset_index()
    pr["share"] = pr["done"] / pr["total"]
    order = {d: i for i, d in enumerate(exp["datasets"])}
    pr = pr.sort_values("dataset", key=lambda c: c.map(order))
    write(out, "progress", pr, f"{a.config} grid vs {a.exp}/summary/*.csv")

    # model accuracy over the seeds that have models
    acc = []
    for ds in sorted(set(s["dataset"])):
        for f in sorted((M.MODELS_DIR / ds).glob("seed*/metrics.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            for kind in ("lr", "rf", "nn"):
                if kind in m:
                    acc.append(dict(dataset=ds, model=kind, seed=int(f.parent.name[4:]),
                                    acc=m[kind]["acc_test"]))
    acc = pd.DataFrame(acc)
    if len(acc):
        ga = acc.groupby(["dataset", "model"])["acc"]
        acc = ga.agg(acc_mean="mean", acc_sd="std", seeds="count").reset_index()
        acc["guidotti"] = [M.GUIDOTTI_ACC.get(r.dataset, {}).get(r.model) for r in acc.itertuples()]
        acc["key"] = acc["dataset"] + "|" + acc["model"]
        write(out, "accuracy", acc, "models/<dataset>/seed*/metrics.json (test split, 70/30)")

    # CARLA success-rate check
    cs = load_summaries(REPO / "results" / "carla_check")
    if len(cs):
        cs["success"] = cs["success"].astype(float)
        cr = cs.groupby(["method", "dataset", "model"]).agg(
            success=("success", "mean"), instances=("instance_id", "nunique"),
            violations=("violations", "sum")).reset_index()
        cr["carla"] = [CARLA.get((r.method, r.dataset, r.model), 1.0) for r in cr.itertuples()]
        cr["key"] = cr["method"] + "|" + cr["dataset"] + "|" + cr["model"]
        write(out, "carla", cr, "results/carla_check/summary/*.csv (budget 10,000, 10 instances, seed 0)")


if __name__ == "__main__":
    main()
