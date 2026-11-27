"""Compare the per-instance summaries of two runs of the same grid (reproducibility, concern C4).

    python scripts/compare_runs.py [--old results/full] [--new results/full_rerun] [--out FILE.csv]

For every configuration present in both runs, the summaries are merged on (instance_id, budget)
and every result column is compared except the timing (``seconds``). Rows of ``dice_random``
that the C9 fix changes on purpose are counted apart: budgets below 2,021 (smaller sample size)
and rows where the new run used the anytime fallback (the old run was cut by the budget and
returned nothing). Prints one line per (dataset, method, code version of the old run).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
COLS = ["n_cfs", "n_valid", "validity", "success", "coverage", "qd_score", "n_cells", "n_strategies",
        "dist_min", "dist_mean", "l0_mean", "violations", "diversity", "redundancy_mean", "n_queries",
        "anytime_fallback"]


def _codes(exp_dir: Path) -> dict:
    p = exp_dir / "manifest.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            out[e["key"]] = e.get("code")
    return out


def _differs(a: pd.Series, b: pd.Series) -> np.ndarray:
    a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    return ~(both_nan | np.isclose(a, b, rtol=1e-9, atol=1e-12))


def compare(old_dir: Path, new_dir: Path) -> pd.DataFrame:
    codes = _codes(old_dir)
    rows = []
    for f in sorted((new_dir / "summary").glob("*.csv")):
        o = old_dir / "summary" / f.name
        if not o.exists():
            continue
        new, old = pd.read_csv(f), pd.read_csv(o)
        m = old.merge(new, on=["instance_id", "budget"], suffixes=("_old", "_new"), how="outer",
                      indicator=True)
        unmatched = int((m["_merge"] != "both").sum())
        m = m[m["_merge"] == "both"]
        method = new["method"].iloc[0]
        expected = np.zeros(len(m), dtype=bool)
        if method == "dice_random":
            expected = (m["budget"] < 2021).to_numpy() | m["anytime_fallback_new"].astype(bool).to_numpy()
        cmp = m[~expected]
        bad = np.zeros(len(cmp), dtype=bool)
        cols_bad = []
        for c in COLS:
            d = _differs(cmp[f"{c}_old"], cmp[f"{c}_new"])
            if d.any():
                cols_bad.append(c)
                bad |= d
        rows.append(dict(key=f.stem, dataset=new["dataset"].iloc[0], model=new["model"].iloc[0],
                         method=method, seed=int(new["seed"].iloc[0]), old_code=codes.get(f.stem),
                         rows=len(m), unmatched=unmatched, expected_change=int(expected.sum()),
                         compared=len(cmp), differing=int(bad.sum()), columns=" ".join(cols_bad)))
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> str:
    if df.empty:
        return "no configuration finished in both runs yet"
    df = df.assign(identical=(df["differing"] == 0) & (df["unmatched"] == 0))
    g = df.groupby(["dataset", "method", "old_code"]).agg(
        configs=("key", "size"), identical=("identical", "sum"),
        rows_compared=("compared", "sum"), rows_differing=("differing", "sum"),
        expected_change=("expected_change", "sum")).reset_index()
    lines = [g.to_string(index=False), "",
             f"TOTAL: {int(df['identical'].sum())}/{len(df)} configurations identical; "
             f"{int(df['differing'].sum())} of {int(df['compared'].sum())} compared rows differ; "
             f"{int(df['expected_change'].sum())} dice_random rows changed on purpose (C9)"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="results/full")
    ap.add_argument("--new", default="results/full_rerun")
    ap.add_argument("--out", help="CSV with one line per configuration")
    a = ap.parse_args(argv)
    df = compare(REPO / a.old, REPO / a.new)
    if a.out:
        df.to_csv(REPO / a.out, index=False)
    print(report(df))


if __name__ == "__main__":
    main()
