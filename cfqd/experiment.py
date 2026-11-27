"""Batch execution (T13): datasets x models x methods x seeds x budgets.

One CONFIGURATION = (dataset, model, method, seed). For each configuration the
same ``n_instances`` test instances (predicted unfavorable by that model) are
explained at every budget of the curve. Budgets are measured in model
queries (``CountingModel``), never in time.

Budget curves: each budget is an independent run with the SAME search seed.
For anytime methods (MAP-Elites) a run at budget b is the exact prefix of a run
at a larger budget, so this gives the exact state of the archive at every
checkpoint; for one-shot methods (DiCE, Wachter, FACE) it is the only honest
way to measure them at a given budget.

Outputs (resumable: a configuration whose files exist is skipped)::

    results/runs/<dataset>__<model>__<method>__seed<k>.csv     one row per counterfactual
    results/summary/<dataset>__<model>__<method>__seed<k>.csv  one row per (instance, budget)
    results/manifest.jsonl                                     one line per configuration
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import datasets as D
from . import models as M
from .archive import GridArchive
from .descriptor import Descriptor
from .metrics import constraint_violations, guidotti_distance, l0, mean_pairwise_distance
from .model import CountingModel
from .results import to_frame

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUDGETS = [500, 1000, 2000, 5000, 10000]


# ================================================================ registry
def _map_elites(p, ds, exp):
    from .methods import MapElites
    p = {"effort_edges": exp.get("effort_edges"), **p}
    return MapElites(**p)


def _dice(method):
    def make(p, ds, exp):
        from .methods import Dice
        p = {"train_predictions": exp.get("train_predictions", "free"), **p}
        return Dice(ds.X_train, ds.y_train, ds.spec, method=method, **p)
    return make


def _wachter(p, ds, exp):
    from .methods import Wachter
    return Wachter(**p)


def _face(p, ds, exp):
    from .methods import Face
    p = {"train_predictions": exp.get("train_predictions", "free"), **p}
    return Face(ds.X_train, **p)


# name -> (factory, needs_gradient). NS, NSLC, surprise and MOC register here.
REGISTRY = {
    "map_elites": (_map_elites, False),
    "dice_genetic": (_dice("genetic"), False),
    "dice_random": (_dice("random"), False),
    "wachter": (_wachter, True),
    "face_knn": (_face, False),
}


def register(name: str, factory, needs_gradient: bool = False) -> None:
    """factory(params: dict, ds: Dataset, experiment: dict) -> CounterfactualMethod"""
    REGISTRY[name] = (factory, needs_gradient)


def build_method(name: str, params: dict, ds, exp: dict):
    factory, _ = REGISTRY[name]
    m = factory(dict(params or {}), ds, exp)
    m.name = name
    return m


# =================================================================== grid
@dataclass(frozen=True)
class RunKey:
    dataset: str
    model: str
    method: str
    seed: int

    @property
    def stem(self) -> str:
        return f"{self.dataset}__{self.model}__{self.method}__seed{self.seed}"


def load_experiment(path: str | Path) -> dict:
    exp = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    exp.setdefault("models", list(M.MODEL_KINDS))
    exp.setdefault("seeds", [0])
    exp.setdefault("budgets", DEFAULT_BUDGETS)
    exp.setdefault("n_instances", 10)
    exp.setdefault("out", "results")
    exp.setdefault("train_predictions", "free")
    exp.setdefault("model_params", {})
    exp["methods"] = {k: (v or {}) for k, v in exp["methods"].items()}
    unknown = set(exp["methods"]) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}; registered: {sorted(REGISTRY)}")
    return exp


def expand_grid(exp: dict) -> tuple[list[RunKey], list[tuple[RunKey, str]]]:
    """Return (runnable configurations, skipped configurations with the reason)."""
    keys, skipped = [], []
    for ds_name in exp["datasets"]:
        spec = D.spec_from_config(D.load_config(ds_name))
        no_action = not spec.mutable_mask.any()
        for model in exp["models"]:
            for method in exp["methods"]:
                for seed in exp["seeds"]:
                    k = RunKey(ds_name, model, method, int(seed))
                    if no_action:
                        skipped.append((k, "no actionable feature (strict actionability)"))
                    elif REGISTRY[method][1] and model == "rf":
                        skipped.append((k, "method needs a gradient; RF has none"))
                    else:
                        keys.append(k)
    return keys, skipped


def out_dir(exp: dict) -> Path:
    p = Path(exp["out"])
    return p if p.is_absolute() else REPO / p


def is_done(key: RunKey, exp: dict) -> bool:
    o = out_dir(exp)
    return (o / "runs" / f"{key.stem}.csv").exists() and (o / "summary" / f"{key.stem}.csv").exists()


# ============================================================== instances
def select_instances(ds, cm: CountingModel, n: int, seed: int) -> np.ndarray:
    """Test rows predicted UNFAVORABLE by this model (uncounted), sampled with the seed.
    Returned as positions in ds.idx_test (stable ids: ds.idx_test[pos] = row of data.parquet)."""
    pred = cm.predict_uncounted(ds.X_test)
    pos = np.flatnonzero(pred != ds.desired)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 17]))
    if len(pos) > n:
        pos = np.sort(rng.choice(pos, size=n, replace=False))
    return pos


def search_seed(seed: int, row: int) -> int:
    return int(np.random.SeedSequence([seed, int(row)]).generate_state(1)[0])


# ================================================================ metrics
def redundancy_batch(cm: CountingModel, x: np.ndarray, cfs: np.ndarray, desired: int) -> np.ndarray:
    """Same definition as metrics.redundancy (CARLA), vectorized: one batch per CF."""
    out = np.zeros(len(cfs), dtype=int)
    for k, c in enumerate(cfs):
        changed = np.flatnonzero(np.abs(x - c) > 1e-9)
        if len(changed) == 0:
            continue
        Y = np.repeat(c[None], len(changed), axis=0)
        Y[np.arange(len(changed)), changed] = x[changed]
        out[k] = int(np.sum(cm.predict_uncounted(Y) == desired))
    return out


def summarize(x, res, cm, spec, desired, desc: Descriptor, truth=None) -> dict:
    cfs = np.asarray(res.cfs, dtype=float)
    if cfs.size == 0:
        cfs = np.empty((0, spec.d))
    valid = cm.predict_uncounted(cfs) == desired if len(cfs) else np.zeros(0, bool)
    V = cfs[valid]
    arch = GridArchive(desc, x)
    arch.add_many(V)
    dists = np.array([guidotti_distance(x, c, spec) for c in V])
    row = dict(
        n_cfs=len(cfs), n_valid=int(valid.sum()),
        validity=float(valid.mean()) if len(cfs) else np.nan,
        success=bool(valid.any()),
        coverage=arch.coverage() if desc.n_feasible_cells else np.nan,
        qd_score=arch.qd_score(), n_cells=len(arch), n_strategies=arch.n_strategies(),
        dist_mean=float(dists.mean()) if len(V) else np.nan,
        dist_min=float(dists.min()) if len(V) else np.nan,
        l0_mean=float(np.mean([l0(x, c) for c in V])) if len(V) else np.nan,
        violations=int(sum(constraint_violations(x, c, spec) for c in cfs)),
        diversity=mean_pairwise_distance(V, spec) if len(V) > 1 else np.nan,
        redundancy_mean=float(redundancy_batch(cm, x, V, desired).mean()) if len(V) else np.nan,
        n_queries=int(res.n_queries), seconds=float(res.seconds),
        anytime_fallback=bool(res.extra.get("anytime_fallback", False)),
    )
    if truth is not None:
        row.update(truth.summary(x, V))
    return row


# =================================================================== runs
def code_version() -> str:
    """Hash of the package sources (the repo is not always a git checkout)."""
    h = hashlib.sha1()
    for p in sorted((REPO / "cfqd").rglob("*.py")):
        h.update(p.read_bytes())
    try:
        import subprocess
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                           text=True, timeout=5)
        if c.returncode == 0:
            return f"git:{c.stdout.strip()}|src:{h.hexdigest()[:10]}"
    except Exception:
        pass
    return f"src:{h.hexdigest()[:10]}"


def run_config(key: RunKey, exp: dict, verbose: bool = False) -> dict:
    """Run one configuration over all instances and budgets; write its files."""
    t0 = time.perf_counter()
    import torch
    torch.set_num_threads(1)
    ds = D.load_dataset(key.dataset, seed=key.seed)
    model = M.get_model(ds, key.model, exp["model_params"].get(key.model))
    cm = CountingModel(model)
    method = build_method(key.method, exp["methods"][key.method], ds, exp)
    desc = Descriptor(ds.spec, exp.get("effort_edges"))
    truth = None
    if ds.cfg["source"]["kind"] == "synthetic":
        from .simulation import GroundTruth
        truth = GroundTruth.from_meta(ds.meta)
    positions = select_instances(ds, cm, exp["n_instances"], key.seed)
    cf_frames, rows = [], []
    for pos in positions:
        row_id = int(ds.idx_test[pos])
        x = ds.X_test[pos]
        s = search_seed(key.seed, row_id)
        for b in sorted(exp["budgets"]):
            res = method.explain(x, cm, ds.spec, desired=ds.desired, budget=int(b), seed=s)
            res.method = key.method
            if len(res.cfs):
                cf_frames.append(to_frame(res, ds.spec, dataset=key.dataset, model=key.model,
                                          seed=key.seed, instance_id=row_id, budget=int(b)))
            r = dict(dataset=key.dataset, model=key.model, method=key.method, seed=key.seed,
                     instance_id=row_id, budget=int(b))
            r.update(summarize(x, res, cm, ds.spec, ds.desired, desc, truth))
            rows.append(r)
        if verbose:
            print(f"  {key.stem} instance {row_id} done", flush=True)
    o = out_dir(exp)
    for sub in ("runs", "summary"):
        (o / sub).mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "model", "method", "seed", "instance_id", "budget", "cf_id", "found_at",
            "n_queries", "seconds"] + ds.spec.names
    cf_df = pd.concat(cf_frames, ignore_index=True) if cf_frames else pd.DataFrame(columns=cols)
    sm_df = pd.DataFrame(rows)
    entry = dict(
        key=key.stem, dataset=key.dataset, model=key.model, method=key.method, seed=key.seed,
        budgets=sorted(exp["budgets"]), n_instances=len(positions),
        method_params=exp["methods"][key.method], effort_edges=list(desc.effort_edges),
        train_predictions=exp.get("train_predictions"), code=code_version(),
        dataset_md5=ds.meta.get("source", {}).get("md5"),
        generator_hash=ds.meta.get("generator_hash"), seconds=round(time.perf_counter() - t0, 2),
        python=platform.python_version(), torch=torch.__version__,
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    # Every file is written to a temporary name and renamed (atomic). One manifest file per
    # configuration: parallel workers appending to a shared manifest.jsonl corrupted lines
    # (full run, 20/11). The summary goes last, since it is what marks a configuration done.
    (o / "manifest").mkdir(parents=True, exist_ok=True)
    for sub, name, write in (
        ("runs", f"{key.stem}.csv", lambda p: cf_df.to_csv(p, index=False)),
        ("manifest", f"{key.stem}.json", lambda p: p.write_text(json.dumps(entry, default=str),
                                                                 encoding="utf-8")),
        ("summary", f"{key.stem}.csv", lambda p: sm_df.to_csv(p, index=False)),
    ):
        tmp = o / sub / f"{name}.tmp"
        write(tmp)
        os.replace(tmp, o / sub / name)
    return entry


def read_manifest(exp_dir: Path) -> list[dict]:
    """Per-configuration manifest files, plus the lines of an older manifest.jsonl
    (unreadable lines are skipped). Per-configuration files win on duplicate keys."""
    exp_dir = Path(exp_dir)
    out: dict[str, dict] = {}
    legacy = exp_dir / "manifest.jsonl"
    if legacy.exists():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out[r["key"]] = r
            except (ValueError, KeyError):
                pass
    for p in sorted((exp_dir / "manifest").glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            out[r["key"]] = r
        except (ValueError, KeyError):
            pass
    return sorted(out.values(), key=lambda r: r.get("finished_at") or "")


def write_manifest_jsonl(exp_dir: Path) -> int:
    """Rebuild manifest.jsonl from read_manifest (one process only, e.g. at the end of a run)."""
    recs = read_manifest(exp_dir)
    path = Path(exp_dir) / "manifest.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in recs), encoding="utf-8")
    os.replace(tmp, path)
    return len(recs)


def _worker(key: RunKey, exp: dict) -> tuple[RunKey, dict | None, str | None]:
    try:
        return key, run_config(key, exp), None
    except Exception:
        return key, None, traceback.format_exc()


def run_grid(exp: dict, jobs: int = 1, force: bool = False, only=None, verbose: bool = True) -> list:
    keys, skipped = expand_grid(exp)
    if only:
        keys = [k for k in keys if only(k)]
    todo = [k for k in keys if force or not is_done(k, exp)]
    if verbose:
        print(f"{len(keys)} configurations, {len(keys) - len(todo)} already done, "
              f"{len(todo)} to run, {len(skipped)} skipped", flush=True)
    for k in todo:  # build datasets/models once, before forking workers
        ds = D.load_dataset(k.dataset, seed=k.seed)
        M.get_model(ds, k.model, exp["model_params"].get(k.model))
    done, failures = [], []
    if jobs <= 1:
        for i, k in enumerate(todo, 1):
            k, e, err = _worker(k, exp)
            (done if e else failures).append((k, e or err))
            if verbose:
                print(f"[{i}/{len(todo)}] {k.stem}: " + (f"{e['seconds']}s" if e else "FAILED"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = [pool.submit(_worker, k, exp) for k in todo]
            for i, f in enumerate(as_completed(futs), 1):
                k, e, err = f.result()
                (done if e else failures).append((k, e or err))
                if verbose:
                    print(f"[{i}/{len(todo)}] {k.stem}: " + (f"{e['seconds']}s" if e else "FAILED"),
                          flush=True)
    for k, err in failures:
        print(f"FAILED {k.stem}\n{err}")
    if todo:
        write_manifest_jsonl(out_dir(exp))
    return done


# ================================================================ dry run
def estimate(exp: dict) -> pd.DataFrame:
    """Grid size and cost in queries; seconds estimated from earlier manifests if any."""
    keys, skipped = expand_grid(exp)
    per_instance = sum(exp["budgets"])
    df = pd.DataFrame([dict(dataset=k.dataset, model=k.model, method=k.method, seed=k.seed,
                            done=is_done(k, exp)) for k in keys])
    if df.empty:
        return df
    df["queries_max"] = exp["n_instances"] * per_instance
    # calibrate with every manifest already produced (smoke runs included)
    dirs = {p.parent for p in (REPO / "results").rglob("manifest.jsonl")}
    dirs |= {p.parent for p in (REPO / "results").rglob("manifest") if p.is_dir()}
    dirs.add(out_dir(exp))
    recs = [r for d in dirs for r in read_manifest(d) if r.get("seconds") is not None]
    rate = {}
    if recs:
        m = pd.DataFrame(recs)
        m["s_per_kquery"] = m["seconds"] / (m["n_instances"].clip(lower=1)
                                            * m["budgets"].map(sum) / 1000)
        rate = m.groupby(["dataset", "method"])["s_per_kquery"].median().to_dict()
    df["est_seconds"] = [rate.get((r.dataset, r.method), np.nan) * r.queries_max / 1000
                         for r in df.itertuples()]
    return df


def dry_run(exp: dict) -> str:
    keys, skipped = expand_grid(exp)
    est = estimate(exp)
    lines = [
        f"experiment: {exp.get('name', '?')}",
        f"datasets={exp['datasets']} models={exp['models']} methods={list(exp['methods'])}",
        f"seeds={exp['seeds']} budgets={exp['budgets']} n_instances={exp['n_instances']}",
        f"configurations: {len(keys)} runnable, {len(skipped)} skipped, "
        f"{int(est['done'].sum()) if len(est) else 0} already done",
        f"explain() calls: {len(keys) * exp['n_instances'] * len(exp['budgets']):,}",
        f"query ceiling: {len(keys) * exp['n_instances'] * sum(exp['budgets']):,}",
    ]
    if len(est) and est["est_seconds"].notna().any():
        known = est["est_seconds"].notna()
        lines.append(f"estimated time (from manifest rates, {known.sum()} of {len(est)} configs "
                     f"covered): {est.loc[known, 'est_seconds'].sum() / 3600:.1f} h sequential")
    else:
        lines.append("estimated time: unknown (run a smoke experiment first to calibrate)")
    reasons = pd.Series([r for _, r in skipped]).value_counts()
    for r, n in reasons.items():
        lines.append(f"skipped {n}: {r}")
    return "\n".join(lines)


def load_summaries(exp: dict) -> pd.DataFrame:
    files = sorted((out_dir(exp) / "summary").glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
