"""Re-run of the full grid on the committed code, plus the full simulation (started 2026-11-21).

    python scripts/rerun_pipeline.py            # run (resumable; keeps Windows awake while it runs)
    python scripts/rerun_pipeline.py --status   # progress, ETA, failures, memory, C4 check so far

Stages, in order:
  1. c4_sample   part of results/full re-run first, so the reproducibility check (C4) has an
                 early answer: every configuration whose manifest code hash is not one of the two
                 main ones, plus one per (dataset, method, main hash);
  2. simulation  configs/experiments/simulation.yaml (3 seeds, 10 instances) -> results/simulation;
  3. full        the rest of configs/experiments/full.yaml -> results/full_rerun, one dataset at a time;
  4. retry       configurations that failed, once more, with 2 processes;
  5. compare     scripts/compare_runs.py -> results/full_rerun/compare_with_full.csv.
The number of processes is chosen at the start of every chunk from the available memory
(about 0.9 GB per worker on GMC, 0.5 GB elsewhere), between 2 and 6. Do not edit cfqd/ while this runs: the workers
import it again at the start of every chunk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cfqd.experiment import expand_grid, is_done, load_experiment, run_grid  # noqa: E402

FULL_OUT = "results/full_rerun"
STATE = REPO / FULL_OUT / "pipeline_state.json"
LOG = REPO / FULL_OUT / "pipeline.log"
MAIN_HASHES = {"src:6f60dc36cd", "src:2144fd74c7"}
# 0.45 GB was measured on a fresh process; workers grow with every configuration they run
# (1.24 GB private on GMC after ~10 configurations), which made Windows thrash on 2026-11-22.
# Hence a larger estimate, at least 2 processes, and a new pool per (dataset, model) in stage 3.
GB_PER_WORKER = {"gmc": 0.9}   # GMC has 150k rows; the other datasets fit in about 0.5 GB
GB_DEFAULT = 0.5
MIN_JOBS, MAX_JOBS = 2, 6


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def full_exp() -> dict:
    exp = load_experiment(REPO / "configs/experiments/full.yaml")
    exp["out"] = FULL_OUT
    return exp


def sim_exp() -> dict:
    return load_experiment(REPO / "configs/experiments/simulation.yaml")


def c4_sample() -> set[str]:
    man = [json.loads(x) for x in (REPO / "results/full/manifest.jsonl").read_text(encoding="utf-8").splitlines()
           if x.strip()]
    pick = {m["key"] for m in man if m["code"] not in MAIN_HASHES}
    seen = set()
    for m in sorted(man, key=lambda m: (m["seed"], m["model"])):
        g = (m["dataset"], m["method"], m["code"])
        if m["code"] in MAIN_HASHES and g not in seen:
            seen.add(g)
            pick.add(m["key"])
    return pick


def jobs_now(dataset: str | None = None) -> int:
    avail = psutil.virtual_memory().available / 2**30
    return int(min(MAX_JOBS, max(MIN_JOBS, avail // GB_PER_WORKER.get(dataset, GB_DEFAULT))))


def keep_awake() -> None:
    """ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process lives."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def write_state(**kw) -> None:
    s = read_state()
    s.update(kw, updated=stamp())
    STATE.write_text(json.dumps(s, indent=1), encoding="utf-8")


def run_chunk(stage: str, chunk: str, exp: dict, only, dataset: str | None = None) -> None:
    jobs = jobs_now(dataset)
    write_state(stage=stage, chunk=chunk, jobs=jobs, chunk_started=stamp())
    avail = psutil.virtual_memory().available / 2**30
    print(f"\n=== {stamp()} | {stage} / {chunk} | {jobs} processes | {avail:.1f} GB available ===",
          flush=True)
    run_grid(exp, jobs=jobs, only=only)


def main() -> None:
    (REPO / FULL_OUT).mkdir(parents=True, exist_ok=True)
    keep_awake()
    write_state(pid=os.getpid(), started=read_state().get("started") or stamp(), finished=None)
    full, sim = full_exp(), sim_exp()
    sample = c4_sample()

    run_chunk("1/5 c4_sample", f"{len(sample)} configurations", full, lambda k: k.stem in sample)
    run_chunk("2/5 simulation", f"{len(expand_grid(sim)[0])} configurations", sim, None)
    for ds in full["datasets"]:
        for model in full["models"]:  # a fresh pool per (dataset, model) releases worker memory
            if any(not is_done(k, full) for k in expand_grid(full)[0] if (k.dataset, k.model) == (ds, model)):
                run_chunk("3/5 full", f"{ds} / {model}", full,
                          lambda k, ds=ds, model=model: (k.dataset, k.model) == (ds, model), dataset=ds)
    for name, exp in (("full", full), ("simulation", sim)):
        if any(not is_done(k, exp) for k in expand_grid(exp)[0]):
            write_state(stage="4/5 retry", chunk=name, jobs=2, chunk_started=stamp())
            print(f"\n=== {stamp()} | 4/5 retry | {name} | 2 processes ===", flush=True)
            run_grid(exp, jobs=2)

    write_state(stage="5/5 compare", chunk="", chunk_started=stamp())
    subprocess.run([sys.executable, str(REPO / "scripts/compare_runs.py"),
                    "--out", f"{FULL_OUT}/compare_with_full.csv"], check=False)
    write_state(stage="done", chunk="", finished=stamp())
    print(f"\n=== {stamp()} | pipeline finished ===", flush=True)


# ------------------------------------------------------------------ status
def _weights(exp: dict, kind: str) -> dict:
    """Estimated seconds per configuration, from the manifests of earlier runs."""
    keys = expand_grid(exp)[0]
    if kind == "full":
        man = pd.read_json(REPO / "results/full/manifest.jsonl", lines=True)
        med = man.groupby(["dataset", "method"])["seconds"].median()
        sec = man.set_index("key")["seconds"]
        return {k.stem: float(sec.get(k.stem) if pd.notna(sec.get(k.stem)) else med.get((k.dataset, k.method), 300))
                for k in keys}
    quick = REPO / "results/simulation_quick/manifest.jsonl"
    est = {}
    if quick.exists():
        q = pd.read_json(quick, lines=True)
        est = q.groupby(["dataset", "model", "method"])["seconds"].mean().to_dict()
    scale = exp["n_instances"] / 5  # the quick run used 5 instances
    return {k.stem: float(est.get((k.dataset, k.model, k.method), 60)) * scale for k in keys}


def status() -> None:
    s = read_state()
    if not s:
        print("pipeline not started")
        return
    full, sim = full_exp(), sim_exp()
    wf, ws = _weights(full, "full"), _weights(sim, "sim")
    done_f = {k.stem for k in expand_grid(full)[0] if is_done(k, full)}
    done_s = {k.stem for k in expand_grid(sim)[0] if is_done(k, sim)}
    total = sum(wf.values()) + sum(ws.values())
    done = sum(wf[k] for k in done_f) + sum(ws[k] for k in done_s)
    frac = done / total
    alive = bool(s.get("pid")) and psutil.pid_exists(s["pid"])
    started = datetime.strptime(s["started"], "%Y-%m-%d %H:%M:%S")
    elapsed = (datetime.now() - started).total_seconds()
    eta = (datetime.now() + pd.Timedelta(seconds=elapsed * (1 - frac) / frac)).strftime("%H:%M") \
        if 0 < frac < 1 else "-"
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.exists() else ""
    failed = sorted({ln.split()[1] for ln in log.splitlines() if ln.startswith("FAILED ")})
    quiet = (time.time() - LOG.stat().st_mtime) / 60 if LOG.exists() else float("nan")
    vm = psutil.virtual_memory()
    sample = c4_sample()
    print(f"time {stamp()} | process {'ALIVE' if alive else 'NOT RUNNING'} (pid {s.get('pid')})"
          f" | stage {s.get('stage')} / {s.get('chunk')} | {s.get('jobs')} processes")
    print(f"progress {100 * frac:.1f}% (weighted by expected time) | started {s['started']}"
          f" | elapsed {elapsed / 3600:.1f} h | ETA {eta}")
    print(f"  c4_sample {len(done_f & sample)}/{len(sample)} | simulation {len(done_s)}/{len(ws)}"
          f" | full {len(done_f)}/{len(wf)}")
    print(f"failures {len(failed)}{': ' + ', '.join(failed[:10]) if failed else ''}"
          f" | log last written {quiet:.0f} min ago"
          f" | memory available {vm.available / 2**30:.1f} GB | CPU {psutil.cpu_percent(interval=1):.0f}%")
    if s.get("finished"):
        print(f"finished {s['finished']}")
    sys.path.insert(0, str(REPO / "scripts"))
    from compare_runs import compare, report
    print("\nC4 (reproducibility of results/full):")
    print(report(compare(REPO / "results/full", REPO / FULL_OUT)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    if ap.parse_args().status:
        status()
    else:
        main()
