"""Simulation study with known routes, quick run (plan, section 7).

Builds the frozen synthetic dataset, trains the model and prints the route
recall of each method on a few unfavorable instances. The full study runs
through the batch script:

    python scripts/run_batch.py --config configs/experiments/simulation.yaml

Run from the repository root:  python examples/simulation.py [--model nn] [--budget 5000]
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402

from cfqd import CountingModel  # noqa: E402
from cfqd import datasets as D, models as M  # noqa: E402
from cfqd.methods import Dice, Face, MapElites, Wachter  # noqa: E402
from cfqd.simulation import GroundTruth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="simulation", choices=["simulation", "simulation_shuffled"])
ap.add_argument("--model", default="nn", choices=["lr", "rf", "nn"])
ap.add_argument("--budget", type=int, default=5000)
ap.add_argument("--n", type=int, default=5, help="number of unfavorable test instances")
a = ap.parse_args()

ds = D.load_dataset(a.dataset, seed=0)
truth = GroundTruth.from_meta(ds.meta)
cm = CountingModel(M.get_model(ds, a.model))
print(f"{a.dataset}: generator hash {ds.meta['generator_hash']}, model {a.model} "
      f"acc={M.evaluate(cm.model, ds)['acc_test']:.3f}, thresholds={truth.thresholds}")

methods = [MapElites(), Dice(ds.X_train, ds.y_train, ds.spec), Face(ds.X_train)]
if a.model != "rf":
    methods.append(Wachter())
pos = np.flatnonzero(cm.predict_uncounted(ds.X_test) == 0)[: a.n]
for m in methods:
    recall, near, far = [], [], []
    for p in pos:
        x = ds.X_test[p]
        res = m.explain(x, cm, ds.spec, desired=1, budget=a.budget, seed=int(p))
        V = res.cfs[cm.predict_uncounted(res.cfs) == 1] if len(res.cfs) else res.cfs
        s = truth.summary(x, V)
        recall.append(s["route_recall"])
        near.append(s["near_route_found"])
        far.append(s["far_routes_found"])
    print(f"  {m.name:13} route recall={np.mean(recall):.2f}  near route found={np.mean(near):.2f}  "
          f"far routes found (of {len(truth.routes) - 1})={np.mean(far):.2f}")
