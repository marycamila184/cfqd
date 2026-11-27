"""FACE (Poyiadzi et al., 2020), extracted from CARLA's ``graph_search`` (T11).

The method needs no gradient: CARLA's restriction to TF/PyTorch came only from
how it called the model. Here the model is queried through ``predict_proba``
(``CountingModel``), so FACE runs on LR, RF and NN.

CARLA logic kept (FACE-KNN, CARLA defaults n_neighbors=50, frac=0.4, p_norm=2):

1. take a random fraction of the data (plus the factual)
2. build a kNN graph (``connectivity`` weights, i.e. hop counts) in the min-max /
   one-hot space; edges only link points that share the immutable categorical
   values ("paths only take same sex / same race routes")
3. candidates = points predicted in the desired class and reachable from x
4. among the candidates at the minimum graph distance, return the closest in L2

Bugs of the CARLA code NOT carried over (documented for the paper):

- ``np.multiply(adj, m1, m2)`` passes m2 as the *output* buffer, and the loop over
  immutable keys overwrites the matrices: only the last immutable feature was
  enforced. Here every frozen categorical feature is enforced.
- the ``for min_dist in [min_distance, inf, 0]`` loop keeps only the last
  condition; here candidates are ranked by (graph distance, L2 distance).
- CARLA builds a dense n x n adjacency; here the graph stays sparse.

Additions (needed by the benchmark protocol):

- frozen NUMERIC features (age, past delinquencies) cannot be matched exactly
  by data points, so the chosen point is projected with ``repair`` and its
  validity is checked with a counted query; the next candidate is tried if
  the projection lost the desired class (up to ``max_checks``)
- ``train_predictions``: "free" labels the sampled data with uncounted queries
  (the labels of the training data are treated as known, amortized over all
  instances); "counted" charges them to the budget, and the sample size then
  shrinks to fit the budget. OPEN DECISION for the group, shared with DiCE.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

from ..operators import repair
from ..schema import FeatureType
from ._utils import frozen_for, scaled_onehot
from .base import CFResult, CounterfactualMethod


class Face(CounterfactualMethod):
    name = "face_knn"

    def __init__(self, X_train: np.ndarray, mode: str = "knn", n_neighbors: int = 50,
                 radius: float = 0.25, frac: float = 0.4, max_pool: int = 2000,
                 weights: str = "connectivity", n_cfs: int = 1, max_checks: int = 500,
                 train_predictions: str = "free"):
        if mode not in ("knn", "epsilon"):
            raise ValueError("mode must be 'knn' or 'epsilon'")
        self.X_train = np.asarray(X_train, dtype=float)
        self.mode = mode
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.frac = frac
        self.max_pool = max_pool
        self.weights = weights
        self.n_cfs = n_cfs
        self.max_checks = max_checks
        self.train_predictions = train_predictions
        if mode == "epsilon":
            self.name = "face_eps"

    def _search(self, x, model, spec, desired, rng) -> CFResult:
        self._found: list[np.ndarray] = []
        self._found_at: list[int] = []
        # Only points that share x's frozen categorical values can be reached.
        # (equivalent to removing the edges between different groups: x's
        # connected component can only contain points of its own group)
        frz = frozen_for(x, spec)
        cat_frozen = [j for j, f in enumerate(spec.features)
                      if f.ftype != FeatureType.NUMERIC and frz[j]]
        same = np.all(self.X_train[:, cat_frozen] == x[cat_frozen], axis=1) if cat_frozen \
            else np.ones(len(self.X_train), dtype=bool)
        pool_idx = np.flatnonzero(same)
        n_pool = min(self.max_pool, int(round(self.frac * len(pool_idx))))
        counted = self.train_predictions == "counted"
        if counted and model.remaining is not None:
            n_pool = min(n_pool, model.remaining - self.max_checks)
        if n_pool < 2:
            return self._result()
        P = self.X_train[rng.choice(pool_idx, size=n_pool, replace=False)]
        pred = model.predict(P) if counted else model.predict_uncounted(P)

        nodes = np.vstack([x[None], P])
        Z = scaled_onehot(nodes, spec)
        mode = "connectivity" if self.weights == "connectivity" else "distance"
        if self.mode == "knn":
            G = kneighbors_graph(Z, n_neighbors=min(self.n_neighbors, len(Z) - 1), mode=mode)
        else:
            G = radius_neighbors_graph(Z, radius=self.radius, mode=mode)
        G = csr_matrix(G)
        dist = dijkstra(G, directed=False, indices=0)
        cand = np.flatnonzero((pred == desired) & np.isfinite(dist[1:])) + 1
        if len(cand) == 0:
            return self._result()
        l2 = np.linalg.norm(Z[cand] - Z[0], axis=1)
        order = cand[np.lexsort((l2, dist[cand]))]   # graph distance first, then L2
        for i in order[: self.max_checks]:
            y = repair(nodes[i], x, spec)
            if np.allclose(y, x):
                continue
            if model.predict(y[None])[0] == desired:
                self._found.append(y)
                self._found_at.append(model.n_queries)
                if len(self._found) >= self.n_cfs:
                    break
        return self._result()

    def _result(self) -> CFResult:
        d = self.X_train.shape[1]
        cfs = np.array(self._found) if self._found else np.empty((0, d))
        return CFResult(self.name, cfs, 0, 0.0, found_at=list(self._found_at))

    def _on_budget_exceeded(self) -> CFResult:
        return self._result()
