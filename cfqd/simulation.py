"""Simulation study with KNOWN routes to flip the decision (plan, section 7).

Synthetic data with K routes: route k is a group of features whose (signed)
mean must exceed a threshold t_k. Route 0 is NEAR the typical instance
(t_0 = near_distance); the others are FAR (t_k = far_distance). Because the
routes are known by construction, we can measure how many of them each method
finds: route recall = distinct routes found / routes that exist.

Hypothesis to test: classical methods get stuck on the near route; QD finds all.

Guard against "data designed to favor our hypothesis" (Mary's concern):
- the generator and its parameters live in configs/simulation.yaml with a
  ``frozen_on`` date, and their hash is written to the data manifest and to
  every run manifest; they must be fixed BEFORE running any method
- ``descriptor: shuffled`` assigns the features to the descriptor groups at
  random, so the strategy grid is NOT aligned with the routes (robustness check)
- report the cases where QD does not find every route

Label rule (on the TRUE generative function, not the model)::

    margin_k(x) = sign_k * mean(x[route_k]) - t_k
    y = 1  iff  max_k (margin_k(x) + eps_k) > 0,   eps_k ~ N(0, label_noise)

Sampling: a base cloud x ~ N(0, I) (mostly unfavorable) plus, for each route,
a cloud shifted along that route across its boundary, so that every model
can learn every route.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULTS = dict(seed=20261120, n_samples=6000, n_routes=4, features_per_route=2, n_noise=2,
                n_immutable=1, near_distance=1.0, far_distance=3.0, label_noise=0.1,
                base_fraction=0.6, directions="up", descriptor="aligned")


def params_hash(params: dict) -> str:
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def _layout(p: dict):
    K, m = p["n_routes"], p["features_per_route"]
    names, routes = [], []
    for k in range(K):
        routes.append(list(range(len(names), len(names) + m)))
        names += [f"r{k}_f{i}" for i in range(m)]
    noise = [f"noise_{i}" for i in range(p["n_noise"])]
    immut = [f"immutable_{i}" for i in range(p["n_immutable"])]
    signs = [1 if (p["directions"] == "up" or k % 2 == 0) else -1 for k in range(K)]
    thresholds = [p["near_distance"]] + [p["far_distance"]] * (K - 1)
    return names + noise + immut, routes, signs, thresholds, noise, immut


def expand_config(cfg: dict) -> dict:
    """Fill features/groups of a synthetic config from its generator parameters."""
    p = {**DEFAULTS, **(cfg.get("generator") or {})}
    names, routes, signs, thr, noise, immut = _layout(p)
    K = p["n_routes"]
    route_feats = [j for r in routes for j in r]
    if p["descriptor"] == "aligned":
        group_of = {j: f"route_{k}" for k, r in enumerate(routes) for j in r}
        groups = [f"route_{k}" for k in range(K)]
    elif p["descriptor"] == "shuffled":
        rng = np.random.default_rng(p["seed"] + 1)
        perm = rng.permutation(route_feats)
        size = p["features_per_route"]
        group_of = {int(j): f"group_{i // size}" for i, j in enumerate(perm)}
        groups = [f"group_{k}" for k in range(K)]
    else:
        raise ValueError("descriptor must be 'aligned' or 'shuffled'")
    feats = []
    for j, n in enumerate(names):
        if n in immut:
            feats.append(dict(name=n, type="numeric", constraint="immutable"))
        elif n in noise:
            feats.append(dict(name=n, type="numeric", constraint="free"))
        else:
            feats.append(dict(name=n, type="numeric", constraint="free", group=group_of[j]))
    return {**cfg, "generator": p, "groups": groups, "ungrouped": list(noise),
            "features": feats, "target": cfg.get("target", "y"),
            "favorable_class": cfg.get("favorable_class", 1)}


@dataclass
class GroundTruth:
    routes: list
    signs: list
    thresholds: list

    def margins(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.column_stack([s * X[:, r].mean(axis=1) - t
                                for r, s, t in zip(self.routes, self.signs, self.thresholds)])

    def label(self, X: np.ndarray) -> np.ndarray:
        return (self.margins(X).max(axis=1) > 0).astype(int)

    def routes_hit(self, V: np.ndarray) -> set:
        if len(V) == 0:
            return set()
        return set(np.flatnonzero((self.margins(V) > 0).any(axis=0)).tolist())

    def summary(self, x: np.ndarray, V: np.ndarray) -> dict:
        """Per-instance route metrics over the VALID counterfactuals V."""
        hit = self.routes_hit(V)
        K = len(self.routes)
        return dict(n_routes=K, routes_found=len(hit), route_recall=len(hit) / K,
                    near_route_found=0 in hit, far_routes_found=len(hit - {0}),
                    cf_on_no_true_route=int(np.sum(self.margins(V).max(axis=1) <= 0)) if len(V) else 0)

    @classmethod
    def from_meta(cls, meta: dict) -> "GroundTruth":
        g = meta["ground_truth"]
        return cls(g["routes"], g["signs"], g["thresholds"])


def generate(params: dict) -> tuple[pd.DataFrame, dict]:
    p = {**DEFAULTS, **(params or {})}
    names, routes, signs, thr, noise, immut = _layout(p)
    rng = np.random.default_rng(p["seed"])
    n, d, K = p["n_samples"], len(names), p["n_routes"]
    n_base = int(round(p["base_fraction"] * n))
    X = [rng.normal(0, 1, (n_base, d))]
    per = (n - n_base) // K
    for k in range(K):
        Xk = rng.normal(0, 1, (per, d))
        shift = rng.uniform(0, thr[k] + 1.0, size=per)       # spread across the boundary
        Xk[:, routes[k]] += signs[k] * shift[:, None]
        X.append(Xk)
    X = np.vstack(X)
    gt = GroundTruth(routes, signs, thr)
    eps = rng.normal(0, p["label_noise"], (len(X), K))
    y = ((gt.margins(X) + eps).max(axis=1) > 0).astype(int)
    df = pd.DataFrame(X, columns=names)
    df["y"] = y
    df = df.sample(frac=1.0, random_state=p["seed"]).reset_index(drop=True)
    route_share = np.bincount(gt.margins(X[y == 1]).argmax(axis=1), minlength=K) / max(1, (y == 1).sum())
    meta = dict(generator=p, generator_hash=params_hash(p),
                ground_truth=dict(routes=routes, signs=signs, thresholds=thr),
                favorable_share_by_route=[round(float(s), 3) for s in route_share])
    return df, meta
