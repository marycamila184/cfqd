"""Metrics (plan 5.5), computed in a single place for every method.

Starting point for T12. Formulas follow Guidotti (2024) and CARLA.
To do: plausibility (nearest neighbor, LOF) and yNN; add a test for each metric.
"""
from __future__ import annotations

import numpy as np

from .schema import Constraint, DatasetSpec


def guidotti_distance(x: np.ndarray, x_cf: np.ndarray, spec: DatasetSpec) -> float:
    """MAD-weighted L1 on numeric features + fraction of differing categorical features."""
    num = [j for j, f in enumerate(spec.features) if not f.is_categorical]
    cat = [j for j, f in enumerate(spec.features) if f.is_categorical]
    d_num = 0.0
    if num:
        mads = np.array([spec.features[j].mad or 1.0 for j in num])
        d_num = float(np.mean(np.abs(x[num] - x_cf[num]) / mads))
    d_cat = float(np.mean(x[cat] != x_cf[cat])) if cat else 0.0
    w_num, w_cat = len(num) / spec.d, len(cat) / spec.d
    return w_num * d_num + w_cat * d_cat


def l0(x: np.ndarray, x_cf: np.ndarray) -> int:
    return int(np.sum(np.abs(x - x_cf) > 1e-9))


def validity(model, cfs: np.ndarray, desired: int) -> float:
    if len(cfs) == 0:
        return 0.0
    return float(np.mean(model.predict_uncounted(cfs) == desired))


def redundancy(model, x: np.ndarray, x_cf: np.ndarray, desired: int) -> int:
    """CARLA: how many changed features can be reverted without losing the desired class."""
    count = 0
    for j in np.flatnonzero(np.abs(x - x_cf) > 1e-9):
        y = x_cf.copy()
        y[j] = x[j]
        if model.predict_uncounted(y)[0] == desired:
            count += 1
    return count


def constraint_violations(x: np.ndarray, x_cf: np.ndarray, spec: DatasetSpec) -> int:
    v = 0
    for j, f in enumerate(spec.features):
        if f.sentinel is not None and (x[j] == f.sentinel) != (x_cf[j] == f.sentinel):
            v += 1  # left "does not apply", or moved into it
        elif f.constraint.frozen and abs(x_cf[j] - x[j]) > 1e-9:
            v += 1
        elif f.constraint == Constraint.INCREASE_ONLY and x_cf[j] < x[j] - 1e-9:
            v += 1
    return v


def mean_pairwise_distance(cfs: np.ndarray, spec: DatasetSpec) -> float:
    k = len(cfs)
    if k < 2:
        return 0.0
    ds = [guidotti_distance(cfs[i], cfs[j], spec) for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(ds))


def fitness(x: np.ndarray, x_cf: np.ndarray, spec: DatasetSpec) -> float:
    """Elite quality: validity is a constraint (only valid CFs enter the archive);
    among valid ones, closer is better. Bounded in (0, 1].
    OPEN DECISION (plan 5.3): include plausibility?"""
    return 1.0 / (1.0 + guidotti_distance(x, x_cf, spec))
