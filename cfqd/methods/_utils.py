"""Helpers shared by the baseline adapters (DiCE, Wachter, FACE)."""
from __future__ import annotations

import numpy as np

from ..schema import Constraint, DatasetSpec, FeatureType


def frozen_for(x: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    """Per-instance frozen mask: frozen constraints + features holding their sentinel."""
    return np.array([f.constraint.frozen or (f.sentinel is not None and x[j] == f.sentinel)
                     for j, f in enumerate(spec.features)])


def instance_bounds(x: np.ndarray, spec: DatasetSpec):
    """Feasible box around x: (lo, hi, frozen). Frozen features get lo = hi = x."""
    frozen = frozen_for(x, spec)
    lo = np.array([f.low for f in spec.features], dtype=float)
    hi = np.array([f.high for f in spec.features], dtype=float)
    for j, f in enumerate(spec.features):
        if frozen[j]:
            lo[j] = hi[j] = x[j]
        elif f.constraint == Constraint.INCREASE_ONLY:
            lo[j] = max(lo[j], x[j])
    return lo, hi, frozen


def minmax(spec: DatasetSpec):
    low = np.array([f.low for f in spec.features], dtype=float)
    span = np.array([(f.high - f.low) or 1.0 for f in spec.features], dtype=float)
    return low, span


def nominal_mask(spec: DatasetSpec) -> np.ndarray:
    return np.array([f.ftype == FeatureType.NOMINAL for f in spec.features])


def ordinal_mask(spec: DatasetSpec) -> np.ndarray:
    return np.array([f.ftype == FeatureType.ORDINAL for f in spec.features])


def scaled_onehot(X: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    """CARLA-style space for distances: min-max numeric/ordinal, one-hot nominal."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    low, span = minmax(spec)
    nom = nominal_mask(spec)
    parts = [(X[:, ~nom] - low[~nom]) / span[~nom]]
    for j in np.flatnonzero(nom):
        k = int(spec.features[j].n_categories)
        parts.append(np.eye(k)[np.clip(np.rint(X[:, j]).astype(int), 0, k - 1)])
    return np.hstack(parts)


def empty(spec: DatasetSpec) -> np.ndarray:
    return np.empty((0, spec.d))
