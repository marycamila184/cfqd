"""Constraint-aware variation operators.

Shared by MAP-Elites, NS, NSLC, surprise and the MOC port, so that the only
difference between the algorithms is the selection mechanism.
"""
from __future__ import annotations

import numpy as np

from .schema import Constraint, DatasetSpec


def repair(x_new: np.ndarray, x_orig: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    """Project x_new onto the feasible set relative to the original instance."""
    x = spec.clip(x_new.copy())  # clip() also pulls a sentinel back into the valid range
    for j, f in enumerate(spec.features):
        if f.constraint.frozen or (f.sentinel is not None and x_orig[j] == f.sentinel):
            x[j] = x_orig[j]
        elif f.constraint == Constraint.INCREASE_ONLY and x[j] < x_orig[j]:
            x[j] = x_orig[j]
    return x


def mutate(
    x: np.ndarray,
    x_orig: np.ndarray,
    spec: DatasetSpec,
    rng: np.random.Generator,
    p_mut: float = 0.2,
    sigma: float = 0.1,
) -> np.ndarray:
    """Per-feature mutation.

    Numeric/ordinal: Gaussian noise with std sigma * range.
    Nominal: resampled uniformly from all categories (may keep the same value).
    At least one mutable feature is always selected for mutation; after
    repair() the result may still equal the input.
    """
    y = x.copy()
    mutable = np.flatnonzero(spec.mutable_mask)
    mask = rng.random(len(mutable)) < p_mut
    if not mask.any():
        mask[rng.integers(len(mutable))] = True
    for j in mutable[mask]:
        f = spec.features[j]
        if f.ftype.value == "nominal":
            y[j] = rng.integers(f.n_categories)
        else:
            y[j] = y[j] + rng.normal(0.0, sigma * (f.high - f.low or 1.0))
    return repair(y, x_orig, spec)


def crossover(
    a: np.ndarray, b: np.ndarray, x_orig: np.ndarray, spec: DatasetSpec, rng: np.random.Generator
) -> np.ndarray:
    """Uniform crossover."""
    take_a = rng.random(len(a)) < 0.5
    return repair(np.where(take_a, a, b), x_orig, spec)


def random_neighbor(x_orig, spec, rng, n_changes: int = 2, sigma: float = 0.3):
    """Starting point: perturb a few mutable features of the original instance."""
    y = x_orig.copy()
    mutable = np.flatnonzero(spec.mutable_mask)
    for j in rng.choice(mutable, size=min(n_changes, len(mutable)), replace=False):
        f = spec.features[j]
        if f.ftype.value == "nominal":
            y[j] = rng.integers(f.n_categories)
        else:
            y[j] = y[j] + rng.normal(0.0, sigma * (f.high - f.low or 1.0))
    return repair(y, x_orig, spec)
