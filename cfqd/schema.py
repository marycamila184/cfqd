"""Data schema: features, types, constraints and actionable groups.

Every instance is a numpy float vector. Categorical features are encoded
as integers (0..n_categories-1) before they reach this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class FeatureType(str, Enum):
    NUMERIC = "numeric"        # continuous or integer
    ORDINAL = "ordinal"        # ordered categorical (has a direction)
    NOMINAL = "nominal"        # unordered categorical (no direction)


class Constraint(str, Enum):
    FREE = "free"                  # may go up or down
    IMMUTABLE = "immutable"        # never changes (race, sex, country...)
    INCREASE_ONLY = "increase_only"  # can only increase (age, education...)
    IRREVERSIBLE = "irreversible"  # past event no action can undo (delinquencies)
    EXCLUDED = "excluded"          # left out of mutation (fnlwgt, ExternalRiskEstimate)

    @property
    def frozen(self) -> bool:
        return self in (Constraint.IMMUTABLE, Constraint.IRREVERSIBLE, Constraint.EXCLUDED)


@dataclass
class Feature:
    name: str
    ftype: FeatureType
    constraint: Constraint = Constraint.FREE
    group: str | None = None       # actionable group used by the descriptor
    low: float | None = None       # valid range (filled in from the data)
    high: float | None = None
    n_categories: int | None = None
    mad: float | None = None       # median absolute deviation (Guidotti distance)
    # Out-of-range value meaning "does not apply" (HELOC missing values, T6 3.3).
    # Per-instance rule enforced by operators.repair: if the ORIGINAL instance
    # holds the sentinel the feature is frozen; otherwise it may never move to it.
    sentinel: float | None = None

    @property
    def is_categorical(self) -> bool:
        return self.ftype != FeatureType.NUMERIC

    @property
    def mutable(self) -> bool:
        return not self.constraint.frozen


@dataclass
class DatasetSpec:
    name: str
    target: str
    features: list[Feature]
    groups: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ util
    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def d(self) -> int:
        return len(self.features)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def group_indices(self, group: str) -> list[int]:
        return [i for i, f in enumerate(self.features) if f.group == group]

    @property
    def mutable_mask(self) -> np.ndarray:
        return np.array([f.mutable for f in self.features])

    # ---------------------------------------------------------- construction
    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetSpec":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        feats = [
            Feature(
                name=f["name"],
                ftype=FeatureType(f["type"]),
                constraint=Constraint(f.get("constraint", "free")),
                group=f.get("group"),
                sentinel=f.get("sentinel"),
            )
            for f in cfg["features"]
        ]
        groups = cfg.get("groups") or sorted({f.group for f in feats if f.group})
        return cls(name=cfg["name"], target=cfg["target"], features=feats, groups=groups)

    def fit_ranges(self, X: pd.DataFrame | np.ndarray) -> "DatasetSpec":
        """Extract valid ranges and MAD from the dataset itself (T3, decision 4)."""
        X = np.asarray(X, dtype=float)
        for j, f in enumerate(self.features):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if f.sentinel is not None:  # the sentinel is not part of the valid range
                col = col[col != f.sentinel]
            f.low, f.high = float(col.min()), float(col.max())
            if f.is_categorical:
                f.n_categories = int(f.high) + 1
            mad = float(np.median(np.abs(col - np.median(col))))
            f.mad = mad if mad > 0 else (float(col.std()) or 1.0)
        return self

    def clip(self, x: np.ndarray) -> np.ndarray:
        lo = np.array([f.low for f in self.features])
        hi = np.array([f.high for f in self.features])
        x = np.clip(x, lo, hi)
        cat = np.array([f.is_categorical for f in self.features])
        x[cat] = np.round(x[cat])
        return x
