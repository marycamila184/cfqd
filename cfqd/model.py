"""Explained model wrapped with a query counter and a budget.

Every method (QD and baselines) queries the model through this wrapper. This
guarantees that queries are counted the same way for all of them, which the
budget curves require.
"""
from __future__ import annotations

import numpy as np


class BudgetExceeded(RuntimeError):
    """Raised when a method tries to exceed its query budget."""


class CountingModel:
    def __init__(self, model, budget: int | None = None):
        self.model = model
        self.budget = budget
        self.n_queries = 0          # each evaluated row counts as one query
        self.history: list[int] = []  # cumulative n_queries after each call

    def reset(self, budget: int | None = None) -> None:
        self.n_queries = 0
        self.history = []
        if budget is not None:
            self.budget = budget

    @property
    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.n_queries

    def charge(self, n: int) -> None:
        """Count n queries made outside predict_proba (e.g. Wachter's gradient
        forward passes, which go straight to the torch module)."""
        if self.budget is not None and self.n_queries + n > self.budget:
            raise BudgetExceeded(f"query budget of {self.budget} exhausted")
        self.n_queries += n
        self.history.append(self.n_queries)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self.charge(len(X))
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # Uncounted query: for evaluation (metrics) only, never inside a method.
    def predict_uncounted(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_uncounted(X).argmax(axis=1)

    def predict_proba_uncounted(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(np.asarray(X, dtype=float)))
