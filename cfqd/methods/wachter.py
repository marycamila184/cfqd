"""Wachter et al. (2017), ported from CARLA's ``wachter_recourse`` (T9).

CARLA is NOT a dependency (frozen since 2022, Python 3.7). What is kept from
CARLA, because the literature check depends on it (plan 5.1: CARLA reports a
0.50 success rate on Adult + ANN; 1.00 would mean we changed the stopping
rule or the lambda schedule):

- loss = BCE(model probabilities, one-hot target) + lambda * L1 distance,
  measured in the min-max normalized space
- Adam(lr=0.01, amsgrad=True); inner loop of up to n_iter=1000 steps that
  stops as soon as P(desired) > 0.5 (the decision threshold)
- after every failed inner loop: lambda -= 0.05 (starting at 0.01)
- categorical features get no gradient: in CARLA they pass through
  torch.round, whose gradient is zero, so they never change. Here nominal
  features are simply held fixed (same behavior, without the dead parameters).

Deliberate deviations (documented for the paper):

- the stop is the QUERY BUDGET (one query per forward pass, counted with
  ``CountingModel.charge``), not CARLA's 0.5-minute timeout (kept as an option)
- ORDINAL features are optimized as continuous values and rounded with a
  straight-through estimator (CARLA has no ordinal type; freezing them would
  cripple Wachter on German Credit, where most features are ordinal)
- constraints are enforced (CARLA's Wachter ignores them, and its
  ``x_new.clone().clamp_(0, 1)`` is a no-op): frozen features are not
  optimized, increase-only features are clamped at their original value,
  everything is clamped to the valid range, and ``repair`` is applied at the end

Runs only on differentiable models (LR, NN). Never on RF: a limitation of the
method, not a failure of the benchmark.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..operators import repair
from ._utils import instance_bounds, minmax, nominal_mask, ordinal_mask
from .base import CFResult, CounterfactualMethod


class Wachter(CounterfactualMethod):
    name = "wachter"
    requires_gradient = True

    def __init__(self, lr: float = 0.01, lambda_: float = 0.01, lambda_step: float = 0.05,
                 n_iter: int = 1000, threshold: float = 0.5, max_outer: int = 100,
                 t_max_seconds: float | None = None):
        self.lr = lr
        self.lambda_ = lambda_
        self.lambda_step = lambda_step
        self.n_iter = n_iter
        self.threshold = threshold
        self.max_outer = max_outer
        self.t_max_seconds = t_max_seconds

    def _search(self, x, model, spec, desired, rng) -> CFResult:
        tm = model.model
        if not getattr(tm, "differentiable", False):
            raise TypeError("Wachter needs a differentiable model (LR or NN), not RF")
        torch.manual_seed(int(rng.integers(2**31)))
        self._x, self._spec, self._last, self._found = x, spec, x.copy(), 0

        lo, hi, frozen = instance_bounds(x, spec)
        low, span = minmax(spec)
        var = ~frozen & ~nominal_mask(spec) & (hi > lo)
        ords = torch.as_tensor(ordinal_mask(spec)[var])
        if not var.any():
            return CFResult(self.name, x[None].copy(), 0, 0.0)

        f32 = torch.float32
        low_t, span_t = torch.as_tensor(low, dtype=f32), torch.as_tensor(span, dtype=f32)
        x_t = torch.as_tensor(x, dtype=f32)
        s0 = (x_t - low_t) / span_t                      # original, normalized
        var_t = torch.as_tensor(var)
        s_lo = ((torch.as_tensor(lo, dtype=f32) - low_t) / span_t)[var_t]
        s_hi = ((torch.as_tensor(hi, dtype=f32) - low_t) / span_t)[var_t]
        v = s0[var_t].clone().requires_grad_(True)       # optimized coordinates
        target = torch.zeros(2, dtype=f32)
        target[desired] = 1.0
        bce = torch.nn.BCELoss()

        def build(v):
            xv = low_t[var_t] + v * span_t[var_t]
            xv = torch.where(ords, xv + (torch.round(xv) - xv).detach(), xv)  # straight-through
            x_new = x_t.clone()
            x_new[var_t] = xv
            return x_new

        model.charge(1)
        with torch.no_grad():
            p = float(tm.proba_torch(x_t[None])[0, desired])
        opt = torch.optim.Adam([v], lr=self.lr, amsgrad=True)
        lam, outer, t0 = self.lambda_, 0, time.perf_counter()
        while p <= self.threshold and outer < self.max_outer:
            it = 0
            while p <= self.threshold and it < self.n_iter:
                opt.zero_grad()
                x_new = build(v)
                model.charge(1)                          # one forward pass = one query
                prob = tm.proba_torch(x_new[None])[0]
                p = float(prob[desired].detach())
                self._last = x_new.detach().numpy().astype(float)
                self._found = model.n_queries
                cost = torch.sum(torch.abs((x_new - x_t) / span_t))   # L1, normalized space
                loss = bce(prob, target) + lam * cost
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v.copy_(torch.max(torch.min(v, s_hi), s_lo))
                it += 1
            lam -= self.lambda_step
            outer += 1
            if self.t_max_seconds and time.perf_counter() - t0 > self.t_max_seconds:
                break
        return self._result()

    def _result(self) -> CFResult:
        cf = repair(self._last, self._x, self._spec)
        return CFResult(self.name, cf[None], 0, 0.0, found_at=[self._found])

    def _on_budget_exceeded(self) -> CFResult:
        return self._result()  # the last iterate (valid or not), as CARLA returns it
