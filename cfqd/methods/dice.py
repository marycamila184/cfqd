"""DiCE (Mothilal et al., 2020) through ``dice-ml`` 0.12 (T9).

``method="genetic"`` is the main baseline (the closest to QD and it runs on all
three models); ``method="random"`` is the floor.

Every model query made by dice-ml goes through ``CountingModel`` via a small
proxy, so DiCE pays for its queries like every other method. One exception,
shared with FACE and controlled by ``train_predictions``:

- "free" (default): the genetic KD-tree initialization predicts the WHOLE
  training set for every query instance. Those predictions are served
  uncounted (the labels of the training data are treated as known and
  amortized). Counting them would charge N_train queries per instance
  (31k on Adult, 84k on GMC) and DiCE would fail at every budget.
- "counted": everything is charged.
OPEN DECISION for the group (plan, section 8).

Budget handling of ``method="random"`` (C9, 2026-11-21). dice-ml predicts the query
once and then batches of ``sample_size`` (1000) candidates, and always evaluates at
least two batches, so a run costs about 2,031 queries and returned nothing at budgets
<= 2,000. Two changes, neither of which touches a run that fits in its budget:

- ``sample_size`` = min(1000, (budget - 1 - total_cfs) // 2), so the two mandatory
  batches fit (still 1000 at budgets >= 2,021; set ``sample_size`` to override);
- anytime fallback: every valid candidate of an evaluated batch is kept; if the
  budget runs out, up to ``total_cfs`` of them are drawn at random, which is the rule
  dice-ml's random mode applies to the candidates it found.

Constraints, given that DiCE has no notion of "increase only":
- ``features_to_vary`` = mutable features (minus features holding their sentinel)
- ``permitted_range`` per instance: [x_j, high] for increase-only features,
  [low, high] otherwise; for categorical features the list of allowed codes.
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ..schema import Constraint, FeatureType
from ._utils import empty, instance_bounds
from .base import CFResult, CounterfactualMethod

os.environ.setdefault("TQDM_DISABLE", "1")  # dice-ml wraps every call in a tqdm bar
OUTCOME = "__favorable__"


class _Proxy:
    """sklearn-like model handed to dice-ml; forwards to the current CountingModel."""

    def __init__(self, names: list[str]):
        self.names = names
        self.counting = None
        self.free = False

    def _array(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.names].astype(float).to_numpy()

    def predict_proba(self, df):
        X = self._array(df)
        if self.free:
            return self.counting.predict_proba_uncounted(X)
        return self.counting.predict_proba(X)

    def predict(self, df):
        return self.predict_proba(df).argmax(axis=1)

    @contextmanager
    def uncounted(self):
        self.free = True
        try:
            yield
        finally:
            self.free = False


def _kd_init_fill(exp, budget_left):
    """Replacement for dice-ml 0.12 ``DiceGenetic.do_KD_init`` (same rules per row).

    dice-ml retrieves population_size * total_CFs nearest neighbours of the desired
    class, builds the population from the FIRST population_size only and, when those
    collapse into duplicates (non-varying features are set to the query's values),
    fills the gap with ``do_random_init``: uniform draws predicted ONE ROW AT A TIME
    until enough are VALID, with no limit. On COMPAS a uniform draw is valid with
    probability 0.035%, so DiCE burned the whole budget in that loop and returned
    nothing (CARLA check, 20/11).

    Here, in order: (1) the first population_size neighbours, exactly as dice-ml;
    (2) dice-ml's random draws, capped at a fifth of the remaining budget;
    (3) the remaining retrieved neighbours. The GA regrows a smaller population by
    crossover, so the population does not need to be full. dice-ml's uncapped random
    init is kept only if fewer than two candidates exist.
    """
    def do_KD_init(features_to_vary, query_instance, cfs, desired_class, desired_range):
        cfs = exp.label_encode(cfs).reset_index(drop=True)
        names = exp.data_interface.feature_names
        cont = set(exp.data_interface.continuous_feature_names)
        fr = exp.feature_range
        size = exp.population_size

        def project(kx):
            one = np.zeros(exp.data_interface.number_of_features)
            for jx, feature in enumerate(names):
                v, q = cfs.iat[kx, jx], query_instance[jx]
                if feature not in features_to_vary:
                    one[jx] = q
                elif feature in cont:
                    lo, hi = fr[feature][0], fr[feature][1]
                    one[jx] = v if lo <= v <= hi else (q if lo <= q <= hi else np.random.uniform(lo, hi))
                else:
                    one[jx] = v if v in fr[feature] else (q if q in fr[feature]
                                                          else np.random.choice(fr[feature]))
            return one

        rows, seen = [], set()

        def add(one):
            key = tuple(one)
            if key not in seen:
                seen.add(key)
                rows.append(one)

        for kx in range(min(size, len(cfs))):          # (1) dice-ml's own block
            add(project(kx))
        missing = size - len(rows)
        if missing > 0:                                # (2) capped random draws
            rem = budget_left()
            cap = 10 * missing if rem is None else min(10 * missing, max(rem // 5, 0))
            for one in _random_valid(exp, missing, cap, features_to_vary, query_instance):
                add(one)
        kx = size
        while len(rows) < size and kx < len(cfs):     # (3) the other neighbours
            add(project(kx))
            kx += 1
        if len(rows) < 2:                              # dice-ml's original fallback
            for one in exp.do_random_init(size - len(rows), features_to_vary, query_instance,
                                          desired_class, desired_range):
                add(one)
        exp.cfs = np.array(rows)
    return do_KD_init


def _random_valid(exp, n: int, cap: int, features_to_vary, query_instance) -> list:
    """Up to n valid uniform draws, trying at most ``cap`` single-row queries."""
    precisions = exp.data_interface.get_decimal_precisions()
    cont = exp.data_interface.continuous_feature_names
    out = []
    for _ in range(int(cap)):
        if len(out) >= n:
            break
        one = np.zeros(exp.data_interface.number_of_features)
        for jx, feature in enumerate(exp.data_interface.feature_names):
            if feature in features_to_vary:
                if feature in cont:
                    one[jx] = np.round(np.random.uniform(exp.feature_range[feature][0],
                                                         exp.feature_range[feature][1]), precisions[jx])
                else:
                    one[jx] = np.random.choice(exp.feature_range[feature])
            else:
                one[jx] = query_instance[jx]
        if exp.is_cf_valid(exp.predict_fn_scores(one)):
            out.append(one)
    return out


class Dice(CounterfactualMethod):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, spec, method: str = "genetic",
                 total_cfs: int = 10, train_predictions: str = "free", kd_fill: bool = True,
                 **gen_kwargs):
        self.method = method
        self.name = f"dice_{method}"
        self.total_cfs = total_cfs
        self.train_predictions = train_predictions
        self.kd_fill = kd_fill
        self.gen_kwargs = gen_kwargs
        self._spec = spec
        self._cat = [f.name for f in spec.features if f.ftype != FeatureType.NUMERIC]
        self._num = [f.name for f in spec.features if f.ftype == FeatureType.NUMERIC]
        self._int_num = [f.name for j, f in enumerate(spec.features)
                         if f.ftype == FeatureType.NUMERIC
                         and np.all(np.mod(X_train[:, j], 1) == 0)]
        self._train_df = self._frame(X_train)
        self._train_df[OUTCOME] = np.asarray(y_train).astype(int)
        self._exp = None
        self._last_pop = None
        self._random_found: list[np.ndarray] = []
        self._proxy = _Proxy(spec.names)

    # ------------------------------------------------------------ frames
    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(np.atleast_2d(X), columns=self._spec.names)
        for c in self._cat:
            df[c] = df[c].round().astype(int).astype(str)
        for c in self._int_num:
            df[c] = df[c].round().astype(np.int64)
        return df

    def _explainer(self):
        if self._exp is None:
            import dice_ml
            import dice_ml.explainer_interfaces.explainer_base as eb
            eb.tqdm = lambda it, *a, **k: it  # silence the per-call progress bar
            data = dice_ml.Data(dataframe=self._train_df, continuous_features=self._num,
                                outcome_name=OUTCOME)
            m = dice_ml.Model(model=self._proxy, backend="sklearn", model_type="classifier")
            exp = dice_ml.Dice(data, m, method=self.method)
            if self.train_predictions == "free" and hasattr(exp, "build_KD_tree"):
                orig_kd = exp.build_KD_tree

                def kd_tree_uncounted(*a, **k):
                    with self._proxy.uncounted():
                        return orig_kd(*a, **k)
                exp.build_KD_tree = kd_tree_uncounted
            if self.kd_fill and hasattr(exp, "do_KD_init"):
                exp.do_KD_init = _kd_init_fill(exp, lambda: self._proxy.counting.remaining)
            if hasattr(exp, "predict_fn_scores"):
                # Anytime fallback: remember the last population the GA evaluated
                # (those queries are already paid) so that a budget cut does not
                # throw it away. See _on_budget_exceeded.
                orig_scores = exp.predict_fn_scores

                def scores_spy(inp):
                    out = orig_scores(inp)
                    arr = np.asarray(inp)
                    if arr.ndim == 2 and len(arr) > 1:
                        self._last_pop = (arr.copy(), np.asarray(out).copy())
                    return out
                exp.predict_fn_scores = scores_spy
            if self.method == "random" and hasattr(exp, "predict_fn"):
                # Anytime fallback for the random mode: keep the valid candidates of
                # every evaluated batch (already paid for).
                orig_predict = exp.predict_fn

                def predict_spy(inp):
                    out = orig_predict(inp)
                    if isinstance(inp, pd.DataFrame) and len(inp) > 1:
                        valid = np.asarray(out).argmax(axis=1) == self._desired
                        if valid.any():
                            self._random_found.append(
                                inp[self._spec.names][valid].astype(float).to_numpy())
                    return out
                exp.predict_fn = predict_spy
            self._exp = exp
        return self._exp

    def _on_budget_exceeded(self) -> CFResult:
        """Valid members of the last evaluated population, closest first (the same
        rule DiCE applies to its final population), capped at total_cfs."""
        spec = self._spec
        if self.method == "random":
            if not self._random_found:
                return CFResult(self.name, empty(spec), 0, 0.0)
            cfs = np.unique(np.vstack(self._random_found), axis=0)
            if len(cfs) > self.total_cfs:
                cfs = cfs[np.sort(np.random.choice(len(cfs), self.total_cfs, replace=False))]
            return CFResult(self.name, cfs, 0, 0.0, found_at=[self._proxy.counting.n_queries] * len(cfs),
                            extra={"anytime_fallback": True})
        if self._last_pop is None:
            return CFResult(self.name, empty(spec), 0, 0.0)
        pop, scores = self._last_pop
        valid = np.asarray(scores).argmax(axis=1) == self._desired
        if not valid.any():
            return CFResult(self.name, empty(spec), 0, 0.0)
        df = self._exp.label_decode(pop[valid])
        cfs = np.unique(df[spec.names].astype(float).to_numpy(), axis=0)
        from ..metrics import guidotti_distance
        d = [guidotti_distance(self._x, c, spec) for c in cfs]
        cfs = cfs[np.argsort(d)][: self.total_cfs]
        return CFResult(self.name, cfs, 0, 0.0, found_at=[self._proxy.counting.n_queries] * len(cfs),
                        extra={"anytime_fallback": True})

    # ------------------------------------------------------------ search
    def _search(self, x, model, spec, desired, rng) -> CFResult:
        from raiutils.exceptions import UserConfigValidationException
        exp = self._explainer()
        self._proxy.counting = model
        self._last_pop, self._x, self._desired = None, x, int(desired)
        self._random_found = []
        s = int(rng.integers(2**31))
        random.seed(s)
        np.random.seed(s)

        lo, hi, frozen = instance_bounds(x, spec)
        to_vary, permitted = [], {}
        for j, f in enumerate(spec.features):
            if frozen[j] or hi[j] < lo[j]:
                continue
            to_vary.append(f.name)
            if f.ftype == FeatureType.NUMERIC:
                if f.name in self._int_num:
                    permitted[f.name] = [int(np.ceil(lo[j])), int(np.floor(hi[j]))]
                else:
                    permitted[f.name] = [float(lo[j]), float(hi[j])]
            else:
                seen = sorted(int(v) for v in self._train_df[f.name].unique())
                if f.constraint == Constraint.INCREASE_ONLY:
                    seen = [v for v in seen if v >= x[j]]
                permitted[f.name] = [str(v) for v in seen]
        if not to_vary:
            return CFResult(self.name, empty(spec), 0, 0.0)

        query = self._frame(x[None])
        kw = dict(total_CFs=self.total_cfs, desired_class=int(desired),
                  features_to_vary=to_vary, permitted_range=permitted, verbose=False)
        if self.method == "random":
            kw["random_seed"] = s
            if model.budget is not None:  # C9: the two mandatory batches must fit
                kw["sample_size"] = int(min(1000, max(1, (model.budget - 1 - self.total_cfs) // 2)))
        kw.update(self.gen_kwargs)
        try:
            out = exp.generate_counterfactuals(query, **kw)
        except UserConfigValidationException:  # "No counterfactuals found"
            return CFResult(self.name, empty(spec), 0, 0.0)
        ex = out.cf_examples_list[0]
        df = ex.final_cfs_df_sparse if ex.final_cfs_df_sparse is not None else ex.final_cfs_df
        if df is None or len(df) == 0:
            return CFResult(self.name, empty(spec), 0, 0.0)
        cfs = df[spec.names].astype(float).to_numpy()
        return CFResult(self.name, cfs, 0, 0.0, found_at=[model.n_queries] * len(cfs))
