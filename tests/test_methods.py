"""Baselines (T9/T11) and models (T5) on small synthetic data: interface, budget, constraints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from cfqd import Constraint, CountingModel, DatasetSpec, Feature, FeatureType
from cfqd import models as M
from cfqd.experiment import redundancy_batch
from cfqd.methods import Dice, Face, Wachter
from cfqd.metrics import constraint_violations, redundancy


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n = 1500
    X = np.column_stack([
        rng.normal(50, 15, n),        # income   numeric free
        rng.normal(20, 10, n),        # debt     numeric free
        rng.integers(0, 5, n),        # education ordinal increase-only
        rng.integers(0, 4, n),        # housing  nominal free
        rng.integers(18, 70, n),      # age      immutable
        rng.integers(0, 3, n),        # delinq   ordinal irreversible
    ]).astype(float)
    y = ((X[:, 0] - X[:, 1] + 5 * X[:, 2] - 8 * X[:, 5] + rng.normal(0, 8, n)) > 35).astype(int)
    spec = DatasetSpec(name="syn", target="y", groups=["income", "debt", "education"], features=[
        Feature("income", FeatureType.NUMERIC, Constraint.FREE, "income"),
        Feature("debt", FeatureType.NUMERIC, Constraint.FREE, "debt"),
        Feature("education", FeatureType.ORDINAL, Constraint.INCREASE_ONLY, "education"),
        Feature("housing", FeatureType.NOMINAL, Constraint.FREE, "income"),
        Feature("age", FeatureType.NUMERIC, Constraint.IMMUTABLE),
        Feature("delinq", FeatureType.ORDINAL, Constraint.IRREVERSIBLE),
    ]).fit_ranges(X)
    enc = M.Encoder.fit(spec, X)
    lr = M.train_lr_torch(X, y, enc, seed=0)
    rf = M.train_rf(X, y, seed=0, n_estimators=30)
    x = X[np.flatnonzero(lr.predict(X) == 0)[0]]
    return X, y, spec, enc, lr, rf, x


def test_torch_and_sklearn_lr_agree(setup):
    X, y, spec, enc, lr, rf, x = setup
    sk = M.train_lr_sklearn(X, y, enc, seed=0)
    assert abs(np.mean(lr.predict(X) == y) - np.mean(sk.predict(X) == y)) < 0.01


def test_models_share_predict_proba(setup):
    X, y, spec, enc, lr, rf, x = setup
    for m in (lr, rf):
        p = m.predict_proba(X[:5])
        assert p.shape == (5, 2) and np.allclose(p.sum(axis=1), 1)


def test_torch_model_roundtrip(setup, tmp_path):
    X, y, spec, enc, lr, rf, x = setup
    lr.save(tmp_path / "m.pt")
    assert np.allclose(M.TorchModel.load(tmp_path / "m.pt").predict_proba(X[:20]), lr.predict_proba(X[:20]))


@pytest.mark.parametrize("make", [
    lambda X, y, spec: Wachter(),
    lambda X, y, spec: Face(X, n_neighbors=10),
    lambda X, y, spec: Dice(X, y, spec, total_cfs=4),
])
def test_method_respects_constraints_and_budget(setup, make):
    X, y, spec, enc, lr, rf, x = setup
    cm = CountingModel(lr)
    m = make(X, y, spec)
    res = m.explain(x, cm, spec, desired=1, budget=3000, seed=0)
    assert res.n_queries <= 3000
    assert res.cfs.ndim == 2 and res.cfs.shape[1] == spec.d
    for c in res.cfs:
        assert constraint_violations(x, c, spec) == 0
    assert len(res.cfs) > 0 and (cm.predict_uncounted(res.cfs) == 1).any()


@pytest.mark.parametrize("budget", [300, 1000, 2000])
def test_dice_random_works_below_its_fixed_cost(setup, budget):
    """C9: dice-ml's random mode needs ~2,031 queries with sample_size=1000; with the
    budget-aware sample size and the anytime fallback it returns valid CFs below that."""
    X, y, spec, enc, lr, rf, x = setup
    cm = CountingModel(lr)
    res = Dice(X, y, spec, method="random", total_cfs=4).explain(x, cm, spec, desired=1,
                                                                  budget=budget, seed=0)
    assert res.n_queries <= budget
    assert 0 < len(res.cfs) <= 4 and (cm.predict_uncounted(res.cfs) == 1).all()
    for c in res.cfs:
        assert constraint_violations(x, c, spec) == 0


def test_wachter_refuses_rf(setup):
    X, y, spec, enc, lr, rf, x = setup
    with pytest.raises(TypeError):
        Wachter().explain(x, CountingModel(rf), spec, desired=1, budget=100)


def test_wachter_counts_gradient_queries(setup):
    X, y, spec, enc, lr, rf, x = setup
    cm = CountingModel(lr)
    res = Wachter(lr=1e-6, lambda_step=0.0).explain(x, cm, spec, desired=1, budget=50)
    assert res.n_queries == 50 and len(res.cfs) == 1   # budget hit, last iterate returned


def test_redundancy_batch_matches_reference(setup):
    X, y, spec, enc, lr, rf, x = setup
    cm = CountingModel(lr)
    res = Face(X, n_neighbors=10, n_cfs=3).explain(x, cm, spec, desired=1, budget=500)
    ref = [redundancy(cm, x, c, 1) for c in res.cfs]
    assert list(redundancy_batch(cm, x, res.cfs, 1)) == ref
