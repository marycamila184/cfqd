"""T6 acceptance criteria that can be checked without downloading anything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from cfqd import datasets as D
from cfqd import Constraint, DatasetSpec, Feature, FeatureType
from cfqd.metrics import constraint_violations
from cfqd.operators import mutate, repair

CONFIGS = D.list_dataset_configs()


def test_all_expected_configs_exist():
    for name in ["german_credit", "adult", "heloc", "gmc", "compas", "compas_guidotti", "simulation"]:
        assert name in CONFIGS


@pytest.mark.parametrize("name", CONFIGS)
def test_config_passes_schema_validator(name):
    assert D.validate_config(D.load_config(name)) == []


@pytest.mark.parametrize("name", CONFIGS)
def test_no_feature_in_two_constraint_categories(name):
    names = [f["name"] for f in D.load_config(name)["features"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", CONFIGS)
def test_every_feature_grouped_frozen_or_declared_out(name):
    cfg = D.load_config(name)
    out = set(cfg.get("ungrouped") or [])
    for f in cfg["features"]:
        frozen = Constraint(f.get("constraint", "free")).frozen
        assert f.get("group") or frozen or f["name"] in out, f["name"]


def test_validator_catches_errors():
    cfg = dict(name="t", target="y", favorable_class=1, source={"kind": "url"}, groups=["a", "b"],
               features=[dict(name="x", type="numeric", constraint="free", group="a"),
                         dict(name="x", type="numeric", constraint="immutable"),
                         dict(name="z", type="nominal", constraint="free"),
                         dict(name="w", type="numeric", constraint="free", group="c")])
    err = "\n".join(D.validate_config(cfg))
    assert "more than once" in err
    assert "z: mutable but in no group" in err
    assert "group 'c' is not declared" in err
    assert "group 'b' has no mutable feature" in err


def test_compas_has_no_label_leakage():
    for name in ("compas", "compas_guidotti"):
        names = {f["name"] for f in D.load_config(name)["features"]}
        assert not names & D.COMPAS_LEAKY


def test_heloc_missing_tests_logic():
    df = pd.DataFrame({
        "MSinceMostRecentDelq": [np.nan, np.nan, 3.0, 5.0],
        "PercentTradesNeverDelq": [100, 100, 80, 90],
        "NetFractionInstallBurden": [np.nan, 10.0, np.nan, 5.0],
        "PercentInstallTrades": [0, 20, 30, 10],
        "MSinceMostRecentInqexcl7days": [np.nan, 1.0, 2.0, np.nan],
        "NumInqLast6Mexcl7days": [0, 1, 1, 0],
    })
    t = D.heloc_missing_tests(df).set_index("column")
    assert t.loc["MSinceMostRecentDelq", "necessity"] == 1.0
    assert t.loc["NetFractionInstallBurden", "necessity"] == 0.5
    assert bool(t.loc["MSinceMostRecentDelq", "confirmed"])
    assert not bool(t.loc["NetFractionInstallBurden", "confirmed"])


# ------------------------------------------------------------ sentinel rule
@pytest.fixture
def sentinel_spec():
    X = np.array([[-1, 0, 0], [10, 5, 1], [3, 2, 0]], dtype=float)
    return DatasetSpec(name="s", target="y", groups=["g"], features=[
        Feature("months", FeatureType.NUMERIC, Constraint.INCREASE_ONLY, "g", sentinel=-1.0),
        Feature("burden", FeatureType.NUMERIC, Constraint.FREE, "g"),
        Feature("months_NA", FeatureType.NOMINAL, Constraint.EXCLUDED),
    ]).fit_ranges(X)


def test_sentinel_excluded_from_range(sentinel_spec):
    f = sentinel_spec.features[0]
    assert (f.low, f.high) == (3.0, 10.0)


def test_sentinel_is_frozen_and_never_entered(sentinel_spec):
    rng = np.random.default_rng(0)
    x_na = np.array([-1.0, 2.0, 1.0])   # "does not apply"
    x_ok = np.array([4.0, 2.0, 0.0])
    for _ in range(300):
        y = mutate(x_na, x_na, sentinel_spec, rng, p_mut=1.0, sigma=2.0)
        assert y[0] == -1.0
        y = mutate(x_ok, x_ok, sentinel_spec, rng, p_mut=1.0, sigma=2.0)
        assert y[0] >= 4.0 and y[0] != -1.0
    assert constraint_violations(x_ok, np.array([-1.0, 2.0, 0.0]), sentinel_spec) == 1
    assert constraint_violations(x_na, np.array([5.0, 2.0, 1.0]), sentinel_spec) == 1
    assert repair(np.array([-1.0, 2.0, 0.0]), x_ok, sentinel_spec)[0] == 4.0
