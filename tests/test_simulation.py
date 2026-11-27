import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cfqd import simulation as S
from cfqd import datasets as D


def test_generator_is_deterministic():
    a, ma = S.generate({"n_samples": 800})
    b, mb = S.generate({"n_samples": 800})
    assert a.equals(b) and ma["generator_hash"] == mb["generator_hash"]


def test_every_route_produces_favorable_points():
    df, meta = S.generate({"n_samples": 4000})
    assert all(s > 0.05 for s in meta["favorable_share_by_route"])


def test_route_hits_and_recall():
    gt = S.GroundTruth(routes=[[0, 1], [2, 3]], signs=[1, 1], thresholds=[1.0, 3.0])
    x = np.zeros(4)
    near = np.array([1.5, 1.5, 0, 0])
    far = np.array([0, 0, 3.5, 3.5])
    assert gt.routes_hit(np.array([near])) == {0}
    s = gt.summary(x, np.array([near, far]))
    assert s["route_recall"] == 1.0 and s["far_routes_found"] == 1


def test_shuffled_descriptor_is_not_aligned():
    cfg = S.expand_config({"name": "s", "source": {"kind": "synthetic"},
                           "generator": {"descriptor": "shuffled"}})
    by_group = {}
    for f in cfg["features"]:
        if f.get("group"):
            by_group.setdefault(f["group"], set()).add(f["name"].split("_")[0])
    assert any(len(v) > 1 for v in by_group.values())   # groups mix routes
    assert D.validate_config(cfg) == []
