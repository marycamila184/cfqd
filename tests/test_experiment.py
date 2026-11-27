"""Batch runner (T13): output files and manifest robustness."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfqd import datasets as D
from cfqd import models as M
from cfqd.experiment import RunKey, is_done, read_manifest, run_config, write_manifest_jsonl


def test_read_manifest_skips_lines_broken_by_concurrent_appends(tmp_path):
    good = {"key": "a__lr__wachter__seed0", "seconds": 1.0, "finished_at": "2026-11-20T22:13:49+00:00"}
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(good) + "\n" + 'pu", "finished_at": "2026-11-20T22:13:51+00:00"}\n', encoding="utf-8")
    (tmp_path / "manifest").mkdir()
    newer = {"key": "b__lr__wachter__seed0", "seconds": 2.0, "finished_at": "2026-11-20T22:14:00+00:00"}
    (tmp_path / "manifest" / "b__lr__wachter__seed0.json").write_text(json.dumps(newer), encoding="utf-8")
    recs = read_manifest(tmp_path)
    assert [r["key"] for r in recs] == ["a__lr__wachter__seed0", "b__lr__wachter__seed0"]
    assert write_manifest_jsonl(tmp_path) == 2
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line) for line in lines) and len(lines) == 2


def test_run_config_writes_one_manifest_file_per_configuration(tmp_path, monkeypatch):
    # build the synthetic data and train the model in temporary folders, so the test never
    # touches the versioned data/manifest.jsonl or the repository's data/ and models/
    monkeypatch.setattr(D, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(M, "MODELS_DIR", tmp_path / "models")
    out = tmp_path / "results"
    exp = dict(name="t", datasets=["simulation"], models=["lr"], methods={"wachter": {}}, seeds=[0],
               budgets=[100], n_instances=1, out=str(out), train_predictions="free",
               model_params={})
    key = RunKey("simulation", "lr", "wachter", 0)
    entry = run_config(key, exp)
    assert is_done(key, exp)
    assert (out / "manifest" / f"{key.stem}.json").exists()
    assert not list(out.rglob("*.tmp"))
    assert read_manifest(out)[0]["key"] == entry["key"] == key.stem
    assert not (out / "manifest.jsonl").exists()   # only run_grid's main process writes it
    assert (tmp_path / "data" / "manifest.jsonl").exists()   # the build went to the temporary folder
