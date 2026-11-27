"""Training of the three explained models (T5).

| kind  | model                          | used by                         |
|-------|--------------------------------|---------------------------------|
| lr    | logistic regression, PyTorch   | every method (Wachter included) |
| lr_sk | logistic regression, sklearn   | conference only (T5 criterion)  |
| rf    | random forest, sklearn         | model-agnostic methods only     |
| nn    | MLP, PyTorch, 2 hidden, ReLU   | every method (Wachter included) |

Wachter needs a gradient, so LR and NN are PyTorch. Every model takes the
SAME input as the rest of cfqd (float vector in spec order, categorical
codes 0..k-1) and exposes ``predict_proba``; the encoding (standardization,
one-hot) lives inside the model. ``CountingModel`` wraps any of them.

Files (plan, section 2; one folder per seed because one seed controls the
split, the training and the search)::

    models/<dataset>/seed<k>/logreg_pt.pt  logreg_sk.pkl  trees.pkl  mlp.pt  metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn

from .schema import DatasetSpec, FeatureType

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
FILES = {"lr": "logreg_pt.pt", "lr_sk": "logreg_sk.pkl", "rf": "trees.pkl", "nn": "mlp.pt"}
MODEL_KINDS = ("lr", "rf", "nn")      # the three explained models
DEFAULTS = {
    "lr": {"C": 1.0},
    "rf": {"n_estimators": 100, "min_samples_leaf": 5},
    "nn": {"hidden": [32, 16], "lr": 1e-3, "batch_size": 256, "max_epochs": 100,
           "patience": 10, "weight_decay": 1e-4},
}

# Published accuracies (Guidotti 2024, 70/30 split): compare as a RANGE, not equality.
GUIDOTTI_ACC = {
    "adult": {"rf": 0.85, "nn": 0.84},
    "german_credit": {"rf": 0.76, "nn": 0.81},
    "heloc": {"rf": 0.68, "nn": 0.67},     # raw FICO file (10,459 rows), not OpenML 45554
    "compas": {"rf": 0.56, "nn": 0.61},
    "compas_guidotti": {"rf": 0.56, "nn": 0.61},
}


# ================================================================ encoding
class Encoder:
    """Standardize numeric/ordinal features, one-hot nominal ones (fitted on train)."""

    def __init__(self, num_idx, nom_idx, nom_sizes, mean, std):
        self.num_idx = list(map(int, num_idx))
        self.nom_idx = list(map(int, nom_idx))
        self.nom_sizes = list(map(int, nom_sizes))
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)

    @classmethod
    def fit(cls, spec: DatasetSpec, X_train: np.ndarray) -> "Encoder":
        num = [j for j, f in enumerate(spec.features) if f.ftype != FeatureType.NOMINAL]
        nom = [j for j, f in enumerate(spec.features) if f.ftype == FeatureType.NOMINAL]
        sizes = [int(spec.features[j].n_categories) for j in nom]
        mean = X_train[:, num].mean(axis=0) if num else np.zeros(0)
        std = X_train[:, num].std(axis=0) if num else np.ones(0)
        std = np.where(std > 0, std, 1.0)
        return cls(num, nom, sizes, mean, std)

    @property
    def dim(self) -> int:
        return len(self.num_idx) + sum(self.nom_sizes)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        parts = [(X[:, self.num_idx] - self.mean) / self.std]
        for j, k in zip(self.nom_idx, self.nom_sizes):
            codes = np.clip(np.rint(X[:, j]).astype(int), 0, k - 1)
            parts.append(np.eye(k)[codes])
        return np.hstack(parts)

    def state(self) -> dict:
        return dict(num_idx=self.num_idx, nom_idx=self.nom_idx, nom_sizes=self.nom_sizes,
                    mean=self.mean.tolist(), std=self.std.tolist())


class EncodedNet(nn.Module):
    """Raw cfqd vector -> encoding -> MLP (hidden=[] gives logistic regression).

    The encoding is differentiable w.r.t. numeric/ordinal inputs, which is
    what Wachter optimizes; nominal inputs go through a (constant) one-hot.
    """

    def __init__(self, enc: Encoder, hidden: list[int]):
        super().__init__()
        self.enc_state = enc.state()
        self.hidden = list(hidden)
        self.num_idx = enc.num_idx
        self.nom = list(zip(enc.nom_idx, enc.nom_sizes))
        self.register_buffer("mean", torch.tensor(enc.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(enc.std, dtype=torch.float32))
        layers, d = [], enc.dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        parts = [(X[:, self.num_idx] - self.mean) / self.std]
        for j, k in self.nom:
            codes = torch.clamp(torch.round(X[:, j]).long(), 0, k - 1)
            parts.append(nn.functional.one_hot(codes, k).to(X.dtype))
        return torch.cat(parts, dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(self.encode(X)).squeeze(-1)  # logit of the favorable class


# ================================================================ wrappers
class TorchModel:
    differentiable = True

    def __init__(self, net: EncodedNet, kind: str):
        self.net = net.eval()
        self.kind = kind

    def proba_torch(self, X: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.net(X))
        return torch.stack([1 - p, p], dim=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Xt = torch.as_tensor(np.atleast_2d(np.asarray(X, dtype=np.float32)))
            return self.proba_torch(Xt).numpy().astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        torch.save({"kind": self.kind, "enc": self.net.enc_state, "hidden": self.net.hidden,
                    "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: Path) -> "TorchModel":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        e = ck["enc"]
        enc = Encoder(e["num_idx"], e["nom_idx"], e["nom_sizes"], e["mean"], e["std"])
        net = EncodedNet(enc, ck["hidden"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck["kind"])


class SklearnModel:
    differentiable = False

    def __init__(self, est, encoder: Encoder | None, kind: str):
        self.est = est
        self.encoder = encoder
        self.kind = kind

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self.est.predict_proba(self.encoder.transform(X) if self.encoder else X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "SklearnModel":
        return joblib.load(path)


# ================================================================ training
def train_lr_torch(X, y, enc: Encoder, seed: int, C: float = 1.0) -> TorchModel:
    """Full-batch L-BFGS on the same objective as sklearn's LogisticRegression(C):
    mean log-loss + ||w||^2 / (2 C n), intercept not penalized."""
    torch.manual_seed(seed)
    net = EncodedNet(enc, hidden=[])
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    lin = net.net[0]
    opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=500, tolerance_grad=1e-7,
                            tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss()
    reg = 1.0 / (2.0 * C * len(y))

    def closure():
        opt.zero_grad()
        loss = bce(net(Xt), yt) + reg * (lin.weight ** 2).sum()
        loss.backward()
        return loss

    net.train()
    for _ in range(5):
        opt.step(closure)
    return TorchModel(net, "lr")


def train_lr_sklearn(X, y, enc: Encoder, seed: int, C: float = 1.0) -> SklearnModel:
    est = LogisticRegression(C=C, max_iter=5000, random_state=seed).fit(enc.transform(X), y)
    return SklearnModel(est, enc, "lr_sk")


def train_rf(X, y, seed: int, n_estimators: int = 100, min_samples_leaf: int = 5) -> SklearnModel:
    """Trees take the integer codes directly (no one-hot)."""
    est = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                 n_jobs=-1, random_state=seed).fit(X, y)
    est.set_params(n_jobs=1)  # methods query small batches: threading overhead dominates
    return SklearnModel(est, None, "rf")


def train_mlp(X, y, enc: Encoder, seed: int, hidden=(32, 16), lr: float = 1e-3,
              batch_size: int = 256, max_epochs: int = 100, patience: int = 10,
              weight_decay: float = 1e-4) -> TorchModel:
    """Adam + early stopping on a 10% stratified validation split of the TRAIN set."""
    torch.manual_seed(seed)
    tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=seed)
    Xtr = torch.as_tensor(X[tr], dtype=torch.float32)
    ytr = torch.as_tensor(y[tr], dtype=torch.float32)
    Xva = torch.as_tensor(X[va], dtype=torch.float32)
    yva = torch.as_tensor(y[va], dtype=torch.float32)
    net = EncodedNet(enc, hidden=list(hidden))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(len(ytr), generator=gen)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            bce(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(bce(net(Xva), yva))
        if v < best - 1e-5:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return TorchModel(net, "nn")


# ============================================================== train/load
def model_dir(dataset: str, seed: int) -> Path:
    return MODELS_DIR / dataset / f"seed{seed}"


def _train(kind: str, ds, params: dict):
    X, y = ds.X_train, ds.y_train
    p = {**DEFAULTS.get(kind.replace("_sk", ""), {}), **(params or {})}
    if kind == "rf":
        return train_rf(X, y, ds.seed, **p)
    enc = Encoder.fit(ds.spec, X)
    if kind == "lr":
        return train_lr_torch(X, y, enc, ds.seed, **p)
    if kind == "lr_sk":
        return train_lr_sklearn(X, y, enc, ds.seed, **p)
    if kind == "nn":
        return train_mlp(X, y, enc, ds.seed, **p)
    raise ValueError(f"unknown model kind {kind!r}")


def load_model(dataset: str, seed: int, kind: str):
    path = model_dir(dataset, seed) / FILES[kind]
    return TorchModel.load(path) if path.suffix == ".pt" else SklearnModel.load(path)


def get_model(ds, kind: str, params: dict | None = None, force: bool = False):
    """Load the model of (dataset, seed, kind), training and saving it if missing."""
    path = model_dir(ds.name, ds.seed) / FILES[kind]
    if path.exists() and not force:
        return load_model(ds.name, ds.seed, kind)
    model = _train(kind, ds, params or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    _write_metrics(ds, kind, model)
    return model


def evaluate(model, ds) -> dict:
    p_tr = model.predict(ds.X_train)
    p_te = model.predict(ds.X_test)
    yte = ds.y_test
    tpr = float(np.mean(p_te[yte == 1] == 1))
    tnr = float(np.mean(p_te[yte == 0] == 0))
    return dict(acc_train=float(np.mean(p_tr == ds.y_train)), acc_test=float(np.mean(p_te == yte)),
                balanced_acc_test=(tpr + tnr) / 2, majority_baseline=float(max(yte.mean(), 1 - yte.mean())),
                unfavorable_rate_test=float(np.mean(p_te == 0)))


def _write_metrics(ds, kind: str, model) -> None:
    path = model_dir(ds.name, ds.seed) / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    m[kind] = evaluate(model, ds)
    path.write_text(json.dumps(m, indent=2), encoding="utf-8")


def train_all(ds, kinds=("lr", "lr_sk", "rf", "nn"), params: dict | None = None,
              force: bool = False) -> dict:
    params = params or {}
    return {k: get_model(ds, k, params.get(k), force=force) for k in kinds}
