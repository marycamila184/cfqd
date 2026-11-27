"""Dataset acquisition, preprocessing and constraint files (T3/T6).

One YAML per dataset in ``configs/`` (same schema read by ``DatasetSpec.from_yaml``)
plus a few extra keys used only here:

    source            where the raw file comes from (kind: openml | uci | url | synthetic)
    favorable_class   raw label of the favorable class. ``y`` is ALWAYS re-coded as
                      1 = favorable, 0 = unfavorable, so ``desired = 1`` for every dataset
                      and the class is never inferred from the label order.
    base              reuse the raw data/preprocessing of another config (compas_guidotti)
    preprocessing     dataset-specific options (HELOC missing values, GMC tails...)
    ungrouped         mutable features deliberately outside every group

Layout (plan, section 2)::

    data/raw/<base>/                 file as downloaded + SOURCE.json (md5)
    data/processed/<name>/data.parquet  features (integer-coded categoricals) + y
    data/processed/<name>/meta.json  category mappings, rows dropped, HELOC tests...
    data/manifest.jsonl              one line per dataset: id, version, md5, rows

Every instance is a float vector in the order of the YAML ``features`` list;
categorical features are integer codes 0..k-1 (required by schema.py).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .schema import Constraint, DatasetSpec, Feature, FeatureType

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
DATA_DIR = REPO / "data"
TEST_SIZE = 0.3  # 70/30 stratified, same protocol as Guidotti (plan, section 2)

# COMPAS columns that leak the label (or are the COMPAS score itself).
COMPAS_LEAKY = {
    "two_year_recid", "is_recid", "is_violent_recid", "decile_score", "decile_score.1",
    "score_text", "v_decile_score", "v_score_text", "violent_recid", "event", "start", "end",
    "r_case_number", "r_charge_degree", "r_days_from_arrest", "r_offense_date",
    "r_charge_desc", "r_jail_in", "r_jail_out", "vr_case_number", "vr_charge_degree",
    "vr_offense_date", "vr_charge_desc",
}

# HELOC "does not apply" hypothesis (plan 3.3): if the column is missing, the
# sibling column (which has no missing values) should hold the expected value.
HELOC_SIBLINGS = {
    "MSinceMostRecentDelq": ("PercentTradesNeverDelq", 100),
    "NetFractionInstallBurden": ("PercentInstallTrades", 0),
    "MSinceMostRecentInqexcl7days": ("NumInqLast6Mexcl7days", 0),
}
HYPOTHESIS_THRESHOLD = 0.95

GERMAN_NAMES = {  # UCI 522 German column names -> English names (Groemping's codebook)
    "laufkont": "status", "laufzeit": "duration", "moral": "credit_history", "verw": "purpose",
    "hoehe": "amount", "sparkont": "savings", "beszeit": "employment_duration",
    "rate": "installment_rate", "famges": "personal_status_sex", "buerge": "other_debtors",
    "wohnzeit": "present_residence", "verm": "property", "alter": "age",
    "weitkred": "other_installment_plans", "wohn": "housing", "bishkred": "number_credits",
    "beruf": "job", "pers": "people_liable", "telef": "telephone", "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]


# =============================================================== configs
def config_path(name: str) -> Path:
    return CONFIG_DIR / f"{name}.yaml"


def load_config(name: str) -> dict:
    cfg = yaml.safe_load(config_path(name).read_text(encoding="utf-8"))
    if cfg.get("source", {}).get("kind") == "synthetic":
        from . import simulation
        cfg = simulation.expand_config(cfg)
    return cfg


def list_dataset_configs() -> list[str]:
    out = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if "features" in cfg or cfg.get("source", {}).get("kind") == "synthetic":
            out.append(p.stem)
    return out


def spec_from_config(cfg: dict) -> DatasetSpec:
    feats = [
        Feature(name=f["name"], ftype=FeatureType(f["type"]),
                constraint=Constraint(f.get("constraint", "free")),
                group=f.get("group"), sentinel=f.get("sentinel"))
        for f in cfg["features"]
    ]
    groups = cfg.get("groups")
    if groups is None:
        groups = sorted({f.group for f in feats if f.group})
    return DatasetSpec(name=cfg["name"], target=cfg["target"], features=feats, groups=list(groups))


VALID_TYPES = {t.value for t in FeatureType}
VALID_CONSTRAINTS = {c.value for c in Constraint}
REQUIRED_KEYS = ("name", "target", "favorable_class", "source", "groups", "features")


def validate_config(cfg: dict, columns: list[str] | None = None) -> list[str]:
    """Schema validator for the T6 constraint files. Returns a list of errors (empty = ok).

    Checks the T6 acceptance criteria:
      - required keys, valid types and constraint categories
      - no feature listed twice (a feature in two constraint categories at once)
      - every feature is in a group, OR frozen, OR declared in ``ungrouped``
      - every declared group exists and has at least one MUTABLE feature
      - sentinels only on numeric features; ``ungrouped`` only lists mutable, groupless features
      - if ``columns`` is given: the features are exactly the processed columns
    """
    err: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in cfg:
            err.append(f"missing key '{k}'")
    if err:
        return err
    names = [f.get("name") for f in cfg["features"]]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        err.append(f"features listed more than once: {dup}")
    groups = list(cfg["groups"] or [])
    ungrouped = set(cfg.get("ungrouped") or [])
    used_groups: dict[str, int] = {g: 0 for g in groups}
    for f in cfg["features"]:
        n = f.get("name")
        if f.get("type") not in VALID_TYPES:
            err.append(f"{n}: invalid type {f.get('type')!r}")
        c = f.get("constraint", "free")
        if c not in VALID_CONSTRAINTS:
            err.append(f"{n}: invalid constraint {c!r}")
            continue
        frozen = Constraint(c).frozen
        g = f.get("group")
        if g is not None and g not in used_groups:
            err.append(f"{n}: group {g!r} is not declared in 'groups'")
        if g is not None and g in used_groups and not frozen:
            used_groups[g] += 1
        if g is None and not frozen and n not in ungrouped:
            err.append(f"{n}: mutable but in no group and not declared in 'ungrouped'")
        if n in ungrouped and (g is not None or frozen):
            err.append(f"{n}: listed in 'ungrouped' but has a group or is frozen")
        if f.get("sentinel") is not None and f.get("type") != "numeric":
            err.append(f"{n}: sentinel is only supported on numeric features")
    for g, k in used_groups.items():
        if k == 0:
            err.append(f"group {g!r} has no mutable feature")
    missing_ungrouped = ungrouped - set(names)
    if missing_ungrouped:
        err.append(f"'ungrouped' lists unknown features: {sorted(missing_ungrouped)}")
    if columns is not None:
        cols = [c for c in columns if c != "y"]
        if set(cols) != set(names):
            err.append(f"features != data columns. only in YAML: {sorted(set(names) - set(cols))}; "
                       f"only in data: {sorted(set(cols) - set(names))}")
    return err


# ============================================================== download
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cfqd-benchmark"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def base_name(cfg: dict) -> str:
    return cfg.get("base", cfg["name"])


def raw_dir(cfg: dict) -> Path:
    return DATA_DIR / "raw" / base_name(cfg)


def fetch_raw(cfg: dict, force: bool = False) -> dict:
    """Download the raw file(s) once and record id/version/md5 in raw/<base>/SOURCE.json.

    OpenML: sklearn.fetch_openml verifies the ARFF md5 against the OpenML API;
    we also check it against the md5 pinned in the YAML.
    UCI / URL: the md5 is recorded on first download and verified afterwards.
    """
    src = cfg["source"]
    d = raw_dir(cfg)
    info_path = d / "SOURCE.json"
    if info_path.exists() and not force:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if src["kind"] in ("uci", "url"):
            f = d / info["file"]
            if f.exists() and _md5(f) == info["md5"]:
                return info
        elif src["kind"] == "openml":
            return info
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if src["kind"] == "openml":
        from sklearn.datasets import fetch_openml
        bunch = fetch_openml(data_id=int(src["id"]), data_home=str(d / "openml"), as_frame=True,
                             parser="pandas")
        det = bunch.details
        if str(src.get("md5", det["md5_checksum"])) != det["md5_checksum"]:
            raise ValueError(f"md5 mismatch for OpenML {src['id']}: "
                             f"YAML {src.get('md5')} vs OpenML {det['md5_checksum']}")
        if "version" in src and int(src["version"]) != int(det["version"]):
            raise ValueError(f"version mismatch for OpenML {src['id']}")
        bunch.frame.to_parquet(d / "frame.parquet")
        info = dict(kind="openml", id=int(det["id"]), name=det["name"], version=int(det["version"]),
                    md5=det["md5_checksum"], url=det["url"], file="frame.parquet",
                    n_rows=int(len(bunch.frame)), downloaded_at=now)
    elif src["kind"] in ("uci", "url"):
        url = src["url"]
        fname = url.rsplit("/", 1)[-1].replace("+", "_")
        dest = d / fname
        _download(url, dest)
        info = dict(kind=src["kind"], id=src.get("id"), url=url, file=fname, md5=_md5(dest),
                    size=dest.stat().st_size, downloaded_at=now)
        if fname.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(d)
    else:
        raise ValueError(f"unknown source kind {src['kind']!r}")
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ========================================================= preprocessing
def _codes(s: pd.Series) -> tuple[pd.Series, list]:
    """Alphabetical integer coding of a nominal column."""
    cats = sorted(s.dropna().unique().tolist(), key=str)
    return s.map({c: i for i, c in enumerate(cats)}).astype(float), [str(c) for c in cats]


def _favorable(y_raw: pd.Series, favorable) -> pd.Series:
    return (y_raw.astype(str).str.strip() == str(favorable)).astype(int)


def _prep_german(cfg: dict, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(raw_dir(cfg) / cfg["source"].get("member", "SouthGermanCredit.asc"), sep=" ")
    df = df.rename(columns=GERMAN_NAMES)
    y = _favorable(df.pop(cfg["target"]), cfg["favorable_class"])
    meta["code_shift"] = {}
    for f in cfg["features"]:
        if f["type"] != "numeric":
            shift = int(df[f["name"]].min())
            df[f["name"]] = df[f["name"]] - shift  # 1-based -> 0-based
            meta["code_shift"][f["name"]] = shift
    df["y"] = y
    return df


def _prep_adult(cfg: dict, meta: dict) -> pd.DataFrame:
    d = raw_dir(cfg)
    kw = dict(names=ADULT_COLUMNS, skipinitialspace=True, na_values="?", header=None)
    tr = pd.read_csv(d / "adult.data", **kw)
    te = pd.read_csv(d / "adult.test", skiprows=1, **kw)
    df = pd.concat([tr, te], ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    n0 = len(df)
    df = df.dropna().reset_index(drop=True)
    meta["rows_dropped_missing"] = int(n0 - len(df))
    df = df.drop(columns=["fnlwgt", "education"])
    meta["removed_columns"] = ["fnlwgt", "education"]
    y = _favorable(df.pop(cfg["target"]), cfg["favorable_class"])
    meta["categories"] = {}
    for f in cfg["features"]:
        if f["type"] == "nominal":
            df[f["name"]], meta["categories"][f["name"]] = _codes(df[f["name"]])
    df["y"] = y
    return df


def heloc_missing_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Plan 3.3: is "missing" in the three big columns "never happened"?

    necessity   = P(sibling == expected | column missing)   <- the hypothesis
    sufficiency = P(column missing | sibling == expected)   <- informative only
    """
    rows = []
    for col, (sib, val) in HELOC_SIBLINGS.items():
        miss = df[col].isna()
        hit = df[sib] == val
        rows.append(dict(
            column=col, missing=int(miss.sum()), missing_pct=round(100 * miss.mean(), 1),
            sibling=sib, expected=val,
            necessity=float(hit[miss].mean()) if miss.any() else float("nan"),
            sufficiency=float(miss[hit].mean()) if hit.any() else float("nan"),
            n_missing_but_sibling_differs=int((miss & ~hit).sum()),
        ))
    out = pd.DataFrame(rows)
    out["confirmed"] = out["necessity"] >= HYPOTHESIS_THRESHOLD
    return out


def _prep_heloc(cfg: dict, meta: dict) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir(cfg) / "frame.parquet")
    pp = cfg.get("preprocessing", {})
    y = _favorable(df.pop(cfg["target"]), cfg["favorable_class"])
    meta["rows_with_any_missing"] = int(df.isna().any(axis=1).sum())
    meta["rows_if_dropped"] = int((~df.isna().any(axis=1)).sum())
    # ordinal columns that arrive as nominal: -9 -> missing BEFORE ranking
    meta["ordinal_levels"] = {}
    for c in pp.get("ordinal_from_nominal", []):
        v = pd.to_numeric(df[c].astype(str), errors="coerce")
        v = v.where(v != -9)
        levels = sorted(v.dropna().unique().tolist())
        df[c] = v.map({lv: i for i, lv in enumerate(levels)})
        meta["ordinal_levels"][c] = [int(lv) for lv in levels]
    tests = heloc_missing_tests(df)
    meta["missing_tests"] = tests.to_dict(orient="records")
    for r in tests.itertuples():
        if not r.confirmed:
            warnings.warn(f"HELOC: 'does not apply' hypothesis NOT confirmed for {r.column} "
                          f"(necessity={r.necessity:.3f}); its sentinel means 'unknown'")
    sentinel = float(pp.get("sentinel", -1))
    ind_cols = pp.get("missing_indicator", [])
    for c in ind_cols:
        if (df[c].dropna() <= sentinel).any():
            raise ValueError(f"sentinel {sentinel} is inside the range of {c}")
        df[f"{c}_NA"] = df[c].isna().astype(float)
        df[c] = df[c].fillna(sentinel)
    meta["missing_pct_before_imputation"] = {
        c: round(100 * float(df[c].isna().mean()), 2) for c in df.columns if df[c].isna().any()}
    meta["imputed"] = {}
    types = {f["name"]: f["type"] for f in cfg["features"]}
    for c in df.columns:
        if df[c].isna().any():  # categorical codes take the mode, so they stay integer
            val = float(df[c].median() if types.get(c) == "numeric" else df[c].mode().iloc[0])
            df[c] = df[c].fillna(val)
            meta["imputed"][c] = val
    df["y"] = y
    return df


def _prep_gmc(cfg: dict, meta: dict) -> pd.DataFrame:
    df = pd.read_parquet(raw_dir(cfg) / "frame.parquet")
    pp = cfg.get("preprocessing", {})
    n0 = len(df)
    df = df.dropna().reset_index(drop=True)
    meta["rows_dropped_missing"] = int(n0 - len(df))
    dc = pp.get("drop_codes")
    if dc:
        bad = df[dc["columns"]].isin(dc["values"]).any(axis=1)
        df = df[~bad].reset_index(drop=True)
        meta["rows_dropped_error_codes"] = int(bad.sum())
    w = pp.get("winsorize_upper")
    if w:
        meta["winsorize_upper"] = {}
        for c in w["columns"]:
            cap = float(df[c].quantile(w["quantile"]))
            meta["winsorize_upper"][c] = {"cap": cap, "n_capped": int((df[c] > cap).sum())}
            df[c] = df[c].clip(upper=cap)
    y = _favorable(df.pop(cfg["target"]), cfg["favorable_class"])
    df = df.astype(float)
    df["y"] = y
    mr = pp.get("max_rows")
    if mr and len(df) > mr:
        df, _ = train_test_split(df, train_size=int(mr), stratify=df["y"], random_state=0)
        df = df.reset_index(drop=True)
        meta["subsampled_to"] = int(mr)
    return df


def _prep_compas(cfg: dict, meta: dict) -> pd.DataFrame:
    raw = pd.read_csv(raw_dir(cfg) / "compas-scores-two-years.csv")
    f = raw[(raw.days_b_screening_arrest <= 30) & (raw.days_b_screening_arrest >= -30)
            & (raw.is_recid != -1) & (raw.c_charge_degree != "O") & (raw.score_text != "N/A")]
    f = f.reset_index(drop=True)
    meta["rows_raw"], meta["rows_filtered"] = int(len(raw)), int(len(f))
    los = (pd.to_datetime(f["c_jail_out"]) - pd.to_datetime(f["c_jail_in"])).dt.days
    meta["length_of_stay_missing"] = int(los.isna().sum())
    df = pd.DataFrame({"length_of_stay": los.fillna(los.median()).clip(lower=0)})
    names = [x["name"] for x in cfg["features"]]
    leaky = sorted(set(names) & COMPAS_LEAKY)
    if leaky:
        raise ValueError(f"COMPAS label leakage: {leaky} cannot be inputs")
    meta["categories"] = {}
    for x in cfg["features"]:
        n = x["name"]
        if n == "length_of_stay":
            continue
        if x["type"] == "nominal":
            df[n], meta["categories"][n] = _codes(f[n])
        else:
            df[n] = f[n].astype(float)
    df["y"] = _favorable(f[cfg["target"]], cfg["favorable_class"])
    return df


def _prep_simulation(cfg: dict, meta: dict) -> pd.DataFrame:
    from . import simulation
    df, sim_meta = simulation.generate(cfg["generator"])
    meta.update(sim_meta)
    return df


PREPROCESSORS = {
    "german_credit": _prep_german, "adult": _prep_adult, "heloc": _prep_heloc,
    "gmc": _prep_gmc, "compas": _prep_compas, "simulation": _prep_simulation,
}


def processed_dir(name: str) -> Path:
    return DATA_DIR / "processed" / name


def build(name: str, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Download (if needed), preprocess, validate and cache one dataset."""
    out = processed_dir(name)
    if (out / "data.parquet").exists() and not force:
        return (pd.read_parquet(out / "data.parquet"),
                json.loads((out / "meta.json").read_text(encoding="utf-8")))
    cfg = load_config(name)
    kind = cfg["source"]["kind"]
    info = {"kind": "synthetic"} if kind == "synthetic" else fetch_raw(cfg)
    meta: dict = {"name": name, "base": base_name(cfg), "source": info}
    prep = PREPROCESSORS[base_name(cfg) if kind != "synthetic" else "simulation"]
    df = prep(cfg, meta)
    order = [f["name"] for f in cfg["features"]]
    errors = validate_config(cfg, columns=list(df.columns))
    if errors:
        raise ValueError(f"{name}: invalid constraint file:\n  " + "\n  ".join(errors))
    df = df[order + ["y"]].astype({c: float for c in order})
    if df[order].isna().any().any():
        raise ValueError(f"{name}: missing values left after preprocessing")
    meta.update(n_rows=int(len(df)), n_features=len(order),
                favorable_rate=round(float(df["y"].mean()), 4),
                built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "data.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    _update_manifest(name, meta)
    return df, meta


def _update_manifest(name: str, meta: dict) -> None:
    path = DATA_DIR / "manifest.jsonl"
    lines = []
    if path.exists():
        lines = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    lines = [r for r in lines if r.get("dataset") != name]
    src = meta.get("source", {})
    lines.append(dict(dataset=name, kind=src.get("kind"), id=src.get("id"),
                      version=src.get("version"), md5=src.get("md5"), url=src.get("url"),
                      n_rows=meta["n_rows"], n_features=meta["n_features"],
                      favorable_rate=meta["favorable_rate"], built_at=meta["built_at"],
                      generator_hash=meta.get("generator_hash")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, default=str) for r in lines) + "\n", encoding="utf-8")


# ================================================================ loading
@dataclass
class Dataset:
    name: str
    cfg: dict
    spec: DatasetSpec
    X: np.ndarray
    y: np.ndarray
    idx_train: np.ndarray
    idx_test: np.ndarray
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def X_train(self) -> np.ndarray:
        return self.X[self.idx_train]

    @property
    def X_test(self) -> np.ndarray:
        return self.X[self.idx_test]

    @property
    def y_train(self) -> np.ndarray:
        return self.y[self.idx_train]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.idx_test]

    @property
    def desired(self) -> int:
        return 1  # y is re-coded so that 1 is always the favorable class

    def frame(self, which: str = "all") -> pd.DataFrame:
        idx = {"all": slice(None), "train": self.idx_train, "test": self.idx_test}[which]
        df = pd.DataFrame(self.X[idx], columns=self.spec.names)
        df["y"] = self.y[idx]
        return df


def split_indices(y: np.ndarray, seed: int, test_size: float = TEST_SIZE):
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def load_dataset(name: str, seed: int = 0, test_size: float = TEST_SIZE) -> Dataset:
    """One seed controls the split (and, downstream, training and search)."""
    df, meta = build(name)
    cfg = load_config(name)
    spec = spec_from_config(cfg)
    X = df[spec.names].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    spec.fit_ranges(X)  # valid ranges and MAD come from the data itself
    tr, te = split_indices(y, seed, test_size)
    return Dataset(name=name, cfg=cfg, spec=spec, X=X, y=y, idx_train=np.sort(tr),
                   idx_test=np.sort(te), seed=seed, meta=meta)


def describe_spec(spec: DatasetSpec) -> pd.DataFrame:
    """Readable table of a fitted spec (for notebooks and the paper appendix)."""
    return pd.DataFrame([dict(feature=f.name, type=f.ftype.value, constraint=f.constraint.value,
                              group=f.group, low=f.low, high=f.high, n_categories=f.n_categories,
                              mad=f.mad, sentinel=f.sentinel) for f in spec.features])
