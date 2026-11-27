# CFQD

**A benchmark of Quality-Diversity algorithms for counterfactual explanations.**

Course project for INFO7070 (Explainable Machine Learning, PPGInf/UFPR, Prof. Anderson Ara).
Working title: *CFQD: Benchmarking Quality-Diversity Algorithms for Counterfactual Explanations*
(formerly *Beyond Optimization: Benchmarking Quality-Diversity Algorithms for Counterfactual Explanations*).

## The idea

Most counterfactual methods treat generation as **optimization**: they return one solution,
or a small set, and tend to converge to variations of the same change strategy.
CFQD evaluates **Quality-Diversity (QD)** algorithms, which search for a repertoire of solutions
that are both good and different from each other. All methods are compared under a common protocol.

The strategy map is the common yardstick: a **strategy × effort** grid.

- **Strategy:** for each group of actionable features, whether the group increased, decreased or did not change (3^g combinations).
- **Effort:** the normalized magnitude of the change, in bins.

The counterfactuals of **every** method, QD or baseline, are projected onto this grid to measure
coverage and QD-score. MAP-Elites is the only method that uses the grid during the search.

### Research questions

1. **QD vs. optimization:** do QD algorithms find more distinct strategies than DiCE, Wachter, FACE and MOC, without losing validity, proximity and plausibility?
2. **QD vs. QD:** among MAP-Elites, NS, NSLC and surprise, which works best for counterfactuals, and does it depend on the dataset and the model?

## Methods

| Family | Method | Idea | Models |
| --- | --- | --- | --- |
| QD | MAP-Elites | keeps the best solution in each grid cell | LR, RF, NN |
| QD | Novelty Search (NS) | rewards novel behavior, ignoring quality | LR, RF, NN |
| QD | NSLC | novelty plus local competition on quality | LR, RF, NN |
| QD | Surprise / Novelty + Surprise | rewards deviation from predicted behavior | LR, RF, NN |
| Baseline | Wachter (2017) | smallest change via gradient; returns 1 counterfactual | LR, NN (no gradient for RF) |
| Baseline | DiCE (2020) | set with a diversity term (`genetic` mode for all models) | LR, RF, NN |
| Baseline | FACE (2020) | paths through dense regions of the data | LR, RF, NN (after adaptation) |
| Baseline | MOC (2020) | NSGA-II with 4 objectives; returns the Pareto front | LR, RF, NN |

LR = logistic regression, RF = random forest, NN = shallow neural network.

## Datasets

| Dataset | Source | Role |
| --- | --- | --- |
| South German Credit | UCI 522 (not the classic Statlog one) | main; validates the MOC port |
| Adult | UCI 2 (no `fnlwgt`; `education-num` only) | main |
| HELOC | OpenML **45554** v2 (rows with only −9 dropped; −9/−8/−7 as missing) | main |
| Give Me Some Credit | OpenML 45577 (= Kaggle data) | optional |
| COMPAS | ProPublica (`compas.yaml` strict, `compas_guidotti.yaml` runnable) | appendix |
| Simulation | `configs/simulation.yaml` (known routes, parameters frozen 2026-11-20) | simulation study |

`data/manifest.jsonl` pins id, version and md5 of every dataset. `cfqd.datasets.build(name)` downloads and
preprocesses one dataset (cached under `data/`); `cfqd.datasets.load_dataset(name, seed)` returns the split.

## Layout

✅ = done · ➕ = to do

```text
cfqd/
├── configs/
│   ├── german_credit.yaml       ✅ checked against the T6 table and the UCI codebook
│   ├── adult.yaml  heloc.yaml  gmc.yaml                      ✅
│   ├── compas.yaml (strict, g=0)  compas_guidotti.yaml (g=1)  ✅ open decision 3
│   ├── simulation.yaml  simulation_shuffled.yaml             ✅ frozen generator parameters
│   └── experiments/ smoke.yaml  v1.yaml  full.yaml  simulation.yaml   ✅ batch grids
├── cfqd/
│   ├── schema.py                ✅ features, types, constraints, groups, ranges and MAD (+ sentinel)
│   ├── datasets.py              ✅ download (md5), preprocessing, HELOC tests, YAML validator, manifest
│   ├── models.py                ✅ LR and NN in PyTorch, LR sklearn (check), RF; one predict_proba
│   ├── model.py                 ✅ CountingModel: query counter and budget (+ charge())
│   ├── operators.py             ✅ constraint-aware mutation and crossover (shared with MOC)
│   ├── descriptor.py            ✅ strategy × effort and feasible cells
│   ├── archive.py               ✅ grid, coverage, QD-score, baseline projection
│   ├── metrics.py               ✅ distance, L0, validity, redundancy, violations, diversity
│   │                            ➕ plausibility (nearest neighbor, LOF), yNN
│   ├── results.py               ✅ CSV with one row per counterfactual
│   ├── experiment.py            ✅ batch logic: grid, registry, budget curves, summaries, manifest
│   ├── simulation.py            ✅ generator with known routes + route recall
│   ├── plots.py                 ✅ budget-curve figures (fixed color per method)
│   └── methods/
│       ├── base.py              ✅ common interface
│       ├── map_elites.py        ✅ MAP-Elites (minimal version)
│       ├── novelty.py           ➕ NS and NSLC
│       ├── surprise.py          ➕ surprise and novelty + surprise
│       ├── moc.py               ➕ port with pymoo (moc_original and moc_shared)
│       ├── dice.py              ✅ dice-ml adapter (genetic + random), anytime fallback
│       ├── wachter.py           ✅ CARLA port (same loss, lambda schedule and stop rule)
│       └── face.py              ✅ extracted from CARLA, predict_proba, CARLA bugs fixed
├── scripts/
│   ├── run_batch.py             ✅ datasets × models × methods × seeds × budgets (--dry-run, --jobs, resumable)
│   ├── build_report_data.py     ✅ aggregates results/full into the data files of the results page
│   ├── repair_manifest.py       ✅ rebuilds manifest entries lost by parallel appends (runs before 2026-11-21)
│   └── validate_moc.R           ➕ original MOC (counterfactuals package) to validate the port
├── report/
│   └── index.html               ✅ source of the group results page (5 datasets, budget curves, checks)
├── notebooks/
│   ├── 01_t6_datasets_and_constraints.ipynb    ✅ acquisition, HELOC tests, YAML validation, T6 checklist
│   ├── 02_t5_models.ipynb                      ✅ training, Guidotti accuracies, T5 checklist
│   ├── 03_t9_t11_baselines.ipynb               ✅ DiCE / Wachter / FACE, CARLA success-rate check
│   ├── 04_t13_batch_execution.ipynb            ✅ dry run, cost estimate, budget curves
│   ├── 05_simulation_local_optima.ipynb        ✅ route recall, shuffled-descriptor robustness
│   └── analysis.ipynb           ➕ Friedman, Nemenyi and map figures
├── examples/
│   ├── demo_synthetic.py        ✅ full pipeline on made-up data
│   └── simulation.py            ✅ simulation study, quick run
└── tests/
    ├── test_basic.py            ✅ constraints, feasible cells and budget
    ├── test_datasets.py         ✅ T6 criteria: schema, categories, groups, sentinel, leakage
    ├── test_methods.py          ✅ DiCE/Wachter/FACE: constraints, budget; LR torch vs sklearn
    ├── test_simulation.py       ✅ generator, route recall, shuffled descriptor
    ├── test_experiment.py       ✅ atomic per-configuration manifest, tolerant manifest reader
    └── test_metrics.py          ➕ one test per metric
```

### What is versioned and what is generated

| Path | In git | How to get it |
| --- | --- | --- |
| `data/manifest.jsonl` | yes | pins id, version and md5 of every dataset |
| `data/raw/`, `data/processed/` | no | `cfqd.datasets.build(name)` downloads (md5-checked) and preprocesses |
| `models/` | no | trained on demand per seed by `cfqd.models.get_model` (deterministic) |
| `results/full/summary/`, `results/full/manifest.jsonl` | yes | one row per instance × budget of the 700-configuration run (~12 MB) |
| `results/carla_check/summary/`, `results/carla_check/manifest.jsonl` | yes | the CARLA success-rate check (notebook 03) |
| `results/full/runs/` | no | one row per counterfactual (~200 MB); rerun `configs/experiments/full.yaml` (~6 h on 6 cores) or ask for the copy on the group Drive |
| `results/report/` | no | `python scripts/build_report_data.py` |
| other `results/*` (smoke, curve_demo, simulation_quick) | no | rebuilt by the notebooks |

## Division of work

| Who | Files | Plan tasks |
| --- | --- | --- |
| **Mary** | `operators.py`, `descriptor.py`, `archive.py`, `methods/map_elites.py`, `novelty.py`, `surprise.py` | T4, T8, T10 |
| **Marcelo** | `configs/*.yaml`, `datasets.py`, `models.py`, `methods/dice.py`, `wachter.py`, `face.py`, `scripts/run_batch.py` (with Mary) | T3, T5, T6, T9, T11, T13 |
| **Vitor** | `metrics.py`, `tests/test_metrics.py`, `scripts/validate_moc.R` | T1, T7, T12 |
| **Israel** | `notebooks/analysis.ipynb` (or Vitor), literature review, writing, references | T2, T14, T18 |
| **TBD** | `methods/moc.py`, `examples/simulation.py` | T11, simulation study |

Order that unblocks the rest: `datasets.py` and `models.py` → `dice.py` and `wachter.py` → `moc.py` → `novelty.py` and `surprise.py` → `run_batch.py`.

## How to run

```bash
pip install -r requirements.txt
python -m pytest -q tests
python examples/demo_synthetic.py
python scripts/run_batch.py --config configs/experiments/smoke.yaml            # ~2 min
python scripts/run_batch.py --config configs/experiments/v1.yaml --dry-run    # grid and cost
python scripts/run_batch.py --config configs/experiments/v1.yaml --jobs 4     # resumable
```

Registering a new method for the batch runner (NS, NSLC, surprise, MOC):

```python
from cfqd.experiment import register
register("novelty", lambda params, ds, exp: NoveltySearch(**params))   # needs_gradient=False
```

## Plugging in a new method

```python
from cfqd.methods.base import CFResult, CounterfactualMethod

class Dice(CounterfactualMethod):
    name = "dice_genetic"

    def _search(self, x, model, spec, desired, rng):
        # ALWAYS query the model through model.predict / model.predict_proba,
        # never through the original model: that is what counts the queries.
        cfs = ...  # array (k, d), in the same encoding as spec
        return CFResult(self.name, cfs, n_queries=0, seconds=0.0)
```

`explain()` resets the counter, applies the budget, measures time and fills in `n_queries` and `seconds`.
External libraries that call the model on their own (such as DiCE) must receive the
`CountingModel` as their model, or a wrapper that forwards calls to it.

Projecting any method onto the grid:

```python
arch = GridArchive(Descriptor(spec), x)
arch.add_many(res.cfs[valid_mask])
arch.coverage(), arch.qd_score(), arch.n_strategies()
```

## Decisions made

- **Language:** Python. R is used only to validate the MOC port.
- **MOC:** Python port using pymoo's NSGA-II, in two versions: `moc_original` (the R operators, for validation) and `moc_shared` (the CFQD operators, for the main comparison).
- **CARLA:** not a dependency; the FACE code is extracted and adapted.
- **DiCE:** `genetic` mode for all three models.
- **Wachter:** does not run on RF (no gradient). Statistics: a global Friedman test without Wachter, plus a separate analysis with LR and NN only.
- **Budget:** curves across budgets instead of a single budget. Baselines may use the extra budget.
- **Validity is a constraint:** only valid counterfactuals enter the grid. **Fitness:** `1 / (1 + Guidotti distance)`.
- **Coverage:** computed over **feasible** cells, excluding cells made impossible by constraints.
- **Metrics:** names and formulas from Guidotti (2024) and CARLA, plus redundancy.
- **Constraints:** immutable, increase-only, irreversible and excluded-from-mutation are distinct categories.
- **Sentinel (HELOC):** `Feature.sentinel` is a per-instance rule (in `operators.repair` and
  `metrics.constraint_violations`): an instance holding the sentinel keeps it; others can never move into it.
- **Target:** `y` is re-coded so that 1 = favorable for every dataset (`favorable_class` in the YAML is the raw
  label); `desired = 1` always.
- **Seeds:** one seed controls split (70/30 stratified), training and search; models live in `models/<dataset>/seed<k>/`.
- **Budget curves:** each budget is an independent run with the same search seed (exact archive snapshot for MAP-Elites).
- **Wachter:** CARLA's loss, Adam(0.01, amsgrad), 1000-step inner loop, lambda -= 0.05; the budget replaces the
  timeout; ordinal features optimized with straight-through rounding; constraints enforced.
- **DiCE:** anytime fallback: if the budget runs out, the valid members of the last evaluated population are returned.
  KD-tree init fix (`_kd_init_fill`): dice-ml's uncapped one-row-at-a-time random init stalled on COMPAS (a uniform
  draw is valid with probability 0.035%), so success was 0.00; random draws are now capped at a fifth of the remaining
  budget and the other retrieved neighbours fill the population. Identical to dice-ml where it already worked.
- **FACE:** up to 500 candidates (in graph order) are projected onto the constraints and checked (was 50).

## Open issues

- [ ] Fitness: include plausibility in elite quality? (plan, 5.3)
- [ ] Nominal features in the descriptor: today a change only in a nominal feature counts as "increased". Keep it, add a "changed" state, or handle it separately?
- [ ] Calibrate the effort bins on one dataset before the full run.
- [x] Check `german_credit.yaml` against the T6 constraint table.
- [x] HELOC: OpenML 45554 v2 confirmed (md5 pinned). Missing values: indicator + sentinel on the three big columns.
- [ ] HELOC: the "missing = never happened" test holds only for `MSinceMostRecentDelq` (95%); it FAILS for
  `NetFractionInstallBurden` (6%) and `MSinceMostRecentInqexcl7days` (54%). Their sentinel now means "unknown". Group decision.
- [ ] `train_predictions`: DiCE's KD-tree and FACE's graph label the training data uncounted ("free"). Keep, or count them?
- [ ] COMPAS: strict (g=0, nothing runs) vs Guidotti actionability (g=1). The plan's "accuracy > 0.65 = leakage" rule
  fires (0.67) although no leaking column is used; 0.67 matches Dressel & Farid (2018).
- [ ] German Credit NN: 0.74 (mean of 5 seeds) vs 0.81 in Guidotti; RF within range. Not tuned on the test set.
- [ ] Integer-valued numeric features (age, counts) are mutated to fractional values by `operators.mutate`
  (DiCE rounds them). Add an `integer` flag to `Feature` and round in `clip`?
- [ ] MOC owner (plan, pending 1).
- [ ] Wachter reaches 1.00 on Adult + NN vs 0.50 in CARLA (plan flag). Stop rule and lambda schedule are CARLA's;
  the difference is the model and CARLA's timeout. Check with CARLA's ANN architecture.
- [ ] Simulation study: small version by Dec 3; full design in the paper extension.
