# Beyond Optimization: Quality-Diversity Search for Counterfactual Explanations

**Draft — Abstract, Introduction and Related Work.** Written 2026-11-27 against branch `marcelo-21nov`
(HEAD `f554cd9`). Numbers come from `reports/full_run_results.pt-BR.md` (full grid, re-run of 2026-11-22).

This draft takes the **cautious calibration**: claims are attached to the repertoire the method returns, not to
the algorithm family, and the three structural asymmetries of the comparison (C1, C2, C3 in the run report) are
stated in the text rather than left for a reviewer to find. See *Notes on this draft* at the end for the open
items, including the title question raised by C6.

Authors and order: to be settled by the group. GECCO venues require anonymisation.

---

## Abstract

Counterfactual explanations answer the question of what would have to change in an input for a model to decide
otherwise, and are among the forms of explanation closest to human reasoning. Most established methods formulate
their generation as an optimisation problem that returns a single point, or a small set obtained by adding a
diversity penalty to the objective. Both formulations tend to converge on variations of one change strategy, and
the first depends on gradients or on a well-behaved cost function.

We formulate counterfactual generation as a quality-diversity (QD) problem, in which the search returns a
repertoire of solutions that are simultaneously valid and behaviourally distinct. Behaviour is described along two
axes: the *strategy*, that is, which domain-defined groups of attributes changed and in which direction, and the
*effort*, that is, how many attributes changed and by how much. Each cell of the resulting archive corresponds to
one recourse path at one level of effort, so a filled archive is itself a statement about which paths the model
admits around an instance.

We implement MAP-Elites over this descriptor and compare the repertoire it returns against the sets produced by
DiCE (genetic and random variants), Wachter's gradient method and FACE, over five tabular decision datasets, three
model families, ten seeds and five model-query budgets, for 700 configurations in total. At the largest budget the
MAP-Elites archive occupies 40.9% of the feasible cells of the strategy-by-effort grid against 6.0% for the best
baseline, and this ordering holds in every one of the 150 (dataset, model, seed) blocks without exception. No method
violated the documented mutability constraints, and validity is comparable across methods. In a controlled
simulation whose recourse paths are known by construction, MAP-Elites recovers 0.66 of the planted paths against
0.34 for the strongest baseline, and is the only method that regularly reaches the distant ones.

We argue that these results should not yet be read as a settled comparison between search paradigms, and we report
why. Coverage is bounded by the number of counterfactuals a method returns, which ranges from one to 85 across the
methods compared; per returned counterfactual, DiCE attains as many distinct strategies as MAP-Elites or more. The
MAP-Elites fitness is the same distance used to score proximity. An equal query budget does not imply equal search
effort, since the median number of queries actually consumed ranges from 1 to 10,000 depending on the method.
Plausibility, which the hypothesis also concerns, is not measured here. We therefore present this work as a study
of MAP-Elites as a generator of counterfactual repertoires, together with an open evaluation framework and an
explicit account of what a fair comparison between QD and optimisation would still require.

*Keywords:* counterfactual explanations, algorithmic recourse, quality diversity, MAP-Elites, explainable
artificial intelligence, evolutionary computation, benchmarking.

---

## 1. Introduction

### 1.1 Counterfactual explanations and the single-solution formulation

A counterfactual explanation of a model decision is a second input, close to the original, for which the model
produces a different and desired outcome. Miller (2019) argues that contrastive explanations of this kind are the
form closest to how people actually explain things to one another, and Wachter, Mittelstadt and Russell (2017)
gave the formulation that most of the subsequent literature inherits: find the smallest change to the input that
flips the decision, by minimising a loss that combines the distance to the desired output with the distance to the
original point.

That formulation returns one point. Its practical limitation is well known and was stated early: the nearest change
is not necessarily a change the affected person is able to perform. A credit applicant told that the shortest path
to approval runs through a higher income receives an explanation that is correct and useless if their income is not
the variable they control. Mothilal, Sharma and Tan (2020) responded by arguing that a diverse set of
counterfactuals is more useful than a single one, and built that argument into DiCE by adding a diversity term to
the objective, alongside validity and proximity.

Adding diversity as a penalty, however, keeps it subordinate. The search still has one scalar objective, and
diversity competes with proximity inside it through a weight the user must set. The set that comes back is the set
that happens to optimise the weighted sum, not a set constructed to span the space of possible strategies.

### 1.2 Why this problem has the shape of a quality-diversity problem

Two features of counterfactual generation suggest a different family of search algorithms.

First, validity is a step function of the model output. A candidate either crosses the decision boundary or it does
not, and in the region before it crosses, an objective built on validity provides no gradient and no ranking.
Lehman and Stanley (2011) showed that objectives with this character send search into dead ends, and that
rewarding only the novelty of behaviour can reach solutions that direct optimisation does not find. The
quality-diversity family that followed combines that divergent search with local competition for quality, and
returns, from a single run, a repertoire of solutions indexed by characteristics the user chooses rather than by
the objective (Mouret and Clune, 2015; Pugh, Soros and Stanley, 2016).

Second, and more importantly for explanation, what a user needs is often not the single optimal counterfactual but
a view of the distinct strategies available: change income, change credit history, or change both by less. That
view is precisely the output a QD algorithm produces. If the behaviour descriptor encodes which attributes changed
and in which direction, then each cell of the archive is one change strategy, and the filled archive reports which
strategies the model admits for that instance and what each one costs. The archive stops being an intermediate data
structure and becomes the explanation.

### 1.3 The gap

The closest evolutionary relative of this idea is the multi-objective counterfactual (MOC) method of Dandl et al.
(2020), which treats validity, distance, sparsity and plausibility as separate objectives of an NSGA-II run and
returns a Pareto front. The difference is the criterion by which solutions are kept apart: a Pareto front organises
solutions by the objectives themselves, whereas a QD archive organises them by behaviour descriptors defined
independently of the objectives. Two counterfactuals that are equivalent on the objective vector may sit in
different cells of a QD archive if they change different attributes, and that is the distinction the explanation
needs.

The gap is on record in the community. The roadmap of Zhou et al. (2024) on evolutionary computation and
explainable AI reviews evolutionary approaches to counterfactual generation, naming CERTIFAI, GeCo and MOC, and
then lists quality diversity applied to counterfactual explanation among the directions not yet taken. To our
knowledge there is no systematic comparison between QD algorithms and established counterfactual methods under a
common protocol and a common budget of model queries.

### 1.4 What this paper does

We formulate counterfactual generation as a QD problem and study it empirically. An individual is a point in the
attribute space, mutated only in mutable attributes and within documented valid ranges. Quality combines validity
with proximity to the original instance. The behaviour descriptor has two axes: a *strategy* axis, which records
for each domain-defined group of attributes whether it increased, decreased or was left unchanged, and an *effort*
axis, which records how many attributes changed and by how much. With `g` groups the strategy axis has at most
`3^g` states, of which only a subset is feasible once mutability constraints are applied.

We implement MAP-Elites over this descriptor in an open framework whose common interface exposes each method to the
same model wrapper and the same query counter, and we compare the repertoire it returns against DiCE in its genetic
and random variants, the gradient method of Wachter et al., and FACE. The evaluation covers five tabular decision
datasets (German Credit, Adult, HELOC, Give Me Some Credit and COMPAS), three model families (logistic regression,
random forest and a shallow neural network), ten seeds, twenty test instances per configuration drawn from the rows
each model itself predicts unfavourably, and five query budgets from 500 to 10,000, for 700 configurations and 36.5
hours of recorded CPU time. A separate controlled simulation plants a known set of recourse paths, one near the
instance and the others far from it, so that the fraction of true paths each method recovers can be measured
against ground truth rather than against another method.

### 1.5 Findings

At the largest budget, the MAP-Elites archive covers 40.9% of the feasible cells of the strategy-by-effort grid,
against 6.0% for the best baseline, and it has the highest coverage in all 150 (dataset, model, seed) blocks. The
margin is between 11-fold and 30-fold on German Credit, Adult, HELOC and Give Me Some Credit, and 2.2-fold on
COMPAS, whose grid has only ten feasible cells because the dataset admits a single actionable group of attributes.
No method produced a counterfactual that violates the documented constraints, in any of the 700 configurations.
Validity is comparable: MAP-Elites finds at least one valid counterfactual for 97.7% of instances, Wachter for
98.5%, DiCE for roughly 94%, and FACE for 85.5%. The nearest counterfactual returned by MAP-Elites is nearer than
that of DiCE-genetic, Wachter and FACE in 99% of paired instances.

In the simulation, MAP-Elites recovers 0.66 of the planted paths at 10,000 queries against 0.34 for DiCE-genetic,
in every (model, seed) block, and it is the only method that regularly reaches the distant paths. Shuffling the
mapping between attribute groups and descriptor axes, which removes the alignment between the descriptor and the
planted structure, costs 0.07 of that recall; the advantage therefore does not depend entirely on a descriptor
chosen with knowledge of the answer.

### 1.6 Contributions

1. A formulation of counterfactual generation as a quality-diversity problem, with a two-axis behaviour descriptor
   whose cells are interpretable as recourse paths at a given level of effort, and an account of how the grid size
   follows from the number of attribute groups.
2. An open evaluation framework in which counterfactual methods of different families are driven through one
   interface against one model wrapper and one model-query counter, with documented mutability constraints per
   dataset and automated tests over the constraint and metric modules.
3. An empirical study of MAP-Elites as a counterfactual generator over 700 configurations, reporting coverage,
   validity, proximity, sparsity, redundancy, constraint violations and cost, as curves over five query budgets
   rather than at a single operating point.
4. A controlled simulation with known recourse paths, which allows recall of true paths to be measured directly,
   together with the fraction of valid counterfactuals that flip the model without lying on any planted path.
5. An explicit account of the asymmetries that a comparison of this kind carries, and of what a fair comparison
   between quality diversity and optimisation would still require.

### 1.7 Scope, and what this paper does not claim

We state the boundaries of the present version in the introduction rather than only in a closing section, because
they bear directly on how the results above should be read.

**Coverage depends on how many counterfactuals a method returns.** Coverage counts occupied cells, and a method
that returns one point can occupy at most one cell. Wachter and FACE return one counterfactual, DiCE returns at
most ten, and the MAP-Elites archive holds 85.5 on average and up to 358. Part of the observed margin is therefore
a property of the output size rather than of the search. Measured per returned counterfactual, DiCE attains as many
distinct strategies as MAP-Elites or more: 0.67 against 0.38 on German Credit, 0.44 against 0.30 on HELOC, 0.41
against 0.30 on Give Me Some Credit. A comparison at equal set size is required before the coverage result can be
attributed to the paradigm, and we identify it as the first piece of work still to be done.

**The MAP-Elites fitness is the metric used to score proximity.** Elite quality is a monotone transform of the same
distance that the evaluation uses to report proximity, whereas Wachter minimises an L1 distance in min-max space
and DiCE minimises its own proximity-and-diversity loss. The proximity result is therefore not symmetric between
methods, and we report it as such.

**An equal budget is not equal effort.** All methods receive the same ceiling of model queries, but the median
number actually consumed at a ceiling of 10,000 is 10,000 for MAP-Elites, about 830 for DiCE-genetic, 2,031 for
DiCE-random, nine for Wachter and one for FACE. Methods that build a structure over the training data, namely the
k-d tree used by DiCE and the density graph used by FACE, additionally receive predictions over the training set
that the counter does not charge. The budget is a common ceiling, not a common expenditure.

**Plausibility is not measured.** The hypothesis concerns diversity obtained without loss of validity, proximity or
plausibility, and the first three of those are measured here while the fourth is not. There is also evidence
against it that we report rather than omit: 42% of the valid counterfactuals MAP-Elites returns in the simulation
flip the model without satisfying any planted path, and mutation produces fractional values in attributes that are
integer by nature.

**One QD algorithm, not a family.** Novelty search, novelty search with local competition and surprise search are
part of the formulation but are not implemented in this version, and neither is MOC, which, because it returns a
Pareto front, is the baseline most likely to compete on coverage. A comparison among QD algorithms is consequently
outside the scope of this version, and the strongest evolutionary baseline is absent from it.

---

## 2. Related Work

### 2.1 Counterfactual explanations: definition and desiderata

A counterfactual explanation of an instance `x` under a model `f` is a point `x'` such that `f(x')` is the desired
outcome, and that satisfies a set of requirements the literature has converged on. It should be *valid*, that is,
actually change the decision; *proximate*, that is, close to `x` under some distance; *sparse*, that is, change few
attributes; *actionable*, that is, respect which attributes the affected person can change and in which direction;
and *plausible*, that is, lie in a region the data supports. Guidotti (2024) and Verma et al. (2024) organise these
requirements and classify methods by search strategy, by the access they need to the model, and by how they handle
constraints.

Two of these requirements deserve emphasis because they bear on the design choices in this work. Actionability is
not a property that can be read off the data: no column states that age is immutable or that a count of past
delinquencies cannot be reduced. It is a modelling decision, and two studies on the same dataset may legitimately
differ. We therefore document it per dataset as an explicit constraint table, and distinguish attributes that can
only increase, such as education, from attributes that record past events and cannot be reversed at all, such as
delinquency counts; treating the two alike would either discard valid strategies or produce counterfactuals that
ask the subject to undo the past. Plausibility matters for a different reason: Laugel et al. (2019) showed that
counterfactuals generated without regard for the data distribution may land in regions without support and explain
artefacts of the model rather than the phenomenon.

### 2.2 Gradient-based optimisation

Wachter, Mittelstadt and Russell (2017) minimise a loss combining the squared distance of the model output to the
target with the distance to the original instance, increasing the weight on the first term until the class changes.
The method returns one counterfactual, is simple, and requires a differentiable model; on a tree ensemble it does
not apply at all.

The contrastive explanation method of Dhurandhar et al. (2018) reformulates the problem as finding the *pertinent
negative*: the minimal set of features whose absence changes the decision, with an autoencoder term that keeps the
result near the data manifold. CFProto (Van Looveren and Klaise, 2021) adds a term that pulls the candidate toward
a prototype of the target class, which speeds convergence and improves plausibility, and can operate with numerical
gradient estimates when gradients are unavailable. DiCE's original gradient variant optimises a set of `k`
counterfactuals jointly, adding a determinantal point process term that rewards dissimilarity among them
(Mothilal, Sharma and Tan, 2020).

### 2.3 Latent-space and generative methods

A second family moves the search off the raw attribute space. C-CHVAE (Pawelczyk et al., 2020), REVISE (Joshi et
al., 2019) and CRUDS (Downs et al., 2020) train a variational autoencoder and search for the counterfactual in the
latent space, decoding back; because the decoder was trained on the data, the result is by construction closer to
the data distribution. CLUE (Antorán et al., 2021) is related in machinery but answers a different question: it
searches for changes that reduce the uncertainty of a Bayesian model rather than changes that flip a class.

These methods address plausibility directly, at the cost of a generative model to train and of a search that is
harder to constrain: a mutability constraint stated in the attribute space does not translate into a simple
restriction of the latent space.

### 2.4 Model-specific methods for tree ensembles

Because gradient methods exclude tree ensembles, a family of methods exploits their structure instead. Feature
Tweaking (Tolomei et al., 2017) enumerates the root-to-leaf paths that end in the desired class and computes the
minimal adjustment to `x` that satisfies a chosen path. FOCUS (Lucic et al., 2022) replaces the hard splits with
sigmoids, producing a differentiable approximation of the ensemble over which gradient descent can then run.

These methods are exact or near-exact for the model family they target and do not transfer beyond it, which makes
them useful as strong per-model references but unsuitable as cross-model baselines.

### 2.5 Instance-based and graph-based methods

A third family builds the counterfactual from points that already exist in the data. NICE (Brughmans, Leyman and
Martens, 2024) starts from the nearest training instance of the opposite class and copies attributes from it one at
a time until the decision changes, selecting at each step by sparsity, proximity or plausibility; it is
model-agnostic and always returns a solution when one exists in the data. Growing Spheres (Laugel et al., 2018)
samples points in growing spherical shells around `x` until one of the opposite class is found, then reduces the
number of changed attributes.

FACE (Poyiadzi et al., 2020) differs from all of the above in what it optimises. It builds a graph over the
training data weighted by density and returns the shortest path to a point of the desired class, so the
counterfactual is a real data point reached through dense regions. Feasibility is thereby built into the
construction rather than added as a penalty. The method needs no gradients; the limitation we encountered is one of
available implementations rather than of the method, and is discussed in Section 2.9.

### 2.6 Constraint-based recourse

Ustun, Spangher and Liu (2019) approach the problem from the recourse side rather than the explanation side. For
linear classifiers they solve the minimum-cost action problem exactly with integer programming, subject to declared
actionability constraints, and approximate other models locally. The MACE method of Karimi et al. (2020) encodes
model, distance and plausibility constraints as a satisfiability problem and obtains counterfactuals with
optimality guarantees from a solver. The implementation distributed in OmniXAI under the same acronym follows a
different strategy, retrieving candidates from the training data and refining them with numerically estimated
gradients, and the two should not be confused.

These methods offer the strongest guarantees available, and pay for them in the restrictions they place on the
model or in solver cost.

### 2.7 Evolutionary and metaheuristic methods

The closest relatives of the present work treat counterfactual generation as a population-based search, and need no
gradient.

MOC (Dandl et al., 2020) runs a modified NSGA-II with mixed integer evolution strategies over four objectives,
namely validity, distance, number of changed attributes and plausibility, and returns a Pareto front rather than a
point. It is the natural baseline for any claim about diversity obtained by evolutionary means, because a front is
already a set of mutually non-dominated alternatives. CERTIFAI (Sharma, Henderson and Ghosh, 2020) uses a genetic
algorithm to generate counterfactuals and derives from them measures of robustness and fairness. GeCo (Schleich et
al., 2021) also searches genetically but concentrates on plausibility constraints and on the database machinery
needed to make the search interactive. The genetic variant of DiCE belongs here as well: although the method is
presented through its diversity term, its model-agnostic implementation evolves a population under the DiCE loss.
CFNOW (de Oliveira and Martens, 2023) uses a two-phase tabu search, first finding any valid counterfactual and then
refining it to escape local optima, and requires only `predict_proba`.

Zhou et al. (2024) survey this family and argue that evolutionary computation is well suited to counterfactual
generation precisely because it is a black-box method with mature multi-objective machinery. The same survey notes
that counterfactual methods remain focused on single instances and therefore say little about global model
behaviour, that better metrics for explanation quality are needed, and that quality diversity has not yet been
brought to bear on the problem.

### 2.8 Quality diversity

Novelty search (Lehman and Stanley, 2011) replaces the objective with the average distance of an individual to its
`k` nearest neighbours in a behaviour space, keeping an archive of what has already been found. Novelty search with
local competition adds a second objective under which each individual competes on quality only against its
behavioural neighbours, so that quality is preserved without collapsing diversity. MAP-Elites (Mouret and Clune,
2015) discretises the descriptor space into a grid and retains the best solution found in each cell, which yields
in one run a map of how quality varies across behaviours; CVT-MAP-Elites (Vassiliades, Chatzilygeroudis and Mouret,
2018) replaces the regular grid with centroids obtained by a centroidal Voronoi tessellation, which keeps the
number of cells fixed as the descriptor dimension grows. Surprise search (Gravina, Liapis and Yannakakis, 2019)
rewards deviation from the behaviour predicted from recent generations rather than deviation from behaviours
already visited, and combined with novelty improved both efficiency and robustness on deceptive tasks. Pugh, Soros
and Stanley (2016) set out the family as a whole, and the pyribs library provides reference implementations.

What makes this family relevant here is not its optimisation performance but its output type. A QD algorithm is
designed to return a set indexed by user-chosen characteristics, which is the same object an explanation of
available recourse strategies requires.

### 2.9 Benchmarks, implementations and evaluation measures

CARLA (Pawelczyk et al., 2021) is the most complete catalogue of counterfactual and recourse methods behind a
common interface, with datasets, trained models and an evaluation module reporting success rate, distance,
redundancy, yNN and constraint violations. Its practical limitation for new work is that the distribution has not
been updated since 2022 and requires a Python version that is no longer supported, and that most of its methods are
implemented only for differentiable models; FACE, for instance, is available there only for neural networks even
though the method itself needs no gradients. ReLax (Guo et al., 2024) reimplements a set of methods in JAX with a
vectorised benchmarking protocol and reports order-of-magnitude reductions in benchmarking time, at the cost of
explaining only models written in JAX. Guidotti (2024) provides both a review and a benchmark over several datasets
and model families, and is the source we follow for metric definitions, including the mixed distance that
normalises continuous attributes by the median absolute deviation and counts categorical mismatches.

Five of the six families of measures this work reports already have published definitions in that literature:
validity, proximity, sparsity, plausibility and actionability. Diversity is the exception. The counterfactual
literature measures it as an average pairwise distance within a returned set, which says nothing about how much of
the space of possible strategies the set spans; archive coverage and QD-score come instead from the
quality-diversity literature. Bringing the two together is part of what this work has to do, and the mismatch is
also why we report coverage normalised by the number of feasible cells, so that datasets whose grids differ in size
remain comparable.

### 2.10 Positioning

Against this background, the present work sits in a specific place. It is not a new counterfactual method competing
on proximity, and it is not a QD contribution competing on archive quality. It asks whether the output type that QD
algorithms produce, a repertoire indexed by behaviour, is the output type that counterfactual explanation needs,
and it tries to answer that question under a protocol in which the comparison with established methods is explicit
about what is and is not being held equal.

---

## Notes on this draft

**Calibration.** Written to the cautious calibration agreed on 2026-11-27. Claims attach to the returned
repertoire, not to the algorithm family. Concerns C1, C2, C3, C5 and C6 of `reports/full_run_results.pt-BR.md`
appear in Section 1.7 as declared scope limits rather than as an appendix.

**Title and C6.** The project title promises quality-diversity *algorithms* in the plural, and only MAP-Elites is
implemented. The heading above drops the plural by speaking of "quality-diversity search". This is a group
decision: either the title is scoped to this version, or Section 1.7 carries the whole weight of the discrepancy.

**Numbers.** All figures are from the full re-run of 2026-11-22. Not yet incorporated, because they are not yet
measured: plausibility (C5) and the equal-set-size comparison (C1). Section 1.5 and the abstract will need one
pass each once those exist.

**Open in the text.**
- Section 1.4 says twenty instances per configuration are drawn from the rows each model predicts unfavourably.
  Under C16 this makes cross-model comparisons unpaired, and the Methods section should say so.
- Section 2.9 claims coverage is normalised by feasible cells; confirm this matches what `metrics.py` computes.
- The balanced accuracies of C8 belong in Methods, not here, but they weaken any claim built on Give Me Some
  Credit and should not be omitted from the paper.

**References.** The reference list is deliberately not reproduced here. The bibliography inherited from the work
plan states that its entries were compiled from memory and that volumes, pages and DOIs must be checked before
submission, and that applies to every entry carried over. The entries added in this draft, and not present in the
plan, are: Zhou et al. (2024) on evolutionary computation and explainable AI, verified at arXiv:2406.07811;
Ustun, Spangher and Liu (2019); Karimi et al. (2020); Sharma, Henderson and Ghosh (2020) on CERTIFAI; Schleich et
al. (2021) on GeCo; Dhurandhar et al. (2018); Van Looveren and Klaise (2021); Pawelczyk et al. (2020); Joshi et al.
(2019); Downs et al. (2020); Antorán et al. (2021); Tolomei et al. (2017); Lucic et al. (2022); Brughmans, Leyman
and Martens (2024); Laugel et al. (2018) on Growing Spheres; de Oliveira and Martens (2023) on CFNOW; Vassiliades,
Chatzilygeroudis and Mouret (2018) on CVT-MAP-Elites; and Guo et al. (2024) on ReLax. Author lists and years for
these were written from knowledge of the literature and **must be verified against the published records before
submission**; do not treat them as checked.
