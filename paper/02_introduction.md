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

### References

Dandl, S., Molnar, C., Binder, M., & Bischl, B. (2020). Multi-objective counterfactual explanations. In *Parallel Problem Solving from Nature – PPSN XVI* (Lecture Notes in Computer Science, pp. 448–469). https://doi.org/10.1007/978-3-030-58112-1_31

Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*, 19(2), 189–223. https://doi.org/10.1162/evco_a_00025

Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence*, 267, 1–38. https://doi.org/10.1016/j.artint.2018.07.007

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency* (pp. 607–617). https://doi.org/10.1145/3351095.3372850

Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv preprint*. https://arxiv.org/abs/1504.04909

Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation. *Frontiers in Robotics and AI*, 3. https://doi.org/10.3389/frobt.2016.00040

Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3063289
<!-- MARY: o registro do arXiv (1711.00399) diz que saiu no Harvard Journal of Law & Technology (2018); volume e páginas não foram verificados. Citar a versão do periódico? -->

Zhou, R., Bacardit, J., Brownlee, A. E. I., Cagnoni, S., Fyvie, M., Iacca, G., McCall, J., van Stein, N., Walker, D., & Hu, T. (2024). Evolutionary computation and explainable AI: A roadmap to understandable intelligent systems. *arXiv preprint*. https://arxiv.org/abs/2406.07811
<!-- MARY: já saiu em periódico: IEEE Transactions on Evolutionary Computation, 29(5), 2213–2228 (2025), https://doi.org/10.1109/tevc.2024.3476443. Trocar para a versão publicada e citar como Zhou et al. (2025)? -->
