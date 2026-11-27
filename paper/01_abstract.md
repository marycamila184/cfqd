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