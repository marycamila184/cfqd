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

### References

Antorán, J., Bhatt, U., Adel, T., Weller, A., & Hernández-Lobato, J. M. (2021). Getting a CLUE: A method for explaining uncertainty estimates. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2006.06848

Brughmans, D., Leyman, P., & Martens, D. (2024). NICE: An algorithm for nearest instance counterfactual explanations. *Data Mining and Knowledge Discovery*, 38(5), 2665–2703. https://doi.org/10.1007/s10618-023-00930-y

Dandl, S., Molnar, C., Binder, M., & Bischl, B. (2020). Multi-objective counterfactual explanations. In *Parallel Problem Solving from Nature – PPSN XVI* (Lecture Notes in Computer Science, pp. 448–469). https://doi.org/10.1007/978-3-030-58112-1_31

de Oliveira, R. M. B., Sörensen, K., & Martens, D. (2024). A model-agnostic and data-independent tabu search algorithm to generate counterfactuals for tabular, image, and text data. *European Journal of Operational Research*, 317(2), 286–302. https://doi.org/10.1016/j.ejor.2023.08.031
<!-- MARY: o texto cita "de Oliveira and Martens (2023)", mas o artigo do CFNOW tem três autores (de Oliveira, Sörensen e Martens) e saiu no volume de 2024 (o README do CFNOW cita como 2023). Sugestão: "de Oliveira, Sörensen and Martens (2024)". -->

Dhurandhar, A., Chen, P.-Y., Luss, R., Tu, C.-C., Ting, P., Shanmugam, K., & Das, P. (2018). Explanations based on the missing: Towards contrastive explanations with pertinent negatives. In *Advances in Neural Information Processing Systems (NeurIPS 2018)*. https://papers.nips.cc/paper_files/paper/2018/hash/c5ff2543b53f4cc0ad3819a36752467b-Abstract.html

Downs et al. (2020). CRUDS.
<!-- MARY: referência NÃO encontrada em registros públicos (Crossref, arXiv). Faltam autores, título completo e onde foi publicada. Marcelo, pode completar com a fonte? -->

Gravina, D., Liapis, A., & Yannakakis, G. N. (2019). Quality diversity through surprise. *IEEE Transactions on Evolutionary Computation*, 23(4), 603–616. https://doi.org/10.1109/tevc.2018.2877215

Guidotti, R. (2024). Counterfactual explanations and how to find them: Literature review and benchmarking. *Data Mining and Knowledge Discovery*, 38(5), 2770–2824. https://doi.org/10.1007/s10618-022-00831-6

Guo, H., Xiong, X., Zhang, W., & Yadav, A. (2024). ReLax: Efficient and scalable recourse explanation benchmarking using JAX. *Journal of Open Source Software*, 9(103), 6567. https://doi.org/10.21105/joss.06567

Joshi, S., Koyejo, O., Vijitbenjaronk, W., Kim, B., & Ghosh, J. (2019). Towards realistic individual recourse and actionable explanations in black-box decision making systems. *arXiv preprint*. https://arxiv.org/abs/1907.09615

Karimi, A.-H., Barthe, G., Balle, B., & Valera, I. (2020). Model-agnostic counterfactual explanations for consequential decisions. In *Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics (AISTATS 2020)*, PMLR 108, 895–905. https://proceedings.mlr.press/v108/

Laugel, T., Lesot, M.-J., Marsala, C., Renard, X., & Detyniecki, M. (2018). Comparison-based inverse classification for interpretability in machine learning. In *Information Processing and Management of Uncertainty in Knowledge-Based Systems. Theory and Foundations* (Communications in Computer and Information Science, pp. 100–111). https://doi.org/10.1007/978-3-319-91473-2_9

Laugel, T., Lesot, M.-J., Marsala, C., Renard, X., & Detyniecki, M. (2019). The dangers of post-hoc interpretability: Unjustified counterfactual explanations. In *Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19)* (pp. 2801–2807). https://doi.org/10.24963/ijcai.2019/388

Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*, 19(2), 189–223. https://doi.org/10.1162/evco_a_00025

Lucic, A., Oosterhuis, H., Haned, H., & de Rijke, M. (2022). FOCUS: Flexible optimizable counterfactual explanations for tree ensembles. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(5), 5313–5322. https://doi.org/10.1609/aaai.v36i5.20468

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. In *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency* (pp. 607–617). https://doi.org/10.1145/3351095.3372850

Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv preprint*. https://arxiv.org/abs/1504.04909

Pawelczyk, M., Bielawski, S., van den Heuvel, J., Richter, T., & Kasneci, G. (2021). CARLA: A Python library to benchmark algorithmic recourse and counterfactual explanation algorithms. *NeurIPS 2021 Track on Datasets and Benchmarks*. https://arxiv.org/abs/2108.00783

Pawelczyk, M., Broelemann, K., & Kasneci, G. (2020). Learning model-agnostic counterfactual explanations for tabular data. In *Proceedings of The Web Conference 2020* (pp. 3126–3132). https://doi.org/10.1145/3366423.3380087

Poyiadzi, R., Sokol, K., Santos-Rodriguez, R., De Bie, T., & Flach, P. (2020). FACE: Feasible and actionable counterfactual explanations. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society* (pp. 344–350). https://doi.org/10.1145/3375627.3375850

Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation. *Frontiers in Robotics and AI*, 3. https://doi.org/10.3389/frobt.2016.00040

Schleich, M., Geng, Z., Zhang, Y., & Suciu, D. (2021). GeCo: Quality counterfactual explanations in real time. *Proceedings of the VLDB Endowment*, 14(9), 1681–1693. https://doi.org/10.14778/3461535.3461555

Sharma, S., Henderson, J., & Ghosh, J. (2020). CERTIFAI: A common framework to provide explanations and analyse the fairness and robustness of black-box models. In *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society* (pp. 166–172). https://doi.org/10.1145/3375627.3375812

Tolomei, G., Silvestri, F., Haines, A., & Lalmas, M. (2017). Interpretable predictions of tree-based ensembles via actionable feature tweaking. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 465–474). https://doi.org/10.1145/3097983.3098039

Ustun, B., Spangher, A., & Liu, Y. (2019). Actionable recourse in linear classification. In *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 10–19). https://doi.org/10.1145/3287560.3287566

Van Looveren, A., & Klaise, J. (2021). Interpretable counterfactual explanations guided by prototypes. In *Machine Learning and Knowledge Discovery in Databases. Research Track* (Lecture Notes in Computer Science, pp. 650–665). https://doi.org/10.1007/978-3-030-86520-7_40

Vassiliades, V., Chatzilygeroudis, K., & Mouret, J.-B. (2018). Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. *IEEE Transactions on Evolutionary Computation*, 22(4), 623–630. https://doi.org/10.1109/tevc.2017.2735550

Verma, S., Boonsanong, V., Hoang, M., Hines, K., Dickerson, J., & Shah, C. (2024). Counterfactual explanations and algorithmic recourses for machine learning: A review. *ACM Computing Surveys*, 56(12), 1–42. https://doi.org/10.1145/3677119

Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box: Automated decisions and the GDPR. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3063289
<!-- MARY: o registro do arXiv (1711.00399) diz que saiu no Harvard Journal of Law & Technology (2018); volume e páginas não foram verificados. Citar a versão do periódico? -->

Zhou, R., Bacardit, J., Brownlee, A. E. I., Cagnoni, S., Fyvie, M., Iacca, G., McCall, J., van Stein, N., Walker, D., & Hu, T. (2024). Evolutionary computation and explainable AI: A roadmap to understandable intelligent systems. *arXiv preprint*. https://arxiv.org/abs/2406.07811
<!-- MARY: já saiu em periódico: IEEE Transactions on Evolutionary Computation, 29(5), 2213–2228 (2025), https://doi.org/10.1109/tevc.2024.3476443. Trocar para a versão publicada e citar como Zhou et al. (2025)? -->
