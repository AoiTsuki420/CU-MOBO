# Calibrated-Uncertainty Multi-Objective Bayesian Optimization for Expensive, Uncertain Oracles in Inverse Molecular Design

**Xinran Wang, Haijuan Zhang, Yi Deng, Jiahuan Zhao, Yang Li\***

School of Pharmacy, Lanzhou University, Lanzhou 730000, China

\*Corresponding author: Yang Li. E-mail: to be provided (Lanzhou University, lzu.edu.cn)

---

## Abstract

Inverse molecular and process design increasingly relies on *learned* oracles for enzyme turnover (kcat), binding, or structure. These scores can be costly to obtain and imperfect. Optimizing a point prediction `μ` can favor designs whose predicted activity is high but whose true activity is low. We call the difference between prediction and truth the **oracle-gap**.

We present **CU-MOBO** (Calibrated-Uncertainty Multi-Objective Bayesian Optimization), which uses the oracle's uncertainty estimate `σ` in two places: (1) as an explicit Pareto objective alongside predicted activity and molecular properties, and (2) as input-dependent noise `σ²` in the activity Gaussian-process surrogate (`train_Yvar`). The latter down-weights observations assigned higher uncertainty; interpreting the posterior as an estimate of true activity additionally requires that `σ²` reflects the prediction error of `μ` relative to that activity. Constrained qNEHVI selects batches from the resulting model.

In a **controlled synthetic oracle-gap benchmark** with hidden ground-truth activity (5 seeds), CU-MOBO recovers designs with **2.9× higher true activity** and **13× smaller oracle-gap** than μ-greedy; trap-region frequency falls from 32.9 % to 0.7 %. A 2×2 ablation separates the contributions of the two `σ` mechanisms. In a **retrospective study using experimental kcat labels**, generated molecules are enriched for held-out high-kcat substrates (**1.26 ± 0.11× over random**; specificity **1.74×**). A scaffold-split screening analysis and near-neighbour-removal control characterize analogue bias. On a third-party TDC QED scorer, CU-MOBO achieves similar or higher scores than μ-greedy with a small oracle-gap. A calibration audit on real kcat labels finds Spearman correlations of 0.18–0.21 between `σ` and absolute prediction error.

---

## 1. Introduction

Generative models can propose vast numbers of candidate molecules or process configurations; the bottleneck is *scoring* them. The scoring function may be a **learned oracle** for enzyme turnover [@li2022dlkcat; @yu2023unikp; @kroll2023turnup], structure [@wohlwend2024boltz; @jumper2021alphafold], or bioactivity. Recent molecular-design work has sharpened two related concerns: expensive oracle calls make sample efficiency central [@guo2026saturn], and predictive models may fail beyond the chemical space represented in their training data [@vantilborg2026edge].

1. **Evaluation is costly.** Each query may require substantial computation or an assay. Evaluation budget is therefore a useful unit of comparison, as in the *Practical Molecular Optimization* (PMO) benchmark [@gao2022pmo].
2. **Predictions have error.** Ensembles [@lakshminarayanan2017ensembles], conformal methods [@angelopoulos2021conformal], and evidential models [@amini2020evidential] can attach uncertainty estimates to predictions, but their calibration must be checked for the task and chemical region being optimized. Chen and Li used uncertainty-aware probability-of-improvement objectives for molecular design, including multi-objective tasks [@chen2025uqdesign].

`μ`-greedy design can select molecules with high predicted activity but low true activity `a_true`, creating an **oracle-gap**. Optimization can amplify prediction errors, especially under distribution shift [@vantilborg2026edge]. In multi-objective molecular design, DyRAMO addresses related reward hacking with applicability-domain constraints [@yoshizawa2025dyramo].

**Contributions.**
- **CU-MOBO**, which uses an oracle uncertainty estimate both as a Pareto objective and as heteroscedastic GP observation noise, with the latter conditional on an error-variance interpretation of `σ²`.
- A **controlled synthetic benchmark** with hidden `a_true` that measures oracle-gap independently of the optimized predictor, with a **2×2 ablation** of the two `σ` mechanisms.
- A **real-label retrospective study** (kcat) that measures enrichment relative to random generation and characterizes scaffold and near-neighbour effects.
- A **calibration audit** showing that both RF tree variance and out-of-fold residual estimates rank errors only weakly on the real-kcat surrogate.

The inner loop uses established Bayesian optimization methods for expensive multi-objective search [@shahriari2016bo; @frazier2018bo]. The methodological question here is how oracle uncertainty enters the objective and the surrogate, and how those choices affect recommendations when the oracle is imperfect.

---

## 2. Related Work

**Multi-objective BO.** qEHVI [@daulton2020qehvi] and qNEHVI [@daulton2021qnehvi] are hypervolume-based acquisition functions for expensive multi-objective optimization; qParEGO [@knowles2006parego] provides a random-scalarization baseline. These methods provide the acquisition framework used here; CU-MOBO additionally specifies how an external oracle error estimate enters the objectives and activity surrogate.

**Uncertainty in molecular optimization.** Chen and Li [@chen2025uqdesign] combine graph-network uncertainty with probability-of-improvement objectives and report gains on multi-objective molecular tasks. DyRAMO [@yoshizawa2025dyramo] adjusts applicability-domain thresholds to reduce reward hacking. CU-MOBO tests a different placement of the uncertainty estimate: `σ` is an objective and `σ²` supplies input-dependent noise to the activity GP. These uses require distinct forms of calibration (§4.4).

**Oracle exploitation.** Design-Bench [@trabucco2022designbench] separates a learned proxy from the oracle used to evaluate recommended designs; Conservative Objective Models [@trabucco2021coms] demonstrate how proxy optimization can improve predicted values while worsening independently evaluated values. Our controlled benchmark likewise hides `a_true` from the optimizer. The 2026 study by van Tilborg and colleagues [@vantilborg2026edge] further documents the challenge of molecular prediction beyond training chemical space.

---

## 3. Method: CU-MOBO

![Figure 1](../results/fig1_main.png)
**Figure 1.** CU-MOBO overview. **(a)** In the controlled benchmark, μ-greedy optimization can favor a high-μ, low-`a_true` trap, while an informative σ separates it from a high-activity region. **(b)** σ enters the Pareto objectives and, when it estimates prediction-error variance, supplies heteroscedastic activity-GP noise (`train_Yvar`); constrained qNEHVI selects batches.

### 3.1 Problem setting
Search `x ∈ X` (e.g. a generative-model latent `z`). Each query of an expensive, uncertain oracle returns predicted activity `μ(x)`, calibrated uncertainty `σ(x)`, deterministic objectives (QED [@bickerton2012qed], SA [@ertl2009sa]), and constraints (validity, Lipinski [@lipinski1997ro5], PAINS [@baell2010pains], size), all computed with RDKit [@landrum_rdkit]. The quantity of interest is the **true** activity `a_true`, not `μ`.

### 3.2 Failure mode
μ-greedy may favor regions where `μ` is inflated relative to `a_true`. An uncertainty estimate helps distinguish such regions only when it tracks the relevant prediction error; the synthetic benchmark explicitly constructs that relationship.

### 3.3 Three mechanisms
1. **`σ` as a first-class Pareto objective.** Optimize `[μ↑, −σ↑, QED↑, −SA↑]`. The `μ−λσ` scalarization becomes an *explicit front*; risk preference is chosen *post hoc* on the front, with no `λ` hyperparameter.
2. **`σ²` as heteroscedastic observation noise.** The activity GP uses `train_Yvar = σ²(x)` per observation, down-weighting observations assigned higher variance. For its posterior mean to estimate true activity, `σ²` must approximate the conditional error variance of `μ` relative to `a_true`, and systematic bias must be addressed. Ranking uncertainty alone does not establish this interpretation.
3. **Risk-aware batch acquisition + constraints.** qNEHVI selects `q` points/round on the above objectives; constraints are handled by feasibility.

A **2×2 ablation** (σ-as-objective × σ-as-noise) isolates the contribution of mechanisms 1 and 2: naive(F,F) → σ-obj(T,F) → σ-noise(F,T) → CU-MOBO(T,T).

### 3.4 Implementation
BoTorch [@balandat2020botorch] `ModelListGP` of `SingleTaskGP`s (input `Normalize`, outcome `Standardize`); the `μ` model takes `train_Yvar=σ²`; `qNoisyExpectedHypervolumeImprovement` [@daulton2021qnehvi] with a Sobol QMC sampler; `optimize_acqf` sequential q-batch. Budget is measured in **real oracle evaluations**. See `algorithms/cu_mobo.py`.

---

## 4. Experiments

### 4.1 Controlled synthetic oracle-gap benchmark (mechanism, with hidden ground truth)

**Setup.** `problems/uncertain_oracle.py`: a known `a_true(x)` the optimizer never sees; the oracle returns `μ = a_true + trap(x)` and a *calibrated* `σ = σ0 + γ·trap(x)`. A reliable optimum `r*` (high `a_true`) and a trap `t*` (inflated `μ`, low `a_true`) are placed symmetrically so QED/SA are identical at both — the *only* discriminator is whether a method trusts `μ` blindly or reads `σ`. Methods: naive / σ-obj / σ-noise / CU-MOBO (2×2 ablation) + qParEGO [@knowles2006parego] + NSGA-II [@deb2002nsga2] (via pymoo [@blank2020pymoo]). Budget 80 oracle evals, n_init 20, batch 10, **5 seeds**. Metrics on the recommended front, computed with the hidden `a_true`: top-K true activity↑, oracle-gap `|μ−a_true|`↓, trap fraction↓.

**Results.** *(from `results/oracle_gap_summary.csv`, 5 seeds, budget 80, mean ± sd)*

| method | top-K true activity ↑ | oracle-gap ↓ | trap frac ↓ | PMO-AUC ↑ |
|---|---|---|---|---|
| naive (= qNEHVI on μ) | 0.284 ± 0.362 | 0.264 | 0.329 | 0.327 |
| σ-objective only | 0.715 ± 0.256 | 0.069 | 0.106 | 0.444 |
| σ-noise only | 0.791 ± 0.120 | 0.041 | 0.023 | 0.506 |
| **CU-MOBO (both)** | **0.835 ± 0.104** | **0.020** | **0.007** | **0.555** |
| qParEGO | 0.175 ± 0.277 | 0.462 | 0.548 | 0.194 |
| NSGA-II | 0.276 ± 0.131 | 0.166 | 0.242 | — |

CU-MOBO recovers designs with **2.9× higher true activity** and **13× smaller oracle-gap** than μ-greedy, landing in the trap region **0.7 % vs 32.9 %** of the time. The 2×2 ablation increases top-K true activity from naive (0.284) through σ-objective (0.715) and σ-noise (0.791) to CU-MOBO (0.835; Fig. 2). Either σ mechanism captures much of the gain; using both produces a smaller additional gain in true activity and a larger reduction in oracle-gap. With five seeds, Cliff's δ for true activity is +0.84 versus naive, +0.92 versus qParEGO, and +1.0 versus NSGA-II; it is approximately +0.20 versus either single-σ ablation. The naive cell uses qNEHVI on μ with both σ mechanisms disabled, isolating their effect within the same acquisition framework.

![Figure 2](../results/oracle_gap.png)
**Figure 2.** Controlled synthetic oracle-gap benchmark (5 seeds, hidden ground truth). *Left:* true activity of the top-K recommended designs (↑) — CU-MOBO and its σ-ablations dominate the no-σ baselines (naive, qParEGO, NSGA-II). *Right:* front oracle-gap |μ−`a_true`| (↓). The 2×2 ablation is monotone: each σ mechanism helps and the two are complementary.

### 4.2 PMO sample efficiency

Top-K (by `μ`) true activity vs real-evaluation budget; area-under-curve (AUC, normalized) summarizes how quickly each method reaches high *true* activity. CU-MOBO has the **highest PMO-AUC (0.555)**, ahead of σ-noise (0.506), σ-objective (0.444), naive (0.327), and far ahead of qParEGO (0.194) — the μ-only baseline spends its budget chasing oracle-gap regions (Fig. 3). I.e. CU-MOBO reaches high *true* activity *sooner* per expensive oracle call, the quantity PMO cares about.

![Figure 3](../results/oracle_gap_convergence.png)
**Figure 3.** PMO sample efficiency: true activity of the top-K (by μ) designs vs real-evaluation budget (mean ± sd, 5 seeds). CU-MOBO reaches high *true* activity soonest per expensive oracle call; the μ-only baselines (naive, qParEGO) plateau low because they spend budget in oracle-gap regions.

### 4.3 Retrospective enrichment on real kcat labels (real-label evidence)

**Setup.** `benchmark/run_retrospective_kcat.py`. Judge = **real experimental kcat labels** (`unified_train_mixed_phys.csv`, 4.3k unique substrates after canonicalization; high = ≥p75, low = ≤p25). A held-out HIGH/LOW set is excluded from surrogate training (leakage-controlled). CU-MOBO optimizes the leakage-controlled real-label surrogate in the kcat-substrate VAE latent space; a random VAE-sampling baseline matches count. Metric: recovery@τ = fraction of held-out HIGH substrates with a generated molecule at Tanimoto ≥ τ; enrichment = recovery_CU / recovery_random; specificity = recovery(HIGH)/recovery(LOW). 3 seeds.

**Results.** *(from `results/retrospective_kcat_summary.csv`, 3 seeds, mean ± sd)*

- recovery@0.4 (HIGH): CU **0.406 ± 0.052** vs random **0.322 ± 0.034** → enrichment **1.26 ± 0.11×** (all 3 seeds > 1; 4,306 unique substrates, 60 held-out HIGH/LOW each)
- specificity (HIGH/LOW): **1.74 ± 0.31×** (CU-guided generation resembles held-out *high*-kcat substrates ~1.7× more than *low*-kcat ones)
- **near-neighbour-removal control:** after deleting training molecules with Tc≥0.6 to the held-out actives, generative enrichment is **1.66 ± 0.30×** (specificity 2.17×; `run_retrospective_kcat.py --nn-remove 0.6`).

We additionally assess how well the surrogate ranks held-out high-kcat substrates under random and scaffold-based splits.

**Hardened surrogate-screening enrichment (rigorous companion).** The recovery@0.4 metric and random hold-out above are lenient (random molecule pairs sit at Tanimoto ≈ 0.1–0.2, and random splits leave analogues of held-out actives in training). We therefore add a standard virtual-screening analysis of the *leakage-controlled surrogate itself*: rank a held-out pool (60 true-high-kcat actives among 540 background decoys, none in training) by predicted μ, reporting Enrichment Factor, BEDROC(α=80.5) [@truchon2007bedroc] and ROC-AUC under both random and **Bemis–Murcko scaffold** [@bemis1996scaffolds] splits, against a **1-NN Tanimoto memorization baseline**, with bootstrap 95 % CIs over 5 seeds (`run_retrospective_kcat_hardened.py`, Fig. 4):

| split | surrogate AUC [95 % CI] | surrogate BEDROC | EF₁% | 1-NN AUC | 1-NN BEDROC |
|---|---|---|---|---|---|
| random | 0.799 [0.78, 0.82] | 0.485 | 5.0× | 0.691 | 0.510 |
| **scaffold** | **0.651 [0.62, 0.69]** | 0.218 | 1.33× | 0.599 | 0.210 |

The scaffold-split AUC is 0.651 [0.62, 0.69], above the 1-NN baseline's 0.599. Under this split, EF₁% is 1.33 versus 5.0 for a random split, and BEDROC is 0.218 versus 0.485. The random-split 1-NN baseline has comparable BEDROC (0.510 vs 0.485). Together, these comparisons quantify the contribution of analogue overlap to retrospective enrichment. *(BEDROC [@truchon2007bedroc]; scaffold splitting [@wu2018moleculenet]; virtual-screening controls [@trannguyen2020litpcba].)*

![Figure 4](../results/retrospective_hardened.png)
**Figure 4.** Hardened retrospective screening on real kcat labels. Enrichment Factor / BEDROC(α=80.5) / ROC-AUC of the leakage-controlled surrogate under random vs Bemis–Murcko **scaffold** splits, against a 1-NN memorization baseline, with bootstrap 95 % CIs (5 seeds). The surrogate generalises to unseen scaffolds (AUC 0.65, CI excludes 0.5) and beats 1-NN; the random→scaffold drop quantifies analogue bias.

### 4.4 Calibration of σ

The two σ mechanisms need different evidence. Using σ as an objective requires it to distinguish more from less reliable predictions; we measure this with Spearman ρ(σ,|μ−y|). Using `σ²` as `train_Yvar` additionally requires an error-variance interpretation; we assess dispersion with ENCE [@levi2022ence] and a reliability diagram [@kuleshov2018calibrated]. In the synthetic benchmark σ is an affine function of the constructed error (ρ = 1.0).

On the real-kcat surrogate (4,306 unique substrates, 25 % held out, log₁₀kcat spanning ~13 units), both RF-derived σ estimates show **weak error correlation**, and the OOF-residual estimate does not beat tree variance:

| σ source | Spearman ρ(σ,\|err\|) ↑ | ENCE ↓ | miscalibration area ↓ | NLL ↓ |
|---|---|---|---|---|
| RF tree-variance | **0.213** (p=1e-12) | **0.258** | **0.017** | **1.908** |
| OOF-residual ("calibrated") | 0.180 (p=3e-9) | 0.404 | 0.022 | 1.964 |

Both σ estimates rank errors weakly (ρ ≈ 0.2); the OOF-residual estimate is worse than tree variance on all four reported metrics (Fig. 5). The difference between constructed σ in the synthetic task and weakly calibrated σ on real-kcat labels limits what the former can predict about real-data performance. Neither estimate establishes that `σ²` is the conditional variance of oracle prediction error. More suitable uncertainty estimation and calibration remain necessary for testing that interpretation [@hirschfeld2020uq; @chen2025uqdesign].

![Figure 5](../results/calibration.png)
**Figure 5.** σ-calibration audit on the real-kcat surrogate. Reliability diagram, ENCE and Spearman ρ(σ,|error|) for tree variance and OOF-residual σ. Both correlate weakly with error (ρ≈0.2); the OOF estimate performs worse on the reported calibration metrics.

### 4.5 Diversity and novelty of the recommended molecules

We measure diversity within the recommended set and novelty relative to the training molecules (MOSES [@polykovskiy2020moses], GuacaMol [@brown2019guacamol], and #Circles [@xie2023circles]), comparing CU-MOBO with μ-greedy and random sampling in the kcat latent (`run_diversity.py`, 3 seeds, Fig. 6):

| method | μ (quality) ↑ | IntDiv₁ ↑ | #Circles ↑ | #scaffolds ↑ | SNN→train | novel-frac ↑ |
|---|---|---|---|---|---|---|
| **CU-MOBO** | **1.84** | **0.860** | **19.0** | **5.7** | 0.755 | 0.095 |
| naive (μ-greedy) | 1.78 | 0.839 | 16.3 | 4.0 | 0.660 | 0.120 |
| random | 1.46 | 0.889 | 19.0 | 4.7 | 0.596 | 0.210 |

CU-MOBO has higher predicted quality and slightly higher internal diversity than μ-greedy (IntDiv₁ 0.860 vs 0.839; #Circles 19.0 vs 16.3), but its recommendations lie closer to the training set (mean nearest-neighbour Tanimoto 0.755 vs 0.660; novel fraction 0.095 vs 0.120). The results show a diversity–novelty trade-off: within-set diversity is maintained while distance from known molecules decreases. This pattern is consistent with σ-aware selection favoring regions near training data, although the present comparison does not isolate that mechanism.

![Figure 6](../results/diversity.png)
**Figure 6.** Diversity and novelty of the recommended set (3 seeds): IntDiv₁, #Circles, scaffold count and nearest-neighbour Tanimoto to the oracle's training set, for CU-MOBO, μ-greedy and random sampling. CU-MOBO maintains internal diversity but yields fewer novel molecules.

### 4.6 External evaluation on a TDC QED scorer

We use a **Therapeutics Data Commons (TDC) QED scorer** [@huang2021tdc; @bickerton2012qed] as an external judge (`run_tdc_anchor.py`, 3 seeds). The optimizer sees a Morgan-fingerprint [@rogers2010ecfp] random-forest [@breiman2001rf] surrogate trained on a limited label budget, with out-of-fold residuals used to estimate σ; the TDC scorer evaluates its generated molecules. Baselines are μ-greedy, the two single-σ ablations, qParEGO, NSGA-II and random search.

QED is a deterministic drug-likeness score that the fingerprint surrogate approximates well in this experiment. We therefore use two label budgets, 600 and 120, to assess whether CU-MOBO preserves QED performance when the oracle-gap is small. This task does not test resistance to a large predictive gap.

| method | QED-rich top-K-true ↑ | gap ↓ | QED-scarce top-K-true ↑ | gap ↓ |
|---|---|---|---|---|
| **CU-MOBO** | **0.894** ± .025 | **0.056** | 0.864 ± .029 | **0.078** |
| naive (μ-greedy) | 0.863 ± .053 | 0.068 | 0.826 ± .038 | 0.078 |
| σ-objective | 0.848 ± .052 | 0.088 | 0.854 ± .004 | 0.102 |
| σ-noise | 0.855 ± .036 | 0.087 | **0.881** ± .018 | 0.085 |
| qParEGO | 0.829 ± .051 | 0.074 | 0.878 ± .038 | 0.095 |
| random | 0.821 ± .023 | 0.177 | 0.821 ± .021 | 0.183 |
| NSGA-II | 0.879 ± .027 | 0.119 | 0.863 ± .060 | 0.095 |

At 600 labels, CU-MOBO has the highest mean top-K QED (0.894 ± 0.025) and lowest gap (0.056); μ-greedy reaches 0.863 ± 0.053 and gap 0.068. At 120 labels, CU-MOBO reaches 0.864 ± 0.029 and gap 0.078, while μ-greedy reaches 0.826 ± 0.038 and gap 0.078. The σ-noise-only ablation has the highest mean QED at 120 labels (0.881 ± 0.018). The small gaps and three-seed sample limit conclusions about method ranking on this scorer.

![Figure 7](../results/tdc_anchor.png)
**Figure 7.** TDC QED evaluation (3 seeds) at surrogate label budgets of 600 and 120. **(a)** QED of top-K designs; **(b)** front oracle-gap. Mean CU-MOBO QED exceeds μ-greedy at both budgets, while gaps remain small.

---

## 5. Preliminary extensions (D1–D4): relaxing the preconditions

The σ-objective can divert search from a high-uncertainty optimum, and GP-based search can degrade as latent dimension grows. We explored four preliminary variants on synthetic or proxy tasks; their results are separate from the core experiments in §4. Further implementation details are in `docs/cu_mobo_improvements.md`.

**E1 — Trust-region search.** A TuRBO-style [@eriksson2019turbo] local-GP trust region (`trust_region=True`) improves the synthetic objective at n_var=8 (0.54 vs 0.38) but worsens it at n_var=16 (0.19 vs 0.39). We therefore do not use it in the core method. Out-of-fold residuals supply a separate σ estimate, whose calibration is assessed in §4.4.

**E2 — Adaptive reliability.** When the true optimum has moderate σ, minimizing σ can favor a low-σ decoy. A one-sided penalty, `μ_eff = μ − λ·relu(σ − τ)` with `τ` a σ quantile, permits moderate-σ candidates while penalizing extreme σ. On a synthetic task with this structure, the variant reaches mean true activity **0.779 ± 0.077** and optimum frequency **0.175** (3 seeds), versus 0.730 / 0.112 for the static σ-objective and 0.736 ± 0.251 for naive search. This is a task-specific result; robustness to weakly calibrated σ remains untested. *(cf. DyRAMO [@yoshizawa2025dyramo].)*

**E3 — Property-aware latent.** A property head on the VAE [@gomezbombarelli2018vae] latent (`z→activity`) raises held-out linear-probe AUC from 0.635 to 1.000 in a 16-dimensional latent (raw-fingerprint AUC 0.975). At supervision weight `w_prop=5`, random-latent validity falls from 174 to 33/200; at `w_prop=1`, it is 129/200. These results characterize the balance between property organization and decoder validity. *(cf. LOL-BO [@maus2022lolbo] / CoBO [@lee2023cobo].)*

**E4 — Flow-guided generation resists reward hacking (couples CU-MOBO with a generative flow).** On the property-aware latent we train a rectified-flow [@liu2023rectifiedflow] velocity field and steer sampling by the property gradient [@dhariwal2021guidance], `z ← z + dt·(v(z,t) + s·∇g(z))`. Scored by an **independent real oracle**, moderate guidance raises true activity **2.3× (0.108 → 0.247)** and the active fraction 11%→20%. Sweeping the guidance strength `s` traces a **Conservative-Objective-Models-style over-optimization curve** (Fig. 8; cf. COMs, Trabucco et al. ICML 2021, Fig. 3): as `s` rises (0 → 0.3 → 0.6 → ∞), the proxy head's self-belief climbs monotonically (0.11 → 0.74 → 0.99 → 1.00) while true activity **peaks at s=0.6 then collapses to 0** when the flow prior is removed entirely (gradient ascent alone) — the reward-hacking gap widening from 0.003 to **1.00**. **The generative flow prior is the regularizer that keeps acquisition-gradient guidance on-manifold:** the same oracle-gap principle of §1 reappearing one level up, inside the generator, and the same "optimize-the-proxy-harder → true-value-falls" signature that the offline-MBO literature uses as its canonical reward-hacking diagnostic. Guidance strength `s` is the resulting validity↔activity Pareto knob. *(cf. LaMBO-2 [@gruver2023lambo2]; COMs [@trabucco2021coms] / Design-Bench [@trabucco2022designbench].)*

![Figure 8](../results/overoptimization.png)
**Figure 8.** E4 over-optimization (COMs-style) curve. As the flow-guidance strength `s` increases (0 → 0.3 → 0.6 → ∞ = gradient-ascent only), the proxy head's self-belief rises *monotonically* while true activity (scored by an *independent* oracle) peaks at `s`=0.6 then **collapses** once the flow prior is removed — the reward-hacking signature of §1 reappearing inside the generator; the flow prior is the on-manifold regulariser.

These variants identify possible changes to the search objective, latent representation and proposal mechanism for subsequent evaluation.

---

## 6. Limitations

- **Uncertainty estimation.** In the synthetic benchmark, σ tracks constructed error by design; on the real-kcat surrogate, its error-ranking correlation is about 0.2. This difference makes stronger error-variance calibration a priority. The adaptive objective in §5 also addresses cases where promising regions have moderate σ.
- **Task and scale.** The main search uses a 16-dimensional latent and limited oracle budgets. Trust-region behavior varies with dimension (§5), while the TDC QED task has a small oracle-gap. Higher-dimensional and more error-prone oracle tasks are natural next evaluations.

---

## 7. Conclusion

CU-MOBO inserts an oracle uncertainty estimate into the Pareto objectives and activity-GP noise model. In a controlled benchmark with informative σ, both mechanisms improve the true activity of recommended designs and reduce oracle-gap. On experimental kcat data, the method enriches generated molecules for similarity to held-out high-kcat substrates. The calibration audit identifies uncertainty estimation as a priority for extending the method to more demanding oracle settings.

---

## References

::: {#refs}
:::
