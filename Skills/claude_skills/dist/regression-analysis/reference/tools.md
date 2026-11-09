# Tool reference: regression-analysis

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

## `check_method_appropriateness`

`from statkit.validation import check_method_appropriateness`

```python
check_method_appropriateness(df: 'DataFrame', method: 'str', columns: 'list[str]') -> 'MethodAppropriatenessReport'
```

Flag a structural mismatch between a declared method and the columns it
is being applied to -- e.g. Pearson correlation requested on a binary
column, or a two-sample test requested on a column with more than two
groups.

StatABench's `judger-open/modeling_groundedness.py` is an LLM-judge prompt
that scores, after the fact, whether a write-up justified its model choice.
This is not a port of it: it is a small set of deterministic rules covering
the structural part of that concern (does the method fit these column
types and group counts?), run before execution, so a verdict is auditable.

Unknown methods return verdict "unchecked" (ok=False), never "appropriate":
a gate that passes whatever it has no rule for gives false assurance.

Returns `MethodAppropriatenessReport` with: `method`, `columns`, `verdict`, `reason`

## `linear_regression`

`from statkit.modeling import linear_regression`

```python
linear_regression(df: 'DataFrame', x: 'str', y: 'str') -> 'RegressionReport'
```

Simple OLS of y on one predictor (cf. StatABench simple_linear_regression).

Returns `RegressionReport` with: `y`, `predictors`, `n`, `n_dropped`, `r_squared`, `adj_r_squared`, `coefficients`, `vif`, `residual_normality_p`

## `multivariable_regression`

`from statkit.modeling import multivariable_regression`

```python
multivariable_regression(df: 'DataFrame', y: 'str', predictors: 'list[str]') -> 'RegressionReport'
```

OLS with several predictors, standard errors, 95% CIs and VIF (cf. StatABench multivariable_linear_regression).

Returns `RegressionReport` with: `y`, `predictors`, `n`, `n_dropped`, `r_squared`, `adj_r_squared`, `coefficients`, `vif`, `residual_normality_p`

## `glm_fit`

`from statkit.modeling import glm_fit`

```python
glm_fit(df: 'DataFrame', y: 'str', predictors: 'list[str]', family: 'str' = 'binomial') -> 'GLMReport'
```

Generalized linear model (cf. StatABench run_glm). Families: gaussian, binomial (logistic),
poisson, gamma, negative_binomial. Categorical predictors are dummy-coded as in
multivariable_regression.

Beyond the reference: `exp_coef` (odds ratio for binomial, rate ratio for poisson /
negative binomial) is returned, because that is the number a reader interprets, and the
Pearson dispersion ratio is checked, since overdispersed Poisson data give p-values that
are far too small.

Returns `GLMReport` with: `family`, `y`, `n`, `coefficients`, `aic`, `deviance`, `pseudo_r_squared`, `dispersion_ratio`

## `robust_regression`

`from statkit.modeling import robust_regression`

```python
robust_regression(df: 'DataFrame', y: 'str', predictors: 'list[str]') -> 'RobustRegressionReport'
```

Huber M-estimation (cf. StatABench huber_regression). Returns the OLS coefficient beside each
robust one and counts the observations given weight < 1: a large OLS-vs-Huber gap is itself
the finding (the OLS fit is being driven by a few points).

Returns `RobustRegressionReport` with: `y`, `n`, `coefficients`, `n_downweighted`, `scale`

## `regularized_regression`

`from statkit.modeling import regularized_regression`

```python
regularized_regression(df: 'DataFrame', y: 'str', predictors: 'list[str]', method: 'str' = 'lasso', cv: 'int' = 5, seed: 'int' = 0) -> 'RegularizedReport'
```

Lasso / ridge / elastic-net with the penalty chosen by cross-validation (cf. StatABench
advanced_regression, which scores on one 80/20 split). Predictors are standardized, so coefficients
are comparable; `selected`/`dropped` say what the penalty kept. Reports both in-sample R^2 and
out-of-fold R^2 from an outer K-fold, with scaling and penalty choice refit inside every fold so
the score is not leaked.

Returns `RegularizedReport` with: `method`, `alpha`, `r_squared`, `cv_r_squared`, `coefficients`, `selected`, `dropped`, `n`

## `dimensionality_reduction`

`from statkit.modeling import dimensionality_reduction`

```python
dimensionality_reduction(df: 'DataFrame', columns: 'list[str] | None' = None, n_components: 'int' = 2, method: 'str' = 'pca', seed: 'int' = 0) -> 'DimReductionReport'
```

PCA or sparse PCA on standardized numeric columns (cf. StatABench sparse_pca_analysis). Returns loadings,
not just scores: a component nobody can name is not an analytical result.

Returns `DimReductionReport` with: `method`, `n_components`, `explained_variance_ratio`, `cumulative_variance`, `loadings`, `n`

## `feature_selection`

`from statkit.modeling import feature_selection`

```python
feature_selection(df: 'DataFrame', target: 'str | None' = None, columns: 'list[str] | None' = None, method: 'str' = 'variance', threshold: 'float' = 0.0, top_k: 'int | None' = None) -> 'FeatureSelectionReport'
```

`general`: drop all-missing, constant, infinite-valued and ID-like text columns (mirrors MetaGPT
GeneralSelection). `variance`: drop numeric columns with variance <= threshold (MetaGPT
VarianceBasedSelection). `correlation`: rank numeric columns by |Pearson r| with the target; this one
has no MetaGPT counterpart and is univariate, so it misses predictors that matter only jointly.

Returns `FeatureSelectionReport` with: `method`, `scores`, `selected`, `dropped`

## `scale_numeric`

`from statkit.cleaning import scale_numeric`

```python
scale_numeric(df: 'DataFrame', columns: 'list[str] | None' = None, method: 'str' = 'standard') -> 'TransformResult'
```

minmax / standard / robust / maxabs scaling (cf. MetaGPT *Scale transformers).

Returns `TransformResult` with: `df`, `details`

## `encode_categorical`

`from statkit.cleaning import encode_categorical`

```python
encode_categorical(df: 'DataFrame', columns: 'list[str]', method: 'str' = 'onehot', target: 'str | None' = None, order: 'dict | None' = None, n_splits: 'int' = 5) -> 'TransformResult'
```

ordinal / onehot / label / target_mean / kfold_target_mean (cf. MetaGPT *Encode*, TargetMeanEncoder).

Returns `TransformResult` with: `df`, `details`

## `polynomial_expand`

`from statkit.cleaning import polynomial_expand`

```python
polynomial_expand(df: 'DataFrame', columns: 'list[str]', degree: 'int' = 2, interaction_only: 'bool' = False) -> 'TransformResult'
```

Add polynomial and interaction terms of numeric columns (cf. MetaGPT PolynomialExpansion,
which keeps only the 10 columns most correlated with the label; here the caller chooses the
columns explicitly, so nothing is selected behind their back). Rows with missing values get
NaN features rather than being imputed with 0 as MetaGPT does.

Returns `TransformResult` with: `df`, `details`

## `engineer_categorical_interactions`

`from statkit.cleaning import engineer_categorical_interactions`

```python
engineer_categorical_interactions(df: 'DataFrame', method: 'str', columns: 'list[str]', agg_col: 'str | None' = None, agg_funcs: 'list[str] | None' = None, max_categories: 'int' = 100) -> 'TransformResult'
```

New features from categorical columns, added beside the originals (cf. MetaGPT CatCount,
CatCross, GroupStat):

- `count`: how often each value of columns[0] occurs (`<col>_cnt`).
- `cross`: every pair of the given columns combined into one categorical (`<a>_<b>`); pairs
  whose cross would exceed `max_categories` levels are skipped and reported (MetaGPT applies
  the same cap).
- `group_stat`: per-group aggregates of `agg_col` by columns[0] (`<agg_col>_<func>_by_<col>`).

Group statistics computed on the full data leak the outcome when `agg_col` is the target;
that case is refused.

Returns `TransformResult` with: `df`, `details`

## `cox_regression`

`from statkit.survival import cox_regression`

```python
cox_regression(df: 'DataFrame', duration: 'str', event: 'str', covariates: 'list[str]', ties: 'str' = 'efron') -> 'CoxReport'
```

Cox proportional-hazards regression (cf. StatABench fit_cox_model, which uses lifelines; this
uses statsmodels PHReg with Efron ties, the R `coxph` default). Categorical covariates are
dummy-coded. The proportional-hazards assumption is checked per covariate by correlating
Schoenfeld residuals with event time (Grambsch-Therneau style, rank of time); p < 0.05 means
the hazard ratio changes over time and a single HR misleads.

Returns `CoxReport` with: `n`, `n_events`, `covariates`, `hazard_ratios`, `proportional_hazards_p`

## `propensity_score_matching`

`from statkit.causal import propensity_score_matching`

```python
propensity_score_matching(df: 'DataFrame', treatment: 'str', outcome: 'str', covariates: 'list[str]', caliper: 'float' = 0.2, n_bootstrap: 'int' = 500, seed: 'int' = 0) -> 'PSMReport'
```

Average treatment effect on the treated by 1:1 nearest-neighbour matching on the logit of the
propensity score, with replacement (cf. StatABench estimate_ATT_with_psm).

The caliper is in standard deviations of the logit propensity score (0.2 is Austin's 2011
recommendation), not in raw probability units. Covariate balance is reported as standardized
mean differences before and after matching; |SMD| > 0.1 after matching means the groups are
still not comparable on that covariate. The CI is a bootstrap over matched pairs, which ignores
uncertainty in the propensity model and so is somewhat too narrow.

Returns `PSMReport` with: `att`, `ci_lower`, `ci_upper`, `naive_difference`, `n_treated`, `n_matched`, `n_unmatched`, `max_abs_smd_before`, `max_abs_smd_after`, `balance`

## `difference_in_differences`

`from statkit.causal import difference_in_differences`

```python
difference_in_differences(df: 'DataFrame', outcome: 'str', group_col: 'str', post_col: 'str', time_col: 'str | None' = None, cluster_col: 'str | None' = None) -> 'DiDReport'
```

Two-group, two-period difference-in-differences: OLS of outcome on group, post and group x post;
the interaction is the effect (cf. StatABench estimate_did_effect). `group_col` and `post_col`
are 0/1. Standard errors are clustered on `cluster_col` when given (units observed repeatedly),
else heteroskedasticity-robust (HC1).

The estimate rests on parallel trends. When `time_col` has 2+ pre-treatment periods, the tool
tests whether the groups' trends already differed before treatment (group x time on the
pre-period only); a small p-value there undermines the whole estimate.

Returns `DiDReport` with: `effect`, `se`, `p_value`, `ci_lower`, `ci_upper`, `cell_means`, `pre_trend_p`, `n`

## `interpret_result`

`from statkit.interpretation import interpret_result`

```python
interpret_result(method: 'str', p_value: 'float | None' = None, effect_size: 'float | None' = None, effect_size_type: 'str' = 'cohens_d', n: 'int | None' = None, question: 'str' = '') -> 'Interpretation'
```

Produce a structured interpretation, not narrative prose, from a
statistical result. `effect_size_type` selects which convention table to
bucket against ("cohens_d" or "cramers_v"); extend the tables above
rather than the function body when adding a new test family.

The distinction this tool exists to enforce: a p-value answers "is this
effect distinguishable from zero," and an effect size answers "is this
effect big enough to matter." Collapsing the two into one sentence is
exactly the kind of unexamined interpretation StatABench's
`result_bias.py` anchors penalise, and this tool keeps them as separate
fields so a caller cannot report one while implying the other.

Returns `Interpretation` with: `method`, `statistically_significant`, `effect_size_label`, `practically_significant`, `headline`, `caveats`

## `critique_findings`

`from statkit.interpretation import critique_findings`

```python
critique_findings(method: 'str', n: 'int | None' = None, group_sizes: 'list[int] | None' = None, normality_p: 'float | None' = None, variance_equality_p: 'float | None' = None, p_value: 'float | None' = None, n_tests: 'int' = 1, observational: 'bool' = True, missing_pct: 'float | None' = None, expected_min_cell: 'float | None' = None) -> 'Critique'
```

Check a result against the assumptions its method relies on and report
violations, ranked by severity. StatABench's judger-open/result_bias.py is
an LLM-judge prompt that scores, after the fact, whether a write-up
identified and corrected biases; this tool asks the checkable subset of
those questions (sample size, assumptions, multiplicity, confounding,
missingness) as deterministic rules, *before* the claim is written.

Returns `Critique` with: `method`, `violations`, `n_high`

## `verify_numeric_consistency`

`from statkit.verification import verify_numeric_consistency`

```python
verify_numeric_consistency(claimed: 'float', recomputed: 'float', rel_tol: 'float' = 0.001) -> 'NumericConsistencyReport'
```

Compare the number a narrative claims against an independent recompute and
name the *kind* of disagreement. Targets DOCUMENTATION §7.1 #2: the model
printed 38.38% against a ground truth of 0.3838 -- the same quantity in a
different unit, which a plain equality check scores as simply wrong.

Returns `NumericConsistencyReport` with: `claimed`, `recomputed`, `relation`, `scale_factor`
