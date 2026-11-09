# Tool reference: group-comparison

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

## `ab_test`

`from statkit.inference import ab_test`

```python
ab_test(df: 'DataFrame', group_col: 'str', value_col: 'str', group_a, group_b, alternative: 'str' = 'two-sided') -> 'ABTestReport'
```

Welch's two-sample t-test (unequal variances assumed by default, since
that assumption is the safer default and StatABench's own `ab_ttest`
tool makes the same choice), plus Cohen's d.

Reporting Cohen's d alongside the p-value is deliberate: DOCUMENTATION.md
§6.2 found that none of the three benchmarked systems distinguished
statistical from practical significance in their narrative output, and a
downstream `interpret_result` call needs an effect-size number to make
that distinction rather than inventing one from the p-value alone.

Returns `ABTestReport` with: `group_col`, `value_col`, `group_a`, `group_b`, `mean_a`, `mean_b`, `n_a`, `n_b`, `t_statistic`, `p_value`, `cohens_d`, `significant_at_05`

## `two_sample_confidence_interval`

`from statkit.hypothesis import two_sample_confidence_interval`

```python
two_sample_confidence_interval(df: 'DataFrame', group_col: 'str', value_col: 'str', group_a, group_b, level: 'float' = 0.95) -> 'CIReport'
```

Welch CI for mean(a) - mean(b) (cf. StatABench ci_two_normal).

Returns `CIReport` with: `estimate`, `lower`, `upper`, `level`, `n`, `method`

## `nonparametric_test`

`from statkit.hypothesis import nonparametric_test`

```python
nonparametric_test(df: 'DataFrame', group_col: 'str', value_col: 'str', groups: 'list | None' = None, pair_col: 'str | None' = None) -> 'NonparametricReport'
```

Mann-Whitney U (2 groups), Wilcoxon signed-rank (2 groups, paired by `pair_col`) or
Kruskal-Wallis (3+), chosen from the group count (cf. StatABench nonparametric_test). Effect size is
rank-biserial r (two groups) or eta-squared-H (Kruskal), so interpret_result can bucket it.

Pairing is by an explicit subject id, never by row position: StatABench's wilcoxon branch truncates
both columns to the shorter length and pairs by position, which silently mispairs subjects as soon
as one side has a missing value.

Returns `NonparametricReport` with: `method`, `statistic`, `p_value`, `effect_size`, `effect_size_type`, `groups`, `significant_at_05`

## `chi_square_association`

`from statkit.inference import chi_square_association`

```python
chi_square_association(df: 'DataFrame', col_a: 'str', col_b: 'str') -> 'ChiSquareReport'
```

Chi-square test of independence between two categorical columns, plus
Cramer's V so the result carries an effect size alongside the p-value --
a p-value alone is exactly the kind of output DOCUMENTATION.md §6.2
flags as uninterpreted: a number with no sense of how large the effect
actually is.

Generalizes StatABench's `contingency_test`.

Returns `ChiSquareReport` with: `col_a`, `col_b`, `chi2`, `p_value`, `dof`, `cramers_v`, `contingency_table`, `significant_at_05`, `n`, `expected_min_cell`

## `variance_equality_test`

`from statkit.hypothesis import variance_equality_test`

```python
variance_equality_test(df: 'DataFrame', group_col: 'str', value_col: 'str', group_a, group_b) -> 'VarianceTestReport'
```

Mood's two-sample scale test (cf. StatABench mood_variance_test); robust to non-normality, unlike F.

Returns `VarianceTestReport` with: `method`, `statistic`, `p_value`, `var_a`, `var_b`, `significant_at_05`

## `ks_normality_test`

`from statkit.hypothesis import ks_normality_test`

```python
ks_normality_test(df: 'DataFrame', column: 'str') -> 'NormalityReport'
```

KS-type normality test with the Lilliefors correction (cf. StatABench ks_test).

StatABench's ks_test fits the distribution's parameters to the sample and then reads a plain KS
p-value. Because the parameters came from the same data, that p-value is far too large: the test
almost never rejects. Lilliefors uses the same KS statistic with the correct null distribution
for estimated mean/sd. Skew is returned as a second signal.

Returns `NormalityReport` with: `column`, `statistic`, `p_value`, `n`, `looks_normal`, `skew`

## `bootstrap_ab_test`

`from statkit.hypothesis import bootstrap_ab_test`

```python
bootstrap_ab_test(df: 'DataFrame', group_col: 'str', value_col: 'str', group_a, group_b, n_resamples: 'int' = 5000, seed: 'int' = 0) -> 'BootstrapABReport'
```

Percentile-bootstrap CI for mean(a) - mean(b) plus a permutation-style p-value (cf. StatABench bootstrap_abtest).
Seeded, so the same call returns the same number -- an agent's result must be reproducible.

Returns `BootstrapABReport` with: `mean_diff`, `ci_lower`, `ci_upper`, `p_value`, `n_resamples`, `significant_at_05`

## `ab_power_analysis`

`from statkit.hypothesis import ab_power_analysis`

```python
ab_power_analysis(effect_size: 'float | None' = None, alpha: 'float' = 0.05, power: 'float | None' = 0.8, n_per_group: 'int | None' = None, proportions: 'tuple[float, float] | None' = None) -> 'PowerReport'
```

Two-sided, two-sample power analysis (cf. StatABench ab_power_analysis, which also uses
statsmodels). Means: pass Cohen's d as `effect_size`; exact noncentral-t (TTestIndPower).
Proportions: pass `proportions=(baseline, target)`; Cohen's h with NormalIndPower.
Give `power` to solve for n per group, or `n_per_group` (with power=None) to solve for power.

Returns `PowerReport` with: `effect_size`, `alpha`, `power`, `n_per_group`, `solved_for`

## `multiple_comparison_correction`

`from statkit.hypothesis import multiple_comparison_correction`

```python
multiple_comparison_correction(p_values: 'list[float]', method: 'str' = 'fdr_bh', alpha: 'float' = 0.05) -> 'MultipleComparisonReport'
```

Adjust a family of p-values: fdr_bh (Benjamini-Hochberg), bonferroni, or holm (StatABench
fdr_control_df / fwer_control_df). Reports how many discoveries the correction removed.

Returns `MultipleComparisonReport` with: `method`, `alpha`, `p_values`, `adjusted`, `rejected`, `n_rejected`, `n_rejected_uncorrected`

## `bayesian_estimate`

`from statkit.bayesian import bayesian_estimate`

```python
bayesian_estimate(df: 'DataFrame', model: 'str', column: 'str', group_col: 'str | None' = None, groups: 'list | None' = None, level: 'float' = 0.95, prior_a: 'float' = 1.0, prior_b: 'float' = 1.0, n_draws: 'int' = 20000, seed: 'int' = 0) -> 'BayesReport'
```

Three models:

- `proportion`: Beta-Binomial for a 0/1 column; with `group_col` and two `groups`, also
  P(p_a > p_b). Prior Beta(prior_a, prior_b), default uniform.
- `mean`: normal data, unknown variance, Jeffreys prior. The posterior for the mean is a
  Student t, so its credible interval equals the frequentist t interval; the value added is the
  direct statement P(mu > 0). With two groups, the posterior of the difference by simulation.
- `hierarchical`: normal-normal partial pooling of group means (empirical Bayes, method of
  moments for the between-group variance). Small groups are shrunk toward the grand mean.

Returns `BayesReport` with: `model`, `posterior`, `prob_statements`, `prior`, `n`

## `kaplan_meier_curve`

`from statkit.survival import kaplan_meier_curve`

```python
kaplan_meier_curve(df: 'DataFrame', duration: 'str', event: 'str') -> 'KaplanMeierReport'
```

Product-limit survival curve with a 95% CI and median survival (cf. StatABench kaplan_meier_plot,
which returns the curve but no interval). The interval is Greenwood's variance on the log(-log)
scale, the usual default (also lifelines'), which keeps bounds inside [0, 1] without clipping.
The curve is returned as data; plot `times` vs `survival` as a step function.

Returns `KaplanMeierReport` with: `n`, `n_events`, `n_censored`, `median_survival`, `times`, `survival`, `ci_lower`, `ci_upper`

## `logrank_test`

`from statkit.survival import logrank_test`

```python
logrank_test(df: 'DataFrame', duration: 'str', event: 'str', group_col: 'str', group_a, group_b) -> 'LogRankReport'
```

Two-group log-rank test (cf. StatABench logrank_test_compare). It compares whole curves, so it
has little power when the curves cross -- the warning says so when it detects a crossing.

Returns `LogRankReport` with: `group_a`, `group_b`, `n_a`, `n_b`, `events_a`, `events_b`, `chi2`, `p_value`, `significant_at_05`

## `describe_column`

`from statkit.descriptive import describe_column`

```python
describe_column(df: 'DataFrame', column: 'str') -> 'ColumnDescription'
```

Numeric columns get mean/median/std/min/max/skew/kurtosis; categorical
columns get mode, top-5 value counts, and cardinality. StatABench's
`calculate_statistic` returns one named statistic and leaves the caller to
know the column is numeric (a `mean` on a text column just raises); this
tool dispatches on dtype itself and returns a self-describing summary, so
an interpretation step doesn't need to re-inspect the dataframe.

Returns `ColumnDescription` with: `column`, `dtype`, `is_numeric`, `n`, `n_missing`, `stats`

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
