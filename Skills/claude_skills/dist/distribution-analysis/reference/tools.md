# Tool reference: distribution-analysis

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

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

## `detect_outliers_iqr`

`from statkit.cleaning import detect_outliers_iqr`

```python
detect_outliers_iqr(df: 'DataFrame', column: 'str', k: 'float' = 1.5) -> 'OutlierReport'
```

Tukey fences (Q1 - k*IQR, Q3 + k*IQR). Reports the rows; never silently drops them.

Returns `OutlierReport` with: `column`, `q1`, `q3`, `lower_fence`, `upper_fence`, `n_outliers`, `pct_outliers`, `outlier_index`

## `confidence_interval`

`from statkit.hypothesis import confidence_interval`

```python
confidence_interval(df: 'DataFrame', column: 'str', level: 'float' = 0.95) -> 'CIReport'
```

One-sample CI for the mean using the t distribution (cf. StatABench ci_normal).

Returns `CIReport` with: `estimate`, `lower`, `upper`, `level`, `n`, `method`

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

## `bin_continuous`

`from statkit.cleaning import bin_continuous`

```python
bin_continuous(df: 'DataFrame', column: 'str', bins: 'int | list[float]' = 5, strategy: 'str' = 'quantile', new_column: 'str | None' = None) -> 'TransformResult'
```

Discretize a numeric column into equal-width or quantile bins (cf. MetaGPT SplitBins).

Returns `TransformResult` with: `df`, `details`

## `frequency_table`

`from statkit.exploration import frequency_table`

```python
frequency_table(df: 'DataFrame', column: 'str', top_n: 'int | None' = None, include_missing: 'bool' = False) -> 'FrequencyTable'
```

value_counts with proportions. Proportions are over non-missing rows unless
include_missing=True, and the denominator is reported, because "what fraction"
questions silently change answer with the denominator (Titanic age, deck).

Returns `FrequencyTable` with: `column`, `n`, `n_missing`, `table`

## `recommend_chart`

`from statkit.visualization import recommend_chart`

```python
recommend_chart(df: 'DataFrame', x: 'str | None' = None, y: 'str | None' = None, color: 'str | None' = None, columns: 'list[str] | None' = None) -> 'ChartSpec'
```

Pick a chart type from the roles of the requested columns (cf. LIDA goal()+visualize(), rule-based).
Pass `columns` (3+ numeric columns) instead of x/y to get a correlation heatmap.

Returns `ChartSpec` with: `chart_type`, `x`, `y`, `color`, `columns`, `aggregate`, `rationale`

## `render_chart`

`from statkit.visualization import render_chart`

```python
render_chart(df: 'DataFrame', spec: 'ChartSpec', path: 'str | pathlib.Path', title: 'str | None' = None) -> 'RenderReport'
```

Draw a ChartSpec to a PNG with matplotlib (cf. LIDA execute()). Grounding is re-checked at draw time,
so a spec built against one frame cannot silently be drawn against another.

Returns `RenderReport` with: `path`, `chart_type`, `n_points`

## `explain_chart`

`from statkit.visualization import explain_chart`

```python
explain_chart(df: 'DataFrame', spec: 'ChartSpec') -> 'ChartExplanation'
```

State what the chart *shows about the data*, computed from the data, never from the spec's styling
(cf. LIDA explain(), redefined per §6.2). Every sentence is backed by a number in `facts`.

Returns `ChartExplanation` with: `chart_type`, `finding`, `facts`

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
