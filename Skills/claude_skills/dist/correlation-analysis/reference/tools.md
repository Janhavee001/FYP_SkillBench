# Tool reference: correlation-analysis

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

## `correlation_matrix`

`from statkit.descriptive import correlation_matrix`

```python
correlation_matrix(df: 'DataFrame', columns: 'list[str] | None' = None, method: 'str' = 'pearson') -> 'CorrelationReport'
```

Pairwise correlation over the given (or all-numeric) columns, plus the
single strongest off-diagonal pair called out explicitly.

Generalizes StatABench's `correlation_analysis`, which computes one pair
at a time; the shared test set's wine-quality questions ask "which
variable correlates most with quality," which needs the matrix reduced
to its strongest entry, not just the raw matrix.

Returns `CorrelationReport` with: `method`, `columns`, `matrix`, `strongest_pair`, `strongest_r`

## `linear_regression`

`from statkit.modeling import linear_regression`

```python
linear_regression(df: 'DataFrame', x: 'str', y: 'str') -> 'RegressionReport'
```

Simple OLS of y on one predictor (cf. StatABench simple_linear_regression).

Returns `RegressionReport` with: `y`, `predictors`, `n`, `n_dropped`, `r_squared`, `adj_r_squared`, `coefficients`, `vif`, `residual_normality_p`

## `multiple_comparison_correction`

`from statkit.hypothesis import multiple_comparison_correction`

```python
multiple_comparison_correction(p_values: 'list[float]', method: 'str' = 'fdr_bh', alpha: 'float' = 0.05) -> 'MultipleComparisonReport'
```

Adjust a family of p-values: fdr_bh (Benjamini-Hochberg), bonferroni, or holm (StatABench
fdr_control_df / fwer_control_df). Reports how many discoveries the correction removed.

Returns `MultipleComparisonReport` with: `method`, `alpha`, `p_values`, `adjusted`, `rejected`, `n_rejected`, `n_rejected_uncorrected`

## `dimensionality_reduction`

`from statkit.modeling import dimensionality_reduction`

```python
dimensionality_reduction(df: 'DataFrame', columns: 'list[str] | None' = None, n_components: 'int' = 2, method: 'str' = 'pca', seed: 'int' = 0) -> 'DimReductionReport'
```

PCA or sparse PCA on standardized numeric columns (cf. StatABench sparse_pca_analysis). Returns loadings,
not just scores: a component nobody can name is not an analytical result.

Returns `DimReductionReport` with: `method`, `n_components`, `explained_variance_ratio`, `cumulative_variance`, `loadings`, `n`

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
