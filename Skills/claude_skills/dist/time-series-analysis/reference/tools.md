# Tool reference: time-series-analysis

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

## `column_summary_for_viz`

`from statkit.exploration import column_summary_for_viz`

```python
column_summary_for_viz(df: 'DataFrame', max_samples: 'int' = 3) -> 'VizColumnSummary'
```

Per-column role (quantitative / temporal / nominal / ordinal-ish / id-like), range and sample
values -- the compact summary a chart recommender needs and nothing more (cf. LIDA summarize()).

Returns `VizColumnSummary` with: `columns`

## `groupby_aggregate`

`from statkit.exploration import groupby_aggregate`

```python
groupby_aggregate(df: 'DataFrame', by: 'str | list[str]', value: 'str', agg: 'str' = 'mean') -> 'GroupAggregateReport'
```

Aggregate `value` by one or more grouping columns, returning group sizes
alongside the aggregate so a mean over n=3 is never read like a mean over n=300.

Returns `GroupAggregateReport` with: `by`, `value`, `agg`, `result`, `group_sizes`

## `stationarity_test`

`from statkit.timeseries import stationarity_test`

```python
stationarity_test(series: 'pd.Series | DataFrame', column: 'str | None' = None, alpha: 'float' = 0.05, regression: 'str' = 'c', time_col: 'str | None' = None) -> 'StationarityReport'
```

ADF (H0: unit root) and KPSS (H0: stationary) run together, because each
alone is easy to over-read; agreement is evidence, disagreement is flagged
(cf. StatABench test_stationarity, which runs one or the other).

With a DataFrame, pass `time_col` so the rows are put in time order first:
both tests read the series in row order, and a shuffled or grouped file
gives a meaningless answer.

Returns `StationarityReport` with: `n`, `adf_statistic`, `adf_p_value`, `kpss_statistic`, `kpss_p_value`, `verdict`

## `seasonal_decompose`

`from statkit.timeseries import seasonal_decompose`

```python
seasonal_decompose(data: 'pd.Series | DataFrame', column: 'str | None' = None, period: 'int | None' = None, time_col: 'str | None' = None) -> 'DecompositionReport'
```

STL decomposition into trend + seasonal + remainder (cf. StatABench decompose_stl, which returns
component means). Reports trend and seasonal *strength* (Hyndman & Athanasopoulos, FPP3 s.4.3:
1 - Var(R)/Var(T+R) and 1 - Var(R)/Var(S+R)), which answer "is there really seasonality?" with a
number. `period` is required: guessing it from the data is a common silent error (7 for daily
data with weekly cycles, 12 for monthly data with yearly cycles).

Returns `DecompositionReport` with: `period`, `n`, `trend_strength`, `seasonal_strength`, `seasonal_amplitude`, `trend_change`, `trend`, `seasonal`, `resid`

## `arima_forecast`

`from statkit.timeseries import arima_forecast`

```python
arima_forecast(data: 'pd.Series | DataFrame', column: 'str | None' = None, horizon: 'int' = 12, order: 'tuple | None' = None, max_p: 'int' = 3, max_q: 'int' = 3, seasonal_period: 'int | None' = None, time_col: 'str | None' = None, criterion: 'str' = 'bic') -> 'ARIMAReport'
```

ARIMA with (p, q) chosen over a small grid (cf. StatABench auto_arima_modeling, which uses
pmdarima with AIC). The default criterion here is BIC: on a simulated AR(1), AIC preferred an
ARMA(2,3) by 1.2 points whose coefficients look nothing like the truth, while BIC picked AR(1).
`criterion` accepts "bic", "aic" or "aicc". d is chosen first with the same ADF+KPSS pair as
stationarity_test, because information criteria are not comparable across different d. Returns a forecast with 95% intervals
and a Ljung-Box check on the residuals.

Returns `ARIMAReport` with: `order`, `seasonal_order`, `criterion`, `criterion_value`, `n`, `coefficients`, `forecast`, `forecast_lower`, `forecast_upper`, `residual_autocorrelation_p`, `candidates_tried`

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
