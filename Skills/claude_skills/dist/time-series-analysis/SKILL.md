---
name: time-series-analysis
description: Analyse how a measure changes over time - put data in time order, aggregate to a regular frequency, test stationarity (ADF + KPSS), measure trend and seasonality strength with STL decomposition, forecast with ARIMA (prediction intervals, residual checks), and draw line charts. Use when the user asks about trends, seasonality, "how does X change over time", peaks, cycles, or forecasts, or when the data has a date or time column. Uses the bundled statkit Python tools.
---

# Time-series and trend analysis

Time-series rows are not independent: today resembles yesterday. Methods built for independent
rows (a t-test between years, a regression on time) overstate certainty. This skill orders
the data, characterises trend and seasonality with explicit strength measures, and only then
forecasts.

## When to use

- "How has demand changed?", "Is there a weekly pattern?", "Forecast the next 30 days.",
  "When is the peak?"

**Not for:** time until an event (use survival tools in `group-comparison` or
`regression-analysis`), or before/after comparisons of a policy across groups
(`difference_in_differences` in `regression-analysis`).

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Load data with `statkit.loading.load_dataset`. Tool signatures are in `reference/tools.md`.

## Procedure

1. **Find the time column.** `column_summary_for_viz(df)` marks date strings as `temporal`.
2. **One row per time point.** If timestamps repeat (several stores per day, hourly rows for a
   daily question), aggregate first:
   `groupby_aggregate(df, "date", "value", "sum")`, or pandas `resample` on a parsed date. Every
   tool here refuses repeated timestamps rather than mixing series.
3. **Look first:** `recommend_chart(df, time_col, value)` gives a line chart. `render_chart` and
   `explain_chart` report start, end, trend and peak.
4. **Stationarity:** `stationarity_test(df, value, time_col=time_col)`. "non_stationary" means
   difference the series before modelling. ADF and KPSS disagreeing means detrend or difference
   and re-test.
5. **Trend and seasonality:** `seasonal_decompose(df, value, period=P, time_col=time_col)`.
   Choose P from the data's frequency and the cycle in question: 7 for daily data with a weekly
   cycle, 12 for monthly data with a yearly cycle, 24 for hourly data with a daily cycle. Report
   `seasonal_strength` and `trend_strength` (0 to 1). A warning means the claimed cycle is no
   stronger than in shuffled data, so do not claim it.
6. **Forecast only if asked:** `arima_forecast(df, value, horizon=h, time_col=time_col)`
   (add `seasonal_period=P` when the decomposition showed seasonality). Report the forecast
   with its 95% interval, never a point alone. If `residual_autocorrelation_p` < 0.05, say the
   model missed structure.
7. **Critique:** `critique_findings(method, n=len(series), observational=True)`.

## Common errors to avoid

- Analysing rows in file order without sorting by time.
- Treating stacked series (many stores) as one series.
- Claiming seasonality from one or two cycles, or from a chart alone.
- A forecast without an interval, or far beyond the data (more than about 20% of the series
  length).
- Testing "this year vs last year" with a t-test on autocorrelated daily values.

## Interpretation guidance

Describe in this order:

1. Level and overall trend, in units per period.
2. Seasonal pattern: when it peaks, its amplitude (`seasonal_amplitude`), and its strength.
3. Unusual periods (large remainder values).
4. The forecast, with its interval and how quickly the interval widens.

## Report format

Line chart, then trend sentence, then seasonality sentence with strength, then a forecast table
(period, forecast, lower, upper) if asked, then caveats.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.timeseries import stationarity_test, seasonal_decompose, arima_forecast
from statkit.visualization import recommend_chart, render_chart, explain_chart

df = load_dataset("<path to the uploaded file>").df
spec = recommend_chart(df, "date", "cnt")
render_chart(df, spec, "<outputs folder>/trend.png")
print(explain_chart(df, spec).finding)
print(stationarity_test(df, "cnt", time_col="date").verdict)
dec = seasonal_decompose(df, "cnt", period=7, time_col="date")
print(dec.trend_strength, dec.seasonal_strength, dec.warnings)
fc = arima_forecast(df, "cnt", horizon=14, seasonal_period=7, time_col="date")
print(fc.order, fc.forecast[:3], fc.forecast_lower[:3], fc.forecast_upper[:3])
```
