---
name: distribution-analysis
description: Describe the shape of one variable's distribution - centre, spread, skewness, normality, outliers, and an interval estimate for its mean or proportion - and pick the right plot (histogram, box plot, bar chart). Use when the user asks how a variable is distributed, whether it is normal, what a typical value is, how spread out it is, or which values are unusual; and before choosing a test that assumes normality. Uses the bundled statkit Python tools.
---

# Distribution analysis

A distribution answers three questions: where is the centre, how spread out is it, and what
shape is it? The shape decides which summary is honest (mean or median) and which later tests
are valid (t-test or rank-based).

## When to use

- "How is X distributed?", "What is a typical X?", "Is X normally distributed?", "Are there
  unusual values?"
- Before `group-comparison` or `regression-analysis`, to check the normality assumption.

**Not for:** comparing groups (`group-comparison`), relationships (`correlation-analysis`), or
change over time (`time-series-analysis`).

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Load the data with `statkit.loading.load_dataset` (see the `data-profiling` skill) rather than
`pd.read_csv` with a guessed separator. Tool signatures are in `reference/tools.md`.

## Procedure

1. **Summarise:** `describe_column(df, col)`. Numeric: mean, median, sd, min, max, skew,
   kurtosis. Categorical: mode and top counts (then use `frequency_table`).
2. **Shape:** read the skew. |skew| < 0.5 is roughly symmetric, 0.5 to 1 is moderately skewed,
   > 1 is strongly skewed.
3. **Normality, only if a later method needs it:** `ks_normality_test(df, col)`
   (Lilliefors-corrected). With n in the thousands it flags trivial departures, so weigh it
   together with skew and the histogram, never alone.
4. **Outliers:** `detect_outliers_iqr(df, col)`. If more than 10% of values are flagged, the
   distribution is skewed, not contaminated. Say that rather than calling them errors.
5. **Interval for the centre:** `confidence_interval(df, col)` for a mean. For a 0/1 column use
   `bayesian_estimate(df, "proportion", col)`, which gives a credible interval for the
   proportion that stays inside [0, 1].
6. **Plot:** `recommend_chart(df, col)` (histogram for numeric, bar for categorical), then
   `render_chart(df, spec, path)` and `explain_chart(df, spec)`. A box plot by group comes from
   `recommend_chart(df, group_col, col)`.
7. **Grouped distributions:** `bin_continuous` to make bands (e.g. age groups), then
   `frequency_table` on the new column.

## Assumptions to check

- Enough data: under ~30 observations, skew and normality estimates are unstable. Say so.
- The column really is a measurement (not an ID, not a code).

## Common errors to avoid

- Reporting the mean of a strongly skewed variable (income, fares, counts) as "typical".
- Saying "the data are normal" because a normality test was not significant on a small sample.
  Absence of evidence is not evidence of normality.
- Deleting outliers without being asked.

## Interpretation guidance

- Lead with the typical value and spread in the variable's units: "Most fares were between
  £7.9 and £31 (IQR); the median was £14.5."
- Describe the shape in words ("long right tail: a few very expensive tickets") before numbers.
- An interval is about the mean, not about individual values.

## Report format

Typical value and spread (in units), then shape, then unusual values, then the chart, then
what it implies for later analysis (e.g. "use a rank-based test for fare").

## Worked example

```python
from statkit.loading import load_dataset
from statkit.descriptive import describe_column
from statkit.hypothesis import ks_normality_test, confidence_interval
from statkit.cleaning import detect_outliers_iqr
from statkit.visualization import recommend_chart, render_chart, explain_chart

df = load_dataset("<path to the uploaded file>").df
d = describe_column(df, "fare")
print(d.stats["median"], d.stats["skew"])
print(ks_normality_test(df, "fare").p_value)
print(detect_outliers_iqr(df, "fare").pct_outliers)
print(confidence_interval(df, "fare"))
spec = recommend_chart(df, "fare")
render_chart(df, spec, "<outputs folder>/fare_hist.png")
print(explain_chart(df, spec).finding)
```
