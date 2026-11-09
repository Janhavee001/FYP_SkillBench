---
name: correlation-analysis
description: Measure and explain relationships between variables - Pearson, Spearman or Kendall correlation, the strongest pairs in a dataset, correlation heatmaps and scatter plots, with checks for binary or categorical columns and for many-comparison false positives, and wording that avoids causal claims. Use when the user asks which variables are related, how strongly, "what correlates with X", or "is X associated with Y" for numeric variables. Uses the bundled statkit Python tools.
---

# Correlation analysis

A correlation says how closely two variables move together. It does not say why. The two
errors this skill exists to prevent are using Pearson's r where it does not apply, and reading
an association as a cause.

## When to use

- "Which variables are related to quality?", "How strongly are X and Y related?",
  "Show me the correlations."

**Not for:** two categorical variables (use `chi_square_association` in `group-comparison`),
predicting an outcome or adjusting for other variables (`regression-analysis`), or comparing
group means (`group-comparison`).

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

1. **Check each pair is suitable:** `check_method_appropriateness(df, "correlation", [a, b])`.
   A binary or text column gives "questionable" or "inappropriate". Follow its reason.
2. **Choose the coefficient:**
    - Pearson for roughly linear, roughly symmetric numeric data.
    - Spearman when either variable is skewed, ordinal (e.g. a 1 to 10 rating) or has outliers.
    - Kendall for small samples with many ties.
3. **Compute:** `correlation_matrix(df, columns, method=...)` returns the matrix and the
   strongest pair. For "what correlates most with Y", read the Y row.
4. **Significance and multiplicity:** for a single pair, `linear_regression(df, x, y)` gives the
   slope's p-value and CI. When scanning many pairs, collect the p-values and run
   `multiple_comparison_correction(pvals, "fdr_bh")`. Report how many survive.
5. **Size:** `interpret_result("correlation", p_value=p, effect_size=r, effect_size_type="r", n=n)`.
   It buckets r on Cohen's r scale (0.1 small, 0.3 medium, 0.5 large).
6. **Plot:** a scatter for one pair, `recommend_chart(df, x, y)`. A heatmap for many variables,
   `recommend_chart(df, columns=[...])`. Then `render_chart` and `explain_chart`.
7. **Critique before reporting:** `critique_findings("correlation", n=n, p_value=p,
   n_tests=k, observational=True)`.

## Assumptions to check

- Pearson measures linear association only. A strong curved relationship can give r near 0,
  so look at the scatter.
- Each row is an independent unit. Repeated rows for one entity inflate apparent strength.

## Common errors to avoid

- Causal language: "alcohol increases quality". Say "higher-alcohol wines tend to be rated
  higher".
- Treating r = 0.3 as "strong". It is medium, and explains about 9% of the variance (r squared).
- Reporting only the significant pairs out of 66 tested.
- Correlating a 0/1 column and calling it Pearson without saying it is a point-biserial r.

## Interpretation guidance

State direction, strength (bucket and r), n, and r squared as "shared variance". Then add one
sentence on what could explain it besides causation (a third variable, selection, reverse
direction).

## Report format

Headline relationship with r, CI or p and n, then the ranked list or heatmap, then the scatter
for the key pair, then caveats.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.descriptive import correlation_matrix
from statkit.validation import check_method_appropriateness
from statkit.modeling import linear_regression
from statkit.interpretation import interpret_result, critique_findings
from statkit.visualization import recommend_chart, render_chart, explain_chart

df = load_dataset("<path to the uploaded file>").df
print(check_method_appropriateness(df, "correlation", ["alcohol", "quality"]).verdict)
cm = correlation_matrix(df, method="spearman")
print(sorted(((c, r) for c, r in cm.matrix["quality"].items() if c != "quality"),
             key=lambda kv: -abs(kv[1] or 0))[:3])
fit = linear_regression(df, "alcohol", "quality")
r = cm.matrix["quality"]["alcohol"]
print(interpret_result("correlation", p_value=fit.coefficients["alcohol"]["p"], effect_size=r,
                       effect_size_type="r", n=fit.n).headline)
print(critique_findings("correlation", n=fit.n, n_tests=11).violations)
spec = recommend_chart(df, columns=["alcohol", "density", "volatile_acidity", "quality"])
render_chart(df, spec, "<outputs folder>/corr_heatmap.png")
print(explain_chart(df, spec).finding)
```
