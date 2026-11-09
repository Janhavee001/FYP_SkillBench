---
name: regression-analysis
description: Model an outcome from several predictors and estimate adjusted effects - linear (OLS) and multivariable regression with VIF, logistic/Poisson/negative-binomial/gamma GLMs (odds and rate ratios), robust Huber regression, lasso/ridge/elastic-net with cross-validated accuracy, PCA, feature selection and feature engineering, Cox proportional-hazards survival regression, and causal-effect estimation from observational data (propensity score matching, difference-in-differences). Use when the user asks what predicts or explains an outcome, the effect of X controlling for Y, odds or hazard ratios, or the effect of a treatment or policy. Uses the bundled statkit Python tools.
---

# Regression and modelling

A regression coefficient is the expected difference in the outcome per unit of a predictor,
holding the other predictors fixed. That is only meaningful if the model family fits the
outcome, the predictors are not redundant, and the reader is told which effects are
associations and which are causal estimates under stated assumptions.

## When to use

- "What predicts quality?", "Effect of age on survival, controlling for class?",
  "Odds ratio for...", "Which features matter?", "Did the policy change the outcome?"

**Not for:** a single pairwise relationship (`correlation-analysis`), comparing two groups
without adjustment (`group-comparison`), forecasting a series (`time-series-analysis`).

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Load data with `statkit.loading.load_dataset`. Tool signatures are in `reference/tools.md`.

## Choosing the model

| Outcome | Tool |
| --- | --- |
| Continuous | `multivariable_regression` (one predictor: `linear_regression`) |
| Continuous, with outliers | `robust_regression` (compare its coefficients with OLS) |
| 0/1 or two-level text | `glm_fit(..., family="binomial")`, gives odds ratios |
| Count | `glm_fit(..., family="poisson")`; if it warns of overdispersion, `"negative_binomial"` |
| Positive, right-skewed (costs, durations) | `glm_fit(..., family="gamma")` |
| Many predictors, prediction is the goal | `regularized_regression` (quote `cv_r_squared`) |
| Time to event with censoring | `cox_regression`, gives hazard ratios |
| Effect of a binary treatment, observational data | `propensity_score_matching` |
| Effect of a change at a point in time, treated vs control groups | `difference_in_differences` |

## Procedure

1. **Gate:** `check_method_appropriateness(df, "multivariable_regression", [y, *xs])`. It
   rejects OLS on a binary outcome and flags ordinal outcomes.
2. **Prepare** only as needed:
    - Categorical predictors are dummy-coded automatically (the warning names the reference
      level).
    - `scale_numeric` when comparing coefficient sizes.
    - `encode_categorical`, `polynomial_expand` and `engineer_categorical_interactions` for
      deliberate feature engineering.
    - `feature_selection(df, target, method="general")` to drop unusable columns.
3. **Fit** with the model from the table.
4. **Diagnostics** from the result:
    - VIF > 10 means the coefficients are unstable (multicollinearity).
    - Residual normality matters only at small n.
    - Dispersion ratio (Poisson).
    - Proportional-hazards p-values (Cox).
    - Balance SMD after matching (propensity score matching).
    - Pre-trend p (difference-in-differences).
5. **Interpret** each key coefficient in units: "each extra 1% alcohol is associated with
   +0.31 quality points (95% CI ...), holding the others fixed". For GLMs report `exp_coef`
   (odds or rate ratio); for Cox, `hr`.
6. **Critique:** `critique_findings("multivariable_regression", n=..., observational=True, ...)`.

## Common errors to avoid

- OLS on a 0/1 outcome. Use logistic.
- Reading coefficients as causal from observational data. Only the causal tools give causal
  estimates, and only under their stated assumptions (no unmeasured confounding; parallel
  trends).
- Quoting in-sample R squared as predictive accuracy.
- Interpreting a dummy coefficient without naming its reference category.
- Dropping a "non-significant" confounder to get a cleaner story.

## Interpretation guidance

Separate three claims: how well the model fits (R squared or pseudo R squared, CV R squared),
which predictors matter (coefficient, CI, p), and what can be concluded causally (usually
nothing, unless a causal tool's assumptions hold). Give each its own sentence.

## Report format

Model and why it fits the outcome, then fit statistics, then a coefficient table (estimate, CI,
p, in units or as ratios), then diagnostics with warnings, then a plain-language takeaway with
an explicit association-or-causation statement.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.validation import check_method_appropriateness
from statkit.modeling import multivariable_regression, glm_fit, regularized_regression
from statkit.interpretation import critique_findings

df = load_dataset("<path to the uploaded file>").df
xs = ["alcohol", "volatile_acidity", "sulphates"]
print(check_method_appropriateness(df, "multivariable_regression", ["quality", *xs]).reason)
ols = multivariable_regression(df, "quality", xs)
print(ols.r_squared, ols.vif, ols.coefficients["alcohol"])
lasso = regularized_regression(df, "quality", [c for c in df.columns if c != "quality"], "lasso")
print(lasso.cv_r_squared, lasso.dropped)
good = df.assign(good=(df["quality"] >= 7).astype(int))
logit = glm_fit(good, "good", xs, family="binomial")
print({k: round(v["exp_coef"], 2) for k, v in logit.coefficients.items()})
print(critique_findings("multivariable_regression", n=ols.n, observational=True).violations)
```
