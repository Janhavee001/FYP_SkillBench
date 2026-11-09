---
name: group-comparison
description: Compare groups and test hypotheses - two-group and multi-group comparisons of means or distributions (Welch t-test, Mann-Whitney, Wilcoxon, Kruskal-Wallis), associations between categorical variables (chi-square, Cramer's V), proportions, confidence intervals, bootstrap and Bayesian comparisons, power and sample size, multiple-comparison correction, and survival-curve comparisons (Kaplan-Meier, log-rank). Use when the user asks whether groups differ, whether a difference is significant, "does X depend on Y", A/B tests, or how many samples are needed. Uses the bundled statkit Python tools.
---

# Group comparison and hypothesis testing

Choose the test from the data, not from habit. Then report how big the difference is, not
just whether p < 0.05. Every test here returns an effect size or interval next to its p-value;
use it.

## When to use

- "Do survivors and non-survivors differ in fare?", "Is survival related to sex?",
  "Did the new version perform better?", "How many users do we need?",
  "Do the groups survive for different lengths of time?"

**Not for:** relationships between two numeric variables (`correlation-analysis`), effects
adjusted for other variables (`regression-analysis`), causal effects from observational data
(`regression-analysis` covers matching and difference-in-differences).

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Load data with `statkit.loading.load_dataset`. Tool signatures are in `reference/tools.md`.

## Choosing the test

| Outcome | Groups | Test (tool) |
| --- | --- | --- |
| Numeric, roughly normal or n > 30 per group | 2 independent | Welch t-test: `ab_test` (+ `two_sample_confidence_interval`) |
| Numeric, skewed or small n | 2 independent | Mann-Whitney: `nonparametric_test` |
| Numeric, same subjects measured twice | 2 paired | Wilcoxon: `nonparametric_test(..., pair_col=id)` |
| Numeric | 3 or more | Kruskal-Wallis: `nonparametric_test`, then pairwise tests + correction |
| Categorical | any | Chi-square: `chi_square_association` |
| 0/1 outcome, want P(A > B) | 2 | `bayesian_estimate(df, "proportion", col, group_col=g)` |
| Numeric, no distributional assumption | 2 | `bootstrap_ab_test` |
| Time until an event, with censoring | 2 | `kaplan_meier_curve` per group, then `logrank_test` |
| Planning: sample size or power | 2 | `ab_power_analysis` |

## Procedure

1. **Gate the plan:** `check_method_appropriateness(df, method, [group_col, value_col])`
   (e.g. method `"ab_test"`, `"nonparametric_test"`, `"chi_square_association"`). If the verdict
   is not "appropriate", follow its reason. If it is "unchecked", say the plan was not
   validated.
2. **Check assumptions** for t-tests: group sizes, skew (`describe_column`) and
   `ks_normality_test` per group with small n. Unequal spread is handled by Welch; check it
   anyway with `variance_equality_test` if the question is about variability.
3. **Run the test** from the table above.
4. **Many comparisons?** Collect all the p-values and run
   `multiple_comparison_correction(pvals, "holm")` (or `"fdr_bh"` when exploring).
5. **Interpret:** `interpret_result(method, p_value=p, effect_size=es, effect_size_type=..., n=n)`.
   Use `"cohens_d"` for `ab_test`, `"cramers_v"` for chi-square, and whatever
   `effect_size_type` the nonparametric result reports.
6. **Critique:** `critique_findings(method, n=..., group_sizes=[...], p_value=p, n_tests=k,
   observational=True, expected_min_cell=chi.expected_min_cell)`. Report every "high" item.
7. **Non-significant result?** Run `ab_power_analysis(effect_size=d, n_per_group=n, power=None)`.
   Low power means "inconclusive", not "no difference".

## Common errors to avoid

- A t-test on 3+ groups, or on a categorical outcome. The gate catches both.
- "p < 0.05 so the effect is important". Check the effect size bucket.
- "p > 0.05 so the groups are the same". Check power.
- Pairing by row order instead of by subject id.
- Testing 20 subgroups and reporting the one that was significant.
- Causal wording for observational groups ("being female caused survival").

## Interpretation guidance

Lead with the difference in the outcome's units and its interval ("survivors paid £26 more on
average, 95% CI £18 to £34"), then the effect size bucket, then p. Name the comparison
direction explicitly (group A minus group B).

## Report format

Question, then the test and why it was chosen, then the result (difference, CI, effect size,
p, n per group), then caveats from the critique, then a one-sentence plain-language answer.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.validation import check_method_appropriateness
from statkit.inference import ab_test, chi_square_association
from statkit.hypothesis import two_sample_confidence_interval, nonparametric_test
from statkit.interpretation import interpret_result, critique_findings

df = load_dataset("<path to the uploaded file>").df
print(check_method_appropriateness(df, "ab_test", ["survived", "fare"]).verdict)
t = ab_test(df, "survived", "fare", 1, 0)
ci = two_sample_confidence_interval(df, "survived", "fare", 1, 0)
mw = nonparametric_test(df, "survived", "fare")  # fare is skewed: confirm with a rank test
print(t.mean_a - t.mean_b, (ci.lower, ci.upper), t.cohens_d, t.p_value, mw.p_value)
print(interpret_result("ab_test", p_value=t.p_value, effect_size=t.cohens_d, n=t.n_a + t.n_b).headline)

chi = chi_square_association(df, "sex", "survived")
print(chi.cramers_v, chi.p_value, chi.warnings)
print(critique_findings("chi_square_association", n=chi.n, p_value=chi.p_value,
                        expected_min_cell=chi.expected_min_cell).violations)
```
