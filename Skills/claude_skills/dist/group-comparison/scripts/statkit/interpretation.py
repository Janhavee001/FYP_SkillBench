"""
Category J: the other gap identified in DOCUMENTATION.md.

§6.2's finding, stated plainly: all three benchmarked systems produce
narrative text, and the judge -- applying StatABench's own anchors for
"interpretation of results and context" -- found that almost none of it
interprets the result. LIDA describes its colour palette and axis labels.
Data Formulator 2 returns a one-line encoding recommendation. nvAgent
returns reasoning about the schema. None of them says what the data means.

interpret_result is scoped narrowly on purpose: it is not a general
narrative generator (that would just reproduce the failure, since an LLM
asked to "explain the chart" tends to describe the chart). It is a pure
function from (method, statistic, effect size) to a small set of structured
claims about *meaning* -- effect-size bucket, statistical vs. practical
significance, and the caveats a result of this shape usually carries. An
agent narrates from this structure; it does not ask the LLM to interpret
from scratch.
"""
from __future__ import annotations

import dataclasses

from .common import ToolResult


@dataclasses.dataclass
class Interpretation(ToolResult):
    method: str = ""
    statistically_significant: bool | None = None
    effect_size_label: str = ""
    practically_significant: bool | None = None
    headline: str = ""
    caveats: list[str] = dataclasses.field(default_factory=list)


# Cohen's conventional buckets, cited explicitly so the thresholds are
# auditable rather than an LLM's ad hoc judgement call.
_COHENS_D_BUCKETS = [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")]
_CRAMERS_V_BUCKETS = [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large")]
# Cohen (1988): r .1/.3/.5 (also used for rank-biserial r); eta-squared .01/.06/.14.
_R_BUCKETS = [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large")]
_ETA_SQ_BUCKETS = [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large")]
_BUCKETS = {
    "cohens_d": _COHENS_D_BUCKETS,
    "cramers_v": _CRAMERS_V_BUCKETS,
    "r": _R_BUCKETS, "pearson_r": _R_BUCKETS, "rank_biserial": _R_BUCKETS,
    "eta_squared": _ETA_SQ_BUCKETS, "eta_squared_h": _ETA_SQ_BUCKETS,
}


def _bucket(value: float, table: list[tuple[float, str]]) -> str:
    v = abs(value)
    for threshold, label in table:
        if v < threshold:
            return label
    return table[-1][1]


def interpret_result(method: str, p_value: float | None = None,
                      effect_size: float | None = None,
                      effect_size_type: str = "cohens_d",
                      n: int | None = None,
                      question: str = "") -> Interpretation:
    """
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
    """
    warnings: list[str] = []
    sig = None if p_value is None else p_value < 0.05

    effect_label = ""
    practical = None
    if effect_size is not None:
        table = _BUCKETS.get(effect_size_type)
        if table is None:
            # Bucketing an unknown effect size against Cohen's d would mislabel it
            # (an r of 0.476 is "medium", but "small" on the d scale), so refuse instead.
            warnings.append(f"Unknown effect_size_type {effect_size_type!r}; effect size not bucketed. "
                            f"Known: {sorted(_BUCKETS)}.")
        else:
            effect_label = _bucket(effect_size, table)
            practical = effect_label not in ("negligible", "small")

    caveats = []
    if n is not None and n < 30:
        caveats.append(
            f"n={n} is small; p-values and effect-size estimates are less "
            "stable and confidence intervals should be reported alongside "
            "any point estimate.")
    if sig is True and practical is False:
        caveats.append(
            "Statistically significant but the effect size is small/negligible -- "
            "with a large enough sample, trivial differences become "
            "statistically detectable without being practically meaningful.")
    if sig is False and effect_size is not None and practical:
        caveats.append(
            "Not statistically significant despite a non-trivial effect size -- "
            "this may reflect low statistical power rather than a genuinely "
            "null effect; consider whether n is adequate before concluding "
            "'no effect'.")

    if sig is None:
        headline = "No p-value supplied; only effect size is reported."
    elif sig and practical:
        headline = f"A statistically significant, {effect_label} effect."
    elif sig and practical is False:
        headline = f"Statistically significant but the effect is {effect_label}."
    elif sig and practical is None:
        headline = "Statistically significant (no effect size supplied)."
    else:
        headline = "No statistically significant effect detected."

    return Interpretation(
        ok=True, warnings=warnings, method=method,
        statistically_significant=sig, effect_size_label=effect_label,
        practically_significant=practical, headline=headline, caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# critique_findings
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Critique(ToolResult):
    method: str = ""
    violations: list[dict] = dataclasses.field(default_factory=list)  # {"severity", "issue"}
    n_high: int = 0


def critique_findings(method: str, n: int | None = None, group_sizes: list[int] | None = None,
                       normality_p: float | None = None, variance_equality_p: float | None = None,
                       p_value: float | None = None, n_tests: int = 1, observational: bool = True,
                       missing_pct: float | None = None, expected_min_cell: float | None = None) -> Critique:
    """
    Check a result against the assumptions its method relies on and report
    violations, ranked by severity. StatABench's judger-open/result_bias.py is
    an LLM-judge prompt that scores, after the fact, whether a write-up
    identified and corrected biases; this tool asks the checkable subset of
    those questions (sample size, assumptions, multiplicity, confounding,
    missingness) as deterministic rules, *before* the claim is written.
    """
    v: list[dict] = []
    add = lambda sev, msg: v.append({"severity": sev, "issue": msg})  # noqa: E731
    m = method.lower()

    if n is not None and n < 30:
        add("high" if n < 10 else "medium", f"Sample size n={n} is small; estimates and p-values are unstable.")
    if group_sizes:
        if min(group_sizes) < 5:
            add("high", f"Smallest group has {min(group_sizes)} observation(s); group-level conclusions are unreliable.")
        if max(group_sizes) / max(min(group_sizes), 1) > 3:
            add("low", f"Group sizes are unbalanced ({group_sizes}); test power is limited by the smallest group.")
    if m in ("t_test", "welch_t_test", "ab_test", "linear_regression", "multivariable_regression", "anova"):
        if normality_p is not None and normality_p < 0.05 and (n is None or n < 100):
            add("medium", f"Normality is rejected (p={normality_p:.3g}) at modest n; consider a nonparametric alternative.")
        if variance_equality_p is not None and variance_equality_p < 0.05 and m in ("anova", "t_test"):
            add("medium", "Variances differ across groups; use Welch's correction or a robust test.")
    if m in ("chi_square", "chi_square_association") and expected_min_cell is not None and expected_min_cell < 5:
        add("high", f"Smallest expected cell count is {expected_min_cell:.1f} (< 5); chi-square approximation is invalid, use Fisher's exact.")
    if n_tests > 1 and p_value is not None:
        bonferroni = min(1.0, p_value * n_tests)
        if p_value < 0.05 <= bonferroni:
            add("high", f"p={p_value:.3g} is one of {n_tests} tests; it does not survive Bonferroni correction (adjusted p={bonferroni:.3g}).")
        else:
            add("low", f"This is one of {n_tests} tests; report all of them, not only the significant ones.")
    if observational:
        add("medium", "Observational data: association is not causation; unmeasured confounders could explain the result.")
    if missing_pct is not None and missing_pct > 5:
        add("high" if missing_pct > 30 else "medium", f"{missing_pct:.0f}% of values are missing; check whether missingness is related to the outcome.")
    if p_value is not None and 0.04 <= p_value < 0.05:
        add("low", f"p={p_value:.3g} is borderline; small analytic choices could flip significance.")

    order = {"high": 0, "medium": 1, "low": 2}
    v.sort(key=lambda d: order[d["severity"]])
    return Critique(ok=not any(d["severity"] == "high" for d in v), method=method, violations=v,
                    n_high=sum(d["severity"] == "high" for d in v),
                    warnings=[d["issue"] for d in v if d["severity"] == "high"])
