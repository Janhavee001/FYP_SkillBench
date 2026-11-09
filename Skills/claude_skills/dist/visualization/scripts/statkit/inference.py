"""
Category D: hypothesis testing. Two tools chosen because they are the two
methods the shared test set actually exercises most (chi-square for
association questions, Welch t-test for group-comparison questions), per
DOCUMENTATION.md §6.1 -- the plain baseline already picks these correctly
98% of the time, so the point of implementing them here is not to beat that
selection rate, it's to make them composable with the Category A validators
(a caller is expected to run check_method_appropriateness before trusting
the result, not after).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from .common import ToolResult


@dataclasses.dataclass
class ChiSquareReport(ToolResult):
    col_a: str = ""
    col_b: str = ""
    chi2: float = 0.0
    p_value: float = 0.0
    dof: int = 0
    cramers_v: float = 0.0
    contingency_table: dict = dataclasses.field(default_factory=dict)
    significant_at_05: bool = False
    n: int = 0
    expected_min_cell: float | None = None  # pass straight to critique_findings(expected_min_cell=...)


def chi_square_association(df: pd.DataFrame, col_a: str, col_b: str) -> ChiSquareReport:
    """
    Chi-square test of independence between two categorical columns, plus
    Cramer's V so the result carries an effect size alongside the p-value --
    a p-value alone is exactly the kind of output DOCUMENTATION.md §6.2
    flags as uninterpreted: a number with no sense of how large the effect
    actually is.

    Generalizes StatABench's `contingency_test`.
    """
    for c in (col_a, col_b):
        if c not in df.columns:
            return ChiSquareReport(ok=False, warnings=[f"Column {c!r} not found."],
                                    col_a=col_a, col_b=col_b)

    table = pd.crosstab(df[col_a], df[col_b])
    if min(table.shape) < 2:
        return ChiSquareReport(ok=False, col_a=col_a, col_b=col_b,
                                warnings=["Each variable needs at least two observed categories."])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    # Cramer's V from the uncorrected statistic: scipy applies Yates' correction to
    # 2x2 tables by default, which would shrink V and make it incomparable across table sizes.
    chi2_raw = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2_raw / (n * min_dim)))

    warnings = []
    min_expected = float(expected.min())
    if min_expected < 5:
        # The rule of thumb is about *expected* counts; a sparse observed cell is not by itself a problem.
        warnings.append(
            f"Smallest expected cell count is {min_expected:.1f} (< 5); the chi-square "
            "approximation may be unreliable. Consider Fisher's exact test.")

    return ChiSquareReport(
        ok=True, warnings=warnings, col_a=col_a, col_b=col_b,
        chi2=float(chi2), p_value=float(p), dof=int(dof), cramers_v=cramers_v,
        contingency_table={str(k): {str(i): int(v) for i, v in col.items()}
                           for k, col in table.to_dict().items()},
        significant_at_05=p < 0.05, n=int(n), expected_min_cell=min_expected,
    )


@dataclasses.dataclass
class ABTestReport(ToolResult):
    group_col: str = ""
    value_col: str = ""
    group_a: str = ""
    group_b: str = ""
    mean_a: float = 0.0
    mean_b: float = 0.0
    n_a: int = 0
    n_b: int = 0
    t_statistic: float = 0.0
    p_value: float = 0.0
    cohens_d: float = 0.0
    significant_at_05: bool = False


def ab_test(df: pd.DataFrame, group_col: str, value_col: str,
            group_a, group_b, alternative: str = "two-sided") -> ABTestReport:
    """
    Welch's two-sample t-test (unequal variances assumed by default, since
    that assumption is the safer default and StatABench's own `ab_ttest`
    tool makes the same choice), plus Cohen's d.

    Reporting Cohen's d alongside the p-value is deliberate: DOCUMENTATION.md
    §6.2 found that none of the three benchmarked systems distinguished
    statistical from practical significance in their narrative output, and a
    downstream `interpret_result` call needs an effect-size number to make
    that distinction rather than inventing one from the p-value alone.
    """
    for c in (group_col, value_col):
        if c not in df.columns:
            return ABTestReport(ok=False, warnings=[f"Column {c!r} not found."],
                                 group_col=group_col, value_col=value_col)

    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    warnings = []
    if len(a) < 2 or len(b) < 2:
        return ABTestReport(ok=False,
                             warnings=[f"Need at least 2 observations per group; "
                                       f"got {len(a)} for {group_a!r}, {len(b)} for {group_b!r}."],
                             group_col=group_col, value_col=value_col,
                             group_a=str(group_a), group_b=str(group_b))

    t_stat, p = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                          / (len(a) + len(b) - 2))
    cohens_d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return ABTestReport(
        ok=True, warnings=warnings, group_col=group_col, value_col=value_col,
        group_a=str(group_a), group_b=str(group_b),
        mean_a=float(a.mean()), mean_b=float(b.mean()), n_a=len(a), n_b=len(b),
        t_statistic=float(t_stat), p_value=float(p), cohens_d=cohens_d,
        significant_at_05=p < 0.05,
    )
