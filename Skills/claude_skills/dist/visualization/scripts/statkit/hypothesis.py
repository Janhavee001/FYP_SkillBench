"""
Category D, remaining tools. Same convention as inference.py: every test
returns an effect size or interval next to its p-value, because the
interpretation step (interpret_result, critique_findings) needs one.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from .common import ToolResult, is_numeric



def _num(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns or not is_numeric(df[col]):
        return None
    return df[col].dropna()


def _bad_col(col: str) -> list[str]:
    return [f"Column {col!r} not found or not numeric."]


# --------------------------------------------------------------------------- #
# confidence intervals
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CIReport(ToolResult):
    estimate: float = 0.0
    lower: float = 0.0
    upper: float = 0.0
    level: float = 0.95
    n: int = 0
    method: str = "t"


def confidence_interval(df: pd.DataFrame, column: str, level: float = 0.95) -> CIReport:
    """One-sample CI for the mean using the t distribution (cf. StatABench ci_normal)."""
    s = _num(df, column)
    if s is None:
        return CIReport(ok=False, warnings=_bad_col(column))
    if len(s) < 2:
        return CIReport(ok=False, warnings=["Need at least 2 observations."], n=len(s))
    m, se = float(s.mean()), float(stats.sem(s))
    lo, hi = stats.t.interval(level, len(s) - 1, loc=m, scale=se)
    warnings = ["n < 30: interval relies on approximate normality of the data."] if len(s) < 30 else []
    return CIReport(ok=True, warnings=warnings, estimate=m, lower=float(lo), upper=float(hi),
                    level=level, n=len(s))


def two_sample_confidence_interval(df: pd.DataFrame, group_col: str, value_col: str,
                                    group_a, group_b, level: float = 0.95) -> CIReport:
    """Welch CI for mean(a) - mean(b) (cf. StatABench ci_two_normal)."""
    if group_col not in df.columns or (_num(df, value_col) is None):
        return CIReport(ok=False, warnings=[f"Column {group_col!r}/{value_col!r} missing or value not numeric."])
    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    if len(a) < 2 or len(b) < 2:
        return CIReport(ok=False, warnings=["Need at least 2 observations per group."], n=len(a) + len(b))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = float(a.mean() - b.mean())
    lo, hi = stats.t.interval(level, dof, loc=diff, scale=se)
    return CIReport(ok=True, estimate=diff, lower=float(lo), upper=float(hi), level=level,
                    n=len(a) + len(b), method="welch")


# --------------------------------------------------------------------------- #
# distribution / variance / nonparametric
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class NormalityReport(ToolResult):
    column: str = ""
    statistic: float = 0.0
    p_value: float = 0.0
    n: int = 0
    looks_normal: bool = False
    skew: float = 0.0


def ks_normality_test(df: pd.DataFrame, column: str) -> NormalityReport:
    """KS-type normality test with the Lilliefors correction (cf. StatABench ks_test).

    StatABench's ks_test fits the distribution's parameters to the sample and then reads a plain KS
    p-value. Because the parameters came from the same data, that p-value is far too large: the test
    almost never rejects. Lilliefors uses the same KS statistic with the correct null distribution
    for estimated mean/sd. Skew is returned as a second signal."""
    from statsmodels.stats.diagnostic import lilliefors
    s = _num(df, column)
    if s is None or len(s) < 5:
        return NormalityReport(ok=False, column=column, warnings=_bad_col(column) + ["(or fewer than 5 values)"])
    stat, p = lilliefors(s.to_numpy(float), dist="norm", pvalmethod="table")
    warnings = []
    if len(s) > 5000:
        warnings.append("Very large n: KS flags trivial departures from normality; inspect skew too.")
    return NormalityReport(ok=True, warnings=warnings, column=column, statistic=float(stat),
                           p_value=float(p), n=len(s), looks_normal=p >= 0.05, skew=float(s.skew()))


@dataclasses.dataclass
class VarianceTestReport(ToolResult):
    method: str = "mood"
    statistic: float = 0.0
    p_value: float = 0.0
    var_a: float = 0.0
    var_b: float = 0.0
    significant_at_05: bool = False


def variance_equality_test(df: pd.DataFrame, group_col: str, value_col: str,
                            group_a, group_b) -> VarianceTestReport:
    """Mood's two-sample scale test (cf. StatABench mood_variance_test); robust to non-normality, unlike F."""
    if group_col not in df.columns or _num(df, value_col) is None:
        return VarianceTestReport(ok=False, warnings=[f"Column {group_col!r}/{value_col!r} missing or value not numeric."])
    a = df.loc[df[group_col] == group_a, value_col].dropna()
    b = df.loc[df[group_col] == group_b, value_col].dropna()
    if len(a) < 3 or len(b) < 3:
        return VarianceTestReport(ok=False, warnings=["Need at least 3 observations per group."])
    z, p = stats.mood(a, b)
    return VarianceTestReport(ok=True, statistic=float(z), p_value=float(p), var_a=float(a.var(ddof=1)),
                              var_b=float(b.var(ddof=1)), significant_at_05=p < 0.05)


@dataclasses.dataclass
class NonparametricReport(ToolResult):
    method: str = ""
    statistic: float = 0.0
    p_value: float = 0.0
    effect_size: float | None = None
    effect_size_type: str = ""
    groups: dict = dataclasses.field(default_factory=dict)
    significant_at_05: bool = False


def nonparametric_test(df: pd.DataFrame, group_col: str, value_col: str,
                        groups: list | None = None, pair_col: str | None = None) -> NonparametricReport:
    """Mann-Whitney U (2 groups), Wilcoxon signed-rank (2 groups, paired by `pair_col`) or
    Kruskal-Wallis (3+), chosen from the group count (cf. StatABench nonparametric_test). Effect size is
    rank-biserial r (two groups) or eta-squared-H (Kruskal), so interpret_result can bucket it.

    Pairing is by an explicit subject id, never by row position: StatABench's wilcoxon branch truncates
    both columns to the shorter length and pairs by position, which silently mispairs subjects as soon
    as one side has a missing value."""
    if group_col not in df.columns or _num(df, value_col) is None:
        return NonparametricReport(ok=False, warnings=[f"Column {group_col!r}/{value_col!r} missing or value not numeric."])
    levels = groups or sorted(df[group_col].dropna().unique().tolist(), key=str)
    samples = {str(g): df.loc[df[group_col] == g, value_col].dropna().to_numpy() for g in levels}
    if len(samples) < 2 or any(len(v) < 2 for v in samples.values()):
        return NonparametricReport(ok=False, warnings=["Need at least 2 groups with 2+ observations each."])
    arrs = list(samples.values())
    sizes = {k: len(v) for k, v in samples.items()}
    if pair_col is not None:
        if len(arrs) != 2:
            return NonparametricReport(ok=False, warnings=["A paired (Wilcoxon) test compares exactly two groups."])
        if pair_col not in df.columns:
            return NonparametricReport(ok=False, warnings=[f"Pair column {pair_col!r} not found."])
        sub = df[df[group_col].isin(levels[:2])]
        if sub.duplicated([pair_col, group_col]).any():
            return NonparametricReport(ok=False, warnings=[f"{pair_col!r} repeats within a group; pairs are ambiguous."])
        wide = sub.pivot(index=pair_col, columns=group_col, values=value_col)
        complete = wide.dropna()
        warnings = []
        if len(complete) < len(wide):
            warnings.append(f"{len(wide) - len(complete)} subject(s) missing one side of the pair were dropped.")
        if len(complete) < 6:
            return NonparametricReport(ok=False, warnings=warnings + [f"Only {len(complete)} complete pair(s)."])
        x, y = complete[levels[0]].to_numpy(float), complete[levels[1]].to_numpy(float)
        stat, p = stats.wilcoxon(x, y)
        n_pairs = int(np.count_nonzero(x - y))
        # scipy returns min(W+, W-) for a two-sided test, so this is the magnitude of the rank-biserial r.
        r = float(1 - 2 * stat / (n_pairs * (n_pairs + 1) / 2)) if n_pairs else 0.0
        return NonparametricReport(ok=True, warnings=warnings, method="wilcoxon", statistic=float(stat),
                                   p_value=float(p), effect_size=r, effect_size_type="rank_biserial",
                                   groups={"pairs": len(complete)}, significant_at_05=p < 0.05)
    if len(arrs) == 2:
        stat, p = stats.mannwhitneyu(arrs[0], arrs[1], alternative="two-sided")
        r = float(2 * stat / (len(arrs[0]) * len(arrs[1])) - 1)
        return NonparametricReport(ok=True, method="mann_whitney", statistic=float(stat), p_value=float(p),
                                   effect_size=r, effect_size_type="rank_biserial", groups=sizes, significant_at_05=p < 0.05)
    stat, p = stats.kruskal(*arrs)
    n = sum(sizes.values())
    eta2_h = max(0.0, float((stat - len(arrs) + 1) / (n - len(arrs))))  # Tomczak & Tomczak (2014)
    return NonparametricReport(
        ok=True, method="kruskal_wallis", statistic=float(stat), p_value=float(p), effect_size=eta2_h,
        effect_size_type="eta_squared_h", groups=sizes, significant_at_05=p < 0.05,
        warnings=["Kruskal-Wallis says *some* group differs, not which; follow with pairwise tests "
                  "and multiple_comparison_correction."])


# --------------------------------------------------------------------------- #
# A/B tooling
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class BootstrapABReport(ToolResult):
    mean_diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 0.0
    n_resamples: int = 0
    significant_at_05: bool = False


def bootstrap_ab_test(df: pd.DataFrame, group_col: str, value_col: str, group_a, group_b,
                       n_resamples: int = 5000, seed: int = 0) -> BootstrapABReport:
    """Percentile-bootstrap CI for mean(a) - mean(b) plus a permutation-style p-value (cf. StatABench bootstrap_abtest).
    Seeded, so the same call returns the same number -- an agent's result must be reproducible."""
    if group_col not in df.columns or _num(df, value_col) is None:
        return BootstrapABReport(ok=False, warnings=[f"Column {group_col!r}/{value_col!r} missing or value not numeric."])
    a = df.loc[df[group_col] == group_a, value_col].dropna().to_numpy()
    b = df.loc[df[group_col] == group_b, value_col].dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return BootstrapABReport(ok=False, warnings=["Need at least 2 observations per group."])
    rng = np.random.default_rng(seed)
    # Vectorised: one (n_resamples x n) index draw per group instead of a Python loop.
    diffs = (a[rng.integers(0, len(a), (n_resamples, len(a)))].mean(axis=1)
             - b[rng.integers(0, len(b), (n_resamples, len(b)))].mean(axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    pooled, obs = np.concatenate([a, b]), a.mean() - b.mean()
    perms = rng.permuted(np.tile(pooled, (n_resamples, 1)), axis=1)
    null = perms[:, :len(a)].mean(axis=1) - perms[:, len(a):].mean(axis=1)
    # (count + 1) / (n + 1): a permutation p-value is never exactly zero.
    p = float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_resamples + 1))
    return BootstrapABReport(ok=True, mean_diff=float(obs), ci_lower=float(lo), ci_upper=float(hi),
                             p_value=p, n_resamples=n_resamples, significant_at_05=p < 0.05)


@dataclasses.dataclass
class PowerReport(ToolResult):
    effect_size: float = 0.0
    alpha: float = 0.05
    power: float = 0.0
    n_per_group: float = 0.0
    solved_for: str = ""


def ab_power_analysis(effect_size: float | None = None, alpha: float = 0.05, power: float | None = 0.8,
                       n_per_group: int | None = None,
                       proportions: tuple[float, float] | None = None) -> PowerReport:
    """Two-sided, two-sample power analysis (cf. StatABench ab_power_analysis, which also uses
    statsmodels). Means: pass Cohen's d as `effect_size`; exact noncentral-t (TTestIndPower).
    Proportions: pass `proportions=(baseline, target)`; Cohen's h with NormalIndPower.
    Give `power` to solve for n per group, or `n_per_group` (with power=None) to solve for power."""
    from statsmodels.stats.power import NormalIndPower, TTestIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    if proportions is not None:
        effect_size = float(proportion_effectsize(proportions[1], proportions[0]))
        engine = NormalIndPower()
    else:
        engine = TTestIndPower()
    if not effect_size:
        return PowerReport(ok=False, warnings=["Give a non-zero effect_size, or two different proportions."])
    es = abs(effect_size)
    if n_per_group is None:
        n = float(np.ceil(engine.solve_power(effect_size=es, alpha=alpha, power=power, alternative="two-sided")))
        return PowerReport(ok=True, effect_size=effect_size, alpha=alpha, power=power,
                           n_per_group=n, solved_for="n_per_group")
    pw = float(engine.power(effect_size=es, nobs1=n_per_group, alpha=alpha, alternative="two-sided"))
    warnings = [f"Power {pw:.2f} is below the conventional 0.8; a null result here is weak evidence."] if pw < 0.8 else []
    return PowerReport(ok=True, warnings=warnings, effect_size=effect_size, alpha=alpha, power=pw,
                       n_per_group=float(n_per_group), solved_for="power")


@dataclasses.dataclass
class MultipleComparisonReport(ToolResult):
    method: str = ""
    alpha: float = 0.05
    p_values: list[float] = dataclasses.field(default_factory=list)
    adjusted: list[float] = dataclasses.field(default_factory=list)
    rejected: list[bool] = dataclasses.field(default_factory=list)
    n_rejected: int = 0
    n_rejected_uncorrected: int = 0


def multiple_comparison_correction(p_values: list[float], method: str = "fdr_bh",
                                    alpha: float = 0.05) -> MultipleComparisonReport:
    """Adjust a family of p-values: fdr_bh (Benjamini-Hochberg), bonferroni, or holm (StatABench
    fdr_control_df / fwer_control_df). Reports how many discoveries the correction removed."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0 or np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        return MultipleComparisonReport(ok=False, warnings=["p_values must be a non-empty list of numbers in [0, 1]."])
    m = p.size
    order = np.argsort(p)
    if method == "bonferroni":
        adj = np.minimum(p * m, 1.0)
    elif method == "holm":
        stepped = np.maximum.accumulate((m - np.arange(m)) * p[order])
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    elif method == "fdr_bh":
        stepped = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(stepped, 1.0)
    else:
        return MultipleComparisonReport(ok=False, warnings=[f"Unknown method {method!r}; use fdr_bh, bonferroni or holm."])
    # <= matches statsmodels' multipletests and StatABench's fdr/fwer_control_df.
    rejected = adj <= alpha
    raw = int((p <= alpha).sum())
    warnings = []
    if raw > int(rejected.sum()):
        warnings.append(f"{raw - int(rejected.sum())} of {raw} nominally significant result(s) did not survive {method}.")
    return MultipleComparisonReport(ok=True, warnings=warnings, method=method, alpha=alpha,
                                    p_values=p.tolist(), adjusted=[float(x) for x in adj],
                                    rejected=[bool(x) for x in rejected], n_rejected=int(rejected.sum()),
                                    n_rejected_uncorrected=raw)
