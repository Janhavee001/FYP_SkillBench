"""
Category H (causal): two estimators of a treatment effect from observational data.

Both return the naive difference next to the adjusted estimate, because the gap
between them is what tells a reader how much confounding the adjustment removed,
and both state the assumption the estimate rests on. Neither can make an
observational study causal on its own: the warnings say what has to be true.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .common import ToolResult, is_numeric


@dataclasses.dataclass
class PSMReport(ToolResult):
    att: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    naive_difference: float = 0.0
    n_treated: int = 0
    n_matched: int = 0
    n_unmatched: int = 0
    max_abs_smd_before: float = 0.0
    max_abs_smd_after: float = 0.0
    balance: dict = dataclasses.field(default_factory=dict)  # covariate -> {smd_before, smd_after}


def _smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    pooled = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
    return float((x_t.mean() - x_c.mean()) / pooled) if pooled > 0 else 0.0


def propensity_score_matching(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str],
                               caliper: float = 0.2, n_bootstrap: int = 500, seed: int = 0) -> PSMReport:
    """
    Average treatment effect on the treated by 1:1 nearest-neighbour matching on the logit of the
    propensity score, with replacement (cf. StatABench estimate_ATT_with_psm).

    The caliper is in standard deviations of the logit propensity score (0.2 is Austin's 2011
    recommendation), not in raw probability units. Covariate balance is reported as standardized
    mean differences before and after matching; |SMD| > 0.1 after matching means the groups are
    still not comparable on that covariate. The CI is a bootstrap over matched pairs, which ignores
    uncertainty in the propensity model and so is somewhat too narrow.
    """
    from sklearn.linear_model import LogisticRegression

    missing = [c for c in [treatment, outcome] + covariates if c not in df.columns]
    if missing:
        return PSMReport(ok=False, warnings=[f"Column(s) not found: {missing}"])
    sub = df[[treatment, outcome] + covariates].dropna()
    if not set(sub[treatment].unique()) <= {0, 1}:
        return PSMReport(ok=False, warnings=[f"{treatment!r} must be 0/1 (1 = treated)."])
    if not is_numeric(sub[outcome]):
        return PSMReport(ok=False, warnings=[f"Outcome {outcome!r} must be numeric."])
    X = pd.get_dummies(sub[covariates], drop_first=True, dtype=float).to_numpy(float)
    names = list(pd.get_dummies(sub[covariates], drop_first=True, dtype=float).columns)
    t = sub[treatment].to_numpy(int)
    y = sub[outcome].to_numpy(float)
    if t.sum() < 5 or (1 - t).sum() < 5:
        return PSMReport(ok=False, warnings=["Need at least 5 treated and 5 control units."])
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
    ps = LogisticRegression(C=1e6, max_iter=5000).fit(Xs, t).predict_proba(Xs)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    logit = np.log(ps / (1 - ps))
    width = caliper * logit.std()
    treated, control = np.where(t == 1)[0], np.where(t == 0)[0]
    dist = np.abs(logit[treated][:, None] - logit[control][None, :])
    nearest = dist.argmin(axis=1)
    ok = dist[np.arange(len(treated)), nearest] <= width
    tm, cm = treated[ok], control[nearest[ok]]
    if len(tm) < 5:
        return PSMReport(ok=False, warnings=[f"Only {len(tm)} treated unit(s) found a match within the caliper; "
                                             "the groups barely overlap."])
    diffs = y[tm] - y[cm]
    att = float(diffs.mean())
    rng = np.random.default_rng(seed)
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_bootstrap)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    balance = {nm: {"smd_before": _smd(X[treated, j], X[control, j]), "smd_after": _smd(X[tm, j], X[cm, j])}
               for j, nm in enumerate(names)}
    after = max(abs(b["smd_after"]) for b in balance.values())
    warnings = ["ATT assumes no unmeasured confounders: every variable that drives both treatment and "
                "outcome must be in `covariates`."]
    if after > 0.1:
        warnings.append(f"Balance not achieved (max |SMD| after matching = {after:.2f} > 0.1); the estimate is still confounded.")
    if (~ok).sum():
        warnings.append(f"{int((~ok).sum())} treated unit(s) had no control within the caliper and were dropped; "
                        "the ATT describes only the matched treated units.")
    return PSMReport(ok=True, warnings=warnings, att=att, ci_lower=float(lo), ci_upper=float(hi),
                     naive_difference=float(y[t == 1].mean() - y[t == 0].mean()), n_treated=int(t.sum()),
                     n_matched=int(len(tm)), n_unmatched=int((~ok).sum()),
                     max_abs_smd_before=max(abs(b["smd_before"]) for b in balance.values()),
                     max_abs_smd_after=after, balance=balance)


@dataclasses.dataclass
class DiDReport(ToolResult):
    effect: float = 0.0
    se: float = 0.0
    p_value: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    cell_means: dict = dataclasses.field(default_factory=dict)  # "treated_pre" etc. -> mean
    pre_trend_p: float | None = None
    n: int = 0


def difference_in_differences(df: pd.DataFrame, outcome: str, group_col: str, post_col: str,
                               time_col: str | None = None, cluster_col: str | None = None) -> DiDReport:
    """
    Two-group, two-period difference-in-differences: OLS of outcome on group, post and group x post;
    the interaction is the effect (cf. StatABench estimate_did_effect). `group_col` and `post_col`
    are 0/1. Standard errors are clustered on `cluster_col` when given (units observed repeatedly),
    else heteroskedasticity-robust (HC1).

    The estimate rests on parallel trends. When `time_col` has 2+ pre-treatment periods, the tool
    tests whether the groups' trends already differed before treatment (group x time on the
    pre-period only); a small p-value there undermines the whole estimate.
    """
    import statsmodels.api as sm

    cols = [outcome, group_col, post_col] + [c for c in (time_col, cluster_col) if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return DiDReport(ok=False, warnings=[f"Column(s) not found: {missing}"])
    sub = df[cols].dropna()
    for c in (group_col, post_col):
        if not set(sub[c].unique()) <= {0, 1}:
            return DiDReport(ok=False, warnings=[f"{c!r} must be 0/1."])
    g, p = sub[group_col].to_numpy(float), sub[post_col].to_numpy(float)
    cells = {f"{'treated' if gi else 'control'}_{'post' if pi else 'pre'}": float(sub[outcome][(g == gi) & (p == pi)].mean())
             for gi in (0, 1) for pi in (0, 1)}
    if any(np.isnan(v) for v in cells.values()):
        return DiDReport(ok=False, warnings=["Every group x period cell needs observations."])
    X = sm.add_constant(np.column_stack([g, p, g * p]))
    model = sm.OLS(sub[outcome].to_numpy(float), X)
    res = (model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub[cluster_col])[0]})
           if cluster_col else model.fit(cov_type="HC1"))
    lo, hi = res.conf_int()[3]
    warnings = ["The effect assumes parallel trends: without treatment, both groups would have changed by the same amount."]
    pre_p = None
    if time_col:
        pre = sub[sub[post_col] == 0]
        if pre[time_col].nunique() >= 2:
            tt = pd.factorize(pre[time_col], sort=True)[0].astype(float)
            gg = pre[group_col].to_numpy(float)
            r = sm.OLS(pre[outcome].to_numpy(float), sm.add_constant(np.column_stack([gg, tt, gg * tt]))).fit(cov_type="HC1")
            pre_p = float(r.pvalues[3])
            if pre_p < 0.05:
                warnings.append(f"Pre-treatment trends differ between groups (p = {pre_p:.3g}); parallel trends is doubtful.")
        else:
            warnings.append("Only one pre-treatment period: parallel trends cannot be checked.")
    else:
        warnings.append("No time_col given: parallel trends was not checked.")
    if cluster_col and sub[cluster_col].nunique() < 30:
        warnings.append(f"Only {sub[cluster_col].nunique()} clusters; clustered SEs are unreliable below ~30.")
    return DiDReport(ok=True, warnings=warnings, effect=float(res.params[3]), se=float(res.bse[3]),
                     p_value=float(res.pvalues[3]), ci_lower=float(lo), ci_upper=float(hi),
                     cell_means=cells, pre_trend_p=pre_p, n=len(sub))
