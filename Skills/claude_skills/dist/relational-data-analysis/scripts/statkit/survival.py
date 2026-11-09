"""
Category G: survival analysis, implemented directly on numpy/scipy (product-limit
estimator and the log-rank statistic are short and fully auditable) rather
than adding a dependency. Convention: `event` is 1 if the event was observed,
0 if the observation was censored.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from .common import ToolResult


def _clean(df: pd.DataFrame, duration: str, event: str):
    missing = [c for c in (duration, event) if c not in df.columns]
    if missing:
        return None, [f"Column(s) not found: {missing}"]
    sub = df[[duration, event]].dropna()
    if (sub[duration] < 0).any():
        return None, [f"{duration!r} contains negative durations."]
    if not set(sub[event].unique()) <= {0, 1, True, False}:
        return None, [f"{event!r} must be 0/1 (1 = event observed)."]
    return sub, []


def _km(t: np.ndarray, e: np.ndarray):
    times = np.unique(t[e == 1])
    surv, s, at_risk_list, events_list = [], 1.0, [], []
    for ti in times:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & (e == 1)).sum())
        s *= 1 - d / at_risk
        surv.append(s)
        at_risk_list.append(at_risk)
        events_list.append(d)
    return times, np.array(surv), at_risk_list, events_list


@dataclasses.dataclass
class KaplanMeierReport(ToolResult):
    n: int = 0
    n_events: int = 0
    n_censored: int = 0
    median_survival: float | None = None
    times: list[float] = dataclasses.field(default_factory=list)
    survival: list[float] = dataclasses.field(default_factory=list)
    ci_lower: list[float] = dataclasses.field(default_factory=list)
    ci_upper: list[float] = dataclasses.field(default_factory=list)


def kaplan_meier_curve(df: pd.DataFrame, duration: str, event: str) -> KaplanMeierReport:
    """Product-limit survival curve with a 95% CI and median survival (cf. StatABench kaplan_meier_plot,
    which returns the curve but no interval). The interval is Greenwood's variance on the log(-log)
    scale, the usual default (also lifelines'), which keeps bounds inside [0, 1] without clipping.
    The curve is returned as data; plot `times` vs `survival` as a step function."""
    sub, problems = _clean(df, duration, event)
    if sub is None:
        return KaplanMeierReport(ok=False, warnings=problems)
    t, e = sub[duration].to_numpy(float), sub[event].astype(int).to_numpy()
    if e.sum() == 0:
        return KaplanMeierReport(ok=False, n=len(t), n_censored=len(t),
                                  warnings=["No events observed; the survival curve is undefined."])
    times, surv, at_risk, events = _km(t, e)
    var_sum = np.cumsum([d / (n * (n - d)) if n > d else np.inf for n, d in zip(at_risk, events)])
    # Greenwood's variance is infinite once everyone at risk has the event (S = 0); the curve is exactly 0 there.
    lo, hi = np.zeros_like(surv), np.zeros_like(surv)
    pos = surv > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = 1.96 * np.sqrt(var_sum[pos]) / np.abs(np.log(surv[pos]))
    lo[pos], hi[pos] = surv[pos] ** np.exp(c), surv[pos] ** np.exp(-c)
    below = np.where(surv <= 0.5)[0]
    median = float(times[below[0]]) if len(below) else None
    warnings = []
    if median is None:
        warnings.append("Survival never drops to 0.5: the median is not reached within follow-up.")
    if e.mean() < 0.2:
        warnings.append(f"Only {e.mean():.0%} of observations are events; the late tail of the curve is imprecise.")
    return KaplanMeierReport(ok=True, warnings=warnings, n=len(t), n_events=int(e.sum()),
                             n_censored=int(len(e) - e.sum()), median_survival=median,
                             times=times.tolist(), survival=surv.tolist(),
                             ci_lower=lo.tolist(), ci_upper=hi.tolist())


@dataclasses.dataclass
class LogRankReport(ToolResult):
    group_a: str = ""
    group_b: str = ""
    n_a: int = 0
    n_b: int = 0
    events_a: int = 0
    events_b: int = 0
    chi2: float = 0.0
    p_value: float = 0.0
    significant_at_05: bool = False


def logrank_test(df: pd.DataFrame, duration: str, event: str, group_col: str,
                  group_a, group_b) -> LogRankReport:
    """Two-group log-rank test (cf. StatABench logrank_test_compare). It compares whole curves, so it
    has little power when the curves cross -- the warning says so when it detects a crossing."""
    sub, problems = _clean(df, duration, event)
    if sub is None or group_col not in df.columns:
        return LogRankReport(ok=False, warnings=problems or [f"Column {group_col!r} not found."])
    sub = df.loc[sub.index, [duration, event, group_col]]
    a = sub[sub[group_col] == group_a]
    b = sub[sub[group_col] == group_b]
    if len(a) < 2 or len(b) < 2:
        return LogRankReport(ok=False, warnings=["Need at least 2 observations per group."])
    ta, ea = a[duration].to_numpy(float), a[event].astype(int).to_numpy()
    tb, eb = b[duration].to_numpy(float), b[event].astype(int).to_numpy()
    all_t = np.concatenate([ta, tb])
    event_times = np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]]))
    o_minus_e, var = 0.0, 0.0
    for ti in event_times:
        na, nb = (ta >= ti).sum(), (tb >= ti).sum()
        n = na + nb
        d = ((ta == ti) & (ea == 1)).sum() + ((tb == ti) & (eb == 1)).sum()
        da = ((ta == ti) & (ea == 1)).sum()
        o_minus_e += da - d * na / n
        if n > 1:
            var += d * (na / n) * (nb / n) * (n - d) / (n - 1)
    if var == 0:
        return LogRankReport(ok=False, warnings=["No variance in the log-rank statistic (no events?)."])
    chi2 = float(o_minus_e ** 2 / var)
    p = float(stats.chi2.sf(chi2, 1))

    warnings = []
    grid = np.linspace(0, all_t.max(), 50)
    def surv_at(t, e):
        times, s, *_ = _km(t, e)
        return np.array([s[times <= g][-1] if (times <= g).any() else 1.0 for g in grid])
    try:
        diff = surv_at(ta, ea) - surv_at(tb, eb)
        if diff.max() > 0.05 and diff.min() < -0.05:
            warnings.append("The survival curves appear to cross; the log-rank test has low power against "
                            "crossing hazards, so a non-significant result is not evidence of no difference.")
    except (ValueError, IndexError):
        pass
    return LogRankReport(ok=True, warnings=warnings, group_a=str(group_a), group_b=str(group_b),
                         n_a=len(a), n_b=len(b), events_a=int(ea.sum()), events_b=int(eb.sum()),
                         chi2=chi2, p_value=p, significant_at_05=p < 0.05)


@dataclasses.dataclass
class CoxReport(ToolResult):
    n: int = 0
    n_events: int = 0
    covariates: list[str] = dataclasses.field(default_factory=list)
    hazard_ratios: dict = dataclasses.field(default_factory=dict)  # name -> {coef, hr, se, p, hr_lower, hr_upper}
    proportional_hazards_p: dict = dataclasses.field(default_factory=dict)  # name -> p of the PH check


def cox_regression(df: pd.DataFrame, duration: str, event: str, covariates: list[str],
                    ties: str = "efron") -> CoxReport:
    """
    Cox proportional-hazards regression (cf. StatABench fit_cox_model, which uses lifelines; this
    uses statsmodels PHReg with Efron ties, the R `coxph` default). Categorical covariates are
    dummy-coded. The proportional-hazards assumption is checked per covariate by correlating
    Schoenfeld residuals with event time (Grambsch-Therneau style, rank of time); p < 0.05 means
    the hazard ratio changes over time and a single HR misleads.
    """
    from statsmodels.duration.hazard_regression import PHReg

    sub, problems = _clean(df, duration, event)
    if sub is None:
        return CoxReport(ok=False, warnings=problems)
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        return CoxReport(ok=False, warnings=[f"Column(s) not found: {missing}"])
    frame = df.loc[sub.index, [duration, event] + covariates].dropna()
    cats = [c for c in covariates if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])]
    warnings = []
    if cats:
        frame = pd.get_dummies(frame, columns=cats, drop_first=True, dtype=float)
        warnings.append(f"Dummy-coded {cats} (first level is the reference).")
    xs = [c for c in frame.columns if c not in (duration, event)]
    t, e = frame[duration].to_numpy(float), frame[event].astype(int).to_numpy()
    if e.sum() < 10 * len(xs):
        warnings.append(f"{int(e.sum())} events for {len(xs)} covariate(s); fewer than ~10 events per covariate "
                        "makes the estimates unstable.")
    res = PHReg(t, frame[xs].to_numpy(float), status=e, ties=ties).fit()
    ci = res.conf_int()
    hrs = {nm: {"coef": float(b), "hr": float(np.exp(b)), "se": float(s), "p": float(p),
                "hr_lower": float(np.exp(lo)), "hr_upper": float(np.exp(hi))}
           for nm, b, s, p, (lo, hi) in zip(xs, res.params, res.bse, res.pvalues, ci)}
    ph = {}
    sch = res.schoenfeld_residuals
    ev = np.isfinite(sch).all(axis=1)
    if ev.sum() > 3:
        ranks = stats.rankdata(t[ev])
        for j, nm in enumerate(xs):
            ph[nm] = float(stats.pearsonr(ranks, sch[ev, j])[1]) if np.std(sch[ev, j]) > 0 else 1.0
        bad = [nm for nm, p in ph.items() if p < 0.05]
        if bad:
            warnings.append(f"Proportional hazards looks violated for {bad}: the hazard ratio changes over time.")
    return CoxReport(ok=True, warnings=warnings, n=len(frame), n_events=int(e.sum()), covariates=xs,
                     hazard_ratios=hrs, proportional_hazards_p=ph)
