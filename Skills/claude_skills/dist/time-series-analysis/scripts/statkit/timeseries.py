"""
Category F: time series. First pass is the gatekeeping tool -- a forecast or
a regression on a non-stationary series is the time-series version of the
"method does not fit the data" failure check_method_appropriateness guards.
"""
from __future__ import annotations

import dataclasses
import warnings as _warnings

import numpy as np
import pandas as pd

from .common import ToolResult, is_numeric


@dataclasses.dataclass
class StationarityReport(ToolResult):
    n: int = 0
    adf_statistic: float = 0.0
    adf_p_value: float = 0.0
    kpss_statistic: float = 0.0
    kpss_p_value: float = 0.0
    verdict: str = ""  # stationary | non_stationary | trend_stationary | difference_stationary | inconclusive


def stationarity_test(series: pd.Series | pd.DataFrame, column: str | None = None,
                       alpha: float = 0.05, regression: str = "c",
                       time_col: str | None = None) -> StationarityReport:
    """
    ADF (H0: unit root) and KPSS (H0: stationary) run together, because each
    alone is easy to over-read; agreement is evidence, disagreement is flagged
    (cf. StatABench test_stationarity, which runs one or the other).

    With a DataFrame, pass `time_col` so the rows are put in time order first:
    both tests read the series in row order, and a shuffled or grouped file
    gives a meaningless answer.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    if isinstance(series, pd.DataFrame) and column:
        frame = series
        if time_col is not None:
            if time_col not in frame.columns:
                return StationarityReport(ok=False, warnings=[f"time_col {time_col!r} not found."])
            order = pd.to_datetime(frame[time_col], errors="coerce", format="mixed")
            if order.isna().all():
                order = frame[time_col]
            frame = frame.assign(_order=order).sort_values("_order")
            if frame[time_col].duplicated().any():
                return StationarityReport(ok=False, warnings=[
                    f"{time_col!r} has repeated timestamps (several series stacked?); aggregate or filter first."])
        series = frame[column]
    s = series
    if isinstance(s, pd.DataFrame) or not is_numeric(s):
        return StationarityReport(ok=False, warnings=["Pass a numeric series (or a frame plus `column`)."])
    s = s.dropna()
    if len(s) < 20:
        return StationarityReport(ok=False, n=len(s), warnings=[f"n={len(s)} is too short for a reliable stationarity test (need ~20+)."])
    adf_stat, adf_p, *_ = adfuller(s, regression=regression, autolag="AIC", result_object=False)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # KPSS warns when p is outside its lookup table; we report the clipped p
        kpss_stat, kpss_p, *_ = kpss(s, regression=regression, nlags="auto")
    adf_stationary, kpss_stationary = adf_p < alpha, kpss_p >= alpha
    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non_stationary"
    elif adf_stationary and not kpss_stationary:
        verdict = "difference_stationary"
    else:
        verdict = "trend_stationary"
    notes = []
    if verdict in ("difference_stationary", "trend_stationary"):
        notes.append("ADF and KPSS disagree; difference or detrend and re-test before modelling.")
    if verdict == "non_stationary":
        notes.append("Both tests indicate non-stationarity; difference the series before ARIMA-style modelling.")
    return StationarityReport(ok=True, warnings=notes, n=len(s), adf_statistic=float(adf_stat),
                              adf_p_value=float(adf_p), kpss_statistic=float(kpss_stat),
                              kpss_p_value=float(kpss_p), verdict=verdict)


def _ordered_series(data: pd.Series | pd.DataFrame, column: str | None, time_col: str | None):
    """Shared by the time-series tools: a numeric series in time order, or a reason it is not one."""
    if isinstance(data, pd.DataFrame):
        if column is None or column not in data.columns:
            return None, f"Column {column!r} not found."
        frame = data
        if time_col is not None:
            if time_col not in frame.columns:
                return None, f"time_col {time_col!r} not found."
            order = pd.to_datetime(frame[time_col], errors="coerce", format="mixed")
            frame = frame.assign(_order=order if order.notna().any() else frame[time_col]).sort_values("_order")
            if frame[time_col].duplicated().any():
                return None, f"{time_col!r} has repeated timestamps; aggregate first."
        s = frame[column]
    else:
        s = data
    if not is_numeric(s):
        return None, "Series is not numeric."
    return s.dropna().reset_index(drop=True), None


@dataclasses.dataclass
class DecompositionReport(ToolResult):
    period: int = 0
    n: int = 0
    trend_strength: float = 0.0     # Hyndman's F_T, 0..1
    seasonal_strength: float = 0.0  # Hyndman's F_S, 0..1
    seasonal_amplitude: float = 0.0  # median peak-to-trough range within one cycle
    trend_change: float = 0.0       # trend at end minus trend at start
    trend: list[float] = dataclasses.field(default_factory=list)
    seasonal: list[float] = dataclasses.field(default_factory=list)
    resid: list[float] = dataclasses.field(default_factory=list)


def seasonal_decompose(data: pd.Series | pd.DataFrame, column: str | None = None, period: int | None = None,
                        time_col: str | None = None) -> DecompositionReport:
    """
    STL decomposition into trend + seasonal + remainder (cf. StatABench decompose_stl, which returns
    component means). Reports trend and seasonal *strength* (Hyndman & Athanasopoulos, FPP3 s.4.3:
    1 - Var(R)/Var(T+R) and 1 - Var(R)/Var(S+R)), which answer "is there really seasonality?" with a
    number. `period` is required: guessing it from the data is a common silent error (7 for daily
    data with weekly cycles, 12 for monthly data with yearly cycles).
    """
    from statsmodels.tsa.seasonal import STL

    s, err = _ordered_series(data, column, time_col)
    if err:
        return DecompositionReport(ok=False, warnings=[err])
    if period is None or period < 2:
        return DecompositionReport(ok=False, warnings=["Give `period` (e.g. 7 for daily data with a weekly cycle, "
                                                       "12 for monthly data with a yearly cycle)."])
    if len(s) < 2 * period:
        return DecompositionReport(ok=False, warnings=[f"Need at least two full periods ({2 * period} points); got {len(s)}."])
    var = lambda a: float(np.var(a))  # noqa: E731

    def strengths(values):
        fit = STL(values, period=period, robust=True).fit()
        tt, ss, rr = fit.trend, fit.seasonal, fit.resid
        ft = max(0.0, 1 - var(rr) / var(tt + rr)) if var(tt + rr) else 0.0
        fs = max(0.0, 1 - var(rr) / var(ss + rr)) if var(ss + rr) else 0.0
        return fit, ft, fs

    res, f_t, f_s = strengths(s.to_numpy(float))
    t, se, r = res.trend, res.seasonal, res.resid
    warnings = []
    # STL always finds *some* seasonal pattern: pure noise scores ~0.3. So compare with the strength the
    # same values reach in random order (19 shuffles: the observed value beats all of them with p < 0.05
    # if there is no seasonality). Skipped for long series, where 19 refits get slow.
    if len(s) <= 5000:
        rng = np.random.default_rng(0)
        null = [strengths(rng.permutation(s.to_numpy(float)))[2] for _ in range(19)]
        if f_s <= max(null):
            warnings.append(f"Seasonal strength {f_s:.2f} is no higher than shuffled data reaches "
                            f"(up to {max(null):.2f}); there is no evidence of a period-{period} cycle.")
    elif f_s < 0.4:
        warnings.append(f"Seasonal strength is low ({f_s:.2f}); the period {period} may be wrong or absent.")
    return DecompositionReport(ok=True, warnings=warnings, period=period, n=len(s), trend_strength=f_t,
                               seasonal_strength=f_s, seasonal_amplitude=float(np.median([np.ptp(se[i:i + period]) for i in range(0, len(se) - period + 1, period)])),
                               trend_change=float(t[-1] - t[0]), trend=t.tolist(), seasonal=se.tolist(),
                               resid=r.tolist())


@dataclasses.dataclass
class ARIMAReport(ToolResult):
    order: tuple = ()
    seasonal_order: tuple = ()
    criterion: str = "bic"
    criterion_value: float = 0.0
    n: int = 0
    coefficients: dict = dataclasses.field(default_factory=dict)
    forecast: list[float] = dataclasses.field(default_factory=list)
    forecast_lower: list[float] = dataclasses.field(default_factory=list)
    forecast_upper: list[float] = dataclasses.field(default_factory=list)
    residual_autocorrelation_p: float | None = None  # Ljung-Box; < 0.05 = model missed structure
    candidates_tried: int = 0


def arima_forecast(data: pd.Series | pd.DataFrame, column: str | None = None, horizon: int = 12,
                    order: tuple | None = None, max_p: int = 3, max_q: int = 3,
                    seasonal_period: int | None = None, time_col: str | None = None,
                    criterion: str = "bic") -> ARIMAReport:
    """
    ARIMA with (p, q) chosen over a small grid (cf. StatABench auto_arima_modeling, which uses
    pmdarima with AIC). The default criterion here is BIC: on a simulated AR(1), AIC preferred an
    ARMA(2,3) by 1.2 points whose coefficients look nothing like the truth, while BIC picked AR(1).
    `criterion` accepts "bic", "aic" or "aicc". d is chosen first with the same ADF+KPSS pair as
    stationarity_test, because information criteria are not comparable across different d. Returns a forecast with 95% intervals
    and a Ljung-Box check on the residuals.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA

    if criterion not in ("bic", "aic", "aicc"):
        return ARIMAReport(ok=False, warnings=[f"Unknown criterion {criterion!r}; use bic, aic or aicc."])
    score = lambda f: getattr(f, criterion)  # noqa: E731
    s, err = _ordered_series(data, column, time_col)
    if err:
        return ARIMAReport(ok=False, warnings=[err])
    if len(s) < 30:
        return ARIMAReport(ok=False, n=len(s), warnings=[f"n={len(s)} is too short for ARIMA (need ~30+)."])
    warnings = []
    seasonal = (1, 0, 0, seasonal_period) if seasonal_period else (0, 0, 0, 0)
    if order is None:
        d, x = 0, s
        while d < 2 and stationarity_test(x).verdict == "non_stationary":
            d, x = d + 1, x.diff().dropna()
        best, tried = None, 0
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    try:
                        fit = ARIMA(s.to_numpy(float), order=(p, d, q), seasonal_order=seasonal).fit()
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    tried += 1
                    if best is None or score(fit) < score(best):
                        best = fit
        if best is None:
            return ARIMAReport(ok=False, warnings=["No ARIMA order in the grid could be fitted."])
        fit = best
    else:
        tried = 1
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            fit = ARIMA(s.to_numpy(float), order=tuple(order), seasonal_order=seasonal).fit()
    fc = fit.get_forecast(horizon)
    ci = fc.conf_int(alpha=0.05)
    lags = min(10, len(s) // 5)
    p_, _, q_ = fit.model.order
    # model_df: the p + q fitted ARMA parameters use up degrees of freedom of the Ljung-Box statistic.
    lb_p = float(acorr_ljungbox(fit.resid[1:], lags=[max(lags, p_ + q_ + 1)], model_df=p_ + q_).iloc[0]["lb_pvalue"])
    if lb_p < 0.05:
        warnings.append(f"Residuals are still autocorrelated (Ljung-Box p = {lb_p:.3g}); the model missed structure.")
    if fit.mle_retvals and not fit.mle_retvals.get("converged", True):
        warnings.append("The optimiser did not converge; treat coefficients with caution.")
    warnings.append("Forecast intervals assume the model is right; they widen in reality when it is not.")
    return ARIMAReport(ok=True, warnings=warnings, order=tuple(int(v) for v in fit.model.order),
                       seasonal_order=tuple(int(v) for v in fit.model.seasonal_order), criterion=criterion, criterion_value=float(score(fit)),
                       n=len(s), coefficients={k: float(v) for k, v in zip(fit.param_names, fit.params)},
                       forecast=[float(v) for v in fc.predicted_mean],
                       forecast_lower=[float(v) for v in ci[:, 0]], forecast_upper=[float(v) for v in ci[:, 1]],
                       residual_autocorrelation_p=lb_p, candidates_tried=tried)
