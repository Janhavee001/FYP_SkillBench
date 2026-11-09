"""
Tests for the last 10 catalog tools. Ground truth is independent of the code under test:
sklearn for logistic regression, R coxph's published estimate for the Freireich trial,
analytic conjugate posteriors, and synthetic data with planted effects.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skills.bayesian import bayesian_estimate
from skills.causal import difference_in_differences, propensity_score_matching
from skills.cleaning import engineer_categorical_interactions, polynomial_expand
from skills.common import BENCHMARK_DATASETS
from skills.modeling import glm_fit, robust_regression
from skills.survival import cox_regression
from skills.timeseries import arima_forecast, seasonal_decompose

TITANIC_CSV = BENCHMARK_DATASETS / "titanic" / "titanic.csv"
needs_data = pytest.mark.skipif(not TITANIC_CSV.exists(), reason="benchmark datasets not found")


# ------------------------------------------------------------- feature engineering
def test_categorical_interactions_count_cross_groupstat():
    df = pd.DataFrame({"a": ["x", "x", "y", "z"], "b": ["p", "q", "p", "p"], "v": [1.0, 3.0, 5.0, 7.0]})
    c = engineer_categorical_interactions(df, "count", ["a"])
    assert c.df["a_cnt"].tolist() == [2, 2, 1, 1]
    x = engineer_categorical_interactions(df, "cross", ["a", "b"])
    assert x.df["a_b"].tolist() == ["x_p", "x_q", "y_p", "z_p"]
    capped = engineer_categorical_interactions(df, "cross", ["a", "b"], max_categories=2)
    assert capped.details["new_columns"] == [] and capped.warnings
    g = engineer_categorical_interactions(df, "group_stat", ["a"], agg_col="v", agg_funcs=["mean", "max"])
    assert g.df["v_mean_by_a"].tolist() == [2.0, 2.0, 5.0, 7.0] and any("leak" in w for w in g.warnings)


def test_polynomial_expand_terms_and_missing_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, None], "b": [3.0, 4.0, 5.0]})
    r = polynomial_expand(df, ["a", "b"], degree=2)
    assert set(r.details["new_columns"]) == {"a^2", "a*b", "b^2"}
    assert r.df.loc[1, "a*b"] == 8.0 and np.isnan(r.df.loc[2, "a^2"])
    assert polynomial_expand(df, ["a", "b"], interaction_only=True).details["new_columns"] == ["a*b"]


# ------------------------------------------------------------- GLM / robust
@needs_data
def test_logistic_glm_matches_unpenalised_sklearn():
    from sklearn.linear_model import LogisticRegression
    ti = pd.read_csv(TITANIC_CSV)
    r = glm_fit(ti, "survived", ["sex", "pclass", "fare"], family="binomial")
    sub = ti[["survived", "sex", "pclass", "fare"]].dropna()
    X = np.column_stack([(sub.sex == "male").astype(float), sub.pclass, sub.fare])
    sk = LogisticRegression(C=1e10, max_iter=10000).fit(X, sub.survived)
    assert r.coefficients["sex_male"]["coef"] == pytest.approx(sk.coef_[0][0], abs=1e-3)
    assert r.coefficients["sex_male"]["exp_coef"] < 0.2  # men had far lower odds of survival


def test_poisson_glm_recovers_rate_ratio_and_flags_overdispersion():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    df = pd.DataFrame({"x": x, "y": rng.poisson(np.exp(0.5 + 0.3 * x))})
    r = glm_fit(df, "y", ["x"], family="poisson")
    assert r.coefficients["x"]["exp_coef"] == pytest.approx(np.exp(0.3), rel=0.05)
    assert not any("Overdispersion" in w for w in r.warnings)
    df["y"] = rng.negative_binomial(1, 1 / (1 + np.exp(0.5 + 0.3 * x)))  # variance >> mean
    assert any("Overdispersion" in w for w in glm_fit(df, "y", ["x"], family="poisson").warnings)
    assert not glm_fit(df.assign(y=-1), "y", ["x"], family="poisson").ok


def test_robust_regression_resists_outliers_where_ols_does_not():
    rng = np.random.default_rng(2)
    x = np.arange(100.0)
    y = 2 * x + rng.normal(0, 1, 100)
    y[95:] = -500  # five gross outliers at high x
    r = robust_regression(pd.DataFrame({"x": x, "y": y}), "y", ["x"])
    assert r.coefficients["x"]["coef"] == pytest.approx(2.0, abs=0.1)
    assert r.coefficients["x"]["ols_coef"] < 1.5 and r.n_downweighted >= 5 and r.warnings


# ------------------------------------------------------------- time series
def _seasonal_series(n=144, period=12, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return pd.Series(10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n))


def test_stl_finds_strong_seasonality_and_trend():
    r = seasonal_decompose(_seasonal_series(), period=12)
    assert r.seasonal_strength > 0.9 and r.trend_strength > 0.9
    assert r.seasonal_amplitude == pytest.approx(6.0, rel=0.15)  # sine of amplitude 3: peak to trough 6
    assert not seasonal_decompose(_seasonal_series(), period=None).ok


def test_stl_flags_absent_seasonality():
    noise = pd.Series(np.random.default_rng(4).normal(size=120))
    assert seasonal_decompose(noise, period=12).warnings


def test_arima_recovers_ar1_and_forecasts():
    rng = np.random.default_rng(5)
    e = rng.normal(size=400)
    x = np.zeros(400)
    for i in range(1, 400):
        x[i] = 0.7 * x[i - 1] + e[i]
    r = arima_forecast(pd.Series(x), horizon=5)
    assert r.order == (1, 0, 0) and r.criterion == "bic"
    assert r.coefficients["ar.L1"] == pytest.approx(0.7, abs=0.1)
    assert len(r.forecast) == 5 and all(lo < f < hi for lo, f, hi in zip(r.forecast_lower, r.forecast, r.forecast_upper))
    assert 0 <= r.residual_autocorrelation_p <= 1  # a correct model still fails Ljung-Box 5% of the time


def test_arima_differences_a_random_walk():
    walk = pd.Series(np.random.default_rng(6).normal(size=300).cumsum())
    assert arima_forecast(walk, horizon=3, max_p=1, max_q=1).order[1] == 1


# ------------------------------------------------------------- survival
def test_cox_matches_r_coxph_on_freireich():
    t6 = [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35]
    tp = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
    df = pd.DataFrame({"t": t6 + tp, "e": [1] * 9 + [0] * 12 + [1] * 21, "placebo": [0] * 21 + [1] * 21})
    r = cox_regression(df, "t", "e", ["placebo"])
    # R: coxph(Surv(time, cens) ~ treat, data = gehan), Efron ties: coef 1.572, HR 4.82
    assert r.hazard_ratios["placebo"]["coef"] == pytest.approx(1.572, abs=0.005)
    assert r.hazard_ratios["placebo"]["hr"] == pytest.approx(4.82, abs=0.03)


# ------------------------------------------------------------- causal
def test_psm_removes_confounding_bias():
    rng = np.random.default_rng(7)
    n = 3000
    age = rng.normal(50, 10, n)
    treated = (rng.random(n) < 1 / (1 + np.exp(-(age - 50) / 5))).astype(int)  # older people get treated
    y = 2.0 * treated + 0.5 * age + rng.normal(0, 1, n)  # true effect 2; age also raises y
    r = propensity_score_matching(pd.DataFrame({"t": treated, "y": y, "age": age}), "t", "y", ["age"])
    assert r.naive_difference > 5  # badly confounded
    assert r.att == pytest.approx(2.0, abs=0.4) and r.ci_lower < 2.0 < r.ci_upper
    assert r.max_abs_smd_after < 0.1 < r.max_abs_smd_before


def test_did_recovers_planted_effect_and_flags_pretrend():
    rng = np.random.default_rng(8)
    rows = []
    for unit in range(200):
        g = unit % 2
        for period in range(4):
            post = int(period >= 2)
            y = 5 + 1.0 * g + 0.5 * period + 3.0 * g * post + rng.normal(0, 1)
            rows.append({"unit": unit, "g": g, "period": period, "post": post, "y": y})
    df = pd.DataFrame(rows)
    r = difference_in_differences(df, "y", "g", "post", time_col="period", cluster_col="unit")
    assert r.effect == pytest.approx(3.0, abs=0.3) and r.pre_trend_p > 0.05
    df.loc[(df.g == 1) & (df.post == 0), "y"] += df.loc[(df.g == 1) & (df.post == 0), "period"] * 2.0
    assert difference_in_differences(df, "y", "g", "post", time_col="period").pre_trend_p < 0.05


# ------------------------------------------------------------- Bayesian
def test_beta_binomial_posterior_is_analytic():
    df = pd.DataFrame({"x": [1] * 30 + [0] * 70})
    r = bayesian_estimate(df, "proportion", "x")
    post = stats.beta(31, 71)
    assert r.posterior["p"]["mean"] == pytest.approx(post.mean())
    assert r.posterior["p"]["lower"] == pytest.approx(post.ppf(0.025))


@needs_data
def test_two_group_proportion_gives_probability_statement():
    ti = pd.read_csv(TITANIC_CSV)
    r = bayesian_estimate(ti, "proportion", "survived", group_col="sex", groups=["female", "male"])
    assert r.prob_statements["P(p[female] > p[male])"] > 0.999
    assert r.posterior["p[female]"]["mean"] == pytest.approx(0.742, abs=0.01)


def test_jeffreys_mean_interval_equals_t_interval():
    x = np.random.default_rng(9).normal(1, 2, 40)
    r = bayesian_estimate(pd.DataFrame({"x": x}), "mean", "x")
    lo, hi = stats.t.interval(0.95, 39, loc=x.mean(), scale=stats.sem(x))
    assert r.posterior["mu"]["lower"] == pytest.approx(lo) and r.posterior["mu"]["upper"] == pytest.approx(hi)


def test_hierarchical_shrinks_small_groups_more():
    rng = np.random.default_rng(10)
    rows = [{"g": f"g{i}", "y": v} for i, n in enumerate([200, 200, 200, 5, 5])
            for v in rng.normal([0, 1, 2, 4, -3][i], 2, n)]
    r = bayesian_estimate(pd.DataFrame(rows), "hierarchical", "y", group_col="g")
    big, small = r.prob_statements["shrinkage[g0]"], r.prob_statements["shrinkage[g3]"]
    assert small > big
    grand = r.prob_statements["grand_mean"]
    p3 = r.posterior["g3"]
    assert abs(p3["mean"] - grand) < abs(p3["raw_mean"] - grand)  # pulled toward the grand mean
