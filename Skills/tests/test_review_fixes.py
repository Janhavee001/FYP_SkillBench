"""
Regression tests for the issues found in the code review against the reference
repositories (StatABench mcp_server.py, MetaGPT data_preprocess/feature_engineering).
Each test pins one defect so it cannot quietly return.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

from skills.common import BENCHMARK_DATASETS, is_numeric
from skills.cleaning import encode_categorical, impute_missing
from skills.descriptive import describe_column
from skills.exploration import column_summary_for_viz
from skills.hypothesis import (ab_power_analysis, ks_normality_test, multiple_comparison_correction,
                                nonparametric_test)
from skills.inference import chi_square_association
from skills.interpretation import interpret_result
from skills.loading import load_dataset
from skills.modeling import feature_selection, multivariable_regression, regularized_regression
from skills.survival import kaplan_meier_curve
from skills.timeseries import stationarity_test
from skills.validation import check_method_appropriateness
from skills.visualization import explain_chart, recommend_chart, render_chart

TITANIC_CSV = BENCHMARK_DATASETS / "titanic" / "titanic.csv"
WINE_CSV = BENCHMARK_DATASETS / "wine_quality" / "winequality-red.csv"
needs_data = pytest.mark.skipif(not TITANIC_CSV.exists() or not WINE_CSV.exists(),
                                reason="benchmark datasets not found")


@pytest.fixture(scope="module")
def titanic():
    return pd.read_csv(TITANIC_CSV)


@pytest.fixture(scope="module")
def wine():
    return pd.read_csv(WINE_CSV)


# ------------------------------------------------------------- shared dtype test
def test_small_and_nullable_ints_are_numeric_bool_is_not():
    assert is_numeric(pd.Series([1, 2], dtype="int8"))
    assert is_numeric(pd.Series([1, None], dtype="Int64"))
    assert not is_numeric(pd.Series([True, False]))
    assert describe_column(pd.DataFrame({"a": np.array([1, 2, 3], dtype="int8")}), "a").is_numeric


# ------------------------------------------------------------- method gate
def test_gate_recognises_the_librarys_own_tool_names(titanic):
    assert check_method_appropriateness(titanic, "chi_square_association", ["sex", "embarked"]).verdict == "appropriate"
    assert check_method_appropriateness(titanic, "ab_test", ["pclass", "fare"]).verdict == "inappropriate"
    assert check_method_appropriateness(titanic, "ab_test", ["survived", "sex"]).verdict == "inappropriate"  # text outcome


def test_gate_does_not_silently_pass_unknown_methods(titanic):
    r = check_method_appropriateness(titanic, "magic_test", ["sex", "fare"])
    assert r.verdict == "unchecked" and not r.ok and r.warnings


def test_gate_flags_ols_on_binary_target(titanic):
    assert check_method_appropriateness(titanic, "linear_regression", ["survived", "fare"]).verdict == "inappropriate"


# ------------------------------------------------------------- chi-square
@needs_data
def test_chi_square_warns_on_expected_not_observed_counts():
    # Observed cell of 3 but every expected count >= 5: no warning should fire.
    df = pd.DataFrame({"a": ["x"] * 20 + ["y"] * 20, "b": ["p"] * 3 + ["q"] * 17 + ["p"] * 17 + ["q"] * 3})
    r = chi_square_association(df, "a", "b")
    assert r.expected_min_cell == pytest.approx(10.0) and not r.warnings
    # Expected count < 5 (tiny margin): warning fires.
    df2 = pd.DataFrame({"a": ["x"] * 36 + ["y"] * 4, "b": ["p"] * 18 + ["q"] * 18 + ["p"] * 2 + ["q"] * 2})
    assert chi_square_association(df2, "a", "b").warnings


def test_cramers_v_uses_uncorrected_chi2_on_2x2():
    df = pd.DataFrame({"a": ["x"] * 50 + ["y"] * 50, "b": ["p"] * 35 + ["q"] * 15 + ["p"] * 15 + ["q"] * 35})
    r = chi_square_association(df, "a", "b")
    # 2x2 with no correction: V equals |phi| = |ad - bc| / sqrt(row and column products)
    phi = abs(35 * 35 - 15 * 15) / np.sqrt(50 * 50 * 50 * 50)
    assert r.cramers_v == pytest.approx(phi)


# ------------------------------------------------------------- interpretation
def test_correlation_is_bucketed_on_r_scale_not_d_scale():
    assert interpret_result("correlation", effect_size=0.476, effect_size_type="r").effect_size_label == "medium"
    unknown = interpret_result("x", effect_size=0.476, effect_size_type="mystery")
    assert unknown.effect_size_label == "" and unknown.warnings


# ------------------------------------------------------------- hypothesis tests
def test_normality_uses_lilliefors_not_plain_ks_with_fitted_params():
    rng = np.random.default_rng(10)
    x = rng.standard_t(df=4, size=150)
    r = ks_normality_test(pd.DataFrame({"x": x}), "x")
    assert r.p_value == pytest.approx(lilliefors(x, dist="norm", pvalmethod="table")[1])
    plain_p = stats.kstest(x, "norm", args=(x.mean(), x.std(ddof=1))).pvalue  # StatABench-style
    assert r.p_value < plain_p


def test_power_matches_cohens_table_and_supports_proportions():
    assert ab_power_analysis(0.5).n_per_group == 64  # Cohen (1988), exact noncentral t
    n = ab_power_analysis(proportions=(0.02, 0.025)).n_per_group
    assert 13000 < n < 15000


def test_wilcoxon_pairs_by_subject_id_not_row_position():
    rng = np.random.default_rng(11)
    before = rng.normal(10, 2, 30)
    after = before + rng.normal(0.8, 0.5, 30)
    df = pd.DataFrame({"id": list(range(30)) * 2, "t": ["pre"] * 30 + ["post"] * 30, "v": np.r_[before, after]})
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle rows
    df = df[~((df.id == 0) & (df.t == "post"))]  # subject 0 lost its second measurement
    r = nonparametric_test(df, "t", "v", groups=["pre", "post"], pair_col="id")
    ref = stats.wilcoxon(before[1:], after[1:])
    assert r.method == "wilcoxon" and r.p_value == pytest.approx(ref.pvalue)
    assert r.groups == {"pairs": 29} and r.warnings  # the dropped subject is reported


def test_kruskal_effect_size_is_eta_squared_h():
    rng = np.random.default_rng(12)
    df = pd.DataFrame({"g": list("abc") * 30, "v": rng.normal(size=90)})
    r = nonparametric_test(df, "g", "v")
    assert r.effect_size_type == "eta_squared_h"
    assert interpret_result("kruskal", effect_size=r.effect_size, effect_size_type=r.effect_size_type).effect_size_label


def test_multiple_comparison_rejects_at_exactly_alpha():
    assert multiple_comparison_correction([0.05], "bonferroni").rejected == [True]


# ------------------------------------------------------------- modelling
@needs_data
def test_regression_dummy_codes_categorical_predictors(titanic):
    r = multivariable_regression(titanic, "fare", ["age", "sex"])
    assert r.ok and "sex_male" in r.coefficients
    assert any("Dummy-coded" in w for w in r.warnings)


@needs_data
def test_regression_accepts_mixed_scale_predictors(wine):
    r = multivariable_regression(wine, "quality", [c for c in wine.columns if c != "quality"])
    assert r.ok and r.n == 1599


@needs_data
def test_regularized_reports_cross_validated_r2(wine):
    r = regularized_regression(wine, "quality", [c for c in wine.columns if c != "quality"], "lasso")
    assert r.cv_r_squared is not None and 0 < r.cv_r_squared <= r.r_squared + 0.02
    assert regularized_regression(wine, "quality", ["alcohol", "sulphates"], "elasticnet").method == "elastic_net"


def test_feature_selection_general_mirrors_metagpt_general_selection():
    df = pd.DataFrame({"empty": [np.nan] * 5, "const": [1] * 5, "inf": [1.0, np.inf, 2, 3, 4],
                       "name": list("abcde"), "good": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 1]})
    r = feature_selection(df, target="y", method="general")
    assert r.selected == ["good"] and set(r.dropped) == {"empty", "const", "inf", "name"}


# ------------------------------------------------------------- time series
def test_stationarity_sorts_by_time_col():
    rng = np.random.default_rng(13)
    walk = rng.normal(size=300).cumsum()
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=300).astype(str), "v": walk})
    shuffled = df.sample(frac=1, random_state=1)
    assert stationarity_test(shuffled, "v").verdict != "non_stationary"  # row order destroys the walk
    assert stationarity_test(shuffled, "v", time_col="date").verdict == "non_stationary"


# ------------------------------------------------------------- survival
def test_km_log_log_ci_matches_textbook():
    t = [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35]
    e = [1] * 9 + [0] * 12
    r = kaplan_meier_curve(pd.DataFrame({"t": t, "e": e}), "t", "e")
    # Freireich 6-MP at t=6: S = 0.857, log-log 95% CI (0.620, 0.952), as in R survfit / lifelines
    assert r.ci_lower[0] == pytest.approx(0.620, abs=0.002)
    assert r.ci_upper[0] == pytest.approx(0.952, abs=0.002)
    assert all(lo <= s <= hi for lo, s, hi in zip(r.ci_lower, r.survival, r.ci_upper))


# ------------------------------------------------------------- visualization
def test_date_strings_are_temporal_and_get_a_line_chart():
    rng = np.random.default_rng(14)
    cnt = (np.arange(60) * 40 + rng.normal(0, 150, 60) + 1000).round().astype(int)
    df = pd.DataFrame({"dteday": pd.date_range("2011-01-01", periods=60).astype(str), "cnt": cnt})
    assert column_summary_for_viz(df).columns["dteday"]["role"] == "temporal"
    spec = recommend_chart(df, "dteday", "cnt")
    assert spec.chart_type == "line"
    assert "upward" in explain_chart(df, spec).finding


def test_consecutive_int_column_is_id_like_but_unique_counts_are_not():
    roles = column_summary_for_viz(pd.DataFrame({"instant": range(1, 101),
                                                  "cnt": np.arange(100) * 7 + 3})).columns
    assert roles["instant"]["role"] == "id_like" and roles["cnt"]["role"] == "quantitative"


def test_box_explanation_reports_medians():
    df = pd.DataFrame({"g": ["a"] * 30 + ["b"] * 30, "v": [1] * 29 + [1000] + [5] * 30})
    spec = recommend_chart(df, "g", "v")
    e = explain_chart(df, spec)
    assert spec.chart_type == "box" and e.facts["highest_median"] == ("b", 5.0)  # mean would say "a"


@needs_data
def test_correlation_heatmap(wine, tmp_path):
    spec = recommend_chart(wine, columns=["alcohol", "density", "quality", "ph"])
    assert spec.chart_type == "correlation_heatmap"
    assert render_chart(wine, spec, tmp_path / "corr.png").ok
    a, b, r = explain_chart(wine, spec).facts["strongest_pair"]
    assert {a, b} == {"alcohol", "density"} and r == pytest.approx(wine.alcohol.corr(wine.density))


def test_scatter_color_is_drawn(tmp_path):
    df = pd.DataFrame({"x": range(40), "y": range(40), "g": ["a", "b"] * 20})
    assert render_chart(df, recommend_chart(df, "x", "y", color="g"), tmp_path / "c.png").ok


# ------------------------------------------------------------- loading / cleaning
def test_load_dataset_reports_parse_errors_instead_of_crashing(tmp_path):
    bad = tmp_path / "not_really.xlsx"
    bad.write_text("this is not a spreadsheet")
    r = load_dataset(bad)
    assert not r.ok and "Could not read" in r.warnings[0]


def test_cleaning_edge_cases_report_instead_of_crash():
    df = pd.DataFrame({"n": [1.0, None], "c": ["lo", "mid"]})
    assert not impute_missing(df, ["n"], "constant:abc").ok
    r = encode_categorical(df, ["c"], "ordinal", order={"c": ["lo", "hi"]})
    assert any("became NaN" in w for w in r.warnings)
