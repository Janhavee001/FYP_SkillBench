"""
Tests for the 32 tools added in the second implementation pass. No API key
needed. Where possible, ground truth is independent of the code under test:
the benchmark's verified Titanic/Wine values, scipy reference calls, and the
classic Freireich 6-MP leukaemia data for the survival tools (published
S(6)=0.857, S(23)=0.448, log-rank chi2=16.79).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from skills.common import BENCHMARK_DATASETS
from skills.loading import load_dataset
from skills.verification import (verify_numeric_consistency, check_repair_scope,
                                  flag_unmeasurable_dimension)
from skills.cleaning import (impute_missing, scale_numeric, encode_categorical,
                              detect_outliers_iqr, bin_continuous)
from skills.exploration import frequency_table, groupby_aggregate, column_summary_for_viz
from skills.hypothesis import (confidence_interval, two_sample_confidence_interval, ks_normality_test,
                                variance_equality_test, nonparametric_test, bootstrap_ab_test,
                                ab_power_analysis, multiple_comparison_correction)
from skills.modeling import (linear_regression, multivariable_regression, regularized_regression,
                              dimensionality_reduction, feature_selection)
from skills.timeseries import stationarity_test
from skills.survival import kaplan_meier_curve, logrank_test
from skills.visualization import recommend_chart, render_chart, explain_chart
from skills.interpretation import critique_findings

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


# ----------------------------- loading / verification ------------------------ #
@needs_data
def test_load_dataset_wine_is_comma_12_cols():
    r = load_dataset(WINE_CSV, expected_min_columns=12)
    assert r.ok and r.n_cols == 12 and r.separator == ","


def test_load_dataset_missing_file(tmp_path):
    assert not load_dataset(tmp_path / "nope.csv").ok


def test_numeric_consistency_catches_the_38_percent_bug():
    r = verify_numeric_consistency(claimed=38.38, recomputed=0.3838)
    assert r.relation == "percent_vs_fraction" and r.ok and r.warnings


def test_numeric_consistency_match_and_scale_error_and_mismatch():
    assert verify_numeric_consistency(0.3838, 0.38380001).relation == "match"
    bad = verify_numeric_consistency(1000.0, 1.0)
    assert not bad.ok and bad.relation == "scale_mismatch" and bad.scale_factor == 1000
    assert verify_numeric_consistency(5.0, 7.0).relation == "mismatch"


def test_repair_scope_rejects_fix_far_from_failure():
    orig = "\n".join(f"line{i}" for i in range(1, 21))
    far = orig.replace("line18", "changed")
    assert not check_repair_scope(orig, far, failing_line=3).ok
    near = orig.replace("line3", "changed")
    assert check_repair_scope(orig, near, failing_line=3).touched_failing_region


def test_repair_scope_noop_is_not_a_fix():
    assert not check_repair_scope("a\nb", "a\nb", failing_line=1).ok


def test_unmeasurable_dimension_reports_na_not_zero():
    na = flag_unmeasurable_dimension("interpretation", [None] * 13 + [0.0, 0.0, 0.0])
    assert na.status == "n/a" and na.score is None and na.coverage == "3/16"
    real_zero = flag_unmeasurable_dimension("interpretation", [0.0, 0.0, 0.0, 0.0])
    assert real_zero.status == "scored" and real_zero.score == 0.0


# ----------------------------- cleaning -------------------------------------- #
@needs_data
def test_impute_median_fills_age_and_keeps_input(titanic):
    before = int(titanic["age"].isna().sum())
    r = impute_missing(titanic, ["age"], "median")
    assert r.ok and r.df["age"].isna().sum() == 0
    assert r.details["age"]["n_filled"] == before
    assert titanic["age"].isna().sum() == before  # input not mutated


@needs_data
def test_impute_warns_when_mostly_missing(titanic):
    r = impute_missing(titanic, ["deck"], "most_frequent")
    assert any("invents data" in w for w in r.warnings)


def test_scale_numeric_standard_and_minmax():
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5], "b": [10.0, 10, 10, 10, 10]})
    s = scale_numeric(df, ["a", "b"], "standard")
    assert abs(s.df["a"].mean()) < 1e-12 and abs(s.df["a"].std(ddof=0) - 1) < 1e-12
    assert any("'b'" in w for w in s.warnings) and (s.df["b"] == 10).all()
    m = scale_numeric(df, ["a"], "minmax")
    assert m.df["a"].min() == 0 and m.df["a"].max() == 1


def test_encode_onehot_label_ordinal():
    df = pd.DataFrame({"c": ["x", "y", "x", "z"], "t": [1, 0, 1, 0]})
    oh = encode_categorical(df, ["c"], "onehot")
    assert {"c_x", "c_y", "c_z"} <= set(oh.df.columns) and "c" not in oh.df.columns
    assert encode_categorical(df, ["c"], "label").df["c"].tolist() == [0, 1, 0, 2]
    od = encode_categorical(df, ["c"], "ordinal", order={"c": ["z", "y", "x"]})
    assert od.df["c"].tolist() == [2, 1, 2, 0] and not od.warnings
    assert encode_categorical(df, ["c"], "ordinal").warnings  # alphabetical fallback is flagged


def test_encode_target_mean_and_kfold_no_leak():
    df = pd.DataFrame({"c": list("aabbccaabbcc") * 5, "t": [1, 1, 0, 0, 1, 0] * 10})
    plain = encode_categorical(df, ["c"], "target_mean", target="t")
    assert any("leaks" in w for w in plain.warnings)
    kf = encode_categorical(df, ["c"], "kfold_target_mean", target="t")
    assert kf.ok and kf.df["c"].notna().all()
    assert encode_categorical(df, ["c"], "target_mean").ok is False


def test_outliers_iqr_flags_planted_outlier():
    df = pd.DataFrame({"v": [10, 11, 12, 11, 10, 12, 11, 10, 11, 500]})
    r = detect_outliers_iqr(df, "v")
    assert r.n_outliers == 1 and 9 in r.outlier_index
    assert not detect_outliers_iqr(df.assign(v="s"), "v").ok


@needs_data
def test_bin_continuous_quantile_and_width(titanic):
    q = bin_continuous(titanic, "age", bins=4)
    assert q.ok and q.df["age_bin"].nunique() == 4
    w = bin_continuous(titanic, "age", bins=[0, 18, 65, 120], strategy="width")
    assert w.df["age_bin"].nunique() == 3
    assert not bin_continuous(titanic, "nope").ok


# ----------------------------- exploration ----------------------------------- #
@needs_data
def test_frequency_table_matches_benchmark_survival(titanic):
    r = frequency_table(titanic, "survived")
    assert r.table["0"]["count"] + r.table["1"]["count"] == 891
    assert abs(r.table["1"]["proportion"] - 0.3838) < 1e-3  # DOCUMENTATION §4.2


@needs_data
def test_frequency_table_reports_denominator_when_missing(titanic):
    r = frequency_table(titanic, "age")
    assert r.n_missing > 0 and r.n == 891 - r.n_missing and r.warnings


@needs_data
def test_groupby_aggregate_survival_by_sex(titanic):
    r = groupby_aggregate(titanic, "sex", "survived", "mean")
    assert abs(r.result["female"] - 0.742) < 1e-3  # benchmark ground truth
    assert r.group_sizes["male"] + r.group_sizes["female"] == 891
    assert not groupby_aggregate(titanic, "sex", "embarked", "mean").ok  # mean of strings


@needs_data
def test_column_summary_for_viz_roles(titanic):
    cols = column_summary_for_viz(titanic).columns
    assert cols["fare"]["role"] == "quantitative"
    assert cols["sex"]["role"] == "nominal"
    assert cols["survived"]["role"] == "nominal"  # binary


# ----------------------------- hypothesis ------------------------------------ #
@needs_data
def test_confidence_interval_matches_scipy(wine):
    r = confidence_interval(wine, "alcohol")
    lo, hi = stats.t.interval(0.95, len(wine) - 1, loc=wine["alcohol"].mean(), scale=stats.sem(wine["alcohol"]))
    assert abs(r.lower - lo) < 1e-9 and abs(r.upper - hi) < 1e-9


@needs_data
def test_two_sample_ci_contains_zero_iff_welch_not_significant(titanic):
    r = two_sample_confidence_interval(titanic, "survived", "fare", 1, 0)
    assert r.lower > 0  # survivors paid significantly more
    assert r.lower <= r.estimate <= r.upper


def test_ks_normality_detects_normal_vs_exponential():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"n": rng.normal(size=500), "e": rng.exponential(size=500)})
    assert ks_normality_test(df, "n").looks_normal
    e = ks_normality_test(df, "e")
    assert not e.looks_normal and e.skew > 1


def test_variance_equality_detects_unequal_spread():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200,
                       "v": np.r_[rng.normal(0, 1, 200), rng.normal(0, 4, 200)]})
    assert variance_equality_test(df, "g", "v", "a", "b").significant_at_05
    df["v"] = rng.normal(0, 1, 400)
    assert not variance_equality_test(df, "g", "v", "a", "b").significant_at_05


def test_nonparametric_picks_method_by_group_count():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"g": list("abc") * 40, "v": rng.normal(size=120)})
    assert nonparametric_test(df, "g", "v").method == "kruskal_wallis"
    two = nonparametric_test(df[df.g != "c"], "g", "v")
    assert two.method == "mann_whitney" and -1 <= two.effect_size <= 1
    ref = stats.mannwhitneyu(df.v[df.g == "a"], df.v[df.g == "b"])
    assert abs(two.p_value - ref.pvalue) < 1e-12


@needs_data
def test_bootstrap_ab_agrees_with_welch_and_is_reproducible(titanic):
    r1 = bootstrap_ab_test(titanic, "survived", "fare", 1, 0, n_resamples=1000)
    r2 = bootstrap_ab_test(titanic, "survived", "fare", 1, 0, n_resamples=1000)
    assert r1.significant_at_05 and r1.ci_lower > 0
    assert r1.mean_diff == r2.mean_diff and r1.ci_lower == r2.ci_lower


def test_power_analysis_textbook_values():
    # d=0.5, alpha=.05, power=.8 -> ~64 per group (Cohen); normal approx gives 63-64
    assert 62 <= ab_power_analysis(0.5, power=0.8).n_per_group <= 65
    p = ab_power_analysis(0.5, n_per_group=64, power=None)
    assert 0.78 < p.power < 0.82
    assert ab_power_analysis(0.2, n_per_group=20, power=None).warnings  # underpowered flagged


def test_multiple_comparison_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    bonf = multiple_comparison_correction(p, "bonferroni")
    assert np.allclose(bonf.adjusted, [0.04, 0.16, 0.12, 0.02])
    bh = multiple_comparison_correction(p, "fdr_bh")
    assert np.allclose(bh.adjusted, [0.02, 0.04, 0.04, 0.02])
    assert multiple_comparison_correction(p, "holm").n_rejected == 2
    assert bonf.n_rejected_uncorrected == 4 and bonf.warnings
    assert not multiple_comparison_correction([1.5], "holm").ok


# ----------------------------- modeling -------------------------------------- #
@needs_data
def test_linear_regression_matches_benchmark_correlation(wine):
    r = linear_regression(wine, "alcohol", "quality")
    assert abs(np.sqrt(r.r_squared) - 0.476) < 1e-3  # DOCUMENTATION §4.2
    ref = stats.linregress(wine["alcohol"], wine["quality"])
    assert abs(r.coefficients["alcohol"]["coef"] - ref.slope) < 1e-9
    assert abs(r.coefficients["alcohol"]["p"] - ref.pvalue) < 1e-9


def test_multivariable_regression_recovers_coefficients_and_flags_collinearity():
    rng = np.random.default_rng(4)
    x1, x2 = rng.normal(size=500), rng.normal(size=500)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": 2 * x1 - 3 * x2 + 1 + rng.normal(scale=0.1, size=500)})
    r = multivariable_regression(df, "y", ["x1", "x2"])
    assert abs(r.coefficients["x1"]["coef"] - 2) < 0.05 and abs(r.coefficients["x2"]["coef"] + 3) < 0.05
    assert all(v < 1.2 for v in r.vif.values())
    df["x3"] = df["x1"] * 1.0 + rng.normal(scale=0.01, size=500)
    assert any("VIF" in w for w in multivariable_regression(df, "y", ["x1", "x3"]).warnings)
    df["dup"] = df["x1"]
    assert not multivariable_regression(df, "y", ["x1", "dup"]).ok  # perfectly collinear


def test_regularized_lasso_drops_noise_features():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 6))
    df = pd.DataFrame(X, columns=list("abcdef"))
    df["y"] = 3 * df["a"] - 2 * df["b"] + rng.normal(scale=0.5, size=400)
    r = regularized_regression(df, "y", list("abcdef"), "lasso")
    assert {"a", "b"} <= set(r.selected) and len(r.dropped) >= 3
    assert regularized_regression(df, "y", list("abcdef"), "ridge").ok
    assert regularized_regression(df, "y", list("abcdef"), "elastic_net").ok
    assert not regularized_regression(df, "y", list("abcdef"), "magic").ok


@needs_data
def test_pca_variance_sums_and_loadings(wine):
    r = dimensionality_reduction(wine.drop(columns="quality"), n_components=3)
    assert r.ok and 0 < r.cumulative_variance[-1] <= 1 and r.explained_variance_ratio[0] > r.explained_variance_ratio[1]
    assert set(r.loadings["PC1"]) == set(wine.drop(columns="quality").columns)
    assert dimensionality_reduction(wine.drop(columns="quality"), n_components=2, method="sparse_pca").ok


def test_feature_selection_variance_and_correlation():
    df = pd.DataFrame({"const": [1.0] * 50, "x": np.arange(50.0), "noise": np.random.default_rng(6).normal(size=50)})
    df["y"] = df["x"] * 2
    assert "const" in feature_selection(df, "y", method="variance").dropped
    c = feature_selection(df, "y", method="correlation", top_k=1)
    assert c.selected == ["x"]
    assert not feature_selection(df, None, method="correlation").ok


# ----------------------------- time series ----------------------------------- #
def test_stationarity_noise_vs_random_walk():
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(size=300))
    walk = pd.Series(rng.normal(size=300).cumsum())
    assert stationarity_test(noise).verdict == "stationary"
    assert stationarity_test(walk).verdict == "non_stationary"
    assert not stationarity_test(pd.Series([1.0, 2.0, 3.0])).ok


# ----------------------------- survival -------------------------------------- #
MP6_T = [6, 6, 6, 7, 10, 13, 16, 22, 23, 6, 9, 10, 11, 17, 19, 20, 25, 32, 32, 34, 35]
MP6_E = [1] * 9 + [0] * 12
PLACEBO_T = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]


def test_kaplan_meier_matches_published_freireich_values():
    r = kaplan_meier_curve(pd.DataFrame({"t": MP6_T, "e": MP6_E}), "t", "e")
    surv = dict(zip(r.times, r.survival))
    assert abs(surv[6] - 18 / 21) < 1e-9 and abs(surv[6] - 0.857) < 1e-3
    assert abs(surv[23] - 0.448) < 1e-3
    assert r.median_survival == 23 and r.n_events == 9 and r.n_censored == 12


def test_kaplan_meier_ci_is_finite_when_curve_reaches_zero():
    r = kaplan_meier_curve(pd.DataFrame({"t": PLACEBO_T, "e": [1] * 21}), "t", "e")
    assert r.survival[-1] == 0 and np.isfinite(r.ci_lower).all() and np.isfinite(r.ci_upper).all()


def test_kaplan_meier_rejects_no_events_and_bad_event_column():
    assert not kaplan_meier_curve(pd.DataFrame({"t": [1, 2, 3], "e": [0, 0, 0]}), "t", "e").ok
    assert not kaplan_meier_curve(pd.DataFrame({"t": [1, 2, 3], "e": [0, 2, 1]}), "t", "e").ok


def test_logrank_matches_published_freireich_chi2():
    df = pd.DataFrame({"t": MP6_T + PLACEBO_T, "e": MP6_E + [1] * 21,
                       "g": ["6mp"] * 21 + ["placebo"] * 21})
    r = logrank_test(df, "t", "e", "g", "6mp", "placebo")
    assert abs(r.chi2 - 16.79) < 0.05 and r.p_value < 1e-4 and r.significant_at_05


# ----------------------------- visualization --------------------------------- #
@needs_data
def test_recommend_chart_by_roles(titanic, wine):
    assert recommend_chart(wine, "alcohol", "quality").chart_type in ("scatter", "box", "bar")
    assert recommend_chart(wine, "alcohol", "density").chart_type == "scatter"
    assert recommend_chart(titanic, "age").chart_type == "histogram"
    assert recommend_chart(titanic, "sex", "survived").chart_type == "heatmap"
    assert not recommend_chart(titanic, "ghost").ok


@needs_data
def test_render_chart_writes_png_and_regrounds(titanic, tmp_path):
    spec = recommend_chart(titanic, "age")
    out = render_chart(titanic, spec, tmp_path / "age.png")
    assert out.ok and (tmp_path / "age.png").stat().st_size > 1000
    assert not render_chart(titanic.drop(columns="age"), spec, tmp_path / "x.png").ok  # frame changed


@needs_data
def test_explain_chart_states_finding_with_numbers(wine):
    spec = recommend_chart(wine, "alcohol", "density")
    e = explain_chart(wine, spec)
    assert e.ok and "association" in e.finding and abs(e.facts["pearson_r"] - wine["alcohol"].corr(wine["density"])) < 1e-12
    assert "palette" not in e.finding.lower() and "color" not in e.finding.lower()


# ----------------------------- interpretation -------------------------------- #
def test_critique_findings_ranks_and_blocks_on_high():
    c = critique_findings("t_test", n=8, p_value=0.03, n_tests=10, normality_p=0.01, observational=True)
    assert not c.ok and c.n_high >= 2
    assert c.violations[0]["severity"] == "high"
    assert any("Bonferroni" in v["issue"] for v in c.violations)


def test_critique_findings_clean_result_passes():
    c = critique_findings("t_test", n=500, p_value=0.0001, observational=False)
    assert c.ok and c.violations == []


def test_critique_chi_square_small_expected_cells():
    c = critique_findings("chi_square_association", n=100, expected_min_cell=2.1, observational=False)
    assert not c.ok
