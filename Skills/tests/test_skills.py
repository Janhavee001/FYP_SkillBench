"""
Regression tests for the first-pass skill implementations. No API key
required -- every tool here is a plain function, matching the convention in
Benchmark/benchmark-fyp/harness/test_rubric.py.

Ground truth values (0.3838 / 0.742 / 0.1889 survival rates, r=0.476 for
wine alcohol-quality) are the same ones already independently verified in
DOCUMENTATION.md §4.2, so these tests double as a cross-check against
numbers already computed for the benchmark, not freshly invented here.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from skills.common import BENCHMARK_DATASETS
from skills.profiling import detect_separator_and_encoding, profile_dataset, missing_value_report
from skills.validation import validate_grounding, check_method_appropriateness
from skills.descriptive import describe_column, correlation_matrix
from skills.inference import chi_square_association, ab_test
from skills.interpretation import interpret_result

TITANIC_CSV = BENCHMARK_DATASETS / "titanic" / "titanic.csv"
WINE_CSV = BENCHMARK_DATASETS / "wine_quality" / "winequality-red.csv"

pytestmark = pytest.mark.skipif(
    not TITANIC_CSV.exists() or not WINE_CSV.exists(),
    reason="benchmark datasets not found; run from a checkout with Benchmark/benchmark-fyp/datasets present")


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return pd.read_csv(TITANIC_CSV)


@pytest.fixture(scope="module")
def wine() -> pd.DataFrame:
    return pd.read_csv(WINE_CSV)


# --------------------------------------------------------------------------- #
# profiling
# --------------------------------------------------------------------------- #
def test_detect_separator_correct_file_is_comma(wine):
    report = detect_separator_and_encoding(WINE_CSV, expected_min_columns=12)
    assert report.ok
    assert report.detected_separator == ","
    assert report.columns_at_detected_sep == 12
    assert not report.warnings


def test_detect_separator_catches_the_wine_bug(tmp_path):
    """
    Reproduces DOCUMENTATION.md §6.4 exactly: a comma-separated file, sniffed
    or forced with sep=';', should be caught rather than silently collapsed
    into one column.
    """
    bad_path = tmp_path / "wine_bad.csv"
    bad_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")

    df_wrong = pd.read_csv(bad_path, sep=";")
    assert len(df_wrong.columns) == 1  # reproduces the silent failure

    report = detect_separator_and_encoding(bad_path, expected_min_columns=3)
    assert report.detected_separator == ","
    assert report.columns_at_detected_sep == 3


def test_profile_dataset_titanic_shape(titanic):
    profile = profile_dataset(titanic, expected_columns=15)
    assert profile.ok
    assert profile.n_rows == 891
    assert profile.n_cols == 15
    assert "survived" in profile.columns


def test_profile_dataset_flags_column_count_mismatch(titanic):
    profile = profile_dataset(titanic, expected_columns=99)
    assert not profile.ok
    assert profile.warnings


def test_missing_value_report_deck_is_structural(titanic):
    report = missing_value_report(titanic, columns=["deck", "age", "survived"])
    assert "deck" in report.structurally_missing
    assert "age" in report.sparse
    assert "survived" not in report.structurally_missing
    assert "survived" not in report.sparse
    # matches DOCUMENTATION.md's own spot-check: deck ~77% missing, age ~20%
    assert report.per_column["deck"]["missing_pct"] > 70
    assert 15 < report.per_column["age"]["missing_pct"] < 25


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_grounding_flags_missing_column(titanic):
    profile = profile_dataset(titanic)
    report = validate_grounding(profile, referenced_columns=["survived", "not_a_real_column"])
    assert not report.ok
    assert "not_a_real_column" in report.unknown


def test_validate_grounding_catches_nvagent_style_table_bug(titanic):
    """Reproduces DOCUMENTATION.md §6.6: querying a table name that was never
    actually loaded."""
    profile = profile_dataset(titanic)
    code = "SELECT * FROM wine WHERE quality > 5"
    report = validate_grounding(profile, referenced_columns=[], code=code,
                                 known_tables=["titanic"])
    assert not report.ok
    assert "wine" in report.unknown_tables


def test_validate_grounding_passes_when_everything_exists(titanic):
    profile = profile_dataset(titanic)
    report = validate_grounding(profile, referenced_columns=["survived", "sex"])
    assert report.ok
    assert not report.unknown


def test_method_appropriateness_flags_correlation_on_binary(titanic):
    report = check_method_appropriateness(titanic, "correlation", ["survived", "fare"])
    assert report.verdict == "questionable"


def test_method_appropriateness_flags_ttest_on_three_groups(titanic):
    report = check_method_appropriateness(titanic, "t_test", ["pclass", "fare"])
    assert report.verdict == "inappropriate"


def test_method_appropriateness_accepts_chi_square_on_categoricals(titanic):
    report = check_method_appropriateness(titanic, "chi_square_test", ["sex", "embarked"])
    assert report.verdict == "appropriate"


# --------------------------------------------------------------------------- #
# descriptive
# --------------------------------------------------------------------------- #
def test_describe_column_numeric(titanic):
    desc = describe_column(titanic, "fare")
    assert desc.ok
    assert desc.is_numeric
    assert desc.stats["mean"] > 0


def test_describe_column_categorical(titanic):
    desc = describe_column(titanic, "embarked")
    assert desc.ok
    assert not desc.is_numeric
    assert desc.stats["mode"] == "S"


def test_correlation_matrix_wine_alcohol_quality(wine):
    report = correlation_matrix(wine, columns=["alcohol", "quality", "ph"])
    assert report.ok
    # ground truth from DOCUMENTATION.md §4.2: r = 0.476
    assert report.matrix["alcohol"]["quality"] == pytest.approx(0.476, abs=0.01)
    assert report.strongest_pair is not None


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def test_chi_square_sex_survived_is_significant(titanic):
    report = chi_square_association(titanic, "sex", "survived")
    assert report.ok
    assert report.significant_at_05
    assert report.cramers_v > 0.3  # a large, well-known effect in this dataset


def test_ab_test_fare_by_survival_matches_known_direction(titanic):
    report = ab_test(titanic, "survived", "fare", 1, 0)
    assert report.ok
    assert report.mean_a > report.mean_b  # survivors paid more, on average
    assert report.significant_at_05


def test_ab_test_warns_on_tiny_groups():
    tiny = pd.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    report = ab_test(tiny, "group", "value", "a", "b")
    assert not report.ok
    assert report.warnings


# --------------------------------------------------------------------------- #
# interpretation
# --------------------------------------------------------------------------- #
def test_interpret_result_significant_and_large_effect():
    interp = interpret_result("ab_test", p_value=0.001, effect_size=0.9, n=500)
    assert interp.statistically_significant
    assert interp.practically_significant
    assert interp.effect_size_label == "large"
    assert not interp.caveats  # nothing to flag: n is large, sig and effect agree


def test_interpret_result_significant_but_negligible_effect_is_flagged():
    """The exact case DOCUMENTATION.md §6.2 says the benchmarked systems miss:
    large n makes a trivial effect statistically significant."""
    interp = interpret_result("ab_test", p_value=0.001, effect_size=0.05, n=100000)
    assert interp.statistically_significant
    assert interp.practically_significant is False
    assert any("trivial" in c or "significant" in c for c in interp.caveats)


def test_interpret_result_small_n_caveat():
    interp = interpret_result("t_test", p_value=0.2, effect_size=0.7, n=12)
    assert any("small" in c for c in interp.caveats)
