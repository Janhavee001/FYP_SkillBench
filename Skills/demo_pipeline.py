"""
End-to-end composition demo: load -> detect -> profile -> validate -> describe
-> test -> interpret, run twice on the wine dataset -- once with the correct
separator, once reproducing the exact mistake DOCUMENTATION.md §6.4 recorded
the model making -- to show the grounding step catching what execution-feedback
repair did not.

Run: python demo_pipeline.py
"""
from __future__ import annotations

import pandas as pd

from skills.common import BENCHMARK_DATASETS
from skills.profiling import detect_separator_and_encoding, profile_dataset
from skills.validation import validate_grounding, check_method_appropriateness
from skills.descriptive import correlation_matrix
from skills.inference import ab_test
from skills.interpretation import interpret_result

WINE_CSV = BENCHMARK_DATASETS / "wine_quality" / "winequality-red.csv"
TITANIC_CSV = BENCHMARK_DATASETS / "titanic" / "titanic.csv"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_wine_pipeline(force_wrong_separator: bool) -> None:
    section(f"WINE PIPELINE  (force_wrong_separator={force_wrong_separator})")

    sep_report = detect_separator_and_encoding(WINE_CSV, expected_min_columns=12)
    print(f"[detect_separator_and_encoding] separator={sep_report.detected_separator!r} "
          f"confidence={sep_report.confidence} warnings={sep_report.warnings}")

    sep = ";" if force_wrong_separator else sep_report.detected_separator
    df = pd.read_csv(WINE_CSV, sep=sep)

    profile = profile_dataset(df, expected_columns=12)
    print(f"[profile_dataset] n_cols={profile.n_cols} warnings={profile.warnings}")

    if not profile.ok:
        print(">>> STOPPING: grounding check failed before any analysis ran. "
              "This is the fix for DOCUMENTATION.md §6.4 -- the wrong-separator "
              "bug that a repair loop never caught even when shown the traceback.")
        return

    grounding = validate_grounding(profile, referenced_columns=["alcohol", "quality"])
    print(f"[validate_grounding] ok={grounding.ok} unknown={grounding.unknown}")

    appropriateness = check_method_appropriateness(df, "correlation", ["alcohol", "quality"])
    print(f"[check_method_appropriateness] verdict={appropriateness.verdict}")

    corr = correlation_matrix(df, columns=["alcohol", "quality"])
    r = corr.matrix["alcohol"]["quality"]
    print(f"[correlation_matrix] alcohol-quality r={r}")

    interp = interpret_result("correlation", effect_size=r, effect_size_type="r")
    print(f"[interpret_result] {interp.headline}")


def run_titanic_pipeline() -> None:
    section("TITANIC PIPELINE")
    df = pd.read_csv(TITANIC_CSV)
    profile = profile_dataset(df, expected_columns=15)
    print(f"[profile_dataset] n_rows={profile.n_rows} n_cols={profile.n_cols}")

    report = ab_test(df, "survived", "fare", 1, 0)
    print(f"[ab_test] mean_a(survived)={report.mean_a:.2f} mean_b(died)={report.mean_b:.2f} "
          f"p={report.p_value:.4g} cohens_d={report.cohens_d:.3f}")

    interp = interpret_result("ab_test", p_value=report.p_value,
                               effect_size=report.cohens_d, n=report.n_a + report.n_b)
    print(f"[interpret_result] {interp.headline}")
    for c in interp.caveats:
        print(f"    caveat: {c}")


if __name__ == "__main__":
    run_wine_pipeline(force_wrong_separator=False)
    run_wine_pipeline(force_wrong_separator=True)
    run_titanic_pipeline()
