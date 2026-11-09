"""
Build the eight uploadable skills.

For each skill folder here (holding a hand-written SKILL.md) this writes dist/<skill>/ with:
  SKILL.md                    copied as written
  reference/tools.md          generated from the tool signatures and docstrings, so it cannot drift
  scripts/statkit/            a copy of ../skills (the tool library), importable as `statkit`
and zips it to dist/<skill>.zip with the skill folder at the zip root, the layout the Claude app's
skill upload expects.

Each skill carries its own copy of the library because uploaded skills are installed
independently; any one of them must work without the others.

Run:  python claude_skills/build.py
"""
from __future__ import annotations

import importlib
import inspect
import pathlib
import re
import shutil
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
LIB = HERE.parent / "skills"
DIST = HERE / "dist"
sys.path.insert(0, str(HERE.parent))

# Tools documented in each skill's reference/tools.md. Every tool is importable from every skill;
# this list is what each skill tells Claude about.
SKILL_TOOLS = {
    "data-profiling": [
        "loading.load_dataset", "profiling.detect_separator_and_encoding", "profiling.profile_dataset",
        "profiling.missing_value_report", "descriptive.describe_column", "exploration.frequency_table",
        "exploration.groupby_aggregate", "cleaning.detect_outliers_iqr", "cleaning.impute_missing",
        "validation.validate_grounding", "verification.verify_numeric_consistency"],
    "distribution-analysis": [
        "descriptive.describe_column", "hypothesis.ks_normality_test", "cleaning.detect_outliers_iqr",
        "hypothesis.confidence_interval", "bayesian.bayesian_estimate", "cleaning.bin_continuous",
        "exploration.frequency_table", "visualization.recommend_chart", "visualization.render_chart",
        "visualization.explain_chart", "interpretation.interpret_result",
        "verification.verify_numeric_consistency"],
    "correlation-analysis": [
        "validation.check_method_appropriateness", "descriptive.correlation_matrix",
        "modeling.linear_regression", "hypothesis.multiple_comparison_correction",
        "modeling.dimensionality_reduction", "interpretation.interpret_result",
        "interpretation.critique_findings", "visualization.recommend_chart", "visualization.render_chart",
        "visualization.explain_chart", "verification.verify_numeric_consistency"],
    "group-comparison": [
        "validation.check_method_appropriateness", "inference.ab_test",
        "hypothesis.two_sample_confidence_interval", "hypothesis.nonparametric_test",
        "inference.chi_square_association", "hypothesis.variance_equality_test",
        "hypothesis.ks_normality_test", "hypothesis.bootstrap_ab_test", "hypothesis.ab_power_analysis",
        "hypothesis.multiple_comparison_correction", "bayesian.bayesian_estimate",
        "survival.kaplan_meier_curve", "survival.logrank_test", "descriptive.describe_column",
        "interpretation.interpret_result", "interpretation.critique_findings",
        "verification.verify_numeric_consistency"],
    "regression-analysis": [
        "validation.check_method_appropriateness", "modeling.linear_regression",
        "modeling.multivariable_regression", "modeling.glm_fit", "modeling.robust_regression",
        "modeling.regularized_regression", "modeling.dimensionality_reduction", "modeling.feature_selection",
        "cleaning.scale_numeric", "cleaning.encode_categorical", "cleaning.polynomial_expand",
        "cleaning.engineer_categorical_interactions", "survival.cox_regression",
        "causal.propensity_score_matching", "causal.difference_in_differences",
        "interpretation.interpret_result", "interpretation.critique_findings",
        "verification.verify_numeric_consistency"],
    "time-series-analysis": [
        "exploration.column_summary_for_viz", "exploration.groupby_aggregate", "timeseries.stationarity_test",
        "timeseries.seasonal_decompose", "timeseries.arima_forecast", "visualization.recommend_chart",
        "visualization.render_chart", "visualization.explain_chart", "interpretation.critique_findings",
        "verification.verify_numeric_consistency"],
    "visualization": [
        "exploration.column_summary_for_viz", "visualization.recommend_chart", "visualization.render_chart",
        "visualization.explain_chart", "validation.validate_grounding"],
    "relational-data-analysis": [
        "relational.discover_tables", "relational.infer_keys", "relational.build_relationship_graph",
        "relational.plan_join_path", "relational.aggregate_to_grain", "relational.validate_join",
        "relational.execute_join_plan", "relational.check_unit_of_analysis",
        "relational.detect_aggregation_reversal", "validation.validate_grounding",
        "verification.verify_numeric_consistency"],
}


def tool_reference(skill: str) -> str:
    lines = [f"# Tool reference: {skill}", "",
             "Generated from the code by `build.py`; do not edit by hand.", "",
             "Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` "
             "to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) "
             "and the fields named below; `.to_dict()` converts it.", ""]
    for dotted in SKILL_TOOLS[skill]:
        mod_name, fn_name = dotted.split(".")
        fn = getattr(importlib.import_module(f"skills.{mod_name}"), fn_name)
        sig = str(inspect.signature(fn)).replace("pd.DataFrame", "DataFrame")
        doc = inspect.getdoc(fn) or ""
        ret = inspect.signature(fn).return_annotation
        ret_cls = getattr(importlib.import_module(f"skills.{mod_name}"), str(ret), None) if isinstance(ret, str) else ret
        fields = [f for f in getattr(ret_cls, "__dataclass_fields__", {}) if f not in ("ok", "warnings")]
        lines += [f"## `{fn_name}`", "", f"`from statkit.{mod_name} import {fn_name}`", "",
                  "```python", f"{fn_name}{sig}", "```", "", doc, ""]
        if fields:
            lines += [f"Returns `{getattr(ret_cls, '__name__', ret)}` with: " + ", ".join(f"`{f}`" for f in fields), ""]
    return "\n".join(lines)


def check_frontmatter(skill: str, text: str) -> None:
    m = re.match(r"---\nname: ([a-z0-9-]+)\ndescription: (.+?)\n---\n", text, re.S)
    if not m or m.group(1) != skill:
        raise SystemExit(f"{skill}: SKILL.md frontmatter must start with name: {skill}")
    if len(m.group(1)) > 64 or len(m.group(2)) > 1024:
        raise SystemExit(f"{skill}: name must be <= 64 chars and description <= 1024 chars")


def build() -> list[pathlib.Path]:
    if DIST.exists():
        shutil.rmtree(DIST)
    zips = []
    for skill in SKILL_TOOLS:
        src = HERE / skill / "SKILL.md"
        text = src.read_text(encoding="utf-8")
        check_frontmatter(skill, text)
        out = DIST / skill
        (out / "reference").mkdir(parents=True)
        (out / "SKILL.md").write_text(text, encoding="utf-8")
        (out / "reference" / "tools.md").write_text(tool_reference(skill), encoding="utf-8")
        shutil.copytree(LIB, out / "scripts" / "statkit",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        zpath = DIST / f"{skill}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(DIST).as_posix())
        zips.append(zpath)
        print(f"built {zpath.name}: {zpath.stat().st_size // 1024} KB")
    return zips


if __name__ == "__main__":
    build()
