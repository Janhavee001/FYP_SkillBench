"""
Category A: the two grounding checks with no equivalent in any of the five
benchmarked papers.

validate_grounding closes the nvAgent failure in DOCUMENTATION.md §6.6: right
chart, right statistical intent, querying a table that does not exist.
check_method_appropriateness closes the general case of "chose a real method
that does not fit this data." It takes the concern StatABench's
judger-open/modeling_groundedness.py raises (an LLM-judge prompt that scores,
after the fact, whether a write-up's model choice and assumptions were
justified) and turns the structural part of it into deterministic rules that
run before execution.
"""
from __future__ import annotations

import dataclasses
import re

import pandas as pd

from .common import ToolResult, is_numeric
from .profiling import DatasetProfile


@dataclasses.dataclass
class GroundingReport(ToolResult):
    referenced: list[str] = dataclasses.field(default_factory=list)
    unknown: list[str] = dataclasses.field(default_factory=list)
    known_tables: list[str] = dataclasses.field(default_factory=list)
    unknown_tables: list[str] = dataclasses.field(default_factory=list)


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def validate_grounding(profile: DatasetProfile | None, referenced_columns: list[str],
                        code: str | None = None,
                        known_tables: list[str] | None = None,
                        schema: dict[str, list[str]] | None = None) -> GroundingReport:
    """
    Check that every column (and, if given, every table name) a plan or a
    piece of generated code references actually exists in the profiled
    schema, before that code is executed.

    This is the direct fix for nvAgent's failure mode in DOCUMENTATION.md
    §6.6: it emitted syntactically valid SQL, chose the right chart, and
    expressed the right statistical intent (a correlation ranking), and
    still produced nothing, because it queried a table called `wine` when
    the actual table was `wine_quality`. A validator scoped to "does this
    reference something that exists" would have caught that in one line,
    which is the whole argument for this tool existing separately from
    execution-feedback repair (§6.5's point: a validator can only fix what
    it is scoped to touch, so make grounding one of the things it is
    scoped to touch).

    Multi-table plans: pass `schema` ({table: [columns]}, e.g. from discover_tables(...).schema)
    and reference columns as "table.column". A dotted reference is checked against that table
    only, so `orders.customer_id` fails if the column lives in `customers` but not `orders`.
    """
    unknown_cols = []
    for ref in referenced_columns:
        if schema is not None and "." in ref:
            table, col = ref.split(".", 1)
            if table not in schema or col not in schema[table]:
                unknown_cols.append(ref)
        elif profile is None or ref not in profile.columns:
            unknown_cols.append(ref)

    known_tables = known_tables or (list(schema) if schema else [])
    unknown_tables: list[str] = []
    if code and known_tables:
        # Heuristic table-reference scan: look for FROM/JOIN <identifier> in
        # SQL-shaped code, or a bare identifier used as a dataframe name.
        for m in re.finditer(r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", code, re.IGNORECASE):
            table = m.group(1)
            if table not in known_tables:
                unknown_tables.append(table)

    warnings = []
    if unknown_cols:
        warnings.append(
            f"References column(s) not present in the loaded data: {unknown_cols}. "
            + (f"Available columns: {profile.columns}." if profile is not None else
               f"Known schema: { {t: list(c) for t, c in (schema or {}).items()} }."))
    if unknown_tables:
        warnings.append(
            f"References table(s) not present in the known schema: "
            f"{sorted(set(unknown_tables))}. Known tables: {known_tables}.")

    return GroundingReport(
        ok=not unknown_cols and not unknown_tables,
        warnings=warnings,
        referenced=list(referenced_columns),
        unknown=unknown_cols,
        known_tables=known_tables,
        unknown_tables=sorted(set(unknown_tables)),
    )


@dataclasses.dataclass
class MethodAppropriatenessReport(ToolResult):
    method: str = ""
    columns: list[str] = dataclasses.field(default_factory=list)
    verdict: str = ""  # "appropriate" | "questionable" | "inappropriate"
    reason: str = ""


# (method, required dtype kind, min/max distinct categories) rules, deliberately
# small and explicit rather than ML-learned, so every verdict is auditable --
# the same design instinct behind the benchmark's own rubric (§4.4: criteria
# taken from the papers' own judges, not invented ad hoc).
#
# Method names map to a rule family. Both the tool names in this library and
# the generic names an LLM plan tends to use are accepted, so the gate fires on
# whatever the planner actually writes. Column conventions per family:
#   two_group / k_group: [group_col, value_col]    linear_model: [target, *predictors]
#   survival: [duration, event]   time_series: [column]   correlation / chi_square: [a, b]
_METHOD_FAMILIES = {
    **dict.fromkeys(["correlation", "pearson_correlation", "correlation_matrix", "linear_correlation"], "correlation"),
    **dict.fromkeys(["t_test", "welch_t_test", "ab_test", "ab_ttest", "two_sample_confidence_interval",
                     "bootstrap_ab_test", "mann_whitney", "wilcoxon", "variance_equality_test"], "two_group"),
    **dict.fromkeys(["anova", "kruskal", "kruskal_wallis", "nonparametric_test"], "k_group"),
    **dict.fromkeys(["chi_square_test", "chi_square_association", "contingency_test", "chi_square"], "chi_square"),
    **dict.fromkeys(["linear_regression", "multivariable_regression", "regularized_regression", "ols"], "linear_model"),
    **dict.fromkeys(["logistic_regression"], "logistic"),
    **dict.fromkeys(["kaplan_meier_curve", "logrank_test", "kaplan_meier", "survival"], "survival"),
    **dict.fromkeys(["stationarity_test", "adf", "kpss"], "time_series"),
}


def check_method_appropriateness(df: pd.DataFrame, method: str,
                                  columns: list[str]) -> MethodAppropriatenessReport:
    """
    Flag a structural mismatch between a declared method and the columns it
    is being applied to -- e.g. Pearson correlation requested on a binary
    column, or a two-sample test requested on a column with more than two
    groups.

    StatABench's `judger-open/modeling_groundedness.py` is an LLM-judge prompt
    that scores, after the fact, whether a write-up justified its model choice.
    This is not a port of it: it is a small set of deterministic rules covering
    the structural part of that concern (does the method fit these column
    types and group counts?), run before execution, so a verdict is auditable.

    Unknown methods return verdict "unchecked" (ok=False), never "appropriate":
    a gate that passes whatever it has no rule for gives false assurance.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return MethodAppropriatenessReport(
            ok=False, warnings=[f"Column(s) not found: {missing}"],
            method=method, columns=columns, verdict="inappropriate",
            reason=f"Cannot evaluate: {missing} not in dataframe.")

    dtypes = {c: str(df[c].dtype) for c in columns}
    nunique = {c: int(df[c].nunique(dropna=True)) for c in columns}
    m = method.lower()

    def numeric(c: str) -> bool:
        return is_numeric(df[c])

    verdict, reason = "appropriate", "No structural mismatch detected."
    family = _METHOD_FAMILIES.get(m)

    if family is None:
        # An unknown method must not pass silently: "no rule fired" is not
        # evidence the method fits, so say so instead of returning "appropriate".
        return MethodAppropriatenessReport(
            ok=False, method=method, columns=columns, verdict="unchecked",
            reason=f"No appropriateness rules for method {method!r}.",
            warnings=[f"No appropriateness rules for method {method!r}; the plan was not checked. "
                      f"Known methods: {sorted(_METHOD_FAMILIES)}."])

    if family == "two_group":
        group_col, value_col = columns[0], (columns[1] if len(columns) > 1 else None)
        if value_col is not None and not numeric(value_col):
            verdict = "inappropriate"
            reason = f"Outcome {value_col!r} is not numeric; compare proportions with a chi-square test instead."
        elif nunique[group_col] > 2:
            verdict = "inappropriate"
            reason = (f"{group_col} has {nunique[group_col]} groups; this test compares exactly two. "
                      "Use Kruskal-Wallis/ANOVA, or pairwise tests with multiple_comparison_correction.")
        elif nunique[group_col] < 2:
            verdict = "inappropriate"
            reason = f"{group_col} has fewer than two groups; there is nothing to compare."

    elif family == "k_group":
        group_col, value_col = columns[0], (columns[1] if len(columns) > 1 else None)
        if value_col is not None and not numeric(value_col):
            verdict = "inappropriate"
            reason = f"Outcome {value_col!r} is not numeric."
        elif nunique[group_col] == 2:
            verdict = "questionable"
            reason = f"{group_col} has exactly two groups; a two-group test (Welch / Mann-Whitney) is the direct choice."

    elif family == "linear_model":
        target, predictors = columns[0], columns[1:]
        if not numeric(target):
            verdict = "inappropriate"
            reason = f"Target {target!r} is not numeric; linear regression needs a continuous outcome."
        elif nunique[target] <= 2:
            verdict = "inappropriate"
            reason = f"Target {target!r} is binary; use logistic regression, not OLS."
        elif nunique[target] <= 10:
            verdict = "questionable"
            reason = (f"Target {target!r} has only {nunique[target]} levels (ordinal-like); OLS treats the gaps "
                      "as equal. Report it, or consider an ordinal model.")
        elif any(nunique[p] > 50 and not numeric(p) for p in predictors):
            verdict = "questionable"
            reason = "A text predictor has more than 50 levels; dummy-coding it will overfit."

    elif family == "survival":
        event = columns[1] if len(columns) > 1 else None
        if not numeric(columns[0]):
            verdict = "inappropriate"
            reason = f"Duration {columns[0]!r} is not numeric."
        elif event is not None and not set(df[event].dropna().unique()) <= {0, 1}:
            verdict = "inappropriate"
            reason = f"Event column {event!r} must be 0/1 (1 = event observed)."

    elif family == "time_series":
        if not numeric(columns[0]):
            verdict = "inappropriate"
            reason = f"{columns[0]!r} is not numeric."
        elif df[columns[0]].notna().sum() < 20:
            verdict = "inappropriate"
            reason = "Fewer than 20 observations; stationarity tests are unreliable this short."

    elif family == "correlation" and len(columns) == 2:
        non_numeric = [c for c in columns if not numeric(c)]
        binary = [c for c in columns if numeric(c) and nunique[c] <= 2]
        if non_numeric:
            verdict = "inappropriate"
            reason = f"Correlation requires numeric columns; {non_numeric} is not numeric."
        elif binary:
            verdict = "questionable"
            reason = (f"{binary} has only {nunique[binary[0]]} distinct value(s). "
                       "A point-biserial interpretation is defensible, but a plain "
                       "Pearson r on a near-constant or binary column is easy to "
                       "over-interpret; consider reporting group means instead.")

    elif family == "chi_square" and len(columns) == 2:
        numeric_high_card = [c for c in columns
                             if numeric(c) and nunique[c] > 12]
        if numeric_high_card:
            verdict = "questionable"
            reason = (f"{numeric_high_card} looks continuous ({nunique[numeric_high_card[0]]} "
                      "distinct values). Chi-square association tests are for "
                      "categorical variables; binning first or using correlation "
                      "may be more appropriate.")

    elif family == "logistic" and len(columns) >= 1:
        target = columns[-1]
        if numeric(target) and nunique.get(target, 0) > 2:
            verdict = "inappropriate"
            reason = (f"Target {target} has {nunique[target]} distinct numeric values; "
                      "logistic regression expects a binary (or, for multinomial/"
                      "ordinal variants, explicitly categorical) target.")

    return MethodAppropriatenessReport(
        ok=verdict == "appropriate",
        warnings=[] if verdict == "appropriate" else [reason],
        method=method, columns=columns, verdict=verdict, reason=reason,
    )
