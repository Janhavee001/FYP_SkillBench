"""
Category C: descriptive/exploratory statistics.

Both tools generalize a single StatABench function into something closer to
what a real question needs -- describe_column handles both numeric and
categorical columns instead of requiring the caller to already know the
dtype, and correlation_matrix reports the *strongest* pair directly, since
"which two variables are most related" is how the shared test set's
correlation questions are actually phrased (DOCUMENTATION.md §4.2).
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from .common import ToolResult, is_numeric



@dataclasses.dataclass
class ColumnDescription(ToolResult):
    column: str = ""
    dtype: str = ""
    is_numeric: bool = False
    n: int = 0
    n_missing: int = 0
    stats: dict = dataclasses.field(default_factory=dict)


def describe_column(df: pd.DataFrame, column: str) -> ColumnDescription:
    """
    Numeric columns get mean/median/std/min/max/skew/kurtosis; categorical
    columns get mode, top-5 value counts, and cardinality. StatABench's
    `calculate_statistic` returns one named statistic and leaves the caller to
    know the column is numeric (a `mean` on a text column just raises); this
    tool dispatches on dtype itself and returns a self-describing summary, so
    an interpretation step doesn't need to re-inspect the dataframe.
    """
    if column not in df.columns:
        return ColumnDescription(ok=False, warnings=[f"Column {column!r} not found."],
                                  column=column)

    series = df[column]
    numeric = is_numeric(series)
    n_missing = int(series.isna().sum())
    warnings = []
    if n_missing:
        warnings.append(f"{n_missing} missing value(s) excluded from statistics.")

    clean = series.dropna()
    if numeric:
        stats = {
            "mean": float(clean.mean()) if len(clean) else None,
            "median": float(clean.median()) if len(clean) else None,
            "std": float(clean.std()) if len(clean) > 1 else None,
            "min": float(clean.min()) if len(clean) else None,
            "max": float(clean.max()) if len(clean) else None,
            "skew": float(clean.skew()) if len(clean) > 2 else None,
            "kurtosis": float(clean.kurtosis()) if len(clean) > 3 else None,
        }
    else:
        vc = clean.value_counts().head(5)
        stats = {
            "mode": clean.mode().iloc[0] if not clean.mode().empty else None,
            "n_unique": int(clean.nunique()),
            "top_5_value_counts": {str(k): int(v) for k, v in vc.items()},
        }

    return ColumnDescription(
        ok=True, warnings=warnings, column=column, dtype=str(series.dtype),
        is_numeric=numeric, n=len(clean), n_missing=n_missing, stats=stats,
    )


@dataclasses.dataclass
class CorrelationReport(ToolResult):
    method: str = "pearson"
    columns: list[str] = dataclasses.field(default_factory=list)
    matrix: dict = dataclasses.field(default_factory=dict)
    strongest_pair: tuple | None = None
    strongest_r: float | None = None


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None,
                        method: str = "pearson") -> CorrelationReport:
    """
    Pairwise correlation over the given (or all-numeric) columns, plus the
    single strongest off-diagonal pair called out explicitly.

    Generalizes StatABench's `correlation_analysis`, which computes one pair
    at a time; the shared test set's wine-quality questions ask "which
    variable correlates most with quality," which needs the matrix reduced
    to its strongest entry, not just the raw matrix.
    """
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    non_numeric = [c for c in cols if not is_numeric(df[c])]
    warnings = []
    if non_numeric:
        warnings.append(f"Dropping non-numeric column(s) from correlation: {non_numeric}")
        cols = [c for c in cols if c not in non_numeric]

    if len(cols) < 2:
        return CorrelationReport(ok=False, warnings=warnings + ["Fewer than 2 numeric columns."],
                                  method=method, columns=cols)

    corr = df[cols].corr(method=method)
    matrix = {r: {c: (None if pd.isna(corr.loc[r, c]) else round(float(corr.loc[r, c]), 4))
                  for c in cols} for r in cols}

    best_pair, best_r = None, 0.0
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if abs(r) > abs(best_r) or best_pair is None:
                best_pair, best_r = (a, b), float(r)

    return CorrelationReport(
        ok=True, warnings=warnings, method=method, columns=cols, matrix=matrix,
        strongest_pair=best_pair, strongest_r=best_r,
    )
