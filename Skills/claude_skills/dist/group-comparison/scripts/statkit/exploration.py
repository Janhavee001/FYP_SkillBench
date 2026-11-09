"""
Category C, remaining tools: frequency tables, group aggregation and the
compact column summary a chart recommender conditions on (cf. LIDA summarize()).
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from .common import ToolResult, is_numeric

_AGGS = {"mean", "median", "sum", "count", "min", "max", "std", "nunique"}


@dataclasses.dataclass
class FrequencyTable(ToolResult):
    column: str = ""
    n: int = 0
    n_missing: int = 0
    table: dict = dataclasses.field(default_factory=dict)  # value -> {count, proportion}


def frequency_table(df: pd.DataFrame, column: str, top_n: int | None = None,
                     include_missing: bool = False) -> FrequencyTable:
    """value_counts with proportions. Proportions are over non-missing rows unless
    include_missing=True, and the denominator is reported, because "what fraction"
    questions silently change answer with the denominator (Titanic age, deck)."""
    if column not in df.columns:
        return FrequencyTable(ok=False, column=column, warnings=[f"Column {column!r} not found."])
    s = df[column]
    n_missing = int(s.isna().sum())
    vc = s.value_counts(dropna=not include_missing)
    total = int(vc.sum())
    if top_n:
        vc = vc.head(top_n)
    warnings = []
    if n_missing:
        warnings.append(f"{n_missing} missing value(s); proportions use denominator {total}"
                        + (" (missing included)." if include_missing else " (missing excluded)."))
    if is_numeric(s) and s.nunique() > 50:
        warnings.append(f"{column!r} has {s.nunique()} distinct numeric values; consider bin_continuous first.")
    table = {str(k): {"count": int(v), "proportion": round(v / total, 4)} for k, v in vc.items()}
    return FrequencyTable(ok=True, warnings=warnings, column=column, n=total,
                           n_missing=n_missing, table=table)


@dataclasses.dataclass
class GroupAggregateReport(ToolResult):
    by: list[str] = dataclasses.field(default_factory=list)
    value: str = ""
    agg: str = ""
    result: dict = dataclasses.field(default_factory=dict)
    group_sizes: dict = dataclasses.field(default_factory=dict)


def groupby_aggregate(df: pd.DataFrame, by: str | list[str], value: str,
                       agg: str = "mean") -> GroupAggregateReport:
    """Aggregate `value` by one or more grouping columns, returning group sizes
    alongside the aggregate so a mean over n=3 is never read like a mean over n=300."""
    by = [by] if isinstance(by, str) else list(by)
    missing = [c for c in by + [value] if c not in df.columns]
    if missing:
        return GroupAggregateReport(ok=False, warnings=[f"Column(s) not found: {missing}"],
                                     by=by, value=value, agg=agg)
    if agg not in _AGGS:
        return GroupAggregateReport(ok=False, warnings=[f"Unknown agg {agg!r}; use one of {sorted(_AGGS)}."],
                                     by=by, value=value, agg=agg)
    if agg in ("mean", "median", "sum", "std") and not is_numeric(df[value]):
        return GroupAggregateReport(ok=False, warnings=[f"{agg} needs a numeric column; {value!r} is {df[value].dtype}."],
                                     by=by, value=value, agg=agg)
    grouped = df.groupby(by, observed=True)[value]
    res, sizes = grouped.agg(agg), grouped.size()
    warnings = []
    small = [str(k) for k, v in sizes.items() if v < 10]
    if small:
        warnings.append(f"Group(s) {small} have fewer than 10 rows; their aggregates are unstable.")
    key = lambda k: str(k)  # noqa: E731 - tuple keys for multi-column groupings
    return GroupAggregateReport(
        ok=True, warnings=warnings, by=by, value=value, agg=agg,
        result={key(k): float(v) for k, v in res.items()},
        group_sizes={key(k): int(v) for k, v in sizes.items()})


def looks_like_dates(s: pd.Series, sample: int = 50) -> bool:
    """True when a text column parses as dates (e.g. Bike Sharing's `dteday`, stored as
    '2011-01-01' strings). Without this, a date column read from CSV is treated as an
    identifier, and time-series questions never get a line chart."""
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) or s.empty:
        return False
    head = s.astype(str).head(sample)
    if not head.str.contains(r"\d{1,4}[-/.]\d{1,2}", regex=True).all():
        return False  # cheap pre-filter: free text like 'male' never reaches the parser
    parsed = pd.to_datetime(head, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.9)


@dataclasses.dataclass
class VizColumnSummary(ToolResult):
    columns: dict = dataclasses.field(default_factory=dict)


def column_summary_for_viz(df: pd.DataFrame, max_samples: int = 3) -> VizColumnSummary:
    """Per-column role (quantitative / temporal / nominal / ordinal-ish / id-like), range and sample
    values -- the compact summary a chart recommender needs and nothing more (cf. LIDA summarize())."""
    cols = {}
    for c in df.columns:
        s = df[c].dropna()
        n_unique = int(s.nunique())
        dtype = str(df[c].dtype)
        if "datetime" in dtype or looks_like_dates(s):
            role = "temporal"
        elif is_numeric(df[c]):
            # A row id is unique *and* consecutive (0..n-1, 1..n); a unique count series is not an id.
            consecutive = "int" in dtype.lower() and len(s) and int(s.max()) - int(s.min()) + 1 == len(s)
            role = "id_like" if n_unique == len(s) and len(s) > 20 and consecutive \
                else ("nominal" if n_unique <= 2 else ("ordinal_like" if n_unique <= 10 else "quantitative"))
        else:
            role = "id_like" if n_unique == len(s) and len(s) > 20 else "nominal"
        info = {"dtype": dtype, "role": role, "n_unique": n_unique,
                "missing_pct": round(float(df[c].isna().mean()) * 100, 2),
                "samples": [str(v) for v in s.drop_duplicates().head(max_samples)]}
        if role in ("quantitative", "ordinal_like") and len(s):
            info["min"], info["max"] = float(s.min()), float(s.max())
        cols[c] = info
    return VizColumnSummary(ok=True, columns=cols)
