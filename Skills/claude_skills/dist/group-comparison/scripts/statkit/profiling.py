"""
Category A/B: grounding-oriented profiling.

These three tools exist to catch, before any analysis runs, the two failure
classes DOCUMENTATION.md documents as the ones a repair loop does not catch:
a wrong separator collapsing every column into one (§6.4), and an agent
never actually seeing the shape of what it loaded (§7.1 bug #11, DataSciBench
analysing invented data because inputs were never staged).
"""
from __future__ import annotations

import csv
import dataclasses
import pathlib

import pandas as pd

from .common import ToolResult, load_dataframe


@dataclasses.dataclass
class SeparatorReport(ToolResult):
    path: str = ""
    detected_separator: str = ","
    detected_encoding: str = "utf-8"
    confidence: float = 0.0
    columns_at_detected_sep: int = 0
    columns_at_comma: int = 0


def detect_separator_and_encoding(path: str | pathlib.Path,
                                   expected_min_columns: int = 2) -> SeparatorReport:
    """
    Sniff the real delimiter of a CSV before anyone calls pd.read_csv on it.

    Directly targets the bug in DOCUMENTATION.md §6.4: the model "remembered"
    that the UCI wine dataset is semicolon-separated and wrote
    `pd.read_csv(path, sep=';')` against a file that was actually
    comma-separated. That parses without error -- it just returns one column
    -- so nothing downstream raises until a much later, more confusing
    `KeyError: 'quality'`. Sniffing the file's *actual* delimiter up front,
    rather than trusting what the model recalls about a similarly-named
    public dataset, is the fix the benchmark's own repair loop never found
    (0 of 3 repairs fixed this class of error).
    """
    path = pathlib.Path(path)
    warnings: list[str] = []

    raw = path.read_bytes()
    encoding = "utf-8"
    try:
        sample = raw[:65536].decode("utf-8")
    except UnicodeDecodeError:
        encoding = "cp1252"
        sample = raw[:65536].decode("cp1252", errors="replace")
        warnings.append(
            "File is not valid UTF-8; falling back to cp1252. Confirm this "
            "is the intended encoding before analysing text columns.")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
        warnings.append(
            "csv.Sniffer could not confidently detect a delimiter; "
            "defaulting to ',' -- verify column count below before trusting this.")

    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    cols_at_detected = len(next(csv.reader([first_line], delimiter=sep)))
    cols_at_comma = len(next(csv.reader([first_line], delimiter=",")))

    confidence = 1.0
    if cols_at_detected < expected_min_columns:
        confidence = 0.0
        warnings.append(
            f"Detected separator {sep!r} yields only {cols_at_detected} "
            f"column(s), below the expected minimum of {expected_min_columns}. "
            "The file is almost certainly being mis-parsed.")
    elif sep != "," and cols_at_comma > cols_at_detected:
        # The sniffed separator produced fewer columns than a plain comma
        # split would -- exactly the wine_q1-class mistake, just caught
        # before execution instead of after.
        confidence = 0.3
        warnings.append(
            f"Sniffed separator {sep!r} gives {cols_at_detected} columns, but "
            f"splitting on ',' gives {cols_at_comma}. Prefer the comma "
            "parse unless you have independent evidence this file uses "
            f"{sep!r}.")
        sep = ","
        cols_at_detected = cols_at_comma

    return SeparatorReport(
        ok=confidence > 0,
        warnings=warnings,
        path=str(path),
        detected_separator=sep,
        detected_encoding=encoding,
        confidence=confidence,
        columns_at_detected_sep=cols_at_detected,
        columns_at_comma=cols_at_comma,
    )


@dataclasses.dataclass
class DatasetProfile(ToolResult):
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = dataclasses.field(default_factory=list)
    dtypes: dict = dataclasses.field(default_factory=dict)
    n_unique: dict = dataclasses.field(default_factory=dict)
    missing_pct: dict = dataclasses.field(default_factory=dict)


def profile_dataset(df: pd.DataFrame, expected_columns: int | None = None) -> DatasetProfile:
    """
    Schema fingerprint an agent should inspect before writing any code
    against a dataframe.

    Generalizes StatABench's `check_column_type_is` (which checks one column
    against one expected type) into a whole-frame profile, and is the
    concrete answer to DataSciBench's bug #11 (§7.1): if the agent had been
    made to print `profile_dataset(df).n_cols` before analysing, "1 column
    where 12 were expected" would have been visible immediately instead of
    surfacing as a plausible-looking analysis of dummy data.
    """
    warnings: list[str] = []
    if expected_columns is not None and len(df.columns) != expected_columns:
        warnings.append(
            f"Loaded {len(df.columns)} column(s), expected {expected_columns}. "
            "Check the separator before proceeding -- see "
            "detect_separator_and_encoding().")
    if len(df.columns) == 1:
        warnings.append(
            "Only one column loaded. This is the signature of a wrong "
            "delimiter, not necessarily a genuinely single-column dataset.")

    return DatasetProfile(
        ok=not warnings,
        warnings=warnings,
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        n_unique={c: int(df[c].nunique(dropna=True)) for c in df.columns},
        missing_pct={c: round(float(df[c].isna().mean()) * 100, 2) for c in df.columns},
    )


@dataclasses.dataclass
class MissingValueReport(ToolResult):
    per_column: dict = dataclasses.field(default_factory=dict)
    structurally_missing: list[str] = dataclasses.field(default_factory=list)
    sparse: list[str] = dataclasses.field(default_factory=list)


def missing_value_report(df: pd.DataFrame, columns: list[str] | None = None,
                          structural_threshold: float = 0.5) -> MissingValueReport:
    """
    Per-column missing-value breakdown, distinguishing columns that are
    mostly absent (>50% missing, e.g. Titanic's `deck` at 77%) from columns
    with ordinary sparse missingness (e.g. `age` at 20%), since those call
    for different handling and conflating them is a real failure mode in the
    shared test set's missing-data questions.

    Inspired by StatABench's `check_missing_values`, extended with the
    structural/sparse split so the *report* -- not just the raw percentage --
    is what an agent conditions its next step on.
    """
    cols = columns or list(df.columns)
    per_column = {}
    structural, sparse = [], []
    for c in cols:
        pct = float(df[c].isna().mean())
        per_column[c] = {"missing_count": int(df[c].isna().sum()),
                          "missing_pct": round(pct * 100, 2)}
        if pct >= structural_threshold:
            structural.append(c)
        elif pct > 0:
            sparse.append(c)

    return MissingValueReport(
        ok=True,
        per_column=per_column,
        structurally_missing=structural,
        sparse=sparse,
    )
