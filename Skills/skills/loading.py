"""
Category B: dataset loading. Kept separate from profiling.py because it
*composes* detect_separator_and_encoding rather than duplicating it.
"""
from __future__ import annotations

import dataclasses
import pathlib

import pandas as pd

from .common import ToolResult, load_dataframe
from .profiling import detect_separator_and_encoding


@dataclasses.dataclass
class LoadReport(ToolResult):
    path: str = ""
    n_rows: int = 0
    n_cols: int = 0
    separator: str = ","
    encoding: str = "utf-8"
    df: pd.DataFrame | None = None


def load_dataset(path: str | pathlib.Path, expected_min_columns: int = 2) -> LoadReport:
    """
    Load CSV / TSV / Parquet / Excel, with the delimiter and encoding sniffed
    first (detect_separator_and_encoding) instead of recalled from memory.
    The frame comes back inside the result envelope so grounding warnings
    travel with the data rather than being a separate step a caller can skip.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return LoadReport(ok=False, warnings=[f"File not found: {path}"], path=str(path))
    try:
        return _load(path, expected_min_columns)
    except (ValueError, OSError, ImportError, pd.errors.ParserError) as e:
        # A tool reports failure in its envelope; it does not crash the agent's loop.
        return LoadReport(ok=False, warnings=[f"Could not read {path.name}: {type(e).__name__}: {e}"], path=str(path))


def _load(path: pathlib.Path, expected_min_columns: int) -> LoadReport:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        return LoadReport(ok=True, path=str(path), n_rows=len(df), n_cols=len(df.columns), df=df)
    if suffix in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path)
        return LoadReport(ok=True, path=str(path), n_rows=len(df), n_cols=len(df.columns), df=df)

    sep_report = detect_separator_and_encoding(path, expected_min_columns=expected_min_columns)
    if not sep_report.ok:
        return LoadReport(ok=False, warnings=sep_report.warnings, path=str(path),
                          separator=sep_report.detected_separator,
                          encoding=sep_report.detected_encoding)
    df = load_dataframe(path, sep=sep_report.detected_separator,
                        encoding=sep_report.detected_encoding)
    return LoadReport(ok=True, warnings=sep_report.warnings, path=str(path),
                      n_rows=len(df), n_cols=len(df.columns),
                      separator=sep_report.detected_separator,
                      encoding=sep_report.detected_encoding, df=df)
