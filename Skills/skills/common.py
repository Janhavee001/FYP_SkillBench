"""
Shared helpers for the skill library: dataframe loading and the canonical
method-name vocabulary, reused from the benchmark harness rather than
reinvented, so a tool's `declared_method` output is directly comparable to
Benchmark/benchmark-fyp/harness/common.py's METHOD_PATTERNS vocabulary.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCHMARK_DATASETS = ROOT.parent / "Benchmark" / "benchmark-fyp" / "datasets"


def _force_utf8_console() -> None:
    """Same fix as Benchmark/benchmark-fyp/harness/common.py bug #1: Windows
    consoles default to cp1252, and print()-ing a section mark or an
    en-dash raises UnicodeEncodeError otherwise."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_force_utf8_console()


def is_numeric(series: pd.Series) -> bool:
    """One numeric test for every tool. A hard-coded {"int64", "float64", ...}
    set misses int8/uint8, nullable Int64 and float16, so a column could be
    numeric to one tool and categorical to the next. bool is excluded on
    purpose: True/False is a category, not a measurement."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def load_dataframe(path: str | pathlib.Path, **read_csv_kwargs) -> pd.DataFrame:
    """Thin wrapper so every tool loads data the same way and the same kwargs
    (sep, encoding) that detect_separator_and_encoding recommends can be
    passed straight through."""
    return pd.read_csv(path, **read_csv_kwargs)


@dataclasses.dataclass
class ToolResult:
    """Common envelope every skill returns.

    `warnings` is not an afterthought: it is how a grounding/validation tool
    communicates "ran fine, but here is what you should not trust" without
    forcing the caller to inspect internals. An agent composing tools should
    treat a non-empty `warnings` list as something to surface, not swallow.
    """
    ok: bool
    warnings: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
