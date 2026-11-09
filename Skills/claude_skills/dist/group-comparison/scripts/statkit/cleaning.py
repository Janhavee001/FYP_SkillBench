"""
Category B: cleaning and feature preparation. Each tool returns the
transformed frame inside its result alongside a record of what it did, so an
agent can narrate the change and a validator can check it. None mutates its
input.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .common import ToolResult, is_numeric



@dataclasses.dataclass
class TransformResult(ToolResult):
    df: pd.DataFrame | None = None
    details: dict = dataclasses.field(default_factory=dict)


def _numeric_cols(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    return columns or [c for c in df.columns if is_numeric(df[c])]


def _missing_cols(df: pd.DataFrame, columns: list[str]) -> TransformResult | None:
    bad = [c for c in columns if c not in df.columns]
    return TransformResult(ok=False, warnings=[f"Column(s) not found: {bad}"]) if bad else None


def impute_missing(df: pd.DataFrame, columns: list[str] | None = None,
                    strategy: str = "median") -> TransformResult:
    """mean / median / most_frequent / constant:<value> imputation (cf. MetaGPT FillMissingValue)."""
    cols = columns or [c for c in df.columns if df[c].isna().any()]
    if (err := _missing_cols(df, cols)):
        return err
    out, filled, warnings = df.copy(), {}, []
    for c in cols:
        n = int(out[c].isna().sum())
        if not n:
            continue
        numeric = is_numeric(out[c])
        if strategy.startswith("constant:"):
            raw = strategy.split(":", 1)[1]
            try:
                value = float(raw) if numeric else raw
            except ValueError:
                return TransformResult(ok=False, warnings=[f"{c!r} is numeric but constant {raw!r} is not a number."])
        elif strategy in ("mean", "median") and numeric:
            value = out[c].mean() if strategy == "mean" else out[c].median()
        elif strategy in ("mean", "median", "most_frequent"):
            if strategy != "most_frequent":
                warnings.append(f"{c!r} is not numeric; used most_frequent instead of {strategy}.")
            mode = out[c].mode()
            if mode.empty:
                warnings.append(f"{c!r} is entirely missing; left unfilled.")
                continue
            value = mode.iloc[0]
        else:
            return TransformResult(ok=False, warnings=[f"Unknown strategy {strategy!r}."])
        out[c] = out[c].fillna(value)
        filled[c] = {"n_filled": n, "value": value.item() if hasattr(value, "item") else value}
        if n / len(df) > 0.5:
            warnings.append(f"{c!r} was {n / len(df):.0%} missing; imputing it mostly invents data.")
    return TransformResult(ok=True, warnings=warnings, df=out, details=filled)


def scale_numeric(df: pd.DataFrame, columns: list[str] | None = None,
                   method: str = "standard") -> TransformResult:
    """minmax / standard / robust / maxabs scaling (cf. MetaGPT *Scale transformers)."""
    cols = _numeric_cols(df, columns)
    if (err := _missing_cols(df, cols)):
        return err
    out, params, warnings = df.copy(), {}, []
    for c in cols:
        s = out[c].astype(float)
        if method == "minmax":
            shift, scale = s.min(), s.max() - s.min()
        elif method == "standard":
            shift, scale = s.mean(), s.std(ddof=0)
        elif method == "robust":
            shift, scale = s.median(), s.quantile(0.75) - s.quantile(0.25)
        elif method == "maxabs":
            shift, scale = 0.0, s.abs().max()
        else:
            return TransformResult(ok=False, warnings=[f"Unknown method {method!r}."])
        if not scale or np.isnan(scale):
            warnings.append(f"{c!r} has zero spread under {method}; left unscaled.")
            continue
        out[c] = (s - shift) / scale
        params[c] = {"shift": float(shift), "scale": float(scale)}
    return TransformResult(ok=True, warnings=warnings, df=out, details=params)


def encode_categorical(df: pd.DataFrame, columns: list[str], method: str = "onehot",
                        target: str | None = None, order: dict | None = None,
                        n_splits: int = 5) -> TransformResult:
    """ordinal / onehot / label / target_mean / kfold_target_mean (cf. MetaGPT *Encode*, TargetMeanEncoder)."""
    if (err := _missing_cols(df, columns + ([target] if target else []))):
        return err
    if method in ("target_mean", "kfold_target_mean") and target is None:
        return TransformResult(ok=False, warnings=["target_mean encoding needs `target`."])
    out, details, warnings = df.copy(), {}, []
    for c in columns:
        if method == "onehot":
            dummies = pd.get_dummies(out[c], prefix=c, dtype=int)
            out = pd.concat([out.drop(columns=c), dummies], axis=1)
            details[c] = list(dummies.columns)
            if dummies.shape[1] > 20:
                warnings.append(f"{c!r} expanded to {dummies.shape[1]} columns (high cardinality).")
        elif method in ("label", "ordinal"):
            explicit = (order or {}).get(c)
            cats = explicit or sorted(out[c].dropna().unique().tolist(), key=str)
            if method == "ordinal" and not explicit:
                warnings.append(f"{c!r}: no explicit order given; used alphabetical, which may be meaningless.")
            mapping = {v: i for i, v in enumerate(cats)}
            before_na = int(out[c].isna().sum())
            out[c] = out[c].map(mapping)
            new_na = int(out[c].isna().sum()) - before_na
            if new_na:
                warnings.append(f"{c!r}: {new_na} value(s) not in the given order became NaN.")
            details[c] = {str(k): v for k, v in mapping.items()}
        elif method == "target_mean":
            means = out.groupby(c)[target].mean()
            out[c] = out[c].map(means)
            details[c] = {str(k): float(v) for k, v in means.items()}
            warnings.append(f"{c!r}: plain target-mean encoding leaks the target; "
                            "prefer kfold_target_mean for modelling.")
        elif method == "kfold_target_mean":
            idx = np.arange(len(out))
            folds = np.array_split(np.random.default_rng(0).permutation(idx), n_splits)
            enc = np.full(len(out), np.nan)
            global_mean = out[target].mean()
            for f in folds:
                train = out.iloc[np.setdiff1d(idx, f)]
                means = train.groupby(c)[target].mean()
                enc[f] = out.iloc[f][c].map(means).fillna(global_mean).to_numpy()
            out[c] = enc
            details[c] = {"n_splits": n_splits}
        else:
            return TransformResult(ok=False, warnings=[f"Unknown method {method!r}."])
    return TransformResult(ok=True, warnings=warnings, df=out, details=details)


@dataclasses.dataclass
class OutlierReport(ToolResult):
    column: str = ""
    q1: float = 0.0
    q3: float = 0.0
    lower_fence: float = 0.0
    upper_fence: float = 0.0
    n_outliers: int = 0
    pct_outliers: float = 0.0
    outlier_index: list = dataclasses.field(default_factory=list)


def detect_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> OutlierReport:
    """Tukey fences (Q1 - k*IQR, Q3 + k*IQR). Reports the rows; never silently drops them."""
    if column not in df.columns or not is_numeric(df[column]):
        return OutlierReport(ok=False, column=column,
                              warnings=[f"{column!r} missing or not numeric."])
    s = df[column].dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (s < lo) | (s > hi)
    warnings = []
    if mask.mean() > 0.10:
        warnings.append(f"{mask.mean():.0%} of values flagged; the distribution may be skewed, not contaminated.")
    return OutlierReport(ok=True, warnings=warnings, column=column, q1=q1, q3=q3,
                          lower_fence=lo, upper_fence=hi, n_outliers=int(mask.sum()),
                          pct_outliers=round(float(mask.mean()) * 100, 2),
                          outlier_index=list(s.index[mask][:100]))


def bin_continuous(df: pd.DataFrame, column: str, bins: int | list[float] = 5,
                    strategy: str = "quantile", new_column: str | None = None) -> TransformResult:
    """Discretize a numeric column into equal-width or quantile bins (cf. MetaGPT SplitBins)."""
    if (err := _missing_cols(df, [column])):
        return err
    out, warnings = df.copy(), []
    name = new_column or f"{column}_bin"
    try:
        if isinstance(bins, list) or strategy == "width":
            out[name] = pd.cut(out[column], bins=bins, include_lowest=True)
        else:
            out[name] = pd.qcut(out[column], q=bins, duplicates="drop")
    except ValueError as e:
        return TransformResult(ok=False, warnings=[f"Could not bin {column!r}: {e}"])
    if not isinstance(bins, list) and out[name].nunique() < bins:
        warnings.append(f"Only {out[name].nunique()} bins produced (requested {bins}); duplicate edges were merged.")
    return TransformResult(ok=True, warnings=warnings, df=out,
                           details={"new_column": name,
                                    "counts": out[name].astype(str).value_counts().to_dict()})


def engineer_categorical_interactions(df: pd.DataFrame, method: str, columns: list[str],
                                       agg_col: str | None = None,
                                       agg_funcs: list[str] | None = None,
                                       max_categories: int = 100) -> TransformResult:
    """
    New features from categorical columns, added beside the originals (cf. MetaGPT CatCount,
    CatCross, GroupStat):

    - `count`: how often each value of columns[0] occurs (`<col>_cnt`).
    - `cross`: every pair of the given columns combined into one categorical (`<a>_<b>`); pairs
      whose cross would exceed `max_categories` levels are skipped and reported (MetaGPT applies
      the same cap).
    - `group_stat`: per-group aggregates of `agg_col` by columns[0] (`<agg_col>_<func>_by_<col>`).

    Group statistics computed on the full data leak the outcome when `agg_col` is the target;
    that case is refused.
    """
    if (err := _missing_cols(df, columns + ([agg_col] if agg_col else []))):
        return err
    out, details, warnings = df.copy(), {}, []
    if method == "count":
        c = columns[0]
        out[f"{c}_cnt"] = out[c].map(out[c].value_counts())
        details["new_columns"] = [f"{c}_cnt"]
    elif method == "cross":
        if len(columns) < 2:
            return TransformResult(ok=False, warnings=["cross needs at least two columns."])
        made = []
        for i, a in enumerate(columns):
            for b in columns[i + 1:]:
                levels = out[a].nunique() * out[b].nunique()
                if levels > max_categories:
                    warnings.append(f"Skipped {a} x {b}: up to {levels} levels exceeds max_categories={max_categories}.")
                    continue
                name = f"{a}_{b}"
                out[name] = out[a].astype(str) + "_" + out[b].astype(str)
                made.append(name)
        details["new_columns"] = made
    elif method == "group_stat":
        if agg_col is None or not is_numeric(out[agg_col]):
            return TransformResult(ok=False, warnings=["group_stat needs a numeric agg_col."])
        funcs = agg_funcs or ["mean"]
        c = columns[0]
        stats_ = out.groupby(c, observed=True)[agg_col].agg(funcs)
        made = []
        for f in funcs:
            name = f"{agg_col}_{f}_by_{c}"
            out[name] = out[c].map(stats_[f])
            made.append(name)
        details["new_columns"] = made
        warnings.append(f"If {agg_col!r} is the outcome you will model, these features leak it; "
                        "compute them on training folds only.")
    else:
        return TransformResult(ok=False, warnings=[f"Unknown method {method!r}; use count, cross or group_stat."])
    return TransformResult(ok=True, warnings=warnings, df=out, details=details)


def polynomial_expand(df: pd.DataFrame, columns: list[str], degree: int = 2,
                       interaction_only: bool = False) -> TransformResult:
    """
    Add polynomial and interaction terms of numeric columns (cf. MetaGPT PolynomialExpansion,
    which keeps only the 10 columns most correlated with the label; here the caller chooses the
    columns explicitly, so nothing is selected behind their back). Rows with missing values get
    NaN features rather than being imputed with 0 as MetaGPT does.
    """
    from sklearn.preprocessing import PolynomialFeatures

    if (err := _missing_cols(df, columns)):
        return err
    bad = [c for c in columns if not is_numeric(df[c])]
    if bad:
        return TransformResult(ok=False, warnings=[f"Non-numeric column(s): {bad}"])
    if not 2 <= degree <= 4:
        return TransformResult(ok=False, warnings=["degree must be between 2 and 4."])
    pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    complete = df[columns].dropna()
    feats = pf.fit_transform(complete.to_numpy(float))
    names = [n.replace(" ", "*") for n in pf.get_feature_names_out(columns)]
    new = [n for n in names if n not in columns]
    out = df.copy()
    block = pd.DataFrame(feats, columns=names, index=complete.index)[new]
    out = out.join(block)
    warnings = []
    if len(new) > 50:
        warnings.append(f"{len(new)} new columns; consider interaction_only=True or fewer inputs.")
    if len(complete) < len(df):
        warnings.append(f"{len(df) - len(complete)} row(s) with missing inputs get NaN features.")
    warnings.append("Polynomial terms are highly correlated with their inputs; centre or scale before regression.")
    return TransformResult(ok=True, warnings=warnings, df=out, details={"new_columns": new})
