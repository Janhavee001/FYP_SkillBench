"""
Category I: visualization, deliberately the thinnest category (DOCUMENTATION §6.3:
the published systems are already comparatively strong here). Two choices matter:

* recommend_chart is rule-based over column *roles*, not a fixed grammar like
  nvAgent's VQL, so a question outside a grammar's ceiling still gets a chart.
* explain_chart explains the *finding* (computed from the data), not the chart's
  design -- LIDA's explain() describes palette and axes, which §6.2 scored ~0.

Chart types cover the project brief's list: histogram, bar, line, scatter, box,
correlation heatmap (plus a count heatmap for two categorical variables).
"""
from __future__ import annotations

import dataclasses
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import ToolResult, is_numeric  # noqa: E402
from .exploration import column_summary_for_viz, looks_like_dates  # noqa: E402

_QUANT = {"quantitative", "ordinal_like"}
_BLUE = "#4C78A8"


@dataclasses.dataclass
class ChartSpec(ToolResult):
    chart_type: str = ""
    x: str | None = None
    y: str | None = None
    color: str | None = None
    columns: list[str] = dataclasses.field(default_factory=list)  # correlation_heatmap only
    aggregate: str | None = None
    rationale: str = ""


def recommend_chart(df: pd.DataFrame, x: str | None = None, y: str | None = None, color: str | None = None,
                    columns: list[str] | None = None) -> ChartSpec:
    """Pick a chart type from the roles of the requested columns (cf. LIDA goal()+visualize(), rule-based).
    Pass `columns` (3+ numeric columns) instead of x/y to get a correlation heatmap."""
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return ChartSpec(ok=False, warnings=[f"Column(s) not found: {missing}"])
        numeric = [c for c in columns if is_numeric(df[c])]
        warnings = [f"Dropped non-numeric column(s) {sorted(set(columns) - set(numeric))}."] if len(numeric) < len(columns) else []
        if len(numeric) < 3:
            return ChartSpec(ok=False, warnings=warnings + ["A correlation heatmap needs 3+ numeric columns; use x/y for a pair."])
        return ChartSpec(ok=True, warnings=warnings, chart_type="correlation_heatmap", columns=numeric,
                         rationale="Several numeric variables: show every pairwise correlation at once.")
    if x is None:
        return ChartSpec(ok=False, warnings=["Give x (and optionally y/color), or columns for a heatmap."])
    cols = [c for c in (x, y, color) if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return ChartSpec(ok=False, warnings=[f"Column(s) not found: {missing}"])
    roles = {c: i["role"] for c, i in column_summary_for_viz(df[cols]).columns.items()}
    rx, ry = roles[x], (roles[y] if y else None)
    warnings = []
    if "id_like" in (rx, ry):
        warnings.append("An identifier-like column was requested for an axis; the chart is unlikely to be informative.")
    if color and df[color].nunique() > 10:
        warnings.append(f"color={color!r} has {df[color].nunique()} levels; more than ~10 colours cannot be told apart.")
    if y is None:
        if rx in _QUANT:
            spec = ("histogram", None, "One numeric variable: show its distribution.")
        else:
            spec = ("bar", "count", "One categorical variable: show category counts.")
    elif rx == "temporal" and ry in _QUANT:
        spec = ("line", None, "Numeric measure over time.")
    elif rx in _QUANT and ry in _QUANT:
        spec = ("scatter", None, "Two numeric variables: show their relationship point by point.")
        if len(df) > 5000:
            warnings.append("Over 5,000 points will overplot; consider a hexbin or sampling.")
    elif rx in ("nominal", "id_like") and ry in _QUANT:
        n_levels = int(df[x].nunique())
        if n_levels > 15:
            warnings.append(f"{x!r} has {n_levels} categories; consider showing the top N only.")
        spec = ("box", None, "Compare the distribution of a numeric variable across categories.") \
            if len(df) / max(n_levels, 1) >= 20 else ("bar", "mean", "Few rows per category: compare means.")
    elif rx in ("nominal", "ordinal_like") and ry in ("nominal", "ordinal_like"):
        spec = ("heatmap", "count", "Two categorical variables: show co-occurrence counts.")
    else:
        spec = ("scatter", None, "Fallback for an unrecognized role combination.")
        warnings.append(f"No rule for roles ({rx}, {ry}); defaulted to scatter.")
    return ChartSpec(ok=True, warnings=warnings, chart_type=spec[0], x=x, y=y, color=color,
                     aggregate=spec[1], rationale=spec[2])


def _needed(spec: ChartSpec) -> list[str]:
    return spec.columns if spec.chart_type == "correlation_heatmap" else [c for c in (spec.x, spec.y, spec.color) if c]


def _as_time(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed") if looks_like_dates(s) else s


@dataclasses.dataclass
class RenderReport(ToolResult):
    path: str = ""
    chart_type: str = ""
    n_points: int = 0


def render_chart(df: pd.DataFrame, spec: ChartSpec, path: str | pathlib.Path,
                  title: str | None = None) -> RenderReport:
    """Draw a ChartSpec to a PNG with matplotlib (cf. LIDA execute()). Grounding is re-checked at draw time,
    so a spec built against one frame cannot silently be drawn against another."""
    if not spec.ok:
        return RenderReport(ok=False, warnings=["Refusing to render an invalid ChartSpec."])
    needed = _needed(spec)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return RenderReport(ok=False, warnings=[f"Spec references column(s) not in this dataframe: {missing}"])
    t = spec.chart_type
    fig, ax = plt.subplots(figsize=(7, 4.5) if t != "correlation_heatmap" else (7.5, 6))
    try:
        if t == "histogram":
            ax.hist(df[spec.x].dropna(), bins=30, color=_BLUE)
            ax.set_xlabel(spec.x); ax.set_ylabel("count")
        elif t in ("scatter", "line"):
            sub = df[needed].dropna().copy()
            if t == "line":
                sub[spec.x] = _as_time(sub[spec.x])
                sub = sub.sort_values(spec.x)
            groups = sub.groupby(spec.color, observed=True) if spec.color else [(None, sub)]
            for key, g in groups:
                if t == "scatter":
                    ax.scatter(g[spec.x], g[spec.y], s=12, alpha=0.5, label=key, color=None if spec.color else _BLUE)
                else:
                    ax.plot(g[spec.x], g[spec.y], label=key, color=None if spec.color else _BLUE)
            if spec.color:
                ax.legend(title=spec.color, frameon=False)
            ax.set_xlabel(spec.x); ax.set_ylabel(spec.y)
        elif t == "box":
            # Labels and data come from the same groupby, so they cannot drift apart
            # (sorting labels as strings would put "10" before "2").
            grouped = [(k, g[spec.y].dropna().to_numpy()) for k, g in df.groupby(spec.x, observed=True)]
            ax.boxplot([v for _, v in grouped], tick_labels=[str(k) for k, _ in grouped])
            ax.set_xlabel(spec.x); ax.set_ylabel(spec.y)
        elif t == "bar":
            s = df.groupby(spec.x, observed=True)[spec.y].mean() if spec.y and spec.aggregate == "mean" \
                else df[spec.x].value_counts()
            ax.bar([str(i) for i in s.index], s.values, color=_BLUE)
            ax.set_xlabel(spec.x); ax.set_ylabel(f"mean {spec.y}" if spec.y else "count")
        elif t == "heatmap":
            ct = pd.crosstab(df[spec.x], df[spec.y])
            im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
            ax.set_xticks(range(ct.shape[1]), [str(c) for c in ct.columns])
            ax.set_yticks(range(ct.shape[0]), [str(i) for i in ct.index])
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(spec.y); ax.set_ylabel(spec.x)
        elif t == "correlation_heatmap":
            corr = df[spec.columns].corr()
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr)), corr.index)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    v = corr.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.6 else "black")
            fig.colorbar(im, ax=ax, label="Pearson r")
        else:
            return RenderReport(ok=False, warnings=[f"Unsupported chart_type {t!r}."])
        if title:
            ax.set_title(title)
        elif t == "correlation_heatmap":
            ax.set_title("Correlation matrix")
        else:
            ax.set_title(f"{spec.y} by {spec.x}" if spec.y else spec.x)
        fig.tight_layout()
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
    return RenderReport(ok=True, path=str(path), chart_type=t, n_points=len(df[needed].dropna()))


@dataclasses.dataclass
class ChartExplanation(ToolResult):
    chart_type: str = ""
    finding: str = ""
    facts: dict = dataclasses.field(default_factory=dict)


def explain_chart(df: pd.DataFrame, spec: ChartSpec) -> ChartExplanation:
    """State what the chart *shows about the data*, computed from the data, never from the spec's styling
    (cf. LIDA explain(), redefined per §6.2). Every sentence is backed by a number in `facts`."""
    if not spec.ok:
        return ChartExplanation(ok=False, warnings=["Cannot explain an invalid ChartSpec."])
    t, x, y = spec.chart_type, spec.x, spec.y
    try:
        if t == "histogram":
            s = df[x].dropna()
            skew = float(s.skew())
            shape = "roughly symmetric" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
            facts = {"n": len(s), "median": float(s.median()), "mean": float(s.mean()), "skew": skew}
            finding = f"{x} is {shape} (skew {skew:.2f}); median {facts['median']:.3g}, mean {facts['mean']:.3g}, n={len(s)}."
        elif t == "scatter":
            sub = df[[x, y]].dropna()
            r = float(sub[x].corr(sub[y]))
            strength = "negligible" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong"
            facts = {"n": len(sub), "pearson_r": r}
            finding = (f"{x} and {y} show a {strength} {'positive' if r > 0 else 'negative'} linear association "
                       f"(r = {r:.2f}, n = {len(sub)}); correlation alone does not establish a causal link.")
        elif t == "box":
            g = df.groupby(x, observed=True)[y].median().sort_values()  # a box shows medians, so report medians
            facts = {"highest_median": (str(g.index[-1]), float(g.iloc[-1])),
                     "lowest_median": (str(g.index[0]), float(g.iloc[0])), "n_groups": len(g)}
            finding = (f"Median {y} is highest for {x} = {g.index[-1]} ({g.iloc[-1]:.3g}) and lowest for "
                       f"{g.index[0]} ({g.iloc[0]:.3g}) across {len(g)} groups; run a test before calling the gap real.")
        elif t == "bar" and y:
            g = df.groupby(x, observed=True)[y].mean().sort_values()
            facts = {"highest": (str(g.index[-1]), float(g.iloc[-1])), "lowest": (str(g.index[0]), float(g.iloc[0])),
                     "n_groups": len(g)}
            finding = (f"Mean {y} is highest for {x} = {g.index[-1]} ({g.iloc[-1]:.3g}) and lowest for {g.index[0]} "
                       f"({g.iloc[0]:.3g}) across {len(g)} groups; run a test before calling the gap real.")
        elif t == "bar":
            vc = df[x].value_counts()
            facts = {"top": (str(vc.index[0]), int(vc.iloc[0])), "n": int(vc.sum())}
            finding = f"{vc.index[0]} is the most common {x} ({vc.iloc[0]} of {vc.sum()}, {vc.iloc[0] / vc.sum():.0%})."
        elif t == "line":
            sub = df[[x, y]].dropna().copy()
            sub[x] = _as_time(sub[x])
            sub = sub.sort_values(x)
            idx = np.arange(len(sub))
            slope = float(np.polyfit(idx, sub[y].to_numpy(float), 1)[0])
            peak = sub[x].iloc[int(sub[y].to_numpy().argmax())]
            if isinstance(peak, pd.Timestamp) and peak == peak.normalize():
                peak = peak.date()  # "2012-12-17", not "2012-12-17 00:00:00"
            facts = {"first": float(sub[y].iloc[0]), "last": float(sub[y].iloc[-1]), "slope_per_step": slope,
                     "max_at": str(peak)}
            finding = (f"{y} moves from {facts['first']:.3g} to {facts['last']:.3g} "
                       f"({'upward' if slope > 0 else 'downward'} linear trend, {slope:+.3g} per step), "
                       f"peaking at {x} = {facts['max_at']}.")
        elif t == "heatmap":
            ct = pd.crosstab(df[x], df[y])
            i, j = np.unravel_index(ct.values.argmax(), ct.shape)
            facts = {"largest_cell": (str(ct.index[i]), str(ct.columns[j]), int(ct.values[i, j]))}
            finding = f"The most common combination is {x} = {ct.index[i]} with {y} = {ct.columns[j]} ({ct.values[i, j]} rows)."
        elif t == "correlation_heatmap":
            corr = df[spec.columns].corr()
            pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
            a, b, r = max(pairs, key=lambda p: abs(p[2]))
            facts = {"strongest_pair": (a, b, r), "n_pairs": len(pairs),
                     "n_abs_r_above_0_5": sum(abs(p[2]) >= 0.5 for p in pairs)}
            finding = (f"The strongest association among {len(spec.columns)} variables is {a} with {b} (r = {r:.2f}); "
                       f"{facts['n_abs_r_above_0_5']} of {len(pairs)} pairs have |r| >= 0.5.")
        else:
            return ChartExplanation(ok=False, warnings=[f"No explainer for chart_type {t!r}."])
    except (KeyError, ValueError, IndexError) as e:
        return ChartExplanation(ok=False, warnings=[f"Could not compute the finding: {e}"])
    return ChartExplanation(ok=True, chart_type=t, finding=finding, facts=facts)
