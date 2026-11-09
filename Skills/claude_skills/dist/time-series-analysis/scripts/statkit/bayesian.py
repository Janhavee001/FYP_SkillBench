"""
Category H (Bayesian): closed-form posteriors only.

Compared with StatABench: its bayesian_inference is also conjugate (Beta-Binomial, and a normal
model that plugs in the sample variance as if it were known, which understates uncertainty at
small n; here the variance is unknown under a Jeffreys prior). Its bayesian_linear_regression is
sklearn's BayesianRidge (not covered here; use regression-analysis). Its fit_hierarchical_model
samples with PyMC, a heavy dependency that takes minutes and can fail to converge silently, which
is a poor fit for a tool an agent calls inside a chat sandbox. Every model here has an exact
conjugate or empirical-Bayes solution, so results are instant, deterministic and checkable.
The trade-off, stated in each warning: priors are fixed families, and the hierarchical model is
empirical Bayes (the between-group variance is estimated, not given a prior), which slightly
understates uncertainty when there are few groups.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats

from .common import ToolResult, is_numeric


@dataclasses.dataclass
class BayesReport(ToolResult):
    model: str = ""
    posterior: dict = dataclasses.field(default_factory=dict)  # quantity -> {mean, lower, upper}
    prob_statements: dict = dataclasses.field(default_factory=dict)  # e.g. "P(p_a > p_b)" -> 0.93
    prior: str = ""
    n: int = 0


def _interval(dist, level):
    a = (1 - level) / 2
    return float(dist.ppf(a)), float(dist.ppf(1 - a))


def bayesian_estimate(df: pd.DataFrame, model: str, column: str, group_col: str | None = None,
                       groups: list | None = None, level: float = 0.95, prior_a: float = 1.0,
                       prior_b: float = 1.0, n_draws: int = 20000, seed: int = 0) -> BayesReport:
    """
    Three models:

    - `proportion`: Beta-Binomial for a 0/1 column; with `group_col` and two `groups`, also
      P(p_a > p_b). Prior Beta(prior_a, prior_b), default uniform.
    - `mean`: normal data, unknown variance, Jeffreys prior. The posterior for the mean is a
      Student t, so its credible interval equals the frequentist t interval; the value added is the
      direct statement P(mu > 0). With two groups, the posterior of the difference by simulation.
    - `hierarchical`: normal-normal partial pooling of group means (empirical Bayes, method of
      moments for the between-group variance). Small groups are shrunk toward the grand mean.
    """
    rng = np.random.default_rng(seed)
    need = [column] + ([group_col] if group_col else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        return BayesReport(ok=False, warnings=[f"Column(s) not found: {missing}"], model=model)
    sub = df[need].dropna()
    if model in ("mean", "hierarchical") and not is_numeric(sub[column]):
        return BayesReport(ok=False, warnings=[f"{column!r} must be numeric."], model=model)

    if model == "proportion":
        if not set(sub[column].unique()) <= {0, 1}:
            return BayesReport(ok=False, warnings=[f"{column!r} must be 0/1."], model=model)
        prior = f"Beta({prior_a:g}, {prior_b:g})"
        if group_col is None:
            k, n = int(sub[column].sum()), len(sub)
            post = stats.beta(prior_a + k, prior_b + n - k)
            lo, hi = _interval(post, level)
            return BayesReport(ok=True, model=model, prior=prior, n=n,
                               posterior={"p": {"mean": float(post.mean()), "lower": lo, "upper": hi}})
        levels = groups or sorted(sub[group_col].unique(), key=str)[:2]
        if len(levels) != 2:
            return BayesReport(ok=False, warnings=["Give exactly two groups."], model=model)
        posts, draws = {}, {}
        for g in levels:
            s = sub.loc[sub[group_col] == g, column]
            d = stats.beta(prior_a + s.sum(), prior_b + len(s) - s.sum())
            lo, hi = _interval(d, level)
            posts[f"p[{g}]"] = {"mean": float(d.mean()), "lower": lo, "upper": hi}
            draws[g] = d.rvs(n_draws, random_state=rng)
        diff = draws[levels[0]] - draws[levels[1]]
        posts[f"p[{levels[0]}] - p[{levels[1]}]"] = {"mean": float(diff.mean()),
                                                      "lower": float(np.percentile(diff, 100 * (1 - level) / 2)),
                                                      "upper": float(np.percentile(diff, 100 * (1 + level) / 2))}
        return BayesReport(ok=True, model=model, prior=prior, n=len(sub), posterior=posts,
                           prob_statements={f"P(p[{levels[0]}] > p[{levels[1]}])": float((diff > 0).mean())})

    if model == "mean":
        prior = "Jeffreys (flat on mu, 1/sigma^2 on the variance)"

        def post_t(x):
            n = len(x)
            return stats.t(df=n - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(n))

        if group_col is None:
            x = sub[column].to_numpy(float)
            if len(x) < 3:
                return BayesReport(ok=False, warnings=["Need at least 3 observations."], model=model)
            d = post_t(x)
            lo, hi = _interval(d, level)
            return BayesReport(ok=True, model=model, prior=prior, n=len(x),
                               posterior={"mu": {"mean": float(x.mean()), "lower": lo, "upper": hi}},
                               prob_statements={"P(mu > 0)": float(d.sf(0))},
                               warnings=["With this prior the credible interval equals the t confidence interval; "
                                         "the gain is the direct probability statement."])
        levels = groups or sorted(sub[group_col].unique(), key=str)[:2]
        xs = {g: sub.loc[sub[group_col] == g, column].to_numpy(float) for g in levels}
        if len(levels) != 2 or any(len(v) < 3 for v in xs.values()):
            return BayesReport(ok=False, warnings=["Need exactly two groups with 3+ observations each."], model=model)
        draws = {g: post_t(v).rvs(n_draws, random_state=rng) for g, v in xs.items()}
        diff = draws[levels[0]] - draws[levels[1]]
        key = f"mu[{levels[0]}] - mu[{levels[1]}]"
        return BayesReport(ok=True, model=model, prior=prior, n=len(sub),
                           posterior={key: {"mean": float(diff.mean()),
                                            "lower": float(np.percentile(diff, 100 * (1 - level) / 2)),
                                            "upper": float(np.percentile(diff, 100 * (1 + level) / 2))}},
                           prob_statements={f"P({key} > 0)": float((diff > 0).mean())})

    if model == "hierarchical":
        if group_col is None:
            return BayesReport(ok=False, warnings=["hierarchical needs group_col."], model=model)
        g = sub.groupby(group_col, observed=True)[column]
        means, ns, var = g.mean(), g.size(), g.var(ddof=1)
        if len(means) < 3:
            return BayesReport(ok=False, warnings=["Need at least 3 groups to estimate between-group variance."],
                               model=model)
        sigma2 = float(np.nanmean(var[ns > 1])) if (ns > 1).any() else float(sub[column].var(ddof=1))
        se2 = sigma2 / ns
        grand = float(np.average(means, weights=1 / se2))
        tau2 = max(0.0, float(np.var(means, ddof=1) - se2.mean()))
        posts, shrink = {}, {}
        for k in means.index:
            b = se2[k] / (se2[k] + tau2) if tau2 > 0 else 1.0
            m = b * grand + (1 - b) * means[k]
            v = (1 - b) * se2[k] if tau2 > 0 else se2.min() / len(means)
            half = stats.norm.ppf((1 + level) / 2) * np.sqrt(v)
            posts[str(k)] = {"mean": float(m), "lower": float(m - half), "upper": float(m + half),
                             "raw_mean": float(means[k]), "n": int(ns[k])}
            shrink[str(k)] = float(b)
        warnings = ["Empirical Bayes: the between-group variance is estimated, not given a prior, so intervals "
                    "are slightly too narrow when there are few groups."]
        if tau2 == 0:
            warnings.append("Estimated between-group variance is 0: the data are consistent with all groups "
                            "sharing one mean, and every estimate is pooled to it.")
        return BayesReport(ok=True, warnings=warnings, model=model, prior="normal-normal, empirical Bayes",
                           n=len(sub), posterior=posts,
                           prob_statements={"grand_mean": grand, "tau": float(np.sqrt(tau2)),
                                            **{f"shrinkage[{k}]": s for k, s in shrink.items()}})

    return BayesReport(ok=False, warnings=[f"Unknown model {model!r}; use proportion, mean or hierarchical."],
                       model=model)
