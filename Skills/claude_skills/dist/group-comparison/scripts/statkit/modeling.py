"""
Categories E: regression and modelling. Implemented on numpy/scipy/sklearn so
coefficient tables carry standard errors and p-values without pulling in a
formula interface. Every regression reports n, rows dropped for missingness,
and (for multivariable fits) VIF, because a coefficient table without those is
exactly the "number with no context" failure the catalog is built around.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from .common import ToolResult, is_numeric



def _check_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    bad = [c for c in cols if c not in df.columns or not is_numeric(df[c])]
    return [f"Column(s) missing or not numeric: {bad}"] if bad else []


@dataclasses.dataclass
class RegressionReport(ToolResult):
    y: str = ""
    predictors: list[str] = dataclasses.field(default_factory=list)
    n: int = 0
    n_dropped: int = 0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    coefficients: dict = dataclasses.field(default_factory=dict)  # name -> {coef, se, t, p, ci_lower, ci_upper}
    vif: dict = dataclasses.field(default_factory=dict)
    residual_normality_p: float | None = None


def _design(df: pd.DataFrame, y: str, predictors: list[str]):
    """Rows complete on the model columns, with text/category/bool predictors dummy-coded
    (drop_first, as StatABench's multivariable_linear_regression does). Returns
    (frame of y + numeric design columns, design column names, notes, problems)."""
    missing = [c for c in [y] + predictors if c not in df.columns]
    if missing:
        return None, [], [], [f"Column(s) not found: {missing}"]
    if not is_numeric(df[y]):
        return None, [], [], [f"Outcome {y!r} is not numeric."]
    sub = df[[y] + predictors].dropna()
    categorical = [c for c in predictors if not is_numeric(sub[c])]
    notes = []
    if categorical:
        sub = pd.get_dummies(sub, columns=categorical, drop_first=True, dtype=float)
        notes.append(f"Dummy-coded {categorical} (first level is the reference category).")
    xs = [c for c in sub.columns if c != y]
    return sub, xs, notes, []


def _ols(df: pd.DataFrame, y: str, predictors: list[str], with_vif: bool) -> RegressionReport:
    sub, xs, notes, problems = _design(df, y, predictors)
    if problems:
        return RegressionReport(ok=False, warnings=problems, y=y, predictors=predictors)
    n, k = len(sub), len(xs)
    if n <= k + 1:
        return RegressionReport(ok=False, warnings=[f"n={n} is too small for {k} predictor(s)."], y=y, predictors=predictors)
    X = np.column_stack([np.ones(n), sub[xs].to_numpy(float)])
    yv = sub[y].to_numpy(float)
    # Rank of the column-standardised design, so the check does not depend on units
    # (a cond(X'X) threshold rejects legitimate fits that mix e.g. density ~1 with sulphur ~100).
    Xs = X[:, 1:] - X[:, 1:].mean(axis=0)
    spread = Xs.std(axis=0)
    if (spread == 0).any() or np.linalg.matrix_rank(np.column_stack([np.ones(n), Xs / np.where(spread == 0, 1, spread)])) < k + 1:
        return RegressionReport(ok=False, y=y, predictors=predictors,
                                 warnings=["Design matrix is singular: a predictor is constant or an exact "
                                           "combination of others."])
    xtx = X.T @ X
    beta = np.linalg.solve(xtx, X.T @ yv)
    resid = yv - X @ beta
    dof = n - k - 1
    sigma2 = float(resid @ resid / dof)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(xtx)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    tcrit = stats.t.ppf(0.975, dof)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot else 0.0
    names = ["intercept"] + xs
    coefs = {nm: {"coef": float(b), "se": float(s), "t": float(tt), "p": float(pp),
                  "ci_lower": float(b - tcrit * s), "ci_upper": float(b + tcrit * s)}
             for nm, b, s, tt, pp in zip(names, beta, se, t, p)}

    warnings = list(notes)
    if len(df) - n:
        warnings.append(f"{len(df) - n} row(s) dropped for missing values in the model columns.")
    vif = {}
    if with_vif and k > 1:
        for i, nm in enumerate(xs):
            others = np.column_stack([np.ones(n), np.delete(sub[xs].to_numpy(float), i, axis=1)])
            b, *_ = np.linalg.lstsq(others, sub[nm].to_numpy(float), rcond=None)
            r = sub[nm].to_numpy(float) - others @ b
            ss = float(((sub[nm] - sub[nm].mean()) ** 2).sum())
            r2_i = 1 - float(r @ r) / ss if ss else 0.0
            vif[nm] = float(1 / (1 - r2_i)) if r2_i < 1 else float("inf")
        high = [nm for nm, v in vif.items() if v > 10]
        if high:
            warnings.append(f"VIF > 10 for {high}: multicollinearity makes their individual coefficients unstable.")
    norm_p = float(stats.shapiro(resid[:5000]).pvalue) if n >= 3 else None
    if norm_p is not None and norm_p < 0.01 and n < 200:
        warnings.append("Residuals depart from normality at small n; coefficient p-values may be unreliable.")
    return RegressionReport(
        ok=True, warnings=warnings, y=y, predictors=predictors, n=n, n_dropped=len(df) - n,
        r_squared=r2, adj_r_squared=1 - (1 - r2) * (n - 1) / dof, coefficients=coefs,
        vif=vif, residual_normality_p=norm_p)


def linear_regression(df: pd.DataFrame, x: str, y: str) -> RegressionReport:
    """Simple OLS of y on one predictor (cf. StatABench simple_linear_regression)."""
    return _ols(df, y, [x], with_vif=False)


def multivariable_regression(df: pd.DataFrame, y: str, predictors: list[str]) -> RegressionReport:
    """OLS with several predictors, standard errors, 95% CIs and VIF (cf. StatABench multivariable_linear_regression)."""
    return _ols(df, y, list(predictors), with_vif=True)


@dataclasses.dataclass
class RegularizedReport(ToolResult):
    method: str = ""
    alpha: float = 0.0
    r_squared: float = 0.0  # in-sample
    cv_r_squared: float | None = None  # mean out-of-fold R^2: the number to quote for predictive accuracy
    coefficients: dict = dataclasses.field(default_factory=dict)  # on standardized predictors
    selected: list[str] = dataclasses.field(default_factory=list)
    dropped: list[str] = dataclasses.field(default_factory=list)
    n: int = 0


def regularized_regression(df: pd.DataFrame, y: str, predictors: list[str],
                            method: str = "lasso", cv: int = 5, seed: int = 0) -> RegularizedReport:
    """Lasso / ridge / elastic-net with the penalty chosen by cross-validation (cf. StatABench
    advanced_regression, which scores on one 80/20 split). Predictors are standardized, so coefficients
    are comparable; `selected`/`dropped` say what the penalty kept. Reports both in-sample R^2 and
    out-of-fold R^2 from an outer K-fold, with scaling and penalty choice refit inside every fold so
    the score is not leaked."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    method = {"elasticnet": "elastic_net"}.get(method, method)
    sub, xs, notes, problems = _design(df, y, list(predictors))
    if problems:
        return RegularizedReport(ok=False, warnings=problems, method=method)
    if len(sub) < cv * 4:
        return RegularizedReport(ok=False, warnings=[f"n={len(sub)} too small for nested {cv}-fold CV."], method=method)
    makers = {
        "lasso": lambda: LassoCV(cv=cv, random_state=seed, max_iter=20000),
        "ridge": lambda: RidgeCV(alphas=np.logspace(-3, 3, 25), cv=cv),
        "elastic_net": lambda: ElasticNetCV(cv=cv, random_state=seed, l1_ratio=[0.2, 0.5, 0.8], max_iter=20000),
    }
    if method not in makers:
        return RegularizedReport(ok=False, warnings=[f"Unknown method {method!r}; use lasso, ridge or elastic_net."], method=method)
    Xraw, yv = sub[xs].to_numpy(float), sub[y].to_numpy(float)
    pipe = make_pipeline(StandardScaler(), makers[method]())
    outer = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_r2 = float(cross_val_score(pipe, Xraw, yv, cv=outer, scoring="r2").mean())
    pipe.fit(Xraw, yv)
    model = pipe[-1]
    coefs = {c: float(b) for c, b in zip(xs, model.coef_)}
    selected = [c for c, b in coefs.items() if abs(b) > 1e-8]
    in_r2 = float(pipe.score(Xraw, yv))
    warnings = list(notes)
    if in_r2 - cv_r2 > 0.1:
        warnings.append(f"In-sample R^2 {in_r2:.2f} vs cross-validated {cv_r2:.2f}: the model overfits; quote the CV figure.")
    return RegularizedReport(
        ok=True, warnings=warnings, method=method, alpha=float(model.alpha_), r_squared=in_r2, cv_r_squared=cv_r2,
        coefficients=coefs, selected=selected, dropped=[c for c in xs if c not in selected], n=len(sub))


@dataclasses.dataclass
class DimReductionReport(ToolResult):
    method: str = ""
    n_components: int = 0
    explained_variance_ratio: list[float] = dataclasses.field(default_factory=list)
    cumulative_variance: list[float] = dataclasses.field(default_factory=list)
    loadings: dict = dataclasses.field(default_factory=dict)  # component -> {column: loading}
    n: int = 0


def dimensionality_reduction(df: pd.DataFrame, columns: list[str] | None = None, n_components: int = 2,
                              method: str = "pca", seed: int = 0) -> DimReductionReport:
    """PCA or sparse PCA on standardized numeric columns (cf. StatABench sparse_pca_analysis). Returns loadings,
    not just scores: a component nobody can name is not an analytical result."""
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    problems = _check_numeric(df, cols)
    if problems or len(cols) < 2:
        return DimReductionReport(ok=False, warnings=problems or ["Need at least 2 numeric columns."], method=method)
    sub = df[cols].dropna()
    n_components = min(n_components, len(cols))
    X = StandardScaler().fit_transform(sub.to_numpy(float))
    if method == "pca":
        model = PCA(n_components=n_components, random_state=seed).fit(X)
        evr = model.explained_variance_ratio_
    elif method == "sparse_pca":
        model = SparsePCA(n_components=n_components, random_state=seed).fit(X)
        scores = model.transform(X)
        evr = scores.var(axis=0) / X.var(axis=0).sum()
    else:
        return DimReductionReport(ok=False, warnings=[f"Unknown method {method!r}."], method=method)
    loadings = {f"PC{i + 1}": {c: float(v) for c, v in zip(cols, comp)} for i, comp in enumerate(model.components_)}
    warnings = [f"{len(df) - len(sub)} row(s) dropped for missing values."] if len(df) - len(sub) else []
    if method == "sparse_pca":
        warnings.append("Sparse components are not orthogonal, so the per-component variance shares overlap "
                        "and are approximate; do not add them up as if they were PCA's.")
    return DimReductionReport(ok=True, warnings=warnings, method=method, n_components=n_components,
                              explained_variance_ratio=[float(v) for v in evr],
                              cumulative_variance=[float(v) for v in np.cumsum(evr)], loadings=loadings, n=len(sub))


@dataclasses.dataclass
class FeatureSelectionReport(ToolResult):
    method: str = ""
    scores: dict = dataclasses.field(default_factory=dict)
    selected: list[str] = dataclasses.field(default_factory=list)
    dropped: list[str] = dataclasses.field(default_factory=list)


def feature_selection(df: pd.DataFrame, target: str | None = None, columns: list[str] | None = None,
                       method: str = "variance", threshold: float = 0.0, top_k: int | None = None) -> FeatureSelectionReport:
    """
    `general`: drop all-missing, constant, infinite-valued and ID-like text columns (mirrors MetaGPT
    GeneralSelection). `variance`: drop numeric columns with variance <= threshold (MetaGPT
    VarianceBasedSelection). `correlation`: rank numeric columns by |Pearson r| with the target; this one
    has no MetaGPT counterpart and is univariate, so it misses predictors that matter only jointly.
    """
    if method == "general":
        cand = [c for c in (columns or df.columns) if c != target]
        scores, dropped_why = {}, {}
        for c in cand:
            s = df[c]
            if s.isna().all():
                dropped_why[c] = "all missing"
            elif s.nunique(dropna=True) == 1:
                dropped_why[c] = "constant"
            elif is_numeric(s) and np.isinf(s.to_numpy(float, na_value=np.nan)).any():
                dropped_why[c] = "contains infinite values"
            elif not is_numeric(s) and s.nunique() == len(s):
                dropped_why[c] = "ID-like (every value unique)"
            scores[c] = 0.0 if c in dropped_why else 1.0
        return FeatureSelectionReport(ok=True, method=method, scores=scores,
                                      selected=[c for c in cand if c not in dropped_why],
                                      dropped=list(dropped_why),
                                      warnings=[f"Dropped {c!r}: {why}." for c, why in dropped_why.items()])
    cols = [c for c in (columns or df.columns) if c != target and is_numeric(df[c])]
    if not cols:
        return FeatureSelectionReport(ok=False, method=method, warnings=["No numeric candidate columns."])
    if method == "variance":
        scores = {c: float(df[c].var(ddof=0)) for c in cols}
        selected = [c for c, v in scores.items() if v > threshold]
    elif method == "correlation":
        if target is None or target not in df.columns:
            return FeatureSelectionReport(ok=False, method=method, warnings=["correlation method needs a numeric `target`."])
        with np.errstate(invalid="ignore", divide="ignore"):  # constant columns have undefined r; scored 0 below
            scores = {c: abs(float(df[c].corr(df[target]))) for c in cols}
        scores = {c: (0.0 if np.isnan(v) else v) for c, v in scores.items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        selected = ranked[:top_k] if top_k else [c for c in ranked if scores[c] > threshold]
    else:
        return FeatureSelectionReport(ok=False, method=method, warnings=[f"Unknown method {method!r}."])
    warnings = []
    if method == "variance":
        warnings.append("Raw variance depends on units; scale_numeric first if columns are on different scales.")
    return FeatureSelectionReport(ok=True, warnings=warnings, method=method, scores=scores,
                                  selected=selected, dropped=[c for c in cols if c not in selected])


@dataclasses.dataclass
class GLMReport(ToolResult):
    family: str = ""
    y: str = ""
    n: int = 0
    coefficients: dict = dataclasses.field(default_factory=dict)  # name -> {coef, se, p, ci_lower, ci_upper, exp_coef}
    aic: float | None = None
    deviance: float | None = None
    pseudo_r_squared: float | None = None
    dispersion_ratio: float | None = None  # Pearson chi2 / df_resid; ~1 if the family's variance is right


_FAMILIES = {"gaussian", "binomial", "logistic", "poisson", "gamma", "negative_binomial"}


def glm_fit(df: pd.DataFrame, y: str, predictors: list[str], family: str = "binomial") -> GLMReport:
    """
    Generalized linear model (cf. StatABench run_glm). Families: gaussian, binomial (logistic),
    poisson, gamma, negative_binomial. Categorical predictors are dummy-coded as in
    multivariable_regression.

    Beyond the reference: `exp_coef` (odds ratio for binomial, rate ratio for poisson /
    negative binomial) is returned, because that is the number a reader interprets, and the
    Pearson dispersion ratio is checked, since overdispersed Poisson data give p-values that
    are far too small.
    """
    import statsmodels.api as sm

    family = {"logistic": "binomial"}.get(family, family)
    if family not in _FAMILIES:
        return GLMReport(ok=False, warnings=[f"Unknown family {family!r}; use one of {sorted(_FAMILIES)}."])
    if y not in df.columns:
        return GLMReport(ok=False, warnings=[f"Outcome {y!r} not found."], family=family, y=y)
    data, coding = df, []
    if not is_numeric(df[y]):
        if family != "binomial" or df[y].nunique() != 2:
            return GLMReport(ok=False, warnings=[f"Outcome {y!r} must be numeric (or two-level text for binomial)."],
                             family=family, y=y)
        levels = sorted(df[y].dropna().unique(), key=str)
        data = df.assign(**{y: (df[y] == levels[1]).astype(float).where(df[y].notna())})
        coding = [f"Outcome coded 1 = {levels[1]!r}, 0 = {levels[0]!r}."]
    sub, xs, notes, problems = _design(data, y, predictors)
    if problems:
        return GLMReport(ok=False, warnings=problems, family=family, y=y)
    yv = sub[y].to_numpy(float)
    warnings = coding + notes
    if family == "binomial" and not set(np.unique(yv)) <= {0.0, 1.0}:
        return GLMReport(ok=False, warnings=["binomial needs a 0/1 outcome."], family=family, y=y)
    if family in ("poisson", "negative_binomial") and ((yv < 0).any() or (yv % 1 != 0).any()):
        return GLMReport(ok=False, warnings=[f"{family} needs a non-negative integer count outcome."], family=family, y=y)
    if family == "gamma" and (yv <= 0).any():
        return GLMReport(ok=False, warnings=["gamma needs a strictly positive outcome."], family=family, y=y)
    fam = {"gaussian": sm.families.Gaussian(), "binomial": sm.families.Binomial(),
           "poisson": sm.families.Poisson(), "gamma": sm.families.Gamma(sm.families.links.Log()),
           "negative_binomial": None}[family]
    X = sm.add_constant(sub[xs].to_numpy(float), has_constant="add")
    if family == "negative_binomial":
        # Estimate the dispersion alpha from a Poisson fit by method of moments (Var = mu + alpha mu^2)
        # instead of statsmodels' silent default of alpha = 1.
        mu = sm.GLM(yv, X, family=sm.families.Poisson()).fit().mu
        alpha = max(1e-8, float(np.mean(((yv - mu) ** 2 - yv) / mu ** 2)))
        fam = sm.families.NegativeBinomial(alpha=alpha)
        warnings.append(f"Negative binomial dispersion alpha = {alpha:.3g}, estimated by method of moments.")
    try:
        res = sm.GLM(yv, X, family=fam).fit()
    except (np.linalg.LinAlgError, ValueError) as e:
        return GLMReport(ok=False, warnings=[f"GLM did not fit: {e}"], family=family, y=y)
    if family == "binomial" and np.any(np.abs(res.params) > 15):
        warnings.append("A coefficient is extreme (|b| > 15): likely perfect separation; odds ratios are unreliable.")
    ci = res.conf_int()
    names = ["intercept"] + xs
    coefs = {nm: {"coef": float(b), "se": float(s), "p": float(p), "ci_lower": float(lo), "ci_upper": float(hi),
                  "exp_coef": float(np.exp(b)) if family != "gaussian" else None}
             for nm, b, s, p, (lo, hi) in zip(names, res.params, res.bse, res.pvalues, ci)}
    disp = float(res.pearson_chi2 / res.df_resid) if res.df_resid > 0 else None
    if family == "poisson" and disp and disp > 1.5:
        warnings.append(f"Overdispersion (Pearson chi2/df = {disp:.2f}); standard errors are too small. "
                        "Refit with family='negative_binomial'.")
    pseudo = float(1 - res.deviance / res.null_deviance) if res.null_deviance else None
    if len(df) - len(sub):
        warnings.append(f"{len(df) - len(sub)} row(s) dropped for missing values.")
    return GLMReport(ok=True, warnings=warnings, family=family, y=y, n=len(sub), coefficients=coefs,
                     aic=float(res.aic), deviance=float(res.deviance), pseudo_r_squared=pseudo,
                     dispersion_ratio=disp)


@dataclasses.dataclass
class RobustRegressionReport(ToolResult):
    y: str = ""
    n: int = 0
    coefficients: dict = dataclasses.field(default_factory=dict)  # name -> {coef, se, p, ols_coef}
    n_downweighted: int = 0
    scale: float = 0.0


def robust_regression(df: pd.DataFrame, y: str, predictors: list[str]) -> RobustRegressionReport:
    """
    Huber M-estimation (cf. StatABench huber_regression). Returns the OLS coefficient beside each
    robust one and counts the observations given weight < 1: a large OLS-vs-Huber gap is itself
    the finding (the OLS fit is being driven by a few points).
    """
    import statsmodels.api as sm

    sub, xs, notes, problems = _design(df, y, list(predictors))
    if problems:
        return RobustRegressionReport(ok=False, warnings=problems, y=y)
    X = sm.add_constant(sub[xs].to_numpy(float), has_constant="add")
    yv = sub[y].to_numpy(float)
    res = sm.RLM(yv, X, M=sm.robust.norms.HuberT()).fit()
    ols = np.linalg.lstsq(X, yv, rcond=None)[0]
    names = ["intercept"] + xs
    coefs = {nm: {"coef": float(b), "se": float(s), "p": float(p), "ols_coef": float(o)}
             for nm, b, s, p, o in zip(names, res.params, res.bse, res.pvalues, ols)}
    down = int((res.weights < 0.999).sum())
    warnings = list(notes)
    big = [nm for nm, c in coefs.items() if nm != "intercept"
           and abs(c["coef"] - c["ols_coef"]) > 2 * c["se"]]
    if big:
        warnings.append(f"OLS and Huber disagree on {big} by more than 2 SE: outliers are driving the OLS fit.")
    return RobustRegressionReport(ok=True, warnings=warnings, y=y, n=len(sub), coefficients=coefs,
                                  n_downweighted=down, scale=float(res.scale))
