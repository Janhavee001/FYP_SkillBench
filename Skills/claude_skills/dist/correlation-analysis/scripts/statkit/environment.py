"""
Environment check for running the library inside a sandbox (e.g. the Claude app's code
execution). Run first: it says which tools will work, so a missing package is found before an
analysis is half done rather than as an ImportError in the middle of one.
"""
from __future__ import annotations

import dataclasses
import importlib

from .common import ToolResult

# Tools that need a package beyond pandas / numpy / scipy.
_NEEDS = {
    "sklearn": ["regularized_regression", "dimensionality_reduction", "polynomial_expand",
                "propensity_score_matching"],
    "statsmodels": ["ks_normality_test", "ab_power_analysis", "stationarity_test", "seasonal_decompose",
                    "arima_forecast", "glm_fit", "robust_regression", "cox_regression",
                    "difference_in_differences"],
    "matplotlib": ["render_chart"],
}


@dataclasses.dataclass
class EnvironmentReport(ToolResult):
    versions: dict = dataclasses.field(default_factory=dict)
    missing_packages: list[str] = dataclasses.field(default_factory=list)
    unavailable_tools: list[str] = dataclasses.field(default_factory=list)


def check_environment() -> EnvironmentReport:
    versions, missing = {}, []
    for pkg in ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "matplotlib"]:
        try:
            versions[pkg] = getattr(importlib.import_module(pkg), "__version__", "?")
        except ImportError:
            missing.append(pkg)
    core_missing = [p for p in missing if p in ("pandas", "numpy", "scipy")]
    unavailable = sorted(t for p in missing for t in _NEEDS.get(p, []))
    warnings = []
    if missing:
        warnings.append(f"Missing: {missing}. Try `pip install {' '.join(missing)}`; if installs are blocked, "
                        f"these tools cannot run: {unavailable}. Say so in the answer rather than substituting "
                        "a hand-written method.")
    return EnvironmentReport(ok=not core_missing, warnings=warnings, versions=versions,
                             missing_packages=missing, unavailable_tools=unavailable)
