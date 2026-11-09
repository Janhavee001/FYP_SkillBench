"""
Category A, second half: the three checks that guard *outputs* rather than
inputs -- the number a narrative claims, the "fix" a repair loop reports, and
the rubric dimension a system cannot produce. validation.py guards what goes
into an analysis; this file guards what comes out.
"""
from __future__ import annotations

import dataclasses
import difflib

from .common import ToolResult


# --------------------------------------------------------------------------- #
# verify_numeric_consistency
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class NumericConsistencyReport(ToolResult):
    claimed: float = 0.0
    recomputed: float = 0.0
    relation: str = ""  # match | percent_vs_fraction | scale_mismatch | mismatch
    scale_factor: float | None = None


def verify_numeric_consistency(claimed: float, recomputed: float,
                                rel_tol: float = 1e-3) -> NumericConsistencyReport:
    """
    Compare the number a narrative claims against an independent recompute and
    name the *kind* of disagreement. Targets DOCUMENTATION §7.1 #2: the model
    printed 38.38% against a ground truth of 0.3838 -- the same quantity in a
    different unit, which a plain equality check scores as simply wrong.
    """
    def close(a: float, b: float) -> bool:
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-12)

    if close(claimed, recomputed):
        return NumericConsistencyReport(ok=True, claimed=claimed, recomputed=recomputed,
                                         relation="match", scale_factor=1.0)
    if recomputed != 0 and close(claimed, recomputed * 100):
        return NumericConsistencyReport(
            ok=True, claimed=claimed, recomputed=recomputed,
            relation="percent_vs_fraction", scale_factor=100.0,
            warnings=[f"Claimed {claimed} is the recomputed {recomputed} x 100: the same "
                      "quantity as a percentage. Consistent, but state the unit explicitly."])
    if claimed != 0 and recomputed != 0:
        ratio = claimed / recomputed
        for k in (10, 100, 1000, 1e6, 0.1, 0.01, 0.001, 1e-6):
            if close(ratio, k):
                return NumericConsistencyReport(
                    ok=False, claimed=claimed, recomputed=recomputed,
                    relation="scale_mismatch", scale_factor=k,
                    warnings=[f"Claimed value is {k}x the recomputed value: likely a unit/scale error."])
    return NumericConsistencyReport(
        ok=False, claimed=claimed, recomputed=recomputed, relation="mismatch",
        warnings=[f"Claimed {claimed} does not match recomputed {recomputed}."])


# --------------------------------------------------------------------------- #
# check_repair_scope
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RepairScopeReport(ToolResult):
    failing_line: int | None = None
    changed_lines: list[int] = dataclasses.field(default_factory=list)
    touched_failing_region: bool = False


def check_repair_scope(original_code: str, repaired_code: str,
                        failing_line: int | None = None,
                        context_lines: int = 2) -> RepairScopeReport:
    """
    Before accepting `fixed=True` from a self-correction loop, check that the
    diff touched the region responsible for the failure (the traceback line,
    +/- context_lines). Targets §6.5: nvAgent's Validator reported fixed=True
    on 14/16 questions while the fault lived in scaffolding its refiner could
    not edit, so the "fix" changed nothing that mattered.
    """
    orig, new = original_code.splitlines(), repaired_code.splitlines()
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=orig, b=new).get_opcodes():
        if tag != "equal":
            changed.update(range(i1 + 1, max(i2, i1 + 1) + 1))
    changed_lines = sorted(changed)

    if not changed_lines:
        return RepairScopeReport(ok=False, failing_line=failing_line,
                                  warnings=["Repair is a no-op: the code is unchanged."])
    if failing_line is None:
        return RepairScopeReport(ok=True, changed_lines=changed_lines,
                                  warnings=["No failing line supplied; scope cannot be verified."])
    touched = any(abs(ln - failing_line) <= context_lines for ln in changed_lines)
    warnings = [] if touched else [
        f"Repair changed lines {changed_lines} but the failure is at line {failing_line}. "
        "A 'fixed' flag here is not evidence the fault was addressed."]
    return RepairScopeReport(ok=touched, warnings=warnings, failing_line=failing_line,
                              changed_lines=changed_lines, touched_failing_region=touched)


# --------------------------------------------------------------------------- #
# flag_unmeasurable_dimension
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DimensionFlag(ToolResult):
    dimension: str = ""
    status: str = ""  # scored | n/a
    score: float | None = None
    coverage: str = ""
    n_observed: int = 0
    n_total: int = 0


def flag_unmeasurable_dimension(dimension: str, scores: list[float | None],
                                 min_coverage: float = 0.5) -> DimensionFlag:
    """
    Report `n/a` plus a coverage count instead of a misleading 0 when a system
    structurally cannot produce a rubric dimension (§4.5, §7.3-7.4). `None`
    entries mean "could not be measured"; a real 0.0 stays a 0.0.
    """
    observed = [s for s in scores if s is not None]
    n_total, n_obs = len(scores), len(observed)
    coverage = f"{n_obs}/{n_total}"
    if n_total == 0 or n_obs / n_total < min_coverage:
        return DimensionFlag(
            ok=True, dimension=dimension, status="n/a", coverage=coverage,
            n_observed=n_obs, n_total=n_total,
            warnings=[f"{dimension}: measurable on only {coverage} items; reported n/a, not 0."])
    return DimensionFlag(ok=True, dimension=dimension, status="scored",
                          score=sum(observed) / n_obs, coverage=coverage,
                          n_observed=n_obs, n_total=n_total)
