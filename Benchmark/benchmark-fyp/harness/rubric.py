"""
Shared scoring rubric applied uniformly to all five systems.

Four dimensions, each scored 0.0-1.0. The criteria language is deliberately
borrowed from the two benchmark papers' own evaluation sections so that the
scores here stay commensurable with their published numbers:

  D1 method_selection      <- StatABench judger-open/modeling_groundedness.py
                              ("Statistical Theory and Model Specification",
                               "Methodology and Estimation") and the Stat-Closed
                              "decision-making" question format, which tests
                              whether the model picks the right test at all.
  D2 code_executability    <- DataSciBench's Completion Rate / Task-Function-Code
                              framework, which scores "code execution outcomes
                              using defined metrics and programmatic rules".
  D3 output_correctness    <- StatABench Stat-Closed exact/numeric matching
                              (evaluation/eval_utils.py compare_numeric uses a
                               hard tolerance) combined with DataSciBench's
                              value-vs-ground-truth aggregate functions.
  D4 interpretation_quality<- StatABench judger-open/analysis_groundedness.py
                              ("Interpretation of Results and Context",
                               "Critical Evaluation of Findings") and
                              result_bias.py, on their 0/0.25/0.5/0.75/1.0
                              anchor scale.

Two honesty rails, because a benchmark that overstates its own precision is
worse than no benchmark:

  * Every dimension carries a `coverage` flag. A system that structurally cannot
    do a thing (LIDA never runs a hypothesis test; a chart tool has no numeric
    answer to check) scores `null` with coverage=False rather than 0.0, and
    compare.py reports the covered mean separately from the all-dimension mean.
    Scoring a chart tool 0.0 on statistical correctness would manufacture a
    finding that is really just a category mismatch.
  * D4 uses an LLM judge when OPENAI_API_KEY is set. Without a key it falls back
    to a transparent keyword heuristic and stamps judge="heuristic" on the
    result, so a keyless run is never mistaken for a judged one.

Usage:
    from harness.rubric import score_record, score_system
    scored = score_system(json.load(open("results/lida.json")))
"""
from __future__ import annotations

import json
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness.common import (  # noqa: E402
    extract_numbers, flatten_numeric, have_key, load_env, normalise_chart,
    normalise_declared_chart, normalise_method,
)

DIMENSIONS = ["method_selection", "code_executability", "output_correctness",
              "interpretation_quality"]

# Weights used for the single headline number. Correctness-heavy on purpose:
# the FYP's research questions are about reliability, not fluency.
WEIGHTS = {"method_selection": 0.30, "code_executability": 0.20,
           "output_correctness": 0.30, "interpretation_quality": 0.20}


# --------------------------------------------------------------------------- #
# D1 -- method / chart selection correctness
# --------------------------------------------------------------------------- #
def score_method_selection(rec: dict) -> tuple[float | None, bool, str]:
    """
    1.00  picked a method in expected_method (the canonical choice)
    0.70  picked something in acceptable_methods (defensible alternative)
    0.30  picked a real method from the vocabulary, but not a defensible one
    0.00  no identifiable method at all

    Charts are scored the same way against expected_chart. For kind="mixed"
    the two halves are averaged; "stat" uses method only, "viz" chart only.
    """
    expected = set(rec.get("expected_method") or [])
    acceptable = set(rec.get("acceptable_methods") or []) | expected
    exp_chart = set(rec.get("expected_chart") or [])

    evidence = " \n".join(str(x) for x in [
        rec.get("declared_method") or "", rec.get("code") or "",
        rec.get("final_answer") or "", rec.get("stdout") or ""])
    found_m = set(normalise_method(evidence))

    # A system's declared chart type goes through the exact-token alias map
    # (so Vega-Lite's "point" counts as a scatter); its code and prose go
    # through the looser regex scan.
    found_c = set(normalise_declared_chart(rec.get("declared_chart")))
    found_c |= set(normalise_chart(" \n".join(str(x) for x in [
        rec.get("code") or "", rec.get("final_answer") or ""])))

    def grade(found, exp, acc):
        if not exp:
            return None
        if found & exp:
            return 1.0
        if found & acc:
            return 0.7
        if found:
            return 0.3
        return 0.0

    m = grade(found_m, expected, acceptable)
    # "none" in expected_chart means a chart is optional -> don't punish absence.
    chart_optional = "none" in exp_chart
    c = grade(found_c, exp_chart - {"none"}, exp_chart - {"none"})
    if chart_optional and not found_c:
        c = None

    kind = rec.get("kind", "mixed")
    if kind == "stat":
        parts = [m]
    elif kind == "viz":
        parts = [c]
    else:
        parts = [m, c]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None, False, "no expected method or chart defined for this item"

    score = sum(parts) / len(parts)
    detail = (f"methods_found={sorted(found_m) or '-'} vs expected={sorted(expected)}; "
              f"charts_found={sorted(found_c) or '-'} vs expected={sorted(exp_chart)}")
    return round(score, 3), True, detail


# --------------------------------------------------------------------------- #
# D2 -- code executability  (DataSciBench Completion Rate style)
# --------------------------------------------------------------------------- #
def native_verdict(rec: dict) -> float | None:
    """
    The benchmark's own adjudication of this item, if it has one.

    StatABench and DataSciBench are benchmarks, not systems: they define their
    own data and their own scoring, and their verdict is far more precise than
    anything a generic text scan can recover. Where they have ruled, we carry
    that ruling through rather than re-deriving a worse version of it.
    """
    nums = rec.get("numeric_outputs") or {}
    for field in ("native_correct", "completion_rate"):
        if field in nums:
            try:
                return float(nums[field])
            except (TypeError, ValueError):
                return None
    return None


def is_native_record(rec: dict) -> bool:
    return native_verdict(rec) is not None


def score_code_executability(rec: dict) -> tuple[float | None, bool, str]:
    """
    1.00  code ran to completion and produced output (stdout or an artifact)
    0.75  code ran to completion but produced nothing observable
    0.25  code was generated but raised on execution
    0.00  no code was generated at all

    A system that is not code-generating by design gets coverage=False.
    """
    code = (rec.get("code") or "").strip()
    if rec.get("error") == "system_not_code_generating":
        return None, False, "system does not emit code by design"
    if not code and is_native_record(rec):
        # StatABench's MCP agent answers by calling registered tools, not by
        # writing a script. Scoring that 0.0 for "no code emitted" measures the
        # interaction style, not the capability.
        return None, False, ("benchmark agent answers via tool calls, not "
                             "generated code")
    if not code:
        return 0.0, True, "no code emitted"
    if not rec.get("executed"):
        err = (rec.get("execution_error") or "unknown error").splitlines()
        return 0.25, True, f"code emitted but failed: {err[-1][:160] if err else 'error'}"
    produced = bool((rec.get("stdout") or "").strip()) or bool(rec.get("artifact"))
    if produced:
        return 1.0, True, "executed and produced output"
    return 0.75, True, "executed but produced no observable output"


# --------------------------------------------------------------------------- #
# D3 -- output correctness against computed ground truth
# --------------------------------------------------------------------------- #
def _close(got: float, want: float, tol: float) -> bool:
    if want == 0:
        return abs(got) <= max(tol, 1e-6)
    return abs(got - want) / abs(want) <= tol or abs(got - want) <= tol


def _rel_match(got: float, want: float, tol: float) -> tuple[bool, bool]:
    """
    Returns (matched, matched_only_after_rescaling).

    Rates are the single commonest source of false misses: questions.json
    stores survival rate as 0.3838 while a system prints "38.38%". Both are
    correct and it would be wrong to score the second as a miss, so a
    percentage/proportion rescale is accepted as a fallback -- but only after
    a direct match has failed, only for values in the range where the
    conversion is meaningful, and it is reported in the score detail so a
    reader can audit which facts were credited this way.
    """
    if _close(got, want, tol):
        return True, False
    aw = abs(want)
    if 0 < aw <= 1 and _close(got, want * 100.0, tol):
        return True, True          # ground truth proportion, output percent
    if 1 < aw <= 100 and _close(got, want / 100.0, tol):
        return True, True          # ground truth percent, output proportion
    return False, False


def score_output_correctness(rec: dict) -> tuple[float | None, bool, str]:
    """
    Fraction of ground-truth facts recoverable from the system's output.

    Numeric facts match within the item's tolerance (relative or absolute,
    whichever is kinder -- mirroring StatABench's compare_numeric, which uses a
    fixed 0.001 absolute band). Boolean and string facts match on keyword
    presence.

    Known looseness, stated rather than hidden: a fact counts as recovered if a
    matching value appears anywhere in numeric_outputs, stdout, or the final
    answer. A very long stdout can therefore match by coincidence. Structured
    `numeric_outputs` is checked first and preferred; the text scan is the
    fallback for systems that only return prose.
    """
    nv = native_verdict(rec)
    if nv is not None:
        return (round(nv, 3), True,
                "benchmark's own adjudication carried through "
                "(its scoring is more precise than a generic text scan)")

    gt = rec.get("ground_truth") or {}
    if not gt:
        return None, False, "no ground truth for this item"
    if rec.get("error") == "system_not_answering":
        return None, False, "system produces charts, not numeric answers"

    # A system whose answer IS a rendered chart has no numeric channel for this
    # check to read. LIDA is the clear case: it executed 12/16 questions and
    # produced a PNG each time, but emitted zero stdout across all 16, and its
    # explain() text describes the chart's design rather than its values.
    # Scoring that near-zero would report "bad at statistics" when the honest
    # statement is "not measurable by this method" -- recovering the numbers
    # would mean reading them off the image, which is what VisEval's MLLM judge
    # does and this rubric does not.
    #
    # Only applies when the system actually succeeded and produced a chart. A
    # run that executed nothing and produced nothing is a real 0.0, not an
    # exemption.
    has_numeric_channel = (bool((rec.get("stdout") or "").strip())
                           or bool(rec.get("numeric_outputs")))
    if not has_numeric_channel and rec.get("executed") and rec.get("artifact"):
        return None, False, ("answer is a rendered chart with no numeric "
                             "channel; not measurable without reading values "
                             "off the image")

    tol = float(rec.get("tolerance") or 0.05)
    numeric_gt = flatten_numeric(gt)

    declared = rec.get("numeric_outputs") or {}
    declared_vals = [v for v in flatten_numeric(declared).values()]
    text = " ".join(str(x) for x in [rec.get("stdout") or "",
                                     rec.get("final_answer") or ""])
    text_vals = extract_numbers(text)
    pool = declared_vals + text_vals

    hits, total, misses, rescaled = 0, 0, [], []
    for path, want in numeric_gt.items():
        total += 1
        results = [_rel_match(g, want, tol) for g in pool]
        matched = [r for r in results if r[0]]
        if matched:
            hits += 1
            # Credited only via a percent/proportion conversion -- worth
            # surfacing so a reader can see it was not a literal match.
            if all(r[1] for r in matched):
                rescaled.append(path)
        else:
            misses.append(f"{path}={want}")

    # Non-numeric facts: booleans and short strings, matched on keywords.
    def walk(obj, prefix=""):
        nonlocal hits, total
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, bool):
            total += 1
            want_words = (r"significant|reject|yes|true|is associated"
                          if obj else r"not significant|fail to reject|no\b|false|"
                                      r"not associated|does not")
            if re.search(want_words, text, re.IGNORECASE):
                hits += 1
            else:
                misses.append(f"{prefix}={obj}")
        elif isinstance(obj, str) and len(obj) < 120:
            total += 1
            # Match on the distinctive tokens of the expected string.
            toks = [t for t in re.split(r"[^a-zA-Z]+", obj.lower())
                    if len(t) > 3][:4]
            if toks and sum(t in text.lower() for t in toks) >= max(1, len(toks) // 2):
                hits += 1
            else:
                misses.append(f"{prefix}='{obj[:40]}'")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")

    for k, v in gt.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            walk(v, k)

    if total == 0:
        return None, False, "ground truth has no checkable facts"
    score = hits / total
    return (round(score, 3), True,
            f"{hits}/{total} ground-truth facts recovered"
            + (f"; {len(rescaled)} via %/proportion rescale {rescaled[:3]}"
               if rescaled else "")
            + (f"; missed {misses[:4]}" if misses else ""))


# --------------------------------------------------------------------------- #
# D4 -- interpretation quality (StatABench anchors)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = (
    "You are an expert judge evaluating an automated statistical analysis "
    "system, in the style of the StatABench open-track judger. You assess only "
    "the quality of the interpretation, not whether the arithmetic is right."
)

JUDGE_PROMPT = """Evaluate the INTERPRETATION QUALITY of the system's answer.

Score each aspect from 0-1, starting at 0 and requiring justification for any
increase. These anchors are taken from the StatABench evaluation criteria.

1. Interpretation of Results and Context (0-1):
   0.00: No interpretation. Presents a table or a plot with no explanation.
   0.25: Literal interpretation. States a number without translating it into the
         problem's domain.
   0.50: Clear contextual interpretation. Explains what the result means for the
         question that was asked.
   0.75: Thorough interpretation with limitations. Distinguishes correlation from
         causation, discusses scope, uses uncertainty measures.
   1.00: Exceptional interpretation. Weaves statistical findings, domain
         knowledge and confounders into a nuanced account of what the analysis
         does and does not show.

2. Critical Evaluation of Findings (0-1):
   0.00: No critical thinking. Accepts all output as absolute truth.
   0.25: Basic criticism. Mentions an obvious limitation without its impact.
   0.50: Standard analysis of limitations. Identifies key assumptions and data
         limitations and how they affect the conclusion.
   0.75: Strong analysis. Investigates robustness, checks assumptions, or
         qualifies the result against an alternative explanation.
   1.00: Exceptional critique.

QUESTION ASKED:
{question}

GROUND TRUTH (for your reference only -- do NOT score arithmetic accuracy here):
{ground_truth}

SYSTEM'S ANSWER:
{answer}

Reply with strict JSON only:
{{"interpretation": <0-1>, "critical_evaluation": <0-1>, "reason": "<one sentence>"}}
"""

HEURISTIC_MARKERS = [
    (r"however|but |although|caveat|limitation|caution|note that", 0.15),
    (r"not (imply|mean) causation|correlation.{0,20}causation|causal", 0.15),
    (r"p[\s\-]?value|confidence interval|significan|95%|std|standard error", 0.15),
    (r"missing|imputed|dropped|excluded|sample size|\bn\s*=", 0.15),
    (r"suggests|indicates|implies|means that|therefore|because", 0.15),
    (r"skew|outlier|imbalanc|assumption|robust", 0.15),
]


CACHE_PATH = pathlib.Path(__file__).resolve().parent.parent / "results" / ".judge_cache.json"
_cache: dict | None = None


def _cache_load() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cache = {}
    return _cache


def _cache_save() -> None:
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, indent=0), encoding="utf-8")


def _cache_key(rec: dict, model: str) -> str:
    import hashlib
    blob = "\x00".join([model, str(rec.get("question", "")),
                        str(rec.get("final_answer", "")),
                        json.dumps(rec.get("ground_truth", {}), sort_keys=True)])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def judge_many(records: list[dict], cfg: dict | None = None,
               use_llm: bool | None = None, workers: int = 8) -> list[tuple]:
    """
    Score the interpretation dimension for many records at once.

    Two reasons this exists rather than one call per record inline: the judge is
    the only part of the rubric that costs money and latency, and re-running
    compare.py while writing up should not re-pay for answers that have not
    changed. Results are cached on a hash of (judge model, question, answer,
    ground truth), and the uncached ones are fetched concurrently.
    """
    if use_llm is None:
        use_llm = have_key()
    if not use_llm:
        return [score_interpretation(r, cfg=cfg, use_llm=False) for r in records]

    cfg = cfg or load_env()
    cache = _cache_load()
    model = cfg["judge_model"]
    out: list = [None] * len(records)
    todo = []

    for i, rec in enumerate(records):
        key = _cache_key(rec, model)
        hit = cache.get(key)
        if hit is not None:
            out[i] = (hit[0], hit[1], hit[2] + " [cached]")
        else:
            todo.append((i, rec, key))

    if todo:
        from concurrent.futures import ThreadPoolExecutor
        def work(item):
            i, rec, key = item
            return i, key, score_interpretation(rec, cfg=cfg, use_llm=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, key, res in pool.map(work, todo):
                out[i] = res
                # Only cache a real judgement; a transient API failure falls
                # back to the heuristic and must not be frozen in.
                if "llm-judge" in (res[2] or ""):
                    cache[key] = [res[0], res[1], res[2]]
        _cache_save()

    return out


def score_interpretation(rec: dict, cfg: dict | None = None,
                         use_llm: bool | None = None) -> tuple[float | None, bool, str]:
    answer = (rec.get("final_answer") or "").strip()
    if rec.get("error") == "system_not_interpreting":
        return None, False, "system emits charts/specs without narrative by design"
    if is_native_record(rec):
        # These items ask for a closed-form answer ("The answer is X"), so the
        # response is short by design. Judging it as an interpretation measures
        # the question format, not the system.
        return None, False, ("native benchmark item expects a closed-form "
                             "answer, not an interpretation")
    if not answer:
        return 0.0, True, "no narrative answer produced"

    if use_llm is None:
        use_llm = have_key()

    if use_llm:
        from harness.common import chat
        cfg = cfg or load_env()
        prompt = JUDGE_PROMPT.format(
            question=rec.get("question", ""),
            ground_truth=json.dumps(rec.get("ground_truth", {}))[:1200],
            answer=answer[:6000])
        try:
            raw = chat(prompt, system=JUDGE_SYSTEM, model=cfg["judge_model"],
                       cfg=cfg, temperature=0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group() if m else raw)
            a = float(data.get("interpretation", 0))
            b = float(data.get("critical_evaluation", 0))
            return (round((a + b) / 2, 3), True,
                    f"llm-judge: {data.get('reason', '')[:180]}")
        except Exception as exc:  # noqa: BLE001
            return _heuristic(answer, note=f"llm judge failed ({exc}); ")
    return _heuristic(answer)


def _heuristic(answer: str, note: str = "") -> tuple[float, bool, str]:
    score = 0.1 if len(answer) > 40 else 0.0
    fired = []
    for pat, w in HEURISTIC_MARKERS:
        if re.search(pat, answer, re.IGNORECASE):
            score += w
            fired.append(pat.split("|")[0])
    score = min(1.0, score)
    return round(score, 3), True, f"{note}heuristic (no LLM judge); markers: {fired}"


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def score_record(rec: dict, cfg: dict | None = None,
                 use_llm: bool | None = None,
                 interp: tuple | None = None) -> dict:
    """
    Score one record. `interp` lets a caller supply an already-computed
    interpretation result (from judge_many) so the LLM call is not repeated.
    """
    scores, coverage, detail = {}, {}, {}
    for dim, fn in [
        ("method_selection", score_method_selection),
        ("code_executability", score_code_executability),
        ("output_correctness", score_output_correctness),
    ]:
        s, cov, why = fn(rec)
        scores[dim], coverage[dim], detail[dim] = s, cov, why
    s, cov, why = (interp if interp is not None
                   else score_interpretation(rec, cfg=cfg, use_llm=use_llm))
    scores["interpretation_quality"], coverage["interpretation_quality"] = s, cov
    detail["interpretation_quality"] = why

    covered = {d: v for d, v in scores.items() if v is not None}
    if covered:
        wsum = sum(WEIGHTS[d] for d in covered)
        overall = round(sum(scores[d] * WEIGHTS[d] for d in covered) / wsum, 3)
    else:
        overall = None

    return {"id": rec.get("id"), "dataset": rec.get("dataset"),
            "kind": rec.get("kind"), "scores": scores, "coverage": coverage,
            "detail": detail, "overall": overall,
            "repair_attempts": rec.get("repair_attempts", 0),
            "self_corrected": bool(rec.get("self_corrected"))}


def score_system(payload: dict, cfg: dict | None = None,
                 use_llm: bool | None = None) -> dict:
    """Score every record in one results/*.json payload."""
    records = payload.get("records", [])
    # Batch the judge across the whole system: cached hits are free and the
    # misses go out concurrently, instead of one blocking call per record.
    interps = judge_many(records, cfg=cfg, use_llm=use_llm)
    scored = [score_record(r, cfg=cfg, use_llm=use_llm, interp=it)
              for r, it in zip(records, interps)]

    # Whether a system has a numeric channel at all is a property of the SYSTEM,
    # not of individual records. Deciding it per record leaves a chart-only tool
    # scored on exactly the handful of items where it crashed and produced no
    # chart -- i.e. its correctness average would be computed purely from its
    # own failures, which is a more misleading number than the one it replaced.
    # If a system never emitted a numeric value anywhere, output correctness is
    # simply not measurable for it, and code_executability already carries the
    # information about the runs that failed.
    emits_numeric = any(bool((r.get("stdout") or "").strip())
                        or bool(r.get("numeric_outputs")) for r in records)
    if records and not emits_numeric:
        why = ("system emitted no numeric output on any question; correctness "
               "is not measurable without reading values off its charts")
        for s in scored:
            s["scores"]["output_correctness"] = None
            s["coverage"]["output_correctness"] = False
            s["detail"]["output_correctness"] = why
            covered = {d: v for d, v in s["scores"].items() if v is not None}
            s["overall"] = (round(sum(s["scores"][d] * WEIGHTS[d] for d in covered)
                                  / sum(WEIGHTS[d] for d in covered), 3)
                            if covered else None)

    agg, cov_counts = {}, {}
    for dim in DIMENSIONS:
        vals = [s["scores"][dim] for s in scored if s["scores"][dim] is not None]
        agg[dim] = round(sum(vals) / len(vals), 3) if vals else None
        cov_counts[dim] = len(vals)

    overalls = [s["overall"] for s in scored if s["overall"] is not None]
    return {
        "system": payload.get("system"),
        "paper": payload.get("paper"),
        "architecture": payload.get("architecture"),
        "native_evaluation": payload.get("native_evaluation"),
        "meta": payload.get("meta", {}),
        "n_records": len(scored),
        "dimension_scores": agg,
        "dimension_coverage": cov_counts,
        "overall": round(sum(overalls) / len(overalls), 3) if overalls else None,
        "judge": "llm" if (use_llm if use_llm is not None else have_key())
                 else "heuristic",
        "records": scored,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score one results/*.json file.")
    ap.add_argument("path")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the heuristic judge even if a key is present")
    a = ap.parse_args()
    payload = json.loads(pathlib.Path(a.path).read_text(encoding="utf-8"))
    out = score_system(payload, use_llm=False if a.no_llm else None)
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=2))
