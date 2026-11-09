"""
Score SkillBench results.

Primary metric (deterministic): is the final answer correct?
  number  value within tolerance of the reference. For percentage questions a proportion is also
          accepted (flagged `unit_slip`), so a correct analysis is not failed for writing 0.25 instead of 25.
  label   the system's final label equals the reference or an alias (case/punctuation-insensitive).
  choice  the system's final choice equals the reference.
If the system gave no numeric `value`, or for labels and choices, an LLM extractor reads the final answer
out of the free text. It only extracts; it never judges correctness.

Trap analysis: a wrong answer that matches the precomputed `naive` answer is counted as `fell_into_trap`.

Secondary metrics (LLM judge, 0/1 each): method appropriate for the data; important caveats stated;
no overclaiming (e.g. causal language for an association). The judge sees the question, the reference,
the trap description and the system's method and interpretation, but not which condition produced it.

    python score.py            -> results/scores.csv, results/summary.md
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark-fyp"))
from harness.common import load_env, openai_client  # noqa: E402

RESULTS = ROOT / "results"
JUDGE = RESULTS / "judgements"
CONDITIONS = ["plain", "agent", "agent_skills"]

EXTRACT = """A data-analysis system answered a question. Extract its FINAL answer only; do not judge it.

QUESTION: {question}
ANSWER TYPE: {atype}{choices}
SYSTEM'S ANSWER: {answer}
SYSTEM'S VALUE FIELD: {value}

Reply with JSON only: {{"final": <the final number, or the final label/choice as a short string, or null if none>}}.
For a number, give the number the system reports as its answer in the units the question asks for.
For a choice, use exactly one of the listed choices."""

RUBRIC = """You are grading the analytic quality of an answer to a data-analysis question. The correct final answer
is known and is scored separately; grade only the three properties below, each 0 or 1.

QUESTION: {question}
REFERENCE ANSWER: {reference}
WHAT A CAREFUL ANALYST MUST NOTICE: {trap}
REFERENCE METHOD: {method}

SYSTEM'S METHOD: {smethod}
SYSTEM'S ANSWER: {answer}
SYSTEM'S INTERPRETATION: {interp}

1. method_appropriate: the method fits the data and question (e.g. medians or rank tests for heavily skewed data,
   aggregation to the right unit before comparing, a test that matches the variable types). 1 if appropriate.
2. caveats_stated: the answer mentions the main limitation or data issue relevant here (the trap above if one is
   given; otherwise any material assumption). 1 if stated.
3. no_overclaiming: the answer does not claim causation from an association, certainty it does not have, or
   precision the data cannot support. 1 if it avoids overclaiming.

Reply with JSON only: {{"method_appropriate": 0 or 1, "caveats_stated": 0 or 1, "no_overclaiming": 0 or 1,
"note": "<one short sentence>"}}"""

TRAP_TEXT = {
    "none": "nothing in particular", "unit_of_analysis": "rows repeat the same entity; analyse at the right unit",
    "missing_code": "missing values are coded as '?' or 'None', not as blanks", "skew": "the variable is heavily skewed",
    "multiple_testing": "many tests are run; correct for multiple comparisons",
    "multi_section_file": "the mapping file holds several stacked tables",
    "population_definition": "patients who died cannot be readmitted; define the population",
    "cancellations": "cancelled invoices (Invoice starting with C) and non-positive quantities/prices",
    "grain": "rows are lines/records, the question is about a higher-level unit; aggregate first",
    "cohort_definition": "define the cohort by first purchase", "duplicates": "exact duplicate rows exist",
    "out_of_range_dates": "some timestamps fall outside January 2024",
    "unrecorded_values": "cash tips are not recorded in this data, so the comparison cannot be made",
    "unequal_exposure": "weekdays occur a different number of times in the month",
    "invalid_records": "invalid records (non-positive durations or distances, extreme values)",
    "confounding": "a third variable confounds the raw association", "time_gaps": "some hourly records are missing",
    "small_sample": "a category or estimate rests on very few observations", "trend": "the series has a strong trend",
    "missing_data_rule": "days with too few readings should not be counted",
    "missing_values": "the series has missing values", "join": "variables live in different tables",
    "multi_valued_field": "a field holds several '|'-separated values", "orphans": "some records have no matching parent",
    "parsing": "a value must be parsed from text or a timestamp", "hidden_entity_key": "customer_id is per order; "
    "customer_unique_id identifies the person", "fan_out": "a one-to-many relation repeats rows when joined",
    "multi_hop_join": "the answer requires joining several tables", "ratio_of_sums": "use the ratio of totals",
    "unclassified_finishers": "retired drivers have no finishing position", "era_effects": "points systems changed",
    "simpsons_paradox": "the pooled comparison is reversed within regions (confounding by region)",
    "interaction": "the effect differs between groups", "cross_table_consistency": "compare totals across tables",
}


def norm(s) -> str:
    return re.sub(r"[^a-z0-9.]+", "", str(s).lower())


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def chat_json(client, cfg, prompt):
    for _ in range(4):
        try:
            r = client.chat.completions.create(model=cfg["judge_model"], temperature=0,
                                               messages=[{"role": "user", "content": prompt}])
            t = r.choices[0].message.content or ""
            return json.loads(t[t.index("{"):t.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            continue
    return {}


def judge_one(q, rec, cfg):
    path = JUDGE / rec["condition"] / f"{q['id']}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    client = openai_client(cfg)
    sub = rec.get("submission") or {}
    answer = str(sub.get("answer", ""))[:3000]
    out = {}
    if q["answer_type"] != "number" or num(sub.get("value")) is None:
        ch = f"\nCHOICES: {q['choices']}" if q["choices"] else ""
        out["extracted"] = chat_json(client, cfg, EXTRACT.format(
            question=q["question"], atype=q["answer_type"], choices=ch, answer=answer, value=sub.get("value"))).get("final")
    out["rubric"] = chat_json(client, cfg, RUBRIC.format(
        question=q["question"], reference=q["reference"], trap=TRAP_TEXT.get(q["trap"], q["trap"]),
        method=q["expected_method"], smethod=str(sub.get("method", ""))[:500], answer=answer,
        interp=str(sub.get("interpretation", ""))[:2000]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def correct(q, final):
    """Returns (correct, unit_slip, matches_naive)."""
    if final is None:
        return False, False, False
    if q["answer_type"] == "number":
        v = num(final)
        if v is None:
            return False, False, False
        ref, tol = q["reference"], q["tolerance"]

        def close(a, b):
            if "abs" in tol and abs(a - b) <= tol["abs"] + 1e-9:
                return True
            if "rel" in tol and abs(a - b) <= tol["rel"] * max(abs(b), 1e-12):
                return True
            return not tol and abs(a - b) <= 1e-6 * max(abs(b), 1)
        ok = close(v, ref)
        if not ok:
            # Audit fix (v2): a value reported to d >= 1 decimals that equals the reference rounded to d decimals
            # is correct, e.g. 25.8 for 25.78 when the question did not ask for a precision. Flagged as rounded.
            d = decimals(v)
            ok = d >= 1 and abs(round(ref, d) - v) < 1e-9 and abs(v - ref) <= 0.01 * abs(ref)  # and within 1%
            ROUNDED.add((q["id"], v)) if ok else None
        slip = False
        if not ok and q["pct"] and close(v * 100, ref):
            ok, slip = True, True
        nv = q.get("naive")
        naive_hit = (not ok) and isinstance(nv, (int, float)) and (close(v, nv) or (q["pct"] and close(v * 100, nv)))
        return ok, slip, bool(naive_hit)
    ref_list = [q["reference"], *q["aliases"]]
    if q["dataset"] == "olist" and str(q["reference"]).upper() in BR_STATES:
        ref_list.append(BR_STATES[str(q["reference"]).upper()])
    golds = {norm(a) for a in ref_list}
    f = norm(final)
    ok = f in golds
    if not ok and q["answer_type"] == "label":
        # The reference as a whole word inside a short final answer: "The United Kingdom" for united-kingdom,
        # "Autodromo Nazionale Monza" for monza, "Occupation code 20" for 20. Whole words only, so "1" never
        # matches "17".
        fw = words(final)
        ok = len(fw.split()) <= 8 and any(re.search(rf"(?<![a-z0-9]){re.escape(words(g))}(?![a-z0-9])", fw)
                                          for g in ref_list if words(g))
    naive_hit = (not ok) and q.get("naive") is not None and f == norm(q["naive"])
    return ok, False, naive_hit


ROUNDED: set = set()  # (question id, value) accepted only through the rounding rule, for the audit report


def decimals(v: float) -> int:
    s = repr(float(v))
    return 0 if ("e" in s or "." not in s) else len(s.split(".")[1].rstrip("0"))


def words(s) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


BR_STATES = {"AC": "Acre", "AL": "Alagoas", "AP": "Amapa", "AM": "Amazonas", "BA": "Bahia", "CE": "Ceara",
             "DF": "Distrito Federal", "ES": "Espirito Santo", "GO": "Goias", "MA": "Maranhao", "MT": "Mato Grosso",
             "MS": "Mato Grosso do Sul", "MG": "Minas Gerais", "PA": "Para", "PB": "Paraiba", "PR": "Parana",
             "PE": "Pernambuco", "PI": "Piaui", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
             "RS": "Rio Grande do Sul", "RO": "Rondonia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "Sao Paulo",
             "SE": "Sergipe", "TO": "Tocantins"}


def main():
    cfg = load_env()
    questions = {q["id"]: q for q in json.loads((ROOT / "questions" / "questions.json").read_text(encoding="utf-8"))}
    jobs = []
    for cond in CONDITIONS:
        for f in sorted((RESULTS / cond).glob("*.json")) if (RESULTS / cond).exists() else []:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec["id"] in questions:
                jobs.append((questions[rec["id"]], rec))
    rows = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge_one, q, rec, cfg): (q, rec) for q, rec in jobs if not rec.get("error")}
        judged = {}
        for f in cf.as_completed(futs):
            q, rec = futs[f]
            judged[(rec["condition"], q["id"])] = f.result()
    for q, rec in jobs:
        sub = rec.get("submission") or {}
        j = judged.get((rec["condition"], q["id"]), {})
        final = sub.get("value") if q["answer_type"] == "number" and num(sub.get("value")) is not None else j.get("extracted")
        ok, slip, naive_hit = correct(q, final) if not rec.get("error") else (False, False, False)
        rb = j.get("rubric", {})
        rows.append({"condition": rec["condition"], "id": q["id"], "dataset": q["dataset"], "difficulty": q["difficulty"],
                     "category": q["category"], "trap": q["trap"], "has_trap": q["trap"] != "none",
                     "multi_table": q["dataset"] in ("olist", "f1db", "movielens", "shop"),
                     "correct": int(ok), "unit_slip": int(slip), "fell_into_trap": int(naive_hit),
                     "final": final, "reference": q["reference"], "harness_error": int(bool(rec.get("error"))),
                     "method_appropriate": rb.get("method_appropriate"), "caveats_stated": rb.get("caveats_stated"),
                     "no_overclaiming": rb.get("no_overclaiming"), "steps": rec.get("steps"),
                     "code_errors": rec.get("code_errors"), "skills_loaded": ",".join(rec.get("skills_loaded") or []),
                     "prompt_tokens": rec.get("prompt_tokens"), "completion_tokens": rec.get("completion_tokens"),
                     "seconds": rec.get("seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "scores.csv", index=False)
    print(df.groupby("condition")[["correct", "fell_into_trap", "method_appropriate", "caveats_stated",
                                   "no_overclaiming"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
