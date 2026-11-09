"""
Read every results/*.json, apply the shared rubric, emit one comparison table.

Outputs results/comparison.csv and prints the same table to the terminal, plus
three secondary views that the headline table cannot show:
  * per-question-kind scores (stat vs viz), which is what separates "this tool
    is bad" from "this tool was asked the wrong kind of question"
  * self-correction counts, the direct RQ3 evidence
  * a status line for any system that did not produce results, so a missing row
    is visibly a missing run rather than a silent zero

    venvs/harness/Scripts/python.exe harness/compare.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness.common import RESULTS  # noqa: E402
from harness.rubric import DIMENSIONS, score_system  # noqa: E402

# ---------------------------------------------------------------------------
# Architecture taxonomy.
#
# These tags are not copied from the papers' self-descriptions -- they were set
# by reading each cloned codebase, because several papers describe an
# architecture the shipped code does not implement. The `evidence` field records
# where each judgement came from so it can be checked.
# ---------------------------------------------------------------------------
ARCH = {
    "baseline_plain": {
        "type": "single-prompt",
        "n_agents": 1, "validator": "no", "skill_registry": "no",
        "self_correction": "no",
        "evidence": "control defined in harness/runners/run_baseline.py",
    },
    "baseline_selfcheck": {
        "type": "single-prompt + repair",
        "n_agents": 1, "validator": "no", "skill_registry": "no",
        "self_correction": "yes (1 execution-feedback pass)",
        "evidence": "control defined in harness/runners/run_baseline.py",
    },
    "statabench": {
        "type": "tool-using agent over skill registry",
        "n_agents": 1, "validator": "no", "skill_registry": "yes (~36 MCP tools)",
        "self_correction": "no (ReAct retries only)",
        "evidence": "mcp_scripts/mcp_server.py exposes ~36 stat tools; "
                    "mcp_pipeline.py builds a langgraph create_react_agent",
    },
    "datascibench": {
        "type": "multi-agent planner-executor",
        "n_agents": "planner + executor (MetaGPT roles)",
        "validator": "no (reflection retry, opt-in)",
        "skill_registry": "no (free-form code)",
        "self_correction": "optional (--use_reflection / --max_retry)",
        "evidence": "experiments/run_examples.py flags; role/sci_data_interpreter",
    },
    "lida": {
        "type": "multi-stage pipeline (single agent)",
        "n_agents": 1,
        "validator": "partial (evaluate()/repair() exist but are not automatic)",
        "skill_registry": "no (free-form viz code)",
        "self_correction": "on demand (repair() must be called explicitly)",
        "evidence": "lida/components/manager.py: summarize/goals/visualize/"
                    "execute/repair/evaluate are separate public calls",
    },
    "data_formulator2": {
        "type": "multi-agent (UI-driven)",
        "n_agents": "3+ (concept-derive, transform, data-rec)",
        "validator": "no",
        "skill_registry": "no (free-form transform code)",
        "self_correction": "no",
        "evidence": "grep for repair/retry/self-correct across "
                    "data-formulator-v2 agents/ returns nothing",
    },
    "nvagent": {
        "type": "multi-agent",
        "n_agents": 3,
        "validator": "yes (dedicated Validator agent)",
        "skill_registry": "no (VQL -> code translation)",
        "self_correction": "yes (execution-feedback refine loop, MAX_ROUND=3)",
        "evidence": "core/agents.py Validator.talk sets message['fixed'] and "
                    "loops back on _is_need_refine; core/const.py MAX_ROUND=3",
    },
}

ORDER = ["baseline_plain", "baseline_selfcheck", "statabench", "datascibench",
         "lida", "data_formulator2", "nvagent"]

SHORT = {"method_selection": "method", "code_executability": "exec",
         "output_correctness": "correct", "interpretation_quality": "interp"}


def fmt(v, width=7):
    if v is None:
        return "n/a".rjust(width)
    return f"{v:.3f}".rjust(width)


def render(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    line = "-+-".join("-" * w for w in widths)
    out = [" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
           line]
    out += [" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers)))
            for r in rows]
    return "\n".join(out)


def wrap(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true",
                    help="force the heuristic interpretation judge")
    ap.add_argument("--width", type=int, default=46,
                    help="truncation width for long text columns")
    args = ap.parse_args()

    scored, missing = {}, []
    for name in ORDER:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            missing.append(name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scored[name] = score_system(payload,
                                    use_llm=False if args.no_llm else None)

    if not scored:
        print("No results/*.json found. Run the runners in harness/runners/ first.")
        print("Missing:", ", ".join(missing))
        return

    judge = next(iter(scored.values()))["judge"]

    # ----------------------------- main table ------------------------------
    headers = ["paper", "architecture", "n", "native"] + \
              [SHORT[d] for d in DIMENSIONS] + ["overall"]
    rows, csv_rows = [], []
    for name in ORDER:
        s = scored.get(name)
        if not s:
            continue
        a = ARCH[name]
        arch_label = a["type"]
        if a["validator"].startswith("yes"):
            arch_label += " + validator"
        elif a["validator"].startswith("partial"):
            arch_label += " + opt. validator"
        rows.append([
            wrap(s["paper"] or name, args.width),
            wrap(arch_label, 40),
            s["n_records"],
            fmt(s["meta"].get("native_accuracy")
                or s["meta"].get("native_completion_rate")),
            *[fmt(s["dimension_scores"][d]) for d in DIMENSIONS],
            fmt(s["overall"]),
        ])
        csv_rows.append({
            "system": name,
            "paper": s["paper"],
            "architecture_type": a["type"],
            "n_agents": a["n_agents"],
            "includes_validator": a["validator"],
            "skill_registry": a["skill_registry"],
            "self_correction": a["self_correction"],
            "architecture_evidence": a["evidence"],
            "native_evaluation": s["native_evaluation"],
            "native_score": (s["meta"].get("native_accuracy")
                             or s["meta"].get("native_completion_rate")),
            "n_records": s["n_records"],
            **{f"rubric_{d}": s["dimension_scores"][d] for d in DIMENSIONS},
            **{f"coverage_{d}": s["dimension_coverage"][d] for d in DIMENSIONS},
            "rubric_overall": s["overall"],
            "interpretation_judge": s["judge"],
            "model": s["meta"].get("model"),
        })

    print()
    print("=" * 118)
    print(f"  BENCHMARK COMPARISON  (rubric 0-1; interpretation judge: {judge})")
    print("=" * 118)
    print(render(rows, headers))
    print("  n = questions attempted. A dimension shows n/a when it is not "
          "measurable for that\n  system, never 0 -- see the coverage table "
          "below for how many items each score rests on.")

    # ------------------------- coverage -----------------------------------
    # Without this, a dimension scored on 4 of 16 items looks identical to one
    # scored on all 16, and "not measurable" gets read as "scored badly".
    cov_rows = []
    for name in ORDER:
        s = scored.get(name)
        if not s:
            continue
        cov_rows.append([
            wrap(s["paper"] or name, 34),
            *[f"{s['dimension_coverage'][d]}/{s['n_records']}" for d in DIMENSIONS],
        ])
    print()
    print("  Items each dimension was actually scored on")
    print(render(cov_rows, ["paper"] + [SHORT[d] for d in DIMENSIONS]))

    # -------------------- per-kind breakdown (stat vs viz) -----------------
    kind_rows = []
    for name in ORDER:
        s = scored.get(name)
        if not s:
            continue
        row = [wrap(s["paper"] or name, 34)]
        for kind in ("stat", "viz", "mixed"):
            vals = [r["overall"] for r in s["records"]
                    if r.get("kind") == kind and r["overall"] is not None]
            row.append(fmt(sum(vals) / len(vals) if vals else None))
        kind_rows.append(row)
    if any(r[1:] != ["    n/a"] * 3 for r in kind_rows):
        print()
        print("  Overall score by question kind "
              "(separates capability gaps from category mismatch)")
        print(render(kind_rows, ["paper", "stat", "viz", "mixed"]))

    # ------------------------ self-correction (RQ3) ------------------------
    sc_rows = []
    for name in ORDER:
        s = scored.get(name)
        if not s:
            continue
        recs = s["records"]
        attempts = sum(r.get("repair_attempts", 0) for r in recs)
        fixed = sum(1 for r in recs if r.get("self_corrected"))
        execs = [r["scores"]["code_executability"] for r in recs
                 if r["scores"]["code_executability"] is not None]
        sc_rows.append([
            wrap(s["paper"] or name, 34),
            ARCH[name]["self_correction"],
            attempts, fixed,
            fmt(sum(1 for e in execs if e == 1.0) / len(execs) if execs else None),
        ])
    print()
    print("  Self-correction behaviour (RQ3)")
    print(render(sc_rows, ["paper", "mechanism", "repair attempts",
                           "repairs that fixed it", "clean-exec rate"]))

    # ------------------------------- csv -----------------------------------
    out = RESULTS / "comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n[write] {out}")

    detail = RESULTS / "comparison_detail.json"
    detail.write_text(json.dumps(scored, indent=2, default=str), encoding="utf-8")
    print(f"[write] {detail}")

    if missing:
        print("\n  NOT RUN (no results file -- absent from the table above, "
              "not scored as zero):")
        for m in missing:
            print(f"    - {m}: results/{m}.json missing")

    if judge == "heuristic":
        print("\n  NOTE: interpretation_quality used the keyword heuristic, not "
              "the LLM judge.\n        Set OPENAI_API_KEY in .env and re-run for "
              "StatABench-anchored judging.")


if __name__ == "__main__":
    main()
