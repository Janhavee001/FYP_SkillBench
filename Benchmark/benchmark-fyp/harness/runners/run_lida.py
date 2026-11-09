"""
Run LIDA (arXiv 2303.02927) against the shared datasets/questions.

LIDA's pipeline is summarize -> goal -> visualize -> (execute) -> explain, with
an optional repair() step. We drive it question-by-question, treating each of
our natural-language questions as a LIDA Goal, and we deliberately exercise
repair() on failures because LIDA's self-repair is exactly the kind of
validation step RQ3 asks about -- so we log whether repair changed the outcome.

Must run under the LIDA venv (it pins an old llmx/openai stack):
    venvs/lida/Scripts/python.exe harness/runners/run_lida.py
"""
from __future__ import annotations

import argparse
import base64
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from harness.common import (  # noqa: E402
    RESULTS, Record, load_env, load_questions, normalise_chart, require_key,
    write_results,
)

ARCH = "single-agent, multi-stage pipeline (summarizer -> goal explorer -> " \
       "viz generator -> executor -> self-evaluator/repairer)"
NATIVE_EVAL = ("Self-evaluation: an LLM scores generated visualisation code on "
               "6 dimensions (code accuracy, data transformation, goal "
               "compliance, visualization type, data encoding, aesthetics), "
               "plus a visualization error rate (VER) over generated charts. "
               "No external ground-truth answers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--library", default="seaborn")
    ap.add_argument("--no-repair", action="store_true",
                    help="disable LIDA's repair() step (ablation for RQ3)")
    args = ap.parse_args()

    cfg = require_key("venvs/lida/Scripts/python.exe harness/runners/run_lida.py")

    from lida import Manager, TextGenerationConfig, llm
    from lida.datamodel import Goal

    textgen_config = TextGenerationConfig(n=1, temperature=cfg["temperature"],
                                          model=cfg["bench_model"], use_cache=False)
    manager = Manager(text_gen=llm("openai", api_key=cfg["openai_api_key"]))

    charts_dir = RESULTS / "artifacts" / "lida"
    charts_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    # Summaries are per-dataset, so compute once and reuse across questions.
    summaries = {}
    records = []
    for q in questions:
        rec = Record(id=q["id"], dataset=q["dataset"], question=q["question"],
                     ground_truth=q["reference_answer"],
                     expected_method=q["expected_method"],
                     acceptable_methods=q["acceptable_methods"],
                     expected_chart=q["expected_chart"],
                     tolerance=q["tolerance"], kind=q["kind"])
        try:
            if q["dataset"] not in summaries:
                summaries[q["dataset"]] = manager.summarize(
                    q["csv_path"], summary_method="default",
                    textgen_config=textgen_config)
            summary = summaries[q["dataset"]]

            goal = Goal(question=q["question"], visualization=q["question"],
                        rationale="")
            charts = manager.visualize(summary=summary, goal=goal,
                                       textgen_config=textgen_config,
                                       library=args.library, return_error=True)
            if not charts:
                rec.error = "lida returned no chart"
                records.append(rec)
                print(f"  {q['id']}: no chart returned")
                continue

            chart = charts[0]
            rec.code = chart.code or ""
            rec.executed = bool(chart.status)
            rec.execution_error = None if chart.status else str(chart.error)

            # --- LIDA's own repair loop: the RQ3-relevant behaviour ----------
            if not chart.status and not args.no_repair:
                rec.repair_attempts = 1
                try:
                    repaired = manager.repair(
                        code=chart.code, goal=goal, summary=summary,
                        feedback=str(chart.error),
                        textgen_config=textgen_config, library=args.library,
                        return_error=True)
                    if repaired:
                        r = repaired[0]
                        if r.status and not chart.status:
                            rec.self_corrected = True
                        chart = r
                        rec.code = r.code or rec.code
                        rec.executed = bool(r.status)
                        rec.execution_error = None if r.status else str(r.error)
                except Exception as exc:  # noqa: BLE001
                    rec.execution_error = f"{rec.execution_error} | repair failed: {exc}"

            if chart.raster:
                png = charts_dir / f"{q['id']}.png"
                png.write_bytes(base64.b64decode(chart.raster))
                rec.artifact = str(png)

            rec.declared_chart = ", ".join(normalise_chart(rec.code)) or None

            # LIDA's explain() is its narrative channel; use it for the
            # interpretation dimension so the comparison is not unfair to a
            # tool whose main output is a picture.
            try:
                expl = manager.explain(code=rec.code,
                                       textgen_config=textgen_config,
                                       library=args.library)
                parts = []
                for group in expl or []:
                    for item in (group if isinstance(group, list) else [group]):
                        if isinstance(item, dict):
                            parts.append(f"{item.get('section', '')}: "
                                         f"{item.get('explanation', '')}")
                rec.final_answer = "\n".join(parts).strip()
            except Exception as exc:  # noqa: BLE001
                rec.final_answer = ""
                rec.raw["explain_error"] = str(exc)

            rec.raw["chart_status"] = bool(chart.status)
            print(f"  {q['id']}: executed={rec.executed} "
                  f"chart={rec.declared_chart} repaired={rec.self_corrected}")

        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.raw["traceback"] = traceback.format_exc()[-1500:]
            print(f"  {q['id']}: HARNESS ERROR {rec.error}")

        records.append(rec)

    write_results("lida", "LIDA (Dibia, 2023)", ARCH, NATIVE_EVAL, records,
                  meta={"model": cfg["bench_model"], "library": args.library,
                        "provider": cfg["provider"],
                        "base_url": cfg["openai_base_url"],
                        "repair_enabled": not args.no_repair})


if __name__ == "__main__":
    main()
