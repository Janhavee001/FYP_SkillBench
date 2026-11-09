"""
Run Data Formulator 2 (arXiv 2408.16119) against the shared questions.

Fidelity note: `repos/data-formulator` is the current main branch (0.8.0b1),
which is several architectural generations past the DF2 paper. This runner
targets `repos/data-formulator-v2`, pinned at tag 0.2, which is the release the
paper describes. Benchmarking main and calling it "Data Formulator 2" would
have been wrong.

DF2 is a UI-first tool: the human drags fields onto a chart shelf and the agent
writes the transformation code that produces the fields the chart needs. There
is no headless "answer this question" entry point, so we drive
PythonDataRecAgent directly -- the agent that takes a natural-language goal and
both recommends the chart encoding and generates the transform. That is the
closest faithful mapping from our question format onto DF2's design.

Worth recording while reading the scores: DF2 v0.2 has no repair or retry path
anywhere in its agent code (verified by grep), and its chart vocabulary is
restricted by prompt to point/bar/line/boxplot.

Run under the DF2 venv:
    venvs/df2/Scripts/python.exe harness/runners/run_data_formulator2.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.common import (  # noqa: E402
    Record, load_questions, normalise_chart, require_key, write_results,
)

ARCH = "multi-agent, UI-driven (concept-derive + data-transform + data-rec " \
       "agents); no validator, no repair loop"
NATIVE_EVAL = ("No quantitative benchmark in the paper. Evaluated by a user "
               "study (n=10) on task completion and subjective workload, plus "
               "qualitative walkthroughs of the data-threads interaction.")

# DF2's prompt restricts recommendations to these four marks.
DF2_CHART_VOCAB = {"point", "bar", "line", "boxplot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rows", type=int, default=400,
                    help="rows sampled per table (DF2 inlines rows into the prompt)")
    args = ap.parse_args()

    cfg = require_key(
        "venvs/df2/Scripts/python.exe harness/runners/run_data_formulator2.py")

    import pandas as pd
    from data_formulator.agents.agent_py_data_rec import PythonDataRecAgent
    from data_formulator.agents.client_utils import Client

    # DF2's Client only routes through litellm for non-"openai" endpoints. With
    # endpoint="openai" it builds a raw openai.OpenAI client and forwards the
    # model string verbatim (see client_utils.py get_completion), so the model
    # id must be exactly what the endpoint expects -- "openai/gpt-oss-120b" for
    # OpenRouter, "gpt-4o-mini" for OpenAI. No provider prefix, and none is
    # stripped. api_base has to be passed explicitly; Client only sets it when
    # given one.
    model_id = cfg["bench_model"]
    client = Client(endpoint="openai", model=model_id,
                    api_key=cfg["openai_api_key"],
                    api_base=cfg["openai_base_url"])
    agent = PythonDataRecAgent(client=client)
    print(f"[df2] provider={cfg['provider']} model={model_id} "
          f"base_url={cfg['openai_base_url']}")

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    tables_cache: dict[str, list] = {}
    records = []
    for q in questions:
        rec = Record(id=q["id"], dataset=q["dataset"], question=q["question"],
                     ground_truth=q["reference_answer"],
                     expected_method=q["expected_method"],
                     acceptable_methods=q["acceptable_methods"],
                     expected_chart=q["expected_chart"],
                     tolerance=q["tolerance"], kind=q["kind"])
        try:
            if q["dataset"] not in tables_cache:
                df = pd.read_csv(q["csv_path"])
                # DF2 serialises rows into the prompt; sample to stay in budget.
                sample = df if len(df) <= args.rows else df.sample(
                    args.rows, random_state=cfg["seed"])
                tables_cache[q["dataset"]] = [{
                    "name": q["dataset"],
                    "rows": json.loads(sample.to_json(orient="records")),
                }]
            input_tables = tables_cache[q["dataset"]]

            candidates = agent.run(input_tables, q["question"], n=1)
            if not candidates:
                rec.error = "data formulator returned no candidate"
                records.append(rec)
                continue

            cand = candidates[0]
            if not isinstance(cand, dict):
                rec.error = f"data formulator returned a {type(cand).__name__}"
                rec.raw["candidate"] = str(cand)[:2000]
                records.append(rec)
                print(f"  {q['id']}: non-dict candidate")
                continue

            rec.code = cand.get("code", "") or ""
            status = cand.get("status", "")
            rec.executed = status == "ok"
            rec.execution_error = None if rec.executed else str(cand.get("content"))[:2000]

            # DF2 sets refined_goal to the first JSON block it can parse out of
            # the reply (agent_py_data_rec.py), which is not guaranteed to be an
            # object -- a bare JSON string or list comes back just as happily.
            # Keep whatever it was for the record, but only treat a dict as a
            # goal, otherwise this raises and loses an otherwise-good result.
            raw_goal = cand.get("refined_goal")
            rec.raw["refined_goal"] = raw_goal
            goal = raw_goal if isinstance(raw_goal, dict) else {}
            if raw_goal is not None and not isinstance(raw_goal, dict):
                rec.raw["refined_goal_not_an_object"] = type(raw_goal).__name__
            rec.raw["status"] = status
            chart_type = (goal.get("chart_type") or "").strip().lower()
            rec.declared_chart = chart_type or (", ".join(normalise_chart(rec.code))
                                                or None)
            rec.raw["chart_type_in_df2_vocab"] = chart_type in DF2_CHART_VOCAB
            rec.raw["visualization_fields"] = goal.get("visualization_fields")

            # The transformed table IS DF2's answer; surface it as stdout so the
            # rubric can look for ground-truth values in it.
            #
            # `content` is only a {"rows": [...]} dict on success. When the
            # sandbox raises, DF2 puts a plain error STRING there instead
            # (agent_py_data_rec.py), so it must never be treated as a dict
            # unconditionally -- doing so turns a legitimate "DF2 failed to
            # execute" record into a harness crash and destroys the real signal.
            content = cand.get("content")
            rows = content.get("rows", []) if isinstance(content, dict) else []
            if rec.executed:
                rec.numeric_outputs = {"transformed_rows": rows[:200]}
                rec.stdout = pd.DataFrame(rows).to_string(max_rows=60) if rows else ""
                rec.artifact = None

            # DF2's narrative channel is the recommendation/reason text.
            rec.final_answer = " ".join(str(goal.get(k, "")) for k in
                                        ("recommendation", "reason", "instruction")).strip()

            print(f"  {q['id']}: status={status} chart={rec.declared_chart} "
                  f"rows={len(rows)}")
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.raw["traceback"] = traceback.format_exc()[-1500:]
            print(f"  {q['id']}: HARNESS ERROR {rec.error}")
        records.append(rec)

    write_results("data_formulator2",
                  "Data Formulator 2 (Wang et al., 2024)", ARCH, NATIVE_EVAL,
                  records,
                  meta={"model": cfg["bench_model"],
                        "provider": cfg["provider"],
                        "model_id_sent": model_id,
                        "base_url": cfg["openai_base_url"],
                        "repo_tag": "0.2 (paper-faithful; main is 0.8.0b1)",
                        "entry_point": "PythonDataRecAgent.run",
                        "rows_sampled_per_table": args.rows,
                        "chart_vocabulary": sorted(DF2_CHART_VOCAB)})


if __name__ == "__main__":
    main()
