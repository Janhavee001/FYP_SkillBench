"""
Run StatABench (arXiv 2606.22977) natively and record its own scores.

Unlike the three systems, StatABench is a benchmark: it defines its own data
(Stat-Closed, 404 items across 18 statistical topics) and its own scoring
(evaluation/eval.py -> extract "The answer is <x>" then match by answer_type).
We run its MCP-agent pipeline with an OpenAI model as the model under test,
then invoke its own evaluator, then translate the graded rows into our shared
Record format so rubric.py can score it on the same four dimensions as
everything else.

Two things worth knowing about the native pipeline, both reflected in the
output:
  * mcp_pipeline.py skips every row with code == 0, so the MCP track covers
    198 of the 404 Stat-Closed items (the ones that require tool use). The
    206 pure-theory items are not attempted by this track.
  * The MCP server exposes ~36 hand-written statistical tools. Selecting the
    right one is the task -- which makes this the closest thing in the five
    papers to an explicit skill registry, and the main reason it belongs in
    this comparison.

Run under the StatABench venv:
    venvs/statabench/Scripts/python.exe harness/runners/run_statabench.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
STATABENCH = ROOT / "repos" / "StatABench"
sys.path.insert(0, str(ROOT))

from harness.common import (  # noqa: E402
    RESULTS, Record, load_env, normalise_method, require_key, write_results,
)

ARCH = "tool-using ReAct agent over an MCP registry of ~36 statistical tools " \
       "(benchmark also ships AutoGen / CrewAI / Qwen-Agent / SmolAgents runners)"
NATIVE_EVAL = ("Stat-Closed: 404 items, 18 topics, answer-type-aware matching "
               "(choice / decision / numeric / open) after extracting 'The "
               "answer is <x>'; numeric compared at 1e-3. Stat-Open: 30 "
               "modelling tasks graded by an LLM-as-Judge across structural "
               "coherency, data/modeling/analysis groundedness, result & bias "
               "analysis, innovativeness and practical science.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="gpt4omini",
                    help="key in keys.json / agent_map (default gpt4omini)")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full 404-item Stat-Closed set")
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse an existing generation file and only re-score")
    ap.add_argument("--track", choices=["tool", "all"], default="tool",
                    help="tool = the native MCP track (198 code==1 items, what "
                         "mcp_pipeline.py attempts as shipped); all = every "
                         "Stat-Closed item, including the 206 theory questions")
    ap.add_argument("--shards", type=int, default=1,
                    help="parallel pipeline processes (each spawns its own MCP "
                         "tool server)")
    ap.add_argument("--resume", action="store_true",
                    help="keep answers already in the generation file; only "
                         "(re)generate items with no answer or an 'Error:' answer")
    args = ap.parse_args()

    cfg = require_key(
        "venvs/statabench/Scripts/python.exe harness/runners/run_statabench.py")

    # keys.json must exist before the pipeline imports get_model_cfg.
    # Regenerate every run rather than only when missing: a keys.json left over
    # from an earlier provider or model would otherwise silently override .env
    # and send the whole run to the wrong endpoint.
    subprocess.run([sys.executable, str(ROOT / "harness" / "setup_keys.py")],
                   check=True)

    # mcp_pipeline.py spawns the tool server with StdioServerParameters(
    # command="python"), which resolves through PATH. Run from an unactivated
    # venv that finds the *system* interpreter, which lacks statsmodels, pymc,
    # pingouin and the rest -- so the server dies and every tool call fails.
    # Putting this interpreter's directory first in PATH makes the bare "python"
    # resolve to the venv without editing the repo.
    env = dict(os.environ)
    env["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    # StatABench's pipeline prints progress with emoji and writes model replies
    # straight to a cp1252 stdout on Windows. Without this every single response
    # comes back as the literal string
    #   "Error: 'charmap' codec can't encode character '⏳'"
    # and the benchmark scores 0% for an encoding reason rather than a
    # statistical one. harness.common forces UTF-8 for processes that import it;
    # a spawned script needs it in the environment.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    work = RESULTS / "artifacts" / "statabench"
    work.mkdir(parents=True, exist_ok=True)
    src = STATABENCH / "data" / "stat-closed.json"
    gen = work / f"mcp_{args.model_key}.json"
    response_col = f"mcp_response_{args.model_key}"

    if not args.skip_generation:
        items = json.loads(src.read_text(encoding="utf-8"))
        if args.track == "tool":
            items = [r for r in items if r.get("code") == 1]
        if args.limit:
            # stat-closed.json is ordered with the pure-theory items first, so
            # a naive items[:N] under the tool track would select rows the
            # pipeline never attempts. Subset after filtering instead.
            items = items[:args.limit]
        print(f"[statabench] track={args.track}: {len(items)} items selected")

        # mcp_pipeline.py skips every row with code == 0. To attempt the theory
        # questions with the same agent, mark them code=1 for generation only.
        # get_mcp_prompt() builds an identical prompt for a code==1 row whose
        # dataset is 0 (every theory row), and the original flag is restored
        # before scoring because eval_utils uses it to pick the ground-truth
        # column.
        orig_code = {str(r["id"]): r.get("code") for r in items}
        for r in items:
            r["code"] = 1

        done = {}
        if args.resume and gen.exists():
            for r in json.loads(gen.read_text(encoding="utf-8")):
                resp = r.get(response_col)
                if resp and not str(resp).startswith("Error"):
                    done[str(r["id"])] = resp
        pending = [r for r in items if str(r["id"]) not in done]
        n = max(1, min(args.shards, len(pending) or 1))
        print(f"[statabench] {len(done)} answers kept, {len(pending)} to "
              f"generate across {n} shard(s)")

        answers = dict(done)
        failed = []
        if pending:
            procs = []
            for i in range(n):
                if i:
                    # Stagger start-up: the MCP tool server imports pymc ->
                    # arviz, which writes a once-a-day stamp file on import;
                    # two servers starting in the same instant race on it and
                    # one dies with "Connection closed" before any question.
                    time.sleep(20)
                shard_path = work / f"shard_{i}.json"
                shard_path.write_text(
                    json.dumps(pending[i::n], ensure_ascii=False, indent=2),
                    encoding="utf-8")
                log = (work / f"generation_shard_{i}.log").open(
                    "w", encoding="utf-8")
                procs.append((shard_path, log, subprocess.Popen(
                    [sys.executable,
                     str(STATABENCH / "mcp_scripts" / "mcp_pipeline.py"),
                     "--model", args.model_key, "--begin_index", "0",
                     "--input_path", str(shard_path),
                     "--output_path", str(shard_path)],
                    cwd=str(STATABENCH), stdout=log, stderr=subprocess.STDOUT,
                    text=True, env=env)))
            for shard_path, log, proc in procs:
                rc = proc.wait()
                log.close()
                if rc != 0:
                    failed.append(shard_path.name)
                # Merge whatever the shard saved, even on failure: the pipeline
                # checkpoints every few rows, and --resume picks up the rest.
                try:
                    for r in json.loads(shard_path.read_text(encoding="utf-8")):
                        if r.get(response_col):
                            answers[str(r["id"])] = r[response_col]
                except (json.JSONDecodeError, OSError):
                    failed.append(shard_path.name + " (unreadable)")

        for r in items:
            r["code"] = orig_code[str(r["id"])]
            if str(r["id"]) in answers:
                r[response_col] = answers[str(r["id"])]
        gen.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        n_err = sum(1 for r in items
                    if str(r.get(response_col, "")).startswith("Error"))
        n_missing = sum(1 for r in items if not r.get(response_col))
        print(f"[statabench] generation file: {len(items)} items, "
              f"{n_missing} unanswered, {n_err} 'Error:' answers")
        if failed:
            print(f"[statabench] GENERATION FAILED in {failed} -- see "
                  f"{work}/generation_shard_*.log; re-run with --resume")
            raise SystemExit(1)
        if n_missing or n_err:
            print("[statabench] WARNING: re-run with --resume to retry the "
                  "unanswered / errored items before quoting a score")

    # --- native scoring ----------------------------------------------------
    print("[statabench] running native evaluation/eval.py")
    p = subprocess.run(
        [sys.executable, str(STATABENCH / "evaluation" / "eval.py"),
         "--input_path", str(gen), "--response_col", response_col],
        cwd=str(STATABENCH / "evaluation"), capture_output=True, text=True,
        timeout=21600, env=env)
    (work / "evaluation.log").write_text(
        (p.stdout or "") + "\n=== STDERR ===\n" + (p.stderr or ""),
        encoding="utf-8")
    if p.returncode != 0:
        print("[statabench] NATIVE EVAL FAILED -- see "
              f"{work / 'evaluation.log'}")
        print((p.stderr or "")[-2500:])
        raise SystemExit(1)

    graded = json.loads(gen.read_text(encoding="utf-8"))
    match_col = f"{response_col}_match"

    attempted = [r for r in graded if r.get(response_col)]
    correct = [r for r in attempted if r.get(match_col) is True]
    native_accuracy = len(correct) / len(attempted) if attempted else None

    def acc(rows):
        rows = list(rows)
        hits = sum(1 for r in rows if r.get(match_col) is True)
        return (round(hits / len(rows), 4) if rows else None), len(rows)
    acc_tool, n_tool = acc(r for r in attempted if r.get("code") == 1)
    acc_theory, n_theory = acc(r for r in attempted if r.get("code") != 1)
    by_type = {}
    for t in sorted({str(r.get("answer_type")) for r in attempted}):
        a, k = acc(r for r in attempted if str(r.get("answer_type")) == t)
        by_type[t] = {"accuracy": a, "n": k}

    records = []
    for row in graded:
        resp = row.get(response_col) or ""
        if not resp:
            continue  # not attempted (code == 0 under --track tool)
        gt = row.get("ground truth2") if row.get("code") == 1 else row.get("ground truth")
        matched = row.get(match_col)
        rec = Record(
            id=str(row.get("id")), dataset="stat-closed",
            question=row.get("question", ""),
            declared_method=", ".join(normalise_method(resp)) or None,
            code="", executed=bool(row.get("code") == 1),
            stdout="", final_answer=resp,
            ground_truth={"answer": gt} if gt is not None else {},
            expected_method=[], acceptable_methods=[], expected_chart=[],
            tolerance=0.001, kind="stat",
            raw={"task": row.get("task"), "answer_type": row.get("answer_type"),
                 "native_match": matched,
                 "extracted": row.get(f"{response_col}_extract")})
        # StatABench has already adjudicated correctness far more precisely than
        # a generic text scan could, so carry its verdict straight through
        # rather than re-deriving it.
        rec.numeric_outputs = {"native_correct": 1.0 if matched is True else 0.0}
        records.append(rec)

    write_results(
        "statabench", "StatABench (Zhu et al., 2026)", ARCH, NATIVE_EVAL, records,
        meta={"model": cfg["bench_model"], "provider": cfg["provider"],
              "base_url": cfg["openai_base_url"],
              "track": "Stat-Closed (MCP agent), --track "
                       + ("all" if n_theory else "tool"),
              "accuracy_tool_items": acc_tool, "n_tool_items": n_tool,
              "accuracy_theory_items": acc_theory, "n_theory_items": n_theory,
              "accuracy_by_answer_type": by_type,
              "native_accuracy": round(native_accuracy, 4) if native_accuracy else None,
              "n_attempted": len(attempted), "n_total_in_file": len(graded),
              "native_score_definition":
                  "answer-type-aware match rate over attempted items",
              "generation_logs": str(work / "generation_shard_*.log"),
              "note": ("theory items (code==0) are attempted only under "
                       "--track all; as shipped, mcp_pipeline.py skips them")})

    print(f"[statabench] native accuracy: {native_accuracy} over "
          f"{len(attempted)} attempted items (tool items {acc_tool} over "
          f"{n_tool}, theory items {acc_theory} over {n_theory})")


if __name__ == "__main__":
    main()
