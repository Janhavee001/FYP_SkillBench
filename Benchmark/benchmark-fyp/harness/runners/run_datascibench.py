"""
Run DataSciBench (arXiv 2502.13897) natively and record its own scores.

DataSciBench evaluates a MetaGPT Data-Interpreter agent on 222 task folders
(167 BigCodeBench-derived + 55 curated csv/human/dl tasks) using its
Task-Function-Code (TFC) framework: each task carries a metric.yaml naming the
aggregate functions to check, and completion rate (CR) is computed from whether
each planned step executed and whether its output matches ground truth.

Setup this runner performs for you:
  * writes ~/.metagpt/config2.yaml from OPENAI_API_KEY (MetaGPT reads config
    from the home directory, not the environment)
  * checks for the ground-truth files that must be downloaded from
    huggingface.co/datasets/zd21/DataSciBench into data/<task_id>/gt/ before
    experiments.evaluate can score anything
  * points the repository's VLM chart judge (src/vlm_utils.py) at the
    configured provider -- as shipped it targets a third-party proxy with no
    key, so every VLM-judged metric raised and scored 0

The two task families are scored by two different native evaluators:
  * csv / human / dl  -> experiments.evaluate      (per-step TFC checks, CR)
  * bcb               -> experiments.evaluate_tmc  (exec the extracted
                         task_func against the ground-truth function, CR)

Generation is resumable: experiments.run_examples skips any run whose
logs.txt is already non-empty, so re-running this script only fills gaps.

The --use-reflection / --max-retry flags map onto DataSciBench's own
self-correction knobs and are exposed here because they are the RQ3-relevant
ablation for this system.

Run under the DataSciBench venv:
    venvs/datascibench/Scripts/python.exe harness/runners/run_datascibench.py
    venvs/datascibench/Scripts/python.exe harness/runners/run_datascibench.py \
        --task-id csv_excel_0 --data-types csv          # a single task
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DSB = ROOT / "repos" / "DataSciBench"
sys.path.insert(0, str(ROOT))

from harness.common import (  # noqa: E402
    RESULTS, Record, normalise_chart, normalise_method, require_key,
    write_results,
)

ARCH = "multi-agent planner-executor (MetaGPT Data Interpreter: plan into a " \
       "task DAG, generate + execute code per node, optional reflection retry)"
NATIVE_EVAL = ("Task-Function-Code (TFC) framework: per-task metric.yaml names "
               "the aggregate functions to verify; Completion Rate scores each "
               "planned step on execution success and output-vs-ground-truth "
               "match, with LLM self-consistency plus human verification used "
               "to establish the ground truth.")

# folder prefix for each --data_type accepted by experiments.run_examples
TYPE_PREFIX = {"csv": "csv_excel", "human": "human", "dl": "dl", "bcb": "bcb"}
VLM_MODEL = "openai/gpt-4o-mini"   # the judge model src/vlm_utils.py names


def write_metagpt_config(cfg, config_name: str = "test_config.yaml") -> pathlib.Path:
    """
    Write BOTH config files MetaGPT/DataSciBench expect under ~/.metagpt/.

    MetaGPT itself reads config2.yaml. DataSciBench additionally passes a named
    config through `--config` (default test_config.yaml) into
    Config.from_home(), and src/logs.py opens that same file directly to derive
    the model name for its log directory -- so a missing test_config.yaml is a
    hard FileNotFoundError before any task runs.
    """
    home = pathlib.Path(os.path.expanduser("~")) / ".metagpt"
    home.mkdir(parents=True, exist_ok=True)
    body = ("llm:\n"
            '  api_type: "openai"\n'
            f'  model: "{cfg["bench_model"]}"\n'
            f'  base_url: "{cfg["openai_base_url"]}"\n'
            f'  api_key: "{cfg["openai_api_key"]}"\n'
            "  temperature: 0\n")
    written = []
    for name in ("config2.yaml", config_name):
        path = home / name
        path.write_text(body, encoding="utf-8")
        written.append(path)
    for p in written:
        print(f"[write] {p}")
    return written[-1]


def ground_truth_present() -> tuple[int, int]:
    """How many non-BCB task folders have the downloaded gt/ directory."""
    have = total = 0
    for d in sorted((DSB / "data").iterdir()):
        if not d.is_dir() or d.name.startswith("bcb"):
            continue
        total += 1
        if (d / "gt").is_dir() and any((d / "gt").iterdir()):
            have += 1
    return have, total


def stage_inputs() -> tuple[int, list[str]]:
    """
    Copy each task's input files next to its prompt.

    The prompts tell the agent that "all the input source data is at the `../`
    folder", i.e. data/<task_id>/. The clone ships only prompt.json there; the
    inputs live in the HF download under DataSciBench-data/<task_id>/ alongside
    the ground truth. Without them the agent cannot read its data -- and
    gpt-oss-120b responds by fabricating a "dummy" input file and analysing
    that, which scores as a plausible, entirely meaningless run. Copies only
    what is missing, never overwrites.
    """
    src_root = DSB / "_gt_download" / "DataSciBench-data"
    copied, no_source = 0, []
    for d in sorted((DSB / "data").iterdir()):
        if not d.is_dir() or d.name.startswith("bcb"):
            continue
        src = src_root / d.name
        if not src.is_dir():
            no_source.append(d.name)
            continue
        for item in src.iterdir():
            if item.name in ("prompt.json", ".DS_Store") or (d / item.name).exists():
                continue
            if item.is_dir():
                shutil.copytree(item, d / item.name)
            else:
                shutil.copy2(item, d / item.name)
            copied += 1
    return copied, no_source


def tasks_of(data_type: str) -> list[str]:
    prefix = TYPE_PREFIX[data_type]
    return sorted(d.name for d in (DSB / "data").iterdir()
                  if d.is_dir() and d.name.startswith(prefix)
                  and (d / "prompt.json").exists())


def type_of(task: str) -> str:
    for t, prefix in TYPE_PREFIX.items():
        if task.startswith(prefix):
            return t
    return "?"


def bcb_tmc_path(task: str, model_id: str) -> pathlib.Path:
    # BCB runs keep the full model id ("openai/gpt-oss-120b") as a path, so
    # outputs land in data/<task>/openai/gpt-oss-120b_outputs.jsonl.
    return DSB / "data" / task / f"{model_id}_tmc_results.jsonl"


def outputs_path(task: str, model_id: str) -> pathlib.Path:
    name = model_id if task.startswith("bcb") else model_id.split("/")[-1]
    return DSB / "data" / task / f"{name}_outputs.jsonl"


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="all",
                    help="a single task folder, or 'all'")
    ap.add_argument("--data-types", default="csv,human,dl,bcb",
                    help="comma-separated subset of csv,human,dl,bcb")
    ap.add_argument("--max-runs", type=int, default=3,
                    help="independent runs per task (DataSciBench default 3)")
    ap.add_argument("--shards", type=int, default=4,
                    help="concurrent run_examples processes")
    ap.add_argument("--chunk", type=int, default=6,
                    help="tasks per run_examples process")
    ap.add_argument("--model-id", default=None,
                    help="model id DataSciBench uses for its output paths")
    ap.add_argument("--use-reflection", action="store_true",
                    help="enable DataSciBench's reflection retry (RQ3 ablation)")
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--config", default="test_config.yaml",
                    help="named config DataSciBench passes to Config.from_home")
    ap.add_argument("--bcb-timeout", type=int, default=900,
                    help="seconds allowed for evaluating one BCB task")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-evaluation", action="store_true",
                    help="aggregate existing native results only")
    args = ap.parse_args()

    cfg = require_key(
        "venvs/datascibench/Scripts/python.exe harness/runners/run_datascibench.py")
    model_id = args.model_id or cfg["bench_model"]
    write_metagpt_config(cfg, args.config)

    types = [t.strip() for t in args.data_types.split(",") if t.strip()]
    if args.task_id != "all":
        tasks = [args.task_id]
        types = [type_of(args.task_id)]
    else:
        tasks = [t for dt in types for t in tasks_of(dt)]
    print(f"[datascibench] {len(tasks)} tasks across types {types}, "
          f"{args.max_runs} runs each")

    # DataSciBench's run_examples.py reads its own log back with a bare
    # open(), which on Windows defaults to cp1252 and dies on the non-ASCII
    # content the model produces (the task data here is Chinese). PYTHONUTF8=1
    # makes open() default to UTF-8 for the whole child process;
    # PYTHONIOENCODING covers its stdout. Same root cause as the StatABench
    # encoding failure -- a third distinct process boundary.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # evaluate_tmc exec()s the agent's task_func in-process; many BCB
    # completions end with plt.show(), which under the default TkAgg backend
    # opens a "Figure 1" window and blocks until someone closes it (bcb198
    # hung the whole evaluation for 90 minutes on 2026-09-21). Agg renders
    # off-screen and makes plt.show() a no-op.
    env["MPLBACKEND"] = "Agg"
    # src/vlm_utils.py reads these (see the adaptation note in README.md).
    env["DSB_VLM_BASE_URL"] = cfg["openai_base_url"]
    env["DSB_VLM_API_KEY"] = cfg["openai_api_key"]
    env["DSB_VLM_MODEL"] = VLM_MODEL if cfg["provider"] == "openrouter" \
        else VLM_MODEL.split("/")[-1]

    # BCB runs keep the full model id as a path component, and src/logs.py
    # opens logs/<provider>/<model>_<run>_time.txt without creating the
    # provider folder -- so every BCB run dies with FileNotFoundError before
    # the agent starts unless it already exists.
    (DSB / "logs" / pathlib.PurePosixPath(model_id).parent).mkdir(
        parents=True, exist_ok=True)

    work = RESULTS / "artifacts" / "datascibench"
    (work / "gen_logs").mkdir(parents=True, exist_ok=True)
    (work / "eval_logs").mkdir(parents=True, exist_ok=True)

    copied, no_source = stage_inputs()
    print(f"[datascibench] staged {copied} missing input file(s) from the HF "
          f"download" + (f"; no input folder for {no_source}" if no_source else ""))
    if no_source and any(t != "bcb" for t in types):
        print("[datascibench] WARNING: run harness/fetch_ground_truth.py -- tasks "
              "without their input data will be scored on fabricated inputs")

    have_gt, total_gt = ground_truth_present()
    print(f"[datascibench] ground-truth folders present: {have_gt}/{total_gt}")
    if have_gt == 0 and any(t != "bcb" for t in types):
        print("[datascibench] WARNING: no data/<task_id>/gt/ folders found. "
              "Download them from huggingface.co/datasets/zd21/DataSciBench "
              "(needs HF_TOKEN) or experiments.evaluate will score nothing.")

    # --- generation --------------------------------------------------------
    if not args.skip_generation:
        jobs = []
        for dt in types:
            ts = [t for t in tasks if type_of(t) == dt]
            for i in range(0, len(ts), args.chunk):
                jobs.append((dt, ts[i:i + args.chunk]))
        print(f"[datascibench] generation: {len(jobs)} chunks, "
              f"{args.shards} concurrent")
        lock = threading.Lock()
        finished = [0]

        def run_chunk(k_job):
            k, (dt, ids) = k_job
            cmd = [sys.executable, "-m", "experiments.run_examples",
                   "--data_type", dt, "--task_id", repr(ids),
                   "--max_runs", str(args.max_runs),
                   "--max_retry", str(args.max_retry), "--config", args.config]
            if args.use_reflection:
                cmd += ["--use_reflection"]
            log_path = work / "gen_logs" / f"{dt}_{k:03d}.log"
            t0 = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n")
                log.flush()
                rc = subprocess.run(cmd, cwd=str(DSB), stdout=log,
                                    stderr=subprocess.STDOUT, env=env).returncode
            with lock:
                finished[0] += 1
                print(f"[datascibench] chunk {finished[0]}/{len(jobs)} done "
                      f"({dt}: {ids[0]}..{ids[-1]}, rc={rc}, "
                      f"{(time.time() - t0) / 60:.1f} min)", flush=True)
            return rc

        with concurrent.futures.ThreadPoolExecutor(args.shards) as pool:
            rcs = list(pool.map(run_chunk, enumerate(jobs)))
        if any(rcs):
            print(f"[datascibench] {sum(1 for r in rcs if r)} chunk(s) exited "
                  f"non-zero -- see {work / 'gen_logs'}; re-running resumes")

    # --- native evaluation -------------------------------------------------
    res_csv = DSB / "evaluation_results" / f"{model_id.split('/')[-1]}_results.csv"
    bcb_timeouts = []
    if not args.skip_evaluation:
        if any(t != "bcb" for t in types):
            print("[datascibench] running experiments.evaluate (csv/human/dl)")
            with (work / "eval_logs" / "evaluate.log").open("w", encoding="utf-8") as log:
                p = subprocess.run(
                    [sys.executable, "-m", "experiments.evaluate",
                     "--task_id", args.task_id, "--model_id", model_id],
                    cwd=str(DSB), stdout=log, stderr=subprocess.STDOUT,
                    env=env, timeout=4 * 3600)
            if p.returncode != 0:
                print(f"[datascibench] NATIVE EVAL FAILED -- see "
                      f"{work / 'eval_logs' / 'evaluate.log'}")
                raise SystemExit(1)

        bcb_tasks = [t for t in tasks if t.startswith("bcb")]
        if bcb_tasks:
            # evaluate_tmc exec()s each extracted task_func in-process with no
            # overall timeout, and matches --task_id by substring. Evaluate
            # one task per subprocess so a hanging completion costs one task,
            # not the whole family. Stale results are removed first so a
            # timeout can never leave an old score in place.
            for t in bcb_tasks:
                bcb_tmc_path(t, model_id).unlink(missing_ok=True)
            print(f"[datascibench] running experiments.evaluate_tmc on "
                  f"{len(bcb_tasks)} BCB tasks")
            with (work / "eval_logs" / "evaluate_tmc.log").open("w", encoding="utf-8") as log:
                for i, t in enumerate(bcb_tasks, 1):
                    if bcb_tmc_path(t, model_id).exists():
                        continue  # written as a substring match of an earlier id
                    log.write(f"\n=== {t}\n")
                    log.flush()
                    p = subprocess.Popen(
                        [sys.executable, "-m", "experiments.evaluate_tmc",
                         "--task_id", t, "--model_id", model_id],
                        cwd=str(DSB), stdout=log, stderr=subprocess.STDOUT,
                        env=env)
                    try:
                        p.wait(timeout=args.bcb_timeout)
                    except subprocess.TimeoutExpired:
                        # A venv python.exe on Windows is a launcher that
                        # spawns the real interpreter; killing only the
                        # launcher leaves the task_func running. Kill the tree.
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL)
                        else:
                            p.kill()
                        p.wait()
                        bcb_timeouts.append(t)
                        log.write(f"\n=== {t} TIMED OUT after {args.bcb_timeout}s\n")
                    if i % 20 == 0:
                        print(f"[datascibench] evaluate_tmc {i}/{len(bcb_tasks)}",
                              flush=True)

    # --- aggregate -----------------------------------------------------------
    # Non-BCB: evaluation_results/<model>_results.csv has one row per
    # (task, run, metric function) plus one "Completion Rate" row per run.
    #   * `task_name` is the METRIC name; the task is `data_name`.
    #   * per-function rows use a 0/1/2 scale; only rows whose result_type is
    #     "Completion Rate" are the 0-1 CR DataSciBench reports.
    # BCB: data/<task>/<model>_tmc_results.jsonl has one row per run with `cr`.
    # A run that produced nothing scorable counts as 0 (divide by max_runs),
    # so a crashed run cannot silently raise a task's average.
    crs_by_task: dict[str, list[float]] = {t: [] for t in tasks}
    if res_csv.exists():
        for r in csv.DictReader(res_csv.open(encoding="utf-8")):
            if (r.get("result_type") or "").strip() != "Completion Rate":
                continue
            if r.get("data_name") in crs_by_task:
                try:
                    crs_by_task[r["data_name"]].append(float(r["result_cr"]))
                except (TypeError, ValueError):
                    pass
    for t in tasks:
        if t.startswith("bcb"):
            crs_by_task[t] = [float(x.get("cr") or 0)
                              for x in read_jsonl(bcb_tmc_path(t, model_id))]

    records = []
    per_type: dict[str, list[float]] = {}
    for task in tasks:
        vals = crs_by_task.get(task, [])[:args.max_runs]
        cr = sum(vals) / args.max_runs
        per_type.setdefault(type_of(task), []).append(cr)

        runs = read_jsonl(outputs_path(task, model_id))
        plan = runs[0].get("plan", []) if runs else []
        code_text = "\n\n".join(str(s.get("code", "")) for s in plan)
        rec = Record(
            id=task, dataset="datascibench", question=task,
            declared_method=", ".join(normalise_method(code_text)) or None,
            declared_chart=", ".join(normalise_chart(code_text)) or None,
            code=code_text,
            executed=all(s.get("is_success") for s in plan) if plan else bool(cr > 0),
            stdout="\n".join(str(s.get("result", ""))[:500] for s in plan),
            final_answer="\n".join(str(s.get("instruction", "")) for s in plan),
            numeric_outputs={"completion_rate": cr},
            ground_truth={"completion_rate": 1.0},
            expected_method=[], acceptable_methods=[], expected_chart=[],
            tolerance=0.0, kind="stat",
            raw={"data_type": type_of(task),
                 "cr_per_run": vals,
                 "runs_generated": len(runs),
                 "runs_scored": len(vals),
                 "eval_timed_out": task in bcb_timeouts,
                 "plan_steps": len(plan)})
        records.append(rec)

    all_crs = [r.numeric_outputs["completion_rate"] for r in records]
    native_cr = sum(all_crs) / len(all_crs) if all_crs else None
    by_type = {t: {"n_tasks": len(v), "completion_rate": round(sum(v) / len(v), 4)}
               for t, v in per_type.items()}
    unscored = [r.id for r in records if not r.raw["runs_scored"]]
    write_results(
        "datascibench", "DataSciBench (Zhang et al., 2025)", ARCH, NATIVE_EVAL,
        records,
        meta={"model": cfg["bench_model"], "provider": cfg["provider"],
              "base_url": cfg["openai_base_url"],
              "data_types": types,
              "native_completion_rate": round(native_cr, 4) if native_cr is not None else None,
              "completion_rate_by_type": by_type,
              "n_tasks_scored": len(records),
              "max_runs": args.max_runs,
              "n_tasks_with_no_scored_run": len(unscored),
              "tasks_with_no_scored_run": unscored,
              "bcb_eval_timeouts": bcb_timeouts,
              "ground_truth_folders": f"{have_gt}/{total_gt}",
              "vlm_judge_model": env["DSB_VLM_MODEL"],
              "use_reflection": args.use_reflection,
              "max_retry": args.max_retry,
              "cr_definition": "mean over tasks of (sum of per-run CR / max_runs); "
                               "an unscored run counts as 0",
              "results_csv": str(res_csv)})
    print(f"[datascibench] native completion rate: {native_cr} over "
          f"{len(records)} tasks; by type: {by_type}; "
          f"{len(unscored)} task(s) with no scored run")


if __name__ == "__main__":
    main()
