"""
Run the full SkillBench v2 design from one process, so a single semaphore (run.CODE_SLOTS) caps how many
analysis scripts load multi-million-row tables at the same time, while many model calls wait on the API.

Design (per model):  plain x 1 run;  agent, agent_skills, agent_skills_oracle x 3 runs each.
Temperature 0.7; run k uses seed 1000 + k. Jobs are ordered run by run, so a complete run 1 exists early.
Resumable: finished answers are skipped.

    python run_all.py [--models openai/gpt-oss-120b,openai/gpt-oss-20b] [--workers 20] [--ids ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import time

import run as R
from harness.common import load_env  # noqa: E402

DESIGN = [("plain", [1]), ("agent", [1, 2, 3]), ("agent_skills", [1, 2, 3]), ("agent_skills_oracle", [1, 2, 3])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="openai/gpt-oss-120b,openai/gpt-oss-20b")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    base = load_env()
    questions = json.loads(R.QUESTIONS.read_text(encoding="utf-8"))
    if args.ids:
        keep = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in keep]
    catalog, skill_texts = R.skill_catalog()

    jobs = []
    for k in (1, 2, 3):
        for model in args.models.split(","):
            cfg = copy.deepcopy(base)
            cfg.update(bench_model=model, temperature=args.temperature, seed=1000 + k)
            tag = f"v2/{model.split('/')[-1]}/run{k}"
            for cond, runs in DESIGN:
                if k in runs:
                    jobs += [(q, cond, cfg, tag) for q in questions
                             if not (R.RESULTS / tag / cond / f"{q['id']}.json").exists()]
    print(f"{len(jobs)} jobs to run, {args.workers} workers, code slots = {R.CODE_SLOTS._value}", flush=True)
    t0, done, errors = time.time(), 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(R.run_one, q, cond, cfg, catalog, skill_texts, tag) for q, cond, cfg, tag in jobs]
        for f in cf.as_completed(futs):
            done += 1
            errors += f.result() == "error"
            if done % 50 == 0 or done == len(futs):
                rate = done / (time.time() - t0) * 60
                print(f"  {done}/{len(futs)} done, {errors} errors, {rate:.1f}/min, "
                      f"~{(len(futs) - done) / max(rate, 0.1):.0f} min left", flush=True)


if __name__ == "__main__":
    main()
