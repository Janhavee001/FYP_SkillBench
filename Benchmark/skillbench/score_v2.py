"""
Score SkillBench v2 results: results/v2/<model>/run<k>/<condition>/<id>.json.

Same deterministic correctness rules as score.py (imported from it). The judge for the extraction step and
the secondary quality rubric is a model from a DIFFERENT family to the systems under test
(default qwen/qwen3-235b-a22b-2507), to avoid self-preference bias.
Answers accepted only through the rounding rule in score.correct are counted in scores.csv column `rounded`.

    python score_v2.py [--judge qwen/qwen3-235b-a22b-2507]   ->   results/v2/scores.csv
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib

import pandas as pd

import score as v1
from harness.common import load_env, openai_client  # noqa: E402  (path set up by score.py)

ROOT = pathlib.Path(__file__).resolve().parent
V2 = ROOT / "results" / "v2"
SYNTHETIC = {"clinic", "school", "saas", "shop"}
MULTI = {"olist", "f1db", "movielens", "shop", "flights", "imdb", "nyc_taxi", "clinic", "school", "saas"}


def judge(q, rec, cfg, cache: pathlib.Path):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    client = openai_client(cfg)
    sub = rec.get("submission") or {}
    answer = str(sub.get("answer", ""))[:3000]
    out = {}
    if q["answer_type"] != "number" or v1.num(sub.get("value")) is None:
        ch = f"\nCHOICES: {q['choices']}" if q["choices"] else ""
        out["extracted"] = v1.chat_json(client, cfg, v1.EXTRACT.format(
            question=q["question"], atype=q["answer_type"], choices=ch, answer=answer,
            value=sub.get("value"))).get("final")
    out["rubric"] = v1.chat_json(client, cfg, v1.RUBRIC.format(
        question=q["question"], reference=q["reference"], trap=v1.TRAP_TEXT.get(q["trap"], q["trap"]),
        method=q["expected_method"], smethod=str(sub.get("method", ""))[:500], answer=answer,
        interp=str(sub.get("interpretation", ""))[:2000]))
    if out["rubric"] and ("extracted" not in out or out["extracted"] is not None or not answer.strip()):
        # only cache a complete judgement, so a failed judge call is retried on the next scoring pass
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()
    cfg = load_env()
    cfg["judge_model"] = args.judge
    questions = {q["id"]: q for q in json.loads((ROOT / "questions" / "questions.json").read_text(encoding="utf-8"))}
    jobs = []
    for f in sorted(V2.glob("*/run*/*/*.json")):
        model, run, cond = f.parts[-4], f.parts[-3], f.parts[-2]
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["id"] in questions:
            jobs.append((model, run, cond, questions[rec["id"]], rec,
                         V2 / "judgements" / model / run / cond / f.name))
    print(f"{len(jobs)} answers to score, judge = {args.judge}", flush=True)
    judged = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, q, rec, cfg, cache): (m, r, c, q["id"]) for m, r, c, q, rec, cache in jobs
                if not rec.get("error")}
        for i, f in enumerate(cf.as_completed(futs), 1):
            judged[futs[f]] = f.result()
            if i % 500 == 0:
                print(f"  judged {i}/{len(futs)}", flush=True)
    rows = []
    for m, r, c, q, rec, _ in jobs:
        sub = rec.get("submission") or {}
        j = judged.get((m, r, c, q["id"]), {})
        final = sub.get("value") if q["answer_type"] == "number" and v1.num(sub.get("value")) is not None \
            else j.get("extracted")
        ok, slip, naive_hit = v1.correct(q, final) if not rec.get("error") else (False, False, False)
        rb = j.get("rubric", {}) or {}
        rows.append({"model": m, "run": r, "condition": c, "id": q["id"], "dataset": q["dataset"],
                     "synthetic": q["dataset"] in SYNTHETIC, "multi_table": q["dataset"] in MULTI,
                     "difficulty": q["difficulty"], "category": q["category"], "trap": q["trap"],
                     "has_trap": q["trap"] != "none", "correct": int(ok), "unit_slip": int(slip),
                     "rounded": int(ok and (q["id"], v1.num(final)) in v1.ROUNDED),
                     "fell_into_trap": int(naive_hit), "final": final, "reference": q["reference"],
                     "harness_error": int(bool(rec.get("error"))), "recovered": (sub.get("recovered") or ""),
                     "method_appropriate": rb.get("method_appropriate"), "caveats_stated": rb.get("caveats_stated"),
                     "no_overclaiming": rb.get("no_overclaiming"), "steps": rec.get("steps"),
                     "code_errors": rec.get("code_errors"), "skills_loaded": ",".join(rec.get("skills_loaded") or []),
                     "prompt_tokens": rec.get("prompt_tokens"), "completion_tokens": rec.get("completion_tokens"),
                     "seconds": rec.get("seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(V2 / "scores.csv", index=False)
    print(df.groupby(["model", "condition"]).correct.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
