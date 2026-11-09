"""
External benchmark: StatABench (Zhu et al. 2026), Stat-Closed, the 198 tool-requiring items (code == 1).

The items, their data files (datasets83) and their reference answers ("ground truth2") are used unchanged. Each
item is run through the same four SkillBench conditions with the same harness (run.py), and graded with
StatABench's own comparison prompt (evaluation/eval_utils.py, prompt_compare_str), copied verbatim below, with
the judge model from a different family to the models under test.

    python statabench.py run   [--models ...] [--runs 1,2,3] [--workers 40]
    python statabench.py score [--judge qwen/qwen3-235b-a22b-2507]
    python statabench.py analyse          ->  results/statabench/summary.md, stats.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import pathlib
import re
import time

import numpy as np
import pandas as pd

import run as R
from harness.common import load_env, openai_client  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
SB = ROOT.parents[0] / "benchmark-fyp" / "repos" / "StatABench"
DATA = SB / "data" / "datasets83"
OUT = ROOT / "results" / "statabench"
CONDS = ["plain", "agent", "agent_skills", "agent_skills_oracle", "agent_skills_validated"]
DESIGN = [("plain", [1]), ("agent", [1, 2, 3]), ("agent_skills", [1, 2, 3]), ("agent_skills_oracle", [1, 2, 3])]

# StatABench task -> (SkillBench category used by the oracle condition, skills a statistician would load)
TASK_SKILLS = {
    "Correlation Analysis": ["correlation-analysis"],
    "Distribution Compliance Test": ["distribution-analysis"],
    "Contingency Table Test": ["group-comparison"],
    "Descriptive Statistics": ["data-profiling", "distribution-analysis"],
    "Variance Test": ["group-comparison"],
    "generalized linear model": ["regression-analysis"],
    "survial analysis": ["group-comparison", "regression-analysis"],
    "ABtest": ["group-comparison"],
    "confidence interval": ["distribution-analysis", "group-comparison"],
    "time series": ["time-series-analysis"],
    "EDA": ["data-profiling"],
    "regression": ["regression-analysis"],
    "bayesian method": ["group-comparison"],
    "NonpStat": ["group-comparison"],
    "High dimension data": ["regression-analysis"],
    "causal inference": ["regression-analysis"],
    "multiple comparison": ["group-comparison"],
}

# Verbatim from StatABench evaluation/eval_utils.py
PROMPT_COMPARE = """
For the following question:
{question}

There are two answers:
<ground truth> {ground} </ground truth>

<response> {response} </response>

Please determine whether the response correctly conveys the meaning of the ground. You should follow these rules:
- A response should be considered correct ONLY IF it either:
    (a) explicitly states a conclusion that answers the question (e.g., whether a difference exists or not), and this conclusion aligns with the ground truth; OR
    (b) provides explicit and sufficient results (e.g., test statistics, p-values, model outputs) from which the correct conclusion can be directly inferred, and these results are consistent with the ground truth, even if the response does not explicitly state the conclusion.
- If the response only discusses methodology, analysis plans, or conditional statements (e.g., "if significant, then...") without providing a conclusion or sufficient results, it must be judged as false.
- If the main conclusion is wrong, missing, or includes factual information that contradicts the ground truth, output "false".

Important: Your answer must be strictly "true" or "false" only, without any additional explanation.
"""


def resolve(name: str) -> pathlib.Path | None:
    files = {p.stem.lower(): p for p in DATA.iterdir()}
    stem = re.sub(r"\.(csv|xlsx?)$", "", name.strip(), flags=re.I).lower()
    return files.get(stem)


def questions() -> list[dict]:
    items = json.loads((SB / "data" / "stat-closed.json").read_text(encoding="utf-8"))
    out = []
    for it in items:
        if it["code"] != 1:
            continue
        files = []
        if isinstance(it["dataset"], str):
            p = resolve(it["dataset"])
            if p is None:
                raise FileNotFoundError(f"{it['id']}: {it['dataset']}")
            df = pd.read_csv(p, nrows=None, low_memory=False, encoding_errors="replace")
            files = [{"path": p.as_posix(), "rows": len(df), "columns": list(map(str, df.columns))}]
            desc = f"Dataset: {p.name}"
        else:
            desc = "The data needed is given in the question itself."
        out.append({"id": it["id"], "dataset": p.stem if files else "inline", "question": it["question"],
                    "category": it["task"], "difficulty": "n/a", "trap": "none", "answer_type": "text",
                    "files": files, "dataset_description": desc, "ground": it["ground truth2"],
                    "function": it["function"], "source": it["reference"]})
    return out


def oracle_for(q):
    names = TASK_SKILLS.get(q["category"])
    return list(names) if names else R._oracle_for_skillbench(q)


def cmd_run(args):
    base = load_env()
    qs = questions()[:args.limit or None]
    R._oracle_for_skillbench = R.oracle_for
    R.oracle_for = oracle_for  # run_agent looks the oracle skill up through this name
    R.RESULTS = OUT
    catalog, skill_texts = R.skill_catalog()
    jobs = []
    for k in [int(x) for x in args.runs.split(",")]:
        for model in args.models.split(","):
            cfg = copy.deepcopy(base)
            cfg.update(bench_model=model, temperature=args.temperature, seed=1000 + k)
            tag = f"{model.split('/')[-1]}/run{k}"
            for cond, runs in DESIGN:
                if k in runs:
                    jobs += [(q, cond, cfg, tag) for q in qs if not (OUT / tag / cond / f"{q['id']}.json").exists()]
    print(f"{len(jobs)} StatABench jobs, {args.workers} workers", flush=True)
    t0, done, errors = time.time(), 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(R.run_one, q, c, cfg, catalog, skill_texts, tag) for q, c, cfg, tag in jobs]
        for f in cf.as_completed(futs):
            done += 1
            errors += f.result() == "error"
            if done % 50 == 0 or done == len(futs):
                rate = done / (time.time() - t0) * 60
                print(f"  {done}/{len(futs)} done, {errors} errors, {rate:.1f}/min, "
                      f"~{(len(futs) - done) / max(rate, 0.1):.0f} min left", flush=True)


def response_text(rec) -> str:
    sub = rec.get("submission") or {}
    return " ".join(str(sub.get(k) or "") for k in ("answer", "interpretation")).strip()


def judge(q, rec, cfg, cache):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["match"]
    resp = response_text(rec)
    if not resp:
        return False
    if q["ground"].strip().lower() == resp.lower():
        match = True
    else:
        client = openai_client(cfg)
        verdict = None
        for _ in range(4):
            try:
                r = client.chat.completions.create(
                    model=cfg["judge_model"], temperature=0,
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": PROMPT_COMPARE.format(question=q["question"],
                                                                                 ground=q["ground"], response=resp)}])
                t = (r.choices[0].message.content or "").strip().lower()
                if t in ("true", "false"):
                    verdict = t == "true"
                    break
            except Exception:  # noqa: BLE001
                continue
        if verdict is None:
            return None
        match = verdict
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"match": match}), encoding="utf-8")
    return match


def cmd_score(args):
    cfg = load_env()
    cfg["judge_model"] = args.judge
    qs = {q["id"]: q for q in questions()}
    jobs = []
    for f in sorted(OUT.glob("*/run*/*/*.json")):
        model, run, cond = f.parts[-4], f.parts[-3], f.parts[-2]
        rec = json.loads(f.read_text(encoding="utf-8"))
        jobs.append((model, run, cond, qs[rec["id"]], rec, OUT / "judgements" / model / run / cond / f.name))
    print(f"{len(jobs)} StatABench answers to judge, judge = {args.judge}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge, q, rec, cfg, cache): (m, r, c, q, rec) for m, r, c, q, rec, cache in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            m, r, c, q, rec = futs[fut]
            ok = fut.result()
            rows.append({"model": m, "run": r, "condition": c, "id": q["id"], "task": q["category"],
                         "inline_data": not q["files"], "source": str(q["source"])[:20],
                         "correct": None if ok is None else int(ok), "harness_error": int(bool(rec.get("error"))),
                         "skills_loaded": ",".join(rec.get("skills_loaded") or []), "steps": rec.get("steps"),
                         "prompt_tokens": rec.get("prompt_tokens"), "seconds": rec.get("seconds")})
            if i % 500 == 0:
                print(f"  judged {i}/{len(futs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)
    print(f"unjudged (judge failed): {df.correct.isna().sum()}")
    print(df.groupby(["model", "condition"]).correct.mean().unstack().round(3).to_string())


def cmd_analyse(_args):
    import analyze_v2 as A  # same bootstrap / permutation code as SkillBench v2
    df = pd.read_csv(OUT / "scores.csv").dropna(subset=["correct"])
    models = sorted(df.model.unique(), key=lambda m: ("20b" in m, m))
    stats = {"accuracy": [], "paired": [], "by_task": [], "selection": []}
    out = ["# StatABench (Stat-Closed, 198 tool items) under SkillBench conditions\n",
           "Items, data and reference answers unchanged from StatABench; graded with StatABench's own comparison "
           "prompt (judge qwen3-235b, temperature 0).\n",
           "| Model | Condition | Accuracy (mean of runs) | SD across runs | 95% CI | Runs |", "|---|---|---|---|---|---|"]
    for m in models:
        for c in CONDS:
            d = df[(df.model == m) & (df.condition == c)]
            if d.empty:
                continue
            by_run = d.groupby("run").correct.mean()
            pq = d.groupby("id").correct.mean()
            lo, hi = A.boot_ci(pq)
            sd = by_run.std(ddof=1) if len(by_run) > 1 else 0.0
            stats["accuracy"].append({"model": m, "condition": c, "acc": by_run.mean(), "sd": sd, "lo": lo, "hi": hi,
                                      "runs": len(by_run), "n": int(pq.size)})
            out.append(f"| {m} | {A.LABEL[c]} | {A.pct(by_run.mean())} | {100 * sd:.1f} pts | {A.pct(lo)} to "
                       f"{A.pct(hi)} | {len(by_run)} |")
    out += ["\n## Paired comparisons\n", "| Model | Comparison | Difference | 95% CI | p (permutation) |",
            "|---|---|---|---|---|"]
    for m in models:
        for a, b in [("plain", "agent"), ("agent", "agent_skills"), ("agent", "agent_skills_oracle"),
                     ("agent_skills", "agent_skills_oracle"), ("plain", "agent_skills"),
                     ("agent_skills", "agent_skills_validated")]:
            qa = df[(df.model == m) & (df.condition == a)].groupby("id").correct.mean()
            qb = df[(df.model == m) & (df.condition == b)].groupby("id").correct.mean()
            common = qa.index.intersection(qb.index)
            if not len(common):
                continue
            diff = (qb[common] - qa[common]).to_numpy()
            lo, hi = A.boot_ci(diff)
            p = A.perm_test(diff)
            stats["paired"].append({"model": m, "a": a, "b": b, "diff": diff.mean(), "lo": lo, "hi": hi, "p": p,
                                    "n": len(diff)})
            out.append(f"| {m} | {A.LABEL[a]} -> {A.LABEL[b]} | {100 * diff.mean():+.1f} pts | {100 * lo:+.1f} to "
                       f"{100 * hi:+.1f} | {p:.3g} |")
    out += ["\n## By StatABench task\n"]
    for m in models:
        t = df[df.model == m].groupby(["task", "condition"]).correct.mean().unstack()
        cs = [c for c in CONDS if c in t.columns]
        n = df[(df.model == m) & (df.condition == cs[0])].groupby("task").id.nunique()
        out += [f"\n**{m}**\n", "| Task | n | " + " | ".join(A.LABEL[c] for c in cs) + " |",
                "|---" * (len(cs) + 2) + "|"]
        for k in t.index:
            out.append(f"| {k} | {n.get(k, 0)} | " + " | ".join(A.pct(t.loc[k, c]) for c in cs) + " |")
            stats["by_task"] += [{"model": m, "task": k, "n": int(n.get(k, 0)), "condition": c, "acc": t.loc[k, c]}
                                 for c in cs]
    out += ["\n## Skill selection (Agent + skills)\n", "| Model | Loaded any skill | Loaded a fitting skill |",
            "|---|---|---|"]
    qs = {q["id"]: q for q in questions()}
    for m in models:
        s = df[(df.model == m) & (df.condition == "agent_skills")].copy()
        if s.empty:
            continue
        s["loaded"] = s.skills_loaded.fillna("").astype(str).str.len() > 0
        s["hit"] = s.apply(lambda r: bool(set(oracle_for(qs[r.id]) if qs[r.id]["category"] in TASK_SKILLS else [])
                                          & set(str(r.skills_loaded).split(","))), axis=1)
        stats["selection"].append({"model": m, "loaded": s.loaded.mean(), "hit": s.hit.mean()})
        out.append(f"| {m} | {A.pct(s.loaded.mean())} | {A.pct(s.hit.mean())} |")
    out += A.section_validator(df, models, root=OUT)
    stats["validator"] = A.STATS["validator"]
    (OUT / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=lambda x: None if pd.isna(x) else float(x)),
                                    encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "analyse", "check"])
    ap.add_argument("--models", default="openai/gpt-oss-120b,openai/gpt-oss-20b")
    ap.add_argument("--runs", default="1")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--judge", default="qwen/qwen3-235b-a22b-2507")
    a = ap.parse_args()
    if a.cmd == "check":
        qs = questions()
        print(len(qs), "items;", sum(not q["files"] for q in qs), "with inline data;",
              len({q['dataset'] for q in qs}), "datasets")
    else:
        {"run": cmd_run, "score": cmd_score, "analyse": cmd_analyse}[a.cmd](a)
