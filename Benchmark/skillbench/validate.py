"""
RQ3: does a validation step catch and repair analysis errors?

Condition `agent_skills_validated` starts from the SAME first answer as `agent_skills` (its saved transcript), so
the comparison is exactly paired. After the agent submits:

  1. Deterministic checks
     - grounding: the agent ran code that finished without an error;
     - numeric consistency: a submitted number appears in the agent's own code output (it was computed, not
       written from memory).
  2. Checklist critic: the same model, at temperature 0, reviews the question, schema, the code that was run,
     its output and the answer against a fixed checklist of analysis errors, and lists concrete concerns.
  3. Repair: if anything is flagged, the concerns go back into the agent's conversation and it may run up to
     REPAIR_STEPS more code calls, then resubmit (keeping its answer if a concern does not apply).

Measured: how often wrong first answers are flagged (detection), how often correct ones are (false alarms), and
how many answers are fixed versus broken by the repair.

    python validate.py --bench skillbench|statabench [--runs 1,2,3] [--workers 40]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import pathlib
import re
import time

import run as R
from harness.common import load_env, openai_client  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
COND_IN, COND_OUT = "agent_skills", "agent_skills_validated"
REPAIR_STEPS = 4

CHECKLIST = """1. Unit of analysis: is the answer computed at the unit the question asks about (e.g. per invoice, per person,
   per day), not per row, when rows repeat an entity?
2. Population: were the right records included or excluded (e.g. cancelled, invalid, out-of-period, deceased,
   or records outside the question's scope)?
3. Missing values: are missing-value codes ('?', 'None', '\\N', sentinel numbers) or blanks handled correctly,
   including blanks that mean zero or "not recorded"?
4. Joins: if tables were joined, could the join have duplicated or dropped rows (one-to-many fan-out, orphans,
   the wrong key)?
5. Method: does the method match the question and the data (variable types, skew, paired vs independent,
   censoring, the exact test or model the question names, controls the question asks for)?
6. Measurability: can the question actually be answered from this data, or is the needed quantity not recorded?
7. Answer: does the submitted answer state exactly what the question asks, in its units and direction?"""

CRITIC = """You are a strict reviewer checking another analyst's work before it is accepted. Do not redo the
analysis; look for concrete mistakes.

QUESTION:
{question}

DATA:
{data}

CODE THE ANALYST RAN (most recent last):
{code}

OUTPUT OF THAT CODE (most recent last, truncated):
{output}

SUBMITTED ANSWER:
{answer}

Check the work against this list:
{checklist}

Report only problems that are likely to change the answer, each with specific evidence from the code or
output. Do not report style issues or missing extra analyses. If the work is sound, report no issues.
Reply with JSON only: {{"issues": ["<concrete problem and why it changes the answer>", ...]}}"""

REPAIR = """VALIDATOR REVIEW. Before your answer is accepted, a validation step raised these concerns:
{concerns}

Re-examine them with run_python (at most {n} more calls). If a concern is correct, fix the analysis. If it does
not apply, keep your answer. Then call submit_answer again with your final answer."""


def numbers(text: str) -> list[float]:
    out = []
    for t in re.findall(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text or ""):
        try:
            out.append(float(t.replace(",", "")))
        except ValueError:
            pass
    return out


def deterministic(rec) -> list[str]:
    msgs = rec.get("transcript") or []
    outputs = [m["content"] for m in msgs if m.get("role") == "tool" and m.get("content") not in ("Submitted.",)]
    code_outputs = [o for o in outputs if not o.startswith("---") and "Unknown skill" not in o]
    issues = []
    ran_ok = any("Traceback" not in o for o in code_outputs if o and not o.startswith("Step limit"))
    if not ran_ok:
        issues.append("Grounding: no code ran successfully, so the answer is not backed by any computation.")
    v = R_num((rec.get("submission") or {}).get("value"))
    if v is not None and ran_ok:
        seen = [x for o in code_outputs for x in numbers(o)]
        ok = any(abs(x - v) <= max(0.005 * abs(v), 0.01) or abs(100 * x - v) <= max(0.005 * abs(v), 0.01)
                 for x in seen)
        if not ok:
            issues.append(f"Numeric consistency: the submitted value {v} does not appear in any code output; "
                          "it may have been miscopied or computed by hand.")
    return issues


def R_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def critic(client, cfg, q, rec) -> list[str]:
    msgs = rec.get("transcript") or []
    code = []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            name, args = R.parse_tool_call(tc["function"]["name"], tc["function"]["arguments"])
            if name == "run_python":
                code.append(args.get("code", ""))
    outs = [m["content"] for m in msgs if m.get("role") == "tool" and m.get("content") != "Submitted."
            and not str(m.get("content", "")).startswith("---")]
    sub = rec.get("submission") or {}
    prompt = CRITIC.format(question=q["question"], data=R.data_block(q),
                           code=("\n# ---- next call ----\n".join(code))[-7000:] or "(none)",
                           output=("\n---\n".join(outs))[-5000:] or "(none)",
                           answer=json.dumps({k: sub.get(k) for k in ("answer", "value", "method", "interpretation")},
                                             ensure_ascii=False)[:2500],
                           checklist=CHECKLIST)
    c2 = dict(cfg, temperature=0)
    for _ in range(3):
        try:
            r = R.call(client, c2, [{"role": "user", "content": prompt}])
            t = r.choices[0].message.content or ""
            obj = json.loads(t[t.index("{"):t.rindex("}") + 1])
            return [str(i) for i in obj.get("issues", []) if str(i).strip()][:5], r.usage
        except Exception:  # noqa: BLE001
            continue
    return [], None


def repair(client, cfg, q, rec, concerns, wd, skill_texts):
    msgs = copy.deepcopy(rec["transcript"])
    msgs.append({"role": "user", "content": REPAIR.format(concerns="\n".join(f"- {c}" for c in concerns),
                                                          n=REPAIR_STEPS)})
    tools = R.TOOLS_BASE + [R.LOAD_SKILL]
    steps, usage, submitted = 0, [], None
    for _ in range(REPAIR_STEPS + 2):
        r = R.call(client, cfg, msgs, tools)
        usage.append(r.usage)
        m = r.choices[0].message
        entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in m.tool_calls]
        msgs.append(entry)
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Call submit_answer with your final answer."})
            continue
        for tc in m.tool_calls:
            name, args = R.parse_tool_call(tc.function.name, tc.function.arguments)
            if name == "submit_answer":
                submitted, result = args, "Submitted."
            elif name == "load_skill":
                result = skill_texts.get(str(args.get("name", "")).strip(), "Unknown skill.")
            elif name == "run_python":
                steps += 1
                result = ("Step limit reached. Call submit_answer now." if steps > REPAIR_STEPS
                          else R.run_python(args.get("code", ""), wd, with_statkit=True))
            else:
                result = "Unknown tool."
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if submitted is not None:
            break
    if submitted is None:
        rec_sub = R.recover_submission(msgs[len(rec["transcript"]):])
        submitted = rec_sub if rec_sub.get("answer") != "(no answer submitted)" else rec["submission"]
    return submitted, msgs, steps, usage


def validate_one(src: pathlib.Path, dst: pathlib.Path, q, cfg, skill_texts, wd):
    if dst.exists():
        return "skip"
    rec = json.loads(src.read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        if rec.get("error") or not rec.get("transcript"):
            raise RuntimeError("source answer missing")
        client = openai_client(cfg)
        det = deterministic(rec)
        crit, u0 = critic(client, cfg, q, rec)
        concerns = det + crit
        usage = [u0]
        if concerns:
            sub, msgs, steps, u = repair(client, cfg, q, rec, concerns, wd, skill_texts)
            usage += u
        else:
            sub, msgs, steps = rec["submission"], rec["transcript"], 0
        out = {"id": rec["id"], "condition": COND_OUT, "model": cfg["bench_model"], "run": rec.get("run"),
               "submission": sub, "initial_submission": rec["submission"], "flagged": bool(concerns),
               "deterministic_issues": det, "critic_issues": crit, "repair_steps": steps,
               "steps": (rec.get("steps") or 0) + steps, "code_errors": rec.get("code_errors"),
               "skills_loaded": rec.get("skills_loaded"),
               "prompt_tokens": (rec.get("prompt_tokens") or 0) + R.tokens(usage)["prompt_tokens"],
               "completion_tokens": (rec.get("completion_tokens") or 0) + R.tokens(usage)["completion_tokens"],
               "seconds": round((rec.get("seconds") or 0) + time.time() - t0, 1), "error": None,
               "transcript": msgs}
    except Exception as exc:  # noqa: BLE001
        out = {"id": rec.get("id"), "condition": COND_OUT, "submission": None, "error": f"{type(exc).__name__}: {exc}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return "error" if out["error"] else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["skillbench", "statabench"], required=True)
    ap.add_argument("--models", default="gpt-oss-120b,gpt-oss-20b")
    ap.add_argument("--runs", default="1,2,3")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    base = load_env()
    if args.bench == "skillbench":
        qs = {q["id"]: q for q in json.loads(R.QUESTIONS.read_text(encoding="utf-8"))}
        root = ROOT / "results" / "v2"
    else:
        import statabench as S
        qs = {q["id"]: q for q in S.questions()}
        root = S.OUT
    _, skill_texts = R.skill_catalog()
    jobs = []
    for k in args.runs.split(","):
        for m in args.models.split(","):
            cfg = copy.deepcopy(base)
            cfg.update(bench_model=f"openai/{m}", temperature=0.7, seed=1000 + int(k))
            srcs = sorted((root / m / f"run{k}" / COND_IN).glob("*.json"))[:args.limit or None]
            for src in srcs:
                dst = root / m / f"run{k}" / COND_OUT / src.name
                wd = R.WORK / "validated" / args.bench / m / f"run{k}" / src.stem
                if not dst.exists() and src.stem in qs:
                    jobs.append((src, dst, qs[src.stem], cfg, wd))
    print(f"{len(jobs)} validations ({args.bench}), {args.workers} workers", flush=True)
    t0, done, errors = time.time(), 0, 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(validate_one, s, d, q, c, skill_texts, w) for s, d, q, c, w in jobs]
        for f in cf.as_completed(futs):
            done += 1
            errors += f.result() == "error"
            if done % 50 == 0 or done == len(futs):
                rate = done / (time.time() - t0) * 60
                print(f"  {done}/{len(futs)} done, {errors} errors, {rate:.1f}/min, "
                      f"~{(len(futs) - done) / max(rate, 0.1):.0f} min left", flush=True)


if __name__ == "__main__":
    main()
