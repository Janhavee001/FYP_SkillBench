"""
Run SkillBench under one condition. Every condition uses the same model, temperature, data and
question text; only the scaffolding differs, so differences are attributable to the scaffolding.

  plain         One shot: the model writes one script, it runs once, the model then answers from
                its printed output. No iteration, no error feedback. (Weak baseline.)
  agent         Tool loop: the model may call run_python as often as it needs (up to MAX_STEPS),
                sees stdout/stderr each time, then calls submit_answer. (Strong baseline.)
  agent_skills  The same loop plus load_skill(name), which returns one of the eight skills'
                SKILL.md + generated tool reference, and the bundled `statkit` library on the
                import path. (Treatment.)

    python run.py --condition agent_skills [--limit N] [--ids q1,q2] [--workers 6]

Results go to results/<condition>/<question id>.json (an existing result is skipped, so a run can
be resumed). The full message transcript is kept with each result.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback

CODE_SLOTS = threading.Semaphore(int(os.environ.get("SKILLBENCH_CODE_SLOTS", "6")))

ROOT = pathlib.Path(__file__).resolve().parent
FYP = ROOT.parent.parent
sys.path.insert(0, str(ROOT.parent / "benchmark-fyp"))
from harness.common import load_env, openai_client, strip_code_fence  # noqa: E402

SKILLS_DIST = FYP / "Skills" / "claude_skills" / "dist"
STATKIT_PATH = SKILLS_DIST / "data-profiling" / "scripts"  # every skill bundles the same statkit
QUESTIONS = ROOT / "questions" / "questions.json"
RESULTS = ROOT / "results"
WORK = ROOT / "work"

# Oracle condition: the skill(s) a statistician would pick for each question, preloaded. Separates
# "does the skill help when it is used" (RQ1) from "does the agent pick it" (RQ2).
ORACLE_SKILLS = {
    "data_quality": ["data-profiling"], "descriptive": ["data-profiling"],
    "group_comparison": ["group-comparison"], "correlation": ["correlation-analysis"],
    "regression": ["regression-analysis"], "time_series": ["time-series-analysis"],
    "causal": ["group-comparison", "regression-analysis"],
}

MAX_STEPS = 10
CODE_TIMEOUT = 240
OUTPUT_CHARS = 6000

SYSTEM = ("You are a careful data analyst. You answer questions about real datasets by running Python, "
          "never by guessing. You check the data before trusting it, choose methods that fit the data, "
          "and state what results mean, including caveats.")

TASK = """DATA (files are real and may contain data-quality problems):
{data_block}

QUESTION:
{question}
"""

AGENT_RULES = """Use the run_python tool to inspect the data and compute the answer (pandas, numpy, scipy,
statsmodels, scikit-learn and matplotlib are installed; each call is a fresh Python process, but files you
write in the working directory persist). When you are confident, call submit_answer exactly once.
You have at most {max_steps} tool calls."""

SKILLS_RULES = """You also have analysis SKILLS. A skill is a procedure written by a statistician: when a
method applies, what to check first, which tested library function to call, and how to report the result.
Call load_skill(name) to read one before analysing; follow its procedure. Its functions come from the
`statkit` library, which is already importable in run_python (`from statkit.<module> import <function>`).
In skill examples, ignore the sys.path setup lines and the `SKILL_DIR` placeholder; use the real file paths
given above.

Available skills:
{catalog}"""

PLAIN_PROMPT = """{task}
Write ONE self-contained Python script that answers the question and print()s every number needed.
It will be run once; you will see its output and then give your final answer.
Reply with only the script in a ```python fence."""

PLAIN_ANSWER = """Your script printed:
----
{output}
----
{error_note}Now give your final answer as JSON with keys: "answer" (one sentence), "value" (the single
headline number, or null if the answer is not a number), "method" (the method used), "interpretation"
(2-4 sentences, including caveats). Reply with only the JSON."""

TOOLS_BASE = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run a Python script in a fresh process and return its stdout and stderr (truncated).",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the final answer. Call exactly once, at the end.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "The answer in one or two sentences."},
            "value": {"type": ["number", "null"], "description": "The single headline number, or null."},
            "method": {"type": "string", "description": "The method used, e.g. 'Welch t-test'."},
            "interpretation": {"type": "string", "description": "2-4 sentences: what it means, with caveats."},
        }, "required": ["answer", "value", "method", "interpretation"]}}},
]
LOAD_SKILL = {"type": "function", "function": {
    "name": "load_skill",
    "description": "Read one analysis skill (its procedure and the reference of the statkit functions it uses).",
    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}


def skill_catalog() -> tuple[str, dict]:
    lines, texts = [], {}
    for d in sorted(p for p in SKILLS_DIST.iterdir() if p.is_dir()):
        md = (d / "SKILL.md").read_text(encoding="utf-8")
        desc = md.split("description:", 1)[1].split("\n---", 1)[0].strip()
        lines.append(f"- {d.name}: {desc}")
        ref = (d / "reference" / "tools.md").read_text(encoding="utf-8")
        texts[d.name] = md.replace("<this skill's folder>", d.as_posix()) + "\n\n" + ref
    return "\n".join(lines), texts


def data_block(q: dict) -> str:
    out = []
    for f in q["files"]:
        out.append(f"- {f['path']}  ({f['rows']:,} rows)\n  columns: {f['columns']}")
    return f"{q['dataset_description']}\n" + "\n".join(out)


def run_python(code: str, workdir: pathlib.Path, with_statkit: bool) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / f"step_{int(time.time() * 1000)}.py"
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + code, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    if with_statkit:
        env["PYTHONPATH"] = str(STATKIT_PATH)
    try:
        with CODE_SLOTS:  # cap simultaneous analysis processes so multi-million-row loads fit in memory
            p = subprocess.run([sys.executable, str(script)], cwd=workdir, capture_output=True, text=True,
                               timeout=CODE_TIMEOUT, env=env, encoding="utf-8", errors="replace")
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", f"TimeoutExpired after {CODE_TIMEOUT}s", 1
    text = (out[-OUTPUT_CHARS:] if out else "(no stdout)")
    if rc != 0 or err.strip():
        text += "\n[stderr]\n" + err[-2500:]
    return text


def call(client, cfg, messages, tools=None):
    for attempt in range(6):
        try:
            kw = dict(model=cfg["bench_model"], messages=messages, temperature=cfg["temperature"])
            if cfg.get("seed") is not None:
                kw["seed"] = cfg["seed"]
            if tools:
                kw["tools"] = tools
            return client.chat.completions.create(**kw)
        except Exception as exc:  # noqa: BLE001 - provider hiccups: back off and retry
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
            last = exc  # noqa: F841


def run_plain(q, client, cfg, wd):
    task = TASK.format(data_block=data_block(q), question=q["question"])
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PLAIN_PROMPT.format(task=task)}]
    r = call(client, cfg, msgs)
    code = strip_code_fence(r.choices[0].message.content or "")
    output = run_python(code, wd, with_statkit=False)
    failed = "[stderr]" in output and "Traceback" in output
    msgs += [{"role": "assistant", "content": r.choices[0].message.content or ""},
             {"role": "user", "content": PLAIN_ANSWER.format(
                 output=output, error_note="(The script raised an error.)\n" if failed else "")}]
    r2 = call(client, cfg, msgs)
    raw = r2.choices[0].message.content or ""
    try:
        sub = json.loads(strip_code_fence(raw) if "```" in raw else raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        sub = {"answer": raw, "value": None, "method": "", "interpretation": ""}
    msgs.append({"role": "assistant", "content": raw})
    usage = [r.usage, r2.usage]
    return sub, msgs, {"steps": 1, "code_errors": int(failed), "skills_loaded": []}, usage


def oracle_for(q) -> list[str]:
    names = list(ORACLE_SKILLS.get(q["category"], ["data-profiling"]))
    if len(q["files"]) > 1:
        names = ["relational-data-analysis"] + names
    return names


def run_agent(q, client, cfg, wd, skills: bool, catalog=None, skill_texts=None, oracle=False):
    rules = AGENT_RULES.format(max_steps=MAX_STEPS)
    if skills:
        rules += "\n\n" + SKILLS_RULES.format(catalog=catalog)
    if oracle:
        rules += ("\n\nThe following skill(s) are the right ones for this question and are already loaded; "
                  "follow their procedure:\n\n" + "\n\n".join(skill_texts[n] for n in oracle_for(q)))
    tools = TOOLS_BASE + ([LOAD_SKILL] if skills else [])
    msgs = [{"role": "system", "content": SYSTEM + "\n\n" + rules},
            {"role": "user", "content": TASK.format(data_block=data_block(q), question=q["question"])}]
    stats = {"steps": 0, "code_errors": 0, "skills_loaded": oracle_for(q) if oracle else []}
    usage, submitted = [], None
    for _ in range(MAX_STEPS + 2):
        r = call(client, cfg, msgs, tools)
        usage.append(r.usage)
        m = r.choices[0].message
        entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            entry["tool_calls"] = [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in m.tool_calls]
        msgs.append(entry)
        if not m.tool_calls:
            if stats["steps"] >= MAX_STEPS:
                break
            msgs.append({"role": "user", "content": "Continue: use run_python, then call submit_answer."})
            continue
        for tc in m.tool_calls:
            name, args = parse_tool_call(tc.function.name, tc.function.arguments)
            if name == "submit_answer":
                submitted = args
                result = "Submitted."
            elif name == "load_skill" and skills:
                sk = str(args.get("name", "")).strip()
                result = skill_texts.get(sk, f"Unknown skill {sk!r}. Available: {sorted(skill_texts)}")
                if sk in skill_texts:
                    stats["skills_loaded"].append(sk)
            elif name == "run_python":
                stats["steps"] += 1
                if stats["steps"] > MAX_STEPS:
                    result = "Step limit reached. Call submit_answer now."
                else:
                    result = run_python(args.get("code", ""), wd, with_statkit=skills)
                    if "Traceback" in result:
                        stats["code_errors"] += 1
            else:
                result = f"Unknown tool {name!r}."
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if submitted is not None:
            break
    if submitted is None:
        submitted = recover_submission(msgs)
    return submitted, msgs, stats, usage


TOOL_NAMES = ("run_python", "submit_answer", "load_skill")


def parse_tool_call(name: str, raw: str) -> tuple[str, dict]:
    """Lenient tool-call parsing. gpt-oss sometimes leaks its chat-format tokens into tool calls: a name such as
    'run_python<|channel|>commentary', JSON with literal newlines inside strings, or the bare code instead of
    {"code": ...}. Before this parser such calls ran an empty script and returned '(no stdout)', which misled the
    model. Applied identically to every condition."""
    clean = (name or "").split("<|")[0].strip()
    if clean not in TOOL_NAMES:
        clean = next((t for t in TOOL_NAMES if t in (name or "")), clean)
    raw = raw or ""
    try:
        obj = json.loads(raw, strict=False)
        if isinstance(obj, dict):
            if clean == "run_python" and "code" not in obj and len(obj) == 1:
                obj = {"code": str(next(iter(obj.values())))}
            return clean, obj
        if isinstance(obj, str):
            raw = obj
    except json.JSONDecodeError:
        if raw.lstrip().startswith("{") and "}" in raw:
            try:
                obj = json.loads(raw[raw.index("{"):raw.rindex("}") + 1], strict=False)
                if isinstance(obj, dict):
                    return clean, obj
            except json.JSONDecodeError:
                pass
    if clean == "run_python":
        code = raw.strip()
        if "\n" not in code and "\\n" in code:  # escaped newlines in a bare string
            code = code.replace("\\n", "\n").replace('\\"', '"').replace("\\t", "\t")
        return clean, {"code": code}
    if clean == "submit_answer":
        return clean, {"answer": raw.strip()[:3000], "value": None, "method": "", "interpretation": ""}
    if clean == "load_skill":
        return clean, {"name": raw.strip().strip('"')}
    return clean, {}


def old_parse_failed(name: str, raw: str) -> bool:
    """True when the pre-fix harness mishandled this call (see parse_tool_call)."""
    if name not in TOOL_NAMES:
        return any(t in (name or "") for t in TOOL_NAMES)
    try:
        obj = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return True
    return name == "run_python" and (not isinstance(obj, dict) or "code" not in obj)


def recover_submission(msgs) -> dict:
    """The model sometimes writes its submit_answer JSON as plain text instead of calling the tool.
    Take the latest assistant message that parses as a submission; failing that, the latest non-empty
    assistant text, so the extractor can read the answer. Applied identically to every condition."""
    texts = [m.get("content") or "" for m in msgs if m["role"] == "assistant" and (m.get("content") or "").strip()]
    for t in reversed(texts):
        if "{" in t and "}" in t:
            try:
                obj = json.loads(t[t.index("{"):t.rindex("}") + 1])
                if isinstance(obj, dict) and "answer" in obj:
                    return {"answer": str(obj.get("answer", "")), "value": obj.get("value"),
                            "method": str(obj.get("method", "")), "interpretation": str(obj.get("interpretation", "")),
                            "recovered": "json_in_text"}
            except json.JSONDecodeError:
                pass
    if texts:
        return {"answer": texts[-1][:3000], "value": None, "method": "", "interpretation": "", "recovered": "last_text"}
    return {"answer": "(no answer submitted)", "value": None, "method": "", "interpretation": ""}


def tokens(usage) -> dict:
    p = sum(getattr(u, "prompt_tokens", 0) or 0 for u in usage if u)
    c = sum(getattr(u, "completion_tokens", 0) or 0 for u in usage if u)
    return {"prompt_tokens": p, "completion_tokens": c}


def run_one(q, condition, cfg, catalog, skill_texts, tag=""):
    out = RESULTS / tag / condition / f"{q['id']}.json"
    if out.exists():
        return "skip"
    client = openai_client(cfg)
    wd = WORK / tag / condition / q["id"]
    t0 = time.time()
    try:
        if condition == "plain":
            sub, msgs, stats, usage = run_plain(q, client, cfg, wd)
        else:
            sub, msgs, stats, usage = run_agent(q, client, cfg, wd, skills=condition.startswith("agent_skills"),
                                                catalog=catalog, skill_texts=skill_texts,
                                                oracle=condition == "agent_skills_oracle")
        rec = {"id": q["id"], "condition": condition, "model": cfg["bench_model"], "run": tag,
               "temperature": cfg["temperature"], "submission": sub, **stats, **tokens(usage),
               "seconds": round(time.time() - t0, 1), "error": None, "transcript": msgs}
    except Exception as exc:  # noqa: BLE001
        rec = {"id": q["id"], "condition": condition, "submission": None, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return "error" if rec["error"] else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["plain", "agent", "agent_skills", "agent_skills_oracle"])
    ap.add_argument("--model", default="", help="override BENCH_MODEL, e.g. openai/gpt-oss-20b")
    ap.add_argument("--run", type=int, default=0, help="v2 repeat index (1, 2, 3); 0 = the v1 layout")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_env()
    if args.model:
        cfg["bench_model"] = args.model
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.run:
        cfg["seed"] = 1000 + args.run
    tag = f"v2/{cfg['bench_model'].split('/')[-1]}/run{args.run}" if args.run else ""
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.ids:
        keep = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in keep]
    if args.limit:
        questions = questions[:args.limit]
    catalog, skill_texts = skill_catalog()
    print(f"{args.condition}: {len(questions)} questions, model={cfg['bench_model']}, "
          f"temperature={cfg['temperature']}, run={args.run or 'v1'}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, q, args.condition, cfg, catalog, skill_texts, tag): q["id"] for q in questions}
        for i, f in enumerate(cf.as_completed(futs), 1):
            print(f"  [{i}/{len(futs)}] {futs[f]}: {f.result()}", flush=True)


if __name__ == "__main__":
    main()
