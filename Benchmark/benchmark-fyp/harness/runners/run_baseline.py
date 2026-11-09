"""
Plain-LLM baselines on the shared questions.

Not one of the five papers -- added because RQ1 ("do structured skills improve
correctness vs a plain LLM baseline?") and RQ3 ("does validation/self-correction
improve reliability?") are unanswerable without controls. The five systems all
differ in model, scaffolding and output format at once, so comparing them to
each other alone cannot separate "the scaffolding helped" from "the model is
good". These two modes hold everything constant except the one variable:

  --mode plain      one prompt, one shot, no execution feedback, no retry.
                    The RQ1 control.
  --mode selfcheck  identical prompt and model, but on an execution error the
                    traceback is fed back for one repair attempt.
                    plain vs selfcheck isolates RQ3 inside a fixed architecture.

Run under the harness venv:
    venvs/harness/Scripts/python.exe harness/runners/run_baseline.py --mode plain
    venvs/harness/Scripts/python.exe harness/runners/run_baseline.py --mode selfcheck
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from harness.common import (  # noqa: E402
    RESULTS, Record, chat, load_questions, normalise_chart, normalise_method,
    require_key, run_code, strip_code_fence, write_results,
)

SYSTEM = ("You are a careful data analyst. You write correct, runnable Python "
          "and you state what your results mean, including their limitations.")

PROMPT = """You are given a CSV file at this exact path:

    {csv_path}

Columns and dtypes:
{columns}

Dataset: {description}

QUESTION:
{question}

Write a single self-contained Python script that answers the question.
Requirements:
- Load the CSV from the exact path above.
- print() every number a reader would need to answer the question.
- If a chart is appropriate, create it with matplotlib/seaborn and save it to
  "{chart_path}". Do not call plt.show().
- Do not ask follow-up questions.

Reply in exactly this format:

METHOD: <the statistical method or chart type you chose, a few words>
```python
<your script>
```
INTERPRETATION: <2-4 sentences saying what the result means, including any
caveat, assumption or limitation a reader should know about>
"""

REPAIR = """Your script failed to run.

SCRIPT:
```python
{code}
```

ERROR:
{error}

Return the corrected full script in a single ```python fence. No commentary.
"""


def parse(reply: str) -> tuple[str, str, str]:
    method = ""
    for line in reply.splitlines():
        if line.strip().upper().startswith("METHOD:"):
            method = line.split(":", 1)[1].strip()
            break
    code = strip_code_fence(reply)
    interp = ""
    if "INTERPRETATION:" in reply.upper():
        idx = reply.upper().rindex("INTERPRETATION:")
        interp = reply[idx + len("INTERPRETATION:"):].strip()
    return method, code, interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plain", "selfcheck"], default="plain")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = require_key(f"harness/runners/run_baseline.py --mode {args.mode}")
    system_name = f"baseline_{args.mode}"
    workdir = RESULTS / "artifacts" / system_name
    workdir.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    records = []
    for q in questions:
        rec = Record(id=q["id"], dataset=q["dataset"], question=q["question"],
                     ground_truth=q["reference_answer"],
                     expected_method=q["expected_method"],
                     acceptable_methods=q["acceptable_methods"],
                     expected_chart=q["expected_chart"],
                     tolerance=q["tolerance"], kind=q["kind"])
        wd = workdir / q["id"]
        wd.mkdir(parents=True, exist_ok=True)
        chart_path = wd / "chart.png"
        try:
            reply = chat(PROMPT.format(
                csv_path=str(q["csv_path"]).replace("\\", "/"),
                columns=json.dumps(q["dataset_columns"], indent=2),
                description=q["dataset_description"],
                question=q["question"],
                chart_path=str(chart_path).replace("\\", "/")),
                system=SYSTEM, cfg=cfg)
            method, code, interp = parse(reply)
            rec.declared_method = method or None
            rec.code = code
            rec.final_answer = interp
            rec.raw["raw_reply"] = reply[:6000]

            ok, out, err = run_code(code, wd, python=sys.executable, timeout=120)
            rec.executed, rec.stdout = ok, out[:8000]
            rec.execution_error = None if ok else err[-2000:]

            if not ok and args.mode == "selfcheck":
                rec.repair_attempts = 1
                fixed = strip_code_fence(chat(
                    REPAIR.format(code=code, error=err[-2000:]),
                    system=SYSTEM, cfg=cfg))
                ok2, out2, err2 = run_code(fixed, wd, python=sys.executable,
                                           timeout=120)
                if ok2:
                    rec.self_corrected = True
                rec.code = fixed
                rec.executed, rec.stdout = ok2, out2[:8000]
                rec.execution_error = None if ok2 else err2[-2000:]

            if chart_path.exists():
                rec.artifact = str(chart_path)
            rec.declared_chart = (", ".join(normalise_chart(method + " " + code))
                                  or None)
            if not rec.declared_method:
                rec.declared_method = ", ".join(normalise_method(code)) or None

            print(f"  {q['id']}: executed={rec.executed} method={method!r} "
                  f"repaired={rec.self_corrected}")
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.raw["traceback"] = traceback.format_exc()[-1500:]
            print(f"  {q['id']}: HARNESS ERROR {rec.error}")
        records.append(rec)

    arch = ("single-prompt LLM, no scaffolding, no validation"
            if args.mode == "plain" else
            "single-prompt LLM + one execution-feedback repair pass")
    write_results(system_name, f"(control) plain LLM -- {args.mode}", arch,
                  "n/a -- control condition defined by this harness", records,
                  meta={"model": cfg["bench_model"], "mode": args.mode,
                        "provider": cfg["provider"],
                        "base_url": cfg["openai_base_url"]})


if __name__ == "__main__":
    main()
