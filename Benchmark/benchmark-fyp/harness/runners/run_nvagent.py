"""
Run nvAgent (ACL 2025 long 960 / arXiv 2502.05036) against the shared questions.

nvAgent is a three-agent workflow: Processor (schema description + context
filtering) -> Composer (plans the visualisation, emits VQL) -> Validator
(translates VQL to matplotlib/seaborn, executes it, and refines on failure).
The Validator is the reason this system is in the comparison at all: it is the
only one of the five with an explicit, separately-prompted validation agent, so
its `fixed` / `try_times` message fields are direct evidence for RQ3.

Two adaptations, both logged in the output so they can be reported honestly:

  1. nvAgent ships pointed at Azure OpenAI. Rather than editing the clone, we
     monkeypatch core.llm.api_func onto a plain OpenAI client built from
     OPENAI_API_KEY. The prompts and agent logic are untouched.
  2. nvAgent expects a VisEval-shaped corpus (folder/databases/<db_id>/*.csv).
     We materialise our two CSVs into exactly that layout instead of changing
     its loader.

Run under the nvAgent venv:
    venvs/nvagent/Scripts/python.exe harness/runners/run_nvagent.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
import traceback
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
NVAGENT = ROOT / "repos" / "nvAgent"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NVAGENT))


def _stub_cairosvg():
    """
    nvAgent's viseval/__init__.py eagerly imports Evaluator, which imports
    cairosvg at module level; cairocffi then needs a native Cairo library that
    is not present on a stock Windows install. cairosvg is used in exactly one
    place -- viseval/evaluate.py svg2png, part of the native VisEval
    MLLM-as-judge scoring, which this shared-dataset run does not use.

    So we register a stub before importing chat_manager rather than making the
    whole runner depend on a system library it never calls. If you later want
    nvAgent's *native* VisEval scores, install real Cairo (GTK runtime on
    Windows) and delete this.
    """
    if "cairosvg" in sys.modules:
        return False
    try:
        import cairosvg  # noqa: F401
        return False
    except Exception:  # noqa: BLE001
        stub = types.ModuleType("cairosvg")

        def svg2png(*_args, **_kwargs):
            raise RuntimeError(
                "cairosvg is stubbed out in run_nvagent.py: native Cairo is not "
                "installed. Only nvAgent's native VisEval rendering needs it.")

        stub.svg2png = svg2png
        sys.modules["cairosvg"] = stub
        return True

from harness.common import (  # noqa: E402
    RESULTS, Record, load_questions, normalise_chart, require_key, run_code,
    write_results,
)

ARCH = "multi-agent (Processor -> Composer -> Validator), explicit validator " \
       "agent with execution-feedback refinement loop"
NATIVE_EVAL = ("VisEval benchmark: per-chart pass rate decomposed into code "
               "execution validity, data accuracy (deconstructed chart vs "
               "ground-truth table), chart-type/ordering/scale checks, and a "
               "readability score from an MLLM-as-judge over the rendered PNG.")

WORKSPACE = RESULTS / "artifacts" / "nvagent_workspace"


def build_workspace(questions) -> dict:
    """Materialise the shared CSVs in the folder/databases/<db_id>/ layout."""
    dbs = {}
    for q in questions:
        db_id = q["dataset"]
        if db_id in dbs:
            continue
        dest_dir = WORKSPACE / "databases" / db_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = pathlib.Path(q["csv_path"])
        # The copied file is named after the db_id, NOT after the source file.
        # nvAgent derives the SQL table name from the filename stem and emits
        # `CREATE VIEW {stem}` unquoted, so "winequality-red.csv" becomes
        # `CREATE VIEW winequality-red` and DuckDB fails on the hyphen -- every
        # wine question dies while every titanic one passes, purely because
        # "titanic.csv" happens to be a legal identifier. Naming the file after
        # the db_id also keeps the table name consistent with what the Processor
        # shows the Composer, so the generated SQL references a table that
        # actually exists.
        dest = dest_dir / f"{db_id}.csv"
        shutil.copyfile(src, dest)
        # Forward slashes for the same reason as workspace_posix below.
        dbs[db_id] = [str(dest).replace("\\", "/")]
    return dbs


def patch_llm(cfg):
    """Swap nvAgent's AzureOpenAI call for a plain OpenAI one, in-memory."""
    from openai import OpenAI
    from core import llm as nv_llm

    client = OpenAI(api_key=cfg["openai_api_key"], base_url=cfg["openai_base_url"])
    model = cfg["bench_model"]

    def api_func(prompt: str):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0)
        text = (resp.choices[0].message.content or "").strip()
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    nv_llm.api_func = api_func
    nv_llm.MODEL_NAME = model
    return nv_llm


def chart_from_vql(vql: str) -> str | None:
    """nvAgent's VQL starts 'Visualize <TYPE> SELECT ...'."""
    if not vql:
        return None
    m = re.search(r"Visualize\s+([\w\s]+?)\s+SELECT", vql, re.IGNORECASE)
    return m.group(1).strip().lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--library", default="matplotlib")
    args = ap.parse_args()

    cfg = require_key(
        "venvs/nvagent/Scripts/python.exe harness/runners/run_nvagent.py")

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    dbs = build_workspace(questions)

    stubbed = _stub_cairosvg()
    if stubbed:
        print("[nvagent] cairosvg stubbed (native Cairo absent; only needed "
              "for native VisEval rendering)")

    patch_llm(cfg)
    from core.chat_manager import ChatManager
    from core.const import SYSTEM_NAME

    # nvAgent's Validator injects the database folder into generated code as a
    # raw single-quoted Python string ("data_folder = '{db_path}'"). A Windows
    # path puts backslashes inside that literal, so "...\\Users\\..." is parsed
    # as a \U unicode escape and every single script dies with SyntaxError
    # before running -- which reads as "nvAgent executes 0/16" when nothing is
    # wrong with nvAgent at all. Its README uses "E:/visEval_dataset", so
    # forward slashes are the supported form. Python accepts them on Windows.
    workspace_posix = str(WORKSPACE).replace("\\", "/")
    manager = ChatManager(data_path=workspace_posix,
                          log_path=str(WORKSPACE / "logs" / "nvagent.log"))

    records = []
    for q in questions:
        rec = Record(id=q["id"], dataset=q["dataset"], question=q["question"],
                     ground_truth=q["reference_answer"],
                     expected_method=q["expected_method"],
                     acceptable_methods=q["acceptable_methods"],
                     expected_chart=q["expected_chart"],
                     tolerance=q["tolerance"], kind=q["kind"])
        msg = {"db_id": q["dataset"], "query": q["question"],
               "tables": dbs[q["dataset"]], "send_to": SYSTEM_NAME,
               "library": args.library}
        try:
            code = manager.start(msg)
            rec.code = code or ""
            vql = msg.get("final_vql", "")
            rec.raw["vql"] = vql
            rec.raw["query_difficulty"] = msg.get("query_difficulty")
            rec.raw["augmented_explanation"] = msg.get("augmented_explanation", "")[:2000]
            rec.declared_chart = (chart_from_vql(vql)
                                  or ", ".join(normalise_chart(code)) or None)
            # The Validator sets `fixed` when it rewrote failing code, and bumps
            # try_times on every pass. This is our RQ3 signal.
            rec.repair_attempts = int(msg.get("try_times", 0) or 0)
            rec.self_corrected = bool(msg.get("fixed", False))

            # nvAgent's own exec path swallows stdout; run the code ourselves in
            # the workspace so executability is measured the same way for
            # every system.
            wd = WORKSPACE / "exec" / q["id"]
            ok, out, err = run_code(rec.code, wd, python=sys.executable, timeout=120)
            rec.executed = ok
            rec.stdout = out[:8000]
            rec.execution_error = None if ok else err[-2000:]

            # nvAgent emits code, not narrative. Its augmented_explanation is
            # the closest thing it has to an interpretation, so score that
            # rather than crediting it with nothing.
            rec.final_answer = msg.get("augmented_explanation", "") or ""
            print(f"  {q['id']}: executed={ok} chart={rec.declared_chart} "
                  f"tries={rec.repair_attempts} fixed={rec.self_corrected}")
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.raw["traceback"] = traceback.format_exc()[-1500:]
            print(f"  {q['id']}: HARNESS ERROR {rec.error}")
        records.append(rec)

    write_results("nvagent", "nvAgent (Ge et al., ACL 2025)", ARCH, NATIVE_EVAL,
                  records,
                  meta={"model": cfg["bench_model"], "library": args.library,
                        "provider": cfg["provider"],
                        "base_url": cfg["openai_base_url"],
                        "adaptations": [
                            "core.llm.api_func monkeypatched from AzureOpenAI to OpenAI",
                            "shared CSVs materialised into VisEval folder layout",
                        ]})


if __name__ == "__main__":
    main()
