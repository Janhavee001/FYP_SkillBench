"""
Reference-answer review sheet for a human reviewer: one row per question with the question, the reference answer,
the careless answer, the method and the exact code that computes the reference. The reviewer marks each row
Agree / Disagree / Unsure. Sheet "Sample" holds a stratified random sample (fixed seed) of real-data questions;
sheet "All questions" holds all of them.

    python make_review_sheet.py [--n 30]   ->   results/reference_review.xlsx
After review:  python make_review_sheet.py --summarise   ->   agreement rate in results/reference_review.md
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "reference_review.xlsx"
SYNTHETIC = {"clinic", "school", "saas", "shop"}


def reference_code() -> dict[str, str]:
    """For each q("ID", ...) call: the statements since the previous q call, i.e. the code that builds it."""
    code = {}
    for src_file in ["build_questions.py", "questions_v2.py"]:
        src = (ROOT / src_file).read_text(encoding="utf-8")
        for fn in ast.walk(ast.parse(src)):
            if not isinstance(fn, ast.FunctionDef):
                continue
            pending = []
            for st in fn.body:
                pending.append(st)
                if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and getattr(st.value.func, "id", None) == "q" and st.value.args
                        and isinstance(st.value.args[0], ast.Constant)):
                    seg = "\n".join(ast.get_source_segment(src, s) for s in pending)
                    code[st.value.args[0].value] = seg[-3000:]
                    pending = []
    return code


def fmt(v):
    if isinstance(v, float):
        return round(v, 6)
    return v if v is not None else ""


def build(n):
    qs = pd.DataFrame(json.loads((ROOT / "questions" / "questions.json").read_text(encoding="utf-8")))
    code = reference_code()
    real = qs[~qs.dataset.isin(SYNTHETIC)]
    # stratified by difficulty, trap questions first in each stratum's draw
    share = real.difficulty.value_counts(normalize=True)
    parts = [real[real.difficulty == d].sample(n=max(1, round(n * share[d])), random_state=7) for d in share.index]
    sample = pd.concat(parts).sort_values("id")

    wb = Workbook()
    head = ["ID", "Dataset", "Difficulty", "Trap", "Question", "Reference answer", "Tolerance", "Careless answer",
            "Method", "Code computing the reference", "Verdict", "Comment"]
    widths = [7, 13, 9, 16, 50, 16, 11, 14, 36, 80, 11, 30]
    for title, df in [("Sample", sample), ("All questions", qs)]:
        ws = wb.active if title == "Sample" else wb.create_sheet(title)
        ws.title = title
        ws.append(head)
        for c, w in zip("ABCDEFGHIJKL", widths):
            ws.column_dimensions[c].width = w
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="EEEEEE")
        for _, r in df.iterrows():
            ws.append([r.id, r.dataset, r.difficulty, r.trap, r.question, fmt(r.reference), json.dumps(r.tolerance),
                       fmt(r.naive), r.expected_method, code.get(r.id, "(not found)"), "", ""])
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            row[9].font = Font(name="Consolas", size=9)
        dv = DataValidation(type="list", formula1='"Agree,Disagree,Unsure"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"K2:K{ws.max_row}")
        ws.freeze_panes = "F2"
    ins = wb.create_sheet("How to review", 0)
    for line in [
        "Reference-answer review (SkillBench v2)",
        "",
        "For each row in 'Sample': read the question, then the code that computes the reference answer.",
        "Mark Verdict = Agree if the code answers the question as worded and the reference is right.",
        "Mark Disagree if the code answers a different question, misreads the data, or the number is wrong; say why.",
        "Mark Unsure if the question is ambiguous (two reasonable readings give different answers).",
        "The 'Careless answer' is what a shortcut gives; it should be wrong, and the trap should be real.",
        "About 1-2 minutes per row. The data files are in Benchmark/skillbench/data/ if a check is needed.",
    ]:
        ins.append([line])
    ins.column_dimensions["A"].width = 110
    ins["A1"].font = Font(bold=True, size=14)
    OUT.parent.mkdir(exist_ok=True)
    wb.save(OUT)
    print(f"saved {OUT}: {len(sample)} sampled, {len(qs)} total, code found for "
          f"{sum(i in code for i in qs.id)}/{len(qs)}")


# How each Disagree / Unsure was resolved (checked against the data after the review)
RESOLUTIONS = {
    "I11": "Checked: crew.parquet is filtered to the feature films at build time (build_data.py, crew.tconst.isin(ids)); "
           "all 759,901 crew rows are in movies.parquet, and restricting to films gives the same 60,907. "
           "Reference kept.",
}


def summarise():
    ws = load_workbook(OUT)["Sample"]
    rows = [[c.value for c in r] for r in ws.iter_rows(min_row=2)]
    v = pd.Series([r[10] for r in rows if r[10]])
    n = len(rows)
    lines = ["# Reference-answer review", "", f"{len(v)} of {n} sampled questions reviewed.", ""]
    for k in ["Agree", "Disagree", "Unsure"]:
        lines.append(f"- {k}: {int((v == k).sum())}")
    if len(v):
        lines.append(f"\nAgreement rate: {(v == 'Agree').mean():.0%}")
    lines += ["", "Disagreements and unsure:", ""]
    lines += [f"- {r[0]}: {r[10]} - {r[11] or ''}\n  - Resolution: {RESOLUTIONS.get(r[0], 'open')}"
              for r in rows if r[10] in ("Disagree", "Unsure")]
    open_ = [r[0] for r in rows if r[10] in ("Disagree", "Unsure") and r[0] not in RESOLUTIONS]
    lines += ["", f"After resolution: {sum(r[10] == 'Agree' for r in rows) + len(RESOLUTIONS)} of {len(v)} references "
                  f"confirmed; {len(open_)} open."]
    (ROOT / "results" / "reference_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    summarise() if a.summarise else build(a.n)
