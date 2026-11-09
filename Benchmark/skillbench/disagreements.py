"""
Reference-answer audit by consensus (run after score_v2.py). For each question, the final answers of all runs are
grouped (numbers within the question's tolerance of each other count as the same answer). A question is flagged
when a majority of agent answers agree on one answer that is NOT the reference: either the reference is wrong or
the question has a trap most systems fall into. Each flag is then checked by hand and the verdict recorded in
AUDIT below, which the report cites.

    python disagreements.py   ->   results/v2/disagreements.md
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

import score as v1

ROOT = pathlib.Path(__file__).resolve().parent
V2 = ROOT / "results" / "v2"

# Hand-audit verdicts for flagged questions: id -> (verdict, note). verdict: "reference correct (trap)",
# "reference corrected", "question reworded", "ambiguous - excluded".
AUDIT: dict[str, tuple[str, str]] = {
    # consensus = the careless answer: the trap works as designed
    "C01": ("reference correct (trap)", "pooled comparison reverses within groups"),
    "S03": ("reference correct (trap)", "Simpson's paradox; confounder in another table"),
    "C03": ("reference correct (trap)", "quantity is not recorded; a number cannot be determined"),
    "T02": ("reference correct (trap)", "cash tips are not recorded"),
    "T19": ("reference correct (trap)", "cash tips are not recorded"),
    "D23": ("reference correct (trap)", "'?' is a missing-value code, not a category"),
    "D03": ("reference correct (trap)", "weight is coded '?' when missing (97% of encounters); 100% counts the code as data"),
    "O02": ("reference correct (trap)", "customer_id is per order; customer_unique_id identifies the person"),
    # consensus is a different, wrong method (recomputed by hand)
    "B14": ("reference correct", "consensus 0.0748 = hour and season as numbers; question says categorical"),
    "R05": ("reference correct", "agent condition reproduces 157.03; 208.85 from a wrong method"),
    "SA2": ("reference correct (trap)", "322.5 = censored accounts treated as cancelled; Kaplan-Meier gives 407"),
    "T08": ("reference correct (trap)", "Saturday only if out-of-month trips are kept; Thursday within January"),
    # correct analyses reported at lower precision: fixed in scoring (rounding rule, all conditions)
    "FL11": ("scoring fixed", "25.8 vs 25.78: rounding rule"),
    "FL12": ("scoring fixed", "3.9 vs 3.892: rounding rule"),
    "I02": ("scoring fixed", "64.3 vs 64.27: rounding rule"),
    # two defensible readings give different answers: excluded from the analysis, all conditions
    "R18": ("ambiguous - excluded", "timestamp gaps (41.94) vs calendar-day gaps (41)"),
    "CB06": ("ambiguous - excluded", "same station by name (1.961%) vs by station id (1.91%)"),
}
EXCLUDED = {k for k, (v, _) in AUDIT.items() if v.startswith("ambiguous")}


def main(min_share=0.5):
    qs = {q["id"]: q for q in json.loads((ROOT / "questions" / "questions.json").read_text(encoding="utf-8"))}
    df = pd.read_csv(V2 / "scores.csv")
    df = df[df.condition != "plain"]
    rows = []
    for qid, g in df.groupby("id"):
        q = qs[qid]
        finals = [f for f in g.final.tolist() if isinstance(f, str) or pd.notna(f)]
        if not finals:
            continue
        groups = []  # [representative, count]
        for f in finals:
            for grp in groups:
                if same(q, f, grp[0]):
                    grp[1] += 1
                    break
            else:
                groups.append([f, 1])
        rep, cnt = max(groups, key=lambda x: x[1])
        share = cnt / len(g)
        ref_ok = v1.correct(q, rep)[0]
        if share >= min_share and not ref_ok:
            naive = v1.correct(q, rep)[2]
            rows.append((qid, q["dataset"], q["trap"], q["reference"], rep, cnt, len(g), naive))
    out = ["# Consensus disagreements with the reference", "",
           f"Questions where at least {min_share:.0%} of agent answers (all models, runs, agent conditions) agree on "
           "one answer that differs from the reference.", "",
           f"{len(rows)} of {df.id.nunique()} questions flagged.", "",
           "| ID | Dataset | Trap | Reference | Consensus answer | Support | = careless answer | Audit |",
           "|---|---|---|---|---|---|---|---|"]
    for qid, ds, trap, ref, rep, cnt, n, naive in sorted(rows):
        a = AUDIT.get(qid, ("pending", ""))
        out.append(f"| {qid} | {ds} | {trap} | {short(ref)} | {short(rep)} | {cnt}/{n} | {'yes' if naive else 'no'} "
                   f"| {a[0]}{': ' + a[1] if a[1] else ''} |")
    (V2 / "disagreements.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


def same(q, a, b):
    if q["answer_type"] == "number":
        x, y = v1.num(a), v1.num(b)
        if x is None or y is None:
            return False
        tol = q["tolerance"]
        lim = max(tol.get("abs", 0), tol.get("rel", 0) * abs(y), 1e-6 * max(abs(y), 1))
        return abs(x - y) <= lim
    return v1.norm(a) == v1.norm(b)


def short(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)[:40]


if __name__ == "__main__":
    main()
