"""
Analyse scored SkillBench results (run score.py first) and write results/summary.md plus figures.

Statistics:
  - accuracy per condition with a percentile bootstrap 95% CI (10,000 resamples over questions)
  - paired comparisons on the same questions: McNemar's exact test (binomial on discordant pairs)
  - breakdowns by difficulty, trap vs no trap, single vs multi-table, category
  - trap-fall rate: wrong answers that equal the precomputed naive answer
  - skill selection (agent_skills only): share of questions where a skill was loaded, and where a
    loaded skill matches the question's category
  - efficiency: steps, tokens, seconds
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = RES / "figures"
ORDER = ["plain", "agent", "agent_skills"]
LABEL = {"plain": "Plain (one shot)", "agent": "Agent (no skills)", "agent_skills": "Agent + skills"}
EXPECTED_SKILLS = {
    "data_quality": {"data-profiling"}, "descriptive": {"data-profiling", "distribution-analysis"},
    "group_comparison": {"group-comparison"}, "correlation": {"correlation-analysis"},
    "regression": {"regression-analysis"}, "time_series": {"time-series-analysis"},
    "causal": {"group-comparison", "regression-analysis", "relational-data-analysis"},
}
MULTI = {"olist", "f1db", "movielens", "shop"}


def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def mcnemar(a, b):
    """Exact McNemar on paired 0/1 vectors: returns (b_only, a_only, p)."""
    a, b = np.asarray(a), np.asarray(b)
    a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else stats.binomtest(min(a_only, b_only), n, 0.5).pvalue
    return b_only, a_only, p


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    df = pd.read_csv(RES / "scores.csv")
    df["multi_table"] = df.dataset.isin(MULTI)
    conds = [c for c in ORDER if c in set(df.condition)]
    common = set.intersection(*(set(df[df.condition == c].id) for c in conds))
    df = df[df.id.isin(common)]
    n_q = len(common)
    FIG.mkdir(parents=True, exist_ok=True)
    out = [f"# SkillBench results\n", f"{n_q} questions answered under all {len(conds)} conditions; "
           f"model `{'openai/gpt-oss-120b'}`, temperature 0.\n"]

    # ---------------- headline
    out += ["## Accuracy\n", "| Condition | Correct | 95% CI | Fell into trap | Unit slips | Harness errors |",
            "|---|---|---|---|---|---|"]
    head = {}
    for c in conds:
        d = df[df.condition == c]
        lo, hi = boot_ci(d.correct)
        head[c] = (d.correct.mean(), lo, hi)
        trap_rate = d[d.has_trap].fell_into_trap.mean()
        out.append(f"| {LABEL[c]} | {pct(d.correct.mean())} ({d.correct.sum()}/{len(d)}) | {pct(lo)} to {pct(hi)} | "
                   f"{pct(trap_rate)} of trap questions | {d.unit_slip.sum()} | {d.harness_error.sum()} |")

    # ---------------- paired tests
    out += ["\n## Paired comparisons (McNemar exact test, same questions)\n",
            "| Comparison | Only first correct | Only second correct | p |", "|---|---|---|---|"]
    piv = df.pivot_table(index="id", columns="condition", values="correct")
    for a, b in [("plain", "agent"), ("agent", "agent_skills"), ("plain", "agent_skills")]:
        if a in piv and b in piv:
            b_only, a_only, p = mcnemar(piv[a], piv[b])
            out.append(f"| {LABEL[a]} vs {LABEL[b]} | {a_only} | {b_only} | {p:.3g} |")

    # ---------------- breakdowns
    def table(by, title):
        t = df.groupby([by, "condition"]).correct.mean().unstack()[conds]
        n = df[df.condition == conds[0]].groupby(by).size()
        rows = [f"\n## By {title}\n", "| " + title + " | n | " + " | ".join(LABEL[c] for c in conds) + " |",
                "|---" * (len(conds) + 2) + "|"]
        for k in t.index:
            rows.append(f"| {k} | {n[k]} | " + " | ".join(pct(t.loc[k, c]) for c in conds) + " |")
        return rows, t
    for by, title in [("difficulty", "difficulty"), ("has_trap", "trap present"), ("multi_table", "multi-table"),
                      ("category", "category"), ("dataset", "dataset")]:
        rows, t = table(by, title)
        out += rows
        if by in ("difficulty", "has_trap", "multi_table", "category"):
            ax = t.loc[[i for i in (["easy", "medium", "hard"] if by == "difficulty" else t.index)]].plot.bar(
                figsize=(7, 3.8), color=["#9AA5B1", "#4C78A8", "#2F6F7E"][: len(conds)], width=0.8)
            ax.set_ylabel("share correct"); ax.set_ylim(0, 1); ax.set_xlabel(title)
            ax.legend([LABEL[c] for c in conds], frameon=False, fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            plt.xticks(rotation=0 if by != "category" else 30, ha="right" if by == "category" else "center")
            plt.tight_layout(); plt.savefig(FIG / f"accuracy_by_{by}.png", dpi=150); plt.close()

    # ---------------- traps
    tr = df[df.has_trap].groupby(["trap", "condition"]).correct.mean().unstack()[conds]
    out += ["\n## Trap questions: share correct by trap type\n",
            "| Trap | n | " + " | ".join(LABEL[c] for c in conds) + " |", "|---" * (len(conds) + 2) + "|"]
    ntr = df[(df.condition == conds[0]) & df.has_trap].groupby("trap").size()
    for k in tr.sort_index().index:
        out.append(f"| {k} | {ntr[k]} | " + " | ".join(pct(tr.loc[k, c]) for c in conds) + " |")

    # ---------------- quality (judge)
    out += ["\n## Analytic quality (LLM judge, share of answers)\n",
            "| Condition | Method appropriate | Caveats stated | No overclaiming |", "|---|---|---|---|"]
    for c in conds:
        d = df[df.condition == c]
        out.append(f"| {LABEL[c]} | {pct(d.method_appropriate.mean())} | {pct(d.caveats_stated.mean())} | "
                   f"{pct(d.no_overclaiming.mean())} |")

    # ---------------- skill selection
    if "agent_skills" in conds:
        s = df[df.condition == "agent_skills"].copy()
        s["loaded"] = s.skills_loaded.fillna("").astype(str).str.len() > 0
        s["skills"] = s.skills_loaded.fillna("").astype(str).str.split(",")

        def fits(r):
            exp = set(EXPECTED_SKILLS.get(r.category, set()))
            if r.multi_table:
                exp |= {"relational-data-analysis"}
            return bool(exp & set(r.skills))
        s["fits"] = s.apply(fits, axis=1)
        out += ["\n## Skill selection (Agent + skills)\n",
                f"- Loaded at least one skill: {pct(s.loaded.mean())} of questions",
                f"- Loaded a skill that fits the question category: {pct(s.fits.mean())} "
                f"({pct(s[s.loaded].fits.mean())} of questions where any skill was loaded)",
                f"- Accuracy when a skill was loaded: {pct(s[s.loaded].correct.mean())} (n={s.loaded.sum()}); "
                f"when none was loaded: {pct(s[~s.loaded].correct.mean())} (n={(~s.loaded).sum()})",
                "\n| Skill | Times loaded |", "|---|---|"]
        counts = pd.Series([k for ks in s.skills for k in ks if k]).value_counts()
        for k, v in counts.items():
            out.append(f"| {k} | {v} |")

    # ---------------- efficiency
    out += ["\n## Cost per question (mean)\n", "| Condition | Code runs | Code errors | Prompt tokens | "
            "Completion tokens | Seconds |", "|---|---|---|---|---|---|"]
    for c in conds:
        d = df[df.condition == c]
        out.append(f"| {LABEL[c]} | {d.steps.mean():.1f} | {d.code_errors.mean():.2f} | {d.prompt_tokens.mean():,.0f} | "
                   f"{d.completion_tokens.mean():,.0f} | {d.seconds.mean():.0f} |")

    # ---------------- headline figure
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = np.arange(len(conds))
    vals = [head[c][0] for c in conds]
    err = np.array([[head[c][0] - head[c][1] for c in conds], [head[c][2] - head[c][0] for c in conds]])
    ax.bar(xs, vals, yerr=err, capsize=5, color=["#9AA5B1", "#4C78A8", "#2F6F7E"][: len(conds)])
    ax.set_xticks(xs, [LABEL[c] for c in conds]); ax.set_ylim(0, 1); ax.set_ylabel("share correct (95% CI)")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "accuracy_overall.png", dpi=150); plt.close()

    (RES / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
