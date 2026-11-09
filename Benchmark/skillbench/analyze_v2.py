"""
Analyse SkillBench v2 (run score_v2.py first). Writes results/v2/summary.md and results/v2/figures/.

Headline numbers use REAL datasets only; the synthetic appendix is reported separately.

Statistics
  - accuracy per model x condition: mean over runs, SD across runs, 95% CI by bootstrap over questions
    (each question's accuracy averaged over its runs)
  - paired condition comparisons: mean per-question difference, its bootstrap 95% CI, and a two-sided
    sign-flip permutation test (20,000 permutations) on the per-question differences
  - skill selection loss (RQ2): oracle-skills minus selected-skills accuracy
  - model x condition: is the skills gain larger for the weaker model?
  - breakdowns by difficulty, trap, single vs multi-table, category; trap-fall rates; skill-selection rates; cost
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
V2 = ROOT / "results" / "v2"
FIG = V2 / "figures"
CONDS = ["plain", "agent", "agent_skills", "agent_skills_oracle", "agent_skills_validated"]
LABEL = {"plain": "Plain (one shot)", "agent": "Agent", "agent_skills": "Agent + skills",
         "agent_skills_oracle": "Agent + oracle skill",
         "agent_skills_validated": "Agent + skills + validator"}
COLORS = {"plain": "#9AA5B1", "agent": "#4C78A8", "agent_skills": "#2F6F7E", "agent_skills_oracle": "#B07AA1", "agent_skills_validated": "#E39C37"}
RNG = np.random.default_rng(0)
STATS: dict = {"accuracy": [], "paired": [], "breakdown": [], "selection": [], "quality": [], "cost": [],
              "validator": []}


def pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def per_question(df, model, cond):
    d = df[(df.model == model) & (df.condition == cond)]
    return d.groupby("id").correct.mean()


def boot_ci(v, n=10_000):
    v = np.asarray(v, float)
    return np.percentile(RNG.choice(v, (n, len(v))).mean(axis=1), [2.5, 97.5])


def perm_test(diff, n=20_000):
    diff = np.asarray(diff, float)
    obs = diff.mean()
    signs = RNG.choice([-1, 1], (n, len(diff)))
    null = (signs * diff).mean(axis=1)
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def section_accuracy(df, models, title):
    out = [f"\n## {title}\n", "| Model | Condition | Accuracy (mean of runs) | SD across runs | 95% CI | Runs | "
           "Fell into trap |", "|---|---|---|---|---|---|---|"]
    for m in models:
        for c in CONDS:
            d = df[(df.model == m) & (df.condition == c)]
            if d.empty:
                continue
            by_run = d.groupby("run").correct.mean()
            pq = d.groupby("id").correct.mean()
            lo, hi = boot_ci(pq)
            trap = d[d.has_trap].fell_into_trap.mean()
            STATS["accuracy"].append({"set": title, "model": m, "condition": c, "acc": by_run.mean(),
                                      "sd": by_run.std(ddof=1) if len(by_run) > 1 else 0.0, "lo": lo, "hi": hi,
                                      "runs": len(by_run), "trap_fall": trap})
            out.append(f"| {m} | {LABEL[c]} | {pct(by_run.mean())} | {100 * by_run.std(ddof=1) if len(by_run) > 1 else 0:.1f} pts | "
                       f"{pct(lo)} to {pct(hi)} | {len(by_run)} | {pct(trap)} |")
    return out


def section_paired(df, models):
    out = ["\n## Paired comparisons (real data; per-question accuracy averaged over runs)\n",
           "| Model | Comparison | Difference | 95% CI | p (permutation) | Questions differing |", "|---|---|---|---|---|---|"]
    pairs = [("plain", "agent"), ("agent", "agent_skills"), ("agent", "agent_skills_oracle"),
             ("agent_skills", "agent_skills_oracle"), ("plain", "agent_skills"),
             ("agent_skills", "agent_skills_validated")]
    for m in models:
        for a, b in pairs:
            qa, qb = per_question(df, m, a), per_question(df, m, b)
            common = qa.index.intersection(qb.index)
            if len(common) == 0:
                continue
            diff = (qb[common] - qa[common]).to_numpy()
            lo, hi = boot_ci(diff)
            p = perm_test(diff)
            STATS["paired"].append({"model": m, "a": a, "b": b, "diff": diff.mean(), "lo": lo, "hi": hi, "p": p,
                                    "n_diff": int((diff != 0).sum()), "n": len(diff)})
            out.append(f"| {m} | {LABEL[a]} -> {LABEL[b]} | {100 * diff.mean():+.1f} pts | {100 * lo:+.1f} to "
                       f"{100 * hi:+.1f} | {p:.3g} | {(diff != 0).sum()} of {len(diff)} |")
    return out


def section_breakdown(df, models, by, title, order=None):
    out = [f"\n## By {title} (real data)\n"]
    for m in models:
        d = df[df.model == m]
        t = d.groupby([by, "condition"]).correct.mean().unstack()
        cs = [c for c in CONDS if c in t.columns]
        n = d[d.condition == cs[0]].groupby(by).id.nunique()
        idx = [i for i in (order or sorted(t.index, key=str)) if i in t.index]
        out += [f"\n**{m}**\n", f"| {title} | n | " + " | ".join(LABEL[c] for c in cs) + " |",
                "|---" * (len(cs) + 2) + "|"]
        for k in idx:
            out.append(f"| {k} | {n.get(k, 0)} | " + " | ".join(pct(t.loc[k, c]) for c in cs) + " |")
            STATS["breakdown"] += [{"by": by, "model": m, "level": str(k), "n": int(n.get(k, 0)), "condition": c,
                                    "acc": t.loc[k, c]} for c in cs]
        ax = t.loc[idx, cs].plot.bar(figsize=(7.5, 3.8), color=[COLORS[c] for c in cs], width=0.82)
        ax.set_ylim(0, 1); ax.set_ylabel("share correct"); ax.set_xlabel(title); ax.set_title(m, fontsize=10)
        ax.legend([LABEL[c] for c in cs], frameon=False, fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        plt.xticks(rotation=0 if by != "category" else 30, ha="right" if by == "category" else "center")
        plt.tight_layout(); plt.savefig(FIG / f"{m}_by_{by}.png", dpi=150); plt.close()
    return out


def section_validator(df, models, root=V2):
    """RQ3: flags and repairs of the validator, relative to the same first answer (agent_skills, same run)."""
    out = ["\n## Validator (RQ3): agent + skills answers, before and after validation\n",
           "| Model | Answers | Flagged | Wrong answers flagged (detection) | Correct answers flagged (false alarm) | "
           "Fixed (wrong to right) | Broken (right to wrong) | Net |", "|---|---|---|---|---|---|---|---|"]
    for m in models:
        a = df[(df.model == m) & (df.condition == "agent_skills")].set_index(["run", "id"]).correct
        v = df[(df.model == m) & (df.condition == "agent_skills_validated")].set_index(["run", "id"]).correct
        if v.empty:
            continue
        flags = {}
        for f in root.glob(f"{m}/run*/agent_skills_validated/*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            flags[(f.parts[-3], r.get("id"))] = bool(r.get("flagged"))
        idx = a.index.intersection(v.index)
        a, v = a[idx], v[idx]
        fl = pd.Series([flags.get(i, False) for i in idx], index=idx)
        n, wrong, right = len(idx), a == 0, a == 1
        row = {"model": m, "n": n, "flagged": fl.mean(), "detection": fl[wrong].mean() if wrong.any() else None,
               "false_alarm": fl[right].mean() if right.any() else None, "fixed": int(((a == 0) & (v == 1)).sum()),
               "broken": int(((a == 1) & (v == 0)).sum())}
        row["net"] = row["fixed"] - row["broken"]
        STATS["validator"].append(row)
        out.append(f"| {m} | {n} | {pct(row['flagged'])} | {pct(row['detection'])} | {pct(row['false_alarm'])} | "
                   f"{row['fixed']} | {row['broken']} | {row['net']:+d} |")
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    from disagreements import EXCLUDED  # questions found ambiguous in the consensus audit; dropped for all conditions
    df = pd.read_csv(V2 / "scores.csv")
    df = df[~df.id.isin(EXCLUDED)]
    models = sorted(df.model.unique(), key=lambda m: ("20b" in m, m))
    real, syn = df[~df.synthetic], df[df.synthetic]
    nq = real.id.nunique()
    out = ["# SkillBench v2 results\n",
           f"{nq} real-data questions ({real.dataset.nunique()} datasets) and {syn.id.nunique()} synthetic appendix "
           f"questions. Models: {', '.join(models)}. Temperature 0.7, {df.run.nunique()} runs per condition.\n"]
    out += section_accuracy(real, models, "Accuracy on real data (headline)")
    out += section_paired(real, models)

    # model x condition: skills gain by model
    out += ["\n## Does the skills gain depend on model strength? (real data)\n",
            "| Model | Agent | Agent + skills | Gain | Agent + oracle skill | Oracle gain |", "|---|---|---|---|---|---|"]
    for m in models:
        a, s, o = (per_question(real, m, c) for c in ("agent", "agent_skills", "agent_skills_oracle"))
        if a.empty:
            continue
        out.append(f"| {m} | {pct(a.mean())} | {pct(s.mean())} | {100 * (s.mean() - a.mean()):+.1f} pts | "
                   f"{pct(o.mean()) if not o.empty else 'n/a'} | "
                   f"{(100 * (o.mean() - a.mean())) if not o.empty else float('nan'):+.1f} pts |")

    out += section_breakdown(real, models, "difficulty", "difficulty", ["easy", "medium", "hard"])
    out += section_breakdown(real, models, "has_trap", "trap present")
    out += section_breakdown(real, models, "multi_table", "multi-table")
    out += section_breakdown(real, models, "category", "category")

    # traps
    out += ["\n## Trap types (real data, share correct)\n"]
    for m in models:
        d = real[(real.model == m) & real.has_trap]
        t = d.groupby(["trap", "condition"]).correct.mean().unstack()
        cs = [c for c in CONDS if c in t.columns]
        n = d[d.condition == cs[0]].groupby("trap").id.nunique()
        out += [f"\n**{m}**\n", "| Trap | n | " + " | ".join(LABEL[c] for c in cs) + " |", "|---" * (len(cs) + 2) + "|"]
        for k in t.sort_index().index:
            out.append(f"| {k} | {n.get(k, 0)} | " + " | ".join(pct(t.loc[k, c]) for c in cs) + " |")

    # skill selection
    out += ["\n## Skill selection, Agent + skills (real data)\n",
            "| Model | Loaded any skill | Loaded the oracle skill | Accuracy when loaded | Accuracy when not |",
            "|---|---|---|---|---|"]
    import run as runner
    qs = {q["id"]: q for q in __import__("json").loads((ROOT / "questions" / "questions.json").read_text(encoding="utf-8"))}
    for m in models:
        s = real[(real.model == m) & (real.condition == "agent_skills")].copy()
        if s.empty:
            continue
        s["loaded"] = s.skills_loaded.fillna("").astype(str).str.len() > 0
        s["hit"] = s.apply(lambda r: bool(set(runner.oracle_for(qs[r.id])) & set(str(r.skills_loaded).split(","))),
                           axis=1)
        STATS["selection"].append({"model": m, "loaded": s.loaded.mean(), "hit": s.hit.mean(),
                                   "acc_loaded": s[s.loaded].correct.mean(), "acc_not": s[~s.loaded].correct.mean()})
        out.append(f"| {m} | {pct(s.loaded.mean())} | {pct(s.hit.mean())} | {pct(s[s.loaded].correct.mean())} | "
                   f"{pct(s[~s.loaded].correct.mean())} |")

    # quality + cost
    out += ["\n## Analytic quality (judge: different model family; real data)\n",
            "| Model | Condition | Method appropriate | Caveats stated | No overclaiming |", "|---|---|---|---|---|"]
    for m in models:
        for c in CONDS:
            d = real[(real.model == m) & (real.condition == c)]
            if not d.empty:
                STATS["quality"].append({"model": m, "condition": c, "method": d.method_appropriate.mean(),
                                         "caveats": d.caveats_stated.mean(), "no_overclaim": d.no_overclaiming.mean()})
                out.append(f"| {m} | {LABEL[c]} | {pct(d.method_appropriate.mean())} | {pct(d.caveats_stated.mean())} | "
                           f"{pct(d.no_overclaiming.mean())} |")
    out += ["\n## Cost per question (mean, real data)\n",
            "| Model | Condition | Code runs | Code errors | Prompt tokens | Completion tokens | Seconds |",
            "|---|---|---|---|---|---|---|"]
    for m in models:
        for c in CONDS:
            d = real[(real.model == m) & (real.condition == c)]
            if not d.empty:
                STATS["cost"].append({"model": m, "condition": c, "steps": d.steps.mean(),
                                      "code_errors": d.code_errors.mean(), "prompt_tokens": d.prompt_tokens.mean(),
                                      "completion_tokens": d.completion_tokens.mean(), "seconds": d.seconds.mean()})
                out.append(f"| {m} | {LABEL[c]} | {d.steps.mean():.1f} | {d.code_errors.mean():.2f} | "
                           f"{d.prompt_tokens.mean():,.0f} | {d.completion_tokens.mean():,.0f} | {d.seconds.mean():.0f} |")

    out += section_validator(real, models)

    # synthetic appendix
    out += section_accuracy(syn, models, "Appendix: synthetic planted-trap databases")

    # headline figure
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.6), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        cs = [c for c in CONDS if not real[(real.model == m) & (real.condition == c)].empty]
        vals, errs = [], []
        for c in cs:
            pq = per_question(real, m, c)
            lo, hi = boot_ci(pq)
            vals.append(pq.mean()); errs.append([pq.mean() - lo, hi - pq.mean()])
        ax.bar(range(len(cs)), vals, yerr=np.array(errs).T, capsize=4, color=[COLORS[c] for c in cs])
        ax.set_xticks(range(len(cs)), [LABEL[c].replace(" + ", "\n+ ") for c in cs], fontsize=8)
        ax.set_title(m, fontsize=10); ax.set_ylim(0, 1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("share correct (95% CI)")
    plt.tight_layout(); plt.savefig(FIG / "accuracy_overall.png", dpi=150); plt.close()

    (V2 / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    STATS["meta"] = {"n_real": int(nq), "n_syn": int(syn.id.nunique()), "answers": int(len(df)), "models": models,
                     "real_datasets": int(real.dataset.nunique()), "runs": sorted(df.run.unique().tolist()),
                     "excluded": sorted(EXCLUDED),
                     "rounded": int(df["rounded"].sum()) if "rounded" in df else 0}
    (V2 / "stats.json").write_text(json.dumps(STATS, indent=1, default=lambda x: None if pd.isna(x) else float(x)),
                                   encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
