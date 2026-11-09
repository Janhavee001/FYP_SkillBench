# SkillBench v1: results report

Model `openai/gpt-oss-120b` (via OpenRouter), temperature 0, one run per condition. 108 questions over
9 datasets (5 single-table, 4 multi-table), 72 containing a data trap. Full tables: `summary.md`.
Per-answer scores: `scores.csv`.

## Headline

| Condition | Correct | 95% CI (bootstrap) |
|---|---|---|
| Plain (one shot) | 86.1% (93/108) | 79.6 to 92.6% |
| Agent (no skills) | 93.5% (101/108) | 88.9 to 97.2% |
| Agent + skills | **95.4% (103/108)** | 90.7 to 99.1% |

Paired McNemar tests on the same questions:

| Comparison | Result |
|---|---|
| Plain vs Agent + skills | 11 vs 1 discordant answers, **p = 0.006** |
| Plain vs Agent | 10 vs 2, p = 0.039 |
| Agent vs Agent + skills | 4 vs 2, p = 0.69 (not significant) |

## Findings

1. **Skills plus an agent loop clearly beat one-shot analysis.** Most of that gain comes from the loop:
   letting the model inspect the data before answering. Skills add a smaller gain on top, which is not
   statistically significant at n = 108.
2. **The benefit concentrates on hard questions.** Accuracy on the 25 hard questions:

   | Plain | Agent | Agent + skills |
   |---|---|---|
   | 68% | 80% | 92% |

   On easy questions all conditions are at about 100%. The skills condition is the only one that:
   - excluded deceased patients before computing a readmission rate (D12);
   - aggregated invoice lines to invoices before comparing countries (R05).
3. **Three traps defeat every condition.** These are the benchmark's discriminating frontier:
   - **T02, unrecorded values:** concluding "card payers tip more" when cash tips are not recorded.
   - **T08, out-of-range dates:** counting February-dated trips in a January average.
   - **S03, Simpson's paradox:** the confounder is in another table.

   No skill currently covers "values that are structurally unrecorded". The relational skill covers
   Simpson's paradox, but the agent did not run its reversal check.
4. **Skill selection (RQ2) is the weak link.**
   - The agent loaded a skill for only 52% of questions.
   - It loaded one that fits the question's category for only 39%.
   - `data-profiling` dominated (38 of 58 loads), while `group-comparison` was loaded twice and
     `time-series-analysis` once.

   So the skill content is mostly not being used yet; the measured effect is a lower bound on what the
   skills could contribute.
5. **Cost:** the skills condition used about 2.4 times the prompt tokens of the agent condition (20k vs
   8k per question), because skill text is re-sent on each turn. Wall time rose from 51 s to 62 s.
6. **Judge-rated quality is mixed.**
   - **Overclaiming:** the skills condition overclaimed least (98% free of overclaiming, vs 94% for the agent).
   - **Caveats:** it stated caveats less often (78% vs 83%).
   - **One-shot answers** stated caveats most often (94%), but were also least often correct.

## Errors the audit fixed before these numbers

The first scoring pass was audited answer by answer; four problems were fixed, identically for every condition:
1. **Category answers were matched too strictly.** "The United Kingdom", "Autodromo Nazionale Monza" and
   "Roraima" were marked wrong.
2. **Six agent answers were lost** because the model wrote its submission as text instead of calling the
   tool. They were recovered from the transcripts.
3. **Three ambiguous questions** (R10, R11, R12) were reworded and rerun under all conditions.
4. **References re-verified:** T08, P07 and the rest were checked and kept.

## Limitations

- **One run per condition.** Run-to-run variance is not yet measured. With a 2-answer gap between
  agent and skills, a repeat could change the order.
- **n = 108 is too small** to detect a 2-point difference between strong conditions. Roughly 600 paired
  questions would be needed for 80% power at that effect size.
- **The model and judge are the same family.** Correctness is deterministic, but the quality scores may
  carry self-preference bias.
- **References were written and checked by one person** (with assistance). A second independent check of
  the gold answers is needed before publication.
- **The strong agent baseline is near ceiling on easy and medium questions.**

## What would make this publication-ready

1. **More hard questions:** grow the hard and trap-heavy portion (about 300 questions, most of them hard),
   modelled on the three traps that defeat every condition today.
2. **Three or more seeds per condition,** reporting the mean and spread.
3. **A forced-skill ablation:** give the agent the right skill up front, separating "the skill helps when
   used" (RQ1) from "the agent picks it" (RQ2).
4. **A second, weaker model,** where skills are expected to matter more.
5. **Independent validation of every reference answer** by a second person.
