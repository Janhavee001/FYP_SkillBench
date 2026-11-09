# Skills for Reliable Data Analysis with LLM Agents: A Tested Tool Library, Procedural Skills, and Two Benchmarks

Janhavee Singh · Final Year Project, Nanyang Technological University · Draft, November 2026

**Abstract.** LLM agents that analyse data make confident errors on real, messy data. The errors come from
skipped procedural steps (wrong grain, missing-value codes, fan-out joins, unrecorded quantities) more than from
choosing the wrong method. We test whether procedural knowledge packaged as agent skills, on top of a library of
61 tested deterministic tools, makes such analysis more reliable.

The evaluation uses two benchmarks:
- **SkillBench:** 206 questions, 183 of them on 11 uncleaned public datasets of up to 3 million rows; 123 need
  several tables and 140 contain a documented trap.
- **StatABench:** the 198 published tool items, used unchanged.

Both are run under five conditions (plain, agent, agent + skills, agent + oracle skill, and agent + skills +
validator), with gpt-oss-120b and gpt-oss-20b and three runs per agent condition: 10,452 analysed answers in all.

Results:
- **The large model** is unaffected by skills on both benchmarks: 93.0% vs 92.8% on SkillBench, and 64.5% vs
  63.8% on StatABench.
- **The small model** gains from skills on SkillBench: 75.5% → 80.5%, +5.0 points, p = 0.012. On StatABench the
  gain is in the same direction but smaller (+2.2 points, n.s.; +8.1 points over a one-shot baseline,
  p = 0.011).
- **Skill selection is weak:** the small model loads a skill for only 9% of SkillBench questions, and handing it
  the correct skill gives no significant gain.
- **A validation loop** (deterministic grounding checks plus a checklist critic, then repair) adds +6.3 points for
  the small model on SkillBench (p < 0.001) and +3.9 on StatABench (p = 0.013), but is neutral for the large
  model, where it breaks about as many answers as it fixes.

Skills and validation help where the model is weak, not where it is near its ceiling. Some traps (unrecorded
values, Simpson's paradox, censoring) defeat most runs of every condition.

## 1. Introduction

Language-model agents can now write and run analysis code, but their answers on real data are often confident
and wrong. In practice the errors are rarely about knowing a statistical method. They come from the steps an
experienced analyst performs without thinking:

- analysing at the wrong unit (invoice lines instead of invoices);
- treating a missing-value code such as `?` or `\N` as data;
- joining tables so that rows are silently duplicated;
- averaging values that are only recorded for a subset of rows;
- ignoring censoring or confounding.

This project asks whether **procedural knowledge packaged as agent skills** makes LLM data analysis more
reliable. A skill here is a short written procedure, in Anthropic's Agent Skills format. It states:

- when a method applies;
- what to check first;
- which tested function to call;
- how to report the result without overclaiming.

The skills sit on a library of deterministic, tested statistical tools.

Contributions:

1. **statkit**, a library of 61 deterministic analysis tools in 11 categories, including a 9-tool relational
   layer for multi-table data. Every tool returns a typed result with warnings, and the library is covered by
   137 tests that compare tools against answers they did not produce.
2. **Eight skills** packaged for the Claude app and for agent harnesses, each bundling the library.
3. **SkillBench**, a benchmark of 206 questions on 11 large, uncleaned real datasets (up to 3 million rows) and
   4 synthetic databases. 123 of the questions need more than one table, and 140 contain a documented trap.
4. **An evaluation on two benchmarks:** SkillBench, and the published StatABench (Zhu et al. 2026), used
   unchanged. Both are run under four conditions, with two model sizes and three runs. Results are reported with
   bootstrap confidence intervals and paired permutation tests.
5. **A validation of the answer key:**
   - independent reference code;
   - publisher record counts;
   - a consensus audit across all runs;
   - a human review sample.

## 2. Background and phase-1 findings

**StatABench** (Zhu et al. 2026) evaluates statistical analysis with 404 closed items (Stat-Closed):
- 206 theory items and 198 items that require data and tools;
- 83 CSV datasets;
- answers graded by an LLM comparison prompt.

**DataSciBench** (Zhang et al. 2025) evaluates data-science agents on 222 tasks:
- 167 adapted from BigCodeBench;
- 45 curated CSV, human-written and deep-learning tasks;
- graded by task-specific checks of output files.

In phase 1 of this project both benchmarks were run natively with gpt-oss-120b, alongside three analysis
systems (LIDA, Data Formulator 2, nvAgent):

- **StatABench's own 36-tool agent reached 66.2%** on the 198 tool items. Its weakest tasks were generalized
  linear models (40%) and survival analysis (55%).
- **DataSciBench's agent completed about 55%** of checked steps on the curated analysis tasks.
- **A plain LLM chose the correct statistical method 98% of the time** on standard questions over well-known
  datasets.

Two conclusions shaped this work:
1. Method choice on clean, familiar data is largely solved.
2. Neither benchmark tests messy real data, analysis at scale, or questions that span several tables.

## 3. Tool library and skills

**Tools.** Each tool follows one pattern:
1. Check the inputs: columns exist, types fit, there is enough data. If not, return `ok = False` with a reason.
2. Compute with a trusted library (scipy, statsmodels, scikit-learn).
3. Add context: effect size, interval, sample size, rows dropped, assumption checks.
4. Return a typed result with warnings.

No tool contains a language model.

The relational layer covers the multi-table workflow:
- discovering tables;
- inferring primary and foreign keys with evidence;
- planning a join path at the analysis grain;
- aggregating child tables before joining;
- validating each join (rows before and after, fan-out, orphans);
- detecting Simpson's-paradox reversals.

**Verification.** 137 automated tests compare tools against:
- benchmark ground truth;
- reference libraries (agreement to 1e-9);
- published results (e.g. the Freireich 6-MP survival data, checked against R);
- planted structure in synthetic data;
- reproductions of earlier failures.

**Skills.** There are eight skills:
- data-profiling;
- distribution-analysis;
- correlation-analysis;
- group-comparison;
- regression-analysis;
- time-series-analysis;
- visualization;
- relational-data-analysis.

Each SKILL.md has a description used for selection, when to use and when not to, a numbered procedure naming
the tool for each step, assumptions and common errors, a report format, and a worked example. The tool
reference is generated from the code on every build, and every example is executed from the packaged zip in
a clean process.

## 4. Benchmarks

### 4.1 SkillBench

**Data.** Eleven public datasets, kept exactly as published:

| Dataset | Rows |
|---|---|
| NYC Yellow Taxi, Jan 2024 | 2.96 M trips |
| Citi Bike, Jan 2024 | 1.89 M rides |
| US flights, Q1 2024 (BTS) | 1.66 M flights |
| Online Retail II | 1.07 M lines |
| MovieLens 1M | 1.0 M ratings |
| IMDb feature films | 760 k films |
| Olist (9 tables) | 99 k orders |
| Diabetes 130-US | 102 k stays |
| Beijing PM2.5 | 43.8 k hours |
| F1DB (47 tables) | 27.7 k results |
| Bike Sharing DC | 17.4 k hours |

Four small synthetic databases (clinic, school, SaaS, shop) test specific situations under control. They are
reported only as an appendix.

**Questions.** There are 206 questions:
- 183 on real data and 23 synthetic;
- 65 easy, 86 medium and 55 hard;
- 123 multi-table.

Each question has one unambiguous answer: a number with a tolerance, a label, or a choice.

**Construction.** Every question follows one pipeline:
1. Profile the real data to find its genuine problems.
2. Write the question.
3. Compute the reference answer with pandas, scipy and statsmodels only, never with statkit, so the key cannot
   favour the library under test.
4. Compute the *careless* answer, i.e. the result of the obvious shortcut. Keep a trap only if the careless
   answer differs from the reference.

**Traps.** 140 questions contain a trap, in five families:
- grain and unit of analysis;
- joins and linkage;
- missing and unrecorded values;
- dirty or invalid records;
- statistical reasoning.

### 4.2 StatABench (external)

The 198 Stat-Closed tool items are used unchanged: questions, datasets and reference answers ("ground truth2").
Grading uses StatABench's comparison prompt verbatim. It marks a response correct only if it states a
conclusion, or gives sufficient results, consistent with the reference.

## 5. Experimental setup

**Conditions.** Each condition adds one thing:

| Condition | What it gets |
|---|---|
| Plain | Writes one script, which runs once, and answers from its output |
| Agent | A tool loop: run Python up to 10 times, then submit |
| Agent + skills | The same loop plus `load_skill`, with statkit importable; the model chooses the skill |
| Agent + oracle skill | The same loop with the correct skill already loaded |
| Agent + skills + validator (RQ3) | Starts from the *same* first answer as agent + skills, then runs a validator: (a) deterministic checks (some code ran successfully; a submitted number appears in the agent's own code output); (b) a checklist critic (the same model, temperature 0) covering unit of analysis, population, missing values, joins, method, measurability and the form of the answer. If anything is flagged, the concerns go back into the conversation and the agent may run up to 4 more code calls and resubmit. |

The oracle condition separates whether a skill helps when used (RQ1) from whether the agent selects it (RQ2).

**Models.** gpt-oss-120b and gpt-oss-20b via OpenRouter, at temperature 0.7 with fixed seeds.
- Plain: 1 run per model.
- Each agent condition: 3 runs per model.
- Answers per benchmark: 4,120 for SkillBench and 3,960 for StatABench, plus one validator answer per agent + skills answer (1,236 and 1,188).

All conditions receive identical data descriptions, file schemas and question text.

**Scoring.**
- *SkillBench correctness* is deterministic: a number within tolerance, or an exact label or choice match after
  normalisation.
- An LLM extracts the final value only when the submission gave none, and it never judges correctness.
- A *trap fall* is a wrong answer equal to the precomputed careless answer.
- An LLM judge from a different model family (qwen3-235b) scores secondary analytic quality: method fit,
  caveats stated, no overclaiming. It is blind to condition.
- *StatABench* is graded with its own prompt by the same judge, at temperature 0.

**Harness fix found during the run.** The audit of the transcripts showed that gpt-oss sometimes leaks its
chat-format tokens into tool calls, in three ways:
- tool names such as `run_python<|channel|>commentary`;
- JSON arguments with literal newlines inside strings;
- the bare code sent instead of `{"code": ...}`.

The first harness parsed these strictly. It then either ran an empty script and returned "(no stdout)" or
rejected the call as an unknown tool, so the model was given false feedback that its code printed nothing.
- **How common it was:** up to half of gpt-oss-20b's code calls in some runs, apparently depending on which
  provider OpenRouter routed the request to, and far fewer for gpt-oss-120b.
- **The fix:** tool calls are now parsed leniently, identically for every condition.
- **What was redone:** every answer containing at least one affected call (1,982 of 7,188 at the time) was
  moved to quarantine with its cached judgement, together with any validator output derived from it, and was
  rerun.

Answers without an affected call would be identical under the fixed parser, so the result is equivalent to
rerunning everything with the fixed harness. After the fix, 0 of 1,107 sampled code calls that print returned
empty output.

A second check after the rerun searched every transcript for what the agent was actually shown:
- an empty script that failed with no output;
- a rejected tool name containing a chat-format token.

It found 2,683 further answers, almost all in StatABench runs 2 and 3. They were most likely written by a runner
process started before the fix that was still running. These answers were quarantined and rerun in the same way, and the final check finds no
remaining affected answers in either benchmark.

**Statistics.**
- Accuracy is the mean over runs, with the SD across runs.
- The 95% CI is a bootstrap over questions, using per-question accuracy averaged over runs.
- Conditions are compared with a two-sided paired sign-flip permutation test on per-question differences
  (20,000 permutations).
- Headline SkillBench numbers use real data only.

## 6. Results

### 6.1 Accuracy (RQ1)

Table 1 gives SkillBench accuracy on its 181 real-data questions; the two excluded questions are left out.
Agent conditions show the mean of three runs; plain is a single run. Intervals are 95% bootstrap CIs over
questions.

**Table 1. SkillBench accuracy (real data)**

| Condition | gpt-oss-120b | gpt-oss-20b |
|---|---|---|
| Plain | 90.6% (86.2–94.5) | 77.3% (71.3–83.4) |
| Agent | 93.0% (90.1–95.8) | 75.5% (71.1–79.9) |
| Agent + skills | 92.8% (89.5–95.8) | 80.5% (75.7–85.1) |
| Agent + oracle skill | 91.9% (88.6–94.8) | 78.8% (74.6–82.9) |
| Agent + skills + validator | 93.2% (90.1–95.9) | 86.7% (82.9–90.4) |

**Table 2. StatABench accuracy (198 items, StatABench's own grading)**

| Condition | gpt-oss-120b | gpt-oss-20b |
|---|---|---|
| Plain | 61.6% (54.5–68.2) | 48.5% (41.4–55.6) |
| Agent | 64.5% (58.4–70.2) | 54.4% (48.7–59.9) |
| Agent + skills | 63.8% (57.7–69.7) | 56.6% (50.8–62.1) |
| Agent + oracle skill | 64.8% (58.9–70.5) | 59.1% (53.2–64.8) |
| Agent + skills + validator | 63.1% (57.1–69.2) | 60.4% (54.7–66.2) |

**Table 3. Paired differences in points (95% CI; permutation p)**

| Comparison | SkillBench 120b | SkillBench 20b | StatABench 120b | StatABench 20b |
|---|---|---|---|---|
| Agent → agent + skills | −0.2 (−2.4, 2.0); 1.00 | **+5.0 (1.3, 8.7); 0.012** | −0.7 (−4.2, 2.9); 0.78 | +2.2 (−2.4, 6.7); 0.38 |
| Plain → agent + skills | +2.2 (−1.8, 6.4); 0.33 | +3.1 (−2.4, 8.8); 0.30 | +2.2 (−2.7, 7.1); 0.42 | **+8.1 (2.4, 14.0); 0.011** |
| Agent + skills → oracle | −0.9 (−3.1, 1.3); 0.53 | −1.7 (−5.5, 2.2); 0.45 | +1.0 (−2.7, 4.7); 0.66 | +2.5 (−1.7, 6.9); 0.28 |
| Agent + skills → validator | +0.4 (−1.7, 2.6); 0.87 | **+6.3 (3.9, 8.8); < 0.001** | −0.7 (−3.2, 1.9); 0.71 | **+3.9 (1.0, 6.9); 0.013** |

**The large model is unaffected by skills.**
- On SkillBench it is near the ceiling (91–93% in every condition).
- On StatABench there is real headroom (about 36% of answers are wrong), and still no condition differs
  significantly.
- Its agent scores 64.5% on StatABench, close to the 66.2% of StatABench's own 36-tool agent in phase 1, which
  indicates the harness is comparable.

**The small model benefits.**
- On SkillBench the plain agent loop does not beat one-shot analysis (−1.8 points). Skills then add a
  significant +5.0 points.
- On StatABench the ordering replicates (plain < agent < skills < oracle), but the skills-over-agent difference is
  smaller and not significant on its own.

**Difficulty.** For the small model the gain is largest on hard questions: 61.6% (agent), 69.6% (skills) and
77.5% (validator). For the large model, skills lowered hard-question accuracy from 89.1% to 83.3%. That
difference is on 46 questions and is not significant, but it suggests a generic procedure can occasionally
displace an analysis the model would otherwise get right.

### 6.2 Skill selection (RQ2)

**How often a skill is loaded:**

| Model | SkillBench: any skill | SkillBench: fitting skill | StatABench: any skill | StatABench: fitting skill |
|---|---|---|---|---|
| gpt-oss-120b | 38% | 30% | 67% | 47% |
| gpt-oss-20b | 9% | 8% | 27% | 18% |

Two observations follow:
- Handing the model the correct skill (oracle) gives no significant gain over self-selection for either model on
  either benchmark.
- For the small model, accuracy is the same with and without a loaded skill: 80.4% vs 80.5% on SkillBench.

Its SkillBench gain therefore cannot come from reading skill text. It must come from the other parts of the
skills condition: the skill catalog in the prompt, which names what to check, and the tested library being
importable. Separating these is the next ablation.

**Cost.** Skills roughly double prompt tokens per question (9.1 k → 19.0 k for the 120b model). The oracle,
which preloads the skill text, raises them to 52.6 k.

### 6.3 Validation (RQ3)

| Benchmark | Model | Wrong answers flagged | Correct answers flagged | Fixed | Broken | Net |
|---|---|---|---|---|---|---|
| SkillBench | 120b | 56% | 15% | 15 | 13 | +2 |
| SkillBench | 20b | 65% | 21% | 39 | 5 | +34 |
| StatABench | 120b | 49% | 30% | 23 | 27 | −4 |
| StatABench | 20b | 57% | 22% | 45 | 22 | +23 |

The validator detects half to two thirds of wrong first answers on both benchmarks.
- **Small model:** the repair step converts this into significant gains on both benchmarks. On SkillBench it
  fixes 39 answers for every 5 it breaks.
- **Large model:** the critic's false alarms (15–30% of correct answers) talk the agent out of right answers about
  as often as they correct wrong ones, so the net effect is zero.

Because the critic is the model under test, it tends to share the blind spots behind the original error.
- The preliminary transcripts include cases where the critic raised a generic concern and missed the actual
  issue: on T02 it questioned negative tips instead of noting that cash tips are not recorded.
- A critic from a different family is the obvious next test.

The validator adds about 6–8 k prompt tokens per question.

### 6.4 Multi-table questions, traps and answer quality

**Multi-table.** On the 100 real multi-table questions:
- the small model goes from 74% (agent) to 78% (skills) and 86% (skills + validator);
- the large model is at 90–93% in every agent condition.

Neither published benchmark contains multi-table questions, so this is the main coverage SkillBench adds.

**Traps.** Trap falls (a wrong answer equal to the precomputed careless answer) are 3–5% of answers in every
condition.
- The errors that remain cluster on a few trap types that most runs of every condition miss:
  - quantities that are not recorded (cash tips in the taxi data);
  - Simpson's paradox with the confounder in another table;
  - right-censored durations.
- On StatABench, variance tests (at most 31%), GLMs and survival analysis remain the weakest tasks in every
  condition, as they were for StatABench's own agent.

**Answer quality.** The judge rates method fit at 94–96% (120b) in every condition. Stated caveats, however,
fall from 87% of answers in plain to 57% for the agent and 39% with skills. Agents tend to report the number and
stop, which argues for making caveats a required part of each skill's report format.

## 7. Validation of the answer key

The reference answers were produced by scripts written with LLM assistance (Claude), so they were validated
independently:

1. **Independent code.** No reference script imports the library under test.
2. **Publisher counts.** For every dataset with a documented record count, the benchmark's copy reproduces it
   exactly (11 of 11 counts). This rules out download and parsing errors.
3. **Consensus audit.**
   - For each question, all agent answers across models, runs and conditions are grouped. A question is
     flagged when at least half agree on an answer that is not the reference.
   - The audit was run during the experiment and again on the final data. 17 distinct questions were flagged
     across the two passes, and each was recomputed by hand.
   - In 12 the reference is correct:
     - in 10 the consensus fell into the documented trap (for example, treating the missing-value code `?` as
       a recorded weight, or giving a tip figure when cash tips are not recorded);
     - in 2 the consensus used a wrong method.
   - 3 were correct analyses reported at lower precision (e.g. 25.8 for 25.78). A rounding rule now accepts a
     value that equals the reference at the reported precision and is within 1%. It is applied to every
     condition (62 answers in total).
   - 2 had two defensible readings (R18, CB06) and are excluded from the analysis for all conditions.
   - On the final data, 13 of 206 questions are flagged, and none is unresolved.
4. **Human review.** A stratified random sample of 30 real-data questions (seed 7) was reviewed by the author,
   reading the exact reference code for each one.
   - 29 were marked agree and 1 unsure: I11, which asked whether the crew table holds only feature films.
   - The build script confirms it does: restricting to films gives the same count, 60,907.
   - After resolution, 30 of 30 references are confirmed.

The models under test (gpt-oss), the author of the answer-key code (Claude) and the judge (qwen) are three
different model families.

## 8. Limitations

- **Two models from one family.** The effect of skills may differ for other model families.
- **The SkillBench reference answers were reviewed by one person** with LLM assistance. A second reviewer on
  the full set is still needed.
- **The StatABench judge differs from the paper's.** Absolute StatABench scores are therefore not directly
  comparable to the paper's published numbers; the comparison across conditions uses one judge throughout.
- **The skill text is re-sent on every turn**, which raises token cost. Prompt caching would reduce it but was
  not used.
- **The inference provider was not pinned.** OpenRouter may route the same model to different providers; runs
  were made on two dates (20 October and 5 November), and provider differences may add variance across runs.
- **The validator's critic is the model under test.** A stronger or different-family critic may detect more.

## 9. Conclusion

Packaging analysis procedure as skills over a tested tool library makes a smaller model's data analysis more
reliable on messy, large and multi-table data:
- +5.0 points on SkillBench;
- a consistent but smaller gain on the published StatABench.

Adding a validation loop raises this further to +11.2 points over the plain agent.

For a stronger model, neither skills nor validation changes accuracy, and the critic's false alarms cancel its
corrections.

Two findings shape the next phase:
1. **The gain does not come from the skill text.** The small model rarely loads a skill, and an oracle-loaded
   skill does not beat self-selection. The benefit therefore appears to come from the skill catalog and the
   available library, which an ablation will separate.
2. **A few trap types defeat almost every run of every condition:** unrecorded quantities, Simpson's paradox
   across tables, and censoring. They are the clearest targets for new skills and for a stronger,
   different-family validator.

## References

- Zhu et al. (2026). StatABench: Dataset and Framework for Evaluating Statistical Analysis Capabilities of LLMs.
  arXiv 2606.22977.
- Zhang et al. (2025). DataSciBench: An LLM Agent Benchmark for Data Science. arXiv 2502.13897.
- Anthropic (2025). Agent Skills.
