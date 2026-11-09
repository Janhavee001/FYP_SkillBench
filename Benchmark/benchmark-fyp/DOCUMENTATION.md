# Benchmarking Five Reference Papers

**Final Year Project:** *Reliable Data Analysis with LLM Agents: Skill Design, Selection, Composition, and Validation*

**Run dates:** 3 September 2026 (shared-dataset systems and controls); 15 September 2026 (full StatABench); 15–21 September 2026 (full DataSciBench)
**Model under test:** `openai/gpt-oss-120b` served through OpenRouter, `temperature=0`, held constant across every system
**Status:** complete. Shared-dataset benchmark: 5 conditions × 16 questions. StatABench: all 404 Stat-Closed items. DataSciBench: all 222 tasks × 3 runs, scored by both of its native evaluators.

---

## 1. Purpose and scope

Before designing my own agent I wanted to know what the existing work actually does when you run it, rather than what the papers claim. My supervisor asked me to benchmark five reference papers and draw conclusions relevant to my own design. This document records what I built, what I measured, what I concluded, and — at some length in §7 — which of my own numbers I do not trust and why.

Anyone reading this should read §7 before quoting any figure from §5.

---

## 2. The five papers

Two of these are benchmarks, which I ran natively using their own data and metrics. Three are agent or tool systems, which have no common evaluation between them, so I ran those against a shared test set I defined.

| Paper | Type | Repository | What it is |
|---|---|---|---|
| **StatABench** (Zhu et al., 2026), arXiv 2606.22977 | Benchmark | `youxin01/StatABench` | 404 closed statistical questions over 18 topics, plus 30 open modelling tasks. Ships an MCP server exposing 36 hand-written statistical tools. |
| **DataSciBench** (Zhang et al., 2025), arXiv 2502.13897 | Benchmark | `THUDM/DataSciBench` | 222 task folders scored by a Task–Function–Code framework and a Completion Rate metric, run against a MetaGPT Data Interpreter. |
| **LIDA** (Dibia, 2023), arXiv 2303.02927 | System | `microsoft/lida` | Visualisation generator: summarize → goal → visualize → execute → explain, with an optional `repair()`. |
| **Data Formulator 2** (Wang et al., 2024), arXiv 2408.16119 | System | `microsoft/data-formulator` at tag `0.2` | UI-driven multi-agent tool. The analyst drags fields onto a chart shelf and agents write the transformation code that produces them. |
| **nvAgent** (Ge et al., ACL 2025, long 960) | System | `geliang0114/nvAgent` | Three-agent NL2VIS workflow: Processor → Composer → Validator. The only one of the five with a dedicated validation agent. |

### 2.1 The StatABench identifier

My brief flagged `2606.22977` as suspicious, since the numbering implies June 2026. It is correct. 2606 means June 2026, which is three months before this run rather than in the future. The title, author list and the `youxin01/StatABench` repository all match, so I used the paper as given and substituted nothing.

### 2.2 Data Formulator versioning

The `main` branch of `microsoft/data-formulator` is now version 0.8.0b1, which is several architectural generations past the paper I am supposed to be benchmarking. Running `main` and calling it Data Formulator 2 would have been wrong, so I pinned a second checkout at tag `0.2` — the release the paper describes — and every result below comes from that. I kept the `main` clone only for reference.

---

## 3. How the five systems differ architecturally

I assigned these tags by reading the cloned source rather than by taking the papers' own descriptions, because in two cases the shipped code does not implement what the abstract implies. The evidence column is there so the classification can be checked.

| System | Architecture | Agents | Validator | Skill registry | Self-correction | Evidence |
|---|---|---|---|---|---|---|
| baseline `plain` | single-prompt | 1 | no | no | no | control, `harness/runners/run_baseline.py` |
| baseline `selfcheck` | single-prompt + repair | 1 | no | no | one execution-feedback pass | control |
| StatABench | tool-using ReAct agent over a skill registry | 1 | no | yes, 36 MCP tools | ReAct retries only | `mcp_scripts/mcp_server.py` has 36 `@mcp.tool()` registrations; `mcp_pipeline.py` builds a langgraph `create_react_agent` |
| DataSciBench | multi-agent planner–executor | planner + executor (MetaGPT roles) | no | no | optional (`--use_reflection`, `--max_retry`) | `experiments/run_examples.py`; `role/sci_data_interpreter.py` |
| LIDA | multi-stage pipeline, single agent | 1 | partial — `evaluate()` and `repair()` exist but are not automatic | no | on demand only | `lida/components/manager.py` exposes summarize/goals/visualize/execute/repair/evaluate as separate public calls |
| Data Formulator 2 | multi-agent, UI-driven | 3+ | no | no | none | grepping the `data-formulator-v2` agents directory for repair, retry or self-correct returns nothing |
| nvAgent | multi-agent | 3 | yes, dedicated Validator | no | execution-feedback refine loop, `MAX_ROUND=3` | `core/agents.py` `Validator.talk` sets `message['fixed']` and loops on `_is_need_refine`; `core/const.py` |

The observation that matters most for my project is that StatABench's MCP server is the only thing among the five papers that resembles an explicit skill registry — 36 named, documented statistical tools the agent has to select among. It belongs to a benchmark rather than to an agent. **None of the three published systems composes from a skill library at all.** That is the gap my FYP can occupy.

---

## 4. Method

### 4.1 Why a shared test set was necessary

The five papers cannot be compared as published. Two are benchmarks with their own data and metrics. Of the three systems, LIDA self-evaluates with an LLM, Data Formulator 2 has no quantitative evaluation whatsoever (only a ten-person user study), and nvAgent uses VisEval with an MLLM judging rendered images. Placing their published numbers side by side would mean nothing.

So I ran the three systems against a shared test set of my own, and ran the two benchmarks natively before mapping their results onto the same rubric.

### 4.2 The shared datasets and questions

I used two small, public, well-understood datasets so that ground truth would be uncontroversial:

| Dataset | Rows | Columns | Why |
|---|---|---|---|
| Titanic (seaborn distribution) | 891 | 15 | mixed categorical and numeric, with real missingness — `deck` is 77% missing, `age` 20% |
| Wine Quality (UCI, red) | 1,599 | 12 | eleven continuous predictors against an ordinal target, heavily imbalanced (82% of wines are quality 5 or 6) |

I wrote eight questions per dataset, sixteen in total, covering descriptive statistics, hypothesis testing, correlation, distribution analysis, outlier detection, regression and classification, missing-data handling, and pure visualisation. Each is tagged `stat`, `viz` or `mixed`.

Every ground-truth value is computed rather than typed by hand. `datasets/build_questions.py` derives each `reference_answer` from the actual CSV using scipy and scikit-learn, then writes `questions.json`. I spot-checked the output against values I already knew: Titanic overall survival 0.3838, female 0.742, male 0.1889; wine alcohol-to-quality correlation *r* = 0.476. Re-running the script regenerates everything, so the ground truth is reproducible rather than asserted.

Each question also carries `expected_method`, `acceptable_methods`, `expected_chart` and a numeric `tolerance`, which is what lets method selection and output correctness be scored programmatically instead of by eye.

### 4.3 Two controls

Neither RQ1 nor RQ3 can be answered from the five papers alone, because the systems differ in model, scaffolding and output format all at once. I added two controls that hold everything constant except a single variable:

- **`baseline_plain`** — one prompt, one shot, no execution feedback, no retry. This is the RQ1 control.
- **`baseline_selfcheck`** — the same prompt and the same model; when the generated code raises, the traceback is fed back for exactly one repair attempt. Comparing `plain` against `selfcheck` isolates RQ3 inside a fixed architecture.

### 4.4 The rubric

Four dimensions, each scored 0 to 1, implemented in `harness/rubric.py`. I deliberately borrowed the criteria language from the two benchmark papers so that my scores stay commensurable with their published work.

| Dimension | Criteria taken from | How it is scored |
|---|---|---|
| Method selection | StatABench `judger-open/modeling_groundedness.py` | 1.0 for the canonical method, 0.7 for a defensible alternative, 0.3 for a real but inappropriate method, 0.0 for none identifiable. Charts are scored the same way. `stat` items use method only, `viz` items chart only, `mixed` averages the two. |
| Code executability | DataSciBench's Completion Rate and TFC framework | 1.0 ran and produced output, 0.75 ran silently, 0.25 produced code that raised, 0.0 produced no code. |
| Output correctness | StatABench `eval_utils.compare_numeric` plus DataSciBench's value-versus-ground-truth checks | the fraction of ground-truth facts recoverable from the output, within each question's tolerance. |
| Interpretation quality | StatABench `judger-open/analysis_groundedness.py` and `result_bias.py`, on their 0 / 0.25 / 0.5 / 0.75 / 1.0 anchors | an LLM judge, fixed model, scoring "Interpretation of Results and Context" and "Critical Evaluation of Findings". |

The headline number weights method 0.30, executability 0.20, correctness 0.30 and interpretation 0.20. The weighting is deliberately correctness-heavy, because my project is about reliability rather than fluency.

### 4.5 Two safeguards

I built two things into the rubric after deciding that a benchmark which overstates its own precision is worse than no benchmark at all.

The first is a coverage flag. When a system structurally cannot exhibit a dimension, that dimension scores `null` with `coverage=False`, never 0.0, and the comparison prints how many items each score actually rests on. Section 7 explains why this turned out to matter far more than I expected.

The second is judge disclosure. Without an API key the interpretation dimension falls back to a transparent keyword heuristic and stamps `judge="heuristic"` on the output, so a keyless run can never be mistaken for a judged one.

The rubric has a 23-check test suite in `harness/test_rubric.py` which asserts that it genuinely discriminates: that chi-square outscores a t-test on a chi-square question, that a pie chart loses to a bar chart, that executability is strictly ordered, that wrong numbers score below right ones. It runs without an API key.

---

## 5. Results

### 5.1 Headline comparison

All seven conditions on `openai/gpt-oss-120b`, with the LLM judge active. The `native` column is each benchmark's own score; the four rubric columns are mine.

| Paper | Architecture | n | native | method | exec | correct | interp | overall |
|---|---|---|---|---|---|---|---|---|
| (control) plain LLM — plain | single-prompt | 16 | — | 0.981 | 0.719 | 0.637 | 0.527 | 0.735 |
| (control) plain LLM — selfcheck | single-prompt + repair | 16 | — | 0.981 | 0.859 | 0.771 | 0.531 | **0.804** |
| StatABench | tool-using agent over skill registry | 404 | **0.710** | n/a | n/a | 0.710 | (0.018, 7 items) | 0.704 |
| DataSciBench | multi-agent planner-executor | 222 | **0.752** | n/a | 0.833 | 0.752 | n/a | 0.784 |
| LIDA | multi-stage pipeline + optional validator | 16 | — | 0.869 | 0.812 | n/a | 0.070 | 0.625 |
| Data Formulator 2 | multi-agent (UI-driven) | 16 | — | 0.803 | 0.875 | 0.418 | 0.000 | 0.541 |
| nvAgent | multi-agent + validator | 16 | — | 0.487 | 0.766 | 0.337 | 0.000 | 0.400 |

The two benchmark rows and the five shared-dataset rows answer different questions on different items, so the `overall` column is not a ranking across that divide. For StatABench and DataSciBench the number to quote is `native`. StatABench's interpretation score rests on 7 of 404 closed-form answers that happened to be long enough to judge and should be ignored.

### 5.1a Native benchmark results in full

**StatABench, Stat-Closed, all 404 items** (MCP agent over 36 statistical tools):

| Slice | n | Accuracy |
|---|---|---|
| **All items** | 404 | **71.0%** |
| Tool-requiring items (the only ones the shipped pipeline attempts) | 198 | 66.2% |
| Theory items | 206 | 75.7% |
| Answer type: decision (reject / do not reject) | 104 | 82.7% |
| Answer type: multiple choice | 58 | 67.2% |
| Answer type: numeric | 6 | 66.7% |
| Answer type: open / other | 236 | 67.0% |

By topic (30 items unless stated): correlation 83%, contingency tables 80% (n=20), causal inference 80%, high-dimensional 80% (n=20), EDA 80% (n=15), confidence intervals 82% (n=11), descriptive 77%, non-parametric 77%, point estimation 75% (n=20), time series 73% (n=15), multiple comparison 70%, A/B testing 70% (n=20), distribution tests 67%, Bayesian 67% (n=18), regression 65% (n=20), variance tests 60% (n=20), survival analysis 55% (n=20), **generalized linear models 40% (n=25)**. For reference, the StatABench paper reports GPT-5.1 at 68.6% on Stat-Closed.

**DataSciBench, all 222 tasks × 3 runs** (MetaGPT Data Interpreter):

| Family | Tasks | Mean completion rate (harness definition) |
|---|---|---|
| **All** | 222 | **0.752** |
| BigCodeBench-derived (`bcb`) | 167 | 0.816 |
| Curated: human-written (`human`) | 25 | 0.611 |
| Curated: CSV/Excel (`csv`) | 20 | 0.518 |
| Curated: deep learning (`dl`) | 10 | 0.493 |

Of the 167 BCB tasks, 111 were fully solved on all three runs and 15 scored 0 on every run. The 0.752 headline is dominated by BCB (167 of 222 tasks); on the 55 curated, multi-step data-analysis tasks the agent completes only about 55% of the checked steps.

To place this on the paper's own leaderboard I recomputed the curated-task metrics with the exact formulas in `evaluation_results/calculate_final_metric.py`, scaled from its 10 runs to my 3:

| Model | Pass@1 | Average CR |
|---|---|---|
| GPT-4o (the paper's best Pass@1) | 19.82 | 17.89 |
| **gpt-oss-120b (this study)** | **16.36** | **18.29** |
| GPT-4-turbo | 17.27 | 17.36 |
| GPT-4o-mini | 12.73 | 17.35 |
| deepseek-coder-33b | 12.55 | 13.53 |

So the model under test sits with the strongest models in the paper. The low absolute numbers come from DataSciBench's own definitions: Pass@1 requires every checked function in a task to match ground truth, and Average CR divides by the 167 metric functions, not by the task count.

The paper's scoring script has a bug of its own: DL Pass@1 is computed with `if success_dict[key] == "Success"`, comparing a number with a string, so it is always 0. That is why every model in the published table shows 0.00 for DL Pass@1. Computed correctly, gpt-oss-120b's DL Pass@1 is 3.33.

I do not compare BCB against the repository's `bcb_results.csv`. It lists GPT-4-turbo at Pass@1 0.96 with Average CR 0.55, which the repo's own `show_bcb_eval_results.py` makes impossible, since a run only passes when its CR is 1, so Pass@1 can never exceed Average CR. Computed with that script's definitions over 3 runs, gpt-oss-120b scores BCB Pass@1 0.778 and Average CR 0.816.

### 5.2 Coverage

A dimension marked `n/a` above is not measurable for that system, and this table shows how many items each score actually rests on.

| Paper | method | exec | correct | interp |
|---|---|---|---|---|
| plain | 16/16 | 16/16 | 16/16 | 16/16 |
| selfcheck | 16/16 | 16/16 | 16/16 | 16/16 |
| StatABench | 0/404 | 0/404 | 404/404 | 7/404 |
| DataSciBench | 0/222 | 222/222 | 222/222 | 0/222 |
| LIDA | 16/16 | 16/16 | **0/16** | 16/16 |
| Data Formulator 2 | 16/16 | 16/16 | 16/16 | 16/16 |
| nvAgent | 16/16 | 16/16 | 16/16 | 16/16 |

### 5.3 Score by question type

This is the table that does most of the explanatory work, because it separates a genuine capability gap from a category mismatch.

| Paper | stat | viz | mixed |
|---|---|---|---|
| plain | 0.764 | 0.639 | 0.747 |
| selfcheck | 0.793 | 0.833 | 0.801 |
| StatABench | 0.704 | — | — |
| DataSciBench | 0.784 | — | — |
| LIDA | 0.473 | **0.750** | 0.738 |
| Data Formulator 2 | 0.466 | 0.625 | 0.588 |
| nvAgent | 0.329 | 0.591 | 0.388 |

### 5.4 Self-correction

| Paper | Mechanism | Repair attempts | Repairs that fixed it | Clean-execution rate |
|---|---|---|---|---|
| plain | none | 0 | 0 | 0.625 |
| selfcheck | one execution-feedback pass | 4 | 1 | **0.812** |
| DataSciBench | reflection off (the shipped default); plan steps may re-run up to `max_retry=3` | — | — | 0.333 (all plan steps of the first run succeeded) |
| LIDA | on demand, `repair()` called explicitly | 7 | 3 | 0.750 |
| Data Formulator 2 | none | 0 | 0 | 0.875 |
| nvAgent | validator refine loop, `MAX_ROUND=3` | 22 | 8 | 0.750 |

### 5.5 Output channels

This explains why the systems are not directly comparable on every dimension.

| System | stdout | numeric outputs | chart artifact | mean narrative length |
|---|---|---|---|---|
| plain | 10/16 | — | 10/16 | 585 characters |
| selfcheck | 14/16 | — | 13/16 | 593 characters |
| LIDA | **0/16** | — | 12/16 | 1,734 characters |
| Data Formulator 2 | 14/16 | 14/16 | — | 140 characters |
| nvAgent | 12/16 | — | — | 109 characters |

---

## 6. What I take from the results

### 6.1 The plain baseline wins, but only narrowly and only here

`baseline_selfcheck` scores 0.804 and beats every published system. That result is real, but it is much narrower than it looks. It holds on standard questions over well-known datasets, where an unscaffolded model already selects the correct method 98% of the time — it chose chi-square for the association question, a Welch t-test for the group comparison, and logistic regression for the binary outcome, essentially every time.

This is not evidence that scaffolding is useless. It is evidence that my test set is not hard enough to require scaffolding. Any claim that structured skills improve method selection has to be tested where the baseline does not already saturate: unusual data shapes, methods that must be named rather than inferred, or multi-step compositions.

### 6.2 None of the three systems interprets its own results

All three produce narrative text — LIDA averages over 1,700 characters of it — but the judge, applying StatABench's anchors, found that almost none of it interprets the *result*. LIDA describes the chart's design: the colour palette it chose, the dashed reference line, the axis labels. Data Formulator 2 returns a single-line encoding recommendation. nvAgent returns reasoning about the schema.

None of them tells the user what the data means. For a project about reliable analysis this is a substantive gap in the literature rather than an artefact of my scoring. Every one of these tools hands back an artefact and leaves interpretation entirely to the human.

### 6.3 The visualisation systems are much stronger at visualisation

LIDA scores 0.473 on statistical questions and 0.750 on visualisation ones. Quoting only the overall figure would misrepresent all three systems. nvAgent is the clearest case: it is an NL2VIS system whose VQL grammar is `Visualize <TYPE> SELECT ...`, a chart-over-SQL language with no way to express fitting a logistic regression. It emitted no VQL at all for three of my sixteen questions. That is an expressiveness ceiling rather than a defect, and on its own VisEval benchmark it outperforms its baselines.

### 6.4 Repair fixes about a third of crashes; the plain-vs-selfcheck gap is mostly run-to-run variation

The headline comparison looks like strong evidence: the `selfcheck` control beats `plain` on clean execution (0.625 → 0.812), output correctness (0.637 → 0.771) and overall score (0.735 → 0.804). Checked question by question, **the repair step accounts for almost none of that gain**:

| | plain | selfcheck |
|---|---|---|
| Questions that ran | 10/16 | 13/16 |
| Repair attempts | — | 4 |
| Repairs that fixed the crash | — | 1 (`titanic_q7`, which `plain` got right first time anyway) |
| Wine questions where the model wrote `sep=';'` | 5 (all crashed) | 3 (all crashed; 0 repaired) |

The shared Wine CSV is comma-separated, but the original UCI file is semicolon-separated. In the `plain` run the model "remembered" the UCI format and wrote `pd.read_csv(path, sep=';')` on 5 of 8 Wine questions. That loads the whole file as one column, so every later step fails with `KeyError: 'quality'`. In the `selfcheck` run it happened to make that mistake on only 3 questions, so `wine_q1`, `wine_q4` and `wine_q8` ran on the **first** attempt, without any repair. Where the mistake did occur, the repair prompt showed the model a traceback listing the single mashed-together column, and it still did not fix the separator (0 of 3).

What the evidence supports:

* **Repair mechanisms fix roughly one crash in three:** selfcheck 1/4, LIDA 3/7, nvAgent 8/22, pooled **12/33 (36%)**. That is a real but modest effect.
* **One run of 16 questions cannot measure an effect this size.** Two runs of the same model and prompt at temperature 0 differed by 3 questions on execution alone (the provider is not deterministic). The Phase 7 ablation needs several repeats per condition and per-question paired comparison.
* **The dominant failure was grounding, not a crash a repair can see.** The semicolon assumption is the model trusting its prior over the actual file, which is the RQ2 grounding problem again (§6.6). A validator that compared the loaded frame against a dataset profile (12 columns expected, 1 found) would catch it before any analysis runs.

One confound is worth stating plainly. Data Formulator 2 has the highest execution rate in the study, 0.875, with no repair mechanism at all. It wins that column because it generates short data-transformation snippets rather than whole analyses. Execution reliability tracks task complexity at least as much as it tracks architecture, so that column is not a clean ranking of validators.

### 6.5 A validator only repairs what it is allowed to touch

I measured nvAgent three times, for reasons explained in §7.2. In the first run its Validator burned all three rounds on fourteen of sixteen questions and reported `fixed=True` each time, yet never resolved the error — because the fault lay in its own scaffolding template rather than in the LLM-generated code the refiner is able to rewrite.

That is a design lesson I intend to act on. An execution-feedback validator detects that something failed but can only fix the region it is scoped to edit. A validator that checked schema conformance or method appropriateness would catch a different and, I think, more important class of error.

### 6.6 Grounding fails silently, and that is the RQ2 problem

nvAgent produced this for the wine correlation question:

```sql
SELECT property, corr FROM (SELECT 'fixed_acidity' AS property,
  CORR(fixed_acidity, quality) AS corr FROM wine UNION ALL ...) AS t ORDER BY corr DESC
```

The chart choice is right and the statistical intent is right, but it queries a table called `wine`, which does not exist — the Processor had given it `wine_quality`. The system invented a plausible-looking table name and failed on that alone.

This is a composition and grounding failure rather than a selection failure, and it is exactly the reliability problem RQ2 targets. The agent chose correctly and still produced nothing, because it did not bind to the data it had been given.

---

## 7. Limitations, and bugs I found in my own harness

### 7.1 Fifteen environment and harness bugs

I am recording these in full because they turned out to be one of the more useful findings of the exercise. Every one of them would have produced a confident, wrong result.

| # | Bug | Would have been reported as | Actual cause |
|---|---|---|---|
| 1 | Windows console cp1252 | every system penalised on code executability | the model emits U+2011 and U+2192; this crashed both my own `print()` and the generated script's `print()` inside the subprocess, so correct code was recorded as failed |
| 2 | percentage versus proportion | correctness understated across the board (0.625 → 0.875 on the smoke test) | ground truth stores 0.3838 and the model printed 38.38%, which is exactly right and scored 0.0 |
| 3 | Vega `point` treated as unknown | Data Formulator penalised for vocabulary rather than for wrong charts | `normalise_chart('point')` returned nothing, but Data Formulator emits Vega-Lite mark names in which `point` *is* the scatter mark |
| 4 | Windows `\U` path escape | **nvAgent scoring 0/16, the opposite of the truth** | nvAgent injects the data folder into generated code as a raw string, so `C:\Users\...` was parsed as a unicode escape and every script died before running |
| 5 | hyphen in a SQL table name | nvAgent 7/16, with all Titanic passing and all Wine failing | I copied the CSV as `winequality-red.csv`, giving `CREATE VIEW winequality-red` and a DuckDB parser error. Titanic passed only because `titanic.csv` happens to be a legal identifier |
| 6 | cp1252 in a spawned subprocess | **StatABench scoring 0%** | StatABench prints progress with an emoji; all 25 responses came back as the literal string `Error: 'charmap' codec can't encode character '⏳'`. Fixing bug 1 did not cover a separately spawned process |
| 7 | missing `test_config.yaml` | DataSciBench unable to start | it passes a named config to `Config.from_home()` and `src/logs.py` opens the same file directly, so `config2.yaml` alone is not enough |
| 8 | `httpx >= 0.28` | DataSciBench generating nothing | MetaGPT's pinned `openai==1.39.0` passes the removed `proxies=` argument; the same incompatibility had already hit nvAgent |
| 9 | wrong ground-truth tree | DataSciBench scoring 0.0 on everything, cleanly and plausibly | I copied `DataSciBench-data/<task>/`, which holds prompts, when the evaluator wants `gt_data/<task>/gt/output.csv` |
| 10 | mixed scales in the results CSV | a "completion rate" of 1.36 | per-function rows use a 1/2 scale while only rows typed `Completion Rate` are the 0–1 metric, and `task_name` is the metric name rather than the task |

The full runs of the two benchmarks surfaced five more:

| # | Bug | Would have been reported as | Actual cause |
|---|---|---|---|
| 11 | DataSciBench input files never staged | plausible scores on analyses of **invented data** | the prompts point at `../`, but the clone ships only `prompt.json` there. Given no input, gpt-oss-120b wrote "source.xlsx not found – creating a dummy file" and analysed that. The first full run was discarded (`results/_invalid_dsb_runs_missing_inputs_2026-09-15/`) |
| 12 | VLM chart judge pointed at a keyless third-party proxy | 0 on all 14 chart-judged DataSciBench metrics | `src/vlm_utils.py` hard-codes `api.turboai.one`; now configurable and pointed at the configured provider |
| 13 | `logs/openai/` never created | every BCB run failing before the agent started | `src/logs.py` opens a file inside a folder named after the model's provider prefix without creating it |
| 14 | `plt.show()` under the TkAgg backend | the BCB evaluation hanging indefinitely (90 minutes on `bcb198`) | `evaluate_tmc` `exec()`s the agent's code in-process, which opened a "Figure 1" window and waited for a human to close it. The per-task timeout also failed to fire, because on Windows a venv `python.exe` is a launcher, and killing it left the real interpreter running. Fixed with `MPLBACKEND=Agg` and a process-tree kill |
| 15 | laptop sleep mid-run | two BCB tasks (`bcb381`, `bcb41`) scoring a run as 0 | the lid closed at the exact second their second run started, so the Jupyter kernel timed out ("Kernel didn't respond in 60 seconds"). Those two runs were regenerated before evaluation |

Fixes 2 and 3, and the coverage rule, are locked in by regression tests.

The pattern is itself worth reporting. All fifteen made a system look **worse** than it is, or (bug 11) produced a score for work on the wrong data; none flattered one. Most were invisible in the aggregate scores and only surfaced when I read individual records, because a plausible-looking number — 0/16, 7/16, 0%, 0.091, 1.36 — is indistinguishable from a real result until you inspect the output underneath it. Bug 6 also shows that fixing a class of error once is not sufficient: the same cp1252 root cause reappeared at two further process boundaries hours later.

For a project about reliable analysis this is direct evidence. An agent that reports a confident aggregate without exposing per-item provenance will mislead its user, and the failure mode looks systematically pessimistic while actually being arbitrary.

### 7.2 nvAgent was measured three times

| Run | Result | Cause |
|---|---|---|
| first | 0/16 | bug 4, mine |
| second | 7/16 | bug 5, mine |
| third | **12/16** | honest |

Only the third is reported. Had I published the first, my report would have claimed that the one paper whose entire contribution is a validator agent cannot execute anything, when in fact it was producing correct VQL every time.

### 7.3 LIDA's correctness column

Output correctness is `n/a` for LIDA rather than a low score. LIDA emitted no stdout on any of the sixteen questions; its answer is a rendered image. Its correctness is not measurable by my rubric, because recovering the numbers would mean reading them off the chart, which is what VisEval's MLLM judge does and mine does not.

An intermediate version of this fix was worse than the original bug. Marking only the twelve successful chart records as unmeasurable left the four failures scored, so LIDA's correctness was being computed entirely from the questions it had failed. The rule is now applied at the system level.

### 7.4 The benchmark rows needed the same treatment

Before I applied coverage properly, StatABench appeared with an overall of 0.042 against a native accuracy of 75% (on the 20-item subset run at the time). Its records carry no generated code, because the agent answers by calling MCP tools, and scoring that 0.0 for "no code emitted" measures interaction style rather than capability. Its answers are also closed-form by design ("The answer is (63.43 kg, 72.57 kg)"), so judging them as interpretations measures the question format. Both dimensions are now marked as not applicable, and the benchmark's own adjudication is carried through as the correctness score.

### 7.5 How complete the native runs are

Both benchmarks were run in full. StatABench covers all 404 Stat-Closed items. As shipped, its pipeline skips every item flagged `code == 0` and so attempts only the 198 tool-requiring ones; `--track all` includes the 206 theory items too, and the two groups are reported separately in §5.1a. Stat-Open, the 30 open modelling tasks graded by an LLM judge, was not run. DataSciBench covers all 222 tasks with 3 runs each, against the paper's 10, so its per-task figures have wider variance than the published ones. Reflection was left off, the repository default.

### 7.6 nvAgent is not deterministic

Even at `temperature=0`, `titanic_q8` produced valid VQL in one run and nothing in another. Its single-run figures carry more uncertainty than the others'.

### 7.7 My test set favours systems that print numbers

My questions ask for things like the test statistic and the p-value, which rewards systems whose output channel is text and penalises systems whose output channel is an image. This is the main threat to the validity of the headline table, and it is why §5.2 and §5.3 have to be read alongside it.

### 7.8 The model differs from the papers'

I ran an open-weights model through OpenRouter rather than the models the papers used, so my absolute figures are not comparable to their published ones — StatABench reports GPT-5.1 at 68.6% on Stat-Closed. This study compares architectures on a fixed model; it does not reproduce published scores.

---

## 8. Reproducibility

Three of the five repositories do not run as shipped. The full list of adaptations is in `README.md`; these are the ones that matter.

| Repository | Problem | What I did |
|---|---|---|
| nvAgent | `core/chat_manager.py` imports `PROVIDER_NAME` from `core/const.py`, which does not define it. The module cannot be imported at all as distributed, which also breaks the repository's own `run_evaluate.py` | two-line patch to `PROCESSOR_NAME`, visible in `git -C repos/nvAgent diff` |
| nvAgent | pins `openai==1.51.2`, which passes `proxies=` to httpx, removed in 0.28 | pinned `httpx<0.28` in that environment |
| nvAgent | `viseval/__init__.py` eagerly imports `cairosvg`, which needs a native Cairo library absent on Windows | stubbed in my runner; only nvAgent's native VisEval rendering needs it |
| DataSciBench | its vendored MetaGPT, and upstream `metagpt==0.8.1`, both pin `lancedb==0.4.0`, which has been delisted from PyPI, so `pip install` fails outright | installed 0.8.1 with that single pin dropped; lancedb is used only for RAG and the Data Interpreter path never touches it |
| StatABench | `mcp_pipeline.py` takes the model from a hardcoded `agent_map`, ignoring `keys.json` | one-line patch to prefer the `keys.json` entry |
| StatABench | spawns its MCP tool server with a bare `command="python"`, which resolves to the system interpreter and therefore lacks statsmodels, pymc and pingouin | my runner prepends the virtual environment's `Scripts/` to `PATH` |
| StatABench | `eval_utils.py` calls `gpt_chat(model="deepseek")` with a hardcoded provider name when adjudicating open-ended answers | added a `deepseek` alias to the generated `keys.json` |
| Data Formulator | `main` is 0.8.0b1, far past the paper | pinned a second checkout at tag `0.2` |

I used six isolated virtual environments because the repositories pin mutually incompatible dependencies; there is no single environment that runs all five.

The harness reads its credentials from `.env` and supports OpenAI, OpenRouter and Hugging Face through an `LLM_PROVIDER` switch.

One security note: `repos/StatABench/keys.json` is a tracked file inside that clone, because the authors committed a template, and it now holds a real credential. I must never commit or push from inside that clone; this project's own `.gitignore` cannot protect it, since a nested clone has its own index.

---

## 9. Answers to the research questions

These are stated at the confidence the current evidence supports.

### RQ1 — Do structured skills improve correctness compared with a plain LLM baseline?

Not demonstrated on this test set, and the test set is the reason. The plain baseline reaches 0.981 on method selection, which leaves essentially no headroom. Structured skills cannot be shown to help where an unscaffolded model already chooses correctly 98% of the time.

The useful finding here is a negative one: none of the three published systems beats the plain baseline overall, and all three lose heavily on statistical questions specifically. Scaffolding designed for visualisation does not transfer to analysis.

To answer RQ1 properly I need harder tasks — unusual data shapes, methods that must be named rather than inferred, or multi-step compositions that a single prompt cannot hold.

### RQ2 — Can an agent reliably select and compose skills for unseen data?

Selection is largely solved on familiar data; composition and grounding are not. The failures I observed were not cases of choosing the wrong test. They were:

- referencing a table that does not exist while choosing the right chart (nvAgent)
- emitting no plan at all for a question outside the grammar (nvAgent, three questions)
- returning prose instead of code (Data Formulator 2, two questions)

StatABench's 36-tool MCP registry is the only skill-registry-shaped artefact among the five papers, and it belongs to a benchmark rather than a system. No published system in this set composes from a skill library, which is the clearest opening for my own work.

### RQ3 — Does validation or self-correction improve reliability?

Partly, and less than the headline numbers suggest. Execution-feedback repair fixes about a third of the crashes it attempts (12 of 33 across the selfcheck control, LIDA and nvAgent). The larger plain-versus-selfcheck gap (clean execution 0.625 → 0.812) came mostly from run-to-run variation, not from the repair step (§6.4), so this study does not isolate the effect of self-correction. Doing that needs repeated runs.

Three qualifications. The dominant failure in the controls, reading the Wine CSV with the wrong separator, was never repaired, and a data-profile check would have caught it. Execution rate confounds with task complexity, since Data Formulator leads that column with no validator at all. And a validator only repairs what it is scoped to edit, so execution-feedback validation catches crashes but not analyses that are wrong yet runnable.

---

## 10. What I intend to build

Following from the above:

1. **Include a validator, and give it more to check than execution.** Execution-feedback repair fixes only about a third of crashes, and §6.4–6.6 show that the errors that matter most (a wrong separator, a non-existent table, a fault outside the editable code) are invisible to it. Mine should check the loaded data against a dataset profile, method appropriateness, and whether reported numbers match the computation.
2. **Evaluate with repeated runs.** Two identical runs differed by 3 of 16 questions, so every ablation condition needs several runs and a per-question paired comparison.
3. **Treat interpretation as a first-class output.** All three published systems score near zero here. Producing a chart or a number without saying what it means is the clearest unmet need in this literature.
4. **Use an explicit skill registry, but justify it on composition rather than selection.** Selection is close to saturated for a strong model on familiar data; the defensible argument for a registry is reliable composition and grounding.
5. **Avoid a rigid intermediate grammar.** nvAgent's VQL is the cautionary case. It guarantees a valid chart but cannot express a hypothesis test or a model fit, and it fails silently by emitting nothing.
6. **Design the evaluation before the agent.** Fifteen measurement bugs in this study all favoured a wrong conclusion, and most were invisible without per-record inspection.

---

## 11. Repository layout

```
benchmark-fyp/
  README.md                 setup, run commands, environment adaptations
  DOCUMENTATION.md          this file
  .env.example              provider and key template
  datasets/
    build_questions.py      regenerates questions.json with computed ground truth
    titanic/                titanic.csv and questions.json (8 questions)
    wine_quality/           winequality-red.csv and questions.json (8 questions)
  harness/
    common.py               config, provider switch, Record, method/chart vocabulary
    rubric.py               the four-dimension rubric, LLM judge, judge cache
    compare.py              scores results/*.json into comparison.csv and the tables
    test_rubric.py          23 regression checks, no API key required
    setup_keys.py           writes repos/StatABench/keys.json from .env
    fetch_ground_truth.py   downloads DataSciBench's gated ground truth
    runners/                one runner per system, plus the two controls
  results/                  per-system JSON, comparison.csv, comparison_detail.json
  repos/                    six clones (five papers plus the pinned DF2 v0.2)
  venvs/                    six isolated environments
```
