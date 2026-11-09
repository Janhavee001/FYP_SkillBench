# Skill Catalog

**Project:** *Reliable Data Analysis with LLM Agents: Skill Design, Selection, Composition, and Validation*

This is the tool/skill catalog for my own agent, designed after benchmarking five reference
papers (see `../Benchmark/benchmark-fyp/DOCUMENTATION.md`). Every tool below traces to a
specific finding from that benchmark, not just to "what a data science agent probably needs."
Section 0 explains the design principle; Section 1 is the full catalog; Section 2 says which
tools are implemented so far and why those were chosen first.

---

## 0. Design principle: why this catalog is not just StatABench's tool list copied over

The benchmark's conclusion (`DOCUMENTATION.md` §10) was specific about what to copy and what
not to:

> "Use an explicit skill registry, but justify it on composition rather than selection... the
> defensible argument for a registry is reliable composition and grounding" — not method
> selection, which a plain LLM already gets right 98% of the time on familiar data (§9, RQ1).

So this catalog differs from StatABench's 36-tool MCP registry (the closest prior art) in one
structural way: **every statistical/analytical tool is paired with a grounding or validation
tool that a plain tool-calling loop does not have**, because the benchmark's own failure
analysis (§6.4–6.6, §7.1) showed that the failures that matter are not "wrong test chosen" but:

- a wrong CSV separator silently producing a single mashed-together column (§6.4, bug in the
  `selfcheck` control — never repaired even when shown the traceback)
- a query referencing a table (`wine`) that does not exist, despite choosing the right chart and
  the right statistical intent (§6.6, nvAgent)
- a validator that can only fix the region it is scoped to edit, so it "fixes" nothing when the
  fault is upstream of the generated code (§6.5)
- narrative output that describes the chart or the code rather than what the result *means*
  (§6.2 — none of the three published systems clear this bar)

Category **A (Grounding & Validation)** below exists because of this, and is the part of the
catalog with no direct equivalent in any of the five papers. Everything else is inspired by, and
cited against, specific tools already read out of the cloned repositories.

**Sources cited by short name:**
- **StatABench** — Zhu et al. 2026, arXiv 2606.22977. Tool file:
  `Benchmark/benchmark-fyp/repos/StatABench/mcp_scripts/mcp_server.py` (36 `@mcp.tool()`
  functions, enumerated directly from source, not from the paper's prose).
- **DataSciBench / MetaGPT Data Interpreter** — Zhang et al. 2025 (arXiv 2502.13897) running
  Hong et al. 2024's Data Interpreter (arXiv 2402.18679). Tool files:
  `venvs/datascibench/Lib/site-packages/metagpt/tools/libs/{data_preprocess,feature_engineering}.py`
  (23 `@register_tool` functions total across the package).
- **LIDA** — Dibia 2023, arXiv 2303.02927. Stages read from
  `repos/lida/lida/components/manager.py`: summarize → goal → visualize → execute → explain →
  repair → evaluate.
- **Data Formulator 2** — Wang et al. 2024, arXiv 2408.16119. Shelf-based encoding-to-code
  agents, pinned at tag `0.2`.
- **nvAgent** — Ge et al., ACL 2025. Three-agent NL2VIS with a dedicated Validator
  (`core/agents.py`), and the VQL grammar's expressiveness ceiling documented in §6.3.

---

## 1. Full catalog (52 tools, 10 categories)

### A. Grounding & Validation — the gap identified in §3, §6.4–6.6, §10 (no direct equivalent among the five papers)

| # | Tool | Purpose | Inspired by / justified by |
|---|---|---|---|
| 1 | `profile_dataset` | Schema fingerprint: shape, dtypes, memory, per-column cardinality | Generalizes StatABench `check_column_type_is`; also closes DataSciBench bug #11 (analysing invented data because inputs were never staged) by making "what did I actually load" an explicit, inspectable step |
| 2 | `detect_separator_and_encoding` | Sniff CSV delimiter/encoding before parsing; flag low column-count-vs-expected mismatches | Directly targets bug found in DOCUMENTATION §6.4: the model wrote `sep=';'` from UCI-format memory on a comma-separated file, collapsing 12 columns into 1, and repair never fixed it even when shown the traceback |
| 3 | `validate_grounding` | Check that every column/table a plan or generated code references actually exists in the profiled schema, before execution | Directly targets nvAgent's failure in §6.6: correct chart, correct statistical intent, querying a table (`wine`) that does not exist |
| 4 | `check_method_appropriateness` | Given a declared method + the dtypes/cardinality of the columns it's applied to, flag structural mismatches (e.g. Pearson correlation on a binary column, t-test on 3+ groups) | Draws on the concern in StatABench's `judger-open/modeling_groundedness.py` (an LLM-judge prompt scoring whether a write-up justified its model choice), turned into deterministic rules run before execution. Not a port: that file contains no rules |
| 5 | `verify_numeric_consistency` | Recompute the headline number independently and compare it against what the narrative claims, flagging silent unit/scale errors | Targets the bug class in DOCUMENTATION §7.1 #2: the model printed `38.38%` against a ground truth of `0.3838` — numerically identical, syntactically different, and the harness itself scored it wrong until this was checked |
| 6 | `check_repair_scope` | Before accepting a "fixed" flag from a self-correction loop, confirm the diff touched the region actually responsible for the failure | Targets §6.5: nvAgent's Validator reported `fixed=True` on 14/16 questions while the fault was in its own scaffolding template, outside the region the refiner could edit |
| 7 | `flag_unmeasurable_dimension` | Mark a rubric dimension `n/a` with a coverage count, rather than scoring 0, when a system structurally cannot produce it | Formalizes the benchmark's own coverage-flag safeguard (§4.5, §7.3–7.4), so the *agent* self-reports what it cannot show, instead of a human patching the rubric afterward |

### B. Data Loading, Profiling & Cleaning

| # | Tool | Inspired by |
|---|---|---|
| 8 | `load_dataset` | generic — CSV/Parquet/Excel loader used by every system in the benchmark |
| 9 | `missing_value_report` | StatABench `check_missing_values` |
| 10 | `impute_missing` | MetaGPT `FillMissingValue` |
| 11 | `scale_numeric` (minmax / standard / robust / maxabs) | MetaGPT `MinMaxScale`, `StandardScale`, `RobustScale`, `MaxAbsScale` |
| 12 | `encode_categorical` (ordinal / one-hot / label / target-mean / k-fold target-mean) | MetaGPT `OrdinalEncode`, `OneHotEncode`, `LabelEncode`, `TargetMeanEncoder`, `KFoldTargetMeanEncoder` |
| 13 | `detect_outliers_iqr` | StatABench `detect_outliers` |
| 14 | `engineer_categorical_interactions` (count encode, cross, group-stat) | MetaGPT `CatCount`, `CatCross`, `GroupStat` |
| 15 | `bin_continuous` | MetaGPT `SplitBins` |
| 16 | `polynomial_expand` | MetaGPT `PolynomialExpansion` |

### C. Descriptive & Exploratory Statistics

| # | Tool | Inspired by |
|---|---|---|
| 17 | `describe_column` | StatABench `calculate_statistic` |
| 18 | `correlation_matrix` | StatABench `correlation_analysis` |
| 19 | `frequency_table` | generic, `value_counts`-class questions in the shared test set |
| 20 | `groupby_aggregate` | the `groupby_aggregation` method vocabulary already used in `harness/common.py` |
| 21 | `column_summary_for_viz` | LIDA `summarize()` stage |

### D. Inferential Statistics / Hypothesis Testing

| # | Tool | Inspired by |
|---|---|---|
| 22 | `confidence_interval` (one-sample) | StatABench `ci_normal` |
| 23 | `two_sample_confidence_interval` | StatABench `ci_two_normal` |
| 24 | `chi_square_association` | StatABench `contingency_test` |
| 25 | `ks_normality_test` | StatABench `ks_test` |
| 26 | `variance_equality_test` (Mood's) | StatABench `mood_variance_test` |
| 27 | `nonparametric_test` (Mann–Whitney / Wilcoxon / Kruskal) | StatABench `nonparametric_test` |
| 28 | `ab_test` (Welch t) | StatABench `ab_ttest` |
| 29 | `bootstrap_ab_test` | StatABench `bootstrap_abtest` |
| 30 | `ab_power_analysis` | StatABench `ab_power_analysis` |
| 31 | `multiple_comparison_correction` (FDR / FWER) | StatABench `fdr_control_df`, `fwer_control_df` |

### E. Regression & Modeling

| # | Tool | Inspired by |
|---|---|---|
| 32 | `linear_regression` (simple) | StatABench `simple_linear_regression` |
| 33 | `multivariable_regression` (+VIF) | StatABench `multivariable_linear_regression` |
| 34 | `glm_fit` | StatABench `run_glm` |
| 35 | `robust_regression` (Huber) | StatABench `huber_regression` |
| 36 | `regularized_regression` (lasso / ridge / elastic-net) | StatABench `advanced_regression` |
| 37 | `dimensionality_reduction` (PCA / sparse PCA) | StatABench `sparse_pca_analysis` |
| 38 | `feature_selection` (variance-based / general) | MetaGPT `VarianceBasedSelection`, `GeneralSelection` |

### F. Time Series

| # | Tool | Inspired by |
|---|---|---|
| 39 | `stationarity_test` (ADF / KPSS) | StatABench `test_stationarity` |
| 40 | `seasonal_decompose` (STL) | StatABench `decompose_stl` |
| 41 | `arima_forecast` | StatABench `auto_arima_modeling` |

### G. Survival Analysis

| # | Tool | Inspired by |
|---|---|---|
| 42 | `kaplan_meier_curve` | StatABench `kaplan_meier_plot` |
| 43 | `logrank_test` | StatABench `logrank_test_compare` |
| 44 | `cox_regression` | StatABench `fit_cox_model` |

### H. Causal Inference / Bayesian

| # | Tool | Inspired by |
|---|---|---|
| 45 | `propensity_score_matching` | StatABench `estimate_ATT_with_psm` |
| 46 | `difference_in_differences` | StatABench `estimate_did_effect` |
| 47 | `bayesian_estimate` (incl. hierarchical) | StatABench `bayesian_inference`, `bayesian_linear_regression`, `fit_hierarchical_model` |

### I. Visualization

| # | Tool | Inspired by |
|---|---|---|
| 48 | `recommend_chart` | LIDA `goal()` + `visualize()`, deliberately **not** restricted to a fixed grammar like nvAgent's VQL — §6.3 documents VQL silently emitting nothing for 3/16 questions outside its expressiveness |
| 49 | `render_chart` | LIDA `execute()` |
| 50 | `explain_chart` | LIDA `explain()`, redefined below (Category J's framing) to explain the *finding*, not the chart's design choices — see §6.2 |

*(these three form one pipeline stage in the implementation; visualization is deliberately the thinnest category here, since the benchmark's finding is that the published systems are already comparatively strong at it — §6.3 — and comparatively weak at interpretation, which is where I chose to spend the implementation budget instead.)*

### J. Interpretation — the other gap identified in §6.2, §10 item 3

| # | Tool | Purpose | Justified by |
|---|---|---|---|
| 51 | `interpret_result` | Turn a (method, statistic, question) triple into a structured interpretation: effect-size bucket, practical vs. statistical significance, and what the number means for the question asked — not a description of the chart or the code | §6.2: LIDA averages 1,734 characters of narrative that describes its own colour palette and axis labels; Data Formulator 2 and nvAgent return encoding/schema reasoning. All three score ≈0 on StatABench's own "interpretation of results" anchors. This tool is scoped to *only* produce result-meaning statements, structurally preventing the chart-description substitution the benchmark caught |
| 52 | `critique_findings` | Check the result against its own assumptions (sample size, distributional assumption, confounders) and report violations | Asks the checkable subset of the questions in StatABench's `judger-open/result_bias.py` (an LLM-judge prompt on bias and result analysis) as deterministic rules, before the claim is written |

---

### K. Relational layer — multi-table analysis (added after the catalog; see the multi-table research plan)

Not part of the original 52. With several related tables, most errors happen at the join and are silent, so
this category makes every step from "several tables" to "one analysis table at a stated grain" explicit and
checked. No counterpart in StatABench or MetaGPT; key inference follows the approach of HoPF (Jiang & Naumann
2020): primary keys from unique non-null columns, foreign keys from inclusion dependencies plus name evidence.

| # | Tool | Purpose | Failure it targets |
|---|---|---|---|
| 53 | `discover_tables` | Load a folder of files, a list of files or a SQLite database; schema, row counts, declared keys | Agent never seeing the whole schema |
| 54 | `infer_keys` | Primary keys (incl. composite) and foreign keys, each with its evidence and the rejected look-alikes | Wrong join key |
| 55 | `build_relationship_graph` | Tables and links with 1:1 / 1:N cardinality, bridge (N:M) tables, disconnected groups | Ambiguous structure |
| 56 | `plan_join_path` | Join steps from a chosen grain, each marked many-to-one (safe) or one-to-many (fans out); rival routes reported | Fan-out, ambiguous path |
| 57 | `aggregate_to_grain` | Roll a child table up to one row per key before joining | Fan-out, grain mismatch |
| 58 | `validate_join` | Runs a join and reports rows before/after, fan-out factor, unmatched rows, observed vs expected relationship | Fan-out, orphans, lost rows, many-to-many |
| 59 | `execute_join_plan` | Runs a plan step by step through validate_join, stopping at the first bad step | Unchecked multi-step joins |
| 60 | `check_unit_of_analysis` | Whether an entity spans several rows, and which columns are entity-level attributes | Non-independence |
| 61 | `detect_aggregation_reversal` | Pooled vs within-group association; flags Simpson's paradox (Pearl 2014) | Aggregation reversal |

`validate_grounding` also accepts `table.column` references against a multi-table schema.

---

## 2. What is implemented: all 52 catalog tools, plus 9 relational tools

First pass (10 tools) walked one pipeline end to end, the second added 32, the third the last 10. All are
plain typed functions returning a dataclass with no LLM call inside, so `tests/` (137 checks) runs with no
API key. The eight Claude skills in `claude_skills/` call these tools (see its README).

| Category | Implemented | File |
|---|---|---|
| A Grounding & validation | all 7: `detect_separator_and_encoding`, `profile_dataset` (`profiling.py`); `validate_grounding`, `check_method_appropriateness` (`validation.py`); `verify_numeric_consistency`, `check_repair_scope`, `flag_unmeasurable_dimension` | `verification.py` for the last three |
| B Loading & cleaning | all 9: `load_dataset` (`loading.py`), `missing_value_report` (`profiling.py`), `impute_missing`, `scale_numeric`, `encode_categorical`, `detect_outliers_iqr`, `bin_continuous`, `engineer_categorical_interactions`, `polynomial_expand` | `cleaning.py` |
| C Descriptive | all 5: `describe_column`, `correlation_matrix` (`descriptive.py`); `frequency_table`, `groupby_aggregate`, `column_summary_for_viz` | `exploration.py` |
| D Hypothesis testing | all 10: `chi_square_association`, `ab_test` (`inference.py`); `confidence_interval`, `two_sample_confidence_interval`, `ks_normality_test`, `variance_equality_test`, `nonparametric_test`, `bootstrap_ab_test`, `ab_power_analysis`, `multiple_comparison_correction` | `hypothesis.py` |
| E Regression & modelling | all 7: `linear_regression`, `multivariable_regression`, `glm_fit`, `robust_regression`, `regularized_regression`, `dimensionality_reduction`, `feature_selection` | `modeling.py` |
| F Time series | all 3: `stationarity_test`, `seasonal_decompose`, `arima_forecast` | `timeseries.py` |
| G Survival | all 3: `kaplan_meier_curve`, `logrank_test`, `cox_regression` | `survival.py` |
| H Causal / Bayesian | all 3: `propensity_score_matching`, `difference_in_differences` (`causal.py`); `bayesian_estimate` (`bayesian.py`) | |
| I Visualization | all 3: `recommend_chart`, `render_chart`, `explain_chart` | `visualization.py` |
| J Interpretation | all 2: `interpret_result`, `critique_findings` | `interpretation.py` |
| K Relational (added) | 9: `discover_tables`, `infer_keys`, `build_relationship_graph`, `plan_join_path`, `aggregate_to_grain`, `validate_join`, `execute_join_plan`, `check_unit_of_analysis`, `detect_aggregation_reversal` | `relational.py` |

**Verification.** Ground truth is independent of the code under test: the benchmark's verified
Titanic/Wine values (survival 0.3838 / 0.742, alcohol-quality r = 0.476), scipy reference calls, and the
synthetic five-table shop database with planted traps for the relational layer
(`demo/make_shop_tables.py`), the Freireich 6-MP trial for survival (S(6) = 0.857, S(23) = 0.448, log-rank chi2 = 16.79, Cox coefficient
1.572 as in R `coxph`), unpenalised sklearn for logistic regression, and analytic conjugate posteriors.

Slides: `../docs/slides/Skills_Library.pptx`.

---

## 3. Provenance: what each tool was checked against

The tools were written from the statistical methods themselves, not adapted from the reference
repositories' code. Afterwards, each tool whose catalog row cites a reference function was compared
against that function's source:

- StatABench: `repos/StatABench/mcp_scripts/mcp_server.py`
- MetaGPT (used by DataSciBench): `repos/DataSciBench/MetaGPT/.../tools/libs/{data_preprocess,feature_engineering}.py`
- LIDA: `repos/lida/lida/components/summarizer.py`

"cf." in a docstring means "compare with", not "derived from". Every name the catalog cites exists in
the source. The table records where behaviour differs, and which differences are deliberate fixes.

| Tool | Reference | Same purpose? | Differences (★ = deliberate improvement over the reference) |
|---|---|---|---|
| `missing_value_report` | StatABench `check_missing_values` | yes | Adds the structural (>50%) vs sparse split |
| `detect_outliers_iqr` | StatABench `detect_outliers` | yes | Same Tukey fences; one column per call |
| `profile_dataset` | StatABench `check_column_type_is` | broader | Whole-frame profile instead of a yes/no type check |
| `describe_column` | StatABench `calculate_statistic` | broader | Full summary dispatched on dtype; theirs returns one named statistic |
| `correlation_matrix` | StatABench `correlation_analysis` | partly | Pearson/Spearman/Kendall plus the strongest pair; no partial correlation (theirs supports it) |
| `confidence_interval` | StatABench `ci_normal` | partly | Computes from data with t; theirs takes summary statistics and also does known-σ and variance intervals |
| `two_sample_confidence_interval` | StatABench `ci_two_normal` | partly | Welch only; theirs also has pooled, known-σ and variance-ratio intervals |
| `chi_square_association` | StatABench `contingency_test` | partly | ★ Cramér's V from uncorrected χ²; ★ expected-count warning. No Fisher or Mantel-Haenszel (theirs has both) |
| `ks_normality_test` | StatABench `ks_test` | yes | ★ Lilliefors correction. Theirs fits the parameters and then reads a plain KS p-value, which is far too large (test `test_normality_uses_lilliefors...`) |
| `variance_equality_test` | StatABench `mood_variance_test` | yes | Two groups of one column instead of two columns |
| `nonparametric_test` | StatABench `nonparametric_test` | yes | ★ Wilcoxon pairs by subject id. Theirs truncates to the shorter column and pairs by row position. ★ Adds effect sizes; picks the test from the group count |
| `ab_test` | StatABench `ab_ttest` | yes | Same Welch test; ★ adds Cohen's d |
| `bootstrap_ab_test` | StatABench `bootstrap_abtest` | yes | Difference is mean(a) − mean(b) (theirs is b − a); ★ adds a permutation p-value; mean only (theirs also does median) |
| `ab_power_analysis` | StatABench `ab_power_analysis` | yes | Same statsmodels engines; two-sided by default (theirs is one-sided `larger`); also solves for power |
| `multiple_comparison_correction` | StatABench `fdr_control_df`, `fwer_control_df` | yes | Same BH / Bonferroni / Holm and the same `<= alpha` rule; no BY (theirs has it) |
| `linear_regression` | StatABench `simple_linear_regression` | yes | ★ Adds CIs and a residual-normality check |
| `multivariable_regression` | StatABench `multivariable_linear_regression` | yes | Same dummy coding (drop_first); ★ reports dropped rows and a singular-design error |
| `regularized_regression` | StatABench `advanced_regression` | yes | ★ Nested-CV R² instead of one 80/20 split. (Theirs also reports MSE under the name `rmse`.) |
| `dimensionality_reduction` | StatABench `sparse_pca_analysis` | broader | Adds ordinary PCA; ★ flags that sparse-PCA variance shares are approximate |
| `stationarity_test` | StatABench `test_stationarity` | yes | ★ Runs ADF and KPSS together and reports disagreement; ★ sorts by a time column |
| `kaplan_meier_curve` | StatABench `kaplan_meier_plot` | yes | ★ Adds a log(-log) Greenwood CI and the median; one curve per call (theirs can group) |
| `logrank_test` | StatABench `logrank_test_compare` | partly | Two groups only (theirs handles k groups); ★ warns when curves cross |
| `impute_missing` | MetaGPT `FillMissingValue` | yes | Default median (theirs mean); ★ warns on heavy imputation |
| `scale_numeric` | MetaGPT `MinMaxScale` / `StandardScale` / `RobustScale` / `MaxAbsScale` | yes | ★ Skips and reports zero-spread columns |
| `encode_categorical` | MetaGPT `OrdinalEncode` / `OneHotEncode` / `LabelEncode` / `TargetMeanEncoder` / `KFoldTargetMeanEncoder` | yes | Target encodings replace the column (theirs add a new one); ★ warns on leakage and on values that become NaN |
| `bin_continuous` | MetaGPT `SplitBins` | yes | ★ Leaves missing values as missing. Theirs fills NaN with 0 before binning, which moves them into the lowest bin |
| `feature_selection` | MetaGPT `GeneralSelection`, `VarianceBasedSelection` | yes | `general` and `variance` modes mirror both. `correlation` mode has no MetaGPT counterpart. *Corrected*: an earlier docstring wrongly attributed it to GeneralSelection |
| `column_summary_for_viz` | LIDA `summarizer.get_column_properties` | yes | Same fields (dtype, min/max, 3 samples, n_unique). Role is rule-based where LIDA asks the LLM for a `semantic_type`. ★ Detects date strings |
| `recommend_chart` / `render_chart` / `explain_chart` | LIDA `goal()` / `visualize()` / `execute()` / `explain()` | stage-level only | Same pipeline stages, no shared code; LIDA generates chart code with an LLM |
| `check_method_appropriateness`, `critique_findings` | StatABench `judger-open/modeling_groundedness.py`, `result_bias.py` | concern only | Those files are LLM-judge prompts, not rules. These tools turn the checkable part of those concerns into deterministic rules |

| `engineer_categorical_interactions` | MetaGPT `CatCount`, `CatCross`, `GroupStat` | yes | Same three operations and the same category cap on crosses; ★ warns that group statistics of the target leak it |
| `polynomial_expand` | MetaGPT `PolynomialExpansion` | yes | Caller chooses the columns (given more than 10, theirs silently keeps the 10 most target-correlated); keeps the original columns (theirs drops them); ★ missing inputs stay missing (theirs fills 0) |
| `glm_fit` | StatABench `run_glm` | yes | Same families except inverse Gaussian and multinomial; ★ odds/rate ratios, overdispersion check, separation warning, negative-binomial dispersion estimated (statsmodels' default silently fixes it at 1) |
| `robust_regression` | StatABench `huber_regression` | yes | Same Huber RLM; ★ OLS coefficients alongside and a count of down-weighted points |
| `seasonal_decompose` | StatABench `decompose_stl` | yes | Same STL; ★ trend and seasonal strength, with seasonality tested against shuffled data |
| `arima_forecast` | StatABench `auto_arima_modeling` | yes | Grid search in statsmodels instead of pmdarima; d chosen by ADF+KPSS; ★ BIC by default (AIC overfit a simulated AR(1)); forecast intervals and a Ljung-Box residual check |
| `cox_regression` | StatABench `fit_cox_model` | yes | statsmodels PHReg with Efron ties instead of lifelines; ★ proportional-hazards check per covariate |
| `propensity_score_matching` | StatABench `estimate_ATT_with_psm` | yes | Matches with replacement (theirs without); ★ caliper of 0.2 SD on the logit scale (Austin 2011; theirs 0.05 on the raw probability); ★ balance (SMD) before and after matching, which theirs does not report; bootstrap CI |
| `difference_in_differences` | StatABench `estimate_did_effect` | yes | Both offer clustered SEs and a pre-period check (theirs a placebo treatment date, this a group x time pre-trend test); ★ unclustered SEs are heteroskedasticity-robust HC1 (theirs plain OLS) |
| `bayesian_estimate` | StatABench `bayesian_inference`, `fit_hierarchical_model` | partly | Beta-Binomial as theirs; ★ normal mean with unknown variance (theirs plugs in the sample variance as known); ★ adds two-group P(A > B); hierarchical by empirical Bayes instead of PyMC sampling (instant and deterministic, but tau is estimated, not given a prior). Their `bayesian_linear_regression` (sklearn BayesianRidge) has no counterpart here |

Grounding tools with no counterpart in any reference repository: `detect_separator_and_encoding`,
`validate_grounding`, `verify_numeric_consistency`, `check_repair_scope`, `flag_unmeasurable_dimension`.
