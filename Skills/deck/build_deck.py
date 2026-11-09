"""
Builds Skill_Library_and_SkillBench.pptx: black text on white, native (editable) PowerPoint charts.
Numbers are read from the project files where possible so the deck matches the code.

    python deck/build_deck.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

FYP = pathlib.Path(__file__).resolve().parents[2]
BENCH = FYP / "Benchmark" / "skillbench"
OUT = FYP / "docs" / "slides" / "Skill_Library_and_SkillBench.pptx"
sys.path.insert(0, str(FYP / "Skills" / "claude_skills"))
sys.path.insert(0, str(FYP / "Skills"))
import build as skill_build  # noqa: E402

BLACK, DARK, MID, LIGHT, RULE, FILL = (RGBColor(0x11, 0x11, 0x11), RGBColor(0x33, 0x33, 0x33),
                                       RGBColor(0x6B, 0x6B, 0x6B), RGBColor(0xA6, 0xA6, 0xA6),
                                       RGBColor(0xD0, 0xD0, 0xD0), RGBColor(0xF2, 0xF2, 0xF2))
GREYS = [RGBColor(0xBF, 0xBF, 0xBF), RGBColor(0x80, 0x80, 0x80), RGBColor(0x40, 0x40, 0x40), RGBColor(0x11, 0x11, 0x11)]
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
state = {"n": 0}
TITLES: list[str] = []  # one per slide, used to attach speaker notes by title


# ----------------------------------------------------------------------------- helpers
def text(slide, x, y, w, h, content, size=14, bold=False, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space=4, font=FONT, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    items = content if isinstance(content, list) else [content]
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        runs = item if isinstance(item, list) else [(item, {})] if isinstance(item, str) else [item]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = t
            f = run.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = o.get("color", color)
    return tb


def bullets(slide, x, y, w, h, items, size=14, space=6):
    tb = text(slide, x, y, w, h, [f"•  {i}" if isinstance(i, str) else i for i in items], size=size, space=space)
    return tb


def new_slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    state["n"] += 1
    TITLES.append(title)
    text(s, 0.6, 0.4, 12.1, 0.7, title, size=30, bold=True, color=BLACK)
    if subtitle:
        text(s, 0.6, 1.08, 12.1, 0.4, subtitle, size=14, color=MID)
    text(s, 12.2, 7.02, 0.6, 0.3, str(state["n"]), size=10, color=LIGHT, align=PP_ALIGN.RIGHT)
    text(s, 0.6, 7.02, 9, 0.3, "Reliable Data Analysis with LLM Agents", size=10, color=LIGHT)
    return s


def divider(label, title, sub):
    s = prs.slides.add_slide(BLANK)
    state["n"] += 1
    TITLES.append(f"{label.title()}: {title}")
    text(s, 0.9, 2.6, 11.5, 0.4, label, size=14, bold=True, color=MID)
    text(s, 0.9, 3.05, 11.5, 0.9, title, size=40, bold=True, color=BLACK)
    text(s, 0.9, 4.05, 11.5, 0.8, sub, size=16, color=MID)
    return s


def box(slide, x, y, w, h, fill=FILL, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    sh.shadow.inherit = False
    return sh


def table(slide, x, y, w, rows, col_w, size=11, header=True, row_h=0.3, bold_first_col=False, mono_first=False):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = header
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = FILL if (header and i == 0) else RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = ""
            run = p.add_run()
            run.text = str(val)
            f = run.font
            mono = mono_first and j == 0 and i > 0
            f.size = Pt(size - 1.5 if (mono and len(str(val)) > 30) else size)
            f.name = "Consolas" if mono else FONT
            f.bold = (header and i == 0) or (bold_first_col and j == 0)
            f.color.rgb = BLACK if (header and i == 0) else DARK
    return shape


def style_chart(chart, size=11, legend=True, value_fmt='0%', labels=True, max_v=None):
    chart.has_title = False
    chart.font.size = Pt(size)
    chart.font.name = FONT
    chart.font.color.rgb = DARK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RULE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size)
    va.tick_labels.font.color.rgb = MID
    va.tick_labels.number_format = value_fmt
    va.tick_labels.number_format_is_linked = False
    if max_v is not None:
        va.maximum_scale = max_v
        va.minimum_scale = 0
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.color.rgb = DARK
    ca.format.line.color.rgb = LIGHT
    plot = chart.plots[0]
    plot.gap_width = 60
    if labels:
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.font.size = Pt(size - 1)
        dl.font.color.rgb = DARK
        dl.number_format = value_fmt
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for i, series in enumerate(plot.series):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = GREYS[min(i + (4 - len(plot.series)), 3)] if len(plot.series) > 1 else DARK


def bar_chart(slide, x, y, w, h, cats, series: dict, kind=XL_CHART_TYPE.COLUMN_CLUSTERED, **kw):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    style_chart(gf.chart, legend=len(series) > 1, **kw)
    return gf.chart


def stat(slide, x, y, w, number, label):
    text(slide, x, y, w, 0.9, number, size=48, bold=True, color=BLACK)
    text(slide, x, y + 0.95, w, 0.6, label, size=13, color=MID)


# ----------------------------------------------------------------------------- data
qs = pd.DataFrame(json.loads((BENCH / "questions" / "questions.json").read_text(encoding="utf-8")))
SYN = {"clinic", "school", "saas", "shop"}
qs["synthetic"] = qs.dataset.isin(SYN)
v1 = pd.read_csv(BENCH / "results" / "scores.csv")
SKILL_TOOLS = skill_build.SKILL_TOOLS

TOOL_CATS = [
    ("A", "Grounding & validation", ["detect_separator_and_encoding", "profile_dataset", "validate_grounding",
                                     "check_method_appropriateness", "verify_numeric_consistency", "check_repair_scope",
                                     "flag_unmeasurable_dimension"]),
    ("B", "Loading & cleaning", ["load_dataset", "missing_value_report", "impute_missing", "scale_numeric",
                                 "encode_categorical", "detect_outliers_iqr", "bin_continuous",
                                 "engineer_categorical_interactions", "polynomial_expand"]),
    ("C", "Descriptive statistics", ["describe_column", "correlation_matrix", "frequency_table", "groupby_aggregate",
                                     "column_summary_for_viz"]),
    ("D", "Hypothesis testing", ["ab_test", "chi_square_association", "confidence_interval",
                                 "two_sample_confidence_interval", "ks_normality_test", "variance_equality_test",
                                 "nonparametric_test", "bootstrap_ab_test", "ab_power_analysis",
                                 "multiple_comparison_correction"]),
    ("E", "Regression & modelling", ["linear_regression", "multivariable_regression", "glm_fit", "robust_regression",
                                     "regularized_regression", "dimensionality_reduction", "feature_selection"]),
    ("F", "Time series", ["stationarity_test", "seasonal_decompose", "arima_forecast"]),
    ("G", "Survival analysis", ["kaplan_meier_curve", "logrank_test", "cox_regression"]),
    ("H", "Causal & Bayesian", ["propensity_score_matching", "difference_in_differences", "bayesian_estimate"]),
    ("I", "Visualization", ["recommend_chart", "render_chart", "explain_chart"]),
    ("J", "Interpretation", ["interpret_result", "critique_findings"]),
    ("K", "Relational (multi-table)", ["discover_tables", "infer_keys", "build_relationship_graph", "plan_join_path",
                                       "aggregate_to_grain", "validate_join", "execute_join_plan",
                                       "check_unit_of_analysis", "detect_aggregation_reversal"]),
]
CAT_OF = {t: c for c, _, ts in TOOL_CATS for t in ts}
N_TOOLS = sum(len(ts) for _, _, ts in TOOL_CATS)

DESC = {
    "detect_separator_and_encoding": "Sniffs delimiter and encoding before parsing",
    "profile_dataset": "Shape, types, cardinality, missing share per column",
    "validate_grounding": "Every referenced table and column must exist",
    "check_method_appropriateness": "Rejects methods that do not fit the data",
    "verify_numeric_consistency": "Recomputes a claimed number; names the mismatch",
    "check_repair_scope": "Checks a 'fix' touched the failing code region",
    "flag_unmeasurable_dimension": "Reports n/a with coverage instead of 0",
    "load_dataset": "CSV, Parquet, Excel with sniffed delimiter",
    "missing_value_report": "Structural vs sparse missingness",
    "impute_missing": "Mean, median, mode, constant; warns on heavy use",
    "scale_numeric": "Min-max, standard, robust, max-abs scaling",
    "encode_categorical": "One-hot, ordinal, label, (k-fold) target encoding",
    "detect_outliers_iqr": "Tukey fences; reports, never deletes",
    "bin_continuous": "Equal-width or quantile bins",
    "engineer_categorical_interactions": "Counts, crosses, group statistics",
    "polynomial_expand": "Polynomial and interaction terms",
    "describe_column": "Summary dispatched on data type",
    "correlation_matrix": "Pearson, Spearman, Kendall; strongest pair",
    "frequency_table": "Counts and proportions with denominator",
    "groupby_aggregate": "Group aggregates with group sizes",
    "column_summary_for_viz": "Column roles for chart selection",
    "ab_test": "Welch t-test with Cohen's d",
    "chi_square_association": "Chi-square with Cramer's V",
    "confidence_interval": "t interval for a mean",
    "two_sample_confidence_interval": "Welch interval for a difference",
    "ks_normality_test": "Lilliefors-corrected normality test",
    "variance_equality_test": "Mood's test for scale",
    "nonparametric_test": "Mann-Whitney, Wilcoxon, Kruskal-Wallis",
    "bootstrap_ab_test": "Bootstrap CI and permutation p-value",
    "ab_power_analysis": "Sample size or power",
    "multiple_comparison_correction": "Bonferroni, Holm, Benjamini-Hochberg",
    "linear_regression": "OLS with SE, p and CI",
    "multivariable_regression": "OLS with VIF, dummy coding",
    "glm_fit": "Logistic, Poisson, NB, gamma; odds/rate ratios",
    "robust_regression": "Huber fit beside OLS",
    "regularized_regression": "Lasso, ridge, elastic net; CV R-squared",
    "dimensionality_reduction": "PCA or sparse PCA with loadings",
    "feature_selection": "General, variance, correlation filters",
    "stationarity_test": "ADF and KPSS together",
    "seasonal_decompose": "STL with trend and seasonal strength",
    "arima_forecast": "BIC order search, intervals, residual check",
    "kaplan_meier_curve": "Survival curve, log(-log) CI, median",
    "logrank_test": "Two-group survival comparison",
    "cox_regression": "Hazard ratios, proportional-hazards check",
    "propensity_score_matching": "ATT with covariate balance",
    "difference_in_differences": "Policy effect with pre-trend test",
    "bayesian_estimate": "Conjugate and partial-pooling posteriors",
    "recommend_chart": "Chart from column roles",
    "render_chart": "Draws the chart; re-checks columns",
    "explain_chart": "States the finding with numbers",
    "interpret_result": "Effect size vs significance",
    "critique_findings": "Assumption and multiplicity checks",
    "discover_tables": "Loads folders, files or SQLite",
    "infer_keys": "Primary and foreign keys with evidence",
    "build_relationship_graph": "Links with 1:1 / 1:N, bridges",
    "plan_join_path": "Join order from the analysis grain",
    "aggregate_to_grain": "Child table rolled up before joining",
    "validate_join": "Rows before/after, fan-out, orphans",
    "execute_join_plan": "Runs a plan with a check per step",
    "check_unit_of_analysis": "Detects repeated entities",
    "detect_aggregation_reversal": "Flags Simpson's paradox",
}

# ============================================================================= 1 title
s = prs.slides.add_slide(BLANK)
state["n"] += 1
TITLES.append("Title")
text(s, 0.9, 2.3, 11.5, 1.2, "Reliable Data Analysis with LLM Agents", size=44, bold=True, color=BLACK)
text(s, 0.9, 3.45, 11.5, 0.6, "A statistical skill library and the SkillBench benchmark", size=22, color=DARK)
text(s, 0.9, 5.6, 11.5, 0.4, "Janhavee Singh  ·  Final Year Project  ·  Progress review", size=14, color=MID)

# ============================================================================= 2 overview
s = new_slide("Overview")
stat(s, 0.6, 1.6, 3.0, f"{N_TOOLS}", "deterministic analysis tools\nin 11 categories")
stat(s, 3.7, 1.6, 3.0, "8", "skills for the Claude app,\neach calling the tools")
stat(s, 6.8, 1.6, 3.0, f"{len(qs)}", f"benchmark questions on\n{qs.dataset.nunique()} datasets")
stat(s, 9.9, 1.6, 3.0, "137", "automated tests,\nno API key required")
rows = [["Component", "Purpose", "Status"],
        ["Tool library", "Tested functions for profiling, statistics, modelling, joins and interpretation", "Complete"],
        ["Skills", "Instructions: when a method applies, what to check, how to report", "Complete; tested in the app"],
        ["SkillBench v1", "108 questions, 9 datasets, 3 conditions, 1 model", "Complete"],
        ["SkillBench v2", "206 questions, 15 datasets (123 multi-table), 4 conditions, 2 models, 3 runs",
         "Complete" if (BENCH / "results" / "v2" / "stats.json").exists() else "Running"],
        ["StatABench (external)", "198 published items, unchanged; same 4 conditions, 2 models, 3 runs",
         "Complete" if (BENCH / "results" / "statabench" / "stats.json").exists() else "Running"]]
table(s, 0.6, 3.75, 12.1, rows, [2.2, 7.4, 2.5], size=13, row_h=0.48)

# ============================================================================= tools
divider("PART 1", "Tool library", f"{N_TOOLS} deterministic functions with input checks, typed results and warnings")

s = new_slide("Tool library at a glance", "Number of tools per category")
cats = [f"{c} {n}" for c, n, _ in TOOL_CATS]
ch = bar_chart(s, 0.6, 1.6, 8.2, 5.2, [n for _, n, _ in TOOL_CATS], {"Tools": [len(t) for _, _, t in TOOL_CATS]},
               kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.has_major_gridlines = False
ch.value_axis.visible = False
bullets(s, 9.2, 1.75, 3.6, 5.0, [
    "Plain, typed Python functions",
    "Same result envelope: ok, values, warnings",
    "No language model inside a tool",
    "Statistics from scipy, statsmodels, scikit-learn",
    "Categories A and K have no equivalent in the five reference systems",
    f"{N_TOOLS} tools = 52 catalog tools + 9 relational tools"], size=14, space=10)


def tool_slide(title, letters):
    s = new_slide(title)
    groups = [g for g in TOOL_CATS if g[0] in letters]
    cols = [[], []]
    load = [0, 0]
    for g in groups:
        i = 0 if load[0] <= load[1] else 1
        cols[i].append(g)
        load[i] += len(g[2]) + 1.4
    cols = [c for c in cols if c]
    for ci, col in enumerate(cols):
        x = 0.6 + ci * 6.2
        rows = [["Tool", "Function"]]
        for c, name, ts in col:
            rows.append([f"{c}  {name}", ""])
            rows += [[t, DESC[t]] for t in ts]
        widths = [3.0, 2.95] if len(cols) == 2 else [4.0, 8.1]
        total = sum(widths)
        rh = min(0.42, 5.0 / len(rows))
        shape = table(s, x, 1.45, total, rows, widths, size=11.5, row_h=rh, mono_first=True)
        tb = shape.table
        for i, r in enumerate(rows):
            if i and r[1] == "":
                c0 = tb.cell(i, 0)
                c0.merge(tb.cell(i, 1))
                c0.fill.solid()
                c0.fill.fore_color.rgb = FILL
                run = c0.text_frame.paragraphs[0].runs[0]
                run.font.bold = True
                run.font.name = FONT
    return s


tool_slide("Tools: grounding, loading and descriptive statistics", "ABC")
tool_slide("Tools: hypothesis testing and modelling", "DE")
tool_slide("Tools: time series, survival, causal, visualization, interpretation", "FGHIJ")
tool_slide("Tools: relational analysis for multiple tables", "K")

s = new_slide("Tool construction", "Every tool follows one pattern: check inputs, compute with a trusted library, return context")
steps = [("1  Check", "Columns exist, types fit, enough data;\nreturn ok = False with a reason"),
         ("2  Compute", "scipy / statsmodels / scikit-learn\nfor the statistics"),
         ("3  Add context", "Effect size, interval, n, rows dropped,\nassumption checks"),
         ("4  Return", "Typed result: ok, values, warnings\n(never a silent failure)")]
for i, (h, b) in enumerate(steps):
    x = 0.6 + i * 3.1
    box(s, x, 1.75, 2.85, 1.75)
    text(s, x + 0.2, 1.9, 2.5, 0.4, h, size=17, bold=True, color=BLACK)
    text(s, x + 0.2, 2.45, 2.5, 1.0, b, size=13, color=DARK)
code = ['@dataclass', 'class ABTestReport(ToolResult):      # ok, warnings', '    mean_a: float; mean_b: float',
        '    p_value: float; cohens_d: float', '', 'def ab_test(df, group_col, value_col, a, b):',
        '    if value_col not in df.columns:', '        return ABTestReport(ok=False, warnings=["Column not found"])',
        '    t, p = stats.ttest_ind(xa, xb, equal_var=False)   # Welch',
        '    return ABTestReport(ok=True, p_value=p, cohens_d=d, ...)']
box(s, 0.6, 3.85, 7.4, 2.95, fill=FILL)
text(s, 0.8, 3.98, 7.1, 2.8, code, size=11.5, color=BLACK, font="Consolas", space=1)
bullets(s, 8.35, 3.95, 4.4, 2.9, [
    "Design rules derived from the phase-1 benchmark failures",
    "Each tool cross-checked against its reference implementation (StatABench, MetaGPT, LIDA)",
    "Documented deviations where the reference is statistically weaker"], size=13.5, space=10)

s = new_slide("Tool verification", "137 automated tests; each compares a tool with an answer it did not produce")
rows = [["Evidence", "Example", "Tests"],
        ["Benchmark ground truth", "Titanic survival 0.3838; wine alcohol-quality r = 0.476", "Phase-1 values"],
        ["Reference libraries", "scipy t interval, Mann-Whitney, linregress, sklearn logistic", "Agreement to 1e-9"],
        ["Published results", "Freireich 6-MP: S(6) = 0.857, log-rank 16.79, Cox 1.572 (R coxph)", "Textbook values"],
        ["Planted structure", "Known coefficients, effects, outliers, Simpson's reversal", "Synthetic data"],
        ["Failure reproduction", "Wrong separator, missing table, 38.38 vs 0.3838 unit slip", "Regression tests"],
        ["Skill examples", "Every SKILL.md example run from the packaged zip", "Subprocess tests"]]
table(s, 0.6, 1.55, 8.0, rows, [2.3, 4.2, 1.5], size=12, row_h=0.5)
files = {"Initial tools": 20, "Extended tools": 42, "Review fixes": 25, "Remaining tools": 16, "Relational": 18,
         "Skill packages": 16}
ch = bar_chart(s, 8.9, 1.5, 3.9, 4.7, list(files), {"Tests": list(files.values())}, kind=XL_CHART_TYPE.BAR_CLUSTERED,
               value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
text(s, 8.9, 6.3, 3.9, 0.4, "Tests per test file", size=11, color=MID)

# ============================================================================= skills
divider("PART 2", "Skills", "Procedures that decide when a method applies, what to check and how to report the result")

s = new_slide("The eight skills", "Each skill is uploaded to the Claude app as one package")
SK = [("data-profiling", "Safe loading, schema, missing values, outliers, summaries"),
      ("distribution-analysis", "Shape, centre, spread, normality, intervals"),
      ("correlation-analysis", "Relationships, heatmaps, no causal claims"),
      ("group-comparison", "Tests, effect sizes, power, survival curves"),
      ("regression-analysis", "OLS, GLMs, Cox, matching, difference-in-differences"),
      ("time-series-analysis", "Ordering, stationarity, trend, seasonality, forecasts"),
      ("visualization", "Choose, draw and explain the chart"),
      ("relational-data-analysis", "Keys, safe joins, fan-out, orphans, Simpson's paradox")]
for i, (n, d) in enumerate(SK):
    c, r = i % 4, i // 4
    x, y = 0.6 + c * 3.1, 1.65 + r * 2.55
    box(s, x, y, 2.85, 2.3)
    text(s, x + 0.2, y + 0.2, 2.5, 0.4, n, size=14, bold=True, color=BLACK, font="Consolas")
    text(s, x + 0.2, y + 0.7, 2.5, 1.0, d, size=13, color=DARK)
    text(s, x + 0.2, y + 1.8, 2.5, 0.3, f"{len(SKILL_TOOLS[n])} tools documented", size=11, color=MID)

s = new_slide("Skill structure", "A skill encodes the research plan's definition of a skill")
rows = [["SKILL.md section", "Content"],
        ["Name and description", "Matched against the user's request to select the skill"],
        ["When to use / not to use", "Scope, with pointers to neighbouring skills"],
        ["Setup", "Makes the bundled library importable; checks the environment"],
        ["Procedure", "Numbered steps, each naming the tool and what to read from it"],
        ["Assumptions and common errors", "What must hold; mistakes to avoid"],
        ["Interpretation and report format", "How to state results without overclaiming"],
        ["Worked example", "Runnable code on a real dataset, tested on every build"]]
table(s, 0.6, 1.55, 7.6, rows, [2.9, 4.7], size=12.5, row_h=0.52)
box(s, 8.6, 1.55, 4.15, 2.3, fill=FILL)
text(s, 8.8, 1.7, 3.8, 2.1, ["group-comparison.zip", "  group-comparison/", "    SKILL.md",
                             "    reference/tools.md", "    scripts/statkit/"], size=13, color=BLACK, font="Consolas",
     space=3)
text(s, 8.6, 4.05, 4.15, 0.4, "Use in the Claude app", size=15, bold=True, color=BLACK)
text(s, 8.6, 4.5, 4.15, 2.3, ["1  The description matches the question", "2  Claude reads SKILL.md and the tool reference",
                              "3  The tools run in the code sandbox", "4  The report follows the skill's format"],
     size=13, space=8)

s = new_slide("Skill-tool relationship", "Tools documented by each skill, by tool category")
letters = [c for c, _, _ in TOOL_CATS]
rows = [["Skill"] + letters + ["Total"]]
for sk, tools in SKILL_TOOLS.items():
    counts = {c: 0 for c in letters}
    for t in tools:
        counts[CAT_OF[t.split(".")[1]]] += 1
    rows.append([sk] + [str(counts[c]) if counts[c] else "" for c in letters] + [str(len(tools))])
shape = table(s, 0.6, 1.55, 12.1, rows, [2.95] + [0.76] * 11 + [0.79], size=12.5, row_h=0.46, mono_first=True)
for i in range(1, len(rows)):
    for j in range(1, 13):
        cell = shape.table.cell(i, j)
        cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        v = rows[i][j]
        if v and j <= 11:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xE4, 0xE4, 0xE4) if int(v) < 3 else RGBColor(0xC8, 0xC8, 0xC8)
text(s, 0.6, 5.85, 12.1, 0.9, [" · ".join(f"{c} {n}" for c, n, _ in TOOL_CATS),
                              "Every skill bundles the full library, so any tool remains available when a question needs it."],
     size=11.5, color=MID)

s = new_slide("Skill construction and validation")
flow = [("Write", "SKILL.md by hand:\nprocedure, checks,\ninterpretation"),
        ("Generate", "reference/tools.md from\nthe code's signatures\nand docstrings"),
        ("Bundle", "Copy the tool library\ninto scripts/statkit\n(self-contained)"),
        ("Test", "Run every example from\nthe zip in a clean\nprocess"),
        ("Package", "One zip per skill,\nfolder at the root,\nfor upload")]
for i, (h, b) in enumerate(flow):
    x = 0.6 + i * 2.48
    box(s, x, 1.55, 2.2, 2.0)
    text(s, x + 0.18, 1.7, 1.9, 0.4, h, size=17, bold=True, color=BLACK)
    text(s, x + 0.18, 2.2, 1.9, 1.3, b, size=12.5, color=DARK)
    if i < len(flow) - 1:
        text(s, x + 2.2, 2.3, 0.28, 0.4, "›", size=24, color=LIGHT, align=PP_ALIGN.CENTER)
bullets(s, 0.6, 3.95, 6.0, 2.9, [
    "One build command produces all eight packages",
    "The tool reference cannot drift from the code: it is regenerated on every build",
    "Validated end to end in the Claude app: the matching skill loads without being named"], size=14, space=10)
bullets(s, 6.9, 3.95, 5.9, 2.9, [
    "Example: \"Did survivors pay more?\" loads group-comparison",
    "Result: +£26.28 (95% CI £18.73 to £33.83), Cohen's d = 0.55, medium effect",
    "Reported as an association: fare tracks passenger class"], size=14, space=10)

# ============================================================================= benchmark
divider("PART 3", "SkillBench", "Measuring whether skills make LLM data analysis more reliable on real, messy data")

s = new_slide("Motivation", "The phase-1 benchmark could not separate the systems")
stat(s, 0.6, 1.7, 3.6, "98%", "method-selection accuracy of a\nplain LLM on the phase-1 questions")
bullets(s, 4.6, 1.75, 8.1, 2.4, [
    "Phase-1 data (Titanic, Wine) is small, clean and widely known",
    "Errors that matter in practice are not method choice but data handling: wrong grain, missing codes, "
    "silent joins, confounding",
    "SkillBench targets these: large real datasets, uncleaned, with documented traps"], size=15, space=10)
rows = [["Requirement", "SkillBench design"],
        ["Realistic data", "Public datasets up to 3 million rows, not cleaned"],
        ["Discriminating questions", "68% of questions contain a trap that changes the answer"],
        ["Unbiased ground truth", "Reference answers computed independently of the tool library"],
        ["Fair comparison", "Same model, data and wording in every condition"],
        ["Statistical rigour", "Repeated runs, confidence intervals, paired tests"]]
table(s, 0.6, 4.2, 12.1, rows, [3.5, 8.6], size=13, row_h=0.42)

s = new_slide("Benchmark versions", "v1 is the pilot; v2 is the full design. Both are reported.")
rows = [["", "v1 (pilot, complete)", "v2 (full design, complete)"],
        ["Questions", "108", f"{len(qs)} ({(~qs.synthetic).sum()} real, {qs.synthetic.sum()} synthetic)"],
        ["Datasets", "9", f"{qs.dataset.nunique()} (11 real, 4 synthetic)"],
        ["Conditions", "Plain, agent, agent + skills", "Adds agent + oracle skill"],
        ["Models", "gpt-oss-120b", "gpt-oss-120b, gpt-oss-20b"],
        ["Runs", "1 per condition", "3 per agent condition, temperature 0.7"],
        ["Scoring", "Deterministic match", "Deterministic match + independent judge"],
        ["Statistics", "McNemar test", "Bootstrap CI, paired permutation test"],
        ["Answers", "324", "4,120"]]
shape = table(s, 0.6, 1.55, 8.0, rows, [1.6, 2.9, 3.5], size=13, row_h=0.55, bold_first_col=True)
ch = bar_chart(s, 9.0, 1.5, 3.8, 5.0, ["v1", "v2"], {"Answers": [324, 4120]}, value_fmt="#,##0")
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
text(s, 9.0, 6.55, 3.8, 0.4, "Graded answers per version", size=11, color=MID)

s = new_slide("Benchmark setup: datasets", "11 real datasets (headline results) and 4 synthetic databases (appendix)")
rows = [["Dataset", "Source", "Tables", "Size", "Main difficulties"],
        ["NYC Yellow Taxi, Jan 2024", "NYC TLC", "2", "2.96 M trips", "Out-of-month dates, negative fares, unrecorded cash tips"],
        ["Citi Bike, Jan 2024", "Citi Bike", "1", "1.89 M rides", "December rides, >24 h rides, missing end stations"],
        ["US flights, Q1 2024", "BTS", "3", "1.66 M flights", "Conditional delay-cause recording, Latin-1 lookup"],
        ["Online Retail II", "UCI", "1", "1.07 M lines", "Cancellations, line vs invoice grain, duplicates"],
        ["MovieLens 1M", "GroupLens", "3", "1.0 M ratings", "Rating vs user level, multi-valued genres"],
        ["IMDb feature films", "IMDb", "4", "760 k films", "'\\N' missing code, vote skew, multi-director cells"],
        ["Olist e-commerce", "Olist", "9", "99 k orders", "Hidden person key, payment fan-out"],
        ["Diabetes 130-US", "UCI", "1+map", "102 k stays", "'?' and 'None' codes, repeat patients, deaths"],
        ["Beijing PM2.5", "UCI", "1", "43.8 k hours", "Missing readings, completeness rules"],
        ["F1DB", "F1DB", "47", "27.7 k results", "Large schema, unclassified finishers, pit fan-out"],
        ["Bike Sharing (DC)", "UCI", "1", "17.4 k hours", "Missing hours, confounding by hour"]]
table(s, 0.6, 1.5, 12.1, rows, [2.75, 1.25, 0.8, 1.6, 5.7], size=12.5, row_h=0.44)

s = new_slide("Benchmark setup: question bank", f"{len(qs)} questions: {(~qs.synthetic).sum()} on real data, "
              f"{qs.synthetic.sum()} synthetic")
by_ds = qs.groupby("dataset").size().sort_values(ascending=False)
ch = bar_chart(s, 0.6, 1.5, 7.3, 5.3, list(by_ds.index), {"Questions": list(by_ds.values)},
               kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
diff = qs.difficulty.value_counts()
ch2 = bar_chart(s, 8.2, 1.5, 4.6, 2.6, ["Easy", "Medium", "Hard"],
                {"Questions": [int(diff["easy"]), int(diff["medium"]), int(diff["hard"])]}, value_fmt="0")
ch2.value_axis.visible = False
ch2.value_axis.has_major_gridlines = False
cat = qs.category.value_counts()
ch3 = bar_chart(s, 8.2, 4.25, 4.6, 2.6, [c.replace("_", " ") for c in cat.index], {"Questions": list(cat.values)},
                kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch3.category_axis.reverse_order = True
ch3.value_axis.visible = False
ch3.value_axis.has_major_gridlines = False

s = new_slide("Question construction", "Every question is built and checked by the same pipeline")
flow = [("1  Inspect", "Profile the real data;\nfind its genuine\nproblems"),
        ("2  Write", "Question with one\nunambiguous answer\n(number, label, choice)"),
        ("3  Compute", "Gold answer in pandas,\nscipy, statsmodels;\nnever the tool library"),
        ("4  Trap", "Also compute the\ncareless answer;\ndrop traps that do not bite"),
        ("5  Audit", "Review every reference;\nreword ambiguous\nquestions; rerun")]
for i, (h, b) in enumerate(flow):
    x = 0.6 + i * 2.48
    box(s, x, 1.55, 2.2, 2.1)
    text(s, x + 0.18, 1.7, 1.9, 0.4, h, size=16, bold=True, color=BLACK)
    text(s, x + 0.18, 2.2, 1.95, 1.4, b, size=12.5, color=DARK)
rows = [["Field", "Example (FL05, flights)"],
        ["Question", "Average weather delay per operated flight, in minutes"],
        ["Reference", "0.99 min (cause minutes are recorded only for flights 15+ min late; blank means 0)"],
        ["Careless answer", "4.89 min (averaging recorded values only)"],
        ["Tolerance, labels", "abs 0.01; difficulty hard; trap: unrecorded values"]]
table(s, 0.6, 4.0, 12.1, rows, [2.4, 9.7], size=13, row_h=0.5)

trap_fam = {
    "Grain and unit of analysis": ["grain", "unit_of_analysis", "fan_out", "hidden_entity_key", "ratio_of_sums"],
    "Joins and linkage": ["join", "multi_hop_join", "orphans", "cross_table_consistency"],
    "Missing and unrecorded values": ["missing_code", "missing_values", "unrecorded_values", "censoring",
                                      "missing_data_rule", "time_gaps"],
    "Dirty or invalid records": ["duplicates", "invalid_records", "cancellations", "out_of_range_dates", "parsing",
                                 "multi_section_file", "multi_valued_field", "currency_units"],
    "Statistical reasoning": ["skew", "small_sample", "confounding", "simpsons_paradox", "multiple_testing",
                              "interaction", "trend", "unequal_exposure", "population_definition",
                              "cohort_definition", "era_effects", "unclassified_finishers"]}
tv = qs[qs.trap != "none"].trap.value_counts()
fam_counts = {k: int(sum(tv.get(t, 0) for t in v)) for k, v in trap_fam.items()}
s = new_slide("Trap taxonomy", f"{int((qs.trap != 'none').sum())} of {len(qs)} questions contain a trap that changes "
              "the correct answer")
ch = bar_chart(s, 0.6, 1.5, 6.4, 5.2, list(fam_counts), {"Questions": list(fam_counts.values())},
               kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
rows = [["Trap", "Example"],
        ["Grain", "Median invoice value from invoice lines (£304 vs £10)"],
        ["Hidden entity key", "Olist repeat customers: 3.1% vs 0% with the wrong key"],
        ["Unrecorded values", "Cash tips are not recorded in taxi data"],
        ["Censoring", "Account lifetime 407 vs 228 days"],
        ["Currency units", "Revenue $0.85 M vs 9.1 M unconverted"],
        ["Simpson's paradox", "Pooled effect reverses within groups"],
        ["Unequal exposure", "Weekday averages with 4 vs 5 occurrences"]]
table(s, 7.3, 1.55, 5.45, rows, [1.85, 3.6], size=11.5, row_h=0.6)

s = new_slide("Experimental design")
rows = [["Condition", "Scaffolding", "Tests"],
        ["Plain", "One script, run once, answer from its output", "Weak baseline"],
        ["Agent", "Tool loop: run Python up to 10 times, then submit", "Strong baseline"],
        ["Agent + skills", "Same loop plus load_skill and the bundled tool library", "RQ1 and RQ2"],
        ["Agent + oracle skill", "Same loop with the correct skill preloaded", "Skill content alone"]]
table(s, 0.6, 1.5, 7.6, rows, [2.1, 4.0, 1.5], size=12.5, row_h=0.6)
rows = [["Setting", "Value"],
        ["Models", "gpt-oss-120b, gpt-oss-20b"],
        ["Runs", "3 per agent condition, 1 plain"],
        ["Temperature", "0.7, fixed seeds"],
        ["Step limit", "10 code executions"],
        ["Judge", "qwen3-235b (different family)"],
        ["Answers", "4,120 SkillBench + 3,960 StatABench"]]
table(s, 8.6, 1.5, 4.15, rows, [1.5, 2.65], size=12.5, row_h=0.43)
bullets(s, 0.6, 4.75, 12.1, 2.0, [
    "All conditions receive identical data, file schemas and question text; only the scaffolding differs",
    "Oracle minus selected-skill accuracy isolates the cost of poor skill selection (RQ2)",
    "Comparing the two models tests whether skills help weaker models more"], size=14, space=8)

s = new_slide("Scoring and statistics")
rows = [["Measure", "Definition"],
        ["Correctness (primary)", "Deterministic: number within tolerance; label or choice exact match"],
        ["Trap fall", "Wrong answer equal to the precomputed careless answer"],
        ["Analytic quality", "Judge (different model family): method fit, caveats, no overclaiming"],
        ["Skill selection", "Skill loaded; overlap with the oracle skill"],
        ["Cost", "Code runs, errors, tokens, time"]]
table(s, 0.6, 1.5, 7.2, rows, [2.4, 4.8], size=12.5, row_h=0.6)
rows = [["Analysis", "Method"],
        ["Accuracy", "Mean of runs, SD across runs"],
        ["Uncertainty", "Bootstrap 95% CI over questions"],
        ["Condition comparison", "Paired sign-flip permutation test"],
        ["Reporting", "Real data headline; synthetic appendix"],
        ["Audit", "Every all-wrong question reviewed"]]
table(s, 8.2, 1.5, 4.55, rows, [1.8, 2.75], size=12.5, row_h=0.6)
text(s, 0.6, 5.5, 12.1, 1.2, ["Audit of v1 before reporting: category matching relaxed for equivalent names, six "
                              "answers recovered from transcripts, three ambiguous questions reworded and rerun; "
                              "all fixes applied equally to every condition."], size=13, color=MID)

# results (v1)
conds = ["plain", "agent", "agent_skills"]
lab = {"plain": "Plain", "agent": "Agent", "agent_skills": "Agent + skills"}
acc = v1.groupby("condition").correct.mean()
s = new_slide("Preliminary results (v1)", "108 questions, 9 datasets, gpt-oss-120b, one run per condition")
bar_chart(s, 0.6, 1.5, 5.6, 5.0, [lab[c] for c in conds], {"Accuracy": [float(acc[c]) for c in conds]}, max_v=1.0)
d = v1.groupby(["difficulty", "condition"]).correct.mean().unstack()
bar_chart(s, 6.5, 1.5, 6.3, 5.0, ["Easy", "Medium", "Hard"],
          {lab[c]: [float(d.loc[k, c]) for k in ["easy", "medium", "hard"]] for c in conds}, max_v=1.0)
text(s, 0.6, 6.55, 12.1, 0.4, "Accuracy overall (left) and by difficulty (right). 95% CI: Plain 79.6-92.6%, "
     "Agent 88.9-97.2%, Agent + skills 90.7-99.1%.", size=11, color=MID)

s = new_slide("Preliminary results (v1): comparisons")
rows = [["Comparison", "Discordant answers", "p (McNemar)"],
        ["Plain vs Agent + skills", "1 vs 11", "0.006"],
        ["Plain vs Agent", "2 vs 10", "0.039"],
        ["Agent vs Agent + skills", "2 vs 4", "0.69"]]
table(s, 0.6, 1.5, 6.0, rows, [2.8, 1.9, 1.3], size=14, row_h=0.6)
rows = [["Observation", "Value"],
        ["Skill loaded", "52% of questions"],
        ["Fitting skill loaded", "39% of questions"],
        ["Prompt tokens per question", "8.2 k (agent) vs 20.1 k (skills)"],
        ["Traps no condition solved", "3 (unrecorded tips, out-of-month dates, Simpson)"]]
table(s, 7.0, 1.5, 5.75, rows, [2.5, 3.25], size=14, row_h=0.6)
bullets(s, 0.6, 4.75, 12.1, 2.2, [
    "Gains concentrate on hard questions: 68% (plain), 80% (agent), 92% (agent + skills)",
    "Skills preserve correct handling of population definitions and grain (deceased patients, invoice level)",
    "Skill selection is the main bottleneck: half of the questions run without any skill",
    "One run and 108 questions are too few for a 2-point difference; addressed in v2"], size=16, space=10)

# v2 progress (read from the result files at build time)
V2 = BENCH / "results" / "v2"
PLAN = {"plain": 1, "agent": 3, "agent_skills": 3, "agent_skills_oracle": 3}
CLAB = {"plain": "Plain", "agent": "Agent", "agent_skills": "Agent + skills", "agent_skills_oracle": "Oracle"}
MODELS = ["gpt-oss-120b", "gpt-oss-20b"]
done_n = {(m, c): len(list(V2.glob(f"{m}/run*/{c}/*.json"))) for m in MODELS for c in PLAN}
errors = 0
for f in V2.glob("*/run*/*/*.json"):
    try:
        errors += json.loads(f.read_text(encoding="utf-8")).get("error") is not None
    except Exception:
        pass
total_done = sum(done_n.values())
progress = {"done": f"{total_done:,}", "errors": errors,
            "oracle20": len(list(V2.glob("gpt-oss-20b/run1/agent_skills_oracle/*.json")))}
RUN_DONE = total_done >= 4120
STATS_FILE = V2 / "stats.json"
SCORED = RUN_DONE and STATS_FILE.exists()
s = new_slide("SkillBench v2 run", f"{total_done:,} of 4,120 answers complete ({total_done / 4120:.0%}), "
              f"{errors} errors")
cats = [f"{CLAB[c]} ({m.split('-')[-1]})" for m in MODELS for c in PLAN]
ch = bar_chart(s, 0.6, 1.5, 7.6, 5.2, cats,
               {"Complete": [done_n[(m, c)] for m in MODELS for c in PLAN],
                "Planned": [PLAN[c] * len(qs) for m in MODELS for c in PLAN]},
               kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.plots[0].series[0].format.fill.fore_color.rgb = DARK
ch.plots[0].series[1].format.fill.fore_color.rgb = RULE
rows = [["Stage", "Status"],
        ["Questions, reference answers", "Complete"],
        ["Datasets downloaded, uncleaned", "Complete"],
        ["Run harness, 4 conditions", "Complete"],
        ["Run 1, both models", "Complete"],
        ["Runs 2 and 3", "Complete" if RUN_DONE else "In progress"],
        ["Judge scoring, statistics", "Complete" if SCORED else "After the run"],
        ["Reference-answer audit", "Complete" if SCORED else "After scoring"]]
table(s, 8.6, 1.5, 4.15, rows, [2.55, 1.6], size=12, row_h=0.5)
text(s, 8.6, 5.65, 4.15, 0.9, "Models via OpenRouter; judge qwen3-235b. v1 results are kept; v2 results follow.",
     size=11, color=MID)

# ----------------------------------------------------------------------------- v2 results (from analyze_v2.py)
v2ctx = {}
if SCORED:
    ST = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    C4 = ["plain", "agent", "agent_skills", "agent_skills_oracle"]
    SHORT = {"gpt-oss-120b": "120b", "gpt-oss-20b": "20b"}
    HEAD = "Accuracy on real data (headline)"
    A = {(r["model"], r["condition"]): r for r in ST["accuracy"] if r["set"] == HEAD}
    P = {(r["model"], r["a"], r["b"]): r for r in ST["paired"]}
    meta = ST["meta"]

    def pc(x, d=1):
        return "n/a" if x is None else f"{100 * x:.{d}f}%"

    def pts(x):
        return f"{100 * x:+.1f}"

    def pval(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    s = new_slide("Results (v2): accuracy", f"{meta['n_real']} real-data questions, {meta['real_datasets']} datasets; "
                  "mean of 3 runs (plain: 1 run); 95% bootstrap CI over questions")
    bar_chart(s, 0.6, 1.5, 6.6, 5.2, [CLAB[c] for c in C4],
              {f"gpt-oss-{SHORT[m]}": [A[(m, c)]["acc"] for c in C4] for m in MODELS}, max_v=1.0)
    rows = [["Model", "Condition", "Accuracy", "95% CI", "SD runs"]]
    for m in MODELS:
        for c in C4:
            r = A[(m, c)]
            rows.append([SHORT[m], CLAB[c], pc(r["acc"]), f"{100 * r['lo']:.1f}-{100 * r['hi']:.1f}",
                         f"{100 * r['sd']:.1f}" if r["runs"] > 1 else "-"])
    table(s, 7.5, 1.5, 5.25, rows, [0.7, 1.6, 1.05, 1.15, 0.75], size=11.5, row_h=0.5)

    s = new_slide("Results (v2): paired comparisons", "Difference in accuracy (points), per-question paired; "
                  "95% bootstrap CI; two-sided sign-flip permutation test")
    rows = [["Comparison"] + [f"gpt-oss-{SHORT[m]}: difference (95% CI)" for m in MODELS] + ["p (120b)", "p (20b)"]]
    PAIRS = [("plain", "agent"), ("agent", "agent_skills"), ("agent", "agent_skills_oracle"),
             ("agent_skills", "agent_skills_oracle"), ("plain", "agent_skills")]
    for a, b in PAIRS:
        rs = [P[(m, a, b)] for m in MODELS]
        rows.append([f"{CLAB[a]} → {CLAB[b]}"] + [f"{pts(r['diff'])} ({pts(r['lo'])} to {pts(r['hi'])})" for r in rs]
                    + [pval(r["p"]) for r in rs])
    table(s, 0.6, 1.55, 12.1, rows, [3.1, 3.2, 3.2, 1.3, 1.3], size=13, row_h=0.55)
    for m in MODELS:
        k = SHORT[m]
        v2ctx.update({f"acc_{k}_{c}": pc(A[(m, c)]["acc"]) for c in C4})
        v2ctx.update({f"d_{k}_{a}_{b}": pts(P[(m, a, b)]["diff"]) for a, b in PAIRS})
        v2ctx.update({f"p_{k}_{a}_{b}": pval(P[(m, a, b)]["p"]) for a, b in PAIRS})

    BD = ST["breakdown"]
    s = new_slide("Results (v2): difficulty", "Accuracy by question difficulty, real data, mean over runs")
    for i, m in enumerate(MODELS):
        lv = ["easy", "medium", "hard"]
        vals = {CLAB[c]: [next((r["acc"] for r in BD if r["by"] == "difficulty" and r["model"] == m
                                and r["level"] == L and r["condition"] == c), 0) for L in lv] for c in C4}
        bar_chart(s, 0.6 + i * 6.2, 1.85, 6.0, 4.9, ["Easy", "Medium", "Hard"], vals, max_v=1.0)
        text(s, 0.6 + i * 6.2, 1.45, 6.0, 0.4, f"gpt-oss-{SHORT[m]}", size=15, bold=True, color=BLACK)
        for c in C4:
            v2ctx[f"hard_{SHORT[m]}_{c}"] = pc(vals[CLAB[c]][2], 0)

    s = new_slide("Results (v2): traps and multi-table questions", "Accuracy, real data, mean over runs")
    for i, (by, lv, lab) in enumerate([("has_trap", ["False", "True"], ["No trap", "Trap"]),
                                       ("multi_table", ["False", "True"], ["Single table", "Multi-table"])]):
        cats, vals = [], {CLAB[c]: [] for c in C4}
        for m in MODELS:
            for L, nm in zip(lv, lab):
                cats.append(f"{nm} ({SHORT[m]})")
                for c in C4:
                    vals[CLAB[c]].append(next((r["acc"] for r in BD if r["by"] == by and r["model"] == m
                                               and r["level"] == L and r["condition"] == c), 0))
        bar_chart(s, 0.6 + i * 6.2, 1.5, 6.0, 5.3, cats, vals, max_v=1.0, labels=False)
    trap_fall = {(r["model"], r["condition"]): r["trap_fall"] for r in ST["accuracy"] if r["set"] == HEAD}
    for m in MODELS:
        for c in C4:
            v2ctx[f"tf_{SHORT[m]}_{c}"] = pc(trap_fall[(m, c)], 0)

    SEL = {r["model"]: r for r in ST["selection"]}
    Q = {(r["model"], r["condition"]): r for r in ST["quality"]}
    CO = {(r["model"], r["condition"]): r for r in ST["cost"]}
    s = new_slide("Results (v2): skill selection, quality and cost")
    rows = [["Agent + skills", "120b", "20b"],
            ["Loaded any skill", *(pc(SEL[m]["loaded"], 0) for m in MODELS)],
            ["Loaded the fitting skill", *(pc(SEL[m]["hit"], 0) for m in MODELS)],
            ["Accuracy, skill loaded", *(pc(SEL[m]["acc_loaded"], 0) for m in MODELS)],
            ["Accuracy, no skill loaded", *(pc(SEL[m]["acc_not"], 0) for m in MODELS)]]
    table(s, 0.6, 1.5, 5.4, rows, [2.8, 1.3, 1.3], size=12.5, row_h=0.5)
    rows = [["Judge (share of answers)", "Method fits", "Caveats", "No overclaim"]]
    for m in MODELS:
        for c in C4:
            r = Q[(m, c)]
            rows.append([f"{CLAB[c]} ({SHORT[m]})", pc(r["method"], 0), pc(r["caveats"], 0), pc(r["no_overclaim"], 0)])
    table(s, 6.4, 1.5, 6.35, rows, [2.6, 1.25, 1.2, 1.3], size=11.5, row_h=0.4)
    rows = [["Cost per question", *(f"{CLAB[c]}" for c in C4[1:])],
            *[[f"Prompt tokens ({SHORT[m]})", *(f"{CO[(m, c)]['prompt_tokens'] / 1000:.1f} k" for c in C4[1:])]
              for m in MODELS],
            *[[f"Code runs ({SHORT[m]})", *(f"{CO[(m, c)]['steps']:.1f}" for c in C4[1:])] for m in MODELS]]
    table(s, 0.6, 4.3, 5.4, rows, [2.1, 1.0, 1.15, 1.15], size=11.5, row_h=0.45)
    for m in MODELS:
        k = SHORT[m]
        v2ctx.update({f"sel_{k}": pc(SEL[m]["loaded"], 0), f"hit_{k}": pc(SEL[m]["hit"], 0),
                      f"accl_{k}": pc(SEL[m]["acc_loaded"], 0), f"accn_{k}": pc(SEL[m]["acc_not"], 0),
                      f"tok_{k}_agent": f"{CO[(m, 'agent')]['prompt_tokens'] / 1000:.1f}",
                      f"tok_{k}_skills": f"{CO[(m, 'agent_skills')]['prompt_tokens'] / 1000:.1f}"})

# ----------------------------------------------------------------------------- external benchmark: StatABench
SBD = BENCH / "results" / "statabench"
SB_STATS = SBD / "stats.json"
sbq = json.loads((FYP / "Benchmark" / "benchmark-fyp" / "repos" / "StatABench" / "data" / "stat-closed.json")
                 .read_text(encoding="utf-8"))
sbt = [x for x in sbq if x["code"] == 1]
TASK_NAME = {"survial analysis": "Survival analysis", "ABtest": "A/B testing", "NonpStat": "Non-parametric",
             "EDA": "Exploratory analysis", "High dimension data": "High-dimensional"}


def tname(t):
    return TASK_NAME.get(t, t[0].upper() + t[1:])


s = new_slide("External benchmark: StatABench", "Zhu et al. 2026: the 198 Stat-Closed items that require data "
              "and tools, used unchanged")
tc = pd.Series([tname(x["task"]) for x in sbt]).value_counts()
ch = bar_chart(s, 0.6, 1.5, 5.6, 5.3, list(tc.index), {"Items": [int(v) for v in tc.values]},
               kind=XL_CHART_TYPE.BAR_CLUSTERED, value_fmt="0")
ch.category_axis.reverse_order = True
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
src = pd.Series([("StatQA" if "StatQA" in str(x["reference"]) else "Gemini-generated" if "Gemini" in str(x["reference"])
                  else "Textbook / Kaggle / other") for x in sbt]).value_counts()
rows = [["Property", "Value"],
        ["Items", f"{len(sbt)} (17 statistical tasks)"],
        ["Data", f"{len({x['dataset'] for x in sbt if isinstance(x['dataset'], str)})} CSV datasets; "
                 f"{sum(not isinstance(x['dataset'], str) for x in sbt)} items with data in the text"],
        ["Item sources", ", ".join(f"{k} {v}" for k, v in src.items())],
        ["Reference answers", "StatABench's own ('ground truth2')"],
        ["Grading", "StatABench's comparison prompt, verbatim"],
        ["Judge", "qwen3-235b, temperature 0"],
        ["Conditions", "Same four as SkillBench, same harness"],
        ["Prior result", "StatABench's own 36-tool agent: 66.2% (gpt-oss-120b, phase 1)"]]
table(s, 6.6, 1.5, 6.15, rows, [1.9, 4.25], size=12, row_h=0.56)

sbctx = {}
if SB_STATS.exists():
    SS = json.loads(SB_STATS.read_text(encoding="utf-8"))
    SA = {(r["model"], r["condition"]): r for r in SS["accuracy"]}
    SPm = {(r["model"], r["a"], r["b"]): r for r in SS["paired"]}
    C4 = ["plain", "agent", "agent_skills", "agent_skills_oracle"]
    SHORT = {"gpt-oss-120b": "120b", "gpt-oss-20b": "20b"}
    runs_sb = max(r["runs"] for r in SS["accuracy"])

    def pc(x, d=1):
        return "n/a" if x is None else f"{100 * x:.{d}f}%"

    def pts(x):
        return f"{100 * x:+.1f}"

    def pval(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    s = new_slide("Results: StatABench", f"{len(sbt)} items; mean of {runs_sb} runs per agent condition (plain: 1); "
                  "95% bootstrap CI; paired permutation test")
    bar_chart(s, 0.6, 1.5, 5.6, 5.2, [CLAB[c] for c in C4],
              {f"gpt-oss-{SHORT[m]}": [SA[(m, c)]["acc"] if (m, c) in SA else 0 for c in C4] for m in MODELS},
              max_v=1.0)
    rows = [["Comparison", "120b: difference (95% CI)", "p", "20b: difference (95% CI)", "p"]]
    SPAIRS = [("plain", "agent"), ("agent", "agent_skills"), ("agent", "agent_skills_oracle"),
              ("agent_skills", "agent_skills_oracle")]
    for a, b in SPAIRS:
        row = [f"{CLAB[a]} → {CLAB[b]}"]
        for m in MODELS:
            r = SPm.get((m, a, b))
            row += [f"{pts(r['diff'])} ({pts(r['lo'])} to {pts(r['hi'])})", pval(r["p"])] if r else ["-", "-"]
        rows.append(row)
    table(s, 6.5, 1.5, 6.25, rows, [1.85, 1.6, 0.6, 1.6, 0.6], size=10.5, row_h=0.55)
    rows = [["Accuracy", "Plain", "Agent", "Skills", "Oracle"]]
    rows += [[f"gpt-oss-{SHORT[m]}"] + [pc(SA[(m, c)]["acc"]) if (m, c) in SA else "-" for c in C4] for m in MODELS]
    table(s, 6.5, 4.5, 6.25, rows, [1.65, 1.15, 1.15, 1.15, 1.15], size=12, row_h=0.45)
    for m in MODELS:
        k = SHORT[m]
        sbctx.update({f"sb_{k}_{c}": pc(SA[(m, c)]["acc"]) for c in C4 if (m, c) in SA})
        sbctx.update({f"sbd_{k}_{a}_{b}": pts(SPm[(m, a, b)]["diff"]) for a, b in SPAIRS if (m, a, b) in SPm})
        sbctx.update({f"sbp_{k}_{a}_{b}": pval(SPm[(m, a, b)]["p"]) for a, b in SPAIRS if (m, a, b) in SPm})

    s = new_slide("Results: StatABench by task", "Accuracy per task: agent vs agent + skills vs oracle skill")
    BT = SS["by_task"]
    tasks = sorted({r["task"] for r in BT}, key=lambda t: -next(r["n"] for r in BT if r["task"] == t))
    for i, m in enumerate(MODELS):
        vals = {CLAB[c]: [next((r["acc"] for r in BT if r["model"] == m and r["task"] == t and r["condition"] == c), 0)
                          for t in tasks] for c in ["agent", "agent_skills", "agent_skills_oracle"]}
        c_ = bar_chart(s, 0.6 + i * 6.2, 1.85, 6.0, 5.0, [tname(t) for t in tasks], vals,
                       kind=XL_CHART_TYPE.BAR_CLUSTERED, max_v=1.0, labels=False, size=9)
        c_.category_axis.reverse_order = True
        text(s, 0.6 + i * 6.2, 1.45, 6.0, 0.4, f"gpt-oss-{SHORT[m]}", size=15, bold=True, color=BLACK)
    for m in MODELS:
        gain = {t: (next((r["acc"] for r in BT if r["model"] == m and r["task"] == t and r["condition"] == "agent_skills"), 0)
                    - next((r["acc"] for r in BT if r["model"] == m and r["task"] == t and r["condition"] == "agent"), 0))
                for t in tasks}
        best = sorted(gain, key=gain.get, reverse=True)[:3]
        sbctx[f"sb_best_{SHORT[m]}"] = ", ".join(f"{tname(t).lower()} ({pts(gain[t])} pts)" for t in best)
    SEL = {r["model"]: r for r in SS["selection"]}
    for m in MODELS:
        if m in SEL:
            sbctx[f"sb_sel_{SHORT[m]}"] = pc(SEL[m]["loaded"], 0)
            sbctx[f"sb_hit_{SHORT[m]}"] = pc(SEL[m]["hit"], 0)
    sbctx["sb_runs"] = runs_sb

# ----------------------------------------------------------------------------- RQ3: validator
vctx = {}
VAL = []
for bname, fpath in [("SkillBench", V2 / "stats.json"), ("StatABench", SB_STATS)]:
    if not fpath.exists():
        continue
    S_ = json.loads(fpath.read_text(encoding="utf-8"))
    acc = {(r["model"], r["condition"]): r["acc"] for r in S_["accuracy"]
           if r.get("set", "Accuracy on real data (headline)") == "Accuracy on real data (headline)"}
    prd = {(r["model"], r["a"], r["b"]): r for r in S_["paired"]}
    for r in S_.get("validator", []):
        pr = prd.get((r["model"], "agent_skills", "agent_skills_validated"))
        VAL.append({"bench": bname, **r, "acc_s": acc.get((r["model"], "agent_skills")),
                    "acc_v": acc.get((r["model"], "agent_skills_validated")), "pair": pr})
if VAL:
    s = new_slide("Validation loop (RQ3)", "Same first answers as agent + skills; a validator checks them, and flagged "
                  "answers are re-examined (up to 4 more code runs)")
    rows = [["Benchmark", "Model", "Skills", "+ validator", "Difference (p)", "Wrong flagged", "Correct flagged",
             "Fixed", "Broken"]]
    for r in VAL:
        k = r["model"].split("-")[-1]
        pr = r["pair"]
        ptxt = ("< 0.001" if pr["p"] < 0.001 else f"{pr['p']:.3f}") if pr else ""
        rows.append([r["bench"], k, f"{100 * r['acc_s']:.1f}%" if r["acc_s"] is not None else "-",
                     f"{100 * r['acc_v']:.1f}%" if r["acc_v"] is not None else "-",
                     f"{100 * pr['diff']:+.1f} ({ptxt})" if pr else "-",
                     f"{100 * r['detection']:.0f}%" if r["detection"] is not None else "-",
                     f"{100 * r['false_alarm']:.0f}%" if r["false_alarm"] is not None else "-",
                     str(r["fixed"]), str(r["broken"])])
        tag = f"{r['bench'][:2].lower()}_{k}"
        vctx.update({f"v_{tag}_det": f"{100 * (r['detection'] or 0):.0f}%",
                     f"v_{tag}_fa": f"{100 * (r['false_alarm'] or 0):.0f}%",
                     f"v_{tag}_fixed": r["fixed"], f"v_{tag}_broken": r["broken"],
                     f"v_{tag}_diff": f"{100 * pr['diff']:+.1f}" if pr else "n/a",
                     f"v_{tag}_p": (f"{pr['p']:.3f}" if pr else "n/a")})
    table(s, 0.6, 1.55, 12.1, rows, [1.6, 0.8, 1.1, 1.3, 1.8, 1.5, 1.6, 1.2, 1.2], size=12.5, row_h=0.52)
    bullets(s, 0.6, 4.4, 12.1, 2.4, [
        "Validator = deterministic checks (code ran; submitted number appears in the code output) + a checklist "
        "critic (same model, temperature 0) covering grain, population, missing codes, joins, method, measurability",
        "'Wrong flagged' = detection rate on wrong first answers; 'Correct flagged' = false-alarm rate",
        "Fixed / broken = answers changed from wrong to right / right to wrong by the re-examination",
        "The checklist is generic and was not tuned on benchmark answers"], size=13.5, space=8)

# reference-answer validation (always shown; parts fill in as checks complete)
anchors = (BENCH / "results" / "anchors.md")
anc = anchors.read_text(encoding="utf-8").strip().splitlines()[-1] if anchors.exists() else "pending"
dis = V2 / "disagreements.md"
dis_line = "After scoring"
if dis.exists():
    _rows = [l for l in dis.read_text(encoding="utf-8").splitlines() if l.startswith("| ") and not l.startswith("| ID")]
    _n_trap = sum("reference correct (trap)" in l for l in _rows)
    _n_ok = sum("| reference correct:" in l for l in _rows)
    _n_amb = sum("ambiguous" in l for l in _rows)
    dis_line = (f"{len(_rows)} of 206 flagged: {_n_trap} trap working as designed, {_n_ok} consensus used a "
                f"wrong method, {_n_amb} ambiguous (excluded)")
rev = BENCH / "results" / "reference_review.md"
rev_line = None
if rev.exists():
    _rl = rev.read_text(encoding="utf-8").splitlines()
    _after = next((l for l in _rl if l.startswith("After resolution")), None)
    _agree = next((l for l in _rl if l.startswith("- Agree")), "").replace("- Agree: ", "")
    if _after:
        rev_line = (f"{_after.split(':', 1)[1].split(';')[0].strip()} ({_agree} agree; 1 unsure, "
                    "resolved by checking the data build)")
s = new_slide("Reference-answer validation", "Four independent checks that the answer key is right")
rows = [["Check", "What it rules out", "Result"],
        ["Independent code", "Key rewarding the tool library", "No statkit import in any reference script"],
        ["Publisher counts", "Wrong download or parsing", anc],
        ["Consensus audit", "Wrong reference that most runs avoid", dis_line],
        ["Human review (sample of 30)", "Misread question, wrong method", rev_line or "Sheet ready; review pending"]]
table(s, 0.6, 1.55, 12.1, rows, [2.8, 3.9, 5.4], size=13.5, row_h=0.62)
bullets(s, 0.6, 4.95, 12.1, 1.9, [
    "Reference answers computed with pandas / scipy / statsmodels only, never with the library under test",
    "Models under test (gpt-oss), answer-key author (Claude) and judge (qwen) are three different model families",
    "Every flagged question is checked by hand; corrections apply to all conditions equally"], size=14, space=8)

# ----------------------------------------------------------------------------- inference and status (final numbers)
ST_ = json.loads(STATS_FILE.read_text(encoding="utf-8"))
SS_ = json.loads(SB_STATS.read_text(encoding="utf-8"))
PV = {(r["model"].split("-")[-1], r["a"], r["b"]): r for r in ST_["paired"]}
PS = {(r["model"].split("-")[-1], r["a"], r["b"]): r for r in SS_["paired"]}
AV = {(r["model"].split("-")[-1], r["condition"]): r["acc"] for r in ST_["accuracy"]
      if r["set"] == "Accuracy on real data (headline)"}
SELV = {r["model"].split("-")[-1]: r for r in ST_["selection"]}
SELS = {r["model"].split("-")[-1]: r for r in SS_["selection"]}
BDV = {(r["by"], r["model"].split("-")[-1], r["level"], r["condition"]): r["acc"] for r in ST_["breakdown"]}
QV = {(r["model"].split("-")[-1], r["condition"]): r for r in ST_["quality"]}


def dp(r):
    p = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
    return f"{100 * r['diff']:+.1f} pts, {p}"


icx = {
    "i_sk20": dp(PV[("20b", "agent", "agent_skills")]), "i_sb20": dp(PS[("20b", "agent", "agent_skills")]),
    "i_sbp20": dp(PS[("20b", "plain", "agent_skills")]),
    "i_sk120": f"{100 * PV[('120b', 'agent', 'agent_skills')]['diff']:+.1f}",
    "i_sb120": f"{100 * PS[('120b', 'agent', 'agent_skills')]['diff']:+.1f}",
    "i_v20": dp(PV[("20b", "agent_skills", "agent_skills_validated")]),
    "i_vs20": dp(PS[("20b", "agent_skills", "agent_skills_validated")]),
    "i_v120": f"{100 * PV[('120b', 'agent_skills', 'agent_skills_validated')]['diff']:+.1f}",
    "i_vs120": f"{100 * PS[('120b', 'agent_skills', 'agent_skills_validated')]['diff']:+.1f}",
    "i_selv120": f"{100 * SELV['120b']['loaded']:.0f}%", "i_selv20": f"{100 * SELV['20b']['loaded']:.0f}%",
    "i_sels120": f"{100 * SELS['120b']['loaded']:.0f}%", "i_sels20": f"{100 * SELS['20b']['loaded']:.0f}%",
    "i_or20": f"{100 * PV[('20b', 'agent_skills', 'agent_skills_oracle')]['diff']:+.1f}",
    "i_ors20": f"{100 * PS[('20b', 'agent_skills', 'agent_skills_oracle')]['diff']:+.1f}",
    "i_mt20": " → ".join(f"{100 * BDV[('multi_table', '20b', 'True', c)]:.0f}%"
                         for c in ["agent", "agent_skills", "agent_skills_validated"]),
    "i_mt120": "–".join(f"{100 * x:.0f}%" for x in sorted(
        [min(BDV[("multi_table", "120b", "True", c)] for c in ["agent", "agent_skills", "agent_skills_validated"]),
         max(BDV[("multi_table", "120b", "True", c)] for c in ["agent", "agent_skills", "agent_skills_validated"])])),
    "i_cav_plain": f"{100 * QV[('120b', 'plain')]['caveats']:.0f}%",
    "i_cav_skills": f"{100 * QV[('120b', 'agent_skills')]['caveats']:.0f}%",
    "i_acc20s": f"{100 * AV[('20b', 'agent_skills')]:.1f}%", "i_acc20v": f"{100 * AV[('20b', 'agent_skills_validated')]:.1f}%",
}

s = new_slide("Inference", "Final v2 results: SkillBench (181 real-data questions) and StatABench (198 items)")
rows = [["Research question", "Evidence", "Confidence"],
        ["RQ1: do skills improve correctness?",
         f"Small model: yes on SkillBench ({icx['i_sk20']}); on StatABench {icx['i_sb20']} over the agent, "
         f"{icx['i_sbp20']} over plain. Large model: no change ({icx['i_sk120']} and {icx['i_sb120']} pts)",
         "Moderate"],
        ["RQ2: does the agent select skills?",
         f"Rarely: skill loaded for {icx['i_selv120']} / {icx['i_sels120']} of questions (120b, SkillBench / "
         f"StatABench), {icx['i_selv20']} / {icx['i_sels20']} (20b). Handing over the right skill adds no "
         f"significant gain ({icx['i_or20']} / {icx['i_ors20']} pts, 20b)", "Moderate"],
        ["RQ3: does validation catch errors?",
         f"Small model: yes on both ({icx['i_v20']}; {icx['i_vs20']}). Large model: neutral "
         f"({icx['i_v120']} / {icx['i_vs120']} pts); it breaks about as many answers as it fixes", "Moderate"],
        ["Multi-table analysis",
         f"20b on 100 multi-table questions: {icx['i_mt20']} (agent → skills → validator); "
         f"120b {icx['i_mt120']} in every agent condition", "Moderate"]]
table(s, 0.6, 1.5, 12.1, rows, [3.0, 7.6, 1.5], size=12.5, row_h=0.82)
bullets(s, 0.6, 5.75, 12.1, 1.3, [
    "Skills and validation help where the model is weak, not where it is already near its ceiling",
    "Traps that remain: unrecorded values, Simpson's paradox and censoring are missed by most runs of every condition",
    f"Agents state fewer caveats than one-shot answers (judge, 120b: {icx['i_cav_plain']} plain vs "
    f"{icx['i_cav_skills']} with skills)"], size=13.5, space=6)

s = new_slide("Status and next steps")
rows = [["Item", "Status"],
        [f"SkillBench v2 ({ST_['meta']['answers']:,} answers)", "Complete"],
        ["StatABench, unchanged (5,148 answers)", "Complete"],
        ["Validation loop (RQ3)", "Complete"],
        ["Answer key: four checks", "Complete; second reviewer needed"],
        ["Ablation: skill text vs library vs prompt", "Next"],
        ["Different-family critic; other model families", "Next"]]
table(s, 0.6, 1.5, 7.0, rows, [4.3, 2.7], size=14, row_h=0.7)
bullets(s, 8.0, 1.6, 4.75, 5.0, [
    "Explain why the 20b model gains from skills it rarely loads: separate the skill catalog, the library and "
    "the skill text",
    "Test a validator critic from another model family",
    "New skills for unrecorded values, censoring and Simpson's paradox",
    "Require caveats in the skill report format",
    "Pin the inference provider; add DataSciBench and a second model family"], size=14, space=12)

# speaker notes, and the same text as a phone-readable script
from speaker_notes import NOTES  # noqa: E402

class _Ctx(dict):
    def __missing__(self, k):
        return f"[{k}?]"  # visible gap instead of a crash if a number is missing


NOTE_BY_TITLE = dict(NOTES)
missing = [t for t in TITLES if t not in NOTE_BY_TITLE]
assert not missing, f"no speaker notes for: {missing}"
ctx = _Ctx({**progress, **v2ctx, **sbctx, **vctx, **icx})
md = ["# Speaker Script: Skill Library and SkillBench", "",
      f"{len(TITLES)} slides. Notes are also in each slide's speaker notes.", ""]
for i, (slide, title) in enumerate(zip(prs.slides, TITLES), 1):
    note = NOTE_BY_TITLE[title]
    note = (note[RUN_DONE] if isinstance(note, tuple) else note).format_map(ctx)
    slide.notes_slide.notes_text_frame.text = note
    md += ["---", "", f"## Slide {i} · {title}", "", note, ""]
(FYP / "docs" / "slides" / "Speaker_Script.md").write_text("\n".join(md), encoding="utf-8")

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
