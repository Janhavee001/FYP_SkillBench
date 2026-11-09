# Demo script: statistical analysis skills in the Claude app

Eight scenes, about 15 minutes. Every expected number below was produced by running the same tools
on the same files, so a skill that follows its procedure should give these values. Claude may choose
a slightly different method or wording; the numbers match whenever the method matches.

## Before the demo

- [ ] Code execution is on, and all eight skills are uploaded and enabled.
- [ ] The files in this folder are ready to drag in: `titanic.csv`, `winequality-red.csv`, `bike_demand_daily.csv`,
      and the five CSVs in `shop/`.
- [ ] A dry run of the whole script has been done in the app, and no "missing package" warning appeared.
- [ ] Each scene starts in a new chat with the file attached, unless it says "same chat".

`bike_demand_daily.csv` is **synthetic**, made by `make_bike_demand.py`, which plants a growth trend, a
weekly cycle and weather effects. Say so when you show it.

---

## 1. Profile a new file (titanic.csv)

**Ask:** "Profile this dataset."

**Expect (data-profiling skill):**
- 891 rows × 15 columns; delimiter detected as comma.
- `deck` 77.2% missing, flagged structurally missing (unusable). `age` is 19.9% missing, and `embarked` and `embark_town` 0.2% each.
- 342 of 891 survived (38.4%).

**Point out:** the file is profiled before anything is computed, and missing values come with denominators.

## 2. A two-group comparison (same chat)

**Ask:** "Did survivors pay more for their tickets?"

**Expect (group-comparison):**
- The method check passes (`ab_test` on survived × fare: appropriate).
- Survivors' mean fare was £48.40 vs £22.12 for non-survivors: a difference of **£26.28 (95% CI £18.73 to £33.83)**, Cohen's d = 0.55 (**medium**), p = 2.7 × 10⁻¹¹; n = 342 vs 549.
- Fare is strongly skewed (skew 4.79), so a Mann-Whitney test should be run as a check: p = 4.6 × 10⁻²².

**Point out:** the answer leads with the difference and its interval, not just "significant".

## 3. The method check catches a wrong test (same chat)

**Ask:** "Run a t-test of fare across the three passenger classes."

**Expect:**
- The method check returns **inappropriate**: "pclass has 3 groups; this test compares exactly two."
- It switches to Kruskal-Wallis: p = 7.6 × 10⁻⁹⁶, effect size η²H = 0.49 (**large**).
- Median fares: 1st class £60.29, 2nd class £14.25, 3rd class £8.05.

**Point out:** this is the grounding/validation idea from the benchmark. The plan is checked before it runs.

## 4. Two categorical variables (same chat)

**Ask:** "Is survival related to sex?"

**Expect:**
- 74.2% of women survived vs 18.9% of men; chi-square p = 1.2 × 10⁻⁵⁸; Cramér's V = 0.54 (**large**).
- Optional follow-up, "What's the probability women had a higher survival rate?": the Bayesian estimate gives a difference of 0.55 (95% interval 0.49 to 0.61), with P(women > men) ≈ 1.00.
- Wording should be association, not cause.

## 5. Relationships and a chart (winequality-red.csv, new chat)

**Ask:** "Which variables are most related to quality? Show a chart."

**Expect (correlation-analysis and visualization):**
- Spearman, because quality is a 0 to 10 rating. Alcohol **0.48**, volatile acidity **−0.38**, sulphates **0.38**.
- A correlation heatmap, explained by its finding. The strongest pair overall is fixed acidity and pH (r = −0.68), and 6 of 66 pairs have |r| ≥ 0.5.

**Point out:** the explanation states what the data shows, not the chart's colours. That was the gap the benchmark found in LIDA.

## 6. A regression with an honest caveat (same chat)

**Ask:** "Model quality from alcohol, volatile acidity and sulphates."

**Expect (regression-analysis):**
- The method check says **questionable**: quality has only 6 levels, so ordinary regression treats the gaps between levels as equal.
- R² = 0.336, n = 1,599.
- Alcohol +0.31 per % (95% CI 0.28 to 0.34). Volatile acidity −1.22 (−1.41 to −1.03). Sulphates +0.68 (0.48 to 0.88).
- VIF 1.04 to 1.11, so collinearity is not a problem. Effects are described as associations.

## 7. Trend, seasonality, forecast (bike_demand_daily.csv, new chat)

**Ask:** "How has demand changed over time? Is there a weekly pattern? Forecast the next two weeks."

**Expect (time-series-analysis):**
- A line chart with an upward trend of about +2.8 rentals per day (the planted trend is 2.5). The peak is on 2024-07-18.
- The series is non-stationary (stationarity test).
- STL with period 7: seasonal strength 0.42, which beats shuffled data, so the weekly cycle is real. Trend strength is 0.79.
- Weekday mean 3,997 vs weekend 3,173 (the planted gap is about 850). STL's within-week amplitude comes out larger (about 1,400) because it also absorbs some weather noise. If it's quoted, say so.
- ARIMA (1,1,3) with a weekly seasonal term; a 14-day forecast with 95% intervals (first day about 4,019, interval 2,652 to 5,386).
- **Residual check warns the model missed structure** (Ljung-Box p < 0.001). That is correct: weather drives demand, and a single-series forecast cannot see it.

**Point out:** the tool flags its own limitation instead of presenting a confident forecast.

## 8. Several related tables (all five files in shop/, new chat)

`shop/` is **synthetic**, made by `make_shop_tables.py`, with traps planted on purpose. Say so.

**Ask:** "These are tables from an online shop. Are express orders delivered on time more often than standard ones?"

**Expect (relational-data-analysis, then group-comparison):**
- Five tables and five links found. Orders link to customers; order_items link orders to products (many-to-many); reviews link customers to products. `reviewer_id` is recognised as a customer key despite its name.
- **Orphans flagged:** 48 of 1,200 orders (4%) belong to customers that do not exist.
- Joined at one row per order (1,200 rows, no fan-out).
- **Simpson's paradox:** overall, express orders are on time **17 points less** often (−0.173), but **within each region about 6 points more** (+0.062). Region lives in the customers table, so this only shows up after the join. Express is mostly used in the South, where deliveries are slower.

**Follow-up:** "What is the average order value?" The correct answer is **£148.24** over 1,200 orders. A naive join with order_items gives £182.38, because the rows fan out (×2.5). The skill should flag the fan-out and avoid it.

**Point out:** both wrong answers would look perfectly plausible. Only the checks reveal them.

---

## Before/after comparison (RQ1 in miniature)

Repeat scenes **2, 3, 6 and 8** in a new chat with the skills **turned off**: same file, same question. Compare:

| Check | With skills | Without |
| --- | --- | --- |
| Effect size and interval reported | | |
| Test checked against the data (3 groups, skew, ordinal outcome) | | |
| Assumptions or warnings mentioned | | |
| Causal wording avoided | | |

Fill it in live or during the dry run. If the plain model already does well here, say so: the benchmark
found these easy questions near saturation, which is why the harder benchmark comes next.

## If something goes wrong

- **A skill is not picked up:** say "Use the group-comparison skill." Mention that you prompted it.
- **A missing-package warning:** the skill names the tools affected. Continue with the scenes that do not need them.
- **The numbers differ:** check which method Claude chose. A different test gives different numbers, which is itself worth discussing.
