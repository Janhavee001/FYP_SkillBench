"""
Build the SkillBench question bank with COMPUTED ground truth.

Every reference answer is computed here, directly with pandas / numpy / scipy / statsmodels. Nothing
in this file imports `statkit` (the library the skills use), so the benchmark cannot be biased toward
the system under test. Each question records:

  id, dataset, files, question, category, difficulty (easy|medium|hard), trap, answer_type,
  reference (number | label | choice), tolerance, expected_method, rationale (why this is the answer),
  naive (the answer a careless analyst gets by falling into the trap, where one exists).

answer_type:
  number  numeric; correct if |value - reference| <= abs tolerance, or within rel tolerance.
          pct=True marks a percentage: a proportion (reference / 100) is also accepted, flagged as a unit slip.
  label   one category; `aliases` lists accepted spellings.
  choice  one of `choices` (e.g. yes / no / cannot_determine).

    python build_questions.py   ->   questions/questions.json  +  questions/questions.csv (for reading)
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "questions"

QUESTIONS: list[dict] = []


def q(id, dataset, question, category, difficulty, trap, answer_type, reference, *, tol=None, pct=False,
      aliases=None, choices=None, method="", rationale="", naive=None, files=None):
    assert difficulty in ("easy", "medium", "hard")
    if answer_type == "number":
        reference = float(reference)
        if naive is not None:
            naive = float(naive)
    QUESTIONS.append(dict(id=id, dataset=dataset, question=question, category=category, difficulty=difficulty,
                          trap=trap, answer_type=answer_type, reference=reference,
                          tolerance=tol or {}, pct=pct, aliases=aliases or [], choices=choices or [],
                          expected_method=method, rationale=rationale, naive=naive, files=files))


def r(x, nd=4):
    return round(float(x), nd)


# =========================================================================== diabetes
def diabetes():
    d = pd.read_csv(DATA / "diabetes" / "diabetic_data.csv", na_values=[], keep_default_na=False)
    ds = "diabetes"
    n = len(d)
    lt30 = (d.readmitted == "<30")

    q("D01", ds, "How many distinct patients are in this dataset?", "descriptive", "easy", "unit_of_analysis",
      "number", d.patient_nbr.nunique(), tol={"abs": 0}, method="count distinct patient_nbr",
      rationale="Rows are hospital encounters; patients are identified by patient_nbr.", naive=n)

    q("D02", ds, "What percentage of encounters were followed by a readmission within 30 days?", "descriptive",
      "easy", "none", "number", 100 * lt30.mean(), tol={"abs": 0.05}, pct=True,
      method="proportion of readmitted == '<30'", rationale="readmitted has values NO, >30, <30.")

    q("D03", ds, "What percentage of encounters have the patient's weight recorded?", "data_quality", "medium",
      "missing_code", "number", 100 * (d.weight != "?").mean(), tol={"abs": 0.05}, pct=True,
      method="count non-'?' weight", rationale="Missing weight is coded as '?', not as an empty cell.",
      naive=100.0)

    q("D04", ds, "What percentage of encounters had an HbA1c (A1C) test result?", "data_quality", "medium",
      "missing_code", "number", 100 * d.A1Cresult.isin([">7", ">8", "Norm"]).mean(), tol={"abs": 0.05}, pct=True,
      method="share of A1Cresult in {>7, >8, Norm}",
      rationale="A1Cresult 'None' means the test was not done (pandas reads the string 'None' as missing by default).")

    rec = d[d.race != "?"]
    rates = rec.groupby("race").apply(lambda g: (g.readmitted == "<30").mean(), include_groups=False)
    q("D05", ds, "Among encounters with race recorded, which race category has the highest 30-day readmission rate?",
      "group_comparison", "medium", "none", "label", rates.idxmax(), aliases=[rates.idxmax().lower()],
      method="readmission rate by race excluding '?'",
      rationale=f"Rates: {rates.round(4).to_dict()}; '?' is the missing code.",
      naive=d.groupby("race").apply(lambda g: (g.readmitted == "<30").mean(), include_groups=False).idxmax())

    diff = 100 * (lt30[d.diabetesMed == "Yes"].mean() - lt30[d.diabetesMed == "No"].mean())
    q("D06", ds, "How much higher (in percentage points) is the 30-day readmission rate for encounters where "
      "diabetes medication was prescribed (diabetesMed = Yes) than where it was not?", "group_comparison", "easy",
      "none", "number", diff, tol={"abs": 0.05}, method="difference in proportions",
      rationale="Yes-rate minus No-rate.")

    a, b = d.time_in_hospital[lt30], d.time_in_hospital[~lt30]
    q("D07", ds, "Is length of stay (time_in_hospital) different for encounters readmitted within 30 days versus "
      "all other encounters? Report the difference in MEDIAN length of stay in days (readmitted <30 minus others).",
      "group_comparison", "medium", "skew", "number", a.median() - b.median(), tol={"abs": 0},
      method="median difference; Mann-Whitney U (skewed counts)",
      rationale=f"Medians {a.median()} vs {b.median()}; Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.2e}.")

    meds = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
            "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
            "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
    tested, sig_raw, sig_bonf = 0, 0, 0
    pvals = {}
    for m in meds:
        given = d[m] != "No"
        if given.sum() == 0:
            continue
        tab = pd.crosstab(given, lt30)
        if tab.shape != (2, 2):
            continue
        tested += 1
        pvals[m] = stats.chi2_contingency(tab)[1]
    sig_raw = sum(p < 0.05 for p in pvals.values())
    sig_bonf = sum(p < 0.05 / len(pvals) for p in pvals.values())
    q("D08", ds, "For each of the 23 diabetes medication columns that at least one encounter received, test whether "
      "receiving it (any value other than 'No') is associated with 30-day readmission using a chi-square test. "
      "After a Bonferroni correction for the number of medications tested, for how many medications is the "
      "association significant at the 5% level?", "group_comparison", "hard", "multiple_testing", "number",
      sig_bonf, tol={"abs": 0}, method="chi-square per medication + Bonferroni",
      rationale=f"{len(pvals)} medications tested; {sig_raw} significant before correction, {sig_bonf} after.",
      naive=sig_raw)

    q("D09", ds, "What percentage of patients (not encounters) appear in more than one encounter?", "descriptive",
      "medium", "unit_of_analysis", "number", 100 * (d.patient_nbr.value_counts() > 1).mean(), tol={"abs": 0.05},
      pct=True, method="count encounters per patient_nbr")

    y = lt30.astype(int)
    X = sm.add_constant(d[["number_inpatient", "number_emergency", "time_in_hospital", "num_medications"]].astype(float))
    logit = sm.Logit(y, X).fit(disp=0)
    q("D10", ds, "Fit a logistic regression of 30-day readmission (readmitted = '<30' as 1, else 0) on "
      "number_inpatient, number_emergency, time_in_hospital and num_medications. What is the odds ratio for one "
      "additional prior inpatient visit (number_inpatient)?", "regression", "hard", "none", "number",
      np.exp(logit.params["number_inpatient"]), tol={"abs": 0.01}, method="logistic regression, exp(coef)",
      rationale="Odds ratio = exp(coefficient).", naive=logit.params["number_inpatient"])

    mapping = (DATA / "diabetes" / "IDS_mapping.csv").read_text().split("\n,")[1]
    expired_ids = [int(m.group(1)) for m in re.finditer(r"^(\d+),\"?Expired", mapping, re.M)]
    died = d.discharge_disposition_id.isin(expired_ids)
    q("D11", ds, "Using the discharge descriptions in IDS_mapping.csv, what percentage of encounters ended with the "
      "patient's death?", "data_quality", "hard", "multi_section_file", "number", 100 * died.mean(), tol={"abs": 0.02},
      pct=True, method="parse the discharge_disposition section; ids whose description starts with 'Expired'",
      rationale=f"Expired ids {expired_ids} (IDS_mapping.csv holds three stacked tables).")

    alive = d[~died]
    q("D12", ds, "Excluding encounters that ended in the patient's death (they cannot be readmitted), what percentage "
      "of encounters were readmitted within 30 days?", "descriptive", "hard", "population_definition", "number",
      100 * (alive.readmitted == "<30").mean(), tol={"abs": 0.02}, pct=True,
      method="exclude Expired discharge ids, then proportion <30", naive=100 * lt30.mean())

    age_mid = d.age.str.extract(r"\[(\d+)-(\d+)\)").astype(float).mean(axis=1)
    first = d.sort_values("encounter_id").drop_duplicates("patient_nbr")
    rho = stats.spearmanr(first.age.str.extract(r"\[(\d+)")[0].astype(int), first.number_inpatient)[0]
    q("D13", ds, "Using one encounter per patient (each patient's earliest encounter_id), what is the Spearman "
      "correlation between age group and number_inpatient?", "correlation", "hard", "unit_of_analysis", "number",
      rho, tol={"abs": 0.01}, method="dedupe to first encounter; Spearman on ordered age bands",
      naive=stats.spearmanr(age_mid, d.number_inpatient)[0])

    q("D14", ds, "How many encounters have an unknown or invalid gender?", "data_quality", "easy", "none", "number",
      (~d.gender.isin(["Male", "Female"])).sum(), tol={"abs": 0}, method="value counts of gender")


# =========================================================================== online retail
def retail():
    df = pd.read_csv(DATA / "online_retail" / "online_retail_ii.csv", dtype={"Invoice": str, "StockCode": str})
    ds = "online_retail"
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    cancelled = df.Invoice.str.startswith("C")
    sales = df[~cancelled & (df.Quantity > 0) & (df.Price > 0)].copy()
    sales["value"] = sales.Quantity * sales.Price

    q("R01", ds, "What percentage of rows have no Customer ID?", "data_quality", "easy", "none", "number",
      100 * df["Customer ID"].isna().mean(), tol={"abs": 0.05}, pct=True, method="share of missing Customer ID")

    q("R02", ds, "What percentage of invoices are cancellations? (Cancelled invoice numbers start with 'C'.)",
      "data_quality", "easy", "none", "number", 100 * df.loc[cancelled, "Invoice"].nunique() / df.Invoice.nunique(),
      tol={"abs": 0.05}, pct=True, method="distinct invoices starting with C / all distinct invoices",
      naive=100 * cancelled.mean())

    monthly = sales.groupby(sales.InvoiceDate.dt.to_period("M"))["value"].sum()
    q("R03", ds, "Which calendar month (YYYY-MM) had the highest sales value (Quantity x Price), counting only "
      "genuine sales (no cancellations, positive quantity and price)?", "time_series", "medium", "cancellations",
      "label", str(monthly.idxmax()), aliases=[str(monthly.idxmax())], method="monthly sum of line values")

    inv = sales.groupby("Invoice").agg(value=("value", "sum"), country=("Country", "first"))
    q("R04", ds, "What is the median value of an invoice (sum of Quantity x Price over its lines), counting only "
      "genuine sales (no cancellations, positive quantity and price)?", "descriptive", "medium", "grain",
      "number", inv.value.median(), tol={"rel": 0.005}, method="aggregate lines to invoices, then median",
      rationale="One row is an invoice LINE; the question is about invoices.", naive=sales.value.median())

    uk, non = inv.value[inv.country == "United Kingdom"], inv.value[inv.country != "United Kingdom"]
    q("R05", ds, "Do invoices from customers outside the United Kingdom have a higher typical value than UK invoices? "
      "Report the difference in median invoice value (non-UK minus UK), genuine sales only.", "group_comparison",
      "hard", "grain", "number", non.median() - uk.median(), tol={"rel": 0.01},
      method="invoice-level medians; Mann-Whitney (heavily skewed)",
      rationale=f"Mann-Whitney p = {stats.mannwhitneyu(non, uk).pvalue:.1e}.")

    cust_by_country = df.dropna(subset=["Customer ID"]).groupby("Country")["Customer ID"].nunique()
    cbc = cust_by_country.drop("United Kingdom")
    q("R06", ds, "Which country other than the United Kingdom has the most distinct customers?", "descriptive",
      "easy", "none", "label", cbc.idxmax(), aliases=[cbc.idxmax().lower()], method="distinct Customer ID by country")

    dow = df.drop_duplicates("Invoice").InvoiceDate.dt.day_name().value_counts()
    alldays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    missing_days = [x for x in alldays if x not in dow.index]
    q("R07", ds, "On which day of the week are the fewest invoices issued?", "time_series", "medium",
      "none", "label", missing_days[0] if missing_days else dow.idxmin(),
      aliases=[(missing_days[0] if missing_days else dow.idxmin()).lower()],
      method="count invoices by weekday, including weekdays with zero",
      rationale=f"Counts by weekday: {dow.to_dict()}; absent days: {missing_days}.", naive=dow.idxmin())

    q("R08", ds, "How many rows have a zero or negative unit Price?", "data_quality", "easy", "none", "number",
      (df.Price <= 0).sum(), tol={"abs": 0}, method="count Price <= 0")

    c = sales.dropna(subset=["Customer ID"])
    firsts = c.groupby("Customer ID").InvoiceDate.min()
    cohort = firsts[(firsts >= "2009-12-01") & (firsts < "2010-01-01")].index
    later = c[c["Customer ID"].isin(cohort) & (c.InvoiceDate >= "2010-01-01")]["Customer ID"].nunique()
    q("R09", ds, "Of the customers whose first genuine purchase was in December 2009, what percentage made another "
      "genuine purchase in 2010 or later?", "descriptive", "hard", "cohort_definition", "number",
      100 * later / len(cohort), tol={"abs": 0.1}, pct=True, method="first-purchase cohort, then retention",
      rationale=f"Cohort size {len(cohort)}; retained {later}.")

    dups = df.duplicated().sum()
    q("R10", ds, "If exact duplicate rows were removed, keeping the first occurrence of each, how many rows would be "
      "removed?", "data_quality", "easy", "duplicates", "number",
      dups, tol={"abs": 0}, method="DataFrame.duplicated()")

    daily = sales.groupby(sales.InvoiceDate.dt.date)["value"].sum()
    daily.index = pd.to_datetime(daily.index)
    wk = daily.resample("W").sum()
    q("R11", ds, "Were genuine sales (no cancellations, positive quantity and positive price) higher in the first "
      "half of 2011 than in the first half of 2010? Report the percentage change in total sales value "
      "(Quantity x Price), January-June 2011 versus January-June 2010.",
      "time_series", "medium", "cancellations", "number",
      100 * (daily["2011-01-01":"2011-06-30"].sum() / daily["2010-01-01":"2010-06-30"].sum() - 1), tol={"abs": 0.2},
      method="sum by period, percentage change")

    corr_q = sales.groupby("StockCode").agg(price=("Price", "median"), qty=("Quantity", "sum"))
    corr_q = corr_q[corr_q.qty > 0]
    q("R12", ds, "Across products (StockCode), what is the Spearman correlation between a product's median unit "
      "price and its total quantity sold, using genuine sales only (no cancellations, positive quantity and "
      "positive price)?", "correlation", "medium", "skew", "number",
      stats.spearmanr(corr_q.price, corr_q.qty)[0], tol={"abs": 0.01}, method="aggregate to product, Spearman")
    return wk


# =========================================================================== NYC taxi
def taxi():
    t = pd.read_parquet(DATA / "nyc_taxi" / "yellow_tripdata_2024-01.parquet")
    ds = "nyc_taxi"
    inmonth = (t.tpep_pickup_datetime >= "2024-01-01") & (t.tpep_pickup_datetime < "2024-02-01")

    q("T01", ds, "How many trips started in January 2024?", "data_quality", "easy", "out_of_range_dates", "number",
      inmonth.sum(), tol={"abs": 0}, method="filter pickup datetime to January 2024", naive=len(t))

    q("T02", ds, "Do passengers who pay by credit card tip more than passengers who pay in cash?", "group_comparison",
      "hard", "unrecorded_values", "choice", "cannot_determine", choices=["yes", "no", "cannot_determine"],
      method="recognise that cash tips are not recorded",
      rationale="In the TLC data, tip_amount is only populated for card tips; cash tips are not recorded "
                f"(mean recorded tip: card {t.tip_amount[t.payment_type == 1].mean():.2f}, cash "
                f"{t.tip_amount[t.payment_type == 2].mean():.4f}).", naive="yes")

    pos = t.trip_distance[(t.trip_distance > 0) & inmonth]
    q("T03", ds, "What is the median trip distance in miles among January 2024 trips with a positive distance?",
      "descriptive", "easy", "none", "number", pos.median(), tol={"abs": 0.005}, method="median")

    hrs = t[inmonth].tpep_pickup_datetime.dt.hour.value_counts()
    q("T04", ds, "At which hour of the day (0-23) do the most January 2024 trips begin?", "time_series", "easy",
      "none", "label", str(hrs.idxmax()), aliases=[str(hrs.idxmax()), f"{hrs.idxmax()}:00"], method="count by hour")

    q("T05", ds, "What percentage of trips have a negative total_amount?", "data_quality", "easy", "none", "number",
      100 * (t.total_amount < 0).mean(), tol={"abs": 0.01}, pct=True, method="share total_amount < 0")

    v = t[inmonth & (t.payment_type == 1) & (t.fare_amount >= 3) & (t.fare_amount <= 200) & (t.trip_distance > 0)
          & (t.trip_distance <= 100)]
    slope = np.polyfit(v.trip_distance, v.fare_amount, 1)[0]
    q("T06", ds, "For January 2024 credit-card trips with a fare between $3 and $200 and a distance above 0 and at "
      "most 100 miles, what is the OLS slope of fare_amount on trip_distance (dollars per mile)?", "regression",
      "medium", "none", "number", slope, tol={"abs": 0.01}, method="OLS slope")

    cc = t[inmonth & (t.payment_type == 1) & (t.fare_amount > 0)].copy()
    cc["tip_pct"] = 100 * cc.tip_amount / cc.fare_amount
    jfk, std = cc.tip_pct[cc.RatecodeID == 2], cc.tip_pct[cc.RatecodeID == 1]
    q("T07", ds, "Among January 2024 credit-card trips with a positive fare, what is the difference in median tip "
      "percentage (tip / fare x 100) between JFK flat-rate trips (RatecodeID 2) and standard-rate trips "
      "(RatecodeID 1)? Report JFK minus standard, in percentage points.", "group_comparison", "medium", "skew",
      "number", jfk.median() - std.median(), tol={"abs": 0.05}, method="median difference; Mann-Whitney")

    dow = t[inmonth].groupby(t.tpep_pickup_datetime.dt.day_name()).size() / \
        t[inmonth].groupby(t.tpep_pickup_datetime.dt.day_name()).apply(lambda g: g.tpep_pickup_datetime.dt.date.nunique())
    q("T08", ds, "Which day of the week has the highest AVERAGE number of trips per day in January 2024? "
      "(January 2024 has five Mondays, Tuesdays and Wednesdays but four of each other day.)", "time_series",
      "medium", "unequal_exposure", "label", dow.idxmax(), aliases=[dow.idxmax().lower()],
      method="trips per weekday / number of such days",
      naive=t[inmonth].tpep_pickup_datetime.dt.day_name().value_counts().idxmax())

    dur = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    q("T09", ds, "What percentage of trips have a dropoff time that is not after the pickup time?", "data_quality",
      "medium", "invalid_records", "number", 100 * (dur <= 0).mean(), tol={"abs": 0.005}, pct=True,
      method="share of non-positive durations")

    vv = t[inmonth & (t.trip_distance > 0) & (t.trip_distance <= 50) & (dur > 1) & (dur <= 180)]
    speed = vv.trip_distance / (dur[vv.index] / 60)
    by_hr = speed.groupby(vv.tpep_pickup_datetime.dt.hour).median()
    q("T10", ds, "For January 2024 trips with distance in (0, 50] miles and duration between 1 and 180 minutes, at "
      "which pickup hour (0-23) is the median speed (miles per hour) lowest?", "time_series", "hard",
      "invalid_records", "label", str(by_hr.idxmin()), aliases=[str(by_hr.idxmin())],
      method="filter, compute speed, median by hour")


# =========================================================================== bike sharing
def bike():
    b = pd.read_csv(DATA / "bike_sharing" / "hour.csv")
    ds = "bike_sharing"
    tot = b.groupby("yr").cnt.sum()
    q("B01", ds, "By what percentage did total rentals grow from 2011 to 2012?", "descriptive", "easy", "none",
      "number", 100 * (tot[1] / tot[0] - 1), tol={"abs": 0.05}, pct=True, method="yearly totals")

    wd = b[b.workingday == 1].groupby("hr").cnt.mean()
    q("B02", ds, "On working days, at which hour of the day is the average number of rentals highest?",
      "time_series", "easy", "none", "label", str(wd.idxmax()), aliases=[str(wd.idxmax())], method="mean by hour")

    nwd = b[b.workingday == 0].groupby("hr").cnt.mean()
    q("B03", ds, "On non-working days, at which hour of the day is the average number of rentals highest?",
      "time_series", "easy", "none", "label", str(nwd.idxmax()), aliases=[str(nwd.idxmax())], method="mean by hour")

    q("B04", ds, "What is the Pearson correlation between temperature (temp) and the hourly rental count (cnt)?",
      "correlation", "easy", "none", "number", b.temp.corr(b.cnt), tol={"abs": 0.005}, method="Pearson r")

    resid_h = sm.OLS(b.hum, pd.get_dummies(b.hr, dtype=float)).fit().resid
    resid_c = sm.OLS(b.cnt, pd.get_dummies(b.hr, dtype=float)).fit().resid
    q("B05", ds, "Humidity is lowest in the afternoon, when rentals peak, so the raw humidity-rentals correlation is "
      "partly driven by time of day. What is the partial correlation between humidity (hum) and rentals (cnt) "
      "after controlling for hour of day (as a categorical variable)?", "correlation", "hard", "confounding",
      "number", np.corrcoef(resid_h, resid_c)[0, 1], tol={"abs": 0.01},
      method="residualise both on hour dummies, correlate residuals", naive=b.hum.corr(b.cnt))

    days = pd.date_range("2011-01-01", "2012-12-31").size
    q("B06", ds, "The data are meant to hold one row per hour for 2011-2012. How many hourly records are missing?",
      "data_quality", "medium", "time_gaps", "number", days * 24 - len(b), tol={"abs": 0},
      method="expected hours minus rows")

    share = b.groupby("workingday").apply(lambda g: g.casual.sum() / g.cnt.sum(), include_groups=False)
    q("B07", ds, "What share of rentals are made by casual (unregistered) users on non-working days, minus that share "
      "on working days? Answer in percentage points.", "group_comparison", "medium", "none", "number",
      100 * (share[0] - share[1]), tol={"abs": 0.05}, method="ratio of sums by group",
      naive=100 * ((b.casual / b.cnt)[b.workingday == 0].mean() - (b.casual / b.cnt)[b.workingday == 1].mean()))

    w4 = b[b.weathersit == 4]
    q("B08", ds, "How many hourly observations fall in the most severe weather category (weathersit = 4)?",
      "data_quality", "easy", "small_sample", "number", len(w4), tol={"abs": 0}, method="count",
      rationale="Only a handful of hours: any estimate for this category is unreliable.")

    from statsmodels.tsa.stattools import adfuller, kpss
    daily = b.groupby("dteday").cnt.sum()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p, kpss_p = adfuller(daily, result_object=False)[1], kpss(daily, nlags="auto", result_object=False)[1]
    q("B09", ds, "Is the daily total rental series stationary?", "time_series", "medium", "trend", "choice",
      "no", choices=["yes", "no"], method="ADF + KPSS on daily totals",
      rationale=f"Strong growth trend and yearly cycle; ADF p = {adf_p:.3f}, KPSS p = {kpss_p:.3f}.")

    m = sm.OLS(b.cnt, sm.add_constant(b[["temp", "hum", "windspeed"]].join(pd.get_dummies(b.hr, prefix="h", drop_first=True, dtype=float)))).fit()
    q("B10", ds, "In an OLS regression of hourly rentals (cnt) on temp, hum and windspeed plus hour-of-day dummy "
      "variables, what is the coefficient on temp?", "regression", "hard", "none", "number", m.params["temp"],
      tol={"rel": 0.005}, method="OLS with hour fixed effects")


# =========================================================================== Beijing PM2.5
def pm25():
    p = pd.read_csv(DATA / "beijing_pm25" / "beijing_pm25.csv")
    ds = "beijing_pm25"
    v = p["pm2.5"]
    q("P01", ds, "What percentage of hourly PM2.5 readings are missing?", "data_quality", "easy", "none", "number",
      100 * v.isna().mean(), tol={"abs": 0.01}, pct=True, method="share NaN")

    yr = p.groupby("year")["pm2.5"].mean()
    q("P02", ds, "Which year had the highest mean PM2.5 concentration?", "descriptive", "easy", "none", "label",
      str(yr.idxmax()), aliases=[str(yr.idxmax())], method="mean by year")

    mo = p.groupby("month")["pm2.5"].mean()
    q("P03", ds, "Which calendar month (1-12), pooled over all years, has the highest mean PM2.5?", "time_series",
      "easy", "none", "label", str(mo.idxmax()), aliases=[str(mo.idxmax())], method="mean by month")

    nw, se = p.loc[p.cbwd == "NW", "pm2.5"].dropna(), p.loc[p.cbwd == "SE", "pm2.5"].dropna()
    q("P04", ds, "What is the difference in median PM2.5 between hours with north-west wind (cbwd = NW) and hours with "
      "south-east wind (cbwd = SE)? Report NW minus SE.", "group_comparison", "medium", "skew", "number",
      nw.median() - se.median(), tol={"abs": 0.5}, method="median difference; Mann-Whitney")

    sub = p[["Iws", "pm2.5"]].dropna()
    q("P05", ds, "What is the Spearman correlation between cumulated wind speed (Iws) and PM2.5?", "correlation",
      "medium", "skew", "number", stats.spearmanr(sub.Iws, sub["pm2.5"])[0], tol={"abs": 0.01}, method="Spearman")

    p["date"] = pd.to_datetime(p[["year", "month", "day"]])
    g = p.groupby("date")["pm2.5"].agg(["mean", "count"])
    valid = g[g["count"] >= 18]
    q("P06", ds, "How many days had a daily mean PM2.5 above 75 ug/m3 (China's 24-hour Grade II limit), counting only "
      "days with at least 18 hourly readings?", "descriptive", "hard", "missing_data_rule", "number",
      (valid["mean"] > 75).sum(), tol={"abs": 0}, method="daily mean with completeness rule",
      naive=(g["mean"] > 75).sum())

    s = p.sort_values(["year", "month", "day", "hour"])["pm2.5"].reset_index(drop=True)
    q("P07", ds, "What is the lag-24 (one day) autocorrelation of the hourly PM2.5 series?", "time_series", "medium",
      "missing_values", "number", s.autocorr(24), tol={"abs": 0.01}, method="Series.autocorr(24), pairwise complete")

    rain = p[["Ir", "pm2.5"]].dropna()
    rained, dry = rain.loc[rain.Ir > 0, "pm2.5"], rain.loc[rain.Ir == 0, "pm2.5"]
    q("P08", ds, "Is PM2.5 lower in hours with accumulated rain (Ir > 0) than in dry hours? Report the difference in "
      "median PM2.5 (rain minus dry).", "group_comparison", "medium", "skew", "number", rained.median() - dry.median(),
      tol={"abs": 0.5}, method="median difference; Mann-Whitney")


# =========================================================================== MovieLens
def movielens():
    u = pd.read_csv(DATA / "movielens" / "users.csv")
    m = pd.read_csv(DATA / "movielens" / "movies.csv")
    rt = pd.read_csv(DATA / "movielens" / "ratings.csv")
    ds = "movielens"
    q("M01", ds, "How many ratings are there in total?", "descriptive", "easy", "none", "number", len(rt),
      tol={"abs": 0}, method="count")

    j = rt.merge(u, on="user_id")
    diff = j.loc[j.gender == "F", "rating"].mean() - j.loc[j.gender == "M", "rating"].mean()
    q("M02", ds, "What is the average of all ratings given by female users minus the average of all ratings given by "
      "male users?", "group_comparison", "medium", "join", "number", diff, tol={"abs": 0.002}, method="join users, mean")

    pu = j.groupby(["user_id", "gender"]).rating.mean().reset_index()
    du = pu.loc[pu.gender == "F", "rating"].mean() - pu.loc[pu.gender == "M", "rating"].mean()
    q("M03", ds, "Treating each user as one observation (their own average rating), what is the difference in mean "
      "per-user average rating, female minus male?", "group_comparison", "medium", "unit_of_analysis", "number",
      du, tol={"abs": 0.002}, method="aggregate to users, then compare", naive=diff)

    g = rt.merge(m, on="movie_id")
    g = g.assign(genre=g.genres.str.split("|")).explode("genre")
    gs = g.groupby("genre").rating.agg(["mean", "size"])
    gs = gs[gs["size"] >= 1000]
    q("M04", ds, "Movies can have several genres (separated by '|'). Among genres with at least 1,000 ratings, which "
      "genre has the highest average rating?", "group_comparison", "medium", "multi_valued_field", "label",
      gs["mean"].idxmax(), aliases=[gs["mean"].idxmax().lower()], method="split genres, explode, mean by genre",
      naive=rt.merge(m, on="movie_id").groupby("genres").rating.mean().idxmax())

    q("M05", ds, "How many movies in the movie list have no ratings at all?", "data_quality", "easy", "orphans",
      "number", (~m.movie_id.isin(rt.movie_id)).sum(), tol={"abs": 0}, method="anti-join")

    per_user = rt.groupby("user_id").size().rename("n").reset_index().merge(u, on="user_id")
    occ = per_user.groupby("occupation").n.mean()
    q("M06", ds, "Which occupation code has the highest mean number of ratings per user?", "group_comparison",
      "medium", "grain", "label", str(occ.idxmax()), aliases=[str(occ.idxmax())],
      method="ratings per user, then mean by occupation",
      naive=str(j.groupby("occupation").size().idxmax()))

    ms = rt.groupby("movie_id").rating.agg(["mean", "size"])
    ms20 = ms[ms["size"] >= 20]
    q("M07", ds, "Across movies with at least 20 ratings, what is the Spearman correlation between a movie's number of "
      "ratings and its average rating?", "correlation", "medium", "grain", "number",
      stats.spearmanr(ms20["size"], ms20["mean"])[0], tol={"abs": 0.005}, method="aggregate to movies, Spearman")

    m["year"] = m.title.str.extract(r"\((\d{4})\)\s*$").astype(float)
    m["decade"] = (m.year // 10 * 10).astype("Int64")
    dm = rt.merge(m, on="movie_id")
    pop = ms[ms["size"] >= 50].index
    dec = dm[dm.movie_id.isin(pop)].groupby("decade").rating.mean()
    q("M08", ds, "Taking the release year from the movie title, which release decade has the highest average rating, "
      "using all ratings of movies that have at least 50 ratings? Give the decade as its first year (e.g. 1970).",
      "group_comparison", "hard", "parsing", "label", str(dec.idxmax()), aliases=[str(dec.idxmax()), f"{dec.idxmax()}s"],
      method="parse year, filter movies, mean by decade")

    pu2 = rt.groupby("user_id").rating.mean().rename("avg").reset_index().merge(u, on="user_id")
    q("M09", ds, "Using one value per user (their average rating), what is the Spearman correlation between the age "
      "group code and the user's average rating?", "correlation", "medium", "unit_of_analysis", "number",
      stats.spearmanr(pu2.age, pu2.avg)[0], tol={"abs": 0.005}, method="aggregate to users, Spearman",
      naive=stats.spearmanr(j.age, j.rating)[0])

    ts = pd.to_datetime(rt.timestamp, unit="s")
    q("M10", ds, "In which calendar year were the most ratings made?", "time_series", "easy", "parsing", "label",
      str(ts.dt.year.value_counts().idxmax()), aliases=[str(ts.dt.year.value_counts().idxmax())],
      method="convert Unix timestamp")


# =========================================================================== Olist
def olist():
    D = DATA / "olist"
    o = pd.read_csv(D / "olist_orders_dataset.csv")
    it = pd.read_csv(D / "olist_order_items_dataset.csv")
    pay = pd.read_csv(D / "olist_order_payments_dataset.csv")
    rv = pd.read_csv(D / "olist_order_reviews_dataset.csv")
    cu = pd.read_csv(D / "olist_customers_dataset.csv")
    pr = pd.read_csv(D / "olist_products_dataset.csv")
    tr = pd.read_csv(D / "product_category_name_translation.csv")
    se = pd.read_csv(D / "olist_sellers_dataset.csv")
    ds = "olist"

    q("O01", ds, "How many orders are there?", "descriptive", "easy", "none", "number", o.order_id.nunique(),
      tol={"abs": 0}, method="count orders")

    oc = o.merge(cu, on="customer_id")
    per = oc.groupby("customer_unique_id").order_id.nunique()
    q("O02", ds, "What percentage of customers (people, not customer records) placed more than one order?",
      "descriptive", "medium", "hidden_entity_key", "number", 100 * (per > 1).mean(), tol={"abs": 0.02}, pct=True,
      method="group orders by customer_unique_id",
      rationale="customer_id is unique per order; the same person keeps one customer_unique_id.",
      naive=100 * (oc.groupby("customer_id").order_id.nunique() > 1).mean())

    ptot = pay.groupby("order_id").payment_value.sum()
    q("O03", ds, "What is the average total payment per order (adding up all payments made for the same order), over "
      "orders that have at least one payment?", "descriptive", "medium", "fan_out", "number", ptot.mean(),
      tol={"abs": 0.05}, method="sum payments per order, then mean", naive=pay.payment_value.mean())

    n_items = it.groupby("order_id").size()
    rscore = rv.groupby("order_id").review_score.mean()
    both = pd.DataFrame({"n": n_items, "score": rscore}).dropna()
    q("O04", ds, "Using each order's average review score, what is the mean review score of orders with more than one "
      "item minus that of orders with exactly one item?", "group_comparison", "hard", "fan_out", "number",
      both.score[both.n > 1].mean() - both.score[both.n == 1].mean(), tol={"abs": 0.005},
      method="items per order, review per order, then compare")

    dl = o[o.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"]).copy()
    dl["days"] = (pd.to_datetime(dl.order_delivered_customer_date) - pd.to_datetime(dl.order_purchase_timestamp)).dt.total_seconds() / 86400
    q("O05", ds, "For delivered orders, what is the median time from purchase to delivery to the customer, in days?",
      "descriptive", "easy", "none", "number", dl.days.median(), tol={"abs": 0.05}, method="datetime difference")

    dl["late"] = pd.to_datetime(dl.order_delivered_customer_date) > pd.to_datetime(dl.order_estimated_delivery_date)
    lr = dl.set_index("order_id").join(rscore.rename("score")).dropna(subset=["score"])
    q("O06", ds, "Do late deliveries get worse reviews? For delivered orders, using each order's average review score, "
      "report the mean score of orders delivered after the estimated delivery date minus that of orders delivered on "
      "time or early.", "group_comparison", "medium", "join", "number",
      lr.score[lr.late].mean() - lr.score[~lr.late].mean(), tol={"abs": 0.005}, method="join reviews, compare means")

    ip = it.merge(pr, on="product_id", how="left").merge(tr, on="product_category_name", how="left")
    rev = ip.groupby("product_category_name_english").price.sum()
    q("O07", ds, "Which product category, by its English name, generated the most revenue from item prices "
      "(excluding freight)?", "descriptive", "medium", "multi_hop_join", "label", rev.idxmax(),
      aliases=[rev.idxmax().replace("_", " ")], method="items -> products -> translation, sum price")

    st = it.merge(o[["order_id", "customer_id"]], on="order_id").merge(cu[["customer_id", "customer_state"]], on="customer_id")
    st = st.groupby("customer_state").agg(f=("freight_value", "sum"), p=("price", "sum"))
    ratio = (100 * st.f / st.p)
    q("O08", ds, "Which customer state pays the highest freight as a percentage of item price (total freight / total "
      "item price)?", "group_comparison", "hard", "ratio_of_sums", "label", ratio.idxmax(),
      aliases=[ratio.idxmax().lower()], method="join to customers, ratio of sums by state")

    q("O09", ds, "How many orders have more than one review?", "data_quality", "easy", "duplicates", "number",
      (rv.groupby("order_id").size() > 1).sum(), tol={"abs": 0}, method="count reviews per order")

    same = it.merge(o[["order_id", "customer_id"]], on="order_id").merge(cu[["customer_id", "customer_state"]], on="customer_id") \
        .merge(se[["seller_id", "seller_state"]], on="seller_id")
    q("O10", ds, "What percentage of order items are shipped by a seller located in the same state as the customer?",
      "descriptive", "medium", "multi_hop_join", "number", 100 * (same.customer_state == same.seller_state).mean(),
      tol={"abs": 0.05}, pct=True, method="items -> orders -> customers, items -> sellers")


# =========================================================================== F1DB
def f1():
    D = DATA / "f1db"
    races = pd.read_csv(D / "races.csv", low_memory=False)
    rr = pd.read_csv(D / "races_race_results.csv", low_memory=False)
    drivers = pd.read_csv(D / "drivers.csv", low_memory=False)
    pits = pd.read_csv(D / "races_pit_stops.csv", low_memory=False)
    standings = pd.read_csv(D / "seasons_driver_standings.csv", low_memory=False)
    ds = "f1db"

    q("F01", ds, "How many World Championship races were held in 2023?", "descriptive", "easy", "none", "number",
      (races.year == 2023).sum(), tol={"abs": 0}, method="count races")

    wins = rr[rr.positionNumber == 1].groupby("driverId").size()
    q("F02", ds, "Which driver has won the most races?", "descriptive", "easy", "none", "label", wins.idxmax(),
      aliases=[wins.idxmax().replace("-", " ")], method="count positionNumber == 1")

    dec = rr[(rr.year >= 2010) & (rr.year <= 2019) & (rr.positionNumber == 1)].groupby("constructorId").size()
    q("F03", ds, "Which constructor won the most races from 2010 to 2019?", "descriptive", "easy", "none", "label",
      dec.idxmax(), aliases=[dec.idxmax().replace("-", " ")], method="filter years, count wins")

    since = rr[rr.year >= 2000]
    pole = since[since.gridPositionNumber == 1]
    q("F04", ds, "Since 2000, in what percentage of race results where the driver started from first on the grid did "
      "that driver win the race?", "descriptive", "medium", "none", "number", 100 * (pole.positionNumber == 1).mean(),
      tol={"abs": 0.1}, pct=True, method="filter grid == 1, share winning")

    mod = rr[(rr.year >= 2014) & rr.positionNumber.notna() & rr.gridPositionNumber.notna()]
    q("F05", ds, "From 2014 onwards, among classified finishers who started from a grid position, what is the Spearman "
      "correlation between starting grid position and finishing position?", "correlation", "medium",
      "unclassified_finishers", "number", stats.spearmanr(mod.gridPositionNumber, mod.positionNumber)[0],
      tol={"abs": 0.005}, method="filter classified, Spearman")

    pc = pits[pits.year >= 2012].groupby(["raceId", "driverId"]).size().rename("stops").reset_index()
    pj = pc.merge(rr[["raceId", "driverId", "positionNumber"]], on=["raceId", "driverId"]).dropna(subset=["positionNumber"])
    naive = pits.loc[pits.year >= 2012, ["raceId", "driverId", "stop"]] \
        .merge(rr[["raceId", "driverId", "positionNumber"]], on=["raceId", "driverId"]).dropna(subset=["positionNumber"])
    q("F06", ds, "From 2012 onwards, for each driver in each race with at least one pit stop, count the pit stops. "
      "Among classified finishers, what is the Spearman correlation between the number of pit stops and finishing "
      "position?", "correlation", "hard", "fan_out", "number",
      stats.spearmanr(pj.stops, pj.positionNumber)[0], tol={"abs": 0.005},
      method="aggregate pit stops to driver-race, join results, Spearman",
      naive=stats.spearmanr(naive.stop, naive.positionNumber)[0])

    circ = races.circuitId.value_counts()
    q("F07", ds, "Which circuit has hosted the most World Championship races?", "descriptive", "easy", "none", "label",
      circ.idxmax(), aliases=[circ.idxmax().replace("-", " ")], method="count races by circuit")

    s10 = rr[(rr.year >= 2010) & (rr.year <= 2025)]
    q("F08", ds, "In race results from 2010 to 2025, what percentage are recorded as 'DNF' (did not finish)?",
      "descriptive", "easy", "none", "number", 100 * (s10.positionText == "DNF").mean(), tol={"abs": 0.05}, pct=True,
      method="share positionText == 'DNF'")

    champs = standings[standings.championshipWon == True].drop_duplicates("driverId")  # noqa: E712
    nat = champs.merge(drivers[["id", "nationalityCountryId"]], left_on="driverId", right_on="id").nationalityCountryId.value_counts()
    q("F09", ds, "Which nationality has produced the most different World Drivers' Champions?", "descriptive",
      "medium", "join", "label", nat.idxmax(), aliases=[nat.idxmax().replace("-", " ")],
      method="champions (distinct drivers) joined to nationality",
      naive=standings[standings.championshipWon == True].merge(drivers, left_on="driverId", right_on="id").nationalityCountryId.value_counts().idxmax())  # noqa: E712

    pts = standings.loc[standings.points.idxmax()]
    q("F10", ds, "Which driver scored the most championship points in a single season, and in which year? Give the "
      "driver id.", "descriptive", "medium", "era_effects", "label", pts.driverId,
      aliases=[pts.driverId.replace("-", " ")], method="max of season standings points",
      rationale=f"{pts.driverId} in {pts.year} with {pts.points} points (points systems changed over time).")

    age = rr[rr.positionNumber == 1].merge(races[["id", "date"]], left_on="raceId", right_on="id") \
        .merge(drivers[["id", "dateOfBirth"]], left_on="driverId", right_on="id", suffixes=("", "_d"))
    yrs = (pd.to_datetime(age.date) - pd.to_datetime(age.dateOfBirth)).dt.days / 365.25
    q("F11", ds, "What was the median age, in years, of race winners on the day of their win?", "descriptive",
      "medium", "multi_hop_join", "number", yrs.median(), tol={"abs": 0.05}, method="join results, races, drivers")


# =========================================================================== synthetic shop
def shop():
    D = DATA / "shop"
    c, o, it, p = (pd.read_csv(D / f"{n}.csv") for n in ("customers", "orders", "order_items", "products"))
    ds = "shop"
    q("S01", ds, "How many orders belong to a customer who does not appear in the customer table?", "data_quality",
      "easy", "orphans", "number", (~o.customer_id.isin(c.customer_id)).sum(), tol={"abs": 0}, method="anti-join")

    q("S02", ds, "What is the average order_total per order?", "descriptive", "easy", "fan_out", "number",
      o.order_total.mean(), tol={"abs": 0.01}, method="mean of orders.order_total",
      naive=o.merge(it, on="order_id").order_total.mean())

    j = o.merge(c, on="customer_id")
    pooled = j.loc[j.express == 1, "on_time"].mean() - j.loc[j.express == 0, "on_time"].mean()
    q("S03", ds, "Does express shipping make orders more likely to arrive on time?", "causal", "hard",
      "simpsons_paradox", "choice", "yes", choices=["yes", "no"],
      method="compare within customer region (the confounder)",
      rationale=f"Pooled difference {pooled:+.3f} (express looks worse) but positive within every region; "
                "region (in the customers table) confounds the pooled comparison.", naive="no")

    buyers = c[c.customer_id.isin(o.customer_id)]
    q("S04", ds, "What is the mean age of customers who placed at least one order, counting each customer once?",
      "descriptive", "medium", "unit_of_analysis", "number", buyers.age.mean(), tol={"abs": 0.01},
      method="dedupe customers", naive=j.age.mean())

    cat = it.merge(p, on="product_id").groupby("category").quantity.sum()
    q("S05", ds, "Which product category has the highest total quantity sold?", "descriptive", "easy", "join",
      "label", cat.idxmax(), aliases=[cat.idxmax()], method="join items to products, sum quantity")


# =========================================================================== extra hard, multi-step questions
def extra_hard():
    # --- diabetes: adjusted model with categorical controls
    d = pd.read_csv(DATA / "diabetes" / "diabetic_data.csv", na_values=[], keep_default_na=False)
    d["y"] = (d.readmitted == "<30").astype(int)
    d["a1c_measured"] = d.A1Cresult.isin([">7", ">8", "Norm"]).astype(float)
    X = pd.get_dummies(d[["age"]], drop_first=True, dtype=float)
    X["number_inpatient"] = d.number_inpatient.astype(float)
    X["a1c_measured"] = d.a1c_measured
    fit = sm.Logit(d.y, sm.add_constant(X)).fit(disp=0)
    raw = sm.Logit(d.y, sm.add_constant(d[["a1c_measured"]])).fit(disp=0)
    q("D15", "diabetes", "Is having an HbA1c test result associated with the odds of 30-day readmission once age group "
      "(as categories) and number_inpatient are controlled for? Report the odds ratio for 'HbA1c measured' from a "
      "logistic regression of readmission within 30 days on HbA1c measured, age group dummies and number_inpatient.",
      "regression", "hard", "confounding", "number", np.exp(fit.params["a1c_measured"]), tol={"abs": 0.01},
      method="logistic regression with categorical controls", naive=np.exp(raw.params["a1c_measured"]),
      rationale="A1Cresult 'None' = not measured; must not be dropped as missing.")

    # --- retail: customer-level analysis after cleaning
    df = pd.read_csv(DATA / "online_retail" / "online_retail_ii.csv", dtype={"Invoice": str, "StockCode": str})
    canc = df.Invoice.str.startswith("C")
    sales = df[~canc & (df.Quantity > 0) & (df.Price > 0)].dropna(subset=["Customer ID"]).copy()
    sales["value"] = sales.Quantity * sales.Price
    inv = sales.groupby(["Customer ID", "Invoice"]).value.sum().reset_index()
    cust = inv.groupby("Customer ID").agg(n=("Invoice", "size"), med=("value", "median"))
    cust2 = cust[cust.n >= 2]
    q("R13", "online_retail", "Among customers with at least two genuine invoices (no cancellations, positive quantity "
      "and price), what is the Spearman correlation between a customer's number of invoices and their median invoice "
      "value?", "correlation", "hard", "grain", "number", stats.spearmanr(cust2.n, cust2.med)[0], tol={"abs": 0.01},
      method="lines -> invoices -> customers, Spearman")
    cq = df[canc].assign(q=lambda x: -x.Quantity).groupby("StockCode").q.sum()
    q("R14", "online_retail", "Which StockCode has the largest total cancelled quantity?", "data_quality", "medium",
      "cancellations", "label", str(cq.idxmax()), aliases=[str(cq.idxmax())], method="filter cancellations, sum |qty|")

    # --- taxi
    t = pd.read_parquet(DATA / "nyc_taxi" / "yellow_tripdata_2024-01.parquet",
                        columns=["tpep_pickup_datetime", "payment_type", "fare_amount", "tip_amount", "PULocationID",
                                 "trip_distance", "tpep_dropoff_datetime"])
    inm = (t.tpep_pickup_datetime >= "2024-01-01") & (t.tpep_pickup_datetime < "2024-02-01")
    cc = t[inm & (t.payment_type == 1) & (t.fare_amount > 0)]
    q("T11", "nyc_taxi", "Among January 2024 credit-card trips with a positive fare, what percentage recorded no tip "
      "(tip_amount = 0)?", "descriptive", "medium", "none", "number", 100 * (cc.tip_amount == 0).mean(),
      tol={"abs": 0.02}, pct=True, method="share tip == 0")
    pu = t[inm].PULocationID.value_counts()
    q("T12", "nyc_taxi", "Which pickup zone (PULocationID) had the most January 2024 trips?", "descriptive", "easy",
      "none", "label", str(pu.idxmax()), aliases=[str(pu.idxmax())], method="value counts")
    v = t[inm & (t.trip_distance > 0) & (t.trip_distance <= 50)].copy()
    v["dur"] = (v.tpep_dropoff_datetime - v.tpep_pickup_datetime).dt.total_seconds() / 60
    v = v[(v.dur > 1) & (v.dur <= 180)]
    v["weekend"] = v.tpep_pickup_datetime.dt.dayofweek >= 5
    sp = v.trip_distance / (v.dur / 60)
    q("T13", "nyc_taxi", "For January 2024 trips with distance in (0, 50] miles and duration between 1 and 180 minutes, "
      "how much faster (in mph) is the median speed on weekends than on weekdays?", "group_comparison", "hard",
      "invalid_records", "number", sp[v.weekend].median() - sp[~v.weekend].median(), tol={"abs": 0.05},
      method="filter invalid trips, speed, median by weekend flag")

    # --- bike sharing: detrended autocorrelation and interaction
    b = pd.read_csv(DATA / "bike_sharing" / "hour.csv")
    daily = b.groupby("dteday").cnt.sum().reset_index(drop=True)
    idx = np.arange(len(daily))
    resid = daily - np.polyval(np.polyfit(idx, daily, 1), idx)
    q("B11", "bike_sharing", "Aggregate to daily totals, remove a linear time trend (OLS on the day index), and report "
      "the lag-7 autocorrelation of the detrended daily series.", "time_series", "hard", "trend", "number",
      pd.Series(resid).autocorr(7), tol={"abs": 0.01}, method="daily totals, linear detrend, autocorr(7)",
      naive=daily.autocorr(7))
    Xb = b[["temp", "workingday"]].astype(float).assign(temp_x_workingday=b.temp * b.workingday) \
        .join(pd.get_dummies(b.hr, prefix="h", drop_first=True, dtype=float))
    mi = sm.OLS(b.cnt, sm.add_constant(Xb)).fit()
    q("B12", "bike_sharing", "Does the effect of temperature on hourly rentals differ between working and non-working "
      "days? Fit OLS: cnt ~ temp + workingday + temp x workingday + hour-of-day dummies. Report the coefficient on the "
      "temp x workingday interaction.", "regression", "hard", "interaction", "number",
      mi.params["temp_x_workingday"], tol={"rel": 0.01}, method="OLS with interaction and hour fixed effects")

    # --- PM2.5
    p = pd.read_csv(DATA / "beijing_pm25" / "beijing_pm25.csv")
    win = p[p.month.isin([12, 1, 2])].groupby("hour")["pm2.5"].median()
    q("P09", "beijing_pm25", "In winter months (December, January, February), at which hour of the day is median PM2.5 "
      "lowest?", "time_series", "medium", "none", "label", str(win.idxmin()), aliases=[str(win.idxmin())],
      method="filter months, median by hour")
    pp = p.dropna(subset=["pm2.5"]).copy()
    Xp = pp[["DEWP", "TEMP", "PRES", "Iws"]].join(pd.get_dummies(pp.month, prefix="m", drop_first=True, dtype=float))
    mp = sm.OLS(np.log1p(pp["pm2.5"]), sm.add_constant(Xp.astype(float))).fit()
    q("P10", "beijing_pm25", "Regress log(1 + PM2.5) on dew point (DEWP), temperature (TEMP), pressure (PRES) and "
      "cumulated wind speed (Iws), with month-of-year dummies, using hours with a PM2.5 reading. What is the "
      "coefficient on Iws?", "regression", "hard", "skew", "number", mp.params["Iws"], tol={"abs": 0.0002},
      method="OLS on log-transformed outcome with month fixed effects")

    # --- MovieLens
    rt = pd.read_csv(DATA / "movielens" / "ratings.csv")
    m = pd.read_csv(DATA / "movielens" / "movies.csv")
    per_user = rt.groupby("user_id").size().sort_values(ascending=False)
    top = per_user.iloc[: int(np.ceil(0.1 * len(per_user)))]
    q("M11", "movielens", "What percentage of all ratings were made by the 10% most active users (rounding the number of "
      "users up)?", "descriptive", "medium", "grain", "number", 100 * top.sum() / len(rt), tol={"abs": 0.05},
      pct=True, method="ratings per user, sort, share of top decile")
    ms = rt.groupby("movie_id").rating.agg(["mean", "size"])
    best = ms[ms["size"] >= 1000]["mean"].idxmax()
    title = m.set_index("movie_id").title[best]
    q("M12", "movielens", "Among movies with at least 1,000 ratings, which has the highest average rating? Give its "
      "title.", "descriptive", "medium", "small_sample", "label", title,
      aliases=[title.lower(), re.sub(r"\s*\(\d{4}\)$", "", title).lower()], method="filter by count, max mean",
      naive=m.set_index("movie_id").title[ms["mean"].idxmax()])

    # --- Olist: cross-table consistency and per-state delivery
    D = DATA / "olist"
    o = pd.read_csv(D / "olist_orders_dataset.csv")
    it = pd.read_csv(D / "olist_order_items_dataset.csv")
    pay = pd.read_csv(D / "olist_order_payments_dataset.csv")
    cu = pd.read_csv(D / "olist_customers_dataset.csv")
    ptot = pay.groupby("order_id").payment_value.sum()
    itot = it.groupby("order_id").apply(lambda g: (g.price + g.freight_value).sum(), include_groups=False)
    both = pd.concat([ptot.rename("paid"), itot.rename("billed")], axis=1).dropna()
    q("O11", "olist", "For orders that have both items and payments, what percentage have total payments differing from "
      "total item price plus freight by more than 1 (in BRL)?", "data_quality", "hard", "cross_table_consistency",
      "number", 100 * ((both.paid - both.billed).abs() > 1).mean(), tol={"abs": 0.02}, pct=True,
      method="aggregate both tables to orders, compare")
    dl = o[o.order_status == "delivered"].dropna(subset=["order_delivered_customer_date"]).merge(cu, on="customer_id")
    dl["days"] = (pd.to_datetime(dl.order_delivered_customer_date) - pd.to_datetime(dl.order_purchase_timestamp)).dt.total_seconds() / 86400
    st = dl.groupby("customer_state").days.median()
    q("O12", "olist", "Which customer state has the slowest median delivery time (purchase to customer delivery) for "
      "delivered orders?", "group_comparison", "medium", "join", "label", st.idxmax(), aliases=[st.idxmax().lower()],
      method="join customers, median by state")
    inst = pay[pay.payment_type == "credit_card"]
    q("O13", "olist", "What percentage of credit-card payments were split into more than one installment?",
      "descriptive", "easy", "none", "number", 100 * (inst.payment_installments > 1).mean(), tol={"abs": 0.05},
      pct=True, method="share installments > 1")

    # --- F1: grain and multi-hop home advantage
    F = DATA / "f1db"
    rr = pd.read_csv(F / "races_race_results.csv", low_memory=False)
    races = pd.read_csv(F / "races.csv", low_memory=False)
    drivers = pd.read_csv(F / "drivers.csv", low_memory=False)
    circuits = pd.read_csv(F / "circuits.csv", low_memory=False)
    pits = pd.read_csv(F / "races_pit_stops.csv", low_memory=False)
    s23 = pits[pits.year == 2023].groupby(["raceId", "driverId"]).size()
    q("F12", "f1db", "In 2023, what was the average number of pit stops per driver per race, among drivers who made at "
      "least one pit stop in that race?", "descriptive", "medium", "grain", "number", s23.mean(), tol={"abs": 0.005},
      method="count stops per driver-race, mean")
    e = rr[(rr.year >= 2014)].groupby("constructorId").agg(pts=("points", "sum"), n=("points", "size"))
    e = e[e.n >= 50]
    best = (e.pts / e.n).idxmax()
    q("F13", "f1db", "From 2014 onwards, among constructors with at least 50 race-result entries, which has the highest "
      "average points per entry?", "descriptive", "medium", "grain", "label", best, aliases=[best.replace("-", " ")],
      method="sum points / entries by constructor")
    h = rr[(rr.year >= 2000) & rr.positionNumber.notna()].merge(races[["id", "circuitId"]], left_on="raceId", right_on="id") \
        .merge(circuits[["id", "countryId"]], left_on="circuitId", right_on="id", suffixes=("", "_c")) \
        .merge(drivers[["id", "nationalityCountryId"]], left_on="driverId", right_on="id", suffixes=("", "_d"))
    home = h.countryId == h.nationalityCountryId
    q("F14", "f1db", "Since 2000, among classified finishers, do drivers finish better at races held in their own "
      "country (circuit country equals driver nationality)? Report the mean finishing position at home minus away "
      "(negative means better at home).", "group_comparison", "hard", "multi_hop_join", "number",
      h.positionNumber[home].mean() - h.positionNumber[~home].mean(), tol={"abs": 0.01},
      method="results -> races -> circuits, results -> drivers; compare means")


DESCRIPTIONS = {
    "diabetes": "Diabetes 130-US hospitals, 1999-2008 (UCI). One row per inpatient encounter of a diabetic patient. "
                "readmitted: NO / >30 / <30 days. IDS_mapping.csv gives the meaning of admission_type_id, "
                "discharge_disposition_id and admission_source_id.",
    "online_retail": "Online Retail II (UCI): transactions of a UK-based online gift retailer, Dec 2009 - Dec 2011. "
                     "One row per invoice line. Invoice numbers starting with 'C' are cancellations.",
    "nyc_taxi": "NYC TLC Yellow Taxi trip records, January 2024, with the official taxi zone lookup table "
                "(LocationID -> Borough, Zone). One row per trip. payment_type: 0 = flex fare, "
                "1 = credit card, 2 = cash, 3 = no charge, 4 = dispute. RatecodeID: 1 = standard, 2 = JFK, "
                "3 = Newark, 4 = Nassau/Westchester, 5 = negotiated, 6 = group ride, 99 = unknown.",
    "bike_sharing": "Capital Bikeshare hourly rentals, Washington DC, 2011-2012 (UCI). temp, atemp, hum, windspeed are "
                    "normalised to 0-1. yr: 0 = 2011, 1 = 2012. weathersit: 1 clear ... 4 heavy rain/snow. "
                    "cnt = casual + registered.",
    "beijing_pm25": "Beijing PM2.5 hourly data from the US Embassy, 2010-2014 (UCI). pm2.5 in ug/m3; cbwd = combined "
                    "wind direction; Iws = cumulated wind speed; Is / Ir = cumulated hours of snow / rain.",
    "movielens": "MovieLens 1M (GroupLens): three linked tables. Ratings 1-5 with Unix timestamps; users have gender, an "
                 "age-group code (1, 18, 25, 35, 45, 50, 56) and an occupation code (0-20); movies have a title "
                 "with the release year and '|'-separated genres.",
    "olist": "Olist Brazilian e-commerce public dataset, 2016-2018: nine linked tables (orders, items, payments, reviews, "
             "customers, products, sellers, geolocation, category-name translation).",
    "f1db": "F1DB: the Formula 1 World Championship database, 1950 to date, as 47 linked tables (races, results, "
            "drivers, constructors, standings, pit stops, qualifying, ...). Text ids such as 'lewis-hamilton'.",
    "shop": "A small online shop: customers, products, orders, order_items and reviews (five linked tables).",
    "flights": "US domestic flights, January-March 2024, from the Bureau of Transportation Statistics On-Time "
               "Performance data, with airport (airports.csv) and carrier (carriers.csv) lookup tables. Times are "
               "local (hhmm). Delays in minutes; ArrDel15 / DepDel15 flag delays of 15+ minutes. The five *Delay "
               "cause columns give minutes attributed to each cause.",
    "imdb": "IMDb non-commercial datasets restricted to feature films: movies (title, year, runtime, genres), "
            "ratings (average rating, number of votes), crew (director and writer ids) and directors (people).",
    "citibike": "Citi Bike (New York) system data for January 2024: one row per ride, with start/end times, "
                "stations, coordinates, bike type and rider type (member or casual).",
    "clinic": "A hospital clinic database: patients, admissions and laboratory results. Data documentation: patient "
              "age is -1 when unknown; readmitted_30d is blank when 30 days of follow-up after discharge are not yet "
              "complete; HbA1c results are stored only when abnormal (6.5 or above).",
    "school": "A university's student records: students, courses (with credits) and enrolments (grade points 0-4, "
              "status passed / failed / withdrawn).",
    "saas": "A subscription software company: accounts (with billing currency), invoices (amounts in the account's "
            "currency), fx_rates (US dollars per unit of each currency), product usage events and cancellations. "
            "Data run to 2024-12-31.",
}


def file_info(dataset: str) -> list[dict]:
    out = []
    for f in sorted((DATA / dataset).iterdir()):
        if f.suffix not in (".csv", ".parquet"):
            continue
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq
            meta = pq.ParquetFile(f)
            rows, cols = meta.metadata.num_rows, meta.schema.names
        else:
            cols = list(pd.read_csv(f, nrows=0).columns)
            with open(f, "rb") as fh:
                rows = sum(1 for _ in fh) - 1
        out.append({"path": f.as_posix(), "rows": rows, "columns": cols})
    return out


def main():
    import questions_v2
    for fn in (diabetes, retail, taxi, bike, pm25, movielens, olist, f1, shop, extra_hard):
        print(f"building {fn.__name__} ...", flush=True)
        fn()
    for fn in (questions_v2.existing_real, questions_v2.big_real, questions_v2.synthetic):
        print(f"building v2 {fn.__name__} ...", flush=True)
        fn(q, DATA)
    # A trap only counts if falling into it changes the answer. Where the naive answer equals the reference,
    # drop the trap label so trap statistics are not diluted by traps that do not bite.
    for qq in QUESTIONS:
        nv, ref = qq.get("naive"), qq["reference"]
        if nv is None or qq["trap"] == "none":
            continue
        if qq["answer_type"] == "number":
            tol = qq["tolerance"]
            same = abs(float(nv) - float(ref)) <= max(tol.get("abs", 0), tol.get("rel", 0) * abs(float(ref)), 1e-9)
        else:
            same = str(nv).strip().lower() == str(ref).strip().lower()
        if same:
            qq["rationale"] = (qq["rationale"] + f" [Trap '{qq['trap']}' does not change the answer here.]").strip()
            qq["trap"], qq["naive"] = "none", None
    infos = {}
    for qq in QUESTIONS:
        ds = qq["dataset"]
        if ds not in infos:
            infos[ds] = file_info(ds)
        qq["files"] = infos[ds]
        qq["dataset_description"] = DESCRIPTIONS[ds]
        if isinstance(qq["reference"], (np.integer, np.floating)):
            qq["reference"] = qq["reference"].item()
    OUT.mkdir(exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(QUESTIONS, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in x.items() if k not in ("files", "dataset_description")} for x in QUESTIONS]) \
        .to_csv(OUT / "questions.csv", index=False)
    df = pd.DataFrame(QUESTIONS)
    print(f"\n{len(df)} questions")
    print(df.groupby(["dataset"]).size().to_string())
    print(df.difficulty.value_counts().to_string())
    print(df.category.value_counts().to_string())
    print(f"with a trap: {(df.trap != 'none').sum()}")


if __name__ == "__main__":
    main()
