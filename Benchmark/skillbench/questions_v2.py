"""
SkillBench v2 questions on REAL data. Added to the v1 bank by build_questions.py. As in v1, every
reference answer is computed here with pandas / numpy / scipy / statsmodels only (no statkit), and trap
questions store the `naive` answer a careless analysis produces.

The trap types that separated systems in v1 (grain, unit of analysis, missing codes, unrecorded values,
unequal exposure, multi-hop joins) are each given several more instances here, on more datasets.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def existing_real(q, DATA):
    # ------------------------------------------------------------------ diabetes
    d = pd.read_csv(DATA / "diabetes" / "diabetic_data.csv", na_values=[], keep_default_na=False)
    ds = "diabetes"
    lt30 = d.readmitted == "<30"
    first = d.sort_values("encounter_id").drop_duplicates("patient_nbr")
    q("D16", ds, "Using each patient's first encounter (the lowest encounter_id), what percentage of patients were "
      "readmitted within 30 days?", "descriptive", "medium", "unit_of_analysis", "number",
      100 * (first.readmitted == "<30").mean(), tol={"abs": 0.05}, pct=True, method="dedupe to first encounter",
      naive=100 * lt30.mean())

    spec = d[d.medical_specialty != "?"].groupby("medical_specialty").filter(lambda g: len(g) >= 1000)
    sr = spec.groupby("medical_specialty").readmitted.apply(lambda s: (s == "<30").mean())
    allr = d.groupby("medical_specialty").filter(lambda g: len(g) >= 1000) \
        .groupby("medical_specialty").readmitted.apply(lambda s: (s == "<30").mean())
    q("D17", ds, "Among medical specialties with at least 1,000 encounters, which has the highest 30-day readmission "
      "rate?", "group_comparison", "hard", "missing_code", "label", sr.idxmax(), aliases=[sr.idxmax().lower()],
      method="exclude the '?' code, filter by size, rate by specialty", naive=allr.idxmax(),
      rationale="'?' (missing specialty) is the largest group and must not be treated as a specialty.")

    a1c = d.A1Cresult.where(d.A1Cresult.isin([">7", ">8", "Norm"]), "not measured")
    tab = pd.crosstab(a1c, d.readmitted)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    q("D18", ds, "Treating 'not measured' as its own HbA1c category (alongside Norm, >7 and >8), what is Cramer's V "
      "for the association between HbA1c result and the three readmission categories?", "group_comparison",
      "hard", "missing_code", "number", np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))),
      tol={"abs": 0.002}, method="chi-square on 4x3 table, Cramer's V")

    X = pd.get_dummies(d.insulin, prefix="insulin", dtype=float).drop(columns="insulin_No")
    X[["number_inpatient", "time_in_hospital"]] = d[["number_inpatient", "time_in_hospital"]].astype(float)
    lg = sm.Logit(lt30.astype(int), sm.add_constant(X)).fit(disp=0)
    q("D19", ds, "In a logistic regression of 30-day readmission on insulin status (No as the reference, with Down, "
      "Steady and Up as dummies), number_inpatient and time_in_hospital, what is the odds ratio for insulin 'Up'?",
      "regression", "hard", "none", "number", np.exp(lg.params["insulin_Up"]), tol={"abs": 0.01},
      method="logistic regression with dummy coding")

    q("D20", ds, "What percentage of encounters have a primary diagnosis (diag_1) that is a diabetes code (ICD-9 codes "
      "starting with 250)?", "data_quality", "medium", "parsing", "number",
      100 * d.diag_1.astype(str).str.startswith("250").mean(), tol={"abs": 0.02}, pct=True,
      method="string prefix on the code (codes mix numbers and letters such as V57)")

    fm = d[d.gender.isin(["Female", "Male"])]
    q("D21", ds, "What is the 30-day readmission rate of women minus that of men, in percentage points (excluding "
      "unknown gender)?", "group_comparison", "easy", "none", "number",
      100 * ((fm.readmitted[fm.gender == "Female"] == "<30").mean() - (fm.readmitted[fm.gender == "Male"] == "<30").mean()),
      tol={"abs": 0.02}, method="difference in proportions")

    q("D22", ds, "What percentage of encounters had a serum glucose test (max_glu_serum) result?", "data_quality",
      "medium", "missing_code", "number", 100 * d.max_glu_serum.isin([">200", ">300", "Norm"]).mean(),
      tol={"abs": 0.02}, pct=True, method="'None' means not measured")

    pc = d[d.payer_code != "?"].payer_code.value_counts()
    q("D23", ds, "Which payer code covers the most encounters?", "descriptive", "medium", "missing_code", "label",
      pc.idxmax(), aliases=[pc.idxmax().lower()], method="exclude '?'", naive=d.payer_code.value_counts().idxmax())

    # ------------------------------------------------------------------ online retail
    df = pd.read_csv(DATA / "online_retail" / "online_retail_ii.csv", dtype={"Invoice": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
    ds = "online_retail"
    canc = df.Invoice.str.startswith("C")
    sales = df[~canc & (df.Quantity > 0) & (df.Price > 0)].copy()
    sales["value"] = sales.Quantity * sales.Price
    gq = "genuine sales (no cancellations, positive quantity and positive price)"

    per_inv = sales.groupby("Invoice").StockCode.nunique()
    q("R15", ds, f"Using {gq}, what is the average number of distinct products (StockCode) per invoice?",
      "descriptive", "medium", "grain", "number", per_inv.mean(), tol={"abs": 0.05},
      method="distinct StockCode per invoice, mean", naive=sales.groupby("Invoice").size().mean())

    cv = sales.dropna(subset=["Customer ID"]).groupby("Customer ID").value.sum().sort_values(ascending=False)
    top = cv.iloc[: int(np.ceil(0.01 * len(cv)))]
    q("R16", ds, f"Among customers with a Customer ID, what percentage of their total value from {gq} comes from the "
      "top 1% of customers by value (rounding the number of customers up)?", "descriptive", "hard", "grain",
      "number", 100 * top.sum() / cv.sum(), tol={"abs": 0.1}, pct=True, method="customer totals, Pareto share")

    ds_ = sales.groupby("Description").Quantity.sum()
    q("R17", ds, f"Which product Description has the largest total quantity sold, using {gq}?", "descriptive",
      "medium", "cancellations", "label", ds_.idxmax(), aliases=[ds_.idxmax().lower()],
      method="sum quantity by description",
      naive=df.groupby("Description").Quantity.sum().idxmax())

    s5 = sales.dropna(subset=["Customer ID"]).drop_duplicates("Invoice")[["Customer ID", "InvoiceDate"]]
    s5 = s5.sort_values(["Customer ID", "InvoiceDate"])
    s5["gap"] = s5.groupby("Customer ID").InvoiceDate.diff().dt.total_seconds() / 86400
    n_inv = s5.groupby("Customer ID").size()
    med_gap = s5[s5["Customer ID"].isin(n_inv[n_inv >= 5].index)].groupby("Customer ID").gap.median()
    q("R18", ds, f"For customers with at least 5 invoices from {gq}, compute each customer's median number of days "
      "between consecutive invoices. What is the median of those per-customer medians?", "time_series", "hard",
      "grain", "number", med_gap.median(), tol={"abs": 0.1},
      method="invoice-level dates per customer, diff, median of medians")

    q("R19", ds, f"What percentage of the total value of {gq} has no Customer ID?", "data_quality", "medium",
      "none", "number", 100 * sales.value[sales["Customer ID"].isna()].sum() / sales.value.sum(), tol={"abs": 0.05},
      pct=True, method="share of value with missing ID", naive=100 * df["Customer ID"].isna().mean())

    hr = sales.groupby(sales.InvoiceDate.dt.hour).value.sum()
    q("R20", ds, f"At which hour of the day (0-23) is the total value of {gq} highest?", "time_series", "easy",
      "none", "label", str(hr.idxmax()), aliases=[str(hr.idxmax())], method="sum by hour")

    # ------------------------------------------------------------------ NYC taxi + zones
    t = pd.read_parquet(DATA / "nyc_taxi" / "yellow_tripdata_2024-01.parquet")
    z = pd.read_csv(DATA / "nyc_taxi" / "taxi_zone_lookup.csv")
    ds = "nyc_taxi"
    inm = (t.tpep_pickup_datetime >= "2024-01-01") & (t.tpep_pickup_datetime < "2024-02-01")
    tj = t[inm].merge(z.add_prefix("PU_"), left_on="PULocationID", right_on="PU_LocationID", how="left") \
        .merge(z.add_prefix("DO_"), left_on="DOLocationID", right_on="DO_LocationID", how="left")
    bor = tj.PU_Borough.value_counts()
    q("T14", ds, "Using taxi_zone_lookup.csv, which borough had the most January 2024 pickups?", "descriptive", "easy",
      "join", "label", bor.idxmax(), aliases=[bor.idxmax().lower()], method="join zone lookup on PULocationID")

    known = ~tj.PU_Borough.isin(["Unknown", "N/A"]) & ~tj.DO_Borough.isin(["Unknown", "N/A"]) \
        & tj.PU_Borough.notna() & tj.DO_Borough.notna()
    q("T15", ds, "Among January 2024 trips whose pickup and dropoff zones both have a known borough (not 'Unknown' "
      "or 'N/A'), what percentage start and end in different boroughs?", "descriptive", "medium", "join", "number",
      100 * (tj.PU_Borough[known] != tj.DO_Borough[known]).mean(), tol={"abs": 0.02}, pct=True,
      method="join the lookup twice (pickup and dropoff)")

    jfk_man = tj[(tj.PU_Zone == "JFK Airport") & (tj.DO_Borough == "Manhattan") & (tj.RatecodeID == 2)]
    q("T16", ds, "What is the median fare_amount of January 2024 trips from the JFK Airport zone to Manhattan charged "
      "at the JFK flat rate (RatecodeID 2)?", "descriptive", "medium", "multi_hop_join", "number",
      jfk_man.fare_amount.median(), tol={"abs": 0.01}, method="join zones, filter, median")

    cc = tj[(tj.payment_type == 1) & (tj.fare_amount > 0) & (tj.PU_Borough == "Manhattan")].copy()
    cc["tip_pct"] = 100 * cc.tip_amount / cc.fare_amount
    zt = cc.groupby("PU_Zone").filter(lambda g: len(g) >= 1000).groupby("PU_Zone").tip_pct.median()
    q("T17", ds, "Among Manhattan pickup zones with at least 1,000 January 2024 credit-card trips with a positive fare, "
      "which zone has the highest median tip percentage (tip / fare x 100)?", "group_comparison", "hard",
      "multi_hop_join", "label", zt.idxmax(), aliases=[zt.idxmax().lower()], method="join, filter, median by zone")

    air = tj.PU_Zone.astype(str).str.contains("Airport")
    dist = tj.trip_distance
    ok = dist > 0
    q("T18", ds, "For January 2024 trips with a positive distance, how much longer (in miles) is the median trip that "
      "starts in an airport zone (zone name containing 'Airport') than the median trip starting elsewhere?",
      "group_comparison", "medium", "join", "number", dist[ok & air].median() - dist[ok & ~air].median(),
      tol={"abs": 0.01}, method="join zones, median difference")

    q("T19", ds, "What is the average tip percentage (tip / fare x 100) for trips paid in cash?", "descriptive",
      "hard", "unrecorded_values", "choice", "cannot_determine", choices=["a number", "cannot_determine"],
      method="recognise that cash tips are not recorded",
      rationale=f"Recorded cash tips average {t.tip_amount[t.payment_type == 2].mean():.4f}: cash tips are not "
                "captured in TLC data, so any computed percentage (about 0) is not the true tip rate.",
      naive="a number")

    jfk = t[inm & (t.PULocationID == 132)]
    per_day = jfk.groupby(jfk.tpep_pickup_datetime.dt.date).size()
    per_day.index = pd.to_datetime(per_day.index)
    wd = per_day.groupby(per_day.index.day_name()).mean()
    all_jfk = t[t.PULocationID == 132]
    pd2 = all_jfk.groupby(all_jfk.tpep_pickup_datetime.dt.date).size()
    pd2.index = pd.to_datetime(pd2.index)
    q("T20", ds, "For pickups in the JFK Airport zone (PULocationID 132), which day of the week has the highest "
      "average number of pickups per day in January 2024?", "time_series", "hard", "unequal_exposure", "label",
      wd.idxmax(), aliases=[wd.idxmax().lower()], method="filter month and zone, daily counts, mean by weekday",
      naive=all_jfk.tpep_pickup_datetime.dt.day_name().value_counts().idxmax())

    # ------------------------------------------------------------------ bike sharing
    b = pd.read_csv(DATA / "bike_sharing" / "hour.csv")
    ds = "bike_sharing"
    daily = b.groupby(["dteday", "weekday"]).casual.sum().reset_index()
    wdc = daily.groupby("weekday").casual.mean()
    q("B13", ds, "On which weekday code (0 = Sunday ... 6 = Saturday) is the average DAILY total of casual rentals "
      "highest?", "time_series", "medium", "grain", "label", str(wdc.idxmax()), aliases=[str(wdc.idxmax())],
      method="aggregate hours to days, mean by weekday")
    Xw = pd.get_dummies(b[["hr", "season"]].astype(str), dtype=float, drop_first=True)
    rw = sm.OLS(b.windspeed, sm.add_constant(Xw)).fit().resid
    rc = sm.OLS(b.cnt, sm.add_constant(Xw)).fit().resid
    q("B14", ds, "What is the partial correlation between windspeed and hourly rentals (cnt) after controlling for "
      "hour of day and season (both as categorical)?", "correlation", "hard", "confounding", "number",
      np.corrcoef(rw, rc)[0, 1], tol={"abs": 0.01}, method="residualise on dummies", naive=b.windspeed.corr(b.cnt))

    # ------------------------------------------------------------------ Beijing PM2.5
    p = pd.read_csv(DATA / "beijing_pm25" / "beijing_pm25.csv")
    ds = "beijing_pm25"
    p["date"] = pd.to_datetime(p[["year", "month", "day"]])
    g = p.groupby("date")["pm2.5"].agg(["mean", "count"])
    v = g[g["count"] >= 18]
    yb = (v["mean"] > 150).groupby(v.index.year).sum()
    q("P11", ds, "Counting only days with at least 18 hourly readings, which year had the most days with a daily mean "
      "PM2.5 above 150?", "descriptive", "hard", "missing_data_rule", "label", str(yb.idxmax()),
      aliases=[str(yb.idxmax())], method="daily means with completeness rule, count by year",
      naive=str((g["mean"] > 150).groupby(g.index.year).sum().idxmax()))
    w = p[p.month.isin([12, 1, 2])][["DEWP", "pm2.5"]].dropna()
    q("P12", ds, "In winter months (December-February), what is the Spearman correlation between dew point (DEWP) "
      "and PM2.5?", "correlation", "medium", "none", "number", stats.spearmanr(w.DEWP, w["pm2.5"])[0],
      tol={"abs": 0.01}, method="filter months, Spearman")
    q("P13", ds, "What percentage of hours that have a PM2.5 reading exceed 300?", "descriptive", "easy", "none",
      "number", 100 * (p["pm2.5"].dropna() > 300).mean(), tol={"abs": 0.02}, pct=True,
      method="share above 300 among non-missing", naive=100 * (p["pm2.5"] > 300).mean())
    miss = p.sort_values(["year", "month", "day", "hour"])["pm2.5"].isna().astype(int).to_numpy()
    runs, cur = [], 0
    for m in miss:
        cur = cur + 1 if m else 0
        runs.append(cur)
    q("P14", ds, "What is the longest run of consecutive hours with a missing PM2.5 reading?", "data_quality",
      "medium", "missing_values", "number", max(runs), tol={"abs": 0}, method="run lengths of missing values")

    # ------------------------------------------------------------------ MovieLens
    u = pd.read_csv(DATA / "movielens" / "users.csv")
    m = pd.read_csv(DATA / "movielens" / "movies.csv")
    rt = pd.read_csv(DATA / "movielens" / "ratings.csv")
    ds = "movielens"
    per = rt.groupby("user_id").rating.agg(["std", "size"])
    q("M13", ds, "Among users who rated at least 100 movies, what is the average of their per-user rating standard "
      "deviations?", "descriptive", "hard", "grain", "number", per[per["size"] >= 100]["std"].mean(),
      tol={"abs": 0.002}, method="per-user SD, then mean")
    j = rt.merge(u[["user_id", "gender"]], on="user_id")
    pop = rt.groupby("movie_id").size()
    jj = j[j.movie_id.isin(pop[pop >= 200].index)]
    gm = jj.groupby(["movie_id", "gender"]).rating.mean().unstack()
    gm = gm.dropna()
    diff = (gm["F"] - gm["M"])
    best = m.set_index("movie_id").title[diff.idxmax()]
    q("M14", ds, "Among movies with at least 200 ratings, which movie has the largest gap between female and male "
      "average rating (female minus male)? Give its title.", "group_comparison", "hard", "join", "label", best,
      aliases=[best.lower(), re.sub(r"\s*\(\d{4}\)$", "", best).lower()], method="join gender, per-movie means")
    q("M15", ds, "What percentage of all ratings are 5 stars?", "descriptive", "easy", "none", "number",
      100 * (rt.rating == 5).mean(), tol={"abs": 0.02}, pct=True, method="share")
    q("M16", ds, "How many (user, movie) pairs have more than one rating?", "data_quality", "easy", "duplicates",
      "number", rt.duplicated(["user_id", "movie_id"]).sum(), tol={"abs": 0}, method="duplicated pairs")

    # ------------------------------------------------------------------ Olist
    D = DATA / "olist"
    o = pd.read_csv(D / "olist_orders_dataset.csv")
    it = pd.read_csv(D / "olist_order_items_dataset.csv")
    pay = pd.read_csv(D / "olist_order_payments_dataset.csv")
    rv = pd.read_csv(D / "olist_order_reviews_dataset.csv")
    cu = pd.read_csv(D / "olist_customers_dataset.csv")
    pr = pd.read_csv(D / "olist_products_dataset.csv")
    se = pd.read_csv(D / "olist_sellers_dataset.csv")
    ds = "olist"
    dl = o[(o.order_status == "delivered")].dropna(subset=["order_delivered_customer_date"])
    late = pd.to_datetime(dl.order_delivered_customer_date) > pd.to_datetime(dl.order_estimated_delivery_date)
    q("O14", ds, "Among delivered orders with a delivery date, what percentage arrived after the estimated delivery "
      "date?", "descriptive", "easy", "none", "number", 100 * late.mean(), tol={"abs": 0.02}, pct=True,
      method="compare dates")
    rscore = rv.groupby("order_id").review_score.mean()
    iss = it.merge(se[["seller_id", "seller_state"]], on="seller_id")
    iss["score"] = iss.order_id.map(rscore)
    ss = iss.dropna(subset=["score"]).groupby("seller_state").filter(lambda g: len(g) >= 1000) \
        .groupby("seller_state").score.mean()
    q("O15", ds, "Assign each order item the mean review score of its order. Among seller states with at least 1,000 "
      "such items, which seller state has the highest average score?", "group_comparison", "hard",
      "multi_hop_join", "label", ss.idxmax(), aliases=[ss.idxmax().lower()],
      method="items -> sellers, items -> reviews via order, mean by state")
    first_pay = pay[pay.payment_sequential == 1].set_index("order_id").payment_type
    n_items = it.groupby("order_id").size()
    dfp = pd.DataFrame({"type": first_pay, "n": n_items}).dropna()
    q("O16", ds, "Using each order's first payment (payment_sequential = 1) to define its payment type, what is the "
      "mean number of items per order for credit_card orders minus boleto orders?", "group_comparison", "hard",
      "fan_out", "number", dfp.n[dfp.type == "credit_card"].mean() - dfp.n[dfp.type == "boleto"].mean(),
      tol={"abs": 0.002}, method="aggregate items per order, join first payment")
    oc = o.merge(cu, on="customer_id")
    q("O17", ds, "What is the average number of orders per customer, counting customers as people "
      "(customer_unique_id)?", "descriptive", "medium", "hidden_entity_key", "number",
      oc.groupby("customer_unique_id").order_id.nunique().mean(), tol={"abs": 0.001},
      method="group by customer_unique_id", naive=oc.groupby("customer_id").order_id.nunique().mean())
    ip = it.merge(pr[["product_id", "product_weight_g"]], on="product_id").dropna(subset=["product_weight_g"])
    q("O18", ds, "At the order-item level, what is the Spearman correlation between product weight (grams) and "
      "freight value?", "correlation", "medium", "join", "number",
      stats.spearmanr(ip.product_weight_g, ip.freight_value)[0], tol={"abs": 0.005}, method="join products, Spearman")

    # ------------------------------------------------------------------ F1DB
    F = DATA / "f1db"
    rr = pd.read_csv(F / "races_race_results.csv", low_memory=False)
    qr = pd.read_csv(F / "races_qualifying_results.csv", low_memory=False)
    pits = pd.read_csv(F / "races_pit_stops.csv", low_memory=False)
    ds = "f1db"
    w10 = rr[(rr.year >= 2010) & (rr.positionNumber == 1)]
    q("F15", ds, "Since 2010, in what percentage of race wins did the winner start from outside the front row (grid "
      "position 3 or worse)?", "descriptive", "medium", "none", "number",
      100 * (w10.gridPositionNumber >= 3).mean(), tol={"abs": 0.1}, pct=True, method="filter winners, grid >= 3")
    poles = qr[(qr.year >= 2014) & (qr.positionNumber == 1)].constructorId.value_counts()
    q("F16", ds, "From 2014 onwards, which constructor has the most qualifying first places (pole positions in the "
      "qualifying results table)?", "descriptive", "medium", "join", "label", poles.idxmax(),
      aliases=[poles.idxmax().replace("-", " ")], method="qualifying results, count position 1")
    q("F17", ds, "How many different drivers have won at least one World Championship race?", "descriptive", "easy",
      "none", "number", rr[rr.positionNumber == 1].driverId.nunique(), tol={"abs": 0}, method="distinct winners")
    p23 = pits[(pits.year == 2023)].timeMillis.dropna() / 1000
    q("F18", ds, "What was the median pit-stop time in seconds in 2023, as recorded in the pit stops table?",
      "descriptive", "medium", "parsing", "number", p23.median(), tol={"abs": 0.01},
      method="use timeMillis (time strings mix seconds and minute:second formats)")


def synthetic(q, DATA):
    """Appendix: planted-trap databases (see make_synthetic.py). Reported separately from real data."""
    # ------------------------------------------------------------------ clinic
    C = DATA / "clinic"
    pt, ad, lb = (pd.read_csv(C / f"{n}.csv") for n in ("patients", "admissions", "labs"))
    ds = "clinic"
    j = ad.merge(pt, on="patient_id")
    within = []
    for sev, g in j.dropna(subset=["readmitted_30d"]).groupby("severity"):
        within.append(g.readmitted_30d[g.treatment == "B"].mean() - g.readmitted_30d[g.treatment == "A"].mean())
    jj = j.dropna(subset=["readmitted_30d"])
    pooled = jj.readmitted_30d[jj.treatment == "B"].mean() - jj.readmitted_30d[jj.treatment == "A"].mean()
    q("C01", ds, "Does treatment B reduce the risk of 30-day readmission compared with treatment A?", "causal", "hard",
      "simpsons_paradox", "choice", "yes", choices=["yes", "no"],
      rationale=f"Pooled B-A {pooled:+.3f} (B looks worse), within severity {[round(x, 3) for x in within]}: "
                "severity (in patients) confounds treatment assignment.", naive="no",
      method="stratify by severity")
    q("C02", ds, "What percentage of admissions with complete 30-day follow-up were readmitted within 30 days?",
      "descriptive", "medium", "censoring", "number", 100 * ad.readmitted_30d.dropna().mean(), tol={"abs": 0.05},
      pct=True, method="exclude admissions without complete follow-up (missing readmitted_30d)",
      naive=100 * ad.readmitted_30d.fillna(0).mean())
    q("C03", ds, "What percentage of HbA1c tests had an abnormal result (6.5 or above)?", "descriptive", "hard",
      "unrecorded_values", "choice", "cannot_determine", choices=["a number", "cannot_determine"],
      rationale="Only abnormal HbA1c results are stored, so every stored HbA1c is abnormal; the denominator of "
                "tests performed is unknown.", naive="a number", method="read the data documentation")
    q("C04", ds, "What is the mean age of patients?", "descriptive", "medium", "missing_code", "number",
      pt.age[pt.age >= 0].mean(), tol={"abs": 0.01}, method="exclude the -1 unknown code", naive=pt.age.mean())
    q("C05", ds, "How many admissions reference a patient who is not in the patients table?", "data_quality", "easy",
      "orphans", "number", (~ad.patient_id.isin(pt.patient_id)).sum(), tol={"abs": 0}, method="anti-join")
    q("C06", ds, "How many lab rows are exact duplicates of an earlier row?", "data_quality", "easy", "duplicates",
      "number", lb.duplicated().sum(), tol={"abs": 0}, method="duplicated(keep='first')")
    disc = ad.dropna(subset=["discharge_date"])
    los = (pd.to_datetime(disc.discharge_date) - pd.to_datetime(disc.admit_date)).dt.days
    q("C07", ds, "Among admissions that have been discharged, what is the median length of stay in days?",
      "descriptive", "easy", "none", "number", los.median(), tol={"abs": 0}, method="date difference")
    glu = lb[lb.test == "glucose"].drop_duplicates()
    per_adm = glu.groupby("admission_id").value.mean()
    q("C08", ds, "Using each admission's mean glucose value (after removing duplicated lab rows), what is the mean "
      "glucose across admissions that had at least one glucose test?", "descriptive", "hard", "grain", "number",
      per_adm.mean(), tol={"abs": 0.01}, method="dedupe, per-admission mean, then mean",
      naive=lb[lb.test == "glucose"].value.mean())

    # ------------------------------------------------------------------ school
    S = DATA / "school"
    st, co, en = (pd.read_csv(S / f"{n}.csv") for n in ("students", "courses", "enrolments"))
    ds = "school"
    e = en.merge(st, on="student_id").merge(co, on="course_id")
    done = e[e.status != "withdrawn"]
    done = done.assign(passed=(done.status == "passed").astype(int))
    pooled = done.passed[done.gender == "F"].mean() - done.passed[done.gender == "M"].mean()
    q("SC1", ds, "Do female students pass courses at a higher rate than male students in the same department?",
      "causal", "hard", "simpsons_paradox", "choice", "yes", choices=["yes", "no"], naive="no",
      rationale=f"Pooled F-M {pooled:+.3f}, but women pass more in every department: they enrol more in "
                "harder departments.", method="compare within department")
    g2 = done.dropna(subset=["grade_points"])
    gpa = g2.assign(w=g2.grade_points * g2.credits).groupby("student_id").apply(
        lambda x: x.w.sum() / x.credits.sum(), include_groups=False)
    q("SC2", ds, "What is the average student GPA, where a student's GPA is the credit-weighted mean of their grade "
      "points over completed (not withdrawn) courses?", "descriptive", "hard", "grain", "number", gpa.mean(),
      tol={"abs": 0.001}, method="credit-weighted per student, then mean over students",
      naive=en.grade_points.mean())
    q("SC3", ds, "What percentage of enrolments were withdrawn?", "descriptive", "easy", "none", "number",
      100 * (en.status == "withdrawn").mean(), tol={"abs": 0.02}, pct=True, method="share")
    dept_rate = done.groupby("department").passed.mean()
    q("SC4", ds, "Which department has the lowest pass rate among completed enrolments?", "group_comparison", "easy",
      "none", "label", dept_rate.idxmin(), aliases=[dept_rate.idxmin().lower()], method="pass rate by department")
    sch = done.groupby("scholarship").passed.mean()
    q("SC5", ds, "What is the pass rate of scholarship students minus that of other students, in percentage points, "
      "among completed enrolments?", "group_comparison", "medium", "join", "number", 100 * (sch[1] - sch[0]),
      tol={"abs": 0.05}, method="join students, difference in rates")

    # ------------------------------------------------------------------ saas
    A = DATA / "saas"
    acc, inv, fx, ev, ca = (pd.read_csv(A / f"{n}.csv") for n in ("accounts", "invoices", "fx_rates", "events",
                                                                  "cancellations"))
    ds = "saas"
    iv = inv.merge(acc[["account_id", "currency", "plan"]], on="account_id").merge(fx, on="currency")
    paid = iv[iv.status == "paid"]
    in24 = paid.invoice_date.str.startswith("2024")
    q("SA1", ds, "What was the total revenue from paid invoices dated in 2024, in US dollars?", "descriptive", "hard",
      "currency_units", "number", (paid.amount * paid.usd_per_unit)[in24].sum(), tol={"rel": 0.001},
      method="convert each invoice with fx_rates, exclude refunds", naive=paid.amount[in24].sum())
    acc2 = acc.merge(ca, on="account_id", how="left")
    end = pd.to_datetime(acc2.cancel_date).fillna(pd.Timestamp("2024-12-31"))
    dur = (end - pd.to_datetime(acc2.signup_date)).dt.days.to_numpy()
    event = acc2.cancel_date.notna().astype(int).to_numpy()
    s, med, n_at = 1.0, None, len(dur)
    for tt in np.unique(dur):
        here = dur == tt
        d_ = int(event[here].sum())
        if d_:
            s *= 1 - d_ / n_at
            if med is None and s <= 0.5:
                med = float(tt)
        n_at -= int(here.sum())
    q("SA2", ds, "What is the median account lifetime in days (from signup to cancellation), accounting for accounts "
      "that are still active at 2024-12-31?", "descriptive", "hard", "censoring", "number", med, tol={"rel": 0.01},
      method="Kaplan-Meier median (censored at 2024-12-31)", naive=float(np.median(dur[event == 1])))
    q("SA3", ds, "How many event rows are exact duplicates of an earlier row?", "data_quality", "easy", "duplicates",
      "number", ev.duplicated().sum(), tol={"abs": 0}, method="duplicated()")
    churn = acc.assign(churned=acc.account_id.isin(ca.account_id)).groupby("plan").churned.mean()
    q("SA4", ds, "Which plan has the highest share of accounts that have cancelled?", "group_comparison", "easy",
      "none", "label", churn.idxmax(), aliases=[churn.idxmax()], method="share cancelled by plan")
    upa = ev.drop_duplicates().groupby("account_id").user_id.nunique()
    q("SA5", ds, "Among accounts with at least one event, what is the average number of distinct active users per "
      "account?", "descriptive", "medium", "duplicates", "number", upa.mean(), tol={"abs": 0.01},
      method="distinct users per account", naive=ev.groupby("account_id").size().mean())


def big_real(q, DATA):
    """The three large real datasets added in v2: US flights (BTS), IMDb (feature films), Citi Bike."""
    NA = "\\N"  # IMDb's missing-value code: a literal backslash-N

    # ------------------------------------------------------------------ flights
    f = pd.read_parquet(DATA / "flights" / "flights_2024q1.parquet")
    ap = pd.read_csv(DATA / "flights" / "airports.csv", encoding="latin-1")
    cr = pd.read_csv(DATA / "flights" / "carriers.csv")
    ds = "flights"
    op = f[(f.Cancelled == 0) & (f.Diverted == 0)]
    q("FL01", ds, "How many flights were scheduled in January-March 2024 in this data?", "descriptive", "easy",
      "none", "number", len(f), tol={"abs": 0}, method="count rows")
    q("FL02", ds, "What percentage of scheduled flights were cancelled?", "descriptive", "easy", "none", "number",
      100 * f.Cancelled.mean(), tol={"abs": 0.01}, pct=True, method="mean of Cancelled")
    top = f.Reporting_Airline.value_counts().idxmax()
    name = cr.set_index("Code").Description[top]
    q("FL03", ds, "Which airline operated the most flights? Give its name (not its code).", "descriptive", "easy",
      "join", "label", name, aliases=[name.lower(), "southwest airlines", "southwest"],
      method="count by carrier code, look up the name in carriers.csv")
    q("FL04", ds, "What percentage of flights that actually operated (not cancelled and not diverted) arrived on time, "
      "meaning less than 15 minutes late?", "descriptive", "medium", "population_definition", "number",
      100 * (op.ArrDel15 == 0).mean(), tol={"abs": 0.02}, pct=True, method="exclude cancelled/diverted; ArrDel15 == 0",
      naive=100 * (f.ArrDel15 == 0).sum() / len(f))
    q("FL05", ds, "Across all flights that operated (not cancelled and not diverted), what is the average weather "
      "delay in minutes per flight?", "descriptive", "hard", "unrecorded_values", "number",
      op.WeatherDelay.fillna(0).mean(), tol={"abs": 0.01},
      method="delay-cause minutes are only recorded for flights 15+ minutes late; blank means 0 for the others",
      naive=op.WeatherDelay.mean())
    causes = {c.replace("Delay", ""): f[c].sum() for c in ["CarrierDelay", "WeatherDelay", "NASDelay",
                                                           "SecurityDelay", "LateAircraftDelay"]}
    big = max(causes, key=causes.get)
    q("FL06", ds, "Which delay cause (carrier, weather, NAS, security or late aircraft) accounts for the most total "
      "delay minutes?", "descriptive", "medium", "none", "label", big,
      aliases=[big.lower(), {"LateAircraft": "late aircraft", "NAS": "national air system"}.get(big, big.lower())],
      method="sum of each cause column", rationale=str({k: int(v) for k, v in causes.items()}))
    q("FL07", ds, "What is the median arrival delay in minutes of flights that operated (not cancelled and not "
      "diverted)?", "descriptive", "medium", "skew", "number", op.ArrDelay.median(), tol={"abs": 0},
      method="median (the distribution is very skewed; the mean is far larger)", naive=op.ArrDelay.mean())
    nc = f[f.Cancelled == 0].merge(ap, left_on="OriginAirportID", right_on="Code")
    big_ap = nc.groupby("Description").filter(lambda g: len(g) >= 20000)
    late = big_ap.groupby("Description").DepDel15.mean()
    q("FL08", ds, "Among origin airports with at least 20,000 non-cancelled departures, which airport (by its name in "
      "airports.csv) had the highest share of departures 15 or more minutes late?", "group_comparison", "hard",
      "multi_hop_join", "label", late.idxmax(),
      aliases=[late.idxmax().lower(), late.idxmax().split(": ")[-1].lower(),
               nc[nc.Description == late.idxmax()].Origin.iloc[0].lower()],
      method="join airports on OriginAirportID (file is Latin-1 encoded), filter, rate")
    days = f.groupby("FlightDate").size()
    days.index = pd.to_datetime(days.index)
    wd = days.groupby(days.index.day_name()).mean()
    q("FL09", ds, "Which day of the week has the highest average number of scheduled flights per day?",
      "time_series", "medium", "unequal_exposure", "label", wd.idxmax(), aliases=[wd.idxmax().lower()],
      method="daily counts, mean by weekday", naive=pd.to_datetime(f.FlightDate).dt.day_name().value_counts().idxmax())
    o2 = op.dropna(subset=["DepDelay", "ArrDelay"])
    q("FL10", ds, "For flights that operated, what is the OLS slope of arrival delay on departure delay (minutes per "
      "minute)?", "regression", "medium", "none", "number", np.polyfit(o2.DepDelay, o2.ArrDelay, 1)[0],
      tol={"abs": 0.003}, method="OLS slope")
    l60 = op[op.DepDelay >= 60]
    q("FL11", ds, "Among operated flights that departed at least 60 minutes late, what percentage made up at least 15 "
      "minutes in the air (arrival delay at least 15 minutes smaller than departure delay)?", "descriptive", "hard",
      "none", "number", 100 * (l60.ArrDelay <= l60.DepDelay - 15).mean(), tol={"abs": 0.02}, pct=True,
      method="filter, compare delays")
    tails = op.dropna(subset=["Tail_Number"]).groupby(["Tail_Number", "FlightDate"]).size()
    q("FL12", ds, "On a day when an aircraft (identified by Tail_Number) flies, how many flights does it operate on "
      "average? Use operated flights with a tail number.", "descriptive", "hard", "grain", "number", tails.mean(),
      tol={"abs": 0.005}, method="count flights per (tail number, date), mean")
    big_c = f.groupby("Reporting_Airline").filter(lambda g: len(g) >= 20000).groupby("Reporting_Airline").Cancelled.mean()
    cname = cr.set_index("Code").Description[big_c.idxmax()]
    q("FL13", ds, "Among airlines with at least 20,000 scheduled flights, which has the highest cancellation rate? "
      "Give its name.", "group_comparison", "medium", "join", "label", cname,
      aliases=[cname.lower(), big_c.idxmax().lower(), cname.split(" ")[0].lower()],
      method="cancellation rate by carrier, look up the name")
    o3 = op.dropna(subset=["ArrDelay"])
    q("FL14", ds, "For flights that operated, what is the Spearman correlation between flight distance and arrival "
      "delay?", "correlation", "medium", "skew", "number", stats.spearmanr(o3.Distance, o3.ArrDelay)[0],
      tol={"abs": 0.005}, method="Spearman")

    # ------------------------------------------------------------------ IMDb
    m = pd.read_parquet(DATA / "imdb" / "movies.parquet")
    r = pd.read_parquet(DATA / "imdb" / "ratings.parquet")
    cw = pd.read_parquet(DATA / "imdb" / "crew.parquet")
    dr = pd.read_parquet(DATA / "imdb" / "directors.parquet")
    ds = "imdb"
    q("I01", ds, "How many feature films are in the movies table?", "descriptive", "easy", "none", "number", len(m),
      tol={"abs": 0}, method="count rows")
    q("I02", ds, "What percentage of films have a known runtime?", "data_quality", "medium", "missing_code", "number",
      100 * (m.runtimeMinutes != NA).mean(), tol={"abs": 0.02}, pct=True,
      method="IMDb codes missing values as the text \\N, not as blanks", naive=100.0)
    rt_ = pd.to_numeric(m.runtimeMinutes.replace(NA, np.nan))
    q("I03", ds, "What is the median runtime, in minutes, of films with a known runtime?", "descriptive", "medium",
      "missing_code", "number", rt_.median(), tol={"abs": 0}, method="convert \\N to missing, median")
    q("I04", ds, "What percentage of films have a user rating?", "descriptive", "easy", "join", "number",
      100 * m.tconst.isin(r.tconst).mean(), tol={"abs": 0.02}, pct=True, method="join movies and ratings")
    mr = m.merge(r, on="tconst")
    pop = mr[mr.numVotes >= 25000]
    best = pop.loc[pop.averageRating.idxmax()]
    q("I05", ds, "Among films with at least 25,000 votes, which has the highest average rating? Give its title.",
      "descriptive", "medium", "small_sample", "label", best.primaryTitle, aliases=[best.primaryTitle.lower()],
      method="filter by votes, max rating", naive=mr.loc[mr.averageRating.idxmax()].primaryTitle)
    g = mr[(mr.numVotes >= 1000) & (mr.genres != NA)].assign(genre=lambda x: x.genres.str.split(",")).explode("genre")
    gs = g.groupby("genre").averageRating.agg(["mean", "size"])
    gs = gs[gs["size"] >= 500]
    q("I06", ds, "Among rated films with at least 1,000 votes, which genre has the highest average rating? Films list "
      "up to three comma-separated genres; count a film in each of its genres, and only consider genres with at "
      "least 500 such films.", "group_comparison", "hard", "multi_valued_field", "label", gs["mean"].idxmax(),
      aliases=[gs["mean"].idxmax().lower()], method="split genres, explode, mean")
    pop10 = mr[mr.numVotes >= 10000][["tconst"]].merge(cw, on="tconst")
    ex = pop10[pop10.directors != NA].assign(d=lambda x: x.directors.str.split(",")).explode("d")
    top_d = ex.d.value_counts()
    dname = dr.set_index("nconst").primaryName.get(top_d.idxmax(), top_d.idxmax())
    q("I07", ds, "Which director has directed the most films that have at least 10,000 votes? (A film can have "
      "several comma-separated directors.) Give the director's name.", "descriptive", "hard", "multi_hop_join",
      "label", dname, aliases=[dname.lower()], method="ratings -> crew (split directors) -> directors",
      rationale=f"{dname}: {top_d.max()} films")
    yr = mr[(mr.numVotes >= 1000) & (mr.startYear != NA)]
    q("I08", ds, "For films with at least 1,000 votes and a known release year, what is the Spearman correlation "
      "between release year and average rating?", "correlation", "medium", "missing_code", "number",
      stats.spearmanr(yr.startYear.astype(int), yr.averageRating)[0], tol={"abs": 0.005}, method="Spearman")
    dec = (m.startYear[m.startYear != NA].astype(int) // 10 * 10).value_counts()
    q("I09", ds, "Which decade (by release year) has the most films in the table? Give it as its first year, e.g. "
      "1990.", "descriptive", "easy", "missing_code", "label", str(dec.idxmax()),
      aliases=[str(dec.idxmax()), f"{dec.idxmax()}s"], method="decade from startYear")
    q("I10", ds, "What is the average rating across rated films when each film is weighted by its number of votes?",
      "descriptive", "medium", "grain", "number", np.average(r.averageRating, weights=r.numVotes), tol={"abs": 0.005},
      method="vote-weighted mean", naive=r.averageRating.mean())
    q("I11", ds, "How many films list more than one director?", "data_quality", "medium", "multi_valued_field",
      "number", cw.directors.str.contains(",").sum(), tol={"abs": 0}, method="count commas in directors")
    q("I12", ds, "What is the median number of votes among rated films?", "descriptive", "easy", "skew", "number",
      r.numVotes.median(), tol={"abs": 0}, method="median", naive=r.numVotes.mean())
    adult = mr[mr.isAdult == "1"]
    q("I13", ds, "What is the mean rating of rated films flagged as adult films (isAdult = 1) minus that of other "
      "rated films?", "group_comparison", "medium", "none", "number",
      adult.averageRating.mean() - mr[mr.isAdult == "0"].averageRating.mean(), tol={"abs": 0.005},
      method="compare means")
    dmov = ex.merge(mr[["tconst", "averageRating"]], on="tconst").merge(dr, left_on="d", right_on="nconst")
    dmov = dmov[dmov.birthYear != NA]
    agg = dmov.groupby(["d", "birthYear"]).averageRating.agg(["mean", "size"]).reset_index()
    agg = agg[agg["size"] >= 5]
    q("I14", ds, "Among directors with a known birth year who directed at least 5 films with 10,000 or more votes, "
      "what is the Spearman correlation between birth year and the director's mean rating over those films?",
      "correlation", "hard", "multi_hop_join", "number", stats.spearmanr(agg.birthYear.astype(int), agg["mean"])[0],
      tol={"abs": 0.01}, method="crew -> directors, per-director mean, Spearman")

    # ------------------------------------------------------------------ Citi Bike
    cb = pd.read_parquet(DATA / "citibike" / "citibike_202401.parquet")
    ds = "citibike"
    st = pd.to_datetime(cb.started_at)
    en = pd.to_datetime(cb.ended_at)
    jan = (st >= "2024-01-01") & (st < "2024-02-01")
    dur = (en - st).dt.total_seconds() / 60
    J = cb[jan].assign(dur=dur[jan], hour=st[jan].dt.hour, date=st[jan].dt.date)
    q("CB01", ds, "How many rides started in January 2024?", "data_quality", "easy", "out_of_range_dates", "number",
      int(jan.sum()), tol={"abs": 0}, method="filter started_at to January 2024", naive=len(cb))
    q("CB02", ds, "What percentage of rides that started in January 2024 were taken by members?", "descriptive",
      "easy", "none", "number", 100 * (J.member_casual == "member").mean(), tol={"abs": 0.02}, pct=True,
      method="share")
    q("CB03", ds, "What is the median ride duration, in minutes, of rides that started in January 2024?",
      "descriptive", "easy", "skew", "number", J.dur.median(), tol={"abs": 0.01}, method="median",
      naive=J.dur.mean())
    ok = J[J.dur <= 24 * 60]
    q("CB04", ds, "Excluding rides longer than 24 hours (bikes not returned), what is the median duration of casual "
      "riders' rides minus that of members, in minutes, for rides started in January 2024?", "group_comparison",
      "medium", "invalid_records", "number",
      ok.dur[ok.member_casual == "casual"].median() - ok.dur[ok.member_casual == "member"].median(),
      tol={"abs": 0.01}, method="filter, median difference")
    stn = J.start_station_name.value_counts()
    q("CB05", ds, "Which start station had the most rides that started in January 2024?", "descriptive", "easy",
      "none", "label", stn.idxmax(), aliases=[stn.idxmax().lower()], method="value counts")
    ke = J.dropna(subset=["end_station_name"])
    q("CB06", ds, "Among rides that started in January 2024 and have an end station recorded, what percentage ended "
      "at the same station they started from?", "descriptive", "medium", "missing_values", "number",
      100 * (ke.start_station_name == ke.end_station_name).mean(), tol={"abs": 0.02}, pct=True,
      method="exclude missing end station, compare names")
    hs = J.groupby("hour").member_casual.apply(lambda s: (s == "casual").mean())
    q("CB07", ds, "At which hour of the day (0-23) is the share of rides taken by casual riders highest, for rides "
      "started in January 2024?", "time_series", "medium", "ratio_of_sums", "label", str(hs.idxmax()),
      aliases=[str(hs.idxmax())], method="share casual by hour",
      naive=str(J[J.member_casual == "casual"].hour.value_counts().idxmax()))
    pdd = J.groupby("date").size()
    pdd.index = pd.to_datetime(pdd.index)
    wdb = pdd.groupby(pdd.index.day_name()).mean()
    q("CB08", ds, "Which day of the week had the highest average number of rides per day in January 2024?",
      "time_series", "medium", "unequal_exposure", "label", wdb.idxmax(), aliases=[wdb.idxmax().lower()],
      method="daily counts, mean by weekday", naive=st.dt.day_name().value_counts().idxmax())
    q("CB09", ds, "What percentage of rides that started in January 2024 have no end station recorded?",
      "data_quality", "easy", "none", "number", 100 * J.end_station_name.isna().mean(), tol={"abs": 0.005}, pct=True,
      method="share missing")

    def hav(a1, o1, a2, o2):
        a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
        h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(h))
    dd = J.dropna(subset=["end_lat", "end_lng", "end_station_name"])
    dd = dd[dd.start_station_name != dd.end_station_name]
    km = hav(dd.start_lat, dd.start_lng, dd.end_lat, dd.end_lng)
    q("CB10", ds, "For rides started in January 2024 that ended at a different, recorded station, what is the median "
      "straight-line (great-circle) distance between start and end coordinates, in km?", "descriptive", "hard",
      "parsing", "number", km.median(), tol={"abs": 0.005}, method="haversine on coordinates")
    sp = dd.assign(km=km)
    sp = sp[(sp.dur >= 2) & (sp.dur <= 120)]
    speed = sp.km / (sp.dur / 60)
    q("CB11", ds, "For rides started in January 2024 between two different recorded stations and lasting 2-120 "
      "minutes, how much faster is the median straight-line speed (km/h) of electric bikes than of classic bikes?",
      "group_comparison", "hard", "invalid_records", "number",
      speed[sp.rideable_type == "electric_bike"].median() - speed[sp.rideable_type == "classic_bike"].median(),
      tol={"abs": 0.02}, method="haversine distance / duration, median by bike type")
    q("CB12", ds, "How many rides that started in January 2024 lasted more than 24 hours?", "data_quality", "easy",
      "invalid_records", "number", int((J.dur > 24 * 60).sum()), tol={"abs": 0}, method="duration filter")
