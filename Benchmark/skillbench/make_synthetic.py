"""
Three SYNTHETIC multi-table databases with planted traps, for SkillBench v2.

Real data cannot guarantee the hardest traps (a Simpson's reversal, values that were never recorded,
censored follow-up) or a known true causal effect. These databases plant them deliberately, with fixed
seeds, so the correct answer is known by construction. Results on synthetic and real data are reported
separately.

  clinic  patients, admissions, labs            Simpson (treatment x severity), censored follow-up,
                                                -1 age code, abnormal-only lab storage, orphans, duplicates
  school  students, courses, enrolments         Simpson (gender x department), credit-weighted GPA grain,
                                                withdrawn grades, ecological correlation
  saas    accounts, invoices, fx_rates,         multi-currency revenue, refunds, censored churn,
          events, cancellations                 duplicate events, fan-out

    python make_synthetic.py   ->   data/clinic, data/school, data/saas
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2024-12-31")


def clinic(seed: int = 11) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 3000
    severity = rng.choice(["mild", "severe"], n, p=[0.6, 0.4])
    age = rng.integers(20, 90, n).astype(int)
    age_recorded = np.where(rng.random(n) < 0.06, -1, age)                       # -1 = unknown
    patients = pd.DataFrame({"patient_id": [f"P{i:05d}" for i in range(1, n + 1)], "sex": rng.choice(["F", "M"], n),
                             "age": age_recorded, "severity": severity,
                             "site": rng.choice(["North", "Central", "South"], n)})
    rows = []
    aid = 0
    for i in range(n):
        for _ in range(1 + rng.poisson(0.6)):
            aid += 1
            sev = severity[i]
            treat = "B" if rng.random() < (0.75 if sev == "severe" else 0.25) else "A"   # B mostly for severe
            base = 0.30 if sev == "severe" else 0.08
            p_readm = base - 0.04 * (treat == "B")                                     # B helps in both strata
            admit = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 730)))
            los = int(max(1, rng.gamma(2.0, 2.5 if sev == "mild" else 4.0)))
            discharge = admit + pd.Timedelta(days=los)
            still_in = discharge > CUTOFF
            follow_complete = (CUTOFF - discharge).days >= 30
            readm = int(rng.random() < p_readm) if follow_complete and not still_in else np.nan
            rows.append({"admission_id": f"A{aid:06d}", "patient_id": patients.patient_id[i],
                         "admit_date": admit.date().isoformat(),
                         "discharge_date": None if still_in else discharge.date().isoformat(),
                         "treatment": treat, "readmitted_30d": readm})
    admissions = pd.DataFrame(rows)
    orphan_idx = rng.choice(len(admissions), 60, replace=False)
    admissions.loc[orphan_idx, "patient_id"] = [f"P9{i:04d}" for i in range(60)]
    labs = []
    lid = 0
    for a in admissions.admission_id:
        for test in rng.choice(["glucose", "creatinine", "hba1c"], 1 + rng.poisson(1.2)):
            val = {"glucose": rng.normal(120, 35), "creatinine": rng.lognormal(0, 0.35),
                   "hba1c": rng.normal(6.3, 1.1)}[test]
            if test == "hba1c" and val < 6.5:
                continue                                   # normal HbA1c results are not stored
            lid += 1
            labs.append({"lab_id": f"L{lid:07d}", "admission_id": a, "test": test, "value": round(float(val), 2)})
    labs = pd.DataFrame(labs)
    labs = pd.concat([labs, labs.sample(150, random_state=seed)], ignore_index=True)   # duplicated rows
    return {"patients": patients, "admissions": admissions, "labs": labs}


def school(seed: int = 22) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 4000
    students = pd.DataFrame({"student_id": [f"S{i:05d}" for i in range(1, n + 1)],
                             "gender": rng.choice(["F", "M"], n), "cohort_year": rng.choice([2021, 2022, 2023], n),
                             "scholarship": (rng.random(n) < 0.2).astype(int)})
    depts = {"Math": 0.55, "Physics": 0.60, "History": 0.85, "Art": 0.90, "Biology": 0.70, "Economics": 0.65}
    courses = []
    for d in depts:
        for k in range(1, 7):
            courses.append({"course_id": f"{d[:3].upper()}{100 * k + k}", "department": d,
                            "level": 100 * ((k + 1) // 2), "credits": int(rng.choice([2, 3, 4]))})
    courses = pd.DataFrame(courses)
    hard = {"Math", "Physics", "Economics"}
    rows = []
    for _, s in students.iterrows():
        pref = ["Math", "Physics", "Economics"] if s.gender == "F" else ["History", "Art", "Biology"]
        for _ in range(rng.integers(4, 9)):
            dept = rng.choice(pref) if rng.random() < 0.7 else rng.choice(list(depts))
            c = courses[courses.department == dept].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            p_pass = depts[dept] + (0.05 if s.gender == "F" else 0.0)        # women pass more in every dept
            withdrawn = rng.random() < 0.06
            passed = (not withdrawn) and rng.random() < p_pass
            gp = np.nan if withdrawn else (float(np.clip(rng.normal(3.0 if passed else 1.0, 0.5), 0, 4)))
            rows.append({"student_id": s.student_id, "course_id": c.course_id,
                         "term": f"{rng.choice([2022, 2023, 2024])}-{rng.choice(['S1', 'S2'])}",
                         "status": "withdrawn" if withdrawn else ("passed" if passed else "failed"),
                         "grade_points": None if np.isnan(gp) else round(gp, 2)})
    enrolments = pd.DataFrame(rows)
    return {"students": students, "courses": courses, "enrolments": enrolments}


def saas(seed: int = 33) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 1500
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.35, 0.10])
    country = rng.choice(["US", "DE", "FR", "GB", "IN"], n, p=[0.4, 0.2, 0.15, 0.15, 0.1])
    currency = pd.Series(country).map({"US": "USD", "DE": "EUR", "FR": "EUR", "GB": "GBP", "IN": "INR"}).to_numpy()
    signup = [pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d)) for d in rng.integers(0, 900, n)]
    accounts = pd.DataFrame({"account_id": [f"AC{i:05d}" for i in range(1, n + 1)], "plan": plan, "country": country,
                             "currency": currency, "signup_date": [d.date().isoformat() for d in signup]})
    fx = pd.DataFrame({"currency": ["USD", "EUR", "GBP", "INR"], "usd_per_unit": [1.0, 1.08, 1.27, 0.012]})
    price_usd = {"basic": 20, "pro": 60, "enterprise": 400}
    hazard = {"basic": 1 / 400, "pro": 1 / 900, "enterprise": 1 / 2500}               # daily churn hazard
    inv, canc, ev = [], [], []
    iid = eid = 0
    usd_rate = dict(zip(fx.currency, fx.usd_per_unit))
    for i in range(n):
        life = rng.exponential(1 / hazard[plan[i]])
        end = signup[i] + pd.Timedelta(days=int(life))
        if end <= CUTOFF:
            canc.append({"account_id": accounts.account_id[i], "cancel_date": end.date().isoformat()})
        last = min(end, CUTOFF)
        d = signup[i]
        while d <= last:
            iid += 1
            local = price_usd[plan[i]] / usd_rate[currency[i]] * rng.uniform(0.95, 1.05)
            status = "refunded" if rng.random() < 0.04 else "paid"
            inv.append({"invoice_id": f"IN{iid:07d}", "account_id": accounts.account_id[i],
                        "invoice_date": d.date().isoformat(), "amount": round(float(local), 2), "status": status})
            d += pd.Timedelta(days=30)
        n_users = {"basic": 2, "pro": 6, "enterprise": 25}[plan[i]]
        for _ in range(int(rng.poisson(n_users * 6))):
            eid += 1
            t = signup[i] + pd.Timedelta(days=int(rng.uniform(0, max(1, (last - signup[i]).days))))
            ev.append({"event_id": f"E{eid:08d}", "account_id": accounts.account_id[i],
                       "user_id": f"{accounts.account_id[i]}-U{int(rng.integers(1, n_users + 1))}",
                       "event_type": rng.choice(["login", "report", "export"], p=[0.7, 0.2, 0.1]),
                       "event_time": t.date().isoformat()})
    events = pd.DataFrame(ev)
    events = pd.concat([events, events.sample(400, random_state=seed)], ignore_index=True)   # duplicated events
    return {"accounts": accounts, "invoices": pd.DataFrame(inv), "fx_rates": fx, "events": events,
            "cancellations": pd.DataFrame(canc)}


def main():
    for name, fn in (("clinic", clinic), ("school", school), ("saas", saas)):
        out = DATA / name
        out.mkdir(parents=True, exist_ok=True)
        for t, df in fn().items():
            df.to_csv(out / f"{t}.csv", index=False)
            print(f"{name}/{t}.csv: {len(df):,} rows")


if __name__ == "__main__":
    main()
