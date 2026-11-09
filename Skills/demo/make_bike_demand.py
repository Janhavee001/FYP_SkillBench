"""
Generates demo/bike_demand_daily.csv: two years of SYNTHETIC daily bike-rental counts, shaped like
the UCI Bike Sharing data (a growth trend, a yearly cycle, a weekly cycle, weather effects, noise).
It exists so the time-series skill can be demonstrated before the real Bike Sharing data is added.
Because the components are planted, the true answers are known.

Run: python demo/make_bike_demand.py
"""
import pathlib

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
t = np.arange(len(days))
doy = days.dayofyear.to_numpy()

temp = 14 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 2.5, len(days))  # deg C
weekday = days.dayofweek.to_numpy()
workingday = (weekday < 5).astype(int)
weather = rng.choice(["clear", "mist", "rain"], size=len(days), p=[0.62, 0.28, 0.10])

cnt = (3000
       + 2.5 * t                                          # growth: +2.5 rentals per day
       + 120 * (temp - temp.mean())                       # warmer days, more rentals
       + np.where(workingday == 1, 350, -500)             # weekly cycle: commuters on weekdays
       + np.select([weather == "mist", weather == "rain"], [-300, -1400], 0)
       + rng.normal(0, 250, len(days)))
cnt = np.clip(cnt, 50, None).round().astype(int)

out = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "temp_c": temp.round(1), "workingday": workingday,
                    "weather": weather, "cnt": cnt})
path = pathlib.Path(__file__).with_name("bike_demand_daily.csv")
out.to_csv(path, index=False)
print(f"wrote {path} ({len(out)} rows)")
