"""
External anchors for the reference answers: the benchmark's copies of the data must reproduce the record counts
that the data publishers document. If a count matches, the download and parsing are intact; the reference answers
are then computed on the same data the publisher describes.

    python check_anchors.py   ->   results/anchors.md
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent
D = ROOT / "data"


def rows_csv(p, **kw):
    return len(pd.read_csv(p, low_memory=False, **kw))


def rows_pq(p):
    return pq.ParquetFile(p).metadata.num_rows


# (dataset, quantity, documented value, source of the documented value, function computing our value)
ANCHORS = [
    ("MovieLens 1M", "ratings", 1_000_209, "GroupLens ml-1m README", lambda: rows_csv(D / "movielens/ratings.csv")),
    ("MovieLens 1M", "users", 6_040, "GroupLens ml-1m README", lambda: rows_csv(D / "movielens/users.csv")),
    ("MovieLens 1M", "movies", 3_883, "movies.dat in ml-1m", lambda: rows_csv(D / "movielens/movies.csv")),
    ("Bike Sharing (DC)", "hourly records", 17_379, "UCI repository page", lambda: rows_csv(D / "bike_sharing/hour.csv")),
    ("Beijing PM2.5", "hourly records", 43_824, "UCI repository page", lambda: rows_csv(D / "beijing_pm25/beijing_pm25.csv")),
    ("Diabetes 130-US", "encounters", 101_766, "UCI repository page", lambda: rows_csv(D / "diabetes/diabetic_data.csv")),
    ("Online Retail II", "invoice lines", 1_067_371, "UCI repository page",
     lambda: rows_csv(D / "online_retail/online_retail_ii.csv")),
    ("Olist", "orders", 99_441, "Kaggle dataset description", lambda: rows_csv(D / "olist/olist_orders_dataset.csv")),
    ("Olist", "order items", 112_650, "Kaggle dataset description",
     lambda: rows_csv(D / "olist/olist_order_items_dataset.csv")),
    ("Olist", "unique customers", 96_096, "Kaggle dataset description",
     lambda: pd.read_csv(D / "olist/olist_customers_dataset.csv").customer_unique_id.nunique()),
    ("NYC Yellow Taxi", "trips, Jan 2024", 2_964_624, "NYC TLC trip record file",
     lambda: rows_pq(D / "nyc_taxi/yellow_tripdata_2024-01.parquet")),
]


def main():
    out = ["# External anchors", "",
           "Record counts in the benchmark's data versus the counts documented by each publisher.", "",
           "| Dataset | Quantity | Documented | Benchmark data | Source | Match |", "|---|---|---:|---:|---|---|"]
    ok = 0
    for ds, what, doc, src, fn in ANCHORS:
        try:
            v = int(fn())
            m = v == doc
        except Exception as exc:  # noqa: BLE001
            v, m = f"error: {exc}", False
        ok += m
        out.append(f"| {ds} | {what} | {doc:,} | {v:,} | {src} | {'yes' if m else 'NO'} |" if isinstance(v, int)
                   else f"| {ds} | {what} | {doc:,} | {v} | {src} | NO |")
    out += ["", f"{ok} of {len(ANCHORS)} documented counts reproduced."]
    (ROOT / "results" / "anchors.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
