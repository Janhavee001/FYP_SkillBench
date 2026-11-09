"""
Generates demo/shop/: a SYNTHETIC five-table online-shop database for testing and demonstrating
the relational-data-analysis skill. Every trap from the multi-table research plan is planted on
purpose, so the correct answers are known:

  customers(customer_id, region, age)               customer_id is text: C0001..C0300
  products(product_id, category, price)             product_id 1..60
  orders(order_id, customer_id, order_date, express, on_time, order_total)   order_id 1..1200
  order_items(order_id, line_no, product_id, quantity, unit_price)           no single-column key
  reviews(review_id, reviewer_id, product_id, rating)                        reviewer_id -> customers

Planted traps
  - Orphans: 48 of 1,200 orders (4%) reference customers that do not exist.
  - Fan-out: order_items has 1-6 lines per order, so averaging order_total after joining
    order_items over-weights big orders.
  - False key: quantity (1..4) is contained in both order_id and product_id ranges but is not a key.
  - Differently named key: reviews.reviewer_id refers to customers.customer_id.
  - Ambiguous path: customers reach products through orders/order_items and through reviews.
  - Simpson's paradox: express orders are on time LESS often overall, but MORE often within each
    region, because express is mostly used in the South where deliveries are slower. The
    confounder (region) lives in customers, so it only appears after a join.

Run: python demo/make_shop_tables.py   (writes demo/shop/*.csv)
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


def make_shop(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    n_cust = 300
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_cust + 1)],
        "region": rng.choice(["North", "South"], n_cust),
        "age": rng.integers(18, 71, n_cust),
    })

    products = pd.DataFrame({
        "product_id": np.arange(1, 61),
        "category": rng.choice(["books", "garden", "kitchen", "toys"], 60),
        "price": np.round(rng.lognormal(3.0, 0.6, 60), 2),
    })

    n_orders = 1200
    cust_ids = rng.choice(customers["customer_id"], n_orders)
    orphan_rows = rng.choice(n_orders, 48, replace=False)
    cust_ids[orphan_rows] = [f"C9{i:03d}" for i in range(48)]                 # customers that do not exist
    region_of = dict(zip(customers["customer_id"], customers["region"]))
    region = np.array([region_of.get(c, rng.choice(["North", "South"])) for c in cust_ids])
    express = (rng.random(n_orders) < np.where(region == "South", 0.8, 0.2)).astype(int)
    p_on_time = np.where(region == "South", 0.50, 0.90) + 0.05 * express    # express helps in every region
    on_time = (rng.random(n_orders) < p_on_time).astype(int)
    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": cust_ids,
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 366, n_orders), unit="D"),
        "express": express,
        "on_time": on_time,
    })
    orders["order_date"] = orders["order_date"].dt.strftime("%Y-%m-%d")

    lines = []
    price_of = dict(zip(products["product_id"], products["price"]))
    for oid in orders["order_id"]:
        for line in range(1, min(6, 1 + rng.poisson(1.5)) + 1):
            pid = int(rng.integers(1, 61))
            lines.append({"order_id": int(oid), "line_no": line, "product_id": pid,
                          "quantity": int(rng.integers(1, 5)), "unit_price": price_of[pid]})
    order_items = pd.DataFrame(lines)
    totals = (order_items["quantity"] * order_items["unit_price"]).groupby(order_items["order_id"]).sum()
    orders["order_total"] = orders["order_id"].map(totals).round(2)

    n_rev = 500
    reviews = pd.DataFrame({
        "review_id": [f"R{i:04d}" for i in range(1, n_rev + 1)],
        "reviewer_id": rng.choice(customers["customer_id"], n_rev),
        "product_id": rng.integers(1, 61, n_rev),
        "rating": rng.integers(1, 6, n_rev),
    })
    return {"customers": customers, "products": products, "orders": orders,
            "order_items": order_items, "reviews": reviews}


if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("shop")
    out.mkdir(exist_ok=True)
    for name, df in make_shop().items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote shop/{name}.csv ({len(df)} rows)")
