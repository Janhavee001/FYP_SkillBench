"""
Tests for the relational layer (skills/relational.py). Ground truth is the synthetic shop database
in demo/make_shop_tables.py, where every trap is planted on purpose, so the right answers are known:
the five real links, the 48 orphan orders, the fan-out, the false keys, the ambiguous route and the
Simpson's reversal (express is worse overall, better within each region).
"""
import importlib.util
import pathlib
import sqlite3

import pandas as pd
import pytest

from skills.relational import (aggregate_to_grain, build_relationship_graph, check_unit_of_analysis,
                               detect_aggregation_reversal, discover_tables, execute_join_plan, infer_keys,
                               plan_join_path, validate_join)
from skills.profiling import profile_dataset
from skills.validation import validate_grounding

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_shop_tables", ROOT / "demo" / "make_shop_tables.py")
_shop_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shop_mod)

TRUE_LINKS = {("orders", "customer_id", "customers", "customer_id"),
              ("order_items", "order_id", "orders", "order_id"),
              ("order_items", "product_id", "products", "product_id"),
              ("reviews", "reviewer_id", "customers", "customer_id"),
              ("reviews", "product_id", "products", "product_id")}


@pytest.fixture(scope="module")
def shop():
    return _shop_mod.make_shop()


@pytest.fixture(scope="module")
def shop_dir(tmp_path_factory, shop):
    d = tmp_path_factory.mktemp("shop")
    for name, df in shop.items():
        df.to_csv(d / f"{name}.csv", index=False)
    return d


@pytest.fixture(scope="module")
def keys(shop):
    return infer_keys(shop)


@pytest.fixture(scope="module")
def graph(shop, keys):
    return build_relationship_graph(shop, keys)


# ------------------------------------------------------------- discover
def test_discover_tables_from_a_folder(shop_dir):
    r = discover_tables(shop_dir)
    assert r.ok and set(r.tables) == {"customers", "products", "orders", "order_items", "reviews"}
    assert r.row_counts["orders"] == 1200 and "customer_id" in r.schema["orders"]


def test_discover_tables_reads_declared_keys_from_sqlite(tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE customers (customer_id TEXT PRIMARY KEY, region TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id TEXT REFERENCES customers(customer_id));
        INSERT INTO customers VALUES ('C1','North'),('C2','South');
        INSERT INTO orders VALUES (1,'C1'),(2,'C1'),(3,'C2');""")
    con.close()
    r = discover_tables(db)
    assert r.declared_primary_keys == {"customers": ["customer_id"], "orders": ["order_id"]}
    assert r.declared_foreign_keys == [{"child_table": "orders", "child_column": "customer_id",
                                        "parent_table": "customers", "parent_column": "customer_id"}]
    k = infer_keys(r.tables, r.declared_primary_keys, r.declared_foreign_keys)
    assert k.foreign_keys[0]["source"] == "declared"


# ------------------------------------------------------------- keys
def test_infer_keys_finds_primary_keys_including_composite(keys):
    assert keys.primary_keys["customers"] == ["customer_id"]
    assert keys.primary_keys["orders"] == ["order_id"]
    assert keys.primary_keys["order_items"] == ["order_id", "line_no"]  # no single-column key exists


def test_infer_keys_finds_exactly_the_true_links(keys):
    found = {(f["child_table"], f["child_column"], f["parent_table"], f["parent_column"]) for f in keys.foreign_keys}
    assert found == TRUE_LINKS  # includes reviewer_id, whose name differs from customer_id


def test_infer_keys_rejects_small_integers_that_merely_fit(keys):
    rejected = {(r["child_table"], r["child_column"]) for r in keys.rejected_candidates}
    assert {("order_items", "quantity"), ("order_items", "line_no"), ("reviews", "rating")} <= rejected


def test_infer_keys_measures_orphans_by_rows(keys):
    fk = next(f for f in keys.foreign_keys if f["child_table"] == "orders")
    assert fk["inclusion"] == pytest.approx(1 - 48 / 1200)
    assert any("orphans" in w for w in keys.warnings)


# ------------------------------------------------------------- graph and plan
def test_graph_cardinalities_and_bridge(graph):
    cards = {(e["child_table"], e["parent_table"]): e["cardinality"] for e in graph.edges}
    assert cards[("orders", "customers")] == "1:N"
    assert graph.bridges == {"order_items": ["orders", "products"]}
    assert len(graph.components) == 1


def test_plan_orders_with_customers_is_safe(graph):
    p = plan_join_path(graph, ["orders", "customers"])
    assert p.base_table == "orders" and p.steps[0]["direction"] == "many_to_one" and not p.fan_out_steps


def test_plan_flags_the_ambiguous_route(graph):
    p = plan_join_path(graph, ["customers", "products"])
    assert p.alternatives  # via reviews vs via orders/order_items
    assert any("More than one route" in w for w in p.warnings)


def test_plan_flags_fan_out_when_base_is_the_parent(graph):
    p = plan_join_path(graph, ["orders", "order_items"], base_table="orders")
    assert p.fan_out_steps and "aggregate_to_grain" in p.warnings[0]


# ------------------------------------------------------------- joins
def test_naive_join_inflates_the_mean_and_is_caught(shop):
    orders, items = shop["orders"], shop["order_items"]
    r = validate_join(orders, items, "order_id", expected="many_to_one", left_name="orders", right_name="order_items")
    assert not r.ok and r.fan_out_factor > 2
    assert r.df["order_total"].mean() > orders["order_total"].mean() * 1.1  # the planted inflation


def test_aggregate_to_grain_fixes_the_fan_out(shop):
    agg = aggregate_to_grain(shop["order_items"], "order_id", {"quantity": "sum"})
    assert agg.rows_after == 1200 and agg.rows_before == len(shop["order_items"])
    r = validate_join(shop["orders"], agg.df, "order_id", expected="one_to_one")
    assert r.ok and r.fan_out_factor == 1.0
    assert r.df["order_total"].mean() == pytest.approx(shop["orders"]["order_total"].mean())


def test_join_reports_orphans_and_inner_join_loses_them(shop):
    left = validate_join(shop["orders"], shop["customers"], "customer_id", how="left")
    assert left.ok and left.left_unmatched == 48 and left.rows_result == 1200
    inner = validate_join(shop["orders"], shop["customers"], "customer_id", how="inner", max_unmatched_pct=3)
    assert not inner.ok and inner.rows_result == 1152


def test_join_refuses_mismatched_key_types():
    a = pd.DataFrame({"k": [1, 2]})
    b = pd.DataFrame({"k": ["1", "2"]})
    assert not validate_join(a, b, "k").ok


def test_execute_join_plan_end_to_end(shop, graph):
    p = plan_join_path(graph, ["orders", "customers"])
    run = execute_join_plan(shop, p, columns={"orders": ["express", "on_time"], "customers": ["region", "age"]})
    assert run.ok and len(run.df) == 1200 and run.steps[0]["fan_out_factor"] == 1.0
    assert {"express", "on_time", "region", "age"} <= set(run.df.columns)


# ------------------------------------------------------------- unit of analysis and reversal
def test_unit_of_analysis_flags_repeated_customers(shop, graph):
    run = execute_join_plan(shop, plan_join_path(graph, ["orders", "customers"]))
    u = check_unit_of_analysis(run.df, "customer_id", ["age", "region", "order_total"])
    assert u.repeated and u.n_entities < u.n_rows
    assert set(u.entity_level_columns) == {"age", "region"}  # customer attributes, not order attributes


def test_simpsons_reversal_is_detected_across_tables(shop, graph):
    run = execute_join_plan(shop, plan_join_path(graph, ["orders", "customers"]))
    r = detect_aggregation_reversal(run.df, "express", "on_time", "region")
    assert r.reversed and r.pooled_effect < 0 < r.within_effect
    assert r.within_effect == pytest.approx(0.05, abs=0.04)  # planted within-region benefit
    no_rev = detect_aggregation_reversal(run.df, "express", "on_time", "customer_id", min_group_size=10_000)
    assert not no_rev.ok  # no group big enough: reported, not guessed


# ------------------------------------------------------------- grounding across tables
def test_validate_grounding_checks_table_dot_column(shop_dir):
    schema = discover_tables(shop_dir).schema
    ok = validate_grounding(None, ["orders.customer_id", "customers.region"], schema=schema)
    assert ok.ok
    bad = validate_grounding(None, ["orders.region"], schema=schema)  # region lives in customers
    assert not bad.ok and "orders.region" in bad.unknown
    single = validate_grounding(profile_dataset(pd.DataFrame({"a": [1]})), ["a"])  # old behaviour unchanged
    assert single.ok
