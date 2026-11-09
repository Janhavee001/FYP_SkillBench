---
name: relational-data-analysis
description: Analyse data spread across several related tables (multiple CSV files or a SQLite database) - discover the tables, infer primary and foreign keys, map how the tables relate (one-to-many, many-to-many), plan and run joins safely, and catch the silent join errors that corrupt statistics (wrong keys, row fan-out, orphan rows, grain mismatch, repeated entities, Simpson's paradox) before handing one clean table to the statistical skills. Use whenever the user uploads two or more related tables, a .db/.sqlite file, or asks a question whose variables live in different tables ("do repeat customers spend more?", "average order value by customer region"). Uses the bundled statkit Python tools.
---

# Relational data analysis (multiple tables)

With several tables, most errors happen at the join, and they are silent: the join runs, the
numbers look plausible, and they are wrong. An order with five items appears five times after a
naive join, so averages of order values are inflated. Orders whose customer is missing quietly
disappear. This skill turns "several tables" into "one analysis table at a stated grain" with every
step checked. Then the other skills (group comparison, correlation, regression...) take over.

## When to use

- Two or more related files are uploaded, or a SQLite database.
- The question's variables live in different tables.

**Not for:** a single table (go straight to `data-profiling`), or unrelated datasets that would
need fuzzy matching. Say that this skill only handles tables linked by keys.

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Tool signatures are in `reference/tools.md`.

## Procedure

1. **Discover:** `discover_tables(folder_or_list_of_paths_or_db)`. Read `row_counts` and `schema`.
   Pass the exact uploaded files when the upload folder also holds unrelated files.
2. **Infer keys:** `infer_keys(t.tables, t.declared_primary_keys, t.declared_foreign_keys)`.
   - Read every link's `evidence`.
   - Read `rejected_candidates`: columns that fit by value but are not keys (quantities, ratings).
   - Orphan warnings ("4% of rows have no match") must reach the user.
   - If a link the user expects is missing, ask; never invent a join key.
3. **Map the relationships:** `build_relationship_graph(t.tables, keys)`. Tell the user the structure
   in plain words from `description` ("each customer has zero or more orders").
4. **Decide the grain** before joining: what should one row of the analysis table be? "Average order
   value by region" means one row per order. "Do older customers spend more?" means one row per
   customer.
5. **Plan:** `plan_join_path(graph, [tables needed], base_table=<the grain table>)`.
    - `many_to_one` steps are safe.
    - A `one_to_many` step fans out. Aggregate the child first: `aggregate_to_grain(child, key, {...})`,
      then join the aggregate.
    - If `alternatives` is non-empty, the question is ambiguous (products a customer reviewed vs
      bought). Say which route you used, or ask.
6. **Join with checks:** `execute_join_plan(t.tables, plan, columns={...})`, or `validate_join` for a
   single join. Report rows before and after, the fan-out factor and unmatched rows. If `ok` is
   False, fix the cause (aggregate, change the key, or the join type); do not analyse that table.
7. **Check the unit of analysis:** `check_unit_of_analysis(df, entity_col)`. If one entity spans
   several rows, aggregate to the entity before any test, or say the rows are not independent.
8. **Hand over:** run the matching statistical skill's procedure on the final table, stating the grain.
9. **Check for reversal:** when comparing groups across a variable from another table, run
   `detect_aggregation_reversal(df, x, y, group_col)` for plausible confounders (region, segment,
   class). If `reversed`, report the within-group result and explain the reversal.

## Common errors to avoid

- Joining on a column because it shares a name or a value range. Use only links from `infer_keys` or
  declared keys.
- Averaging a parent-level number (order total, customer age) after a one-to-many join.
- Using an inner join without saying how many rows it dropped.
- Running a t-test on rows that repeat the same customer.
- Answering from the pooled result when the per-group result reverses it.

## Interpretation guidance

State the grain in the answer ("per order, n = 1,200"). Name the rows lost or null-filled by joins
and why. When a reversal is found, lead with the within-group result and explain the paradox in one
sentence: which variable confounds, and how.

## Report format

1. Table map: tables, rows, how they link.
2. Join path and checks: rows before and after, orphans, fan-out handled.
3. The analysis result, via the statistical skill.
4. Caveats from the joins: lost rows, reversal, ambiguity.

## Worked example

```python
from statkit.relational import (discover_tables, infer_keys, build_relationship_graph, plan_join_path,
                                execute_join_plan, aggregate_to_grain, validate_join,
                                check_unit_of_analysis, detect_aggregation_reversal)

t = discover_tables("<folder with the uploaded files>")
print(t.row_counts)
keys = infer_keys(t.tables, t.declared_primary_keys, t.declared_foreign_keys)
print([(f["child_table"], f["child_column"], "->", f["parent_table"]) for f in keys.foreign_keys], keys.warnings)
graph = build_relationship_graph(t.tables, keys)
print("\n".join(graph.description))

# Question: are express orders delivered on time more often? Grain: one row per order.
plan = plan_join_path(graph, ["orders", "customers"], base_table="orders")
run = execute_join_plan(t.tables, plan, columns={"orders": ["express", "on_time", "order_total"],
                                                 "customers": ["region", "age"]})
print(run.steps, run.warnings)
rev = detect_aggregation_reversal(run.df, "express", "on_time", "region")
print(rev.pooled_effect, rev.within_effect, rev.reversed)

# The fan-out trap and its fix: average order value
naive = validate_join(t.tables["orders"], t.tables["order_items"], "order_id", expected="many_to_one")
print(naive.ok, naive.fan_out_factor)
lines = aggregate_to_grain(t.tables["order_items"], "order_id", {"quantity": "sum"})
safe = validate_join(t.tables["orders"], lines.df, "order_id", expected="one_to_one")
print(safe.ok, safe.df["order_total"].mean())
print(check_unit_of_analysis(run.df, "customer_id").warnings)
```
