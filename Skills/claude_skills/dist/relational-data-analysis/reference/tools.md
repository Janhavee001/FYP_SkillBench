# Tool reference: relational-data-analysis

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

## `discover_tables`

`from statkit.relational import discover_tables`

```python
discover_tables(source) -> 'TablesReport'
```

Load every table the analysis may need and say what is there before anything is joined.

`source` can be a folder (every CSV/TSV/Parquet/Excel file in it), a list of file paths, a
SQLite database file (all tables, plus its declared primary and foreign keys), or a dict of
{name: DataFrame}. Table names come from file names ("Order Items.csv" -> "order_items").
CSVs go through load_dataset, so each delimiter is sniffed rather than assumed.

Returns `TablesReport` with: `tables`, `schema`, `row_counts`, `declared_primary_keys`, `declared_foreign_keys`, `sources`

## `infer_keys`

`from statkit.relational import infer_keys`

```python
infer_keys(tables: 'dict', declared_primary_keys: 'dict | None' = None, declared_foreign_keys: 'list | None' = None, min_inclusion: 'float' = 0.9) -> 'KeysReport'
```

Infer primary keys and foreign keys from the data, and say why.

Primary key: a column with no missing values and no duplicates, preferring ID-like names
(`customer_id` in `customers`). If no single column qualifies, an ID-like pair is tried
(e.g. order_id + line_no for order lines).

Foreign key: a column whose values are (almost) all contained in another table's primary key
(an inclusion dependency), with compatible types. Containment is measured over rows: the share
of non-missing child rows whose value exists in the parent, so it equals 1 - the orphan rate.
It must also have name evidence that points at the parent (`customer_id` -> customers), unless
it holds 10+ distinct text IDs that are all contained (`reviewer_id` holding customer IDs).
Small integer columns (quantities, ratings, line numbers) fit inside almost every integer ID
range by chance, so numbers alone never make a key.
Each child column gets at most one parent: the best-scoring one. `min_inclusion` below 1
tolerates orphans (a few child rows whose parent is missing); validate_join counts them.

Declared keys (from a SQLite schema) are taken as given and marked source="declared".

Returns `KeysReport` with: `primary_keys`, `pk_candidates`, `foreign_keys`, `rejected_candidates`

## `build_relationship_graph`

`from statkit.relational import build_relationship_graph`

```python
build_relationship_graph(tables: 'dict', keys: 'KeysReport') -> 'RelationshipGraph'
```

Turn the key links into a graph: tables are nodes, foreign keys are edges labelled with their
cardinality from the parent's side, "1:1" or "1:N" (one customer, many orders). A table with
two or more foreign keys and no single-column key of its own is a bridge: through it, its
parents are N:M (orders and products, via order_items). Tables in separate components cannot
be joined at all.

Returns `RelationshipGraph` with: `tables`, `edges`, `bridges`, `components`, `description`

## `plan_join_path`

`from statkit.relational import plan_join_path`

```python
plan_join_path(graph: 'RelationshipGraph', needed_tables: 'list[str]', base_table: 'str | None' = None, max_path_length: 'int' = 5) -> 'JoinPlan'
```

Plan how to join the tables a question needs, starting from a base table whose rows are the
unit of analysis (the grain). Each step is labelled many_to_one (safe: looks up one parent row
per row) or one_to_many (fans out: each row is repeated once per child row, which inflates sums
and means of the repeated columns). Without `base_table`, the table that reaches the others with
the fewest fan-out steps is chosen; say which grain the question is really about and pass it.

When two routes of similar length connect the same tables (customers reach products through
reviews, or through orders and order_items), the shortest is planned and the others are reported
in `alternatives`, because the routes answer different questions ("products a customer reviewed"
vs "products a customer bought").

Returns `JoinPlan` with: `base_table`, `grain`, `steps`, `fan_out_steps`, `alternatives`

## `aggregate_to_grain`

`from statkit.relational import aggregate_to_grain`

```python
aggregate_to_grain(df: 'DataFrame', key: 'str | list[str]', aggregations: 'dict', count_column: 'str | None' = 'n_rows') -> 'GrainResult'
```

Roll a child table up to one row per key before joining it to its parent (order lines to
orders: total quantity, number of lines). This is the fix for fan-out: joining the aggregate
is many_to_one, so the parent's rows are not repeated.

`aggregations` maps column -> function name or list of names ("sum", "mean", "max", "nunique",
...). New columns are named <column>_<function>.

Returns `GrainResult` with: `df`, `key`, `rows_before`, `rows_after`, `new_columns`

## `validate_join`

`from statkit.relational import validate_join`

```python
validate_join(left: 'DataFrame', right: 'DataFrame', left_on: 'str', right_on: 'str | None' = None, how: 'str' = 'left', expected: 'str' = 'many_to_one', left_name: 'str' = 'left', right_name: 'str' = 'right', max_unmatched_pct: 'float' = 5.0) -> 'JoinReport'
```

Do the join and report what it did to the rows, so a bad join is caught before any statistic.

- fan_out_factor: result rows / left rows. Above 1 means left rows were repeated, so any sum or
  mean of a left-table column is now weighted by how many partners each row had.
- left_unmatched: left rows with no partner. An inner join drops them silently (biasing the
  sample); a left join keeps them with missing right-hand values.
- observed vs expected relationship: e.g. expected many_to_one (each order has one customer)
  but the right key is duplicated, so the join is really many_to_many.

ok=False when the join repeats rows that were expected to stay unique, or drops/orphans more
than `max_unmatched_pct` percent of rows.

Returns `JoinReport` with: `df`, `how`, `observed`, `expected`, `rows_left`, `rows_right`, `rows_result`, `fan_out_factor`, `left_unmatched`, `left_unmatched_pct`, `right_unmatched`, `duplicate_left_keys`, `duplicate_right_keys`

## `execute_join_plan`

`from statkit.relational import execute_join_plan`

```python
execute_join_plan(tables: 'dict', plan: 'JoinPlan', how: 'str' = 'left', columns: 'dict | None' = None) -> 'JoinRun'
```

Run a JoinPlan step by step through validate_join, stopping at the first step that fails, and
return the final table with a per-step record of rows before/after and unmatched rows.
`columns` ({table: [cols]}) keeps only the columns the question needs from each table (keys are
added automatically), which avoids name clashes and very wide tables.

Returns `JoinRun` with: `df`, `steps`, `grain`

## `check_unit_of_analysis`

`from statkit.relational import check_unit_of_analysis`

```python
check_unit_of_analysis(df: 'DataFrame', entity_col: 'str', columns: 'list[str] | None' = None) -> 'UnitReport'
```

Check whether one entity (a customer, an order) now spans several rows, as it does after a
one-to-many join. If it does, rows are not independent: a t-test or correlation at row level
counts an entity with 10 rows ten times. `entity_level_columns` lists the columns that are
constant within each entity (attributes of the entity, e.g. customer age): analyse those after
aggregating to one row per entity.

Returns `UnitReport` with: `entity`, `n_rows`, `n_entities`, `max_rows_per_entity`, `mean_rows_per_entity`, `repeated`, `entity_level_columns`

## `detect_aggregation_reversal`

`from statkit.relational import detect_aggregation_reversal`

```python
detect_aggregation_reversal(df: 'DataFrame', x: 'str', y: 'str', group_col: 'str', min_group_size: 'int' = 10) -> 'ReversalReport'
```

Compare the association of x with y overall (pooled) against within each level of group_col.
A reversal (Simpson's paradox) is when the pooled association points one way and the
within-group associations the other: the grouping variable is confounding the pooled result
(Pearl 2014). Joins make this easy to miss, because the confounder often sits in another table.

x with two levels: difference in mean y between the levels (works for 0/1 outcomes too: a
difference in rates). x numeric: Pearson r between x and y.

Returns `ReversalReport` with: `measure`, `pooled_effect`, `within_effect`, `group_effects`, `reversed`, `n_groups_opposite`

## `validate_grounding`

`from statkit.validation import validate_grounding`

```python
validate_grounding(profile: 'DatasetProfile | None', referenced_columns: 'list[str]', code: 'str | None' = None, known_tables: 'list[str] | None' = None, schema: 'dict[str, list[str]] | None' = None) -> 'GroundingReport'
```

Check that every column (and, if given, every table name) a plan or a
piece of generated code references actually exists in the profiled
schema, before that code is executed.

This is the direct fix for nvAgent's failure mode in DOCUMENTATION.md
§6.6: it emitted syntactically valid SQL, chose the right chart, and
expressed the right statistical intent (a correlation ranking), and
still produced nothing, because it queried a table called `wine` when
the actual table was `wine_quality`. A validator scoped to "does this
reference something that exists" would have caught that in one line,
which is the whole argument for this tool existing separately from
execution-feedback repair (§6.5's point: a validator can only fix what
it is scoped to touch, so make grounding one of the things it is
scoped to touch).

Multi-table plans: pass `schema` ({table: [columns]}, e.g. from discover_tables(...).schema)
and reference columns as "table.column". A dotted reference is checked against that table
only, so `orders.customer_id` fails if the column lives in `customers` but not `orders`.

Returns `GroundingReport` with: `referenced`, `unknown`, `known_tables`, `unknown_tables`

## `verify_numeric_consistency`

`from statkit.verification import verify_numeric_consistency`

```python
verify_numeric_consistency(claimed: 'float', recomputed: 'float', rel_tol: 'float' = 0.001) -> 'NumericConsistencyReport'
```

Compare the number a narrative claims against an independent recompute and
name the *kind* of disagreement. Targets DOCUMENTATION §7.1 #2: the model
printed 38.38% against a ground truth of 0.3838 -- the same quantity in a
different unit, which a plain equality check scores as simply wrong.

Returns `NumericConsistencyReport` with: `claimed`, `recomputed`, `relation`, `scale_factor`
