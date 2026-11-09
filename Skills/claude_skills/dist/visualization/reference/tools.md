# Tool reference: visualization

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

## `column_summary_for_viz`

`from statkit.exploration import column_summary_for_viz`

```python
column_summary_for_viz(df: 'DataFrame', max_samples: 'int' = 3) -> 'VizColumnSummary'
```

Per-column role (quantitative / temporal / nominal / ordinal-ish / id-like), range and sample
values -- the compact summary a chart recommender needs and nothing more (cf. LIDA summarize()).

Returns `VizColumnSummary` with: `columns`

## `recommend_chart`

`from statkit.visualization import recommend_chart`

```python
recommend_chart(df: 'DataFrame', x: 'str | None' = None, y: 'str | None' = None, color: 'str | None' = None, columns: 'list[str] | None' = None) -> 'ChartSpec'
```

Pick a chart type from the roles of the requested columns (cf. LIDA goal()+visualize(), rule-based).
Pass `columns` (3+ numeric columns) instead of x/y to get a correlation heatmap.

Returns `ChartSpec` with: `chart_type`, `x`, `y`, `color`, `columns`, `aggregate`, `rationale`

## `render_chart`

`from statkit.visualization import render_chart`

```python
render_chart(df: 'DataFrame', spec: 'ChartSpec', path: 'str | pathlib.Path', title: 'str | None' = None) -> 'RenderReport'
```

Draw a ChartSpec to a PNG with matplotlib (cf. LIDA execute()). Grounding is re-checked at draw time,
so a spec built against one frame cannot silently be drawn against another.

Returns `RenderReport` with: `path`, `chart_type`, `n_points`

## `explain_chart`

`from statkit.visualization import explain_chart`

```python
explain_chart(df: 'DataFrame', spec: 'ChartSpec') -> 'ChartExplanation'
```

State what the chart *shows about the data*, computed from the data, never from the spec's styling
(cf. LIDA explain(), redefined per §6.2). Every sentence is backed by a number in `facts`.

Returns `ChartExplanation` with: `chart_type`, `finding`, `facts`

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
