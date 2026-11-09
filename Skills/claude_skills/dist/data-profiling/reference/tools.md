# Tool reference: data-profiling

Generated from the code by `build.py`; do not edit by hand.

Import as `from statkit.<module> import <tool>` after adding `<this skill's folder>/scripts` to `sys.path`. Every tool returns a result object with `ok` (bool), `warnings` (list of str) and the fields named below; `.to_dict()` converts it.

## `load_dataset`

`from statkit.loading import load_dataset`

```python
load_dataset(path: 'str | pathlib.Path', expected_min_columns: 'int' = 2) -> 'LoadReport'
```

Load CSV / TSV / Parquet / Excel, with the delimiter and encoding sniffed
first (detect_separator_and_encoding) instead of recalled from memory.
The frame comes back inside the result envelope so grounding warnings
travel with the data rather than being a separate step a caller can skip.

Returns `LoadReport` with: `path`, `n_rows`, `n_cols`, `separator`, `encoding`, `df`

## `detect_separator_and_encoding`

`from statkit.profiling import detect_separator_and_encoding`

```python
detect_separator_and_encoding(path: 'str | pathlib.Path', expected_min_columns: 'int' = 2) -> 'SeparatorReport'
```

Sniff the real delimiter of a CSV before anyone calls pd.read_csv on it.

Directly targets the bug in DOCUMENTATION.md §6.4: the model "remembered"
that the UCI wine dataset is semicolon-separated and wrote
`pd.read_csv(path, sep=';')` against a file that was actually
comma-separated. That parses without error -- it just returns one column
-- so nothing downstream raises until a much later, more confusing
`KeyError: 'quality'`. Sniffing the file's *actual* delimiter up front,
rather than trusting what the model recalls about a similarly-named
public dataset, is the fix the benchmark's own repair loop never found
(0 of 3 repairs fixed this class of error).

Returns `SeparatorReport` with: `path`, `detected_separator`, `detected_encoding`, `confidence`, `columns_at_detected_sep`, `columns_at_comma`

## `profile_dataset`

`from statkit.profiling import profile_dataset`

```python
profile_dataset(df: 'DataFrame', expected_columns: 'int | None' = None) -> 'DatasetProfile'
```

Schema fingerprint an agent should inspect before writing any code
against a dataframe.

Generalizes StatABench's `check_column_type_is` (which checks one column
against one expected type) into a whole-frame profile, and is the
concrete answer to DataSciBench's bug #11 (§7.1): if the agent had been
made to print `profile_dataset(df).n_cols` before analysing, "1 column
where 12 were expected" would have been visible immediately instead of
surfacing as a plausible-looking analysis of dummy data.

Returns `DatasetProfile` with: `n_rows`, `n_cols`, `columns`, `dtypes`, `n_unique`, `missing_pct`

## `missing_value_report`

`from statkit.profiling import missing_value_report`

```python
missing_value_report(df: 'DataFrame', columns: 'list[str] | None' = None, structural_threshold: 'float' = 0.5) -> 'MissingValueReport'
```

Per-column missing-value breakdown, distinguishing columns that are
mostly absent (>50% missing, e.g. Titanic's `deck` at 77%) from columns
with ordinary sparse missingness (e.g. `age` at 20%), since those call
for different handling and conflating them is a real failure mode in the
shared test set's missing-data questions.

Inspired by StatABench's `check_missing_values`, extended with the
structural/sparse split so the *report* -- not just the raw percentage --
is what an agent conditions its next step on.

Returns `MissingValueReport` with: `per_column`, `structurally_missing`, `sparse`

## `describe_column`

`from statkit.descriptive import describe_column`

```python
describe_column(df: 'DataFrame', column: 'str') -> 'ColumnDescription'
```

Numeric columns get mean/median/std/min/max/skew/kurtosis; categorical
columns get mode, top-5 value counts, and cardinality. StatABench's
`calculate_statistic` returns one named statistic and leaves the caller to
know the column is numeric (a `mean` on a text column just raises); this
tool dispatches on dtype itself and returns a self-describing summary, so
an interpretation step doesn't need to re-inspect the dataframe.

Returns `ColumnDescription` with: `column`, `dtype`, `is_numeric`, `n`, `n_missing`, `stats`

## `frequency_table`

`from statkit.exploration import frequency_table`

```python
frequency_table(df: 'DataFrame', column: 'str', top_n: 'int | None' = None, include_missing: 'bool' = False) -> 'FrequencyTable'
```

value_counts with proportions. Proportions are over non-missing rows unless
include_missing=True, and the denominator is reported, because "what fraction"
questions silently change answer with the denominator (Titanic age, deck).

Returns `FrequencyTable` with: `column`, `n`, `n_missing`, `table`

## `groupby_aggregate`

`from statkit.exploration import groupby_aggregate`

```python
groupby_aggregate(df: 'DataFrame', by: 'str | list[str]', value: 'str', agg: 'str' = 'mean') -> 'GroupAggregateReport'
```

Aggregate `value` by one or more grouping columns, returning group sizes
alongside the aggregate so a mean over n=3 is never read like a mean over n=300.

Returns `GroupAggregateReport` with: `by`, `value`, `agg`, `result`, `group_sizes`

## `detect_outliers_iqr`

`from statkit.cleaning import detect_outliers_iqr`

```python
detect_outliers_iqr(df: 'DataFrame', column: 'str', k: 'float' = 1.5) -> 'OutlierReport'
```

Tukey fences (Q1 - k*IQR, Q3 + k*IQR). Reports the rows; never silently drops them.

Returns `OutlierReport` with: `column`, `q1`, `q3`, `lower_fence`, `upper_fence`, `n_outliers`, `pct_outliers`, `outlier_index`

## `impute_missing`

`from statkit.cleaning import impute_missing`

```python
impute_missing(df: 'DataFrame', columns: 'list[str] | None' = None, strategy: 'str' = 'median') -> 'TransformResult'
```

mean / median / most_frequent / constant:<value> imputation (cf. MetaGPT FillMissingValue).

Returns `TransformResult` with: `df`, `details`

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
