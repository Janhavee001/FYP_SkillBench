---
name: data-profiling
description: Load a tabular dataset safely and profile it before any analysis - real delimiter and encoding, shape, column types, missing values, outliers, summary statistics, group summaries and data-quality problems. Use whenever a user uploads a CSV/Excel/Parquet file, asks to "summarize", "describe", "explore", "profile" or "check the quality of" a dataset, or before any other statistical skill runs on a new file. Uses the bundled statkit Python tools.
---

# Data profiling

Profile first, analyse second. Most wrong analyses are wrong before the first statistic: a
wrong delimiter collapses 12 columns into 1, a column name is guessed instead of read, or
missing values are silently dropped. This skill makes "what did I actually load" an explicit,
checked step.

## When to use

- A new dataset arrives, or the user asks what is in it.
- Before any other statistical skill runs on a file this conversation has not profiled yet.
- The user asks about missing data, outliers, data quality, or summary statistics.

**Not for:** testing a hypothesis (use `group-comparison`), relationships between variables
(`correlation-analysis`), or anything spanning several related tables (load each table here,
then handle the join explicitly before analysing).

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
env = check_environment()
print(env.versions, env.warnings)
```

Every tool returns a result object with `ok`, `warnings` and typed fields. A non-empty
`warnings` list is something to tell the user, never something to swallow. Signatures for every
tool are in `reference/tools.md`.

## Procedure

1. **Load with the delimiter sniffed, not remembered.** `load_dataset(path)` sniffs the
   delimiter and encoding first. Never pass `sep=` from memory of a "similar" public dataset.
   If `ok` is False, stop and report why.
2. **Profile the schema.** `profile_dataset(df)` gives rows, columns, dtypes, unique counts and
   missing %. Read the column names from here, and use only these names from now on.
3. **Missing values.** `missing_value_report(df)` separates structurally missing columns (>50%,
   usually unusable) from sparse ones. Report both groups.
4. **Describe the columns the user cares about.** `describe_column(df, col)` dispatches on
   dtype. Use `frequency_table` for categories (it reports its denominator) and
   `groupby_aggregate` for per-group summaries (it reports group sizes).
5. **Outliers.** `detect_outliers_iqr(df, col)` for key numeric columns. Report the rows; do not
   delete them unless the user asks and you say what changed.
6. **Only if asked to clean:** `impute_missing` (never silently; say the strategy and how many
   values were filled). Imputing a structurally missing column invents data, so say so.
7. **Before reporting any number,** check that each claimed figure came from a tool output.
   Restating a figure in other units? Run `verify_numeric_consistency(claimed, recomputed)`.

## Common errors to avoid

- Guessing a column name (`"Survived"` vs `"survived"`). Use `validate_grounding(profile, [cols])`
  when unsure.
- Reporting a proportion without its denominator when values are missing.
- Treating an ID column or a 0/1 column as a continuous measurement.
- Calling an integer column with a few levels (e.g. `pclass`) "numeric" without saying it is
  really ordinal.

## Interpretation guidance

- A data-quality finding is a finding: lead with the problems that limit what can be concluded.
- Say "n = 714 of 891 passengers have an age" rather than "the mean age is 29.7".
- Mean vs median: when skew > 1, report the median as the typical value and say why.

## Report format

1. One-line summary: rows x columns, what each row is.
2. A short table of columns: type, missing %, key statistic.
3. Data-quality issues, most serious first.
4. What this means for the analysis the user wants next.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.profiling import profile_dataset, missing_value_report
from statkit.descriptive import describe_column
from statkit.exploration import frequency_table, groupby_aggregate
from statkit.cleaning import detect_outliers_iqr

data = load_dataset("<path to the uploaded file>")
print(data.ok, data.separator, data.n_rows, data.n_cols, data.warnings)
df = data.df

profile = profile_dataset(df)
missing = missing_value_report(df)
print("structurally missing:", missing.structurally_missing, "sparse:", missing.sparse)

print(describe_column(df, "age").stats)
print(frequency_table(df, "survived").table)
print(groupby_aggregate(df, "sex", "survived", "mean").result)
print(detect_outliers_iqr(df, "fare").n_outliers)
```
