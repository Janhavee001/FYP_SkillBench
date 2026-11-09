---
name: visualization
description: Choose, draw and explain a chart for a question about tabular data - histogram, bar chart, line chart, scatter plot, box plot, correlation heatmap or count heatmap - chosen from the roles of the columns (numeric, categorical, temporal), drawn with matplotlib, checked against the real columns, and explained by what the data shows rather than how the chart looks. Use when the user asks to plot, chart, graph, visualize or "show" data, or when another analysis needs a figure. Uses the bundled statkit Python tools.
---

# Visualization

A chart is chosen by the question and the column types, not by taste. It is explained by its
finding ("fares are strongly right-skewed: median £14, a few above £500"), never by its
styling ("blue bars on a white background").

## When to use

- "Plot...", "Show me...", "Visualize...", "Make a chart of...", or any analysis whose answer
  reads better as a figure.

## Setup (run once per conversation)

```python
import sys
SKILL_DIR = "<this skill's folder>"  # the folder you read this SKILL.md from
sys.path.insert(0, f"{SKILL_DIR}/scripts")
from statkit.environment import check_environment
print(check_environment().warnings)
```

Load data with `statkit.loading.load_dataset`. Tool signatures are in `reference/tools.md`.

## Chart choice (what `recommend_chart` does)

| Columns requested | Chart |
| --- | --- |
| one numeric | histogram |
| one categorical | bar chart of counts |
| numeric x numeric | scatter (pass `color=` for a third, categorical variable) |
| date x numeric | line chart |
| categorical x numeric | box plot (bar of means when a group has fewer than ~20 rows) |
| categorical x categorical | count heatmap |
| 3+ numeric (`columns=[...]`) | correlation heatmap |

## Procedure

1. **Roles:** `column_summary_for_viz(df)` labels each column quantitative, ordinal-like,
   nominal, temporal or ID-like. ID-like columns make meaningless axes.
2. **Recommend:** `recommend_chart(df, x, y=None, color=None)` or
   `recommend_chart(df, columns=[...])`. Read `rationale` and `warnings` (overplotting, too many
   categories, too many colours).
3. **Render:** `render_chart(df, spec, path, title=...)`. It re-checks that every column exists
   in this dataframe. Save to the outputs folder and show the image.
4. **Explain:** `explain_chart(df, spec).finding`. Each sentence is backed by a computed number
   in `.facts`. Add the question-specific meaning yourself, without adding numbers that are not
   in `facts`.
5. If the chart supports a statistical claim (a difference, a correlation), run the matching
   skill for the test. A chart alone is not evidence of a difference.

## Common errors to avoid

- Describing the palette, axes or chart type instead of the data.
- A line chart for unordered categories, or a pie chart for many categories.
- Plotting an ID column, or more than ~10 colours.
- Truncated axes that exaggerate small differences. Bars start at zero.
- Claiming a difference from overlapping box plots without a test.

## Report format

The image, then one or two sentences of finding (from `explain_chart`, adapted to the
question), then one sentence on what the chart cannot show.

## Worked example

```python
from statkit.loading import load_dataset
from statkit.exploration import column_summary_for_viz
from statkit.visualization import recommend_chart, render_chart, explain_chart

df = load_dataset("<path to the uploaded file>").df
print({c: v["role"] for c, v in column_summary_for_viz(df).columns.items()})
for spec in [recommend_chart(df, "alcohol", "density"),
             recommend_chart(df, "quality"),
             recommend_chart(df, columns=["alcohol", "density", "ph", "quality"])]:
    out = render_chart(df, spec, f"<outputs folder>/{spec.chart_type}.png")
    print(spec.chart_type, out.ok, explain_chart(df, spec).finding)
```
