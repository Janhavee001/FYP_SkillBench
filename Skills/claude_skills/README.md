# Claude skills: statistical analysis

Eight Agent Skills for the Claude app: one per analysis type in the FYP research plan, plus one for
data spread across several related tables. Each
skill is a `SKILL.md` that encodes *when* to use a method, *what assumptions* to check, *which
errors* to avoid and *how* to interpret the result. The skill calls the tested Python tools in
`../skills/` for the computation.

| Skill | Use it for |
| --- | --- |
| `data-profiling` | Loading a file safely, schema, missing values, outliers, summaries |
| `distribution-analysis` | Shape, centre, spread, normality, intervals, histograms and box plots |
| `correlation-analysis` | Relationships between numeric variables, heatmaps, no causal overclaiming |
| `group-comparison` | t-tests, rank tests, chi-square, proportions, power, multiple testing, survival curves |
| `regression-analysis` | OLS, GLMs, robust and regularized regression, Cox, propensity matching, DiD |
| `time-series-analysis` | Trend, seasonality (STL), stationarity, ARIMA forecasts |
| `visualization` | Choosing, drawing and explaining charts |
| `relational-data-analysis` | Several related tables: keys, safe joins, fan-out, orphans, Simpson's paradox |

## Build

```bash
cd Skills
python claude_skills/build.py
```

This writes `claude_skills/dist/<skill>.zip`. Each zip holds:

- `SKILL.md`, as written in `claude_skills/<skill>/`.
- `reference/tools.md`, generated from the code.
- `scripts/statkit/`, a copy of the tool library.

Every skill carries its own copy of the library, so each one works when uploaded alone.

## Upload to the Claude app

1. Settings → Capabilities: turn on code execution (skills run inside it).
2. Settings → Capabilities → Skills → Upload skill, then choose one zip. Repeat for each skill.
3. In a chat, upload a CSV and ask a question. Claude loads the matching skill from its
   description.

Menu names in the app change from time to time. If these steps don't match what you see, check
Anthropic's current help page on using skills in Claude.

## Testing

`tests/test_claude_skills.py` builds the skills and runs every Python block in every built
`SKILL.md` in a subprocess outside the repo, so only the bundled copy can be imported. That
catches any example that would crash in the app.

What it cannot test is the app sandbox's own packages. Each skill's setup step runs
`check_environment()`, which reports any missing package (statsmodels is the most likely one)
and the tools that need it.

## For the RQ1 experiment

Baseline is the same chat with no skills enabled; the treatment is the same questions with
these skills enabled. Keep the model, prompts and files identical across conditions.
