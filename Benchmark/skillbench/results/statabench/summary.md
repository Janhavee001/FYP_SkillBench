# StatABench (Stat-Closed, 198 tool items) under SkillBench conditions

Items, data and reference answers unchanged from StatABench; graded with StatABench's own comparison prompt (judge qwen3-235b, temperature 0).

| Model | Condition | Accuracy (mean of runs) | SD across runs | 95% CI | Runs |
|---|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) | 61.6% | 0.0 pts | 54.5% to 68.2% | 1 |
| gpt-oss-120b | Agent | 64.5% | 0.8 pts | 58.4% to 70.2% | 3 |
| gpt-oss-120b | Agent + skills | 63.8% | 1.9 pts | 57.7% to 69.7% | 3 |
| gpt-oss-120b | Agent + oracle skill | 64.8% | 0.8 pts | 58.9% to 70.5% | 3 |
| gpt-oss-120b | Agent + skills + validator | 63.1% | 0.5 pts | 57.1% to 69.2% | 3 |
| gpt-oss-20b | Plain (one shot) | 48.5% | 0.0 pts | 41.4% to 55.6% | 1 |
| gpt-oss-20b | Agent | 54.4% | 6.1 pts | 48.7% to 59.9% | 3 |
| gpt-oss-20b | Agent + skills | 56.6% | 3.8 pts | 50.8% to 62.1% | 3 |
| gpt-oss-20b | Agent + oracle skill | 59.1% | 2.5 pts | 53.2% to 64.8% | 3 |
| gpt-oss-20b | Agent + skills + validator | 60.4% | 3.0 pts | 54.7% to 66.2% | 3 |

## Paired comparisons

| Model | Comparison | Difference | 95% CI | p (permutation) |
|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) -> Agent | +2.9 pts | -1.9 to +7.4 | 0.259 |
| gpt-oss-120b | Agent -> Agent + skills | -0.7 pts | -4.2 to +2.9 | 0.782 |
| gpt-oss-120b | Agent -> Agent + oracle skill | +0.3 pts | -3.7 to +4.4 | 0.934 |
| gpt-oss-120b | Agent + skills -> Agent + oracle skill | +1.0 pts | -2.7 to +4.7 | 0.658 |
| gpt-oss-120b | Plain (one shot) -> Agent + skills | +2.2 pts | -2.7 to +7.1 | 0.416 |
| gpt-oss-120b | Agent + skills -> Agent + skills + validator | -0.7 pts | -3.2 to +1.9 | 0.706 |
| gpt-oss-20b | Plain (one shot) -> Agent | +5.9 pts | -0.5 to +12.1 | 0.0771 |
| gpt-oss-20b | Agent -> Agent + skills | +2.2 pts | -2.4 to +6.7 | 0.383 |
| gpt-oss-20b | Agent -> Agent + oracle skill | +4.7 pts | -0.5 to +9.9 | 0.0959 |
| gpt-oss-20b | Agent + skills -> Agent + oracle skill | +2.5 pts | -1.7 to +6.9 | 0.284 |
| gpt-oss-20b | Plain (one shot) -> Agent + skills | +8.1 pts | +2.4 to +14.0 | 0.0107 |
| gpt-oss-20b | Agent + skills -> Agent + skills + validator | +3.9 pts | +1.0 to +6.9 | 0.0135 |

## By StatABench task


**gpt-oss-120b**

| Task | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| ABtest | 10 | 40.0% | 43.3% | 43.3% | 36.7% | 53.3% |
| Contingency Table Test | 20 | 85.0% | 78.3% | 81.7% | 81.7% | 76.7% |
| Correlation Analysis | 23 | 69.6% | 63.8% | 58.0% | 52.2% | 59.4% |
| Descriptive Statistics | 18 | 77.8% | 83.3% | 66.7% | 81.5% | 70.4% |
| Distribution Compliance Test | 21 | 81.0% | 84.1% | 84.1% | 81.0% | 81.0% |
| EDA | 8 | 87.5% | 87.5% | 83.3% | 91.7% | 79.2% |
| High dimension data | 6 | 50.0% | 22.2% | 50.0% | 50.0% | 44.4% |
| NonpStat | 6 | 50.0% | 77.8% | 94.4% | 88.9% | 88.9% |
| Variance Test | 15 | 0.0% | 13.3% | 6.7% | 31.1% | 4.4% |
| bayesian method | 8 | 62.5% | 75.0% | 70.8% | 70.8% | 70.8% |
| causal inference | 5 | 40.0% | 53.3% | 46.7% | 40.0% | 40.0% |
| confidence interval | 9 | 88.9% | 100.0% | 96.3% | 100.0% | 100.0% |
| generalized linear model | 14 | 57.1% | 54.8% | 52.4% | 45.2% | 57.1% |
| multiple comparison | 5 | 40.0% | 53.3% | 60.0% | 60.0% | 46.7% |
| regression | 8 | 87.5% | 87.5% | 91.7% | 83.3% | 91.7% |
| survial analysis | 13 | 38.5% | 41.0% | 51.3% | 43.6% | 48.7% |
| time series | 9 | 44.4% | 55.6% | 59.3% | 66.7% | 59.3% |

**gpt-oss-20b**

| Task | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| ABtest | 10 | 30.0% | 30.0% | 43.3% | 33.3% | 53.3% |
| Contingency Table Test | 20 | 60.0% | 50.0% | 65.0% | 66.7% | 71.7% |
| Correlation Analysis | 23 | 52.2% | 68.1% | 63.8% | 63.8% | 63.8% |
| Descriptive Statistics | 18 | 72.2% | 72.2% | 77.8% | 75.9% | 77.8% |
| Distribution Compliance Test | 21 | 61.9% | 57.1% | 69.8% | 73.0% | 77.8% |
| EDA | 8 | 75.0% | 79.2% | 95.8% | 87.5% | 91.7% |
| High dimension data | 6 | 33.3% | 38.9% | 22.2% | 44.4% | 27.8% |
| NonpStat | 6 | 66.7% | 77.8% | 61.1% | 94.4% | 61.1% |
| Variance Test | 15 | 6.7% | 11.1% | 6.7% | 26.7% | 4.4% |
| bayesian method | 8 | 50.0% | 62.5% | 62.5% | 70.8% | 66.7% |
| causal inference | 5 | 40.0% | 66.7% | 53.3% | 46.7% | 53.3% |
| confidence interval | 9 | 55.6% | 59.3% | 88.9% | 88.9% | 85.2% |
| generalized linear model | 14 | 28.6% | 54.8% | 47.6% | 26.2% | 52.4% |
| multiple comparison | 5 | 60.0% | 40.0% | 46.7% | 53.3% | 46.7% |
| regression | 8 | 87.5% | 79.2% | 54.2% | 79.2% | 66.7% |
| survial analysis | 13 | 15.4% | 30.8% | 30.8% | 30.8% | 41.0% |
| time series | 9 | 33.3% | 59.3% | 51.9% | 51.9% | 63.0% |

## Skill selection (Agent + skills)

| Model | Loaded any skill | Loaded a fitting skill |
|---|---|---|
| gpt-oss-120b | 66.7% | 46.8% |
| gpt-oss-20b | 27.4% | 18.4% |

## Validator (RQ3): agent + skills answers, before and after validation

| Model | Answers | Flagged | Wrong answers flagged (detection) | Correct answers flagged (false alarm) | Fixed (wrong to right) | Broken (right to wrong) | Net |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 594 | 36.7% | 49.3% | 29.6% | 23 | 27 | -4 |
| gpt-oss-20b | 594 | 37.4% | 57.0% | 22.3% | 45 | 22 | +23 |
