# SkillBench v2 results

181 real-data questions (11 datasets) and 23 synthetic appendix questions. Models: gpt-oss-120b, gpt-oss-20b. Temperature 0.7, 3 runs per condition.


## Accuracy on real data (headline)

| Model | Condition | Accuracy (mean of runs) | SD across runs | 95% CI | Runs | Fell into trap |
|---|---|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) | 90.6% | 0.0 pts | 86.2% to 94.5% | 1 | 3.4% |
| gpt-oss-120b | Agent | 93.0% | 2.8 pts | 90.1% to 95.8% | 3 | 4.5% |
| gpt-oss-120b | Agent + skills | 92.8% | 1.5 pts | 89.5% to 95.8% | 3 | 5.0% |
| gpt-oss-120b | Agent + oracle skill | 91.9% | 1.2 pts | 88.6% to 94.8% | 3 | 3.9% |
| gpt-oss-120b | Agent + skills + validator | 93.2% | 1.9 pts | 90.1% to 95.9% | 3 | 3.1% |
| gpt-oss-20b | Plain (one shot) | 77.3% | 0.0 pts | 71.3% to 83.4% | 1 | 4.2% |
| gpt-oss-20b | Agent | 75.5% | 1.2 pts | 71.1% to 79.9% | 3 | 3.6% |
| gpt-oss-20b | Agent + skills | 80.5% | 4.6 pts | 75.7% to 85.1% | 3 | 5.0% |
| gpt-oss-20b | Agent + oracle skill | 78.8% | 2.6 pts | 74.6% to 82.9% | 3 | 3.9% |
| gpt-oss-20b | Agent + skills + validator | 86.7% | 0.0 pts | 82.9% to 90.4% | 3 | 3.6% |

## Paired comparisons (real data; per-question accuracy averaged over runs)

| Model | Comparison | Difference | 95% CI | p (permutation) | Questions differing |
|---|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) -> Agent | +2.4 pts | -1.3 to +6.3 | 0.258 | 28 of 181 |
| gpt-oss-120b | Agent -> Agent + skills | -0.2 pts | -2.4 to +2.0 | 1 | 20 of 181 |
| gpt-oss-120b | Agent -> Agent + oracle skill | -1.1 pts | -3.9 to +1.5 | 0.509 | 27 of 181 |
| gpt-oss-120b | Agent + skills -> Agent + oracle skill | -0.9 pts | -3.1 to +1.3 | 0.529 | 24 of 181 |
| gpt-oss-120b | Plain (one shot) -> Agent + skills | +2.2 pts | -1.8 to +6.4 | 0.327 | 28 of 181 |
| gpt-oss-120b | Agent + skills -> Agent + skills + validator | +0.4 pts | -1.7 to +2.6 | 0.867 | 19 of 181 |
| gpt-oss-20b | Plain (one shot) -> Agent | -1.8 pts | -7.4 to +3.9 | 0.564 | 91 of 181 |
| gpt-oss-20b | Agent -> Agent + skills | +5.0 pts | +1.3 to +8.7 | 0.0121 | 82 of 181 |
| gpt-oss-20b | Agent -> Agent + oracle skill | +3.3 pts | -0.4 to +7.0 | 0.102 | 83 of 181 |
| gpt-oss-20b | Agent + skills -> Agent + oracle skill | -1.7 pts | -5.5 to +2.2 | 0.453 | 70 of 181 |
| gpt-oss-20b | Plain (one shot) -> Agent + skills | +3.1 pts | -2.4 to +8.8 | 0.301 | 62 of 181 |
| gpt-oss-20b | Agent + skills -> Agent + skills + validator | +6.3 pts | +3.9 to +8.8 | 0 | 35 of 181 |

## Does the skills gain depend on model strength? (real data)

| Model | Agent | Agent + skills | Gain | Agent + oracle skill | Oracle gain |
|---|---|---|---|---|---|
| gpt-oss-120b | 93.0% | 92.8% | -0.2 pts | 91.9% | -1.1 pts |
| gpt-oss-20b | 75.5% | 80.5% | +5.0 pts | 78.8% | +3.3 pts |

## By difficulty (real data)


**gpt-oss-120b**

| difficulty | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| easy | 55 | 92.7% | 97.0% | 98.8% | 98.2% | 98.8% |
| medium | 80 | 91.2% | 92.5% | 94.2% | 93.3% | 93.8% |
| hard | 46 | 87.0% | 89.1% | 83.3% | 81.9% | 85.5% |

**gpt-oss-20b**

| difficulty | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| easy | 55 | 87.3% | 83.6% | 87.9% | 84.2% | 91.5% |
| medium | 80 | 78.8% | 77.9% | 81.7% | 81.7% | 88.8% |
| hard | 46 | 63.0% | 61.6% | 69.6% | 67.4% | 77.5% |

## By trap present (real data)


**gpt-oss-120b**

| trap present | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| False | 62 | 90.3% | 95.7% | 97.8% | 97.3% | 97.3% |
| True | 119 | 90.8% | 91.6% | 90.2% | 89.1% | 91.0% |

**gpt-oss-20b**

| trap present | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| False | 62 | 88.7% | 80.1% | 85.5% | 81.2% | 91.4% |
| True | 119 | 71.4% | 73.1% | 77.9% | 77.6% | 84.3% |

## By multi-table (real data)


**gpt-oss-120b**

| multi-table | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| False | 81 | 93.8% | 96.3% | 92.2% | 92.2% | 95.5% |
| True | 100 | 88.0% | 90.3% | 93.3% | 91.7% | 91.3% |

**gpt-oss-20b**

| multi-table | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| False | 81 | 75.3% | 77.4% | 83.5% | 78.2% | 88.1% |
| True | 100 | 79.0% | 74.0% | 78.0% | 79.3% | 85.7% |

## By category (real data)


**gpt-oss-120b**

| category | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| correlation | 16 | 93.8% | 91.7% | 87.5% | 89.6% | 85.4% |
| data_quality | 28 | 96.4% | 97.6% | 96.4% | 97.6% | 98.8% |
| descriptive | 74 | 85.1% | 91.0% | 94.1% | 91.9% | 93.2% |
| group_comparison | 35 | 94.3% | 90.5% | 86.7% | 87.6% | 88.6% |
| regression | 8 | 100.0% | 100.0% | 100.0% | 95.8% | 100.0% |
| time_series | 20 | 90.0% | 96.7% | 95.0% | 91.7% | 96.7% |

**gpt-oss-20b**

| category | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| correlation | 16 | 68.8% | 81.2% | 77.1% | 83.3% | 81.2% |
| data_quality | 28 | 67.9% | 77.4% | 77.4% | 79.8% | 83.3% |
| descriptive | 74 | 83.8% | 75.2% | 80.6% | 77.9% | 89.2% |
| group_comparison | 35 | 71.4% | 74.3% | 85.7% | 81.0% | 90.5% |
| regression | 8 | 75.0% | 70.8% | 70.8% | 79.2% | 87.5% |
| time_series | 20 | 85.0% | 73.3% | 81.7% | 73.3% | 80.0% |

## Trap types (real data, share correct)


**gpt-oss-120b**

| Trap | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| cancellations | 3 | 66.7% | 100.0% | 88.9% | 100.0% | 100.0% |
| cohort_definition | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| confounding | 3 | 66.7% | 88.9% | 66.7% | 88.9% | 66.7% |
| cross_table_consistency | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| duplicates | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| era_effects | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fan_out | 4 | 100.0% | 91.7% | 91.7% | 83.3% | 91.7% |
| grain | 14 | 92.9% | 97.6% | 90.5% | 88.1% | 90.5% |
| hidden_entity_key | 2 | 50.0% | 50.0% | 66.7% | 66.7% | 66.7% |
| interaction | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| invalid_records | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| join | 12 | 100.0% | 94.4% | 97.2% | 94.4% | 91.7% |
| missing_code | 9 | 88.9% | 88.9% | 81.5% | 85.2% | 96.3% |
| missing_data_rule | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| missing_values | 2 | 100.0% | 100.0% | 83.3% | 66.7% | 100.0% |
| multi_hop_join | 10 | 90.0% | 90.0% | 90.0% | 86.7% | 80.0% |
| multi_section_file | 1 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| multi_valued_field | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| multiple_testing | 1 | 100.0% | 33.3% | 0.0% | 33.3% | 0.0% |
| orphans | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| out_of_range_dates | 2 | 100.0% | 66.7% | 83.3% | 100.0% | 83.3% |
| parsing | 5 | 80.0% | 93.3% | 100.0% | 86.7% | 93.3% |
| population_definition | 2 | 100.0% | 100.0% | 83.3% | 66.7% | 100.0% |
| ratio_of_sums | 2 | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| skew | 11 | 100.0% | 100.0% | 100.0% | 97.0% | 100.0% |
| small_sample | 3 | 100.0% | 88.9% | 100.0% | 100.0% | 100.0% |
| time_gaps | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| trend | 2 | 100.0% | 100.0% | 100.0% | 66.7% | 100.0% |
| unclassified_finishers | 1 | 100.0% | 66.7% | 66.7% | 66.7% | 33.3% |
| unequal_exposure | 2 | 100.0% | 66.7% | 83.3% | 83.3% | 83.3% |
| unit_of_analysis | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| unrecorded_values | 3 | 33.3% | 22.2% | 22.2% | 33.3% | 55.6% |

**gpt-oss-20b**

| Trap | n | Plain (one shot) | Agent | Agent + skills | Agent + oracle skill | Agent + skills + validator |
|---|---|---|---|---|---|---|
| cancellations | 3 | 100.0% | 55.6% | 88.9% | 88.9% | 77.8% |
| cohort_definition | 1 | 100.0% | 66.7% | 100.0% | 66.7% | 100.0% |
| confounding | 3 | 66.7% | 55.6% | 44.4% | 66.7% | 66.7% |
| cross_table_consistency | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| duplicates | 3 | 100.0% | 77.8% | 88.9% | 77.8% | 77.8% |
| era_effects | 1 | 100.0% | 66.7% | 66.7% | 100.0% | 100.0% |
| fan_out | 4 | 50.0% | 66.7% | 66.7% | 75.0% | 66.7% |
| grain | 14 | 78.6% | 76.2% | 78.6% | 76.2% | 90.5% |
| hidden_entity_key | 2 | 50.0% | 50.0% | 66.7% | 33.3% | 66.7% |
| interaction | 1 | 100.0% | 33.3% | 100.0% | 100.0% | 100.0% |
| invalid_records | 6 | 66.7% | 72.2% | 77.8% | 77.8% | 88.9% |
| join | 12 | 100.0% | 88.9% | 91.7% | 88.9% | 100.0% |
| missing_code | 9 | 22.2% | 70.4% | 63.0% | 66.7% | 85.2% |
| missing_data_rule | 1 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| missing_values | 2 | 50.0% | 66.7% | 50.0% | 66.7% | 50.0% |
| multi_hop_join | 10 | 50.0% | 63.3% | 66.7% | 76.7% | 76.7% |
| multi_section_file | 1 | 0.0% | 0.0% | 0.0% | 33.3% | 0.0% |
| multi_valued_field | 3 | 100.0% | 55.6% | 88.9% | 88.9% | 88.9% |
| multiple_testing | 1 | 0.0% | 66.7% | 100.0% | 33.3% | 100.0% |
| orphans | 1 | 100.0% | 66.7% | 100.0% | 100.0% | 100.0% |
| out_of_range_dates | 2 | 100.0% | 66.7% | 100.0% | 66.7% | 100.0% |
| parsing | 5 | 80.0% | 80.0% | 80.0% | 86.7% | 86.7% |
| population_definition | 2 | 50.0% | 50.0% | 33.3% | 66.7% | 50.0% |
| ratio_of_sums | 2 | 100.0% | 100.0% | 100.0% | 66.7% | 100.0% |
| skew | 11 | 90.9% | 97.0% | 100.0% | 97.0% | 100.0% |
| small_sample | 3 | 66.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| time_gaps | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| trend | 2 | 100.0% | 83.3% | 83.3% | 66.7% | 83.3% |
| unclassified_finishers | 1 | 0.0% | 33.3% | 0.0% | 33.3% | 33.3% |
| unequal_exposure | 2 | 50.0% | 50.0% | 66.7% | 50.0% | 66.7% |
| unit_of_analysis | 6 | 66.7% | 94.4% | 94.4% | 88.9% | 94.4% |
| unrecorded_values | 3 | 33.3% | 0.0% | 11.1% | 22.2% | 11.1% |

## Skill selection, Agent + skills (real data)

| Model | Loaded any skill | Loaded the oracle skill | Accuracy when loaded | Accuracy when not |
|---|---|---|---|---|
| gpt-oss-120b | 38.3% | 30.2% | 94.2% | 91.9% |
| gpt-oss-20b | 9.4% | 7.9% | 80.4% | 80.5% |

## Analytic quality (judge: different model family; real data)

| Model | Condition | Method appropriate | Caveats stated | No overclaiming |
|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) | 95.6% | 87.3% | 99.4% |
| gpt-oss-120b | Agent | 94.8% | 56.6% | 98.5% |
| gpt-oss-120b | Agent + skills | 95.0% | 38.7% | 98.2% |
| gpt-oss-120b | Agent + oracle skill | 94.3% | 41.9% | 98.3% |
| gpt-oss-120b | Agent + skills + validator | 94.8% | 39.4% | 98.9% |
| gpt-oss-20b | Plain (one shot) | 86.2% | 72.9% | 96.1% |
| gpt-oss-20b | Agent | 79.6% | 47.7% | 96.1% |
| gpt-oss-20b | Agent + skills | 83.4% | 39.7% | 96.1% |
| gpt-oss-20b | Agent + oracle skill | 83.2% | 42.6% | 96.1% |
| gpt-oss-20b | Agent + skills + validator | 89.1% | 41.5% | 97.6% |

## Cost per question (mean, real data)

| Model | Condition | Code runs | Code errors | Prompt tokens | Completion tokens | Seconds |
|---|---|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) | 1.0 | 0.04 | 2,374 | 991 | 21 |
| gpt-oss-120b | Agent | 2.5 | 0.08 | 9,055 | 1,201 | 78 |
| gpt-oss-120b | Agent + skills | 2.4 | 0.17 | 19,039 | 1,214 | 114 |
| gpt-oss-120b | Agent + oracle skill | 3.5 | 0.44 | 52,589 | 1,802 | 131 |
| gpt-oss-120b | Agent + skills + validator | 2.7 | 0.17 | 24,763 | 2,023 | 131 |
| gpt-oss-20b | Plain (one shot) | 1.0 | 0.07 | 2,281 | 971 | 21 |
| gpt-oss-20b | Agent | 2.5 | 0.24 | 11,770 | 1,425 | 63 |
| gpt-oss-20b | Agent + skills | 2.7 | 0.21 | 19,127 | 1,304 | 174 |
| gpt-oss-20b | Agent + oracle skill | 3.4 | 0.49 | 59,730 | 1,533 | 134 |
| gpt-oss-20b | Agent + skills + validator | 3.3 | 0.21 | 26,876 | 5,628 | 305 |

## Validator (RQ3): agent + skills answers, before and after validation

| Model | Answers | Flagged | Wrong answers flagged (detection) | Correct answers flagged (false alarm) | Fixed (wrong to right) | Broken (right to wrong) | Net |
|---|---|---|---|---|---|---|---|
| gpt-oss-120b | 543 | 17.7% | 56.4% | 14.7% | 15 | 13 | +2 |
| gpt-oss-20b | 543 | 29.3% | 65.1% | 20.6% | 39 | 5 | +34 |

## Appendix: synthetic planted-trap databases

| Model | Condition | Accuracy (mean of runs) | SD across runs | 95% CI | Runs | Fell into trap |
|---|---|---|---|---|---|---|
| gpt-oss-120b | Plain (one shot) | 65.2% | 0.0 pts | 43.5% to 82.6% | 1 | 15.8% |
| gpt-oss-120b | Agent | 85.5% | 2.5 pts | 72.5% to 97.1% | 3 | 14.0% |
| gpt-oss-120b | Agent + skills | 82.6% | 0.0 pts | 66.7% to 95.7% | 3 | 15.8% |
| gpt-oss-120b | Agent + oracle skill | 81.2% | 2.5 pts | 65.2% to 95.7% | 3 | 17.5% |
| gpt-oss-120b | Agent + skills + validator | 79.7% | 2.5 pts | 63.8% to 94.2% | 3 | 17.5% |
| gpt-oss-20b | Plain (one shot) | 73.9% | 0.0 pts | 56.5% to 91.3% | 1 | 10.5% |
| gpt-oss-20b | Agent | 72.5% | 6.6 pts | 56.5% to 87.0% | 3 | 14.0% |
| gpt-oss-20b | Agent + skills | 72.5% | 9.1 pts | 56.5% to 87.0% | 3 | 15.8% |
| gpt-oss-20b | Agent + oracle skill | 78.3% | 0.0 pts | 62.3% to 91.3% | 3 | 8.8% |
| gpt-oss-20b | Agent + skills + validator | 78.3% | 7.5 pts | 63.8% to 89.9% | 3 | 12.3% |
