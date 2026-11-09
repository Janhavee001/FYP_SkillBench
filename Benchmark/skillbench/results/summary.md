# SkillBench results

108 questions answered under all 3 conditions; model `openai/gpt-oss-120b`, temperature 0.

## Accuracy

| Condition | Correct | 95% CI | Fell into trap | Unit slips | Harness errors |
|---|---|---|---|---|---|
| Plain (one shot) | 86.1% (93/108) | 79.6% to 92.6% | 5.6% of trap questions | 0 | 0 |
| Agent (no skills) | 93.5% (101/108) | 88.9% to 97.2% | 2.8% of trap questions | 0 | 0 |
| Agent + skills | 95.4% (103/108) | 90.7% to 99.1% | 2.8% of trap questions | 0 | 0 |

## Paired comparisons (McNemar exact test, same questions)

| Comparison | Only first correct | Only second correct | p |
|---|---|---|---|
| Plain (one shot) vs Agent (no skills) | 2 | 10 | 0.0386 |
| Agent (no skills) vs Agent + skills | 2 | 4 | 0.688 |
| Plain (one shot) vs Agent + skills | 1 | 11 | 0.00635 |

## By difficulty

| difficulty | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| easy | 37 | 100.0% | 100.0% | 97.3% |
| hard | 25 | 68.0% | 80.0% | 92.0% |
| medium | 46 | 84.8% | 95.7% | 95.7% |

## By trap present

| trap present | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| False | 36 | 100.0% | 97.2% | 97.2% |
| True | 72 | 79.2% | 91.7% | 94.4% |

## By multi-table

| multi-table | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| False | 64 | 84.4% | 93.8% | 95.3% |
| True | 44 | 88.6% | 93.2% | 95.5% |

## By category

| category | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| causal | 1 | 0.0% | 0.0% | 0.0% |
| correlation | 10 | 100.0% | 90.0% | 100.0% |
| data_quality | 19 | 84.2% | 100.0% | 100.0% |
| descriptive | 37 | 89.2% | 94.6% | 97.3% |
| group_comparison | 21 | 81.0% | 90.5% | 95.2% |
| regression | 6 | 83.3% | 100.0% | 100.0% |
| time_series | 14 | 85.7% | 92.9% | 85.7% |

## By dataset

| dataset | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| beijing_pm25 | 10 | 90.0% | 100.0% | 90.0% |
| bike_sharing | 12 | 100.0% | 100.0% | 100.0% |
| diabetes | 15 | 66.7% | 93.3% | 100.0% |
| f1db | 14 | 85.7% | 85.7% | 92.9% |
| movielens | 12 | 100.0% | 100.0% | 100.0% |
| nyc_taxi | 13 | 84.6% | 84.6% | 84.6% |
| olist | 13 | 92.3% | 100.0% | 100.0% |
| online_retail | 14 | 85.7% | 92.9% | 100.0% |
| shop | 5 | 60.0% | 80.0% | 80.0% |

## Trap questions: share correct by trap type

| Trap | n | Plain (one shot) | Agent (no skills) | Agent + skills |
|---|---|---|---|---|
| cancellations | 3 | 66.7% | 100.0% | 100.0% |
| cohort_definition | 1 | 100.0% | 100.0% | 100.0% |
| confounding | 2 | 50.0% | 100.0% | 100.0% |
| cross_table_consistency | 1 | 100.0% | 100.0% | 100.0% |
| duplicates | 2 | 100.0% | 100.0% | 100.0% |
| era_effects | 1 | 100.0% | 100.0% | 100.0% |
| fan_out | 4 | 100.0% | 75.0% | 100.0% |
| grain | 8 | 87.5% | 87.5% | 100.0% |
| hidden_entity_key | 1 | 0.0% | 100.0% | 100.0% |
| interaction | 1 | 100.0% | 100.0% | 100.0% |
| invalid_records | 3 | 100.0% | 100.0% | 100.0% |
| join | 5 | 100.0% | 100.0% | 100.0% |
| missing_code | 2 | 50.0% | 100.0% | 100.0% |
| missing_data_rule | 1 | 100.0% | 100.0% | 100.0% |
| missing_values | 1 | 0.0% | 100.0% | 0.0% |
| multi_hop_join | 4 | 50.0% | 100.0% | 100.0% |
| multi_section_file | 1 | 0.0% | 100.0% | 100.0% |
| multi_valued_field | 1 | 100.0% | 100.0% | 100.0% |
| multiple_testing | 1 | 0.0% | 100.0% | 100.0% |
| orphans | 2 | 100.0% | 100.0% | 100.0% |
| out_of_range_dates | 1 | 100.0% | 100.0% | 100.0% |
| parsing | 2 | 100.0% | 100.0% | 100.0% |
| population_definition | 1 | 0.0% | 0.0% | 100.0% |
| ratio_of_sums | 1 | 100.0% | 100.0% | 100.0% |
| simpsons_paradox | 1 | 0.0% | 0.0% | 0.0% |
| skew | 7 | 100.0% | 100.0% | 100.0% |
| small_sample | 2 | 100.0% | 100.0% | 100.0% |
| time_gaps | 1 | 100.0% | 100.0% | 100.0% |
| trend | 2 | 100.0% | 100.0% | 100.0% |
| unclassified_finishers | 1 | 100.0% | 100.0% | 100.0% |
| unequal_exposure | 1 | 0.0% | 0.0% | 0.0% |
| unit_of_analysis | 6 | 83.3% | 100.0% | 100.0% |
| unrecorded_values | 1 | 0.0% | 0.0% | 0.0% |

## Analytic quality (LLM judge, share of answers)

| Condition | Method appropriate | Caveats stated | No overclaiming |
|---|---|---|---|
| Plain (one shot) | 95.4% | 94.4% | 95.4% |
| Agent (no skills) | 97.2% | 83.3% | 93.5% |
| Agent + skills | 97.2% | 77.8% | 98.1% |

## Skill selection (Agent + skills)

- Loaded at least one skill: 51.9% of questions
- Loaded a skill that fits the question category: 38.9% (75.0% of questions where any skill was loaded)
- Accuracy when a skill was loaded: 94.6% (n=56); when none was loaded: 96.2% (n=52)

| Skill | Times loaded |
|---|---|
| data-profiling | 38 |
| relational-data-analysis | 10 |
| regression-analysis | 4 |
| correlation-analysis | 3 |
| group-comparison | 2 |
| time-series-analysis | 1 |

## Cost per question (mean)

| Condition | Code runs | Code errors | Prompt tokens | Completion tokens | Seconds |
|---|---|---|---|---|---|
| Plain (one shot) | 1.0 | 0.04 | 2,625 | 936 | 24 |
| Agent (no skills) | 2.2 | 0.02 | 8,216 | 1,177 | 51 |
| Agent + skills | 2.3 | 0.20 | 20,078 | 1,204 | 62 |
