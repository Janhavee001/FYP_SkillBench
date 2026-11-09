# SkillBench

A benchmark for **reliable data analysis by LLM agents** on real, messy, large data, both single tables and
multi-table relational data. It is used together with the published benchmark StatABench (run unchanged) to answer:

- **RQ1:** do the statistical skills (`../../Skills`) make an agent's analysis more correct?
- **RQ2:** does the agent pick the right skill?
- **RQ3:** does a validation step catch and repair analysis errors?

The phase-1 benchmark (`../benchmark-fyp`, Titanic and Wine) was too easy: a plain LLM already chose the right
method 98% of the time. SkillBench uses real datasets of up to 3 million rows, with the data-quality problems
that make careful analysis necessary.

## Datasets

Eleven public datasets, kept as published (not cleaned), plus four small synthetic databases with planted traps.
`build_data.py` downloads and prepares the real data (about 1.3 GB of downloads, cached in `data/raw/`).
`make_synthetic.py` writes `clinic`, `school` and `saas`. `shop` comes from `../../Skills/demo/make_shop_tables.py`.

| Dataset | Tables | Size | Source | Licence |
|---|---|---|---|---|
| NYC Yellow Taxi, Jan 2024 | 2 | 2.96 M trips | [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | public |
| Citi Bike, Jan 2024 | 1 | 1.89 M rides | [Citi Bike](https://citibikenyc.com/system-data) | Citi Bike data licence |
| US flights, Q1 2024 | 3 | 1.66 M flights | [BTS](https://www.transtats.bts.gov/) | public |
| Online Retail II, 2009-2011 | 1 | 1.07 M lines | [UCI 502](https://archive.ics.uci.edu/dataset/502) | CC BY 4.0 |
| MovieLens 1M | 3 | 1.0 M ratings | [GroupLens](https://grouplens.org/datasets/movielens/1m/) | research use |
| IMDb feature films | 4 | 760 k films | [IMDb datasets](https://developer.imdb.com/non-commercial-datasets/) | non-commercial |
| Olist Brazilian e-commerce | 9 | 99 k orders | [olist/work-at-olist-data](https://github.com/olist/work-at-olist-data) | CC BY-NC-SA 4.0 |
| Diabetes 130-US hospitals | 2 | 102 k stays | [UCI 296](https://archive.ics.uci.edu/dataset/296) | CC BY 4.0 |
| Beijing PM2.5, 2010-2014 | 1 | 43.8 k hours | [UCI 381](https://archive.ics.uci.edu/dataset/381) | CC BY 4.0 |
| F1DB | 47 | 27.7 k results | [f1db/f1db](https://github.com/f1db/f1db) | CC BY 4.0 |
| Capital Bikeshare hourly | 1 | 17.4 k hours | [UCI 275](https://archive.ics.uci.edu/dataset/275) | CC BY 4.0 |
| clinic, school, SaaS, shop (synthetic) | 3-5 each | small | this project | — |

This repository stores only the build scripts and `data/manifest.json`, not the data. Check each licence
before redistributing the data.

## Questions

`build_questions.py` (with `questions_v2.py`) writes `questions/questions.json` and a readable `questions.csv`.

- **206 questions:** 183 on real data and 23 synthetic; 65 easy, 86 medium, 55 hard; 123 multi-table.
- **140 contain a trap**, in five families: grain and unit of analysis; joins and linkage; missing and
  unrecorded values; dirty or invalid records; statistical reasoning.
- **Reference answers** are computed with pandas, scipy and statsmodels only. Nothing here imports `statkit`,
  the library the skills use, so the answer key cannot favour the system under test.
- **Careless answers:** each trap question also stores the answer the obvious shortcut gives, so a wrong answer
  can be classified as a trap fall.
- **Checks on the key:** `check_anchors.py` reproduces the record counts the data publishers document;
  `disagreements.py` audits every question where most runs agree on a different answer;
  `make_review_sheet.py` produces the sheet for a human review of a stratified sample.

Two questions (R18, CB06) are excluded from the analysis after the audit; see `results/v2/disagreements.md`.

## Conditions

All conditions use the same model, temperature, data and question text. Only the scaffolding differs.

| Condition | Scaffolding | Runs |
|---|---|---|
| `plain` | One script, run once, then an answer from its output | 1 |
| `agent` | Tool loop: run Python as often as needed (up to the step limit), then submit | 3 |
| `agent_skills` | The same loop plus `load_skill` (the eight skills) and the `statkit` library | 3 |
| `agent_skills_oracle` | As `agent_skills`, with the correct skill preloaded | 3 |
| `agent_skills_validated` | Starts from the saved `agent_skills` answer; deterministic checks and a critic, then one chance to repair (`validate.py`) | 3 |

Models: `openai/gpt-oss-120b` and `openai/gpt-oss-20b` through OpenRouter, temperature 0.7, seed 1000 + run.

## Scoring

- **Correctness (primary, deterministic):** numbers within a per-question tolerance (a percentage given as a
  proportion is accepted and flagged); labels and choices must match.
- **Judge:** `qwen/qwen3-235b-a22b-2507`, a different model family from the models under test. It is used only to
  extract a final label from free text and for the secondary quality rubric (method appropriate, caveats stated,
  no overclaiming).
- **Statistics:** 95% bootstrap CIs over questions; paired differences tested with a sign-flip permutation test.

StatABench (`statabench.py`): the 198 Stat-Closed tool items, data and reference answers are used unchanged and
graded with StatABench's own comparison prompt, run through the same harness and conditions.

## Files

| File | What it does |
|---|---|
| `build_data.py`, `make_synthetic.py` | Download and prepare the datasets |
| `build_questions.py`, `questions_v2.py` | Question bank with computed reference and careless answers |
| `check_anchors.py` | Checks the data against counts documented by the publishers |
| `run.py` | The harness: one condition, one model (tool loop, skill loading, sandboxed Python) |
| `run_all.py` | The full SkillBench design in one process |
| `statabench.py` | StatABench: `run`, `score`, `analyse`, `check` |
| `validate.py` | RQ3 validator condition on either benchmark |
| `score.py`, `score_v2.py` | Correctness and judge scoring (`score.py` holds the shared rules) |
| `analyze_v2.py` | Accuracy, paired tests, breakdowns and figures |
| `disagreements.py` | Consensus audit of the reference answers |
| `make_review_sheet.py` | Human review sheet for the reference answers |
| `quarantine_parse.py`, `quarantine_stale.py` | Move answers affected by the tool-call parsing bug aside so they are rerun |
| `pipeline.sh` | Everything above, in order |
| `analyze.py` | v1 analysis (108 questions, three conditions), kept for the v1 results |

## Results

Committed in `results/`:

- `v2/summary.md`, `v2/stats.json`, `v2/scores.csv`, `v2/figures/`, `v2/disagreements.md`: SkillBench;
- `statabench/summary.md`, `statabench/stats.json`, `statabench/scores.csv`: StatABench;
- `reference_review.md` and `.xlsx`: the human review of the answer key;
- `anchors.md`: the data checks;
- `scores.csv`, `summary.md`, `figures/`: the v1 run.

The per-answer files are not in git: the answer transcripts (`v2/gpt-oss-*/`, `statabench/gpt-oss-*/`), the
cached judge verdicts (`*/judgements/`) and the v1 answers (`plain/`, `agent/`, `agent_skills/`). They are
attached to the GitHub release as `skillbench_transcripts.zip`; unzip it into `results/` to restore them. With the
cached verdicts in place, rescoring makes no API calls.

## Reproduce

```bash
sh pipeline.sh
```

The API key comes from `../benchmark-fyp/.env` (`OPENROUTER_API_KEY`). StatABench must be cloned at
`../benchmark-fyp/repos/StatABench`. A full run produces about 10,500 answers across both benchmarks, each from
several model calls, so check API credit first (`run_all.py --ids` and `statabench.py --limit` run a subset).
