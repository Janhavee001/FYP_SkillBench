# Reliable Data Analysis with LLM Agents

**Skill design, selection, composition and validation.** Final Year Project, Nanyang Technological University.
Janhavee Singh.

LLM agents can write analysis code, but on real data they often give confident wrong answers: they analyse at the
wrong grain, treat a missing-value code as data, or join tables in a way that duplicates rows. This project
builds a tested library of statistical tools and *skills* (procedures an agent can load on demand), a benchmark
of real, messy, multi-table data that measures whether they help, and a validation step that checks an answer
before it is reported.

| Research question | Finding |
|---|---|
| **RQ1:** do skills make analysis more correct? | Yes for the smaller model (gpt-oss-20b): +5.0 points on SkillBench (p = 0.012). No change for gpt-oss-120b, which is near its ceiling on SkillBench and shows no significant difference on StatABench either. |
| **RQ2:** does the agent pick the right skill? | Rarely. The 20b model loads a skill for 9% of SkillBench questions, and giving it the correct skill (oracle) adds nothing, so its gain does not come from the skill text. |
| **RQ3:** does validation catch and repair errors? | For the 20b model: +6.3 points on SkillBench (p < 0.001) and +3.9 on StatABench (p = 0.013). For the 120b model the critic's false alarms cancel its fixes. |

Full results: [docs/paper/Paper_Draft.md](docs/paper/Paper_Draft.md).

## Repository layout

```
.
├── Skills/                      the tool library and skills
│   ├── skills/                  statkit: 61 typed statistical tools, no LLM inside
│   ├── claude_skills/           the eight skills (SKILL.md) and build.py, which packages them into dist/
│   ├── tests/                   pytest suite (no API key needed)
│   ├── demo/                    demo datasets and the synthetic shop database
│   ├── deck/                    builds the slide deck and speaker script from the results
│   └── CATALOG.md               every tool, with its source in the literature
├── Benchmark/
│   ├── skillbench/              SkillBench, the StatABench runner, the validator, scoring and analysis
│   │   ├── questions/           the 206 questions with reference answers
│   │   ├── results/             scores, statistics, summaries, figures
│   │   └── pipeline.sh          the full pipeline
│   └── benchmark-fyp/           phase 1: five published systems on a shared test set
│       ├── harness/             shared rubric, LLM client and one runner per system
│       ├── datasets/            Titanic and Wine Quality, with computed answers
│       └── results/             phase-1 results
├── docs/
│   ├── paper/                   report draft
│   ├── slides/                  decks (current and v1) and the speaker script
│   └── proposal/                proposal and research plan
└── requirements.txt
```

Each main folder has its own README with details.

## Setup

Python 3.11.

```bash
python -m pip install -r requirements.txt
cp Benchmark/benchmark-fyp/.env.example Benchmark/benchmark-fyp/.env   # then add OPENROUTER_API_KEY
```

The tests and the demo need no API key:

```bash
cd Skills
python -m pytest tests/ -q
python demo_pipeline.py
```

## Reproducing the results

The benchmarks need the data and the third-party benchmark code, which are not stored in this repository.

1. **Data.** `python Benchmark/skillbench/build_data.py` downloads the eleven public datasets (about 1.3 GB) and
   `python Benchmark/skillbench/make_synthetic.py` writes the synthetic ones.
2. **External benchmarks**, cloned into `Benchmark/benchmark-fyp/repos/` at the commits used:

   | Repo | Commit | Used for |
   |---|---|---|
   | [youxin01/StatABench](https://github.com/youxin01/StatABench) | `3bba4c3` | external benchmark (phase 1 and final) |
   | [THUDM/DataSciBench](https://github.com/THUDM/DataSciBench) | `84ef3d4` | phase 1 |
   | [microsoft/lida](https://github.com/microsoft/lida) | `d892e20` | phase 1 |
   | [microsoft/data-formulator](https://github.com/microsoft/data-formulator) | `2fb5016e` (tag 0.2), cloned as `data-formulator-v2` | phase 1 |
   | [geliang0114/nvAgent](https://github.com/geliang0114/nvAgent) | `a37209e` | phase 1 |

   ```bash
   git clone https://github.com/youxin01/StatABench Benchmark/benchmark-fyp/repos/StatABench
   git -C Benchmark/benchmark-fyp/repos/StatABench checkout 3bba4c3
   ```
3. **Run.** `sh Benchmark/skillbench/pipeline.sh` runs everything, from the questions to the deck. Every runner
   resumes where it stopped.

To check the reported numbers without rerunning any model, download `skillbench_transcripts.zip` from the
release (all answer transcripts and cached judge verdicts), unzip it into `Benchmark/skillbench/results/`, then
run `python score_v2.py` and `python analyze_v2.py` in `Benchmark/skillbench/`. Rescoring makes no API calls.

## Models

- **Under test:** `openai/gpt-oss-120b` and `openai/gpt-oss-20b`, through OpenRouter, temperature 0.7, three runs
  per agent condition.
- **Judge:** `qwen/qwen3-235b-a22b-2507`, from a different model family, used only for answer extraction, the
  quality rubric and StatABench's own grading prompt.

## Known limitations

- The answer-key review (30 sampled questions) was done by one reviewer; a second, independent reviewer is
  planned.
- The validator's critic is the model under test; a critic from another model family has not been tested yet.
- The OpenRouter provider was not pinned, so runs may have been served by different providers.
- A tool-call parsing bug affected some early answers. All affected answers were detected and rerun; see section 5
  of the paper.

## Data licences

The datasets are not redistributed here. Each keeps its own licence (listed in
[Benchmark/skillbench/README.md](Benchmark/skillbench/README.md)); some (Olist, IMDb, MovieLens) are for
non-commercial or research use only.
