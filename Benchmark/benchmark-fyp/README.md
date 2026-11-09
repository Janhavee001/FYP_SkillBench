# benchmark-fyp

Benchmark harness for the FYP *"Reliable Data Analysis with LLM Agents: Skill
Design, Selection, Composition, and Validation"*.

Five reference papers, two of which are benchmarks run natively and three of
which are systems run against a shared test set defined here, all scored
through one common rubric.

```
benchmark-fyp/
  repos/                  cloned reference repos (+ a pinned DF2 checkout)
  harness/                shared evaluation code
    common.py             config, LLM client, normalised Record, method/chart vocab
    rubric.py             the 4-dimension shared rubric
    compare.py            reads results/*.json -> comparison.csv + terminal table
    test_rubric.py        sanity tests proving the rubric discriminates
    setup_keys.py         writes repos/StatABench/keys.json from .env
    fetch_ground_truth.py downloads DataSciBench ground truth from HF
    runners/              one runner per system
  datasets/               Titanic + Wine Quality, each with questions.json
    build_questions.py    regenerates questions.json with COMPUTED ground truth
  results/                per-system output logs + comparison.csv
  venvs/                  one virtualenv per repo (they have incompatible pins)
  report.md               final comparison + conclusion
```

## Setup

```bash
cp .env.example .env      # then put your real keys in .env (it is gitignored)
```

The harness can drive the five systems through **either** OpenAI **or** Hugging
Face, selected by `LLM_PROVIDER` in `.env`. The default is Hugging Face, whose
router at `https://router.huggingface.co/v1` is a drop-in OpenAI-compatible
endpoint, so one `HF_TOKEN` covers both the model under test and DataSciBench's
gated ground truth.

```ini
LLM_PROVIDER=huggingface
HF_TOKEN=hf_...
BENCH_MODEL=openai/gpt-oss-120b
JUDGE_MODEL=openai/gpt-oss-120b
```

Two things that will otherwise bite you:

* **A read-only token is not enough.** Inference needs a *fine-grained* token
  with **"Make calls to Inference Providers"** enabled
  (<https://huggingface.co/settings/tokens>). A token made for downloading
  datasets will return `401 Invalid username or password`.
* **`BENCH_MODEL` and `JUDGE_MODEL` must both be Hugging Face ids** (`org/model`)
  when `LLM_PROVIDER=huggingface`. A `.env` copied from an older template that
  still says `JUDGE_MODEL=gpt-4o-mini` would run every system successfully and
  only fail at scoring time, after the credits are gone. `require_key()` refuses
  to start in that state.

Model ids may carry a routing suffix: `:fastest` (the default), `:cheapest`, or
a named provider such as `:groq` — e.g. `openai/gpt-oss-120b:cheapest`.

Whichever provider is selected, `harness/common.py` exports the resolved
credentials back into `OPENAI_API_KEY` / `OPENAI_BASE_URL`, because some
downstream libraries can only be redirected through the environment.

> **On comparability:** running an open-weights model instead of GPT-4o-mini
> does not invalidate anything, but the absolute numbers will not match the
> papers' published figures (StatABench reports GPT-5.1 at 68.6% on
> Stat-Closed). This harness compares *architectures on a fixed model*, not
> published scores. Say so explicitly in `report.md`.

The virtualenvs under `venvs/` already exist. Each repo pins mutually
incompatible dependencies (LIDA wants a 2023 llmx stack, MetaGPT wants 2024
pins, nvAgent wants `openai==1.51.2`), so **there is no single environment that
runs all five** — every command below names the interpreter it needs.

## Running

Order matters only in that `compare.py` reads whatever results exist.

```bash
# 0. shared datasets (already built; only re-run if you edit a CSV or add questions)
venvs/harness/Scripts/python.exe datasets/build_questions.py

# 1. controls for RQ1 / RQ3 (not one of the five papers -- see report.md)
venvs/harness/Scripts/python.exe harness/runners/run_baseline.py --mode plain
venvs/harness/Scripts/python.exe harness/runners/run_baseline.py --mode selfcheck

# 2. the three systems, on the shared datasets/questions
venvs/lida/Scripts/python.exe      harness/runners/run_lida.py
venvs/df2/Scripts/python.exe       harness/runners/run_data_formulator2.py
venvs/nvagent/Scripts/python.exe   harness/runners/run_nvagent.py

# 3. the two benchmarks, natively
venvs/harness/Scripts/python.exe     harness/setup_keys.py
# full Stat-Closed: all 404 items (198 tool + 206 theory), 6 parallel shards;
# add --resume to retry unanswered / "Error:" items without redoing the rest
venvs/statabench/Scripts/python.exe  harness/runners/run_statabench.py --track all --shards 6

venvs/datascibench/Scripts/python.exe harness/fetch_ground_truth.py
# full DataSciBench: 222 tasks (csv, human, dl, bcb) x 3 runs; resumable --
# re-running skips every run that already has output
venvs/datascibench/Scripts/python.exe harness/runners/run_datascibench.py --shards 5

# 4. score everything and build the comparison table
venvs/harness/Scripts/python.exe harness/compare.py
```

Add `--limit N` to any runner for a cheap smoke test. `harness/compare.py
--no-llm` forces the keyword heuristic for the interpretation dimension instead
of the LLM judge.

Sanity-check the rubric at any time, no API key needed:

```bash
venvs/harness/Scripts/python.exe harness/test_rubric.py
```

## Ground truth

`datasets/*/questions.json` contains 8 questions per dataset. Every value under
`reference_answer` is **computed** by `build_questions.py` from the CSV in that
folder — none of it is hand-typed. Add your own questions by appending objects
with the same schema and re-running the script (it will not overwrite your
additions' text, but it does regenerate the file, so add new questions inside
`build_questions.py` rather than editing the JSON directly).

## Environment adaptations

Every deviation from a repo's stock instructions, and why. These matter for the
FYP's reproducibility section — three of the five repos do not run as shipped.

| Repo | Problem | What was done |
|---|---|---|
| **nvAgent** | `core/chat_manager.py` imports `PROVIDER_NAME` from `core/const.py`, which does not define it (it defines `PROCESSOR_NAME`; the `web_vis` variant uses the Provider/Generator/Corrector naming). The module **cannot be imported as shipped**, which also breaks the repo's own `run_evaluate.py`. | 2-line patch in the clone, `PROVIDER_NAME` → `PROCESSOR_NAME`. See `git -C repos/nvAgent diff`. |
| **nvAgent** | Pins `openai==1.51.2`, which passes `proxies=` to httpx; removed in httpx 0.28. | `httpx<0.28` pinned in `venvs/nvagent`. |
| **nvAgent** | `viseval/__init__.py` eagerly imports `Evaluator` → `cairosvg` → native Cairo, absent on stock Windows. | `cairosvg` stubbed in `run_nvagent.py` before import. Only nvAgent's *native* VisEval rendering needs it; install the GTK runtime and delete the stub if you want native VisEval scores. |
| **nvAgent** | Written against Azure OpenAI, with keys hardcoded in `core/api_config.py`. | `core.llm.api_func` monkeypatched onto a plain OpenAI client from `OPENAI_API_KEY`. Clone untouched; prompts and agent logic unchanged. |
| **nvAgent** | Expects a VisEval corpus at `<folder>/databases/<db_id>/*.csv`. | Shared CSVs materialised into that layout under `results/artifacts/nvagent_workspace/`. |
| **DataSciBench** | Its vendored MetaGPT — and upstream `metagpt==0.8.1` on PyPI — pin `lancedb==0.4.0`, which has been **delisted from PyPI** (earliest available is now 0.14.0). `pip install` fails outright. | Installed upstream `metagpt==0.8.1` with the `lancedb` pin dropped. lancedb is used only for RAG/vector storage, which the Data Interpreter path never touches. `SciDataInterpreter`, `experiments.run_examples` and `experiments.evaluate` all import cleanly against it. |
| **DataSciBench** | MetaGPT reads config from `~/.metagpt/config2.yaml`, not the environment. | `run_datascibench.py` writes it from `.env` on every run. |
| **DataSciBench** | Ships prompts only; every `data/<task_id>/` has just `prompt.json`. Ground truth lives in a **gated** HF dataset. | `harness/fetch_ground_truth.py`. You must accept the gate at <https://huggingface.co/datasets/zd21/DataSciBench> with the account owning your `HF_TOKEN` before it will work. |
| **StatABench** | `keys.json` is a **tracked** file in the clone (the authors committed a placeholder). Once `setup_keys.py` writes a real credential it shows as modified in that clone's `git status`, and this project's `.gitignore` cannot cover it because a nested clone has its own index. | Never commit or push from inside `repos/StatABench`. `setup_keys.py` prints this warning on every run. |
| **StatABench** | Reads credentials from its own `keys.json`; `mcp_pipeline.py` resolves models through a hardcoded `agent_map` with no generic OpenAI entry. | `harness/setup_keys.py` writes a `gpt4omini` entry (the one `agent_map` key pointing at an OpenAI model) from `.env`. Clone untouched. |
| **StatABench** | `mcp_pipeline.py` spawns the tool server with `command="python"`, resolved via `PATH` — which finds the *system* interpreter, not the venv, so the server starts without statsmodels/pymc/pingouin and every tool call fails. | `run_statabench.py` prepends the venv's `Scripts/` to `PATH` for the subprocess. |
| **Data Formulator** | `main` is **0.8.0b1**, several architectural generations past the DF2 paper (arXiv 2408.16119). Benchmarking it would not be benchmarking the paper. | Second checkout at tag **0.2** in `repos/data-formulator-v2`; the runner targets that. `repos/data-formulator` is kept at `main` for reference. |
| **Data Formulator** | UI-first; no headless "answer this question" entry point. | Runner drives `PythonDataRecAgent.run()` directly — the agent that takes an NL goal and returns both a chart recommendation and a transform. |
| **LIDA** | `llmx`'s `OpenAITextGenerator` builds its OpenAI client from a fixed kwarg whitelist (`api_key`, `organization`, `api_version`, `azure_endpoint`) — there is **no `base_url` parameter**, so it cannot be pointed at a non-OpenAI endpoint through its API. | `load_env()` exports `OPENAI_BASE_URL` into the environment, which the openai SDK reads on its own. No change to llmx or LIDA. |
| **Data Formulator** | Talks to models through litellm, which splits provider from model on the first `/` — but Hugging Face ids contain slashes, so `openai/gpt-oss-120b` would be parsed as provider `openai`, model `gpt-oss-120b`, and the wrong id would reach the router. | `harness/common.py:litellm_model()` double-prefixes to `openai/openai/gpt-oss-120b`; litellm strips one and forwards the full id. Verified against `litellm.get_llm_provider`. |
| **StatABench** | `mcp_pipeline.py` skips every row with `code == 0`, so as shipped the MCP track attempts only the 198 tool-requiring Stat-Closed items, not all 404. | `run_statabench.py --track all` marks theory rows `code=1` for generation only (`get_mcp_prompt()` builds the identical prompt for them, since their `dataset` is 0) and restores the flag before `eval.py`, which uses it to choose the ground-truth column. The runner reports tool-item and theory-item accuracy separately. `--track tool` (the default) reproduces the shipped behaviour. |
| **StatABench** | One pipeline process answers ~13 s per item, and the tool server imports pymc -> arviz, which writes a once-a-day stamp file on import; two servers started in the same instant race on it and one exits with `McpError: Connection closed`. | `--shards N` runs N pipeline processes on disjoint slices, started 20 s apart, and merges answers by `id`. `--resume` keeps good answers and regenerates only missing or `Error:` ones. |
| **DataSciBench** | `src/vlm_utils.py` (the chart judge used by 14 task metrics) builds its client for a third-party proxy, `api.turboai.one`, with no key configured, so every VLM-judged metric raised and scored 0. | Patched to read `DSB_VLM_BASE_URL` / `DSB_VLM_API_KEY` / `DSB_VLM_MODEL`, defaulting to the original values; the runner points it at the configured provider with `gpt-4o-mini`, the judge model the file names. See `git -C repos/DataSciBench diff -- src/vlm_utils.py`. |
| **DataSciBench** | The venv held only part of `requirements.txt` (no tensorflow, nltk, gensim, ...) and none of the libraries BigCodeBench-derived tasks import (statsmodels, faker, ...), so those tasks would score the environment rather than the agent. | Installed the missing requirements plus statsmodels, faker, sympy, opencv-python-headless, wordcloud, xlrd, PyPDF2 and related libraries, with numpy/pandas/scikit-learn/openai/httpx/pydantic held at their existing versions and metagpt's `tqdm==4.66.2` / `typing-extensions==4.9.0` pins restored. The pre-install freeze is in `results/_backup_sample_runs_2026-09-03/`. torch is not installed (one dl ground truth uses it; tensorflow/keras are available). |
| **DataSciBench** | Prompts say the input data is at `../` (i.e. `data/<task_id>/`), but the clone ships only `prompt.json` there; inputs are in the HF download under `DataSciBench-data/<task_id>/`, and the earlier setup copied only `gt/`. 45+ of the 55 csv/human/dl tasks had no input, and the agent **fabricated dummy input files and analysed those** (e.g. `csv_excel_1`: "source.xlsx not found – creating a dummy file"). | `run_datascibench.py` now copies missing inputs into `data/<task_id>/` before generating. The first full-run attempt on 2026-09-15 was stopped and its runs (plus the 2026-09-03 `csv_excel_0` sample) moved to `results/_invalid_dsb_runs_missing_inputs_2026-09-15/`. |
| **DataSciBench** | BCB runs keep the full model id as a path component, and `src/logs.py` opens `logs/openai/<model>_<run>_time.txt` without creating `logs/openai/`, so every BCB run raised `FileNotFoundError` before the agent started. | `run_datascibench.py` creates the folder. |
| **DataSciBench** | The two task families use different native evaluators: `experiments.evaluate` for csv/human/dl and `experiments.evaluate_tmc` for bcb. `evaluate_tmc` `exec()`s each completion in-process with no overall timeout and matches `--task_id` by substring. `show_bcb_eval_results.py` averages over 10 runs, padding missing runs with 0. | The runner calls both; BCB tasks are evaluated one per subprocess with a timeout (`--bcb-timeout`, default 900 s). Task CR is the sum of per-run CR divided by `--max-runs` (default 3, the `run_examples` default), so a run that crashed counts as 0 rather than being dropped. |
| **DataSciBench** | `evaluate_tmc` `exec()`s each BCB completion in-process; completions that end in `plt.show()` open a TkAgg "Figure 1" window and block until a human closes it, and on Windows the per-task timeout killed only the venv launcher, not the real interpreter. | `run_datascibench.py` sets `MPLBACKEND=Agg` for every child and kills the whole process tree (`taskkill /T`) on timeout. |
| **StatABench** | `mcp_pipeline.py` picks the model from a hardcoded `agent_map`, so it always sent the literal string `gpt-4o-mini` regardless of `keys.json`. | 1-line patch: prefer `keys.json`'s `model` field over `agent_map`. See `git -C repos/StatABench diff`. |
| **LIDA** | `requirements.txt` is just `.`; pip resolved it against openai 3.7.0 / pandas 3.0.5 / numpy 2.4.6, far newer than llmx 0.0.21a0 targets. | Left as resolved; verified `Manager`, `OpenAITextGenerator` and `summarize()` all work. Watch for drift here if results look odd. |

## Notes on the papers

* **StatABench arXiv ID `2606.22977` is correct.** The numbering looks like a
  future date but 2606 = June 2026, which is in the past. Title, authors and
  the `youxin01/StatABench` repo all match. No substitution was made.
* **nvAgent** is cited in the FYP brief as `2025.acl-long.960`; the repo README
  cites arXiv `2502.05036`. These are the preprint and the ACL 2025 version of
  the same work.
