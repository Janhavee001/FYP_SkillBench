# Skills — my own tool/skill library

Second phase of *Reliable Data Analysis with LLM Agents: Skill Design, Selection, Composition,
and Validation*, following on from the five-paper benchmark in `../Benchmark/benchmark-fyp/`.

- **`CATALOG.md`** — the full 52-tool catalog across 10 categories, each tool cited against a
  specific paper and, for Category A and J, against a specific failure the benchmark documented.
  Read this first.
- **`skills/`** — all 52 catalog tools plus 9 relational tools for multi-table data (`relational.py`;
  see `CATALOG.md`), as plain typed functions with no LLM call inside
  them.
- **`tests/test_skills.py`** — 20 checks (plus `test_skills_extended.py`, `test_review_fixes.py`, `test_remaining_tools.py`, `test_relational.py` and `test_claude_skills.py`: 137 in all), no API key required, run against the same Titanic/Wine
  datasets and ground-truth values already verified in `DOCUMENTATION.md` §4.2.
- **`claude_skills/`** — the eight Claude skills (SKILL.md files) that call these tools, plus
  `build.py`, which packages them as uploadable zips. See its README.
- **`../docs/slides/Skills_Library.pptx`** — slide deck covering the tools and skills (v1).
- **`demo_pipeline.py`** — composes the tools end-to-end (load → detect → profile → validate →
  describe → test → interpret) and reproduces the exact wrong-separator bug from
  `DOCUMENTATION.md` §6.4 to show the grounding step catching it before any analysis runs.

## Run it

```bash
cd Skills
python -m pytest tests/ -v
python demo_pipeline.py
```

Both assume `../Benchmark/benchmark-fyp/datasets/{titanic,wine_quality}` exists (it already
does, from the benchmark phase).

## Why the first 10 came first

The benchmark's own conclusion (`DOCUMENTATION.md` §10) is specific: an explicit skill registry
is defensible for *composition and grounding*, not for method selection, which a plain LLM
already gets right 98% of the time on familiar data (§9, RQ1). So the first implementation pass
is not "the 10 most common stats functions" — it's one complete pipeline that exercises the
grounding story end to end, because that is the part the benchmark found broken:

1. `detect_separator_and_encoding`, `profile_dataset`, `missing_value_report` — see what you
   actually loaded before trusting it (closes the §6.4 wrong-separator bug and the §7.1 #11
   invented-data bug)
2. `validate_grounding`, `check_method_appropriateness` — check the plan against the schema and
   the method against the data before executing (closes the §6.6 nvAgent nonexistent-table bug)
3. `describe_column`, `correlation_matrix`, `chi_square_association`, `ab_test` — the actual
   statistics, deliberately ordinary, because RQ1 says this part isn't where the problem is
4. `interpret_result` — turns a statistic into a claim about *meaning* (effect size vs.
   significance, explicit caveats), which is the other gap the benchmark found: none of the
   three published systems it tested cleared this bar (§6.2)

See `CATALOG.md` §2 for the full implemented-vs-planned breakdown and what's next.
