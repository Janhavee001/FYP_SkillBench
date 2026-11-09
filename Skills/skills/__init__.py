"""
First-pass implementation of the skill catalog in ../CATALOG.md.

Every tool here is a plain typed function returning a dataclass -- no LLM call
inside the tool itself. That is deliberate: these are the building blocks an
agent calls, not the agent. Keeping them LLM-free is also what lets
tests/test_skills.py run with no API key, the same convention already used by
Benchmark/benchmark-fyp/harness/test_rubric.py.
"""
