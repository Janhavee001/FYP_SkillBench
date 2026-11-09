"""
Generate repos/StatABench/keys.json from the project .env.

StatABench reads credentials from its own keys.json rather than the
environment, and its mcp_pipeline.py resolves a model through a hardcoded
agent_map. To run OpenAI as the model under test without editing the clone, we
write a "gpt4omini" entry (the one agent_map key that points at an OpenAI
model) plus a "judge" entry used by the open-track judger.

Careful: keys.json is a TRACKED file inside the StatABench clone (the authors
committed a placeholder template). Once this script runs it holds a real
credential and will show as modified in that clone's `git status`. This
project's .gitignore cannot protect it -- a nested clone has its own index --
so never commit or push from inside repos/StatABench.

    venvs/harness/Scripts/python.exe harness/setup_keys.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness.common import REPOS, load_env  # noqa: E402


def main():
    cfg = load_env()
    if not cfg["openai_api_key"] or cfg["openai_api_key"].startswith("your-"):
        var = "HF_TOKEN" if cfg["provider"] == "huggingface" else "OPENAI_API_KEY"
        raise SystemExit(f"{var} is not set in .env -- nothing to write.")

    base = cfg["openai_base_url"]
    key = cfg["openai_api_key"]

    # "gpt4omini" is the one key in mcp_pipeline.py's hardcoded agent_map that
    # points at an OpenAI model, so it stays the entry we run under. The model
    # actually sent is the "model" field below: mcp_pipeline.py is patched to
    # prefer it over agent_map, which is what lets a Hugging Face model id like
    # "openai/gpt-oss-120b" reach the router instead of the literal string
    # "gpt-4o-mini". See README "Environment adaptations".
    keys = {
        "gpt4omini": {"base_url": base, "api_key": key, "model": cfg["bench_model"]},
        "openai": {"base_url": base, "api_key": key, "model": cfg["bench_model"]},
        "judge": {"base_url": base, "api_key": key, "model": cfg["judge_model"]},
        # evaluation/eval_utils.py calls gpt_chat(model="deepseek") with a
        # hardcoded provider name when it needs an LLM to adjudicate an
        # open-ended ("other") answer. Without this entry the native evaluator
        # dies with ValueError: Unknown provider: deepseek partway through
        # scoring. Alias it onto whatever judge model .env selects.
        "deepseek": {"base_url": base, "api_key": key, "model": cfg["judge_model"]},
        cfg["judge_model"]: {"base_url": base, "api_key": key,
                             "model": cfg["judge_model"]},
    }
    out = REPOS / "StatABench" / "keys.json"
    if not out.parent.exists():
        raise SystemExit(f"{out.parent} does not exist -- clone the repos first.")
    out.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    print(f"[write] {out}")
    print("       provider:", cfg["provider"], "| model:", cfg["bench_model"])
    print("       entries:", ", ".join(keys))
    print()
    print("  WARNING: keys.json is a TRACKED file in the StatABench clone "
          "(the authors")
    print("  committed a template). It now holds a real credential, so "
          "`git status`")
    print("  inside repos/StatABench will show it as modified. Never commit "
          "or push")
    print("  from inside that clone. This project's own .gitignore does not "
          "cover it,")
    print("  because a nested clone has its own git index.")


if __name__ == "__main__":
    main()
