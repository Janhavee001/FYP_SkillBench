"""
Runs every ```python block of every *built* skill (claude_skills/dist/<skill>/SKILL.md) in a fresh
subprocess whose working directory is outside the repo, so the only importable copy of the library
is the one bundled in the skill's scripts/statkit. That is the situation in the Claude app sandbox.
Placeholders in the examples are replaced with real paths; each skill's example runs against the
dataset its column names refer to.
"""
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "claude_skills"))
import build  # noqa: E402

from skills.common import BENCHMARK_DATASETS  # noqa: E402

TITANIC = BENCHMARK_DATASETS / "titanic" / "titanic.csv"
WINE = BENCHMARK_DATASETS / "wine_quality" / "winequality-red.csv"
DATASET = {"data-profiling": TITANIC, "distribution-analysis": TITANIC, "group-comparison": TITANIC,
           "correlation-analysis": WINE, "regression-analysis": WINE, "visualization": WINE,
           "time-series-analysis": None,  # synthetic daily series, made below
           "relational-data-analysis": None}  # synthetic shop database, made below


@pytest.fixture(scope="module")
def built():
    build.build()
    return build.DIST


@pytest.fixture(scope="module")
def daily_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    t = np.arange(730)
    cnt = 4000 + 3 * t + 600 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 150, 730)
    path = tmp_path_factory.mktemp("ts") / "daily.csv"
    pd.DataFrame({"date": pd.date_range("2011-01-01", periods=730).astype(str), "cnt": cnt.round()}).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def shop_dir(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_shop_tables", ROOT / "demo" / "make_shop_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = tmp_path_factory.mktemp("shop")
    for name, df in mod.make_shop().items():
        df.to_csv(d / f"{name}.csv", index=False)
    return d


@pytest.mark.skipif(not TITANIC.exists() or not WINE.exists(), reason="benchmark datasets not found")
@pytest.mark.parametrize("skill", list(build.SKILL_TOOLS))
def test_skill_examples_run_from_the_bundled_copy(skill, built, daily_csv, shop_dir, tmp_path):
    text = (built / skill / "SKILL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 2, "every skill needs a setup block and a worked example"
    data = DATASET[skill] or daily_csv
    code = "\n".join(blocks)
    code = (code.replace("<this skill's folder>", (built / skill).as_posix())
                .replace("<path to the uploaded file>", pathlib.Path(data).as_posix())
                .replace("<outputs folder>", tmp_path.as_posix())
                .replace("<folder with the uploaded files>", shop_dir.as_posix()))
    script = tmp_path / "run.py"
    script.write_text(code, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # nothing but the bundled copy to import
    env.update(PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
                          timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize("skill", list(build.SKILL_TOOLS))
def test_skill_reference_documents_every_listed_tool(skill, built):
    ref = (built / skill / "reference" / "tools.md").read_text(encoding="utf-8")
    for dotted in build.SKILL_TOOLS[skill]:
        assert f"## `{dotted.split('.')[1]}`" in ref
    skill_md = (built / skill / "SKILL.md").read_text(encoding="utf-8")
    for name in re.findall(r"from statkit\.(\w+) import ([\w, ]+)", skill_md):
        for fn in name[1].split(","):
            mod = __import__(f"skills.{name[0]}", fromlist=[fn.strip()])
            assert hasattr(mod, fn.strip()), f"{skill}: SKILL.md imports missing {name[0]}.{fn.strip()}"
