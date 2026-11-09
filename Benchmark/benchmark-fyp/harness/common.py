"""
Shared plumbing for the benchmark: config, LLM client, dataset loading, and the
normalised result record that every runner emits and that rubric.py consumes.

The whole point of this module is that the five systems produce wildly
different native output (a JSON column of chat responses, a MetaGPT plan, a
LIDA ChartExecutorResponse, a Vega-Lite spec, a matplotlib script). Each runner
is responsible for flattening its system into `Record` below; rubric.py then
never needs to know which system it is scoring.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
REPOS = ROOT / "repos"
VENVS = ROOT / "venvs"


def _force_utf8_console() -> None:
    """
    Windows consoles default to cp1252, and models routinely emit characters
    outside it -- non-breaking hyphens in "chi-square", arrows in "x -> y".
    Printing one raises UnicodeEncodeError and aborts the record mid-write, so
    without this a perfectly good run is lost to a console encoding.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable stream
            pass


_force_utf8_console()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HF_ROUTER = "https://router.huggingface.co/v1"
OPENROUTER = "https://openrouter.ai/api/v1"

# Providers whose model ids are "org/model" rather than a bare name.
SLASH_ID_PROVIDERS = {"huggingface", "openrouter"}

# Token prefix expected per provider, used only to catch a key pasted into the
# wrong variable -- the commonest and most confusing .env mistake.
KEY_PREFIX = {"openrouter": "sk-or-", "huggingface": "hf_"}


def load_env() -> dict:
    """
    Load .env from the project root and resolve which LLM provider to use.

    LLM_PROVIDER=openai       -> OPENAI_API_KEY against api.openai.com
    LLM_PROVIDER=huggingface  -> HF_TOKEN against router.huggingface.co/v1,
                                 which is a drop-in OpenAI-compatible endpoint

    Whichever is chosen, the resolved credentials are exported back into
    os.environ as OPENAI_API_KEY / OPENAI_BASE_URL. That matters because not
    every downstream library lets us pass a base URL: LIDA's vendored llmx
    builds its OpenAI client from a fixed whitelist of kwargs with no base_url
    among them, so the only way to redirect it is the environment variable that
    the openai SDK reads on its own.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        # Minimal parser so the harness still works without python-dotenv.
        envfile = ROOT / ".env"
        if envfile.exists():
            for line in envfile.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
    hf_token = os.environ.get("HF_TOKEN", "")

    if provider in ("openrouter", "or"):
        provider = "openrouter"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER)
        default_model = "openai/gpt-oss-120b"
    elif provider in ("huggingface", "hf"):
        provider = "huggingface"
        api_key = hf_token
        base_url = os.environ.get("HF_BASE_URL", HF_ROUTER)
        default_model = "openai/gpt-oss-120b"
    else:
        provider = "openai"
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        default_model = "gpt-4o-mini"

    # Export so libraries that only read the environment pick the right target.
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url

    return {
        "provider": provider,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "hf_token": hf_token,
        "bench_model": os.environ.get("BENCH_MODEL", default_model),
        "judge_model": os.environ.get("JUDGE_MODEL",
                                      os.environ.get("BENCH_MODEL", default_model)),
        "limit": int(os.environ.get("BENCH_LIMIT", "0") or 0),
        "temperature": float(os.environ.get("BENCH_TEMPERATURE", "0")),
        "seed": int(os.environ.get("BENCH_SEED", "42")),
    }


def have_key() -> bool:
    cfg = load_env()
    key = cfg["openai_api_key"]
    return bool(key) and not key.startswith("your-")


_WARNED = set()


def check_model_consistency(cfg: dict) -> list[str]:
    """
    Catch the .env mistakes that would otherwise fail deep inside a run.

    The expensive one: a .env copied from an older template still says
    JUDGE_MODEL=gpt-4o-mini while LLM_PROVIDER=huggingface. Every system run
    would succeed and only the rubric's interpretation dimension would fail,
    hours later, after the credits are spent.
    """
    problems = []
    provider = cfg["provider"]
    models = (("BENCH_MODEL", cfg["bench_model"]),
              ("JUDGE_MODEL", cfg["judge_model"]))

    if provider in SLASH_ID_PROVIDERS:
        for label, model in models:
            # These providers use "org/model", optionally with a suffix
            # (":free" on OpenRouter, ":fastest"/":cheapest" on HF).
            if "/" not in model:
                problems.append(
                    f"{label}={model!r} is not an {provider} model id "
                    f"(expected 'org/model', e.g. 'openai/gpt-oss-120b'). "
                    f"It will fail on {cfg['openai_base_url']}.")
    else:
        for label, model in models:
            if model.count("/") >= 1 and not model.startswith("ft:"):
                problems.append(
                    f"{label}={model!r} looks like an org/model id but "
                    f"LLM_PROVIDER=openai. Set LLM_PROVIDER to openrouter or "
                    f"huggingface, or use a plain OpenAI model name.")

    # A key pasted into the wrong variable is the mistake that wastes the most
    # time, because the failure surfaces as a bare 401 with no hint of cause.
    key = cfg["openai_api_key"]
    expect = KEY_PREFIX.get(provider)
    if key and expect and not key.startswith(expect):
        looks_like = next((p for p, pre in KEY_PREFIX.items()
                           if key.startswith(pre)), None)
        hint = (f" That value looks like a {looks_like} key."
                if looks_like else
                " OpenAI keys start with 'sk-' and are not valid here.")
        var = {"openrouter": "OPENROUTER_API_KEY",
               "huggingface": "HF_TOKEN"}[provider]
        problems.append(
            f"{var} does not start with {expect!r}, which every {provider} "
            f"credential does.{hint} Check LLM_PROVIDER matches the key you "
            f"actually have.")
    return problems


def require_key(what: str) -> dict:
    cfg = load_env()
    problems = check_model_consistency(cfg)
    if problems and what not in _WARNED:
        _WARNED.add(what)
        print("\n  CONFIG PROBLEM in .env:")
        for p in problems:
            print(f"    - {p}")
        print("    Fix .env before spending credits on a run that cannot "
              "score.\n")
        raise SystemExit(1)
    if not have_key():
        var = {"huggingface": "HF_TOKEN",
               "openrouter": "OPENROUTER_API_KEY"}.get(cfg["provider"],
                                                       "OPENAI_API_KEY")
        extra = ""
        if cfg["provider"] == "huggingface":
            extra = (
                "\n\nNote: a plain read token is not enough. The token must be a\n"
                "fine-grained token with 'Make calls to Inference Providers'\n"
                "enabled, from https://huggingface.co/settings/tokens")
        elif cfg["provider"] == "openrouter":
            extra = ("\n\nGet a key at https://openrouter.ai/keys "
                     "(it starts with 'sk-or-').")
        raise SystemExit(
            f"{var} is not set (or still holds the placeholder).\n"
            f"  provider = {cfg['provider']}  (set by LLM_PROVIDER in .env)\n"
            "  1. cp .env.example .env\n"
            f"  2. put your real {var} in .env\n"
            f"  3. re-run: {what}{extra}"
        )
    return cfg


def litellm_model(cfg: dict) -> str:
    """
    Model id in the form litellm needs for an OpenAI-compatible endpoint.

    Only for callers that actually route through litellm. Data Formulator does
    NOT: with endpoint="openai" its Client builds a raw openai.OpenAI client and
    forwards the model string verbatim, so it takes the plain id instead. Check
    which path a library uses before applying this.

    litellm splits provider from model on the first '/', but OpenRouter and
    Hugging Face ids contain slashes ("openai/gpt-oss-120b"), so a single
    prefix would send only "gpt-oss-120b" upstream. Prefixing again makes
    litellm strip one "openai/" and forward the full id. Verified with
    litellm.get_llm_provider.
    """
    model = cfg["bench_model"]
    if cfg["provider"] == "openai":
        return model if model.startswith("openai/") else f"openai/{model}"
    return f"openai/{model}"


def openai_client(cfg: dict | None = None):
    from openai import OpenAI
    cfg = cfg or load_env()
    return OpenAI(api_key=cfg["openai_api_key"], base_url=cfg["openai_base_url"])


def chat(prompt: str, *, system: str = "You are a helpful assistant.",
         model: str | None = None, cfg: dict | None = None,
         temperature: float | None = None) -> str:
    """One-shot chat completion used by runners and by the LLM judge."""
    cfg = cfg or load_env()
    client = openai_client(cfg)
    resp = client.chat.completions.create(
        model=model or cfg["bench_model"],
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=cfg["temperature"] if temperature is None else temperature,
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def load_questions(dataset: str | None = None) -> list[dict]:
    """Return the flat list of questions, optionally for one dataset only."""
    out = []
    for name in ("titanic", "wine_quality"):
        if dataset and name != dataset:
            continue
        payload = json.loads(
            (DATASETS / name / "questions.json").read_text(encoding="utf-8"))
        for q in payload["questions"]:
            q = dict(q)
            q["csv_path"] = str(DATASETS / payload["csv_path"])
            q["dataset_columns"] = payload["columns"]
            q["dataset_description"] = payload["description"]
            out.append(q)
    return out


def dataset_meta(dataset: str) -> dict:
    return json.loads(
        (DATASETS / dataset / "questions.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Normalised record
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Record:
    """One (system, question) outcome, in the shape rubric.py expects."""
    id: str
    dataset: str
    question: str
    declared_method: str | None = None     # method the system said it used
    declared_chart: str | None = None      # chart type the system chose
    code: str = ""                         # generated code, verbatim
    executed: bool = False                 # did the code run without raising
    execution_error: str | None = None
    stdout: str = ""
    artifact: str | None = None            # path to a rendered chart, if any
    final_answer: str = ""                 # narrative / interpretation text
    numeric_outputs: dict = dataclasses.field(default_factory=dict)
    repair_attempts: int = 0               # self-correction loops the system ran
    self_corrected: bool = False           # did a repair change the outcome
    ground_truth: dict = dataclasses.field(default_factory=dict)
    expected_method: list = dataclasses.field(default_factory=list)
    acceptable_methods: list = dataclasses.field(default_factory=list)
    expected_chart: list = dataclasses.field(default_factory=list)
    tolerance: float = 0.05
    kind: str = "mixed"
    error: str | None = None               # harness-level failure, not the system's
    raw: dict = dataclasses.field(default_factory=dict)  # native output, kept verbatim

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def write_results(system: str, paper: str, architecture: str,
                  native_evaluation: str, records: list[Record],
                  meta: dict | None = None) -> pathlib.Path:
    RESULTS.mkdir(exist_ok=True)
    payload = {
        "system": system,
        "paper": paper,
        "architecture": architecture,
        "native_evaluation": native_evaluation,
        "meta": meta or {},
        "n_records": len(records),
        "records": [r.to_dict() if isinstance(r, Record) else r for r in records],
    }
    path = RESULTS / f"{system}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[write] {path}  ({len(records)} records)")
    return path


# --------------------------------------------------------------------------- #
# Method / chart normalisation
#
# Every system names things differently (LIDA says "histogram", nvAgent emits a
# Vega "bar" mark, an LLM writes "independent samples t-test"). rubric.py scores
# selection correctness against the canonical vocabulary used in questions.json,
# so both sides get pushed through these maps first.
# --------------------------------------------------------------------------- #
METHOD_PATTERNS = [
    ("chi_square_test", r"chi[\s\-_]?squar|chi2|chisq|crosstab.*chi|contingency"),
    ("fisher_exact_test", r"fisher.{0,10}exact"),
    ("proportion_z_test", r"proportions?[\s_]z|z[\s\-_]?test for proportion|proportions_ztest"),
    ("welch_t_test", r"welch|equal_var\s*=\s*False"),
    ("t_test", r"\bt[\s\-_]?test\b|ttest_ind|ttest_rel|ttest_1samp|student.?s t"),
    ("mann_whitney_u", r"mann[\s\-_]?whitney|wilcoxon rank|mannwhitneyu"),
    ("anova", r"\banova\b|f_oneway|kruskal"),
    ("logistic_regression", r"logistic\s*regression|LogisticRegression|logit"),
    ("random_forest", r"random\s*forest|RandomForest"),
    ("gradient_boosting", r"gradient\s*boost|XGB|LGBM|GradientBoosting"),
    ("decision_tree", r"decision\s*tree|DecisionTree"),
    ("ridge_regression", r"\bridge\b|Ridge\("),
    ("ordinal_regression", r"ordinal\s*(logistic|regression)|OrderedModel"),
    ("linear_regression", r"linear\s*regression|LinearRegression|\bOLS\b|ols\(|least squares"),
    ("correlation", r"correlat|pearson|spearman|kendall|\.corr\(|corrcoef"),
    ("outlier_detection", r"outlier|\bIQR\b|z[\s\-_]?score|isolation\s*forest"),
    ("normality_test", r"shapiro|normality|kolmogorov|anderson.?darling|jarque"),
    ("missing_data_analysis", r"missing|isna\(|isnull\(|\bNaN\b|imput"),
    ("distribution_analysis", r"distribution|skew|kurtosis|histogram|density"),
    ("frequency_count", r"value_counts|frequency|\bcount\b"),
    ("groupby_aggregation", r"group\s?by|groupby|pivot_table|aggregat|\bmean by\b"),
    ("feature_importance", r"feature[\s_]importance|permutation_importance|coef_"),
    ("proportion_comparison", r"proportion|survival rate|percentage by"),
    ("scatter_analysis", r"scatter.*(trend|fit|regplot)|regplot|lmplot"),
    ("descriptive_statistics", r"describe\(|\bmean\b|\bmedian\b|summary statistic|\bstd\b"),
]

CHART_PATTERNS = [
    ("grouped_bar", r"grouped\s*bar|stacked\s*bar|barmode|hue\s*=.*bar|catplot.*kind=.?bar"),
    ("heatmap", r"heatmap|imshow|matshow|pcolor"),
    ("box", r"boxplot|box_plot|\bbox\b|kind\s*=\s*.box"),
    ("violin", r"violin"),
    ("strip", r"stripplot|swarmplot|jitter"),
    ("hist", r"histogram|\bhist\b|histplot|distplot"),
    ("kde", r"\bkde\b|kdeplot|density plot"),
    ("scatter", r"scatter|regplot|lmplot|jointplot|point mark"),
    ("line", r"lineplot|plt\.plot|\bline chart\b"),
    ("pie", r"pie\s*chart|plt\.pie"),
    ("point", r"pointplot|dot plot"),
    ("bar", r"\bbar\b|barplot|countplot|plt\.bar|catplot"),
]


def normalise_method(text: str | None) -> list[str]:
    """Return every canonical method name the text plausibly refers to."""
    if not text:
        return []
    t = str(text)
    hits = []
    for name, pat in METHOD_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            hits.append(name)
    return hits


# Systems name the same mark differently. Data Formulator and Vega-Lite say
# "point" for what questions.json calls a scatter; Vega says "arc" for a pie.
# These are exact-token aliases applied ONLY to a system's *declared* chart
# type, never to free prose -- matching "line" inside "regression line" would
# invent a line chart that nobody drew.
#
# "point" deliberately maps to both: a Vega point mark genuinely serves as a
# scatter and as a dot/point plot, so crediting either is accuracy, not
# leniency.
CHART_ALIASES = {
    "point": ["point", "scatter"],
    "circle": ["scatter"],
    "boxplot": ["box"],
    "box_plot": ["box"],
    "histogram": ["hist"],
    "arc": ["pie"],
    "rect": ["heatmap"],
    "errorband": ["line"],
    "area": ["line"],
    "grouped bar": ["grouped_bar", "bar"],
    "stacked bar": ["grouped_bar", "bar"],
}


def normalise_declared_chart(value: str | None) -> list[str]:
    """Canonical chart names for a system's explicitly declared chart type."""
    if not value:
        return []
    out = []
    for token in re.split(r"[,\s]*,[,\s]*|\s{2,}", str(value).strip().lower()):
        token = token.strip()
        if not token:
            continue
        if token in CHART_ALIASES:
            out.extend(CHART_ALIASES[token])
        else:
            out.extend(normalise_chart(token) or ([token] if token else []))
    return sorted(set(out))


def normalise_chart(text: str | None) -> list[str]:
    if not text:
        return []
    t = str(text)
    hits = []
    for name, pat in CHART_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            hits.append(name)
    return hits


# --------------------------------------------------------------------------- #
# Numeric extraction, used for output-correctness scoring
# --------------------------------------------------------------------------- #
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(text or ""):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def flatten_numeric(obj: Any, prefix: str = "") -> dict:
    """Flatten a nested ground-truth dict down to {path: float}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


# --------------------------------------------------------------------------- #
# Sandboxed code execution (used by runners that must execute generated code)
# --------------------------------------------------------------------------- #
def run_code(code: str, workdir: pathlib.Path, python: str | None = None,
             timeout: int = 120) -> tuple[bool, str, str]:
    """Execute `code` in `workdir`. Returns (ok, stdout, stderr)."""
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "snippet.py"
    header = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
    )
    script.write_text(header + code, encoding="utf-8")
    # The child inherits a cp1252 stdout on Windows, so a generated script that
    # prints "chi-square" with a non-breaking hyphen dies with
    # UnicodeEncodeError and is recorded as a failed execution. That would
    # penalise every system for a console setting rather than for its code, so
    # force UTF-8 on both ends of the pipe.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        p = subprocess.run([python or sys.executable, str(script)],
                           cwd=str(workdir), capture_output=True, text=True,
                           timeout=timeout, env=env,
                           encoding="utf-8", errors="replace")
        return p.returncode == 0, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"TimeoutExpired after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, "", f"{type(exc).__name__}: {exc}"


def strip_code_fence(text: str) -> str:
    """Pull python out of a ```python ... ``` fence if one is present."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()


def venv_python(name: str) -> str:
    """Path to a per-repo venv interpreter (Windows layout, posix fallback)."""
    win = VENVS / name / "Scripts" / "python.exe"
    nix = VENVS / name / "bin" / "python"
    return str(win if win.exists() else nix)
