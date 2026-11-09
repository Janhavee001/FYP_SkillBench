"""
Move answers produced while the harness mis-parsed tool calls (see run.parse_tool_call) to
results/_quarantine_parse/, together with their cached judgements, so the resumable runners redo them with the
fixed parser. A validated answer is also moved when its source agent_skills answer is. Writes a log for the report.

    python quarantine_parse.py
"""
import json
import pathlib
import shutil

import run as R

ROOT = pathlib.Path(__file__).resolve().parent
Q = ROOT / "results" / "_quarantine_parse"
log = []


def affected(path):
    r = json.loads(path.read_text(encoding="utf-8"))
    return any(R.old_parse_failed(tc["function"]["name"], tc["function"]["arguments"])
               for m in (r.get("transcript") or []) for tc in (m.get("tool_calls") or []))


def move(path, bench_root):
    rel = path.relative_to(ROOT / "results")
    dst = Q / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))
    j = bench_root / "judgements" / path.relative_to(bench_root)
    if j.exists():
        j.unlink()
    log.append(str(rel))


for bench_root in [ROOT / "results" / "v2", ROOT / "results" / "statabench"]:
    files = [f for f in bench_root.glob("*/run*/*/*.json")]
    bad = {f for f in files if affected(f)}
    for f in list(bad):
        if f.parent.name == "agent_skills":
            v = f.parent.parent / "agent_skills_validated" / f.name
            if v.exists():
                bad.add(v)
    for f in sorted(bad):
        move(f, bench_root)
(Q / "moved.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
print(len(log), "answers quarantined")
