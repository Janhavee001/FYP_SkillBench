"""
Second quarantine (5 Nov). Some answers were still written by a harness process that predated the tool-call
parsing fix. They are recognised by what the agent was told, not by the raw call:
  - an empty script was run and failed silently ('(no stdout)\\n[stderr]\\n' with nothing after it), or
  - a leaked tool name was rejected ("Unknown tool 'run_python<|...").
Such answers, any validated answer derived from them, and their cached judgements are moved to
results/_quarantine_stale/ so the resumable runners redo them.

    python quarantine_stale.py [--dry]
"""
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
Q = ROOT / "results" / "_quarantine_stale"
SIGNS = ("(no stdout)\n[stderr]\n", "Unknown tool 'run_python<|", "Unknown tool 'submit_answer<|",
         "Unknown tool 'load_skill<|")


def affected(path):
    r = json.loads(path.read_text(encoding="utf-8"))
    for m in r.get("transcript") or []:
        if m.get("role") == "tool":
            c = str(m.get("content", ""))
            if c == SIGNS[0] or any(c.startswith(s) for s in SIGNS[1:]):
                return True
    return False


dry = "--dry" in sys.argv
log = []
for bench_root in [ROOT / "results" / "v2", ROOT / "results" / "statabench"]:
    bad = {f for f in bench_root.glob("*/run*/*/*.json") if affected(f)}
    for f in list(bad):
        if f.parent.name == "agent_skills":
            v = f.parent.parent / "agent_skills_validated" / f.name
            if v.exists():
                bad.add(v)
    for f in sorted(bad):
        rel = f.relative_to(ROOT / "results")
        log.append(str(rel))
        if dry:
            continue
        dst = Q / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))
        j = bench_root / "judgements" / f.relative_to(bench_root)
        if j.exists():
            j.unlink()
if not dry:
    Q.mkdir(parents=True, exist_ok=True)
    (Q / "moved.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
from collections import Counter
print(Counter("/".join(p.replace("\\", "/").split("/")[:3]) for p in log))
print(len(log), "answers", "would be" if dry else "", "quarantined")
