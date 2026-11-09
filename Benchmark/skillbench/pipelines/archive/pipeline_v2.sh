#!/bin/sh
# Finish the v2 run, then score (judge results are cached, so already-judged answers are skipped) and analyse.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS=8
echo "== run   $(date)";  python run_all.py --workers 40
echo "== score $(date)";  python score_v2.py
echo "== analyse $(date)"; python analyze_v2.py
echo "== done  $(date)"
