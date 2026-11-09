#!/bin/sh
# Redo answers quarantined by quarantine_stale.py, one runner at a time, then validate, score, analyse, build.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS=8
echo "== runs $(date)"
python run_all.py --workers 20 > redo_run_v2.log 2>&1
python statabench.py run --runs 1,2,3 --workers 40 > redo_run_sb.log 2>&1
echo "== validate $(date)"
python validate.py --bench skillbench --runs 1,2,3 --workers 20 > redo_val_v2.log 2>&1
python validate.py --bench statabench --runs 1,2,3 --workers 40 > redo_val_sb.log 2>&1
echo "== check $(date)"
python quarantine_stale.py --dry
echo "== score $(date)"
python score_v2.py > redo_score_v2.log 2>&1
python statabench.py score > redo_score_sb.log 2>&1
echo "== analyse $(date)"
python analyze_v2.py > /dev/null && python disagreements.py > /dev/null && python statabench.py analyse > /dev/null
echo "== deck $(date)"; (cd ../../Skills/deck && python build_deck.py)
echo "== done $(date)"
