#!/bin/sh
# Final chain after the tool-call parsing fix: redo quarantined answers and finish all runs, validate, score, analyse.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS=8
echo "== runs $(date)"
python run_all.py --workers 40 > final_run_v2.log 2>&1 &
python statabench.py run --runs 1,2,3 --workers 40 > final_run_sb.log 2>&1 &
wait
echo "== validate $(date)"
python validate.py --bench skillbench --runs 1,2,3 --workers 30 > final_val_v2.log 2>&1 &
python validate.py --bench statabench --runs 1,2,3 --workers 30 > final_val_sb.log 2>&1 &
wait
echo "== score $(date)"
python score_v2.py > final_score_v2.log 2>&1 &
python statabench.py score > final_score_sb.log 2>&1 &
wait
echo "== analyse $(date)"
python analyze_v2.py > /dev/null && python disagreements.py > /dev/null && python statabench.py analyse > /dev/null
echo "== deck $(date)"; (cd ../../Skills/deck && python build_deck.py)
echo "== done $(date)"
