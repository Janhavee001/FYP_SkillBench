#!/bin/sh
# RQ3: validator condition on both benchmarks, then final scoring, analysis, audit and deck.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS=8
echo "== validate skillbench $(date)"; python validate.py --bench skillbench --runs 1,2,3 --workers 40
echo "== wait for statabench runs $(date)"
while ! grep -q "== done" pipeline_statabench.log; do sleep 60; done
while ! grep -q "== done" pipeline_v2.log; do sleep 60; done
echo "== validate statabench $(date)"; python validate.py --bench statabench --runs 1,2,3 --workers 40
echo "== score skillbench $(date)"; python score_v2.py && python analyze_v2.py > /dev/null && python disagreements.py > /dev/null
echo "== score statabench $(date)"; python statabench.py score && python statabench.py analyse > /dev/null
echo "== deck $(date)"; (cd ../../Skills/deck && python build_deck.py)
echo "== done $(date)"
