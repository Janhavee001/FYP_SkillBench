#!/bin/sh
# StatABench under the four SkillBench conditions: run 1 first (complete design slice), then runs 2-3.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS=8
echo "== run1  $(date)"; python statabench.py run --runs 1 --workers 40
echo "== score $(date)"; python statabench.py score; python statabench.py analyse > /dev/null
echo "== run23 $(date)"; python statabench.py run --runs 2,3 --workers 40
echo "== score $(date)"; python statabench.py score; python statabench.py analyse > /dev/null
echo "== done  $(date)"
