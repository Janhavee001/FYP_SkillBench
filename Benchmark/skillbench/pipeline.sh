#!/bin/sh
# Full pipeline from a fresh checkout: data, questions, skills, both benchmarks, validator, scoring, analysis, deck.
# Every runner is resumable (finished answers are skipped), so the script can be re-run after an interruption.
# Needs OPENROUTER_API_KEY in ../benchmark-fyp/.env, and StatABench cloned at
# ../benchmark-fyp/repos/StatABench (see the top-level README).
#
#     sh pipeline.sh            # logs go to logs/
#
# The scripts that produced the reported results, in the order they were run, are in pipelines/archive/.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 SKILLBENCH_CODE_SLOTS="${SKILLBENCH_CODE_SLOTS:-8}"
mkdir -p logs
step() { echo "== $1 $(date)"; }

step "data";       python build_data.py      > logs/build_data.log 2>&1
                   python make_synthetic.py  > logs/make_synthetic.log 2>&1
step "questions";  python build_questions.py > logs/build_questions.log 2>&1
                   python check_anchors.py   > logs/check_anchors.log 2>&1
step "skills";     python ../../Skills/claude_skills/build.py > logs/build_skills.log 2>&1

step "runs";       python run_all.py --workers 20                     > logs/run_skillbench.log 2>&1
                   python statabench.py run --runs 1,2,3 --workers 40 > logs/run_statabench.log 2>&1
step "validate";   python validate.py --bench skillbench --runs 1,2,3 --workers 20 > logs/val_skillbench.log 2>&1
                   python validate.py --bench statabench --runs 1,2,3 --workers 40 > logs/val_statabench.log 2>&1
step "check";      python quarantine_stale.py --dry

step "score";      python score_v2.py           > logs/score_skillbench.log 2>&1
                   python statabench.py score   > logs/score_statabench.log 2>&1
step "analyse";    python analyze_v2.py > /dev/null
                   python disagreements.py > /dev/null
                   python statabench.py analyse > /dev/null
step "deck";       (cd ../../Skills/deck && python build_deck.py)
step "done"
