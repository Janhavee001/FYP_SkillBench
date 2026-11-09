# Pipelines as run

The shell scripts that produced the reported results, kept as a record. Use `../../pipeline.sh` for a fresh run.
Each script changes to `Benchmark/skillbench/` first, so it can still be run from here.

| Order | Script | Purpose |
|---|---|---|
| 1 | `pipeline_v2.sh` | SkillBench v2 runs, scoring and analysis |
| 2 | `pipeline_statabench.sh` | StatABench under the four agent conditions |
| 3 | `pipeline_validate.sh` | RQ3 validator on both benchmarks |
| 4 | `pipeline_final.sh` | Reruns after the tool-call parsing fix (first quarantine, 1,982 answers) |
| 5 | `pipeline_redo.sh` | Sequential reruns after the second quarantine (2,683 answers); produced the final results |

The two quarantines are described in `docs/paper/Paper_Draft.md`, section 5.
