# Reference-answer review

30 of 30 sampled questions reviewed.

- Agree: 29
- Disagree: 0
- Unsure: 1

Agreement rate: 97%

Disagreements and unsure:

- I11: Unsure - Logic fine (comma = multiple directors, \N has no comma). Unsure whether crew.parquet is restricted to the feature films in movies.parquet; if it holds all title types the count includes non-films. Check, or filter cw to m.tconst.
  - Resolution: Checked: crew.parquet is filtered to the feature films at build time (build_data.py, crew.tconst.isin(ids)); all 759,901 crew rows are in movies.parquet, and restricting to films gives the same 60,907. Reference kept.

After resolution: 30 of 30 references confirmed; 0 open.
