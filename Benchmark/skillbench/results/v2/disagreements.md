# Consensus disagreements with the reference

Questions where at least 50% of agent answers (all models, runs, agent conditions) agree on one answer that differs from the reference.

13 of 206 questions flagged.

| ID | Dataset | Trap | Reference | Consensus answer | Support | = careless answer | Audit |
|---|---|---|---|---|---|---|---|
| B14 | bike_sharing | confounding | -0.02919 | 0.0748 | 15/24 | no | reference correct: consensus 0.0748 = hour and season as numbers; question says categorical |
| C01 | clinic | simpsons_paradox | yes | no | 17/24 | yes | reference correct (trap): pooled comparison reverses within groups |
| C03 | clinic | unrecorded_values | cannot_determine | a number | 19/24 | yes | reference correct (trap): quantity is not recorded; a number cannot be determined |
| CB06 | citibike | missing_values | 1.961 | 1.91 | 21/24 | no | ambiguous - excluded: same station by name (1.961%) vs by station id (1.91%) |
| D03 | diabetes | missing_code | 3.142 | 100.0 | 12/24 | yes | reference correct (trap): weight is coded '?' when missing (97% of encounters); 100% counts the code as data |
| D23 | diabetes | missing_code | MC | ? | 18/24 | yes | reference correct (trap): '?' is a missing-value code, not a category |
| O02 | olist | hidden_entity_key | 3.119 | 0 | 16/24 | yes | reference correct (trap): customer_id is per order; customer_unique_id identifies the person |
| R05 | online_retail | grain | 157 | 208.85 | 12/24 | no | reference correct: agent condition reproduces 157.03; 208.85 from a wrong method |
| S03 | shop | simpsons_paradox | yes | no | 23/24 | yes | reference correct (trap): Simpson's paradox; confounder in another table |
| SA2 | saas | censoring | 407 | 322.5 | 16/24 | no | reference correct (trap): 322.5 = censored accounts treated as cancelled; Kaplan-Meier gives 407 |
| T02 | nyc_taxi | unrecorded_values | cannot_determine | yes | 22/24 | yes | reference correct (trap): cash tips are not recorded |
| T08 | nyc_taxi | unequal_exposure | Thursday | Saturday | 13/24 | no | reference correct (trap): Saturday only if out-of-month trips are kept; Thursday within January |
| T19 | nyc_taxi | unrecorded_values | cannot_determine | a number | 19/24 | yes | reference correct (trap): cash tips are not recorded |
