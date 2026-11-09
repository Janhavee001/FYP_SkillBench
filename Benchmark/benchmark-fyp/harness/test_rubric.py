"""
Sanity tests for the shared rubric.

These run without an API key (the interpretation judge is forced to the
heuristic) and exist to prove the scoring logic actually discriminates: a
correct answer must outscore a wrong one on the dimension that is supposed to
notice the difference. Without this, a rubric can look reasonable and quietly
give everything the same number.

    venvs/harness/Scripts/python.exe harness/test_rubric.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness.common import load_questions, normalise_chart  # noqa: E402
from harness.rubric import score_record  # noqa: E402

FAILURES = []


def check(label, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}{(' -- ' + extra) if extra else ''}")
    if not cond:
        FAILURES.append(label)


def base(q, **kw):
    rec = dict(id=q["id"], dataset=q["dataset"], question=q["question"],
               ground_truth=q["reference_answer"],
               expected_method=q["expected_method"],
               acceptable_methods=q["acceptable_methods"],
               expected_chart=q["expected_chart"],
               tolerance=q["tolerance"], kind=q["kind"],
               code="", executed=False, stdout="", final_answer="",
               numeric_outputs={}, repair_attempts=0, self_corrected=False)
    rec.update(kw)
    return rec


def main():
    qs = {q["id"]: q for q in load_questions()}
    q2 = qs["titanic_q2"]   # chi-square, stat
    q4 = qs["titanic_q4"]   # bar chart, viz
    gt2 = q2["reference_answer"]

    print("\n--- D1 method selection ---")
    right = score_record(base(q2, code="from scipy.stats import chi2_contingency\n"
                                       "chi2_contingency(ct)"), use_llm=False)
    wrong = score_record(base(q2, code="from scipy.stats import ttest_ind\n"
                                       "ttest_ind(a, b)"), use_llm=False)
    none_ = score_record(base(q2, code="print(df.head())"), use_llm=False)
    check("chi-square scores 1.0 on a chi-square question",
          right["scores"]["method_selection"] == 1.0,
          f"got {right['scores']['method_selection']}")
    check("t-test scores below chi-square on a chi-square question",
          wrong["scores"]["method_selection"] < right["scores"]["method_selection"],
          f"t-test={wrong['scores']['method_selection']}")
    check("no identifiable method scores lowest",
          none_["scores"]["method_selection"] <= wrong["scores"]["method_selection"])

    print("\n--- D1 chart selection (viz question) ---")
    bar = score_record(base(q4, code="sns.barplot(x='pclass', y='survived', data=df)"),
                       use_llm=False)
    pie = score_record(base(q4, code="plt.pie(sizes, labels=labels)"), use_llm=False)
    check("bar chart scores 1.0 on a bar-chart question",
          bar["scores"]["method_selection"] == 1.0,
          f"got {bar['scores']['method_selection']}")
    check("pie chart scores lower than bar on a bar-chart question",
          pie["scores"]["method_selection"] < bar["scores"]["method_selection"],
          f"pie={pie['scores']['method_selection']}")

    print("\n--- D1 cross-system chart vocabulary (caught in a live run) ---")
    q8w = qs["wine_q8"]          # expects ['scatter'] only
    q4t = qs["titanic_q4"]       # expects ['bar','grouped_bar','point']
    # Data Formulator / Vega-Lite name the scatter mark "point"; scoring that
    # as a miss penalised DF2 for vocabulary, not for a wrong chart choice.
    vega = score_record(base(q8w, declared_chart="point"), use_llm=False)
    named = score_record(base(q8w, declared_chart="scatter"), use_llm=False)
    check("Vega 'point' scores as a scatter where scatter is expected",
          vega["scores"]["method_selection"]
          == named["scores"]["method_selection"] == 1.0,
          f"point={vega['scores']['method_selection']} "
          f"scatter={named['scores']['method_selection']}")
    check("'point' still counts as a point plot where that is expected",
          score_record(base(q4t, declared_chart="point"),
                       use_llm=False)["scores"]["method_selection"] == 1.0)
    check("'boxplot' scores as 'box'",
          score_record(base(qs["wine_q4"], declared_chart="boxplot"),
                       use_llm=False)["scores"]["method_selection"] == 1.0)
    # The alias map must not become a blanket pass.
    check("a genuinely wrong declared chart still loses credit",
          score_record(base(q8w, declared_chart="pie"),
                       use_llm=False)["scores"]["method_selection"] < 1.0)
    # Prose must not mint charts nobody drew.
    from harness.common import normalise_declared_chart as _ndc
    check("aliases apply to declared type only, not free prose",
          _ndc("point") == ["point", "scatter"] and "line" not in
          set(normalise_chart("we fitted a regression line to the data")),
          f"prose->{sorted(set(normalise_chart('a regression line')))}")

    print("\n--- D2 code executability ---")
    ok = score_record(base(q2, code="x=1", executed=True, stdout="chi2=260.7"),
                      use_llm=False)
    silent = score_record(base(q2, code="x=1", executed=True, stdout=""),
                          use_llm=False)
    broke = score_record(base(q2, code="x=", executed=False,
                              execution_error="SyntaxError: invalid syntax"),
                         use_llm=False)
    nocode = score_record(base(q2), use_llm=False)
    seq = [ok["scores"]["code_executability"], silent["scores"]["code_executability"],
           broke["scores"]["code_executability"], nocode["scores"]["code_executability"]]
    check("executability is strictly ordered: ran>silent>errored>no code",
          seq == sorted(seq, reverse=True) and len(set(seq)) == 4, f"got {seq}")

    print("\n--- D3 output correctness ---")
    truthy = (f"The chi-square statistic is {gt2['chi2_statistic']} with "
              f"{gt2['degrees_of_freedom']} degree of freedom and a p-value of "
              f"{gt2['p_value']}. The association is statistically significant. "
              f"We used a chi-square test of independence.")
    good = score_record(base(q2, stdout=truthy, final_answer=truthy), use_llm=False)
    bad = score_record(base(q2, stdout="chi2 = 3.14, p = 0.87, not significant",
                            final_answer="chi2 = 3.14, p = 0.87, not significant"),
                       use_llm=False)
    empty = score_record(base(q2), use_llm=False)
    check("correct numbers recover most ground-truth facts",
          good["scores"]["output_correctness"] >= 0.75,
          f"got {good['scores']['output_correctness']}")
    check("wrong numbers score lower than correct ones",
          bad["scores"]["output_correctness"] < good["scores"]["output_correctness"],
          f"wrong={bad['scores']['output_correctness']}")
    check("no output scores 0.0",
          empty["scores"]["output_correctness"] == 0.0,
          f"got {empty['scores']['output_correctness']}")

    print("\n--- D3 tolerance behaviour ---")
    near = gt2["chi2_statistic"] * 1.01           # inside 5% tolerance
    far = gt2["chi2_statistic"] * 1.60            # outside
    n_rec = score_record(base(q2, stdout=f"chi2={near}"), use_llm=False)
    f_rec = score_record(base(q2, stdout=f"chi2={far}"), use_llm=False)
    check("a value inside tolerance beats one outside",
          n_rec["scores"]["output_correctness"] > f_rec["scores"]["output_correctness"],
          f"near={n_rec['scores']['output_correctness']} "
          f"far={f_rec['scores']['output_correctness']}")

    print("\n--- D3 percent vs proportion (caught in a live run) ---")
    q1 = qs["titanic_q1"]
    # Ground truth stores rates as proportions; a system printing percentages
    # is equally correct and must not be scored as a miss.
    pct = ("Overall survival rate: 38.38%\nSurvival rate for female: 74.20%\n"
           "Survival rate for male: 18.89%")
    prop = ("overall 0.3838, female 0.742, male 0.1889")
    r_pct = score_record(base(q1, stdout=pct), use_llm=False)
    r_prop = score_record(base(q1, stdout=prop), use_llm=False)
    check("percentages score the same as proportions",
          r_pct["scores"]["output_correctness"]
          == r_prop["scores"]["output_correctness"] == 1.0,
          f"pct={r_pct['scores']['output_correctness']} "
          f"prop={r_prop['scores']['output_correctness']}")
    check("the rescale is disclosed in the score detail",
          "rescale" in r_pct["detail"]["output_correctness"],
          r_pct["detail"]["output_correctness"][:90])
    # The rescale must not make wrong answers pass.
    r_bad = score_record(base(q1, stdout="overall 12.0%, female 55%, male 61%"),
                         use_llm=False)
    check("rescaling does not credit genuinely wrong rates",
          r_bad["scores"]["output_correctness"] == 0.0,
          f"got {r_bad['scores']['output_correctness']}")

    print("\n--- D4 interpretation (heuristic mode) ---")
    rich = score_record(base(q2, final_answer=(
        "The chi-square test shows a significant association (p < 0.001). "
        "However, this is observational data, so it does not imply causation; "
        "the effect may be confounded by passenger class. Sample size is 891.")),
        use_llm=False)
    thin = score_record(base(q2, final_answer="The value is 260.7."), use_llm=False)
    check("a caveated interpretation outscores a bare number",
          rich["scores"]["interpretation_quality"] >
          thin["scores"]["interpretation_quality"],
          f"rich={rich['scores']['interpretation_quality']} "
          f"thin={thin['scores']['interpretation_quality']}")

    print("\n--- coverage flags ---")
    nogt = score_record(base(q2, ground_truth={}), use_llm=False)
    check("missing ground truth marks correctness uncovered, not zero",
          nogt["scores"]["output_correctness"] is None
          and nogt["coverage"]["output_correctness"] is False)
    check("overall stays computable when one dimension is uncovered",
          nogt["overall"] is not None, f"got {nogt['overall']}")

    print("\n--- end-to-end ordering ---")
    check("a fully correct record outscores a fully wrong one overall",
          score_record(base(q2, code="chi2_contingency(ct)", executed=True,
                            stdout=truthy, final_answer=truthy),
                       use_llm=False)["overall"]
          > score_record(base(q2, code="ttest_ind(a,b)", executed=False,
                              execution_error="boom",
                              final_answer="The value is 3.14."),
                         use_llm=False)["overall"])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all rubric checks passed")


if __name__ == "__main__":
    main()
