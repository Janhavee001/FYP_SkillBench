"""
Build questions.json for each shared dataset, with *computed* reference answers.

Every numeric reference value in questions.json is produced here from the actual
CSV -- nothing is hand-typed -- so harness/rubric.py can score "output
correctness" against a trustworthy ground truth rather than a guess.

Run:  python datasets/build_questions.py
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent


def R(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, n)


# --------------------------------------------------------------------------- #
# Titanic
# --------------------------------------------------------------------------- #
def titanic():
    df = pd.read_csv(HERE / "titanic" / "titanic.csv")
    q = []

    # Q1 -- descriptive + group comparison
    overall = df["survived"].mean()
    by_sex = df.groupby("sex")["survived"].mean()
    q.append(dict(
        id="titanic_q1",
        question=("What is the overall survival rate on the Titanic, and how does it "
                  "differ between male and female passengers?"),
        kind="mixed", difficulty="easy",
        expected_variables=["survived", "sex"],
        expected_method=["descriptive_statistics", "groupby_aggregation"],
        acceptable_methods=["descriptive_statistics", "groupby_aggregation",
                            "proportion_comparison"],
        expected_chart=["bar", "grouped_bar"],
        reference_answer={"overall_survival_rate": R(overall),
                          "female_survival_rate": R(by_sex["female"]),
                          "male_survival_rate": R(by_sex["male"])},
        tolerance=0.01,
        notes=("Female survival rate is roughly 3.5x the male rate; a correct answer "
               "must report both group rates, not just the overall mean.")))

    # Q2 -- chi-square test of independence
    ct = pd.crosstab(df["sex"], df["survived"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    q.append(dict(
        id="titanic_q2",
        question=("Is there a statistically significant association between passenger sex "
                  "and survival? Report the test used, the test statistic and the p-value."),
        kind="stat", difficulty="medium",
        expected_variables=["sex", "survived"],
        expected_method=["chi_square_test"],
        acceptable_methods=["chi_square_test", "fisher_exact_test", "proportion_z_test"],
        expected_chart=["bar", "grouped_bar", "none"],
        reference_answer={"test": "chi-square test of independence",
                          "chi2_statistic": R(chi2), "p_value": R(p, 8),
                          "degrees_of_freedom": int(dof),
                          "significant_at_0.05": bool(p < 0.05)},
        tolerance=0.05,
        notes=("Both variables are categorical -> chi-square (or Fisher). Choosing a "
               "t-test or a correlation here is a method-selection error.")))

    # Q3 -- two-sample t-test on age
    a = df.loc[df["survived"] == 1, "age"].dropna()
    b = df.loc[df["survived"] == 0, "age"].dropna()
    t, tp = stats.ttest_ind(a, b, equal_var=False)
    _, up = stats.mannwhitneyu(a, b)
    q.append(dict(
        id="titanic_q3",
        question=("Does the mean age of passengers who survived differ significantly from "
                  "those who did not? Handle the missing age values and state how you "
                  "handled them."),
        kind="stat", difficulty="medium",
        expected_variables=["age", "survived"],
        expected_method=["t_test"],
        acceptable_methods=["t_test", "welch_t_test", "mann_whitney_u"],
        expected_chart=["box", "hist", "violin", "none"],
        reference_answer={"test": "Welch two-sample t-test",
                          "mean_age_survived": R(a.mean()),
                          "mean_age_died": R(b.mean()),
                          "t_statistic": R(t), "p_value": R(tp, 6),
                          "significant_at_0.05": bool(tp < 0.05),
                          "n_missing_age": int(df["age"].isna().sum()),
                          "mann_whitney_p": R(up, 6)},
        tolerance=0.05,
        notes=("177 ages are missing. A correct answer must state a missing-data "
               "strategy; silently dropping without mention is a partial-credit case.")))

    # Q4 -- survival by class, visual
    by_class = df.groupby("pclass")["survived"].mean()
    q.append(dict(
        id="titanic_q4",
        question=("How does survival rate vary across passenger classes? Produce a chart "
                  "that makes the comparison clear."),
        kind="viz", difficulty="easy",
        expected_variables=["pclass", "survived"],
        expected_method=["groupby_aggregation"],
        acceptable_methods=["groupby_aggregation", "descriptive_statistics"],
        expected_chart=["bar", "grouped_bar", "point"],
        reference_answer={"survival_rate_by_pclass":
                          {str(k): R(v) for k, v in by_class.items()},
                          "monotonic_decreasing_with_class":
                          bool(by_class.is_monotonic_decreasing)},
        tolerance=0.01,
        notes=("A bar chart of a rate by an ordinal category. A pie chart or a scatter "
               "plot is a chart-selection error.")))

    # Q5 -- fare distribution / skew
    fare = df["fare"].dropna()
    q.append(dict(
        id="titanic_q5",
        question=("Describe the distribution of ticket fares. Is it skewed, and what does "
                  "that imply for choosing summary statistics?"),
        kind="mixed", difficulty="medium",
        expected_variables=["fare"],
        expected_method=["descriptive_statistics", "distribution_analysis"],
        acceptable_methods=["descriptive_statistics", "distribution_analysis",
                            "normality_test"],
        expected_chart=["hist", "kde", "box"],
        reference_answer={"mean": R(fare.mean()), "median": R(fare.median()),
                          "std": R(fare.std()), "skewness": R(stats.skew(fare)),
                          "right_skewed": bool(stats.skew(fare) > 1),
                          "shapiro_p": R(stats.shapiro(
                              fare.sample(500, random_state=42))[1], 8)},
        tolerance=0.05,
        notes=("Strong right skew (~4.8). A good interpretation prefers the median over "
               "the mean.")))

    # Q6 -- correlation
    sub = df[["age", "fare"]].dropna()
    pr, pp = stats.pearsonr(sub["age"], sub["fare"])
    sr, sp = stats.spearmanr(sub["age"], sub["fare"])
    q.append(dict(
        id="titanic_q6",
        question=("Is there a relationship between a passenger's age and the fare they "
                  "paid? Quantify it and say whether it is strong."),
        kind="stat", difficulty="medium",
        expected_variables=["age", "fare"],
        expected_method=["correlation"],
        acceptable_methods=["correlation", "linear_regression"],
        expected_chart=["scatter", "none"],
        reference_answer={"pearson_r": R(pr), "pearson_p": R(pp, 6),
                          "spearman_rho": R(sr), "spearman_p": R(sp, 6),
                          "strength": "weak positive"},
        tolerance=0.05,
        notes=("r is ~0.1: statistically significant but practically negligible. Calling "
               "this a 'strong relationship' is an interpretation error even if the "
               "number is right.")))

    # Q7 -- logistic regression
    m = df[["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]].dropna().copy()
    m["sex"] = (m["sex"] == "male").astype(int)
    X, y = m.drop(columns="survived"), m["survived"]
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    coefs = dict(zip(X.columns, lr.coef_[0]))
    q.append(dict(
        id="titanic_q7",
        question=("Which passenger attributes best predict survival? Fit an appropriate "
                  "model and report which predictors matter most and in which direction."),
        kind="stat", difficulty="hard",
        expected_variables=["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"],
        expected_method=["logistic_regression"],
        acceptable_methods=["logistic_regression", "random_forest", "decision_tree",
                            "gradient_boosting"],
        expected_chart=["bar", "none"],
        reference_answer={"model": "logistic regression (standardised predictors)",
                          "train_accuracy": R(lr.score(Xs, y)),
                          "standardised_coefficients": {k: R(v) for k, v in coefs.items()},
                          "strongest_predictor": max(coefs, key=lambda k: abs(coefs[k])),
                          "sex_effect_direction": "male -> lower survival",
                          "n_used": int(len(m))},
        tolerance=0.10,
        notes=("Binary outcome -> logistic regression, not linear regression. Sex "
               "dominates; pclass is second. Reporting a linear-regression R^2 here is a "
               "method error.")))

    # Q8 -- missingness
    miss = df.isna().sum()
    miss = miss[miss > 0]
    q.append(dict(
        id="titanic_q8",
        question=("Which columns have missing values, how much is missing in each, and "
                  "what would you do about each one before modelling?"),
        kind="mixed", difficulty="easy",
        expected_variables=list(miss.index),
        expected_method=["missing_data_analysis"],
        acceptable_methods=["missing_data_analysis", "descriptive_statistics"],
        expected_chart=["bar", "heatmap", "none"],
        reference_answer={"missing_counts": {k: int(v) for k, v in miss.items()},
                          "missing_pct": {k: R(v / len(df) * 100, 2)
                                          for k, v in miss.items()},
                          "n_rows": int(len(df))},
        tolerance=0.0,
        notes=("deck is ~77% missing (drop), age ~20% (impute), embarked/embark_town 2 "
               "rows (drop rows). A correct answer differentiates treatment by severity.")))

    for item in q:
        item["dataset"] = "titanic"
    return q


# --------------------------------------------------------------------------- #
# Wine Quality (UCI, red)
# --------------------------------------------------------------------------- #
def wine():
    df = pd.read_csv(HERE / "wine_quality" / "winequality-red.csv")
    q = []

    vc = df["quality"].value_counts().sort_index()
    q.append(dict(
        id="wine_q1",
        question=("What is the distribution of wine quality scores in this dataset? Is it "
                  "balanced?"),
        kind="mixed", difficulty="easy",
        expected_variables=["quality"],
        expected_method=["descriptive_statistics", "distribution_analysis"],
        acceptable_methods=["descriptive_statistics", "distribution_analysis",
                            "frequency_count"],
        expected_chart=["bar", "hist"],
        reference_answer={"counts": {str(k): int(v) for k, v in vc.items()},
                          "mean_quality": R(df["quality"].mean()),
                          "modal_quality": int(df["quality"].mode()[0]),
                          "balanced": False,
                          "pct_in_top_two_classes": R((vc[5] + vc[6]) / len(df) * 100, 2)},
        tolerance=0.01,
        notes=("Heavily imbalanced: ~82% of wines are quality 5 or 6. Noting the "
               "imbalance is part of a correct answer.")))

    corr = df.corr(numeric_only=True)["quality"].drop("quality")
    corr = corr.sort_values(key=abs, ascending=False)
    q.append(dict(
        id="wine_q2",
        question=("Which physicochemical properties are most strongly correlated with "
                  "wine quality? Show the relationships."),
        kind="mixed", difficulty="medium",
        expected_variables=list(df.columns),
        expected_method=["correlation"],
        acceptable_methods=["correlation", "feature_importance", "linear_regression"],
        expected_chart=["heatmap", "bar"],
        reference_answer={"top_3_by_abs_correlation": list(corr.index[:3]),
                          "correlations": {k: R(v) for k, v in corr.items()},
                          "strongest_positive": corr[corr > 0].idxmax(),
                          "strongest_negative": corr[corr < 0].idxmin()},
        tolerance=0.03,
        notes=("alcohol (+0.48), then volatile_acidity (-0.39), then sulphates (+0.25). "
               "A correlation heatmap is the canonical chart.")))

    hi = df.loc[df["quality"] >= 7, "alcohol"]
    lo = df.loc[df["quality"] <= 4, "alcohol"]
    t, tp = stats.ttest_ind(hi, lo, equal_var=False)
    q.append(dict(
        id="wine_q3",
        question=("Do high-quality wines (quality >= 7) have significantly higher alcohol "
                  "content than low-quality wines (quality <= 4)?"),
        kind="stat", difficulty="medium",
        expected_variables=["alcohol", "quality"],
        expected_method=["t_test"],
        acceptable_methods=["t_test", "welch_t_test", "mann_whitney_u"],
        expected_chart=["box", "violin", "hist", "none"],
        reference_answer={"test": "Welch two-sample t-test",
                          "mean_alcohol_high": R(hi.mean()),
                          "mean_alcohol_low": R(lo.mean()),
                          "n_high": int(len(hi)), "n_low": int(len(lo)),
                          "t_statistic": R(t), "p_value": R(tp, 8),
                          "significant_at_0.05": bool(tp < 0.05)},
        tolerance=0.05,
        notes=("Continuous outcome, two independent groups -> t-test. Unequal group "
               "sizes (217 vs 63) make Welch the safer choice.")))

    grp = df.groupby("quality")["alcohol"].mean()
    q.append(dict(
        id="wine_q4",
        question=("Visualise how alcohol content varies across every quality level, "
                  "showing the spread within each level, not just the average."),
        kind="viz", difficulty="medium",
        expected_variables=["alcohol", "quality"],
        expected_method=["groupby_aggregation", "distribution_analysis"],
        acceptable_methods=["groupby_aggregation", "distribution_analysis"],
        expected_chart=["box", "violin", "strip"],
        reference_answer={"mean_alcohol_by_quality": {str(k): R(v) for k, v in grp.items()},
                          "trend": "increasing above quality 5"},
        tolerance=0.05,
        notes=("The question explicitly asks for spread -> box/violin. A plain bar chart "
               "of means answers a different question and loses chart-selection credit.")))

    def iqr_out(s):
        q1, q3 = s.quantile([.25, .75])
        i = q3 - q1
        return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())

    q.append(dict(
        id="wine_q5",
        question=("Are there outliers in residual sugar and total sulfur dioxide? "
                  "Identify them and say whether they should be removed."),
        kind="mixed", difficulty="medium",
        expected_variables=["residual_sugar", "total_sulfur_dioxide"],
        expected_method=["outlier_detection"],
        acceptable_methods=["outlier_detection", "descriptive_statistics",
                            "distribution_analysis"],
        expected_chart=["box", "hist", "scatter"],
        reference_answer={"iqr_outliers_residual_sugar": iqr_out(df["residual_sugar"]),
                          "iqr_outliers_total_sulfur_dioxide":
                              iqr_out(df["total_sulfur_dioxide"]),
                          "max_residual_sugar": R(df["residual_sugar"].max()),
                          "max_total_sulfur_dioxide": R(df["total_sulfur_dioxide"].max()),
                          "method": "1.5 x IQR rule"},
        tolerance=0.05,
        notes=("A correct answer states the outlier rule used. Recommending blind "
               "removal without noting these are legitimate chemical readings is an "
               "interpretation weakness.")))

    X = df.drop(columns="quality")
    y = df["quality"]
    lin = LinearRegression().fit(X, y)
    r2 = lin.score(X, y)
    co = dict(zip(X.columns, lin.coef_))
    q.append(dict(
        id="wine_q6",
        question=("Build a model that predicts wine quality from its physicochemical "
                  "properties. Report how well it performs and which features drive the "
                  "prediction."),
        kind="stat", difficulty="hard",
        expected_variables=list(df.columns),
        expected_method=["linear_regression"],
        acceptable_methods=["linear_regression", "random_forest", "gradient_boosting",
                            "ridge_regression", "ordinal_regression"],
        expected_chart=["bar", "scatter", "none"],
        reference_answer={"model": "OLS linear regression, all 11 predictors",
                          "r_squared_in_sample": R(r2),
                          "coefficients": {k: R(v) for k, v in co.items()},
                          "largest_abs_coefficient": max(co, key=lambda k: abs(co[k]))},
        tolerance=0.10,
        notes=("R^2 is only ~0.36 -- a correct answer reports the weak fit honestly "
               "rather than presenting the model as accurate.")))

    pr, pp = stats.pearsonr(df["volatile_acidity"], df["quality"])
    q.append(dict(
        id="wine_q7",
        question=("Is volatile acidity negatively associated with wine quality? Quantify "
                  "the association and comment on whether it implies causation."),
        kind="stat", difficulty="medium",
        expected_variables=["volatile_acidity", "quality"],
        expected_method=["correlation"],
        acceptable_methods=["correlation", "linear_regression", "t_test"],
        expected_chart=["scatter", "box", "none"],
        reference_answer={"pearson_r": R(pr), "p_value": R(pp, 10),
                          "direction": "negative",
                          "significant_at_0.05": bool(pp < 0.05),
                          "implies_causation": False},
        tolerance=0.03,
        notes=("Explicitly probes the correlation-vs-causation distinction; the "
               "interpretation dimension should penalise a causal claim.")))

    pr2, _ = stats.pearsonr(df["density"], df["alcohol"])
    q.append(dict(
        id="wine_q8",
        question=("Show the relationship between density and alcohol content, with a "
                  "fitted trend line, and explain the pattern."),
        kind="viz", difficulty="easy",
        expected_variables=["density", "alcohol"],
        expected_method=["correlation", "linear_regression"],
        acceptable_methods=["correlation", "linear_regression", "scatter_analysis"],
        expected_chart=["scatter"],
        reference_answer={"pearson_r": R(pr2), "direction": "negative",
                          "strength": "strong",
                          "explanation": ("ethanol is less dense than water, so higher "
                                          "alcohol lowers overall density")},
        tolerance=0.03,
        notes="Scatter plus a regression line is the only correct chart family here."))

    for item in q:
        item["dataset"] = "wine_quality"
    return q


# --------------------------------------------------------------------------- #
DESCRIPTIONS = {
    "titanic": ("Passenger manifest of the RMS Titanic (seaborn distribution, 891 rows). "
                "Mixed categorical/numeric with genuine missingness."),
    "wine_quality": ("UCI Wine Quality, red variant (1599 rows). Eleven continuous "
                     "physicochemical predictors and an ordinal quality score 3-8."),
}
SOURCES = {
    "titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
    "wine_quality": ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "wine-quality/winequality-red.csv"),
}


def main():
    specs = [("titanic", titanic(), "titanic/titanic.csv"),
             ("wine_quality", wine(), "wine_quality/winequality-red.csv")]
    for name, items, csv in specs:
        df = pd.read_csv(HERE / csv)
        payload = {
            "dataset": name,
            "csv_path": csv,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": {c: str(df[c].dtype) for c in df.columns},
            "description": DESCRIPTIONS[name],
            "source": SOURCES[name],
            "note_on_ground_truth": (
                "Every value under reference_answer was computed by "
                "datasets/build_questions.py against the CSV in this folder. Re-run that "
                "script if you edit the CSV. Add your own questions by appending objects "
                "with the same schema."),
            "questions": items,
        }
        out = HERE / name / "questions.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("wrote {}  ({} questions)".format(out, len(items)))


if __name__ == "__main__":
    main()
