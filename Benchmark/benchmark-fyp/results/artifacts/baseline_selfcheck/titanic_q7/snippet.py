import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
# 1. Load data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Define target and feature columns
target = "survived"
X = df.drop(columns=[target])

# ------------------------------------------------------------------
# 3. Identify column types
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
bool_features    = ["adult_male", "alone"]
categorical_features = [
    "sex", "embarked", "class", "who",
    "deck", "embark_town", "alive"
]

# ------------------------------------------------------------------
# 4. Build preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Convert bool → int, then impute most frequent (0 or 1)
bool_transformer = Pipeline(steps=[
    ("to_int", FunctionTransformer(lambda x: x.astype(int), validate=False)),
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("bool", bool_transformer, bool_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ------------------------------------------------------------------
# 5. Append logistic regression model
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("logreg", LogisticRegression(max_iter=1000, solver="liblinear"))
])

# ------------------------------------------------------------------
# 6. Fit the model
clf.fit(X, df[target])

# ------------------------------------------------------------------
# 7. Retrieve feature names after preprocessing
# (required for interpreting coefficients)
ohe = clf.named_steps["preprocess"].named_transformers_["cat"]["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

# combine all feature names in the order they appear in the transformed matrix
feature_names = (
    numeric_features +
    bool_features +          # after bool_transformer they remain as single columns
    list(cat_feature_names)
)

# ------------------------------------------------------------------
# 8. Extract coefficients
coefficients = clf.named_steps["logreg"].coef_[0]

# Create a DataFrame for easy sorting / printing
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coef": np.abs(coefficients)
}).sort_values(by="abs_coef", ascending=False)

# ------------------------------------------------------------------
# 9. Print results that a reader needs
print("\n=== Logistic Regression Coefficients (sorted by absolute magnitude) ===")
for _, row in coef_df.iterrows():
    print(f"{row['feature']:<30} {row['coefficient']:+.4f}")

# Show top 5 predictors explicitly
top5 = coef_df.head(5)
print("\nTop 5 predictors by absolute effect:")
for _, row in top5.iterrows():
    direction = "increases" if row["coefficient"] > 0 else "decreases"
    print(f"- {row['feature']}: {direction} survival probability (coef={row['coefficient']:.4f})")

# ------------------------------------------------------------------
# 10. Visualise the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(
    x="coefficient",
    y="feature",
    data=coef_df,
    palette="vlag"
)
plt.title("Logistic Regression Coefficients – Titanic Survival")
plt.xlabel("Coefficient (log‑odds)")
plt.ylabel("Feature")

# Ensure output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q7"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"\nBar‑chart of coefficients saved to: {out_path}")