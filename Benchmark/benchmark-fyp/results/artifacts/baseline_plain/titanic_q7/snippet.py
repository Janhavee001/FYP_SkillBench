import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ---------- 1. Load data ----------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ---------- 2. Basic preprocessing ----------
# Separate target
y = df["survived"]
X = df.drop(columns=["survived"])

# Identify column types
numeric_cols = ["age", "sibsp", "parch", "fare", "pclass"]
categorical_cols = ["sex", "embarked", "class", "who", "deck",
                    "embark_town", "alive", "alone", "adult_male"]

# Impute missing values
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# ---------- 3. Model ----------
logreg = LogisticRegression(max_iter=1000, solver="liblinear")  # liblinear works well for small data

# Build full pipeline
clf = Pipeline(steps=[("preprocess", preprocessor),
                     ("model", logreg)])

# ---------- 4. Train / evaluate ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf.fit(X_train, y_train)

# Cross‑validated accuracy (5‑fold)
cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"Cross‑validated accuracy (5‑fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------- 5. Extract and display coefficients ----------
# Get feature names after one‑hot encoding
ohe = clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names = numeric_cols + list(cat_feature_names)

coefficients = clf.named_steps["model"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coef": np.abs(coefficients)
}).sort_values(by="abs_coef", ascending=False)

# Print top predictors
print("\nTop predictors (sorted by absolute coefficient):")
print(coef_df.head(10).to_string(index=False))

# ---------- 6. Plot feature importance ----------
plt.figure(figsize=(10, 6))
sns.barplot(
    x="abs_coef",
    y="feature",
    data=coef_df.head(15),
    palette="viridis"
)
plt.title("Top 15 Feature Importances (|log‑odds coefficient|) – Logistic Regression")
plt.xlabel("Absolute coefficient (|log‑odds|)")
plt.ylabel("Feature")

# Ensure output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q7"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()