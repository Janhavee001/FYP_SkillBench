import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
csv_path = Path(r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv")
df = pd.read_csv(csv_path, sep=';')          # the original file uses ';' as delimiter
# Verify columns
expected_cols = ["fixed_acidity","volatile_acidity","citric_acid","residual_sugar",
                 "chlorides","free_sulfur_dioxide","total_sulfur_dioxide",
                 "density","ph","sulphates","alcohol","quality"]
assert list(df.columns) == expected_cols, "Unexpected column order"

# ------------------------------------------------------------------
# 2. Train‑test split
# ------------------------------------------------------------------
X = df.drop(columns="quality")
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------
# 3. Model – Random Forest Regressor
# ------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# For an ordinal target it is also useful to see how often the rounded
# prediction matches the true class.
y_pred_rounded = np.rint(y_pred).astype(int)
# Clip to the valid range (3‑8) to avoid impossible labels after rounding
y_pred_rounded = np.clip(y_pred_rounded, 3, 8)
accuracy = accuracy_score(y_test, y_pred_rounded)

print(f"R² (coefficient of determination): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Rounded‑prediction Accuracy: {accuracy:.4%}")

# ------------------------------------------------------------------
# 5. Feature importance
# ------------------------------------------------------------------
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("\nFeature importances (descending):")
for feat, imp in feat_imp.iteritems():
    print(f"{feat:20s}: {imp:.4f}")

# ------------------------------------------------------------------
# 6. Plot and save feature‑importance chart
# ------------------------------------------------------------------
output_dir = Path(r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q6")
output_dir.mkdir(parents=True, exist_ok=True)
chart_path = output_dir / "chart.png"

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"\nFeature‑importance chart saved to: {chart_path}")