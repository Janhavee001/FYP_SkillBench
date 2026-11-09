import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"

# The original file uses ';' as delimiter
df = pd.read_csv(csv_path, sep=';')

# ------------------------------------------------------------------
# 2. Identify the target column (robust to whitespace / case issues)
# ------------------------------------------------------------------
# Look for a column whose stripped, lower‑cased name equals "quality"
target_candidates = [c for c in df.columns if c.strip().lower() == "quality"]
if not target_candidates:
    raise KeyError(
        "No column matching 'quality' (case‑insensitive, stripped) was found in the dataset. "
        f"Available columns: {list(df.columns)}"
    )
target_col = target_candidates[0]          # use the first match

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# ------------------------------------------------------------------
# 3. Train‑test split
# ------------------------------------------------------------------
# Using stratification on the target helps keep the distribution of quality scores
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------------
# 4. Model: Random Forest Regressor
# ------------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. Predict & evaluate
# ------------------------------------------------------------------
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² (coefficient of determination): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# ------------------------------------------------------------------
# 6. Feature importance
# ------------------------------------------------------------------
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

# Print top 5 drivers
top5 = importance_df.head(5)
print("\nTop 5 most important features:")
for _, row in top5.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# ------------------------------------------------------------------
# 7. Plot feature importances
# ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(
    x="importance",
    y="feature",
    data=importance_df,
    palette="viridis"
)
plt.title("Feature Importances from Random Forest Regressor")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q6"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.savefig(out_path, dpi=300)
# Do NOT call plt.show()