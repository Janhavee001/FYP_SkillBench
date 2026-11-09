import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Function to compute IQR bounds and flag outliers
# ------------------------------------------------------------------
def iqr_outliers(series):
    """Return a boolean mask of outliers using the 1.5*IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return mask, lower, upper

# ------------------------------------------------------------------
# 3. Detect outliers for the two variables of interest
# ------------------------------------------------------------------
variables = ["residual_sugar", "total_sulfur_dioxide"]
for var in variables:
    mask, lower, upper = iqr_outliers(df[var])
    out_idx = df.index[mask].tolist()
    out_vals = df.loc[mask, var].tolist()
    
    print(f"\n{var}:")
    print(f"  Lower bound (1.5*IQR) = {lower:.3f}")
    print(f"  Upper bound (1.5*IQR) = {upper:.3f}")
    print(f"  Number of outliers   = {mask.sum()}")
    if mask.sum() > 0:
        # show up to first 10 outliers for readability
        print(f"  Outlier indices (first 10) = {out_idx[:10]}")
        print(f"  Outlier values  (first 10) = {[round(v,3) for v in out_vals[:10]]}")
    else:
        print("  No outliers detected.")

# ------------------------------------------------------------------
# 4. Visualise with a side‑by‑side boxplot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[variables], palette="pastel")
plt.title("Boxplot of Residual Sugar and Total Sulfur Dioxide")
plt.ylabel("Value")

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q5"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f"\nBoxplot saved to: {out_path}")