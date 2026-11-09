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
# 2. Define IQR outlier helper
# ------------------------------------------------------------------
def iqr_outliers(series):
    """Return lower/upper fences and a boolean mask of outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask = (series < lower_fence) | (series > upper_fence)
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "mask": outlier_mask
    }

# ------------------------------------------------------------------
# 3. Detect outliers for the two variables of interest
# ------------------------------------------------------------------
variables = ["residual_sugar", "total_sulfur_dioxide"]
results = {}

for var in variables:
    res = iqr_outliers(df[var])
    results[var] = res
    out_idx = df.index[res["mask"]].tolist()
    out_vals = df.loc[out_idx, var].tolist()
    print(f"\n=== {var.upper()} ===")
    print(f"Q1 = {res['q1']:.3f}, Q3 = {res['q3']:.3f}, IQR = {res['iqr']:.3f}")
    print(f"Lower fence = {res['lower_fence']:.3f}, Upper fence = {res['upper_fence']:.3f}")
    print(f"Number of outliers = {len(out_idx)}")
    if len(out_idx) > 0:
        print("Outlier indices and values:")
        for i, v in zip(out_idx, out_vals):
            print(f"  index {i}: {v}")

# ------------------------------------------------------------------
# 4. Visualise with boxplots (saved to file, not shown)
# ------------------------------------------------------------------
output_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q5"
os.makedirs(output_dir, exist_ok=True)
chart_path = os.path.join(output_dir, "chart.png")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[variables], orient="h", palette="Set2")
plt.title("Boxplots for Residual Sugar and Total Sulfur Dioxide")
plt.xlabel("Value")
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()

print(f"\nBoxplot saved to: {chart_path}")