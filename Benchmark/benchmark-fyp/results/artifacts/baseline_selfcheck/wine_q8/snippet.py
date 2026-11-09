import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# ------------------------------------------------------------------
# 1. Load data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute Pearson correlation between density and alcohol
corr, p_value = stats.pearsonr(df["density"], df["alcohol"])
print(f"Pearson correlation (density vs alcohol): {corr:.4f}")
print(f"P‑value of correlation: {p_value:.4e}")

# ------------------------------------------------------------------
# 3. Fit a simple linear regression (alcohol = a * density + b)
slope, intercept, r_value, p_val, std_err = stats.linregress(df["density"], df["alcohol"])
print(f"Linear regression slope: {slope:.4f}")
print(f"Linear regression intercept: {intercept:.4f}")
print(f"R‑squared: {r_value**2:.4f}")

# ------------------------------------------------------------------
# 4. Create scatter plot with regression line
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.regplot(x="density", y="alcohol", data=df,
            scatter_kws={"s": 30, "alpha": 0.6},
            line_kws={"color": "red", "lw": 2})

plt.title("Relationship between Density and Alcohol Content (Red Wine)")
plt.xlabel("Density")
plt.ylabel("Alcohol (% vol)")

# ------------------------------------------------------------------
# 5. Save the figure
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q8"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
# Do NOT call plt.show()