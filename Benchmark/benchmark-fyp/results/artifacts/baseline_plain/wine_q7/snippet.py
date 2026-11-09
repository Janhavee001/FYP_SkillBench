import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

# ------------------------------------------------------------------
# Paths
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q7/chart.png"

# Ensure the directory for the chart exists
os.makedirs(os.path.dirname(chart_path), exist_ok=True)

# ------------------------------------------------------------------
# Load data
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 1. Pearson correlation between volatile_acidity and quality
corr_coef, corr_p = pearsonr(df["volatile_acidity"], df["quality"])
print(f"Pearson correlation coefficient: {corr_coef:.4f}")
print(f"Pearson correlation p‑value: {corr_p:.4e}")

# ------------------------------------------------------------------
# 2. Simple linear regression (quality ~ volatile_acidity)
X = sm.add_constant(df["volatile_acidity"])   # adds intercept term
model = sm.OLS(df["quality"], X).fit()
slope = model.params["volatile_acidity"]
intercept = model.params["const"]
r_squared = model.rsquared
reg_p = model.pvalues["volatile_acidity"]

print(f"Linear regression intercept: {intercept:.4f}")
print(f"Linear regression slope (quality per unit volatile_acidity): {slope:.4f}")
print(f"R‑squared: {r_squared:.4f}")
print(f"Slope p‑value: {reg_p:.4e}")

# ------------------------------------------------------------------
# 3. Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x="volatile_acidity", y="quality", data=df,
            scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
plt.title("Wine Quality vs. Volatile Acidity")
plt.xlabel("Volatile Acidity")
plt.ylabel("Quality (score)")
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()