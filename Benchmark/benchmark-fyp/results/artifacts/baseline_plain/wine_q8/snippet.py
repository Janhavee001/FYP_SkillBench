import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# ------------------------------------------------------------------
# 1. Load the data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as delimiter

# ------------------------------------------------------------------
# 2. Compute linear regression statistics for density vs. alcohol
x = df["density"]
y = df["alcohol"]
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Print the numbers a reader needs
print("Linear regression of Alcohol on Density:")
print(f"  slope      = {slope:.6f}")
print(f"  intercept  = {intercept:.6f}")
print(f"  r (Pearson correlation) = {r_value:.6f}")
print(f"  r-squared  = {r_value**2:.6f}")
print(f"  p-value    = {p_value:.6e}")
print(f"  std_err    = {std_err:.6f}")

# ------------------------------------------------------------------
# 3. Create the plot (scatter + fitted line)
plt.figure(figsize=(8, 6))
sns.regplot(x=x, y=y, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("Density vs. Alcohol Content (Red Wine)")
plt.xlabel("Density (g/cm³)")
plt.ylabel("Alcohol (% vol)")

# 4. Save the figure
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q8"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()