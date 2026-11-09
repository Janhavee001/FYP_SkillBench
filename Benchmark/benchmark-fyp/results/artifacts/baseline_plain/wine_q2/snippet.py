import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as delimiter

# ------------------------------------------------------------------
# 2. Compute Pearson correlation of each physicochemical variable with quality
# ------------------------------------------------------------------
target = "quality"
features = [col for col in df.columns if col != target]

corr_series = df[features].corrwith(df[target])
# Sort by absolute correlation (strongest relationship first)
corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

# ------------------------------------------------------------------
# 3. Print the correlations (rounded to 3 decimals)
# ------------------------------------------------------------------
print("Pearson correlation of each physicochemical property with wine quality:")
for var, corr in corr_sorted.items():
    print(f"{var:>22}: {corr:.3f}")

# ------------------------------------------------------------------
# 4. Visualise the correlations as a bar plot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x=corr_sorted.values, y=corr_sorted.index, palette="viridis")
plt.axvline(0, color='grey', linewidth=0.8)
plt.title("Correlation of physicochemical properties with wine quality")
plt.xlabel("Pearson correlation coefficient")
plt.ylabel("Physicochemical property")
plt.tight_layout()

# ------------------------------------------------------------------
# 5. Save the chart
# ------------------------------------------------------------------
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q2"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.savefig(out_path, dpi=300)
plt.close()