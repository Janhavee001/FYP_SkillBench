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
df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as delimiter

# ------------------------------------------------------------------
# 2. Identify the target column (quality) robustly
# ------------------------------------------------------------------
# Some versions of the dataset may have extra whitespace or different capitalisation.
target_candidates = [col for col in df.columns if col.strip().lower() == "quality"]
if not target_candidates:
    raise KeyError(
        "Could not find a column representing wine quality. "
        f"Available columns: {list(df.columns)}"
    )
target = target_candidates[0]  # use the first match

# ------------------------------------------------------------------
# 3. Compute Pearson correlation of each physicochemical variable with quality
# ------------------------------------------------------------------
features = [col for col in df.columns if col != target]

# Pearson correlation (r) for each feature
corr_series = df[features].corrwith(df[target])

# Print the correlations (sorted by absolute magnitude, descending)
print("Pearson correlation of each physicochemical property with wine quality:")
for feat, corr in (
    corr_series.reindex(corr_series.abs().sort_values(ascending=False).index).items()
):
    print(f"{feat:25s}: {corr:.4f}")

# ------------------------------------------------------------------
# 4. Visualise the correlations as a bar plot
# ------------------------------------------------------------------
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q2"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_series.index, y=corr_series.values, palette="viridis")
plt.title("Correlation of Physicochemical Properties with Wine Quality")
plt.ylabel("Pearson r")
plt.xlabel("Physicochemical Property")
plt.ylim(-1, 1)
plt.axhline(0, color="gray", linewidth=0.8)
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()