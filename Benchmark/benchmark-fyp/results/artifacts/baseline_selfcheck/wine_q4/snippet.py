import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute and print descriptive statistics for alcohol per quality level
# ------------------------------------------------------------------
stats = (
    df.groupby("quality")["alcohol"]
    .agg(
        count="size",
        mean="mean",
        std="std",
        min="min",
        q25=lambda x: x.quantile(0.25),
        median="median",
        q75=lambda x: x.quantile(0.75),
        max="max",
    )
    .reset_index()
)

print("Alcohol content statistics by wine quality level:")
print(stats.to_string(index=False))

# ------------------------------------------------------------------
# 3. Visualise the spread of alcohol across quality levels
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x="quality", y="alcohol", data=df, palette="viridis")
plt.title("Alcohol Content Distribution per Wine Quality Level")
plt.xlabel("Quality")
plt.ylabel("Alcohol (% vol)")

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q4"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()