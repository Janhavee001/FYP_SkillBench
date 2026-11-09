import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q8/chart.png"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# Compute missingness statistics
# ------------------------------------------------------------------
total_rows = len(df)
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / total_rows * 100).round(2)

# Print only columns that actually have missing values
print("=== Missing‑value summary ===")
for col in df.columns:
    if missing_counts[col] > 0:
        print(f"{col}: {missing_counts[col]} missing values "
              f"({missing_percent[col]}% of {total_rows} rows)")

# ------------------------------------------------------------------
# Visualise missingness as a heatmap
# ------------------------------------------------------------------
# Create folder if it does not exist
os.makedirs(os.path.dirname(chart_path), exist_ok=True)

plt.figure(figsize=(12, 6))
# The heatmap shows True (missing) as a colored cell
sns.heatmap(df.isnull(),
            cbar=False,
            yticklabels=False,
            cmap="viridis")
plt.title("Missing‑value matrix for Titanic dataset")
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()