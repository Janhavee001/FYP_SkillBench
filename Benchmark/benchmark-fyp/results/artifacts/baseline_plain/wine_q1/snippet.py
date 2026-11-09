import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q1/chart.png"

# Load data
df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as delimiter

# Compute distribution of the 'quality' column
quality_counts = df['quality'].value_counts().sort_index()
total = quality_counts.sum()
quality_percent = (quality_counts / total * 100).round(2)

# Print a readable table
print("Quality score distribution (count, % of total):")
for q in quality_counts.index:
    print(f"  {q}: {quality_counts[q]} ({quality_percent[q]}%)")

# Determine if the dataset is balanced (simple heuristic: max/min ratio)
balance_ratio = quality_counts.max() / quality_counts.min()
print(f"\nBalance ratio (max count / min count): {balance_ratio:.2f}")

# Create and save a bar chart
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette="viridis")
plt.title("Distribution of Wine Quality Scores (Red Wine)")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.tight_layout()

# Ensure the output directory exists
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.savefig(chart_path, dpi=300)
plt.close()