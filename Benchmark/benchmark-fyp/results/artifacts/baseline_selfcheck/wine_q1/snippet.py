import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q1/chart.png"

# Load data
df = pd.read_csv(csv_path)

# Compute distribution of the 'quality' column
quality_counts = df['quality'].value_counts().sort_index()
total = quality_counts.sum()
quality_percent = (quality_counts / total * 100).round(2)

# Print counts and percentages
print("Wine quality distribution (count):")
print(quality_counts.to_string())
print("\nWine quality distribution (percentage):")
print(quality_percent.to_string())

# Create output directory if it does not exist
os.makedirs(os.path.dirname(chart_path), exist_ok=True)

# Plot bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette="viridis")
plt.title("Distribution of Wine Quality Scores (Red Wine)")
plt.xlabel("Quality Score")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()