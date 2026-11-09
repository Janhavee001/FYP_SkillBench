import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path, sep=';')   # the original file uses ';' as delimiter

# -----------------------------
# 2. Compute and print descriptive statistics per quality level
# -----------------------------
stats = df.groupby('quality')['alcohol'].describe(percentiles=[0.25, 0.5, 0.75])
# Rename columns for clarity
stats = stats.rename(columns={
    'count': 'n',
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    '25%': 'Q1',
    '50%': 'median',
    '75%': 'Q3',
    'max': 'max'
})

print("Alcohol content statistics by wine quality level:")
for quality, row in stats.iterrows():
    print(f"\nQuality = {int(quality)}")
    print(f"  n      : {int(row['n'])}")
    print(f"  mean   : {row['mean']:.3f}")
    print(f"  std    : {row['std']:.3f}")
    print(f"  min    : {row['min']:.3f}")
    print(f"  Q1     : {row['Q1']:.3f}")
    print(f"  median : {row['median']:.3f}")
    print(f"  Q3     : {row['Q3']:.3f}")
    print(f"  max    : {row['max']:.3f}")

# -----------------------------
# 3. Visualise the spread with a boxplot
# -----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x='quality', y='alcohol', data=df, palette='Set2')
plt.title('Alcohol Content Distribution Across Wine Quality Levels')
plt.xlabel('Quality Score')
plt.ylabel('Alcohol (% vol)')

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q4"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()