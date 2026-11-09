import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ------------------------------------------------------------------
# Paths
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q6/chart.png"

# Ensure the directory for the chart exists
os.makedirs(os.path.dirname(chart_path), exist_ok=True)

# ------------------------------------------------------------------
# Load data
df = pd.read_csv(csv_path)

# Keep only rows where both age and fare are present
df_clean = df[['age', 'fare']].dropna()

# Number of observations used
n = len(df_clean)

# ------------------------------------------------------------------
# Compute Pearson correlation
r, p_value = pearsonr(df_clean['age'], df_clean['fare'])

# Print the numbers a reader needs
print(f"Number of observations (non‑missing age & fare): {n}")
print(f"Pearson correlation coefficient (r): {r:.4f}")
print(f"Two‑tailed p‑value: {p_value:.4e}")

# ------------------------------------------------------------------
# Create scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_clean,
    x='age',
    y='fare',
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red'}
)
plt.title('Titanic: Age vs. Fare')
plt.xlabel('Age (years)')
plt.ylabel('Fare (USD)')

# Save the figure
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()