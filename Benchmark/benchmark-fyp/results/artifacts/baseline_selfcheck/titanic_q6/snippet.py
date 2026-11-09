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
# 1. Load data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# 2. Keep only rows where both age and fare are present
df_clean = df[['age', 'fare']].dropna()

# 3. Compute Pearson correlation and p‑value
age = df_clean['age'].values
fare = df_clean['fare'].values
r, p_value = pearsonr(age, fare)

# 4. Print the numbers a reader needs
print(f"Number of observations (non‑missing age & fare): {len(df_clean)}")
print(f"Pearson correlation coefficient (r) between age and fare: {r:.4f}")
print(f"P‑value for the correlation: {p_value:.4e}")

# 5. Create and save a scatter plot with a regression line
plt.figure(figsize=(8, 6))
sns.regplot(x='age', y='fare', data=df_clean,
            scatter_kws={'alpha':0.6, 's':30},
            line_kws={'color':'red'})
plt.title('Age vs. Fare (Titanic passengers)')
plt.xlabel('Age (years)')
plt.ylabel('Fare (USD)')

# Ensure the output directory exists
out_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q6"
os.makedirs(out_path, exist_ok=True)
chart_file = os.path.join(out_path, "chart.png")
plt.tight_layout()
plt.savefig(chart_file, dpi=300)
plt.close()