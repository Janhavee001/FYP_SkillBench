import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ------------------------------------------------------------------
# 1. Load data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Handle missing ages
#    We drop rows where age is missing (listwise deletion). This keeps the
#    analysis simple and avoids introducing imputation bias.
df_clean = df.dropna(subset=["age"])

# ------------------------------------------------------------------
# 3. Split by survival status
survived = df_clean[df_clean["survived"] == 1]["age"]
not_survived = df_clean[df_clean["survived"] == 0]["age"]

# ------------------------------------------------------------------
# 4. Compute descriptive statistics
mean_survived = survived.mean()
mean_not_survived = not_survived.mean()
n_survived = survived.shape[0]
n_not_survived = not_survived.shape[0]

print(f"Number of passengers with known age who survived: {n_survived}")
print(f"Number of passengers with known age who did NOT survive: {n_not_survived}")
print(f"Mean age of survivors: {mean_survived:.2f} years")
print(f"Mean age of non‑survivors: {mean_not_survived:.2f} years")

# ------------------------------------------------------------------
# 5. Statistical test (Welch's t‑test, does not assume equal variances)
t_stat, p_val = ttest_ind(survived, not_survived, equal_var=False, nan_policy='omit')
print(f"T‑statistic: {t_stat:.4f}")
print(f"P‑value: {p_val:.4f}")

# ------------------------------------------------------------------
# 6. Optional visualisation
plt.figure(figsize=(8, 5))
sns.boxplot(x="survived", y="age", data=df_clean, palette="Set2")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age (years)")
plt.title("Age distribution by survival status")

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q3"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()