import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------- 1. Load data ----------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ---------- 2. Handle missing ages ----------
# For the comparison we drop rows with missing age (listwise deletion)
df_clean = df.dropna(subset=["age"])

# ---------- 3. Split by survival ----------
age_survived = df_clean.loc[df_clean["survived"] == 1, "age"]
age_not = df_clean.loc[df_clean["survived"] == 0, "age"]

# ---------- 4. Descriptive statistics ----------
mean_surv = age_survived.mean()
std_surv = age_survived.std(ddof=1)
n_surv = age_survived.shape[0]

mean_not = age_not.mean()
std_not = age_not.std(ddof=1)
n_not = age_not.shape[0]

print(f"Survived:  n = {n_surv}, mean age = {mean_surv:.2f}, std = {std_surv:.2f}")
print(f"Did not survive:  n = {n_not}, mean age = {mean_not:.2f}, std = {std_not:.2f}")

# ---------- 5. Statistical test (Welch's t‑test) ----------
t_stat, p_val = stats.ttest_ind(age_survived, age_not, equal_var=False, nan_policy='omit')
print(f"Welch's t‑test: t = {t_stat:.4f}, p‑value = {p_val:.4f}")

# ---------- 6. Visualisation ----------
plt.figure(figsize=(8, 5))
sns.boxplot(x="survived", y="age", data=df_clean, palette="Set2")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.title("Age distribution by Survival status")

# Save the chart
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q3/chart.png"
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()