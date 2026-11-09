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
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path)

# ---------- 2. Define groups ----------
high = df[df["quality"] >= 7]["alcohol"]
low  = df[df["quality"] <= 4]["alcohol"]

# ---------- 3. Descriptive statistics ----------
n_high = high.shape[0]
n_low  = low.shape[0]
mean_high = high.mean()
mean_low  = low.mean()
std_high  = high.std(ddof=1)
std_low   = low.std(ddof=1)

print("Sample sizes:")
print(f"  High‑quality (quality >=7): {n_high}")
print(f"  Low‑quality  (quality <=4): {n_low}")

print("\nAlcohol content (percent) – descriptive stats:")
print(f"  High‑quality: mean = {mean_high:.3f}, std = {std_high:.3f}")
print(f"  Low‑quality : mean = {mean_low:.3f}, std = {std_low:.3f}")

# ---------- 4. Statistical test ----------
t_stat, p_val = stats.ttest_ind(high, low, equal_var=False)  # Welch's t‑test
df_eff = (std_high**2/n_high + std_low**2/n_low)**2 / \
         ((std_high**2/n_high)**2/(n_high-1) + (std_low**2/n_low)**2/(n_low-1))

print("\nWelch's t‑test results:")
print(f"  t = {t_stat:.4f}")
print(f"  df (approx.) = {df_eff:.2f}")
print(f"  two‑sided p‑value = {p_val:.4e}")

# ---------- 5. Effect size (Cohen's d) ----------
pooled_sd = np.sqrt(((n_high-1)*std_high**2 + (n_low-1)*std_low**2) / (n_high+n_low-2))
cohen_d = (mean_high - mean_low) / pooled_sd
print(f"\nEffect size (Cohen's d) = {cohen_d:.3f}")

# ---------- 6. Visualisation ----------
# Prepare directory
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/wine_q3"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")

# Boxplot of alcohol by quality group
plt.figure(figsize=(8, 5))
sns.boxplot(x=pd.cut(df["quality"],
                     bins=[-np.inf, 4, 6, np.inf],
                     labels=["Low (≤4)", "Medium (5‑6)", "High (≥7)"]),
            y="alcohol",
            data=df,
            palette="Set2")
plt.xlabel("Quality group")
plt.ylabel("Alcohol (% vol.)")
plt.title("Alcohol content by wine quality")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"\nBoxplot saved to: {out_path}")