import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ---------- 1. Load data ----------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/wine_quality/winequality-red.csv"
df = pd.read_csv(csv_path)

# ---------- 2. Define groups ----------
high_quality = df[df["quality"] >= 7]["alcohol"]
low_quality  = df[df["quality"] <= 4]["alcohol"]

# ---------- 3. Descriptive statistics ----------
n_high = high_quality.shape[0]
n_low  = low_quality.shape[0]

mean_high = high_quality.mean()
mean_low  = low_quality.mean()

std_high = high_quality.std(ddof=1)
std_low  = low_quality.std(ddof=1)

# ---------- 4. Statistical test ----------
t_stat, p_val = ttest_ind(high_quality, low_quality, equal_var=False)

# ---------- 5. Print results ----------
print("Number of high‑quality wines (quality >= 7):", n_high)
print("Number of low‑quality wines  (quality <= 4):", n_low)
print(f"Mean alcohol (high quality): {mean_high:.4f} %")
print(f"Mean alcohol (low quality):  {mean_low:.4f} %")
print(f"Std dev alcohol (high quality): {std_high:.4f}")
print(f"Std dev alcohol (low quality):  {std_low:.4f}")
print(f"T‑statistic: {t_stat:.4f}")
print(f"P‑value: {p_val:.4e}")

# ---------- 6. Visualisation ----------
plt.figure(figsize=(8, 6))
sns.boxplot(x=pd.cut(df["quality"],
                     bins=[-np.inf, 4, 6, np.inf],
                     labels=["Low (≤4)", "Medium (5‑6)", "High (≥7)"]),
            y="alcohol",
            data=df,
            palette="Set2")
plt.title("Alcohol Content by Wine Quality Group")
plt.xlabel("Quality Group")
plt.ylabel("Alcohol (% vol)")

# Ensure output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/wine_q3"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()