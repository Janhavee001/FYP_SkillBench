import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Build contingency table (sex vs survived)
# ------------------------------------------------------------------
contingency = pd.crosstab(df["sex"], df["survived"])
print("Contingency table (sex x survived):")
print(contingency)

# ------------------------------------------------------------------
# 3. Chi‑squared test of independence
# ------------------------------------------------------------------
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi‑squared statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P‑value: {p:.4e}")

# ------------------------------------------------------------------
# 4. (Optional) Visualise the relationship
# ------------------------------------------------------------------
# Create output directory if it does not exist
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q2"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")

sns.set(style="whitegrid")
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sex", hue="survived", palette="Set2")
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nChart saved to: {out_path}")