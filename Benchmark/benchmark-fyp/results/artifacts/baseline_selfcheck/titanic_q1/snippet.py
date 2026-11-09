import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute overall survival rate
# ------------------------------------------------------------------
overall_survival_rate = df["survived"].mean()  # survived is 0/1, so mean = proportion

# ------------------------------------------------------------------
# 3. Compute survival rate by sex
# ------------------------------------------------------------------
survival_by_sex = df.groupby("sex")["survived"].mean()

male_survival_rate = survival_by_sex.get("male", float("nan"))
female_survival_rate = survival_by_sex.get("female", float("nan"))

# ------------------------------------------------------------------
# 4. Print the results (rounded to 4 decimal places)
# ------------------------------------------------------------------
print(f"Overall survival rate: {overall_survival_rate:.4f} ({overall_survival_rate*100:.2f}%)")
print(f"Male survival rate   : {male_survival_rate:.4f} ({male_survival_rate*100:.2f}%)")
print(f"Female survival rate : {female_survival_rate:.4f} ({female_survival_rate*100:.2f}%)")

# ------------------------------------------------------------------
# 5. Create a bar chart and save it
# ------------------------------------------------------------------
# Prepare output directory
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q1"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")

# Plot
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values, palette="pastel")
plt.ylim(0, 1)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
# Annotate bars with percentages
for i, rate in enumerate(survival_by_sex.values):
    plt.text(i, rate + 0.02, f"{rate*100:.1f}%", ha='center')

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()