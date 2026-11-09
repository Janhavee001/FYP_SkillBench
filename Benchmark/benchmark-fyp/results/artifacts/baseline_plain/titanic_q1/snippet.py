import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths (use raw strings to avoid escape‑character issues)
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q1/chart.png"

# ------------------------------------------------------------------
# Load data
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# Compute overall and sex‑specific survival rates
overall_rate = df["survived"].mean()                     # proportion (0‑1)
male_rate    = df.loc[df["sex"] == "male", "survived"].mean()
female_rate  = df.loc[df["sex"] == "female", "survived"].mean()

# Print the numbers a reader needs
print(f"Overall survival rate: {overall_rate:.2%}")
print(f"Male survival rate:    {male_rate:.2%}")
print(f"Female survival rate:  {female_rate:.2%}")

# ------------------------------------------------------------------
# Create a bar chart of survival rate by sex
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
# Because “survived” is 0/1, the mean equals the survival proportion
ax = sns.barplot(x="sex", y="survived", data=df, ci=None, palette="pastel")
ax.set_ylabel("Survival Rate")
ax.set_xlabel("Sex")
ax.set_ylim(0, 1)
# Show percentages on top of bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.1%}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom", fontsize=10)

# Ensure the output directory exists
os.makedirs(os.path.dirname(chart_path), exist_ok=True)
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()