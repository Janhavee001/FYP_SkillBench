import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load data
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute survival rate per class
#    survived is 0/1, so the mean gives the proportion that survived.
survival_by_class = df.groupby("pclass")["survived"].mean().reset_index()
survival_by_class["survival_rate_%"] = survival_by_class["survived"] * 100

# 3. Print the numbers a reader needs
print("Survival rate by passenger class (percentage):")
for _, row in survival_by_class.iterrows():
    print(f"  Class {int(row['pclass'])}: {row['survival_rate_%']:.2f}%")

# ------------------------------------------------------------------
# 4. Create a clear chart
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.barplot(
    data=survival_by_class,
    x="pclass",
    y="survived",
    palette="viridis"
)

# Convert y‑axis from proportion to percent for readability
ax.set_ylabel("Survival Rate")
ax.set_xlabel("Passenger Class")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([f"{int(t*100)}%" for t in ax.get_yticks()])

# Add exact percentages on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height*100:.1f}%",
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

# ------------------------------------------------------------------
# 5. Save the figure
output_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q4/chart.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()