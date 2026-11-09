import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# ------------------------------------------------------------------
# 2. Compute survival rates per class
# ------------------------------------------------------------------
# total passengers per class
total_per_class = df.groupby('pclass').size()
# survivors per class
survived_per_class = df.groupby('pclass')['survived'].sum()
# survival rate (as a proportion)
survival_rate = survived_per_class / total_per_class

# Print the numbers a reader needs
print("Passenger counts per class:")
print(total_per_class.to_string())
print("\nSurvivors per class:")
print(survived_per_class.to_string())
print("\nSurvival rate per class (as %):")
print((survival_rate * 100).round(2).to_string() + " %")

# ------------------------------------------------------------------
# 3. Visualise
# ------------------------------------------------------------------
# Prepare a DataFrame for seaborn
plot_df = survival_rate.reset_index()
plot_df.columns = ['pclass', 'survival_rate']
plot_df['survival_rate_pct'] = plot_df['survival_rate'] * 100

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
barplot = sns.barplot(
    x='pclass',
    y='survival_rate_pct',
    data=plot_df,
    palette='viridis'
)

# Add exact percentages on top of bars
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.1f%%', padding=3)

plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate (%)')
plt.ylim(0, 100)

# ------------------------------------------------------------------
# 4. Save chart
# ------------------------------------------------------------------
output_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q4/chart.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()