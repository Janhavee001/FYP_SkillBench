import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
chart_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_plain/titanic_q5/chart.png"

# Load data
df = pd.read_csv(csv_path)

# Extract fare column (drop missing values)
fares = df["fare"].dropna()

# Descriptive statistics
count = fares.count()
mean = fares.mean()
median = fares.median()
std = fares.std()
minimum = fares.min()
maximum = fares.max()
skewness = fares.skew()
q1 = fares.quantile(0.25)
q3 = fares.quantile(0.75)

# Print results
print(f"Count: {count}")
print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Std Dev: {std:.2f}")
print(f"Min: {minimum:.2f}")
print(f"Max: {maximum:.2f}")
print(f"25th percentile (Q1): {q1:.2f}")
print(f"75th percentile (Q3): {q3:.2f}")
print(f"Skewness: {skewness:.2f}")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(chart_path), exist_ok=True)

# Plot histogram with KDE
plt.figure(figsize=(8, 5))
sns.histplot(fares, bins=30, kde=True, color="steelblue", edgecolor="black")
plt.title("Distribution of Titanic Ticket Fares")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(chart_path, dpi=300)
plt.close()