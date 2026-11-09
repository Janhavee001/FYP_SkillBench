import matplotlib
matplotlib.use('Agg')
import warnings; warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# -----------------------------
# 1. Load the data
# -----------------------------
csv_path = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/datasets/titanic/titanic.csv"
df = pd.read_csv(csv_path)

# -----------------------------
# 2. Isolate the fare column (drop missing values)
# -----------------------------
fares = df["fare"].dropna()

# -----------------------------
# 3. Compute summary statistics
# -----------------------------
count = fares.count()
mean_fare = fares.mean()
median_fare = fares.median()
std_fare = fares.std()
min_fare = fares.min()
max_fare = fares.max()
q1 = fares.quantile(0.25)
q3 = fares.quantile(0.75)
skew_fare = skew(fares, bias=False)

# Print the numbers a reader needs
print(f"Count: {count}")
print(f"Mean: {mean_fare:.4f}")
print(f"Median: {median_fare:.4f}")
print(f"Std Dev: {std_fare:.4f}")
print(f"Min: {min_fare:.4f}")
print(f"25th percentile (Q1): {q1:.4f}")
print(f"75th percentile (Q3): {q3:.4f}")
print(f"Max: {max_fare:.4f}")
print(f"Skewness: {skew_fare:.4f}")

# -----------------------------
# 4. Visualise the distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(fares, bins=30, kde=True, color="steelblue")
plt.title("Distribution of Titanic Ticket Fares")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

# Ensure the output directory exists
out_dir = r"C:/Users/janha/Downloads/FYP/Benchmark/benchmark-fyp/results/artifacts/baseline_selfcheck/titanic_q5"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "chart.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()